// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional, request-scoped fetching of encoded images for URL-passthrough workers.

#[cfg(test)]
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Result;
use base64::Engine;
#[cfg(test)]
use futures::{StreamExt, TryStreamExt};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use url::Url;

use super::{MediaFetcher, max_data_url_bytes};
use crate::protocols::common::invalid_argument_error;
#[cfg(test)]
use crate::protocols::common::preprocessor::{MultimodalData, MultimodalDataMap};
use dynamo_runtime::error::{DynamoError, ErrorClass};

const MAX_CONCURRENT_FETCHES: usize = 4;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
// Encoded bytes for unique fetched URLs and existing inline images. The
// request plane separately bounds each serialized frame, including text,
// metadata and transport headers, with DYN_TCP_MAX_MESSAGE_SIZE.
const MAX_REQUEST_IMAGE_BYTES: usize = 16 * 1024 * 1024;

fn fetch_timeout() -> anyhow::Error {
    DynamoError::builder()
        .class(ErrorClass::DeadlineExceeded)
        .public_message("Frontend image fetching timed out")
        .build()
        .into()
}

fn fetch_error(error: reqwest::Error) -> anyhow::Error {
    if error.is_timeout() {
        fetch_timeout()
    } else {
        invalid_argument_error("Could not fetch frontend image")
    }
}

#[derive(Debug)]
pub(crate) struct FetchedImage {
    pub(crate) data: Arc<Url>,
    _memory: Vec<OwnedSemaphorePermit>,
}

pub(crate) struct FrontendImageFetcher {
    policy: MediaFetcher,
    client: reqwest::Client,
    permits: Semaphore,
    memory: Arc<Semaphore>,
    timeout: Duration,
    max_request_bytes: usize,
}

impl FrontendImageFetcher {
    pub(crate) fn new(policy: MediaFetcher) -> Result<Self> {
        Ok(Self {
            client: policy.build_direct_http_client()?,
            policy,
            permits: Semaphore::new(MAX_CONCURRENT_FETCHES),
            memory: Arc::new(Semaphore::new(64 * 1024 * 1024)),
            timeout: REQUEST_TIMEOUT,
            max_request_bytes: MAX_REQUEST_IMAGE_BYTES,
        })
    }

    pub(crate) async fn validate(&self, url: &Url) -> Result<()> {
        tokio::time::timeout(self.timeout, self.policy.check_if_url_allowed_with_dns(url))
            .await
            .map_err(|_| fetch_timeout())?
    }

    pub(crate) async fn fetch_with_timeout(
        &self,
        url: &Url,
        budget: &AtomicUsize,
    ) -> Result<FetchedImage> {
        tokio::time::timeout(
            self.timeout,
            self.fetch(url, 1, max_data_url_bytes(), budget),
        )
        .await
        .map_err(|_| fetch_timeout())?
    }

    #[cfg(test)]
    pub(crate) async fn rewrite_images(
        &self,
        media: &mut MultimodalDataMap,
    ) -> Result<Vec<Option<Url>>> {
        let Some(images) = media.get_mut("image_url") else {
            return Ok(Vec::new());
        };
        let mut urls: HashMap<Url, Vec<usize>> = HashMap::new();
        let mut sources = vec![None; images.len()];
        let budget = AtomicUsize::new(0);
        for (index, image) in images.iter().enumerate() {
            if let MultimodalData::Url(url) = image {
                match url.scheme() {
                    "http" | "https" => {
                        sources[index] = Some(url.clone());
                        urls.entry(url.clone()).or_default().push(index);
                    }
                    "data" => self.reserve(&budget, url.as_str().len())?,
                    _ => {}
                }
            }
        }
        let per_image_limit = max_data_url_bytes();
        // No spawned tasks or persistent cache: dropping this future cancels
        // queued and active downloads, and errors discard all partial results.
        let fetches = futures::stream::iter(urls.into_iter().map(|(url, indices)| {
            let budget = &budget;
            async move {
                let data = self
                    .fetch(&url, indices.len(), per_image_limit, budget)
                    .await?;
                Ok::<_, anyhow::Error>((indices, data))
            }
        }))
        .buffer_unordered(MAX_CONCURRENT_FETCHES)
        .try_collect::<Vec<_>>();
        let results = tokio::time::timeout(self.timeout, fetches)
            .await
            .map_err(|_| fetch_timeout())??;
        for (indices, data) in results {
            for index in indices {
                images[index] = MultimodalData::Url(data.data.as_ref().clone());
            }
        }
        Ok(sources)
    }

    pub(crate) fn reserve(&self, budget: &AtomicUsize, bytes: usize) -> Result<()> {
        budget
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(bytes)
                    .filter(|total| *total <= self.max_request_bytes)
            })
            .map_err(|_| {
                invalid_argument_error(format!(
                    "Frontend image payload exceeds the {}-byte request limit",
                    self.max_request_bytes
                ))
            })?;
        Ok(())
    }

    async fn fetch(
        &self,
        url: &Url,
        copies: usize,
        per_image_limit: usize,
        budget: &AtomicUsize,
    ) -> Result<FetchedImage> {
        // Share the concurrency bound across requests using this preprocessor.
        let _permit = self.permits.acquire().await?;
        self.policy.check_if_url_allowed_with_dns(url).await?;
        // Do not include reqwest errors: they may contain signed URLs or credentials.
        let mut response = self
            .client
            .get(url.clone())
            .send()
            .await
            .map_err(fetch_error)?
            .error_for_status()
            .map_err(|_| invalid_argument_error("Image server returned an unsuccessful status"))?;

        // Reserve enough space for any supported image MIME prefix. Sniff the
        // encoded bytes below; a server's Content-Type is not authoritative.
        const PREFIX_BYTES: usize = "data:image/jpeg;base64,".len();
        let size_error = || invalid_argument_error("Fetched image exceeds DYN_MM_MAX_DATA_URL_MB");
        let encoded_size = |bytes: usize| {
            base64::encoded_len(bytes, true)
                .and_then(|len| len.checked_add(PREFIX_BYTES))
                .filter(|len| *len <= per_image_limit)
                .ok_or_else(size_error)
        };
        if let Some(length) = response.content_length() {
            encoded_size(usize::try_from(length).map_err(|_| size_error())?)?;
        }
        let mut body = Vec::new();
        let mut reserved = 0;
        let mut memory = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(fetch_error)? {
            let length = body.len().checked_add(chunk.len()).ok_or_else(size_error)?;
            let total = encoded_size(length)?
                .checked_mul(copies)
                .ok_or_else(size_error)?;
            self.reserve(budget, total - reserved)?;
            memory.push(
                self.memory
                    .clone()
                    .try_acquire_many_owned((total - reserved) as u32)
                    .map_err(|_| {
                        DynamoError::builder()
                            .class(ErrorClass::CapacityExhausted)
                            .public_message("Frontend image buffer is full")
                            .build()
                    })?,
            );
            reserved = total;
            body.extend_from_slice(&chunk);
        }
        let mime = match image::guess_format(&body) {
            Ok(image::ImageFormat::Jpeg) => "image/jpeg",
            Ok(image::ImageFormat::Png) => "image/png",
            Ok(image::ImageFormat::Gif) => "image/gif",
            Ok(image::ImageFormat::WebP) => "image/webp",
            _ => return Err(invalid_argument_error("Unsupported fetched image format")),
        };
        let data = base64::engine::general_purpose::STANDARD.encode(body);
        Ok(FetchedImage {
            data: Arc::new(Url::parse(&format!("data:{mime};base64,{data}"))?),
            _memory: memory,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A valid 1x1 PNG; the payload is forwarded without decoding or re-encoding.
    const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    fn image_bytes() -> Vec<u8> {
        base64::engine::general_purpose::STANDARD
            .decode(PNG)
            .unwrap()
    }

    fn local_fetcher() -> FrontendImageFetcher {
        FrontendImageFetcher::new(MediaFetcher {
            allow_direct_ip: true,
            allow_direct_port: true,
            allow_private_ips: true,
            ..Default::default()
        })
        .unwrap()
    }

    fn media(urls: &[&str]) -> MultimodalDataMap {
        HashMap::from([(
            "image_url".into(),
            urls.iter()
                .map(|url| MultimodalData::Url(Url::parse(url).unwrap()))
                .collect(),
        )])
    }

    #[tokio::test]
    async fn frontend_image_fetch_deduplicates_per_request_and_preserves_other_slots() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/image")
            // Ignore an incorrect MIME header; sniff the encoded image.
            .with_header("content-type", "text/plain")
            .with_body(image_bytes())
            .expect(2)
            .create_async()
            .await;
        let url = format!("{}/image", server.url());
        let inline = format!("data:image/png;base64,{PNG}");
        let mut jpeg = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(2, 1)
            .write_to(&mut jpeg, image::ImageFormat::Jpeg)
            .unwrap();
        let jpeg = jpeg.into_inner();
        let other = server
            .mock("GET", "/other")
            .with_body(jpeg.clone())
            .expect(2)
            .create_async()
            .await;
        let other_url = format!("{}/other", server.url());
        let other_inline = format!(
            "data:image/jpeg;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(jpeg)
        );
        let fetcher = local_fetcher();
        for _ in 0..2 {
            let mut request = media(&[&url, &inline, &other_url, &url]);
            request
                .get_mut("image_url")
                .unwrap()
                .insert(1, MultimodalData::UuidOnly("cached".into()));
            request.insert(
                "video_url".into(),
                vec![MultimodalData::Url(Url::parse(&url).unwrap())],
            );
            request.insert(
                "audio_url".into(),
                vec![MultimodalData::Url(Url::parse(&url).unwrap())],
            );
            fetcher.rewrite_images(&mut request).await.unwrap();
            for index in [0, 2, 4] {
                let MultimodalData::Url(value) = &request["image_url"][index] else {
                    panic!()
                };
                assert_eq!(value.as_str(), inline);
            }
            assert!(
                matches!(&request["image_url"][3], MultimodalData::Url(value) if value.as_str() == other_inline)
            );
            assert!(
                matches!(&request["image_url"][1], MultimodalData::UuidOnly(id) if id == "cached")
            );
            for kind in ["video_url", "audio_url"] {
                assert!(
                    matches!(&request[kind][0], MultimodalData::Url(value) if value.as_str() == url)
                );
            }
        }
        mock.assert_async().await;
        other.assert_async().await;
    }

    #[tokio::test]
    async fn frontend_image_fetch_rejects_initial_private_destination() {
        let fetcher = FrontendImageFetcher::new(MediaFetcher::default()).unwrap();
        for url in [
            "http://127.0.0.1/image",
            "http://localhost/image",
            "http://169.254.169.254/image",
        ] {
            let error = fetcher
                .rewrite_images(&mut media(&[url]))
                .await
                .unwrap_err();
            assert!(MediaFetcher::is_policy_rejection(&error));
        }
    }

    #[tokio::test]
    async fn frontend_image_fetch_revalidates_redirects_without_leaking_urls() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/image?secret=value")
            .with_status(302)
            .with_header("location", "file:///etc/passwd")
            .create_async()
            .await;
        let url = format!("{}/image?secret=value", server.url());
        let error = local_fetcher()
            .rewrite_images(&mut media(&[&url]))
            .await
            .unwrap_err();
        assert!(MediaFetcher::is_policy_rejection(&error));
        assert!(!format!("{error:#}").contains("secret"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn frontend_image_fetch_rejects_error_status_and_non_image_body() {
        let mut server = mockito::Server::new_async().await;
        for (path, status) in [("/missing", 404), ("/html", 200)] {
            let mock = server
                .mock("GET", path)
                .with_status(status)
                .with_body("<html>secret</html>")
                .create_async()
                .await;
            let url = format!("{}{path}", server.url());
            let error = local_fetcher()
                .rewrite_images(&mut media(&[&url]))
                .await
                .unwrap_err();
            assert!(MediaFetcher::is_policy_rejection(&error));
            assert!(!format!("{error:#}").contains("secret"));
            mock.assert_async().await;
        }
    }

    #[tokio::test]
    async fn frontend_image_fetch_bounds_content_length_and_chunked_bodies() {
        let mut server = mockito::Server::new_async().await;
        let fixed = server
            .mock("GET", "/fixed")
            .with_body(image_bytes())
            .create_async()
            .await;
        let chunked = server
            .mock("GET", "/chunked")
            .with_chunked_body(|writer| writer.write_all(&image_bytes()))
            .create_async()
            .await;
        let fetcher = local_fetcher();
        for path in ["/fixed", "/chunked"] {
            let url = Url::parse(&format!("{}{path}", server.url())).unwrap();
            let error = fetcher
                .fetch(&url, 1, 32, &AtomicUsize::new(0))
                .await
                .unwrap_err();
            assert!(error.to_string().contains("DYN_MM_MAX_DATA_URL_MB"));
        }
        fixed.assert_async().await;
        chunked.assert_async().await;
    }

    #[tokio::test]
    async fn frontend_image_fetch_budget_counts_repeated_and_existing_inline_images() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/image")
            .with_body(image_bytes())
            .expect(2)
            .create_async()
            .await;
        let url = format!("{}/image", server.url());
        let inline = format!("data:image/png;base64,{PNG}");
        let mut fetcher = local_fetcher();
        fetcher.max_request_bytes = inline.len() + 1;
        for mut request in [media(&[&url, &url]), media(&[&url, &inline])] {
            let error = fetcher.rewrite_images(&mut request).await.unwrap_err();
            assert!(error.to_string().contains("request limit"));
            // A failure must not leave a partially rewritten request behind.
            assert!(
                matches!(&request["image_url"][0], MultimodalData::Url(value) if value.as_str() == url)
            );
        }
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn frontend_image_fetch_deadline_includes_concurrency_queue() {
        let mut fetcher = local_fetcher();
        fetcher.timeout = Duration::from_millis(10);
        let _held = fetcher
            .permits
            .acquire_many(MAX_CONCURRENT_FETCHES as u32)
            .await
            .unwrap();
        let mut request = media(&["https://example.com/image"]);
        let error = fetcher.rewrite_images(&mut request).await.unwrap_err();
        assert_eq!(
            error.downcast_ref::<DynamoError>().unwrap().class(),
            ErrorClass::DeadlineExceeded
        );
    }

    #[tokio::test]
    async fn frontend_image_fetch_cancellation_releases_active_download() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/image", listener.local_addr().unwrap());
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0; 4096];
            socket.read(&mut request).await.unwrap();
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n")
                .await
                .unwrap();
            started_tx.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        let fetcher = local_fetcher();
        let mut request = media(&[&url]);
        {
            let download = fetcher.rewrite_images(&mut request);
            tokio::pin!(download);
            tokio::select! {
                _ = started_rx => {},
                result = &mut download => panic!("download should be waiting for the body: {result:?}"),
                _ = tokio::time::sleep(Duration::from_secs(5)) => panic!("server did not receive request"),
            }
            assert_eq!(
                fetcher.permits.available_permits(),
                MAX_CONCURRENT_FETCHES - 1
            );
        }
        assert_eq!(fetcher.permits.available_permits(), MAX_CONCURRENT_FETCHES);
        server.abort();
    }
}
