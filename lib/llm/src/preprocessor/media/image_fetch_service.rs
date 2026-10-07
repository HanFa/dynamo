// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-scoped image futures served over the existing Dynamo request plane.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use anyhow::Result;
use async_trait::async_trait;
use dashmap::DashMap;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider};
use dynamo_runtime::error::{DynamoError, ErrorClass};
use dynamo_runtime::pipeline::{ManyOut, ResponseStream, SingleIn, network::Ingress};
use dynamo_runtime::protocols::annotated::Annotated;
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};
use tokio_util::sync::CancellationToken;
use url::Url;
use uuid::Uuid;

use super::frontend_image_fetch::{FetchedImage, FrontendImageFetcher};
use crate::protocols::common::invalid_argument_error;
use crate::protocols::common::preprocessor::{
    ImageFetchReference, MultimodalData, MultimodalDataMap,
};

const MAX_PENDING_IMAGES: usize = 1024;
type Entries = DashMap<Uuid, Weak<PendingImage>>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ImageFetchFailure {
    pub class: ErrorClass,
    pub message: String,
}

impl ImageFetchFailure {
    fn from_error(error: anyhow::Error) -> Self {
        Self {
            class: error
                .downcast_ref::<DynamoError>()
                .map_or(ErrorClass::InvalidArgument, DynamoError::class),
            // Fetch errors are intentionally bounded and never contain origin URLs.
            message: "Frontend could not fetch image".into(),
        }
    }

    fn into_error(self) -> anyhow::Error {
        DynamoError::builder()
            .class(self.class)
            .public_message(self.message)
            .build()
            .into()
    }
}

pub(crate) struct PendingImage {
    result: watch::Receiver<Option<Result<Arc<FetchedImage>, ImageFetchFailure>>>,
    abort: tokio::task::AbortHandle,
    _slot: OwnedSemaphorePermit,
}

impl PendingImage {
    pub(crate) async fn get(&self) -> Result<Arc<FetchedImage>> {
        let mut receiver = self.result.clone();
        loop {
            if let Some(result) = receiver.borrow().clone() {
                return result.map_err(ImageFetchFailure::into_error);
            }
            receiver
                .changed()
                .await
                .map_err(|_| invalid_argument_error("Frontend image request has ended"))?;
        }
    }
}

/// Kept locally until the response stream is dropped, including retries and P/D handoff.
pub(crate) struct ImageFetchLease {
    entries: Arc<Entries>,
    images: HashMap<Uuid, Arc<PendingImage>>,
}

impl std::fmt::Debug for ImageFetchLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageFetchLease")
            .field("images", &self.images.len())
            .finish()
    }
}

impl ImageFetchLease {
    pub(crate) fn image(&self, reference: &ImageFetchReference) -> Option<Arc<PendingImage>> {
        self.images.get(&reference.token).cloned()
    }
}

impl Drop for ImageFetchLease {
    fn drop(&mut self) {
        for (token, image) in &self.images {
            self.entries.remove(token);
            image.abort.abort();
        }
    }
}

pub(crate) struct FrontendImageService {
    endpoint: String,
    entries: Arc<Entries>,
    slots: Arc<Semaphore>,
    shutdown: CancellationToken,
}

impl Drop for FrontendImageService {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

impl FrontendImageService {
    pub(crate) async fn start(runtime: &DistributedRuntime, namespace: &str) -> Result<Arc<Self>> {
        // One unique endpoint per frontend namespace. A reference can never be
        // load-balanced onto another frontend's request-local image store.
        let component = format!("images-{}", Uuid::new_v4().simple());
        let endpoint = runtime
            .namespace(namespace)?
            .component(&component)?
            .endpoint("fetch");
        let service = Self::new(format!("{namespace}.{component}.fetch"));
        let serving = endpoint
            .endpoint_builder()
            .handler(Ingress::for_engine(Arc::new(ImageFetchEngine(
                service.entries.clone(),
            )))?)
            .start_with_registration()
            .await?;
        let shutdown = service.shutdown.clone();
        let runtime_shutdown = runtime.primary_token();
        tokio::spawn(async move {
            tokio::select! {
                _ = shutdown.cancelled() => {},
                _ = runtime_shutdown.cancelled() => {},
            }
            if let Err(error) = serving.shutdown().await {
                tracing::warn!(%error, "image fetch endpoint shutdown failed");
            }
        });
        Ok(service)
    }

    pub(crate) fn new(endpoint: String) -> Arc<Self> {
        Arc::new(Self {
            endpoint,
            entries: Arc::new(DashMap::new()),
            slots: Arc::new(Semaphore::new(MAX_PENDING_IMAGES)),
            shutdown: CancellationToken::new(),
        })
    }

    pub(crate) async fn prefetch(
        &self,
        fetcher: Arc<FrontendImageFetcher>,
        media: &MultimodalDataMap,
    ) -> Result<(Vec<Option<ImageFetchReference>>, Arc<ImageFetchLease>)> {
        let images = media
            .get("image_url")
            .map(Vec::as_slice)
            .unwrap_or_default();
        let mut references = vec![None; images.len()];
        let mut by_url = HashMap::<Url, Uuid>::new();
        let mut lease = ImageFetchLease {
            entries: self.entries.clone(),
            images: HashMap::new(),
        };
        let budget = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for (index, image) in images.iter().enumerate() {
            let MultimodalData::Url(url) = image else {
                continue;
            };
            if url.scheme() == "data" {
                fetcher.reserve(&budget, url.as_str().len())?;
                continue;
            }
            if !matches!(url.scheme(), "http" | "https") {
                continue;
            }
            let token = if let Some(token) = by_url.get(url) {
                *token
            } else {
                // Validate even if the worker later hits its cache. DNS/redirect
                // connection policy is also enforced by the fetch itself.
                fetcher.validate(url).await?;
                let slot = self.slots.clone().try_acquire_owned().map_err(|_| {
                    DynamoError::builder()
                        .class(ErrorClass::CapacityExhausted)
                        .public_message("Too many pending frontend images")
                        .build()
                })?;
                let token = Uuid::new_v4();
                let (sender, receiver) = watch::channel(None);
                let fetcher = fetcher.clone();
                let url_owned = url.clone();
                let budget = budget.clone();
                let task = tokio::spawn(async move {
                    let result = fetcher
                        .fetch_with_timeout(&url_owned, &budget)
                        .await
                        .map(Arc::new)
                        .map_err(ImageFetchFailure::from_error);
                    let _ = sender.send(Some(result));
                });
                let pending = Arc::new(PendingImage {
                    result: receiver,
                    abort: task.abort_handle(),
                    _slot: slot,
                });
                self.entries.insert(token, Arc::downgrade(&pending));
                lease.images.insert(token, pending);
                by_url.insert(url.clone(), token);
                token
            };
            references[index] = Some(ImageFetchReference {
                endpoint: self.endpoint.clone(),
                token,
            });
        }
        Ok((references, Arc::new(lease)))
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ImageFetchRequest {
    #[serde(with = "uuid::serde::simple")]
    pub token: Uuid,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ImageFetchResponse {
    pub data_url: Option<Arc<Url>>,
    pub error: Option<ImageFetchFailure>,
}

struct ImageFetchEngine(Arc<Entries>);

#[async_trait]
impl AsyncEngine<SingleIn<ImageFetchRequest>, ManyOut<Annotated<ImageFetchResponse>>, anyhow::Error>
    for ImageFetchEngine
{
    async fn generate(
        &self,
        request: SingleIn<ImageFetchRequest>,
    ) -> Result<ManyOut<Annotated<ImageFetchResponse>>> {
        let (request, context) = request.into_parts();
        let pending = self.0.get(&request.token).and_then(|entry| entry.upgrade());
        let result = match pending {
            Some(image) => image.get().await,
            None => Err(invalid_argument_error(
                "Frontend image reference has expired",
            )),
        };
        let response = match result {
            Ok(image) => ImageFetchResponse {
                data_url: Some(image.data.clone()),
                error: None,
            },
            Err(error) => ImageFetchResponse {
                data_url: None,
                error: Some(ImageFetchFailure::from_error(error)),
            },
        };
        Ok(ResponseStream::new(
            Box::pin(futures::stream::once(async {
                Annotated::from_data(response)
            })),
            context.context(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::MediaFetcher;
    use super::*;
    use base64::Engine;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    #[test]
    fn frontend_image_fetch_reference_uses_strings_in_json_and_msgpack() {
        let token = Uuid::new_v4();
        let reference = ImageFetchReference {
            endpoint: "test.images-0123456789abcdef0123456789abcdef.fetch".into(),
            token,
        };
        let json = serde_json::to_value(&reference).unwrap();
        let msgpack = rmp_serde::to_vec_named(&reference).unwrap();
        let value: serde_json::Value = rmp_serde::from_slice(&msgpack).unwrap();
        assert_eq!(value, json);
        assert_eq!(value["token"], token.simple().to_string());
        let decoded: ImageFetchReference = rmp_serde::from_slice(&msgpack).unwrap();
        assert_eq!(decoded.token, token);

        // Python sends the token back as a string through the same request plane.
        let request = serde_json::json!({ "token": value["token"] });
        let decoded: ImageFetchRequest =
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&request).unwrap()).unwrap();
        assert_eq!(decoded.token, token);
        let decoded: ImageFetchRequest = serde_json::from_value(request).unwrap();
        assert_eq!(decoded.token, token);
    }

    fn fetcher() -> Arc<FrontendImageFetcher> {
        Arc::new(
            FrontendImageFetcher::new(MediaFetcher {
                allow_direct_ip: true,
                allow_direct_port: true,
                allow_private_ips: true,
                ..Default::default()
            })
            .unwrap(),
        )
    }

    fn media(url: &str) -> MultimodalDataMap {
        HashMap::from([(
            "image_url".into(),
            vec![MultimodalData::Url(Url::parse(url).unwrap())],
        )])
    }

    #[tokio::test]
    async fn frontend_image_fetch_reference_shares_one_download_and_preserves_slots() {
        let mut origin = mockito::Server::new_async().await;
        let mock = origin
            .mock("GET", "/image")
            .with_body(
                base64::engine::general_purpose::STANDARD
                    .decode(PNG)
                    .unwrap(),
            )
            .expect(1)
            .create_async()
            .await;
        let url = format!("{}/image", origin.url());
        let mut images = media(&url);
        images.get_mut("image_url").unwrap().extend([
            MultimodalData::UuidOnly("cached".into()),
            MultimodalData::Url(Url::parse(&url).unwrap()),
            MultimodalData::Url(Url::parse(&format!("data:image/png;base64,{PNG}")).unwrap()),
        ]);
        let service = FrontendImageService::new("test".into());
        let (references, lease) = service.prefetch(fetcher(), &images).await.unwrap();
        assert!(references[1].is_none() && references[3].is_none());
        assert_eq!(
            references[0].as_ref().unwrap().token,
            references[2].as_ref().unwrap().token
        );
        let image = lease.image(references[0].as_ref().unwrap()).unwrap();
        let (prefill, decode) = tokio::join!(image.get(), image.get());
        assert!(Arc::ptr_eq(&prefill.unwrap(), &decode.unwrap()));
        assert!(
            matches!(&images["image_url"][0], MultimodalData::Url(value) if value.as_str() == url)
        );
        drop(lease);
        assert!(service.entries.is_empty());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn frontend_image_fetch_reference_does_not_wait_and_lease_drop_cancels_download() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/image", listener.local_addr().unwrap());
        let (started, received) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0; 4096];
            socket.read(&mut buffer).await.unwrap();
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n")
                .await
                .unwrap();
            started.send(()).unwrap();
            assert_eq!(socket.read(&mut buffer).await.unwrap(), 0);
        });
        let service = FrontendImageService::new("test".into());
        let (references, lease) = tokio::time::timeout(
            Duration::from_secs(2),
            service.prefetch(fetcher(), &media(&url)),
        )
        .await
        .unwrap()
        .unwrap();
        received.await.unwrap();
        let pending = lease.image(references[0].as_ref().unwrap()).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), pending.get())
                .await
                .is_err()
        );
        drop(lease);
        assert!(service.entries.is_empty());
        assert!(
            tokio::time::timeout(Duration::from_secs(2), pending.get())
                .await
                .unwrap()
                .is_err()
        );
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap();
        drop(pending);
        assert_eq!(service.slots.available_permits(), MAX_PENDING_IMAGES);
    }

    #[tokio::test]
    async fn frontend_image_fetch_reference_is_request_scoped_and_expires() {
        let service = FrontendImageService::new("test".into());
        let mut origin = mockito::Server::new_async().await;
        let mock = origin
            .mock("GET", "/missing")
            .with_status(404)
            .expect(2)
            .create_async()
            .await;
        let images = media(&format!("{}/missing", origin.url()));
        let mut tokens = Vec::new();
        for _ in 0..2 {
            let (references, lease) = service.prefetch(fetcher(), &images).await.unwrap();
            let reference = references[0].as_ref().unwrap();
            tokens.push(reference.token);
            assert!(lease.image(reference).unwrap().get().await.is_err());
            drop(lease);
        }
        assert_ne!(tokens[0], tokens[1]);
        assert!(service.entries.is_empty());
        let mut response = ImageFetchEngine(service.entries.clone())
            .generate(dynamo_runtime::pipeline::Context::new(ImageFetchRequest {
                token: tokens[0],
            }))
            .await
            .unwrap();
        use futures::StreamExt;
        assert!(response.next().await.unwrap().data.unwrap().error.is_some());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn frontend_image_fetch_reference_admission_is_bounded() {
        let service = FrontendImageService::new("test".into());
        let _slots = service
            .slots
            .acquire_many(MAX_PENDING_IMAGES as u32)
            .await
            .unwrap();
        let result = service
            .prefetch(fetcher(), &media("http://127.0.0.1/image"))
            .await;
        let error = result.unwrap_err();
        assert_eq!(
            error.downcast_ref::<DynamoError>().unwrap().class(),
            ErrorClass::CapacityExhausted
        );
        assert!(service.entries.is_empty());
    }
}
