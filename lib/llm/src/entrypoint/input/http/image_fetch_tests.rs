// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::local_model::LocalModelBuilder;
use crate::model_card::ModelDeploymentCard;
use crate::preprocessor::media::MediaFetcher;
use crate::protocols::common::llm_backend::{BackendOutput, PreprocessedRequest};
use crate::protocols::common::preprocessor::MultimodalData;
use async_trait::async_trait;
use base64::Engine;
use dynamo_runtime::Runtime;
use dynamo_runtime::discovery::DiscoverySpec;
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider};
use dynamo_runtime::pipeline::{Error, ManyOut, ResponseStream, SingleIn, network::Ingress};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use std::sync::Mutex;
use std::time::Duration;

const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

struct CapturingBackend(Mutex<Vec<PreprocessedRequest>>, DistributedRuntime);

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>
    for CapturingBackend
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<BackendOutput>>, Error> {
        let (request, context) = request.transfer(());
        if let Some(references) = &request.image_fetches {
            let reference = references[0].as_ref().unwrap();
            let parts = reference.endpoint.split('.').collect::<Vec<_>>();
            let client = self
                .1
                .namespace(parts[0])?
                .component(parts[1])?
                .endpoint(parts[2])
                .client()
                .await?;
            client.wait_for_instances().await?;
            let router = dynamo_runtime::pipeline::network::egress::push_router::PushRouter::<
                crate::preprocessor::media::image_fetch_service::ImageFetchRequest,
                Annotated<crate::preprocessor::media::image_fetch_service::ImageFetchResponse>,
            >::from_client_no_fault_detection(
                client,
                dynamo_runtime::pipeline::RouterMode::RoundRobin,
            )
            .await?;
            let mut images = router
                .generate(dynamo_runtime::pipeline::Context::new(
                    crate::preprocessor::media::image_fetch_service::ImageFetchRequest {
                        token: reference.token,
                    },
                ))
                .await?;
            let image = images.next().await.unwrap().data.unwrap();
            assert!(image.error.is_none());
            assert_eq!(
                image.data_url.unwrap().as_str(),
                format!("data:image/png;base64,{PNG}")
            );
        }
        self.0.lock().unwrap().push(request);
        let output = serde_json::from_value(serde_json::json!({
            "token_ids": [42], "tokens": ["ok"], "text": "ok",
            "finish_reason": "stop", "index": 0
        }))?;
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter([Annotated::from_data(output)])),
            context.context(),
        ))
    }
}

#[tokio::test]
async fn http_frontend_image_fetch_reaches_discovered_workers() {
    // The TCP dispatcher is process-global. Exercise each configuration in an
    // isolated process, through the real HTTP entrypoint and worker discovery.
    const ENV: &str = "DYNAMO_HTTP_IMAGE_FETCH_TEST";
    const TEST: &str = concat!(
        module_path!(),
        "::http_frontend_image_fetch_reaches_discovered_workers"
    );
    let test_name = TEST.split_once("::").unwrap().1;
    let enabled = match std::env::var(ENV) {
        Ok(value) => value == "true",
        Err(_) => {
            for value in ["false", "true"] {
                let output = tokio::time::timeout(
                    Duration::from_secs(30),
                    tokio::process::Command::from(crate::test_utils::isolated_command(test_name))
                        .env(ENV, value)
                        .env("DYN_TCP_RPC_HOST", "127.0.0.1")
                        .env("DYN_TCP_RPC_PORT", "0")
                        .env("DYN_TCP_RESPONSE_STREAM_HOST", "127.0.0.1")
                        .env("DYN_TCP_RESPONSE_STREAM_PORT", "0")
                        // The HTTP adapter validates URLs before the model's
                        // MediaFetcher policy; permit this loopback fixture.
                        .env("DYN_MM_ALLOW_INTERNAL", "1")
                        .kill_on_drop(true)
                        .output(),
                )
                .await
                .expect("HTTP image-fetch test must finish")
                .unwrap();
                crate::test_utils::assert_isolated_success(&output);
            }
            return;
        }
    };

    let _ = tracing_subscriber::fmt()
        .with_env_filter("warn,dynamo_llm::http::service::openai=debug")
        .with_ansi(false)
        .try_init();
    let mut origin = mockito::Server::new_async().await;
    let png = base64::engine::general_purpose::STANDARD
        .decode(PNG)
        .unwrap();
    let image = origin
        .mock("GET", "/image")
        .match_header("range", mockito::Matcher::Missing)
        .with_body(&png)
        .expect(usize::from(enabled))
        .create_async()
        .await;
    // MM routing probes dimensions when forwarding URLs; fetching the image
    // should reuse its bytes and eliminate this separate Range request.
    let dimensions = origin
        .mock("GET", "/image")
        .match_header("range", "bytes=0-4095")
        .with_status(206)
        .with_body(&png)
        .expect(usize::from(!enabled && cfg!(feature = "mm-routing")))
        .create_async()
        .await;
    let image_url = format!("{}/image", origin.url());
    let drt = DistributedRuntime::new(
        Runtime::from_current().unwrap(),
        DistributedConfig::process_local(),
    )
    .await
    .unwrap();
    let backend = Arc::new(CapturingBackend(Mutex::new(Vec::new()), drt.clone()));
    let endpoint = drt
        .namespace("image-fetch-test")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("generate");
    let serving = endpoint
        .endpoint_builder()
        .handler(Ingress::for_engine(backend.clone()).unwrap())
        .start_with_registration()
        .await
        .unwrap();
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
    let mut card = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
    card.model_type = ModelType::Chat;
    card.model_input = crate::model_type::ModelInput::Tokens;
    card.kv_cache_block_size = 16;
    card.worker_type = Some(WorkerType::Aggregated);
    card.media_fetcher = Some(MediaFetcher {
        allow_direct_ip: true,
        allow_direct_port: true,
        allow_private_ips: true,
        ..Default::default()
    });
    let model_name = card.display_name.clone();
    let registration = drt
        .discovery()
        .register(
            DiscoverySpec::from_model(
                "image-fetch-test".into(),
                "workers".into(),
                "generate".into(),
                &card,
            )
            .unwrap(),
        )
        .await
        .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let model = LocalModelBuilder::default()
        .http_host(Some("127.0.0.1".into()))
        .http_port(port)
        .frontend_image_fetch(enabled)
        .build()
        .await
        .unwrap();
    let frontend = tokio::spawn(HttpFrontend::default().run(
        drt.clone(),
        EngineConfig::Dynamic {
            model: Box::new(model),
            chat_engine_factory: None,
            prefill_load_estimator: None,
        },
    ));
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if let Ok(response) = client.get(format!("{base}/v1/models")).send().await
                && let Ok(body) = response.json::<serde_json::Value>().await
                && body["data"]
                    .as_array()
                    .is_some_and(|models| models.iter().any(|m| m["id"] == model_name))
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("discovered model must become ready");
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model_name,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe the images."},
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}],
            "max_tokens": 1
        }))
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body = response.text().await.unwrap();
    assert!(status.is_success(), "{status}: {body}");
    {
        let requests = backend.0.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let references = requests[0].image_fetches.as_ref();
        if enabled {
            let references = references.unwrap();
            assert_eq!(
                references[0].as_ref().unwrap().token,
                references[1].as_ref().unwrap().token
            );
        } else {
            assert!(references.is_none());
        }
        let images = &requests[0].multi_modal_data.as_ref().unwrap()["image_url"];
        assert_eq!(images.len(), 2);
        let expected = image_url;
        for image in images {
            assert!(matches!(image, MultimodalData::Url(url) if url.as_str() == expected));
        }
    }
    image.assert_async().await;
    dimensions.assert_async().await;
    drt.discovery().unregister(registration).await.unwrap();
    serving.shutdown().await.unwrap();
    drt.shutdown();
    tokio::time::timeout(Duration::from_secs(5), frontend)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}
