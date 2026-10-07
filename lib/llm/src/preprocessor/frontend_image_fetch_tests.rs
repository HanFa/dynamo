// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use base64::Engine;

const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

fn preprocessor(enabled: bool, decode: bool) -> Arc<OpenAIPreprocessor> {
    let mut card = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    card.media_fetcher = Some(media::MediaFetcher {
        allow_direct_ip: true,
        allow_direct_port: true,
        allow_private_ips: true,
        ..Default::default()
    });
    if decode {
        card.media_decoder = Some(MediaDecoder::default());
    }
    let tokenizer = card.tokenizer().unwrap();
    let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(&card).unwrap();
    OpenAIPreprocessor::new_with_parts_and_image_fetch(
        card,
        formatter,
        tokenizer,
        None,
        enabled.then(|| FrontendImageService::new("dynamo.images-test.fetch".into())),
    )
    .unwrap()
}

#[cfg(feature = "mm-routing")]
fn builder() -> PreprocessedRequestBuilder {
    let mut builder = PreprocessedRequestBuilder::default();
    builder
        .model("test-model".into())
        .token_ids(vec![1, 2, 3])
        .stop_conditions(Default::default())
        .sampling_options(Default::default())
        .output_options(Default::default());
    builder
}

#[tokio::test]
async fn frontend_image_fetch_preserves_uuid_alignment_and_tokens_through_preprocessing() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("GET", "/image")
        .with_body(
            base64::engine::general_purpose::STANDARD
                .decode(PNG)
                .unwrap(),
        )
        .expect(1)
        .create_async()
        .await;
    let url = format!("{}/image", server.url());
    let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [{"role":"user", "content": [
            {"type":"text", "text":"Describe the images."},
            {"type":"image_url", "image_url":{"url":url}, "uuid":"first"},
            {"type":"image_url", "uuid":"cached"},
            {"type":"image_url", "image_url":{"url":url}}
        ]}]
    }))
    .unwrap();
    let mut baseline_tokens = None;
    let mut baseline_prompt = None;
    for enabled in [false, true] {
        let preprocessor = preprocessor(enabled, false);
        let (result, _, _) = preprocessor
            .preprocess_request(&request, None)
            .await
            .unwrap();
        let prompt = result.extra_args.as_ref().unwrap()["formatted_prompt"].clone();
        if enabled {
            assert_eq!(Some(result.token_ids.clone()), baseline_tokens);
            assert_eq!(Some(prompt), baseline_prompt);
            let references = result.image_fetches.as_ref().unwrap();
            assert!(references[1].is_none());
            assert_eq!(
                references[0].as_ref().unwrap().token,
                references[2].as_ref().unwrap().token
            );
            let image = result
                .image_fetch_lease
                .as_ref()
                .unwrap()
                .image(references[0].as_ref().unwrap())
                .unwrap();
            assert_eq!(
                image.get().await.unwrap().data.as_str(),
                format!("data:image/png;base64,{PNG}")
            );
        } else {
            baseline_tokens = Some(result.token_ids.clone());
            baseline_prompt = Some(prompt);
            assert!(result.image_fetches.is_none());
            assert!(
                serde_json::to_value(&result)
                    .unwrap()
                    .get("image_fetches")
                    .is_none()
            );
        }
        let images = &result.multi_modal_data.as_ref().unwrap()["image_url"];
        assert_eq!(images.len(), 3);
        assert!(matches!(&images[1], MultimodalData::UuidOnly(uuid) if uuid == "cached"));
        assert_eq!(
            result.multi_modal_uuids.as_ref().unwrap()["image_url"],
            vec![Some("first".into()), Some("cached".into()), None]
        );
        for index in [0, 2] {
            let MultimodalData::Url(value) = &images[index] else {
                panic!()
            };
            assert_eq!(value.as_str(), url.as_str());
        }
        // Both legacy execution URLs and template metadata preserve the source.
        // Encoded bytes are returned only when a worker requests the reference.
        let messages = &result.extra_args.as_ref().unwrap()["messages"];
        assert!(!messages.to_string().contains(PNG));
        assert_eq!(messages[0]["content"][1]["image_url"]["url"], url);
    }
    mock.assert_async().await;
}

#[test]
#[cfg(feature = "testing-nixl")]
fn frontend_image_fetch_is_disabled_when_frontend_decoding_is_active() {
    let preprocessor = preprocessor(true, true);
    assert!(preprocessor.media_loader.is_some());
    assert!(preprocessor.frontend_image_fetcher.is_none());
}

#[cfg(feature = "mm-routing")]
#[tokio::test]
async fn frontend_image_fetch_preserves_original_hash_and_reuses_bytes_for_dimensions() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("GET", "/image")
        .with_body(
            base64::engine::general_purpose::STANDARD
                .decode(PNG)
                .unwrap(),
        )
        .expect(1)
        .create_async()
        .await;
    let url = format!("{}/image", server.url());
    let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model":"test-model", "messages":[{"role":"user", "content":[
            {"type":"image_url", "image_url":{"url":url}},
            {"type":"image_url", "image_url":{"url":url}}
        ]}]
    }))
    .unwrap();
    let entries = preprocessor(true, false)
        .gather_multi_modal_data(&request, &mut builder(), None, &[1, 2, 3])
        .await
        .unwrap();
    assert_eq!(entries.len(), 2);
    for entry in entries {
        assert_eq!(entry.mm_hash, OpenAIPreprocessor::hash_image_url(&url));
        assert_eq!((entry.width, entry.height), (1, 1));
    }
    mock.assert_async().await;
}
