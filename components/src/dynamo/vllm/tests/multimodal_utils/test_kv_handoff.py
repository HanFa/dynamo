# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the transport boundary without loading a model or the vLLM runtime."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.vllm.kv_handoff import MultimodalKvHandoff

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def handoff():
    # The engine owns this schema; the adapter must not interpret it.
    engine_state = {"opaque_engine_state": ["image-a", 576]}
    rendered = {"type": "multimodal", "prompt_token_ids": [1, 99, 99, 2]}
    api = SimpleNamespace(
        external_kv_handoff_version=1,
        get_external_kv_handoff_error=Mock(return_value=None),
        prepare_multimodal_kv_handoff=AsyncMock(return_value=(rendered, engine_state)),
        restore_multimodal_kv_handoff=Mock(return_value=rendered),
    )
    adapter = MultimodalKvHandoff(SimpleNamespace(input_processor=api), enabled=True)
    request = {
        "token_ids": [1, 99, 2],
        "multi_modal_data": {"image_url": [{"Url": "https://example.com/a.png"}]},
        "image_cache_scope": "tenant-a",
    }
    return adapter, api, request


async def _prepare_decode(handoff):
    adapter, api, request = handoff
    prompt, pending = await adapter.prepare(request, {"prompt_token_ids": [1, 99, 2]})
    assert prompt is api.prepare_multimodal_kv_handoff.return_value[0]
    params = {
        "do_remote_prefill": True,
        "remote_request_id": "prefill-a",
        "remote_block_ids": [[3, 4]],
    }
    decode = deepcopy(request)
    decode["prefill_result"] = {
        "disaggregated_params": {
            "kv_transfer_params": params,
            "embedding_params": {"multimodal_kv_handoff": pending.bind(params)},
        }
    }
    return decode


@pytest.mark.asyncio
async def test_roundtrip_preserves_opaque_engine_state_and_allows_routing_changes(
    handoff,
):
    adapter, api, _ = handoff
    decode = await _prepare_decode(handoff)
    decode["routing"] = {"worker_id": 42}
    decode["stop_conditions"] = {"max_tokens": 16}
    assert adapter.restore(decode) is api.restore_multimodal_kv_handoff.return_value
    api.restore_multimodal_kv_handoff.assert_called_once_with(
        {"opaque_engine_state": ["image-a", 576]}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        "token_ids",
        "image",
        "scope",
        "processor",
        "salt",
        "transfer",
        "version",
        "missing_transfer",
    ],
)
async def test_rejects_mismatched_request_or_kv_before_engine_restore(handoff, changed):
    adapter, api, _ = handoff
    decode = await _prepare_decode(handoff)
    params = decode["prefill_result"]["disaggregated_params"]
    if changed == "token_ids":
        decode["token_ids"][0] = 5
    elif changed == "image":
        decode["multi_modal_data"]["image_url"][0]["Url"] = "https://example.com/b.png"
    elif changed == "scope":
        decode["image_cache_scope"] = "tenant-b"
    elif changed == "processor":
        decode["mm_processor_kwargs"] = {"size": 224}
    elif changed == "salt":
        decode["nvext"] = {"cache_salt": "other"}
    elif changed == "transfer":
        params["kv_transfer_params"]["remote_block_ids"] = [[8, 9]]
    elif changed == "version":
        params["embedding_params"]["multimodal_kv_handoff"]["version"] = 2
    else:
        del params["kv_transfer_params"]
    with pytest.raises(ValueError):
        adapter.restore(decode)
    api.restore_multimodal_kv_handoff.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsupported",
    ["disabled", "engine", "model", "mixed", "logprobs", "tito", "missing"],
)
async def test_unsupported_paths_keep_media_loading_available(handoff, unsupported):
    adapter, api, request = handoff
    if unsupported == "disabled":
        adapter.enabled = False
    elif unsupported == "engine":
        api.external_kv_handoff_version = 0
    elif unsupported == "model":
        api.get_external_kv_handoff_error.return_value = "unsupported model"
    elif unsupported == "mixed":
        request["multi_modal_data"]["video_url"] = [{"Url": "video"}]
    elif unsupported == "logprobs":
        request["output_options"] = {"prompt_logprobs": 1}
    elif unsupported == "tito":
        request["extra_args"] = {"vllm_tito": {"cache_salt": "tenant"}}
    if unsupported != "missing":
        prompt = {"multi_modal_data": "pixels"}
        assert await adapter.prepare(request, prompt) == (prompt, None)
        api.prepare_multimodal_kv_handoff.assert_not_awaited()
    assert adapter.restore(request) is None


@pytest.mark.asyncio
async def test_engine_validation_failure_does_not_silently_load_other_media(handoff):
    adapter, api, _ = handoff
    decode = await _prepare_decode(handoff)
    api.restore_multimodal_kv_handoff.side_effect = ValueError("model mismatch")
    with pytest.raises(ValueError, match="model mismatch"):
        adapter.restore(decode)
