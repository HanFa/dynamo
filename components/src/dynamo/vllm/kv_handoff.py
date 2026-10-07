# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport vLLM-owned multimodal state bound to a request and KV transfer."""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _request_digest(request: dict) -> str:
    # Routing, stop conditions and prefill_result legitimately differ between
    # the two legs. Bind only the prompt, media and cache-isolation inputs.
    extra = request.get("extra_args") or {}
    return _digest(
        {
            "request": {
                key: request.get(key)
                for key in (
                    "model",
                    "token_ids",
                    "multi_modal_data",
                    "multi_modal_uuids",
                    "mm_processor_kwargs",
                    "image_cache_scope",
                    "nvext",
                )
            },
            "extra": {
                key: extra.get(key)
                for key in (
                    "mm_processor_kwargs",
                    "mm_hashes",
                    "mm_hashes_by_modality",
                    "nvext",
                )
            },
        }
    )


@dataclass(frozen=True)
class PendingKvHandoff:
    request_digest: str
    engine_state: dict

    def bind(self, kv_transfer_params: dict) -> dict:
        """Bind engine state only after prefill produces the actual KV handle."""
        return {
            "version": 1,
            "request_digest": self.request_digest,
            "kv_transfer_digest": _digest(kv_transfer_params),
            "engine_state": self.engine_state,
        }


class MultimodalKvHandoff:
    """Adapter for an optional vLLM capability; engine state stays opaque here."""

    def __init__(self, engine_client: Any, *, enabled: bool):
        self.enabled = enabled
        self._api = getattr(engine_client, "input_processor", None)

    def _fallback_reason(self, request: dict) -> str | None:
        if getattr(self._api, "external_kv_handoff_version", None) != 1:
            return "engine lacks multimodal KV handoff v1"
        error = self._api.get_external_kv_handoff_error()
        if error is not None:
            return error
        if "vllm_tito" in (request.get("extra_args") or {}):
            return "pre-rendered token-in-token-out requests use their existing path"
        media = request.get("multi_modal_data") or {}
        if set(media) != {"image_url"} or not media["image_url"]:
            return "request is not image-only"
        if (request.get("output_options") or {}).get("prompt_logprobs") is not None:
            return "prompt logprobs require prompt computation"
        return None

    async def prepare(self, request: dict, prompt: Any):
        """Ask vLLM to render once and export its processed input."""
        if not self.enabled:
            return prompt, None
        reason = self._fallback_reason(request)
        if reason is not None:
            logger.info("Multimodal KV handoff fallback: %s", reason)
            return prompt, None
        rendered, engine_state = await self._api.prepare_multimodal_kv_handoff(prompt)
        return rendered, PendingKvHandoff(_request_digest(request), engine_state)

    def restore(self, request: dict) -> Any:
        """Restore before media loading, or retain the established media path."""
        if not self.enabled:
            return None
        params = (request.get("prefill_result") or {}).get("disaggregated_params") or {}
        state = (params.get("embedding_params") or {}).get("multimodal_kv_handoff")
        if state is None:
            logger.info("Multimodal KV handoff fallback: prefill supplied no state")
            return None
        reason = self._fallback_reason(request)
        if reason is not None:
            logger.info("Multimodal KV handoff fallback: %s", reason)
            return None
        if (
            not isinstance(state, dict)
            or type(state.get("version")) is not int
            or state["version"] != 1
        ):
            raise ValueError("Unsupported multimodal KV handoff version")
        kv_params = params.get("kv_transfer_params")
        if not isinstance(kv_params, dict) or not kv_params.get("do_remote_prefill"):
            raise ValueError("Multimodal KV handoff has no remote prefill transfer")
        if state.get("request_digest") != _request_digest(request):
            raise ValueError("Multimodal KV handoff does not match this request")
        if state.get("kv_transfer_digest") != _digest(kv_params):
            raise ValueError("Multimodal KV handoff does not match this KV transfer")
        return self._api.restore_multimodal_kv_handoff(state.get("engine_state"))
