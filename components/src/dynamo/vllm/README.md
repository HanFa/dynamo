<!-- # SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# vLLM

See [docs/backends/vllm/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/vllm/overview.md) for documentation.

## Experimental multimodal KV handoff

Related: [HanFa/dynamo#2](https://github.com/HanFa/dynamo/issues/2).

The optional image P/D path uses vLLM's processed-input export/restore API to
continue decoding without loading or preprocessing images. It requires the
companion [vLLM branch](https://github.com/HanFa/vllm/tree/codex/mm-kv-handoff-2),
based on v0.31.0, on **both** prefill and decode workers. A stock vLLM installation
retains the existing media-loading path. This feature is off by default.

Set `DYN_VLLM_MULTIMODAL_KV_HANDOFF=1` on both workers. The initial engine capability
supports image-only `LlavaForConditionalGeneration` requests with `NixlConnector`
and `kv_load_failure_policy="fail"`. LoRA, speculative decoding, encoder-cache
connectors, prompt logprobs and models requiring additional position state are
not enabled. Unsupported requests/configurations log a fallback reason and use
the existing image path. Prefix caching and media processor caches remain enabled.

Prefill invokes `InputProcessor.prepare_multimodal_kv_handoff()` once and submits
that exact rendered input to generation. The engine exports expanded tokens,
resolved image hashes, placeholder ranges/masks, cache salt, a model fingerprint
and a versioned position-state contract. Dynamo treats this state as opaque,
binds it to the request's prompt/media/cache scope and the returned KV transfer,
and forwards it in `embedding_params.multimodal_kv_handoff`.

Decode calls `InputProcessor.restore_multimodal_kv_handoff()` before media loading.
The engine distinguishes KV-backed media from an ordinary receiver-cache lookup.
Every scheduling attempt must have KV covering all image spans; a partial load,
missing image-tail block or preemption that loses coverage ends the request with
an error. The caller must retry through prefill. It never recomputes image spans
using missing or dummy media. Cancellation uses the existing request lifecycle.
Malformed or mismatched handoffs are request errors, not silent fallbacks.

### Validation

Run the adapter, processor and handler tests in the Dynamo vLLM test environment:

```bash
python -m pytest components/src/dynamo/vllm/tests/multimodal_utils/test_kv_handoff.py \
  components/src/dynamo/vllm/tests/multimodal_utils/test_vllm_request_processor.py \
  components/src/dynamo/vllm/tests/test_vllm_worker_handler.py
```

In the companion vLLM checkout, run `tests/multimodal/test_kv_handoff.py` and the
`media_free` cases in `tests/v1/core/test_scheduler.py`. The engine tests exercise
wire roundtrips, identity preservation, malformed layouts, cold receiver caches,
partial coverage, asynchronous loading and coverage loss after preemption.

Before deployment, run a 1P1D LLaVA evaluation with prefix caching enabled:
equal-size images A → B → A, cold and warm caches, then multiple images and
changed processor parameters. Compare answers, prompt lengths and block hashes
with the existing media-loading baseline. Instrument decode image loads,
processor calls and vision-encoder calls (all must be zero on the supported
successful path), and inject KV load failures, image-tail boundary misses,
cancellation and preemption. GPU correctness and performance are not established
by the CPU tests alone.
