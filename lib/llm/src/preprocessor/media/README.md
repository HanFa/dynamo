<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Media decoding in the frontend


This component performs media download, base64 decoding, media decoding and NIXL registration. Today, this is used in the OpenAI preprocessor, to transform multimodal inputs (image_url, video_url, audio_url) into fully decoded data (pixel values, ...) accessible to the backends via NIXL.

## Usage

Media decoding is enabled when registering the MDC:

Set HTTP download options:

```python
from dynamo.llm import MediaFetcher
fetcher = MediaFetcher()
fetcher.user_agent("dynamo")
fetcher.timeout_ms(15000)
fetcher.allow_direct_ip(True)
fetcher.allow_direct_port(False)
fetcher.allowed_media_domains(["google.com"])
```

Set media decoding default options and limits:

```python
from dynamo.llm import MediaDecoder
decoder = MediaDecoder()
decoder.enable_image({"limits": {"max_image_width": 4096, "max_image_height": 4096, "max_alloc": 16*1024*1024}})
decoder.enable_video({"fps": 2.0, "max_frames": 128, "limits": {"max_alloc": 1024*1024*128*3}})
```

If `enable_image` or `enable_video` are not called, requests containing the corresponding modality will be rejected.

Register the LLM as usual, adding the media configuration:

```python
register_model(
  ...,
  media_decoder=decoder,
  media_fetcher=fetcher,
)
```


## Known Limitations

> [!WARNING]
> **Incompatible with `Dockerfile.frontend`**: Frontend media decoding, including libjpeg-turbo image decoding, is not supported when using `Dockerfile.frontend`. The standalone frontend image does not include the required NIXL/UCX dependencies or `libturbojpeg` runtime library.

> [!WARNING]
> **Requires GPU node**: The frontend must run on a node with GPU access. During media processing, decoded tensors are written to GPU memory via NIXL, which requires `libcuda.so.1` to be available. Running the frontend on a CPU-only node will fail with something like: `add UCX backend to media-loader NIXL agent: No UCX plugin found`.

> [!NOTE]
> **NIXL progress thread**: `DYN_MM_NIXL_PROGRESS_DELAY_US` sets how long the progress thread of the frontend's NIXL agent sleeps between polls (default 1000 µs). `0` trades one CPU core per agent for lower read latency over TCP. See [NIXL Progress Thread](../../../../../docs/fern/pages/use-cases/multimodal-serving/parallel-media-decoding.md#nixl-progress-thread) for the trade-off.

> [!WARNING]
> **Video decoding**: Video decoding needs to be enabled via the `dynamo-llm/media-ffmpeg` rust feature. The following ffmpeg dynamic libraries must be available on the system: `libavcodec`, `libavdevice`, `libavfilter`, `libavformat`, `libswresample`, `libswscale`. These are available in dynamo dockerfiles rendered with `enable_media_ffmpeg` set to true in `container/context.yaml`.

> [!WARNING]
> **Supported input codecs**: The in-tree ffmpeg is built with a narrow decoder allowlist (VP8/VP9 video in mp4/webm/mkv) — it carries only the media formats we build and use, not ffmpeg's full default set (see `container/templates/wheel_builder.Dockerfile`). Other codecs, including H.264 and H.265, are intentionally **not** decodable in software; decoding them would require enabling the NVDEC hardware decoders (`h264_cuvid`/`hevc_cuvid`), which is not wired up today.

## Image decoding options

### JPEG decoding
- The frontend uses libjpeg-turbo's TurboJPEG API for JPEG inputs by default. Set `DYN_MM_ENABLE_LIBJPEG=0` on the frontend process to use `image::ImageReader` instead.
- Dynamo backend runtime images include `libturbojpeg`; custom images must provide `libturbojpeg.so.0` or Dynamo logs a one-time warning and falls back to `image::ImageReader`. The standalone frontend image is excluded as described above. Non-JPEG inputs and JPEGs that TurboJPEG cannot decode also fall back to `image::ImageReader`.

### Limits (not overridable at runtime via `media_io_kwargs`)
- **limits.max_image_width** (uint32, > 0): If the image width exceeds this value, abort the decoding.
- **limits.max_image_height** (uint32, > 0): If the image height exceeds this value, abort the decoding.
- **limits.max_alloc** (uint64, > 0): Maximum allowed total allocation (RAM) of the decoder in bytes

## Video decoding options
### Sampling
There are two ways to configure video sampling: either with a fixed number of frames, or with FPS-based sampling. Sampled frames are distributed uniformly in both cases.

- **num_frames** (uint32, > 0): Attempt to decode exactly this number of frames from the input video.
- **fps** (float32, > 0) and optionally **max_frames** (uint32, > 0): Attempt to decode at a given framerate, with a potential cap on the number of decoded frames.

### Others
- **strict** (bool): if strict mode is enabled, any failure to decode a requested frame will abort the whole video decoding and error out. When strict mode is disabled, it is possible that the decoding of some requested frame fails, and the resulting set of decoded frames might container fewer frames than expected.

### Limits (not overridable at runtime via `media_io_kwargs`)
- **limits.max_alloc** (usize, > 0): If the total number of bytes in the decoded frames would exceed this value, abort the decoding.


## Runtime media decoding options (`media_io_kwargs`)

Parameters of the decoders, can also be set at runtime via an extension to the OpenAI chat completions API. Limits defined in the MDC such as maximum image size, maximum RAM allocation, cannot be overridden at runtime.

This can be used for example to set the video sampling strategy for a request, that differs from the default one registered in the MDC:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": ...,
    "messages": ...,
    "media_io_kwargs": {
        "video": {
            "fps": 1.0,
            "max_frames": 16
        }
    }
  }'
```

## TODOs

### Modalities

- [x] Image decoding
- [x] Video decoding
- [ ] Audio decoding

### Performance

- [x] Image SW decoding
- [ ] Video HW decoding (NVDEC)
- [ ] JPEG HW decoding (nvJPEG)
- [x] Sparse video sampling (seek-forward)
- [ ] Memory slab pre-allocation/registration

### Memory management
- [ ] Memory spilling to lower storage tiers
- [ ] Early-free memory on client notifications

### Misc
- [ ] Observability on performance, memory usage and input distributions
- [x] Per-request decoding options

## Shared decoder fork validation

The experimental `dynamo-llm/shared-media` feature uses the extracted decoder
in `dynamo-multimodal`. It is off by default until the shared crate is
published. The original implementation remains available for before/after
comparison. Preserve this ordering for upstream rollout: publish the crate,
replace the fork revision with that released version, enable the new path,
then remove the original decoder code and this temporary feature.

The existing `Decoder` interface, serialized model/runtime options, environment
overrides, async CPU offloading, media content hashes and NIXL storage remain
in Dynamo. The shared crate returns owned pixel vectors and metadata. Dynamo
converts these to the same `SystemStorage` and metadata as before, with the
same registration and lifetime management. Its existing `tokio_rayon` bridge
provides request concurrency; leave the shared crate's pool unarmed.

### Build and install

```bash
# Rust image path, no FFmpeg required
cargo test -p dynamo-llm --no-default-features --features shared-media \
  preprocessor::media:: --lib
# Image and video paths, with the existing FFmpeg 9 development environment
cargo test -p dynamo-llm --no-default-features --features shared-media,media-ffmpeg \
  preprocessor::media:: --lib
# Existing Python extension, editable installation
cd lib/bindings/python
maturin develop --uv --features shared-media
# Add media-ffmpeg to the feature list for video.
```

The bindings also forward `shared-media` for `maturin build`. No separate
Python package or Python API is introduced. A clean source build additionally
needs CMake and a C/C++ toolchain for the shared crate's existing TurboJPEG
dependency; NASM enables x86 SIMD. The Dynamo-compatible JPEG path still loads
the optional system TurboJPEG library and preserves fallback when absent.
The shared crate includes full native JPEG notices with its Apache attribution.

Video still links shared FFmpeg from the existing restricted native build.
Keep the `_dynamo` SONAMEs, pkg-config aliases, auditwheel exclusions, loader
configuration and FFmpeg source artifacts. Image-only wheels do not need
FFmpeg. The FFmpeg-enabled container wheel remains dependent on its image's
external FFmpeg libraries, as before.

### Container matrix

`--build-arg ENABLE_SHARED_MEDIA=true` forwards the feature through the source
wheel builder. Its default is false. The existing `ENABLE_MEDIA_FFMPEG` flag
continues selecting video independently. The table records renderer targets;
it does not assert that every architecture has been built in this fork.

| Framework/device | Architectures | Targets and migration treatment | Rust video default |
| --- | --- | --- | --- |
| vLLM CUDA 13.0 | amd64, arm64 | runtime, dev, local-dev, wheel_builder: opt-in feature forwarding; existing native handling retained | on |
| vLLM XPU | amd64 | same source targets and feature forwarding | on |
| vLLM CPU | amd64, arm64 accepted by renderer | same source targets; these unshipped builds retain their existing compliance exclusions | on |
| SGLang CUDA 13.0 | amd64, arm64 | runtime, dev, local-dev, wheel_builder: opt-in feature forwarding | on |
| SGLang XPU | amd64 | same targets; shared image decoding only; existing template excludes Rust FFmpeg | off |
| TensorRT-LLM CUDA 13.1 | amd64, arm64 | runtime, dev, local-dev, wheel_builder: opt-in feature forwarding | off |
| Dynamo CUDA 13.0 | amd64, arm64 | runtime, dev, local-dev, wheel_builder: opt-in feature forwarding | off |
| Dynamo frontend | amd64, arm64 | consumes existing published wheels; no new source feature or media support | unchanged |
| Dynamo planner | amd64, arm64 | no decoder build changes | unchanged |
| Triton CUDA 13.4 | amd64, arm64 | runtime consumes prebuilt wheels; no source wheel_builder/dev targets | unchanged |
| Base images, EPP, AWS/EFA derivatives | existing supported architectures | base/EPP unchanged; EFA derivatives inherit their runtime's feature and compliance handling | inherited |

Generated Dockerfiles must come from `container/render.py`. Runtime FFmpeg
CLI copies also support backend video encoding, so do not remove them when
Rust video is disabled. The existing NOTICES/SBOM generation, license gates,
source harvesting and codec scans remain release checks. The fork's git
dependency notices are harvested alongside registry crate notices; switching
to a published crate uses the existing registry path.

### Parity and performance

The feature-enabled Rust tests compare original and extracted image/video
decoders through Dynamo's actual storage conversion, metadata serialization,
hashing and concurrent async interface. Run the existing frontend-decoding
integration suite on the supported backend/GPU environments as well:
`tests/mm_router/test_router_rust_mm_frontend_decode_e2e.py`.

Use the existing `image_decode` and `video_decode` Criterion benchmarks for
before/after runs on the same machine, once without and once with
`shared-media`. Enable `RUN_IMAGE_DECODE_SWEEP=1` and
`RUN_VIDEO_DECODE_SWEEP=1` for the 1/8/32-concurrency cases. Keep JPEG backend,
video sampling, native libraries, CPU affinity and input fixtures identical.
Record batch latency and throughput separately from per-request latency;
do not infer p95/p99 request latency from Criterion's batch averages.

### Validation record (2026-10-08)

Fork implementation based on Dynamo
`16b2ae50032edbdde70a9e03a7e7056fac3b7c7d`, consuming frontend-crates
`984199d41cfbf56e4083707a386501344a2b7f0c`. Validation used Linux x86_64,
Rust 1.96.1, the restricted FFmpeg 9.0.1/libvpx 1.14.1 build from the shared
crate's `build-ffmpeg.sh`, and system TurboJPEG 2.1.5. The checkout and Cargo
home were isolated from the developer's existing installations.

| Check | Result |
| --- | --- |
| Shared crate, all features | 69 tests passed, including image/video fixtures and armed/unarmed execution; Clippy with warnings denied passed |
| Shared crate, `--no-default-features --features media-decode` | Passed without FFmpeg environment variables; dependency tree excludes FFmpeg, Dynamo runtime and NIXL |
| Packaged crate | `cargo package --locked --offline` and all-feature tests from the extracted package passed; fixture and license files are included |
| Dynamo media tests, `shared-media,media-ffmpeg`, no default features | 70 passed, including concurrent differential pixel/metadata/hash/ownership checks |
| Rust license checks | `cargo deny --all-features check bans licenses` passed in both repositories |
| Python bindings | Development-profile abi3 wheel built with shared image/video support; wheel and editable installations imported successfully in separate fresh Python 3.12 environments; existing image/video configuration APIs passed smoke checks |
| Wheel compliance | Embedded SBOM contains 917 components including the shared crate; all pass Dynamo's license policy; NOTICES injection preserves full shared/native JPEG attribution and its RECORD digest |
| FFmpeg linking/codecs | Extension links external `_dynamo` FFmpeg shared libraries; native build's positive/negative codec guards passed; runtime-layout filesystem scan found 36 allowed files and zero violations |
| Compliance tests | 50 passed across license-text, Cargo SBOM and codec-scan suites |
| Container templates | 66 applicable framework/device/architecture/target combinations rendered |

The wheel smoke check used `maturin build --profile dev --features
shared-media,media-ffmpeg --auditwheel skip`; it validates local installation
with external native libraries, not a repaired manylinux release wheel. The
native filesystem scan used the existing runtime's `/usr/local` layout and
shipped-file selection, not a complete backend image. Python smoke checks do
not exercise NIXL transfers. BuildKit's CPU wheel-builder Dockerfile check
reports the same three existing warnings as the baseline (two undefined
`LD_LIBRARY_PATH` references and an empty continuation), so it is not recorded
as a passing build.

Still required before enabling this upstream: full runtime/dev container builds
for the affected architectures and feature combinations, their complete
NOTICES/SBOM/source bundles and release license/codec gates, repaired release
wheels, and `test_router_rust_mm_frontend_decode_e2e.py` on supported GPU
backends with real NIXL registration and cleanup. Registry publication and
consumption of that released crate also remain pending. The fork deliberately
retains the original implementation until those rollout steps are complete.

### Concurrent decode comparison

Measured on an AMD Ryzen 9 7900X (12 cores / 24 logical CPUs), Linux
6.14.0-37-generic, using the native versions above and the existing Criterion
0.5.1 image/video benchmarks. The original decoder is the retained legacy
path with `shared-media` disabled. The shared path is the implementation in
`2b73f6faf09895add819096a666b7233dd5a8e78`, pinned to the shared crate
revision above. Both use the optimized bench profile (optimization level 3,
one codegen unit, thin LTO). All migration compilation finished before timing.

The image workload is a batch of 100 generated 3840x2160 JPEGs, tested with
both ImageReader and system TurboJPEG at 1/8/32 Rayon threads. The video
workload samples 30 frames from `240p_100.mp4` (VP9, 320x240), with one video
per concurrent thread. Encoded-input cloning is outside the timed section in
both existing harnesses. This measures synchronous decoding and output storage
construction, not request fetching, async queueing, hashing or NIXL transfer.

Each case uses 10 samples, a one-second warmup and a three-second target
measurement time; Criterion increases that time when ten batches require
longer. The original sweep ran before the shared sweep on the same workstation,
without CPU affinity or exclusive host reservation. Treat the numbers as a
local regression check, not production capacity or request p95/p99 latency.

Reproduce with the existing harnesses and the same native library paths:

```bash
export RUN_IMAGE_DECODE_SWEEP=1 RUN_VIDEO_DECODE_SWEEP=1
export DYNAMO_REQUIRE_LIBJPEG_TURBO_TEST=1
for variant in original shared; do
  features=media-ffmpeg
  if [ "$variant" = shared ]; then features=shared-media,media-ffmpeg; fi
  cargo +1.96.1 bench -p dynamo-llm --no-default-features --locked \
    --features "$features" --bench image_decode --bench video_decode -- \
    'batch_100|video_decode_concurrent' --warm-up-time 1 \
    --measurement-time 3 --sample-size 10 --save-baseline "$variant"
done
```

For the recorded run, both executable variants were built first and then run
directly with those Criterion options, so compilation did not overlap timing.

Mean batch latency and decoded items per second (images or videos):

| Decoder | Threads | Batch size | Original ms | Shared ms | Latency change | Original items/s | Shared items/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JPEG ImageReader | 1 | 100 | 8935.71 | 9157.98 | +2.5% | 11.19 | 10.92 |
| JPEG ImageReader | 8 | 100 | 1359.18 | 1369.14 | +0.7% | 73.57 | 73.04 |
| JPEG ImageReader | 32 | 100 | 879.02 | 888.31 | +1.1% | 113.76 | 112.57 |
| JPEG TurboJPEG | 1 | 100 | 8035.25 | 7747.92 | -3.6% | 12.45 | 12.91 |
| JPEG TurboJPEG | 8 | 100 | 1167.24 | 1158.44 | -0.8% | 85.67 | 86.32 |
| JPEG TurboJPEG | 32 | 100 | 683.02 | 684.87 | +0.3% | 146.41 | 146.01 |
| VP9 FFmpeg | 1 | 1 | 66.47 | 65.45 | -1.5% | 15.04 | 15.28 |
| VP9 FFmpeg | 8 | 8 | 75.25 | 76.05 | +1.1% | 106.31 | 105.20 |
| VP9 FFmpeg | 32 | 32 | 162.48 | 162.65 | +0.1% | 196.94 | 196.75 |

The largest observed regression is +2.5% latency for serial ImageReader,
with throughput decreasing from 11.19 to 10.92 images/s. The 8/32-thread
cases show changes between -0.8% and +1.1%. These measurements do not establish
the cause of small differences on a shared host; repeat on the deployment
hardware before deciding a performance acceptance threshold. No speedup is
required or claimed for this extraction.

The [comparison CSV](../../../benches/shared_media_comparison.csv) contains
unrounded means, 95% confidence intervals, deltas and throughput. The
[Criterion samples and estimates](../../../benches/shared_media_samples.json)
retain all ten measurements per case for both implementations.
