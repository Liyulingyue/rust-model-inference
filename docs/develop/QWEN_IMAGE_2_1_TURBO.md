# Qwen-Image-2.1-Turbo

## Entry and checkpoint contract

Turbo reuses the Qwen-Image-2.1 DiT, Qwen3-VL conditioning and RGBA VAE.
Pass **`--qwen-image-turbo`** with the Turbo checkpoint: the GGUF architecture
(`qwen_image`) and tensor names do not distinguish the base and Turbo weights.
The flag selects the checkpoint's fixed eight-step schedule and defaults CFG to 1.
Explicit step counts other than eight are rejected. Ordinary Qwen-Image-2.1
keeps its existing schedule and defaults. CPU execution is the default; optional
hardware Vulkan projection offload uses `--features vulkan` and `--gpu`.

```bash
RMI_SCALAR=1 target/release-fast/rust-model-inference \
  --qwen-image-turbo \
  --model models/Qwen-Image-2.1-Turbo-GGUF/qwen-image-2.1-turbo-Q8_0.gguf \
  --text-encoder models/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-Q8_0.gguf \
  --vae models/Qwen-Image-2.1-Turbo/vae/diffusion_pytorch_model.safetensors \
  --prompt 'A blue cat. 蓝猫 🐈' --width 256 --height 256 \
  --seed 42 --threads 8 --out target/turbo.png
```

Editing uses the existing `--image`, repeated `--reference` and `--mmproj`
arguments. The VAE needs its sibling `config.json`. Dimensions must be multiples
of 32; see [the shared pipeline report](QWEN_IMAGE_2_1.md) for component and
reference-image limits. Prefix KV caching is not implemented by this CPU path.

Official source: [Qwen/Qwen-Image-2.1-Turbo](https://modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo/files),
revision `4954d69c119759f085aa36bd5e66e27647cdee60`. The official README states
that it uses the same 7B architecture, eight denoising steps and default CFG 1.
The checked-in `model_index.json` and scheduler config are copied from that
revision. Sigmas (with appended terminal zero) are:

```text
1, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0
```

The scheduler uses shift=1, no dynamic shifting, no terminal stretching and
non-stochastic Euler updates. Diffusers source was inspected at
`784ae091a80d568f089842d766cac34637eef86e` (`pipeline_qwenimage21.py` and
`scheduling_flow_match_euler_discrete.py`).

The Q8_0 GGUF header has 265 tensors: 196 Q8_0, 65 F32 and 4 BF16. Its complete
names/shapes match the existing 32-block DiT contract; metadata contains
`general.architecture=qwen_image` and `general.quantization_version=2`.

## Validation method

The independent component Oracle is pinned stable-diffusion.cpp
`2f886889e6e8b78738d6b87f7191f6018557c551` / ggml
`4bf5f6000653b7881d00963cd6ddb665ccd62a8d`, using the official Turbo schedule
loaded separately from the JSON fixture. No external acceleration library is
used. The initial results below check the released GGUF CPU path; the Vulkan
section compares GPU projections with the same CPU Oracle. Neither establishes
native BF16 Diffusers parity or GGUF-conversion quality.

```bash
CARGO_TARGET_DIR=target/qi21-turbo/trace-build \
  cargo build --profile release-fast --features parity-trace --bin rust-model-inference
RMI_QI21_ORACLE_ROOT="$PWD/target/qi21-turbo/oracle" \
  tools/oracle/qwen_image_2_1/build_oracle.sh
.venv/bin/python tools/oracle/qwen_image_2_1/pipeline_parity.py \
  --turbo --relative-l2 0.01 --cases context text \
  --dit models/Qwen-Image-2.1-Turbo-GGUF/qwen-image-2.1-turbo-Q8_0.gguf \
  --text models/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-Q8_0.gguf \
  --mmproj models/Qwen3-VL-8B-Instruct-GGUF/mmproj-F16.gguf \
  --vae models/Qwen-Image-2.1-Turbo/vae/diffusion_pytorch_model.safetensors \
  --rust-bin target/qi21-turbo/trace-build/release-fast/rust-model-inference \
  --oracle-bin target/qi21-turbo/oracle/src/build-scalar/bin/qwen-image-2-1-components \
  --out target/qi21-turbo/validation
```

The numerical bound is declared before running: relative L2 <= 0.01 **and**
max absolute error <= `1e-5 + 0.01 * max(abs(reference))` at each computed
checkpoint. RGBA8 bytes allow at most four levels. Token IDs, image slots,
schedule, initial noise, timesteps, checkpoint names/order/count and shapes
remain exact. Derived conditioning and iterative latents use numerical checks.
See [the runner documentation](../../tools/oracle/qwen_image_2_1/README.md).

## Engineering checks

- Local focused integration tests: 11 passed (schedule, CLI defaults/rejection,
  tensor contract and RGBA validation).
- SSH Linux focused integration tests: 11 passed on AMD EPYC 9334.
- Python comparator regressions: 6 passed, including rejection of large errors,
  nonfinite values, structural mismatches and invalid tolerances.
- Full local library suite before the Vulkan follow-up: 1,236 passed, 23 failed,
  80 ignored. The original
  HEAD (`973210b6de572b8226cd576da7ace17fb22d8ad5`) has 1,235 passed and exactly
  the same 23 failures / 80 ignored; this is not a passing full-suite claim.

The Vulkan follow-up additionally passed 12 focused integration tests with and
without the feature. Local Qwen3 session and dense-execution regressions passed
12 tests; the hardware BF16 tiling and raw-hidden/F32-KV projection checks passed
on both MoltenVK and RADV. The CPU-scope regression passed without initializing
hardware. The SSH host also passed the 12 integration tests and these three
projection/policy checks. The full library suite was not repeated for this
follow-up.

## Executed real-weight results

All four local components were downloaded from ModelScope and SHA256-verified.
They reside under the user-selected shared `models/` directory, linked into this
worktree; weights and local environment links are excluded from Git.

| Component | ModelScope revision | Bytes | SHA256 |
| --- | --- | ---: | --- |
| unsloth Turbo Q8_0 DiT | `0331037889cff0eb06d6f9ee97db856b2fb9228d` | 7,640,860,480 | `399d2ddc3636f36456e2f5bfa57187da4e38721bc046430317f42a9eab5b1712` |
| unsloth Qwen3-VL-8B Q8_0 text | `4e2066f88dd47d2862aaf0a83b60cbf1eb784b4a` | 8,709,520,224 | `cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7` |
| unsloth Qwen3-VL F16 mmproj | `4e2066f88dd47d2862aaf0a83b60cbf1eb784b4a` | 1,159,030,336 | `d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4` |
| Official Turbo F32 VAE | `4954d69c119759f085aa36bd5e66e27647cdee60` | 675,508,656 | `71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c` |

On Apple M3 Max (16 CPU cores, 64 GiB), `--cases context` used fixed two-token
F32 conditioning, seed 42, 64×32 output, eight Euler steps and CFG 1. It passed
2,815 checkpoints / 105,386,513 F32 values with measured max absolute error and
relative L2 both **0**; all 8,192 RGBA8 bytes matched. An ordinary build repeated
the same scalar CLI and produced identical pixels (8.075 seconds including
loading, compared with 274.974 seconds with full tracing; these are not prompt
or large-image benchmarks). This case does not exercise the tokenizer or text
encoder. The runner command above can use `--cases context` for this focused case.

The full text case (`A blue cat. 蓝猫 🐈`, same dimensions/seed/eight steps/CFG 1)
also passed: 39 text checkpoints, 2,730 sampler checkpoints and 85 VAE checkpoints,
totalling 2,854 checkpoints / 274,274,866 F32 values. Measured max absolute error
and relative L2 were both **0**, and all 8,192 RGBA8 bytes matched. This includes
the actual Q8_0 text encoder and exact tokenizer IDs; it has no reference images.
The traced Rust CLI took 968.666 seconds, including trace I/O. A normal scalar
build repeated the prompt through the CLI in 32.99 seconds with identical PNG
pixels; this tiny image is not a large-image performance measurement.

The independent Diffusers direct Euler formula `x + (sigma_next-sigma)*velocity`
was also checked on every recorded step. Its difference from the shared sd.cpp
round-trip denoiser evaluation was at most `1.1920928955078125e-7` absolute and
`3.17495123369983e-8` relative L2; this lossy operation was not forced to bitwise
identity.

A normal SIMD CLI additionally generated a 256×256 orange-cat image with the
actual text encoder, Turbo DiT and VAE, seed 42 and the default eight steps / CFG
1. It completed in 142.99 seconds with maximum RSS 9,374,220,288 bytes. The
measurement ran alongside trace validation, so it is not an isolated performance
benchmark. The PNG was inspected visually. Successful generation does not by
itself establish the SIMD numerical bound.

A later 512×512 full text-to-PNG run on Apple M3 Max used the long prompt
recorded in `normal_cli_image_512` in the result JSON: an orange tabby cat on a
sunlit windowsill, a cup of tea, an open book and city rooftops outside. With the
default SIMD path, eight threads, eight steps, CFG 1 and seed 42, it completed
in 1,348.311 seconds with maximum RSS 12,928,778,240 bytes. The cat, cup, book
and window scene were present on visual inspection; steam was not clearly
visible. PNG SHA256:
`d597156d37c8f17d97dd479a927746f14474fb926bcb4ba719651292f6f885fa`.
The initial part ran alongside an optional scalar comparison, and system
memory compression/swap activity was observed. This is a functional result,
not an isolated benchmark or a 512×512 Oracle comparison. The optional scalar
run was stopped after two steps / 1,016.227 seconds; it produced no PNG.

The default SIMD path was checked separately against the same scalar Oracle and
**failed** the declared 1% bounds. Its first failure was
`qwen.debug.attn.seg1`, occurrence 8: max absolute error `0.08545827865600586`
(reference peak `7.57888126373291`), relative L2 `0.0075745305308977115`. The
peak-relative criterion failed here; the final latent also exceeded the L2
criterion (`0.012188912160074497`, max absolute error `0.023451805114746094`).
The final PNG differed by up to 16 RGBA8 levels. The bound was not loosened.
Use **`RMI_SCALAR=1`** for the validated numerical path; successful default SIMD
image generation is a separate functional result. Generic SIMD kernel changes
are not included in this model/schedule adaptation. To reproduce this failure,
repeat the focused runner with `--cases context --optimized --relative-l2 0.01`
and the same scalar `--oracle-bin`, using a fresh output directory.

A second comparison used the same pinned Oracle with native CPU kernels
(`GGML_NATIVE=ON`, `-ffp-contract=off`, external acceleration disabled). It also
failed the unchanged 1% bounds: first failure `qwen.debug.attn.seg1`, occurrence
7, max absolute error `0.08275669813156128` (reference peak `7.771087646484375`),
relative L2 `0.0066251952738719515`. Final latent relative L2 was
`0.012600705953349844`, max absolute error `0.024372607469558716`; final PNG
maximum difference was seven levels, above the allowed four. Changing the
reference build does not establish passing default SIMD parity.

The investigated rounding differences include Q8 activation ties: a real-operator
probe with scale 1 and inputs `0.5, 1.5, 2.5, -0.5, -1.5, -2.5` produced
`0, 2, 2, 0, -2, -2` in NEON and `1, 2, 3, -1, -2, -3` in scalar mode.
The pinned ggml ARM and generic kernels use these respective rules as well.
The first block's attention input differs by only `2.86102294921875e-6` absolute,
but quantized projections amplify the difference. This is a diagnostic result,
not a justification for loosening the end-to-end bound.

The SSH host passed the same 11 focused tests with the final schedule fixture.
The actual DiT, text encoder and VAE reached that host and passed full SHA256
checks. A normal scalar build then completed the real text-to-image CLI with
`A blue cat. 蓝猫 🐈`, 64×32, seed 42, eight threads and the default eight steps /
CFG 1 in 51.335 seconds, maximum RSS 8,181,514,240 bytes. This included component
loading and ran alongside build activity; it is not an isolated benchmark.
Weights reside under `/root/rmi-qi21-turbo-d289/models`, linked from the test
checkout to avoid its smaller filesystem. No external acceleration library or
global dependency was installed there.

The first Linux Oracle build still selected baseline SSE2 SiLU/softmax kernels,
despite `GGML_SSE42=OFF` and all AVX/FMA options being disabled. It failed the
text bound at block 35 (relative L2 `0.017662001171485316`). The scalar build
script now also undefines `__SSE2__`; the pinned reference source and numerical
bounds are unchanged. A linked 256-value operator probe went from 146 differing
exponentials / 64 differing SiLU values to zero; its softmax sum error went from
`-8.8178785517811775e-7` to zero. Against the existing Rust trace, all 39 text
checkpoints / 5,079,073 F32 values then had zero measured error. The script uses
standard `grep` for compiler/target detection so it also runs on hosts without
`rg`.

After all eight Oracle tool/configuration inputs were synchronized and
SHA256-verified, the complete Linux text case passed 2,854 checkpoints /
274,274,866 F32 values. Text and sampling measured zero error. VAE maximum
absolute error was `1.1920928955078125e-7`, maximum relative L2
`5.9441857491399164e-8`; all 8,192 RGBA8 bytes matched. The completed Rust trace
was reused after correcting the remote harness copy, while the independent
Oracle computed its own text conditioning, eight-step sampler and VAE outputs.
The normal build repeated the same PNG pixels in 69.38 seconds. Both remote
production PNGs have SHA256
`195536bb9cf2519fa77fd5fb3495eefb802fe1baafa930f161b87770f645d674`.

A later 512×512 visual smoke with the long orange-cat/windowsill/tea/book prompt
and the default SIMD path on AMD EPYC 9334 failed at its first step with
`Non-finite Qwen-Image-2.1 latent at step 1` (exit 1 after 294.299 seconds).
No PNG was produced. This is a failed large-image SIMD run; the passing remote
64×32 scalar comparison above does not cover that configuration.

Linux versus macOS CPU output pixels differ by up to 18 levels. A successful CLI
run does not imply cross-platform bitwise identity or passing cross-platform
numerical parity; comparisons with the same-platform Oracle are separate.

The exact result summary is kept in
[`turbo_validation.json`](../../tools/oracle/qwen_image_2_1/turbo_validation.json).
Turbo image-editing end-to-end parity, other GGUF quantizations and native
Diffusers DiT safetensors are outside this validation scope.

## Vulkan projection offload

Build with `--features vulkan` and pass `--gpu`. The text encoder's Q/K/V,
output and FFN projections, the DiT's Q8_0/BF16 projections and the VAE's
F16 convolution matrices use the existing Rust/Vulkan kernels. Attention,
normalization, RoPE, activations, F32 KV and the small F32 DiT modulation
projection remain on the host. VAE im2col, input rounding and layout conversion
also remain on the host. Independent attention query rows run in the existing
thread pool; each row retains its original dot/softmax reduction order.
Weights stay cached for the render. An unavailable,
software or failed Vulkan device returns an error; CPU fallback is not counted
as GPU success.

Use **`RMI_SCALAR=1 --gpu`** for the numerical path exercised below. The scalar
flag controls host operations; these explicit Vulkan projections still execute
on the GPU. It does not establish numerical alignment for default host SIMD.
No external acceleration library is used.

```bash
cargo build --profile release-fast --features vulkan --bin rust-model-inference
RMI_SCALAR=1 target/release-fast/rust-model-inference \
  --qwen-image-turbo --gpu \
  --model models/Qwen-Image-2.1-Turbo-GGUF/qwen-image-2.1-turbo-Q8_0.gguf \
  --text-encoder models/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-Q8_0.gguf \
  --vae models/Qwen-Image-2.1-Turbo/vae/diffusion_pytorch_model.safetensors \
  --prompt 'A blue cat. 蓝猫 🐈' --width 512 --height 512 \
  --seed 42 --threads 8 --out target/turbo-gpu.png
```

For component parity, build with `--features vulkan,parity-trace` and append
`--gpu` to the runner command above. The runner keeps the pinned CPU Oracle
independent and retains the same 1% checkpoint and four-level RGBA8 bounds.
`RMI_COMPUTE_TRACE=1 RUST_GPU_DISPATCH_TRACE=1` records real device submissions,
weight uploads and allocation counters. An explicit CPU compute scope skips
offload even after an earlier GPU request in the same process.

On Apple M3 Max / MoltenVK, the fixed-context 64×32 case completed all eight
steps and VAE decoding. All 2,815 checkpoints / 105,386,513 F32 values passed
the unchanged bounds. Sampling measured zero error; VAE maximum absolute error
was `0.11763381958007812`, maximum relative L2 `0.0009098223636591213`.
All 8,192 RGBA8 bytes differed by at most one level. The runtime recorded 1,982
Vulkan submissions and 246 static uploads, with peak tracked device allocations
of 8,097,642,016 bytes. A normal build repeated exactly the same GPU PNG pixels
in 8.363 seconds including loading; the traced CLI took 273.278 seconds.
These tiny-image timings include host execution and are not a speedup benchmark.
The parallel host-attention implementation repeated all these comparisons with
the same errors and pixels. A normal scalar CPU build also reproduced the
pre-change CPU PNG pixels in 15.974 seconds; that run overlapped GPU/build work.

On the supplied SSH host, AMD Radeon Graphics (RADV NAVI31, 51,522,830,336 bytes
reported VRAM) passed the complete Q8_0 text-to-image case with the same scalar
CPU Oracle. All 2,854 checkpoints / 274,274,866 F32 values passed. Text and
sampling measured zero error; VAE maximum absolute error was
`0.20376968383789062`, maximum relative L2 `0.0009691028640275452`.
The 8,192 RGBA8 bytes differed by at most one level. These comparisons reuse
the already completed independent Linux scalar Oracle outputs for the same
fixed inputs, including its own text conditioning, eight-step sampler and VAE.
The GPU CLI was rerun after the VAE Q/K/V capacity correction.

The final parallel-query traced CLI took 390.549 seconds. An ordinary
Vulkan build repeated the same GPU PNG pixels in 26.7 seconds. It recorded 2,234 Vulkan submissions,
498 cumulative static uploads (15,459,581,632 bytes), and peak tracked device
allocations of 8,097,642,016 bytes. Upload bytes are cumulative across components,
not simultaneous VRAM usage. The final local/remote source audit matched all
599 runtime/shader/build files. Timings include loading and host operations;
the final trace and 512×512 visual run overlap. These earlier timings do not
form a controlled CPU/GPU comparison.

The ordinary Vulkan build also completed the actual long-prompt text-to-PNG
pipeline at 512×512 on RADV: eight steps, CFG 1, seed 42, eight threads and
`RMI_SCALAR=1 --gpu`. It took 645.291 seconds, with maximum process RSS
13,963,321,344 bytes (measured for this child process with `wait4`). It recorded
74,437 hardware submissions, 498 cumulative static uploads (15,459,581,632
bytes), and peak tracked device allocations of 8,097,642,016 bytes. The latter
is the runtime's allocation counter, not a measurement of peak physical VRAM.
The PNG was visually inspected: the orange tabby, wooden windowsill, cup,
open book and blurred city background are present; steam is not clearly
visible. PNG SHA256:
`d41fa3587ab5f55e06ca9c7fb82983d6447eb06521d969ce8568f17ff428ce6e`.
The run overlapped the final 64×32 parity check, so this is a functional image
result, not an isolated benchmark. No 512×512 Oracle alignment is claimed.

## CPU and Vulkan CLI timing

On 2026-10-23, both paths used the same `release` binary with the `vulkan`
feature on the supplied EPYC 9334 / RADV NAVI31 host. Both used
`RMI_SCALAR=1`, eight threads pinned to physical CPUs 39, 44, 46, 51–53, 59
and 61 on NUMA node 1, the long orange-cat prompt, seed 42, eight steps and
CFG 1. Each image started a fresh CLI process; the wall clock includes model
loading, device uploads, text conditioning, sampling, VAE decoding and PNG
saving. OS file caches and shader disk caches were warmed with the 64×32 case
first; those warmup PNGs matched the previously validated build exactly.

| Output | CPU seconds | Vulkan seconds | CPU/GPU mean ratio |
| --- | ---: | ---: | ---: |
| 256×256, two runs each | 544.149, 553.001 (mean 548.575) | 135.916, 135.399 (mean 135.658) | 4.044× |

The 256×256 sequence was CPU, Vulkan, Vulkan, CPU. Before each render the
runner waited for known compilation/inference jobs and the selected GPU to
be idle, then sampled processes and device state every five seconds. No known
competing workload was observed during these four accepted samples. The host
is shared, so this is not a claim of physical machine isolation. An earlier
attempt that overlapped a Sage benchmark was stopped and excluded.

A subsequent 512×512 CPU attempt overlapped sustained compilation and
inference work. It was manually terminated before PNG output and excluded;
the corresponding same-release GPU measurement was not started. No comparable
512×512 speed ratio is available. The earlier functional GPU timing used a
different build and concurrency conditions.

Both CPU repeats produced identical PNGs, as did both Vulkan repeats. The
CPU/Vulkan RGBA8 maximum difference was one level (mean absolute difference
0.003414154052734375). This is a final-image CPU/Vulkan check, rather than a
new per-checkpoint Oracle comparison. The default host SIMD path is outside
this comparison because it has not passed the declared numerical bound.

The exact commands, affinity, binary/source hashes, per-run timings, process
RSS and concurrency flags are in `cpu_gpu_cli_benchmark` in the result JSON.
To reproduce, build with `cargo build --release --features vulkan --bin
rust-model-inference`, use the recorded arguments and affinity, and toggle
only `--gpu` between CPU and Vulkan runs (with separate PNG output paths).
