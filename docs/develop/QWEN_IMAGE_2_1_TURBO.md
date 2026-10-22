# Qwen-Image-2.1-Turbo

## Entry and checkpoint contract

Turbo reuses the Qwen-Image-2.1 DiT, Qwen3-VL conditioning and RGBA VAE.
Pass **`--qwen-image-turbo`** with the Turbo checkpoint: the GGUF architecture
(`qwen_image`) and tensor names do not distinguish the base and Turbo weights.
The flag selects the checkpoint's fixed eight-step schedule and defaults CFG to 1.
Explicit step counts other than eight are rejected. Ordinary Qwen-Image-2.1
keeps its existing schedule and defaults. CPU execution only.

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
used. This checks the released GGUF CPU path; it is not a BF16 Diffusers/GPU
comparison or a GGUF-conversion quality measurement.

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
- Full local library suite: 1,236 passed, 23 failed, 80 ignored. The original
  HEAD (`973210b6de572b8226cd576da7ace17fb22d8ad5`) has 1,235 passed and exactly
  the same 23 failures / 80 ignored; this is not a passing full-suite claim.

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

The SSH host passed the same 11 focused tests with the final schedule fixture.
Full remote inference is not claimed: repeated upload attempts stalled/timed out
before a complete hash-verified DiT/VAE pair reached that host. No external
acceleration library or global dependency was installed there.

The exact result summary is kept in
[`turbo_validation.json`](../../tools/oracle/qwen_image_2_1/turbo_validation.json).
Turbo image-editing end-to-end parity, other GGUF quantizations, native Diffusers
DiT safetensors and GPU execution are outside this validation scope.
