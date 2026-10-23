# FLUX 3 Image Klein adaptation and Oracle

This adapter targets the **distilled BF16** release only. It registers `flux3`
explicitly, uses the Qwen3-VL-4B hidden layers 4/8/12/16/20/24/28/32, and owns
its three-stream DiT and flow-map sampler. Base, FP8 and NVFP4 are rejected.

The runtime uses existing Rust CPU/Vulkan operators with F32 activations and
accumulation, BF16 DiT/text weights and F16 VAE convolutions. Text features,
reference features and initial noise are rounded to BF16. This is a numerical
implementation; it does not promise the CUDA BF16 activation bits.

## Fixed inputs

- Hub: `black-forest-labs/flux-3-image-klein`
- Revision: `337337621b8526e120f2275d250242f4a6bedd75`
- Official source: <https://github.com/black-forest-labs/flux-3-image-klein>
- Oracle commit: `bfc8889a4497f846416f253a3972a1941ac698b9`
- Source file hashes are enforced by `reference.py`.
- Weight hashes and complete tensor/payload contracts are enforced by
  `verify_conversion.py` (359 DiT, 398 text, 250 inference VAE tensors).
- Respect the model's license and gated access requirements.

Use the pinned official code defaults: **alpha=16, text max=4096**, right padding
to a multiple of 80, 16 sampling steps by default. The Hub's `config.json` lists
alpha=6.93 and max_tokens=8192; those values are intentionally not silently mixed
with the pinned official pipeline.

## Convert and run

Run Python modules from the repository root, using a repository-local venv.
`MODEL` below is the directory containing `model.safetensors`, `ae.safetensors`
and `text_encoder/`. For example, use `models/flux-3-image-klein` in a checkout with enough disk space.

```sh
python -m tools.converter.flux3.convert_flux3 "$MODEL" "$MODEL/gguf"
python -m tools.oracle.flux3.verify_conversion "$MODEL" "$MODEL/gguf" --output target/flux3-conversion.json
cargo build --offline --profile release-fast --features parity-trace --bin rust-model-inference

target/release-fast/rust-model-inference \
  --model "$MODEL/gguf/flux3-dit.gguf" \
  --text-encoder "$MODEL/gguf/flux3-text.gguf" \
  --vae "$MODEL/gguf/flux3-vae.gguf" \
  --prompt 'A red cube on a white table.' \
  --resolution 64 --steps 2 --seed 42 --threads 8 --out target/flux3.png
```

The converter refuses to replace existing output files. The CLI supports
`--overwrite` explicitly. Use `--image` and repeated `--reference` for image
conditioning (up to ten). Reference pixels are encoded through the deterministic
32-channel mean, quant_conv, patchify and the checkpoint's BN statistics; they
are never posterior samples. This initial VAE integration accepts square images
and references on the 16-pixel grid, 64..4096 pixels. It rejects rectangular
inputs. Supply a full scene caption; the separate prompt upsampler
is not part of this adapter.

### Hybrid Vulkan execution

Build with `--features vulkan` (add `parity-trace` for alignment) and add
`--gpu` or `--compute auto` to the CLI command above. DiT BF16 projections use
the existing Vulkan linear executor with F32 activations, a bounded 128-row
staging buffer, and weight uploads retained across sampling steps. Existing
Vulkan attention and supported F16 VAE convolutions also offload. The text
encoder, normalization, RoPE, sampler and remaining operations run on CPU.
`--compute vulkan` still requires a fully GPU-resident model and is rejected
by the diffusion CLI; use `--compute auto` for this hybrid adapter.

`RMI_PARITY_TRACE` now observes the requested GPU backend instead of disabling
operator offload. `RMI_SCALAR=1` and explicit CPU scopes retain CPU execution.
`RMI_COMPUTE_TRACE=1` reports actual backend selection and upload counters.

## Reproduce alignment

Check out the official source at the commit above into an ignored directory,
for example `target/flux3-oracle`. Do not alter its graph. CPU Python needs
PyTorch, Transformers, NumPy, safetensors and Pillow in the repository venv;
there is no need to install the official GPU/UI dependencies.

```sh
python tools/oracle/flux3/reference.py "$MODEL" target/flux3-reference \
  --oracle-source target/flux3-oracle --side 64 --steps 2 --seed 42 --threads 8
```

Run the same Rust command above with `RMI_PARITY_TRACE=target/flux3-rust.jsonl`,
then compare:

```sh
python tools/oracle/flux3/compare.py \
  target/flux3-reference/trace.jsonl target/flux3-rust.jsonl \
  --output target/flux3-comparison.json
```

Use fresh trace/output paths for each run. `--reference IMAGE` on the Oracle
matches `--reference IMAGE` on the CLI. `--dtype bf16` provides a second official
CPU BF16 reference; the comparator selects its numerical bound from `run.json`.
The F32 reference widens the exact released weights and executes the official
graph, separating graph mistakes from BF16 rounding differences.

The comparison requires canonical checkpoint order, names, occurrence counts
and shapes, including every DiT block and every sampling step. Token IDs and
BF16 initial noise use raw-bit comparison. Floating calculations use relative
L2 <= 0.005, and VAE encode/decode <= 0.03; CPU BF16 comparison uses <= 0.06.
All comparisons reject non-finite values and report max absolute errors. Extra
legacy VAE debugging checkpoints are not part of this canonical trace. The
shared VAE RGB trace has a legacy H,W,C shape label with planar payload; the
Oracle preserves that exact trace schema.

## Validation evidence

The runtime checks all 398 text tensors for the released BF16 dtype, shape and
byte length before entering the shared Qwen loader. It also validates the
released Qwen configuration and rejects additional QKV biases and output or
classification heads. Missing or differently quantized components return an
error before model construction. The rejection tests were verified red/green,
and the released tokenizer, text encoder and VAE all pass after the fix.

The supported numerical path is **F32 activations/accumulation on CPU or hybrid
Vulkan**, with released BF16 DiT/text weights and F16 VAE convolutions. The primary Oracle
widens the released weights and runs the pinned official graph in F32.

Real CLI comparisons use seed 42, 64x64 and two flow-map steps. macOS arm64 uses
8 Rust threads; Linux x86_64 uses 16 (EPYC 9334, cgroup limit 55 GiB / 16 CPUs).
Both platforms are compared with the same pinned F32 CPU Oracle recorded on
arm64 with 8 threads.

| Platform / case | Canonical checkpoints | Text relative L2 | Maximum velocity relative L2 | RGB relative L2 |
| --- | ---: | ---: | ---: | ---: |
| arm64 / text to image | 89 | 0.0000154383 | 0.0000978136 | 0.000585576 |
| arm64 / two prepared references | 103 | 0.0000154383 | 0.00105131 | 0.000157232 |
| x86_64 / text to image | 89 | 0.0000158357 | 0.000188365 | 0.000858396 |
| x86_64 / two prepared references | 103 | 0.0000158357 | 0.000797763 | 0.000319389 |
| x86_64 / Vulkan text to image | 89 | 0.0000158357 | 0.000190077 | 0.000869048 |
| x86_64 / Vulkan two prepared references | 103 | 0.0000158357 | 0.000747608 | 0.000176240 |

All six cases pass every predeclared F32 bound, produce CLI PNGs, and preserve
canonical checkpoint order/shape/count. Token IDs and initial noise are exact.

- All source SHA256 hashes match the fixed revision. All **1007** converted
  tensors are checked: 935 payloads retain source bits; 72 VAE convolution
  payloads match the specified F16 cast (weight relative L2 0.000207374).
- Five released tokenizer fixtures cover English, Chinese/Unicode/NFC,
  whitespace, special literals and empty prompt; padding/positions and
  4096-token truncation are checked.
- Separate real VAE encode/decode: reference features relative L2 0.00135860,
  RGB 0.000483546 on arm64. On Linux x86_64 (EPYC 9334, container limit 55 GiB,
  16 CPUs), RGB is 0.000476291. Both runs pass all four checkpoints.
- The final Vulkan/trace build passes four regular FLUX3 tests (four ignored).
  Explicit real released tokenizer, text encoder and VAE tests pass separately
  (three passed, none ignored). The separate text-only Oracle comparison
  passes both checkpoints on arm64 (text relative L2 0.0000154383) and x86_64
  (0.0000158357). All three GGUF files pass full SHA256 checks on Linux, whose
  changed Rust source files also match the PR source hashes.
- Eleven shared Qwen regression tests pass on both platforms with `parity-trace` (one
  ignored), including hidden-layer capture and key masking. All eight
  converter/comparator checks and asymmetric F16/BF16 downsampling pass.
  The binary dispatches `flux3` explicitly.
- Broader arm64 VAE regression: 25 passed, 1 ignored, 1 existing SiLU bitwise
  assertion failed by 1 ULP. The same values/failure reproduce on unmodified
  initial HEAD `973210b6`.
- The Linux build, three FLUX3 unit tests and released text/VAE tests pass.
  Its existing Rust 1.94.1
  needs temporary `RUSTC_BOOTSTRAP=1` and
  `RUSTFLAGS='-C target-cpu=native -Zcrate-attr=feature(if_let_guard)'` for
  pre-existing syntax in `src/prompt/jinja_compat.rs`; no source workaround or
  global environment change is required.

### GPU execution evidence and regressions

The Linux Vulkan runs use `AMD Radeon Graphics (RADV NAVI31)`, the Radeon ICD,
and a device exposing 51,522,830,336 bytes of VRAM. They use the same released
weights, inputs, pinned F32 Oracle and fixed bounds as the CPU cases above.
No threshold was relaxed for GPU execution.

| Case | Vulkan DiT linear calls | CPU DiT linear calls | Device submissions at last DiT snapshot | Static upload bytes | Peak allocation counter bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text to image | 484 | 0 | 561 | 9,756,476,480 | 9,787,544,400 |
| Two references | 550 | 0 | 695 | 11,165,325,888 | 11,196,783,152 |

Neither run reports a GPU error or fallback. These are the last DiT executor
counter snapshots, not whole-process physical VRAM peaks; VAE decoding follows.
An earlier release-fast 256x256, default 16-step `--compute auto` CLI run exits successfully
in 155.29 seconds and produces the requested red cube scene. A later full trace
of this workload passes **677/677 canonical Oracle checkpoints** and produces
the same PNG SHA256. Token IDs and noise match raw bits. Maximum velocity
relative L2 is **0.0000670378**, final RGB **0.0000950790**, and the worst
intermediate checkpoint is `single_blocks.26`, occurrence 14, at **0.00349466**
against the unchanged 0.005 bound. The trace records 9,760 Vulkan linear calls,
zero CPU linear fallback and no GPU errors.

Reproduce this larger case with `--side 256 --steps 16` on the Oracle and
`--resolution 256 --steps 16 --compute auto` on the Vulkan CLI. Compare every
step with the same comparator. The trace was started before the final
text-input validation fix; its numerical operations are unchanged for the
accepted official component. Source/binary hashes and Oracle provenance are
recorded per case in `validation.json`; final release performance runs exercise
the updated validator. Trace duration is excluded from performance data.

Hardware regressions cover a 257-row BF16 projection (two staging tiles and a
tail), repeated execution without another static upload, scalar-mode CPU
selection, and traced GPU attention with a hand-computed result and a real
submission. Run each in a separate process on a hardware Vulkan device:

```sh
cargo test --offline --profile release-fast --features vulkan,parity-trace --lib \
  vulkan_projection_tiles_rows_and_reuses_weight_uploads -- --ignored --test-threads=1
RMI_SCALAR=1 cargo test --offline --profile release-fast --features vulkan,parity-trace --lib \
  vulkan_projection_tiles_rows_and_reuses_weight_uploads -- --ignored --test-threads=1
RMI_PARITY_TRACE=target/flux3-attention-test.jsonl \
  cargo test --offline --profile release-fast --features vulkan,parity-trace --lib \
  tracing_preserves_requested_gpu_attention -- --ignored --test-threads=1
```

The final whole-PR regression compares against actual `main` commit `973210b6`,
not the intermediate CPU adapter. With identical unavailable ICD overrides
(`VK_ICD_FILENAMES` and `VK_DRIVER_FILES`) and compiled `vulkan,parity-trace`
features, single-thread library tests report **1339 passed / 29 failed / 158
ignored on main**, and **1345 passed / the same 29 failed / 163 ignored on the
PR**. This controlled CPU-fallback comparison has no new failures; real-device
Vulkan checks remain separate.

With the local device enabled, main initially reports 32 failures and the PR
35; three main repeats report 33/33/32. The three extra PR failure names all
pass individually on both revisions, and one also fails in repeated main
whole-suite runs. Thus the device-enabled full suite remains unstable, and is
not presented as green. Independent review found and closed the text-component
input-contract gap; it found no remaining critical or important code issues.
Shared Qwen trunk regressions pass (28 passed, 1 ignored). Earlier baseline
comparisons and failure names remain recorded in `validation.json`.

### Performance (2026-10-23)

The same `release` binary (`--features vulkan`, without `parity-trace`) is timed
on the EPYC 9334 host with 16 Rust threads and a 16-core cgroup quota. GPU runs
select the NAVI31 device at PCI `0000:c3:00.0`, `/dev/dri/renderD134`. All runs use
the same weights, caption, seed 42, 256x256 output, **two steps**, no references,
and explicit `--compute cpu` / `--compute auto` policies. Model files are read
before each timed run; process initialization, model setup and GPU uploads are
included. Separate 64x64/two-step warmups precede measurement. Rounds alternate
backend order, with no concurrent CPU/GPU generation from this benchmark.

| Round | CPU total seconds | Hybrid Vulkan total seconds |
| --- | ---: | ---: |
| 1 | 315.99 | 54.22 |
| 2 | 180.66 | 57.84 |
| 3 | 274.26 | 69.76 |
| Median | 274.26 | 57.84 |

The ratio of total-time medians is **4.74x**. The second denoise step takes
106.61s / 7.64s (CPU / GPU medians, **13.96x**); this is the measured second step,
not an extrapolated 16-step CPU time. VAE medians are 5.02s / 2.64s; process peak
RSS medians are 15.32 / 15.60 GiB. Selected-device GPU memory sampling records
about 9.27 GiB of host-backed GTT memory in GPU rounds 2/3; device VRAM and GTT
are separate counters, and neither is interchangeable with process RSS.

A separate 256x256, **16-step GPU** run produces the red cube image in 154.80s,
with a 5.99s median across denoise steps 2..16. The three two-step PNGs are
deterministic within each backend; between CPU/GPU, decoded RGB8 pixels have
maximum absolute difference 1/255. These two-step PNG checks are separate from
the 16-step official comparison above.

The final runtime also completes a **512x512/two-step single pair**. Latency
through the PNG announcement is **approximately 1593.82s CPU / 635.03s GPU**
(about **2.51x**), with the same final release binary, weights, caption, seed
and 16 threads. Both generations exit with code 0; decoded RGB8 pixels differ
by at most 1/255. This is PNG consistency, not an official 512x512 Oracle check.

This extension has explicit measurement deviations. The container exposes only
`renderD134` at `0000:c3:00.0`; attempting to select an idle host card did not
change the actual render node. That selection assertion failed after successful
generation, and mistakenly sampled card3 memory counters are excluded. An
earlier contended GPU attempt was deliberately terminated and excluded. SSH
disconnected during CPU generation; a detached reader preserved the same run,
recovered its start within 7.18ms of the kernel starttime, and recorded the PNG.
Exit code 0 was recovered from the matching zombie process stat. CPU timing is
therefore approximate. Raw evidence, source/binary hashes and these boundaries
are recorded under `extension_512` in `performance.json`.

This is a **shared-host observation**, with substantial CPU variation and no
exclusive CPU/GPU reservation. Device activity also occurs during CPU rounds
1/3 while those CPU processes have no render node open; no exact cause of the
timing variation is attributed. No 16-step CPU or isolated hardware benchmark
is claimed. Commands, binary/source/PNG hashes, all round timings and memory
sampling scope are recorded in [performance.json](performance.json).

### Cross-precision diagnostic

The native CPU BF16-activation Oracle is also measured. Against the supported
F32 runtime it **fails** the fixed 0.06 bound: the first failure is
`single_blocks.18`, occurrence 1; second velocity relative L2 is 0.0992463,
final RGB 0.310745. Token IDs/noise remain exact. This is recorded as a failed
cross-precision diagnostic, never as native BF16 activation parity. The F32
primary comparisons above are the supported accuracy claim.

The diagnostic includes the official BF16 timestep/embedding arithmetic,
intermediate rounding and BF16 text encoder. Controlled runs with rounded
timestep inputs alone and with widened text features still exceed the BF16
bound; the difference is not resolved by changing a comparison threshold.

No rectangular image, Base/FP8/NVFP4 or native BF16 activation parity claim
is made. This is an experimental CPU/hybrid Vulkan adapter; numerical Oracle
coverage reaches 256x256/16 steps. A fully GPU-resident pipeline remains
unsupported.
