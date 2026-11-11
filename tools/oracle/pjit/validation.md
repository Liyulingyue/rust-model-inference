# P-JiT 21K validation — 2026-11-11

Status: **Experimental / partial adaptation**, not ready for a verified-model check mark.
The remaining work includes real numerical failures, not only missing test runs.

## Inputs and boundaries

Pins, source hash and commands are in [README.md](README.md). The DiT safetensors
SHA256 was verified on the remote host. All nine DiT exports completed there;
text F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q6_K exports completed. Text Q4_K/Q5_K were still
converting at the last successful observation. Exported GGUF hashes were not
collected before the connection was lost; source hashes do not substitute for
them. No model weights were downloaded onto the Mac.

On the replacement host, all source files were downloaded again directly on
Linux, and their size/revision/SHA256 manifest was retrieved to
[source_hashes.json](source_hashes.json). The final rebased code passed 8 targeted
tests (3 asset-dependent tests ignored). DiT F32/F16/BF16/Q8_0/Q4_0/Q4_1 and text
F32/F16/BF16/Q8_0 export completion was observed; later conversion completion is
unknown. A remote-only parallel scheduler reused the shared quantizers, with
all six formats first checked for exact full-tensor versus tiled payload bytes
on a 257×2560 BF16 fixture. The replacement endpoint subsequently returned
`Connection refused` too. No successful final release build, completed transport
matrix or post-fix sampler result was observed there.

Runtime changes are confined to the new model module, necessary architecture
registration, CLI orchestration and an image-generation HTTP capability. Shared
kernels and other model implementations are unchanged. Cargo metadata confirms
all nine existing binary target names/paths remain unchanged. The branch was
rebased onto upstream `d16dc16d96126351b10c13c8458f968c5c57b2e7` before submission;
recorded historical/full-suite baselines refer to the starting commit in plan.md.
The existing x86 CI job also runs the new model/CLI and HTTP validation tests.

The authorized Linux host reported EPYC 9334 (128 logical CPUs), approximately
500 GB host memory, RADV gfx1100, 48 exposed CUs and approximately 48 GB VRAM.
Rust used project-local toolchains/dependencies; no external acceleration library
was added to the Rust runtime. Official references used PyTorch ROCm.

## Acceptance and observed results

Predeclared gates: discrete token IDs/shapes exact; F32 DiT max_abs ≤ 0.002 and
cosine ≥ 0.99999; lossy same-weight reference NRMSE ≤ 0.02 and cosine ≥ 0.999.
These gates were not relaxed after measurement. Native BF16 activation results,
weight-only quantization quality and runtime implementation error are separate.

The single-forward fixture is 33×35, timestep 750, 18 caption tokens, identical
BF16-rounded image/text inputs. F32 CPU max_abs is 1.88947e-5, NRMSE 1.07669e-5;
Vulkan max_abs is 1.10865e-5, NRMSE 7.83528e-6. Both pass. CPU intermediate
checkpoints are below NRMSE 1e-4. F16 and BF16 passed all three requested backends
in the recorded DiT matrix. GPU evidence for F32 is 244 submissions: 218 linear
projections plus 26 attention operations. This is a hybrid pipeline: text
encoding, norms, RoPE, elementwise work and sampler remain on CPU.

Caption BF16 same-GGUF independent decoding gives NRMSE 0.00579369 / cosine
0.99998337. Rust versus the official native BF16 encoder gives 0.0162553 /
0.99986889. Original F32 versus native BF16 itself differs by NRMSE 0.0136496.
F32 caption same-GGUF reference gives NRMSE 0.000111326 / cosine 0.999999994.
The tokenizer matched the original chat template exactly for empty, Unicode,
NFC, emoji, whitespace, special-token and 256-token truncation cases.

Quantized DiT full-forward checks **fail** the 0.02 NRMSE gate. A weight-only
reference initially measured about 0.03–0.054 runtime-to-reference difference; that
reference omits the shared kernels' dynamic activation quantization. Independent
CPU activation-QDQ was then added. Q8 still measures NRMSE 0.0601911 / cosine
0.9981898 and fails. Its patch checkpoint matches to NRMSE 6.8e-7, first image
refiner to 0.000162742, block 21 to 0.0313247. This locates gradual divergence;
it does not prove every shared kernel is correct. The same-input projection
replay was subsequently completed for Q8 on the replacement host: all **217
projections passed**, with worst NRMSE **5.40955e-7** and cosine
**0.999999999999854**. Each reference receives the exact traced Rust projection
input, independently decodes the same GGUF weights and applies NumPy activation
QDQ. This rules out a large local projection error for that fixture and supports
propagation across quantization boundaries as the source of the full-forward
drift; it does not make the failing end-to-end gate pass. Only its terminal
summary was retrieved before the second disconnect. Q4/K projection replay,
activation-QDQ and quantized caption references remain
unverified. Do not describe the weight-only comparison as full kernel alignment.
Weight-only quality loss versus F32 is reported separately in
[results_snapshot.json](results_snapshot.json), including large Q4 degradation.

A seven-step F32 DiT/APG replay used identical Rust caption features and initial
noise with the official implementation. First-step NRMSE is 6.78698e-7; final
NRMSE is 0.0321500 / cosine 0.9994857 and **fails**. This was before the final APG
cast-order repair. The final chain must be rerun, and divergence at BF16 state
boundaries must be investigated; no successful post-fix chain is claimed.

Raw traces/results remain in remote `target/`. Only terminal-observed results
were captured in the committed snapshot; no fabricated raw fixture or completed
image is included. The connection was closed before SSH authentication during
final validation. No completed 1024×1024 / 50-step output was observed (the CPU
run had reached step 7). Default-size image quality is unverified.

## Coverage matrix

Each row distinguishes implementation from recorded validation. All transport
records below predate the last review fixes; the final remote regression is
pending. “Pass” for CLI/HTTP means exit 0 / HTTP 200 and decoded 33×35 PNG, one
step, seed 42, prompt `猫`; it does not imply the numerical gate passed. Each
HTTP run also rejected blank prompt, n=2, unknown model and invalid size with
400. Requested Auto/Vulkan caption rows still execute on CPU.

Command templates (substitute FORMAT and BACKEND for each row):

```sh
target/release-fast/rust-model-inference --model models/pjit-21k/pjit-FORMAT.gguf \
 --text-encoder models/Qwen3.5-4B/text-FORMAT.gguf --threads 8 --compute BACKEND \
 --prompt '猫' --width 33 --height 35 --steps 1 --seed 42 --out target/result.png --overwrite
target/release-fast/rust-model-server --model models/pjit-21k/pjit-FORMAT.gguf \
 --text-encoder models/Qwen3.5-4B/text-FORMAT.gguf --threads 8 --compute BACKEND --port PORT
# POST /v1/images/generations
# {"prompt":"猫","width":33,"height":35,"steps":1,"seed":42,"response_format":"b64_json"}
```

Evidence per transport: remote `target/pjit-transport/FORMAT-BACKEND.json`,
`-cli.log`, `-http.log`, `-cli.png`, `-http.png` and `-response.json`.
DiT metrics: `target/pjit-matrix/results.json`; QDQ traces:
`target/pjit-q8-cpu.jsonl` and `target/pjit-ref-q8-qdq-cpu/`.
Text metrics: `target/pjit-text-matrix/results.json`, plus BF16 checks described
above. Missing final files after the disconnect must be checked, not assumed.

| Component | Storage format | Requested backend | Implementation | CLI | HTTP | Numerical validation | Actual execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DiT | F32 | cpu | Implemented | Pass | Pass | Pass (single forward) | CPU SIMD/pool |
| DiT | F32 | auto | Implemented | Pass | Pass | Pass (single forward) | Vulkan projections/attention + host |
| DiT | F32 | vulkan | Implemented | Not verified | Not verified | Pass (single forward) | Vulkan projections/attention + host |
| DiT | F16 | cpu | Implemented | Pass | Pass | Pass (single forward) | CPU SIMD/pool |
| DiT | F16 | auto | Implemented | Pass | Pass | Pass (single forward) | Vulkan projections/attention + host |
| DiT | F16 | vulkan | Implemented | Not verified | Not verified | Pass (single forward) | Vulkan projections/attention + host |
| DiT | BF16 | cpu | Implemented | Pass | Pass | Pass (single forward) | CPU SIMD/pool |
| DiT | BF16 | auto | Implemented | Pass | Pass | Pass (single forward) | Vulkan projections/attention + host |
| DiT | BF16 | vulkan | Implemented | Not verified | Not verified | Pass (single forward) | Vulkan projections/attention + host |
| DiT | Q8_0 | cpu | Implemented | Pass | Pass | Fail (CPU QDQ and weight-only) | CPU SIMD/pool |
| DiT | Q8_0 | auto | Implemented | Pass | Pass | Fail (weight-only); QDQ pending | Vulkan projections/attention + host |
| DiT | Q8_0 | vulkan | Implemented | Not verified | Not verified | Fail (weight-only); QDQ pending | Vulkan projections/attention + host |
| DiT | Q4_0 | cpu | Implemented | Pass | Pass | Fail (weight-only); QDQ pending | CPU SIMD/pool |
| DiT | Q4_0 | auto | Implemented | Pass | Pass | Fail (weight-only); QDQ pending | Vulkan projections/attention + host |
| DiT | Q4_0 | vulkan | Implemented | Not verified | Not verified | Fail (weight-only); QDQ pending | Vulkan projections/attention + host |
| DiT | Q4_1 | cpu | Implemented | Pass | Pass | Fail (weight-only); QDQ pending | CPU SIMD/pool |
| DiT | Q4_1 | auto | Implemented | Pass | Pass | Fail (weight-only); QDQ pending | Vulkan projections/attention + host |
| DiT | Q4_1 | vulkan | Implemented | Not verified | Not verified | Fail (weight-only); QDQ pending | Vulkan projections/attention + host |
| DiT | Q4_K | cpu | Implemented | Not verified | Not verified | Fail (weight-only); QDQ pending | CPU SIMD/pool |
| DiT | Q4_K | auto | Implemented | Not verified | Not verified | Fail (weight-only); QDQ pending | Vulkan projections/attention + host |
| DiT | Q4_K | vulkan | Implemented | Not verified | Not verified | Fail (weight-only); QDQ pending | Vulkan projections/attention + host |
| DiT | Q5_K | cpu | Implemented | Not verified | Not verified | Fail (weight-only); QDQ pending | CPU SIMD/pool |
| DiT | Q5_K | auto | Implemented | Not verified | Not verified | Fail (weight-only); QDQ pending | Vulkan projections/attention + host |
| DiT | Q5_K | vulkan | Implemented | Not verified | Not verified | Fail (weight-only); QDQ pending | Vulkan projections/attention + host |
| DiT | Q6_K | cpu | Implemented | Not verified | Not verified | Fail (weight-only); QDQ pending | CPU SIMD/pool |
| DiT | Q6_K | auto | Implemented | Not verified | Not verified | Fail (weight-only); QDQ pending | Vulkan projections/attention + host |
| DiT | Q6_K | vulkan | Implemented | Not verified | Not verified | Fail (weight-only); QDQ pending | Vulkan projections/attention + host |
| Caption | F32 | cpu | Implemented (CPU) | Pass | Pass | Pass on CPU | CPU prefixes 8/16/28 |
| Caption | F32 | auto | GPU raw features not implemented; CPU used | Pass | Pass | Pass on CPU | CPU prefixes 8/16/28 |
| Caption | F32 | vulkan | GPU raw features not implemented; CPU used | Not verified | Not verified | Pass on CPU | CPU prefixes 8/16/28 |
| Caption | F16 | cpu | Implemented (CPU) | Pass | Pass | Not verified | CPU prefixes 8/16/28 |
| Caption | F16 | auto | GPU raw features not implemented; CPU used | Pass | Pass | Not verified | CPU prefixes 8/16/28 |
| Caption | F16 | vulkan | GPU raw features not implemented; CPU used | Not verified | Not verified | Not verified | CPU prefixes 8/16/28 |
| Caption | BF16 | cpu | Implemented (CPU) | Pass | Pass | Pass on CPU | CPU prefixes 8/16/28 |
| Caption | BF16 | auto | GPU raw features not implemented; CPU used | Pass | Pass | Pass on CPU | CPU prefixes 8/16/28 |
| Caption | BF16 | vulkan | GPU raw features not implemented; CPU used | Not verified | Not verified | Pass on CPU | CPU prefixes 8/16/28 |
| Caption | Q8_0 | cpu | Implemented (CPU) | Pass | Pass | Not verified | CPU prefixes 8/16/28 |
| Caption | Q8_0 | auto | GPU raw features not implemented; CPU used | Pass | Pass | Not verified | CPU prefixes 8/16/28 |
| Caption | Q8_0 | vulkan | GPU raw features not implemented; CPU used | Not verified | Not verified | Not verified | CPU prefixes 8/16/28 |
| Caption | Q4_0 | cpu | Implemented (CPU) | Pass | Pass | Not verified | CPU prefixes 8/16/28 |
| Caption | Q4_0 | auto | GPU raw features not implemented; CPU used | Pass | Pass | Not verified | CPU prefixes 8/16/28 |
| Caption | Q4_0 | vulkan | GPU raw features not implemented; CPU used | Not verified | Not verified | Not verified | CPU prefixes 8/16/28 |
| Caption | Q4_1 | cpu | Implemented (CPU) | Pass | Pass | Not verified | CPU prefixes 8/16/28 |
| Caption | Q4_1 | auto | GPU raw features not implemented; CPU used | Pass | Pass | Not verified | CPU prefixes 8/16/28 |
| Caption | Q4_1 | vulkan | GPU raw features not implemented; CPU used | Not verified | Not verified | Not verified | CPU prefixes 8/16/28 |
| Caption | Q4_K | cpu | Implemented (CPU) | Not verified | Not verified | Not verified | CPU prefixes 8/16/28 |
| Caption | Q4_K | auto | GPU raw features not implemented; CPU used | Not verified | Not verified | Not verified | CPU prefixes 8/16/28 |
| Caption | Q4_K | vulkan | GPU raw features not implemented; CPU used | Not verified | Not verified | Not verified | CPU prefixes 8/16/28 |
| Caption | Q5_K | cpu | Implemented (CPU) | Not verified | Not verified | Not verified | CPU prefixes 8/16/28 |
| Caption | Q5_K | auto | GPU raw features not implemented; CPU used | Not verified | Not verified | Not verified | CPU prefixes 8/16/28 |
| Caption | Q5_K | vulkan | GPU raw features not implemented; CPU used | Not verified | Not verified | Not verified | CPU prefixes 8/16/28 |
| Caption | Q6_K | cpu | Implemented (CPU) | Not verified | Not verified | Not verified | CPU prefixes 8/16/28 |
| Caption | Q6_K | auto | GPU raw features not implemented; CPU used | Not verified | Not verified | Not verified | CPU prefixes 8/16/28 |
| Caption | Q6_K | vulkan | GPU raw features not implemented; CPU used | Not verified | Not verified | Not verified | CPU prefixes 8/16/28 |

IQ and other decoder-only formats are not new export formats: the existing
shared exporter exposes only the nine formats above (including no Q5_0/Q5_1
encoder). Unsupported tensor types are explicitly rejected by this loader. JEV and
streaming are not applicable to still-image generation without probability or
incremental-output semantics. Video, image editing, VAE input and remote prompt
upsampling are not capabilities of this adapter.

## Checks and review

- Final local (rebased) `cargo test --locked --profile release-fast --lib pjit --features vulkan,parity-trace`:
  8 passed, 3 real-weight checks explicitly ignored without assets; the HTTP
  request-validation test also passed separately with the same features.
- Remote Vulkan-enabled targeted tests: 7 passed, 3 ignored before the last CLI
  review repair. Both existing binaries built with Vulkan before that repair.
- Final local existing inference/server binary build passed.
- Python converter/oracle helper checks: 7 passed, including failure cleanup and
  racing-output overwrite protection. No Torch installation is needed for these.
- Recorded full library suite: 1651 passed / 47 failed / 230 ignored; base:
  1642 passed / 51 failed / 228 ignored. All 47 failure names exist in the base;
  four additional base-only failures are Gemma4 Vulkan tests. A final Vulkan-enabled
  pre-rebase suite recorded 1667 passed / 35 failed / 231 ignored, with no failure
  names absent from the starting-base list. See [regression.json](regression.json).
- Clippy is blocked by ten deny-level lint errors in unchanged ERNIE, Pig,
  FunASR, Qwen vision, YuE2 and torch28 files. Those files match the base commit.
- Touched-file rustfmt and `git diff --check` passed.

Self-review repaired integer timestep generation (torch.linspace FMA/endpoints),
APG F64 projection cast order, negative seed handling, incompatible CLI flags,
request validation, permit lifetime across HTTP cancellation, converter payload
validation and atomic no-overwrite publication. CPU attention partitions heads
through the existing pool; a 23-head ordering test matches the shared CPU
operation exactly with one and four threads. No old model/shared compute code
was changed to force parity.

## Remaining work and ownership

1. Adapter owner: restore remote access, collect export hashes and raw artifacts,
   finish K text and strict Vulkan CLI/HTTP checks, verify Auto fallback with no
   device and strict failure, rerun final code against real weights.
2. Adapter numerical owner: extend the successful Q8 same-input projection replay
   to Q4/K formats and complete every
   activation-QDQ reference; fix a demonstrated local defect if found. Shared
   compute owner must handle any demonstrated prepared-kernel discrepancy with
   old-model baselines before changing shared arithmetic. All failing rows stay
   Experimental until the existing gates pass or a new, independently calibrated
   contract is agreed before a new validation run.
3. Shared compute/Qwen owner (follow-up layer): expose raw multi-layer device
   features, then validate caption CPU/Auto/Vulkan separately. Current CPU prefix
   composition is deliberate and does not claim full GPU text support.
4. Adapter owner: finish 1024×1024 / 50-step task output, visual review and matched
   multi-step comparison; run clean warm release CPU/Vulkan/official benchmarks
   on the same host. See [performance.json](performance.json) for diagnostic
   numbers and their incompatible timing scopes.
