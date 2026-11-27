# SenseNova Loop 9B aarch64 cross-validation

Date: 2026-11-27 (Asia/Shanghai). Base: PR #256 integration tree `cf171cb`
(plus three follow-up commits ending `fd8fba2`; identical model code).
Host: local aarch64 validation machine, NVIDIA GB10 GPU via the NVIDIA Vulkan
ICD, 121 GiB RAM. CPU backend runs the NEON kernel paths that the x86 oracle
matrix (see [release-validation.md](release-validation.md)) never exercised;
`vulkan`/`auto` runs are the aarch64-host GPU mode-1 push path, not the x86
mode-8 FMA branch.

Method: full oracle reproduction with the repository toolchain — official
PyTorch F32 eager reference (`tools/oracle/sensenova_loop/oracle.py --device
cpu`), native three-backend runner (`examples/sensenova_loop_check`), and the
same `compare.py` gates (F32 tight gates by default, `--lossy` for block
formats). All 25 pinned source files downloaded from ModelScope revision
`3fa2faf3` and verified against `release-source-manifest.json` (size + SHA256).
Text fixtures: 3 cases × 8 steps = 570 checkpoints. Image fixture: a stdlib-
generated 256×256 solid-blue RGB PNG, 8 steps = 457 checkpoints.

## Weight and conversion identity

| Artifact | Bytes | Notes |
| --- | ---: | --- |
| F32 text GGUF | 35,838,774,432 | byte count matches the x86 oracle record exactly |
| F16 / Q4_0 / Q4_K / Q4_1 / Q8_0 | oracle byte counts match exactly | |
| mmproj (F32 vision) | 1,824,061,376 | **SHA256 matches the x86 oracle record bit-for-bit** (`dcc42d2b…`) |
| F32 / Q4_0 / Q4_K text GGUF | hashes differ from the x86 records | see below |

The text-GGUF hash difference is fully explained by `convert.py` line 341:
`ssm_a = -np.exp(A_log)` (plus the `+1` bias adjustment) — `np.exp` has
SIMD-implementation ULP differences between x86 and aarch64, so the same source
produces identical byte counts with ULP-level differences in the `ssm_a`
tensors. The projector performs no such math and is bit-identical across
architectures, which confirms the converter itself is deterministic. This is a
platform ULP artifact, not a defect; no numerical gate depends on GGUF hash
equality.

## Text matrix (570 checkpoints, 3 backends)

| Storage | CPU (NEON) | Auto | Vulkan (NVIDIA ICD) | Greedy IDs vs reference |
| --- | --- | --- | --- | --- |
| F32 | **570/570 pass** | **570/570 pass** | **570/570 pass** | exact, all cases |
| F16 | **570/570 pass** (`--lossy`) | — | **570/570 pass** | exact |
| BF16 | **570/570 pass** | — | **570/570 pass** | exact |
| Q4_0 | 142/570 fail | — | 109/570 fail | exact (same-GGUF QDQ reference) |
| Q4_1 | 119/570 fail | — | 115/570 fail | exact |
| Q8_0 | 108/570 fail | — | 113/570 fail | exact |
| Q4_K | 403/570 fail | — | 404/570 fail | exact |

Q4_0 case-0 greedy `[9419, 0, 2500, 628, 353, 7543, 488, 3242]` matches the
official reference token-for-token on all three backends; the same holds for
the other cases and formats. Worst F32 relative RMSE 1.47e-5 (cosine
0.99999999989) — same class as the x86 oracle's F32 record.

## Image matrix (256×256 blue fixture, 457 checkpoints)

| Storage | CPU | Auto | Vulkan | Notes |
| --- | --- | --- | --- | --- |
| F32 | 46/457 fail | 46/457 fail | 46/457 fail | greedy `[1919, 2099, 369, 6105, 13, 248046, 198, 248069]` exact on all three; failure count 46 matches the x86 oracle's F32 image record with the same intermediate-state pattern |
| F16 | **457/457 pass** | — | **457/457 pass** | greedy exact |
| BF16 | **457/457 pass** | — | **457/457 pass** | greedy exact |
| Q4_0 (CLI 8-token run) | "This image is primarily blue…" | "This image is blue." | **"Blue."** | strict-Vulkan output matches the x86 integrated snapshot's strict-Vulkan `Blue.` exactly; each backend names the blue fixture correctly |

The F32 image intermediate-layer failures and the quantized full-model gate
failures reproduce with the same shape as the x86 oracle records — the
aarch64/GB10 paths inherit the same open acceptance gaps, and no new failure
class appears.

## HTTP / JEV interfaces (F16 backend)

`rust-model-server` with the F16 GGUF + F32 projector: nonstream and stream
text generation, image generation via base64 `image_url` (answers "This image
is blue."), `/v1/jev/score` (choice: "去公园散步" 0.99999),
`/v1/jev/grouped` (multi_select with per-group probabilities),
`/v1/jev/image` (blue image → "蓝色" 0.9995), `/v1/jev/image_grouped`
(block_choice: blue → "冷色/蓝色" 0.99985), and the single-option-group 400
validation all pass. Q4_0 image generation also ran on all three backends.

## Scope

Local reproduction only — no 19.34 GiB re-export of the remaining K formats'
numerical chains (Q5_K/Q6_K were converted and loaded but not oracle-compared;
the x86 matrix already records them as failing formats), no Preview rerun, no
official-GGUF interoperability change, no x86 AVX2/mode-8 coverage (those are
x86-only paths). Raw comparison JSON lives under `target/loop-*` (not
committed).
