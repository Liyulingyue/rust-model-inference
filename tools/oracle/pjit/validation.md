# P-JiT 21K validation — 2026-11-11

Status: **Experimental / partial adaptation**. Quantized end-to-end numerical gates and the missing resident GPU caption-feature API prevent a verified-model check mark.

## Contract, assets and execution

The official/model/llama.cpp revisions and direct-download/streaming relay commands are pinned in [README.md](README.md). All ten source files were downloaded directly onto the third authorized Linux host; their exact revision/size/SHA256 match [source_hashes.json](source_hashes.json). All nine DiT and nine caption GGUF exports completed. [gguf_hashes.json](gguf_hashes.json) records each exported file hash, size, architecture, file type and tensor type counts. Weights and raw F32 traces stayed remote; only small JSON/log/PNG evidence was copied onto the Mac.

Host: AMD EPYC 9334, 128 reported logical CPUs, AMD RADV NAVI31/gfx1100, 48 exposed CUs and 51,522,830,336 VRAM bytes. The container is limited to **16 CPUs and 55 GiB RAM**, despite the host reporting about 500 GiB RAM. Rust 1.94.1; PyTorch ROCm 2.9.1, HIP 7.2. Dependencies/builds stayed in repository-local environments; the Rust runtime uses no external acceleration library. Our heavy GPU validation jobs were serialized; other workloads still shared the host.

Runtime changes stay in `models/diffusion/pjit` plus necessary CLI/HTTP/architecture wiring. Shared kernels and other model implementations are unchanged. Cargo metadata retained all nine original binary target names/paths. The branch base is `d16dc16d96126351b10c13c8458f968c5c57b2e7`; historical suite baselines refer to the starting commit in [plan.md](plan.md).

Storage formats: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K. K-unaligned rows retain float storage, recorded in the manifest. Arithmetic uses the shared storage-specific F32-accumulating kernels, including their activation rounding/quantization; this is not native BF16 activation execution. Caption KV is F32. DiT inputs, caption features and APG momentum are BF16-rounded at the official boundaries.

## Numerical acceptance

Predeclared gates were unchanged: discrete shapes/token IDs exact; F32 DiT max_abs ≤ 0.002 and cosine ≥ 0.99999; lossy same-weight NRMSE ≤ 0.02 and cosine ≥ 0.999. Weight-only quantization quality is measured separately from runtime implementation error. No bitwise floating-point acceptance is imposed on lossy paths.

Single-forward fixture: 33×35, timestep 750, 18 text tokens, identical BF16-rounded image/text. Quantized references run the official model on ROCm, independently decode the same GGUF, and model the CPU prepared-kernel activation QDQ contract. Vulkan comparisons retain this CPU-QDQ reference limitation; same-input device projection replay tests the individual operations separately. All measured metrics and reference commands are in [matrix_results.json](matrix_results.json); raw tensor values stay remote.

| DiT storage | CPU NRMSE / gate | Auto NRMSE / gate | Vulkan NRMSE / gate |
| --- | --- | --- | --- |
| F32 | 1.2946124e-05 / Pass | 1.0935635e-05 / Pass | 1.1299469e-05 / Pass |
| F16 | 0.00081502589 / Pass | 0.0016752153 / Pass | 0.00096704771 / Pass |
| BF16 | 6.0414086e-06 / Pass | 8.5494247e-06 / Pass | 7.7380653e-06 / Pass |
| Q8_0 | 0.05142132 / Fail | 0.03935427 / Fail | 0.03935427 / Fail |
| Q4_0 | 0.040798267 / Fail | 0.04466361 / Fail | 0.039428394 / Fail |
| Q4_1 | 0.0418282 / Fail | 0.043201844 / Fail | 0.043201844 / Fail |
| Q4_K | 0.03095753 / Fail | 0.041085753 / Fail | 0.034642739 / Fail |
| Q5_K | 0.044037838 / Fail | 0.038663027 / Fail | 0.049144319 / Fail |
| Q6_K | 0.044340449 / Fail | 0.056909299 / Fail | 0.047473897 / Fail |

F32 per-backend max_abs/cosine are preserved in [results_snapshot.json](results_snapshot.json). Float single-forward passes do not certify the quantized rows. Quantized divergence propagates across discrete activation-quantization boundaries; neither local operation success nor quantization quality loss turns a failing full-forward gate into a pass.

| Caption storage | Same-GGUF NRMSE / gate | Reference device |
| --- | --- | --- |
| F32 | 0.00011132561 / Pass | ROCm |
| F16 | 0.00118355 / Pass | ROCm |
| BF16 | 0.0057936852 / Pass | ROCm |
| Q8_0 | 0.019844337 / Pass | ROCm |
| Q4_0 | 0.021210382 / Fail | ROCm |
| Q4_1 | 0.022507903 / Fail | ROCm |
| Q4_K | 0.049956631 / Fail | ROCm |
| Q5_K | 0.047343279 / Fail | ROCm |
| Q6_K | 0.044786469 / Fail | ROCm |

Historical BF16 caption comparisons: same-GGUF NRMSE 0.00579369 / cosine 0.99998337; original native BF16 encoder 0.0162553 / 0.99986889. The original F32-vs-native-BF16 reference itself differs by 0.0136496 NRMSE. These are distinct precision contracts. Fresh tokenizer checks passed all eight cases (empty, Unicode/NFC/NFD, emoji, whitespace, special tokens and truncation); exact expected IDs are committed in [tokenizer_cases.json](tokenizer_cases.json).

Same-input quantized projection replay independently decodes the exact GGUF and supplies each operation with its traced Rust input. This isolates local projection error from accumulated divergence:

| Storage / backend | Projections | Failed | Worst NRMSE |
| --- | --- | --- | --- |
| q4_0-cpu | 217 | 0 | 5.3591882e-07 |
| q4_0-vulkan | 217 | 0 | 5.3468231e-07 |
| q4_1-cpu | 217 | 0 | 0.0013139778 |
| q4_1-vulkan | 217 | 0 | 0.0013172824 |
| q4_k-cpu | 217 | 0 | 5.5903191e-07 |
| q4_k-vulkan | 217 | 0 | 1.0102545e-06 |
| q5_k-cpu | 217 | 0 | 5.8091466e-07 |
| q5_k-vulkan | 217 | 0 | 1.0098292e-06 |
| q6_k-cpu | 217 | 0 | 5.6734297e-07 |
| q6_k-vulkan | 217 | 0 | 1.0111968e-06 |
| q8_0-cpu | 217 | 0 | 5.4095469e-07 |
| q8_0-vulkan | 217 | 0 | 5.4095469e-07 |

### Official reference sensitivity

Duplicating identical inputs from batch one to batch two changes the GEMM accumulation shape without changing weights, inputs, activation QDQ or the official code/device. This is an independent diagnostic, not a replacement acceptance gate. Every failing runtime row above remains failed.

| Storage | Official DiT batch drift NRMSE | Official caption batch drift NRMSE |
| --- | --- | --- |
| F32 | Not run | 2.6782743e-05 |
| F16 | Not run | 2.2049606e-05 |
| BF16 | Not run | 2.6782743e-05 |
| Q8_0 | 0.047618916 | 0.010330463 |
| Q4_0 | 0.048850647 | 0.0120772 |
| Q4_1 | 0.039171112 | 0.012342825 |
| Q4_K | 0.045442063 | 0.021298324 |
| Q5_K | 0.043408121 | 0.018563758 |
| Q6_K | 0.052311885 | 0.021197768 |

All six quantized DiT reference self-drifts exceed the unchanged 2% gate (3.92–5.23%); caption K-format drift is also material. Together with same-input projection replays, this supports sensitivity at discrete activation-quantization boundaries. It does not establish complete runtime correctness or certify task quality. Commands and per-checkpoint metrics are in `matrix_results.json`.


The earlier Q8 CPU replay also passed 217/217 projections (worst NRMSE 5.40955e-7); only its terminal summary survived the second host disconnect. Fresh per-operation summaries above replace missing evidence where available.

The **post-APG-fix seven-step F32 Vulkan chain passes all seven gates**: final NRMSE 0.019726344, cosine 0.999806538752. First-step NRMSE 5.3991954e-07. Same-input DiT max_abs stays below 1.85e-5 and independent Euler/APG replay below 4.77e-7. BF16 input differences grow across the chain (0, 2, 15, 278, 787, 2175, 2779 elements). [sampler_7_steps.json](sampler_7_steps.json) retains every step and diagnostic. This supersedes the pre-fix 0.03215 failure; the old result remains in the historical snapshot.

## Real CLI and HTTP matrix

All 27 requested backend/storage combinations have real CLI/HTTP transport evidence. CLI/HTTP verification means exit 0 / HTTP 200, decoded 33×35 PNG, one step, seed 42, prompt `猫`; four malformed requests return 400. This is transport coverage, independent of the numerical gate above. CLI and HTTP PNG bytes also match for every completed row. Detailed commands, request bodies, timings and observed device submissions are in [matrix_results.json](matrix_results.json).

```sh
target/release-fast/rust-model-inference --model models/pjit-21k/pjit-FORMAT.gguf \
 --text-encoder models/Qwen3.5-4B/text-FORMAT.gguf --threads 8 --compute BACKEND \
 --prompt '猫' --width 33 --height 35 --steps 1 --seed 42 --out target/result.png --overwrite
target/release-fast/rust-model-server --model models/pjit-21k/pjit-FORMAT.gguf \
 --text-encoder models/Qwen3.5-4B/text-FORMAT.gguf --threads 8 --compute BACKEND --port PORT
# POST /v1/images/generations
# {"prompt":"猫","width":33,"height":35,"steps":1,"seed":42,"response_format":"b64_json"}
```

| Component | Storage | Requested backend | Implementation | CLI | HTTP | Numerical validation | Actual execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DiT | F32 | cpu | Implemented | Pass | Pass | 1.2946124e-05 / Pass | CPU SIMD/pool; F32 KV |
| DiT | F32 | auto | Implemented | Pass | Pass | 1.0935635e-05 / Pass | Vulkan projections/attention + host; 349 cumulative submissions |
| DiT | F32 | vulkan | Implemented | Pass | Pass | 1.1299469e-05 / Pass | Vulkan projections/attention + host; 487 cumulative submissions |
| DiT | F16 | cpu | Implemented | Pass | Pass | 0.00081502589 / Pass | CPU SIMD/pool; F32 KV |
| DiT | F16 | auto | Implemented | Pass | Pass | 0.0016752153 / Pass | Vulkan projections/attention + host; 349 cumulative submissions |
| DiT | F16 | vulkan | Implemented | Pass | Pass | 0.00096704771 / Pass | Vulkan projections/attention + host; 487 cumulative submissions |
| DiT | BF16 | cpu | Implemented | Pass | Pass | 6.0414086e-06 / Pass | CPU SIMD/pool; F32 KV |
| DiT | BF16 | auto | Implemented | Pass | Pass | 8.5494247e-06 / Pass | Vulkan projections/attention + host; 349 cumulative submissions |
| DiT | BF16 | vulkan | Implemented | Pass | Pass | 7.7380653e-06 / Pass | Vulkan projections/attention + host; 487 cumulative submissions |
| DiT | Q8_0 | cpu | Implemented | Pass | Pass | 0.05142132 / Fail | CPU SIMD/pool; F32 KV |
| DiT | Q8_0 | auto | Implemented | Pass | Pass | 0.03935427 / Fail | Vulkan projections/attention + host; 349 cumulative submissions |
| DiT | Q8_0 | vulkan | Implemented | Pass | Pass | 0.03935427 / Fail | Vulkan projections/attention + host; 487 cumulative submissions |
| DiT | Q4_0 | cpu | Implemented | Pass | Pass | 0.040798267 / Fail | CPU SIMD/pool; F32 KV |
| DiT | Q4_0 | auto | Implemented | Pass | Pass | 0.04466361 / Fail | Vulkan projections/attention + host; 349 cumulative submissions |
| DiT | Q4_0 | vulkan | Implemented | Pass | Pass | 0.039428394 / Fail | Vulkan projections/attention + host; 487 cumulative submissions |
| DiT | Q4_1 | cpu | Implemented | Pass | Pass | 0.0418282 / Fail | CPU SIMD/pool; F32 KV |
| DiT | Q4_1 | auto | Implemented | Pass | Pass | 0.043201844 / Fail | Vulkan projections/attention + host; 349 cumulative submissions |
| DiT | Q4_1 | vulkan | Implemented | Pass | Pass | 0.043201844 / Fail | Vulkan projections/attention + host; 487 cumulative submissions |
| DiT | Q4_K | cpu | Implemented | Pass | Pass | 0.03095753 / Fail | CPU SIMD/pool; F32 KV |
| DiT | Q4_K | auto | Implemented | Pass | Pass | 0.041085753 / Fail | Vulkan projections/attention + host; 349 cumulative submissions |
| DiT | Q4_K | vulkan | Implemented | Pass | Pass | 0.034642739 / Fail | Vulkan projections/attention + host; 487 cumulative submissions |
| DiT | Q5_K | cpu | Implemented | Pass | Pass | 0.044037838 / Fail | CPU SIMD/pool; F32 KV |
| DiT | Q5_K | auto | Implemented | Pass | Pass | 0.038663027 / Fail | Vulkan projections/attention + host; 349 cumulative submissions |
| DiT | Q5_K | vulkan | Implemented | Pass | Pass | 0.049144319 / Fail | Vulkan projections/attention + host; 487 cumulative submissions |
| DiT | Q6_K | cpu | Implemented | Pass | Pass | 0.044340449 / Fail | CPU SIMD/pool; F32 KV |
| DiT | Q6_K | auto | Implemented | Pass | Pass | 0.056909299 / Fail | Vulkan projections/attention + host; 349 cumulative submissions |
| DiT | Q6_K | vulkan | Implemented | Pass | Pass | 0.047473897 / Fail | Vulkan projections/attention + host; 487 cumulative submissions |
| Caption features | F32 | cpu | Implemented | Pass | Pass | 0.00011132561 / Pass | CPU SIMD/pool; F32 KV |
| Caption features | F32 | auto | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.00011132561 / Pass | CPU SIMD/pool; F32 KV |
| Caption features | F32 | vulkan | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.00011132561 / Pass | CPU SIMD/pool; F32 KV |
| Caption features | F16 | cpu | Implemented | Pass | Pass | 0.00118355 / Pass | CPU SIMD/pool; F32 KV |
| Caption features | F16 | auto | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.00118355 / Pass | CPU SIMD/pool; F32 KV |
| Caption features | F16 | vulkan | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.00118355 / Pass | CPU SIMD/pool; F32 KV |
| Caption features | BF16 | cpu | Implemented | Pass | Pass | 0.0057936852 / Pass | CPU SIMD/pool; F32 KV |
| Caption features | BF16 | auto | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.0057936852 / Pass | CPU SIMD/pool; F32 KV |
| Caption features | BF16 | vulkan | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.0057936852 / Pass | CPU SIMD/pool; F32 KV |
| Caption features | Q8_0 | cpu | Implemented | Pass | Pass | 0.019844337 / Pass | CPU SIMD/pool; F32 KV |
| Caption features | Q8_0 | auto | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.019844337 / Pass | CPU SIMD/pool; F32 KV |
| Caption features | Q8_0 | vulkan | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.019844337 / Pass | CPU SIMD/pool; F32 KV |
| Caption features | Q4_0 | cpu | Implemented | Pass | Pass | 0.021210382 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q4_0 | auto | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.021210382 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q4_0 | vulkan | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.021210382 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q4_1 | cpu | Implemented | Pass | Pass | 0.022507903 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q4_1 | auto | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.022507903 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q4_1 | vulkan | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.022507903 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q4_K | cpu | Implemented | Pass | Pass | 0.049956631 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q4_K | auto | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.049956631 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q4_K | vulkan | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.049956631 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q5_K | cpu | Implemented | Pass | Pass | 0.047343279 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q5_K | auto | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.047343279 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q5_K | vulkan | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.047343279 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q6_K | cpu | Implemented | Pass | Pass | 0.044786469 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q6_K | auto | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.044786469 / Fail | CPU SIMD/pool; F32 KV |
| Caption features | Q6_K | vulkan | CPU composition implemented; device extraction unimplemented | Pass | Pass | CPU only: 0.044786469 / Fail | CPU SIMD/pool; F32 KV |

Caption device extraction remains **not implemented (shared capability pending)** for all nine formats in Auto/Vulkan. The rows above describe the implemented CPU composition under those requested policies, rather than full GPU caption execution. The existing resident Qwen API exposes normalized final states/logits, not raw block 7/15/27 features; prefix recomputation stays local to P-JiT. DiT device kernels run while caption, norms, RoPE, elementwise work and sampling stay on CPU. A single-forward F32 device counter is 244 including one warmup, 217 projection submissions and 26 attention submissions (243 actual-forward submissions). This corrects the old classification of the warmup as an extra projection.

No-device checks: Auto F16 CLI and HTTP pass with an invalid ICD path and CPU execution. Strict Vulkan CLI and server both exit 2 with a device initialization error; no output image is created. Error/fallback handling does not count as device execution. JEV/probability scoring and generation streaming are not applicable to this pixel-image contract. IQ exports are not applicable because the repository has no corresponding quantization encoder.

## Default-size output and performance

The real default-size **1024×1024 / 50-step F16 DiT + BF16 caption CPU CLI completed** in 5428.649 s on eight threads (contended host, tracing enabled). Visual inspection found a recognizable red cube on a white table. This is one prompt, not a formal perceptual-quality evaluation.
The independent official same-GGUF replay completed all 50 steps: **all step gates pass**, final NRMSE 0.0196753612, cosine 0.999806970603. It starts with the identical traced Rust caption features and initial noise, decodes the stored F16 GGUF into F32, and checks DiT and Euler/APG again on each exact traced input. It does not certify native BF16 autocast or an independent caption-to-image pipeline.

| Step | Chain NRMSE | Cosine | Gate | Official batch-one/two NRMSE |
| --- | --- | --- | --- | --- |
| 1 | 5.4773456e-07 | 1 | Pass | 1.2846482e-08 |
| 10 | 6.3399517e-06 | 0.99999999998 | Pass | 4.8511185e-07 |
| 20 | 0.00028678923 | 0.999999958885 | Pass | 6.5584969e-06 |
| 30 | 0.0017389791 | 0.999998488694 | Pass | 4.3944902e-05 |
| 40 | 0.0052227453 | 0.999986385588 | Pass | 0.00012439797 |
| 50 | 0.019675361 | 0.999806970603 | Pass | 0.0004635121 |

Same-input DiT worst NRMSE 0.0014341983; independent same-input Euler/APG worst max_abs 9.5367432e-07. The original append-only trace also contained three interrupted runs. Only the final complete 153-record invocation was selected (2 captions, noise, 100 predictions, 50 states), without modifying raw tensor values. The selection manifest and all 50 metrics are retained in `matrix_results.json`.

Decoded final PNG comparison: 699894 / 1048576 pixels differ; RGB RMSE 1.1592347 on the 0–255 scale, max_abs 53, PSNR 46.8474 dB. These are observed pixel diagnostics, not a bitwise image acceptance gate.

Warm same-host release measurements for the identical F32 33×35 / 18-token / batch-one forward (one resident session, four repeats, median of the final three): CPU **0.771232 s**, Vulkan **3.058170 s**, official PyTorch ROCm **0.197731 s**. The timed official forwards have no checkpoint hooks or disk writes. Cold first samples and host-contention limits are explicit in [performance.json](performance.json). Vulkan remains slower; synchronous per-projection device transfers/submits and CPU work are part of this hybrid path. These are a performance reference, not a speed acceptance threshold.

The model-local projection chunk was increased from 64 to 256 rows to reduce synchronous device round trips at 1024px; shared kernels/API were not changed. A preserved real 1024×1024 first-forward fixture compares old/new chunking using identical F16 weights/text/noise. Its verification and the final default-size output are reported in the matrix snapshot. F32 default generation was interrupted after four steps because of its measured multi-minute step time; no completed F32 default image is claimed.

## Checks, review and remaining ownership

- Remote final model/CLI tests: 9 passed, 3 asset-dependent tests ignored; real asset tests are recorded separately above. Final inference/server release-fast and full release builds passed (7m51s / 18m25s); final F16 Auto CLI/HTTP smoke passed with equal PNG bytes and 349 observed device submissions. Four CLI-only server flags were rejected before loading components (exit 1, the existing backend-build error convention).
- Python converter/oracle helper checks: 7 passed, including cleanup and racing-output protection; touched Rust formatting and diff checks passed.
- Historical full library suite: 1651 passed / 47 failed / 230 ignored versus starting-base 1642 / 51 / 228. All failure names existed at that base. Final pre-rebase Vulkan suite: 1667 / 35 / 231, again no new failure names; [regression.json](regression.json) retains the lists.
- Clippy has ten deny-level errors in unchanged ERNIE/Pig/FunASR/Qwen-vision/YuE2/torch28 sources matching the starting base.
- CI on preceding runtime commit `4e4df7719`: all three checks passed. Final runtime `80ce7199f` also passed all three checks; re-check the docs/evidence head before merge.

Self-review repaired torch.linspace endpoint/FMA timestep rounding, APG F64 projection cast order, negative seeds/ignored CLI flags, HTTP request validation and cancellation permit lifetime, converter tensor payload validation and atomic no-overwrite publication, and 23-head CPU attention ordering. The Oracle now copies decoded parameters in place and rejects incomplete GGUF inventories; same-weight sampling can reload a GGUF without holding duplicate full-model states.

Remaining numerical owner: keep failing quantized full-forward/caption rows Experimental, inspect the first divergence and independent projection replays. Any demonstrated shared prepared-kernel defect requires a shared-compute follow-up with real old-model baselines; no shared arithmetic was altered here. Do not loosen thresholds after observing these failures.

Shared compute/Qwen owner, follow-up layer: expose raw multi-layer resident device features, validate all nine caption formats under CPU/Auto/Vulkan, then remove local three-prefix recomputation. This is the specific shared API gap preventing full-device adaptation.
