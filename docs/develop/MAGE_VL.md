# Mage-VL-Cowork: quantization and projection offload

## Status (2026-11-09)

Partial adaptation. Text conversion and loading support F32, F16, BF16,
Q8_0, Q4_0, Q4_K, Q5_K and Q6_K. These are the existing shared converter
formats, not mixed Q4_K_M presets or new IQ algorithms. Vision weights and
norm vectors remain F32. The pinned release, tensor contracts and original
CPU Oracle evidence are in [validation.md](../../tools/oracle/mage_vl/validation.md).

`auto` and `vulkan` now use the shared `LinearExecutor` for text Q/K/V/O,
gate/up/down and vision patch/QKV/O/MLP/merger projections. Auto applies the
shared size threshold and falls back to CPU when needed. Strict Vulkan
requires an actual device and propagates projection failures. Both F16/F32
host KV formats work. Text attention/KV, Mage-ViT per-lane RoPE/attention,
normalization and activations still run on CPU. The output head uses shared
prepared-kernel dispatch, whose legacy wrapper can also offload it; it is
outside the explicit executor's strict projection guarantee. Consequently
`mage_vl` is intentionally absent from the complete-device whitelist.

No model weights were downloaded to the Mac. The current port-32276 EPYC/RADV
instance completed BF16/Q8_0/Q4_K exports and full-weight runs. BF16 CPU
production logits pass the official Oracle with F32/F16 KV; BF16 mixed GPU
probes pass with actual submissions. Q8_0/Q4_K independent same-weight logit
gates and quantized image CPU/GPU gates **fail**. Their image CLI CPU/Auto
answers also differ, failing the existing smoke check. Quantized formats remain
experimental; successful loading or matching eight Oracle tokens does not
override these failures. Gates were not widened. Per-case numbers and scope:
[real-weight-results.json](../../tools/oracle/mage_vl/real-weight-results.json).

## Historical interrupted AMD attempt (port 31599)

The host at SSH port 31599 has EPYC 9334, a 16-CPU cgroup quota, about 55 GiB
container memory and a 48 GiB AMD Radeon GPU (`RADV NAVI31`, gfx1100).
The non-traced Vulkan CLI/server/Oracle `release-fast` build completed.
The downloaded shards passed the pinned sizes and SHA256 values. BF16/Q8_0
export commands completed; independent accuracy checks did not complete.

Both Mage device tests failed with `context is lost` / Vulkan device loss.
The existing `linear_gpu_group_preserves_mixed_formats_across_batches` test
also failed on **unmodified main `532d051a0`**, returning CPU backends after
RADV hard recovery. Switching RADV to its LLVM compiler did not fix the
Mage vision failure. This is a baseline hardware/driver blocker, not evidence
of a successful AMD path; strict Vulkan correctly returned the device error.
That instance did not provide a GPU parity or performance pass. The next
instance's healthy GPU results supersede this as a current hardware blocker.

All SSH sessions then closed and new connections were actively closed by the
endpoint. The cause of that disconnection is unknown; do not attribute it to
the shader failure or infer that interrupted processes finished successfully.
Remote logs are under `target/mage-remote-20261109/`. No complete inference
timing or native output was recovered, so this attempt adds no throughput or
quality result. The completed port-32276 runs are recorded below and in
[validation.md](../../tools/oracle/mage_vl/validation.md#completed-replacement-host-runs-2026-11-09-port-32276).

## Coverage matrix

Each text row covers generation and text/image JEV through the shared text
decoder; image tasks also require the matching F32 vision component below.
Accumulation is F32 with the shared format-specific activation preparation;
the F16 export is a storage format, not a promise of F16 accumulation. CLI
generation/JEV and HTTP Chat/Responses/Anthropic/JEV are implemented for every
row. Their **full-weight validation** is listed separately for each entry.

| Component / weight | Policy | CLI full weights | HTTP/JEV full weights | Implementation / current numerical check |
|---|---|---|---|---|
| Text F32 | cpu | Not rerun; historical pass | Not rerun; historical pass | Implemented; CPU regression pass |
| Text F32 | auto | Not run | Not run | Implemented; synthetic device pass |
| Text F32 | vulkan | Not run | Not run | Implemented; synthetic device pass |
| Text F16 | cpu | Not run | Not run | Implemented; shared converter/kernel tests pass |
| Text F16 | auto | Not run | Not run | Implemented; shared linear device test passes |
| Text F16 | vulkan | Not run | Not run | Implemented; shared linear device test passes |
| Text BF16 | cpu | Generation/JEV pass | Generation/JEV and 40 regression assertions pass | Official Oracle: 64 production logit groups pass, F32/F16 KV |
| Text BF16 | auto | Text/image generation and JEV complete | Generation completes; text/image JEV CLI↔HTTP passes | Real-weight text/square GPU probes pass |
| Text BF16 | vulkan | Generation/JEV pass | 29-check smoke passes, including image JEV | Real-weight text/square GPU probes pass; mixed path |
| Text Q8_0 | cpu | Text/image generation and JEV complete | Generation completes; text/image JEV CLI↔HTTP passes | Independent F32-activation Oracle logit gate fails |
| Text Q8_0 | auto | Generation/JEV complete; image differs from CPU | Generation completes; text/image JEV CLI↔HTTP passes | Text logits equal CPU; image logit gates and smoke fail |
| Text Q8_0 | vulkan | Text/image generation and JEV complete; device traced | Generation completes; text/image JEV CLI↔HTTP passes | Text logits equal CPU; image logit gates fail |
| Text Q4_1 | cpu | Not run | Not run | Implemented; shared encoder + loader accept it, unvalidated |
| Text Q4_0 | cpu | Not run | Not run | Implemented; shared converter/kernel tests pass |
| Text Q4_0 | auto | Not run | Not run | Implemented; shared linear device test passes |
| Text Q4_0 | vulkan | Not run | Not run | Implemented; shared linear device test passes |
| Text Q4_K | cpu | Text/image generation and JEV complete | Generation completes; text/image JEV CLI↔HTTP passes | Independent F32-activation Oracle logit gate fails |
| Text Q4_K | auto | Generation/JEV complete; image differs from CPU | Generation completes; text/image JEV CLI↔HTTP passes | Text logits equal CPU; image logit gates and smoke fail |
| Text Q4_K | vulkan | Text/image generation and JEV complete; device traced | Generation completes; text/image JEV CLI↔HTTP passes | Text logits equal CPU; image logit gates fail |
| Text Q5_K | cpu | Not run | Not run | Implemented; shared converter/kernel tests pass |
| Text Q5_K | auto | Not run | Not run | Implemented; shared linear device test passes |
| Text Q5_K | vulkan | Not run | Not run | Implemented; shared linear device test passes |
| Text Q6_K | cpu | Not run | Not run | Implemented; shared converter/kernel tests pass |
| Text Q6_K | auto | Not run | Not run | Implemented; shared linear device test passes |
| Text Q6_K | vulkan | Not run | Not run | Implemented; shared linear device test passes |
| Vision F32 | cpu | Generation pass | BF16 generation/JEV pass | All 24 vision layers pass BF16 CPU diagnostics |
| Vision F32 | auto | Generation/JEV complete; quantized-text answers differ | Generation completes; three-format image JEV CLI↔HTTP passes | Real-weight submissions; quantized-text image logit gates fail |
| Vision F32 | vulkan | Three-format generation/JEV complete; device traced | BF16 smoke passes; quantized image JEV CLI↔HTTP passes | BF16 logit probes pass; quantized-text image gates fail |

IQ exports are outside the existing shared converter's format set. Q4_1 is
exported by the shared encoder and accepted by the loader, but no full-weight
run backs it, so it stays unvalidated. Full device execution is not implemented: subsequent Mage/Vulkan work must add
Mage-ViT's per-lane RoPE and attention kernels, retain intermediates on device,
and calibrate the resident text path. This change implements the review's
projection-offload option rather than claiming full-device support.

## Current interface and effect evidence

The additional JEV matrix passes 16 CLI↔HTTP probability comparisons: text and
image for BF16 CPU/Auto, and Q8_0/Q4_K CPU/Auto/Vulkan. Maximum difference is
zero against `<1e-6`; BF16 strict-Vulkan JEV is covered by its 29-check smoke.
Four separate quantized strict-Vulkan CLI generation requests (two formats ×
text/image) exit zero and show device projections. This is interface evidence
under each policy. Quantized image CPU/Auto generated answers still differ,
and the numerical gates above remain failed.

Ordinary BF16/Q4_K CPU chat repeats thinking and reaches the 512-token cap for
both arithmetic and image questions. The red/blue image's blue square is
misidentified as a rectangle. A computer-use sample uses the pinned model
card's hybrid system prompt, explicit HTTP tools and a synthetic note editor.
Both formats describe Save correctly and say to click it, but actually return
`computer_use {"action":"screenshot"}` without click coordinates. No action
is executed and the save task is not completed. BF16 takes 256.732 s (1964
prompt / 164 completion tokens); Q4_K takes 257.944 s (1964 / 169) on the
shared host. These single CPU/F32-KV requests use context 4096, a 512-token
cap and temperature zero; they are neither isolated benchmarks nor OSWorld
validation. The checkpoint's ordinary chat and agent-task quality remain
unestablished. Exact JEV values and full GUI responses are in
[real-weight-results.json](../../tools/oracle/mage_vl/real-weight-results.json).

## Current full-weight request performance

EPYC/RADV host above, eight threads, `release-fast`, ordinary non-trace HTTP
server, F32 KV, prefill batch 32, context 1024, temperature zero. Each request
is capped at 128 completion tokens (including reasoning); text input has 23
prompt tokens, the synthetic 128×128 red/blue image has 40. Times include
prefill/image encoding/generation, exclude server startup, and are **request
throughput**, not isolated decode speed. CPU has two samples per kind;
Auto/Vulkan have one. The instance is shared, so these are bounded probes.

| Text weight | Policy | Text seconds / token/s | Image seconds / token/s |
|---|---|---:|---:|
| bf16 | cpu | 12.137 / 10.55 | 14.508 / 8.82 |
| bf16 | auto | 36.833 / 3.48 | 36.836 / 3.47 |
| bf16 | vulkan | 86.651 / 1.48 | 84.814 / 1.51 |
| q8_0 | cpu | 9.825 / 13.03 | 12.319 / 10.39 |
| q8_0 | auto | 72.463 / 1.77 | 75.886 / 1.69 |
| q8_0 | vulkan | 87.527 / 1.46 | 90.837 / 1.41 |
| q4_k | cpu | 7.386 / 17.33 | 9.476 / 13.51 |
| q4_k | auto | 72.198 / 1.77 | 89.848 / 1.42 |
| q4_k | vulkan | 79.643 / 1.61 | 98.959 / 1.29 |

CPU is faster on this host. A fresh BF16 text session's trace records about
8.04 GB of static uploads; block projections return intermediates for host
attention, RoPE and postops. The legacy output-head dispatcher can also submit
device work during decode. No isolated per-kernel timing breakdown was taken;
these observed uploads/round trips and missing resident operators explain why
this mixed implementation must not be advertised as a GPU speedup.

CPU whole-CLI wall times (two samples, same 128-token cap) are BF16 text
12.750–14.971 s / image 14.973–15.274 s; Q8_0 10.173–10.898 / 12.033–12.073 s;
Q4_K 7.536–7.688 / 9.697–9.779 s. These include process/model startup.

The same-host official CPU Oracle diagnostic takes 31.038 s text prefill and
16.665 s for seven decode steps; native BF16 CPU takes 0.909 / 0.677 s in its
probe. The Oracle uses F32-storage Torch kernels and prefill checkpoint hooks;
this single diagnostic observation is not a controlled production benchmark.
The native probe's text timer excludes image encoding and loading. Raw request
samples, usage, timing scope and historical results are in
[performance.json](../../tools/oracle/mage_vl/performance.json).

## Historical local hardware evidence

Apple M3 Max hardware Vulkan, macOS 27.2, repository `release-fast`, two CPU
threads, no parity trace/scalar override. `RUST_GPU_MIN_OUTPUTS=0` forces Auto
offload for the small synthetic fixtures; production retains the shared
threshold. These are constructed tensors, not the released model weights.

- Text: one 256-wide block, eight output logits, three-token prefill in chunks
  of two followed by two one-token decode steps. F32/BF16/Q8_0/Q4_K × F32/F16
  KV × Auto/Vulkan all submit device work at every step. The existing logit
  gate is `abs(error) <= 0.002 + 0.002 * abs(cpu)`. Worst max_abs: F32
  `8.046627e-6`, BF16 `9.536743e-6`, Q8_0/Q4_K `0`. This compares CPU/device
  execution of the same synthetic weights, not quantization loss against BF16.
- Vision: real widths, one synthetic block, a 4×4 patch grid, seven projections
  including biases. Both device policies submit seven operations; final
  max_abs `0` against a `<0.002` gate. A final CPU rerun submits no GPU work
  and reproduces the original CPU result.
- Diagnostic vision wall time: CPU **121.57 ms**, Vulkan **411.80 ms**, Auto
  **331.22 ms**. Each encode creates a fresh executor and uploads its weights;
  transfers and small batches dominate this mixed path. Single samples include
  setup/uploads and are not model throughput or a speedup claim. No independent
  PyTorch/llama.cpp timing was run. Machine-readable scope:
  [performance.json](../../tools/oracle/mage_vl/performance.json).
- Shared batched-linear test: F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K,
  rows 1/3/64, device submission and weight-cache checks pass. Existing Qwen
  hidden-row/capture device tests pass. No shared numerical kernel changed.

The newly merged Nemotron FP64 shaders initially prevented **all** operator
pipelines from loading on M3 Max. The unchanged merge baseline reproduces the
same failure. Optional FP64 slots now use the existing placeholder pattern;
their recorders reject missing `shaderFloat64` before dispatch. FP64-capable
devices keep the original shader selection; that device branch was not run.

## Reproduction

```bash
./.venv/bin/python -m pytest -q tools/converter/mage_vl/test_convert_mage_vl.py tools/converter/utils/test_quantize_gguf.py tools/converter/utils/test_kquants.py
env -u RMI_SCALAR -u RMI_PARITY_TRACE RUST_GPU_MIN_OUTPUTS=0 cargo test --locked --profile release-fast --features vulkan --lib mage_projection_session_runs_quantized_prefill_and_decode_on_device -- --ignored --test-threads=1 --nocapture
env -u RMI_SCALAR -u RMI_PARITY_TRACE RUST_GPU_MIN_OUTPUTS=0 cargo test --locked --profile release-fast --features vulkan --lib mage_vision_projections_submit_and_preserve_cpu_rope_attention -- --ignored --test-threads=1 --nocapture
env -u RMI_SCALAR -u RMI_PARITY_TRACE cargo test --locked --profile release-fast --features vulkan --lib batched_linear_device_rows_match_and_reuse_weights -- --ignored --nocapture
```

Raw local logs and the exact baseline failure comparison are in the ignored
`target/mage-review-20261109/` directory. No new binary target was added.
For real weights, run the [Oracle entry checks](../../tools/oracle/mage_vl/README.md)
on a weight host for each format with CPU, Auto and Vulkan; compare quantized
outputs with an independent same-format/dequantized Oracle before declaring
quality validated. Current shared-model regression remains limited to
constructed fixtures; real weights of older models were not retested.
