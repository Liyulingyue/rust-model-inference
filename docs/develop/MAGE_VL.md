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
host KV formats work. Text attention/KV, the output head, Mage-ViT per-lane
RoPE/attention, normalization and activations still run on CPU. Consequently
`mage_vl` is intentionally absent from the complete-device whitelist.

No model weights were downloaded to the Mac. The replacement remote host
downloaded and verified the pinned release, then exported BF16 and Q8_0.
Q4_K conversion and full-weight Oracle checks were interrupted when SSH closed
all sessions. Full-model quantization quality, new CLI/HTTP/JEV runs and
full-model performance remain **unverified**. The 2026-11-03 BF16/F32 CPU
results are historical evidence, not a rerun of the current merged branch.

## Replacement AMD host attempt

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
The physical GPU has therefore not passed parity or performance validation.

All SSH sessions then closed and new connections were actively closed by the
endpoint. The cause of that disconnection is unknown; do not attribute it to
the shader failure or infer that interrupted processes finished successfully.
Remote logs are under `target/mage-remote-20261109/`. No complete inference
timing or native output was recovered, so this attempt adds no throughput or
quality result. Resume full-weight validation only after the host is reachable.

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
| Text BF16 | cpu | Not rerun; historical pass | Not rerun; historical pass | Implemented; CPU regression pass |
| Text BF16 | auto | Not run | Not run | Implemented; synthetic device pass |
| Text BF16 | vulkan | Not run | Not run | Implemented; synthetic device pass |
| Text Q8_0 | cpu | Not run | Not run | Implemented; synthetic decoder reference passes |
| Text Q8_0 | auto | Not run | Not run | Implemented; synthetic device pass |
| Text Q8_0 | vulkan | Not run | Not run | Implemented; synthetic device pass |
| Text Q4_0 | cpu | Not run | Not run | Implemented; shared converter/kernel tests pass |
| Text Q4_0 | auto | Not run | Not run | Implemented; shared linear device test passes |
| Text Q4_0 | vulkan | Not run | Not run | Implemented; shared linear device test passes |
| Text Q4_K | cpu | Not run | Not run | Implemented; synthetic decoder reference passes |
| Text Q4_K | auto | Not run | Not run | Implemented; synthetic device pass |
| Text Q4_K | vulkan | Not run | Not run | Implemented; synthetic device pass |
| Text Q5_K | cpu | Not run | Not run | Implemented; shared converter/kernel tests pass |
| Text Q5_K | auto | Not run | Not run | Implemented; shared linear device test passes |
| Text Q5_K | vulkan | Not run | Not run | Implemented; shared linear device test passes |
| Text Q6_K | cpu | Not run | Not run | Implemented; shared converter/kernel tests pass |
| Text Q6_K | auto | Not run | Not run | Implemented; shared linear device test passes |
| Text Q6_K | vulkan | Not run | Not run | Implemented; shared linear device test passes |
| Vision F32 | cpu | Not rerun; historical pass | Not rerun; historical pass | Implemented; synthetic reference passes |
| Vision F32 | auto | Not run | Not run | Mixed path implemented; one-block synthetic device pass |
| Vision F32 | vulkan | Not run | Not run | Mixed path implemented; one-block synthetic device pass |

Q4_1/IQ exports are outside the existing shared converter's format set. Full
device execution is not implemented: subsequent Mage/Vulkan work must add
Mage-ViT's per-lane RoPE and attention kernels, retain intermediates on device,
and calibrate the resident text path. This change implements the review's
projection-offload option rather than claiming full-device support.

## Current local evidence

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
quality validated. Current shared-model regression is limited to synthetic
tests because the replacement remote session was interrupted.
