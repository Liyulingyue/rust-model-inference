# SenseNova Loop

The release checkpoint uses BF16 source projections, F32 mHC routing and YaRN
factor 2. The explicit `sensenova_loop` model reuses shared Qwen3.5 CPU SIMD and
Vulkan kernels, retaining independent state for all 56 visits of 32 physical layers.
See the [pinned contract and reproduction commands](../../tools/oracle/sensenova_loop/README.md).

Status on 2026-11-19: **Experimental / partial adaptation**. The
[108-row matrix](../../tools/oracle/sensenova_loop/release-matrix.csv) records
18 passed, 44 failed and 46 pending
combinations across nine text storage formats, three backends, text/image
generation and JEV. Implementation, CLI, HTTP and numeric verification are
separate. F32 text passes all 570 official state/logit gates on each backend,
with exact tokenizer and greedy IDs. Auto/Vulkan execute device submissions.
F16 and BF16 text/image each pass 570/172 gates on all three backends. Block
formats still have full-model gate failures; Q4_0 CPU text and device image
cases also have greedy divergence. Fixed-input quantized projection probes pass all 252 comparisons,
but do not clear those full-model failures. No numerical gate was widened.
F32 single-image greedy and final logits pass, but intermediate states exceed
the original gates. Injecting the exact native vision output into the official
decoder now passes all 172 checks after correcting GDN epsilon placement;
this diagnostic does not replace the failing independent image comparison.

The existing CLI/server expose generation, streaming chat and single/grouped
text/image JEV. SenseNova image chat honors the selected Jinja/custom template;
image bytes, order and marker expansion are preserved. Request validation runs
before SSE. Vision stays on CPU when the decoder uses Vulkan. One final-user
image is supported; historical/multiple images, audio/video and MTP are outside
this adapter. Decoder/projector source configurations must match. Tested context
is 512 with F32 KV; shared Vulkan attention currently caps capacity at 4096,
with Auto CPU fallback for larger capacities.

Same-host F32 text process times include loading and tracing, using 16 threads:

| Implementation | Backend | Seconds | Peak RSS KiB |
| --- | --- | ---: | ---: |
| official PyTorch | ROCm | 71.401 | 19,448,616 |
| native | cpu | 73.017 | 31,372,416 |
| native | auto -> Vulkan | 179.743 | 30,820,660 |
| native | vulkan | 170.249 | 30,156,988 |
| official PyTorch (recorded placement) | ROCm decoder; CPU input/head | 28.442 | 29,193,116 |

These observations come from `j-oxuoqo5tstkquvgqhepkaa5cm6`, SSH port 31667,
with gfx1100/RADV NAVI31. Earlier disconnected hosts are not mixed into this
table. Native constructs three sessions and captures device state; official
PyTorch loads once. Native text uploads 95,278,067,264 bytes and reads back
7,221,903,712 bytes for tracing. Other jobs share the host. These are process
wall times, not controlled throughput measurements. The earlier 518-second CPU
run overlapped exports/other jobs; its difference is not evidence of a GDN speedup.

Review repaired GDN normalization, tracing lock lifetime, grouped-image CLI JEV,
HTTP JEV/image generation template wiring, streaming/atomic quantizer publication,
Q4_K/Q6_K scale errors and reference CPU-offload materialization. Qwen3.5 real
CPU/GPU logits remain bitwise unchanged against the same-host baseline; the full
library retains the same 25 existing failures. Broader K-quant quality,
long-context and arbitrary image resizing remain unvalidated.

[Validation and review evidence](../../tools/oracle/sensenova_loop/release-validation.md),
[machine-readable results](../../tools/oracle/sensenova_loop/release-results.json)
and [timings](../../tools/oracle/sensenova_loop/performance.json) give the measured
scope and failures. Preview's historical evidence predates the GDN correction;
the corrected Preview path has not been rerun.
