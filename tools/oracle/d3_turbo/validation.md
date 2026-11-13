# d3-turbo validation

Status: **Experimental / partial adaptation**. A successful entry or projection
probe is recorded separately from whole-graph numerical acceptance. Failed
thresholds below were not relaxed.

## Fixed contract and assets

- ModelScope `vllm-sr/d3-turbo` revision
  `0c9281c53740765406dcec7f20140179838a6fa5`.
- Eleven original BF16 shards: 52,170,802,640 bytes; downloaded and SHA256-verified
  directly on the authorized model host. Readout SHA256:
  `f2579067a488138e2ef00bb3e98989cac7311afbf48700dfad02244e3671f241`.
- The per-file source manifest and converted-file hashes are in the small
  accompanying reports. Weights, GGUFs and checkpoint payloads stay on the model
  host; no model was downloaded to the Mac.
- Qwen3.5 dense 27B, 64 layers, hidden 5120, FFN 17408; every fourth layer full
  attention, the others GatedDeltaNet. Prefix-bidirectional state, causal
  question, last/readout pooling, 255 F32 code rows, temperature 1.
- Interleaved mRoPE `[11,11,10]`, theta 10,000,000; vision uses temporal patch
  pairs. This standalone revision is distinct from the HF turbo pin used by
  the older Vela integration.
- No causal LM head, generation, KV continuation or streaming generation is
  part of this decision checkpoint. Those are not applicable. Long-context
  and multiwindow scale acceptance remain unrun, not inapplicable.

## Environment and measurement boundaries

2026-11-14, authorized Linux x86_64 host (`d3-navi31`): EPYC 9334,
16-core cgroup CPU quota, 59,055,800,320-byte memory cgroup, RADV NAVI31 / gfx1100
with 49,136 MB VRAM. Rust 1.98.1, `release-fast`, features
`vulkan,parity-trace`. CPU Oracle: Torch 2.10.0+cpu, Transformers 5.17.0,
NumPy 2.1.2, external `gguf` 0.17.1. Torch GPU diagnostic environment: ROCm
7.2.1 / Torch 2.9.1. All environments/toolchains are project-local.

The native original directory and GGUF BF16/F32 use F32 activations and
accumulation. F16 prepared projections round activations to F16; Q8_0/Q4_0/
Q4_1 use Q8 block activations, K formats Q8_K. Visual/readout and sensitive
auxiliary tensors remain F32 in the exports. No native BF16 acceleration or
full device residency is claimed.

## Independent references and gates

`trace.py` loads the pinned official runtime/Transformers graph. F32 diagnostics
stream active layers to F32 without holding a second full F32 model. GGUF
references use the independent external `gguf` dequantizer and the same official
graph, with CPU activation rounding added explicitly; they do not call the Rust
weight decoder. Q4_1 additionally models its separately rounded Q8_1 sum.

- Token IDs, positions, checkpoint order, shape and count: exact.
- Floating arithmetic: `atol=5e-4`, `rtol=1e-4` at each checkpoint.
- Lossy CPU activations: relative RMSE ≤ 0.02 and cosine ≥ 0.9998; logits also
  require `atol=0.05`, `rtol=0.01` and identical argmax.
- Same-input isolated projections: `atol=5e-4`, `rtol=1e-4`.
- Prompt/layout goldens and sampled frame indices are discrete contracts.
  Box distribution/calibration goldens use a numerical 1e-6 envelope.
- Additional product checks, fixed before examining the spatial model outputs:
  box edges within two 128-bin steps (0.015625), confidence within 0.02,
  visibility within 0.01, identical timestamps/track length and usage. Product
  acceptance does not override a failed intermediate-checkpoint gate.
- For the separate official-default BF16 CPU versus native F32 CPU text-choice
  product check: identical option labels, selected choice and usage, probability
  and confidence absolute difference ≤ 0.02. This gate is fixed before the
  official-default run and does not change any same-compute-precision gate.

These are implementation comparisons against the same stored weights, not a
broad evaluation of quantization quality against the original model. Lossy
arithmetic is not required to be bitwise equal.

## Results established before the matrix run

Original CPU text, three questions sharing one English/Chinese state:
585 floating checks passed and all three token-ID sequences matched exactly.
Rust completed in 520.95 s (12 threads, max RSS 47,643,276 KiB); official F32
forward took 151.75 s. Conversions ran concurrently and reference/native thread
counts were not controlled as a benchmark: do not infer a speed ratio.

The original CPU image request completed after two visual fixes: BF16 auxiliary
vectors are explicitly widened and F32 temporal patch inputs remain F32.
Projected features passed the unchanged gate (max abs 0.000507355, RMSE
0.0000021624; the relative term accounts for the largest value). Final logits
passed (max abs 0.000009537). The full 197-check graph still failed 68 checks,
first at layer 41 output; worst relative RMSE 0.0000442024. An earlier run had
195 failures due to the visual precision issue; that earlier result is
superseded, not treated as passing.

Q8_0 three-question text completed. After correcting the independent reference's
scalar/Tensor reciprocal multiplication to direct F32 division, 172 of 585
intermediate checks still failed. First failure: layer 44 input norm, relative
RMSE 0.02137452 / cosine 0.99977157. Worst: layer 58 input norm, relative RMSE
0.04357723 / cosine 0.99905035. All three final logit checks passed (max abs
0.003994 / 0.041415 / 0.034006). The whole-graph result remains **failed**.

The final isolated same-input projection probe passed all nine GGUF formats:
BF16, F32, F16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K. Max abs was
0.000034333 or below on that fixture. This narrows the Q8 whole-graph
issue; it does not prove the 64-layer graph passes or establish a root cause.

A follow-up same-input Q8_0 projection probe also passed at layers 0, 44 (the
first whole-graph failing checkpoint) and 58 (the worst reported checkpoint).
Max abs was 0.000026703 / 0.000008583 / 0.000005722 respectively. These
[late-layer probes](late-layer-projections.json) use the independent external
dequantizer and activation rounding; they narrow the investigation without
changing the whole-graph failure or establishing its cause.

Original CPU video passed all 198 floating checks after restoring the outer
`vision_start`/`vision_end` pair that the official processor retains. Both
token sequences contain 120 identical IDs, and all three position axes match
exactly. Final logit max abs was 0.000005722. The previous 118-token run compared
different inputs and is superseded. This single short-clip pass does not prove
spatial tracking, multiwindow or every format/media combination.

Original CPU video box completed with identical 169-token input, positions,
usage and timestamp. All four final logit checks passed (max abs 0.000020981).
Coordinates differed by at most 0.000000264, and confidence matched exactly,
so the fixed spatial product checks passed. One of 201 internal checks failed
at layer 63 output (max abs 0.004516602, relative RMSE 0.0000023583).
Whole-graph acceptance therefore remains failed. This run includes a brief
administrative trace-cleanup pause and is excluded from performance comparisons.

Original CPU tracking passed all 402 floating checks. Track length, timestamps
and usage matched exactly; maximum edge difference was 0.000000346, visibility
difference 0.000000259 and confidence difference zero. Both the internal and
fixed spatial product gates passed on this short fixture. Compilation and other
validation work overlapped the run, so its wall time is diagnostic only.

## Real CPU entries

All nine converted formats completed real CPU CLI and HTTP requests with exit
0 / HTTP 200 and identical native responses. The original directory's CPU
entry also passed. The fixed single-question input has SHA256
`5d5ee51db2e9154308ceca9d147606bc238f69615d7e1a39202b4468154d8e61`;
its byte-identical fixture is `one.json`. These execution results are separate
from whole-graph numerical acceptance.

## Device failure and shared dependency

Strict Vulkan runs of the original BF16 directory, F16 GGUF and Q8_0 GGUF failed
with device loss at `language_model.layers.0.linear_attn.in_proj_qkv.weight`.
Initialization and warmup succeeded, but inference did not. The repository's
existing ignored `batched_linear_device_rows_match_and_reuse_weights` probe also
failed on its first F32 format, independently of D3's model graph. A same-host
ROCm Torch matrix diagnostic succeeded for F16 and F32; this is only a hardware
probe, not a complete reference model benchmark. The complete official BF16 GPU
load exceeded the cgroup memory budget before inference and was stopped.

The adapter uses existing eligible Vulkan projections, with scan, norms,
softmax and small projections on CPU. Complete working GPU execution remains
**unimplemented pending the shared projection/device-loss repair (layer 3)**;
correct strict errors and Auto fallback do not count as GPU acceptance. Owner:
shared Vulkan/kernel maintainers, followed by the D3 adapter matrix rerun.

## Boundaries and self-review

Cross-model changes are limited to facilities already shared with Vela D3 and
Qwen3.5 vision: an explicit standalone constructor with F32 execution, generic
TensorSource storage, optional media/readout interfaces, quantized projection
activation preparation and an opt-in F32 temporal patch path. Old Vela APIs and
normal CPU/BF16 defaults remain unchanged. The strict projection error check
also makes an existing Vulkan failure explicit for legacy D3 callers.

The model-independent decision facade dispatches THX-02 and D3 explicitly;
CLI/HTTP orchestration uses that facade. No model-named binary, new dependency
or external Rust acceleration library was added.

Self-review corrected model-specific JEV prevalidation, local-file acceptance
through named HTTP assets, the inline-video body limit and demux protocols,
FFmpeg 9's removed `-vsync` option, actual readout F32 validation, and spatial
video duration bounds. It also restored the outer video vision tokens, fixed
the reference runtime's lazy sibling-module import, and corrected independent
F16 activation and Q4_1 sum semantics rather than weakening their projection
checks. Lossy embeddings use numerical tolerances; only discrete contracts
require exact equality.

The final library run (`lib-final11.log`) has 1653 passed, 34 failed and 232
ignored, compared with 1644/34/230 on the unchanged base with identical features.
The 34 failure names are identical; no new failing test was introduced. The
real FFmpeg fixture test passed (one test actually executed), including Unix
0600 permissions and temporary-file cleanup. The CLI integration check passed
against the final rebuilt executable. Seven converter and trace-acceptance
Python tests passed. Cargo metadata confirms the same nine production binaries
as the base. The initially unavailable Clippy component was streamed into the
project-local toolchain and SHA256-verified. The actual Clippy run failed with
10 errors; a fresh check of the unchanged base has the same 10 error signatures
and no new error was introduced. No lint level was disabled. The detailed
baseline comparison is in `engineering.json`; this is not a clean lint pass.

Post-PR review also repaired conversion publication: a file created after the
initial existence check can no longer be overwritten, and a competing partial
file is preserved. The injected race check fails against the original PR
converter; the repaired converter passes all seven Python checks, including
all nine small-format round trips and serial/parallel K-quant equality. This
changes publication only, not the existing exported model payloads.

Real legacy Vela model assets are unavailable on this host: small graph tests
and compilation are useful but are **not** a real legacy inference regression.

## Coverage and remaining layers

The accompanying expanded matrix must retain every capability × format ×
backend row, separately recording implementation, CLI/HTTP execution and
numerical acceptance. Missing runs are not passing rows. See the per-format
entry report for commands, request hashes, actual device evidence and results.

Machine-readable evidence: [asset hashes](assets.json),
[engineering and baseline](engineering.json), [numerical gates](numerical.json),
[spatial products](products.json), [real entries](entries.json),
[official-default product comparison](official-product.json),
[CPU/Auto trace comparisons](device-comparison.json),
[510-row matrix](coverage.csv), [performance](performance.json) and
[snapshot status](report-status.json). The snapshot explicitly records whether
the sequential remote validation queue has finished; an in-progress snapshot
does not establish acceptance of its remaining runs.

Layer 2 owns remaining whole-graph numerical gates, unrun media/format/JEV
combinations, large context/multiwindow acceptance and real legacy-model
regression. Layer 3 owns the shared GPU repair and subsequent complete Vulkan/
Auto correctness/performance runs. Only after these layers pass can this model
be called fully adapted.
