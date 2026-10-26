# Sage-A1.5 Q8_0

[ModelScope artifact](https://modelscope.cn/models/InternScience/Sage-A1.5-Q8_0-GGUF/files).
Latest review follow-up: [functional acceptance and current test boundaries](#review-fixes-and-functional-acceptance),
with machine-readable evidence in [review-fixes.json](review-fixes.json).
The GGUF declares `qwen35moe`; it reuses the existing hybrid decoder, not a new
architecture inferred from the model name. The public `Qwen35Config::from_source`
now selects the matching metadata prefix, as the model loader already did.
The shared ARM NEON vision LayerNorm also advances by its actual four-float
load width; previously it skipped half of every eight values.
MoE routing now uses the existing precise softmax and F64 accumulation for
the selected-probability sum, matching the pinned Oracle's reduction contract.

## Artifact contract

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `Sage-A1.5-Q8_0.gguf` | 37802150112 | `f7247f6faef3f6c616b953595e1fc7c3fbf257d02d9915af5fdf15e808ea8ee4` |
| `Sage-A1.5-mmproj.gguf` | 899282880 | `51cd4410057e8dc4152ca771a726fbfb34856f047e7049018c757d42ddd481f2` |

The language GGUF contains 443 Q8_0 and 310 F32 tensors: hidden size 2048,
vocabulary 248320, 256 experts / top-8, shared FFN 512, 16 query heads / 2 KV
heads / head width 256. Every fourth layer uses full attention; the others use
Gated DeltaNet. RoPE sections are `[11, 11, 10, 0]`, dimension 64, base 1e7.
`block_count=41` includes one `nextn_predict_layers` MTP block. Main decoding
executes **40** blocks; the draft head is not executed.

Tokenizer: GPT-2 / `qwen35`, BOS 248044 with `add_bos_token=false`, EOS 248046.
Use `--jinja` to select the artifact's own chat template. Its optional
verification protocol stays at the template default (disabled).

The companion projector declares `clip` / `qwen3vl_merger`, F16, output 2048,
27 blocks / hidden 1152 / 16 heads, patch 16 and spatial merge 2.

## Reproduce

The Oracle is a temporary instrumented copy of llama.cpp commit
`b96806d96061049a5b574269b049bf6241d63d46`. The input checkout remains read-only;
the shared builder also accepts a tarball with matching pinned model blobs.
BLAS, Accelerate, Metal and other external accelerators are disabled. The
builder also disables SSE4.2/AVX kernels, baseline SSE2/NEON activation and
reduction branches, and FMA contraction;
`compare.py` clears `RMI_SCALAR` to exercise default Rust dispatch. The
`qwen35moe` decoder selects reference arithmetic for each synchronous forward
and propagates it to its compute workers, restoring the previous thread-local
setting afterward. Other architectures retain their normal dispatch. This
prevents small reassociations from crossing Q8 activation boundaries and
changing later expert selection; it does disable explicit CPU SIMD kernels
for this architecture. The Vulkan decoder still uses packed integer dots. The lossy comparisons
below use numerical bounds, including on this reproducible scalar CPU path.

Set `RMI_QWEN35_MOE_SIMD=1` before starting a process to opt this architecture
back into native SIMD/FMA dispatch on CPU and native arithmetic on Vulkan.
The default remains reference arithmetic; `RMI_SCALAR` still forces scalar
dispatch even with the opt-in. The setting is read once per process and is
shared by CPU and Vulkan forwards. Native arithmetic can cross Q8 activation
boundaries, change expert selection and generated tokens, and exceed the
reference-mode error bounds. The opt-in is not an aligned performance claim.

```sh
sh tools/oracle/sage_a15/build_oracle.sh /path/to/pinned/llama.cpp target/sage-oracle
cargo test --profile release-fast --features parity-trace --test sage_a15 --no-run

RMI_SAGE_MODEL=/path/to/Sage-A1.5-Q8_0.gguf \
RMI_SAGE_TOKENIZE="$PWD/target/sage-oracle/build/bin/llama-tokenize" \
cargo test --profile release-fast --test sage_a15 sage_contract_and_tokenizer -- --ignored --exact

python3 -m unittest discover -s tools/oracle/sage_a15 -p test_compare.py
python3 tools/oracle/sage_a15/compare.py \
  --model /path/to/Sage-A1.5-Q8_0.gguf \
  --oracle target/sage-oracle/build/bin/llama-eval-callback \
  --rust-test target/release-fast/deps/sage_a15-REPLACE_WITH_EXECUTABLE_HASH \
  --out target/sage-parity
```

`compare.py` refuses a file with the wrong size or SHA-256. The Oracle uses CPU; Rust uses CPU by default or the forced Vulkan decoder with
`--backend vulkan`. Both use eight CPU threads, F32 KV, sequential single-token
prefill and four greedy predictions. This avoids comparing padded SSM ubatch rows with real tokens.
Oracle weight repacking is disabled to keep the Q8_0 path explicit and avoid a
second full weight allocation.
Inputs are raw `Hello` and a Chinese user turn rendered with the GGUF's default
non-thinking template. Traces and per-checkpoint metrics remain under `--out`.

Each processed token requires 55 checkpoints: input embedding, all 40 layer
outputs, final norm/logits, plus router logits/probabilities/top-8/normalized
weights at layers 0, 3 and 39. Missing or duplicate keys, mismatched logical
shapes, truncated data and non-finite values fail. Tokens, dequantized embedding
bits and selected expert IDs/order must agree exactly. Quantized matrix
products, reductions, normalization and transcendental functions use relative
L2 <= 0.01 and cosine >= 0.99995; logits additionally require max absolute error
<= 0.5. These are numerical bounds, **not** a claim of bitwise graph parity.
The comparator reports the first failing selected checkpoint in graph order.

## Verification limits

### Local ARM64 results

Both artifacts passed complete-file size and SHA-256 checks. The real-model
contract, five tokenizer inputs (including Unicode and special tokens), and
default non-thinking Jinja rendering passed against the pinned tokenizer.

| Input | Processed tokens | Checkpoints | Four greedy token IDs |
| --- | ---: | ---: | --- |
| Raw `Hello` | 4 | 220 | `[11, 271, 40, 1044]` |
| Chinese templated user turn | 16 | 880 | `[109266, 6115, 98691, 101009]` |

All 1,100 selected checkpoints passed: exact embeddings and expert IDs/order,
and bounded floating-point values. The observed scalar errors were zero;
the harness intentionally retains numerical assertions for lossy operations.
The small non-ignored MoE fixture exercises production routing and fails with
the previous F32 reductions, without requiring model files.

Default ARM CPU now also passes the same 1,100 selected checkpoints without
`RMI_SCALAR`: observed relative L2 and absolute error 0, cosine 1, with the
same exact token/embedding/expert checks. This replaces the earlier default
NEON result, which exceeded the bounds (relative L2 0.05289 / 0.11320 and
maximum logit absolute error 0.50174 / 1.16432 for raw / Chinese).
Vision encoding remains on its existing default dispatch and has CLI smoke
coverage only, without numerical Oracle comparison.

Example CLI usage (set `SAGE_MODEL_DIR` to the artifact directory):

```sh
target/release-fast/rust-model-inference \
  --model "$SAGE_MODEL_DIR/Sage-A1.5-Q8_0.gguf" \
  --jinja --no-thinking --prompt '你好' --temp 0 --threads 8
target/release-fast/rust-model-inference \
  --model "$SAGE_MODEL_DIR/Sage-A1.5-Q8_0.gguf" \
  --mmproj "$SAGE_MODEL_DIR/Sage-A1.5-mmproj.gguf" --image /path/to/image.png \
  --jinja --no-thinking --prompt '这张图片的主要颜色是什么？' --temp 0 --threads 8
```

### Remote Linux x86_64 results

Both complete artifacts were transferred to the remote host and passed the
same size/SHA-256 checks. The real-model contract and tokenizer/template test
also passed there against its locally built pinned Oracle.

| Input | Processed tokens | Checkpoints | Four greedy token IDs |
| --- | ---: | ---: | --- |
| Raw `Hello` | 4 | 220 | `[11, 353, 1044, 4313]` |
| Chinese templated user turn | 16 | 880 | `[109266, 6115, 98691, 101009]` |

All 1,100 selected checkpoints passed with exact embeddings and expert
IDs/order. The observed floating-point errors were zero (maximum relative L2
and absolute error 0, minimum cosine 1); lossy operations still use the stated
numerical assertions. Each host matches its **same-host Oracle**. The differing
raw greedy sequences do not establish cross-ISA token or bit identity.

Preparing this scalar Oracle required disabling both `GGML_SSE42` (which also
enables SSE3 reductions) and the baseline `__SSE2__` activation branch. Leaving
the latter enabled first diverged at SiLU after an identical recurrent
convolution output; the small discrepancy then entered recurrent state. The
builder selects the pinned implementation's scalar branches without changing
model formulas or comparison thresholds.

Earlier default-dispatch CLI smokes passed on this host for built-in chat wrapping,
the GGUF Jinja template, and the F16 projector identifying a solid-red image
as red. Those smokes establish CLI execution only. The completion validation below
separately checks default text dispatch; vision outputs still lack numerical
Oracle comparison.

### Vulkan MoE decoder

The resident Qwen3.5 executor now also records `qwen35moe`: F32 router,
softmax/top-k/normalization, Q8_0 or F32 routed expert projections, scalar-reference SiLU,
shared expert with sigmoid gate, and residual reduction. Expert IDs remain on
the GPU and index the uploaded contiguous expert tensors; decoding does not
copy selected weights back through the CPU. Unsupported expert formats and
incomplete or malformed expert/shared projections are rejected before upload.
The main decoder still executes 40 layers, excluding the MTP block.

Recurrent SiLU now follows the existing exact/NEON/AVX2 dispatch, including
`RMI_SCALAR=1`. F32 matrix products use compensated accumulation in scalar
Oracle mode, preserving the CPU's wide sum of F32-rounded products without
requiring GPU float64 support. Recurrent state and attention scalar dots also
use compensated sums. The shared F32 shader also packs 64 independent reference outputs into each
workgroup; AVX2 mode packs four outputs and follows its two eight-lane FMA
streams and horizontal sums. An adversarial cancellation fixture failed on
the old shader (0 instead of 1) and passes on the physical AMD adapter in
default and reference modes. Tail widths and grouped outputs are exercised.

On Linux x86_64 GNU builds, a shader specialization selects the same
table/polynomial `expf`/`logf` approximations as the host reference. A correctly
rounded exponential can differ from that libm by half an ULP and cross a later
Q8 rounding boundary. `host_libm.glsl` evaluates the approximations with two
F32 components, based on the MIT-licensed ARM optimized routines pinned in its
header. Other hosts retain the existing exponential path. This calibrates the
validated host; it does not promise identical libm behavior on other systems.

The test device is physical gfx1100 (`AMD Radeon Graphics (RADV NAVI31)`),
with approximately 48 GiB VRAM. Select its hardware ICD explicitly:

```sh
cargo test --profile release-fast --features vulkan,parity-trace \
  --test sage_a15 --no-run
RMI_QWEN35_MODEL="$SAGE_MODEL_DIR/Sage-A1.5-Q8_0.gguf" \
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
  cargo test --profile release-fast --features vulkan,parity-trace \
  --lib vulkan::qwen35::tests:: -- --include-ignored --nocapture --test-threads=1
python3 tools/oracle/sage_a15/compare.py \
  --backend vulkan --model "$SAGE_MODEL_DIR/Sage-A1.5-Q8_0.gguf" \
  --oracle target/sage-oracle/build/bin/llama-eval-callback \
  --rust-test target/release-fast/deps/sage_a15-REPLACE_WITH_EXECUTABLE_HASH \
  --out target/sage-vulkan-parity
RMI_SAGE_MODEL="$SAGE_MODEL_DIR/Sage-A1.5-Q8_0.gguf" \
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
  cargo test --profile release-fast --features vulkan,parity-trace \
  --test sage_a15 sage_vulkan_batched_prefill_and_decode -- --ignored --exact --nocapture
```

Forced Vulkan session creation must succeed; the trace helper prints
`Sage trace backend: vulkan` and cannot silently substitute CPU execution.
GPU trace mode uses one-token chunks to retain the token-major schema; the
separate batched test runs without tracing and compares 44-token prefill=64
and 64 teacher-forced predictions against CPU prefill=1 and prefill=64.
Every full logit vector must satisfy the original numerical bounds, and the
greedy token IDs must match. Structural checks and numerical thresholds
are identical to the CPU harness above. The router fixture contains F32 bit
patterns from layer 0, token 0 of the same pinned scalar Oracle case; it
reproduces the wide-sum softmax boundary without loading the full model.

The physical adapter suite passed all 21 tests, including synthetic batched
prefill/decode, F32 cancellation, malformed payload/format guards, the real
router fixture, 60 real SSM projections and 40 independent real-weight MoE
blocks. The MoE blocks passed exact selected IDs/order and bounded
routed+shared outputs against production CPU operators with observed maximum
absolute error 0. The separate SiLU device regressions include libm boundaries,
NaN/infinities and input preservation.

With `RMI_SCALAR=1`, forced Vulkan passed all 1,100 selected checkpoints against
the same-host pinned CPU Oracle: 220 raw / 880 Chinese, with the x86 greedy IDs
listed above. Maximum observed relative L2, absolute and logit errors were 0;
minimum cosine was 1. Assertions retain the stated numerical tolerances for
lossy operations. This is measured agreement on these inputs, not a generic
bitwise GPU arithmetic guarantee.

The untraced real-model test separately passed CPU prefill=1 versus Vulkan
prefill=3 and four greedy predictions. Forced Vulkan default-dispatch text and
Auto + scalar red-image CLI smokes use `resident_hybrid_decoder backend=Vulkan`.
The image CLI uses the CPU F16 projector. The machine-readable summary is
[verification.json](verification.json).

Full decoder weights occupy `36,351,807,040` uploaded bytes; tiny trace capacity
used approximately 36.45 GB of Vulkan allocations. Capacity and prefill size
increase arena memory. These measurements are not performance benchmarks.
Vision encoder GPU execution and numerical vision parity are outside this
validation; image CLI uses the existing CPU projector and Vulkan text decoder.

### End-to-end generation and runtime measurement

The 2026-10-23 run used a full `--release --features vulkan` CLI, without
`parity-trace` or compute tracing: temperature 0, eight CPU threads, prefill
batch 64, the GGUF Jinja template, no thinking, and an output limit of 192.
All three cases completed before that limit. Full answers, commands, binary
SHA-256 and precise timing fields are in [performance.json](performance.json).

| Case | Input rows | Output tokens | Session init (s) | Prefill (s) | First text including startup (s) | Decode eval/s | Process wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar Vulkan text | 44 | 88 | 58.47 | 17.07 | 75.95 | 0.87 | 179.07 |
| Default Vulkan text | 44 | 104 | 24.61 | 19.44 | 44.33 | 0.40 | 308.92 |
| Default CPU vision + Vulkan decoder | 104 | 15 | 21.85 | 53.97 | 80.99 | 0.75 | 103.39 |

Decode timing counts single-token model evaluations, including the stopping
EOS evaluation. Startup includes model/session preparation and, for the image
case, image processing. File and driver caches were not flushed. The image
used 72 vision tokens; CPU vision encoding took 4.649 seconds. Given
[this synthetic input](performance-shapes.png), the complete actual response
was `图中从左到右依次是红色圆形、蓝色正方形和绿色三角形。`

These are **shared-environment observations**, not a stable hardware benchmark
or a scalar/default speed comparison. The host has eight adapters, but this
container exposes only `card7/renderD134` (PCI `0000:c3:00.0`); its GPU was
98–100% busy before our measured default runs. Each configuration ran once.
At that revision, the process held about 34 GiB in host-visible GTT buffers and
only 211 MiB in VRAM: `UploadedBuffers::upload` used `upload_static`, rather
than the separate `upload_device_static` VRAM path. These historical timings
precede the weight-placement optimization measured below.

Both text cases requested forced Vulkan. The image CLI uses `--gpu` (Auto),
since forced Vulkan CLI mode admits plain text only; there was no logged CPU
decoder fallback. Process DRM engine counters increased during image decoding
on the same physical adapter. The image encoder/projector remains on CPU.
This run does not extend the earlier numerical parity coverage or establish
long-context/concurrent performance.

### Discrete GPU weight placement

The shared dense/Qwen3/Qwen3.5 immutable decoder uploader now calls the existing
`upload_device_static`: discrete GPUs stage weights into `DEVICE_LOCAL` memory,
while UMA and device-allocation OOM retain its existing mapped-memory fallback.
No arithmetic, quantization or precision settings change. The device upload
regression now also exercises this decoder entry, checking bytes, zero padding
and unmapped device storage on a discrete GPU.

The same physical AMD adapter ran the saved baseline and a new full release CLI
with identical parameters and the same complete 104-token answer. The first
pair began with 0% GPU-busy samples; the host was not reserved. Exact commands,
binary/source hashes, answers and metrics are in [acceleration.json](acceleration.json).

| First complete pair | Before | After |
| --- | ---: | ---: |
| Session init (s) | 30.05 | 38.35 |
| Prefill, 44 rows (s) | 17.06 | 1.23 |
| Decode eval/s | 1.04 | 5.29 |
| First text including startup (s) | 47.42 | 39.98 |
| Process wall (s) | 150.09 | 61.25 |
| Process peak GTT (GiB) | 34.01 | 0.65 |
| Process peak VRAM (GiB) | 0.21 | 34.06 |

This pair shows 5.11x decode throughput, 13.86x prefill throughput and 2.45x
end-to-end speed (59.2% less process time), including the **8.31-second increase
in session initialization**. The new peak GTT includes temporary upload
staging. File and driver caches were not flushed; each case uses a new process.

Another model benchmark then occupied about 20 GiB VRAM on this same adapter.
The second new run began at 100/99/93% GPU busy and measured only 0.92 decode
eval/s. Both second-run answers still matched the first pair. These follow-up
results are retained as shared-load observations and excluded from the speedup.
Further repeats stopped; an incomplete third baseline is excluded. There is
**no dedicated-card repeated benchmark or median speedup claim**.

After this change, the physical upload check, all 21 Qwen3.5 adapter checks,
seven Qwen3 checks, real CPU prefill=1 versus Vulkan prefill=3 plus four decode
predictions, and all 1,100 scalar Oracle checkpoints passed again. Observed
maximum relative L2 and absolute errors remain 0, minimum cosine 1; exact
token/embedding/expert checks and the existing lossy bounds are unchanged.
The full default-dispatch synthetic-image CLI also returned the correct colors,
shapes and order. Its timings were under shared load and do not establish an
image speedup or numerical vision parity.

Other real dense/Qwen3 models, UMA hardware, forced VRAM exhaustion, long
contexts and concurrency were not tested in this follow-up. This does not
extend the optimized-graph numerical contract described above.

### CPU versus Vulkan with device-local weights, before the arena change

The weight-placement speedup above compares two Vulkan implementations.
At the weight-upload revision `eed17fba`, the optimized **eight-thread CPU
decoder was faster than Vulkan** for this short-input workload.

Both backends used the same release binary, model, 44-row prompt, temperature
0, no thinking, Jinja template, eight threads and prefill batch 64. Three
alternating CPU/Vulkan rounds each capped output at 64 tokens, with 63 timed
decode evaluations. These capped runs measure speed, not a completed answer.
Values below are per-metric medians; raw measurements and complete uncapped
answers are included under `cpu_comparison` in [acceleration.json](acceleration.json).

| Fixed 64-token output | CPU, 8 threads | Vulkan |
| --- | ---: | ---: |
| Session init (s) | 0.005 | 21.21 |
| Prefill, 44 rows (s) | 2.33 | 1.18 |
| Decode eval/s | 10.20 | 5.03 |
| Process wall including startup (s) | 9.96 | 37.10 |

CPU decode throughput is 2.03x Vulkan; Vulkan prefill throughput is 1.97x CPU.
Vulkan process time is 3.72x CPU, including the full weight-upload cost on every
new process. Decode rates ranged from 9.73–12.62 for CPU and 5.02–5.08 for
Vulkan. CPU processes opened no DRM device; Vulkan used the same physical PCI
device and its engine counters increased. All three GPU starts sampled
0/0/0% busy. The shared container has a 16-CPU quota and 128 allowed logical
CPUs, but these tests use eight threads; no container throttling was observed.

Uncapped CPU and Vulkan answers had 78 and 104 tokens and different text, so
the primary comparison uses equal output counts. Each backend's capped answer
was repeatable across its three rounds. This speed comparison does not extend
the optimized numerical-alignment contract or establish equal answer quality.
The host was not reserved, file/driver caches were not flushed, and persistent
sessions, full-core CPU performance, long contexts and concurrency were not
measured.

### Device-local activation and recurrent arena

Qwen3.5 now reuses the existing device-local operator arena for activations,
KV and recurrent state on discrete GPUs. Inputs transfer before command
recording; results and parity traces use owned staging copies, releasing the
submission guard before trace downloads. CPU shadow synchronization and all
finite checks remain. Shaders, arithmetic, expert dispatches and recording
chunk limits are unchanged. The allocator retains mapped memory on UMA or
allocation OOM; those fallback configurations were not exercised here.

The specified AMD host ran three rounds each of the saved weight-only Vulkan
binary, the new release Vulkan binary, and the new release eight-thread CPU
backend. GPU order reversed in round two. The prompt, 44 input rows, 64-token
output cap, 63 timed decode evaluations, eight threads, batch 64, Jinja,
no-thinking and temperature 0 were identical. Each case started a new process
after three consecutive idle samples. No parity or compute tracing was enabled.
Full commands, hashes, per-run data, ranges and complete answers are in
[device_arena.json](device_arena.json).

| Per-metric median | Old Vulkan | New Vulkan | CPU, 8 threads |
| --- | ---: | ---: | ---: |
| Session init (s) | 20.39 | 21.10 | 0.005 |
| Prefill, 44 rows (s) | 1.21 | 1.16 | 2.63 |
| Decode eval/s | 4.98 | 5.68 | 12.16 |
| Process wall including startup (s) | 36.52 | 35.77 | 8.92 |

Decode throughput increased by 14.0% in this sample. New Vulkan ranged from
5.30 to 5.68 eval/s versus 4.92 to 4.99 before. The process-wall median changed
by only 2.1%; initialization remained substantial and ranged from 19.42 to
38.73 seconds in the new runs, so this is not an initialization-speedup claim.
Session init includes weight upload and pipeline/buffer setup, not just copy
time. The CPU decoder remains 2.14x faster than the new Vulkan decoder here.

Decode-time process-resident VRAM increased from 34.06 to 34.19 GiB, consistent
with moving the arena into device memory. Resident GTT remained about 0.13 GiB
because the staging arena is retained for input transfers and CPU state mirrors.
All six fixed-length GPU answers were identical. A complete 104-token GPU
answer also matched the previous GPU answer; the 15-token image answer correctly
identified the red circle, blue square and green triangle in order. Image
encoding/projection still executes on CPU; no image speedup or numerical vision
alignment claim is made.

After the change, all 21 physical adapter tests, two injected GPU rollback
tests, four-axis mRoPE, real batched prefill/four-step decode, and all 1,100
selected scalar Oracle checkpoints passed again. Observed relative L2 and
maximum absolute error were 0, and minimum cosine was 1; exact structural
checks and existing lossy bounds are unchanged. The local Vulkan namespace
passed 46 tests with 45 ignored. The broader local Qwen3.5 trunk run had 43
passed, nine ignored, and the existing CPU-only
`qwen35_trace_metadata_is_independent_of_prefill_batch_size` failure, also
reproduced with the earlier cached test binary.

The host was shared and caches were not flushed. The third old-GPU case had
a 3.06-second increase in the container-wide CPU throttling counter; all other
cases were at most 8.7 ms. This shared counter cannot be attributed solely to
the inference process. Idle starts do not guarantee isolation throughout a run.
These measurements cover short requests and eight CPU threads, not full-core
CPU performance, persistent sessions, long contexts or concurrency. Optimized
CPU/Vulkan graph alignment remains outside the scalar Oracle claim.

### Completion validation (2026-10-23 snapshot)

These results predate the review follow-up below.

Default dispatch passes the original 1,100 selected checkpoints on ARM64 CPU,
x86_64 CPU and the physical NAVI31 Vulkan decoder, without `RMI_SCALAR`.
The same-host Oracle and cross-ISA boundary described above still apply.
The historical performance tables above used different arithmetic and are
retained as measurements of their respective revisions; the current CPU/GPU
comparison is recorded in [completion.json](completion.json).

A longer 44-token Chinese prompt exposed a Vulkan RoPE defect: rotation still
used FMA even when the MoE graph selected reference arithmetic. The first
visible difference was at layer 23, input row 23. Its tiny attention error
crossed a subsequent Q8 activation boundary (116.500015 versus 116.499992),
and the final prefill logits exceeded the original relative-L2 limit:
0.0259996 versus 0.01. The dense-prepare operator now forwards the active
reference mode and separately rounds the rotation products in that mode;
normal fused dispatch retains FMA. A small cancellation fixture failed on the
physical adapter with 2.039535 instead of 0, and passes after the fix.
Temporary diagnostic flags and batch-trace modifications were removed.

The 44-token case now also passes the pinned Oracle on AMD CPU and Vulkan:
47 processed rows (prompt plus three generated inputs), 2,585 selected
checkpoints per backend and greedy IDs `[101742, 98138, 103725, 96647]`.
The observed maximum relative L2 / absolute error is 0 and minimum cosine is 1.
The separate untraced regression compares CPU batch=1, CPU batch=64 and
Vulkan batch=64 across 64 complete output-logit vectors and their greedy IDs;
all pass, with observed floating error 0. The implementation retains the
original lossy numerical bounds, rather than asserting generic bitwise GPU
arithmetic. Tokens, embedding values and expert IDs/order remain exact checks.

The physical AMD suite passes 22 adapter tests, two injected rollback tests,
four-axis mRoPE and the long batched regression. The local Qwen namespace
passes 80 tests (11 ignored), shader source/binary/manifest validation passes,
and all four comparator self-tests pass. The numerical-scope regression
covers nesting, pool and chunk workers, thread isolation, restoration and
unwind restoration. The config error-message expectation matches the actual
loader validation; the CPU recurrent trace now emits `q_conv_prenorm` once
after copying every row and before normalizing any row.

The final release binary uses the same default arithmetic on both backends:
SHA-256 `7ffd077ce3d334eeb1b7bf37b38c60405d30184c4858c6c2e12f3117a04cde5c`.
Three alternating CPU/GPU rounds used eight threads, batch 64, the same
44-token prompt and 64 output tokens (63 timed decode evaluations). All six
answers match. Every case starts a new process; caches are not flushed.

| Final release, median of three | CPU, 8 threads | NAVI31 Vulkan |
| --- | ---: | ---: |
| Session init (s) | 0.0047 | 21.72 |
| Prefill, 44 rows (s) | 4.82 | 0.96 |
| Decode eval/s | 9.92 | 5.51 |
| First text including startup (s) | 5.10 | 22.99 |
| Process wall (s) | 12.55 | 36.85 |

Vulkan prefill is 5.03x faster in this sample, but CPU decode is 1.80x faster
and the standalone CPU process finishes 2.94x faster. Vulkan decode ranges
from 5.21 to 5.70 eval/s; CPU ranges from 7.90 to 10.10. Weight upload/session
setup remains substantial. These are shared-host observations, not dedicated
hardware or persistent-session benchmarks, and do not imply full-core CPU
performance. The earlier SIMD/default CPU tables are revision-specific.

The complete GPU text run reaches EOS with 80 output tokens and matches the
complete CPU answer. The companion-image run also reaches EOS (15 tokens),
returning `图中从左到右依次是红色圆形、蓝色正方形和绿色三角形。`
for the synthetic image above. Process DRM engine counters verify GPU activity;
the image projector remains on CPU. All eight final runs began after three
idle samples, but the shared host was not reserved. Complete answers, commands,
per-run timing and resource snapshots remain in `completion.json`.

Full-library testing is not green. The comparison against `e3c49e3d` is retained
in the completion evidence: CPU failures decreased from 24 to 22, while the
serial Vulkan run changed from 32 to 31. One VibeVoice failure appeared only
in the current full Vulkan run but passes in isolation on both revisions;
it is not classified as a proven baseline failure. Other failures reproduced
on the baseline. The final RoPE change was verified with the focused and
physical suites above, rather than rerunning unrelated full-library failures.
No whole-repository or clean-clippy claim is made.

For the validated standalone GPU text path, build the release CLI and use:

```sh
cargo build --release --features vulkan --bin rust-model-inference
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
DRI_PRIME='pci-0000_c3_00_0!' \
  target/release/rust-model-inference \
  --model "$SAGE_MODEL_DIR/Sage-A1.5-Q8_0.gguf" --compute vulkan \
  --jinja --no-thinking --temp 0 --threads 8 --prefill-batch-size 64 \
  --max-tokens 192 --prompt '请用三句话向初学者解释混合专家（MoE）模型。'
```

Other GPUs/libm implementations, other expert formats, persistent-session
performance, long contexts, MTP execution and numerical vision alignment
remain outside this validation. HTTP functional checks added after review are
recorded below. The CPU vision encoder/projector is not a GPU
implementation. Other model revisions and quantization types need separate
validation.


### Review fixes and functional acceptance

[review-fixes.json](review-fixes.json) records the follow-up to PR #187's review,
including the exact commands, real-model responses, red/green regressions and
current test failures. `verification.json` is an explicitly historical snapshot;
`completion.json` retains the previous AMD validation and measurements.

The HTTP chat/tool classifier now accepts `qwen35moe` with the existing Qwen3.5
grammar. Before this fix, a real Sage chat request returned HTTP 500 with
`Tool/chat template is unsupported for architecture qwen35moe`. Text and SSE
chat requests now return the same answer, `你好！有什么可以帮助你的吗？`, and SSE
terminates with `[DONE]`. The parser regression exercises every UTF-8 split of
a Qwen3.5 tool call. Actual tool execution remains outside the real-model smoke.

CPU CLI and HTTP JEV single-choice and grouped block-choice requests run with
the published model. For context `The sky is blue.` and question
`What color is the sky?`, options `blue` / `green`, both interfaces choose A and
return identical probability values: single A=0.9934966564178467,
B=0.0065033454447984695; grouped A=0.9999839067459106,
B=0.000016103205780382268. These are functional consistency checks, not
numerical Oracle validation of every JEV mode. An HTTP data-URI image request
also describes the red circle, blue square and green triangle in order; its
48-token budget ends with `finish_reason=length`. Vision remains on CPU.

The legacy Q8 Vulkan cache is keyed by host address and length. A deterministic
real-device regression reuses one host allocation across two kernel lifetimes;
before eviction it returned 32 instead of 64 on the second calculation.
Dropping a borrowed or owned Q8 kernel now evicts that weight, and warmup evicts
its own temporary weights. Cleanup retires an expired host key immediately, but releases
the GPU allocation only after commands are confirmed idle. Failed recovery
retains the allocation until a later successful recovery or context teardown.
An injected fence/idle failure verifies this boundary and subsequent address
reuse on the physical Apple M3 Max Vulkan device. This regression demonstrates
a cache defect; it does not prove the cause of the previously reported
intermittent VibeVoice full-suite failure. Direct legacy raw-slice callers still
need to manage the backing allocation lifetime. `Q8Kernel` retains `Clone` but
can no longer be `Copy` because it now performs cleanup. Temporary kernels
created by `QuantizedTensor` bridge calls evict on each drop; long-lived
`Weight` kernels retain reuse across forwards.

CPU recurrent-state and `last_hidden` fixtures now explicitly request CPU, so
unrelated tests that globally enable GPU cannot switch those fixtures to a
different backend. The SIMD opt-in and scalar override are checked in isolated
subprocesses. The default CPU path was rechecked against the pinned Oracle:
all 1,100 selected checkpoints pass the original exact/numerical contracts.
The opt-in CPU CLI smoke also generates `你好！有什么可以帮助你的吗？`; this does
not establish opt-in numerical parity or a performance gain.

To reproduce the functional checks, build both entry points and start a CPU
server with the existing artifacts:

```sh
cargo build --profile release-fast --features vulkan,parity-trace \
  --bin rust-model-inference --bin rust-model-server

target/release-fast/rust-model-inference \
  --model "$SAGE_MODEL_DIR/Sage-A1.5-Q8_0.gguf" --compute cpu --threads 8 \
  --prefill-batch-size 64 --jev --jev-output json \
  --jev-context 'The sky is blue.' --jev-question 'What color is the sky?' \
  --jev-option blue --jev-option green
# Add --jev-block color before the options for the grouped request.

target/release-fast/rust-model-server \
  --model "$SAGE_MODEL_DIR/Sage-A1.5-Q8_0.gguf" \
  --mmproj "$SAGE_MODEL_DIR/Sage-A1.5-mmproj.gguf" \
  --compute cpu --threads 8 --prefill-batch-size 64 \
  --host 127.0.0.1 --port 8080 --max-context 512 --jinja --no-thinking

curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"temperature":0,"max_tokens":8}'
# Add "stream":true to the body for the SSE request.
curl --fail-with-body http://127.0.0.1:8080/v1/jev/score \
  -H 'Content-Type: application/json' \
  -d '{"context":"The sky is blue.","questions":[{"text":"What color is the sky?","options":["blue","green"]}]}'
curl --fail-with-body http://127.0.0.1:8080/v1/jev/grouped \
  -H 'Content-Type: application/json' \
  -d '{"context":"The sky is blue.","mode":"block_choice","questions":[{"text":"What color is the sky?","groups":[{"label":"color","options":["blue","green"]}]}]}'

cargo test --profile release-fast --features vulkan,parity-trace \
  --lib cached_weight -- --ignored --nocapture --test-threads=1
```

The original AMD endpoint closes SSH during the handshake in this follow-up;
no new AMD result or SIMD speedup is claimed. Whole-suite and clippy results,
including baseline comparisons and unresolved failures, are listed by name in
`review-fixes.json`; neither is claimed green.


| Follow-up test scope | Passed | Failed | Ignored |
| --- | ---: | ---: | ---: |
| Final default library (serial) | 1243 | 21 | 80 |
| Final Vulkan library (serial) | 1352 | 26 | 168 |
| Default library + integration targets (151 targets) | 1739 | 40 | 146 |
| Explicit physical-device cache regressions | 4 | 0 | 0 |

The final Vulkan failure names are a subset of the same-host `e3c49e3d`
baseline's 31 failures; this comparison has no current-only failure.
Clippy still exits 101 at the same 20 primary error locations as that baseline.
The unrestricted default `cargo test --no-fail-fast` also encounters an
unchanged example without a `main`; library and integration targets were run
separately. The integration run preceded the final cache-owner destructor
changes; final library runs and Vulkan clippy's test compilation cover those
changes. The 40 integration/library failures are not all classified as baseline
failures. These counts exclude ignored real-model/device tests unless explicitly
run above. Full failure names, commands, logs and their hashes are retained in
`review-fixes.json`.
