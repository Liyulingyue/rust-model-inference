# Sage-A1.5 Q8_0

[ModelScope artifact](https://modelscope.cn/models/InternScience/Sage-A1.5-Q8_0-GGUF/files).
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
`compare.py` sets Rust's existing `RMI_SCALAR=1` dispatch. The lossy comparisons
below use numerical bounds, including on this reproducible scalar CPU path.

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

Default ARM NEON CLI smokes also passed with built-in chat wrapping, the GGUF
Jinja template, and the F16 projector (a solid-red image was identified as red).
The two text trace cases generated the same four greedy IDs with default
dispatch, but **did not pass the scalar numerical/routing contract**: worst
relative L2 was 0.05289 / 0.11320, minimum cosine 0.99860 / 0.99359, and maximum
logit absolute error 0.50174 / 1.16432 for raw / Chinese respectively; some
selected expert IDs/order differed. These are measured limits, not relaxed
acceptance thresholds. Use `RMI_SCALAR=1` for the validated Oracle contract.

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

Default-dispatch CLI smokes passed on this host for built-in chat wrapping,
the GGUF Jinja template, and the F16 projector identifying a solid-red image
as red. These establish CLI execution only; the x86_64 optimized kernels and
vision outputs were not compared numerically with the Oracle.

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
use compensated sums. Other modes retain their existing reduction contracts.

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
RMI_SCALAR=1 RMI_QWEN35_MODEL="$SAGE_MODEL_DIR/Sage-A1.5-Q8_0.gguf" \
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
  cargo test --profile release-fast --features vulkan,parity-trace \
  --lib vulkan::qwen35::tests:: -- --include-ignored --nocapture --test-threads=1
python3 tools/oracle/sage_a15/compare.py \
  --backend vulkan --model "$SAGE_MODEL_DIR/Sage-A1.5-Q8_0.gguf" \
  --oracle target/sage-oracle/build/bin/llama-eval-callback \
  --rust-test target/release-fast/deps/sage_a15-REPLACE_WITH_EXECUTABLE_HASH \
  --out target/sage-vulkan-parity
RMI_SCALAR=1 RMI_SAGE_MODEL="$SAGE_MODEL_DIR/Sage-A1.5-Q8_0.gguf" \
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
  cargo test --profile release-fast --features vulkan,parity-trace \
  --test sage_a15 sage_vulkan_batched_prefill_and_decode -- --ignored --exact --nocapture
```

Forced Vulkan session creation must succeed; the trace helper prints
`Sage trace backend: vulkan` and cannot silently substitute CPU execution.
GPU trace mode uses one-token chunks to retain the token-major schema; the
separate batched test runs without tracing and compares prefill=3 plus four
greedy steps against CPU prefill=1. Structural checks and numerical thresholds
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

### Focused regression checks

The final local `--features vulkan,parity-trace --lib vulkan::` run passed 46
tests, with 45 device/model/benchmark tests ignored. The physical Sage adapter
and focused SiLU tests above were run separately with ignored tests enabled.
Formatting, shader rebuild/manifest/SPIR-V validation and all four Python
comparator self-tests passed.

Earlier broader Qwen3.5 runs still had two baseline failures reproduced before
these changes: `qwen35_config_rejects_invalid_tensor_dimensions` (error-message
assertion) and `qwen35_trace_metadata_is_independent_of_prefill_batch_size`
(existing batch-dependent numeric outputs). No whole-repository or clean
clippy claim is made. The production CPU MoE regression also passed default
and scalar dispatch; reverting its earlier softmax/reduction fixes fails it.

The optimized x86 decoder graph, other GPUs/hosts, other expert formats,
long-context behavior, MTP execution, HTTP and numerical vision alignment are
not covered. Default ARM NEON limits above still apply; the Vulkan scalar
results do not broaden the CPU dispatch claim. Other model revisions and
quantization types need separate validation.
