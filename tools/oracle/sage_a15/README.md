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

`compare.py` refuses a file with the wrong size or SHA-256. Both implementations
use CPU, eight threads, F32 KV, sequential single-token prefill and four greedy
predictions. This avoids comparing padded SSM ubatch rows with real tokens.
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

### GPU probe: full-model decoding is unsupported

The remote host exposes a physical gfx1100 GPU with approximately 48 GiB VRAM,
reported by Vulkan as `AMD Radeon Graphics (RADV NAVI31)`. Tests explicitly
select `/usr/share/vulkan/icd.d/radeon_icd.json` to exclude software ICDs.

**Sage does not have a complete Vulkan decoder.** The existing executor records
dense FFNs, but has no routed/shared-expert recorder for `qwen35moe`.
`--compute vulkan` rejects this architecture. `--gpu` / `--compute auto` must
fall back to the CPU decoder; successful Auto output is not GPU alignment.

The real-model probe found that the old eligibility builder hard-coded
`qwen35`, accepting Sage while silently skipping every MoE FFN. Auto emitted
blank/whitespace answers for the Chinese and red-image cases while reporting
`resident_hybrid_decoder backend=Vulkan`. The fix derives the architecture
from the parsed MoE configuration and rejects incomplete dense FFNs before
weight upload. Two small tests reproduce the old acceptance bug; both failed
before the fix, then passed. The final Vulkan adapter suite passed all 15
tests, explicitly including the two device-dependent tests.

| Probe | Result |
| --- | --- |
| Generated Q8_0 matmul fixtures, five shapes | Passed `vk_check`'s combined absolute/relative bound |
| Generated F32 projections, 18 individual/grouped cases | Passed; largest observed absolute error `1.144409e-4` |
| Real Sage SSM alpha/beta projections | All 60 Q8_0 matrices passed; observed maximum absolute error 0 |
| Forced full-model Vulkan | Rejected `qwen35moe`, exit 1 |
| Auto Chinese text / red-image CLI | Both match the corresponding CPU answer; trace confirms CPU decoder fallback |

Projection checks compare physical Vulkan dispatches against CPU kernels,
requiring finite values and `abs(gpu - cpu) <= 1e-4 + 1e-4 * abs(cpu)` for
floating-point reductions. The old F32 exact-bit assertion is replaced with
this existing `vk_check` bound. The real-weight test accepts Sage's Q8_0 SSM
projections as well as F32 projections; its CPU reference prepares quantized
activations independently. These are operator checks, not a full-model GPU
comparison with llama.cpp. The CLI output matches establish correct fallback,
not GPU vision execution or numerical parity.

Build and inspect the selected backend with:

```sh
cargo build --profile release-fast --features vulkan \
  --bin rust-model-inference --example vk_check
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
  target/release-fast/examples/vk_check
RMI_COMPUTE_TRACE=1 VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
  target/release-fast/rust-model-inference \
  --model "$SAGE_MODEL_DIR/Sage-A1.5-Q8_0.gguf" \
  --gpu --jinja --no-thinking --prompt '你好' --max-tokens 8 --temp 0 --threads 8
RMI_QWEN35_MODEL="$SAGE_MODEL_DIR/Sage-A1.5-Q8_0.gguf" \
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
  cargo test --profile release-fast --features vulkan --lib vulkan::qwen35::tests:: \
  -- --include-ignored --nocapture --test-threads=1
```

The trace must report `unsupported architecture qwen35moe. Falling back to CPU`
and `scope=hybrid_decoder backend=Cpu`. Replacing `--gpu` with
`--compute vulkan` must exit unsuccessfully with an unsupported-architecture
error. Neither result establishes GPU model accuracy or performance.

### Focused regression checks

The local `--lib models::qwen35::` run reports 74 passed, 1 failed and 4 ignored.
The failure is the pre-existing `qwen35_config_rejects_invalid_tensor_dimensions`
error-message assertion (`head lengths` versus `Invalid ...key_length: 0`).
Excluding that known failure gives 74 passed. The new production-path MoE test
passes with both default dispatch and `RMI_SCALAR=1`; the old implementation
fails, and reverting only its softmax independently fails the second fixture.
The Python comparator has four passing self-tests, including rejection of
identical NaNs in a supposedly lossless embedding checkpoint.

On remote Linux x86_64, the final `--features parity-trace --lib models::qwen35::`
run reports 72 passed, 2 failed and 5 ignored. Both failures were reproduced
before these changes: the same configuration assertion and
`qwen35_trace_metadata_is_independent_of_prefill_batch_size` (which also checks
existing batch-dependent numeric outputs). The new MoE fixtures pass there.

The text Oracle harness covers incremental decoding; it does not establish
batched prefill, long-context, MTP, full-model GPU, HTTP or vision numerical
parity. Other model revisions and quantization types do not inherit these results.
