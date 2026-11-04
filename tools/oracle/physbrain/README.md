# PhysBrain1.5-2B real-weight verification

This directory pins the independent CPU reference and the acceptance boundaries for
PhysBrain1.5-2B (DeepCybo/PhysBrain1.5-2B, a Qwen3-VL-2B-Instruct fine-tune).
`verification.json` records the exact GGUF hashes, formats and measured errors.
Lossless loading/layout/token IDs are checked exactly. Quantized matmul, activation
rounding and approximate nonlinear operators use numerical checks, as requested;
none of the following results is a claim of whole-model bitwise parity.

PhysBrain rides the `qwen3vl` text path and the `qwen3vl_merger` mmproj path, both
already supported by the engine. The PhysBrain-specific surface is the tokenizer
vocab extension to 168576 tokens (vs 151936 base Qwen3), the deepstack visual
indexes `[5, 11, 17]` in the mmproj, and a RoPE `dimension_count` of 128.

## Inputs and reference

On the validation machine `models` points to
`/Users/gouzi/Documents/git/rust-model-inference/models`. All downloads and local
conversions remain there; no weights or temporary reference checkout are committed.
Dependencies use the repository `.venv`, and Cargo uses a repository `target`.

The reference is ggml-org/llama.cpp commit
`11fe02151f79c41d0d4af7da708755d73b9c0da6`. Use a **temporary checkout**:

```sh
git clone https://github.com/ggml-org/llama.cpp target/physbrain/llama.cpp
git -C target/physbrain/llama.cpp checkout 11fe02151f79c41d0d4af7da708755d73b9c0da6
tools/oracle/physbrain/build.sh target/physbrain/llama.cpp target/physbrain/oracle-build
```

The build disables Accelerate, BLAS, GPU backends and KleidiAI. The patch only
adds checkpoint dumps and greedy token reporting, without changing model math.
Dumps are contiguous native little-endian F32 on the tested Apple Silicon machine;
`index.tsv` includes names and full shapes. Create a fresh, empty dump directory
before every run: the index and Rust traces append to existing files.

## Tokenizer and contracts

```sh
export RMI_PHYSBRAIN_1_5_2B_TEXT=models/PhysBrain1.5-2B-GGUF/physbrain-1.5-2b-Q8_0.gguf
export RMI_PHYSBRAIN_1_5_2B_MMPROJ=models/PhysBrain1.5-2B-GGUF/mmproj-physbrain-1.5-2b-F16.gguf
export RMI_LLAMA_TOKENIZE=target/physbrain/oracle-build/bin/llama-tokenize
cargo test --profile release-fast --features parity-trace --test physbrain_tokenizer_oracle -- --ignored --test-threads=1
cargo test --profile release-fast --test physbrain_1_5_2b -- --ignored
```

The contract tests are `#[ignore]`d: without the converted GGUF they have
nothing to check, and a skipped test that reports as passed is worse than no
test at all. Run them with `-- --ignored`.

Tokenizer cases include empty input, English, Unicode, whitespace, ChatML and
ordinary tool/thinking added tokens. The converter preserves CONTROL vs
USER_DEFINED token types and writes llama.cpp-compatible 32-bit metadata.
PhysBrain RoPE spans the full 128-dimension head.

To reproduce the local conversion from the downloaded safetensors:

```sh
./.venv/bin/python -m tools.converter.physbrain.convert_physbrain --hf-dir models/PhysBrain1.5-2B --out-dir models/PhysBrain1.5-2B-GGUF
```

Never replace a GGUF while a process maps it.

## Text logits and generation

PhysBrain requires `--kv-cache f16` in Rust and `-ctk f16 -ctv f16` in llama.cpp.

```sh
prompt=$'<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
mkdir -p target/physbrain/text-reference
RMI_ORACLE_DUMP=target/physbrain/text-reference RMI_ORACLE_FILTER='^result_output$' \
  target/physbrain/oracle-build/bin/llama-eval-callback -m "$RMI_PHYSBRAIN_1_5_2B_TEXT" \
  -p "$prompt" -n 8 -c 512 -ngl 0 -t 8 -fa off -ctk f16 -ctv f16 > target/physbrain/text-reference.log 2>&1
cargo build --profile release-fast --features parity-trace --bin rust-model-inference
RMI_PARITY_TRACE=target/physbrain/text-rust.jsonl RMI_PARITY_FILTER='qwen35.prompt_ids,prompt_ids,result_output' \
  target/release-fast/rust-model-inference --model "$RMI_PHYSBRAIN_1_5_2B_TEXT" \
  --prompt 'What is 2+2?' --max-tokens 8 --max-context 512 --threads 8 --kv-cache f16 > target/physbrain/text-rust.log 2>&1
./.venv/bin/python tools/oracle/physbrain/compare_text.py target/physbrain/text-rust.jsonl target/physbrain/text-reference target/physbrain/text-reference.log --relative-l2 0.05
```

The 0.05 bound is a regression envelope for this fixture, not an accuracy
guarantee for arbitrary prompts. The comparator rejects missing prompt IDs,
steps, shapes, payloads and nonfinite values, and checks every vocabulary logit
at the final prefill row and each decode. It requires exact greedy IDs when both
runs follow their own generated history.

## Vision projections and DeepStack

`llama-mtmd-debug -p encode --image blue -n 512` supplies already normalized RGB
floats `[0,0,1]` directly. It does **not** preprocess a blue PNG. The Rust test uses
that identical input and compares all projected values, including all three
PhysBrain DeepStack outputs in the reference's token/feature order.

PhysBrain enters `models::qwen3::vision`, including DeepStack. Use
`widen_projector.py` to widen the same weight values exactly and separate
architecture/layout errors from the reference's extra F16 activation rounding.

The test runs the tower twice: once with F32 weights, which isolates the
structure of the tower, and once with the Q8_0 weights the CLI actually uses.
Both are compared against the same reference dump.

```sh
PYTHONPATH=target/physbrain/llama.cpp/gguf-py ./.venv/bin/python tools/oracle/physbrain/widen_projector.py "$RMI_PHYSBRAIN_1_5_2B_MMPROJ" models/PhysBrain1.5-2B-GGUF/mmproj-physbrain-1.5-2b-oracle-F32.gguf
mkdir -p target/physbrain/vision-reference
RMI_ORACLE_DUMP=target/physbrain/vision-reference RMI_ORACLE_FILTER='^rmi.vision.output$' \
  target/physbrain/oracle-build/bin/llama-mtmd-debug -m "$RMI_PHYSBRAIN_1_5_2B_TEXT" \
  --mmproj models/PhysBrain1.5-2B-GGUF/mmproj-physbrain-1.5-2b-oracle-F32.gguf \
  -p encode -n 512 --image blue -c 512 --no-warmup --no-repack -ngl 0 -t 8 -fa off > target/physbrain/vision-reference.log 2>&1
RMI_PHYSBRAIN_VISION_REFERENCE=target/physbrain/vision-reference/0000.f32 \
  cargo test --profile release-fast --test physbrain_vision_oracle physbrain_vision -- --ignored --nocapture
```

The widener asserts that every dequantized weight is exactly preserved. The
measured relative L2 against llama.cpp is **0.00341** with F32 weights (bound
0.01) and **0.16355** with the production Q8_0 weights (bound 0.25, the
quantization envelope). With the interleaved axis assignment the same
comparison measured **0.56000**, so the test fails if the rotation goes back
to being wrong. This is numerical validation, not equality with the original
F16 execution or a bitwise parity claim.

The reference build has to match the host: llama.cpp's scalar build and its
NEON build differ by 0.00085 in this comparison, which is small next to the
bounds above but not zero.

## Real CLI smoke

PhysBrain identifies the blue-image fixture correctly:

```sh
target/release-fast/rust-model-inference --model "$RMI_PHYSBRAIN_1_5_2B_TEXT" --mmproj "$RMI_PHYSBRAIN_1_5_2B_MMPROJ" --image tools/oracle/physbrain/fixtures/blue-64.png --prompt 'Reply with one word: what color is this image?' --max-tokens 8 --max-context 256 --threads 8 --temperature 0
```

`fixtures/blue-64.png` is a 64x64 solid blue image in this directory. The
prompt needs `--temperature 0`: without it the CLI samples, and the same
command prints a different sentence on every run.

This is a semantic smoke check; its generation template is not asserted identical
to llama.cpp. The numerical acceptance remains the tokenizer, text-logit and full
projection tests above.

## Regression checks

```sh
./.venv/bin/python -m unittest tools.oracle.physbrain.test_compare_text
cargo fmt --all -- --check
cargo check --profile release-fast --features vulkan,parity-trace --lib --bin rust-model-inference
```

## The `exp_approx` clamp: what it actually fixes

The Qwen3-VL-2B family (including PhysBrain) was first recorded as failing with
"first token = EOS" or a degenerate repetition loop, and the cause was
attributed to `exp_approx_avx2`: its clamp at `+-88.376_26` lets the exponent
reach `-128`, the exponent field overflows, and `silu` collapses to `-0`, so
the FFN channel is dropped.

That attribution does not hold on the code as it stands. `481ec939` (#186,
2026-10-27, already in `main`) added the `_mm256_andnot_ps(cmp(n, -126), ...)`
mask to the AVX2 path, and it zeroes every lane with `n < -126`. Simulating
that instruction sequence for the recorded value: `-131.219` is clamped to
`-88.376_26`, `z = 12582784.0`, `n = -128`, `scale = -inf` -- and the mask
turns it into `0`, so `silu(131.219) = 131.219` and `silu(gate)*up =
-19893.98`, the "correct value" the original record noted, not `0.0`. The
negative side was neutralised by #186. The oracle host was Apple Silicon,
where the NEON variant takes upstream's `|n| > 126` slow path and was never
affected at all, so the AVX2 clamp cannot be the root cause.

What the clamp does fix is the positive side. The fast path builds `2**n` by
shifting float bits and lands it in the exponent field as `n + 127`; `n = 128`
overflows that field to `0xFF`, so `exp_approx_avx2` returned `+inf` for
`x` in `(88.0999, 88.376_26]` even though the true value (2.4e38 at 88.376)
is an f32. The only caller that can hand a large positive value to `exp` is
`silu` (softmax passes `v - max <= 0`, tanh clamps to +-10 first), and there
the true activation is about `-3.7e-37`, so the old and the new value are both
approximately zero. Overflowing to `+inf` is wrong, but it did not cause the
2026-10 symptoms.

The clamp is now `+-88.0` (`n <= 127`), and
`fast_exp_clamp_keeps_the_exponent_inside_the_f32_exponent_field` states the
invariant that keeps it there: the bound has to satisfy
`clamp * log2(e) <= 127`. That check runs on every platform, because the AVX2
path itself cannot be observed from an ARM host. The old bound reached 127.499
and fails it.

## The vision tower's RoPE

The vision tower rotates Q/K with the **contiguous** axis assignment, the same
as llama.cpp's `GGML_ROPE_TYPE_VISION`. It used to call
`rope_mrope_interleaved`, which carries the **text** tower's interleaved
assignment (HF `apply_interleaved_mrope`, llama.cpp's
`GGML_ROPE_TYPE_IMROPE`). Both operators pair the rotation the same way
(split-half / NeoX): what differed was which axis each frequency slot belongs
to, so the image embeddings had the wrong geometry while the model still
answered in fluent sentences.

Measured against the pinned llama.cpp:

| Rust side | relative L2 | notes |
| --- | --- | --- |
| F32 weights | 0.00341 | the structure of the tower: patch embedding, spatial merge, position embedding, RoPE, DeepStack order |
| Q8_0 weights (the production path) | 0.16355 | the tower's weights are quantized to Q8_0; the error compounds through the ViT and is amplified by the massive activations that start at layer 10 |
| Q8_0 weights, interleaved axis | 0.56000 | the bug this fixes |

The Q8_0 bound is the production envelope; the F32 bound is the structural
check, and it is the one that fails if the rotation ever goes wrong again.
The test runs both.

The vision tower used to carry its own hand-rolled copy of this rotation for
the lossless path, and the copy and the shared operator drifted: the copy was
right, the shared operator was not, so the wrong axis assignment only reached
the models that quantize their weights. There is now one implementation.
