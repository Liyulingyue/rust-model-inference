# Intern-Decision-0.8B real-weight verification

This directory pins the independent CPU reference and the acceptance boundaries
for Intern-Decision-0.8B (Shanghai_AI_Laboratory/Intern-Decision-0.8B), a
Qwen3.5-0.8B fine-tune with a `qwen3vl_merger` vision tower.
`verification.json` records the exact GGUF hashes, formats and measured errors.
Lossless loading/layout/token IDs are checked exactly. Quantized matmul,
activation rounding and approximate nonlinear operators use numerical checks, as
requested; none of the following results is a claim of whole-model bitwise
parity.

The engine side needs no model-specific code: the converter writes the
`qwen35` architecture, which this repository already serves, and the vision
tower rides the `qwen3vl_merger` projector path.

## Inputs and reference

On the validation machine `models` points to
`/Users/gouzi/Documents/git/rust-model-inference/models`. All downloads and local
conversions remain there; no weights or temporary reference checkout are committed.
Dependencies use the repository `.venv`, and Cargo uses a repository `target`.

The reference is ggml-org/llama.cpp commit
`11fe02151f79c41d0d4af7da708755d73b9c0da6`. Use a **temporary checkout**:

```sh
git clone https://github.com/ggml-org/llama.cpp target/intern_decision/llama.cpp
git -C target/intern_decision/llama.cpp checkout 11fe02151f79c41d0d4af7da708755d73b9c0da6
tools/oracle/intern_decision/build.sh target/intern_decision/llama.cpp target/intern_decision/oracle-build
RMI_ORACLE_SCALAR=1 tools/oracle/intern_decision/build.sh target/intern_decision/llama.cpp target/intern_decision/oracle-scalar-build
```

The build disables Accelerate, BLAS, GPU backends and KleidiAI. The optional
scalar build also disables NEON and llamafile and turns off FP contraction. The
patch only adds checkpoint dumps and greedy token reporting, without changing
model math. Dumps are contiguous native little-endian F32 on the tested Apple
Silicon machine; `index.tsv` includes names and full shapes. Create a fresh,
empty dump directory before every run: the index and Rust traces append to
existing files.

## Tokenizer and contracts

```sh
export RMI_INTERN_DECISION_0_8B_TEXT=models/Intern-Decision-0.8B-GGUF/intern-decision-0.8b-Q8_0.gguf
export RMI_INTERN_DECISION_0_8B_MMPROJ=models/Intern-Decision-0.8B-GGUF/mmproj-intern-decision-0.8b-F16.gguf
export RMI_LLAMA_TOKENIZE=target/intern_decision/oracle-build/bin/llama-tokenize
cargo test --profile release-fast --features parity-trace --test intern_decision_tokenizer_oracle -- --ignored --test-threads=1
cargo test --profile release-fast --test intern_decision_0_8b
```

Tokenizer cases include empty input, English, Unicode, whitespace, ChatML and
ordinary tool/thinking added tokens. The converter preserves CONTROL vs
USER_DEFINED token types and writes llama.cpp-compatible 32-bit metadata.

To reproduce the local conversion from the downloaded safetensors:

```sh
./.venv/bin/python -m tools.converter.intern_decision.convert_intern_decision --hf-dir models/Intern-Decision-0.8B --variant 0.8b --out-dir models/Intern-Decision-0.8B-GGUF
```

Never replace a GGUF while a process maps it.

## Text logits and generation

```sh
prompt=$'<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
mkdir -p target/intern_decision/text-reference
RMI_ORACLE_DUMP=target/intern_decision/text-reference RMI_ORACLE_FILTER='^result_output$' \
  target/intern_decision/oracle-build/bin/llama-eval-callback -m "$RMI_INTERN_DECISION_0_8B_TEXT" \
  -p "$prompt" -n 8 -c 512 -ngl 0 -t 8 -fa off -ctk f32 -ctv f32 > target/intern_decision/text-reference.log 2>&1
cargo build --profile release-fast --features parity-trace --bin rust-model-inference
RMI_PARITY_TRACE=target/intern_decision/text-rust.jsonl RMI_PARITY_FILTER='qwen35.prompt_ids,prompt_ids,result_output' \
  target/release-fast/rust-model-inference --model "$RMI_INTERN_DECISION_0_8B_TEXT" \
  --prompt 'What is 2+2?' --max-tokens 8 --max-context 512 --threads 8 > target/intern_decision/text-rust.log 2>&1
./.venv/bin/python tools/oracle/intern_decision/compare_text.py target/intern_decision/text-rust.jsonl target/intern_decision/text-reference target/intern_decision/text-reference.log --relative-l2 0.03
```

The 0.03 bound is a regression envelope for this fixture, not an accuracy
guarantee for arbitrary prompts. The comparator rejects missing prompt IDs,
steps, shapes, payloads and nonfinite values, and checks every vocabulary logit
at the final prefill row and each decode. It requires exact greedy IDs when both
runs follow their own generated history.

## Vision projections and DeepStack

`llama-mtmd-debug -p encode --image blue -n 512` supplies already normalized RGB
floats `[0,0,1]` directly. It does **not** preprocess a blue PNG. The Rust test
uses that identical input and compares all projected values.

```sh
PYTHONPATH=target/intern_decision/llama.cpp/gguf-py ./.venv/bin/python tools/oracle/intern_decision/widen_projector.py "$RMI_INTERN_DECISION_0_8B_MMPROJ" models/Intern-Decision-0.8B-GGUF/mmproj-intern-decision-0.8b-oracle-F32.gguf
mkdir -p target/intern_decision/vision-reference
RMI_ORACLE_DUMP=target/intern_decision/vision-reference RMI_ORACLE_FILTER='^rmi.vision.output$' \
  target/intern_decision/oracle-build/bin/llama-mtmd-debug -m "$RMI_INTERN_DECISION_0_8B_TEXT" \
  --mmproj models/Intern-Decision-0.8B-GGUF/mmproj-intern-decision-0.8b-oracle-F32.gguf \
  -p encode -n 512 --image blue -c 512 --no-warmup --no-repack -ngl 0 -t 8 -fa off > target/intern_decision/vision-reference.log 2>&1
RMI_INTERN_VISION_REFERENCE=target/intern_decision/vision-reference/0000.f32 \
  cargo test --profile release-fast --test intern_decision_vision_oracle intern_decision_vision -- --ignored --nocapture
```

The test requires relative L2 < 0.01. The widener asserts that every
dequantized weight is exactly preserved. The measured relative L2 against
llama.cpp is **0.00774** with the widened F32 weights (bound 0.01); the original
F16 execution differs by **0.05106**. This is numerical validation, not
equality with the original F16 execution or a bitwise parity claim.

## Real CLI smoke

Intern-Decision identifies the blue-image fixture correctly:

```sh
target/release-fast/rust-model-inference --model "$RMI_INTERN_DECISION_0_8B_TEXT" --mmproj "$RMI_INTERN_DECISION_0_8B_MMPROJ" --image path/to/blue-64.png --prompt 'Reply with one word: what color is this image?' --max-tokens 8 --max-context 256 --threads 8
```

This is a semantic smoke check; its generation template is not asserted identical
to llama.cpp. The numerical acceptance remains the tokenizer, text-logit and full
projection tests above.

## Regression checks

```sh
./.venv/bin/python -m unittest tools.converter.utils.test_model_metadata tools.oracle.intern_decision.test_compare_text
cargo fmt --all -- --check
cargo check --profile release-fast --features vulkan,parity-trace --lib --bin rust-model-inference
```

## Known limitation: the 4B variant

The converter accepts the 4B variant and checks its shapes, but the engine's
Qwen3.5 recurrent projection does not accept that SSM shape yet, so the 4B
cannot be served. The 0.8B is the verified size.
