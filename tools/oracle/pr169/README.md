# PR #169 real-weight verification

This directory pins the independent CPU reference and the acceptance boundaries for
EmbeddingGemma-2, PhysBrain1.5-2B and Occamy-1.0.

**Scope on this branch**: only the Occamy-1.0 work is here. The EmbeddingGemma-2 and
PhysBrain1.5-2B sections are kept because the Occamy verification shared this record
(one oracle run, one set of hashes); their model code lives on other branches, so
their commands do not run here.
`verification.json` records the exact GGUF hashes, formats and measured errors.
Lossless loading/layout/token IDs are checked exactly. Quantized matmul, activation
rounding and approximate nonlinear operators use numerical checks, as requested;
none of the following results is a claim of whole-model bitwise parity.

## Inputs and reference

On the validation machine `models` points to
`/Users/gouzi/Documents/git/rust-model-inference/models`. All downloads and local
conversions remain there; no weights or temporary reference checkout are committed.
Dependencies use the repository `.venv`, and Cargo uses a repository `target`.

The reference is ggml-org/llama.cpp commit
`11fe02151f79c41d0d4af7da708755d73b9c0da6`. Use a **temporary checkout**:

```sh
git clone https://github.com/ggml-org/llama.cpp target/pr169/llama.cpp
git -C target/pr169/llama.cpp checkout 11fe02151f79c41d0d4af7da708755d73b9c0da6
tools/oracle/pr169/build.sh target/pr169/llama.cpp target/pr169/oracle-build
RMI_ORACLE_SCALAR=1 tools/oracle/pr169/build.sh target/pr169/llama.cpp target/pr169/oracle-scalar-build
```

The build disables Accelerate, BLAS, GPU backends and KleidiAI. The optional scalar
build also disables NEON and llamafile and turns off FP contraction. The patch only
adds checkpoint dumps and greedy token reporting, without changing model math.
Dumps are contiguous native little-endian F32 on the tested Apple Silicon machine;
`index.tsv` includes names and full shapes. Create a fresh, empty dump directory
before every run: the index and Rust traces append to existing files.

## Tokenizer and contracts

```sh
cargo test --profile release-fast --features parity-trace --test pr169_tokenizer_oracle -- --ignored --test-threads=1
cargo test --profile release-fast --test embeddinggemma_2 --test physbrain_1_5_2b
```

Tokenizer cases include empty input, English, Unicode, whitespace, ChatML and
ordinary tool/thinking added tokens. EmbeddingGemma token IDs are checked against
HF fixtures in its text Oracle. PhysBrain converter preserves CONTROL vs
USER_DEFINED token types and write llama.cpp-compatible 32-bit metadata.
PhysBrain RoPE spans the full 128-dimension head.

To reproduce the local conversions from downloaded safetensors:

```sh
./.venv/bin/python -m tools.converter.physbrain.convert_physbrain --hf-dir models/PhysBrain1.5-2B --out-dir models/PhysBrain1.5-2B-GGUF
./.venv/bin/python -m tools.converter.embeddinggemma2.convert_embeddinggemma2 --help
```

Never replace a GGUF while a process maps it. The historical EmbeddingGemma
filename ends in `F16`, but its actual tensor types are Q8_0/BF16/F32.

## Text logits and generation

Example for PhysBrain (Bash/Zsh; preserve the trailing newlines with `-p`):

```sh
prompt=$<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
<think>

</think>

'
mkdir -p target/pr169/text-reference
RMI_ORACLE_DUMP=target/pr169/text-reference RMI_ORACLE_FILTER='^result_output$' \
  target/pr169/oracle-build/bin/llama-eval-callback -m "$RMI_PHYSBRAIN_1_5_2B_TEXT" \
  -p "$prompt" -n 8 -c 512 -ngl 0 -t 8 -fa off -ctk f16 -ctv f16 > target/pr169/text-reference.log 2>&1
cargo build --profile release-fast --features parity-trace --bin rust-model-inference
RMI_PARITY_TRACE=target/pr169/text-rust.jsonl RMI_PARITY_FILTER='qwen35.prompt_ids,prompt_ids,result_output' \
  target/release-fast/rust-model-inference --model "$RMI_PHYSBRAIN_1_5_2B_TEXT" \
  --prompt 'What is 2+2?' --max-tokens 8 --max-context 512 --threads 8 --kv-cache f16 > target/pr169/text-rust.log 2>&1
./.venv/bin/python tools/oracle/pr169/compare_text.py target/pr169/text-rust.jsonl target/pr169/text-reference target/pr169/text-reference.log --relative-l2 0.05
```

Use the other model's text path for the other rows in `verification.json`.
PhysBrain requires `--kv-cache f16` in Rust and `-ctk f16 -ctv f16` in llama.cpp.
Bounds are regression envelopes for these fixtures: PhysBrain 0.05 and
Occamy 0.10. They are not accuracy guarantees for arbitrary prompts.
The comparator rejects missing prompt IDs, steps, shapes, payloads and nonfinite
values, and checks every vocabulary logit at the final prefill row and each decode.
It requires exact greedy IDs when both runs follow their own generated history.

Occamy's scalar comparison uses `RMI_SCALAR=1` on Rust and
`oracle-scalar-build/bin/llama-eval-callback` above. Its eight greedy IDs match.
The normal SIMD reference chooses a different token at step 4 (zero based), with
only a 0.00937 top-two logit margin. To compare default approximate kernels fairly,
replay that reference's tokens rather than comparing logits on different histories:

```sh
RMI_TEXT_MODEL="$RMI_OCCAMY_TEXT" RMI_LLAMA_REFERENCE_LOG=target/pr169/text-reference.log \
  RMI_PARITY_TRACE=target/pr169/teacher.jsonl RMI_PARITY_FILTER='prompt_ids,teacher_forced_ids,result_output' \
  cargo test --profile release-fast --features parity-trace --test pr169_teacher_forcing -- --ignored
./.venv/bin/python tools/oracle/pr169/compare_text.py target/pr169/teacher.jsonl target/pr169/text-reference target/pr169/text-reference.log --relative-l2 0.10 --teacher-forced
```

For that command first regenerate `text-reference` with the Occamy model. Replay
verifies the exact complete history and reports both argmax sequences; it does not
claim that different greedy sequences agree.

## Vision projections and DeepStack

`llama-mtmd-debug -p encode --image blue -n 512` supplies already normalized RGB
floats `[0,0,1]` directly. It does **not** preprocess a blue PNG. The Rust tests use
that identical input and compare all projected values, including all three
PhysBrain DeepStack outputs in the reference's token/feature order.

Occamy enters `models::qwen35::vision` through the actual hybrid CLI. Its
normal F16 kernel uses F32 activations on the tested SIMD host.
PhysBrain enters `models::qwen3::vision`, including DeepStack, and widens weights
while keeping activations in F32. Use `widen_projector.py` for each reference
projector to widen the same weight values exactly and separate architecture/layout
errors from the reference's extra F16 activation rounding. Rust still loads the
original F16 projector in all three tests.

For Occamy, create an exactly widened F32 reference projector and set
`RMI_OCCAMY_VISION_REFERENCE`. For PhysBrain, create a local F32 reference
projector, then substitute that projector and its text model in the reference command above:

```sh
PYTHONPATH=target/pr169/llama.cpp/gguf-py ./.venv/bin/python tools/oracle/pr169/widen_projector.py "$RMI_PHYSBRAIN_1_5_2B_MMPROJ" models/PhysBrain1.5-2B-GGUF/mmproj-physbrain-1.5-2b-oracle-F32.gguf
```

Set `RMI_PHYSBRAIN_VISION_REFERENCE` to that dump and run `physbrain_vision`. Both
tests require relative L2 < 0.01. The widener asserts that every dequantized
weight is exactly preserved. The actual Occamy/PhysBrain projection relative
L2 is 0.3362% / 0.4601% against F32 activations. The original F16
execution differs by 2.68% / about 4.42%, respectively. This is numerical
validation,
not equality with the original F16 execution or a bitwise parity claim.

Loader payload/layout tests also compare lossless operations exactly. Qwen3-VL now
uses the existing vision RoPE operator; production precompute preserves supplied
weights instead of silently requantizing them to Q8. Keeping F32 caches increases memory
use and may affect throughput; no performance improvement is claimed.

## Real CLI smoke

The final Occamy run identifies a solid RGB `[0,0,255]` 64x64 PNG as
`Blue` / `blue`. Each CLI resizes it to 96x96 (nine vision tokens). Use a matching
PNG with the corresponding text/mmproj paths:

```sh
target/release-fast/rust-model-inference --model "$RMI_OCCAMY_TEXT" --mmproj "$RMI_OCCAMY_MMPROJ" --image path/to/blue-64.png --prompt 'Reply with one word: what color is this image?' --max-tokens 8 --max-context 256 --threads 8 --temperature 0
```

The original F16 llama.cpp multimodal CLI also answers `blue` with
`--image-min-tokens 9 --image-max-tokens 9`. This is a semantic smoke check; its
generation template is not asserted identical to Rust. PhysBrain also
identifies the blue-image fixture correctly. Its numerical acceptance remains the
separate tokenizer, text-logit and full projection tests above.

## EmbeddingGemma-2 and regression checks

```sh
cargo test --profile release-fast --test embeddinggemma_2_text_oracle --test embeddinggemma_2_multimodal_oracle --test embeddinggemma_2_smoke --test embeddinggemma_2_vision_smoke --test embeddinggemma_2_audio_smoke -- --nocapture
./.venv/bin/python -m unittest tools.converter.utils.test_model_metadata tools.oracle.pr169.test_compare_text
cargo fmt --all -- --check
cargo check --profile release-fast --features vulkan,parity-trace --lib --bin rust-model-inference
```

HF `transformers` 5.19.0 fixtures cover five text inputs including 552 tokens across
the sliding-window boundary, vision/audio tower outputs, projected soft tokens,
and final 768-dimensional embeddings. Text uses cosine >0.999 and relative L2
<0.03; tower/projection relative L2 <1e-4; final media embedding relative L2 <0.02.
Fixture generators are in `tests/oracle/run_embeddinggemma2_*_oracle.py`.
CLI supports `--embedding --image ...` or `--embedding --audio ...` with the single
GGUF; simultaneous image/audio and video are explicitly rejected.

The focused Qwen3.5 run excludes the previously failing
`qwen35_config_rejects_invalid_tensor_dimensions`. The NEON LayerNorm stride bug
that skipped four of every eight values is repaired; both its new lane/tail
regression and the previously failing grouped-variance test are included. The
prefill trace regression is also fixed and included. No full-suite pass, GPU
numerical alignment, robot-action execution or performance benchmark is claimed.

## Occamy-1.0 re-verification on the aarch64 box (2026-11-05)

The `Occamy-1.0` branch (from `main`) was re-verified with the real weights on
an aarch64 host (NVIDIA GB10 box, 8 inference threads, `release-fast`). Both
files match the hashes in `verification.json`: `occamy-1.0-Q8_0.gguf`
(`9bd6ecea…`, 36,903,139,456 bytes) and `mmproj-occamy-1.0-F16.gguf`
(`8f827a1e…`, 899,282,944 bytes).

| check | result |
| --- | --- |
| greedy ids vs llama.cpp | **8/8 identical**, and identical to the recorded default row |
| vision projection relative L2 | **0.00070317** (bound 0.01; the Apple Silicon run measured 0.00336136) |
| tokenizer vs the pinned llama.cpp tokenizer | exact on every fixture |
| CLI smoke | the 64x64 blue PNG answers `blue`, 9 vision tokens, CPU and Vulkan alike |

The per-step logits envelope does not transfer across hosts. This host measures

```
0.08729171 0.04906173 0.02428209 0.02397511 0.03363240 0.06843563 0.11983501 0.08049776
```

against the recorded Apple Silicon run

```
0.08481082 0.07745719 0.03895792 0.04257759 0.04971924 0.07071221 0.06171023 0.05796181
```

so step 6 lands at 0.11984, above the recorded bound of 0.1, while the recorded
run's own maximum was 0.08481 — the envelope was already close to its edge. The
argmax agrees with llama.cpp at every step on both hosts. The fragile step is
step 4, where llama.cpp's top-two margin is 0.247 but the Rust's is 0.039: a
host difference can flip that token, which is what the recorded run saw.

The CPU and the device path also choose different greedy tokens for this
checkpoint — CPU `2 + 2 = 4`, device `2 + 2 equals **4**.` — and that reproduces
identically on `origin/main` without any of the Occamy work, so it is not
caused by it. The earlier roadmap note that occamy-1.0 matches bit for bit
between GPU and CPU at `--temp 0` was measured with a different weight set.

Commands are in `verification.json` under `occamy_reverification_2026_11_05`.
