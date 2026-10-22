# Holo4-35B-A3B GGUF CPU validation

Model: [Hcompany/Holo4-35B-A3B-GGUF](https://modelscope.cn/models/Hcompany/Holo4-35B-A3B-GGUF),
ModelScope revision `a298a0585d798c32341af11cdcddfb08f87e6744`.
The model card calls this Qwen3.6 MoE; the actual GGUF architecture is
`qwen35moe`, with the existing Qwen3.5 hybrid attention/SSM/MoE contract.
There is no new architecture alias or separate Holo implementation.

## Artifacts and scope

| File | Bytes | SHA-256 |
|---|---:|---|
| `q4_k_m.gguf` | 21297870912 | `4c51212a4b5673d6a4215b9d708cd1730f1b6b4dbb5f75552df18de8341ce669` |
| `mmproj.f16.gguf` | 899282976 | `e1076d4179a6d82dbc7f129f043ac5cb03e4584ef5c1ed4a0c31eca5d1a187e4` |

The LLM has 733 tensors, 40 blocks, 2048 hidden dimensions, 248320 vocabulary
entries, 16 attention heads, 2 KV heads and 256-dimensional Q/K/V heads. Thirty
blocks are recurrent; every fourth block uses full attention. The GDN dimensions
are state 128, groups 16, rank 32, inner 4096 and convolution 4. RoPE uses base
10000000 and sections `[11,11,10,0]`. Every MoE block has 256 experts, top-8
routing, expert width 512 and shared width 512. Importantly, the Q4_K_M artifact
mixes Q4_K up/gate expert matrices with Q6_K down matrices; their types and
strides must be retained separately.

The projector is `clip` / `qwen3vl_merger`: 27 layers, hidden 1152, FFN 4304,
16 heads, patch 16, spatial merge 2, output 2048, mean/std `[0.5,0.5,0.5]`.
It has no DeepStack outputs. Use the GGUF's own chat template (`--jinja`).

**Status: experimental.** Text and image inference run, but complete text
numerical alignment is not established; the Chinese fixture exceeds the tested
0.10 logit envelope. See `verification.json` for passing checks and residuals.

Exact checks cover tokenizer IDs, input embedding words, tensor types, shapes,
and payload sizes. Quantized matmul, F16 activation rounding, SIMD reductions
and nonlinear operators use numerical error bounds. No whole-model bitwise
parity is claimed. This does not validate other Holo4 sizes, quantizations,
GPU execution, video, long context, or a computer-control agent/harness.

## Reference and builds

The independent CPU reference is ggml-org/llama.cpp
`11fe02151f79c41d0d4af7da708755d73b9c0da6`. Work in a temporary checkout:

```sh
git clone https://github.com/ggml-org/llama.cpp target/holo4/llama.cpp
git -C target/holo4/llama.cpp checkout 11fe02151f79c41d0d4af7da708755d73b9c0da6
tools/oracle/holo4/build.sh target/holo4/llama.cpp target/holo4/oracle-build
cargo build --profile release-fast --features parity-trace --bin rust-model-inference
```

The patch only dumps checkpoints and reports prompt/greedy IDs. Build flags
turn off Accelerate, BLAS, KleidiAI and GPU backends. Fresh output directories
are required because checkpoint indexes and Rust traces append to existing files.

Download only the named files, using the shared model directory and its download venv:

```sh
(
cd /Users/gouzi/Documents/git/rust-model-inference/models
.venv/bin/modelscope download --model Hcompany/Holo4-35B-A3B-GGUF \
  --revision a298a0585d798c32341af11cdcddfb08f87e6744 \
  q4_k_m.gguf mmproj.f16.gguf README.md chat_template.jinja \
  --local_dir ./Holo4-35B-A3B-GGUF
)
```

## Tokenizer and model contract

```sh
export RMI_HOLO4_MODEL=/Users/gouzi/Documents/git/rust-model-inference/models/Holo4-35B-A3B-GGUF/q4_k_m.gguf
export RMI_HOLO4_MMPROJ=/Users/gouzi/Documents/git/rust-model-inference/models/Holo4-35B-A3B-GGUF/mmproj.f16.gguf
export RMI_LLAMA_TOKENIZE=target/holo4/oracle-build/bin/llama-tokenize
cargo test --profile release-fast --features parity-trace --test holo4 holo4_contract_and_tokenizer -- --ignored
```

Seven inputs cover empty text, English, Chinese/emoji, whitespace, ChatML,
tool/thinking tokens and image markers. Both sides add special tokens and parse
specials; the artifact sets `add_bos_token=false`, BOS/PAD 248044 and EOS 248046.

## Text checkpoints, logits and generation

Example for one prompt (Bash or Zsh):

```sh
mkdir -p target/holo4/text-reference
prompt=$'<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
RMI_ORACLE_DUMP=target/holo4/text-reference \
RMI_ORACLE_FILTER='^(model.input_embed|l_out-(0|3|39)|result_norm|result_output)$' \
  target/holo4/oracle-build/bin/llama-eval-callback -m "$RMI_HOLO4_MODEL" \
  -p "$prompt" -n 8 -c 512 -ngl 0 -t 8 -fa off -ctk f32 -ctv f32 \
  -b 1 -ub 1 --no-repack \
  > target/holo4/text-reference.log 2>&1
RMI_PARITY_TRACE=target/holo4/text-rust.jsonl \
RMI_PARITY_FILTER='qwen35.prompt_ids,qwen35.embedding,l_out-0,l_out-3,l_out-39,result_norm,result_output,qwen35.greedy_token_ids' \
  target/release-fast/rust-model-inference --model "$RMI_HOLO4_MODEL" \
  --prompt 'What is the capital of France?' --jinja --temp 0 --max-tokens 8 --max-context 512 \
  --threads 8 --kv-cache f32 --prefill-batch-size 1 > target/holo4/text-rust.log 2>&1
.venv/bin/python tools/oracle/holo4/compare.py target/holo4/text-rust.jsonl \
  target/holo4/text-reference target/holo4/text-reference.log --relative-l2 0.10
```

The comparator checks exact prompt IDs and all greedy IDs, checkpoint counts,
shapes and finite payloads. It compares every vocabulary logit at the final
prefill row and each decode, plus the selected intermediate layers and final
normalization when `--graph` is added. The input embedding is compared as raw
F32 words. The English fixture's eight greedy IDs agree and its generation
logits have maximum relative L2 0.08678880, within 0.10. Selected intermediate
checkpoints reach 0.19468832, so the 0.10 bound does **not** pass the full graph.
`--graph --relative-l2 0.20` reproduces that measured graph diagnostic; it is
not a claim of tight whole-model parity.

For prompts whose free-running histories diverge, replay the Oracle history
through the same session and compare matching positions:

```sh
RMI_HOLO4_ORACLE_LOG=target/holo4/text-reference.log \
RMI_PARITY_TRACE=target/holo4/forced-rust.jsonl \
RMI_PARITY_FILTER='qwen35.prompt_ids,qwen35.embedding,teacher_forced_ids,l_out-0,l_out-3,l_out-39,result_norm,result_output' \
  cargo test --profile release-fast --features parity-trace --test holo4 \
  holo4_teacher_forced_text -- --ignored --nocapture
python3 tools/oracle/holo4/compare.py target/holo4/forced-rust.jsonl \
  target/holo4/text-reference target/holo4/text-reference.log --teacher-forced --relative-l2 0.10
```

The math fixture (`What is 2+2?`) differs at greedy step 1 (spacing); matched-history
logits reach 0.10003591 and selected hidden checkpoints 0.18599869. The Chinese
fixture (`中国的首都是哪里？`) produces `北京。` in Rust versus `中国的首都是北京。`
in the reference. Its matched-history generation logits reach 0.14567876
(prefill logits reach 0.37391150) and selected hidden checkpoints 0.36332688; this remains outside the envelope, rather than a passing
alignment check. Scalar CPU math also retains residual error, so disabling SIMD
does not establish full numerical parity.

The earliest inspected math mismatch is at token index 6, layer 0: GDN output
relative L2 is 6.0e-8, router logits 7.17e-5, and routed output 0.00637 while
all eight selected expert IDs still agree. Subsequent expert sets differ. The
quantized path amplifies small upstream differences; the current evidence does
not prove that every downstream residual is harmless.

Three shared fixes are covered by regressions: independent expert projection
dtypes/strides, complete NEON LayerNorm lane coverage, and the current Oracle's
GDN `x / sqrt(sum(x²) + eps)` implemented as RMSNorm(eps/n) then scale(1/sqrt(n)).
Older Oracle snapshots used denominator clamping; their bitwise reports are
historical and do not certify this updated shared GDN implementation.

## Vision projection

The reference `--image blue` supplies already normalized RGB floats `[0,0,1]`,
not a PNG preprocessing path. At 128x128, the full tower produces 16x2048 values.
The Rust encoder loads the original F16 projector. Compare both native F16
reference execution and the same weights losslessly widened to F32, to separate
architecture errors from extra reference activation rounding:

```sh
PYTHONPATH=target/holo4/llama.cpp/gguf-py .venv/bin/python \
  tools/oracle/holo4/widen_projector.py "$RMI_HOLO4_MMPROJ" \
  "$(dirname "$RMI_HOLO4_MMPROJ")/mmproj.oracle-f32.gguf"
mkdir -p target/holo4/vision-reference
RMI_ORACLE_DUMP=target/holo4/vision-reference RMI_ORACLE_FILTER='^rmi.vision.output$' \
  target/holo4/oracle-build/bin/llama-mtmd-debug -m "$RMI_HOLO4_MODEL" \
  --mmproj "$(dirname "$RMI_HOLO4_MMPROJ")/mmproj.oracle-f32.gguf" \
  -p encode -n 128 --image blue -c 512 --no-warmup --no-repack -ngl 0 -t 8 -fa off
RMI_HOLO4_VISION_REFERENCE=target/holo4/vision-reference/0000.f32 \
  cargo test --profile release-fast --features parity-trace --test holo4 holo4_vision_projection -- --ignored --nocapture
```

For native F16, substitute the original projector and a new output directory.
The acceptance bound is relative L2 < 0.02. Install Python helper dependencies
only in the repository `.venv` (`numpy`, `pyyaml`, `tqdm`). Check the comparator:

```sh
python3 -m unittest tools.oracle.holo4.test_compare
```

## CLI and remote checks

The real CLI with the original projector and `--image target/holo4/blue.png
--prompt 'What color is this image? Answer in one word.' --jinja --temp 0
--max-tokens 8 --max-context 512 --threads 8 --kv-cache f32
--prefill-batch-size 1` returned `blue`. The pinned `llama-mtmd-cli` returned
`Blue` with the same image and a template forcing `enable_thinking=false`.
This is an image-entry smoke check; projection numerical comparison is above.

The requested SSH host was tested in `/workspace/rust-model-inference-holo4`
on Ubuntu 24.04 x86_64, using Rust 1.98.1 installed inside its `target/`:
82 focused Qwen3.5 tests passed, 5 ignored, and the real F16 projector passed
against the pinned macOS F32 reference (relative L2 0.00090910, 32768 values).
The 88-pass macOS count includes ARM-only checks. Both runs exclude the same
two failures reproduced at the original base `973210b6`:
`qwen35_config_rejects_invalid_tensor_dimensions` and
`qwen35_trace_metadata_is_independent_of_prefill_batch_size`.

The remote host could not connect to ModelScope. Full LLM upload is incomplete;
a preallocated/sparse destination must not be treated as the verified artifact.
Remote full-model text/image generation has not been validated. Do not infer
GPU coverage from the presence of ROCm on that host.
