# Holo4-35B-A3B GGUF CPU / Vulkan validation

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
video, long context, or a computer-control agent/harness. Vulkan coverage and
remaining numerical failures are recorded below.

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

## Vulkan decoder

The decoder now uploads each complete expert bank once and binds per-expert
ranges. Router, routed/shared expert projections, SiLU, attention and GDN state
updates run on Vulkan. Softmax/top-k routing and the ordered weighted expert sum
run on the host. This is a hybrid execution path, with host synchronization per
MoE row; no performance improvement is claimed. Strict `--compute vulkan`
supports text. For images, `--compute auto` uses the CPU vision encoder followed
by Vulkan text decoding; strict Vulkan image mode is still rejected.

On Apple M3 Max, the 40 real MoE layers each pass a three-row same-input CPU/GPU
comparison (maximum relative L2 `4.095519109388936e-8`, bound `0.02`). Seven-token
prefill in chunks of four, continuation, and reset reproduce the same GPU
logits as single-token execution (relative L2 `0.0` for prefill and continuation).
The original Q4_K_M artifact has 733 static uploads / 20,869,705,280 bytes.
The real text CLI with `--compute vulkan --prefill-batch-size 4` returns `Paris`;
the image CLI with Auto returns `Blue`, with Vulkan decoder trace evidence.

The full eight-step `Hello` comparison keeps all greedy IDs equal, but CPU/GPU
logits reach relative L2 `0.05883709551014592`. **The integration test's `0.02`
bound fails and has not been widened.** Fixed-Oracle matched-history generation
logits reach `0.09326496` (English), `0.11156020` (math) and `0.13002159`
(Chinese); the latter two fail the `0.10` envelope. English/math argmax IDs agree
at all eight positions; Chinese differs at the first position. Chinese prefill
logits reach `0.35463069`. Finite outputs and readable CLI text are not evidence
that these numerical failures are resolved.

The two-token `[1,2]` diagnostic localizes the earliest inspected GPU difference:
step 0, recurrent layer 18, GDN final output relative L2 `1.64278625e-8`, then MoE
input max absolute difference `5.96046448e-8`. MoE input/output agree exactly in
the preceding 18 layers. At step 1, recurrent layer 5 has GDN output relative L2
`3.23057533e-8`. Later quantized projections amplify the discrepancy; isolated
MoE validation does not certify all downstream residuals as harmless.

```sh
cargo build --profile release-fast --features vulkan,parity-trace --bin rust-model-inference
RMI_QWEN35_MODEL="$RMI_HOLO4_MODEL" \
  cargo test --profile release-fast --features vulkan,parity-trace --lib \
  qwen35_real_moe_layers_run_on_vulkan -- --ignored --nocapture
cargo test --profile release-fast --features vulkan,parity-trace --test holo4 \
  holo4_vulkan_batched_prefill_and_reset -- --ignored --nocapture
# Expected to fail the unchanged 0.02 bound; report is written before asserting.
RMI_HOLO4_GPU_REPORT=target/holo4/gpu-logits.json \
  cargo test --profile release-fast --features vulkan,parity-trace --test holo4 \
  holo4_vulkan_text -- --ignored --nocapture
```

For the fixed-Oracle replay above, add `RMI_HOLO4_COMPUTE=vulkan` and build with
`vulkan,parity-trace`. Keep prefill batch 1 for token-major checkpoints.

The checked-in GDN SPIR-V is generated with glslang `15.1.0`, matching the
Ubuntu 24.04 CI package `15.1.0-2~ubuntu0.24.04.2`. macOS glslang `16.6.0`
produces different constant folding/IDs, so its binary fails the repository's
byte-for-byte rebuild check. Regenerate with the CI compiler and run
`bash scripts/vulkan-shaders.sh check`; dependencies can be extracted under
`target/` without installing system packages. The optimized arithmetic graphs
compare equal, and the final binary is rerun on both GPUs before reporting
the numerical results above.

## CLI and remote checks

The real CLI with the original projector and `--image target/holo4/blue.png
--prompt 'What color is this image? Answer in one word.' --jinja --temp 0
--max-tokens 8 --max-context 512 --threads 8 --kv-cache f32
--prefill-batch-size 1` returned `blue`. The pinned `llama-mtmd-cli` returned
`Blue` with the same image and a template forcing `enable_thinking=false`.
This is an image-entry smoke check; projection numerical comparison is above.

The requested SSH host was tested in `/workspace/rust-model-inference-holo4`
on Ubuntu 24.04 x86_64 / EPYC 9334, using Rust 1.98.1 installed inside its
`target/`. The original CPU-only run passed 82 focused Qwen3.5 tests / 5 ignored
(macOS ARM: 88 / 5). With `vulkan,parity-trace`, the final source passes 98 / 14
on x86 and 104 / 14 on ARM. These counts exclude the same two failures
reproduced at the original base `973210b6`:
`qwen35_config_rejects_invalid_tensor_dimensions` and
`qwen35_trace_metadata_is_independent_of_prefill_batch_size`.

The full LLM upload is now complete, with the entire file's SHA256 verified
before renaming the `.incomplete` file. Both original artifacts are under
`/Users/gouzi/Documents/git/rust-model-inference/models/Holo4-35B-A3B-GGUF`
on both machines. The real F16 projector passes against the pinned macOS F32
reference (relative L2 `0.00090910`, 32768 values).

The remote Vulkan device is **AMD Radeon Graphics (RADV NAVI31)**, with
51,522,830,336 bytes VRAM, using
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json`.
All 40 real MoE layers pass the three-row same-input CPU/GPU comparison at
maximum relative L2 `1.2013590525231618e-6`. Batched prefill, continuation and
reset pass with relative L2 `[0.0, 0.0]`. Strict Vulkan uses 733 static uploads /
20,869,705,280 bytes. CPU and Vulkan text CLIs both return
`The capital of France is Paris.`; the image CPU/Auto CLIs return `blue`/`Blue`.
Auto logs explicitly show the Vulkan decoder after CPU vision encoding.

The remote eight-step `Hello` CPU/GPU comparison retains the same eight greedy
IDs, but maximum logits relative L2 is `0.04931125640742786`, failing the
unchanged `0.02` bound. Fixed-Oracle replay checks prompt/history IDs and raw
prompt embedding words exactly (38912/38912/34816 values). It compares every
logit at all 26/26/22 positions, with these measured maxima:

| Fixture | M3 Max generation | NAVI31 generation | NAVI31 all-prefill | NAVI31 generation bound 0.10 |
|---|---:|---:|---:|---|
| English | 0.09326496 | 0.10057801 | 0.18998320 | fail |
| Math | 0.11156020 | 0.09449778 | 0.11373808 | pass |
| Chinese | 0.13002159 | 0.14532538 | 0.27327580 | fail |

All remote argmax IDs agree on the Oracle history; the local Chinese first
argmax differs. The `holo4_teacher_forced_text` test itself checks only finite
execution: its exit status is not numerical acceptance. The separate comparator
reports the failures above. These results leave full text alignment unresolved.

The focused remote operator run passes:

```sh
SILU_SCALE_ONLY=1 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  target/release-fast/examples/vk_ops_check --formats q4_k,q6_k --rows 3
```

This includes Q4_K/Q6_K, exact Q8_K activation quantization, RMSNorm, RoPE,
tiny-Q GDN convolution, SSM and attention/KV checks. The broader operator tool
does **not** fully pass: its separate large SiLU device-local probe stops at
`device arena requires read_f32_into`; adding F32 to the focused formats fails
its strict generic F32 probe (about `3.624e-5` error). The Holo4 host-visible
arena and real F32 router/alpha/beta execution were tested separately through
the model. Neither generic failure was hidden or counted as a passing check.

Host routing/reduction introduces synchronization. One remote CLI smoke run
reported 1.3 generation tokens/s on Vulkan versus 15.6 on CPU; this is not a
controlled benchmark, and GPU performance improvement is not established.
Video, long context, server/multiturn, the computer-control harness, other
quantizations, the full repository suite and older shared-GDN artifacts remain
unverified. The local evidence paths and SHA256 digests are in `verification.json`;
large traces and weights are not committed.
