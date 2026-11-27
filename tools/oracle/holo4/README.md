# Holo4-35B-A3B GGUF CPU / Vulkan validation

For the dense Holo4-27B (`qwen35`) artifacts and validation, see
[the separate 27B guide](27b/README.md). Both sizes share the decoder/vision kernels.

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
ranges. Immutable buffers prefer device-local VRAM on discrete GPUs, with
staging and the existing out-of-memory host fallback; UMA keeps mapped memory.
Router, routed/shared expert projections, SiLU, attention and GDN state
updates run on Vulkan. Softmax/top-k routing and the ordered weighted expert sum
run on the host. This is a hybrid execution path, with host synchronization per
MoE row. AMD/Apple batch up to nine experts per submission; other vendors retain
two-expert batches for the Intel short-submission workaround. Strict `--compute vulkan`
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
controlled benchmark; the resident-forward follow-up below records the optimized path.
Video, long context, server/multiturn, the computer-control harness, other
quantizations, the full repository suite and older shared-GDN artifacts remain
unverified. The local evidence paths and SHA256 digests are in `verification.json`;
large traces and weights are not committed.

## Resident-forward performance reproduction

Build the standalone harness against the release library (no external acceleration
libraries). It uses 32 prompt tokens, 64 teacher-forced decode steps, eight CPU
threads, F32 KV, batch 16 and capacity 512. CPU and strict Vulkan each warm up once;
five alternating rounds exclude load, upload, reset, tokenization and sampling,
and include synchronized logits/state readback. It asserts no uploads during
measurement, the selected backend, finite logits and exact repeated GPU/CPU runs.

```sh
cargo build --locked --release --features vulkan --lib
rustc --edition=2021 tools/oracle/holo4/benchmark.rs \
  --extern rust_model_inference=target/release/librust_model_inference.rlib \
  --extern serde_json="$(find target/release/deps -name 'libserde_json-*.rlib' -print -quit)" \
  -L dependency=target/release/deps -C opt-level=3 -C lto=fat \
  -C codegen-units=1 -C panic=abort -o target/holo4/benchmark
target/holo4/benchmark "$RMI_HOLO4_MODEL" > target/holo4/benchmark.jsonl
```

For before/after numerical regression, build the same harness against each
version's release library, then run `RMI_SPEED_ROUNDS=0` with a different
`RMI_SPEED_SNAPSHOT=target/holo4/before.f32` / `after.f32` path. Compare the raw
F32 words at the last prefill position and every decode step. This compares the
two GPU implementations on identical CPU-generated history; it does not certify
CPU/GPU or Oracle alignment. Snapshot collection is excluded from performance
runs. The existing whole-model numerical failures above remain unresolved.

On 2026-10-23, five-round median decode throughput on M3 Max changed from
3.546 to 4.068 token/s (+14.7%); the new CPU median was 14.434 token/s. GPU
prefill changed from 4.982 to 6.117 token/s. The SSH NAVI31 host recorded
0.647 to 1.930 GPU decode token/s, versus 14.715 CPU token/s, but different
concurrent GPU clients prevent attributing this ratio to the change. The local
host also had background load and substantial swap use; neither is an exclusive
or power-controlled benchmark. GPU remains slower than CPU on these runs.

Submissions per 32+64-token round dropped from 21,972 to 6,612; synchronized host
readback remains 5.822 GiB. Remote DRM fdinfo shows this process's VRAM changing
from about 0.21 to 19.65 GiB, and GTT from 19.61 to 0.17 GiB. On each device,
all 16,140,800 F32 logits at the 65 measured positions compare exactly before
and after this optimization. All 40 same-input MoE checks and batch/continuation/
reset checks retain their previous errors. This regression does not resolve
the whole-model CPU/GPU/Oracle numerical failures. See `verification.json`'s
`gpu_performance_followup` for ranges, raw-evidence digests and baseline failures.

## GPU MoE reduction and initial narrow Q4_K projections

The next optimization keeps expert outputs on the GPU: an ordered weighted
reduction fuses residual addition, replacing the per-layer expert-output
readback and CPU reduction. Router softmax/top-k and the shared-gate sigmoid
retain their host implementation. Scores use separate per-row arena storage;
a sticky GPU NaN/Inf flag is checked before committing the chunk. CPU shadow
state, fallback/abort/reset behavior and the shorter submission policy for
other vendors are preserved. Requests for MoE output traces retain host
reduction so the existing checkpoints remain available.

The initial optimization (`3f13f226`) uses eight lanes for single-row Q4_K
projections with at most 1024 output columns. The follow-up below extends this
to 4096. A 512-column expert projection therefore dispatches 64 rather than
eight workgroups per weight group. Both the block-first and lane-first
reductions preserve the old shader's arithmetic order. At that baseline, larger
or batched projections retain the existing kernel.

Focused hardware checks (run on both M3 Max and the requested NAVI31 host):

```sh
cargo test --locked --profile release-fast --features vulkan --lib \
  q4_k_cooperative_matches_row_kernel_in_both_reductions -- --ignored --nocapture
cargo test --locked --profile release-fast --features vulkan --lib \
  moe_reduce_preserves_slot_order_and_sticky_finite_check -- --ignored --nocapture
```

The Q4_K check compares both modes, three grouped weights and tail dimensions
against the old GPU shader by F32 bits. The reducer checks slot order, tails,
validation before recording, one submission boundary and sticky NaN/Inf
rejection. All 40 real MoE layers retain max same-input CPU/GPU relative L2
`4.095519109388936e-8` on M3 and `1.2013590525231618e-6` on NAVI31; the existing
`0.02` bound is unchanged. Batch/continuation relative L2 remains `[0,0]`, with
exact reset replay. These checks do not resolve the whole-model alignment
failures documented above.

Five-round release medians for this optimization, relative to `c9fbceee`
(same resident-forward protocol above, 2026-10-23):

| Device | CPU decode, new | GPU decode, before | GPU decode, new | GPU prefill, before → new |
|---|---:|---:|---:|---:|
| Apple M3 Max | 12.758 | 4.042 | 5.577 | 6.115 → 9.201 |
| SSH RADV NAVI31 | 18.611 | 2.084 | 3.584 | 3.284 → 8.358 |

All rates are token/s. GPU decode ranges are `3.996–4.056` → `5.256–5.639`
on M3 and `2.063–2.088` → `3.519–3.622` on NAVI31. The observed median changes
are +38.0% and +72.0%; hardware exclusivity and power were not controlled, so
these ratios are not isolated causal speedup claims. M3 had background load
and about 26.8 GiB swap in use. Unlike the earlier remote run, this run's
five-second DRM samples found only the benchmark client on the selected
render node; that does not establish hardware exclusivity. CPU decode is
still about 2.29× / 5.19× faster on the two hosts.

Submissions per 32+64-token round fall from `6612` to `3972`, host writes from
`32,268,288` to `949,248` bytes, and host reads from `6,251,645,952` to
`5,968,530,696` bytes. CPU routing synchronization and approximately 5.56 GiB
of synchronized state/logit readback remain; per-kernel timestamps were not
measured, so these counters do not assign every remaining cost to a kernel.

On each GPU, all `16,140,800` raw F32 logits at the last prefill plus 64 decode
positions remain byte-for-byte equal to `c9fbceee`. This is a regression check
against the previous GPU implementation; reduced-precision CPU/GPU and Oracle
paths retain their numerical bounds and existing failures. Both release strict
Vulkan CLIs finish the discount-price example with `59` yuan and natural EOS
`248046`. Dense four-axis tests also pass on both devices. Regular Qwen35 tests
pass `104` on ARM / `98` on x86, each with `13` ignored and the known config
error-message assertion excluded.

See `verification.json`'s `gpu_reduction_and_q4_followup` for ranges, snapshots,
source/evidence hashes and the build-source formatting note. Raw logs, snapshots
and the local report are under `target/holo4/gpu-kquant-2026-10-23/` (not committed).
The full repository suite, long context and other GPU vendors were not tested.
The full shader rebuild passes with pinned glslang `15.1.0` on the SSH host.
Local `16.6.0` fails byte-for-byte rebuild of the existing `softmax.spv`;
local manifest and the new binaries' SPIR-V validation pass independently.

## Wider single-row Q4_K projections

The follow-up to `3f13f226` extends the existing cooperative-kernel selection
from 1024 to 4096 output columns, including more recurrent and dense projections.
The only production change is that dispatch limit in `src/vulkan/ops.rs`;
GLSL/SPIR-V, reduction order, batched projections and wider heads are unchanged.
The same hardware test now also covers `(n_in,n_out)` pairs `(512,2048)`,
`(4096,2048)`, `(2048,4096)`, `(4096,4096)` and `(12288,4096)`, with both
reduction orders, three grouped weights and unequal output widths/tails.

Temporary release-fast wall-clock instrumentation over 64 decode steps found
that the layer loop and its waits dominated the measured host stages: median
158.31 ms on M3 and 226.02 ms on NAVI31, versus 9.26/7.20 ms for the head stage.
Queued GPU work can cross these stage boundaries; these are not per-kernel GPU
timestamps. The instrumentation was removed before final tests and release
builds. CPU shadow state, finite scans, commit validation and fallback were
preserved. An isolated synthetic operator experiment with 30 repeats measured
old/new elapsed-time ratios of 1.218–1.845 on M3 and 1.188–2.376 on NAVI31 for
the added output widths. Those ratios do not establish whole-model speedup.

Five-round release medians for the wider-kernel follow-up (same resident-forward protocol, baseline `3f13f226`):

| Device | CPU decode, before → new | GPU decode, before → new | GPU prefill, before → new |
|---|---:|---:|---:|
| Apple M3 Max | 11.180 → 10.215 | 5.344 → 5.706 | 8.139 → 5.384 |
| SSH RADV NAVI31 | 16.465 → 20.946 | 3.629 → 4.975 | 8.409 → 10.391 |

All rates are token/s. GPU decode ranges are `5.306–5.430` → `4.420–6.103`
on M3 and `3.616–3.695` → `4.876–5.050` on NAVI31. Observed decode median
changes are +6.8% / +37.1%, while local prefill regressed. CPU controls also
changed, from 11.180 to 10.215 / 16.465 to 20.946 token/s. Local one-minute
load average rose from 23.16 to 114.86 and swap from about 20.09 to 22.71 GiB.
No stable local performance gain is established. Neither host was exclusive
or power-controlled, and these ratios do not isolate the effect of the code.
Remote five-second samples found only the benchmark on the selected render
node; this does not prove exclusivity. CPU remains about 1.79× / 4.21× faster
than GPU for decode in this run.

Each measured GPU round still has `3972` submissions, `5,968,530,696` host-read
bytes and `949,248` host-write bytes. Those counters are identical across all
ten GPU samples; this change only selects a different existing kernel.

On each GPU, all `16,140,800` F32 logits at the 65 measured positions remain
byte-for-byte equal to the previous GPU implementation. The 40-layer
same-input MoE errors remain `4.095519109388936e-8` / `1.2013590525231618e-6`
(unchanged bound `0.02`); batch/continuation errors remain `[0,0]`, reset replay
is exact, and regular Qwen35 tests pass 104 / 98 with 13 ignored and the same
baseline config assertion excluded. Both release strict-Vulkan CLIs complete
the discount-price question with `59` yuan, EOS `248046` and exit code 0.
This optimization does not resolve the existing whole-model CPU/GPU/Oracle
alignment failures; benchmark warmup CPU/GPU relative L2 remains
`0.18000214` / `0.23536753`.

See `verification.json`'s `gpu_wider_q4_followup` for raw ranges, snapshots,
source/evidence hashes and the profiling limits. Raw logs and the report are
under `target/holo4/gpu-shadow-2026-10-23/` (not committed). No shader sources
or binaries changed in this follow-up; the manifest was checked, and a full
shader rebuild was not repeated. The full repository suite, long context,
other GPU vendors and per-kernel GPU timestamps remain untested.
