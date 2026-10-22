# Dory Q5_K_M CPU reference replay

Reference: [NVIDIA Dory llama.cpp fork](https://github.com/brianchmielNV/llama.cpp),
commit `98a74c97ea09ac72f6ceda1559808fc0a2f447ce`. The build script clones the
reference before replacing its example harness; model and GGML sources stay
unchanged. No external acceleration library is enabled.

Artifact: `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`, 3,673,288,864 bytes,
SHA-256 `96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0`.
Its contract is `dory`, 72 physical / 192 virtual layers, four recurrent loops,
`per_loop` KV, and a 1024-token sliding window on SWA layers.

## Reproduce

Run from the repository root. A C/C++ compiler, CMake, Rust and Python 3 are
required; Python uses only the standard library.

```sh
git clone https://github.com/brianchmielNV/llama.cpp.git target/dory-reference
git -C target/dory-reference checkout --detach 98a74c97ea09ac72f6ceda1559808fc0a2f447ce
sh tools/oracle/dory/build_oracle.sh target/dory-reference target/dory-oracle-strict
cargo test --profile release-fast --features parity-trace --test dory_q5_k_m --no-run
python3 -m venv .venv

export RMI_DORY_Q5_MODEL="$PWD/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"
export RMI_DORY_THREADS=4
export RMI_DORY_KV=f32
export RMI_DORY_N_PREDICT=8
case_dir="$PWD/target/dory-parity/replay-hello-f32"
mkdir -p "$case_dir"
printf 'Hello' > "$case_dir/prompt.txt"
export RMI_DORY_PROMPT_FILE="$case_dir/prompt.txt"

RMI_PARITY_TRACE="$case_dir/oracle.jsonl" \
  target/dory-oracle-strict/build/bin/llama-eval-callback \
  "$RMI_DORY_Q5_MODEL" "$RMI_DORY_PROMPT_FILE" "$RMI_DORY_N_PREDICT"
RMI_SCALAR=1 RMI_PARITY_TRACE="$case_dir/rust.jsonl" \
  cargo test --profile release-fast --features parity-trace --test dory_q5_k_m \
  dory_q5_reference_trace -- --ignored --exact --nocapture
.venv/bin/python tools/oracle/dory/check.py \
  "$case_dir/oracle.jsonl" "$case_dir/rust.jsonl" --report "$case_dir/report.json"

.venv/bin/python -m unittest discover -s tools/oracle/dory -p test_check.py
RMI_SCALAR=1 cargo test --profile release-fast --test dory_q5_k_m \
  dory_q5_prefill_chunking_and_reset -- --ignored --exact
```

Use a fresh `case_dir` for every Rust replay: the shared trace writer appends
records. Set `RMI_DORY_KV=f16` to check half-precision KV. The harness uses raw
text with `add_special=true`, `parse_special=true`, CPU attention without flash
attention, and greedy sampling. It evaluates every generated token as well.

For an optional full-size sliding-window replay, write `"hello " * 1025` into the prompt file,
set `RMI_DORY_N_PREDICT=2`, and set `RMI_DORY_TRACE_FROM=1022` for **both**
executables. Pass `--first-step 1022` to the checker. The prefix still executes
in full; Rust prefills it in chunks of 32. Positions 1022–1027 capture both
sides of the SWA boundary and two continuation tokens. This expensive full-size
case is not included in the accepted results.

For the short boundary regression, use `Hello`, eight generated tokens, and
`RMI_DORY_TEST_SWA=2` for both executables, without `RMI_DORY_TRACE_FROM`.
The Oracle uses its public metadata override API and Rust changes only the test
model configuration; production metadata and model files stay unchanged. Run
both F32 and F16 KV. This exercises cache eviction and continuation with the
real weights, but does not establish accuracy at the original 1024-token boundary.

## Recorded results

Validated on macOS ARM64 / Apple M3 Max, Rust 1.98.1, `release-fast`, four
threads on both sides. Dory selects the existing scalar Q5_K/Q6_K reductions,
BF16 input rounding with F64 accumulation, and precise attention operations
through the shared kernels; `RMI_SCALAR` is unset. All seven
cases pass the same fixed gates. Machine-readable results: [validation.json](validation.json).

| Case | KV | Prompt / generated tokens | Checkpoints | Max relative L2 | Max scaled absolute error |
| --- | --- | --- | --- | --- | --- |
| default-hello-f32 | f32 | 1 / 8 | 1755 | 0 | 0 |
| default-math-f32 | f32 | 7 / 8 | 2925 | 0 | 0 |
| default-unicode-f32 | f32 | 13 / 2 | 2925 | 0 | 0 |
| default-hello-f16 | f16 | 1 / 8 | 1755 | 0 | 0 |
| default-math-f16 | f16 | 7 / 8 | 2925 | 0 | 0 |
| window2-f32 | f32 | 1 / 8 | 1755 | 0 | 0 |
| window2-f16 | f16 | 1 / 8 | 1755 | 0 | 0 |

The ARM64 Rust replays were rerun after the projection fix against the existing
pinned scalar Oracle traces. Zero is the observed error for these inputs;
the lossy-path acceptance gates remain unchanged.

Linux x86_64 / AMD EPYC 9334, glibc 2.39, Rust 1.98.1 and `release-fast` also
pass with four threads on both sides. Fresh Oracle traces cover the same seven
default-entry cases plus scalar-entry Hello with F32/F16 KV: nine comparisons,
19,305 checkpoints, and observed maximum relative L2 / scaled absolute error
of zero. Results and source hashes: [validation-linux.json](validation-linux.json).
The real contract/finite-logit tests, F32/F16 prefill chunking/reset, five Dory
unit tests, two reference-arithmetic regressions in both default/scalar modes,
and the production CLI Hello-eight-token replay pass. The arithmetic tests
also pass with the Vulkan wrapper compiled on macOS; this is a CPU check.

The expanded Linux RoPE preflight has two existing failures (1 and 4 ULP),
reproduced independently on main `973210b6de572b8226cd576da7ace17fb22d8ad5`.
Their names and scope are recorded in the Linux result file. No full-suite
pass is claimed.

## Acceptance

The reference build disables NEON on ARM and SSE2/SSE4.2, AVX/AVX2/AVX512, FMA,
F16C and BMI2 on x86. It also disables FMA contraction and compiler fusion
of `sinf`/`cosf`. The initial canonical Rust
replay uses `RMI_SCALAR=1`; rerun the same command with that prefix omitted
to validate the default path, as in the table above. Separate libm calls matter
on macOS: fused `sincos` rounding can cross an activation-quantization
boundary and amplify through recurrent layers.

On x86, disabling the CMake SSE4.2 option leaves `__SSE2__` defined and selects
GGML's vector activation implementation. The build also undefines that macro
to use the scalar implementation. Dory opts in to reference arithmetic inside
the weight kernels: tiny differences in SIMD reduction order can change later
Q8_K activation bins and amplify over the recurrent stack. BF16 projections
round activations to BF16 before their dot product, as in GGML. Other models
retain their existing kernel defaults. The Vulkan wrapper preserves this CPU
contract instead of offloading these projections. This trades x86 SIMD dot
throughput for stability; performance has not been benchmarked.

The checker requires exact prompt/generated token IDs, checkpoint order,
virtual layer/step indices, shapes and counts. Every traced token has 195
checkpoints: `embd`, 192 `l_out`, `result_norm`, and `result_output`. Lossless
embedding decode is compared as raw bytes. Quantized/BF16 projections,
normalization, RoPE, softmax, gates and F16 KV use numerical gates:

- Relative L2 error at most `0.01`.
- Maximum absolute error divided by `max(1, max(abs(reference)))` at most `0.01`.

Both bounds were fixed before investigating differences. Missing, duplicated,
empty, non-finite or truncated checkpoints fail. This is numerical acceptance
of reduced-precision computation, not a claim of end-to-end bitwise equality.

The replay covers this CPU Q5_K_M artifact with `per_loop` KV. GPU execution,
other weight quantizations, `last_loop` KV, the actual 1024-token SWA boundary,
the full 262144-token context and chat-template generation are outside this validation.
