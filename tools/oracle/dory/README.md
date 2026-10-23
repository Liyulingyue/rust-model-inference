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

## Initial GPU probe (2026-10-22)

On 2026-10-22 the same Linux host exposed `AMD Radeon Graphics (RADV NAVI31)`,
PCI `1002:744b`, with 51,522,830,336 bytes of VRAM. The Radeon Vulkan ICD
initialized successfully. The existing `vk_check` passed all five Q8_0
matrix shapes. Q5_K, Q6_K and BF16 batched operator checks also passed with
one and three token rows, including grouped projections, padded strides,
large outputs and long command buffers. These are operator fixtures, not a
Dory GPU accuracy claim.

The real Dory Hello replay with F32/F16 KV and eight generated tokens matched
the CPU logits within the unchanged dual 1% gates (observed errors zero).
However, after Vulkan warmup both replays performed **zero inference GPU
submissions and zero weight uploads**, even with `RUST_GPU_MIN_OUTPUTS=0`.
`--gpu` succeeds through CPU fallback; `--compute vulkan` exits with
`full Vulkan text decoding is not supported for dory`. At that revision Dory remained
CPU-only. That probe does not establish end-to-end GPU alignment or speedup. Results and
source hashes: [validation-gpu.json](validation-gpu.json).

The current probe below enforces resident execution. Use the recorded revision
to reproduce the historical fallback. It requires the same artifact and a hardware Vulkan ICD:

```sh
cargo build --locked --profile release-fast --features vulkan --lib
rustc --edition=2021 -C opt-level=3 -L dependency=target/release-fast/deps \
  --extern rust_model_inference=target/release-fast/librust_model_inference.rlib \
  tools/oracle/dory/gpu_probe.rs -o target/dory-gpu-probe
env -u RMI_PARITY_TRACE -u RMI_SCALAR RUST_GPU_MIN_OUTPUTS=0 \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  target/dory-gpu-probe "$RMI_DORY_Q5_MODEL"
```

The probe asserts that the ICD is hardware and that the operator checks
submit GPU work, then reports Dory counters separately after warmup. It
compares every step's logits and greedy IDs with a CPU-scoped session.
The execution probe keeps `RMI_PARITY_TRACE` and `RMI_SCALAR` unset so it
measures production command recording without intermediate readbacks. The operator helper retains its own
fixture checks; Dory's reduced-precision acceptance remains the dual 1% gate.

The Z-Image SiLU-scale probe now uses `read_f32_into` to copy device-local
output into its existing host `gate` vector. Its previous borrowed `read_f32`
call failed with `device arena requires read_f32_into` on the discrete GPU.
The same remote check fails before the fix and passes afterward:

```sh
SILU_ONLY=1 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  cargo run --locked --profile release-fast --features vulkan \
  --example vk_ops_check -- --formats q5_k,q6_k,bf16 --rows 1
```

All 39,321,600 SiLU outputs are finite. Fresh full `vk_ops_check` runs with
`--rows 1` and `--rows 3` pass SiLU, RMSNorm, RoPE, Q5/Q6/BF16 matvec and Q8_K
quantization, then stop at a separate convolution check: row 0, column 17,
GPU `0xb788d769` versus CPU `0xb788d768` (1 ULP). The same mismatch occurs
with `SILU_SCALE_ONLY=1`, which skips the SiLU probe entirely. No numerical
gate was changed; the full operator suite is still not counted as passing.
The `readback_fix` section of [validation-gpu.json](validation-gpu.json)
records these follow-up checks separately from the original execution probe.

## Resident Dory Vulkan execution

Dory now has a dedicated Vulkan session selected by `--compute vulkan` or
`--gpu` (Auto). Projection weights upload once; normalized loop state, all
per-loop KV slots, RoPE application, attention, gates, MLP, residuals and output
scaling execute on the device. Embedding lookup and precise RoPE coefficient
generation remain on the host. Prefill evaluates tokens sequentially; reset
reuses uploaded weights and arena allocations.

The initial device contract requires `shaderFloat64`. It covers the tested
Q5_K_M artifact's Q5_K/Q6_K/BF16 projections with F32 or F16 KV. Unsupported
formats, insufficient buffer range or missing FP64 reject forced Vulkan;
Auto may select CPU during initialization. An execution failure invalidates
the GPU session until reset and never silently switches its KV to CPU.
Context capacity is also bounded by the device's storage-buffer range.
As in the CPU session, F16 KV is rounded to half precision but stored in F32
slots; it currently saves no cache memory.

The GPU follows reference Q5/Q6 accumulation and BF16 input rounding. It uses
the existing signed Q8_K quantizer, F16 ties-to-even conversion and normalization
helpers. Q5/Q6 keep their per-lane and per-block reduction order; BF16 and
normalization use cooperative FP64 reductions. The FP64
exponential uses Arm's MIT-licensed `expf` reduction/table/evaluation, matching
the Linux Oracle's FMA sequence: a one-ULP activation difference otherwise
crossed quantization boundaries and amplified through the recurrent stack.
This device arithmetic remains experimental outside the recorded hardware.

Machine-readable graph replays, hardware counters, CLI and scope are in
[validation-gpu-resident.json](validation-gpu-resident.json). The initial
CPU-fallback evidence above is retained as history. On the AMD RADV NAVI31
host, seven GPU/Oracle replays passed with 15,795 checkpoints: Hello and math
with both KV formats, Unicode with F32 KV, and window-2 eviction with both
KV formats. Observed relative L2 and scaled absolute errors were zero.

With tracing unset, each nine-position F32/F16 decode performed 486 inference
GPU submissions, zero static uploads during inference, and zero submissions
from its explicit CPU comparison. Initialization uploaded 362 buffers
(2,996,551,680 bytes) per session. The final hardware integration test and
production CLI `--compute vulkan` F32 / `--gpu` F16 / `--compute cpu` F32
all passed. The two GPU CLI runs produced the pinned Hello output.

Before the cooperative-kernel optimization, the alternating nine-position
probe took GPU 26.92s / CPU 7.69s for F32 and GPU 26.67s / CPU 7.53s for F16,
excluding initialization. The host was shared and CLI timing fluctuated;
these are observations, not an isolated benchmark or a speedup claim. Reduced-precision gates
remain relative L2 <= 1% and scaled maximum absolute error <= 1%; token IDs,
checkpoint structure and embedding bytes are exact checks.

The optimized graph assigns eight invocations to each projection row and a
64-invocation workgroup to each normalization vector. Q5's integer minimum
correction is reduced before scaling, so its floating-point block/lane order
is unchanged. BF16 keeps activation rounding and FP64 accumulation; the
normalization reduction also retains FP64, the same epsilon and scale steps.
Tail projection rows participate in barriers without reading weights or
writing outputs. No external acceleration library or new device feature is
required.

Fresh timings and Oracle replays for these kernels are recorded separately in
[validation-gpu-performance.json](validation-gpu-performance.json); the earlier
resident-execution report remains a record of the initial implementation.
On the same EPYC 9334 / RADV NAVI31 device, the per-call medians were:

| KV | Original | Optimized | Observed ratio |
| --- | --- | --- | --- |
| F32 | 3.764s | 0.581s | 6.47x |
| F16 | 4.016s | 0.504s | 7.97x |

Each median uses 27 post-warmup calls. The host/GPU were shared with independent
tasks; an original-build call stalled for 19.68s, and load changed during the
optimized F32 run. All samples are retained, so these observations do not
establish an isolated throughput guarantee. The final seven Oracle replays
passed all 15,795 checkpoints with zero observed numerical error. The untraced
CPU comparison, hardware integration and forced-Vulkan F32 / Auto F16 CLI also
passed without relaxing either 1% gate.

`gpu_bench.rs` measures nine single-token forwards after an excluded warmup
round, then records three rounds with the same session capacity, input IDs and
KV format. Initialization is excluded. Each round checks finite logits, the
eight pinned greedy IDs, actual GPU submissions and zero static uploads.
Link the same source against each revision's library, save each binary before
building the other revision, and alternate their runs on the same device:

```sh
cargo build --locked --profile release-fast --features vulkan,parity-trace --lib
rustc --edition=2021 -C opt-level=3 -L dependency=target/release-fast/deps \
  --extern rust_model_inference=target/release-fast/librust_model_inference.rlib \
  tools/oracle/dory/gpu_bench.rs -o target/dory-gpu-bench
env -u RMI_PARITY_TRACE -u RMI_SCALAR \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  target/dory-gpu-bench "$RMI_DORY_Q5_MODEL" f32
# Repeat with f16. Preserve binaries and hashes before switching revisions.
```

### Shared and cooperative activation quantization

Q/K/V/gate projections now reuse one Q8_K preparation of their unchanged input;
MLP gate/up projections do the same. BF16 consumers still read the original
F32 input. Dory's private quantizer distributes each 256-value block over 64
invocations, with four consecutive values per lane. It retains the first
signed maximum on ties and the existing ties-to-even integer conversion.
The shared quantizer used by other models is unchanged.

[validation-gpu-quantization.json](validation-gpu-quantization.json) compares
this change with commit `eb01be8e`, using fresh ABBA runs on the same shared
EPYC 9334 / RADV NAVI31 host. Each build/KV has 54 measured calls, excluding
initialization and each run's nine warmup calls:

| KV | Before | After | Observed latency reduction |
| --- | --- | --- | --- |
| F32 | 0.4285s | 0.4109s | 4.1% |
| F16 | 0.4113s | 0.3861s | 6.1% |

All samples are retained; these shared-device observations are not an isolated
throughput guarantee and must not be compared with older absolute timings.
A second ABBA comparison isolates the cooperative kernel from input sharing.
The report also records a discarded Q5 reduction experiment that had no speed
benefit. Use the same `gpu_bench.rs` commands above against the two frozen
builds, running before/after/after/before for each KV format.

The hardware quantizer regression compares packed bytes and scale bits with
the original signed Q8_K kernel for 16 fixtures: four widths, all-zero inputs,
positive/negative maximum ties, rounding boundaries and varied values:

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  cargo test --locked --profile release-fast --features vulkan --lib \
  vulkan::dory::tests::cooperative_quantization_matches_signed_q8k \
  -- --ignored --exact --nocapture
```

Seven fresh full-model Oracle replays pass all 15,795 checkpoints with zero
observed error and unchanged reduced-precision gates. The untraced CPU/GPU
comparison, hardware prefill/reset test and forced-Vulkan F32 / Auto F16 CLI
also pass. Inference still performs 486 GPU submissions per nine positions
and zero static weight uploads.

To replay the complete graph against the fixed CPU Oracle, use the CPU replay
commands above with a fresh trace path and replace the Rust command with:

```sh
RMI_DORY_COMPUTE=vulkan RMI_PARITY_TRACE="$case_dir/vulkan.jsonl" \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  cargo test --profile release-fast --features vulkan,parity-trace \
  --test dory_q5_k_m dory_q5_reference_trace -- --ignored --exact --nocapture
.venv/bin/python tools/oracle/dory/check.py \
  "$case_dir/oracle.jsonl" "$case_dir/vulkan.jsonl" --report "$case_dir/gpu-report.json"
```

The explicit Vulkan graph continues to submit work while tracing; CPU kernel
offload switches do not select this graph. Run `gpu_probe.rs` above with tracing
unset to assert actual inference submissions, one-time static uploads, no
uploads during decode, CPU isolation and nine per-step logits comparisons.
It also reports GPU/CPU evaluation time separately, excluding initialization.
The hardware integration test additionally checks chunked prefill and reset:

```sh
RMI_DORY_Q5_MODEL="$RMI_DORY_Q5_MODEL" \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  cargo test --profile release-fast --features vulkan --test dory_q5_k_m \
  dory_q5_vulkan_executes_and_matches_cpu -- --ignored --exact --nocapture
```

Full 1024-token SWA eviction, the 262144-token context, other GGUF formats,
other GPU/driver combinations and full-repository tests remain unverified.
The shared operator suite's unrelated convolution/SiLU one-ULP failure above
has not been relaxed or counted as passing.

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

The CPU and resident Vulkan replays cover this Q5_K_M artifact with `per_loop`
KV, within the device and arithmetic scope recorded above;
other weight quantizations, `last_loop` KV, the actual 1024-token SWA boundary,
the full 262144-token context and chat-template generation are outside this validation.
