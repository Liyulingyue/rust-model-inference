# Task 6A report

Implemented host row-aware dispatch/recording, row-stride push constants, shader row decoding, regenerated SPIR-V/manifest, and accepted `--all-formats --rows N` in `vk_ops_check`.

Validation:

- `cargo test --lib --features vulkan batched_matmul`: passed (2 tests).
- `bash scripts/vulkan-shaders.sh update`: passed.
- `bash scripts/vulkan-shaders.sh check`: passed.
- `cargo test --example vk_ops_check --features vulkan --no-run`: passed.
- Real `cargo run --features vulkan --example vk_ops_check -- --all-formats --rows 3`: reached Apple M3 Max Vulkan device, then failed quantize tie-even check (`gpu=-21`, `cpu=-20` at index 0). This is unresolved and blocks claiming device parity.

Known limitation: the example accepts `--rows`, but its existing operator fixture remains single-row; the full device batch comparison is deferred with the runtime-owner phase.

## Fix round 1 (2026-09-13)

All four review findings are addressed. The previous note deferring the device row comparison is superseded: the comparison now directly exercises `Qwen3Ops::record_weight_matmul_rows`; no runtime owner or Task 6B code was added.

### Changes

- F32, F16 and BF16 now read `input_word + token_row * input_stride + index`, retaining the original accumulation order.
- Quantization and matmul use production push-constant builders that validate the complete last-row span before any command is recorded: `(rows - 1) * stride + used_row_width`. The checks cover raw input, Q8/Q8K, scales, Q4_1 sums, every grouped output, arena boundaries, alignment, and usize/u32 address overflow. Final-row padding is not required.
- Quantized scratch rows use the packed `n_in` byte stride independently of the raw input stride. The public recorder keeps the preceding phase's units: input stride is measured in f32 elements; each output stride is measured in bytes. These units are documented at the recorder.
- Both quantization and matmul dispatches pass through the production device-z check. Matmul uses `token_rows * grouped_weights`, with checked multiplication, against `max_compute_work_group_count[2]`.
- `vk_ops_check --rows N` now drives a real device comparison, including Q8_0 in `--all-formats`. Each format compares one grouped rows=N recording against concatenated rows=1 recordings with separate weight bindings and packed strides. It compares raw `to_bits()`, uses three distinct weights with output widths 65/33/17, input width 512 at stride 515, distinct padded output strides, sentinel padding, and no final-row tail padding. The row comparison runs before the existing CPU parity suite; the latter still executes and propagates failure.

### RED / GREEN evidence

- Added host production-validation and dispatch-limit tests before the new helpers: `cargo test --lib --features vulkan batched_matmul` failed with missing `quantize_rows_push`, `row_dispatch`, and `matmul_dispatch` (log: `/tmp/vulkan-rows-red.log`).
- Added the ignored device comparison before fixing the float shaders. Running it on Apple M3 Max failed at F32, token row 1 / group 0 / index 68: `batched=0xc1594cc9 single=0x41613aff` (log: `/tmp/vulkan-rows-device-red.log`).
- After adding the float token-row offset and regenerating SPIR-V, the same device test passed all nine formats with raw-bit equality (log: `/tmp/vulkan-rows-device-green.log`).
- Added the example parser test before connecting the row option; it failed on missing `parse_options` and then passed after implementation.

### Clean-base comparison

A detached disposable worktree at `/tmp/vulkan-base-JNIVPX` was created from exactly `1f8cc9ea9fd424390f25b62028b818246c38ce5f`. Its HEAD and clean status were verified after the run. No user branch was changed.

The base CLI predates `--all-formats` and `--rows`, so its equivalent existing operator-suite command was:

```sh
cargo run --release --features vulkan --example vk_ops_check \
  --target-dir /Users/gouzi/.codex/worktrees/5328/rust-model-inference/target \
  -- --formats q4_0,q4_1,q4_k,q5_k,q6_k,f16,bf16,f32
```

The base release build completed and exited 1 with:

```text
[GPU] Vulkan device: Apple M3 Max
[GPU] Warming up Vulkan pipeline (driver JIT)...
[GPU] Warmup done in 0.0s
Vulkan operator check failed: quantize tie-even mismatch at 0: gpu=-21 cpu=-20
```

Exact base log: `/tmp/vulkan-base-JNIVPX-release.log`. This establishes that the tie-even failure predates the row changes. Its underlying CPU/GPU numerical discrepancy was not altered in this phase. The clean disposable worktree was removed after validation; the log remains.

### Final validation

- `cargo test --lib --features vulkan batched_matmul`: **6 passed, 1 ignored**. Host tests use the actual production preparation/validation helpers, covering all nine formats at rows 1/2/3, grouped outputs, nonpacked strides, missing tail padding, per-region undersizing, empty rows, overflow, and device z limits.
- `cargo test --lib --features vulkan batched_matmul_device_rows_match_single_row_bits -- --ignored --nocapture`: **1 passed**, all nine formats on Apple M3 Max.
- `cargo test --example vk_ops_check --features vulkan`: **1 passed**.
- `cargo test --example vk_ops_check --features vulkan --no-run`: passed.
- `bash scripts/vulkan-shaders.sh update` and `bash scripts/vulkan-shaders.sh check`: passed; checked-in SPIR-V validates, rebuilds byte-identically, and keeps the existing workgroup limit. The script lacks execute permission in this checkout, so it was invoked through Bash.
- `rustfmt --edition 2021 --config skip_children=true --check src/vulkan/ops.rs src/vulkan.rs examples/vk_ops_check.rs`: passed.
- `git diff --check`: passed.
- Required `cargo run --release --features vulkan --example vk_ops_check -- --all-formats --rows 3`: the release build completed; **all nine new row comparisons passed** with `groups=3 input_stride=515 padded_outputs=true exact_bits=true`. The command then exited 1 at the same clean-base tie-even discrepancy. Log: `/tmp/vulkan-rows-release.log`.

### Self-review / limits

Every fallible size/stride/dispatch check completes before the first quantization or matmul command is emitted. One-row wrappers reuse the same preparation and command recording. No accumulation loops, CPU references, tolerances, or existing checks were weakened. Existing repository compiler warnings remain. The full legacy operator suite is still blocked by the proven base tie-even failure; this report does not claim full CPU/GPU parity or model inference validation.
