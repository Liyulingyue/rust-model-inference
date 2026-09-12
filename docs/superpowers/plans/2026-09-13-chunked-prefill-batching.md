# Chunked Prefill Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement bit-exact single-request chunked prefill for Qwen3, Qwen3.5, and Gemma4 on CPU and each model's approved Vulkan scope, with a default chunk size of 64 and unchanged single-token decode.

**Architecture:** Add one crate-private prepared-row linear primitive, then keep causal attention, KV ownership, Gemma4 shared-KV, and Qwen3.5 recurrent scans inside their model implementations. Each prompt chunk computes into tentative state, commits only after the entire chunk succeeds, and falls back from Vulkan by recomputing the whole failed chunk on CPU.

**Tech Stack:** Rust 2021, existing `ComputePool`/SIMD/quantized kernels, Ash Vulkan, GLSL/SPIR-V, existing tokenizer/parity-trace infrastructure; no new dependency.

**Spec:** `docs/superpowers/specs/2026-09-13-chunked-prefill-batching-design.md`

## Global Constraints

- Scope is single-request prompt prefill only; do not add continuous batching, request scheduling, paged KV, or dynamic batching.
- Support Qwen3, Qwen3.5, and Gemma4 CPU paths; support Qwen3/Qwen3.5 full Vulkan only within existing eligibility, and Gemma4 through generic batched Vulkan linear operations only.
- `--prefill-batch-size N` requires `N >= 1`, defaults to `64`, and `N=1` is the sequential baseline/fallback.
- Do not add BLAS, Accelerate, MKL, oneDNN, a graph runtime, or another dependency.
- Do not expand existing Vulkan model eligibility and do not add tiled SIMD GEMM in this change.
- Preserve the original dot-product accumulation order; same-backend batch=1 and batch=N must match raw F32/F16 state bits, final logits, and greedy tokens.
- Validate full prompt capacity before prefill; an oversized prompt must commit zero prompt rows.
- Scratch allocations are bounded by `min(prefill_batch_size, capacity)`, never by the full prompt length.
- One Qwen3/Qwen3.5 Vulkan chunk uses one command-buffer submission. Gemma4's acceptance boundary is one dispatch per projection per chunk, not one submission for the full model.
- GPU failure aborts the pending chunk and recomputes that whole chunk from the same base position on CPU; never mix GPU and CPU rows inside one chunk.
- Keep `.codex/` and all unrelated dirty/untracked files untouched. Stage explicit files only.

## File Structure

- `src/core/prefill.rs`: default batch size, value validation, and allocation-free chunk range iterator.
- `src/ops/kernel/mod.rs`: crate-private prepared-row activation buffers and batched weight execution; the public `Kernel` trait remains unchanged.
- `src/models/qwen3/trunk/prefill.rs`: Qwen3 CPU chunk execution and transactional KV commit.
- `src/models/qwen3/trunk/session.rs`: prompt/decode orchestration, scratch ownership, Vulkan whole-chunk fallback.
- `src/models/qwen35/trunk/forward.rs`: batched dense/recurrent projections with explicit base positions and caller-owned tentative recurrent state.
- `src/models/qwen35/trunk/session.rs`: chunk loop, bounded scratch, atomic state commit, and Vulkan fallback.
- `src/models/gemma4/trunk/forward.rs`: Gemma4 chunk forward while preserving shared-KV/SWA/full-attention rules.
- `src/models/gemma4/trunk/session.rs`: batch-size ownership and chunk-level KV rollback.
- `src/vulkan/ops.rs`: row-aware generic linear/normalization/attention recording plus one persistent batched-linear adapter that reuses uploaded weights.
- `src/vulkan/qwen3.rs`, `src/vulkan/qwen35.rs`: model-specific chunk command recording and pending-chunk commit state.
- `shaders/glsl/*.comp`, `shaders/bin/*.spv`, `shaders/manifest.sha256`: row-aware Vulkan kernels and reproducible checked-in binaries.
- Existing model test modules: tiny-model raw-bit, chunk-boundary, capacity, and rollback checks.
- `examples/prefill_bench.rs`: one repeatable real-model pp/tg benchmark and same-backend batch comparison harness.
- Existing reference tests and `examples/vk_model_check.rs`: Oracle and Vulkan batch=1 versus batch=N verification.

---

### Task 1: Add the shared batch-size contract

**Files:**
- Create: `src/core/prefill.rs`
- Modify: `src/core/mod.rs`
- Modify: `src/app/cli.rs:19-69,193-523,775-end`
- Modify: `src/main.rs` help/usage text containing `--threads` and `--kv-cache`

**Interfaces:**
- Produces: `pub const DEFAULT_PREFILL_BATCH_SIZE: usize = 64`
- Produces: `pub fn checked_prefill_batch_size(value: Option<usize>) -> Result<usize, String>`
- Produces: `pub(crate) fn prefill_chunks(len: usize, batch_size: usize) -> impl Iterator<Item = Range<usize>>`
- Produces: `CliOptions::prefill_batch_size: Option<usize>` and `CliOptions::effective_prefill_batch_size() -> Result<usize, String>`

- [ ] **Step 1: Write failing contract and CLI tests**

Add these tests in `src/core/prefill.rs` and `src/app/cli.rs`:

```rust
#[test]
fn prefill_batch_size_defaults_to_64_and_rejects_zero() {
    assert_eq!(checked_prefill_batch_size(None).unwrap(), 64);
    assert_eq!(checked_prefill_batch_size(Some(1)).unwrap(), 1);
    assert_eq!(checked_prefill_batch_size(Some(128)).unwrap(), 128);
    assert!(checked_prefill_batch_size(Some(0)).unwrap_err().contains("at least 1"));
}

#[test]
fn prefill_chunks_cover_input_without_padding() {
    let ranges = prefill_chunks(130, 64).collect::<Vec<_>>();
    assert_eq!(ranges, [0..64, 64..128, 128..130]);
}

#[test]
fn cli_parses_prefill_batch_size_strictly() {
    let parsed = parse_cli_options(&args(&["rmi", "--prefill-batch-size", "32"])).unwrap();
    assert_eq!(parsed.effective_prefill_batch_size().unwrap(), 32);
    assert!(parse_cli_options(&args(&["rmi", "--prefill-batch-size", "x"])).is_err());
    let zero = parse_cli_options(&args(&["rmi", "--prefill-batch-size", "0"])).unwrap();
    assert!(zero.effective_prefill_batch_size().is_err());
}
```

- [ ] **Step 2: Run the focused tests and confirm the symbols are absent**

Run:

```bash
cargo test --lib prefill_batch_size_defaults_to_64_and_rejects_zero
cargo test --lib cli_parses_prefill_batch_size_strictly
```

Expected: compilation fails because the new module, field, and methods do not exist.

- [ ] **Step 3: Implement the minimal shared contract**

Use this implementation shape in `src/core/prefill.rs`:

```rust
use std::ops::Range;

pub const DEFAULT_PREFILL_BATCH_SIZE: usize = 64;

pub fn checked_prefill_batch_size(value: Option<usize>) -> Result<usize, String> {
    match value.unwrap_or(DEFAULT_PREFILL_BATCH_SIZE) {
        0 => Err("prefill batch size must be at least 1".into()),
        value => Ok(value),
    }
}

pub(crate) fn prefill_chunks(
    len: usize,
    batch_size: usize,
) -> impl Iterator<Item = Range<usize>> {
    (0..len).step_by(batch_size).map(move |start| start..(start + batch_size).min(len))
}
```

Parse `--prefill-batch-size` with `parse::<usize>()` and an explicit missing-value error; do not use `unwrap_or`. Keep `CliOptions` derived `Default` by storing `Option<usize>`, and make `effective_prefill_batch_size()` call the shared validator.

- [ ] **Step 4: Run focused tests and formatting**

Run:

```bash
cargo test --lib prefill_batch_size
cargo test --lib cli_parses_prefill_batch_size_strictly
cargo fmt --all -- --check
```

Expected: all selected tests pass and formatting is clean.

- [ ] **Step 5: Commit the contract**

```bash
git add src/core/prefill.rs src/core/mod.rs src/app/cli.rs src/main.rs
git commit -m "feat: add prefill batch size contract"
```

---

### Task 2: Add the crate-private prepared-row linear primitive

**Files:**
- Modify: `src/ops/kernel/mod.rs:28-end`
- Modify: `src/ops/matmul_tests.rs:1-40` and append focused tests

**Interfaces:**
- Consumes: existing `Weight`, `Kernel::forward_prepared`, `ComputePool`, `quantize_q8_0_into`, and `quantize_row_q8_k_into`
- Produces: `pub(crate) struct PreparedRows`
- Produces: `PreparedRows::new(max_rows: usize, max_n_in: usize) -> Self`
- Produces: `PreparedRows::prepare(&mut self, input: &[f32], rows: usize, n_in: usize, need_q8: bool, need_q8k: bool) -> Result<(), String>`
- Produces: `PreparedRows::matmul(&self, weight: &Weight<'_>, input: &[f32], output: &mut [f32], pool: &ComputePool) -> Result<(), String>`
- Produces: `Weight::needs_q8_0_activation() -> bool`, complementing `uses_q8_k()` for formats that need neither prepared quantization buffer
- Produces under `#[cfg(test)]`: `PreparedRows::q8_capacity_for_test() -> usize` and `PreparedRows::q8_ptr_for_test() -> *const u8`

- [ ] **Step 1: Write a raw-bit equivalence test for all existing hot formats**

Add one table-driven test that constructs deterministic F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, and Q6_K weights already supported by the test helpers. This covers direct-F32 input, Q8_0 activation preparation, and Q8_K activation preparation without duplicating every IQ kernel test. For each format and `rows` in `[1, 2, 3, 17]`, compare `PreparedRows::matmul` with one `Weight::quantize_and_matmul_with_scratch` call per row:

```rust
#[test]
fn prepared_rows_match_sequential_matmul_bits_and_reuse_storage() {
    for weight in prepared_row_test_weights() {
        for rows in [1, 2, 3, 17] {
            let input = deterministic_rows(rows, weight.n_in);
            let expected = sequential_weight_rows(&weight, &input, rows);
            let mut prepared = PreparedRows::new(rows, weight.n_in);
            prepared
                .prepare(&input, rows, weight.n_in, weight.needs_q8_0_activation(), weight.uses_q8_k())
                .unwrap();
            let q8_ptr = prepared.q8_ptr_for_test();
            let q8_capacity = prepared.q8_capacity_for_test();
            let mut actual = vec![0.0; rows * weight.n_out];
            prepared
                .matmul(&weight, &input, &mut actual, &ComputePool::new(3))
                .unwrap();
            assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected);
            prepared
                .prepare(&input, rows, weight.n_in, weight.needs_q8_0_activation(), weight.uses_q8_k())
                .unwrap();
            assert_eq!(prepared.q8_ptr_for_test(), q8_ptr);
            if weight.needs_q8_0_activation() {
                assert!(q8_capacity >= rows * weight.n_in);
            } else {
                assert_eq!(q8_capacity, 0);
            }
        }
    }
}
```

Keep both inspection methods behind `#[cfg(test)]`. Define `prepared_row_test_weights`, `deterministic_rows`, and `sequential_weight_rows` locally in the same test file; use deterministic nonzero values and the repository's existing quantized test constructors, not random data.
Also assert `needs_q8_0_activation()` is true only for Q8_0/Q4_0/Q4_1 in this matrix, while every K-quant case uses Q8_K and direct floating formats allocate neither prepared quantization buffer.

- [ ] **Step 2: Run the test and verify it fails to compile**

```bash
cargo test --lib prepared_rows_match_sequential_matmul_bits_and_reuse_storage
```

Expected: compilation fails because `PreparedRows` is absent.

- [ ] **Step 3: Implement preparation with bounded reusable buffers**

`PreparedRows` records maximum rows/width and owns reusable `q8`, `scales`, and `q8k` buffers. Grow only the buffers requested by `need_q8`/`need_q8k`, retain them across calls, and never allocate Q8 storage for F32/F16/BF16. Reject zero rows, zero width, shape mismatches, and widths that are not Q8_K aligned when `need_q8k` is true.

Preparation loops token rows once:

```rust
for row in 0..rows {
    let input = &input[row * n_in..(row + 1) * n_in];
    if need_q8 {
        quantize_q8_0_into(input, n_in, &mut self.q8[row * n_in..], &mut self.scales[row * blocks..]);
    }
    if need_q8k {
        quantize_row_q8_k_into(input, &mut self.q8k[row * q8k_blocks..(row + 1) * q8k_blocks]);
    }
}
```

`matmul` calls `pool.compute` exactly once. Inside each worker, loop over token rows and call the existing `forward_prepared` with that worker's unchanged `(ith, nth)`. This preserves each output dot's accumulation order while amortizing the pool barrier:

```rust
pool.compute(|ith, nth| {
    for row in 0..self.rows {
        weight.kernel.forward_prepared(
            &input[row * n_in..(row + 1) * n_in],
            self.q8_row(row),
            self.scales_row(row),
            self.q8k_row_for(weight, row),
            &mut output[row * n_out..(row + 1) * n_out],
            n_in,
            n_out,
            ith,
            nth,
        );
    }
});
```

Do not modify `Kernel` or its public `forward_batched` method in this task.

- [ ] **Step 4: Run the format matrix and existing matmul suite**

```bash
cargo test --lib prepared_rows_match_sequential_matmul_bits_and_reuse_storage
cargo test --lib ops::matmul::neon_tests
cargo test --test fused_ffn_matmul_parity
```

Expected: raw-bit equivalence passes for every tested format and existing matmul tests remain green.

- [ ] **Step 5: Commit the primitive**

```bash
git add src/ops/kernel/mod.rs src/ops/matmul_tests.rs
git commit -m "perf: add prepared row matmul primitive"
```

---

### Task 3: Implement transactional Qwen3 CPU chunk prefill

**Files:**
- Create: `src/models/qwen3/trunk/prefill.rs`
- Modify: `src/models/qwen3/trunk/mod.rs:1-25`
- Modify: `src/models/qwen3/trunk/forward.rs:120-145` and all `Qwen3GenerateOptions` literals in this file
- Modify: `src/models/qwen3/trunk/session.rs:87-244,327-end`
- Modify: `src/models/qwen3/trunk/tests.rs`
- Modify: all current `Qwen3GenerateOptions` literals reported by `rg -n 'Qwen3GenerateOptions \{' src tests examples`

**Interfaces:**
- Consumes: `PreparedRows`, `prefill_chunks`, and `DEFAULT_PREFILL_BATCH_SIZE`
- Produces: `Qwen3GenerateOptions::prefill_batch_size: usize`
- Produces: `pub(super) Qwen3Session::prefill_cpu(input: &Qwen3Input<'_>, batch_size: usize) -> Result<Duration, String>`
- Produces: crate-private `Qwen3Session::forward_cpu_chunk(input: &Qwen3Input<'_>, range: Range<usize>, project_logits: bool) -> Result<(), String>`
- Produces: `Qwen3Session::scratch_bytes() -> usize` for benchmark/reporting only

- [ ] **Step 1: Add tiny-model chunk-boundary and capacity tests**

Extend the Qwen3 fixture with one deterministic dense layer and a helper `fn run_qwen3_fixture(prompt_len: usize, batch_size: usize) -> (Vec<u32>, KvSnapshot)`. `KvSnapshot` stores `seq_len` plus raw cache bytes. Exercise `[1, 2, 3, 63, 64, 65, 127, 128]`:

```rust
#[test]
fn qwen3_cpu_prefill_matches_batch_one_at_chunk_boundaries() {
    for len in [1, 2, 3, 63, 64, 65, 127, 128] {
        let baseline = run_qwen3_fixture(len, 1);
        for batch_size in [16, 32, 64, 128] {
            let actual = run_qwen3_fixture(len, batch_size);
            assert_eq!(actual, baseline, "len={len} batch={batch_size}");
        }
    }
}

#[test]
fn qwen3_oversized_prompt_commits_nothing() {
    let mut fixture = qwen3_fixture_session(4, 64);
    let before = snapshot_qwen3_kv(fixture.kv_state());
    assert!(prefill_qwen3_tokens(&mut fixture, &[1, 2, 3, 4, 5], 64).is_err());
    assert_eq!(snapshot_qwen3_kv(fixture.kv_state()), before);
}
```

The fixture must also compare three greedy decode tokens after prefill, not only prompt logits. Define `prefill_qwen3_tokens(session, ids, batch_size)` locally in `tests.rs`; it builds contiguous `[position, 0, 0, 0]` values from the current `seq_len` and calls `prefill_cpu` with that batch size.
In the same test module, define `qwen3_fixture_session(capacity, batch_size)` from the existing deterministic test model and `snapshot_qwen3_kv(&KvState) -> KvSnapshot`; the latter copies only logically visible cache rows plus `seq_len`, so tentative bytes beyond `seq_len` cannot make a failed transaction look committed.

- [ ] **Step 2: Add and run a failing chunk rollback test**

Add `#[cfg(test)] fail_cpu_prefill_after_layer: Option<usize>` to the session plus a setter. Inject the error after tentative KV writes and before chunk commit:

```rust
#[test]
fn qwen3_failed_cpu_chunk_keeps_visible_kv_at_base_position() {
    let mut session = qwen3_fixture_session(8, 4);
    prefill_qwen3_tokens(&mut session, &[1, 2], 4).unwrap();
    let before = snapshot_qwen3_kv(session.kv_state());
    session.fail_cpu_prefill_after_layer_for_test(0);
    assert!(prefill_qwen3_tokens(&mut session, &[3, 4, 5], 4).is_err());
    assert_eq!(snapshot_qwen3_kv(session.kv_state()), before);
}
```

Run:

```bash
cargo test --lib qwen3_cpu_prefill_matches_batch_one_at_chunk_boundaries
cargo test --lib qwen3_failed_cpu_chunk_keeps_visible_kv_at_base_position
```

Expected: the new tests fail before the chunk path exists.

- [ ] **Step 3: Extract the existing one-row CPU body and generalize it to rows**

Move CPU-only prompt work from `generate_inner` into `prefill.rs`. Allocate row-major `x`, normed, Q/K/V, attention, projection, gate/up/down, Q8/Q8_K buffers once per session with maximum rows `min(batch_size, capacity)`. Keep the existing score reduction order inside each query row.

For each layer:

1. Normalize every row.
2. Prepare the normalized rows once and run Q/K/V projections.
3. Apply per-row QK norm and existing position/deepstack logic using the original absolute position array.
4. Write K/V into physical slots `[base_position, base_position + rows)` without changing `seq_len`.
5. Compute causal attention where row `r` sees committed prefix plus tentative rows `0..=r`.
6. Batch output, gate/up, and down projections with a second prepared input where needed.
7. Project vocab logits only when `project_logits` is true, and only from the final row.

When `batch_size == 1`, call the same extracted one-row operation sequence and preserve the existing trace record order and shape `[1, width]`.

- [ ] **Step 4: Make chunk commit atomic**

`prefill_cpu` validates the entire prompt against capacity before the first chunk. For each range, capture `base_seq_len`; on success set `kv_state.seq_len = base_seq_len + range.len()` once and update access time. On error restore `seq_len`; tentative fixed-cache bytes remain inaccessible and are overwritten on retry.

Change `generate_inner` to:

```rust
let prompt_started = Instant::now();
self.prefill_cpu(&input, options.prefill_batch_size)?;
let prompt_duration = prompt_started.elapsed();
```

Insert those three lines immediately before the existing decode loop. Keep that loop's token/position construction, sampling, rendering, timing, and callback code unchanged.

- [ ] **Step 5: Run Qwen3 focused and parity-trace tests**

```bash
cargo test --lib qwen3_cpu_prefill
cargo test --lib qwen3_failed_cpu_chunk
cargo test --lib models::qwen3
cargo test --test inference_parity --features parity-trace --no-run
cargo fmt --all -- --check
```

Expected: the boundary/rollback tests and the existing Qwen3 suite pass; the reference test compiles with the added option field.

- [ ] **Step 6: Commit Qwen3 CPU prefill**

```bash
git add src/models/qwen3/trunk/prefill.rs src/models/qwen3/trunk/mod.rs
git add src/models/qwen3/trunk/forward.rs src/models/qwen3/trunk/session.rs src/models/qwen3/trunk/tests.rs
git add src/models/qwen3/asr/model.rs src/models/qwen3/text.rs src/app/text.rs src/bin/server.rs
git add examples/vk_model_check.rs
git commit -m "perf: batch qwen3 cpu prefill"
```

---

### Task 4: Implement bounded, transactional Qwen3.5 CPU chunk prefill

**Files:**
- Modify: `src/models/qwen35/trunk/scratch.rs:1-120`
- Modify: `src/models/qwen35/trunk/forward.rs:28-320` and dense/recurrent/FFN helpers below it
- Modify: `src/models/qwen35/trunk/session.rs:74-118,182-end`
- Modify: `src/models/qwen35/trunk/tests.rs`

**Interfaces:**
- Consumes: `PreparedRows`, `prefill_chunks`, `DEFAULT_PREFILL_BATCH_SIZE`
- Produces: `Qwen35Session::new_with_prefill_batch_size(model, capacity, prefill_batch_size, pool) -> Result<Self, String>`
- Preserves: `Qwen35Session::new(model, capacity, pool)` as a wrapper using batch size 64
- Produces: `Qwen35Session::prefill_batch_size: usize`
- Produces: `Qwen35Session::scratch_bytes() -> usize` for benchmark/reporting only
- Produces: `pub(crate) Qwen35Model::forward_chunk(n_tokens, base_position, kv_cache, scratch, conv_states: &mut [Vec<f32>], ssm_states: &mut [Vec<f32>], pool, positions) -> Result<Vec<f32>, String>`
- Preserves: the existing public `Qwen35Model::forward(n_tokens, kv_cache, scratch, pool, positions)` signature as a compatibility wrapper; new session code calls `forward_chunk` with its explicit logical base position
- Produces under `#[cfg(test)]`: `Qwen35Session::fail_cpu_chunk_after_row_for_test(row: usize)`

- [ ] **Step 1: Write bounded-scratch and dense batch equivalence tests**

Use the existing `tiny_dense_session_model` fixture:

```rust
#[test]
fn qwen35_scratch_rows_are_bounded_by_prefill_batch_size() {
    let mut model = tiny_dense_session_model();
    let n_embd = model.config.n_embd;
    let session = Qwen35Session::new_with_prefill_batch_size(
        &mut model,
        16,
        3,
        session_pool(),
    ).unwrap();
    assert_eq!(session.scratch().x.len(), 3 * n_embd);
}

#[test]
fn qwen35_dense_prefill_matches_batch_one_bits() {
    for len in [1, 2, 3, 15, 16] {
        let expected = run_qwen35_dense_fixture(len, 1);
        for batch in [2, 3, 16] {
            assert_eq!(run_qwen35_dense_fixture(len, batch), expected);
        }
    }
}
```

Define `run_qwen35_dense_fixture(prompt_len, batch_size) -> Qwen35Snapshot` in this module. It returns raw logits/KV words, empty recurrent-state slots, `processed_tokens`, `next_position`, and three greedy decode IDs so it shares the `Qwen35Snapshot` comparison type used below.

- [ ] **Step 2: Add a tiny recurrent fixture and rollback test**

Build `tiny_recurrent_session_model()` with `is_recurrent = vec![true]`, deterministic F32 `wqkv`, `wqkv_gate`, conv, SSM, and FFN weights. Test batch=1 versus 2/3 and inject an error after the second row scan:

```rust
#[test]
fn qwen35_recurrent_prefill_and_state_match_batch_one_bits() {
    let expected = run_qwen35_recurrent_fixture(5, 1);
    assert_eq!(run_qwen35_recurrent_fixture(5, 2), expected);
    assert_eq!(run_qwen35_recurrent_fixture(5, 3), expected);
}

#[test]
fn qwen35_failed_chunk_keeps_dense_and_recurrent_state() {
    let mut session = recurrent_fixture_session(8, 4);
    session.step_with_tokens(&[1], &[[0; 4]]).unwrap();
    let before = snapshot_qwen35_state(&session);
    session.fail_cpu_chunk_after_row_for_test(1);
    assert!(session.step_with_tokens(&[2, 3, 4], &[[1; 4], [2; 4], [3; 4]]).is_err());
    assert_eq!(snapshot_qwen35_state(&session), before);
}
```

Define `run_qwen35_recurrent_fixture(prompt_len, batch_size) -> Qwen35Snapshot`, `recurrent_fixture_session(capacity, batch_size)`, and `snapshot_qwen35_state(&Qwen35Session) -> Qwen35Snapshot` in this test module. `Qwen35Snapshot` contains raw logits/KV/conv/SSM words, `processed_tokens`, `next_position`, and three greedy decode IDs; the snapshot reads only logically committed KV rows.

- [ ] **Step 3: Run the new tests and observe the current capacity-sized scratch/in-place state failure**

```bash
cargo test --lib qwen35_scratch_rows_are_bounded_by_prefill_batch_size
cargo test --lib qwen35_dense_prefill_matches_batch_one_bits
cargo test --lib qwen35_recurrent_prefill_and_state_match_batch_one_bits
cargo test --lib qwen35_failed_chunk_keeps_dense_and_recurrent_state
```

Expected: at least the constructor and recurrent transactional tests fail before implementation.

- [ ] **Step 4: Separate bounded row scratch from persistent recurrent state**

Size row-major buffers with `min(prefill_batch_size, capacity)`. Keep persistent conv/SSM state separately and create one working copy per chunk:

```rust
let mut working_conv_states = self.scratch.conv_states.clone();
let mut working_ssm_states = self.scratch.ssm_states.clone();
let logits = self.model.forward_chunk(
    rows,
    base_position,
    &mut self.kv_cache,
    &mut self.scratch,
    &mut working_conv_states,
    &mut working_ssm_states,
    &self.pool,
    positions,
)?;
self.scratch.conv_states = working_conv_states;
self.scratch.ssm_states = working_ssm_states;
```

The two copies are the minimum correct transaction mechanism. Store `prefill_batch_size` on the session and use it when `reset()` rebuilds row scratch; do not add a recurrent-state wrapper or generic state transaction framework.

- [ ] **Step 5: Batch projections while preserving recurrent scan order**

Replace the per-token Q/K/V and gate/up pool calls with `PreparedRows::prepare` plus one `matmul` per projection. Pass `base_position` into dense KV store/attention instead of deriving it by scanning for zero cache rows. Dense tentative K/V written beyond `processed_tokens` stays invisible until commit.

For recurrent layers, batch input projections, then retain the existing scalar conv and SSM bodies inside one outer `for row in 0..n_tokens` loop. Each iteration reads and updates only `working_conv_states[layer]` and `working_ssm_states[layer]` before the next row begins. Batch only the output projection after the ordered scan. Compute vocab logits from the chunk's last row only.

- [ ] **Step 6: Chunk arbitrary `step` inputs and commit once per chunk**

`Qwen35Session::step` validates the full call against capacity, then uses `prefill_chunks`. For each chunk, update `processed_tokens` and `next_position` only after model forward and recurrent working-state swap succeed. A one-token call remains the decode path.

- [ ] **Step 7: Run focused and full Qwen3.5 tests**

```bash
cargo test --lib qwen35_scratch_rows_are_bounded
cargo test --lib qwen35_dense_prefill
cargo test --lib qwen35_recurrent_prefill
cargo test --lib qwen35_failed_chunk
cargo test --lib models::qwen35
cargo test --test qwen35_reference --features parity-trace --no-run
cargo fmt --all -- --check
```

Expected: dense/recurrent state and greedy outputs are raw-bit identical across batch sizes; current validation and GPU-fallback tests still compile.

- [ ] **Step 8: Commit Qwen3.5 CPU prefill**

```bash
git add src/models/qwen35/trunk/scratch.rs src/models/qwen35/trunk/forward.rs
git add src/models/qwen35/trunk/session.rs src/models/qwen35/trunk/tests.rs
git commit -m "perf: batch qwen3.5 cpu prefill"
```

---

### Task 5: Implement transactional Gemma4 CPU chunk prefill

**Files:**
- Modify: `src/models/gemma4/trunk/scratch.rs`
- Modify: `src/models/gemma4/trunk/session.rs:6-92`
- Modify: `src/models/gemma4/trunk/forward.rs:27-546`
- Modify: `src/models/gemma4/trunk/mod.rs`
- Modify: `src/models/gemma4/trunk/tests.rs`
- Modify: `src/models/gemma4/app.rs:12-110` and current `Gemma4Request` literals
- Modify: `tests/gemma4_reference.rs` current `Gemma4Request` literals

**Interfaces:**
- Consumes: `PreparedRows`, `prefill_chunks`, `DEFAULT_PREFILL_BATCH_SIZE`
- Produces: `Gemma4Session::new_with_prefill_batch_size(model, kv_format, prefill_batch_size) -> Result<Self, String>`
- Preserves: `Gemma4Session::new(model, kv_format)` as a default-64 wrapper
- Produces: `Gemma4Session::forward_chunk(rows: &[AssembledInputRow]) -> Result<(), String>`
- Produces: `Gemma4Session::scratch_bytes() -> usize` for benchmark/reporting only
- Produces: `Gemma4Request::prefill_batch_size: usize`

- [ ] **Step 1: Write boundary, last-logit, and shared-KV tests**

Reuse the existing zero/deterministic model helpers and add `run_gemma4_fixture(len, batch) -> Gemma4Snapshot`, returning final logits bits, every base-KV key/value bit, session length, and three greedy decode IDs:

```rust
#[test]
fn gemma4_prefill_matches_batch_one_across_chunk_boundaries() {
    for len in [1, 2, 3, 63, 64, 65, 127, 128] {
        let expected = run_gemma4_fixture(len, 1);
        for batch in [16, 32, 64, 128] {
            assert_eq!(run_gemma4_fixture(len, batch), expected);
        }
    }
}

#[test]
fn gemma4_only_projects_prompt_logits_for_last_row() {
    let calls = run_counting_gemma4_fixture(65, 64);
    assert_eq!(calls.output_projection_calls, 1);
}
```

The deterministic fixture must exercise at least one SWA layer, one full-attention layer, and a dependent layer reading shared base KV.
Define `Gemma4Snapshot` and `run_counting_gemma4_fixture(prompt_len, batch_size)` beside `run_gemma4_fixture`; the counting helper's existing concrete output-weight test double increments `output_projection_calls` without adding a production trait.

- [ ] **Step 2: Strengthen the existing rollback test to fail in the second row**

Keep the existing malformed-weight failure mechanism, but inject it after row one has populated tentative base KV:

```rust
#[test]
fn failed_gemma4_chunk_truncates_every_base_kv_layer() {
    let model = post_kv_failure_model();
    let mut session = Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, 4).unwrap();
    let before = snapshot_gemma4_state(&session);
    assert!(session.forward_rows(&fixture_rows(3)).is_err());
    assert_eq!(snapshot_gemma4_state(&session), before);
}
```

Define `post_kv_failure_model`, `fixture_rows`, and `snapshot_gemma4_state` in the same test module. The snapshot contains `seq_len` plus every base-KV layer's key/value raw words, and the malformed model fails only after at least two tentative rows have been written.

- [ ] **Step 3: Run the focused tests and verify the current row loop fails the call-count expectation**

```bash
cargo test --lib gemma4_prefill_matches_batch_one_across_chunk_boundaries
cargo test --lib gemma4_only_projects_prompt_logits_for_last_row
cargo test --lib failed_gemma4_chunk_truncates_every_base_kv_layer
```

Expected: new batching/call-count tests fail before implementation.

- [ ] **Step 4: Add row-major Gemma4 scratch and batched model flow**

Keep `AssembledInputRow` semantics unchanged. In each chunk:

1. Assemble token/raw embeddings and per-layer token embeddings for all rows.
2. Batch per-layer projection.
3. At each transformer layer, batch Q/K/V and gate/up projections using shared prepared rows.
4. Apply QK norm/RoPE per row with `base_position + row` and existing SWA/full selection.
5. Append K/V only for base-KV layers; dependent layers continue to use `kv_source_layer`.
6. Compute each row's causal attention against the committed prefix plus tentative prefix.
7. Preserve Gemma4 FP16 GeGLU rounding, post norms, per-layer gates, output scales, and softcap order.
8. Run output norm/vocab projection only for the final prompt row.

Do not create a second Gemma4 model implementation.

- [ ] **Step 5: Make `forward_rows` prevalidate and commit per chunk**

Before processing, check `seq_len + rows.len() <= CONTEXT`. For each chunk capture all base-layer `(keys.len(), values.len())`. On failure truncate every base layer and keep `seq_len`; on success increment `seq_len` once by the chunk length.

- [ ] **Step 6: Run Gemma4 unit/reference compile gates**

```bash
cargo test --lib models::gemma4
cargo test --test gemma4_reference --features parity-trace --no-run
cargo fmt --all -- --check
```

Expected: shared-KV/SWA/full-attention tests pass and the real-model reference test compiles with the new request field.

- [ ] **Step 7: Commit Gemma4 CPU prefill**

```bash
git add src/models/gemma4/trunk/scratch.rs src/models/gemma4/trunk/session.rs
git add src/models/gemma4/trunk/forward.rs src/models/gemma4/trunk/mod.rs src/models/gemma4/trunk/tests.rs
git add src/models/gemma4/app.rs tests/gemma4_reference.rs src/app/text.rs
git commit -m "perf: batch gemma4 cpu prefill"
```

---

### Task 6: Add generic row-aware Vulkan linear operators

**Files:**
- Modify: `src/vulkan/ops.rs:140-930`
- Modify: `shaders/glsl/quantize_q8_0.comp`
- Modify: `shaders/glsl/quantize_q8_k.comp`
- Modify: `shaders/glsl/q8_matmul_grouped.comp`
- Modify: `shaders/glsl/q4_0_matmul.comp`
- Modify: `shaders/glsl/q4_1_matmul.comp`
- Modify: `shaders/glsl/q4_k_matmul.comp`
- Modify: `shaders/glsl/q5_k_matmul.comp`
- Modify: `shaders/glsl/q6_k_matmul.comp`
- Modify: `shaders/glsl/f16_matmul.comp`
- Modify: `shaders/glsl/bf16_matmul.comp`
- Modify: `shaders/glsl/f32_matmul.comp`
- Regenerate: matching files under `shaders/bin/`
- Modify: `shaders/manifest.sha256`
- Modify: `examples/vk_ops_check.rs`

**Interfaces:**
- Produces: `Qwen3Ops::record_weight_matmul_rows(&self, commands: &TokenCommands<'_>, bindings: OperatorBindings, input: ArenaRegion, q8: ArenaRegion, q8_scales: ArenaRegion, q4_1_input_sums: ArenaRegion, q8k: ArenaRegion, q8k_scales: ArenaRegion, outputs: &[(ArenaRegion, usize, usize)], n_in: usize, token_rows: usize, input_stride: usize) -> Result<(), VulkanError>`, where each output tuple is `(region, n_out, row_stride)`
- Produces: `BatchedLinearRuntime::new(context: &'static VulkanContext, max_rows: usize, max_n_in: usize, max_n_out: usize, descriptor_capacity: usize) -> Result<Self, VulkanError>`
- Produces: `BatchedLinearRuntime::matmul_rows(&mut self, weight_bytes: &[u8], format: GpuWeightFormat, input: &[f32], rows: usize, n_in: usize, n_out: usize, output: &mut [f32]) -> Result<(), VulkanError>`
- Preserves: existing one-row `record_weight_matvec[_group]` wrappers by calling the row recorder with `rows = 1` and packed strides
- Produces under `#[cfg(test)]`: `matmul_dispatch_for_test(output_rows, token_rows, grouped_weights) -> Result<[u32; 3], VulkanError>`

- [ ] **Step 1: Add host-side shape and dispatch tests**

Test rows 1/2/3, grouped Q/K/V weights, non-packed strides, zero rows, and overflow. The dispatch expectation is:

```rust
#[test]
fn batched_matmul_dispatch_maps_output_weight_and_token_rows() {
    let dispatch = matmul_dispatch_for_test(65, 3, 2).unwrap();
    assert_eq!(dispatch, [65, 1, 6]);
}
```

Extend the ignored real-device operator check to compare batch=1 concatenation against rows=3 for every `GpuWeightFormat`, using `to_bits()` equality.

- [ ] **Step 2: Run the host test and shader check before editing**

```bash
cargo test --lib batched_matmul_dispatch_maps_output_weight_and_token_rows --features vulkan
scripts/vulkan-shaders.sh check
```

Expected: host test fails because the row-aware dispatch helper is absent; current shader check passes as the baseline.

- [ ] **Step 3: Extend push constants and arena validation**

Pass token rows, input row stride, each grouped output's row stride, and grouped weight count. Validate all byte/word ranges with checked multiplication before command recording. Use:

```text
workgroup.x = output row
workgroup.y = 1
workgroup.z = token_row * grouped_weight_count + weight_slot
```

Decode `token_row` and `weight_slot` in each matmul shader. Quantization shaders dispatch one z-plane per input row. Keep the per-output-row dot loop byte-for-byte equivalent to the one-row shader.

Implement `BatchedLinearRuntime` as a thin owner around the existing `Qwen3Ops` recorder: one arena sized by the constructor maxima, one `HashMap<(usize, usize), (GpuBuffer, OperatorBindings)>` keyed by stable source-slice pointer/length, and no model control flow. On the first weight use, upload and bind the source bytes; later calls reuse that buffer/binding. `matmul_rows` validates shapes against its maxima, records one projection command buffer, submits once, and copies `rows * n_out` outputs back. Destroy only buffers owned by this runtime on drop; do not alter `VulkanContext`'s legacy Q8 matvec cache.

- [ ] **Step 4: Regenerate and validate SPIR-V**

```bash
scripts/vulkan-shaders.sh update
scripts/vulkan-shaders.sh check
```

Expected: all checked-in binaries validate, rebuild byte-identically, and remain under the script's 64-invocation workgroup limit.

- [ ] **Step 5: Run Rust Vulkan compile and operator checks**

```bash
cargo test --lib --features vulkan batched_matmul
cargo test --example vk_ops_check --features vulkan --no-run
cargo run --release --features vulkan --example vk_ops_check -- --all-formats --rows 3
```

Expected: host tests pass. The last command passes when a Vulkan device is present; otherwise record the device/environment blocker without weakening the test.

- [ ] **Step 6: Commit generic Vulkan linear batching**

```bash
git add src/vulkan/ops.rs examples/vk_ops_check.rs
git add shaders/glsl/quantize_q8_0.comp shaders/glsl/quantize_q8_k.comp
git add shaders/glsl/q8_matmul_grouped.comp shaders/glsl/q4_0_matmul.comp shaders/glsl/q4_1_matmul.comp
git add shaders/glsl/q4_k_matmul.comp shaders/glsl/q5_k_matmul.comp shaders/glsl/q6_k_matmul.comp
git add shaders/glsl/f16_matmul.comp shaders/glsl/bf16_matmul.comp shaders/glsl/f32_matmul.comp
git add shaders/bin/quantize_q8_0.spv shaders/bin/quantize_q8_k.spv
git add shaders/bin/q8_matmul_grouped.spv shaders/bin/q4_0_matmul.spv shaders/bin/q4_1_matmul.spv
git add shaders/bin/q4_k_matmul.spv shaders/bin/q5_k_matmul.spv shaders/bin/q6_k_matmul.spv
git add shaders/bin/f16_matmul.spv shaders/bin/bf16_matmul.spv shaders/bin/f32_matmul.spv
git add shaders/manifest.sha256
git commit -m "perf: batch vulkan linear operators"
```

---

### Task 7: Reuse generic Vulkan batched linear operations in Gemma4

**Files:**
- Modify: `src/models/gemma4/trunk/forward.rs` batched `matmul` helpers
- Modify: `src/models/gemma4/trunk/session.rs`
- Modify: `src/models/gemma4/trunk/tests.rs`

**Interfaces:**
- Consumes: `BatchedLinearRuntime::matmul_rows`, `Gemma4Model::_source`, and the existing tensor-name labels already passed to Gemma4's `matmul` helper
- Produces: Gemma4 `prefill_matmul_rows` that attempts Vulkan for every prompt chunk, including the batch=1 Vulkan baseline, when GPU is requested, the weight format is supported, and no CPU-fallback scope is active
- Preserves: one-row decode on the existing CPU path

- [ ] **Step 1: Add an injectable dispatcher test**

Use a `#[cfg(test)]` mutex-backed call counter inside the concrete model-local adapter; production continues to call the same adapter without a trait or second implementation. Count one batched call per projection rather than one per token:

```rust
#[test]
fn gemma4_vulkan_linear_dispatches_once_per_projection_chunk() {
    let calls = run_counting_gemma4_vulkan_fixture(3, 3);
    assert!(calls.iter().all(|call| call.rows == 3));
    assert_eq!(calls.iter().filter(|call| call.name == "blk.0.attn_q.weight").count(), 1);
}
```

Define `run_counting_gemma4_vulkan_fixture(prompt_len, batch_size)` in the same test module; it installs the concrete adapter's scoped counter, runs the existing deterministic Gemma4 fixture, removes the counter before returning, and returns the captured `{ name, rows }` records.

- [ ] **Step 2: Run the test and confirm the CPU helper has no row-aware Vulkan call**

```bash
cargo test --lib gemma4_vulkan_linear_dispatches_once_per_projection_chunk --features vulkan
```

Expected: test fails before the adapter exists.

- [ ] **Step 3: Add the explicit Gemma4 row-matmul branch**

Construct one optional `BatchedLinearRuntime` per `Gemma4Session`, sized from `min(prefill_batch_size, CONTEXT)`, the largest multi-row model projection dimensions, and one descriptor for every distinct Gemma4 projection tensor plus the runtime's arena binding. `forward_chunk` calls `prefill_matmul_rows` even when its chunk contains one prompt row, so Vulkan batch=1 and batch=N exercise the same projection backend; the final one-row vocab projection and all decode calls stay on the existing CPU helper. Look up `weight_bytes` from `model._source.tensor_slice(label)` and call the generic Vulkan rows API with `weight.ggml_type`; never copy weight bytes into a second CPU owner. On `UnsupportedShape`, use CPU prepared rows. On runtime/device failure, call the existing global Vulkan failure path once, drop the session's runtime, then use CPU prepared rows. Do not split a projection batch after dispatch starts and do not route Gemma4 attention/KV to Vulkan in this task.

- [ ] **Step 4: Verify same-backend row bits and decode isolation**

```bash
cargo test --lib gemma4_vulkan_linear --features vulkan
cargo test --lib models::gemma4 --features vulkan
cargo test --test gemma4_reference --features 'parity-trace vulkan' --no-run
```

Expected: counting and fallback tests pass; one-row decode never enters the new generic batch branch.

- [ ] **Step 5: Commit Gemma4 Vulkan linear reuse**

```bash
git add src/models/gemma4/trunk/forward.rs src/models/gemma4/trunk/session.rs
git add src/models/gemma4/trunk/tests.rs
git commit -m "perf: use vulkan batched matmul for gemma4 prefill"
```

---

### Task 8: Implement Qwen3 full Vulkan chunk prefill and whole-chunk fallback

**Files:**
- Modify: `src/vulkan/ops.rs:180-315,679-end`
- Modify: `src/vulkan/qwen3.rs:68-178,246-end`
- Modify: `src/models/qwen3/trunk/session.rs` Vulkan dispatch block
- Modify: `shaders/glsl/rms_norm.comp`
- Modify: `shaders/glsl/qk_norm_rope.comp`
- Modify: `shaders/glsl/kv_write.comp`
- Modify: `shaders/glsl/attention_scores.comp`
- Modify: `shaders/glsl/softmax.comp`
- Modify: `shaders/glsl/attention_values.comp`
- Modify: `shaders/glsl/silu_mul.comp`
- Modify: `shaders/glsl/add.comp`
- Regenerate: matching `shaders/bin/*.spv`
- Modify: `shaders/manifest.sha256`
- Modify: `examples/vk_model_check.rs`

**Interfaces:**
- Changes: `TokenCommitState::new(committed_len: usize, capacity: usize)` and `TokenCommitState::begin(base_position: usize, rows: usize) -> Result<(), String>`
- Produces: `commit_shadow_kv_chunk(state: &mut KvState, base_position: usize, rows: usize, k_delta: &[f32], v_delta: &[f32]) -> Result<(), String>`
- Renames/extends: `GpuTokenResult<'a>` to `GpuChunkResult<'a> { logits: &'a [f32], k_delta: &'a [f32], v_delta: &'a [f32] }`
- Produces: `Qwen3VulkanSession::forward_chunk(input: &[f32], base_position: usize, rows: usize, project_logits: bool) -> Result<GpuChunkResult<'_>, VulkanError>`
- Preserves: `forward_token` as a one-row wrapper and single-token decode entry

- [ ] **Step 1: Convert token commit tests to chunk semantics**

```rust
#[test]
fn failed_chunk_does_not_advance_committed_kv() {
    let mut state = TokenCommitState::new(7, 16);
    state.begin(7, 3).unwrap();
    state.abort();
    assert_eq!(state.committed_len(), 7);
}

#[test]
fn committed_chunk_advances_once_and_checks_capacity() {
    let mut state = TokenCommitState::new(7, 9);
    assert!(state.begin(6, 3).is_err());
    assert!(state.begin(7, 0).is_err());
    assert!(state.begin(7, 3).is_err());
    let mut state = TokenCommitState::new(7, 10);
    state.begin(7, 3).unwrap();
    state.commit();
    state.commit();
    assert_eq!(state.committed_len(), 10);
}
```

Add a shadow test with two layers and three rows that verifies wrong delta length leaves `seq_len` and all visible bytes unchanged.

- [ ] **Step 2: Add a session-level whole-chunk fallback test**

Force a GPU failure on row two of a four-row chunk. Compare against a CPU-only session from the same base:

```rust
#[test]
fn qwen3_gpu_chunk_failure_recomputes_the_whole_chunk_on_cpu() {
    let (actual, actual_state) = run_qwen3_forced_gpu_failure(4, 1);
    let (expected, expected_state) = run_qwen3_cpu_only(4);
    assert_eq!(actual, expected);
    assert_eq!(actual_state, expected_state);
}
```

The injection value `1` means failure after GPU has recorded/computed row index one; the CPU retry input range must still start at row zero.
Define `run_qwen3_forced_gpu_failure(prompt_len, fail_after_row)` and `run_qwen3_cpu_only(prompt_len)` in the existing Vulkan session test module; both return final-logit bits, committed KV snapshot, and three greedy decode IDs from the same deterministic model/input.

- [ ] **Step 3: Run failing host tests**

```bash
cargo test --lib token_commit --features vulkan
cargo test --lib shadow_kv_chunk --features vulkan
cargo test --lib qwen3_gpu_chunk_failure_recomputes_the_whole_chunk_on_cpu --features vulkan
```

Expected: tests fail until commit and session APIs accept rows.

- [ ] **Step 4: Make arena regions and causal operators row-aware**

Size activation and KV-delta regions by the maximum chunk rows while keeping persistent KV by capacity. Add `rows`, `base_position`, and row strides to normalization, RoPE, KV write, attention, softmax, SiLU-mul, and add operators. For attention row `r`, set visible length to `base_position + r + 1`.

- [ ] **Step 5: Record one Qwen3 command buffer per chunk**

Upload all input rows once, run layer operators row-wise inside one `TokenCommands`, and copy/read only final-row logits plus all-row KV deltas. Call `begin(base_position, rows)` before recording, `commit()` only after GPU output validation and CPU shadow commit both succeed, and `abort()` for every error path.

- [ ] **Step 6: Recompute the entire failed range on CPU**

In `Qwen3Session`, retain the chunk's original `Range` and base state until GPU+shadow commit completes. On failure disable the full-model GPU session, enter the existing GPU-matmul-disabled scope, and call `forward_cpu_chunk` with the unchanged range. Emit one fallback message for that chunk. If CPU recomputation also fails, return the CPU error with the original Vulkan error appended as context and leave the chunk uncommitted.

- [ ] **Step 7: Regenerate shaders and run checks**

```bash
scripts/vulkan-shaders.sh update
scripts/vulkan-shaders.sh check
cargo test --lib models::qwen3 --features vulkan
cargo test --lib vulkan::qwen3 --features vulkan
cargo test --example vk_model_check --features vulkan --no-run
```

Expected: host tests and shader checks pass.

- [ ] **Step 8: Run real-device same-Vulkan equivalence**

```bash
cargo run --release --features vulkan --example vk_model_check -- qwen3 --model "$RMI_Q4_0_MODEL" --compare-prefill-batches 1,64
```

Expected: final prompt logits and 32 greedy tokens match bit-for-bit, KV state matches, and submission delta equals `ceil(prompt_tokens / 64)` for the batch-64 run.

- [ ] **Step 9: Commit Qwen3 Vulkan prefill**

```bash
git add src/vulkan/ops.rs src/vulkan/qwen3.rs src/models/qwen3/trunk/session.rs
git add shaders/glsl/rms_norm.comp shaders/glsl/qk_norm_rope.comp shaders/glsl/kv_write.comp
git add shaders/glsl/attention_scores.comp shaders/glsl/softmax.comp shaders/glsl/attention_values.comp
git add shaders/glsl/silu_mul.comp shaders/glsl/add.comp
git add shaders/bin/rms_norm.spv shaders/bin/qk_norm_rope.spv shaders/bin/kv_write.spv
git add shaders/bin/attention_scores.spv shaders/bin/softmax.spv shaders/bin/attention_values.spv
git add shaders/bin/silu_mul.spv shaders/bin/add.spv shaders/manifest.sha256 examples/vk_model_check.rs
git commit -m "perf: batch qwen3 vulkan prefill"
```

---

### Task 9: Implement Qwen3.5 full Vulkan chunk prefill with ordered recurrent scan

**Files:**
- Modify: `src/vulkan/qwen35.rs:52-end`
- Modify: `src/models/qwen35/trunk/session.rs` Vulkan dispatch block
- Modify: `shaders/glsl/qwen35_dense_prepare.comp`
- Modify: `shaders/glsl/qwen35_attention.comp`
- Modify: `shaders/glsl/qwen35_recurrent_conv.comp`
- Modify: `shaders/glsl/qwen35_recurrent_ssm.comp`
- Regenerate: matching `shaders/bin/*.spv`
- Modify: `shaders/manifest.sha256`
- Modify: `src/models/qwen35/trunk/tests.rs`
- Modify: `examples/vk_model_check.rs`

**Interfaces:**
- Produces: `commit_shadow_state_chunk(kv_cache: &mut KvCache, conv_states: &mut [Vec<f32>], ssm_states: &mut [Vec<f32>], base_position: usize, rows: usize, capacity: usize, kv_stride: usize, k_delta: &[f32], v_delta: &[f32], conv_state: &[f32], ssm_state: &[f32]) -> Result<(), String>`
- Renames/extends: `Qwen35GpuTokenResult<'a>` to `Qwen35GpuChunkResult<'a> { logits: &'a [f32], k_delta: &'a [f32], v_delta: &'a [f32], conv_state: &'a [f32], ssm_state: &'a [f32] }`
- Produces: `Qwen35VulkanSession::forward_chunk(input: &[f32], base_position: usize, positions: &[[usize; 4]], rows: usize) -> Result<Qwen35GpuChunkResult<'_>, VulkanError>`
- Preserves: `forward_token` as a one-row wrapper for decode

- [ ] **Step 1: Add chunk shadow-commit and fallback tests**

Test three-row dense K/V deltas plus final conv/SSM state. A wrong K/V or recurrent-state length must leave every CPU shadow byte and logical length unchanged. Force a row-two GPU failure and compare whole state/logits/three decode IDs with CPU-only execution.

```rust
#[test]
fn qwen35_gpu_chunk_failure_restarts_from_chunk_base() {
    let expected = run_qwen35_cpu_fixture(5, 5);
    let actual = run_qwen35_gpu_failure_fixture(5, 5, 2);
    assert_eq!(actual, expected);
}
```

Define `run_qwen35_cpu_fixture(prompt_len, batch_size)` and `run_qwen35_gpu_failure_fixture(prompt_len, batch_size, fail_after_row)` in `tests.rs`; both return raw dense-KV/conv/SSM/logit words plus three greedy decode IDs from the same deterministic mixed dense/recurrent model.

- [ ] **Step 2: Run the tests and verify current prefill still loops `forward_token`**

```bash
cargo test --lib qwen35_gpu_chunk_failure_restarts_from_chunk_base --features vulkan
cargo test --lib commit_shadow_state_chunk --features vulkan
```

Expected: tests fail before chunk APIs exist.

- [ ] **Step 3: Batch dense operators and preserve position arrays**

Use the generic row operators for dense projections. Make dense prepare/attention index each token's four-component mRoPE position and use causal visible length `base_position + row + 1`.

- [ ] **Step 4: Scan recurrent rows sequentially inside each state lane**

Dispatch state lanes in x/y, but loop `row = 0..rows` inside `qwen35_recurrent_conv.comp` and `qwen35_recurrent_ssm.comp`. The shader writes only the final chunk state to persistent state buffers while retaining per-row outputs for the batched output projection. Do not dispatch token rows as independent state writers.

- [ ] **Step 5: Commit GPU and CPU shadow state atomically**

Begin the shared pending-chunk state once. Read back all dense deltas plus final recurrent state; validate lengths and finiteness; commit the CPU shadow; then advance GPU commit state. Any failure aborts and invokes CPU for the original whole range under the GPU-matmul-disabled scope. If CPU recomputation fails, return that error with the original Vulkan error appended as context and leave the chunk uncommitted.

- [ ] **Step 6: Regenerate shaders and run focused checks**

```bash
scripts/vulkan-shaders.sh update
scripts/vulkan-shaders.sh check
cargo test --lib models::qwen35 --features vulkan
cargo test --lib vulkan::qwen35 --features vulkan
cargo test --example vk_model_check --features vulkan --no-run
```

Expected: host rollback tests and shader validation pass.

- [ ] **Step 7: Run real-device same-Vulkan equivalence**

```bash
cargo run --release --features vulkan --example vk_model_check -- qwen35 --model "$RMI_QWEN35_MODEL" --compare-prefill-batches 1,64
```

Expected: dense KV, conv/SSM state, final logits, and 32 greedy tokens match raw bits; batch-64 prompt submissions equal `ceil(prompt_tokens / 64)`.

- [ ] **Step 8: Commit Qwen3.5 Vulkan prefill**

```bash
git add src/vulkan/qwen35.rs src/models/qwen35/trunk/session.rs src/models/qwen35/trunk/tests.rs
git add shaders/glsl/qwen35_dense_prepare.comp shaders/glsl/qwen35_attention.comp
git add shaders/glsl/qwen35_recurrent_conv.comp shaders/glsl/qwen35_recurrent_ssm.comp
git add shaders/bin/qwen35_dense_prepare.spv shaders/bin/qwen35_attention.spv
git add shaders/bin/qwen35_recurrent_conv.spv shaders/bin/qwen35_recurrent_ssm.spv
git add shaders/manifest.sha256 examples/vk_model_check.rs
git commit -m "perf: batch qwen3.5 vulkan prefill"
```

---

### Task 10: Wire CLI, library, multimodal, ASR, and server entry points

**Files:**
- Modify: `src/app/mod.rs` dispatch arguments
- Modify: `src/app/text.rs:381-end`
- Modify: `src/models/qwen3/text.rs`
- Modify: `src/models/qwen3/asr/model.rs`
- Modify: `src/models/gemma4/app.rs`
- Modify: `src/bin/server.rs:980-1190`
- Modify: CLI tests in `src/app/cli.rs`
- Modify: affected call sites returned by:
  - `rg -n 'Qwen3GenerateOptions \{' src tests examples`
  - `rg -n 'Gemma4Request \{' src tests examples`
  - `rg -n 'Qwen35Session::new\(' src tests examples`

**Interfaces:**
- Consumes: `CliOptions::effective_prefill_batch_size()` and the three model APIs added above
- Produces: all supported main CLI and server generation paths use the chosen batch size
- Does not add: Gemma4 server support, because the current server backend does not support Gemma4; this task only propagates through existing server model paths

- [ ] **Step 1: Enumerate and update every concrete construction site**

Run the three literal searches listed under **Files** and update each returned construction site explicitly. Resolve the CLI value once and assign that same `usize` to Qwen3 text/multimodal/ASR options, the Qwen3.5 session constructor, and `Gemma4Request`. For server startup, resolve once in `build_text` and store it in the existing backend/session configuration.

```bash
rg -n 'Qwen3GenerateOptions \{' src tests examples
rg -n 'Gemma4Request \{' src tests examples
rg -n 'Qwen35Session::new\(' src tests examples
```

- [ ] **Step 2: Run CLI validation and compile every construction site**

```bash
cargo test --lib prefill_batch_size
cargo test --all-targets --no-run
```

Expected: CLI default/validation tests pass and every explicit request/options/session literal compiles. Do not add a test-only request builder or capture trait solely to test field plumbing; Tasks 3-9 test that the propagated value changes real chunk behavior.

- [ ] **Step 3: Thread one `usize` through existing calls**

Resolve `effective_prefill_batch_size()` once after CLI validation. Pass it to Qwen3 `Qwen3GenerateOptions`, Qwen3.5 `new_with_prefill_batch_size`, and `Gemma4Request`. Ensure media position/deepstack arrays are sliced by the exact Qwen3 chunk range. Keep one-token decode calls unchanged.

For `rust-model-server`, store the configured batch size in the existing text backend/session construction; do not add an OpenAI request JSON field and do not make batch size request-specific.

- [ ] **Step 4: Verify every literal and help surface**

```bash
rg -n 'Qwen3GenerateOptions \{|Gemma4Request \{|Qwen35Session::new\(' src tests examples
cargo run --bin rust-model-inference -- --help | rg -- '--prefill-batch-size.*64'
cargo run --bin rust-model-server -- --help | rg -- '--prefill-batch-size.*64'
cargo test --all-targets --no-run
```

Expected: all literals compile with an intentional default or propagated value, both binaries document the flag/default, and all targets compile.

- [ ] **Step 5: Commit entry-point wiring**

```bash
git add src/app/mod.rs src/app/text.rs src/app/cli.rs src/main.rs src/bin/server.rs
git add src/models/qwen3/text.rs src/models/qwen3/asr/model.rs src/models/qwen3/trunk/forward.rs
git add src/models/qwen35/trunk/tests.rs src/models/gemma4/app.rs
git add tests/gemma4_reference.rs examples/vk_model_check.rs
git commit -m "feat: wire chunked prefill through inference entry points"
```

---

### Task 11: Add repeatable real-model parity and performance gates

**Files:**
- Create: `examples/prefill_bench.rs`
- Modify: `examples/vk_model_check.rs`
- Modify: `tests/inference_parity.rs`
- Modify: `tests/qwen35_reference.rs`
- Modify: `tests/gemma4_reference.rs`
- Modify: `src/models/qwen3/trunk/tests.rs`
- Modify: `src/models/qwen35/trunk/tests.rs`
- Modify: `src/models/gemma4/trunk/tests.rs`

**Interfaces:**
- Produces CLI: `prefill_bench <qwen3|qwen35|gemma4> --model PATH --backend <cpu|vulkan> --threads N --kv <f16|f32> --prompt-tokens N --batch N --samples N --generate 32`
- Produces one machine-readable `kind=sample` line per sample plus one `kind=median` line, each with: model SHA256, model name, backend/device, threads, KV format, prompt tokens, batch size, pp, tg, first-token ms, total ms, model-reported scratch bytes, and Vulkan submission delta when applicable
- Produces: dependency-free `BenchSample`, `median(&mut [f64]) -> f64`, and `summary_line(kind: &str, sample: &BenchSample) -> String`
- Preserves Oracle contract: token IDs, checkpoint name/order/shape/count, F32 `u32`, and greedy IDs

- [ ] **Step 1: Write parser/median/output tests for the benchmark example**

Keep parsing dependency-free. Test that `--prompt-tokens 0`, `--batch 0`, `--samples 0`, and `--generate != 32` are rejected. Test median and the exact output keys:

```rust
#[test]
fn benchmark_summary_contains_reproduction_fields() {
    let line = summary_line("sample", &fixture_sample());
    for key in [
        "kind=sample", "sha256=", "model=", "backend=", "device=", "threads=", "kv=", "prompt_tokens=",
        "batch=", "pp_tps=", "tg_tps=", "first_token_ms=", "total_ms=",
        "scratch_bytes=",
    ] {
        assert!(line.contains(key), "missing {key}: {line}");
    }
}
```

Define `fixture_sample() -> BenchSample` in the example's test module with fixed values for every field. `summary_line` accepts only `sample` or `median`, prints keys in the asserted order, and appends `submission_delta=` only for Vulkan samples. After collecting samples, build the median `BenchSample` by applying `median` independently to pp/tg/latency fields and print it with `kind=median`.

- [ ] **Step 2: Implement exact-length prompt and measurement loops**

Tokenize a fixed seed phrase, retain valid non-special IDs, and cycle them until exactly `prompt_tokens`; preserve the architecture's required BOS at row zero. Warm up once, reset the session, then collect five samples by default. Measure prompt and 32-token greedy decode separately with `Instant`. For Vulkan, subtract `VulkanContext::submission_count()` before/after each phase.

Do not include model load/tokenization time in pp/tg. Print SHA256 from the already installed `sha2` dependency.

- [ ] **Step 3: Extend reference tests to run batch=1 and batch=64**

For each reference test, run Rust twice from clean sessions. First assert the two Rust traces and greedy IDs are raw-bit identical; then compare the batch-64 trace to the pinned Oracle. Under `parity-trace`, emit per-row checkpoint records in the existing order/shape even when computation is batched.

Use existing environment variables:

- Qwen3: `RMI_Q4_0_MODEL` or `RMI_Q4_K_M_MODEL`, `RMI_LLAMA_ORACLE`
- Qwen3.5: `RMI_QWEN35_MODEL`, `RMI_LLAMA_CPP`
- Gemma4: `RMI_GEMMA4_MODEL`, `LLAMA_CPP_DIR` or `LLAMA_GEMMA4_TRACE_BIN`

- [ ] **Step 4: Add the full boundary matrix to tiny-model tests**

Expand model test loops to exactly `[1, 2, 3, 63, 64, 65, 127, 128]`, plus empty-input rejection, capacity-equal, and capacity+1 cases. Assert `scratch_bytes()` is bounded by batch size for all three models. Add one Qwen3 fixture with nontrivial four-component positions and deepstack rows and one Qwen3.5 fixture with nontrivial mRoPE positions; compare batch=1 and batch=64 on CPU, and assert the Qwen3 deepstack case remains ineligible for the full Vulkan executor.

- [ ] **Step 5: Run local non-model gates**

```bash
cargo test --all-targets
cargo test --all-targets --features parity-trace
cargo test --all-targets --features vulkan --no-run
cargo test --example prefill_bench --features vulkan
cargo fmt --all -- --check
scripts/vulkan-shaders.sh check
```

Expected: all available local tests pass; device/model-dependent tests remain explicitly ignored until invoked below.

- [ ] **Step 6: Run fixed-Oracle real-model parity**

Record hashes and Oracle identities first:

```bash
shasum -a 256 "$RMI_Q4_0_MODEL" "$RMI_QWEN35_MODEL" "$RMI_GEMMA4_MODEL"
shasum -a 256 "$RMI_LLAMA_ORACLE"
git -C "$RMI_LLAMA_CPP" rev-parse HEAD
```

For Gemma4, also run `git -C "$LLAMA_CPP_DIR" rev-parse HEAD` when the test builds the Oracle from that checkout; when `LLAMA_GEMMA4_TRACE_BIN` is supplied instead, record `shasum -a 256 "$LLAMA_GEMMA4_TRACE_BIN"`. Record `RMI_Q4_K_M_MODEL` separately when running the optional Q4_K_M case.

Then run:

```bash
cargo test --release --features parity-trace --test inference_parity q4_0_matches_pinned_scalar_oracle_bit_for_bit -- --ignored --nocapture
cargo test --release --features parity-trace --test qwen35_reference qwen38_matches_pinned_llama_cpp_at_lossless_checkpoints -- --ignored --nocapture
cargo test --release --features parity-trace --test gemma4_reference gemma4_matches_pinned_cpu_oracle_before_softmax -- --ignored --nocapture
```

When `RMI_Q4_K_M_MODEL` is available, run its exact optional case separately:

```bash
cargo test --release --features parity-trace --test inference_parity q4_k_m_matches_pinned_scalar_oracle_bit_for_bit -- --ignored --nocapture
```

Do not use a bare `--ignored` filter for `qwen35_reference`; that binary also contains unrelated NeoHorse/model tests with different artifact requirements.

Expected: batch=1, batch=64, and pinned Oracle token/checkpoint/logit/greedy comparisons pass. If an environment variable or artifact is absent, record that precise validation gap; do not report the corresponding model as verified.

- [ ] **Step 7: Run the CPU benchmark matrix**

For every model, prompt size `8 32 128 512`, and batch `1 16 32 64 128`, run five samples with 32 decode tokens. Use model-appropriate KV (`f16` for Qwen3, `f32` for Qwen3.5/Gemma4):

```bash
for prompt_tokens in 8 32 128 512; do
  for batch in 1 16 32 64 128; do
    cargo run --release --example prefill_bench -- qwen3 \
      --model "$RMI_Q4_0_MODEL" --backend cpu --threads 4 --kv f16 \
      --prompt-tokens "$prompt_tokens" --batch "$batch" --samples 5 --generate 32
  done
done
```

Repeat the same command with `qwen35`/F32 and `gemma4`/F32. Keep thread count fixed for comparison; calibrate thread count in a separate preliminary run rather than assuming all cores are faster. For each model/backend/batch combination, also run one representative invocation under `/usr/bin/time -l` on macOS (or `/usr/bin/time -v` on Linux) and record peak RSS next to the harness's `scratch_bytes`; do not add an in-process platform abstraction only to read RSS.

- [ ] **Step 8: Run the Vulkan benchmark matrix**

Repeat the Step 7 loops on a real device, adding the Vulkan feature and backend as shown here:

```bash
cargo run --release --features vulkan --example prefill_bench -- qwen3 \
  --model "$RMI_Q4_0_MODEL" --backend vulkan --threads 4 --kv f16 \
  --prompt-tokens "$prompt_tokens" --batch "$batch" --samples 5 --generate 32
```

Use the corresponding Qwen3.5/Gemma4 model variables and F32 KV. Check Qwen3/Qwen3.5 prompt submission deltas against `ceil(prompt_tokens / batch)` and Gemma4 projection dispatch counts against `ceil(prompt_tokens / batch)` per projection.

Expected default-64 gates for every applicable model/backend:

- 512-token pp median is at least 10% above batch=1.
- 128-token pp median is no more than 3% below batch=1.
- tg median is no more than 3% below batch=1.
- Qwen3/Qwen3.5 Vulkan submission counts match the chunk count.
- Existing unit tests prove scratch grows with batch size, not prompt length.

If any applicable target misses a correctness or performance gate, stop the completion/release step and report the exact model/backend, raw samples, and failed threshold. Do not silently change the approved default from 64, disable only that route, or add auto-tuning; fix the target or obtain an explicit spec revision before completing the work.

- [ ] **Step 9: Commit the reusable verification harness**

```bash
git add examples/prefill_bench.rs examples/vk_model_check.rs
git add tests/inference_parity.rs tests/qwen35_reference.rs tests/gemma4_reference.rs
git add src/models/qwen3/trunk/tests.rs src/models/qwen35/trunk/tests.rs src/models/gemma4/trunk/tests.rs
git commit -m "test: verify chunked prefill parity and performance"
```

---

### Task 12: Document the final supported behavior and run the release gate

**Files:**
- Modify: `README.md` CLI option table and Vulkan limitations
- Modify: `docs/usage/qwen3.md`
- Create: `docs/usage/prefill-batching.md`

**Interfaces:**
- Documents: default 64, batch=1 baseline, supported model/backend matrix, exact benchmark command, and whole-chunk fallback behavior
- Does not document: continuous batching or unsupported Vulkan architectures

- [ ] **Step 1: Write usage text from verified behavior**

Document these exact examples:

```bash
rust-model-inference --model model.gguf --prompt "Hello" --prefill-batch-size 64
rust-model-inference --model model.gguf --prompt "Hello" --prefill-batch-size 1
```

State that 1 is the diagnostic sequential baseline and 64 is the default. Documentation is written only after every applicable model/backend has passed Task 11; if a target has not passed, Task 12 is blocked rather than documenting a conditional or silent fallback. Include the Gemma4 distinction: Vulkan accelerates batched linear projections while model attention/KV remain on the model-controlled CPU path.

- [ ] **Step 2: Run the complete release gate**

```bash
git diff --check
cargo fmt --all -- --check
cargo test --all-targets
cargo test --all-targets --features parity-trace
cargo test --all-targets --features vulkan
scripts/vulkan-shaders.sh check
cargo build --release --bin rust-model-inference
cargo build --release --bin rust-model-server
cargo build --release --features vulkan --example prefill_bench
```

Expected: every available check passes. Separate baseline, environment, and ignored real-model/device failures explicitly; never convert an unrun Oracle/device test into a passing claim.

- [ ] **Step 3: Inspect scope and decode regression**

```bash
git diff --stat 3e952e0..HEAD
git diff --name-only 3e952e0..HEAD
rg -n "continuous batching|paged KV|auto.?tun" src Cargo.toml
```

Expected: only planned files changed, `Cargo.toml` has no new dependency, no scheduling subsystem exists, and Task 11 tg medians remain within 3% of batch=1.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs/usage/qwen3.md docs/usage/prefill-batching.md
git commit -m "docs: document chunked prefill batching"
```

- [ ] **Step 5: Record completion evidence for review**

Provide the final commit list, exact commands run, local test counts, shader validation result, real-model SHA256 values, Oracle revisions, CPU/Vulkan benchmark medians, submission counts, and every validation gap. Do not stage or modify `.codex/`.
