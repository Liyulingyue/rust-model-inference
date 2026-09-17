# DSpark Speculative Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact greedy DSpark speculative decoding for Qwen3-4B and LFM2.5-1.2B-Instruct with matching GGUF sidecars.

**Architecture:** A shared `models::dspark` module loads the DFlash/DSpark sidecar, maintains its encoder/decoder cache, drafts one block, and verifies the block through a small generic target-session interface. Qwen3 reuses its batched CPU prefill path; LFM2.5 moves its existing token loop into a stateful session so rejected tails can restore short-convolution state.

**Tech Stack:** Rust, existing GGUF/TensorSource loader, existing CPU kernels and ComputePool, repository tokenizer, fixed llama.cpp Oracle.

**Spec:** `docs/superpowers/specs/2026-09-17-dspark-speculative-decoding-design.md`

## Global Constraints

- Do not add or call BLAS/LAPACK, OpenBLAS, MKL, Accelerate, oneDNN, cuBLAS, rocBLAS, or another acceleration library.
- Never route an unknown architecture to a nearby model implementation.
- DSpark is CPU-only and greedy-only in this version.
- Enabling DSpark must preserve target-only greedy token IDs exactly.
- Oracle commit: `ggml-org/llama.cpp@84075273c82f7681d43436b692073cbd4ab15fe9`.
- Preserve the untracked `.codex/` directory and stage explicit paths only.

---

### Task 1: CLI contract and routing

**Files:**
- Modify: `src/app/cli.rs`
- Modify: `src/app/text.rs`
- Modify: `src/main.rs`

**Interfaces:**
- Produces: `CliOptions::draft_model: Option<PathBuf>`
- Produces: `CliOptions::spec_draft_n_max: Option<usize>`
- Produces: `CliOptions::spec_draft_conf_min: f32`
- Produces: `DSparkOptions { draft_model: PathBuf, draft_n_max: Option<usize>, confidence_min: f32 }`
- Consumes later: `app::text::run_inference(..., dspark: Option<DSparkOptions>)`

- [ ] **Step 1: Write failing parser and validation tests**

Add tests under `src/app/cli.rs` that assert:

```rust
let parsed = parse_cli_options(&args(&[
    "rmi", "--draft-model", "draft.gguf",
    "--spec-draft-n-max", "7",
    "--spec-draft-conf-min", "0.4",
])).unwrap();
assert_eq!(parsed.draft_model.as_deref(), Some(Path::new("draft.gguf")));
assert_eq!(parsed.spec_draft_n_max, Some(7));
assert_eq!(parsed.spec_draft_conf_min, 0.4);

let mut sampled = parsed;
sampled.temperature = Some(0.1);
assert!(validate_cli_options(&sampled).unwrap_err().contains("greedy"));
```

Also assert missing values, `n_max == 0`, and confidence outside `0.0..=1.0` are rejected.

- [ ] **Step 2: Verify the tests fail for missing fields**

Run: `cargo test app::cli::tests::dspark --lib -- --nocapture`

Expected: compilation fails because the three `CliOptions` fields do not exist.

- [ ] **Step 3: Implement only the three flags and greedy validation**

Parse values with the existing `required_path_value`, `required_usize_value`, and `required_string_value` helpers. Add:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct DSparkOptions {
    pub draft_model: PathBuf,
    pub draft_n_max: Option<usize>,
    pub confidence_min: f32,
}
```

Pass `Option<DSparkOptions>` from `main.rs` into the text entry point. Leave target-only calls as `None`.

- [ ] **Step 4: Run focused tests**

Run: `cargo test app::cli::tests::dspark --lib -- --nocapture`

Expected: all DSpark CLI tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/app/cli.rs src/app/text.rs src/main.rs
git commit -m "feat: add DSpark CLI contract"
```

### Task 2: Sidecar metadata and tensor contract

**Files:**
- Create: `src/models/dspark/mod.rs`
- Create: `src/models/dspark/config.rs`
- Modify: `src/models/mod.rs`

**Interfaces:**
- Produces: `DSparkConfig::from_source(source: &dyn TensorSource, target: TargetShape) -> Result<Self, String>`
- Produces: `TargetShape { hidden: usize, vocab: usize, layers: usize }`
- Produces config fields: `block_size`, `target_layers`, `hidden`, `layers`, `heads`, `kv_heads`, `head_dim`, `ffn`, `vocab`, `eps`, `rope_base`, `markov_rank`

- [ ] **Step 1: Write failing synthetic metadata tests**

Use a local `FixtureSource` implementing `TensorSource`. The valid fixture declares `general.architecture = "dflash"`, `dflash.block_size = 7`, `dflash.target_layer_ids = [1, 9, 17, 25, 33]`, Qwen dimensions, and tensor infos for `markov_w1.weight`, `markov_w2.weight`, and `conf_proj.weight`.

Assert the valid fixture parses, while these fixtures fail with the named key in the error:

```rust
assert!(parse_with_arch("qwen3").unwrap_err().contains("general.architecture"));
assert!(parse_with_block_size(0).unwrap_err().contains("dflash.block_size"));
assert!(parse_with_target_layers(vec![]).unwrap_err().contains("target_layer_ids"));
assert!(parse_with_vocab(TARGET_VOCAB + 1).unwrap_err().contains("vocabulary"));
```

- [ ] **Step 2: Verify the tests fail because `DSparkConfig` is absent**

Run: `cargo test models::dspark::config::tests --lib -- --nocapture`

Expected: compilation fails on unresolved `DSparkConfig`.

- [ ] **Step 3: Implement strict metadata parsing**

Reuse `MetaValue` conversion and checked integer conversion. Accept only `dflash`; reject target layers greater than the target layer count; require exact target hidden and vocabulary compatibility. Derive `markov_rank` from `markov_w1.weight` and check:

```text
markov_w1.weight = [rank, vocab]
markov_w2.weight = [rank, vocab]
conf_proj.weight = [hidden + rank, 1]
conf_proj.bias   = [1] (optional)
fc.weight        = [target_layers.len * target_hidden, hidden]
```

- [ ] **Step 4: Run config tests and formatting**

Run: `cargo test models::dspark::config::tests --lib -- --nocapture`

Run: `cargo fmt --check`

Expected: both commands exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/models/mod.rs src/models/dspark/mod.rs src/models/dspark/config.rs
git commit -m "feat: validate DSpark sidecar contracts"
```

### Task 3: DSpark draft model

**Files:**
- Create: `src/models/dspark/model.rs`
- Modify: `src/models/dspark/mod.rs`
- Test: `src/models/dspark/model.rs`

**Interfaces:**
- Produces: `DSparkModel::from_source(Arc<dyn TensorSource>, SharedHead, Arc<ComputePool>) -> Result<Self, String>`
- Produces: `DSparkSession::new(&DSparkModel, capacity: usize) -> Result<Self, String>`
- Produces: `DSparkSession::inject(&mut self, position: usize, features: &[f32]) -> Result<(), String>`
- Produces: `DSparkSession::draft(&mut self, last_token: u32, n: usize, confidence_min: f32) -> Result<DraftBlock, String>`
- Produces: `DraftBlock { token_ids: Vec<u32>, confidence: Vec<f32> }`
- Consumes: target `token_embd.weight` and `output.weight` through `SharedHead`

- [ ] **Step 1: Write failing Markov-head and confidence tests**

Construct tiny F32 tensors with `vocab=4`, `hidden=2`, `rank=1`, `block_size=3`. Assert the Markov chain uses the previous predicted token:

```rust
let block = fixture_session().draft(1, 3, 0.0).unwrap();
assert_eq!(block.token_ids, vec![2, 3, 0]);
assert_eq!(block.confidence.len(), 3);
```

Set the second confidence below `0.5` and assert `draft(..., 0.5)` returns only the first token.

- [ ] **Step 2: Verify the tests fail because the draft model is absent**

Run: `cargo test models::dspark::model::tests --lib -- --nocapture`

Expected: compilation fails on unresolved `DSparkModel` and `DSparkSession`.

- [ ] **Step 3: Implement the sidecar with existing operators**

Load `fc`, encoder norm, per-layer attention/FFN weights, output norm, Markov weights, and confidence projection with existing `Weight` and F32 tensor helpers. `inject` fuses concatenated target features, normalizes the result, and writes decoder K/V at the target position. `draft` runs one non-causal block whose first input is `last_token` and remaining inputs are the mask token, then applies:

```text
bias_i = markov_w2 * markov_w1[previous_token]
logits_i = base_logits_i + bias_i
confidence_i = sigmoid(conf_proj * concat(hidden_i, markov_w1[previous_token]) + bias)
previous_token = argmax(logits_i)
```

Use the repository's `argmax`, RMS norm, RoPE, attention, quantization, and matmul kernels. Do not introduce a dependency.

- [ ] **Step 4: Run draft-model tests**

Run: `cargo test models::dspark::model::tests --lib -- --nocapture`

Expected: the Markov chain and confidence truncation tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/dspark/mod.rs src/models/dspark/model.rs
git commit -m "feat: implement DSpark draft model"
```

### Task 4: Shared verification engine and Qwen3 adapter

**Files:**
- Create: `src/models/dspark/engine.rs`
- Modify: `src/models/dspark/mod.rs`
- Modify: `src/models/qwen3/trunk/prefill.rs`
- Modify: `src/models/qwen3/trunk/session.rs`
- Modify: `src/models/qwen3/text.rs`

**Interfaces:**
- Produces: `trait DSparkTarget { type Checkpoint; fn checkpoint(&self) -> Self::Checkpoint; fn restore(&mut self, checkpoint: &Self::Checkpoint); fn evaluate(&mut self, token_ids: &[u32], target_layers: &[usize]) -> Result<TargetBatch, String>; fn current_logits(&self) -> &[f32]; }`
- Produces: `TargetBatch { logits: Vec<Vec<f32>>, features: Vec<Vec<f32>> }`
- Produces: `run_greedy<T: DSparkTarget>(target: &mut T, draft: &mut DSparkSession, options: RunOptions, on_token: impl FnMut(u32)) -> Result<DSparkStats, String>`
- Produces: `DSparkStats { drafted: usize, accepted: usize, target_evaluations: usize }`

- [ ] **Step 1: Write failing verification tests**

Use a deterministic fake target returning chosen argmax IDs. Cover full acceptance and first mismatch:

```rust
assert_eq!(verify_ids(&[2, 3, 4], &[2, 3, 4]), Verification::Accepted(3));
assert_eq!(verify_ids(&[2, 3, 4], &[2, 9, 4]), Verification::Rejected { accepted: 1, target: 9 });
```

Assert a rejected tail restores the checkpoint, replays only the accepted prefix, and leaves the replacement token pending for the next target evaluation.

- [ ] **Step 2: Verify the engine tests fail**

Run: `cargo test models::dspark::engine::tests --lib -- --nocapture`

Expected: compilation fails because `Verification` and `run_greedy` are absent.

- [ ] **Step 3: Implement the minimal shared loop**

Use target logits as the authority. After a batched verification, restore the checkpoint and replay only accepted draft IDs; ignore rejected target KV rows. Inject features into the sidecar only for committed, processed tokens. Count every proposed/accepted token and every target batch.

- [ ] **Step 4: Expose Qwen3 per-row logits and requested layer inputs**

Extend CPU chunk prefill with an optional capture descriptor. Copy `x` before each requested layer into token-major feature rows and project logits for every verified row. Implement `DSparkTarget` for a Qwen3 session wrapper; `Checkpoint` is the committed `seq_len`. Disable the Vulkan session when DSpark is selected because the initial version requires CPU layer captures.

- [ ] **Step 5: Run engine and Qwen3 tests**

Run: `cargo test models::dspark::engine::tests models::qwen3::trunk::tests --lib -- --nocapture`

Expected: full acceptance, mismatch rollback, and existing Qwen3 generation tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/models/dspark src/models/qwen3/trunk/prefill.rs src/models/qwen3/trunk/session.rs src/models/qwen3/text.rs
git commit -m "feat: run DSpark with Qwen3 targets"
```

### Task 5: LFM2.5 target adapter

**Files:**
- Modify: `src/models/lfm25/trunk/forward.rs`
- Modify: `src/models/lfm25/trunk/mod.rs`
- Modify: `src/app/text.rs`
- Create: `src/models/lfm25/trunk/tests.rs`

**Interfaces:**
- Produces: `Lfm25Session::new(source, pool, capacity, kv_format) -> Result<Self, String>`
- Produces: `Lfm25Checkpoint { position: usize, shortconv_states: Vec<Vec<f32>> }`
- Implements: `DSparkTarget for Lfm25Session`
- Consumes: shared `run_greedy` from Task 4

- [ ] **Step 1: Write a failing state-restore test**

Build a tiny `FixtureSource` with one attention layer, one short-convolution layer, F32 weights, `hidden=4`, `heads=2`, and `d_conv=2`. Evaluate three tokens, restore the checkpoint taken after one token, evaluate a replacement token, and compare logits and short-convolution state with a fresh session that evaluated only the prefix plus replacement.

```rust
assert_eq!(restored.logit_bits(), fresh.logit_bits());
assert_eq!(restored.shortconv_state_bits(), fresh.shortconv_state_bits());
```

- [ ] **Step 2: Verify the test fails because `Lfm25Session` is absent**

Run: `cargo test models::lfm25::trunk::tests::restore --lib -- --nocapture`

Expected: compilation fails on unresolved `Lfm25Session`.

- [ ] **Step 3: Move existing mutable decode state into `Lfm25Session`**

Move the current scratchpad, layer weights, output head, KV cache, short-convolution state, and position into the session without changing arithmetic. Make target-only `run_inference` call the session one token at a time so its behavior stays unchanged.

- [ ] **Step 4: Implement feature capture and checkpoint restore**

Capture `scratch.x` immediately before each requested layer. Restore `position` and cloned short-convolution state; KV rows are retained but ignored above `position`. Re-evaluation overwrites those rows. Connect `run_greedy` when `DSparkOptions` is present.

- [ ] **Step 5: Run focused LFM2.5 and DSpark tests**

Run: `cargo test models::lfm25 models::dspark --lib -- --nocapture`

Expected: restore equivalence, target-only generation, and shared verification tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/models/lfm25 src/models/dspark src/app/text.rs
git commit -m "feat: run DSpark with LFM2.5 targets"
```

### Task 6: Fixed-artifact Oracle and real CLI validation

**Files:**
- Create: `tests/dspark_reference.rs`
- Create: `docs/usage/dspark.md`
- Modify: `docs/develop/SUPPORTED_MODELS.md`
- Modify: `docs/MODEL_LIST.md`

**Interfaces:**
- Test environment variables: `RMI_QWEN3_DSPARK_TARGET`, `RMI_QWEN3_DSPARK_DRAFT`, `RMI_LFM25_DSPARK_TARGET`, `RMI_LFM25_DSPARK_DRAFT`, `RMI_DSPARK_ORACLE`

- [ ] **Step 1: Fetch and hash fixed real artifacts outside git**

Use the repository revisions and files recorded in the design investigation:

```text
Qwen/Qwen3-4B-GGUF@bc640142c66e1fdd12af0bd68f40445458f3869b
  Qwen3-4B-Q4_K_M.gguf
deepseek-ai/dspark_qwen3_4b_block7@3457dff1417cb84927f6098a5fcb7cee85c934b7
  config.json, model.safetensors
Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c
  config.json, tokenizer.json, tokenizer_config.json, vocab.json, merges.txt
LiquidAI/LFM2.5-1.2B-Instruct-GGUF@6767265158422fb8a19c62ceb45f16f05363615b
  LFM2.5-1.2B-Instruct-Q4_K_M.gguf
LiquidAI/LFM2.5-1.2B-Instruct-DSpark-GGUF@9235d674d3bbda8a775ca42a4275d11a8c0ab008
  LFM2.5-1.2B-Instruct-DSpark-Q4_K_M.gguf
```

Convert the Qwen sidecar with the fixed Oracle checkout's `convert_hf_to_gguf.py --target-model-dir ... --outtype bf16`. Record `sha256sum` output for every target and sidecar.

- [ ] **Step 2: Write ignored real-model tests before running the implementation**

The tests launch target-only Rust, DSpark Rust, and fixed llama.cpp with the same prompt, one thread, F32 KV, greedy decoding, and seven draft tokens. Parse token IDs, draft confidence, and acceptance counters. Assert:

```rust
assert_eq!(rust_dspark.generated_ids, rust_target.generated_ids);
assert_eq!(rust_dspark.generated_ids, oracle.generated_ids);
assert_eq!(rust_dspark.draft_ids, oracle.draft_ids);
assert_eq!(rust_dspark.accepted_per_block, oracle.accepted_per_block);
```

- [ ] **Step 3: Run Qwen3 real-model parity**

Run: `cargo test --release --features parity-trace --test dspark_reference qwen3 -- --ignored --nocapture`

Expected: Rust target-only, Rust DSpark, and Oracle token/draft/acceptance sequences are identical.

- [ ] **Step 4: Run LFM2.5 real-model parity**

Run: `cargo test --release --features parity-trace --test dspark_reference lfm25 -- --ignored --nocapture`

Expected: Rust target-only, Rust DSpark, and Oracle token/draft/acceptance sequences are identical.

- [ ] **Step 5: Document exact commands, hashes, and limits**

Document both CLI invocations, sidecar pairing requirements, artifact hashes, fixed Oracle commit, greedy/CPU restriction, and drafted/accepted counter meanings. Mark only the two tested target/sidecar pairs as `Verified`.

- [ ] **Step 6: Run final verification**

Run: `cargo fmt --check`

Run: `cargo test --lib`

Run: `cargo test --test dspark_reference -- --ignored --nocapture`

Run: `cargo build --release --bin rust-model-inference`

Run: `git diff --check`

Expected: every command exits 0; the two real runs report identical target-only/DSpark tokens and non-zero accepted counts.

- [ ] **Step 7: Commit**

```bash
git add tests/dspark_reference.rs docs/usage/dspark.md docs/develop/SUPPORTED_MODELS.md docs/MODEL_LIST.md
git commit -m "test: verify DSpark target pairs"
```
