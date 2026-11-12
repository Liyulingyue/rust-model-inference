# Speculative decoding infrastructure (MTP / DFlash / DSpark / EAGLE3 / standalone draft)

> **Status (2026-09-23)**: design draft, not yet implemented.
> Reference implementation studied: `references/llama.cpp/src/models/dflash.cpp`
> (DFlash backbone + DSpark extension), `references/llama.cpp/src/models/deepseek4.cpp`
> (MTP module, `graph_mtp`), `references/llama.cpp/common/speculative.{h,cpp}`
> (main loop + acceptance test), `references/llama.cpp/src/llama-context.cpp`
> (`llama_set_embeddings_layer_inp` hook).
>
> **Refreshed 2026-11-11** against the pinned `llama.cpp b96806d` (the same
> commit this repository's Qwen3.5 / Qwen3.8 oracle uses). The upstream
> contract is now spelled out in "The upstream contract" below, the survey
> of models that actually ship these heads is in "Which models ship them",
> and the status of the three pieces this repository already has is in
> "What this repository already has". Nothing here is implemented yet:
> the whole feature is tracked as TODO-069.

## Why

Standard autoregressive decode is "1 token in, 1 token out" — every step
runs a full forward over the entire KV cache. On Qwen3-0.6B Q8_0 we measure
~30 t/s decode, so each token is ~33 ms of forward.

Speculative decoding proposes K future tokens cheaply, then **verifies all K
in one batched main-model forward**. If acceptance rate is α and K is large
enough, the effective throughput is `1 / ((1 - α) × cost_main_step + α × cost_main_step / K)`
≈ `1 / ((1 + (K - 1) × α)) × cost_main_step`. With K = 5 and α = 0.7, that's
~4× speedup.

The four practical draft strategies are **orthogonal**, not mutually
exclusive — any one can be combined with any other:

| Strategy | Draft input | Draft output | Model architecture |
|---|---|---|---|
| **MTP** (Multi-Token Prediction, DeepSeek-V3 style) | 1 token | N future logits | main model + N MTP heads (`nextn.eh_proj` + `nextn.hnorm` + `nextn.enorm`) |
| **DFlash** (Qwen3 style) | 1 token + main hidden state at `target_layers` | K candidate tokens | main model + `dflash_attn_conv_*` + `dflash_selector_*` |
| **DSpark** (DFlash + Markov head) | same as DFlash | K candidate tokens (better calibrated) | DFlash + `dsv4_hc_mult` + `dspark_markov_rank` |
| **EAGLE3** | 1 token + main hidden state at `target_layers` (concat multi-layer) | K candidate tokens | small auxiliary EAGLE3 model (`ctx_dft`) |
| **Standalone draft** | 1 token | K candidate tokens | a separate small LLM (`ctx_dft`) |

A trunk can ship **any combination**: e.g. MTP heads for top-1 multi-step
prediction AND DSpark for top-K draft proposal.

## Orthogonality & combinations

The user's framing is precise — MTP and DSpark are two separate axes:

```
                draft step                verify step
                ──────────                ───────────
MTP             In: append 1 token        Out: N candidate logits
                Out: N logits             Accept: keep all α-matched
                (single forward)          (single batched forward)

DFlash/DSpark   In: append 1 token        Out: K candidate tokens
                Out: K candidates         Accept: keep first m (m ≤ K)
                (main forward + draft)    (single batched forward)

Composition     MTP heads generate N logits → pick top-K candidates as
                DFlash's "drafted K tokens" → batched verify (append K) →
                accept m of them. Draft side and verify side are
                decoupled, both share the same verification machinery.
```

**Verify-stage structure is identical for all four strategies**:

```
Input:  prefix (P tokens) + K draft tokens = P + K tokens in one sequence
Output: P + K positions of main-model logits (one forward)
Per position i ∈ [0, K): compare main_logits[P + i] vs draft_token[i] →
  accept (extend KV, emit) or reject (rollback KV, resample, restart verify).
```

## Foundation we already have (2026-09-23)

From `src/core/prefill.rs::ChunkedPrefill` + per-trunk `forward_chunk`:

- ✅ **Batched prefill**: `ChunkedPrefill::prefill(input, batch_size)` walks
  `prefill_chunks` and calls `forward_chunk(input, rows, base, is_last)`
  per chunk. B > 1 → chunks of `rows` tokens share FFN/QKV/attention compute.
- ✅ **`prefill_batch_size` CLI flag**: wired into every trunk's forward
  entry; default 64 (clamped at 0).
- ✅ **`is_last` flag**: `forward_chunk` knows when it's the final chunk,
  can opt into projecting logits. The default `ChunkedPrefill::prefill`
  only projects on the final chunk + last row — exactly the verify-stage
  pattern.
- ✅ **Per-row SIMD kernels** (`src/ops/kernel/{q4_0,q4_k,q8_0,neon_k,...}*`):
  handle `[rows, n_in] × [n_in, n_out]` shaped matmuls with `rows > 1`.
  Q4_K measured at ~76 t/s with rows=64 vs ~12 t/s with rows=1.
- ✅ **Single-shot `forward_logits(token_ids)`**: returns one vector of
  vocab-sized logits. JevScorer already uses this for batched scoring.
- ✅ **`run_jev_grouped_decision` (MultiSelect / BlockChoice)**:
  demonstrates "K independent evaluations in one forward via per-group
  softmax" — structurally what verify-stage needs (K candidates, one
  forward, per-candidate verdict).
- ✅ **Qwen3 `deepstack_embeddings` plumbing** (`src/models/qwen3/trunk/`):
  a placeholder hook for "intermediate layer hidden state injection" —
  currently all `None`, but the type/contract exists.
- ✅ **`prefill.rs` doc comment mentions DSpark**: "qwen3 captures
  per-layer intermediates for DSpark on the final chunk" — this commit
  documents the intent.

## What's missing

Grouped by dependency order:

### Phase 0: verify-stage plumbing (foundation, no draft strategy yet)

- ❌ **`forward_logits` returns last-layer hidden state too**, not just
  vocab logits. The draft heads need `n_embd × n_tokens` at specific layers.
  Easiest shape: extend the return type to
  `Result<(Vec<f32>, HiddenStates), String>` where `HiddenStates` is
  `Vec<Vec<f32>>` (per-layer) for the last chunk only.
- ❌ **`llama_set_embeddings_layer_inp`-style hook**: a trunk-side knob
  "expose hidden state at layer L" that turns on capture without breaking
  the default last-logits-only path. Three forms:
  - `last_layer_only` (default; current behavior)
  - `layer_set: &[usize]` (capture multiple layers; needed by EAGLE3/DFlash)
  - `last_layer + extra_hidden` (for MTP, which needs the full n_embd grid)
- ❌ **Multi-sequence KV cache** (only needed if we go beyond simple
  padding). Current `KvCache::new_f32/f16` is single-sequence per row.
  For verify-stage, we either (a) pad K candidates to a single P+K sequence
  with row-per-candidate (simple but wastes compute on rejected branches)
  or (b) maintain K parallel sequences with shared prefix KV. (a) is
  sufficient for an initial implementation.
- ❌ **Verify-stage `forward_logits_with_drafts(prefix_logits, draft)`**:
  run main forward on `prefix ∪ draft`, return last P+K positions of
  logits. Standard `forward_logits(token_ids = prefix + draft)` already
  does this — we just need to keep the per-position logits (one row of
  `[vocab]` per token) instead of only the final row.

### Phase 1: a single draft strategy (pick one to ship first)

Easiest-first ranking by cost:

| Strategy | Cost | Why |
|---|---|---|
| NGRAM (3-level cache, no model) | Lowest | No model changes; pure algorithm; good baseline measurement |
| MTP | Medium | Needs `last_layer_hidden` + MTP heads in one trunk (DeepSeek-V3) |
| DFlash | Medium-high | Needs `target_layer_hidden` + DFlash weights + conv+selector in one trunk (Qwen3) |
| EAGLE3 | Highest | Needs `target_layer_hidden` + a separate small EAGLE3 model + multi-layer concat |

Recommendation: ship **NGRAM first** as a baseline to measure verify-stage
speedup honestly. Then **MTP** on whichever trunk gets MTP support first
(DeepSeek-V3, Step3-5). Then **DFlash** on Qwen3 if/when GGUF metadata
includes DFlash weights.

### Phase 2: speculative main loop

After draft + verify plumbing exists, write:

```
loop:
    # 1. Draft K tokens (strategy-specific)
    draft_tokens = draft_strategy.propose(model, last_logits, last_hidden, prefix)
    
    # 2. Verify K tokens in one batched forward
    logits = model.forward_logits(prefix ++ draft_tokens)  # P + K positions
    
    # 3. Acceptance test (per position, independent Gumbel-max)
    n_accepted = 0
    for i in 0..K:
        main_p = softmax(logits[P + i])
        d = draft_tokens[i]
        r = uniform(0, 1)
        if r < min(1, main_p[d] / draft_p[d]):
            accept(d); n_accepted += 1
        else:
            resample from main_p; break
    
    # 4. Extend KV cache with accepted tokens; emit them
    model.kv_extend(n_accepted + 1)  # +1 for the resampled one
    emit(draft_tokens[..n_accepted] + [resampled])
```

`draft_p[d]` is the draft model's predicted probability for `d` (what it
"would have sampled"). For NGRAM this is a lookup; for MTP it's the MTP
logits at position i; for DFlash/DSpark it's the dflash_selector output;
for EAGLE3 it's the small EAGLE3 model's logits; for standalone draft
it's the small draft model's logits.

## Concrete entry points in our codebase

The work fits in three layers:

### 1. `src/core/prefill.rs` — extend `ChunkedPrefill`

```rust
pub trait ChunkedPrefill {
    type Input;
    fn input_len(input: &Self::Input) -> usize;

    fn forward_chunk(
        &mut self,
        input: &Self::Input,
        rows: usize,
        base_position: usize,
        project_logits: bool,
        capture_hidden: Option<&[usize]>,  // NEW: layers to capture
    ) -> Result<Option<(Vec<f32>, Vec<Vec<f32>>)>, String>;  // NEW: (logits, per-layer h)

    // ... existing methods unchanged
}
```

The default `prefill` loop needs only minor changes: pass `is_last` →
`project_logits = is_last && capture_hidden.is_some()` for the final chunk
only. Per-trunk implementations override `forward_chunk` to honor
`capture_hidden`.

### 2. `src/models/<trunk>/trunk/forward.rs` — capture hooks

```rust
// Example: qwen3 trunk (DFlash target)
fn forward_chunk(
    &mut self,
    input: &Qwen3Input,
    rows: usize,
    base_position: usize,
    project_logits: bool,
    capture_hidden: Option<&[usize]>,
) -> Result<Option<(Vec<f32>, Vec<Vec<f32>>)>, String> {
    // ... existing per-row forward, accumulating hidden states per layer
    let captured_layers: Vec<Vec<f32>> = if let Some(layers) = capture_hidden {
        layers.iter().map(|&l| self.layer_hidden[l].clone()).collect()
    } else {
        Vec::new()
    };
    // ...
}
```

Each trunk needs to allocate per-row hidden state buffers for the layers
in `capture_hidden`. For Qwen3 the layers are exposed by
`target_layer_ids` (DFlash GGUF metadata). For DeepSeek-V3 MTP, only the
last layer's hidden state is needed (passed as `n_embd × n_tokens` to the
MTP module).

### 3. `src/app/text.rs` — speculative main loop

After Phase 0/1 land, expose `--spec-strategy {ngram,mtp,dflash,eagle3,draft}`
and `--spec-k N`. Hook into `run_inference`'s decode loop. The draft stage
is strategy-specific; the verify stage is shared (`forward_logits` +
`accept_test` + `kv_extend`).

## Why verify-stage IS batch inference

The user is correct. Verify-stage does this:

```
token_ids = prefix ++ draft_tokens   // P + K tokens
logits = forward_logits(token_ids)   // single batched forward
```

This is exactly what `ChunkedPrefill::prefill` already does for batch_size
≥ P + K. We don't need new infrastructure for the forward itself — we
need:

1. The return value to carry **all P + K positions of logits**, not just
   the final one. Trivially: change return type from `Vec<f32>` to
   `Vec<Vec<f32>>` (rows of vocab-sized logits). Or keep `Vec<f32>` for
   the last position and add a `Vec<Vec<f32>>` for verify-stage use.
2. The KV cache to accept `seq_len += P + K` atomically (it already does
   `seq_len += N` per call; we'd call it once with `N = P + K`).

**The "batch inference" foundation (`prefill_batch_size`, per-row matmul,
`forward_chunk` B > 1) covers the verify stage already.** What's missing
is the draft side and the acceptance loop.

## Open questions

1. **Multi-sequence vs padding**: For verify-stage, do we run K parallel
   candidate sequences (each P + K_i tokens long, K_i decreases as
   branches reject), or do we pad to the longest and run as one batch?
   Padding is simpler; multi-seq is more efficient but needs new KV-cache
   semantics. Start with padding.

2. **Tree-shaped candidates**: EAGLE3 / Medusa produce a tree of
   candidate tokens (each position branches). This needs a tree-aware
   attention mask. Out of scope for an initial implementation.

3. **Per-trunk vs shared capture**: MTP needs only the last layer; DFlash
   needs a configurable set. We could either (a) make capture uniform
   (every layer kept, caller filters) or (b) make capture a set of layer
   indices. (b) is cheaper memory-wise but adds a per-trunk knob.

4. **NGRAM draft quality**: 3-level n-gram cache works surprisingly well
   for chat-style prompts but poorly for code / math. Worth measuring
   before investing in model-based drafts.

5. **Speculative + multi-modal**: For VL models with vision tokens, the
   draft still operates on text positions only (vision prefix is fixed).
   Should work without changes once verify-stage is in place.

## Phasing recommendation

```
Phase 0  (foundation, ~1-2 weeks)
  - Extend ChunkedPrefill::forward_chunk return type
  - Per-trunk: capture_hidden parameter (qwen3 / deepseek4 first)
  - Verify-stage forward_logits returns P+K positions of logits
  - Test: logit-by-logit parity vs existing forward_logits (no draft)

Phase 1a (NGRAM, ~3-5 days)
  - 3-level n-gram cache (no model changes)
  - Accept/reject loop
  - Measure acceptance rate + speedup on chat benchmark
  - Decision: is model-based draft worth the cost?

Phase 1b (MTP, ~1-2 weeks)
  - DeepSeek-V4 / Step3.5 GGUF support
  - MTP module: eh_proj + hnorm + enorm
  - Connect: main forward hidden → MTP module → 1 extra token
  - Measure vs NGRAM

Phase 1c (DFlash, ~2-3 weeks)
  - Qwen3 DFlash weights: target_layers, attn_conv, selector
  - DeepSeek-V4 DSV4 backbone (if we want DSpark too)
  - Connect: target_layer_hidden → conv → selector → K candidates
  - Measure vs NGRAM/MTP

Phase 1d (EAGLE3 / standalone draft, ~3-4 weeks)
  - Separate small model loading (EAGLE3 weights or any small LLM)
  - Multi-model context (draft ctx + target ctx)
  - KV cache sharing across ctx_dft / ctx_tgt (if applicable)
  - Measure vs all earlier strategies

Phase 2  (acceptance / verification hardening)
  - Gumbel-max acceptance (vs naive argmax)
  - Per-position temperature handling
  - Speculative + tool-call integration
  - Speculative + multi-sequence (n_parallel > 1)
```

## The upstream contract (llama.cpp `b96806d`)

Upstream ships **four** draft types side by side and can infer which one a
GGUF is from its metadata alone (`common/speculative.cpp::common_speculative_types_from_gguf`):

| type | auto-detected when | the draft is |
|---|---|---|
| `draft-mtp` | arch is not `dflash` **and** `blk.{block_count-1}.nextn.eh_proj.weight` exists | an extra MTP block that lives inside the main model (DeepSeek-V3 / V4 style) |
| `draft-dflash` | arch is `dflash` **and** `markov_w1.weight` does not exist | a separate small DFlash model |
| `draft-dspark` | arch is `dflash` **and** `markov_w1.weight` exists | DFlash plus a Markov head (and an optional confidence head) |
| `draft-eagle3` | a separate small EAGLE3 model | |

### MTP (`src/models/deepseek4.cpp::graph_mtp`)

The MTP block takes two inputs: the **main model's last-layer hidden state**
`h` and the **token it has to predict**. Then:

```
tok_embd = embed_tokens(tokens)                 # falls back to the main token_embd
h_norm   = RMSNorm(h,     nextn.hnorm)
e_norm   = RMSNorm(tok_embd, nextn.enorm)
inpL     = eh_proj(concat(e_norm, h_norm))      # 2*n_embd -> n_embd
...                                             # then it is an ordinary layer
logits   = shared_head_head(shared_head_norm(out))   # falls back to output_norm + output
```

Tensor names (`src/llama-arch.cpp`): `blk.%d.nextn.eh_proj`, `.embed_tokens`,
`.enorm`, `.hnorm`, `.shared_head_head`, `.shared_head_norm`. Metadata:
`<arch>.nextn_predict_layers` (how many MTP layers) and `<arch>.block_count`
(the total, MTP included).

Upstream also supports a **standalone GGUF that contains only the MTP block**
(`mtp_only`: no `blk.0.attn_norm.weight`) -- that is exactly the kind of file
that ships in an `MTP/` directory.

### DSpark (`src/models/dflash.cpp`)

DFlash is a small model that reads the main model's hidden state at a set of
layers (`target_layers`) and runs conv + selector stages:

- tensors: `blk.%d.attn_conv_base`, `.attn_conv_proj`, `.ffn_conv_base`,
  `.ffn_conv_proj`, `selector_predecessor`, `selector_successor`,
  `selector_hidden`
- metadata: `<arch>.target_layers`, `.block_size`, `.conv_kernel_size`,
  `.conv_group_size`, `.selector_rank`, `.selector_top_k`

**DSpark = DFlash + a Markov head (+ an optional confidence head)**, and both
heads are corrections *on top of* the draft logits:

```
# Markov head: bias the draft logits at position i by the previously committed token
w1_prev   = markov_w1[prev_token]                # [R]
bias      = markov_w2 @ w1_prev                   # [n_vocab]
logits[i] = base_logits[i] + bias                # chained: position i-1's argmax is position i's prev

# confidence head: predict the acceptance probability of each position
conf[i]   = sigmoid(conf_proj . [embd[i]; markov_w1[prev(i)]] + conf_proj_bias)
```

Tensors: `markov_w1` [R, n_vocab], `markov_w2` [R, n_vocab_draft],
`markov_w2.scale`, `conf_proj.weight` [n_embd+R, 1], `conf_proj.bias` [1].
Metadata: `dflash.block_size`, `dflash.sample_from_anchor` (slot 0 is a bonus
anchor, not a prediction slot).

### The sidecar convention

Upstream's downloader looks for `MTP/`, `dflash/`, `dspark/` and `eagle3/`
sub-directories inside a draft repo (`common/arg.cpp`), with `dspark` ranking
above `dflash` because it carries the extra Markov head. The weights
publishers follow that convention: `unsloth/Qwen3.8-Flash-Next-GGUF` ships an
`MTP/` directory and `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` ships a
`dspark/` one.

## What this repository already has

- **`xing4_0` already parses an MTP block.** `src/models/xing4_0/trunk/weights.rs`
  loads `nextn_eh_proj`, `nextn_embed_tokens`, `nextn_enorm`, `nextn_hnorm`,
  `nextn_shared_head_head` and `nextn_shared_head_norm` -- byte for byte the
  same six tensors as the upstream `blk.N.nextn.*` contract above (the
  GLM-4.5 shape). They are loaded and then **not used**: no forward code
  reads them.
- **`qwen35` knows about MTP layers and excludes them.** `Qwen35Config` has
  `n_nextn` and computes `n_layer_impl = n_layer - n_nextn`, so a Qwen3.5 /
  Qwen3.8 GGUF that ships MTP blocks runs its trunk without them
  (`SUPPORTED_MODELS.md` says "excludes MTP" for Sage-A1.5 and Holo4-27B).
- **The verify-stage forward already exists.** `ChunkedPrefill::prefill` walks
  chunks of `rows` tokens and `forward_chunk` knows `is_last`; a
  `prefix ∪ draft` sequence of `P + K` tokens is one batched prefill, which is
  what the verify stage needs. The missing part is only that the return value
  keeps **all** `P + K` rows of logits instead of the last one.
- **The acceptance test has a shape to copy.** `run_jev_grouped_decision`
  already computes "K independent evaluations in one forward via per-group
  softmax".

What is missing is the draft side and the loop. See TODO-069.

## Which models ship them

Survey of the 354 `-GGUF` repositories under the `unsloth` ModelScope owner
(names as of 2026-11-11). The interesting ones are the repositories that
carry a **sidecar** directory, because that is how the heads are shipped:

| repository (ModelScope, owner `unsloth`) | sidecar | size | what the sidecar is |
|---|---|---|---|
| `gemma-4-E2B-it-GGUF`, `gemma-4-E4B-it-GGUF`, `gemma-4-12b-it-GGUF`, `gemma-4-26B-A4B-it-GGUF`, `gemma-4-31B-it-GGUF` (plus the `-qat` and `-qat-mobile` variants) | `MTP/` | 0.06-0.95 GB | arch `gemma4-assistant`, `nextn_predict_layers = 4`, and a different MTP shape: `nextn.pre_projection` / `nextn.post_projection` instead of `eh_proj` |
| `Qwen3.8-27B-GGUF` | `MTP/` | 1.37 GB (Q4_0) | arch `qwen35`, `nextn_predict_layers = 1`, 18 tensors: `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}` plus the main model's `token_embd` / `output` |
| `Qwen3.8-Flash-Next-GGUF` | `MTP/` | 2.79-7.77 GB | arch `qwen4exp`, `nextn_predict_layers = 1`, and a hyper-connection head: `blk.48.nextn.hc_head_{up,down,norm}` plus `blk.48.hc_attn_*` |
| `DeepSeek-V4-Flash-0731-GGUF` | `dspark/` | 11.31 GB (BF16) | arch `dflash`, 3 DeepSeek-V4 blocks, 128 experts top-3 + 1 shared, `markov_w1` / `markov_w2` / `conf_proj` |
| `DeepSeek-V4.1-Flash-1017-GGUF` | `dspark/` | 8.51 GB (BF16) | same shape; `dflash.target_layers = [37, 38, 39]`, `dflash.block_size = 5`, `dflash.has_confidence_head = true` |
| `Mistral-Large-4-1T-A52B-GGUF` | `dspark/` | 4.21 GB (BF16) | arch `dflash` |
| `MiniMax-M3.1-Flash-GGUF` | root-level `dspark-*.gguf` | 2.09 GB (Q8_0) | arch `dflash` |

Two things worth drawing out of that table:

- **The MTP head is often inside the main model, not only in a sidecar.**
  The Qwen3.8-27B GGUF itself carries `qwen35.nextn_predict_layers = 1` and the
  four `blk.64.nextn.*` tensors, while its main stack is blocks 0-63. This
  repository's `qwen35` trunk already knows that split -- `n_layer_impl` is
  `n_layer - n_nextn` -- so **the Qwen3.8 MTP head is currently ignored
  silently**: the weights are in the file, the config knows about them, and
  no forward pass reads them. The same is true of `xing4_0`, which parses the
  full set of six `nextn` tensors and never uses them.
- **Gemma-4 is the cheapest first target.** This repository already has the
  `gemma4` trunk, and the MTP sidecars are 0.06-0.95 GB, so the whole
  experiment (head + verify loop + acceptance measurement) fits in a
  gigabyte of weights. The Gemma-4 MTP is also a *different shape* from
  DeepSeek's (`pre_projection` / `post_projection` instead of `eh_proj`), so
  it is worth reading the upstream Gemma-4 draft code before designing the
  generic MTP path.

## What to build first

1. **MTP, not DSpark.** MTP needs the main model's last-layer hidden state
   and one block; it needs no second model and no new architecture, and
   `xing4_0` has already parsed its weights. DSpark needs a whole second
   model (a DFlash backbone), a Markov chain and a confidence head.
2. **The verify side first, and measure it.** Extend `forward_chunk` to
   return every row of logits, then measure how much a `P + K` batched
   prefill actually saves over `K` single-token steps. If the answer is
   small, no draft strategy can pay for itself -- that is the same order of
   reasoning the "NGRAM first" recommendation in this document already makes.
3. **The hidden state has to be trustworthy.** A draft head consumes the main
   model's last-layer hidden state, so it inherits that model's error. The
   existing bitwise/parity infrastructure (`RMI_PARITY_TRACE`, the
   `tools/oracle/*` harnesses) is what makes that checkable -- a draft head
   built on an unverified hidden state is worse than no draft head.

## See also

- `docs/develop/PREFILL_ABSTRACTION.md` — the existing `ChunkedPrefill`
  trait this proposal extends.
- `docs/develop/TODO.md` JEV section — similar trait refactor for
  scoring-style forward; speculative will follow the same architectural
  pattern (small scorer struct + trunk free function).
- `references/llama.cpp/common/speculative.cpp` — the reference main loop
  this design mirrors.
- `references/llama.cpp/src/models/dflash.cpp` — DFlash backbone +
  DSpark extension; shows what weights / GGUF metadata to expect.
- `references/llama.cpp/src/models/deepseek4.cpp::graph_mtp` (line 1363) —
  shows how MTP module consumes main forward's hidden state.