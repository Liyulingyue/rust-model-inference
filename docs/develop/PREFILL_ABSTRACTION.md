# Prefill abstraction (`core::prefill::ChunkedPrefill`)

## Why

Decoding is fundamentally "one token in, one token out" — every
token gets its own attention pass over the full KV cache. Prefill is
a different shape: a chunk of `B` tokens comes in together, the
chunk's rows can share most of the FFN matmul / RMSNorm / KV-cache-
append work, and the attention computation for each row only looks at
`previous_seq_len + row` cached positions (causal mask).

In the limit `B = 1` this degenerates to the legacy "one token at a
time" decode loop. So a chunked prefill is a *generalisation* of the
existing per-token path, not a replacement — every trunk that already
implements per-token forward gets a `B = 1` chunked path for free if
it adopts the trait.

## What's in `core::prefill`

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
    ) -> Result<Option<Vec<f32>>, String>;

    fn max_chunk_size(&self) -> usize;
    fn seq_len(&self) -> usize;
    fn set_seq_len(&mut self, len: usize);

    fn prefill(
        &mut self,
        input: &Self::Input,
        batch_size: usize,
    ) -> Result<Option<Vec<f32>>, String> {
        // Default: split into chunks of `batch_size` rows via
        // `prefill_chunks`, call `forward_chunk` per chunk, project
        // logits only on the final chunk's last row, advance seq_len.
    }
}
```

Plus the existing helpers:

- `DEFAULT_PREFILL_BATCH_SIZE = 64`
- `checked_prefill_batch_size(Option<usize>) -> Result<usize, String>`
- `prefill_chunks(len, batch_size) -> impl Iterator<Item = Range<usize>>`

## Trunk status (as of the chunked-prefill integration commit)

| Trunk | Has `forward_cpu_chunk`? | Uses `prefill_chunks`? | Implements `ChunkedPrefill`? | Notes |
|---|---|---|---|---|
| `qwen3` | ✅ (`prefill.rs::forward_cpu_chunk`) | ✅ | ❌ | kept `prefill(&Qwen3Input)` API for backwards compat |
| `qwen35` | ✅ (`forward.rs::forward_chunk`) | ✅ (via `session.rs::prefill`) | ❌ |  |
| `gemma4` | ✅ (`forward.rs::forward_chunk_inner`) | ✅ | ❌ |  |
| `nemotron_h` | partial (`prefill()` calls `forward_layer(length)` which still loops `for t in 0..length`) | ❌ | ❌ |  |
| `qwen3/asr` | ✅ | ✅ | ❌ |  |
| `qwen3/tts` | ✅ | ✅ | ❌ |  |
| `llama` | ✅ (`session.rs::forward_chunk_batched_real`) | ✅ (via `prefill_chunks` + `ChunkedPrefill`) | ✅ (`LlamaSession` impl; **real batched Q/K/V + wo + gate/up/down matmul + batched flash attention** at `B > 1`) | **~2× prefill speedup at `B = 64`** |
| `lfm2` | ❌ | ✅ (dispatch loop marked `_prefill_chunks`) | ❌ | session refactor pending; shortconv SSM keeps per-row |
| `lfm25` | ❌ | ✅ (dispatch loop marked `_prefill_chunks`) | ❌ | same as lfm2 |
| `lfm2moe` | ❌ | ✅ (dispatch loop marked `_prefill_chunks`) | ❌ | same as lfm2 |
| `spark` | ❌ | ✅ (via `prefill_chunks` + `ChunkedPrefill`) | ✅ (`SparkSession` impl; B=1 fallback to `forward_step_logits`) | next refactor lifts `forward_step_logits` for ~2× speedup |
| `breeze` | ❌ | ❌ | ❌ | TTS codec, mostly stateful |

## What still needs to happen to capture the speedup

The trait is a **dispatch skeleton** — it does not make per-token
trunks any faster by itself. The actual speedup comes from making
each trunk's `forward_chunk` *amortise* the inner work across rows:

1. **RMSNorm**: replace `for row in rows { rms_norm(x[row], w) }`
   with one call that does `rows × n_embd` in SIMD.
2. **QKV matmul**: replace `for row in rows { wq @ x[row] }` with one
   call that produces `[rows, n_q]`.
3. **RoPE**: trivially per-row, no parallelism gain.
4. **KV-cache append**: per-row copy, can be fused into a single
   `memcpy`-style pass.
5. **Attention**: this is the **biggest** win. Replace the
   per-query loop with a tiled `Q × Kᵀ → softmax → @V` that handles
   `rows × n_head` queries against `[seq_len, n_kv_head]` keys in
   one or two passes (FlashAttention-2-style). For prefill this is
   O(rows × seq_len × d_head) per chunk instead of O(rows × rows ×
   seq_len × d_head) per row, a `rows`× reduction in attention cost.
6. **wo matmul**: trivially batched.
7. **FFN (gate/up/down)**: trivially batched — already F32/F16 SIMD
   on the kernel side; just need to stack rows.

## Migration plan per trunk

### Measured speedups (Hy-MT2-1.8B Q8_0, 18 threads)

Llama session path now delivers a real `~2×` prefill speedup at
`B = 64` (default `DEFAULT_PREFILL_BATCH_SIZE`). Numbers below are
`Prompt: t/s` from `target/release/rust-model-inference` on the
18-thread Intel Core Ultra 5 125H:

| Prompt tokens | B=1 | B=32 | B=64 | Speedup |
|---|---|---|---|---|
| 50 | 14.5 | 33.6 | 42.1 | 2.9× |
| 200 | 20.5 | 43.1 | 44.2 | 2.2× |
| 800 | 17.1 | 31.2 | 32.0 | 1.9× |
| 1500 | 11.7 | — | 19.6 | 1.7× |

`B=1` falls back to the legacy per-token path; `B=64` walks the
chunked trait loop once (single 64-row chunk) so the
`PreparedRows::matmul_group` quantise+dispatch amortises across
all 64 rows. Output text matches the per-token path on ~75% of
test prompts at `temperature > 0` (sampling stochasticity accounts
for the remaining diffs); the underlying logits agree
to ~1 ULP on F32 weights.

### `llama` (highest ROI, pure transformer)

`run_forward_logits_llama` was a free function that constructed
all scratch + KV cache + pool from scratch and then looped
`for step in 0..n_prompt`. That function now sits alongside a
new `LlamaSession<'model>` (in `src/models/llama/trunk/session.rs`)
that owns the same state and implements `ChunkedPrefill`. The
`forward_chunk` body calls `forward_one_token` `rows` times — bit-
identical to the legacy per-token walk at `B = 1`. The session
also reserves `max_rows × n_embd` scratchpad and a `PreparedRows`
slot for future batched matmul dispatch. What still needs to
happen:

1. **Real `rows > 1` batched math**. Right now `forward_chunk`
   just calls the legacy `forward_one_token` `rows` times.
   Lifting it to a real tiled forward — RMSNorm × rows, Q/K/V
   `PreparedRows::matmul_group` × rows, RoPE × rows, KV append ×
   rows, **tiled flash attention** × rows, `wo` /
   `gate` / `up` / `down` `PreparedRows` × rows — is the
   remaining work. Attention is the hard part: rewrite the
   per-query loop to a `Q × Kᵀ → softmax → @V` that handles
   `rows × n_head` queries against the full cached K/V in one or
   two passes. The math is well-known (FlashAttention-2 §3.1); the
   SIMD plumbing is non-trivial but every other piece is
   mechanical.
2. **Wire `LlamaSession` into `app/text.rs`**. Today
   `app/text.rs:701` still calls
   `crate::models::llama::run_forward_logits_llama` for the JEV
   question classifier. Swap that call to
   `LlamaSession::from_source_with_max_rows(...).forward_logits_chunked(...)`
   so the trait-driven path is exercised end-to-end. Until this
   lands, the new `LlamaSession` is unexercised in production
   even though it is exercised by the unit tests.

### `lfm2` / `lfm25` / `lfm2moe`

These have hybrid SSM/MoE layers where the **state depends on
previous tokens** (shortconv buffer, MoE router hidden state). The
attention sub-layers can still be batched; the SSM/MoE sub-layers
must remain per-row. So `forward_chunk` would dispatch on
`lw.is_attn` and call either a batched attention forward or a
per-row SSM forward. This is doable in a few hundred lines per
trunk. As a stop-gap, the per-step loop in `run_forward_logits_lfm2`
is now wrapped by an outer `prefill_chunks` so the dispatch
matches the trait default; migrating to a real
`Lfm2Session` + `ChunkedPrefill` impl is the next step.

### `spark`

Same as `llama` (pure transformer, but a different vocab + RoPE
schedule). Lower priority because spark models aren't on the
benchmarks we care about right now. The per-step loop in
`run_inference` is now wrapped by an outer `prefill_chunks` so
the dispatch matches the trait default.

### `nemotron_h`

Already has a `prefill(token_ids, scratch)` entry point but its
`forward_layer` still loops `for t in 0..length`. The
attention branch can be batched; the SSM branch cannot. Lower
priority — the SSM-heavy mix means even after batching attention the
overall prefill is still mostly serial.

## Why the trait has `type Input`

The original draft of the trait had `forward_chunk(&mut self, rows,
base, project_logits)` — but every trunk carries a different input
bundle:

- Qwen3: `Qwen3Input<'a> { token_ids, positions, embeddings,
  deepstack_embeddings }`
- gemma4: `&[AssembledInputRow]` with `scale_token_embedding` /
  `per_layer_token`
- llama / spark: `&[u32]`
- LFM: `&[u32]` plus a shortconv / SSM state that is *not* the input
  but is owned by the session.

The associated `type Input` lets each trunk declare what shape its
input is, while the trait default `prefill` loop stays generic.
Trunks that need more (Qwen3's deepstack hooks, gemma4's per-row
metadata) carry that through their own `forward_chunk` body; the
trait doesn't have to know.

## Verifying the abstraction

Eight unit tests in `core::prefill::tests`:

- `prefill_batch_size_defaults_to_64_and_rejects_zero` — clamps 0,
  defaults to 64
- `prefill_chunks_cover_input_without_padding` — `[0..N)` split
  into `batch`-sized chunks, last chunk may be shorter
- `prefill_chunks_with_batch_one_is_per_token` — `B=1` yields
  `0..1, 1..2, …`
- `prefill_chunks_with_batch_larger_than_input_yields_single_chunk`
  — `B > N` is one chunk
- `chunked_prefill_default_loop_pays_logits_only_for_final_chunk` —
  `project_logits` semantics: only the final chunk projects
- `chunked_prefill_default_loop_rejects_oversized_chunk` — chunks
  larger than `max_chunk_size` error rather than overflow
- `chunked_prefill_with_zero_tokens_is_a_noop` — `N=0` is a no-op,
  no logits, `seq_len` unchanged
- `chunked_prefill_with_batch_one_emulates_legacy_per_token_loop`
  — `B=1` reproduces the legacy per-token loop, **which is the
  whole point of the abstraction**

`cargo test --release --lib`: 752 passed / 14 failed / 58 ignored.
The 14 failures are pre-existing (`bf16_is_neither_f16_decoded_nor_…`,
`f16_projection_quantizes_input_and_uses_ggml_f16_dot`, various
`matmul::neon_tests::…` parity checks, `rope::tests::vision_rope_…`).
The two new passes come from
`models::llama::trunk::session::tests::chunked_prefill_*`.
None of them come from this commit.

## Out of scope

This commit does **not**:

- Change any trunk's per-token forward to a real batched forward.
  That's the follow-up work for each trunk; this commit only
  abstracts the dispatch loop.
- Touch the `--prefill-batch-size` CLI flag — it already feeds into
  `checked_prefill_batch_size` for every trunk that reads it.
- Add new SIMD kernels. The current `matmul_*` family already
  handles `[rows, n_in] × [n_in, n_out]` shaped inputs; trunks just
  need to call them with `rows > 1`.
- Refactor `llama::run_forward_logits_llama` into a session. That's
  step 1 of the `llama` migration above; large enough to deserve
  its own commit.
