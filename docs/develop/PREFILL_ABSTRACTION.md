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

## Trunk status (as of the commit that introduces the trait)

| Trunk | Has `forward_cpu_chunk`? | Uses `prefill_chunks`? | Implements `ChunkedPrefill`? |
|---|---|---|---|
| `qwen3` | ✅ (`prefill.rs::forward_cpu_chunk`) | ✅ | ❌ (kept `prefill(&Qwen3Input)` API for backwards compat) |
| `qwen35` | ✅ (`forward.rs::forward_chunk`) | ✅ (via `session.rs::prefill`) | ❌ |
| `gemma4` | ✅ (`forward.rs::forward_chunk_inner`) | ✅ | ❌ |
| `nemotron_h` | partial (`prefill()` calls `forward_layer(length)` which still loops `for t in 0..length`) | ❌ | ❌ |
| `qwen3/asr` | ✅ | ✅ | ❌ |
| `qwen3/tts` | ✅ | ✅ | ❌ |
| `llama` | ❌ (`run_forward_logits_llama` does `for step in 0..prompt_tokens.len()`) | ❌ | ❌ |
| `lfm2` | ❌ (`for step in 0..n_prompt`) | ❌ | ❌ |
| `lfm25` | ❌ (`for step in 0..n_prompt`) | ❌ | ❌ |
| `lfm2moe` | ❌ (`for step in 0..n_prompt`) | ❌ | ❌ |
| `spark` | ❌ (`for (pos, &tok) in prompt_tokens.iter().enumerate()`) | ❌ | ❌ |
| `breeze` | ❌ | ❌ | ❌ |

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

### `llama` (highest ROI, pure transformer)

Llama's `run_forward_logits_llama` is a **free function** that
constructs all scratch + KV cache + pool from scratch and then
loops `for step in 0..n_prompt`. To adopt the trait it needs:

1. Refactor `run_forward_logits_llama` into a `LlamaSession<'model>`
   with `weights`, `kv_cache`, `scratch`, `pool` fields. This is a
   moderate change because the current code weaves construction and
   use in one place.
2. Implement `ChunkedPrefill` with `type Input = &[u32]` and
   `forward_chunk` that does the same RMSNorm + Q/K/V + RoPE + KV
   append + attention + wo + FFN as today, **but on a chunk of
   rows**.
3. Attention is the hard part: rewrite the flash-attention loop to
   accept `rows` queries at once. The math is well-known
   (FlashAttention-2 §3.1), but the SIMD plumbing is non-trivial.

### `lfm2` / `lfm25` / `lfm2moe`

These have hybrid SSM/MoE layers where the **state depends on
previous tokens** (shortconv buffer, MoE router hidden state). The
attention sub-layers can still be batched; the SSM/MoE sub-layers
must remain per-row. So `forward_chunk` would dispatch on
`lw.is_attn` and call either a batched attention forward or a
per-row SSM forward. This is doable in a few hundred lines per
trunk.

### `spark`

Same as `llama` (pure transformer, but a different vocab + RoPE
schedule). Lower priority because spark models aren't on the
benchmarks we care about right now.

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

`cargo test --release --lib`: 750 passed / 14 failed / 58 ignored.
The 14 failures are pre-existing (`bf16_is_neither_f16_decoded_nor_…`,
`f16_projection_quantizes_input_and_uses_ggml_f16_dot`, various
`matmul::neon_tests::…` parity checks, `rope::tests::vision_rope_…`).
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
