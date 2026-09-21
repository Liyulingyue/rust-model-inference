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
| `gemma4` | ✅ (`forward.rs::forward_chunk_inner`) | ✅ (via `forward_rows` → `prefill_chunks` + `forward_chunk`) | ✅ (`Gemma4Session` impl; `forward_chunk` delegates to `forward_rows` which already does internal chunking) | `Input = Vec<Gemma4InputRow>` (carries per-layer token metadata) |
| `nemotron_h` | partial (`prefill()` calls `forward_layer(length)` which still loops `for t in 0..length`) | ❌ | ❌ |  |
| `qwen3/asr` | ✅ | ✅ | ❌ |  |
| `qwen3/tts` | ✅ | ✅ | ❌ |  |
| `llama` | ✅ (`session.rs::forward_chunk_batched_real`) | ✅ (via `prefill_chunks` + `ChunkedPrefill`) | ✅ (`LlamaSession` impl; **real batched Q/K/V + wo + gate/up/down matmul + batched flash attention** at `B > 1`; wired into `app/text.rs:701` JEV path; legacy free-function retained as fallback) | **~2× prefill speedup at `B = 64`** |
| `lfm2` | ❌ | ✅ (dispatch loop marked `_prefill_chunks`) | ✅ (`Lfm2Session` impl; B=1 fallback delegates to `run_forward_logits_lfm2_with_batch`; `forward_attention_chunked(rows, base_position, …)` skeleton with rows threaded but `rows > 1` branch still per-row placeholder) | shortconv SSM keeps per-row by construction; per-row attention math makes lifting a 200+ line change with no easy isolation boundary |
| `lfm25` | ❌ | ✅ (dispatch loop marked `_prefill_chunks`) | ✅ (`Lfm25Session` impl; B=1 fallback delegates to `run_forward_logits_lfm25_with_batch`) | same as lfm2 |
| `lfm2moe` | ❌ | ✅ (dispatch loop marked `_prefill_chunks`) | ✅ (`Lfm2MoeSession` impl; B=1 fallback delegates to `run_forward_logits_lfm2moe_with_batch`; new `_inner` body extracted from `run_inference`) | MoE router + SSM shortconv keep per-row by construction |
| `spark` | ❌ | ✅ (via `prefill_chunks` + `ChunkedPrefill`) | ✅ (`SparkSession` impl; B=1 fallback to `forward_step_logits`) | fused QKV layout + per-head attention loop make the per-step forward hard to lift without a full `rows × n_head` batched attention rewrite |
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
18-thread Intel Core Ultra 5 125H, measured with the bench prompt
`"The quick brown fox jumps over the lazy dog. " × N`:

| Prompt tokens | B=1 | B=64 | Speedup |
|---|---|---|---|
| 53 | 15.5 | 37.9 | 2.45× |
| 304 | 18.6 | 43.7 | 2.35× |
| ~1200 | (timed out at 90s) | 23.1 | — |

`B=1` falls back to the legacy per-token path; `B=64` walks the
chunked trait loop once (single 64-row chunk) so the
`PreparedRows::matmul_group` quantise+dispatch amortises across
all 64 rows. Output text matches the per-token path on
`temperature = 0` (echo prompts stay identical across B settings).

### `llama` (highest ROI, pure transformer)

`run_forward_logits_llama` was a free function that constructed
all scratch + KV cache + pool from scratch and then looped
`for step in 0..n_prompt`. The free function is now a thin
wrapper around `run_forward_logits_llama_with_batch(...)` that
threads `--prefill-batch-size` through, and the new
`LlamaSession<'model>` (in `src/models/llama/trunk/session.rs`)
owns the same state and implements `ChunkedPrefill`.

The session's `forward_chunk` dispatches on `rows`:

- `rows == 1` → legacy per-token path (`forward_one_token`), bit-
  identical to the pre-trait baseline.
- `rows > 1` → `forward_chunk_batched_real`: `PreparedRows::prepare`
  quantises `[rows × n_embd]` activations once, then a single
  `PreparedRows::matmul_group` call dispatches Q / K / V / wo /
  gate / up / down in one pass; attention goes through
  `run_attention_chunked` (tiled flash attention — KV cache row
  loaded once per head instead of `rows × n_head` times; online
  softmax with rescale); LM head projects the last row only.

The batched path is wired into `app/text.rs:701` (JEV question
classifier) as a forwarder around `LlamaSession::from_source_with_max_rows
(...).forward_logits_chunked(...)`; on construction failure we fall
back to the legacy `run_forward_logits_llama_inner` so JEV still
works on models that don't fit the new session.

### `lfm2` / `lfm25` / `lfm2moe`

These have hybrid SSM/MoE layers where the **state depends on
previous tokens** (shortconv buffer, MoE router hidden state). The
attention sub-layers can still be batched; the SSM/MoE sub-layers
must remain per-row. So `forward_chunk` would dispatch on
`lw.is_attn` and call either a batched attention forward or a
per-row SSM forward.

`Lfm2Session<'a>` + `Lfm25Session<'a>` are extracted (mirror
`LlamaSession`'s shape) and implement `ChunkedPrefill` with a
`B = 1` fallback that delegates to the legacy free-function path
(`run_forward_logits_lfm2_with_batch` /
`run_forward_logits_lfm25_with_batch`). `forward_attention_chunked
(rows, base_position, …)` is the work-in-progress batched skeleton;
`rows` is threaded through but the `rows > 1` branch still defers
to the legacy per-row attention compute via `let _ = use_batched;`.
Lifting that to a real `PreparedRows::matmul_group` dispatch +
tiled flash attention is the next step, and is the same shape as
the llama lift. As a stop-gap, the per-step loop in
`run_forward_logits_lfm2` is wrapped by an outer `prefill_chunks`
so the dispatch matches the trait default; `--prefill-batch-size`
routes through the new wrappers in `app/text.rs:890, 1083`.

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

`cargo test --release --lib`: 775 passed / 14 failed / 58 ignored.
The 14 failures are pre-existing (`bf16_is_neither_f16_decoded_nor_…`,
`f16_projection_quantizes_input_and_uses_ggml_f16_dot`, various
`matmul::neon_tests::…` parity checks, `rope::tests::vision_rope_…`,
`models::gemma4::trunk::tests::failed_later_gemma4_chunk_preserves_successful_prefix`,
`models::qwen35::vision::tests::projector_keeps_the_existing_spatial_block_order`,
etc.). The new passes come from
`models::llama::trunk::session::tests::chunked_prefill_*` (2),
`models::lfm2::trunk::session::tests::chunked_prefill_*` (2),
`models::lfm25::trunk::session::tests::chunked_prefill_*` (2),
`models::lfm2moe::trunk::session::tests::chunked_prefill_*` (2),
`models::gemma4::trunk::session::tests::chunked_prefill_*` (2),
`models::spark::trunk::forward::tests::chunked_prefill_input_len_*` (1).
None of the 14 failures come from this branch.

## Out of scope

This commit does **not**:

- Lift `lfm2` / `lfm25` / `spark` per-token forward into a real
  batched forward. The traits and B=1 fallbacks are in place; the
  per-row math is preserved. Lifting them is the follow-up work
  per trunk.
- Touch the `--prefill-batch-size` CLI flag — it already feeds into
  `checked_prefill_batch_size` for every trunk that reads it.
- Add new SIMD kernels. The current `matmul_*` family already
  handles `[rows, n_in] × [n_in, n_out]` shaped inputs; trunks just
  need to call them with `rows > 1`.
- Refactor `qwen3` / `qwen35` / `gemma4` to drop their own
  `prefill()` wrappers in favour of `ChunkedPrefill`. The
  `forward_cpu_chunk` / `forward_chunk` API is preserved for
  backwards compatibility.
