//! # Gemma-2 model family
//!
//! Gemma-2 (Google, 2024) is a decoder-only transformer with three
//! architectural changes vs the original Gemma:
//!
//! - **GeGLU** FFN (`gelu(gate) * up`) instead of SwiGLU.
//! - **Logit softcapping** on both attention scores
//!   (`gemma2.attention.attn_logit_softcapping = 50.0`) and the final
//!   LM-head logits (`gemma2.final_logit_softcapping = 30.0`). Both
//!   squash their input through `cap * tanh(x / cap)` so the
//!   softmax / cross-entropy stays well-conditioned at long context.
//! - **Sliding-window attention** (`gemma2.attention.sliding_window =
//!   4096` for the 2B / 9B variants, 1024 for 4B / 27B) — each query
//!   only attends to the most-recent window. Combined with a few
//!   global-attention layers in the 9B / 27B variants, but Gemma-2-2B
//!   is pure sliding-window.
//! - **4-norm sandwich per layer**: `attn_norm` (pre-attn) +
//!   `post_attention_norm` (pre-residual) + `ffn_norm` (pre-FFN) +
//!   `post_ffw_norm` (pre-residual).
//!
//! The tensor layout follows llama.cpp (`blk.{i}.attn_q/k/v/output`
//! + `blk.{i}.ffn_gate/up/down`), so the model reuses the **llama**
//! trunk family unmodified in its tensor-loading path. The
//! gemma-2-specific behaviour (GeGLU, softcap, sliding window) is
//! all wired inside `models::llama::trunk` via arch-aware dispatch on
//! `general.architecture == "gemma2"`.
//!
//! ## Public surface
//!
//! - `compute_embedding` — last-token **hidden state** as the
//!   embedding vector. Reuses the llama trunk's session-driven
//!   prefill and reads `scratch.x` after the final residual add —
//!   `rms_norm_grouped` writes its output to a separate `normed`
//!   buffer, so the residual stream (last hidden state) is preserved
//!   in `scratch.x`. Result dim = `n_embd` (2304 for 2B-it).

use crate::app::cli::KvFormat;
use crate::core::loader::model_config_from_source;
use crate::core::prefill::ChunkedPrefill;
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::{load_tokenizer, EncodeOptions};
use crate::models::llama::trunk::LlamaSession;

/// Last-token hidden state as the embedding vector for a Gemma-2
/// prompt. Returns `Vec<f32>` of length `n_embd` (2304 for 2B-it,
/// 3584 for 4B-it, 4096 for 9B/27B-it). No learnable pooling head —
/// matches the convention used by every other decoder-LM
/// "embedding" path.
pub fn compute_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    let tokenizer = load_tokenizer(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;

    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    if arch != "gemma2" {
        return Err(format!(
            "gemma2::compute_embedding called on non-gemma2 arch {arch:?}"
        ));
    }

    // Embedding-mode tokenization: raw text + BOS, no chat template
    // wrap. The CLI / HTTP generation paths wrap prompts in
    // `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`
    // before tokenization (see `models::llama::trunk::forward`'s
    // `llama_turn_text` for gemma-2-it chat handling), but the
    // embedding API takes raw user text so the residual stream at
    // the last position encodes the prompt rather than the
    // constant chat-template suffix. Matches
    // `models::gemma3::encode_embedding_input` byte-for-byte.
    let prompt_tokens = tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    if prompt_tokens.is_empty() {
        return Err("Gemma-2 embedding input produced no tokens".into());
    }

    let config = model_config_from_source(source)
        .map_err(|error| format!("Failed to parse Gemma-2 model config: {error}"))?;
    let n_embd = config.n_embd;

    // The session caps KV at `min(n_ctx, max_context)` internally.
    // Gemma-2-2B declares `context_length = 8192`; cap at that to
    // avoid allocating 2 GiB of KV cache on tiny machines.
    let max_ctx = config.n_ctx.min(8192);
    let kv_format = KvFormat::F16;

    // Build the session — owns weights, KV cache, scratchpad, and
    // the gemma-2-specific config (GeGLU dispatch + attn softcap +
    // sliding window + 4-norm sandwich). `from_source_with_max_rows`
    // wires all arch-aware behaviour; `max_rows=1` reproduces the
    // legacy per-token forward exactly.
    let mut session =
        LlamaSession::from_source_with_max_rows(source, n_threads_arg, kv_format, max_ctx, 1)
            .map_err(|error| format!("Gemma-2 session init failed: {error}"))?;

    // `prefill` walks the prompt one token at a time. After it
    // returns, `scratch.x[..n_embd]` holds the post-final-residual
    // stream (= last-token hidden state, BEFORE the optional
    // `output_norm` + LM-head pass). The session's per-layer loop
    // applies GeGLU + attn softcap + sliding window + 4-norm sandwich
    // for gemma2; the per-token decode loop in `forward_one_token`
    // writes the final logits to `scratch.logits` but leaves `x`
    // untouched (output_norm reads x, writes to `normed`).
    session
        .prefill(&prompt_tokens, 1)
        .map_err(|error| format!("Gemma-2 session prefill failed: {error}"))?;

    Ok(session.scratch.x[..n_embd].to_vec())
}
