//! The MTP draft head of a Qwen3.5 checkpoint.
//!
//! The head is the block after the main stack: it takes the main model's
//! last hidden state and the embedding of the next token and predicts the
//! token after that. That is the draft a speculative decoder proposes.
//!
//! The graph is llama.cpp's (`src/models/qwen35.cpp::graph_mtp`):
//!
//! ```text
//! concat(RMSNorm(embed(next_token), nextn.enorm), RMSNorm(h_last, nextn.hnorm))
//!     -> nextn.eh_proj -> one attention block -> shared_head(shared_head_norm(.))
//! ```
//!
//! The optional `nextn.embed_tokens`, `nextn.shared_head_head` and
//! `nextn.shared_head_norm` fall back to the main model's embedding, output
//! weight and output norm, exactly as upstream does.

use super::weights::{HybridTrunk, MtpWeights};
use crate::core::scratchpad::KvCache;
use crate::core::thread_pool::ComputePool;

/// The draft head's output: the logits for the token after the one that was
/// embedded, plus the head's own hidden state.
pub struct MtpDraft {
    pub logits: Vec<f32>,
    /// The head's hidden state, after the head norm.
    ///
    /// llama.cpp hands this to the next draft step as the draft context's
    /// `h_nextn` (`src/models/qwen35.cpp::graph_mtp` sets `t_h_nextn` after the
    /// head norm), and the target's `h_nextn` is likewise the hidden state
    /// *after* the output norm. Both sides therefore work on normalized
    /// hidden states.
    pub hidden: Vec<f32>,
}

/// Run the MTP head once.
///
/// `hidden` is the main model's last hidden state *after* the output norm
/// (llama.cpp's `h_nextn`), `token` the token that follows it, and
/// `kv_cache` the head's own KV cache (one layer, as in llama.cpp's separate
/// draft context).
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_mtp(
    trunk: &HybridTrunk<'_>,
    mtp: &MtpWeights<'_>,
    hidden: &[f32],
    token: u32,
    position: usize,
    mrope_position: [usize; 4],
    kv_cache: &mut KvCache,
    scratch: &mut super::scratch::Qwen35Scratchpad,
    pool: &ComputePool,
) -> Result<MtpDraft, String> {
    let cfg = &trunk.config;
    let n_embd = cfg.n_embd;
    let eps = cfg.norm_eps;
    if hidden.len() != n_embd {
        return Err(format!(
            "Qwen3.5 MTP hidden state must be {n_embd} values, got {}",
            hidden.len()
        ));
    }

    // The token embedding and the hidden state are normalized separately and
    // then concatenated: the projection sees them side by side.
    let mut e_norm = trunk.embed_tokens(&[token])?;
    crate::ops::rms_norm_inplace(&mut e_norm, &mtp.enorm, eps);
    let mut h_norm = hidden.to_vec();
    crate::ops::rms_norm_inplace(&mut h_norm, &mtp.hnorm, eps);
    let mut fused = Vec::with_capacity(2 * n_embd);
    fused.extend_from_slice(&e_norm);
    fused.extend_from_slice(&h_norm);

    // nextn.eh_proj: [n_embd, 2 * n_embd] -> n_embd.
    project(&mtp.eh_proj, &fused, scratch, pool);
    let mut projected = scratch.matmul_out[..n_embd].to_vec();

    // One attention block with the head's own weights and its own KV cache.
    // The cache holds one layer, so the layer count is 1 here.
    let attn_out = trunk.forward_dense_attn_layer_with(
        &mtp.layer,
        cfg.n_layer_impl(),
        0,
        1,
        &projected,
        1,
        kv_cache,
        scratch,
        pool,
        &[mrope_position],
        Some(position),
        false,
        #[cfg(feature = "parity-trace")]
        false,
    );
    for (index, value) in attn_out.iter().enumerate() {
        projected[index] += value;
    }

    // The FFN, then the head.
    let mut ffn_input = projected.clone();
    crate::ops::rms_norm_inplace(&mut ffn_input, &mtp.layer.attn_post_norm, eps);
    trunk.forward_ffn_parallel(&mtp.layer, &ffn_input, 1, scratch, pool, false);
    for (index, value) in scratch.buf[..n_embd].iter().enumerate() {
        projected[index] += value;
    }

    let head_norm = mtp
        .shared_head_norm
        .as_deref()
        .unwrap_or(trunk.output_norm.as_slice());
    let mut normed = projected;
    crate::ops::rms_norm_inplace(&mut normed, head_norm, eps);
    let head = mtp
        .shared_head_head
        .as_ref()
        .unwrap_or(&trunk.output_weight);
    project(head, &normed, scratch, pool);
    let logits = scratch.matmul_out[..head.n_out].to_vec();

    Ok(MtpDraft {
        logits,
        hidden: normed,
    })
}

/// One projection with the shared quantized matmul path. The result lands in
/// the scratchpad's matmul buffer, which is sized for the vocabulary.
fn project(
    weight: &crate::ops::kernel::Weight<'_>,
    input: &[f32],
    scratch: &mut super::scratch::Qwen35Scratchpad,
    pool: &ComputePool,
) {
    let output = &mut scratch.matmul_out[..weight.n_out];
    weight.quantize_and_matmul_with_scratch(
        input,
        &mut scratch.q8k_buf,
        &mut scratch.q8_buf,
        &mut scratch.scale_buf,
        output,
        pool,
    );
}
