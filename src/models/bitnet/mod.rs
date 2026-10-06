//! Microsoft BitNet b1.58 (W1.58A8) decoder trunk family.
//!
//! BitNet b1.58 is a Microsoft research line that replaces every
//! `nn.Linear` in a transformer decoder with a 4-step pipeline:
//!
//! ```text
//! x → RMSNorm(x, norm_in.weight)         // pre-projection centering
//!   → per-token absmax quantize(x)         // activations to int8
//!   → matmul(ternary_weights, x_q)         // I2_S × int8
//!   → rescale by (absmax / 127)            // recover F32 magnitude
//! ```
//!
//! The weights are stored at full magnitude `{-1, 0, +1}` in the
//! GGUF I2_S format (Microsoft-proprietary extension; see
//! `TODO.md` for the provenance note), and a per-projection RMSNorm
//! gain absorbs the inverse-of-mean-absweight scale.
//!
//! # Architectural separation from `qwen3` / `gemma3`
//!
//! BitNet has its own decoder trunk family in this repo. It is
//! **not** a sub-mode of `qwen3` or `gemma3`: those trunks are
//! the standard Q8_0 matmul path with no `is_bitnet` branch
//! polluting their forward loops. The BitNet trunks live here:
//!
//! - [`qwen3_arch`] — Qwen3-backbone BitNet (e.g. `bitnet-embedding-0.6b`)
//! - [`gemma3_arch`] — Gemma3-backbone BitNet (e.g. `bitnet-embedding-270m`)
//!
//! Both share the BitLinear data shape ([`slot::BitLinearWeights`])
//! and ops ([`forward::bitlinear_forward`]) but have
//! architecture-specific forward loops (Qwen3 has a 2-norm
//! sandwich; Gemma3 has a 4-norm sandwich plus per-head QK-norm).
//!
//! # Why everything lives in `models/bitnet/` (not split between
//! `ops/bitnet/` and `models/bitnet/`)
//!
//! BitLinear forward is a **composite op**: RMSNorm + absmax int8
//! quant + ternary matmul + rescale. The math (scalar reference +
//! SIMD path) doesn't strictly need model config, but the I2_S GGUF
//! format, the per-projection `*_norm_in` RMSNorm gain, and the
//! 7-slot-per-layer pattern are all BitNet-specific — there's no
//! other model family in this engine that uses BitLinear. Keeping
//! the op primitives and the shape types co-located with the model
//! wiring makes the family boundary explicit; we can split back
//! into `ops/` later if a second non-BitNet model ever adopts W1.58A8.
//!
//! # Pooling convention
//!
//! Both BitNet Embedding models declare `pooling_type = 1` with
//! `file_type = 40`. The engine routes `pooling_type=1` to last-
//! token for BitNet (mean pooling is the Qwen3-Embedding default;
//! BitNet Embedding uses last-token). See `embedding.rs`.

pub mod embedding;
pub mod forward;
pub mod forward_avx2;
pub mod gemma3_arch;
pub mod qwen3_arch;
pub mod slot;

pub use embedding::{
    compute_embedding, print_embedding, print_embedding_for_arch, run_embedding,
    run_embedding_tokens,
};
pub use forward::{
    bitlinear_forward, bitlinear_forward_from_f32, bitlinear_forward_packed,
    bitlinear_forward_scalar, quantize_activation_per_token,
};
pub use slot::{BitLinearSlot, BitLinearSlotPacked, BitLinearWeights, BitLinearWeightsPacked};

/// Detect whether a `TensorSource` carries a Microsoft BitNet b1.58
/// GGUF, irrespective of the inner architecture.
///
/// Heuristic (matches the official `microsoft/BitNet` conversion
/// script's markers):
///
/// 1. `general.file_type == 40` (Microsoft's
///    `LLAMA_FTYPE_MOSTLY_I2_S` constant), OR
/// 2. presence of `blk.0.attn_q_norm_in.weight` (per-projection
///    BitLinear RMSNorm gain — only BitNet GGUFs ship this).
///
/// Both checks are robust against non-Microsoft re-exports that
/// might forget one of the two markers.
pub fn detect_is_bitnet<S: crate::core::tensor::TensorSource + ?Sized>(source: &S) -> bool {
    let file_type_matches = source
        .metadata("general.file_type")
        .and_then(|v| v.to_u64())
        .map(|v| v == 40)
        .unwrap_or(false);
    let has_norm_in = source.tensor_info("blk.0.attn_q_norm_in.weight").is_some();
    file_type_matches || has_norm_in
}

/// Public surface used by `app::run_embedding`'s `"qwen3"` +
/// `is_bitnet=true` and `"gemma3"` arms. Both BitNet architectures
/// share the same `compute_embedding` signature, so we re-export
/// them under a single function name.
pub fn compute_embedding_for_arch(
    source: &dyn crate::core::tensor::TensorSource,
    prompt: &str,
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or("");
    match arch {
        "qwen3" => qwen3_arch::compute_embedding(source, prompt, n_threads_arg),
        "gemma3" => gemma3_arch::compute_embedding(source, prompt, n_threads_arg),
        other => Err(format!(
            "bitnet::compute_embedding_for_arch: unsupported arch {other:?}; \
             expected qwen3 or gemma3"
        )),
    }
}
