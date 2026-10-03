//! Qwen3Config — hyperparameters extracted from GGUF metadata.

use super::util::optional_usize;
use crate::core::loader::{
    check_qwen3_allowed_dimensions, model_config_from_source, qwen3_arch_knobs,
};
use crate::core::tensor::TensorSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3Rope {
    Neox,
    Interleaved { sections: [i32; 4], n_dims: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3Config {
    pub architecture: String,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_ff: usize,
    pub vocab: usize,
    pub n_ctx: usize,
    pub eps: f32,
    pub freq_base: f32,
    pub has_qk_norm: bool,
    pub has_qkv_bias: bool,
    pub n_deepstack_layers: usize,
    pub moe: Option<crate::core::loader::Qwen3MoeConfig>,
    pub rope: Qwen3Rope,
    /// `true` when this Qwen3-architecture GGUF is a Microsoft
    /// BitNet b1.58 I2_S conversion (file_type=40 +
    /// `*_norm_in.weight` per-projection RMSNorm tensors present).
    /// Drives the text_encode forward to use the BitLinear path
    /// (per-projection RMSNorm → per-token absmax int8 quant →
    /// ternary matmul) instead of the Q8_0 matmul path. See
    /// `docs/usage/bitnet_embedding.md` for the full spec.
    pub is_bitnet: bool,
}

impl Qwen3Config {
    pub(crate) fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let config = model_config_from_source(source)?;
        let knobs = qwen3_arch_knobs(source)?;

        let n_embd_head_k =
            optional_usize(source, &format!("{}.attention.key_length", knobs.arch))?
                .unwrap_or(config.n_embd_head);
        let n_embd_head_v =
            optional_usize(source, &format!("{}.attention.value_length", knobs.arch))?
                .unwrap_or(config.n_embd_head);
        if n_embd_head_k == 0 || n_embd_head_v == 0 {
            return Err(format!(
                "Invalid {} attention head lengths: key={n_embd_head_k}, value={n_embd_head_v}",
                knobs.arch
            ));
        }

        if let Some(allowed) = knobs.allowed_dimensions {
            check_qwen3_allowed_dimensions(allowed, &config, n_embd_head_k, n_embd_head_v)?;
        }

        let rope = match knobs.rope_sections {
            Some(sections) => Qwen3Rope::Interleaved {
                sections,
                n_dims: n_embd_head_k,
            },
            None => Qwen3Rope::Neox,
        };
        let n_deepstack_layers =
            optional_usize(source, &format!("{}.n_deepstack_layers", knobs.arch))?.unwrap_or(0);

        // BitNet detection: Microsoft's BitNet b1.58 GGUF conversions
        // (file_type=40) ship `blk.{i}.{attn_q,attn_k,...}_norm_in.weight`
        // per-projection RMSNorm tensors alongside I2_S ternary weights.
        // Detect by either the file_type or the presence of the first
        // BitLinear per-projection norm tensor. Both checks are
        // defensive — the file_type is the canonical marker but the
        // norm presence is the structural one (some custom conversions
        // may not bump file_type).
        let file_type = source
            .metadata("general.file_type")
            .and_then(crate::core::tensor::MetaValue::to_u64)
            .unwrap_or(0);
        let has_norm_in = source.tensor_info("blk.0.attn_q_norm_in.weight").is_some();
        let is_bitnet = file_type == 40 || has_norm_in;
        if has_norm_in && file_type != 0 && file_type != 40 {
            // Defensive warning if the file_type disagrees with the
            // structural marker; not an error (community conversions
            // sometimes omit file_type=40).
        }

        Ok(Self {
            architecture: knobs.arch,
            n_embd: config.n_embd,
            n_layer: config.n_layer,
            n_head: config.n_head,
            n_head_kv: config.n_head_kv,
            n_embd_head_k,
            n_embd_head_v,
            n_ff: config.n_ff,
            vocab: config.vocab_size,
            n_ctx: config.n_ctx,
            eps: config.norm_eps,
            freq_base: config.rope_freq_base,
            has_qk_norm: knobs.has_qk_norm,
            has_qkv_bias: knobs.has_qkv_bias,
            n_deepstack_layers,
            moe: knobs.moe,
            rope,
            is_bitnet,
        })
    }
}
