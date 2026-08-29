//! Qwen3Config — hyperparameters extracted from GGUF metadata.

use crate::core::loader::{check_qwen3_allowed_dimensions, model_config_from_source, qwen3_arch_knobs};
use crate::core::tensor::TensorSource;
use super::util::optional_usize;

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
    pub rope: Qwen3Rope,
}

impl Qwen3Config {
    pub(crate) fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let config = model_config_from_source(source)?;
        let knobs = qwen3_arch_knobs(source)?;

        let n_embd_head_k = optional_usize(source, &format!("{}.attention.key_length", knobs.arch))?
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
            rope,
        })
    }
}
