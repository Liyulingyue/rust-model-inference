//! Qwen2.5-Omni-3B text encoder for AuK-Base 1.5B.
//!
//! Architecture: `qwen2vl` (per GGUF `general.architecture`), 36 layers,
//! hidden=2048, n_head=16, n_kv=2 (GQA), ffn=11008, RoPE base 1_000_000.
//! No QK-norm (Qwen2.5 style), QKV biases present.
//!
//! `qwen3::trunk::Qwen3Model` already supports the `qwen2vl` arch via
//! `qwen3_arch_knobs` (`has_qk_norm=false`, `has_qkv_bias=true`, tensor
//! naming `blk.{i}.{attn_q,attn_k,attn_v,attn_output,attn_norm,
//! ffn_gate,ffn_up,ffn_down,ffn_norm}.weight`). We just wrap the model
//! here and expose `encode(prompt) -> Vec<f32>` returning per-token
//! hidden states of width `TEXT_IN = 2048` (matches AuK's
//! `transformer.txt_proj` input dim).
//!
//! The AukPipeline then projects these to AuK's hidden via `txt_proj`
//! inside `AukDit::denoise`.

use std::sync::Arc;

use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::models::qwen3::trunk::{text_encode as trunk_text_encode, Qwen3Model};
use crate::models::qwen3::trunk::positions::qwen_text_positions;

const TEXT_IN: usize = 2_048;

pub(crate) struct AukTextEncoder {
    model: Qwen3Model,
    pool: Arc<ComputePool>,
}

impl AukTextEncoder {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let tokenizer = Arc::new(BPETokenizer::from_qwen3_embedded_merges()?);
        let model = Qwen3Model::from_source(source, tokenizer, Arc::clone(&pool))?;
        Ok(Self { model, pool })
    }

    /// Encode `prompt` to per-token hidden states of width `TEXT_IN = 2048`.
    /// Output is a single contiguous `Vec<f32>` of shape `[seq_len * TEXT_IN]`,
    /// suitable for direct consumption by `AukDit::denoise`.
    pub(crate) fn encode(&self, prompt: &str) -> Result<Vec<f32>, String> {
        if prompt.is_empty() {
            return Err("AuK prompt is empty".into());
        }
        let ids = self.model.tokenizer().encode(
            prompt,
            crate::core::tokenizer::EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        );
        if ids.is_empty() {
            return Err("AuK prompt produced no tokens".into());
        }
        let positions = qwen_text_positions(ids.len());
        let hidden = trunk_text_encode(&self.model, &ids, &positions)?;
        if hidden.len() % TEXT_IN != 0 {
            return Err(format!(
                "AukTextEncoder produced malformed hidden: len {} not divisible by TEXT_IN={}",
                hidden.len(),
                TEXT_IN,
            ));
        }
        if !hidden.iter().all(|v| v.is_finite()) {
            return Err("AukTextEncoder produced non-finite hidden states".into());
        }
        Ok(hidden)
    }

    /// Number of tokens in the last encoded sequence.
    pub(crate) fn last_token_count(&self, hidden: &[f32]) -> usize {
        hidden.len() / TEXT_IN
    }
}