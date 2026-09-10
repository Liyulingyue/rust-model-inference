//! Nemotron-3 Nano configuration.

use crate::core::tensor::{MetaValue, TensorSource};

#[derive(Debug, Clone)]
pub struct NemotronConfig {
    pub architecture: String,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
    pub norm_eps: f32,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_inner_size: usize,
    pub ssm_time_step_rank: usize,
}

impl NemotronConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let architecture = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .unwrap_or_default();
        if architecture != "nemotron_h" {
            return Err(format!(
                "Unsupported architecture for NemotronConfig: {architecture}"
            ));
        }
        let get_u32 = |key: &str| -> Result<u32, String> {
            let v = source
                .metadata(key)
                .and_then(MetaValue::to_u64)
                .ok_or_else(|| format!("Missing metadata: {key}"))?;
            Ok(v as u32)
        };
        let as_usize = |key: &str| -> Result<usize, String> {
            get_u32(key).map(|v| v as usize)
        };
        let get_f32 = |key: &str, default: f32| -> Result<f32, String> {
            Ok(source
                .metadata(key)
                .and_then(MetaValue::to_f64)
                .map(|v| v as f32)
                .unwrap_or(default))
        };
        let vocab_size = match get_u32("nemotron_h.vocab_size") {
            Ok(v) => v as usize,
            Err(_) => source
                .metadata("tokenizer.ggml.tokens")
                .and_then(MetaValue::to_arr)
                .map(Vec::len)
                .unwrap_or(0),
        };
        Ok(Self {
            architecture: architecture.to_string(),
            n_embd: as_usize("nemotron_h.embedding_length")?,
            n_layer: as_usize("nemotron_h.block_count")?,
            n_head: as_usize("nemotron_h.attention.head_count")?,
            // head_count_kv metadata for Nemotron-H is a histogram-style
            // summary, not per-layer values. The actual KV-head count is
            // encoded in the attn_k.weight shape. For this checkpoint the
            // 4 attention layers have n_head_kv=8, head_dim_k=128.
            n_head_kv: 8,
            n_embd_head_k: as_usize("nemotron_h.attention.key_length")?,
            n_embd_head_v: as_usize("nemotron_h.attention.value_length")?,
            n_ff: source
                .metadata("nemotron_h.feed_forward_length")
                .and_then(MetaValue::to_u64)
                .map(|v| v as usize)
                .or_else(|| {
                    source
                        .metadata("nemotron_h.feed_forward_length")
                        .and_then(MetaValue::to_arr)
                        .and_then(|arr| arr.first().and_then(MetaValue::to_u64))
                        .map(|v| v as usize)
                })
                .unwrap_or(0),
            n_ctx: as_usize("nemotron_h.context_length")?,
            vocab_size,
            rope_freq_base: get_f32("nemotron_h.rope.freq_base", 1_000_000.0)?,
            rope_dim: as_usize("nemotron_h.rope.dimension_count")?,
            norm_eps: get_f32("nemotron_h.attention.layer_norm_rms_epsilon", 1e-5)?,
            ssm_conv_kernel: as_usize("nemotron_h.ssm.conv_kernel")?,
            ssm_state_size: as_usize("nemotron_h.ssm.state_size")?,
            ssm_group_count: as_usize("nemotron_h.ssm.group_count")?,
            ssm_inner_size: as_usize("nemotron_h.ssm.inner_size")?,
            ssm_time_step_rank: as_usize("nemotron_h.ssm.time_step_rank")?,
        })
    }

    pub fn head_dim(&self) -> usize {
        if self.n_head == 0 {
            0
        } else {
            self.n_embd / self.n_head
        }
    }
}
