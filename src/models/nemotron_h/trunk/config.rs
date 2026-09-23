//! Nemotron-3 Nano configuration.

use crate::core::tensor::{MetaValue, TensorSource};

#[derive(Debug, Clone)]
pub struct NemotronConfig {
    pub architecture: String,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub attention_layers: Vec<usize>,
    pub ffn_layers: Vec<usize>,
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
    /// Number of SSM heads. The 4B Nano GGUF stores
    /// `ssm_dt.bias.shape == (dt_rank,)`, and canonical Mamba2 uses
    /// one dt per head. So n_ssm_head == ssm_time_step_rank.
    pub fn ssm_n_head(&self) -> usize {
        self.ssm_time_step_rank
    }
    /// Per-head channel count: d_inner / n_head.
    pub fn ssm_headdim(&self) -> usize {
        self.ssm_inner_size / self.ssm_n_head()
    }
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
        let get_usize = |key: &str| -> Result<usize, String> {
            let v = source
                .metadata(key)
                .and_then(MetaValue::to_u64)
                .ok_or_else(|| format!("Missing metadata: {key}"))?;
            usize::try_from(v).map_err(|_| format!("Invalid metadata: {key} exceeds usize"))
        };
        let layer_values = |key: &str| -> Result<Vec<usize>, String> {
            let values = source
                .metadata(key)
                .and_then(MetaValue::to_arr)
                .ok_or_else(|| format!("Missing array metadata: {key}"))?;
            values
                .iter()
                .map(|value| {
                    value
                        .to_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| format!("Invalid array metadata: {key}"))
                })
                .collect()
        };
        let get_f32 = |key: &str, default: f32| -> Result<f32, String> {
            Ok(source
                .metadata(key)
                .and_then(MetaValue::to_f64)
                .map(|v| v as f32)
                .unwrap_or(default))
        };
        let vocab_size = match get_usize("nemotron_h.vocab_size") {
            Ok(v) => v,
            Err(_) => source
                .metadata("tokenizer.ggml.tokens")
                .and_then(MetaValue::to_arr)
                .map(Vec::len)
                .unwrap_or(0),
        };
        let n_layer = get_usize("nemotron_h.block_count")?;
        let ff_lengths = layer_values("nemotron_h.feed_forward_length")?;
        let kv_heads = layer_values("nemotron_h.attention.head_count_kv")?;
        if ff_lengths.len() != n_layer || kv_heads.len() != n_layer {
            return Err("Nemotron layer metadata length does not match block_count".into());
        }
        let n_ff = ff_lengths.iter().copied().max().unwrap_or(0);
        let n_head_kv = kv_heads.iter().copied().max().unwrap_or(0);
        let attention_layers: Vec<usize> = kv_heads
            .iter()
            .enumerate()
            .filter_map(|(layer, &heads)| (heads != 0).then_some(layer))
            .collect();
        let ffn_layers: Vec<usize> = ff_lengths
            .iter()
            .enumerate()
            .filter_map(|(layer, &width)| (width != 0).then_some(layer))
            .collect();
        if n_ff == 0 || n_head_kv == 0 {
            return Err("Nemotron requires FFN and attention layers".into());
        }
        if ff_lengths.iter().any(|&value| value != 0 && value != n_ff)
            || kv_heads
                .iter()
                .any(|&value| value != 0 && value != n_head_kv)
            || ff_lengths
                .iter()
                .zip(&kv_heads)
                .any(|(&ff, &kv)| ff != 0 && kv != 0)
        {
            return Err("Unsupported Nemotron layer dimensions".into());
        }
        let config = Self {
            architecture: architecture.to_string(),
            n_embd: get_usize("nemotron_h.embedding_length")?,
            n_layer,
            n_head: get_usize("nemotron_h.attention.head_count")?,
            n_head_kv,
            attention_layers,
            ffn_layers,
            n_embd_head_k: get_usize("nemotron_h.attention.key_length")?,
            n_embd_head_v: get_usize("nemotron_h.attention.value_length")?,
            n_ff,
            n_ctx: get_usize("nemotron_h.context_length")?,
            vocab_size,
            rope_freq_base: get_f32("nemotron_h.rope.freq_base", 1_000_000.0)?,
            rope_dim: get_usize("nemotron_h.rope.dimension_count")?,
            norm_eps: get_f32("nemotron_h.attention.layer_norm_rms_epsilon", 1e-5)?,
            ssm_conv_kernel: get_usize("nemotron_h.ssm.conv_kernel")?,
            ssm_state_size: get_usize("nemotron_h.ssm.state_size")?,
            ssm_group_count: get_usize("nemotron_h.ssm.group_count")?,
            ssm_inner_size: get_usize("nemotron_h.ssm.inner_size")?,
            ssm_time_step_rank: get_usize("nemotron_h.ssm.time_step_rank")?,
        };
        if config.n_head == 0
            || config.n_head % config.n_head_kv != 0
            || config.ssm_conv_kernel < 2
            || config.ssm_group_count == 0
            || config.ssm_time_step_rank == 0
            || config.ssm_inner_size % config.ssm_group_count != 0
            || config.ssm_inner_size % config.ssm_time_step_rank != 0
            || config.ssm_time_step_rank % config.ssm_group_count != 0
        {
            return Err("Unsupported Nemotron attention or SSM dimensions".into());
        }
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        if self.n_head == 0 {
            0
        } else {
            self.n_embd / self.n_head
        }
    }
}
