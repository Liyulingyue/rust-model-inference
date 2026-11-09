//! Nemotron-3 Nano configuration.

use crate::core::tensor::{MetaValue, TensorSource};

/// Mixture-of-experts parameters.
///
/// Only the `nemotron_h_moe` architecture (Nemotron-3-Nano 30B-A3B)
/// carries these; the dense `nemotron_h` (4B) has `moe: None`.
#[derive(Debug, Clone)]
pub struct NemotronMoeConfig {
    /// Total number of routed experts (128 for the 30B).
    pub n_expert: usize,
    /// How many experts each token activates (6 for the 30B).
    pub n_expert_used: usize,
    /// Number of expert groups. The 30B uses a single group, i.e. no
    /// grouped routing.
    pub n_expert_group: usize,
    /// How many groups each token activates (1 for the 30B).
    pub n_group_used: usize,
    /// Expert FFN width (1856 for the 30B).
    pub n_ff_exp: usize,
    /// Shared-expert FFN width (3712 for the 30B).
    pub n_ff_shexp: usize,
    /// Number of shared experts (1 for the 30B).
    pub n_expert_shared: usize,
    /// Whether the router weights are normalized before scaling.
    pub expert_weights_norm: bool,
    /// Scale applied to the router weights after normalization (2.5 for
    /// the 30B).
    pub expert_weights_scale: f32,
    /// Latent projection width for the MoE branch, when the model has
    /// one. The 30B does not, so the expert dimension equals `n_embd`.
    pub moe_latent_size: Option<usize>,
}

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
    /// MoE parameters; `None` for the dense architecture.
    pub moe: Option<NemotronMoeConfig>,
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
    /// Whether this model routes through a mixture of experts.
    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }
    /// Whether the architecture string denotes the MoE variant. Used
    /// while the config is being built, before `moe` is populated.
    pub fn is_moe_architecture(&self) -> bool {
        self.architecture == "nemotron_h_moe"
    }
    /// The dimension the MoE branch operates in. With a latent
    /// projection the experts consume `moe_latent_size`, otherwise they
    /// consume the hidden size directly.
    pub fn moe_dim(&self) -> usize {
        match &self.moe {
            Some(moe) => moe.moe_latent_size.unwrap_or(self.n_embd),
            None => self.n_embd,
        }
    }
}

impl NemotronConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let architecture = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .unwrap_or_default();
        // `nemotron_h` is the dense 4B, `nemotron_h_moe` the 30B-A3B
        // mixture-of-experts variant. Both share the hybrid
        // Mamba-Transformer trunk; the MoE one adds the expert layers.
        if architecture != "nemotron_h" && architecture != "nemotron_h_moe" {
            return Err(format!(
                "Unsupported architecture for NemotronConfig: {architecture}"
            ));
        }
        // GGUF namespaces every architecture-specific metadata key under
        // the architecture name, so the key prefix follows the arch.
        let prefix = architecture.clone();
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
        let get_bool = |key: &str| -> Option<bool> {
            match source.metadata(key) {
                Some(MetaValue::Bool(value)) => Some(*value),
                _ => None,
            }
        };
        // Every architecture-specific key is `<arch>.<name>`.
        let key = |name: &str| format!("{prefix}.{name}");
        let vocab_size = match get_usize(&key("vocab_size")) {
            Ok(v) => v,
            Err(_) => source
                .metadata("tokenizer.ggml.tokens")
                .and_then(MetaValue::to_arr)
                .map(Vec::len)
                .unwrap_or(0),
        };
        let n_layer = get_usize(&key("block_count"))?;
        let ff_lengths = layer_values(&key("feed_forward_length"))?;
        let kv_heads = layer_values(&key("attention.head_count_kv"))?;
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
            n_embd: get_usize(&key("embedding_length"))?,
            n_layer,
            n_head: get_usize(&key("attention.head_count"))?,
            n_head_kv,
            attention_layers,
            ffn_layers,
            n_embd_head_k: get_usize(&key("attention.key_length"))?,
            n_embd_head_v: get_usize(&key("attention.value_length"))?,
            n_ff,
            n_ctx: get_usize(&key("context_length"))?,
            vocab_size,
            rope_freq_base: get_f32(&key("rope.freq_base"), 1_000_000.0)?,
            rope_dim: get_usize(&key("rope.dimension_count"))?,
            norm_eps: get_f32(&key("attention.layer_norm_rms_epsilon"), 1e-5)?,
            ssm_conv_kernel: get_usize(&key("ssm.conv_kernel"))?,
            ssm_state_size: get_usize(&key("ssm.state_size"))?,
            ssm_group_count: get_usize(&key("ssm.group_count"))?,
            ssm_inner_size: get_usize(&key("ssm.inner_size"))?,
            ssm_time_step_rank: get_usize(&key("ssm.time_step_rank"))?,
            moe: None,
        };
        // The MoE variant adds the expert metadata. The dense variant
        // has none of these keys, so `moe` stays `None` there.
        let moe = if config.is_moe_architecture() {
            let moe = NemotronMoeConfig {
                n_expert: get_usize(&key("expert_count"))?,
                n_expert_used: get_usize(&key("expert_used_count"))?,
                n_expert_group: get_usize(&key("expert_group_count"))?,
                n_group_used: get_usize(&key("expert_group_used_count"))?,
                n_ff_exp: get_usize(&key("expert_feed_forward_length"))?,
                n_ff_shexp: get_usize(&key("expert_shared_feed_forward_length"))?,
                n_expert_shared: get_usize(&key("expert_shared_count"))?,
                expert_weights_norm: get_bool(&key("expert_weights_norm")).unwrap_or(false),
                expert_weights_scale: get_f32(&key("expert_weights_scale"), 1.0)?,
                // The 30B has no latent projection, so the expert
                // dimension is the hidden size. The key is absent there.
                moe_latent_size: source
                    .metadata(&key("moe_latent_size"))
                    .and_then(MetaValue::to_u64)
                    .and_then(|value| usize::try_from(value).ok()),
            };
            if moe.n_expert == 0
                || moe.n_expert_used == 0
                || moe.n_expert_used > moe.n_expert
                || moe.n_expert_group == 0
                || moe.n_group_used == 0
                || moe.n_group_used > moe.n_expert_group
                || moe.n_ff_exp == 0
                || moe.n_ff_shexp == 0
            {
                return Err("Unsupported Nemotron MoE dimensions".into());
            }
            Some(moe)
        } else {
            None
        };
        let config = Self { moe, ..config };
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
