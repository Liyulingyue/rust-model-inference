use crate::core::loader::model_config_from_source;
use crate::core::tensor::{MetaValue, TensorSource};

#[derive(Clone, Debug)]
pub struct HyConfig {
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
    pub moe: HyMoeConfig,
}

#[derive(Clone, Debug)]
pub struct HyMoeConfig {
    pub expert_count: usize,
    pub expert_used_count: usize,
    pub expert_ffn: usize,
    pub shared_expert_ffn: usize,
    pub normalize_weights: bool,
    pub scale: f32,
}

impl HyConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        if source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            != Some("hy_v3")
        {
            return Err("Hy-SimulMT requires general.architecture=hy_v3".into());
        }
        let get = |name: &str| {
            let key = format!("hy_v3.{name}");
            source
                .metadata(&key)
                .and_then(|value| match value {
                    MetaValue::Float32(_) | MetaValue::Float64(_) => None,
                    _ => value.to_u64(),
                })
                .filter(|&v| v > 0 && v <= isize::MAX as u64)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| format!("Missing or invalid metadata: {key}"))
        };
        let base = model_config_from_source(source)?;
        let n_layer = get("block_count")?;
        let n_ff = get("feed_forward_length")?;
        let expert_count = get("expert_count")?;
        let expert_used_count = get("expert_used_count")?;
        let nonnegative = |value: &MetaValue| match value {
            MetaValue::Float32(_) | MetaValue::Float64(_) => None,
            _ => value
                .to_u64()
                .filter(|&v| v <= isize::MAX as u64)
                .and_then(|v| usize::try_from(v).ok()),
        };
        let expert_width = source
            .metadata("hy_v3.expert_feed_forward_length")
            .and_then(|value| match value {
                MetaValue::Array(_, values)
                    if values.len() == n_layer
                        && values.iter().all(|v| nonnegative(v).is_some()) =>
                {
                    // The pinned hy-v3 loader uses n_ff_exp(0) for every block.
                    values.first().and_then(nonnegative)
                }
                MetaValue::Array(_, _) => None,
                value => nonnegative(value),
            })
            .ok_or("Missing or invalid metadata: hy_v3.expert_feed_forward_length")?;
        let expert_ffn = if expert_width == 0 {
            n_ff / expert_used_count
        } else {
            expert_width
        };
        let optional_integer = |name: &str| -> Result<usize, String> {
            let key = format!("hy_v3.{name}");
            source.metadata(&key).map_or(Ok(0), |value| {
                nonnegative(value).ok_or_else(|| format!("Invalid metadata: {key}"))
            })
        };
        let shared_width = optional_integer("expert_shared_feed_forward_length")?;
        let normalize_weights = match source.metadata("hy_v3.expert_weights_norm") {
            None => false,
            Some(MetaValue::Bool(value)) => *value,
            _ => return Err("Invalid metadata: hy_v3.expert_weights_norm".into()),
        };
        let scale = source
            .metadata("hy_v3.expert_weights_scale")
            .map_or(Some(0.0), MetaValue::to_f64)
            .ok_or("Invalid metadata: hy_v3.expert_weights_scale")?;
        let moe = HyMoeConfig {
            expert_count,
            expert_used_count,
            expert_ffn,
            shared_expert_ffn: if shared_width == 0 {
                expert_ffn
            } else {
                shared_width
            },
            normalize_weights,
            scale: if scale == 0.0 { 1.0 } else { scale as f32 },
        };
        if !matches!(optional_integer("expert_gating_func")?, 0 | 2)
            || !moe.scale.is_finite()
            || moe.expert_ffn == 0
            || moe.expert_used_count > moe.expert_count
        {
            return Err("Unsupported hy_v3 sigmoid/shared-expert routing metadata".into());
        }
        let config = Self {
            n_embd: get("embedding_length")?,
            n_layer,
            n_head: get("attention.head_count")?,
            n_head_kv: get("attention.head_count_kv")?,
            n_embd_head_k: get("attention.key_length")?,
            n_embd_head_v: get("attention.value_length")?,
            n_ff,
            vocab: if source.metadata("hy_v3.vocab_size").is_some() {
                get("vocab_size")?
            } else {
                base.vocab_size
            },
            n_ctx: get("context_length")?,
            eps: source
                .metadata("hy_v3.attention.layer_norm_rms_epsilon")
                .and_then(MetaValue::to_f64)
                .ok_or("Missing or invalid metadata: hy_v3.attention.layer_norm_rms_epsilon")?
                as f32,
            freq_base: source
                .metadata("hy_v3.rope.freq_base")
                .map_or(Some(10_000.0), MetaValue::to_f64)
                .ok_or("Invalid metadata: hy_v3.rope.freq_base")? as f32,
            moe,
        };
        if config.n_embd == 0
            || config.n_layer < 2
            || config.n_head == 0
            || config.n_head_kv == 0
            || config.n_head % config.n_head_kv != 0
            || config.n_embd_head_k != config.n_embd_head_v
            || config.n_embd_head_k % 2 != 0
            || config.n_ff == 0
            || config.vocab == 0
            || config.n_ctx == 0
            || !config.eps.is_finite()
            || config.eps <= 0.0
            || !config.freq_base.is_finite()
            || config.freq_base <= 0.0
        {
            return Err(
                "Invalid Hy-SimulMT decoder dimensions or normalization/RoPE metadata".into(),
            );
        }
        Ok(config)
    }
}
