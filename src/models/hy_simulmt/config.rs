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
        let scale = source
            .metadata("hy_v3.expert_weights_scale")
            .and_then(MetaValue::to_f64)
            .unwrap_or(f64::NAN);
        let moe = HyMoeConfig {
            expert_count: get("expert_count")?,
            expert_used_count: get("expert_used_count")?,
            expert_ffn: get("expert_feed_forward_length")?,
            shared_expert_ffn: get("expert_shared_feed_forward_length")?,
            scale: scale as f32,
        };
        if get("expert_gating_func")? != 2
            || !matches!(
                source.metadata("hy_v3.expert_weights_norm"),
                Some(MetaValue::Bool(true))
            )
            || !scale.is_finite()
            || scale <= 0.0
            || scale > f32::MAX as f64
            || moe.expert_used_count > moe.expert_count
        {
            return Err("Unsupported hy_v3 sigmoid/shared-expert routing metadata".into());
        }
        let config = Self {
            n_embd: get("embedding_length")?,
            n_layer: get("block_count")?,
            n_head: get("attention.head_count")?,
            n_head_kv: get("attention.head_count_kv")?,
            n_embd_head_k: get("attention.key_length")?,
            n_embd_head_v: get("attention.value_length")?,
            n_ff: get("feed_forward_length")?,
            vocab: if source.metadata("hy_v3.vocab_size").is_some() {
                get("vocab_size")?
            } else {
                base.vocab_size
            },
            n_ctx: get("context_length")?,
            eps: base.norm_eps,
            freq_base: base.rope_freq_base,
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
