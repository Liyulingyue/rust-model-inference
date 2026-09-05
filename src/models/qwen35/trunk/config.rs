//! Qwen35Config — hyperparameters for the Qwen3.5 hybrid LLM trunk.

use crate::core::tensor::{MetaValue, MetaValueType, TensorSource};

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub n_embd: usize,
    pub n_layer: usize,
    /// MTP (nextn) predict layers appended beyond the main stack — excluded
    /// from standard inference (llama.cpp: n_layer_all - n_layer_nextn).
    pub n_nextn: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub rope_freq_base: f32,
    pub norm_eps: f32,
    pub rope_dimension_count: usize,
    pub rope_dimension_sections: [i32; 4],
    pub ssm_d_conv: usize,
    pub ssm_d_state: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_d_inner: usize,
    pub full_attention_interval: usize,
    pub is_recurrent: Vec<bool>,
    pub key_length: usize,
    pub value_length: usize,
}

impl Qwen35Config {
    /// Layers actually executed by the trunk: the main stack without the
    /// appended MTP (nextn) predict blocks.
    pub fn n_layer_impl(&self) -> usize {
        self.n_layer - self.n_nextn
    }
}

fn unsigned_u64(value: &MetaValue) -> Option<u64> {
    match value {
        MetaValue::Uint8(value) => Some(*value as u64),
        MetaValue::Uint16(value) => Some(*value as u64),
        MetaValue::Uint32(value) => Some(*value as u64),
        MetaValue::Uint64(value) => Some(*value),
        _ => None,
    }
}

fn full_attention_interval(value: Option<&MetaValue>) -> Result<Option<u64>, String> {
    value
        .map(|value| {
            unsigned_u64(value).ok_or_else(|| {
                "Invalid qwen35.full_attention_interval: expected unsigned integer".into()
            })
        })
        .transpose()
}

fn recurrent_layer_mask(
    n_layer: usize,
    n_nextn: usize,
    recurrent_layers: Option<&MetaValue>,
    full_attention_interval: Option<u64>,
) -> Result<Vec<bool>, String> {
    if let Some(value) = recurrent_layers {
        let MetaValue::Array(_, values) = value else {
            return Err("Invalid qwen35.attention.recurrent_layers: expected array".into());
        };
        if values.len() != n_layer {
            return Err(format!(
                "Invalid qwen35.attention.recurrent_layers length: expected {n_layer}, got {}",
                values.len()
            ));
        }
        return values
            .iter()
            .enumerate()
            .map(|(index, value)| match value {
                MetaValue::Bool(value) => Ok(*value),
                value => match unsigned_u64(value) {
                    Some(0) => Ok(false),
                    Some(1) => Ok(true),
                    _ => Err(format!(
                        "Invalid recurrent selector at layer {index}; expected 0, 1, or bool"
                    )),
                },
            })
            .collect();
    }

    let interval = full_attention_interval.unwrap_or(4);
    if interval == 0 {
        return Err("qwen35.full_attention_interval must be greater than zero".into());
    }
    // MTP (nextn) blocks are dense attention-only and never recurrent.
    let n_layer_impl = n_layer.saturating_sub(n_nextn);
    Ok((0..n_layer)
        .map(|layer| layer < n_layer_impl && ((layer as u64 + 1) % interval) != 0)
        .collect())
}

impl Qwen35Config {
    /// Build a `Qwen35Config` from a GGUF tensor source.
    pub fn from_source<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let base = crate::core::loader::model_config_from_source(source)?;

        let get_u32 = |key: &str| -> Result<u32, String> {
            source
                .metadata(key)
                .and_then(|v| v.to_u64())
                .map(|v| v as u32)
                .ok_or_else(|| format!("Missing qwen35 metadata: {}", key))
        };

        let key_length = source
            .metadata("qwen35.attention.key_length")
            .and_then(|v| v.to_u64())
            .unwrap_or(base.n_embd_head as u64) as usize;
        let value_length = source
            .metadata("qwen35.attention.value_length")
            .and_then(|v| v.to_u64())
            .unwrap_or(base.n_embd_head as u64) as usize;

        let rope_dimension_count = source
            .metadata("qwen35.rope.dimension_count")
            .and_then(|v| v.to_u64())
            .unwrap_or(64) as usize;

        let rope_dimension_sections = match source.metadata("qwen35.rope.dimension_sections") {
            Some(MetaValue::Array(_, vals)) => {
                let s: Vec<i32> = vals
                    .iter()
                    .filter_map(|v| v.to_u64().map(|x| x as i32))
                    .collect();
                [
                    s.first().copied().unwrap_or(16),
                    s.get(1).copied().unwrap_or(16),
                    s.get(2).copied().unwrap_or(16),
                    s.get(3).copied().unwrap_or(16),
                ]
            }
            _ => {
                let sec = rope_dimension_count as i32 / 4;
                [sec, sec, sec, sec]
            }
        };

        let ssm_d_conv = get_u32("qwen35.ssm.conv_kernel")? as usize;
        let ssm_d_state = get_u32("qwen35.ssm.state_size")? as usize;
        let ssm_n_group = get_u32("qwen35.ssm.group_count")? as usize;
        let ssm_dt_rank = get_u32("qwen35.ssm.time_step_rank")? as usize;
        let ssm_d_inner = get_u32("qwen35.ssm.inner_size")? as usize;
        let full_attention_interval_raw =
            full_attention_interval(source.metadata("qwen35.full_attention_interval"))?;
        let n_nextn = source
            .metadata("qwen35.nextn_predict_layers")
            .and_then(unsigned_u64)
            .unwrap_or(0) as usize;
        let n_layer_impl = base
            .n_layer
            .checked_sub(n_nextn)
            .filter(|&layers| layers > 0)
            .ok_or_else(|| {
                format!(
                    "Invalid qwen35 nextn layer split: blocks={}, nextn={n_nextn}",
                    base.n_layer
                )
            })?;
        if key_length == 0 || value_length == 0 || key_length != value_length {
            return Err(format!(
                "Invalid qwen35 attention head lengths: key_length={key_length}, value_length={value_length}"
            ));
        }
        if ssm_d_conv == 0
            || ssm_d_state == 0
            || ssm_n_group == 0
            || ssm_dt_rank == 0
            || ssm_d_inner == 0
            || ssm_d_inner % ssm_dt_rank != 0
        {
            return Err(format!(
                "Invalid qwen35 SSM inner/rank dimensions: conv={ssm_d_conv}, state={ssm_d_state}, groups={ssm_n_group}, inner_size={ssm_d_inner}, time_step_rank={ssm_dt_rank}"
            ));
        }
        let mut is_recurrent = recurrent_layer_mask(
            base.n_layer,
            n_nextn,
            source.metadata("qwen35.attention.recurrent_layers"),
            full_attention_interval_raw,
        )?;
        is_recurrent.truncate(n_layer_impl);
        let full_attention_interval = usize::try_from(full_attention_interval_raw.unwrap_or(4))
            .map_err(|_| "qwen35.full_attention_interval does not fit usize")?;

        Ok(Self {
            n_embd: base.n_embd,
            n_layer: base.n_layer,
            n_nextn,
            n_head: base.n_head,
            n_head_kv: base.n_head_kv,
            n_ff: base.n_ff,
            n_ctx: base.n_ctx,
            vocab_size: base.vocab_size,
            rope_freq_base: base.rope_freq_base,
            norm_eps: base.norm_eps,
            rope_dimension_count,
            rope_dimension_sections,
            ssm_d_conv,
            ssm_d_state,
            ssm_n_group,
            ssm_dt_rank,
            ssm_d_inner,
            full_attention_interval,
            is_recurrent,
            key_length,
            value_length,
        })
    }

    pub fn n_embd_head(&self) -> usize {
        self.key_length
    }

    pub fn n_embd_gqa(&self) -> usize {
        self.n_head_kv * self.n_embd_head()
    }

    pub fn key_dim(&self) -> usize {
        self.ssm_d_state * self.ssm_n_group
    }

    pub fn value_dim(&self) -> usize {
        self.ssm_d_state * self.ssm_dt_rank
    }

    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }

    pub fn head_v_dim(&self) -> usize {
        self.ssm_d_inner / self.ssm_dt_rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValueType, TensorInfo};
    use std::collections::HashMap;

    #[derive(Default)]
    struct ConfigSource {
        metadata: HashMap<String, MetaValue>,
    }

    impl TensorSource for ConfigSource {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.metadata.get(key)
        }

        fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
            None
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    fn qwen38_config_source() -> ConfigSource {
        use MetaValue::{Array, Float32, String as Text, Uint32};

        ConfigSource {
            metadata: HashMap::from([
                ("general.architecture".into(), Text("qwen35".into())),
                ("qwen35.embedding_length".into(), Uint32(5120)),
                ("qwen35.block_count".into(), Uint32(65)),
                ("qwen35.attention.head_count".into(), Uint32(24)),
                ("qwen35.attention.head_count_kv".into(), Uint32(4)),
                ("qwen35.attention.key_length".into(), Uint32(256)),
                ("qwen35.attention.value_length".into(), Uint32(256)),
                ("qwen35.feed_forward_length".into(), Uint32(17408)),
                ("qwen35.context_length".into(), Uint32(262144)),
                ("qwen35.vocab_size".into(), Uint32(248320)),
                ("qwen35.rope.freq_base".into(), Float32(10_000_000.0)),
                (
                    "qwen35.attention.layer_norm_rms_epsilon".into(),
                    Float32(1e-6),
                ),
                ("qwen35.rope.dimension_count".into(), Uint32(64)),
                (
                    "qwen35.rope.dimension_sections".into(),
                    Array(
                        MetaValueType::Uint32,
                        vec![Uint32(11), Uint32(11), Uint32(10), Uint32(0)],
                    ),
                ),
                ("qwen35.ssm.conv_kernel".into(), Uint32(4)),
                ("qwen35.ssm.state_size".into(), Uint32(128)),
                ("qwen35.ssm.group_count".into(), Uint32(16)),
                ("qwen35.ssm.time_step_rank".into(), Uint32(48)),
                ("qwen35.ssm.inner_size".into(), Uint32(6144)),
                ("qwen35.full_attention_interval".into(), Uint32(4)),
                ("qwen35.nextn_predict_layers".into(), Uint32(1)),
            ]),
        }
    }

    #[test]
    fn qwen35_recurrent_layers_metadata_is_authoritative() {
        let values = MetaValue::Array(
            MetaValueType::Uint32,
            [1, 0, 1, 0].into_iter().map(MetaValue::Uint32).collect(),
        );
        assert_eq!(
            recurrent_layer_mask(4, 0, Some(&values), Some(3)).unwrap(),
            vec![true, false, true, false],
        );
    }

    #[test]
    fn qwen35_recurrent_layers_reject_malformed_arrays() {
        let wrong_length = MetaValue::Array(MetaValueType::Uint32, vec![MetaValue::Uint32(1)]);
        assert!(recurrent_layer_mask(2, 0, Some(&wrong_length), Some(4)).is_err());

        let invalid_selector = MetaValue::Array(
            MetaValueType::Uint32,
            vec![MetaValue::Uint32(1), MetaValue::Uint32(2)],
        );
        assert!(recurrent_layer_mask(2, 0, Some(&invalid_selector), Some(4)).is_err());
    }

    #[test]
    fn qwen35_recurrent_layers_reject_float_selectors() {
        let float_selector = MetaValue::Array(
            MetaValueType::Float32,
            vec![MetaValue::Float32(0.5), MetaValue::Uint32(1)],
        );
        assert!(recurrent_layer_mask(2, 0, Some(&float_selector), Some(4)).is_err());
    }

    #[test]
    fn qwen35_rejects_float_interval_with_recurrent_layers() {
        let recurrent_layers = MetaValue::Array(
            MetaValueType::Uint32,
            vec![MetaValue::Uint32(1), MetaValue::Uint32(0)],
        );
        let float_interval = MetaValue::Float32(4.0);
        assert!(full_attention_interval(Some(&float_interval))
            .and_then(|interval| recurrent_layer_mask(2, 0, Some(&recurrent_layers), interval))
            .is_err());
    }

    #[test]
    fn qwen35_zero_interval_is_an_error_not_a_panic() {
        assert!(recurrent_layer_mask(4, 0, None, Some(0)).is_err());
    }

    #[test]
    fn qwen35_interval_fallback_matches_llama() {
        assert_eq!(
            recurrent_layer_mask(8, 0, None, Some(4)).unwrap(),
            vec![true, true, true, false, true, true, true, false],
        );
    }

    #[test]
    fn qwen35_config_accepts_qwen38_dimensions() {
        let config = Qwen35Config::from_source(&qwen38_config_source()).unwrap();

        assert_eq!(
            (config.n_layer, config.n_layer_impl(), config.n_nextn),
            (65, 64, 1)
        );
        assert_eq!((config.n_embd_head(), config.n_embd_gqa()), (256, 1024));
        assert_eq!((config.key_dim(), config.value_dim()), (2048, 6144));
        assert_eq!((config.conv_dim(), config.head_v_dim()), (10240, 128));
        assert_eq!(config.is_recurrent.len(), 64);
        assert_eq!(
            config
                .is_recurrent
                .iter()
                .filter(|&&recurrent| !recurrent)
                .count(),
            16
        );
        assert_eq!(config.rope_dimension_sections, [11, 11, 10, 0]);
    }

    #[test]
    fn qwen35_config_rejects_invalid_tensor_dimensions() {
        for (key, value, expected) in [
            ("qwen35.attention.key_length", 0, "head lengths"),
            ("qwen35.attention.value_length", 0, "head lengths"),
            ("qwen35.attention.value_length", 128, "head lengths"),
            ("qwen35.ssm.time_step_rank", 0, "inner/rank"),
            ("qwen35.ssm.inner_size", 6145, "inner/rank"),
            ("qwen35.nextn_predict_layers", 65, "nextn layer split"),
        ] {
            let mut source = qwen38_config_source();
            source.metadata.insert(key.into(), MetaValue::Uint32(value));
            let error = Qwen35Config::from_source(&source).unwrap_err();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[test]
    #[ignore = "requires RMI_QWEN35_MODEL"]
    fn qwen35_config_uses_real_layer_selection_metadata() {
        let path = std::env::var("RMI_QWEN35_MODEL").unwrap();
        let source =
            crate::open_model_source(std::path::Path::new(&path), crate::ComponentRole::Llm)
                .unwrap();
        let config = Qwen35Config::from_source(source.as_ref()).unwrap();
        let expected = recurrent_layer_mask(
            config.n_layer,
            config.n_nextn,
            source.metadata("qwen35.attention.recurrent_layers"),
            full_attention_interval(source.metadata("qwen35.full_attention_interval")).unwrap(),
        )
        .unwrap();
        assert_eq!(config.is_recurrent, expected[..config.n_layer_impl()]);
    }
}
