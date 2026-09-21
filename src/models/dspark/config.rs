use crate::core::tensor::{MetaValue, TensorSource};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetShape {
    pub hidden: usize,
    pub vocab: usize,
    pub layers: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DSparkConfig {
    pub block_size: usize,
    pub target_layers: Vec<usize>,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub markov_rank: usize,
}

impl DSparkConfig {
    pub fn from_source(source: &dyn TensorSource, target: TargetShape) -> Result<Self, String> {
        if source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            != Some("dflash")
        {
            return Err("Invalid general.architecture: expected dflash".into());
        }
        if target.hidden == 0 || target.vocab == 0 || target.layers == 0 {
            return Err("Invalid DSpark target shape".into());
        }

        let block_size = positive_usize(source, "dflash.block_size")?;
        let target_layers = usize_array(source, "dflash.target_layers")?;
        let invalid_target_layers = target_layers.is_empty()
            || target_layers.iter().any(|&layer| layer >= target.layers)
            || target_layers
                .iter()
                .enumerate()
                .any(|(index, layer)| target_layers[..index].contains(layer));
        if invalid_target_layers {
            return Err(format!(
                "Invalid dflash.target_layers for target with {} layers",
                target.layers
            ));
        }

        let hidden = positive_usize(source, "dflash.embedding_length")?;
        if hidden != target.hidden {
            return Err(format!(
                "DSpark hidden width mismatch: sidecar has {hidden}, target has {}",
                target.hidden
            ));
        }
        let layers = positive_usize(source, "dflash.block_count")?;
        let heads = positive_usize(source, "dflash.attention.head_count")?;
        let kv_heads = positive_usize(source, "dflash.attention.head_count_kv")?;
        let head_dim = if source.metadata("dflash.rope.dimension_count").is_some() {
            positive_usize(source, "dflash.rope.dimension_count")?
        } else {
            positive_usize(source, "dflash.attention.key_length")?
        };
        if source.metadata("dflash.attention.value_length").is_some()
            && positive_usize(source, "dflash.attention.value_length")? != head_dim
        {
            return Err("DSpark key/value head dimensions differ".into());
        }
        let ffn = positive_usize(source, "dflash.feed_forward_length")?;
        let eps = positive_f32(source, "dflash.attention.layer_norm_rms_epsilon")?;
        let rope_base = positive_f32(source, "dflash.rope.freq_base")?;
        if heads.checked_mul(head_dim).is_none() || heads % kv_heads != 0 {
            return Err("Inconsistent dflash attention dimensions".into());
        }

        let markov = tensor_dims(source, "markov_w1.weight")?;
        if markov.len() != 2 || markov[0] == 0 {
            return Err(format!("Invalid tensor markov_w1.weight shape: {markov:?}"));
        }
        if usize::try_from(markov[1]).ok() != Some(target.vocab) {
            return Err(format!(
                "DSpark vocabulary mismatch: sidecar has {}, target has {}",
                markov[1], target.vocab
            ));
        }
        let markov_rank = usize::try_from(markov[0])
            .map_err(|_| "markov_w1.weight rank does not fit usize".to_string())?;

        require_shape(source, "markov_w2.weight", &[markov_rank, target.vocab])?;
        let confidence_input = hidden
            .checked_add(markov_rank)
            .ok_or_else(|| "DSpark confidence input size overflow".to_string())?;
        let confidence_shape = tensor_dims(source, "conf_proj.weight")?;
        if confidence_shape != [confidence_input as u64]
            && confidence_shape != [confidence_input as u64, 1]
        {
            return Err(format!(
                "Invalid tensor conf_proj.weight shape: {confidence_shape:?}"
            ));
        }
        if source.tensor_info("conf_proj.bias").is_some() {
            require_shape(source, "conf_proj.bias", &[1])?;
        }
        let encoder_input = target_layers
            .len()
            .checked_mul(target.hidden)
            .ok_or_else(|| "DSpark encoder input size overflow".to_string())?;
        require_shape(source, "fc.weight", &[encoder_input, hidden])?;

        Ok(Self {
            block_size,
            target_layers,
            hidden,
            layers,
            heads,
            kv_heads,
            head_dim,
            ffn,
            vocab: target.vocab,
            eps,
            rope_base,
            markov_rank,
        })
    }
}

fn integer(value: &MetaValue) -> Option<u64> {
    match value {
        MetaValue::Uint8(value) => Some((*value).into()),
        MetaValue::Uint16(value) => Some((*value).into()),
        MetaValue::Uint32(value) => Some((*value).into()),
        MetaValue::Uint64(value) => Some(*value),
        MetaValue::Int8(value) => u64::try_from(*value).ok(),
        MetaValue::Int16(value) => u64::try_from(*value).ok(),
        MetaValue::Int32(value) => u64::try_from(*value).ok(),
        MetaValue::Int64(value) => u64::try_from(*value).ok(),
        _ => None,
    }
}

fn positive_usize(source: &dyn TensorSource, key: &str) -> Result<usize, String> {
    source
        .metadata(key)
        .and_then(integer)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
        .ok_or_else(|| format!("Missing or invalid metadata: {key}"))
}

fn usize_array(source: &dyn TensorSource, key: &str) -> Result<Vec<usize>, String> {
    let values = match source.metadata(key) {
        Some(MetaValue::Array(_, values)) => values,
        _ => return Err(format!("Missing or invalid metadata: {key}")),
    };
    values
        .iter()
        .map(|value| {
            integer(value)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("Invalid metadata entry: {key}"))
        })
        .collect()
}

fn positive_f32(source: &dyn TensorSource, key: &str) -> Result<f32, String> {
    let value = source
        .metadata(key)
        .and_then(MetaValue::to_f64)
        .ok_or_else(|| format!("Missing or invalid metadata: {key}"))?;
    let value = value as f32;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(format!("Missing or invalid metadata: {key}"))
    }
}

fn tensor_dims<'a>(source: &'a dyn TensorSource, name: &str) -> Result<&'a [u64], String> {
    source
        .tensor_info(name)
        .map(|info| info.dims.as_slice())
        .ok_or_else(|| format!("Missing tensor: {name}"))
}

fn require_shape(source: &dyn TensorSource, name: &str, expected: &[usize]) -> Result<(), String> {
    let expected = expected
        .iter()
        .map(|&dimension| u64::try_from(dimension))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| format!("Tensor {name} shape does not fit u64"))?;
    let actual = tensor_dims(source, name)?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "Invalid tensor {name} shape: {actual:?}; expected {expected:?}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{DSparkConfig, TargetShape};
    use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
    use std::collections::HashMap;

    const TARGET: TargetShape = TargetShape {
        hidden: 4,
        vocab: 16,
        layers: 12,
    };

    #[derive(Default)]
    struct FixtureSource {
        metadata: HashMap<String, MetaValue>,
        tensors: HashMap<String, TensorInfo>,
    }

    impl FixtureSource {
        fn metadata(mut self, key: &str, value: MetaValue) -> Self {
            self.metadata.insert(key.into(), value);
            self
        }

        fn tensor(mut self, name: &str, dims: &[u64]) -> Self {
            self.tensors.insert(
                name.into(),
                TensorInfo {
                    name: name.into(),
                    dims: dims.to_vec(),
                    ggml_type: GGMLType::F32,
                    offset: 0,
                },
            );
            self
        }
    }

    impl TensorSource for FixtureSource {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.metadata.get(key)
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.tensors.get(name)
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    fn valid_source() -> FixtureSource {
        FixtureSource::default()
            .metadata("general.architecture", MetaValue::String("dflash".into()))
            .metadata("dflash.block_size", MetaValue::Uint32(7))
            .metadata(
                "dflash.target_layers",
                MetaValue::Array(
                    MetaValueType::Uint32,
                    vec![
                        MetaValue::Uint32(1),
                        MetaValue::Uint32(5),
                        MetaValue::Uint32(9),
                    ],
                ),
            )
            .metadata("dflash.embedding_length", MetaValue::Uint32(4))
            .metadata("dflash.block_count", MetaValue::Uint32(2))
            .metadata("dflash.attention.head_count", MetaValue::Uint32(2))
            .metadata("dflash.attention.head_count_kv", MetaValue::Uint32(1))
            .metadata("dflash.rope.dimension_count", MetaValue::Uint32(2))
            .metadata("dflash.feed_forward_length", MetaValue::Uint32(8))
            .metadata(
                "dflash.attention.layer_norm_rms_epsilon",
                MetaValue::Float32(1e-6),
            )
            .metadata("dflash.rope.freq_base", MetaValue::Float32(1_000_000.0))
            .tensor("markov_w1.weight", &[2, TARGET.vocab as u64])
            .tensor("markov_w2.weight", &[2, TARGET.vocab as u64])
            .tensor("conf_proj.weight", &[6, 1])
            .tensor("conf_proj.bias", &[1])
            .tensor("fc.weight", &[12, 4])
    }

    #[test]
    fn parses_valid_sidecar_contract() {
        let config = DSparkConfig::from_source(&valid_source(), TARGET).unwrap();
        assert_eq!(config.block_size, 7);
        assert_eq!(config.target_layers, vec![1, 5, 9]);
        assert_eq!(config.hidden, 4);
        assert_eq!(config.layers, 2);
        assert_eq!(config.heads, 2);
        assert_eq!(config.kv_heads, 1);
        assert_eq!(config.head_dim, 2);
        assert_eq!(config.ffn, 8);
        assert_eq!(config.vocab, TARGET.vocab);
        assert_eq!(config.markov_rank, 2);
        let mut source = valid_source();
        source.tensors.get_mut("conf_proj.weight").unwrap().dims = vec![6];
        assert!(DSparkConfig::from_source(&source, TARGET).is_ok());
        source.metadata.remove("dflash.rope.dimension_count");
        source
            .metadata
            .insert("dflash.attention.key_length".into(), MetaValue::Uint32(2));
        assert!(DSparkConfig::from_source(&source, TARGET).is_ok());
    }

    #[test]
    fn rejects_invalid_architecture_block_and_target_layers() {
        let mut source = valid_source();
        source.metadata.insert(
            "general.architecture".into(),
            MetaValue::String("qwen3".into()),
        );
        assert!(DSparkConfig::from_source(&source, TARGET)
            .unwrap_err()
            .contains("general.architecture"));

        let mut source = valid_source();
        source
            .metadata
            .insert("dflash.block_size".into(), MetaValue::Uint32(0));
        assert!(DSparkConfig::from_source(&source, TARGET)
            .unwrap_err()
            .contains("dflash.block_size"));

        let mut source = valid_source();
        source.metadata.insert(
            "dflash.target_layers".into(),
            MetaValue::Array(MetaValueType::Uint32, vec![]),
        );
        assert!(DSparkConfig::from_source(&source, TARGET)
            .unwrap_err()
            .contains("target_layers"));

        let mut source = valid_source();
        source.metadata.insert(
            "dflash.target_layers".into(),
            MetaValue::Array(
                MetaValueType::Uint32,
                vec![
                    MetaValue::Uint32(5),
                    MetaValue::Uint32(1),
                    MetaValue::Uint32(9),
                ],
            ),
        );
        assert_eq!(
            DSparkConfig::from_source(&source, TARGET)
                .unwrap()
                .target_layers,
            vec![5, 1, 9]
        );
    }

    #[test]
    fn rejects_target_and_tensor_shape_mismatches() {
        let mut source = valid_source();
        source.tensors.get_mut("markov_w1.weight").unwrap().dims[1] += 1;
        assert!(DSparkConfig::from_source(&source, TARGET)
            .unwrap_err()
            .contains("vocabulary"));

        let mut source = valid_source();
        source.tensors.get_mut("fc.weight").unwrap().dims[0] += 1;
        assert!(DSparkConfig::from_source(&source, TARGET)
            .unwrap_err()
            .contains("fc.weight"));
    }
}
