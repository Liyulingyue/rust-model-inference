use super::*;
use crate::core::tensor::TensorInfo;

struct BadSource {
    tensor: Option<TensorInfo>,
    metadata: HashMap<String, MetaValue>,
}
impl BadSource {
    fn new(arch: &str, config: &str, tensor: Option<TensorInfo>) -> Self {
        Self {
            tensor,
            metadata: HashMap::from([
                (
                    "general.architecture".into(),
                    MetaValue::String(arch.into()),
                ),
                ("timestral.config".into(), MetaValue::String(config.into())),
            ]),
        }
    }
}
impl TensorSource for BadSource {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }
    fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
        self.tensor.as_ref()
    }
    fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
        None
    }
}
#[test]
fn rejects_architecture_config_and_wrong_shape_with_same_element_count() {
    let s = BadSource::new("llama", CONFIG_JSON, None);
    assert!(TimestralModel::from_source(&s)
        .err()
        .unwrap()
        .contains("architecture"));
    let s = BadSource::new("timestral", "{}", None);
    let err = TimestralModel::from_source(&s).err().unwrap();
    assert!(
        err.contains("d_model") || err.contains("num_layers") || err.contains("block_schedule"),
        "unexpected error: {err}"
    );
    let s = BadSource::new(
        "timestral",
        CONFIG_JSON,
        Some(TensorInfo {
            name: "representation.forecast_query".into(),
            dims: vec![1, D as u64],
            ggml_type: GGMLType::F32,
            offset: 0,
        }),
    );
    assert!(TimestralModel::from_source(&s)
        .err()
        .unwrap()
        .contains("shape"));
}
#[test]
fn contract_has_exact_parameter_count() {
    let config = Config {
        d_model: D,
        num_heads: HEADS,
        mlp_hidden_dim: F,
        num_layers: 24,
    };
    assert_eq!(
        tensor_contracts(&config)
            .iter()
            .map(|(_, contract)| contract.elements())
            .sum::<u64>(),
        209312256
    );
}
