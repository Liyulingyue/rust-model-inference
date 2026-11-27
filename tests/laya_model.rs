use rust_model_inference::{
    core::tensor::{GGMLType, MetaValue, TensorInfo, TensorSource},
    format::ggufrs::{open_model_source, ComponentRole},
    models::laya::LayaModel,
};
use std::{collections::HashMap, path::PathBuf};

struct WrongShape {
    metadata: HashMap<String, MetaValue>,
    info: TensorInfo,
}

impl TensorSource for WrongShape {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        (name == "temperature").then_some(&self.info)
    }

    fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
        None
    }
}

#[test]
fn laya_rejects_same_element_count_with_wrong_shape() {
    let tokenizer = tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
    let metadata = HashMap::from([
        (
            "general.architecture".into(),
            MetaValue::String("laya".into()),
        ),
        (
            "laya.encoder_config".into(),
            MetaValue::String(
                serde_json::json!({"hidden_size":768,"num_hidden_layers":22,
                    "num_attention_heads":12,"intermediate_size":1152})
                .to_string(),
            ),
        ),
        (
            "laya.agent_config".into(),
            MetaValue::String(serde_json::json!({"temperature":[1,1,1]}).to_string()),
        ),
        (
            "laya.tokenizer_json".into(),
            MetaValue::String(tokenizer.to_string(false).unwrap()),
        ),
    ]);
    let source = WrongShape {
        metadata,
        info: TensorInfo {
            name: "temperature".into(),
            dims: vec![1, 3],
            ggml_type: GGMLType::F32,
            offset: 0,
        },
    };
    assert!(LayaModel::from_source(&source)
        .err()
        .unwrap()
        .contains("Invalid tensor shape temperature"));
}

#[test]
fn laya_loads_matching_model_when_available() {
    let path = std::env::var_os("LAYA_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/laya-multilingual/laya-multilingual-F32.gguf"));
    if !path.exists() {
        return;
    }
    let source = open_model_source(&path, ComponentRole::Llm).unwrap();
    let model = LayaModel::from_source(source.as_ref()).unwrap();
    assert_eq!(model.max_len(), 1024);
    assert_eq!(model.tokenizer().token_to_id("<mask>"), Some(4));
}
