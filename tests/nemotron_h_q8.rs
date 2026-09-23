use rust_model_inference::core::tensor::{MetaValue, TensorInfo};
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::models::nemotron_h::trunk::{
    NemotronConfig, NemotronModel, NemotronScratch,
};
use rust_model_inference::{GGUFLoader, TensorSource};
use std::sync::Arc;

fn model_source() -> Option<Arc<dyn TensorSource>> {
    let path = std::env::var_os("RMI_NEMOTRON_Q8_MODEL")?;
    Some(Arc::new(GGUFLoader::from_file(path).unwrap()))
}

struct MissingTensor {
    inner: GGUFLoader,
    missing: &'static str,
}

impl TensorSource for MissingTensor {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.inner.metadata(key)
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        (name != self.missing)
            .then(|| self.inner.tensor_info(name))
            .flatten()
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        (name != self.missing)
            .then(|| self.inner.tensor_slice(name))
            .flatten()
    }
}

#[test]
fn q8_rejects_missing_attention_projection() {
    let Some(path) = std::env::var_os("RMI_NEMOTRON_Q8_MODEL") else {
        return;
    };
    let source: Arc<dyn TensorSource> = Arc::new(MissingTensor {
        inner: GGUFLoader::from_file(path).unwrap(),
        missing: "blk.12.attn_output.weight",
    });
    let error = NemotronModel::from_source(source)
        .err()
        .expect("missing attention must fail");
    assert!(error.contains("layer 12 tensors do not match"), "{error}");
}

#[test]
fn q8_single_token_forward_is_finite() {
    let Some(source) = model_source() else { return };
    let model = NemotronModel::from_source(source).unwrap();
    let mut scratch = NemotronScratch::new(&model.config, 2);
    let logits = model.prefill(&[1], &mut scratch).unwrap();
    assert_eq!(logits.len(), model.config.vocab_size);
    assert!(logits.iter().all(|value| value.is_finite()));
    assert!(scratch.ssm_scan_state.iter().any(|&value| value != 0.0));
}

#[test]
fn q8_chunked_and_incremental_logits_match() {
    let Some(source) = model_source() else { return };
    let model = NemotronModel::from_source(source).unwrap();
    let mut chunked = NemotronScratch::new(&model.config, 2);
    let mut incremental = NemotronScratch::new(&model.config, 2);
    let expected = model.prefill(&[1, 2], &mut chunked).unwrap();
    model.prefill(&[1], &mut incremental).unwrap();
    let actual = model.prefill(&[2], &mut incremental).unwrap();
    let mismatch = expected
        .iter()
        .zip(&actual)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits());
    assert!(mismatch.is_none(), "first logit mismatch: {mismatch:?}");
}

#[test]
fn q8_model_contract_loads() {
    let Some(source) = model_source() else { return };
    let config = NemotronConfig::from_source(source.as_ref()).unwrap();
    assert_eq!(config.n_layer, 42);
    assert_eq!(config.n_ff, 12_544);
    assert_eq!(config.n_head_kv, 8);
    let model = NemotronModel::from_source(source).unwrap();
    assert_eq!(model.layers.len(), 42);
    let attention = &model.layers[12];
    assert_eq!(attention.wq.as_ref().unwrap().n_in, 3136);
    assert_eq!(attention.wq.as_ref().unwrap().n_out, 5120);
    assert_eq!(attention.wv.as_ref().unwrap().n_out, 1024);
    assert_eq!(attention.wo.as_ref().unwrap().n_in, 5120);
    let ffn = &model.layers[1];
    assert_eq!(ffn.w_up.as_ref().unwrap().n_in, 3136);
    assert_eq!(ffn.w_up.as_ref().unwrap().n_out, 12_544);
    assert_eq!(ffn.w_down.as_ref().unwrap().n_in, 12_544);
}

#[test]
fn q8_hello_greedy_matches_scalar_llama_cpp() {
    let Some(source) = model_source() else { return };
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
    let ids = tokenizer.encode(
        "Hello",
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert_eq!(ids, [1, 22177]);
    let model = NemotronModel::from_source(source).unwrap();
    let mut scratch = NemotronScratch::new(&model.config, ids.len() + 3);
    let mut logits = model.prefill(&ids, &mut scratch).unwrap();
    let hash = logits.iter().fold(14695981039346656037u64, |hash, value| {
        (hash ^ value.to_bits() as u64).wrapping_mul(1099511628211)
    });
    assert_eq!(hash, 0x2a8f5cd0499e563b);
    let mut generated = Vec::new();
    for step in 0..4 {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        generated.push(next);
        if step < 3 {
            logits = model.prefill(&[next], &mut scratch).unwrap();
        }
    }
    assert_eq!(generated, [1044, 4304, 1033, 3075]);
}
