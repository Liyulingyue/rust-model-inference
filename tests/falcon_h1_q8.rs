use rust_model_inference::core::loader::model_config_from_source;
use rust_model_inference::core::tensor::{MetaValue, TensorInfo};
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::models::falcon_h1::trunk::{
    FalconH1Config, FalconH1Model, FalconH1Scratch,
};
use rust_model_inference::{GGUFLoader, TensorSource};
use std::sync::Arc;

fn model_loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_FALCON_H1_Q8_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

fn model_source() -> Option<Arc<dyn TensorSource>> {
    model_loader().map(|inner| Arc::new(inner) as Arc<dyn TensorSource>)
}

fn falcon_tokenizer() -> Option<BPETokenizer> {
    let loader = model_loader()?;
    Some(
        BPETokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned())
            .expect("falcon-h1 tokenizer must load"),
    )
}

struct ArchOverride {
    inner: GGUFLoader,
    arch: &'static str,
}

impl TensorSource for ArchOverride {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        if key == "general.architecture" {
            // SAFETY-free alternative: return None and let the config
            // loader report a missing-arch error instead.
            return None;
        }
        self.inner.metadata(key)
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.inner.tensor_info(name)
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.inner.tensor_slice(name)
    }
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
fn q8_contract_loads() {
    let Some(source) = model_source() else { return };
    let config = FalconH1Config::from_source(source.as_ref()).unwrap();
    assert_eq!(config.n_embd, 2048);
    assert_eq!(config.n_layer, 24);
    assert_eq!(config.n_head, 8);
    assert_eq!(config.n_head_kv, 2);
    assert_eq!(config.n_embd_head_k, 128);
    assert_eq!(config.n_embd_head_v, 128);
    assert_eq!(config.n_ff, 4608);
    assert_eq!(config.n_ctx, 131_072);
    assert_eq!(config.vocab_size, 65_537);
    assert_eq!(config.ssm_conv_kernel, 4);
    assert_eq!(config.ssm_state_size, 256);
    assert_eq!(config.ssm_group_count, 1);
    assert_eq!(config.ssm_inner_size, 3072);
    assert_eq!(config.ssm_time_step_rank, 48);
    assert_eq!(config.ssm_n_head(), 48);
    assert_eq!(config.ssm_headdim(), 64);
    assert_eq!(config.ssm_conv_cols(), 3584);
    assert_eq!(config.ssm_in_proj_dim(), 6704);
    let model = FalconH1Model::from_source(source, 1).unwrap();
    assert_eq!(model.layers.len(), 24);
    let layer = &model.layers[0];
    assert_eq!(layer.wq.n_in, 2048);
    assert_eq!(layer.wq.n_out, 1024);
    assert_eq!(layer.wk.n_out, 256);
    assert_eq!(layer.wv.n_out, 256);
    assert_eq!(layer.wo.n_in, 1024);
    assert_eq!(layer.ssm_in.n_out, 6704);
    assert_eq!(layer.ssm_out.n_in, 3072);
}

#[test]
fn q8_rejects_foreign_architecture() {
    let Some(path) = std::env::var_os("RMI_FALCON_H1_Q8_MODEL") else { return };
    let source: Arc<dyn TensorSource> = Arc::new(ArchOverride {
        inner: GGUFLoader::from_file(path).unwrap(),
        arch: "qwen3",
    });
    let error = FalconH1Config::from_source(source.as_ref())
        .err()
        .expect("foreign architecture must fail");
    assert!(error.contains("Unsupported architecture for FalconH1Config"), "{error}");
    let error = model_config_from_source(source.as_ref())
        .err()
        .expect("foreign architecture must fail the generic loader too");
    assert!(error.contains("Unsupported architecture"), "{error}");
}

#[test]
fn q8_rejects_missing_attention_projection() {
    let Some(path) = std::env::var_os("RMI_FALCON_H1_Q8_MODEL") else { return };
    let source: Arc<dyn TensorSource> = Arc::new(MissingTensor {
        inner: GGUFLoader::from_file(path).unwrap(),
        missing: "blk.12.attn_output.weight",
    });
    let error = FalconH1Model::from_source(source, 1)
        .err()
        .expect("missing attention must fail");
    assert!(error.contains("layer 12: attn_output"), "{error}");
}

#[test]
fn q8_single_token_forward_is_finite() {
    let Some(source) = model_source() else { return };
    let model = FalconH1Model::from_source(source, 1).unwrap();
    let mut scratch = FalconH1Scratch::new(&model.config, 2);
    let logits = model.prefill(&[17], &mut scratch).unwrap();
    assert_eq!(logits.len(), model.config.vocab_size);
    assert!(logits.iter().all(|value| value.is_finite()));
    assert!(scratch.ssm_scan_state.iter().any(|&value| value != 0.0));
    assert!(scratch.ssm_conv_hist.iter().any(|&value| value != 0.0));
    assert!(scratch.k.iter().any(|&value| value != 0.0));
    assert!(scratch.v.iter().any(|&value| value != 0.0));
}

#[test]
fn q8_tokenizer_matches_scalar_llama_cpp() {
    let Some(tokenizer) = falcon_tokenizer() else { return };
    assert_eq!(tokenizer.bos_id(), Some(17));
    assert_eq!(tokenizer.eos_id(), Some(228));
    // llama-tokenize @ 171e8846b on the same GGUF:
    //   17 '<|begin_of_text|>', 1563 'The', 11522 ' capital',
    //   840 ' of', 14501 ' France', 900 ' is'
    let ids = tokenizer.encode(
        "The capital of France is",
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert_eq!(ids, [17, 1563, 11522, 840, 14501, 900]);
}

#[test]
fn q8_chunked_and_incremental_logits_match() {
    let Some(source) = model_source() else { return };
    let model = FalconH1Model::from_source(source, 1).unwrap();
    let mut chunked = FalconH1Scratch::new(&model.config, 2);
    let mut incremental = FalconH1Scratch::new(&model.config, 2);
    let expected = model.prefill(&[17, 100], &mut chunked).unwrap();
    model.prefill(&[17], &mut incremental).unwrap();
    let actual = model.prefill(&[100], &mut incremental).unwrap();
    let mismatch = expected
        .iter()
        .zip(&actual)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits());
    assert!(mismatch.is_none(), "first logit mismatch: {mismatch:?}");
}

#[test]
#[ignore = "developer-only parity dump (set RMI_FALCON_PARITY=1 and run with --ignored --nocapture)"]
fn q8_parity_dump_2tok() {
    let Some(source) = model_source() else { return };
    let model = FalconH1Model::from_source(source, 4).unwrap();
    let mut scratch = FalconH1Scratch::new(&model.config, 4);
    let _ = model.prefill(&[17, 227], &mut scratch).unwrap();
}
