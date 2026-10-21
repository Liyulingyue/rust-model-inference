use rust_model_inference::app::cli::KvFormat;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::models::dory::{DoryConfig, DoryModel};
use rust_model_inference::GGUFLoader;
use std::path::PathBuf;
use std::sync::Arc;

fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("RMI_DORY_Q5_MODEL")?);
    path.exists().then_some(path)
}

fn source() -> Option<Arc<dyn TensorSource>> {
    let path = model_path()?;
    Some(Arc::new(
        GGUFLoader::from_file(path).expect("open Dory GGUF"),
    ))
}

#[test]
fn dory_q5_contract_and_tokenizer() {
    let Some(source) = source() else { return };
    let config = DoryConfig::from_source(source.as_ref()).expect("Dory config");
    assert_eq!(config.n_embd, 2560);
    assert_eq!(config.n_layer, 72);
    assert_eq!(config.n_input_layers, 16);
    assert_eq!(config.n_recurrent_layers, 40);
    assert_eq!(config.n_output_layers, 16);
    assert_eq!(config.n_loops, 4);
    assert_eq!(config.n_head, 16);
    assert_eq!(config.n_head_kv, 4);
    assert_eq!(config.head_dim, 256);
    assert_eq!(config.n_ff, 7168);
    assert_eq!(config.n_ctx, 262_144);
    assert_eq!(config.vocab_size, 131_072);
    assert_eq!(config.rope_freq_base, 10_000_000.0);
    assert_eq!(config.rope_freq_base_2, 10_000.0);
    assert_eq!(config.sliding_window, 1024);
    assert_eq!(config.recurrent_kv_cache_mode, "per_loop");
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
        .expect("Dory GPT-2 tokenizer");
    assert_eq!(tokenizer.bos_id(), Some(1));
    assert_eq!(tokenizer.eos_id(), Some(11));
    assert_eq!(
        tokenizer.encode(
            "Hello",
            EncodeOptions {
                add_special: false,
                parse_special: true,
            }
        ),
        vec![22177]
    );
}

#[test]
fn dory_q5_single_token_logits_are_finite() {
    let Some(source) = source() else { return };
    let model = DoryModel::from_source(source, 1).expect("load Dory model");
    let mut session = model.new_session(2, KvFormat::F32);
    let logits = model.prefill(&[1], &mut session).expect("Dory forward");
    assert_eq!(logits.len(), 131_072);
    assert!(logits.iter().all(|value| value.is_finite()));
}
