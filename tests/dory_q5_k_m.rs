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

fn reference_prompt() -> String {
    match std::env::var_os("RMI_DORY_PROMPT_FILE") {
        Some(path) => std::fs::read_to_string(path).expect("read raw prompt"),
        None => std::env::var("RMI_DORY_PROMPT").unwrap_or_else(|_| "Hello".into()),
    }
}

#[test]
#[ignore = "requires the real Q5_K_M GGUF; see tools/oracle/dory/README.md"]
fn dory_q5_reference_trace() {
    let source = source().expect("set RMI_DORY_Q5_MODEL to the real GGUF");
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
    let prompt = tokenizer.encode(
        &reference_prompt(),
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert!(!prompt.is_empty());
    let count: usize = std::env::var("RMI_DORY_N_PREDICT")
        .unwrap_or_else(|_| "8".into())
        .parse()
        .unwrap();
    let threads = std::env::var("RMI_DORY_THREADS")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    let kv = match std::env::var("RMI_DORY_KV").as_deref().unwrap_or("f32") {
        "f32" => KvFormat::F32,
        "f16" => KvFormat::F16,
        other => panic!("unknown KV format {other}"),
    };
    let mut model = DoryModel::from_source(source, threads).unwrap();
    if let Ok(window) = std::env::var("RMI_DORY_TEST_SWA") {
        model.config.sliding_window = window.parse().unwrap();
        assert!(model.config.sliding_window > 0);
    }
    let mut session = model.new_session(prompt.len() + count, kv);
    #[cfg(feature = "parity-trace")]
    rust_model_inference::parity_trace::token_ids("prompt_ids", &prompt).unwrap();
    let mut logits = Vec::new();
    let prefix = std::env::var("RMI_DORY_TRACE_FROM")
        .ok()
        .map(|v| v.parse::<usize>().unwrap().min(prompt.len()))
        .unwrap_or(0);
    for (i, chunk) in prompt[..prefix].chunks(32).enumerate() {
        logits = model.prefill(chunk, &mut session).unwrap();
        eprintln!("prefill {}/{}", ((i + 1) * 32).min(prefix), prefix);
    }
    for token in &prompt[prefix..] {
        logits = model.prefill(&[*token], &mut session).unwrap();
    }
    let mut generated = Vec::new();
    for _ in 0..count {
        assert!(logits.iter().all(|x| x.is_finite()));
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        generated.push(best as u32);
        logits = model.prefill(&[best as u32], &mut session).unwrap();
    }
    #[cfg(feature = "parity-trace")]
    rust_model_inference::parity_trace::token_ids("generated_ids", &generated).unwrap();
    println!("prompt_ids={prompt:?}\ngenerated_ids={generated:?}");
    println!("output={:?}", tokenizer.decode(&generated, true));
}

#[test]
#[ignore = "requires the real Q5_K_M GGUF"]
fn dory_q5_prefill_chunking_and_reset() {
    let model = DoryModel::from_source(source().expect("set RMI_DORY_Q5_MODEL"), 4).unwrap();
    let tokens = [22177, 1897, 1429];
    for kv in [KvFormat::F32, KvFormat::F16] {
        let mut batched = model.new_session(tokens.len(), kv);
        let expected = model.prefill(&tokens, &mut batched).unwrap();
        let mut chunked = model.new_session(tokens.len(), kv);
        model.prefill(&tokens[..2], &mut chunked).unwrap();
        let actual = model.prefill(&tokens[2..], &mut chunked).unwrap();
        let assert_close = |actual: &[f32]| {
            assert_eq!(expected.len(), actual.len());
            assert!(expected.iter().zip(actual).all(|(a, b)| a.is_finite()
                && b.is_finite()
                && (a - b).abs() <= 1e-6 * a.abs().max(1.0)));
        };
        assert_close(&actual);
        assert!(model.prefill(&[], &mut chunked).is_err());
        assert!(model.prefill(&[1], &mut chunked).is_err());
        chunked.reset(&model, kv);
        assert_close(&model.prefill(&tokens, &mut chunked).unwrap());
    }
}
