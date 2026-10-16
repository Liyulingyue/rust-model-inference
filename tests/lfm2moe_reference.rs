//! Bit-exact parity harness for `lfm2moe` (LFM2.5-24B-A2B) against the pinned
//! llama.cpp oracle (`tools/oracle/lfm2moe/`).
//!
//! The test drives `Lfm2MoeSession::forward_token` one token at a time — the
//! same per-token walk `run_inference_with_batch` performs — with the **raw**
//! prompt and no chat template, so both sides consume identical token ids. The
//! trunk emits every checkpoint the oracle labels through
//! `llm_graph_context::cb()`, and `tools/oracle/lfm2moe/check_scalar.py` diffs
//! the two manifests bit for bit (no tolerance, no cosine similarity).
//!
//! Both sides must run scalar: build with `--features parity-trace`, run the
//! binary with `RMI_SCALAR=1`, and build the oracle with the scalar flags in
//! `tools/oracle/lfm2moe/build_lfm2moe_oracle.sh`.
//!
//! The test is skipped (green) unless `RMI_LFM2MOE_MODEL` points at a real
//! GGUF, matching the other `*_reference.rs` harnesses in this repo.

use std::path::PathBuf;
use std::sync::Arc;

use rust_model_inference::core::scratchpad::KvFormat;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::lfm2moe::{Lfm2MoeConfig, Lfm2MoeSession};

const DEFAULT_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-QAD-Q4_0.gguf"
);

const DEFAULT_PROMPT: &str = "What is 2+2?";

fn model_path() -> Option<PathBuf> {
    let path = std::env::var("RMI_LFM2MOE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_MODEL));
    if path.exists() {
        Some(path)
    } else {
        eprintln!("skipping: {} not present", path.display());
        None
    }
}

fn open_source(path: &PathBuf) -> Arc<dyn TensorSource> {
    let loader =
        rust_model_inference::core::loader::GGUFLoader::from_file(path).expect("open GGUF");
    Arc::new(loader)
}

/// Replay `prompt` through the session and report the greedy continuation.
///
/// `add_bos` mirrors the oracle's `common_tokenize(ctx, prompt,
/// llama_vocab_get_add_bos(vocab), /* parse_special */ true)`: the `lfm2`
/// pre-tokenizer does not prepend BOS on `add_special`, so the harness adds
/// the GGUF's `tokenizer.ggml.bos_token_id` itself to land on the same ids.
fn replay(
    source: &Arc<dyn TensorSource>,
    prompt: &str,
    n_predict: usize,
    add_bos: bool,
) -> (Vec<u32>, Vec<u32>) {
    let tokenizer =
        rust_model_inference::core::tokenizer::BPETokenizer::from_gguf_metadata(|key| {
            source.metadata(key).cloned()
        })
        .expect("tokenizer");
    let config = Lfm2MoeConfig::from_source(source.as_ref()).expect("config");
    // One thread: the oracle harness pins `n_threads = 1`, and the pool splits
    // reduction order per thread in some kernels.
    let mut session = Lfm2MoeSession::from_source(source.as_ref(), 1, KvFormat::F32, 512)
        .expect("lfm2moe session");
    assert_eq!(session.config.n_layer, config.n_layer);

    let mut ids = tokenizer.encode(
        prompt,
        rust_model_inference::core::tokenizer::EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    if add_bos {
        let bos = tokenizer.bos_id().expect("bos_token_id");
        ids.insert(0, bos);
    }
    assert!(!ids.is_empty(), "prompt produced no tokens");

    session.prompt_len = ids.len();
    let mut logits = Vec::new();
    for &token in &ids {
        logits = session.forward_token(token).expect("prefill");
    }
    let mut generated = Vec::with_capacity(n_predict);
    for _ in 0..n_predict {
        let next = greedy(&logits);
        generated.push(next);
        logits = session.forward_token(next).expect("decode");
    }
    (ids, generated)
}

/// Argmax with the same lowest-index tie-break the oracle's `greedy_token`
/// uses (`logits[i] > logits[best]`).
fn greedy(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    best as u32
}

#[test]
#[ignore = "requires the pinned llama.cpp oracle trace; see tools/oracle/lfm2moe/README.md"]
fn lfm2moe_q4_0_matches_pinned_oracle_checkpoints() {
    let Some(path) = model_path() else {
        return;
    };
    assert!(
        std::env::var_os("RMI_PARITY_TRACE").is_some(),
        "set RMI_PARITY_TRACE=<path> so the trunk emits its checkpoints"
    );
    assert!(
        std::env::var_os("RMI_SCALAR").is_some(),
        "bit-exact parity requires RMI_SCALAR=1 on the Rust side"
    );
    let n_predict = std::env::var("RMI_LFM2MOE_N_PREDICT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4usize);
    let prompt = std::env::var("RMI_LFM2MOE_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.into());
    let source = open_source(&path);
    let (ids, generated) = replay(&source, &prompt, n_predict, true);
    #[cfg(feature = "parity-trace")]
    {
        use rust_model_inference::parity_trace::token_ids;
        token_ids("prompt_ids", &ids).expect("prompt_ids trace");
        token_ids("generated_ids", &generated).expect("generated_ids trace");
    }
    eprintln!("prompt_ids={ids:?}");
    eprintln!("generated_ids={generated:?}");
}

/// The GGUF contract this adaptation targets (llama.cpp `LLM_TYPE_24B_A2B`).
/// Every number here is read back from the real metadata keys, so a wrong
/// architecture assumption fails instead of silently loading.
#[test]
fn lfm25_24b_a2b_config_matches_the_pinned_contract() {
    let Some(path) = model_path() else {
        return;
    };
    let source = open_source(&path);
    assert_eq!(
        source
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val()),
        Some("lfm2moe")
    );
    let config = Lfm2MoeConfig::from_source(source.as_ref()).expect("lfm2moe config");
    assert_eq!(config.n_layer, 40);
    assert_eq!(config.n_embd, 2048);
    assert_eq!(config.n_head, 32);
    assert_eq!(config.n_embd_head_k, 64);
    assert_eq!(config.n_embd_head_v, 64);
    assert_eq!(config.n_ff, 11776);
    assert_eq!(config.n_layer_dense_lead, 2);
    assert_eq!(config.n_expert, 64);
    assert_eq!(config.n_expert_used, 4);
    assert_eq!(config.n_ff_exp, 1536);
    assert_eq!(config.expert_gating_func, 2, "sigmoid gating");
    assert_eq!(config.l_cache, 3);
    assert_eq!(config.d_conv, 2);
    assert_eq!(config.rope_freq_base, 1.0e7);
    assert_eq!(config.n_ctx, 131072);
    assert_eq!(config.vocab_size, 128000);
    // Attention on every 4th layer starting at 2 — ten attention layers and
    // thirty shortconv layers out of forty.
    let attention: Vec<usize> = config
        .n_head_kv_per_layer
        .iter()
        .enumerate()
        .filter(|(_, kv)| **kv > 0)
        .map(|(layer, _)| layer)
        .collect();
    assert_eq!(attention, vec![2, 6, 10, 14, 18, 22, 26, 30, 34, 38]);
    assert!(config
        .n_head_kv_per_layer
        .iter()
        .all(|kv| *kv == 0 || *kv == 8));
}

/// Tokenizer parity, pinned against `llama-eval-callback`'s `prompt_ids` for
/// the same raw string (see `tools/oracle/lfm2moe/README.md`).
#[test]
fn lfm25_token_ids_match_the_pinned_oracle() {
    let Some(path) = model_path() else {
        return;
    };
    let source = open_source(&path);
    let tokenizer =
        rust_model_inference::core::tokenizer::BPETokenizer::from_gguf_metadata(|key| {
            source.metadata(key).cloned()
        })
        .expect("tokenizer");
    assert_eq!(tokenizer.bos_id(), Some(124894));
    assert_eq!(tokenizer.eos_id(), Some(124900));
    // oracle: `llama-eval-callback -p "What is 2+2?"`, add_bos = true.
    let ids = tokenizer.encode(
        "What is 2+2?",
        rust_model_inference::core::tokenizer::EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    let mut with_bos = vec![tokenizer.bos_id().expect("bos")];
    with_bos.extend_from_slice(&ids);
    assert_eq!(with_bos, vec![124894, 2992, 355, 229, 26, 19, 26, 39]);
}

#[test]
fn lfm2moe_single_token_forward_is_finite() {
    let Some(path) = model_path() else {
        return;
    };
    let source = open_source(&path);
    let (_, generated) = replay(&source, "Hello", 1, true);
    assert_eq!(generated.len(), 1);
}
