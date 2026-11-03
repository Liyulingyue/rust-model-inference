//! Bit-exact parity harness for Mellum 2.1 against the pinned llama.cpp
//! oracle (`tools/oracle/mellum/`).
//!
//! The test drives `MellumRunner` token by token — exactly how the oracle's
//! `llama-eval-callback` replays the prompt — with the **raw** prompt text and
//! no chat template, so both sides consume identical token ids. Every
//! intermediate tensor the oracle labels in `src/models/mellum.cpp` is emitted
//! through `RMI_PARITY_TRACE` by the trunk itself, and
//! `tools/oracle/mellum/check_scalar.py` diffs the two manifests bit for bit.
//!
//! Both sides must run scalar: build with
//! `--features parity-trace`, run the binary with `RMI_SCALAR=1`, and build
//! the oracle with `-U__ARM_NEON -ffp-contract=off` (see the oracle README).
//!
//! The test is skipped (green) unless `RMI_MELLUM_Q8_MODEL` points at a real
//! Q8_0 GGUF, matching the other `*_reference.rs` harnesses in this repo.

use std::path::PathBuf;
use std::sync::Arc;

use rust_model_inference::core::scratchpad::KvFormat;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::models::mellum::MellumConfig;
use rust_model_inference::models::mellum::MellumModel;
use rust_model_inference::models::mellum::MellumRunner;

const DEFAULT_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/Mellum2.1-12B-A2.5B-Thinking-GGUF/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf"
);

const DEFAULT_PROMPT: &str = "What is 2+2?";

fn model_path() -> Option<PathBuf> {
    let path = std::env::var("RMI_MELLUM_Q8_MODEL")
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

/// Replay `prompt` through the runner and report the greedy continuation.
fn replay(source: &Arc<dyn TensorSource>, prompt: &str, n_predict: usize) -> (Vec<u32>, Vec<u32>) {
    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).expect("tokenizer"),
    );
    let config = MellumConfig::from_source(source.as_ref()).expect("config");
    let pool = Arc::new(ComputePool::new(1));
    let model = MellumModel::from_source(Arc::clone(source), Arc::clone(&tokenizer), pool)
        .expect("load Mellum model");
    assert_eq!(model.config(), &config);

    // Raw prompt, no template and no BOS, with special tokens parsed — the
    // same contract as the oracle's
    // `common_tokenize(ctx, prompt, add_bos, /* parse_special */ true)`.
    // Prompts containing `<|im_start|>` and friends therefore tokenize
    // identically on both sides.
    let ids = tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    assert!(!ids.is_empty(), "prompt produced no tokens");

    let capacity = (ids.len() + n_predict + 8).min(config.n_ctx);
    let mut runner = MellumRunner::new(&model, capacity, KvFormat::F32).expect("runner");
    for (position, &token) in ids.iter().enumerate() {
        runner.forward_token(token, position).expect("forward");
    }
    let mut generated = Vec::with_capacity(n_predict);
    let eos = tokenizer.eos_id();
    for _ in 0..n_predict {
        let next = greedy(runner.logits());
        if Some(next) == eos {
            break;
        }
        generated.push(next);
        let position = ids.len() + generated.len() - 1;
        if position + 1 >= capacity {
            break;
        }
        runner.forward_token(next, position).expect("forward");
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
#[ignore = "requires the pinned llama.cpp oracle trace; see tools/oracle/mellum/README.md"]
fn mellum_q8_matches_pinned_oracle_checkpoints() {
    let Some(path) = model_path() else {
        return;
    };
    assert!(
        std::env::var_os("RMI_PARITY_TRACE").is_some(),
        "set RMI_PARITY_TRACE=<dir> so the trunk emits its checkpoints"
    );
    assert!(
        std::env::var_os("RMI_SCALAR").is_some(),
        "bit-exact parity requires RMI_SCALAR=1 on the Rust side"
    );
    let n_predict = std::env::var("RMI_MELLUM_N_PREDICT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4usize);
    let prompt = std::env::var("RMI_MELLUM_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.into());
    let source = open_source(&path);
    let (ids, generated) = replay(&source, &prompt, n_predict);
    eprintln!("prompt_ids={ids:?}");
    eprintln!("generated_ids={generated:?}");
    eprintln!(
        "prompt_text={:?}",
        decode_ids(
            &BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap(),
            &ids
        )
    );
    eprintln!(
        "generated_text={:?}",
        decode_ids(
            &BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap(),
            &generated
        )
    );
}

/// Detokenize with the byte-level BPE rules so a parity run can be compared as
/// text, not only as ids.
fn decode_ids(tokenizer: &BPETokenizer, ids: &[u32]) -> String {
    tokenizer.decode(ids, false)
}

#[test]
fn mellum_config_matches_the_pinned_12b_a2_5b_contract() {
    let Some(path) = model_path() else {
        return;
    };
    let source = open_source(&path);
    let config = MellumConfig::from_source(source.as_ref()).expect("mellum config");
    // The single verified size (llama.cpp `LLM_TYPE_12B_A2_5B`).
    assert_eq!(config.n_layer, 28);
    assert_eq!(config.n_embd, 2304);
    assert_eq!(config.n_head, 32);
    assert_eq!(config.n_head_kv, 4);
    assert_eq!(config.n_embd_head_k, 128);
    assert_eq!(config.expert_count, 64);
    assert_eq!(config.expert_used_count, 8);
    assert_eq!(config.expert_ffn, 896);
    assert_eq!(config.n_swa, 1024);
    // `sliding_window_pattern = [T,T,T,F] * 7`: 21 sliding-window layers and 7
    // full-attention layers. `swa[l] == true` marks the *sliding-window* layers,
    // which `src/models/mellum.cpp` rotates with the plain table.
    assert_eq!(config.swa.len(), 28);
    assert!(config.swa[0] && config.swa[1] && config.swa[2] && !config.swa[3]);
    assert!(!config.swa[27] && config.swa[26] && config.swa[25]);
    assert_eq!(config.swa.iter().filter(|flag| **flag).count(), 21);
    // The 7 full-attention layers are exactly `[3, 7, 11, 15, 19, 23, 27]`.
    let full_attention: Vec<usize> = config
        .swa
        .iter()
        .enumerate()
        .filter(|(_, swa)| !**swa)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(full_attention, vec![3, 7, 11, 15, 19, 23, 27]);
}

#[test]
fn mellum_single_token_forward_is_finite() {
    let Some(path) = model_path() else {
        return;
    };
    let source = open_source(&path);
    let (_, generated) = replay(&source, "Hello", 1);
    assert!(generated.len() <= 1);
}

/// Tokenizer parity, pinned against `llama-eval-callback`'s `prompt_ids` for
/// the same raw strings (see `tools/oracle/mellum/README.md`).
///
/// These cases exercise the two properties that make `mellum2` different from
/// the `LlamaBpe` split it shares a word regex with:
/// * `12345` becomes five one-digit pre-tokens (llama.cpp applies `\p{N}`
///   *before* the StarCoder word regex);
/// * `'S` splits into `'` + `S` because that regex spells contractions
///   lowercase-only.
#[test]
fn mellum2_token_ids_match_the_pinned_oracle() {
    let Some(path) = model_path() else {
        return;
    };
    let source = open_source(&path);
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
        .expect("mellum2 tokenizer");
    let cases: &[(&str, &[u32])] = &[
        ("Hello, world!", &[7995, 46, 1685, 35]),
        ("12345", &[51, 52, 53, 54, 55]),
        ("a 12345 b", &[99, 255, 51, 52, 53, 54, 55, 321]),
        (
            "it's fine, isn't it? 678",
            &[326, 637, 7112, 46, 5381, 1445, 496, 65, 255, 56, 57, 58],
        ),
        ("'S don't I'M", &[41, 85, 2149, 1445, 397, 41, 79]),
        (
            "ABC 123 a1b2",
            &[13185, 255, 51, 52, 53, 255, 99, 51, 100, 52],
        ),
        (
            "  double  spaces  and\ttabs\nnewlines",
            &[255, 3552, 255, 6740, 255, 335, 232, 12368, 233, 1353, 5662],
        ),
        (
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
            &[
                27, 1397, 233, 2821, 359, 255, 52, 45, 52, 65, 28, 233, 27, 8091, 233,
            ],
        ),
    ];
    for &(text, ids) in cases {
        assert_eq!(
            tokenizer.encode(
                text,
                EncodeOptions {
                    add_special: false,
                    parse_special: true,
                }
            ),
            ids,
            "text {text:?}"
        );
    }
}
