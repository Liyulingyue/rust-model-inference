//! The batched prefill must produce exactly the same logits as the single
//! token walk.
//!
//! `MellumRunner::forward_tokens` evaluates a prompt as one batch: every
//! projection carries all the rows at once, so a weight matrix is read once
//! per prompt instead of once per token. The arithmetic must not change —
//! the batched path has to be bit-for-bit the same as
//! `MellumRunner::forward_token` run over the same tokens.
//!
//! The test is skipped (green) unless `RMI_MELLUM_Q8_MODEL` points at a real
//! Q8_0 GGUF, matching the other `*_reference.rs` harnesses in this repo.

use std::path::PathBuf;
use std::sync::Arc;

use rust_model_inference::core::scratchpad::KvFormat;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::models::mellum::{MellumConfig, MellumModel, MellumRunner};

const DEFAULT_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/Mellum2.1-12B-A2.5B-Thinking-GGUF/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf"
);

/// The prompt is long enough to cross the batch chunk boundary
/// (`MAX_BATCH_ROWS`), so the chunking is exercised too.
const DEFAULT_PROMPT: &str = "The history of the Roman empire is a long and complicated subject. \
The history of the Roman empire is a long and complicated subject. \
The history of the Roman empire is a long and complicated subject.";

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

/// Logits of the last prompt position, from the batched prefill and from the
/// single token walk.
fn both_paths(source: &Arc<dyn TensorSource>, prompt: &str) -> (Vec<f32>, Vec<f32>, usize) {
    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).expect("tokenizer"),
    );
    let config = MellumConfig::from_source(source.as_ref()).expect("config");
    let ids = tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    assert!(!ids.is_empty(), "prompt produced no tokens");
    let capacity = (ids.len() + 16).min(config.n_ctx);

    let pool = Arc::new(ComputePool::new(4));
    let model = MellumModel::from_source(Arc::clone(source), Arc::clone(&tokenizer), pool)
        .expect("load Mellum model");

    let mut batched = MellumRunner::new(&model, capacity, KvFormat::F32).expect("runner");
    batched.forward_tokens(&ids, 0).expect("batched prefill");
    let batched_logits = batched.logits().to_vec();

    let mut single = MellumRunner::new(&model, capacity, KvFormat::F32).expect("runner");
    for (position, &token) in ids.iter().enumerate() {
        single.forward_token(token, position).expect("forward");
    }
    let single_logits = single.logits().to_vec();

    (batched_logits, single_logits, ids.len())
}

#[test]
fn batched_prefill_matches_the_single_token_walk_bit_for_bit() {
    let Some(path) = model_path() else {
        return;
    };
    let source = open_source(&path);
    let prompt = std::env::var("RMI_MELLUM_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.into());
    let (batched, single, tokens) = both_paths(&source, &prompt);
    assert_eq!(batched.len(), single.len());
    assert!(!batched.is_empty());
    for (index, (left, right)) in batched.iter().zip(single.iter()).enumerate() {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "logit {index} differs between the batched prefill and the single token walk"
        );
    }
    eprintln!("compared {tokens} prompt tokens, {} logits", batched.len());
}

/// The batched prefill has to stay bit-identical when the device takes a
/// projection, so the device path is checked with the same comparison.
///
/// `RMI_MELLUM_DEVICE_PREFILL=1` hands the large projections to the Vulkan
/// device; without a device the test still runs, on the CPU path.
#[test]
#[ignore = "needs the Vulkan device path; run with --ignored"]
fn batched_prefill_matches_the_single_token_walk_on_the_device() {
    let Some(path) = model_path() else {
        return;
    };
    let source = open_source(&path);
    let prompt = std::env::var("RMI_MELLUM_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.into());
    let (batched, single, tokens) = both_paths(&source, &prompt);
    assert_eq!(batched.len(), single.len());
    for (index, (left, right)) in batched.iter().zip(single.iter()).enumerate() {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "logit {index} differs between the batched prefill and the single token walk"
        );
    }
    eprintln!("compared {tokens} prompt tokens, {} logits", batched.len());
}
