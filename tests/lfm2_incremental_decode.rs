//! The incremental decode path must agree with the re-prefill path.
//!
//! `tests/lfm2_dense_trunk_agreement.rs` verifies the dense trunk by
//! re-prefilling the whole prefix at every greedy step. That is not the path the
//! CLI takes: `run_inference_stream` walks one token at a time, carrying the KV
//! cache and the shortconv state forward. The two only coincide if the state
//! handling is right, and that is exactly where the LFM2-MoE shortconv bug hid
//! (the conv state advanced only on decode steps, so the first generated token
//! convolved a window that was one step stale).
//!
//! This drives the real incremental path in-process, reads the `prompt_ids` and
//! `generated_ids` it records under `RMI_PARITY_TRACE`, and replays those same
//! ids through the prefill path. Equal continuations mean the incremental path
//! reproduces the arithmetic the oracle-verified prefill path produces.
//!
//! Requires the `parity-trace` feature; without it the incremental run records
//! nothing and the test reports what it could not check.

use std::path::PathBuf;
use std::sync::Arc;

use rust_model_inference::core::scratchpad::KvFormat;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::lfm2::Lfm2StreamItem;
use rust_model_inference::models::lfm2::{
    run_forward_logits_lfm2_with_batch, run_inference_stream,
};

const DEFAULT_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/LFM2.5-230M-GGUF/LFM2.5-230M-Q4_K_M.gguf"
);

const MAX_TOKENS: usize = 12;

fn model_path() -> Option<PathBuf> {
    let path = std::env::var("RMI_LFM2_DENSE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_MODEL));
    if path.exists() {
        Some(path)
    } else {
        eprintln!("skipping: {} not present", path.display());
        None
    }
}

/// Minimal reader for the two `token_ids` records the trace writes.
fn read_ids(path: &PathBuf, name: &str) -> Option<Vec<u32>> {
    let text = std::fs::read_to_string(path).ok()?;
    for line in text.lines() {
        let Some(rest) = line.split_once("\"name\":") else {
            continue;
        };
        let quoted = rest.1.split('"').nth(1)?;
        if quoted != name {
            continue;
        }
        let array = line.split("\"token_ids\":").nth(1)?;
        let array = array.split('[').nth(1)?.split(']').next()?;
        let ids: Vec<u32> = array
            .split(',')
            .filter_map(|part| part.trim().parse().ok())
            .collect();
        return Some(ids);
    }
    None
}

#[test]
fn incremental_decode_matches_reprefill() {
    let Some(path) = model_path() else { return };

    if std::env::var_os("RMI_PARITY_TRACE").is_none() {
        panic!(
            "set RMI_PARITY_TRACE to a writable file path; the incremental path \
             records its token ids there only when the parity-trace feature is on"
        );
    }
    let manifest = PathBuf::from(std::env::var_os("RMI_PARITY_TRACE").unwrap());
    let _ = std::fs::remove_file(&manifest);

    let loader =
        rust_model_inference::core::loader::GGUFLoader::from_file(&path).expect("open GGUF");
    let source: Arc<dyn TensorSource> = Arc::new(loader);

    // Build the prompt the same way the CLI does, through the real chat
    // builder, so the incremental run sees the token stream a user gets.
    let tokenizer =
        rust_model_inference::core::tokenizer::BPETokenizer::from_gguf_metadata(|key| {
            source.metadata(key).cloned()
        })
        .expect("tokenizer");
    let prompt = "What is 2+2? Answer briefly.";
    let prompt_ids = rust_model_inference::prompt::build_lfm2_chat_prompt(
        &tokenizer,
        &[rust_model_inference::prompt::Lfm2Message {
            role: "user",
            content: prompt,
        }],
    )
    .expect("prompt");

    // Greedy, single thread: thread count can change reduction order.
    let stream: Vec<Lfm2StreamItem> = prompt_ids
        .iter()
        .map(|&id| Lfm2StreamItem::Token(id))
        .collect();
    run_inference_stream(
        source.as_ref(),
        stream,
        MAX_TOKENS,
        0.0,
        1,
        false,
        KvFormat::F32,
        512,
        1.0,
    )
    .expect("incremental inference");

    let incremental = read_ids(&manifest, "generated_ids")
        .expect("generated_ids missing: was the crate built with --features parity-trace?");
    let recorded_prompt = read_ids(&manifest, "prompt_ids").expect("prompt_ids missing");

    assert_eq!(
        recorded_prompt, prompt_ids,
        "the incremental path consumed a different prompt than the builder produced"
    );
    assert!(
        !incremental.is_empty(),
        "the incremental path generated nothing (hit EOS on the first step?)"
    );

    // Replay the same ids through the verified prefill path.
    let mut ids = prompt_ids.clone();
    let mut expected = Vec::new();
    for _ in 0..incremental.len() {
        let logits =
            run_forward_logits_lfm2_with_batch(source.as_ref(), &ids, 1, KvFormat::F32, 512, 1)
                .expect("prefill inference")
                .0;
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index as u32)
            .expect("non-empty logits");
        expected.push(next);
        ids.push(next);
    }

    assert_eq!(
        incremental, expected,
        "incremental decode and re-prefill disagree: the decode path's KV or \
         shortconv state handling differs from the verified prefill path"
    );
    eprintln!("incremental decode == re-prefill: {incremental:?}");
}
