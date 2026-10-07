//! Smoke test for the converted openjev 0.8B NLI GGUF.
//!
//! Loads the GGUF, tokenizes a premise/hypothesis pair with the standard
//! NLI template, runs `run_classify_qwen35_with_batch`, applies softmax,
//! and prints per-class probabilities. Designed for eyeballing against
//! `OpenJevCrossEncoder.predict(...)` from `modeling_openjev.py`.
//!
//! **Caveat**: `Qwen3.5Model::forward` has accumulated numerical drift
//! even at modest depth — `tests/qwen35_reference.rs` budgets
//! `attn_norm-63` abs_tol=1.25 and `layer_output-63` abs_tol=2.5, which
//! is large enough that the NLI argmax flips on a fair fraction of
//! samples. The 0.8B model has 24 layers so the drift is smaller; the
//! text below shows the full output for reference but the smoke test
//! does NOT assert argmax correctness.

use std::sync::Arc;

use rust_model_inference::core::tokenizer::BPETokenizer;
use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::models::qwen35::trunk::run_classify_qwen35_with_batch;

const GGUF_PATH: &str =
    "/home/liyulingyue/Codes/rust-model-inference/models/openjev-0.8b-nli-probe/Qwen3.5-0.8B-NLI-v2s-long-F32.gguf";

fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

fn fmt_f(x: f32, decimals: usize) -> String {
    format!("{:.*}", decimals, x)
}

fn main() {
    let source: Arc<dyn rust_model_inference::core::tensor::TensorSource> = Arc::from(
        open_model_source(std::path::Path::new(GGUF_PATH), ComponentRole::Llm).expect("load GGUF"),
    );
    let tokenizer =
        BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned()).expect("tokenizer");

    let samples: &[(&str, &str)] = &[
        (
            "The bird is below the centre of the gap.",
            "The bird is 0.05 below the centre of the gap.",
        ),
        ("A man is playing a guitar.", "Someone is making music."),
        (
            "A photograph of a scene: There is no dog in this image.",
            "There is a dog in the image.",
        ),
        ("The cat is on the mat.", "The sky is blue."),
    ];
    let labels = ["contradiction", "entailment", "neutral"];

    println!("=== openjev 0.8B NLI smoke test ===");
    println!("GGUF: {GGUF_PATH}");
    println!();

    for (i, (premise, hypothesis)) in samples.iter().enumerate() {
        let text = format!(
            "Premise: {}\nHypothesis: {}",
            premise.trim(),
            hypothesis.trim()
        );
        let token_ids = tokenizer.encode(
            &text,
            rust_model_inference::core::tokenizer::EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        );
        let (raw_logits, duration) = run_classify_qwen35_with_batch(
            source.as_ref(),
            &token_ids,
            4,
            rust_model_inference::app::cli::KvFormat::F16,
            8192,
            64,
        )
        .expect("classify");
        let probs = softmax(&raw_logits);
        let am = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        println!(
            "sample {i}: argmax={} probs=[con={}, ent={}, neu={}] logits={:?} ({}ms)",
            labels[am],
            fmt_f(probs[0], 4),
            fmt_f(probs[1], 4),
            fmt_f(probs[2], 4),
            raw_logits,
            fmt_f(duration.as_secs_f64() as f32, 1),
        );
    }
}
