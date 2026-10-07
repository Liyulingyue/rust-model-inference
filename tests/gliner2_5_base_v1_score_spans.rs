//! Test the top-level `score_spans` API end-to-end.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.
//!
//! Exercises the full limited pipeline (BoundaryEncoder + QueryHead +
//! Proposer + PairScorer) through the `score_spans` convenience wrapper
//! and checks the shapes / masking / determinism. Byte-exactness of each
//! individual stage is covered by the per-stage parity tests; this test
//! covers composition.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner_boundary::{score_spans, BoundaryModel};

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF").map(std::path::PathBuf::from)
}

fn loaded_model() -> Option<(Box<dyn std::any::Any>, BoundaryModel<'static>)> {
    let path = match gguf_path() {
        Some(path) => path,
        None => {
            eprintln!("skipping: set RMI_GLINER2_5_BASE_V1_GGUF to enable this test");
            return None;
        }
    };
    if !path.exists() {
        eprintln!("skipping: {} does not exist", path.display());
        return None;
    }
    let source = GGUFLoader::from_file(&path).expect("open boundary GGUF");
    let leaked: &'static dyn TensorSource = Box::leak(Box::new(source));
    let model = BoundaryModel::from_source(leaked).expect("load boundary model");
    Some((Box::new(()), model))
}

#[test]
fn score_spans_runs_end_to_end() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let hidden_size = model.config.n_embd;
    let seq_len = 8;
    let valid_tokens = 6;
    let q_count = 2;
    let c_count = 6;

    let text_states: Vec<f32> = (0..seq_len * hidden_size)
        .map(|i| (i as f32 * 0.011).sin() - 1.0)
        .collect();
    let mut text_mask: Vec<Vec<bool>> = vec![vec![false; seq_len]];
    for slot in text_mask[0].iter_mut().take(valid_tokens) {
        *slot = true;
    }
    let mut query_states: Vec<f32> = Vec::with_capacity(q_count * hidden_size);
    query_states.extend_from_slice(&text_states[..hidden_size]);
    query_states.extend_from_slice(&text_states[hidden_size..2 * hidden_size]);
    let query_mask: Vec<Vec<bool>> = vec![vec![true; q_count]];

    // Same candidate layout as the per-stage oracles.
    let indices: Vec<usize> = vec![
        // q = 0
        0, 2, 1, 4, 3, 6, 5, 5, 4, 2, 7, 8, // q = 1
        0, 2, 1, 4, 3, 6, 5, 5, 4, 2, 7, 8,
    ];
    let valid: Vec<bool> = vec![
        true, true, true, false, false, false, // q = 0
        true, true, true, false, false, false, // q = 1
    ];

    let spans = score_spans(
        &model,
        &text_states,
        &text_mask,
        &query_states,
        &query_mask,
        &indices,
        &valid,
        1,
        q_count,
        c_count,
    );

    assert_eq!(spans.len(), q_count * c_count);
    // Indices round-trip.
    for (i, span) in spans.iter().enumerate() {
        assert_eq!(span.start, indices[i * 2]);
        assert_eq!(span.end, indices[i * 2 + 1]);
    }
    // Valid candidates carry finite logits; invalid ones carry MASK_LOGIT.
    for (i, span) in spans.iter().enumerate() {
        if valid[i] {
            assert!(span.logit.is_finite(), "valid candidate {i} not finite");
        } else {
            assert!(
                span.logit <= -1.0e3,
                "invalid candidate {i} should be <= MASK_LOGIT, got {}",
                span.logit,
            );
        }
    }
    // Determinism: a second run produces identical bits.
    let spans2 = score_spans(
        &model,
        &text_states,
        &text_mask,
        &query_states,
        &query_mask,
        &indices,
        &valid,
        1,
        q_count,
        c_count,
    );
    for (a, b) in spans.iter().zip(spans2.iter()) {
        assert_eq!(a.logit.to_bits(), b.logit.to_bits(), "non-deterministic");
    }
}
