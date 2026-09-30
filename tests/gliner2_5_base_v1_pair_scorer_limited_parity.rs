//! Byte-exact parity for the LIMITED `PairScorer.forward`.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_pair_scorer_limited.py`,
//! which constructs the GLiNER2 reference with
//! `enable_span_content=False, use_inside_evidence=False,
//! endpoint_difference_features=False` to match the Rust implementation
//! (those three feature sources are deferred per glinerTODO.md).
//!
//! Pipeline (limited, byte-exact verifiable):
//!  - DeBERTa-v3-base encoder → boundary states via BoundaryEncoder.
//!  - BoundaryQueryHead → per-query start/end logits + inside prefix.
//!  - BoundaryProposer.score_explicit_pairs → compatibility prior.
//!  - PairScorer.forward → final per-candidate score
//!    (start marginal + end marginal + endpoint compat + length features).
//!
//! The classifier (classifier.0 + classifier.3 with GeLU between) is
//! applied later by the higher-level pipeline; this oracle stops at
//! the PairScorer output.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner_boundary::{BoundaryEncoding, BoundaryModel};

const GGUF: &str = "models/gliner2.5-base-v1/gliner2.5-base-v1-f32.gguf";
const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/pair-scorer-limited-golden.json";

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

trait BoundaryEncodingExt {
    fn boundary_len(&self) -> usize;
}
impl BoundaryEncodingExt for BoundaryEncoding {
    fn boundary_len(&self) -> usize {
        self.seq_len + 1
    }
}

#[test]
fn matches_the_reference_stack() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let raw = std::fs::read_to_string(FIXTURE).unwrap_or_else(|_| {
        panic!("missing {FIXTURE}; regenerate with dump_pair_scorer_limited.py")
    });
    let fixture: serde_json::Value =
        serde_json::from_str(&raw).expect("parse pair-scorer-limited-golden.json");

    let hidden_size = model.config.n_embd;
    let seq_len = fixture["config"]["seq_len"].as_u64().expect("seq_len") as usize;
    let valid_tokens = fixture["config"]["valid_tokens"]
        .as_u64()
        .expect("valid_tokens") as usize;
    let q_count = fixture["config"]["q_count"].as_u64().expect("q_count") as usize;
    let c_count = fixture["config"]["c_count"].as_u64().expect("c_count") as usize;
    let batch: usize = 1;

    // Build text_states, run BoundaryEncoder + BoundaryQueryHead.
    let mut text_states: Vec<f32> = (0..seq_len * hidden_size)
        .map(|i| (((i as f32) * 0.011).sin() - 1.0))
        .collect();
    let mut text_mask: Vec<Vec<bool>> = vec![vec![false; seq_len]];
    for slot in text_mask[0].iter_mut().take(valid_tokens) {
        *slot = true;
    }
    let encoding = model.boundary.forward(&text_states, &text_mask);
    let boundary_len = seq_len + 1;

    let mut query_states: Vec<f32> = Vec::with_capacity(q_count * hidden_size);
    query_states.extend_from_slice(&text_states[..hidden_size]);
    query_states.extend_from_slice(&text_states[hidden_size..2 * hidden_size]);
    let query_mask: Vec<Vec<bool>> = vec![vec![true; q_count]];

    let marginals = model.query_head.forward(
        &encoding.states,
        &encoding.mask,
        &text_states,
        &text_mask,
        &query_states,
        &query_mask,
    );

    // Compatibility prior via BoundaryProposer.
    let want_indices: Vec<usize> = fixture["indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let want_valid: Vec<bool> = fixture["valid_mask"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    let want_compat: Vec<f32> = fixture["compatibility"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let compat_logits = model.proposer.score_explicit_pairs(
        &encoding.states,
        boundary_len,
        &query_states,
        batch,
        q_count,
        c_count,
        &want_indices.iter().map(|&i| i as u32).collect::<Vec<_>>(),
        &want_valid,
    );

    let text_lengths: Vec<usize> = vec![valid_tokens];
    let scores = model.pair_scorer.forward(
        &encoding.states,
        boundary_len,
        &query_states,
        batch,
        q_count,
        c_count,
        &marginals.start_logits,
        &marginals.end_logits,
        &compat_logits,
        &want_indices,
        &text_lengths,
        &want_valid,
    );

    let want_scores: Vec<f32> = fixture["scores"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(scores.len(), want_scores.len());
    let mut max_delta = 0.0f32;
    let mut first: Option<(usize, f32, f32)> = None;
    for (i, (g, w)) in scores.iter().zip(want_scores.iter()).enumerate() {
        let d = (g - w).abs();
        if d > max_delta {
            max_delta = d;
            if first.is_none() {
                first = Some((i, *g, *w));
            }
        }
    }
    assert!(
        max_delta < 1e-3,
        "max pair-scorer delta {max_delta} exceeds threshold 1e-3 (first diff at {first:?})"
    );
    // MASK_LOGIT sentinel must match on invalid candidates.
    for (i, valid) in want_valid.iter().enumerate() {
        if !valid {
            assert!(
                scores[i] <= -1.0e3,
                "invalid position {i} should be <= MASK_LOGIT, got {}",
                scores[i],
            );
        }
    }
    let _ = want_compat;
}
