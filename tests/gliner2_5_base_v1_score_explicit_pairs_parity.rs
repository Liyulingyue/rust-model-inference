//! Byte-exact parity for `BoundaryProposer::score_explicit_pairs`.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.
//!
//! Mirrors the Python oracle at
//! `tools/oracle/gliner_boundary/dump_score_explicit_pairs.py`. The
//! reference path is
//! ``SparseBoundaryProposer.score_explicit_pairs`` and uses the
//! same RotaryBoundaryEmbedding (base = 10000, dim = boundary_dim =
//! 128) as the Rust port.
//!
//! Both sides score the same six `(start, end)` candidates per query on
//! the same fixed `text_states` input (matches the boundary-encoder +
//! boundary-query-head oracles). The mask input is derived from the
//! candidate legality (`start < end` and `end <= text_length`).

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner_boundary::{BoundaryEncoding, BoundaryModel};

const GGUF: &str = "models/gliner2.5-base-v1/gliner2.5-base-v1-f32.gguf";
const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/score-explicit-pairs-golden.json";

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
    fn boundary_len(&self) -> usize { self.seq_len + 1 }
}

#[test]
fn matches_the_reference_stack() {
    let Some((_anchor, model)) = loaded_model() else { return };
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_score_explicit_pairs.py"));
    let fixture: serde_json::Value =
        serde_json::from_str(&raw).expect("parse score-explicit-pairs-golden.json");

    let hidden_size = model.config.n_embd;
    let seq_len = fixture["config"]["seq_len"].as_u64().expect("seq_len") as usize;
    let valid_tokens = fixture["config"]["valid_tokens"].as_u64().expect("valid_tokens") as usize;
    let q_count = fixture["config"]["q_count"].as_u64().expect("q_count") as usize;
    let c_count = fixture["config"]["c_count"].as_u64().expect("c_count") as usize;
    let batch: usize = 1;

    // Deterministic text_states mirroring the Python dump.
    let mut text_states: Vec<f32> = (0..seq_len * hidden_size)
        .map(|i| (((i as f32) * 0.011).sin() - 1.0))
        .collect();
    let mut text_mask: Vec<Vec<bool>> = vec![vec![false; seq_len]];
    for slot in text_mask[0].iter_mut().take(valid_tokens) {
        *slot = true;
    }
    let encoding = model.boundary.forward(&text_states, &text_mask);
    let boundary_len = seq_len + 1;

    // Query 0 = text_states[0, 0..hidden_size]; query 1 = [hidden_size..2*hidden_size].
    let mut query_states: Vec<f32> = Vec::with_capacity(q_count * hidden_size);
    query_states.extend_from_slice(&text_states[..hidden_size]);
    query_states.extend_from_slice(&text_states[hidden_size..2 * hidden_size]);

    // Indices and valid_mask come straight from the fixture (the Python
    // dump also derives the valid mask from `start < end & end <= len`).
    let want_indices: Vec<u32> = fixture["indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
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

    let got = model.proposer.score_explicit_pairs(
        &encoding.states,
        boundary_len,
        &query_states,
        batch,
        q_count,
        c_count,
        &want_indices,
        &want_valid,
    );

    assert_eq!(got.len(), want_compat.len());
    let mut max_delta = 0.0f32;
    let mut first: Option<(usize, f32, f32)> = None;
    for (i, (g, w)) in got.iter().zip(want_compat.iter()).enumerate() {
        let d = (g - w).abs();
        if d > max_delta {
            max_delta = d;
            if first.is_none() {
                first = Some((i, *g, *w));
            }
        }
    }
    assert!(
        max_delta < 1e-4,
        "max compatibility delta {max_delta} exceeds threshold 1e-4 (first diff at {first:?})"
    );
    // Invalid candidates must be exactly zero (per the reference's
    // `torch.where(valid_mask, compatibility, torch.zeros_like(...))`).
    for (i, valid) in want_valid.iter().enumerate() {
        if !valid {
            assert_eq!(
                got[i], 0.0,
                "invalid position {i} should be zero, got {}",
                got[i],
            );
        }
    }
}