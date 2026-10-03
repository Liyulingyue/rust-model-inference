//! Byte-exact parity for `DocumentCandidatePool` — the shared span pool.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_document_candidate_pool.py`,
//! which builds the reference `DocumentCandidatePool` from the checkpoint's own
//! settings and calls it exactly as `BoundaryHead.forward` does at inference:
//! no gold injection, no stats.
//!
//! The pool is a discrete algorithm on top of continuous scores, so a mismatch
//! usually means a tie was broken differently rather than a rounding slip.
//! The fixture therefore compares the selected `(start, end)` pairs, not just
//! the scores, and reports which slot first diverged.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner_boundary::BoundaryModel;

const GGUF: &str = "models/gliner2.5-base-v1/gliner2.5-base-v1-f32.gguf";
const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/document-candidate-pool-golden.json";

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF").map(std::path::PathBuf::from)
}

fn loaded_model() -> Option<BoundaryModel<'static>> {
    let path = gguf_path()?;
    if !path.exists() {
        eprintln!("skipping: {} does not exist", path.display());
        return None;
    }
    let source = GGUFLoader::from_file(&path).expect("open boundary GGUF");
    let leaked: &'static dyn TensorSource = Box::leak(Box::new(source));
    Some(BoundaryModel::from_source(leaked).expect("load boundary model"))
}

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE).unwrap_or_else(|_| {
        panic!("missing {FIXTURE}; regenerate with dump_document_candidate_pool.py")
    });
    serde_json::from_str(&raw).expect("parse document-candidate-pool-golden.json")
}

fn f32s(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn usize_list(value: &serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

fn bool_list(value: &serde_json::Value) -> Vec<bool> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect()
}

/// The pool settings the fixture was generated with, read from the GGUF rather
/// than hardcoded here.
#[test]
fn gguf_carries_the_pool_settings() {
    let Some(model) = loaded_model() else {
        return;
    };
    let fixture = fixture();
    let want = &fixture["config"];
    let settings = &model.settings;
    assert!(
        settings.uses_shared_pool(),
        "candidate_pool = {:?}; the shared pool is the mainline only for \"shared\"",
        settings.candidate_pool
    );
    assert_eq!(
        model.pool_builder.pool_boundary_top_k,
        want["pool_boundary_top_k"].as_u64().unwrap() as usize
    );
    assert_eq!(
        model.pool_builder.pool_size,
        want["pool_size"].as_u64().unwrap() as usize
    );
    assert_eq!(
        model.pool_builder.min_pool_per_query,
        want["min_pool_per_query"].as_u64().unwrap() as usize
    );
    assert_eq!(
        model.pool_builder.boundary_dim,
        want["boundary_dim"].as_u64().unwrap() as usize
    );
    // Both attention stacks are off for base-v1, which is what lets the Rust
    // port skip them entirely.
    assert_eq!(settings.candidate_attention_layers, 0);
    assert_eq!(settings.query_attention_layers, 0);
}

#[test]
fn pool_matches_the_reference() {
    let Some(model) = loaded_model() else {
        return;
    };
    let fixture = fixture();
    for (case_index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let seq_len = case["seq_len"].as_u64().unwrap() as usize;
        let q_count = case["q_count"].as_u64().unwrap() as usize;

        let text_states = f32s(&case["text_states"]);
        let flat_mask = bool_list(&case["text_mask"]);
        let text_mask = vec![flat_mask[..seq_len].to_vec()];
        let query_states = f32s(&case["query_states"]);
        let query_mask = vec![bool_list(&case["query_mask"])];

        let encoding = model.boundary.forward(&text_states, &text_mask);
        let marginals = model.query_head.forward(
            &encoding.states,
            &encoding.mask,
            &text_states,
            &text_mask,
            &query_states,
            &query_mask,
        );

        let pooled = model.pool_builder.build(
            &encoding.states,
            &encoding.mask,
            &query_mask,
            &marginals.start_logits,
            &marginals.end_logits,
        );

        let want_indices = usize_list(&case["indices"]);
        let want_mask = bool_list(&case["mask"]);
        let want_proposal = f32s(&case["proposal_logits"]);
        let want_compat = f32s(&case["compat_logits"]);

        assert_eq!(
            pooled.pool_size,
            want_mask.len(),
            "case {case_index}: pool width mismatch (the padded C)"
        );
        assert_eq!(pooled.indices.len(), want_indices.len());
        assert_eq!(pooled.mask.len(), want_mask.len());
        assert_eq!(pooled.proposal_logits.len(), want_proposal.len());
        assert_eq!(pooled.compat_logits.len(), want_compat.len());

        // A wrong tie-break shows up as a different (start, end) pair, so
        // report the first such slot rather than only the score delta.
        for slot in 0..pooled.pool_size {
            let i = slot;
            assert_eq!(
                pooled.indices[i * 2],
                want_indices[i * 2],
                "case {case_index} slot {i}: start diverged (tie-break?)"
            );
            assert_eq!(
                pooled.indices[i * 2 + 1],
                want_indices[i * 2 + 1],
                "case {case_index} slot {i}: end diverged (tie-break?)"
            );
            assert_eq!(
                pooled.mask[i], want_mask[i],
                "case {case_index} slot {i}: mask diverged"
            );
        }

        let mut worst_proposal = 0.0f32;
        let mut worst_compat = 0.0f32;
        let mut worst_at = 0usize;
        for i in 0..pooled.pool_size {
            let dp = (pooled.proposal_logits[i] - want_proposal[i]).abs();
            let dc = (pooled.compat_logits[i] - want_compat[i]).abs();
            if dp > worst_proposal {
                worst_proposal = dp;
                worst_at = i;
            }
            worst_compat = worst_compat.max(dc);
        }
        assert!(
            worst_proposal < 1e-5,
            "case {case_index}: max proposal delta {worst_proposal} at slot {worst_at}"
        );
        assert!(
            worst_compat < 1e-5,
            "case {case_index}: max compat delta {worst_compat}"
        );

        // Padding slots must carry MASK_LOGIT as their proposal logit.
        for (i, valid) in pooled.mask.iter().enumerate() {
            if !valid {
                assert!(
                    pooled.proposal_logits[i] <= -1.0e3,
                    "case {case_index} slot {i}: padding proposal logit should be MASK_LOGIT, got {}",
                    pooled.proposal_logits[i]
                );
            }
        }
        assert!(
            pooled.mask.iter().any(|m| *m),
            "case {case_index}: pool produced no candidates"
        );
    }
}
