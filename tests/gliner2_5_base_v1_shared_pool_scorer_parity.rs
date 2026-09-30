//! Byte-exact parity for `SharedPoolScorer`.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_shared_pool_scorer.py`, which
//! calls the reference scorer exactly as `BoundaryHead.forward` does on the
//! shared-pool branch.
//!
//! Both returned tensors matter: `pair_logits` in the reference's
//! candidate-major `[B, C, Q]` order (the public contract is `[B, Q, C]`, and
//! `PooledCandidates::to_candidate_batch` does the transpose), and
//! `candidate` in `[B, C, pair_dim]` order — that one feeds the record head's
//! `candidate_encoder` and the optional candidate-attention layers, so it is
//! compared for a sample of valid slots rather than taken on faith.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner_boundary::pool::SharedPoolInputs;
use rust_model_inference::models::gliner_boundary::{score_document_candidates, BoundaryModel};

const GGUF: &str = "models/gliner2.5-base-v1/gliner2.5-base-v1-f32.gguf";
const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/shared-pool-scorer-golden.json";

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
        panic!("missing {FIXTURE}; regenerate with dump_shared_pool_scorer.py")
    });
    serde_json::from_str(&raw).expect("parse shared-pool-scorer-golden.json")
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

#[test]
fn matches_the_reference() {
    let Some(model) = loaded_model() else {
        return;
    };
    let fixture = fixture();
    let batch = 1usize;
    let mut worst_score = 0.0f32;
    let mut worst_candidate = 0.0f32;

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

        // The pool is compared field-by-field against the fixture's copy so a
        // scorer mismatch cannot be blamed on a pool difference.
        assert_eq!(
            pooled.indices,
            usize_list(&case["pool_indices"]),
            "case {case_index}: pool indices differ from the fixture"
        );
        assert_eq!(
            pooled.mask,
            bool_list(&case["pool_mask"]),
            "case {case_index}: pool mask differs from the fixture"
        );
        let want_prior = f32s(&case["pool_compat_logits"]);
        for (i, (got, want)) in pooled
            .compat_logits
            .iter()
            .zip(want_prior.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-5,
                "case {case_index} slot {i}: prior logit differs by {}",
                (got - want).abs()
            );
        }

        let text_lengths: Vec<usize> = text_mask
            .iter()
            .map(|row| row.iter().filter(|m| **m).count())
            .collect();
        let (scores, candidates) = model.pool_scorer.forward(
            &SharedPoolInputs {
                boundary_states: &encoding.states,
                query_states: &query_states,
                query_mask: &query_mask,
                inside_prefix: &marginals.inside_prefix,
                inside_prefix_mean: &marginals.inside_prefix_mean,
                text_states: &text_states,
                text_mask: &text_mask,
                start_logits: &marginals.start_logits,
                end_logits: &marginals.end_logits,
                text_lengths: &text_lengths,
                q_count,
            },
            &pooled,
        );

        let c = pooled.pool_size;
        assert_eq!(
            scores.len(),
            batch * c * q_count,
            "case {case_index}: score width"
        );
        assert_eq!(
            candidates.len(),
            batch * c * model.pool_scorer.pair_dim,
            "case {case_index}: candidate width"
        );

        let want_scores = f32s(&case["pair_logits"]);
        for (i, (got, want)) in scores.iter().zip(want_scores.iter()).enumerate() {
            let slot = i / q_count;
            if !pooled.mask[slot] {
                assert!(
                    *got <= -1.0e3,
                    "case {case_index} slot {slot} query {}: padded candidate must be MASK_LOGIT, got {got}",
                    i % q_count
                );
                continue;
            }
            let delta = (got - want).abs();
            if delta > worst_score {
                worst_score = delta;
            }
        }

        let pair_dim = model.pool_scorer.pair_dim;
        let slots = usize_list(&case["candidate_slots"]);
        let want_rows = case["candidate_rows"].as_array().expect("candidate_rows");
        assert_eq!(slots.len(), want_rows.len());
        for (row_index, slot) in slots.iter().enumerate() {
            for d in 0..pair_dim {
                let got = candidates[slot * pair_dim + d];
                let want = want_rows[row_index][d].as_f64().unwrap() as f32;
                worst_candidate = worst_candidate.max((got - want).abs());
            }
        }
    }

    assert!(
        worst_score < 1e-3,
        "max pair-logit delta {worst_score} exceeds threshold 1e-3"
    );
    assert!(
        worst_candidate < 1e-3,
        "max candidate-state delta {worst_candidate} exceeds threshold 1e-3"
    );
}

#[test]
fn a_padded_candidate_can_never_win() {
    // Every query must see MASK_LOGIT for padding slots, otherwise top-K over
    // a padded pool would return a span that does not exist.
    let Some(model) = loaded_model() else {
        return;
    };
    let fixture = fixture();
    let case = &fixture["cases"][2];
    let seq_len = case["seq_len"].as_u64().unwrap() as usize;
    let q_count = case["q_count"].as_u64().unwrap() as usize;
    let text_states = f32s(&case["text_states"]);
    let text_mask = vec![bool_list(&case["text_mask"])[..seq_len].to_vec()];
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
    let text_lengths: Vec<usize> = text_mask
        .iter()
        .map(|row| row.iter().filter(|m| **m).count())
        .collect();
    let (scores, _) = model.pool_scorer.forward(
        &SharedPoolInputs {
            boundary_states: &encoding.states,
            query_states: &query_states,
            query_mask: &query_mask,
            inside_prefix: &marginals.inside_prefix,
            inside_prefix_mean: &marginals.inside_prefix_mean,
            text_states: &text_states,
            text_mask: &text_mask,
            start_logits: &marginals.start_logits,
            end_logits: &marginals.end_logits,
            text_lengths: &text_lengths,
            q_count,
        },
        &pooled,
    );
    let mut valid_seen = 0usize;
    for slot in 0..pooled.pool_size {
        for q in 0..q_count {
            let value = scores[slot * q_count + q];
            if pooled.mask[slot] {
                valid_seen += 1;
                assert!(
                    value > -1.0e3,
                    "slot {slot} query {q} is valid but scored {value}"
                );
            } else {
                assert!(
                    value <= -1.0e3,
                    "slot {slot} query {q} is padding but scored {value}"
                );
            }
        }
    }
    assert!(valid_seen > 0, "the pool produced no candidates");
}

/// The public contract is `[B, Q, C]`, the reference's internal order is
/// `[B, C, Q]`. A transpose bug here would silently associate every query with
/// the wrong candidate, so assert the layout rather than only the values.
#[test]
fn the_top_level_api_returns_the_per_query_order() {
    let Some(model) = loaded_model() else {
        return;
    };
    let fixture = fixture();
    let case = &fixture["cases"][2];
    let seq_len = case["seq_len"].as_u64().unwrap() as usize;
    let q_count = case["q_count"].as_u64().unwrap() as usize;
    let text_states = f32s(&case["text_states"]);
    let text_mask = vec![bool_list(&case["text_mask"])[..seq_len].to_vec()];
    let query_states = f32s(&case["query_states"]);
    let query_mask = vec![bool_list(&case["query_mask"])];

    let batch =
        score_document_candidates(&model, &text_states, &text_mask, &query_states, &query_mask);
    let c = batch.pool_size;
    assert_eq!(c, fixture["config"]["pool_size"].as_u64().unwrap() as usize);
    assert_eq!(batch.indices.len(), q_count * c * 2);
    assert_eq!(batch.pair_logits.len(), q_count * c);
    assert_eq!(batch.valid_mask.len(), q_count * c);

    // The pool is query-agnostic, so the same (start, end) must appear for
    // every query at the same slot.
    let pool_indices = usize_list(&case["pool_indices"]);
    let pool_mask = bool_list(&case["pool_mask"]);
    for q in 0..q_count {
        for slot in 0..c {
            let flat = q * c + slot;
            assert_eq!(batch.indices[flat * 2], pool_indices[slot * 2]);
            assert_eq!(batch.indices[flat * 2 + 1], pool_indices[slot * 2 + 1]);
            assert_eq!(batch.valid_mask[flat], pool_mask[slot]);
        }
    }

    // Values must line up with the candidate-major fixture after transpose.
    let want = f32s(&case["pair_logits"]);
    for q in 0..q_count {
        for slot in 0..c {
            let flat = q * c + slot;
            if !pool_mask[slot] {
                continue;
            }
            let delta = (batch.pair_logits[flat] - want[slot * q_count + q]).abs();
            assert!(
                delta < 1e-3,
                "transposed[{q}][{slot}] differs from candidate-major[{slot}][{q}] by {delta}"
            );
        }
    }
    assert_eq!(
        batch.candidate_states.len(),
        c * model.pool_scorer.pair_dim,
        "candidate states stay candidate-major"
    );
}
