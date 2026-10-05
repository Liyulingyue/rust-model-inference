//! Byte-exact parity for `group_scored_candidates` — the candidate admission
//! step, including count-head-guided (`adaptive_threshold`) admission.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_adaptive_threshold.py`. This is a
//! pure function over a candidate batch, so like `overlap_resolution_parity` it
//! needs no GGUF and always runs.
//!
//! Every shipped boundary checkpoint sets `adaptive_threshold = false`, so
//! nothing here changes the 12/12 model parity. What it pins is that the branch
//! exists and behaves like the reference when a future checkpoint turns it on:
//! the count head **unions** with the threshold rather than filtering by it,
//! ranks are taken over eligible candidates only, ties break by candidate index,
//! and `exp` is applied before the rounding.
//!
//! The `synthetic_batches` section is what makes those four discriminating. On
//! the real-encoder cases the invalid slots' logits saturate to probability
//! `0.0`, so masking them changes no rank and a port that skipped the mask would
//! agree on all of them.

use rust_model_inference::models::gliner_boundary::spans::{
    group_scored_candidates, DocumentCandidateBatch, QueryThresholds,
};

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/adaptive-threshold-golden.json";

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE).unwrap_or_else(|_| {
        panic!("missing {FIXTURE}; regenerate with dump_adaptive_threshold.py")
    });
    serde_json::from_str(&raw).expect("parse adaptive-threshold-golden.json")
}

fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

fn row(value: &serde_json::Value) -> Vec<(f32, usize, usize)> {
    value
        .as_array()
        .expect("span triples")
        .iter()
        .map(|entry| {
            let triple = entry.as_array().expect("span triple");
            (
                triple[0].as_f64().unwrap() as f32,
                triple[1].as_u64().unwrap() as usize,
                triple[2].as_u64().unwrap() as usize,
            )
        })
        .collect()
}

fn matrix(values: &serde_json::Value) -> Vec<Vec<Vec<(f32, usize, usize)>>> {
    values
        .as_array()
        .expect("query rows")
        .iter()
        .map(|queries| {
            queries
                .as_array()
                .expect("query list")
                .iter()
                .map(|query| row(query))
                .collect()
        })
        .collect()
}

fn assert_close(
    got: &[(f32, usize, usize)],
    want: &[(f32, usize, usize)],
    context: &str,
    tolerance: f32,
) {
    assert_eq!(
        got.len(),
        want.len(),
        "{context}: admitted {} candidates, reference admitted {}",
        got.len(),
        want.len()
    );
    for (index, (actual, expected)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            (actual.1, actual.2),
            (expected.1, expected.2),
            "{context}: candidate {index} span differs"
        );
        assert!(
            (actual.0 - expected.0).abs() <= tolerance,
            "{context}: candidate {index} probability {} vs {} (tolerance {tolerance})",
            actual.0,
            expected.0
        );
    }
}

/// Rebuild the candidate batch the oracle scored from its own recorded inputs.
///
/// The fixture is one sample with `Q` queries of `C` candidates, recorded
/// query-major as `[Q][C]`, which is the same `[B, Q, C]` layout the flattened
/// batch uses.
fn batch_from_real_case(case: &serde_json::Value) -> (DocumentCandidateBatch, usize) {
    let rows = case["candidate_indices"]
        .as_array()
        .expect("candidate_indices");
    let queries = rows.len();
    let pool = rows[0].as_array().expect("candidate row").len();

    let mut indices = Vec::with_capacity(queries * pool * 2);
    let mut pair_logits = Vec::with_capacity(queries * pool);
    let mut valid_mask = Vec::with_capacity(queries * pool);
    for query in 0..queries {
        for slot in 0..pool {
            let pair = &rows[query].as_array().expect("candidate row")[slot];
            indices.push(pair[0].as_u64().unwrap() as usize);
            indices.push(pair[1].as_u64().unwrap() as usize);
            pair_logits.push(
                case["pair_logits"][query].as_array().expect("logits")[slot]
                    .as_f64()
                    .unwrap() as f32,
            );
            valid_mask.push(
                case["valid_mask"][query].as_array().expect("mask")[slot]
                    .as_bool()
                    .unwrap(),
            );
        }
    }
    (
        DocumentCandidateBatch {
            indices,
            pair_logits,
            valid_mask,
            candidate_states: Vec::new(),
            pool_candidate_features: Vec::new(),
            pool_size: pool,
        },
        queries,
    )
}

#[test]
fn real_encoder_cases_match_the_reference() {
    let fixture = fixture();
    for case in fixture["cases"].as_array().expect("cases") {
        let threshold = case["threshold"].as_f64().unwrap() as f32;
        let (batch, queries) = batch_from_real_case(case);
        let rates: Vec<f32> = case["count_log_rates"]
            .as_array()
            .expect("count_log_rates")
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        // The reference divides the logits by `pair_temperature` before the
        // sigmoid, and base-v1's is 1.0, so the probabilities are a plain
        // sigmoid here.
        let probabilities: Vec<f32> = batch.pair_logits.iter().copied().map(sigmoid).collect();

        let baseline = group_scored_candidates(
            &batch,
            &probabilities,
            queries,
            &QueryThresholds::Uniform(threshold),
            None,
            false,
        );
        assert_close(
            &baseline[0][0],
            &row(case["threshold_only"].as_array().unwrap().get(0).unwrap()),
            "threshold_only",
            1e-6,
        );

        let adaptive = group_scored_candidates(
            &batch,
            &probabilities,
            queries,
            &QueryThresholds::Uniform(threshold),
            Some(&rates),
            true,
        );
        assert_close(
            &adaptive[0][0],
            &row(case["adaptive_real_counts"]
                .as_array()
                .unwrap()
                .get(0)
                .unwrap()),
            "adaptive_real_counts",
            1e-6,
        );

        for (rate_key, want) in case["adaptive_synthetic_counts"]
            .as_object()
            .expect("rates")
        {
            let rate: f32 = rate_key.parse().expect("rate");
            let shaped = vec![rate; rates.len()];
            let got = group_scored_candidates(
                &batch,
                &probabilities,
                queries,
                &QueryThresholds::Uniform(threshold),
                Some(&shaped),
                true,
            );
            assert_close(
                &got[0][0],
                &row(want.as_array().unwrap().get(0).unwrap()),
                &format!("adaptive_synthetic_counts[{rate_key}]"),
                1e-6,
            );
        }
    }
}

#[test]
fn synthetic_batches_match_the_reference() {
    let fixture = fixture();
    let cases = fixture["synthetic_batches"].as_array().expect("synthetic");
    assert!(!cases.is_empty(), "fixture has no synthetic batches");

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let threshold = case["threshold"].as_f64().unwrap() as f32;
        let rate = case["count_log_rate"].as_f64().unwrap() as f32;
        let logits: Vec<f32> = case["pair_logits"]
            .as_array()
            .expect("logits")
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        let valid: Vec<bool> = case["valid_mask"]
            .as_array()
            .expect("mask")
            .iter()
            .map(|value| value.as_bool().unwrap())
            .collect();
        let pool = logits.len();
        let batch = DocumentCandidateBatch {
            // The oracle builds `arange(c).view(1, 1, c, 1).expand(1, 1, c, 2)`,
            // which repeats one column, so slot `i` carries `[i, i]`. The
            // admission step copies indices without validating them, so the
            // degenerate span is carried through verbatim.
            indices: (0..pool).flat_map(|i| [i, i]).collect(),
            pair_logits: logits.clone(),
            valid_mask: valid.clone(),
            candidate_states: Vec::new(),
            pool_candidate_features: Vec::new(),
            pool_size: pool,
        };
        let probabilities: Vec<f32> = logits.iter().copied().map(sigmoid).collect();

        let baseline = group_scored_candidates(
            &batch,
            &probabilities,
            1, // the synthetic batches are one query each
            &QueryThresholds::Uniform(threshold),
            None,
            false,
        );
        assert_close(
            &baseline[0][0],
            &row(&case["threshold_only"]),
            &format!("{name}/threshold_only"),
            1e-6,
        );

        let adaptive = group_scored_candidates(
            &batch,
            &probabilities,
            1,
            &QueryThresholds::Uniform(threshold),
            Some(&[rate]),
            true,
        );
        assert_close(
            &adaptive[0][0],
            &row(&case["adaptive"]),
            &format!("{name}/adaptive"),
            1e-6,
        );
    }
}

/// The four properties the synthetic batches exist to discriminate, asserted
/// directly so a fixture edit that weakens them fails here rather than silently
/// making the parity test vacuous.
#[test]
fn count_guidance_unions_with_the_threshold() {
    let fixture = fixture();
    let case = |name: &str| {
        fixture["synthetic_batches"]
            .as_array()
            .expect("synthetic")
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("no synthetic case {name}"))
            .clone()
    };

    // A count of zero must not remove a threshold hit.
    let zero = case("zero_count_keeps_threshold_hits");
    assert!(!zero["threshold_only"].as_array().unwrap().is_empty());
    assert_eq!(
        zero["adaptive"].as_array().unwrap(),
        zero["threshold_only"].as_array().unwrap(),
        "a predicted count of 0 changed the thresholded set"
    );

    // Ranking must skip ineligible slots: the two highest scores are padding, so
    // an unmasked rank would admit nothing beyond the threshold hit.
    let masked = case("masked_rank_interleaved");
    let admitted: Vec<u64> = masked["adaptive"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry[1].as_u64().unwrap())
        .collect();
    assert_eq!(
        admitted,
        vec![1, 3],
        "masked_rank_interleaved must admit the two top *eligible* candidates"
    );

    // Ties rank by candidate index, so an all-equal batch admits a prefix.
    let tied = case("all_tied");
    let admitted: Vec<u64> = tied["adaptive"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry[1].as_u64().unwrap())
        .collect();
    assert_eq!(
        admitted,
        vec![0, 1, 2],
        "tied candidates must rank by index"
    );

    // `exp` before rounding: a log-rate of 1.0 is a count of 3, not 1.
    let exp_round = case("exp_before_round");
    assert_eq!(
        exp_round["adaptive"].as_array().unwrap().len(),
        3,
        "round(exp(1.0)) is 3; rounding the log-rate first would admit 1"
    );
}

#[test]
fn per_query_thresholds_override_per_query() {
    let fixture = fixture();
    let case = &fixture["synthetic_batches"]
        .as_array()
        .expect("synthetic")
        .iter()
        .find(|case| case["name"] == "zero_count_keeps_threshold_hits")
        .expect("case")
        .clone();
    let logits: Vec<f32> = case["pair_logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect();
    let pool = logits.len();
    let probabilities: Vec<f32> = logits.iter().copied().map(sigmoid).collect();
    let batch = DocumentCandidateBatch {
        indices: (0..pool).flat_map(|i| [i, i]).collect(),
        pair_logits: logits,
        valid_mask: vec![true; pool],
        candidate_states: Vec::new(),
        pool_candidate_features: Vec::new(),
        pool_size: pool,
    };

    // A per-query threshold of 1.0 admits nothing: the highest sigmoid in the
    // batch is sigmoid(4.0) = 0.982, below 1.0.
    let none_admitted = group_scored_candidates(
        &batch,
        &probabilities,
        1,
        &QueryThresholds::PerQuery {
            values: vec![1.0],
            queries: 1,
        },
        None,
        false,
    );
    assert!(
        none_admitted[0][0].is_empty(),
        "a threshold no candidate clears must admit nothing"
    );

    // The same batch under 0.0 admits every candidate, which is the uniform
    // case the vector form has to agree with.
    let all_admitted = group_scored_candidates(
        &batch,
        &probabilities,
        1,
        &QueryThresholds::PerQuery {
            values: vec![0.0],
            queries: 1,
        },
        None,
        false,
    );
    assert_eq!(all_admitted[0][0].len(), pool);
}
