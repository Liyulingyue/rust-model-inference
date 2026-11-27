//! Byte-exact parity for the full `score_spans` path.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_score_explicit_spans_full.py`,
//! which loads the reference `BoundaryHead` from the checkpoint's own
//! `config.json` and calls `BoundaryHead.score_explicit_spans`. Every feature
//! flag in the fixture therefore comes from the published base-v1 settings:
//! `enable_span_content`, `use_inside_evidence`,
//! `query_conditioned_inside_weight` and `endpoint_difference_features` are
//! all on, which is what distinguishes this from the earlier limited
//! pair-scorer oracle.
//!
//! Chain: `BoundaryEncoder` -> `BoundaryQueryHead` ->
//! `score_explicit_pairs` -> `PairScorer.forward`.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner_boundary::{score_spans, BoundaryModel, ScoredSpan};

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/score-explicit-spans-full-golden.json";

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF").map(std::path::PathBuf::from)
}

fn loaded_model() -> Option<BoundaryModel<'static>> {
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
    Some(model)
}

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE).unwrap_or_else(|_| {
        panic!("missing {FIXTURE}; regenerate with dump_score_explicit_spans_full.py")
    });
    serde_json::from_str(&raw).expect("parse score-explicit-spans-full-golden.json")
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

/// The GGUF must carry the transcoded settings, otherwise the loader would
/// fall back to a limited scorer and every score here would be wrong for a
/// reason that has nothing to do with the math under test.
#[test]
fn gguf_carries_the_published_feature_flags() {
    let Some(model) = loaded_model() else {
        return;
    };
    let fixture = fixture();
    let want = &fixture["config"];
    let got = model.pair_scorer.features;
    assert!(got.use_inside_evidence, "use_inside_evidence must be on");
    assert!(got.enable_span_content, "enable_span_content must be on");
    assert!(
        !got.content_soft_max_pool,
        "base-v1 disables the soft-max content channel"
    );
    assert!(
        got.query_conditioned_inside_weight,
        "query_conditioned_inside_weight must be on"
    );
    assert!(
        got.endpoint_difference_features,
        "endpoint_difference_features must be on"
    );
    assert!(got.enable_rotary_endpoints);
    assert!(got.reranker_endpoint_compat);
    for (name, expected) in [
        (
            "boundary_dim",
            want["boundary_dim"].as_u64().unwrap() as usize,
        ),
        ("pair_dim", want["pair_dim"].as_u64().unwrap() as usize),
        (
            "content_output_dim",
            want["content_dim"].as_u64().unwrap() as usize,
        ),
    ] {
        let actual = match name {
            "boundary_dim" => model.pair_scorer.boundary_dim,
            "pair_dim" => model.pair_scorer.pair_dim,
            _ => model.pair_scorer.content_output_dim,
        };
        assert_eq!(actual, expected, "{name} disagrees with the fixture");
    }
    assert_eq!(
        model.pair_scorer.multihead_pair_compat_heads,
        want["multihead_pair_compat_heads"].as_u64().unwrap() as usize
    );
    assert_eq!(
        model.pair_scorer.query_dim, model.config.n_embd,
        "base-v1 uses the encoder width as the query dim"
    );
}

#[test]
fn score_spans_matches_the_reference() {
    let Some(model) = loaded_model() else {
        return;
    };
    let fixture = fixture();
    let batch = 1usize;
    let mut worst = 0.0f32;

    for (case_index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let seq_len = case["seq_len"].as_u64().unwrap() as usize;
        let q_count = case["q_count"].as_u64().unwrap() as usize;
        let c = case["c_count"].as_u64().unwrap() as usize;
        let text_states = f32s(&case["text_states"]);
        let flat_mask = bool_list(&case["text_mask"]);
        let text_mask = vec![flat_mask[..seq_len].to_vec()];
        let query_states = f32s(&case["query_states"]);
        let query_mask = vec![bool_list(&case["query_mask"])];
        let indices = usize_list(&case["indices"]);
        let valid_mask = bool_list(&case["valid_mask"]);
        let want: Vec<f32> = f32s(&case["scores"]);

        let got: Vec<ScoredSpan> = score_spans(
            &model,
            &text_states,
            &text_mask,
            &query_states,
            &query_mask,
            &indices,
            &valid_mask,
            batch,
            q_count,
            c,
        );

        assert_eq!(
            got.len(),
            want.len(),
            "case {case_index}: candidate count mismatch"
        );
        for (i, (span, expected)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                span.start,
                indices[i * 2],
                "case {case_index} slot {i} start"
            );
            assert_eq!(
                span.end,
                indices[i * 2 + 1],
                "case {case_index} slot {i} end"
            );
            if !valid_mask[i] {
                // MASK_LOGIT, not -inf: sums downstream must stay finite.
                assert!(
                    span.logit <= -1.0e3,
                    "case {case_index} slot {i} should be MASK_LOGIT, got {}",
                    span.logit
                );
                continue;
            }
            let delta = (span.logit - expected).abs();
            if delta > worst {
                worst = delta;
            }
        }
    }

    // F32 accumulation order differs from torch's, so this is a tight-but-not-
    // bitwise bound; the earlier per-stage oracles run at ~1e-6.
    assert!(
        worst < 1e-3,
        "max score_spans delta {worst} exceeds threshold 1e-3"
    );
}
