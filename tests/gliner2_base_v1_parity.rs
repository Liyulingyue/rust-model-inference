//! End-to-end byte parity for `fastino/gliner2-base-v1` against the GLiNER2
//! reference (`tools/oracle/gliner2/dump_golden.py`).
//!
//! The first **size variant** of the SpanExtractor (pre-2.5) family. Decide and
//! `gliner2-large-v1` are both DeBERTa-v3-large, so every golden in this suite
//! so far was produced against 1024-wide weights. base-v1 is 768-wide with 12
//! layers and 12 heads, and the point of this test is that the *same* Rust
//! forward pass reproduces the reference's logits on a different size without a
//! single size-dependent line of Rust having changed — every dimension arrives
//! from GGUF metadata.
//!
//! `dump_golden.py` likewise derives the classifier width from the encoder
//! config rather than hardcoding `Linear(1024, 2048)`, and loads it with
//! `strict=True`, so a mismatched width fails at generation time instead of
//! quietly scoring a differently-shaped head.
//!
//! The tolerance is the same 1e-4 the Decide and large-v1 suites use. The
//! expected delta is F32 accumulation order only; a structural error moves
//! logits by whole units, not by the fourth decimal.
//!
//! Fixture:
//! ```sh
//! models/.venv/bin/python tools/oracle/gliner2/dump_golden.py \
//!     --model-dir models/gliner2-base-v1 \
//!     --encoder-config target/gliner2-deberta-config-base \
//!     --out tests/fixtures/gliner2-base-v1/classify-golden.json
//! ```

use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::models::gliner::prompt::{Label, Task};
use rust_model_inference::models::gliner::GlinerModel;
use serde_json::Value;

const GGUF: &str = "models/gliner2-base-v1/gliner2-base-v1-f32.gguf";
const FIXTURE: &str = "tests/fixtures/gliner2-base-v1/classify-golden.json";

/// Same threshold as `gliner2_classify_parity.rs` (7.2e-6 measured on Decide).
/// Pre-2.5 weights may shift slightly more or less depending on how the
/// reference's BLAS happened to accumulate for this specific checkpoint, but
/// any structural error still moves logits by whole units.
const LOGIT_TOLERANCE: f32 = 1e-4;

fn fixture() -> Value {
    let text = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_golden.py"));
    serde_json::from_str(&text).expect("parse fixture")
}

fn tasks_from(value: &Value) -> Vec<Task> {
    value
        .as_array()
        .expect("tasks array")
        .iter()
        .map(|raw| {
            let mut task = Task::new(
                raw["name"].as_str().expect("task name"),
                raw["labels"]
                    .as_array()
                    .expect("labels")
                    .iter()
                    .map(|label| Label {
                        name: label["name"].as_str().expect("label name").to_string(),
                        description: label
                            .get("description")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        examples: Vec::new(),
                    })
                    .collect(),
            );
            if let Some(prompt) = raw.get("prompt").and_then(Value::as_str) {
                task.prompt = Some(prompt.to_string());
            }
            task.multi_label = raw
                .get("multi_label")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            task.cls_threshold = raw
                .get("cls_threshold")
                .and_then(Value::as_f64)
                .unwrap_or(0.5) as f32;
            task
        })
        .collect()
}

fn ids_from(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .expect("input_ids")
        .iter()
        .map(|id| id.as_u64().expect("token id") as u32)
        .collect()
}

fn positions_from(value: &Value) -> Vec<Vec<usize>> {
    value
        .as_array()
        .expect("marker_positions")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("positions")
                .iter()
                .map(|p| p.as_u64().expect("position") as usize)
                .collect()
        })
        .collect()
}

#[test]
fn matches_the_reference_stack() {
    let path = std::path::Path::new(GGUF);
    if !path.exists() {
        panic!("missing {GGUF}; run tools/converter/gliner/convert_gliner.py first");
    }
    let source = open_model_source(path, ComponentRole::Llm).expect("open gliner2 gguf");
    let model = GlinerModel::from_source(source.as_ref()).expect("load gliner2 base-v1");

    let value = fixture();
    let cases = value["cases"].as_array().expect("cases");
    assert!(
        cases.len() >= 6,
        "fixture looks truncated: {} cases",
        cases.len()
    );

    for case in cases {
        let text = case["text"].as_str().expect("text");
        let tasks = tasks_from(&case["tasks"]);
        let want_ids = ids_from(&case["input_ids"]);
        let want_positions = positions_from(&case["marker_positions"]);

        let encoded = model
            .encode_prompt(&tasks, text)
            .unwrap_or_else(|error| panic!("encode {text:?}: {error}"));
        assert_eq!(encoded.input_ids, want_ids, "input_ids differ for {text:?}");
        let got_positions: Vec<Vec<usize>> = encoded
            .markers
            .iter()
            .map(|markers| markers.positions.clone())
            .collect();
        assert_eq!(
            got_positions, want_positions,
            "marker positions differ for {text:?}"
        );

        let hidden = model
            .forward(&encoded.input_ids, 0)
            .unwrap_or_else(|error| panic!("forward {text:?}: {error}"));
        let results = model
            .score_prompt(&tasks, &encoded, &hidden)
            .unwrap_or_else(|error| panic!("score {text:?}: {error}"));
        assert_eq!(results.len(), tasks.len());

        for (task_index, want_logits) in case["logits"]
            .as_array()
            .expect("logits")
            .iter()
            .enumerate()
        {
            let got = &results[task_index];
            let want: Vec<f32> = want_logits
                .as_array()
                .expect("logits")
                .iter()
                .map(|value| value.as_f64().expect("logit") as f32)
                .collect();
            assert_eq!(
                got.scores.len(),
                want.len(),
                "label count differs for {text:?}"
            );
            for (index, (score, expected)) in got.scores.iter().zip(&want).enumerate() {
                assert!(
                    (score.logit - expected).abs() < LOGIT_TOLERANCE,
                    "task {} label {} ({:?}): logit {} vs {expected} (delta {})",
                    got.task,
                    index,
                    score.label,
                    score.logit,
                    (score.logit - expected).abs()
                );
            }
            // `selected` must be consistent with the decoded probabilities.
            let total: f32 = got.scores.iter().map(|score| score.probability).sum();
            if got.multi_label {
                assert!(got
                    .scores
                    .iter()
                    .all(|score| (0.0..=1.0).contains(&score.probability)));
                assert!(!got.selected.is_empty());
            } else {
                assert!(
                    (total - 1.0).abs() < 1e-4,
                    "single-label head must softmax, got {total}"
                );
                let best = got
                    .scores
                    .iter()
                    .max_by(|a, b| a.probability.total_cmp(&b.probability))
                    .expect("labels");
                assert_eq!(got.selected, vec![best.label.clone()]);
            }
        }
    }
}

#[test]
fn input_ids_match_decide_byte_for_byte() {
    // The pre-2.5 v1 vocab and schema specials are identical to 2.5-Decide, and
    // the encoder *size* does not enter tokenization at all. So a Rust encoding
    // on base-v1 must produce byte-identical input_ids to Decide for the same
    // (text, schema). This is a strong cross-variant regression: if anyone
    // touches the SentencePiece path / added-token mapping and one variant
    // drifts, this fires before the per-variant goldens do.
    let decide_path = std::path::Path::new("tests/fixtures/gliner2-decide/classify-golden.json");
    let base_path = std::path::Path::new(FIXTURE);
    if !decide_path.exists() || !base_path.exists() {
        return;
    }
    let decide: Value =
        serde_json::from_str(&std::fs::read_to_string(decide_path).unwrap()).unwrap();
    let base: Value = serde_json::from_str(&std::fs::read_to_string(base_path).unwrap()).unwrap();
    let decide_cases = decide["cases"].as_array().unwrap();
    let base_cases = base["cases"].as_array().unwrap();
    assert_eq!(decide_cases.len(), base_cases.len());
    for (d, v) in decide_cases.iter().zip(base_cases) {
        assert_eq!(
            d["text"], v["text"],
            "fixture text differs — goldens are out of sync"
        );
        assert_eq!(
            d["input_ids"], v["input_ids"],
            "input_ids differ for {:?}",
            d["text"]
        );
        assert_eq!(
            d["marker_positions"], v["marker_positions"],
            "marker_positions differ for {:?}",
            d["text"]
        );
    }
}
