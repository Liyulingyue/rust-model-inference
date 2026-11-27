//! End-to-end byte parity for `fastino/gliner2-multi-v1` against the GLiNER2
//! reference (`tools/oracle/gliner2/dump_golden.py`).
//!
//! The multilingual member of the pre-2.5 SpanExtractor family. Its
//! `model_name` is `microsoft/mdeberta-v3-base`, which is *not* a DeBERTa-v3
//! checkpoint wearing a different name — but its published config is
//! field-for-field identical to deberta-v3-base apart from `vocab_size`, and
//! HF serves it through the same `DebertaV2Model` with a plain softmax
//! (`XSoftmax` exists only in the TF implementation). So no new inference code
//! is involved; what differs is everything around the vocab.
//!
//! Two vocab-specific facts, both of which bit during the conversion:
//!
//! - mDeBERTa-v3's SPM has **250101** pieces, not 250000, so the embedding
//!   table is 250112 rows against a published `vocab_size` of 251000. The
//!   config's number is an upper bound, not the piece count, so the oracle
//!   reads the table size off the checkpoint and checks it against the
//!   tokenizer's highest added-token id instead of trusting either.
//! - Its `added_tokens_decoder` declares 104 tokens that are *already* in the
//!   SPM vocab: the four base specials plus 100 ALBERT-style `<extra_id_N>`
//!   sentinels. Only the 11 past the SPM boundary are appended tokens. The
//!   converter now splits declarations by id and requires every in-vocab one to
//!   match the piece it claims, which is stricter than ignoring them.
//!
//! Unlike the same-size DeBERTa checkpoints, `input_ids` here are **not**
//! expected to match Decide's — a 250k multilingual SPM segments differently.
//! So there is deliberately no cross-variant `input_ids` test in this file.
//!
//! Fixture:
//! ```sh
//! models/.venv/bin/python tools/oracle/gliner2/dump_golden.py \
//!     --model-dir models/gliner2-multi-v1 \
//!     --encoder-config target/mdeberta-v3-base-config \
//!     --out tests/fixtures/gliner2-multi-v1/classify-golden.json
//! ```

use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::models::gliner::prompt::{Label, Task};
use rust_model_inference::models::gliner::GlinerModel;
use serde_json::Value;

const GGUF: &str = "models/gliner2-multi-v1/gliner2-multi-v1-f32.gguf";
const FIXTURE: &str = "tests/fixtures/gliner2-multi-v1/classify-golden.json";

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
    let model = GlinerModel::from_source(source.as_ref()).expect("load gliner2 multi-v1");

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
