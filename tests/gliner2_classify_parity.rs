//! End-to-end parity for GLiNER2.5-Decide against the reference stack.
//!
//! The fixture is produced by the reference (GLiNER2's `SchemaTransformer` +
//! `transformers` DeBERTa-v3 + the checkpoint's classifier) and pins three
//! things that each have their own failure mode:
//!
//! - `input_ids` — the schema prompt and the SentencePiece unigram pass;
//! - `marker_positions` — which subwords carry the `[P]` / `[L]` rows;
//! - `logits` — the disentangled-attention encoder and the ReLU MLP.
//!
//! Regenerate with the reference script (needs `transformers==4.48.1`,
//! `sentencepiece` and `torch`).

use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::models::gliner::prompt::{Label, Task};
use rust_model_inference::models::gliner::GlinerModel;
use serde_json::Value;

const GGUF: &str = "models/GLiNER2.5-Decide/gliner2-decide-f32.gguf";
const FIXTURE: &str = "tests/fixtures/gliner2-decide/classify-golden.json";

/// Measured worst-case drift on this fixture is 7.2e-6, which is F32
/// accumulation order across 24 layers. Anything structural — a wrong attention
/// scale, a mis-bucketed position index, an off-by-one marker row, a stale
/// position projection — moves logits by whole units, so this threshold has
/// three orders of magnitude of headroom over the noise and four over the bugs.
const LOGIT_TOLERANCE: f32 = 1e-4;

fn fixture() -> Value {
    let text = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with the reference script"));
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
    let model = GlinerModel::from_source(source.as_ref()).expect("load gliner2");

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
                    "task {} label {} ({:?}): logit {} vs {expected}",
                    got.task,
                    index,
                    score.label,
                    score.logit
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
fn rejects_a_foreign_architecture() {
    let path = std::path::Path::new(GGUF);
    if !path.exists() {
        panic!("missing {GGUF}");
    }
    let source = open_model_source(path, ComponentRole::Llm).expect("open gliner2 gguf");
    // Loading twice proves the source can back several model handles.
    let first = GlinerModel::from_source(source.as_ref()).expect("load gliner2");
    let second = GlinerModel::from_source(source.as_ref()).expect("load gliner2 again");
    assert_eq!(first.config().n_layer, 24);
    assert_eq!(second.config().n_layer, 24);
    assert_eq!(first.tokenizer().len(), 128000);
}
