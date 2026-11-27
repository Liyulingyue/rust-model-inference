//! End-to-end byte parity for the three guardrail checkpoints.
//!
//! `GLiNER2-Guardrails-PII-Multi`, `gliner2-privacy-filter-PII-multi` and
//! `gliguard-LLMGuardrails-300M` are pre-2.5 SpanExtractor models, so they reuse
//! the same Rust forward as `gliner2-large-v1` / `gliner2-base-v1` /
//! `gliner2-multi-v1` with no model-specific code. The first two are
//! mDeBERTa-v3-base like `gliner2-multi-v1`; the third is DeBERTa-v3-base with
//! `count_lstm_v2` like `gliner2-base-v1`.
//!
//! They are grouped in one file because the interesting fact about them is
//! shared, and it is not a happy one: **transformers cannot load any of their
//! tokenizers.** `tokenizer_config.json` carries `extra_special_tokens` as a bare
//! list, which makes `AutoTokenizer.from_pretrained` raise for both the fast and
//! the slow class. So the reference cannot have tokenized from these repos — it
//! must have used the base encoder, which is what `dump_golden.py --base-encoder`
//! now does. Our converter reads the same files directly and does not care,
//! because it only needs the ids, which are in `tokenizer.json`.
//!
//! Fixture, per model (the base encoder is the only tokenizer the reference can
//! use):
//!
//! ```sh
//! models/.venv/bin/python tools/oracle/gliner2/dump_golden.py \
//!     --model-dir models/GLiNER2-Guardrails-PII-Multi \
//!     --base-encoder models/mdeberta-v3-base \
//!     --out tests/fixtures/GLiNER2-Guardrails-PII-Multi/classify-golden.json
//! ```
//!
//! Env vars: `RMI_GUARDRAILS_PII_MULTI_GGUF`, `RMI_PRIVACY_FILTER_MULTI_GGUF`,
//! `RMI_GLLIGUARD_300M_GGUF`.

use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::models::gliner::prompt::{Label, Task};
use rust_model_inference::models::gliner::GlinerModel;
use serde_json::Value;

/// `(fixture dir, GGUF env var, expected encoder width)` per guardrail.
const MODELS: [(&str, &str, usize); 3] = [
    (
        "GLiNER2-Guardrails-PII-Multi",
        "RMI_GUARDRAILS_PII_MULTI_GGUF",
        768,
    ),
    (
        "gliner2-privacy-filter-PII-multi",
        "RMI_PRIVACY_FILTER_MULTI_GGUF",
        768,
    ),
    (
        "gliguard-LLMGuardrails-300M",
        "RMI_GLLIGUARD_300M_GGUF",
        768,
    ),
];

/// Same threshold as `gliner2_classify_parity.rs` (7.2e-6 measured on Decide).
/// Pre-2.5 weights may shift slightly more or less depending on how the
/// reference's BLAS happened to accumulate for this specific checkpoint, but
/// any structural error still moves logits by whole units.
const LOGIT_TOLERANCE: f32 = 1e-4;

fn fixture(model: &str) -> Value {
    let path = format!("tests/fixtures/{model}/classify-golden.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!("missing {path}; regenerate with dump_golden.py --model-dir models/{model}")
    });
    serde_json::from_str(&text).expect("parse fixture")
}

/// The model borrows its `TensorSource`, so the source is leaked to `static`
/// for the test's lifetime; the OS reclaims it on exit.
fn open(env: &str) -> Option<&'static dyn rust_model_inference::core::tensor::TensorSource> {
    let path = match std::env::var_os(env) {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            eprintln!("skipping: set {env} to enable this model");
            return None;
        }
    };
    if !path.exists() {
        eprintln!("skipping: {} does not exist", path.display());
        return None;
    }
    let source = open_model_source(&path, ComponentRole::Llm).expect("open guardrail gguf");
    Some(Box::leak(source))
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
    for (name, env, hidden_size) in MODELS {
        let Some(source) = open(env) else {
            continue;
        };
        let model = GlinerModel::from_source(source).expect("load guardrail model");
        assert_eq!(
            model.config().n_embd,
            hidden_size,
            "{name}: encoder width must come from metadata"
        );

        let value = fixture(name);
        let cases = value["cases"].as_array().expect("cases");
        assert!(
            cases.len() >= 6,
            "{name}: fixture looks truncated: {} cases",
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
            assert_eq!(
                encoded.input_ids, want_ids,
                "{name}: input_ids differ for {text:?}"
            );
            let got_positions: Vec<Vec<usize>> = encoded
                .markers
                .iter()
                .map(|markers| markers.positions.clone())
                .collect();
            assert_eq!(
                got_positions, want_positions,
                "{name}: marker positions differ for {text:?}"
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
                        "{name}: task {} label {} ({:?}): logit {} vs {expected} (delta {})",
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
                        "{name}: single-label head must softmax, got {total}"
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
}
