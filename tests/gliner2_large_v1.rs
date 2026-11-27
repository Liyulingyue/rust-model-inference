//! Smoke integration test for `fastino/gliner2-large-v1`.
//!
//! Run with `RMI_GLINER2_LARGE_V1_GGUF=/path/to/gliner2-large-v1-f32.gguf`.
//!
//! What this adds over `tests/gliner2_classify_parity.rs`:
//!
//! - Pre-2.5 configs (this repo's first non-Decide variant) omit
//!   `architecture`, `config_version`, and `token_pooling`. The converter
//!   accepts the relaxed schema; this test pins that the GGUF still feeds
//!   the same Rust encoder path byte-compatibly.
//! - The four fixture cases from `tools/oracle/gliner/fixtures/` drive the
//!   CLI through every branch of the prompt builder (single-label,
//!   multi-label, described labels, examples, long-position) without
//!   needing a reference fixture. Without a reference fixture we can only
//!   check semantic agreement (refund / positive / mixed / keyboard-battery)
//!   — true byte parity requires running the GLiNER2 reference stack
//!   against `gliner2-large-v1` and pinning its logits (see TODO).

use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::models::gliner::prompt::{Label, Task};
use rust_model_inference::models::gliner::GlinerModel;

fn gguf_path() -> Option<std::path::PathBuf> {
    std::env::var_os("RMI_GLINER2_LARGE_V1_GGUF").map(std::path::PathBuf::from)
}

fn load_model() -> Option<(Box<dyn std::any::Any>, GlinerModel<'static>)> {
    let path = gguf_path()?;
    if !path.exists() {
        eprintln!("skipping: {} does not exist", path.display());
        return None;
    }
    let source = open_model_source(&path, ComponentRole::Llm).expect("open gliner2 gguf");
    // The model holds a `&'a TensorSource`. Leak the source to 'static for
    // the test's lifetime; the OS reclaims the bytes on process exit.
    let leaked: &'static dyn rust_model_inference::core::tensor::TensorSource = Box::leak(source);
    let model = GlinerModel::from_source(leaked).expect("load gliner2 large-v1");
    Some((Box::new(()), model))
}

fn case_intent() -> Task {
    Task::new(
        "intent",
        vec![
            Label {
                name: "refund".into(),
                description: None,
                examples: Vec::new(),
            },
            Label {
                name: "other".into(),
                description: None,
                examples: Vec::new(),
            },
        ],
    )
}

fn case_sentiment() -> Task {
    Task::new(
        "sentiment",
        ["positive", "negative", "mixed", "neutral"]
            .into_iter()
            .map(|name| Label {
                name: name.into(),
                description: None,
                examples: Vec::new(),
            })
            .collect(),
    )
}

fn case_aspects() -> Task {
    let mut task = Task::new(
        "aspects",
        vec![
            Label {
                name: "battery".into(),
                description: Some("power life".into()),
                examples: Vec::new(),
            },
            Label {
                name: "keyboard".into(),
                description: Some("typing feel".into()),
                examples: Vec::new(),
            },
            Label {
                name: "camera".into(),
                description: Some("image quality".into()),
                examples: Vec::new(),
            },
        ],
    );
    task.multi_label = true;
    task.cls_threshold = 0.4;
    task
}

#[test]
fn contract_loads_with_pre_2_5_relaxed_config() {
    let Some((_anchor, model)) = load_model() else {
        return;
    };
    // Decide is 24x1024x16; large-v1 should match (same encoder).
    assert_eq!(model.config().n_layer, 24);
    assert_eq!(model.config().n_embd, 1024);
    assert_eq!(model.config().n_head, 16);
    // Pre-2.5 should not advertise any new contract — vocab size is 128011
    // (128000 SPM + 11 schema specials) for both Decide and large-v1.
    assert_eq!(model.config().vocab_size, 128011);
}

#[test]
fn refund_classifier_picks_refund() {
    let Some((_anchor, model)) = load_model() else {
        return;
    };
    let tasks = vec![case_intent()];
    let text = "Refund please";
    let encoded = model.encode_prompt(&tasks, text).expect("encode");
    let hidden = model.forward(&encoded.input_ids, 0).expect("forward");
    let results = model
        .score_prompt(&tasks, &encoded, &hidden)
        .expect("score");
    assert_eq!(results.len(), 1);
    let scores = &results[0].scores;
    assert_eq!(scores.len(), 2);
    assert!(
        scores[0].logit > scores[1].logit,
        "refund should outrank other"
    );
    assert_eq!(results[0].selected, vec!["refund".to_string()]);
}

#[test]
fn multitask_routes_sentiment_and_aspects() {
    let Some((_anchor, model)) = load_model() else {
        return;
    };
    let text = "Battery dies, but the keyboard is excellent";
    let tasks = vec![case_sentiment(), case_aspects()];
    let encoded = model.encode_prompt(&tasks, text).expect("encode");
    let hidden = model.forward(&encoded.input_ids, 0).expect("forward");
    let results = model
        .score_prompt(&tasks, &encoded, &hidden)
        .expect("score");
    assert_eq!(results.len(), 2);

    let sentiment = &results[0];
    assert_eq!(sentiment.task, "sentiment");
    let mass: f32 = sentiment.scores.iter().map(|s| s.probability).sum();
    assert!((mass - 1.0).abs() < 1e-4, "single-label head must softmax");

    let aspects = &results[1];
    assert_eq!(aspects.task, "aspects");
    assert!(aspects.multi_label);
    let by_label: std::collections::HashMap<&str, f32> = aspects
        .scores
        .iter()
        .map(|score| (score.label.as_str(), score.probability))
        .collect();
    assert!(by_label["battery"] > by_label["camera"]);
    assert!(by_label["keyboard"] > by_label["camera"]);
    assert!(aspects.selected.contains(&"battery".to_string()));
    assert!(aspects.selected.contains(&"keyboard".to_string()));
}

#[test]
fn example_conditioning_changes_selection() {
    let Some((_anchor, model)) = load_model() else {
        return;
    };
    let mut task = case_intent();
    task.prompt = Some("Choose the route".into());
    task.labels[0]
        .examples
        .push(("Refund now".into(), "refund".into()));
    let tasks = vec![task];
    let text = "I need a refund";
    let encoded = model.encode_prompt(&tasks, text).expect("encode");
    let hidden = model.forward(&encoded.input_ids, 0).expect("forward");
    let results = model
        .score_prompt(&tasks, &encoded, &hidden)
        .expect("score");
    assert_eq!(results[0].selected, vec!["refund".to_string()]);
}

#[test]
fn long_position_context_still_picks_positive() {
    let Some((_anchor, model)) = load_model() else {
        return;
    };
    // Shorter than `tools/oracle/gliner/fixtures/long-position.json` because
    // debug-mode forward over 24 layers × the full sequence is slow; the
    // exercise is "long-enough context + softmax path", not "max length".
    let text = std::iter::repeat("great ").take(60).collect::<String>();
    let tasks = vec![Task::new(
        "intent",
        ["positive", "negative"]
            .into_iter()
            .map(|name| Label {
                name: name.into(),
                description: None,
                examples: Vec::new(),
            })
            .collect(),
    )];
    let encoded = model.encode_prompt(&tasks, &text).expect("encode");
    let hidden = model.forward(&encoded.input_ids, 0).expect("forward");
    let results = model
        .score_prompt(&tasks, &encoded, &hidden)
        .expect("score");
    assert_eq!(results[0].selected, vec!["positive".to_string()]);
    assert!(results[0].scores.iter().all(|s| s.logit.is_finite()));
}

#[test]
fn inputs_match_the_pinned_schema_marker_positions() {
    let Some((_anchor, model)) = load_model() else {
        return;
    };
    let tasks = vec![case_intent()];
    let text = "Refund please";
    let encoded = model.encode_prompt(&tasks, text).expect("encode");
    let markers = &encoded.markers[0].positions;
    // One [P] (system prompt row) + one [L] per label. For "refund / other"
    // that's 3 marker rows. Decide pins the same shape.
    assert_eq!(
        markers.len(),
        3,
        "expected [P] + 2 [L] rows, got {:?}",
        markers
    );
    assert!(
        markers[0] < markers[1] && markers[1] < markers[2],
        "markers must be in declaration order: {:?}",
        markers
    );
}
