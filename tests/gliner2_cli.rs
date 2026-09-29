//! CLI-surface tests for `--jev --gliner2-decide`.
//!
//! These only exercise flag parsing, validation and the request-shape
//! translation, so they run without the 1.7 GB model. The numerics live in
//! `tests/gliner2_classify_parity.rs`.

use rust_model_inference::app::{
    gliner2_schema, parse_cli_options, parse_schema, schema_from_label_sets,
    schema_from_questions, validate_cli_options, JevQuestionInput, LabelSet,
};

fn options(args: &[&str]) -> rust_model_inference::app::CliOptions {
    let owned: Vec<String> = args.iter().map(|value| (*value).to_string()).collect();
    parse_cli_options(&owned).expect("flags must parse")
}

#[test]
fn gliner2_decide_is_an_explicit_jev_mode() {
    let parsed = options(&[
        "rmi",
        "--model",
        "gliner2-decide-f32.gguf",
        "--jev",
        "--gliner2-decide",
        "--jev-context",
        "hello",
        "--jev-question",
        "intent",
        "--jev-option",
        "a",
        "--jev-option",
        "b",
    ]);
    assert!(parsed.gliner2_decide);
    validate_cli_options(&parsed).unwrap();
}

#[test]
fn gliner2_decide_requires_jev_and_excludes_clm() {
    let no_jev = options(&["rmi", "--model", "m.gguf", "--gliner2-decide"]);
    assert!(validate_cli_options(&no_jev).is_err());

    let both = options(&[
        "rmi", "--model", "m.gguf", "--jev", "--gliner2-decide", "--clm-head", "h.gguf",
    ]);
    assert!(validate_cli_options(&both).is_err());

    let orphan_schema = options(&["rmi", "--model", "m.gguf", "--gliner2-schema", "{}"]);
    assert!(validate_cli_options(&orphan_schema).is_err());
}

#[test]
fn questions_become_one_task_each() {
    let questions = vec![
        JevQuestionInput {
            text: "intent".into(),
            options: vec!["a".into(), "b".into()],
        },
        JevQuestionInput {
            text: "priority".into(),
            options: vec!["low".into(), "high".into()],
        },
    ];
    let tasks = parse_schema(&schema_from_questions(&questions).unwrap()).unwrap();
    assert_eq!(tasks.len(), 2);
    assert_eq!(tasks[0].name, "intent");
    assert_eq!(
        tasks[0].labels.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(tasks[1].name, "priority");
    assert!(!tasks[0].multi_label);
}

#[test]
fn an_empty_label_set_is_an_error_not_an_empty_task() {
    let questions = vec![JevQuestionInput { text: "intent".into(), options: vec![] }];
    assert!(schema_from_questions(&questions).is_err());
    let unnamed = vec![JevQuestionInput { text: "  ".into(), options: vec!["a".into()] }];
    assert!(schema_from_questions(&unnamed).is_err());
}

#[test]
fn label_sets_cover_descriptions_thresholds_and_prompts() {
    let tasks = vec![LabelSet {
        name: "aspects".into(),
        labels: vec!["battery".into(), "keyboard".into()],
        descriptions: Some(vec!["the battery".into(), "the keyboard".into()]),
        multi_label: true,
        cls_threshold: Some(0.4),
        prompt: Some("which parts?".into()),
    }];
    let parsed = parse_schema(&schema_from_label_sets(&tasks).unwrap()).unwrap();
    assert!(parsed[0].multi_label);
    assert_eq!(parsed[0].cls_threshold, 0.4);
    assert_eq!(parsed[0].prompt.as_deref(), Some("which parts?"));
    assert_eq!(parsed[0].labels[0].description.as_deref(), Some("the battery"));
    assert_eq!(parsed[0].labels[1].name, "keyboard");
}

#[test]
fn a_mismatched_description_count_is_rejected() {
    let tasks = vec![LabelSet {
        name: "aspects".into(),
        labels: vec!["a".into(), "b".into()],
        descriptions: Some(vec!["only one".into()]),
        multi_label: false,
        cls_threshold: None,
        prompt: None,
    }];
    assert!(schema_from_label_sets(&tasks).is_err());
}

#[test]
fn the_schema_flag_wins_over_the_question_flags() {
    let parsed = options(&[
        "rmi",
        "--model",
        "m.gguf",
        "--jev",
        "--gliner2-decide",
        "--jev-context",
        "hello",
        "--jev-question",
        "ignored",
        "--jev-option",
        "ignored",
        "--gliner2-schema",
        r#"{"intent":["a","b"],"aspects":{"labels":["x","y"],"multi_label":true}}"#,
    ]);
    validate_cli_options(&parsed).unwrap();
    let questions = vec![JevQuestionInput { text: "ignored".into(), options: vec!["ignored".into()] }];
    let tasks = parse_schema(&gliner2_schema(&parsed, &questions).unwrap()).unwrap();
    assert_eq!(
        tasks.iter().map(|task| task.name.as_str()).collect::<Vec<_>>(),
        ["intent", "aspects"]
    );
    assert!(tasks[1].multi_label);
}

#[test]
fn a_broken_schema_flag_is_reported_as_such() {
    let parsed = options(&[
        "rmi",
        "--model",
        "m.gguf",
        "--jev",
        "--gliner2-decide",
        "--gliner2-schema",
        "{not json",
    ]);
    let error = gliner2_schema(&parsed, &[]).unwrap_err();
    assert!(error.contains("--gliner2-schema"), "{error}");
}

#[test]
fn reference_schema_shapes_all_parse() {
    // The four documented `classify_text` shapes.
    let schema = serde_json::json!({
        "intent": ["order_status", "refund_request"],
        "aspects": {
            "labels": ["battery", "keyboard"],
            "multi_label": true,
            "cls_threshold": 0.4,
        },
        "described": {"labels": {"pin": "wants a new PIN", "lost": "card is missing"}},
        "answer": {
            "labels": ["yes", "no"],
            "prompt": "Did it work?",
            "examples": [["it worked", "yes"]],
        },
    });
    let tasks = parse_schema(&schema).unwrap();
    assert_eq!(tasks.len(), 4);
    assert_eq!(tasks[0].labels.len(), 2);
    assert!(tasks[1].multi_label);
    assert_eq!(tasks[1].cls_threshold, 0.4);
    assert_eq!(tasks[2].labels[0].description.as_deref(), Some("wants a new PIN"));
    assert_eq!(tasks[3].prompt.as_deref(), Some("Did it work?"));
    assert_eq!(tasks[3].labels[0].examples.len(), 1);
    assert_eq!(tasks[3].labels[1].examples.len(), 0);
}

#[test]
fn a_reserved_marker_in_a_label_is_refused() {
    // A label containing [L] would shift every logit onto the wrong label.
    let schema = serde_json::json!({ "intent": ["ok", "[L] sneaky"] });
    assert!(parse_schema(&schema).is_err());
}

#[test]
fn an_example_whose_label_is_not_declared_is_refused() {
    let schema = serde_json::json!({
        "answer": {
            "labels": ["yes", "no"],
            "examples": [["it worked", "maybe"]],
        }
    });
    assert!(parse_schema(&schema).is_err());
}
