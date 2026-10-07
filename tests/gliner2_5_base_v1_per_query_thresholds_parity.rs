//! Byte-exact parity for per-entity, per-field and per-relation thresholds.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_per_query_thresholds.py`. The
//! resolution is pure schema input — no model forward, no GGUF — so this always
//! runs.
//!
//! The reference resolves a configured threshold in three places that do not
//! agree on which key they read or whether the candidate stage sees it at all,
//! and the fixture pins each of them separately:
//!
//! - `entities` queries read `entity_metadata[<label>]["threshold"]`;
//! - `json_structures` queries read
//!   `field_metadata["<group>.<field>"]["threshold"]`;
//! - relations read **neither** at the candidate stage, and get their threshold
//!   re-applied once by the relation scorer from `relation_metadata[<name>]`.
//!
//! So a port that lets a relation's configured threshold reach the candidate
//! stage changes which pairs are *generated*, not just which are kept, and the
//! two mistakes are separable only because the relations are checked on both
//! channels.

use rust_model_inference::models::gliner::prompt::{BoundaryTaskKind, Label, Task};
use rust_model_inference::models::gliner_boundary::extract::{
    query_layout, resolve_query_thresholds, resolve_relation_thresholds,
};

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/per-query-thresholds-golden.json";

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE).unwrap_or_else(|_| {
        panic!("missing {FIXTURE}; regenerate with dump_per_query_thresholds.py")
    });
    serde_json::from_str(&raw).expect("parse per-query-thresholds-golden.json")
}

fn kind(name: &str) -> BoundaryTaskKind {
    match name {
        "entities" => BoundaryTaskKind::Entities,
        "json_structures" => BoundaryTaskKind::JsonStructure,
        "relations" => BoundaryTaskKind::Relation,
        other => panic!("unknown task type {other:?}"),
    }
}

/// Rebuild the `(task, kind)` list the fixture's `query_specs` imply.
///
/// The reference's own ordering is `json_structures` → `entities` → relations,
/// which the fixture's `query_specs` order records directly. Rebuilding from it
/// rather than re-deriving it here means this test fails if my ordering
/// assumption ever drifts from the reference's.
fn layout_from_specs(case: &serde_json::Value) -> Vec<(Task, BoundaryTaskKind)> {
    let mut tasks: Vec<(Task, BoundaryTaskKind)> = Vec::new();
    for spec in case["query_specs"].as_array().expect("query_specs") {
        let task_type = spec["task_type"].as_str().expect("task_type");
        let task_name = spec["task_name"].as_str().expect("task_name");
        let field = spec["field_name"].as_str().expect("field_name");
        let kind = kind(task_type);
        match tasks.last_mut() {
            Some((task, last)) if *last == kind && task.name == task_name => {
                task.labels.push(Label::new(field));
            }
            _ => tasks.push((Task::new(task_name, vec![Label::new(field)]), kind)),
        }
    }
    tasks
}

#[test]
fn query_thresholds_match_the_reference() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let default = case["threshold"].as_f64().unwrap() as f32;
        let schema = &case["schema"];
        let tasks = layout_from_specs(case);
        let kinds: Vec<BoundaryTaskKind> = tasks.iter().map(|(_, kind)| *kind).collect();
        let specs = query_layout(
            &tasks
                .iter()
                .map(|(task, _)| task.clone())
                .collect::<Vec<_>>(),
            &kinds,
        );

        // The layout itself is part of the contract: a wrong order would apply
        // each query's threshold to the wrong field while still "passing" a
        // length check.
        let want_specs: Vec<(String, String, String)> = case["query_specs"]
            .as_array()
            .expect("query_specs")
            .iter()
            .map(|spec| {
                (
                    spec["task_type"].as_str().unwrap().to_string(),
                    spec["task_name"].as_str().unwrap().to_string(),
                    spec["field_name"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        let got_specs: Vec<(String, String, String)> = specs
            .iter()
            .map(|spec| {
                (
                    spec.task_type.clone(),
                    spec.task_name.clone(),
                    spec.field_name.clone(),
                )
            })
            .collect();
        assert_eq!(got_specs, want_specs, "{name}: query layout");

        let got = resolve_query_thresholds(
            &specs,
            schema.get("entity_metadata"),
            schema.get("field_metadata"),
            default,
        );
        let want: Vec<f32> = case["query_thresholds"][0]
            .as_array()
            .expect("query_thresholds row")
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(got.len(), want.len(), "{name}: threshold count");
        for (index, (actual, expected)) in got.iter().zip(&want).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "{name}: query {index} threshold {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn relation_thresholds_are_a_separate_channel() {
    let fixture = fixture();
    for case in fixture["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let default = case["threshold"].as_f64().unwrap() as f32;
        let want = case["relation_thresholds"]
            .as_object()
            .expect("relation_thresholds");
        let got = resolve_relation_thresholds(case["schema"].get("relation_metadata"), default);

        assert_eq!(
            got.len(),
            want.len(),
            "{name}: resolved relation threshold count"
        );
        for (key, expected) in want {
            let actual = got
                .get(key)
                .unwrap_or_else(|| panic!("{name}: no threshold for {key}"));
            assert!(
                (actual - expected.as_f64().unwrap() as f32).abs() < 1e-6,
                "{name}: relation {key} threshold {actual} vs {expected}"
            );
        }
    }
}

/// The properties the fixture's cases exist to discriminate, asserted directly
/// so a fixture edit that weakens them fails here rather than quietly making the
/// parity test vacuous.
#[test]
fn only_entities_and_structures_reach_the_candidate_stage() {
    // Relations: every resolved query threshold equals the caller's, even though
    // `relation_metadata` configures two types.
    let relation_case = fixture()["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["name"] == "per_relation_threshold")
        .expect("case")
        .clone();
    let default = relation_case["threshold"].as_f64().unwrap() as f32;
    let specs = query_layout(&[], &[]);
    assert!(specs.is_empty());
    let resolved = resolve_query_thresholds(
        &[
            rust_model_inference::models::gliner_boundary::extract::QuerySpec {
                task_type: "relations".into(),
                task_name: "works_at".into(),
                field_name: "head".into(),
            },
            rust_model_inference::models::gliner_boundary::extract::QuerySpec {
                task_type: "relations".into(),
                task_name: "works_at".into(),
                field_name: "tail".into(),
            },
        ],
        relation_case["schema"].get("entity_metadata"),
        relation_case["schema"].get("field_metadata"),
        default,
    );
    assert_eq!(
        resolved,
        vec![default; 2],
        "a relation's configured threshold must not reach the candidate stage"
    );

    // ...and the per-type channel did resolve it, independently.
    let per_type =
        resolve_relation_thresholds(relation_case["schema"].get("relation_metadata"), default);
    assert_eq!(per_type.get("works_at").copied(), Some(0.02));
    assert_eq!(per_type.get("studied_in").copied(), Some(0.99));
}

#[test]
fn field_metadata_is_keyed_by_group_and_field() {
    let case = fixture()["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["name"] == "field_metadata_does_not_leak_across_groups")
        .expect("case")
        .clone();
    let default = case["threshold"].as_f64().unwrap() as f32;
    let schema = &case["schema"];
    let spec = |task: &str, field: &str| {
        rust_model_inference::models::gliner_boundary::extract::QuerySpec {
            task_type: "json_structures".into(),
            task_name: task.into(),
            field_name: field.into(),
        }
    };
    let resolved = resolve_query_thresholds(
        &[
            spec("trip", "traveller"),
            spec("trip", "destination"),
            spec("visit", "who"),
            spec("visit", "where"),
        ],
        schema.get("entity_metadata"),
        schema.get("field_metadata"),
        default,
    );
    assert_eq!(
        resolved,
        vec![0.02, 0.02, default, default],
        "an unconfigured group must keep the caller's threshold"
    );
}

#[test]
fn an_override_replaces_the_caller_threshold_in_both_directions() {
    let build = |configured: f32, default: f32| {
        resolve_query_thresholds(
            &[
                rust_model_inference::models::gliner_boundary::extract::QuerySpec {
                    task_type: "entities".into(),
                    task_name: "entities".into(),
                    field_name: "person".into(),
                },
            ],
            Some(&serde_json::json!({ "person": { "threshold": configured } })),
            None,
            default,
        )[0]
    };
    // Below the caller's threshold: honoured, so a port that took the max of the
    // two would report nothing.
    assert_eq!(build(0.02, 0.99), 0.02);
    // Above it: honoured too, so a port that treated it as a floor would not.
    assert_eq!(build(0.99, 0.02), 0.99);
    // No metadata at all: the caller's.
    assert_eq!(
        resolve_query_thresholds(
            &[
                rust_model_inference::models::gliner_boundary::extract::QuerySpec {
                    task_type: "entities".into(),
                    task_name: "entities".into(),
                    field_name: "person".into(),
                }
            ],
            None,
            None,
            0.42,
        )[0],
        0.42
    );
}

#[test]
fn a_null_relation_threshold_falls_back_rather_than_comparing_against_zero() {
    let default = 0.5;
    let resolved = resolve_relation_thresholds(
        Some(&serde_json::json!({ "works_at": { "threshold": null } })),
        default,
    );
    assert_eq!(
        resolved.get("works_at").copied(),
        Some(default),
        "an explicit null carries no value, so the caller decides"
    );

    // A key that is present but has no `threshold` at all also falls back.
    let resolved = resolve_relation_thresholds(
        Some(&serde_json::json!({ "works_at": { "description": "x" } })),
        default,
    );
    assert_eq!(resolved.get("works_at").copied(), Some(default));

    // Absent metadata resolves nothing, which leaves the caller's threshold in
    // place for every type.
    assert!(resolve_relation_thresholds(None, default).is_empty());
}
