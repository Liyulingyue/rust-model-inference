//! End-to-end parity for the record head: real text + record schema -> records.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_records_end_to_end.py`, which runs
//! the reference pipeline (SchemaTransformer → DeBERTa → BoundaryHead pool +
//! `candidate_encoder` → `compile_record_specs` → `RecordHead.forward_group` →
//! `decode_group`) on the same 9 text/schema cases.
//!
//! This is the only oracle that exercises the *real* handoff. The head fixture
//! (`gliner2_5_base_v1_record_head_parity`) feeds the head a synthetic
//! `candidate_states` formula, so it pins the head's arithmetic but not the
//! boundary between the pool and the record head. Here the states come from the
//! checkpoint's `candidate_encoder` over the real refined boundary states, the
//! candidate batch is the real pool (66+ valid slots per query at this length),
//! and the field query ids come from the real prompt routing.
//!
//! That handoff is where coordinate, layout and broadcast mistakes live, and it
//! is not hypothetical: a one-column offset in the assignment cost matrix, and a
//! `width` shadowing bug, both shipped green through the head fixture and only
//! surfaced when the real shapes and the reference were involved.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner::prompt::BoundaryTaskKind;
use rust_model_inference::models::gliner_boundary::{run_mixed_extraction, BoundaryModel};

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/records-e2e-golden.json";
const TOLERANCE: f32 = 2.0e-4;

fn loaded_model() -> Option<(Box<dyn std::any::Any>, BoundaryModel<'static>)> {
    let path = match std::env::var_os("RMI_GLINER2_5_BASE_V1_GGUF") {
        Some(path) => std::path::PathBuf::from(path),
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
    Some((Box::new(()), model))
}

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE).unwrap_or_else(|_| {
        panic!("missing {FIXTURE}; regenerate with dump_records_end_to_end.py")
    });
    serde_json::from_str(&raw).expect("parse records-e2e-golden.json")
}

#[derive(Debug, serde::Deserialize)]
struct BoundField {
    spans: Vec<(usize, usize)>,
    text: Vec<String>,
    scores: Vec<f32>,
}

#[derive(Debug, serde::Deserialize)]
struct Record {
    task: String,
    mode: String,
    score: f32,
    anchor_span: Option<(usize, usize)>,
    fields: std::collections::BTreeMap<String, BoundField>,
}

fn run_case(
    model: &BoundaryModel<'_>,
    schema: &serde_json::Value,
    text: &str,
    threshold: f32,
) -> Vec<Record> {
    let (tasks, kinds) = rust_model_inference::app::parse_boundary_schema(schema).expect("schema");
    assert!(kinds.iter().any(|k| *k == BoundaryTaskKind::JsonStructure));
    let meta = schema.get("record_metadata");
    let extraction = run_mixed_extraction(model, text, &tasks, &kinds, 0, Some(threshold), meta)
        .expect("mixed extraction");
    extraction
        .records
        .iter()
        .map(|record| Record {
            task: record.task.clone(),
            mode: record.mode.clone(),
            score: record.score,
            anchor_span: record.anchor_span,
            fields: record
                .fields
                .iter()
                .map(|(query_id, spans)| {
                    (
                        query_id.to_string(),
                        BoundField {
                            spans: spans.clone(),
                            text: spans
                                .iter()
                                .map(|(s, e)| extraction.words[*s..*e].join(" "))
                                .collect(),
                            scores: record
                                .field_scores
                                .get(query_id)
                                .cloned()
                                .unwrap_or_default(),
                        },
                    )
                })
                .collect(),
        })
        .collect()
}

#[test]
fn records_end_to_end_match_the_reference() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    for case in data["cases"].as_array().expect("cases") {
        let text = case["text"].as_str().unwrap();
        let threshold = case["threshold"].as_f64().unwrap() as f32;
        let got = run_case(&model, &case["schema"], text, threshold);
        let want: Vec<Record> = serde_json::from_value(case["records"].clone()).expect("records");

        // The real handoff must actually be populated: the pool's
        // `candidate_encoder` states are the whole point of this oracle.
        assert_eq!(
            case["candidate_states_present"], true,
            "{text:?}: the reference produced no candidate states"
        );
        assert_eq!(
            case["candidate_state_width"].as_u64().unwrap(),
            768,
            "{text:?}: candidate states are hidden_size wide"
        );

        assert_eq!(
            got.len(),
            want.len(),
            "{text:?} @{threshold}: record count (reference {})",
            want.len()
        );
        for (index, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.task, w.task, "{text:?}[{index}] task");
            assert_eq!(g.mode, w.mode, "{text:?}[{index}] mode");
            assert!(
                (g.score - w.score).abs() < TOLERANCE,
                "{text:?}[{index}]: record score {} vs {}",
                g.score,
                w.score
            );
            assert_eq!(
                g.anchor_span, w.anchor_span,
                "{text:?}[{index}] anchor_span"
            );
            assert_eq!(
                g.fields.len(),
                w.fields.len(),
                "{text:?}[{index}]: field count (got {:?}, want {:?})",
                g.fields.keys().collect::<Vec<_>>(),
                w.fields.keys().collect::<Vec<_>>()
            );
            for (key, got_field) in &g.fields {
                let want_field = w
                    .fields
                    .get(key)
                    .unwrap_or_else(|| panic!("{text:?}[{index}]: no reference field {key}"));
                assert_eq!(
                    got_field.spans, want_field.spans,
                    "{text:?}[{index}]: field {key} spans"
                );
                assert_eq!(
                    got_field.text, want_field.text,
                    "{text:?}[{index}]: field {key} text"
                );
                for (span, score) in got_field.spans.iter().zip(&got_field.scores) {
                    assert!(
                        span.0 < span.1,
                        "{text:?}[{index}]: {key} span {span:?} empty"
                    );
                    let _ = score;
                }
                for (score, want_score) in got_field.scores.iter().zip(&want_field.scores) {
                    assert!(
                        (score - want_score).abs() < TOLERANCE,
                        "{text:?}[{index}]: field {key} score {score} vs {want_score}"
                    );
                }
            }
        }
    }
}

#[test]
fn exclusive_assignment_beats_greedy() {
    // The case the global assignment exists for. Two people, one place, the place
    // declared `exclusive`. Greedily letting the highest-scoring instance claim
    // its favourite candidate pushes the other instance off Paris; the joint
    // assignment keeps one and leaves the other without, which is the reference's
    // answer and this port's.
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| {
            case["text"].as_str().unwrap().contains("Marie Curie")
                && case["threshold"].as_f64().unwrap() == 0.5
                && case["schema"]["record_metadata"]["person"]["mode"] == "natural"
                && case["schema"]["json_structures"][0]["person"]
                    .as_object()
                    .unwrap()
                    .len()
                    == 2
        })
        .expect("the exclusive Marie Curie case");
    let got = run_case(
        &model,
        &case["schema"],
        case["text"].as_str().unwrap(),
        case["threshold"].as_f64().unwrap() as f32,
    );

    let curie_paris = got.iter().any(|r| {
        r.fields
            .get("0")
            .map(|f| f.text.iter().any(|t| t == "marie curie"))
            .unwrap_or(false)
            && r.fields
                .get("1")
                .map(|f| f.text.iter().any(|t| t == "paris"))
                .unwrap_or(false)
    });
    assert!(curie_paris, "Marie Curie should hold Paris, got {got:?}");

    // Paris is exclusive, so it may only be bound once across all records.
    let paris_claims: usize = got
        .iter()
        .map(|r| {
            r.fields
                .get("1")
                .map(|f| f.text.iter().filter(|t| *t == "paris").count())
                .unwrap_or(0)
        })
        .sum();
    assert_eq!(
        paris_claims, 1,
        "exclusive city bound {paris_claims} times: {got:?}"
    );
}
