//! End-to-end parity for the legacy `json_structures` decode.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_structures_end_to_end.py`, which
//! applies the reference's `_resolve_spans` + `_format_structure_field` (the
//! whole body of `_decode_json_structures` for a group with no `choices`) to the
//! same 7 text/schema cases.
//!
//! The point of this path is that it is **not** the record path and **not** the
//! span path. A `json_structures` group is a record only when the schema
//! annotates it with a `mode` in `record_metadata`; unannotated it emits exactly
//! one instance per group, because boundary checkpoints have no count-slot axis to
//! form several instances from. A port that routes an unannotated group through
//! the record head produces something that looks like a structure and is not one.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner::prompt::BoundaryTaskKind;
use rust_model_inference::models::gliner_boundary::structure::StructureField;
use rust_model_inference::models::gliner_boundary::{
    run_mixed_extraction, BoundaryModel, SchemaOptions,
};

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/structures-e2e-golden.json";
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
        panic!("missing {FIXTURE}; regenerate with dump_structures_end_to_end.py")
    });
    serde_json::from_str(&raw).expect("parse structures-e2e-golden.json")
}

fn structures(
    model: &BoundaryModel<'_>,
    schema: &serde_json::Value,
    text: &str,
    threshold: f32,
) -> Vec<rust_model_inference::models::gliner_boundary::ExtractedStructure> {
    let (tasks, kinds) = rust_model_inference::app::parse_boundary_schema(schema).expect("schema");
    assert!(
        kinds.iter().any(|k| *k == BoundaryTaskKind::JsonStructure),
        "{text:?}: the fixture cases all declare a json_structures group"
    );
    let extraction = run_mixed_extraction(
        model,
        text,
        &tasks,
        &kinds,
        0,
        Some(threshold),
        SchemaOptions {
            record_metadata: schema.get("record_metadata"),
            field_metadata: schema.get("field_metadata"),
        },
    )
    .expect("mixed extraction");
    extraction.structures
}

/// A field value as the reference formats it: a dict for a bound scalar, a list
/// for a list field, `null` for an unbound scalar. Mirrors the JSON shape the
/// oracle dumps, so one comparison covers both.
fn value_json(value: &StructureField) -> serde_json::Value {
    match value {
        StructureField::Scalar(None) => serde_json::Value::Null,
        StructureField::Scalar(Some(span)) => serde_json::json!({
            "text": span.text,
            "confidence": span.score,
            "start": span.start,
            "end": span.end,
        }),
        StructureField::List(spans) => serde_json::Value::Array(
            spans
                .iter()
                .map(|span| {
                    serde_json::json!({
                        "text": span.text,
                        "confidence": span.score,
                        "start": span.start,
                        "end": span.end,
                    })
                })
                .collect(),
        ),
    }
}

fn close(got: &serde_json::Value, want: &serde_json::Value) -> bool {
    match (got, want) {
        (serde_json::Value::Null, serde_json::Value::Null) => true,
        (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
            a.len() == b.len()
                && a.iter().zip(b).all(|(x, y)| {
                    x.get("text") == y.get("text")
                        && x.get("start") == y.get("start")
                        && x.get("end") == y.get("end")
                        && (x.get("confidence").unwrap().as_f64().unwrap()
                            - y.get("confidence").unwrap().as_f64().unwrap())
                        .abs()
                            < TOLERANCE as f64
                })
        }
        (a, b) if a.is_object() && b.is_object() => {
            a.get("text") == b.get("text")
                && a.get("start") == b.get("start")
                && a.get("end") == b.get("end")
                && (a.get("confidence").unwrap().as_f64().unwrap()
                    - b.get("confidence").unwrap().as_f64().unwrap())
                .abs()
                    < TOLERANCE as f64
        }
        _ => false,
    }
}

#[test]
fn structures_end_to_end_match_the_reference() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    for case in data["cases"].as_array().expect("cases") {
        let text = case["text"].as_str().unwrap();
        let threshold = case["threshold"].as_f64().unwrap() as f32;
        let got = structures(&model, &case["schema"], text, threshold);
        let want = case["structures"].as_array().expect("structures");

        assert_eq!(
            got.len(),
            want.len(),
            "{text:?} @{threshold}: structure count (reference {})",
            want.len()
        );
        for (index, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.task,
                w["task"].as_str().unwrap(),
                "{text:?}[{index}] task"
            );
            let want_fields = w["fields"].as_object().expect("fields");
            assert_eq!(
                g.fields.len(),
                want_fields.len(),
                "{text:?}[{index}]: field count"
            );
            for (name, value) in &g.fields {
                let expected = want_fields
                    .get(name)
                    .unwrap_or_else(|| panic!("{text:?}[{index}]: no reference field {name}"));
                let actual = value_json(value);
                assert!(
                    close(&actual, expected),
                    "{text:?}[{index}]: field {name}\n  got  {actual}\n  want {expected}"
                );
            }
        }
    }
}

#[test]
fn a_scalar_field_binds_exactly_one_span() {
    // The behaviour a span-per-field port cannot reproduce: a `dtype: "str"`
    // field reports `spans[0]` and discards the rest. The all-scalar case has two
    // city candidates, so binding more than one would be visible.
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| {
            case["text"]
                .as_str()
                .unwrap()
                .contains("Pierre Curie in London")
                && case["schema"]
                    .get("field_metadata")
                    .map(|meta| {
                        meta.get("person.city")
                            .and_then(|entry| entry.get("dtype"))
                            .and_then(|v| v.as_str())
                            == Some("str")
                    })
                    .unwrap_or(false)
                && case["threshold"].as_f64().unwrap() == 0.5
        })
        .expect("the all-scalar case");
    let got = structures(
        &model,
        &case["schema"],
        case["text"].as_str().unwrap(),
        case["threshold"].as_f64().unwrap() as f32,
    );
    assert_eq!(got.len(), 1, "one instance per legacy group, got {got:?}");
    for (name, value) in &got[0].fields {
        match value {
            StructureField::Scalar(Some(span)) => {
                assert!(span.start < span.end, "{name}: empty span {span:?}")
            }
            other => panic!("{name}: expected a bound scalar, got {other:?}"),
        }
    }
    // The scalar city is the higher-scoring of the two candidates, which is the
    // resolver's `(-score, start, end)` order deciding the value.
    let want_city = case["structures"][0]["fields"]["city"]["text"]
        .as_str()
        .unwrap();
    let got_city = got[0]
        .fields
        .iter()
        .find(|(name, _)| name == "city")
        .map(|(_, value)| match value {
            StructureField::Scalar(Some(span)) => span.text.clone(),
            _ => panic!("city is not a scalar"),
        })
        .unwrap();
    assert_eq!(got_city, want_city, "the scalar must be the best span");
}

#[test]
fn an_unannotated_group_does_not_become_a_record() {
    // The mixed case declares an unannotated `person` group and an annotated
    // `trip` group. Only `person` decodes as a legacy structure; if the legacy
    // path were skipped, or routed through the record head, `person` would be
    // missing or would appear as a record.
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["schema"].get("record_metadata").is_some())
        .expect("the mixed case");
    let schema = &case["schema"];
    let (tasks, kinds) = rust_model_inference::app::parse_boundary_schema(schema).expect("schema");
    let extraction = run_mixed_extraction(
        &model,
        case["text"].as_str().unwrap(),
        &tasks,
        &kinds,
        0,
        Some(case["threshold"].as_f64().unwrap() as f32),
        SchemaOptions {
            record_metadata: schema.get("record_metadata"),
            field_metadata: schema.get("field_metadata"),
        },
    )
    .expect("mixed extraction");

    let legacy: Vec<&str> = extraction
        .structures
        .iter()
        .map(|structure| structure.task.as_str())
        .collect();
    assert_eq!(
        legacy,
        vec!["person"],
        "only the unannotated group decodes here"
    );

    let recorded: Vec<&str> = extraction
        .records
        .iter()
        .map(|record| record.task.as_str())
        .collect();
    assert!(
        !recorded.contains(&"person"),
        "the unannotated group must not reach the record head: {recorded:?}"
    );
    for record in &extraction.records {
        assert_eq!(
            record.task, "trip",
            "the annotated group is the one that becomes a record"
        );
    }
}
