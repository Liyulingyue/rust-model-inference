//! End-to-end parity for the legacy `json_structures` decode, across every
//! boundary checkpoint.
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
//! Four checkpoints, one body. The boundary family differs only in the encoder
//! (DeBERTa-v3-base, mDeBERTa-v3-base, DeBERTa-v3-xsmall) and the vocabulary
//! (128011 rows for the DeBERTa tokenizers, 250112 for mDeBERTa-v3's), and both
//! arrive from GGUF metadata, so nothing here is specialised per model. Each is
//! env-gated and skipped when unset, so a checkout with only one GGUF still runs
//! the rest.
//!
//! Per model:
//!
//! ```sh
//! models/.venv/bin/python tools/oracle/gliner_boundary/dump_structures_end_to_end.py \
//!     --model gliner2.5-base-v1 \
//!     --out tests/fixtures/gliner2.5-base-v1/structures-e2e-golden.json
//! ```
//!
//! Env vars: `RMI_GLINER2_5_BASE_V1_GGUF`, `RMI_GLINER2_5_MULTI_V1_GGUF`,
//! `RMI_GLINER2_5_MULTI_DECIDE_GGUF`, `RMI_GLINER2_5_SMALL_V1_GGUF`.

use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner::prompt::BoundaryTaskKind;
use rust_model_inference::models::gliner_boundary::structure::StructureField;
use rust_model_inference::models::gliner_boundary::{
    run_mixed_extraction, BoundaryModel, SchemaOptions,
};

const TOLERANCE: f32 = 2.0e-4;

/// `(fixture dir, GGUF env var)` per boundary checkpoint.
const MODELS: [(&str, &str); 4] = [
    ("gliner2.5-base-v1", "RMI_GLINER2_5_BASE_V1_GGUF"),
    ("gliner2.5-multi-v1", "RMI_GLINER2_5_MULTI_V1_GGUF"),
    ("GLiNER2.5-multi-Decide", "RMI_GLINER2_5_MULTI_DECIDE_GGUF"),
    ("gliner2.5-small-v1", "RMI_GLINER2_5_SMALL_V1_GGUF"),
];

fn fixture_path(model: &str) -> String {
    format!("tests/fixtures/{model}/structures-e2e-golden.json")
}

fn loaded_model(env: &str) -> Option<(Box<dyn std::any::Any>, BoundaryModel<'static>)> {
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
    let source = GGUFLoader::from_file(&path).expect("open boundary GGUF");
    let leaked: &'static dyn TensorSource = Box::leak(Box::new(source));
    let model = BoundaryModel::from_source(leaked).expect("load boundary model");
    Some((Box::new(()), model))
}

fn fixture(model: &str) -> serde_json::Value {
    let path = fixture_path(model);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!("missing {path}; regenerate with dump_structures_end_to_end.py --model {model}")
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
            schema: schema.get("choices_schema"),
            entity_metadata: schema.get("entity_metadata"),
            relation_metadata: schema.get("relation_metadata"),
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
        // No schema in this fixture declares `choices`, so a choice field here
        // means the fixture grew one without a reference value to compare
        // against. Saying so beats a silent mismatch; the choices decode is
        // covered by `gliner2_5_choice_decode_parity`.
        StructureField::ChoiceScalar(_) | StructureField::ChoiceList(_) => {
            panic!("this fixture declares no `choices`, so a choice field cannot appear")
        }
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
    for (name, env) in MODELS {
        let Some((_anchor, model)) = loaded_model(env) else {
            continue;
        };
        let data = fixture(name);
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
                for (field, value) in &g.fields {
                    let expected = want_fields
                        .get(field)
                        .unwrap_or_else(|| panic!("{text:?}[{index}]: no reference field {field}"));
                    let actual = value_json(value);
                    assert!(
                    close(&actual, expected),
                    "{name}[{text:?}][{index}]: field {field}\n  got  {actual}\n  want {expected}"
                );
                }
            }
        }
    }
}

#[test]
fn a_scalar_field_binds_exactly_one_span() {
    // The behaviour a span-per-field port cannot reproduce: a `dtype: "str"`
    // field reports `spans[0]` and discards the rest. The all-scalar case has two
    // city candidates, so binding more than one would be visible.
    //
    // The *expected* structure count comes from the reference, not from this
    // file. GLiNER2.5-multi-Decide finds nothing at threshold 0.5 on this
    // English text (only the 0.02 case yields anything), so a hardcoded "one
    // instance" would fail a checkpoint that is behaving correctly. What must
    // hold for every model is the dtype rule itself: a group whose fields are all
    // `"str"` never reports a list.
    for (name, env) in MODELS {
        let Some((_anchor, model)) = loaded_model(env) else {
            continue;
        };
        let data = fixture(name);
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
        let want = case["structures"].as_array().expect("structures");
        assert_eq!(
            got.len(),
            want.len(),
            "{name}: structure count must match the reference"
        );
        for instance in &got {
            for (field, value) in &instance.fields {
                match value {
                    // No schema here declares `choices`.
                    StructureField::ChoiceScalar(_) | StructureField::ChoiceList(_) => {
                        panic!("{name} {field}: this fixture declares no `choices`")
                    }
                    StructureField::Scalar(Some(span)) => {
                        assert!(span.start < span.end, "{name} {field}: empty span {span:?}")
                    }
                    // The rule under test: a `"str"` field is a scalar, never a
                    // list, however many candidates it had.
                    StructureField::List(spans) => panic!(
                        "{name} {field}: dtype \"str\" reported {} spans: {spans:?}",
                        spans.len()
                    ),
                    StructureField::Scalar(None) => {
                        panic!("{name} {field}: unexpected null scalar")
                    }
                }
            }
        }
        if let (Some(first), Some(expected)) = (got.first(), want.first()) {
            // The scalar city is the higher-scoring of the two candidates, which
            // is the resolver's `(-score, start, end)` order deciding the value.
            let want_city = expected["fields"]["city"]["text"].as_str().unwrap();
            let got_city = first
                .fields
                .iter()
                .find(|(field, _)| field == "city")
                .map(|(_, value)| match value {
                    StructureField::Scalar(Some(span)) => span.text.clone(),
                    _ => panic!("{name}: city is not a scalar"),
                })
                .unwrap();
            assert_eq!(
                got_city, want_city,
                "{name}: the scalar must be the best span"
            );
        }
    }
}

#[test]
fn an_unannotated_group_does_not_become_a_record() {
    // The mixed case declares an unannotated `person` group and an annotated
    // `trip` group. Only `person` may take the legacy structure path; `trip` is
    // the one that becomes a record. If the legacy path were skipped, or routed
    // through the record head, `person` would be missing or would appear as a
    // record.
    //
    // The routing invariant is model-independent, but whether `person` yields a
    // structure is not: GLiNER2.5-multi-Decide reports none at threshold 0.5.
    // So the expected legacy set comes from the reference, and the invariant this
    // test actually owns — `person` never reaches the record head — is asserted
    // outright.
    for (name, env) in MODELS {
        let Some((_anchor, model)) = loaded_model(env) else {
            continue;
        };
        let data = fixture(name);
        let case = data["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["schema"].get("record_metadata").is_some())
            .expect("the mixed case");
        let schema = &case["schema"];
        let (tasks, kinds) =
            rust_model_inference::app::parse_boundary_schema(schema).expect("schema");
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
                schema: schema.get("choices_schema"),
                entity_metadata: schema.get("entity_metadata"),
                relation_metadata: schema.get("relation_metadata"),
            },
        )
        .expect("mixed extraction");

        let legacy: Vec<&str> = extraction
            .structures
            .iter()
            .map(|structure| structure.task.as_str())
            .collect();
        let want_legacy: Vec<&str> = case["structures"]
            .as_array()
            .expect("structures")
            .iter()
            .map(|structure| structure["task"].as_str().unwrap())
            .collect();
        assert_eq!(
            legacy, want_legacy,
            "{name}: only the unannotated group may decode as a structure"
        );
        assert!(
            legacy.iter().all(|task| *task == "person"),
            "{name}: unexpected legacy group in {legacy:?}"
        );

        let recorded: Vec<&str> = extraction
            .records
            .iter()
            .map(|record| record.task.as_str())
            .collect();
        assert!(
            !recorded.contains(&"person"),
            "{name}: the unannotated group must not reach the record head: {recorded:?}"
        );
        for record in &extraction.records {
            assert_eq!(
                record.task, "trip",
                "{name}: the annotated group is the one that becomes a record"
            );
        }
    }
}
