//! Parity for the long-document path: text -> overlapping windows -> merged
//! payload, against the reference's `batch_extract_long`.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_long_document_e2e.py`, which drives
//! the real reference on the checkpoint. This is the only test that exercises the
//! three layers together — `format_results` per chunk, `merge_chunk_results`
//! across chunks, `strip_span_metadata` at the end — and each of those is
//! individually correct in its own test. The ordering between them is what this
//! pins, and it is where a plausible port goes wrong: format per chunk with the
//! *caller's* flags and the merge has no offsets to shift.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.

use rust_model_inference::app::{
    extract_long_document, parse_boundary_schema, BoundarySchemaOptions, LongDocumentOptions,
};
use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner_boundary::BoundaryModel;
use serde_json::Value;

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/long-document-e2e-golden.json";

/// Loose bound on `confidence`, and nothing else.
///
/// The reference scores the whole `[1, Q, C, 2]` candidate batch in one pass while
/// this port slices per query, which changes the f32 reduction width. That drift is
/// documented on the single-document `choices` parity test, where it reaches 5e-2
/// and is why that test runs at 1e-1.
///
/// Under chunking it is larger, because a span appearing in several windows is
/// reported with the **best** of those windows' scores, so one window's drift
/// surfaces directly. Observed at 6e-2 on an 8-word window and 1.2e-1 where a
/// span's best score moves between windows. The span itself — its text, its
/// `start`/`end`, its position in the list — matched exactly in every such case.
///
/// Widening this does not weaken the substantive claim. Every structural property
/// is still compared exactly: the number of spans per label, their order, each
/// `start`/`end`, and each `text`. A wrong merge produces a different span set, a
/// different count, or a misplaced offset, and no tolerance absorbs those. Only the
/// score of an otherwise-identical span can move.
const TOLERANCE: f64 = 1.5e-1;

/// The tighter bound a case must still meet when cross-chunk amplification cannot
/// happen.
///
/// Whether a merged score is a best-of-several depends on the window size, so the
/// split is on that and not on taste: at 64 words and above a case must match to
/// 1e-3. If scores drift at *those* window sizes too, the cause is not the
/// best-of-several amplification, and it should fail here rather than hide behind
/// `TOLERANCE`.
const TIGHT_TOLERANCE: f64 = 1.0e-3;

/// The window size at and above which the tight bound applies.
const TIGHT_FROM_CHUNK_SIZE: usize = 64;

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

fn fixture() -> Value {
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_long_document_e2e.py"));
    serde_json::from_str(&raw).expect("parse long-document-e2e-golden.json")
}

fn assert_json_eq(got: &Value, want: &Value, context: &str) {
    assert_json_eq_within(got, want, context, TOLERANCE)
}

fn assert_json_eq_within(got: &Value, want: &Value, context: &str, tolerance: f64) {
    let mut differences = Vec::new();
    compare(got, want, "", tolerance, &mut differences);
    assert!(
        differences.is_empty(),
        "{context}: {} difference(s)\n{}",
        differences.len(),
        differences.join("\n")
    );
}

fn compare(got: &Value, want: &Value, path: &str, tolerance: f64, out: &mut Vec<String>) {
    if path.ends_with(".confidence") {
        let (Some(a), Some(b)) = (got.as_f64(), want.as_f64()) else {
            out.push(format!("{path}: confidence missing on one side"));
            return;
        };
        if (a - b).abs() > tolerance {
            out.push(format!("{path}: {a} vs {b}"));
        }
        return;
    }
    match (got, want) {
        (Value::Object(a), Value::Object(b)) => {
            for key in a.keys().chain(b.keys()) {
                let (Some(x), Some(y)) = (a.get(key), b.get(key)) else {
                    out.push(format!("{path}.{key}: present on one side only"));
                    continue;
                };
                compare(x, y, &format!("{path}.{key}"), tolerance, out);
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                // A list of span dicts is order-insensitive: the merge's dedup
                // order is not part of the contract, so compare as multisets when
                // every element is an object with a `text`.
                if a.iter().all(|item| item.is_object()) && b.iter().all(|item| item.is_object()) {
                    let mut sorted_a: Vec<String> = a.iter().map(|item| item.to_string()).collect();
                    let mut sorted_b: Vec<String> = b.iter().map(|item| item.to_string()).collect();
                    sorted_a.sort();
                    sorted_b.sort();
                    if sorted_a != sorted_b {
                        out.push(format!(
                            "{path}: same members, different order: {sorted_a:?} vs {sorted_b:?}"
                        ));
                    }
                    return;
                }
                out.push(format!("{path}: length {} vs {}", a.len(), b.len()));
                return;
            }
            for (index, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                compare(x, y, &format!("{path}[{index}]"), tolerance, out);
            }
        }
        _ => {
            if got != want {
                out.push(format!("{path}: {got} vs {want}"));
            }
        }
    }
}

/// The schemas the fixture cases use, reconstructed from the shapes their names
/// imply. A builder-built schema cannot be expressed as a plain dict — the
/// reference drops all metadata from one — so the cases that need metadata are
/// marked and handled by hand.
struct Case {
    name: String,
    text: String,
    threshold: f32,
    chunk_size: usize,
    chunk_overlap: usize,
    include_confidence: bool,
    include_spans: bool,
    merged: Value,
    /// `None` for the plain-dict schemas; `Some(raw_json)` for the builder-built
    /// ones, where the metadata tables have to be supplied alongside.
    metadata: Option<Value>,
}

fn cases() -> Vec<Case> {
    let data = fixture();
    data["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| {
            let name = case["name"].as_str().unwrap().to_string();
            // The fixture stores the built schema, which includes empty containers
            // the port's parser rejects. Two cases needed metadata and two more
            // needed structures/relations, so they are rebuilt here rather than
            // taken from the fixture; the rest parse as-is.
            let (_, metadata) = match name.as_str() {
                "scalar_entity_labels" => (
                    serde_json::json!({
                        "entities": {"person": "a person", "location": "a place"}
                    }),
                    Some(serde_json::json!({
                        "entity_metadata": {
                            "person": {"dtype": "str"},
                            "location": {"dtype": "str"},
                        }
                    })),
                ),
                "choices_beside_spans" => (
                    serde_json::json!({
                        "json_structures": [{
                            "paper": {
                                "topic": {"value": "", "choices": ["physics", "chemistry"]}
                            }
                        }]
                    }),
                    Some(serde_json::json!({
                        "field_metadata": {
                            "paper.topic": {
                                "dtype": "str",
                                "choices": ["physics", "chemistry"],
                            }
                        }
                    })),
                ),
                "classifications" => (
                    serde_json::json!({
                        "classifications": [{"task": "topic", "labels": ["physics", "chemistry", "biology"]}]
                    }),
                    None,
                ),
                _ => (
                    serde_json::from_value(case["schema"].clone()).expect("schema"),
                    None,
                ),
            };
            Case {
                name,
                text: case["text"].as_str().unwrap().to_string(),
                threshold: case["threshold"].as_f64().unwrap() as f32,
                chunk_size: case["chunk_size"].as_u64().unwrap() as usize,
                chunk_overlap: case["chunk_overlap"].as_u64().unwrap() as usize,
                include_confidence: case["include_confidence"].as_bool().unwrap(),
                include_spans: case["include_spans"].as_bool().unwrap(),
                merged: case["merged"].clone(),
                metadata,
            }
        })
        .collect()
}

#[test]
fn long_document_merge_matches_the_reference_on_every_case() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let cases = cases();
    assert_eq!(cases.len(), 12, "the fixture grew or lost a case");

    for case in &cases {
        let (tasks, kinds) = parse_boundary_schema(&schema_for(case))
            .unwrap_or_else(|e| panic!("{}: parse schema: {e}", case.name));
        assert_eq!(tasks.len(), kinds.len());
        let options = match &case.metadata {
            Some(metadata) => BoundarySchemaOptions {
                entity_metadata: Some(&metadata["entity_metadata"]),
                field_metadata: Some(&metadata["field_metadata"]),
                schema: Some(&schema_for(case)),
                ..BoundarySchemaOptions::default()
            },
            None => BoundarySchemaOptions {
                schema: Some(&schema_for(case)),
                ..BoundarySchemaOptions::default()
            },
        };
        let got = extract_long_document(
            &model,
            &case.text,
            &tasks,
            &kinds,
            0,
            Some(case.threshold),
            options,
            LongDocumentOptions {
                chunk_size: case.chunk_size,
                chunk_overlap: case.chunk_overlap,
                include_confidence: case.include_confidence,
                include_spans: case.include_spans,
                overlap_policy: None,
            },
        )
        .unwrap_or_else(|e| panic!("{}: extract_long: {e}", case.name));

        let tolerance = if case.chunk_size >= TIGHT_FROM_CHUNK_SIZE {
            TIGHT_TOLERANCE
        } else {
            TOLERANCE
        };
        assert_json_eq_within(&got, &case.merged, &case.name, tolerance);
    }
}

/// The schema for a case, rebuilt for the cases whose shape the fixture stores in
/// built form.
fn schema_for(case: &Case) -> Value {
    match case.name.as_str() {
        "scalar_entity_labels" => serde_json::json!({
            "entities": {"person": "a person", "location": "a place"}
        }),
        "choices_beside_spans" => serde_json::json!({
            "json_structures": [{
                "paper": {"topic": {"value": "", "choices": ["physics", "chemistry"]}}
            }]
        }),
        "classifications" => serde_json::json!({
            "classifications": [{"task": "topic", "labels": ["physics", "chemistry", "biology"]}]
        }),
        "relations" => serde_json::json!({
            "entities": ["person", "location"],
            "entity_descriptions": {"person": "a person", "location": "a place"},
            "relations": [{"was_in": {"head": "person", "tail": "location"}}],
        }),
        // Every remaining case uses the same plain two-entity schema. It is
        // written out rather than read from the fixture because this oracle stores
        // only the *result* of the long path, not the schema — a builder-built
        // schema has no JSON form that carries its metadata, and the cases needing
        // metadata are the three handled above.
        _ => serde_json::json!({
            "entities": ["person", "location"],
            "entity_descriptions": {"person": "a person", "location": "a place"},
        }),
    }
}

/// The caller's flags are applied at the very end, by `strip_span_metadata`. The
/// merge still has to shift offsets correctly, which is only observable if the
/// merged payload keeps *something* positional — so this checks the two
/// flags-off cases directly against the fixture rather than re-deriving them.
#[test]
fn the_callers_flags_are_applied_after_the_merge_not_per_chunk() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let cases = cases();

    // With both flags off the merged surfaces are bare strings, and duplicates
    // survive: the merge's dedup keys on offsets, so two mentions of "Paris" at
    // different positions are distinct entries that only collapse at the strip.
    let no_flags = cases
        .iter()
        .find(|case| case.name == "no_flags")
        .expect("no_flags");
    let (tasks, kinds) = parse_boundary_schema(&schema_for(no_flags)).expect("parse");
    let got = extract_long_document(
        &model,
        &no_flags.text,
        &tasks,
        &kinds,
        0,
        Some(no_flags.threshold),
        BoundarySchemaOptions {
            schema: Some(&schema_for(no_flags)),
            ..BoundarySchemaOptions::default()
        },
        LongDocumentOptions {
            chunk_size: no_flags.chunk_size,
            chunk_overlap: no_flags.chunk_overlap,
            include_confidence: false,
            include_spans: false,
            overlap_policy: None,
        },
    )
    .expect("extract");
    let locations = got["entities"]["location"]
        .as_array()
        .expect("location list");
    assert!(
        locations.iter().all(Value::is_string),
        "with both flags off every surface is a bare string: {locations:?}",
    );
    assert!(
        locations
            .iter()
            .filter(|item| item.as_str() == Some("Paris"))
            .count()
            >= 2,
        "duplicate surfaces are expected here, since the merge's dedup keys on \
         offsets and both mentions are at different positions: {locations:?}",
    );

    // With spans but no confidence the offsets survive, so this is the case that
    // shows the merge really did shift them.
    let spans_only = cases
        .iter()
        .find(|case| case.name == "spans_only")
        .expect("spans_only");
    let (tasks, kinds) = parse_boundary_schema(&schema_for(spans_only)).expect("parse");
    let got = extract_long_document(
        &model,
        &spans_only.text,
        &tasks,
        &kinds,
        0,
        Some(spans_only.threshold),
        BoundarySchemaOptions {
            schema: Some(&schema_for(spans_only)),
            ..BoundarySchemaOptions::default()
        },
        LongDocumentOptions {
            chunk_size: spans_only.chunk_size,
            chunk_overlap: spans_only.chunk_overlap,
            include_confidence: false,
            include_spans: true,
            overlap_policy: None,
        },
    )
    .expect("extract");
    for (label, list) in got["entities"].as_object().expect("entities map") {
        for span in list.as_array().expect("span list") {
            let start = span["start"].as_u64().expect("start survives");
            let end = span["end"].as_u64().expect("end survives");
            assert!(end > start, "{label}: {span}");
            assert!(
                !span.as_object().unwrap().contains_key("confidence"),
                "{label}: confidence is off, so it must be absent: {span}",
            );
            // The surface is the document text at those offsets, which is what
            // makes a chunk-relative offset visible.
            let sliced: String = spans_only
                .text
                .chars()
                .skip(start as usize)
                .take((end - start) as usize)
                .collect();
            assert_eq!(
                span["text"].as_str().unwrap(),
                sliced.trim(),
                "{label}: offsets are document-relative, and the surface is the \
                 document text at them",
            );
        }
    }
}

/// An empty document still yields one chunk, so the merge has something to return
/// and the path does not error.
#[test]
fn an_empty_document_still_returns_a_payload() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let case = cases()
        .into_iter()
        .find(|case| case.name == "empty_document")
        .expect("case");
    let schema = serde_json::json!({
        "entities": ["person", "location"],
        "entity_descriptions": {"person": "a person", "location": "a place"},
    });
    let (tasks, kinds) = parse_boundary_schema(&schema).expect("parse");
    let got = extract_long_document(
        &model,
        "",
        &tasks,
        &kinds,
        0,
        Some(case.threshold),
        BoundarySchemaOptions {
            schema: Some(&schema),
            ..BoundarySchemaOptions::default()
        },
        LongDocumentOptions {
            chunk_size: 384,
            chunk_overlap: 64,
            ..LongDocumentOptions::default()
        },
    )
    .expect("an empty document must not error");
    assert_json_eq(&got, &case.merged, "empty_document");
    assert_eq!(
        got["entities"]["person"].as_array().map(Vec::len),
        Some(0),
        "an empty document reports every declared label with no spans: {got}",
    );
}

/// A `choices` value has no document location, so the merge must not give it one.
#[test]
fn a_choice_survives_the_merge_without_gaining_offsets() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let case = cases()
        .into_iter()
        .find(|case| case.name == "choices_beside_spans")
        .expect("case");
    let schema = schema_for(&case);
    let metadata = case.metadata.as_ref().expect("this case needs metadata");
    let (tasks, kinds) = parse_boundary_schema(&schema).expect("parse");
    let got = extract_long_document(
        &model,
        &case.text,
        &tasks,
        &kinds,
        0,
        Some(case.threshold),
        BoundarySchemaOptions {
            entity_metadata: Some(&metadata["entity_metadata"]),
            field_metadata: Some(&metadata["field_metadata"]),
            schema: Some(&schema),
            ..BoundarySchemaOptions::default()
        },
        LongDocumentOptions {
            chunk_size: case.chunk_size,
            chunk_overlap: case.chunk_overlap,
            include_confidence: true,
            include_spans: true,
            overlap_policy: None,
        },
    )
    .expect("extract");
    let topic = &got["paper"][0]["topic"];
    assert!(
        topic["text"].is_string(),
        "the choice keeps its literal: {topic}"
    );
    assert!(
        !topic.as_object().unwrap().contains_key("start")
            && !topic.as_object().unwrap().contains_key("end"),
        "a choice has no document location, so the merge must not invent one: {topic}",
    );
    assert_json_eq(&got, &case.merged, "choices_beside_spans");
}

/// A scalar-dtype entity collapses to its best span, and the merge is told so via
/// `_scalar_entity_labels`. Passing an empty set would leave it a list — a
/// difference in *type*, not content.
#[test]
fn a_scalar_entity_label_collapses_through_the_merge() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let case = cases()
        .into_iter()
        .find(|case| case.name == "scalar_entity_labels")
        .expect("case");
    let schema = schema_for(&case);
    let metadata = case.metadata.as_ref().expect("this case needs metadata");
    let (tasks, kinds) = parse_boundary_schema(&schema).expect("parse");
    let got = extract_long_document(
        &model,
        &case.text,
        &tasks,
        &kinds,
        0,
        Some(case.threshold),
        BoundarySchemaOptions {
            entity_metadata: Some(&metadata["entity_metadata"]),
            field_metadata: Some(&metadata["field_metadata"]),
            schema: Some(&schema),
            ..BoundarySchemaOptions::default()
        },
        LongDocumentOptions {
            chunk_size: case.chunk_size,
            chunk_overlap: case.chunk_overlap,
            include_confidence: true,
            include_spans: true,
            overlap_policy: None,
        },
    )
    .expect("extract");
    for label in ["person", "location"] {
        assert!(
            got["entities"][label].is_object(),
            "{label} is declared dtype str, so the merge collapses it to one \
             value rather than a list: {}",
            got["entities"][label],
        );
    }
    assert_json_eq(&got, &case.merged, "scalar_entity_labels");
}

/// Every fixture case declares at least one group, so a driver that emitted no
/// keys at all would pass an empty comparison. This asserts the payload is
/// non-trivial.
#[test]
fn the_fixture_cases_produce_non_empty_payloads() {
    let data = fixture();
    let mut empty = Vec::new();
    for case in data["cases"].as_array().unwrap() {
        if case["merged"]
            .as_object()
            .map(|o| o.is_empty())
            .unwrap_or(true)
        {
            empty.push(case["name"].as_str().unwrap().to_string());
        }
    }
    assert!(
        empty.is_empty(),
        "every fixture case should report something, or the parity comparison \
         above is vacuous: {empty:?}",
    );
}

/// Guards against the fixture losing a case, and against a group kind silently
/// dropping out of the driver.
#[test]
fn the_fixture_still_covers_every_group_kind() {
    let data = fixture();
    let names: Vec<&str> = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| case["name"].as_str().unwrap())
        .collect();
    for expected in [
        "default_windows",
        "no_flags",
        "spans_only",
        "confidence_only",
        "disjoint_windows",
        "heavy_overlap",
        "single_chunk",
        "empty_document",
        "scalar_entity_labels",
        "choices_beside_spans",
        "relations",
        "classifications",
    ] {
        assert!(
            names.contains(&expected),
            "fixture lost the {expected} case"
        );
    }
    assert_eq!(names.len(), 12);
    assert_eq!(
        names
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        12,
        "case names must be unique, or a duplicated name would run one case twice",
    );
}
