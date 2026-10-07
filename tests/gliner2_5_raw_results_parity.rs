//! Byte-level parity for `Extraction` -> the reference's raw results dict.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_raw_results.py`, which drives the
//! real `batch_extract(format_results=False)` on the checkpoint. Because that is
//! a model-backed oracle, this test is too: the fixture's `raw` values are the
//! reference's own output for these texts and schemas, so the comparison covers
//! the word-to-character offset translation and every per-group key shape in one
//! pass. A pure oracle could pin the shapes but not the coordinate conversion,
//! which is the half most likely to be silently wrong.
//!
//! Run with `RMI_GLINER2_5_BASE_V1_GGUF=.../gliner2.5-base-v1-f32.gguf`.

use std::collections::BTreeSet;

use rust_model_inference::app::{run_gliner2_boundary_extract as extract, BoundarySchemaOptions};
use rust_model_inference::core::loader::GGUFLoader;
use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::gliner::prompt::normalize_text;
use rust_model_inference::models::gliner::prompt::BoundaryTaskKind;
use rust_model_inference::models::gliner_boundary::extract::query_layout;
use rust_model_inference::models::gliner_boundary::raw_results::{
    extraction_to_raw_results, EntityOrder, RawShape, ShapeFlags, WordCharSpans,
};
use rust_model_inference::models::gliner_boundary::BoundaryModel;
use serde_json::Value;

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/raw-results-golden.json";

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
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_raw_results.py"));
    serde_json::from_str(&raw).expect("parse raw-results-golden.json")
}

/// Scores differ from the reference's by f32 rounding on a different reduction
/// width, so the shape is compared exactly and the numbers within a tolerance.
/// Every assertion about *shape* is exact; only the confidence values move.
const TOLERANCE: f32 = 1.0e-3;

/// Compare two raw results dicts exactly, except for the confidence values.
fn assert_raw_eq(got: &Value, want: &Value, context: &str) {
    let mut differences = Vec::new();
    compare(got, want, "", &mut differences);
    assert!(
        differences.is_empty(),
        "{context}: {} structural difference(s)\n{}",
        differences.len(),
        differences.join("\n")
    );
}

fn compare(got: &Value, want: &Value, path: &str, out: &mut Vec<String>) {
    // The one value allowed to drift is a confidence, and only where both sides
    // carry one: a shape difference (a missing key, a list where an object
    // belongs) is still an exact failure.
    if path.ends_with(".confidence") {
        let (Some(a), Some(b)) = (got.as_f64(), want.as_f64()) else {
            out.push(format!(
                "{path}: confidence missing on one side: {got} vs {want}"
            ));
            return;
        };
        if (a - b).abs() > TOLERANCE as f64 {
            out.push(format!("{path}: {a} vs {b} (delta {})", (a - b).abs()));
        }
        return;
    }
    match (got, want) {
        (Value::Object(a), Value::Object(b)) => {
            for key in a.keys().chain(b.keys()) {
                if !a.contains_key(key) {
                    out.push(format!(
                        "{path}.{key}: missing on the left, want {}",
                        b[key]
                    ));
                } else if !b.contains_key(key) {
                    out.push(format!("{path}.{key}: extra on the left: {}", a[key]));
                } else {
                    compare(&a[key], &b[key], &format!("{path}.{key}"), out);
                }
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                out.push(format!("{path}: length {} vs {}", a.len(), b.len()));
                return;
            }
            for (index, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                compare(x, y, &format!("{path}[{index}]"), out);
            }
        }
        _ => {
            if got != want {
                out.push(format!("{path}: {got} vs {want}"));
            }
        }
    }
}

/// The schema-level shape, read the way the port's prompt routing already reads
/// it, so the test cannot disagree with the extractor about query order.
fn raw_shape(
    metadata: &Value,
    specs: &[rust_model_inference::models::gliner_boundary::extract::QuerySpec],
    schema: &Value,
) -> RawShape {
    let declared: Vec<String> = metadata["entity_order"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|item| item.as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default();
    let routed: Vec<String> = specs
        .iter()
        .filter(|spec| spec.task_type == "entities")
        .map(|spec| spec.field_name.clone())
        .collect();
    // A builder-built schema carries an explicit order; a plain dict does not,
    // and the reference falls back to the schema's own key order, which is what
    // the routed list already is.
    let labels = if declared.is_empty() {
        routed
    } else {
        declared
    };
    let scalar: Vec<String> = labels
        .iter()
        .filter(|label| metadata["entity_metadata"][label]["dtype"].as_str() == Some("str"))
        .cloned()
        .collect();
    let mut structure_groups: Vec<String> = metadata["structure_groups"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|item| item.as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default();
    if structure_groups.is_empty() {
        // The port routes a structure group by its query name, so the groups it
        // actually decoded are the ones a caller declared.
        structure_groups = specs
            .iter()
            .filter(|spec| spec.task_type == "json_structures")
            .map(|spec| spec.task_name.clone())
            .collect();
        structure_groups.dedup();
    }
    let _ = schema;
    RawShape {
        entities: EntityOrder { labels, scalar },
        classification_tasks: metadata["classification_tasks"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .map(|item| item.as_str().unwrap().to_string())
                    .collect()
            })
            .unwrap_or_default(),
        structure_groups,
        // The converter does not consult this: `relation_types` is what
        // `format_results` needs, and this test compares the *raw* dict, which is
        // shaped before any of that. Filled from the schema's declared relations so
        // the struct is complete.
        relation_types: specs
            .iter()
            .filter(|spec| spec.task_type == "relations")
            .map(|spec| {
                rust_model_inference::models::gliner_boundary::relations::resolve_relation_type(
                    &spec.field_name,
                )
            })
            .collect(),
    }
}

#[test]
fn raw_results_match_the_reference_on_every_case() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    let cases = data["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let text = case["text"].as_str().unwrap();
        let threshold = case["threshold"].as_f64().unwrap() as f32;
        let schema = case["schema"].clone();
        let metadata = case["metadata"].clone();

        let (tasks, kinds) =
            rust_model_inference::app::parse_boundary_schema(&schema).expect("parse schema");
        let options = BoundarySchemaOptions {
            entity_metadata: Some(&metadata["entity_metadata"]),
            field_metadata: Some(&metadata["field_metadata"]),
            schema: Some(&schema),
            ..BoundarySchemaOptions::default()
        };
        // The adapter's `extract`, not `run_mixed_extraction`: the adapter is
        // what decodes spans after the forward pass, and calling the lower-level
        // function returns an `Extraction` with an empty `spans` list. That is
        // exactly the kind of mismatch a converter test can hide — every
        // assertion below still passes on an empty extraction if the spans are
        // compared loosely, so the entry point has to be the real one.
        let extraction =
            extract(&model, text, &tasks, &kinds, 0, Some(threshold), options).expect("extract");

        let specs = query_layout(&tasks, &kinds);
        let shape = raw_shape(&metadata, &specs, &schema);
        let got = extraction_to_raw_results(
            &extraction,
            &WordCharSpans::for_text(text),
            &rust_model_inference::models::gliner::prompt::normalize_text(text),
            &shape,
            ShapeFlags {
                include_confidence: case["include_confidence"].as_bool().unwrap(),
                include_spans: case["include_spans"].as_bool().unwrap(),
            },
        );

        let mut got_value = Value::Object(got);
        sort_keys(&mut got_value);
        let mut want = case["raw"].clone();
        sort_keys(&mut want);
        assert_raw_eq(&got_value, &want, name);
    }
}

/// Key order is not compared — the reference builds a relation side as
/// `{text, start, end}` and then assigns `confidence`, so its key order differs
/// from a span's even though the content matches. Sorting keeps the comparison
/// about content, and this is what makes that difference invisible on purpose
/// rather than by accident.
fn sort_keys(value: &mut Value) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            let sorted: Vec<(String, Value)> = keys
                .into_iter()
                .map(|key| {
                    let mut child = map[&key].clone();
                    sort_keys(&mut child);
                    (key, child)
                })
                .collect();
            let mut rebuilt = serde_json::Map::new();
            for (key, child) in sorted {
                rebuilt.insert(key, child);
            }
            *map = rebuilt;
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                sort_keys(item);
            }
        }
        _ => {}
    }
}

/// The offset translation is the half a shape-only comparison cannot see, so it
/// is checked directly against the word list: every reported character offset
/// must land on a word boundary in the fixture's text, and the reported surface
/// must be what slicing the text at those offsets yields.
#[test]
fn reported_offsets_are_character_offsets_into_the_original_text() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    // The two cases whose texts carry punctuation and repeated whitespace,
    // where a word index used as a character offset would still be in range.
    let wanted = [
        "entities_offsets_are_characters_not_words",
        "entities_offsets_survive_punctuation_and_spacing",
    ];
    let mut checked = 0;

    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if !wanted.contains(&name) {
            continue;
        }
        checked += 1;
        let text = case["text"].as_str().unwrap();
        let schema = case["schema"].clone();
        let metadata = case["metadata"].clone();
        let (tasks, kinds) =
            rust_model_inference::app::parse_boundary_schema(&schema).expect("parse schema");
        let extraction = extract(
            &model,
            text,
            &tasks,
            &kinds,
            0,
            Some(case["threshold"].as_f64().unwrap() as f32),
            BoundarySchemaOptions {
                entity_metadata: Some(&metadata["entity_metadata"]),
                field_metadata: Some(&metadata["field_metadata"]),
                schema: Some(&schema),
                ..BoundarySchemaOptions::default()
            },
        )
        .expect("extract");

        let specs = query_layout(&tasks, &kinds);
        let shape = raw_shape(&metadata, &specs, &schema);
        let got = extraction_to_raw_results(
            &extraction,
            &WordCharSpans::for_text(text),
            &rust_model_inference::models::gliner::prompt::normalize_text(text),
            &shape,
            ShapeFlags {
                include_confidence: case["include_confidence"].as_bool().unwrap(),
                include_spans: case["include_spans"].as_bool().unwrap(),
            },
        );
        let spans = got["entities"][0].as_object().expect("entities map");
        assert!(!spans.is_empty(), "{name}: no entity spans reported");

        let mut saw_multiword = false;
        for (label, list) in spans {
            for span in list.as_array().unwrap() {
                let start = span["start"].as_u64().unwrap() as usize;
                let end = span["end"].as_u64().unwrap() as usize;
                // The surface is the text at the reported offsets, so a
                // converter that reported word indices would produce a string
                // that does not match this slice.
                let sliced: String = text.chars().skip(start).take(end - start).collect();
                assert_eq!(
                    span["text"].as_str().unwrap(),
                    sliced.trim(),
                    "{name}/{label}: surface does not match text[{start}..{end}]",
                );
                // And the offsets must be code points, not bytes: a multi-byte
                // character before the span would make the byte offset overshoot
                // the text while the code point offset does not.
                assert!(
                    end <= text.chars().count(),
                    "{name}: {end} exceeds the text"
                );
                if sliced.trim().contains(' ') {
                    saw_multiword = true;
                }
            }
        }
        assert!(
            saw_multiword,
            "{name}: expected at least one multi-word span, else a word-index \
             converter would pass this case",
        );
    }
    assert_eq!(
        checked,
        wanted.len(),
        "both cases should have been exercised"
    );
}

/// A schema that declares no entities produces no `entities` key at all, rather
/// than an empty one — the key is absent, not present-and-empty.
#[test]
fn a_schema_without_entities_omits_the_key() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "classifications")
        .expect("the classification case declares no entities");
    let text = case["text"].as_str().unwrap();
    let schema = case["schema"].clone();
    let metadata = case["metadata"].clone();
    let (tasks, kinds) =
        rust_model_inference::app::parse_boundary_schema(&schema).expect("parse schema");
    let extraction = extract(
        &model,
        text,
        &tasks,
        &kinds,
        0,
        Some(case["threshold"].as_f64().unwrap() as f32),
        BoundarySchemaOptions::default(),
    )
    .expect("extract");
    let shape = raw_shape(&metadata, &query_layout(&tasks, &kinds), &schema);
    assert!(
        shape.entities.labels.is_empty(),
        "this case exists because its schema declares no entity labels",
    );
    let got = extraction_to_raw_results(
        &extraction,
        &WordCharSpans::for_text(text),
        &rust_model_inference::models::gliner::prompt::normalize_text(text),
        &shape,
        ShapeFlags::default(),
    );
    assert!(
        !got.contains_key("entities"),
        "an undeclared group must be absent, not empty: {}",
        Value::Object(got),
    );
    assert!(
        got.contains_key("sentiment"),
        "the classification key is present"
    );
}

/// Both flags off reduces a span to its bare text, which is what the
/// single-document path does when the caller asks for neither.
#[test]
fn dropping_the_flags_drops_the_corresponding_keys() {
    let Some((_anchor, model)) = loaded_model() else {
        return;
    };
    let data = fixture();
    // The baseline has to be a case that was *itself* dumped with the flags off.
    // Reaching for a both-on case here compares a flags-off result against
    // dicts and reports a difference that looks like a converter bug but is the
    // test pointing at the wrong baseline.
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "entities_no_flags")
        .expect("the flags-off case");
    assert!(
        !case["include_confidence"].as_bool().unwrap() && !case["include_spans"].as_bool().unwrap(),
        "this baseline must have been dumped with both flags off",
    );
    let text = case["text"].as_str().unwrap();
    let schema = case["schema"].clone();
    let metadata = case["metadata"].clone();
    let (tasks, kinds) =
        rust_model_inference::app::parse_boundary_schema(&schema).expect("parse schema");
    let extraction = extract(
        &model,
        text,
        &tasks,
        &kinds,
        0,
        Some(case["threshold"].as_f64().unwrap() as f32),
        BoundarySchemaOptions::default(),
    )
    .expect("extract");
    assert!(!extraction.spans.is_empty(), "the baseline must find spans");
    let shape = raw_shape(&metadata, &query_layout(&tasks, &kinds), &schema);
    let normalized = rust_model_inference::models::gliner::prompt::normalize_text(text);
    let got = extraction_to_raw_results(
        &extraction,
        &WordCharSpans::for_text(text),
        &normalized,
        &shape,
        ShapeFlags {
            include_confidence: false,
            include_spans: false,
        },
    );
    // With neither flag a span is the bare surface string, not a one-key object.
    // The fixture is the authority on that, so the comparison is against it
    // rather than against an expectation written here.
    assert_raw_eq(&Value::Object(got), &case["raw"], "entities_no_flags");
}

/// `WordCharSpans` is the offset machinery, so it gets its own checks: an
/// empty span list cannot panic, and an out-of-range span disappears rather than
/// clamping to a plausible-looking range.
#[test]
fn word_char_spans_resolve_or_report_absent() {
    let spans = WordCharSpans::for_text("Ada Lovelace, 1815 -- London.");
    // Observed split: `Ada` `Lovelace` `,` `1815` `-` `-` `London` `.` — eight
    // tokens, because the reference's whitespace splitter emits each `-` of `--`
    // separately. I expected six when writing this, so the count is asserted
    // against what the splitter actually does rather than against a reading of
    // the text; the ranges below are the part that matters.
    assert_eq!(spans.len(), 8, "the dash pair splits into two tokens");
    // Words 0..2 is "Ada Lovelace", whose end is the comma's offset.
    assert_eq!(spans.char_range(0, 2), Some((0, 12)));
    // Word 2 is the comma alone, so the range is empty-width.
    assert_eq!(spans.char_range(2, 3), Some((12, 13)));
    // A single word.
    assert_eq!(spans.char_range(0, 1), Some((0, 3)));
    // Out of range in either direction, and the empty range. The reference
    // absorbs these with an IndexError guard and skips the span.
    assert_eq!(spans.char_range(0, 99), None);
    assert_eq!(spans.char_range(4, 2), None);
    assert_eq!(spans.char_range(3, 3), None);
    assert!(WordCharSpans::default().char_range(0, 1).is_none());
}

/// The unit-level half: the choice shapes, which carry no offsets and so cannot
/// be covered by the flag test above.
#[test]
fn a_choice_value_has_no_document_location() {
    // `structures_choices_scalar_and_list` is the fixture case; the port's own
    // structure decode decides which of the four `StructureField` variants it is,
    // and the raw output is what distinguishes them. Read it back from the
    // fixture rather than re-decoding, since the point is the *shape*.
    let data = fixture();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "structures_choices_scalar_and_list")
        .expect("the choices case");
    let fields = &case["raw"]["paper"][0];
    for field in ["topic", "mood"] {
        let value = &fields[field];
        let entries: Vec<&Value> = match value {
            Value::Array(items) => items.iter().collect(),
            other => vec![other],
        };
        for entry in entries {
            assert!(
                entry["text"].is_string(),
                "{field}: text is present: {entry}"
            );
            assert!(
                entry["confidence"].is_number(),
                "{field}: confidence is present: {entry}",
            );
            assert!(
                !entry.as_object().unwrap().contains_key("start")
                    && !entry.as_object().unwrap().contains_key("end"),
                "{field}: a choice has no document location, so it must not \
                 carry offsets: {entry}",
            );
        }
    }
    // The `str` field collapsed to a single object; the `list` field stayed a
    // list. Two renderings, not one.
    assert!(
        fields["topic"].is_object(),
        "a scalar choice field is one object"
    );
    assert!(
        fields["mood"].is_array(),
        "a list choice field stays a list"
    );
}

/// The dtype-`str` entity collapse is a shape change with two distinct null-ish
/// outcomes, so the fixture's own two cases are what distinguish them.
#[test]
fn a_scalar_entity_is_a_dict_or_null_and_never_a_list() {
    let data = fixture();
    let cases = data["cases"].as_array().unwrap();
    let collapsed = cases
        .iter()
        .find(|case| case["name"] == "entities_scalar_dtype_collapses_to_one_value")
        .expect("the collapse case");
    let empty = cases
        .iter()
        .find(|case| case["name"] == "entities_scalar_dtype_with_nothing_to_report")
        .expect("the nothing-to-report case");

    let value = &collapsed["raw"]["entities"][0]["person"];
    assert!(
        value.is_object(),
        "a scalar dtype collapses to one object, not a list: {value}",
    );
    assert!(
        value["start"].is_u64() && value["end"].is_u64(),
        "it carries offsets"
    );

    let nothing = &empty["raw"]["entities"][0]["person"];
    assert!(
        nothing.is_null(),
        "with nothing to report a scalar dtype is null, not [] and not {{}}: {nothing}",
    );
}

/// Two mutations of `surface` survive, and both are *equivalent* rather than
/// uncaught, so the gap is recorded instead of papered over with a case.
///
/// `surface` is `text[char_start..char_end].strip()`, and the character range
/// always comes from `WordCharSpans`, which spans `[first_char_of_word,
/// last_char_of_word]`. A word never starts or ends with whitespace, so the slice
/// never has any to strip: dropping the `.trim()` and adjusting the offsets to
/// match the trimmed text both produce byte-identical output on every input this
/// port can produce. The reference keeps the `.strip()` because its word splitter
/// is pluggable and a different one could yield whitespace-bearing spans, so the
/// call is faithful rather than defensive.
///
/// A fixture case cannot close this gap, because reaching the stripping path needs
/// a span whose range covers whitespace, which no word-boundary range can. What
/// *is* pinned is the offset half of the same rule: the reported `start`/`end` are
/// the raw character range, never the trimmed one, and the main parity test
/// compares them against the reference's.
#[test]
fn the_strip_is_unreachable_because_word_spans_never_cover_whitespace() {
    // Stated as a property of the word map rather than of any fixture case, since
    // it is what makes the two surviving mutations equivalent.
    for text in [
        "Ada Lovelace, 1815 -- London.",
        "Marie Curie worked in Paris.",
        "A meeting in Berlin with Dr. Kwame Nkrumah and Ann.",
        "Deep Learning by Yoshua Bengio.",
    ] {
        let spans = WordCharSpans::for_text(text);
        let chars: Vec<char> = normalize_text(text).chars().collect();
        for (start, end) in &spans.0 {
            assert!(
                chars.get(*start).is_some_and(|c| !c.is_whitespace()),
                "{text:?}: word at {start} starts on whitespace",
            );
            assert!(
                chars
                    .get(end.saturating_sub(1))
                    .is_some_and(|c| !c.is_whitespace()),
                "{text:?}: word ending at {end} ends on whitespace",
            );
        }
    }
}

/// Guards against the fixture shrinking unnoticed, which would silently reduce
/// the coverage above.
#[test]
fn the_fixture_still_covers_every_shape() {
    let data = fixture();
    let cases = data["cases"].as_array().unwrap();
    let names: BTreeSet<&str> = cases
        .iter()
        .map(|case| case["name"].as_str().unwrap())
        .collect();
    for expected in [
        "entities",
        "entities_offsets_are_characters_not_words",
        "entities_offsets_survive_punctuation_and_spacing",
        "entities_scalar_dtype_collapses_to_one_value",
        "entities_scalar_dtype_with_nothing_to_report",
        "structures_choices_scalar_and_list",
        "relations_beside_entities",
        "classifications",
        "entities_no_flags",
        "structures_no_flags",
        "relations_no_flags",
        "entities_scalar_dtype_no_flags",
    ] {
        assert!(names.contains(expected), "fixture lost the {expected} case");
    }
    assert_eq!(names.len(), cases.len(), "case names must be unique");
}
