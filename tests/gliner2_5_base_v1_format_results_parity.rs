//! Parity with `format_results` and the three struct formatters.
//!
//! The fixture is `tools/oracle/gliner_boundary/dump_format_results.py`, which
//! drives the reference directly. These are pure functions over plain data, so
//! no GGUF is needed.

use std::collections::BTreeMap;

use rust_model_inference::models::gliner_boundary::formatting::{
    format_classification, format_entity_dict, format_results_as, format_struct,
};
use serde_json::Value;

fn fixture() -> Value {
    let raw = include_str!("fixtures/gliner2.5-base-v1/format-results-golden.json");
    serde_json::from_str(raw).expect("the format-results fixture is valid JSON")
}

struct Case {
    name: String,
    results: Value,
    include_confidence: bool,
    requested_relations: Vec<String>,
    classification_tasks: Vec<String>,
    formatted: Value,
    /// Whether the reference saw the nested pairs as tuples. JSON cannot record
    /// this, and the reference's dispatcher branches on it, so the fixture
    /// states it and the port is told rather than guessing.
    tuple_pairs: bool,
}

fn cases() -> Vec<Case> {
    fixture()["cases"]
        .as_array()
        .expect("the fixture holds a list of cases")
        .iter()
        .map(|case| Case {
            name: case["name"]
                .as_str()
                .expect("every case is named")
                .to_string(),
            results: case["results"].clone(),
            include_confidence: case["options"]
                .get("include_confidence")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            requested_relations: string_list(case, "requested_relations"),
            classification_tasks: string_list(case, "classification_tasks"),
            formatted: case["formatted"].clone(),
            tuple_pairs: case["tuple_pairs"].as_bool().unwrap_or(true),
        })
        .collect()
}

fn string_list(case: &Value, key: &str) -> Vec<String> {
    case["options"][key]
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|item| item.as_str().expect("names are strings").to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// Rebuild the raw results dict. The fixture stores 4-element spans as arrays,
/// which is what the reference's `tuple` branch destructures.
fn raw_results(results: &Value) -> serde_json::Map<String, Value> {
    match results {
        Value::Object(object) => object.clone(),
        other => panic!("results is an object, got {other}"),
    }
}

#[test]
fn format_results_matches_the_reference_on_every_case() {
    let cases = cases();
    assert_eq!(cases.len(), 21, "the fixture grew or lost a case");
    for case in &cases {
        let got = format_results_as(
            &raw_results(&case.results),
            case.include_confidence,
            &case.requested_relations,
            &case.classification_tasks,
            case.tuple_pairs,
        );
        assert_eq!(&got, &case.formatted, "case `{}` diverged", case.name);
    }
}

/// The dispatch is by type sniffing, so the arguments cannot rescue a value the
/// sniff rejects — and cannot reject one it accepts. Pinned separately because
/// it is the property the whole dispatcher rests on.
#[test]
fn relation_sniff_ignores_the_requested_relations_argument() {
    let cases = cases();
    let by_name: BTreeMap<&str, &Case> = cases
        .iter()
        .map(|case| (case.name.as_str(), case))
        .collect();

    let sniffed = by_name["two_tuple_list_is_a_relation_whatever_the_arguments"];
    assert!(
        sniffed.requested_relations.is_empty(),
        "this case exists precisely because no relation was requested",
    );
    assert!(
        sniffed.formatted.get("relation_extraction").is_some(),
        "a 2-element array sniffs as a relation on its own, so it ships anyway",
    );

    // A pair of *arrays* fails the same sniff and is then stored verbatim at
    // top level, unformatted — it does not become a classification either.
    let real_lists = by_name["list_of_real_lists_is_not_sniffed_and_not_formatted"];
    assert!(
        real_lists.formatted.get("relation_extraction").is_none(),
        "an array of arrays is not a 2-element array, so it is not a relation",
    );
    assert_eq!(
        real_lists.formatted["sentiment"],
        serde_json::json!([["positive", 0.9], ["negative", 0.8]]),
        "with no branch left to match, the value passes through untouched",
    );
}

/// `include_confidence` only affects the branches that *build* a value, so a
/// span sequence and a span dict of the same span serialize differently.
#[test]
fn tuple_spans_lose_offsets_while_dict_spans_keep_them() {
    let cases = cases();
    let by_name: BTreeMap<&str, &Case> = cases
        .iter()
        .map(|case| (case.name.as_str(), case))
        .collect();

    let tuple = &by_name["tuple_spans_lose_their_offsets"].formatted["entities"]["person"][0];
    assert_eq!(tuple["text"], "Marie");
    assert_eq!(tuple["confidence"], 0.9);
    assert!(
        tuple.get("start").is_none() && tuple.get("end").is_none(),
        "the tuple branch constructs the object, so the offsets are gone: {tuple}",
    );

    let dict = &by_name["dict_spans_with_distinct_offsets_are_both_kept"].formatted["entities"]
        ["person"][0];
    assert_eq!(dict["text"], "Marie");
    assert_eq!(dict["start"], 0);
    assert_eq!(dict["end"], 5);
}

/// A name that is both a classification task and a requested relation lands in
/// the payload twice, in two different shapes. `strip_span_metadata` cannot
/// reconcile this afterwards, so it has to be right here.
#[test]
fn a_classification_that_is_also_a_requested_relation_appears_twice() {
    let cases = cases();
    let case = cases
        .iter()
        .find(|case| case.name == "classification_and_requested_relation_appear_in_both_places")
        .expect("the fixture pins this collision");
    assert_eq!(case.formatted["mentions"][0]["label"], "Marie");
    assert_eq!(
        case.formatted["relation_extraction"]["mentions"],
        Value::Array(Vec::new()),
        "the classification branch consumed the value, so relations only got \
         the empty entry the requested_relations loop added",
    );
}

/// Only the first entity-type map is formatted; a second type in the same
/// `entities` list is discarded without a word.
#[test]
fn only_the_first_entity_type_map_is_formatted() {
    let cases = cases();
    let formatted = &cases
        .iter()
        .find(|case| case.name == "tuple_spans_lose_their_offsets")
        .expect("the fixture pins this")
        .formatted;
    assert_eq!(
        formatted["entities"].as_object().map(|o| o.len()),
        Some(1),
        "the input carried a `person` and an `org`, but format_results passes \
         value[0] so `org` never reaches the formatter: {formatted}",
    );
    assert!(formatted["entities"].get("org").is_none());
}

/// Both struct formatters turn a falsy scalar field into null. The two branches
/// read differently in the reference (`spans or None` versus an explicit
/// `None`), so it is tempting to assume they disagree — the oracle says they do
/// not, and an empty string is null through `format_entity_dict` too.
#[test]
fn a_falsy_field_is_null_in_both_struct_formatters() {
    let cases = cases();
    let by_name: BTreeMap<&str, &Case> = cases
        .iter()
        .map(|case| (case.name.as_str(), case))
        .collect();

    let as_struct = &by_name["falsy_struct_fields_become_none"].formatted["record"];
    assert!(
        as_struct["author"].is_null() && as_struct["role"].is_null(),
        "an empty string is falsy, so format_struct nulls it: {as_struct}",
    );

    let as_entity = &by_name["falsy_entity_fields_become_none_too"].formatted["entities"];
    assert!(
        as_entity["author"].is_null() && as_entity["role"].is_null(),
        "`spans or None` in format_entity_dict reaches the same null: {as_entity}",
    );
}

/// The unit-level entry points, exercised directly on the shapes the dispatcher
/// routes to them. `format_results` above covers the same code through the
/// public door; these pin the sub-formatters' own contract, including the
/// empty-input cases the fixture reaches only indirectly.
#[test]
fn format_classification_handles_every_input_shape() {
    let pair = serde_json::json!(["positive", 0.9]);
    assert_eq!(
        format_classification(&pair, true),
        serde_json::json!({"label": "positive", "confidence": 0.9}),
    );
    assert_eq!(
        format_classification(&pair, false),
        Value::String("positive".into()),
    );

    // A list of pairs.
    let pairs = serde_json::json!([["a", 0.9], ["b", 0.8]]);
    assert_eq!(
        format_classification(&pairs, false),
        serde_json::json!(["a", "b"]),
    );

    // A single-element list is still a list of pairs: there is no "too short"
    // special case, and the earlier fixture case that claimed otherwise was
    // renamed to match.
    let one = serde_json::json!([["a", 0.9]]);
    assert_eq!(
        format_classification(&one, true),
        serde_json::json!([{"label": "a", "confidence": 0.9}]),
    );

    // A 2-element array whose second item is a bool is not a (label, score)
    // pair — bool is a subclass of int in Python, hence `_is_score`'s exclusion.
    let boolish = serde_json::json!(["yes", true]);
    assert_eq!(format_classification(&boolish, true), boolish);

    // A 2-element array of strings is a list, not a pair.
    let strings = serde_json::json!(["ml", "safety"]);
    assert_eq!(format_classification(&strings, true), strings);

    // Empty and non-array inputs pass through.
    let empty = serde_json::json!([]);
    assert_eq!(format_classification(&empty, true), empty);
    assert_eq!(
        format_classification(&Value::String("x".into()), true),
        Value::String("x".into()),
    );
}

#[test]
fn dedup_keeps_the_first_occurrence_not_the_highest_score() {
    // The value is appended, never replaced, so when two spans collapse the
    // earlier confidence survives even though it is the lower one.
    let entities = serde_json::json!({
        "person": [["Marie", 0.4, 0, 5], ["marie", 0.9, 0, 5]]
    });
    let got = format_entity_dict(entities.as_object().unwrap(), true);
    assert_eq!(got["person"].as_array().map(Vec::len), Some(1));
    assert_eq!(got["person"][0]["confidence"], 0.4);
}

#[test]
fn an_empty_text_span_is_dropped_rather_than_formatted() {
    let entities = serde_json::json!({"person": [["", 0.9, 0, 0], ["Marie", 0.5, 3, 8]]});
    let got = format_entity_dict(entities.as_object().unwrap(), true);
    assert_eq!(got["person"].as_array().map(Vec::len), Some(1));
    assert_eq!(got["person"][0]["text"], "Marie");
}

#[test]
fn an_offset_less_dict_dedupes_by_text_alone() {
    // Without both offsets the key is (text.lower(), None, None), so two
    // dicts of the same text collapse even though they are different spans.
    // This is the contrast the fixture's surviving dict case relies on.
    let entities = serde_json::json!({
        "person": [
            {"text": "Marie", "confidence": 0.9, "start": 0, "end": 5},
            {"text": "Marie", "confidence": 0.4, "start": 20, "end": 25},
        ]
    });
    let got = format_entity_dict(entities.as_object().unwrap(), true);
    assert_eq!(
        got["person"].as_array().map(Vec::len),
        Some(2),
        "distinct offsets keep them apart: {got}",
    );

    let bare = serde_json::json!({
        "person": [{"text": "Marie", "confidence": 0.9}, {"text": "Marie", "confidence": 0.4}]
    });
    let got = format_entity_dict(bare.as_object().unwrap(), true);
    assert_eq!(
        got["person"].as_array().map(Vec::len),
        Some(1),
        "with no offsets the key is (text, None, None), so they collapse: {got}",
    );
}

#[test]
fn a_scalar_span_sequence_collapses_to_one_labelled_value() {
    let structure = serde_json::json!({"role": ["engineer", 0.9, 4, 12]});
    let got = format_struct(structure.as_object().unwrap(), true);
    assert_eq!(
        got["role"],
        serde_json::json!({"text": "engineer", "confidence": 0.9})
    );
    assert!(
        got["role"].get("start").is_none(),
        "a 4-tuple in scalar position takes the same lossy branch as a list one",
    );

    let without = format_struct(structure.as_object().unwrap(), false);
    assert_eq!(without["role"], Value::String("engineer".into()));

    // An empty surface collapses to the empty string rather than a null, so the
    // `include_confidence and text` guard shows up as a bare string.
    let empty_surface = serde_json::json!({"role": ["", 0.9, 0, 0]});
    let got = format_struct(empty_surface.as_object().unwrap(), true);
    assert_eq!(got["role"], Value::String(String::new()));
}
