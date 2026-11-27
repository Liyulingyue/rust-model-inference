//! The public payload formatters — `runtime.py:43-211`.
//!
//! These are the last thing every extract call does, and they are what makes
//! `merge_chunk_results` possible: the merge walks *formatted* JSON, while the
//! decode stage produces a typed [`Extraction`](super::extract::Extraction).
//! Nothing upstream can produce the reference's public shape without them.
//!
//! The dispatch is **type sniffing**, not declaration. A key is a relation if
//! its first element is a 2-element sequence or a dict carrying `head`/`tail`,
//! and once that fires `requested_relations` is never consulted again. Because
//! JSON has no tuple, the equivalent test here is "a 2-element array" — which
//! is why a 2-element array of arrays is *not* a relation (its first element is
//! an array, not a string or a number).
//!
//! `include_confidence` is not a uniform toggle. It only affects the branches
//! that *build* a value: the span-sequence branch constructs
//! `{"text", "confidence"}` and **drops the offsets**, while the dict branch
//! passes spans through untouched and so keeps `start`/`end`. The same logical
//! span therefore serializes differently depending on which branch saw it, and
//! `strip_span_metadata` cannot recover the offsets afterwards.

use std::collections::HashSet;

use serde_json::{json, Map, Value};

/// `_is_score` (`runtime.py:44`): a confidence scalar, excluding bools.
///
/// `serde_json`'s `Value::Bool` is not a number, so the only way a bool reaches
/// this is if a caller built the raw payload by hand. The reference's
/// `isinstance(value, bool)` exclusion exists because `bool` subclasses `int`
/// in Python, and a bool that slipped into a score slot would otherwise read as
/// a confidence. Preserved here so a hand-built payload behaves the same.
fn is_score(value: &Value) -> bool {
    value.is_number()
}

/// `_format_classification` (`runtime.py:49`).
///
/// A single- or multi-label classification, including JSON arrays. Returns a
/// label, a `{label, confidence}` object, or a list of those.
///
/// A 2-element array is a `(label, score)` pair only if its first element is a
/// string *and* its second is a real score; anything else is a list that
/// survives untouched. Note the asymmetry with the relation sniff, which accepts
/// a 2-element array whose first element is a *number* — so `[1, 2]` is a
/// relation while `["a", true]` is neither.
pub fn format_classification(value: &Value, include_confidence: bool) -> Value {
    let Some(items) = value.as_array() else {
        return value.clone();
    };
    if items.is_empty() {
        return value.clone();
    }
    // A list of pairs.
    if items[0].is_array() {
        if include_confidence {
            return Value::Array(
                items
                    .iter()
                    .map(|item| labeled(item, include_confidence))
                    .collect(),
            );
        }
        return Value::Array(items.iter().map(|item| item[0].clone()).collect());
    }
    // A pair, but only if the score really is a score.
    if items.len() == 2 && items[0].is_string() && is_score(&items[1]) {
        return labeled(value, include_confidence);
    }
    value.clone()
}

/// One `(label, score)` pair as `{label, confidence}`, or the bare label.
fn labeled(value: &Value, include_confidence: bool) -> Value {
    if include_confidence {
        json!({"label": value[0], "confidence": value[1]})
    } else {
        value[0].clone()
    }
}

/// `format_entity_dict` (`runtime.py:134`): deduplicate and optionally keep
/// confidence on an entity-type map.
///
/// The dedup key differs per branch, which is load-bearing. A span sequence
/// keys on `(lower, start, end)`; a span dict keys on `(lower, start, end)` when
/// it carries both offsets and `(lower, None, None)` when it does not, so two
/// offset-less dicts of the same text collapse while two with distinct offsets
/// do not. A bare string keys on the lowercased text alone.
///
/// The survivor is always the **first** occurrence — values are appended, never
/// replaced — so when two collapse the earlier confidence wins even if it is
/// lower.
pub fn format_entity_dict(entities: &Map<String, Value>, include_confidence: bool) -> Value {
    let mut formatted = Map::new();
    for (name, spans) in entities {
        formatted.insert(name.clone(), dedupe_value(spans, include_confidence));
    }
    Value::Object(formatted)
}

/// `format_struct` (`runtime.py:172`): deduplicate a structure instance.
///
/// The same rules as [`format_entity_dict`], applied to a structure's fields.
pub fn format_struct(structure: &Map<String, Value>, include_confidence: bool) -> Value {
    let mut formatted = Map::new();
    for (field, value) in structure {
        formatted.insert(field.clone(), dedupe_value(value, include_confidence));
    }
    Value::Object(formatted)
}

/// The shared body of both struct formatters.
///
/// A list of predictions is deduplicated per branch (below); a scalar
/// `(text, conf, start, end)` collapses to one labelled value; anything else is
/// stored as-is unless it is falsy, which becomes an explicit `null`.
///
/// The falsy rule is the same in both formatters — `spans or None` and
/// `None` respectively — despite the two branches being written differently.
/// They differ in the *list* branch instead: `format_entity_dict` dedupes a
/// list of spans, `format_struct` a list of field values, but a value that is
/// neither a list nor a 4-tuple takes the same path in both.
fn dedupe_value(value: &Value, include_confidence: bool) -> Value {
    // A bare 4-element array of scalars is the reference's scalar *tuple*, not
    // a list of predictions. Python distinguishes them by type — `isinstance
    // (value, list)` is False for a tuple — and JSON has no such distinction, so
    // the shape has to stand in: a list of predictions is an array whose
    // elements are themselves sequences or objects, and a span tuple is an array
    // of four scalars. Getting this backwards sends every scalar span down the
    // list branch, where each of its four elements is judged as its own
    // prediction.
    if is_span_tuple(value) {
        let items = value.as_array().unwrap();
        let text = items[0].as_str().unwrap_or_default();
        return if include_confidence && !text.is_empty() {
            json!({"text": text, "confidence": items[1]})
        } else {
            Value::String(text.to_string())
        };
    }
    if let Some(items) = value.as_array() {
        let mut unique = Vec::new();
        let mut seen: HashSet<Value> = HashSet::new();
        for item in items {
            if let Some(key) = span_sequence_key(item) {
                if item[0].as_str().is_some_and(|text| !text.is_empty()) && seen.insert(key) {
                    unique.push(built_span(item, include_confidence));
                }
            } else if item.is_object() {
                if let Some(key) = span_dict_key(item) {
                    if item["text"].as_str().is_some_and(|t| !t.is_empty()) && seen.insert(key) {
                        unique.push(item.clone());
                    }
                }
            } else if item.as_str().is_some_and(|text| !text.is_empty()) {
                let folded = json!([
                    item.as_str().unwrap().to_lowercase(),
                    Value::Null,
                    Value::Null
                ]);
                if seen.insert(folded) {
                    unique.push(item.clone());
                }
            }
        }
        return Value::Array(unique);
    }
    if is_truthy(value) {
        value.clone()
    } else {
        Value::Null
    }
}

/// A 4-element array of scalars: the reference's scalar span tuple.
fn is_span_tuple(value: &Value) -> bool {
    let Some(items) = value.as_array() else {
        return false;
    };
    items.len() == 4
        && items[0].is_string()
        && is_score(&items[1])
        && items[2].is_number()
        && items[3].is_number()
}

/// `span` 4-tuple branch: `key = (text.lower(), start, end)`.
fn span_sequence_key(item: &Value) -> Option<Value> {
    let items = item.as_array()?;
    if items.len() != 4 {
        return None;
    }
    Some(json!([
        items[0].as_str()?.to_lowercase(),
        items[2],
        items[3],
    ]))
}

/// `span` dict branch: `key = (text.lower(), start, end)` when both offsets are
/// present, `(text.lower(), None, None)` otherwise.
fn span_dict_key(item: &Value) -> Option<Value> {
    let text = item["text"].as_str()?.to_lowercase();
    if item.get("start").is_some() && item.get("end").is_some() {
        Some(json!([text, item["start"], item["end"]]))
    } else {
        Some(json!([text, Value::Null, Value::Null]))
    }
}

/// The tuple branch **constructs** `{text, confidence}` and drops the offsets.
/// This is the asymmetry that makes a tuple and a dict of the same span
/// serialize differently.
fn built_span(item: &Value, include_confidence: bool) -> Value {
    if include_confidence {
        json!({"text": item[0], "confidence": item[1]})
    } else {
        Value::String(item[0].as_str().unwrap_or_default().to_string())
    }
}

/// Python truthiness, for the `format_struct` falsy branch.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(object) => !object.is_empty(),
    }
}

/// `format_results` (`runtime.py:73`): format extraction results into the
/// public payload.
///
/// `results` is the **raw** dict, where a prediction is still a 4-element
/// sequence rather than a dict. `requested_relations` and
/// `classification_tasks` come from the schema, and both are order-sensitive:
/// a name in `classification_tasks` short-circuits the relation sniff, so a name
/// that is both a classification task and a requested relation appears twice —
/// formatted as a classification at top level *and* as an empty list inside
/// `relation_extraction`, because the classification branch consumed the value
/// and never populated `relations`.
pub fn format_results(
    results: &Map<String, Value>,
    include_confidence: bool,
    requested_relations: &[String],
    classification_tasks: &[String],
) -> Value {
    format_results_as(
        results,
        include_confidence,
        requested_relations,
        classification_tasks,
        true,
    )
}

/// [`format_results`], with the tuple/list distinction under the caller's
/// control.
///
/// The reference's dispatcher branches on `isinstance(value[0], tuple)`, and
/// JSON cannot record that: `["a", 0.9]` is the same document whether the
/// reference saw a tuple or a list, yet it routes to opposite branches — as a
/// relation in one case and as an unformatted pass-through in the other. Rather
/// than infer it from shape and be wrong half the time, the caller states it.
/// `format_results` assumes tuples, which is what the extractor produces: the
/// decode stages build Python-style `(text, score, start, end)` sequences.
pub fn format_results_as(
    results: &Map<String, Value>,
    include_confidence: bool,
    requested_relations: &[String],
    classification_tasks: &[String],
    nested_pairs_are_tuples: bool,
) -> Value {
    let mut formatted = Map::new();
    let mut relations: Vec<(String, Value)> = Vec::new();
    for (key, value) in results {
        let is_classification = classification_tasks.iter().any(|task| task == key);
        let is_relation = !is_classification && {
            requested_relations.iter().any(|name| name == key)
                || value
                    .as_array()
                    .filter(|items| !items.is_empty())
                    .is_some_and(|items| is_relation_item(&items[0], nested_pairs_are_tuples))
        };

        if is_classification {
            formatted.insert(
                key.clone(),
                format_classification(value, include_confidence),
            );
        } else if is_relation {
            relations.push((
                key.clone(),
                match value {
                    Value::Array(_) => value.clone(),
                    _ => Value::Array(Vec::new()),
                },
            ));
        } else if let Some(items) = value.as_array() {
            if items.is_empty() {
                // Empty is not absent: `entities` becomes an object, every
                // other key stays a list.
                formatted.insert(
                    key.clone(),
                    if key == "entities" {
                        Value::Object(Map::new())
                    } else {
                        value.clone()
                    },
                );
            } else if items[0].is_object() {
                formatted.insert(
                    key.clone(),
                    if key == "entities" {
                        // Only the FIRST entity-type map is formatted; a second
                        // one in the same list is discarded silently.
                        format_entity_dict(items[0].as_object().unwrap(), include_confidence)
                    } else {
                        Value::Array(
                            items
                                .iter()
                                .map(|item| {
                                    format_struct(item.as_object().unwrap(), include_confidence)
                                })
                                .collect(),
                        )
                    },
                );
            } else {
                formatted.insert(key.clone(), value.clone());
            }
        } else if value.is_object() {
            formatted.insert(
                key.clone(),
                format_struct(value.as_object().unwrap(), include_confidence),
            );
        } else {
            formatted.insert(key.clone(), value.clone());
        }
    }

    // Requested relations always appear, empty list when nothing was found, so
    // a caller can index every requested relation without a presence check.
    for name in requested_relations {
        if !relations.iter().any(|(key, _)| key == name) {
            relations.push((name.clone(), Value::Array(Vec::new())));
        }
    }
    // Sniffed relations join them here even when unrequested, so the whole set
    // nests under one key rather than staying at top level.
    if !relations.is_empty() {
        formatted.insert(
            "relation_extraction".to_string(),
            Value::Object(relations.into_iter().collect()),
        );
    }
    Value::Object(formatted)
}

/// The relation sniff: a 2-element *scalar pair*, or a dict with `head`/`tail`.
///
/// The reference tests `isinstance(value[0], tuple) and len(value[0]) == 2`, so
/// the pair must be a tuple of two leaves. JSON erases the tuple/list
/// distinction, so "two leaves" has to carry the meaning on its own: a
/// 2-element array *of scalars* is the pair, while a 2-element array *of arrays*
/// is a list of predictions and is not a relation. Accepting any 2-element
/// array here misclassifies `[[label, score], [label, score]]` as a relation and
/// moves it under `relation_extraction` unformatted.
fn is_relation_item(item: &Value, nested_pairs_are_tuples: bool) -> bool {
    match item {
        // The reference tests `isinstance(value[0], tuple) and len == 2`, so the
        // pair must be a tuple of two leaves. `nested_pairs_are_tuples` says
        // whether it is: when the caller says the nested pairs are tuples, any
        // 2-element array qualifies; when it says they are lists — the case the
        // fixture records for a pair of pairs — the sniff must fail, because
        // the reference saw a list there and stored the value unformatted.
        Value::Array(items) => {
            items.len() == 2
                && (nested_pairs_are_tuples
                    || !items
                        .iter()
                        .all(|item| !item.is_array() && !item.is_object()))
        }
        Value::Object(object) => object.contains_key("head") && object.contains_key("tail"),
        _ => false,
    }
}
