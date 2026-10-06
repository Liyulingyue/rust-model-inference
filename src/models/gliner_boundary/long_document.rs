//! Long-document chunking and chunk-result merging.
//!
//! `gliner2/inference/chunking.py` is model-agnostic: it splits a document into
//! overlapping word windows, shifts each chunk's local span offsets back to
//! document offsets, merges the per-chunk predictions, and finally strips the
//! span metadata the caller did not ask for. Every function here is pure, so the
//! tests never need a model.
//!
//! The merge is where the behaviour is type-dependent — a classification dict
//! takes the max confidence, a bare string takes a majority vote, lists
//! concatenate and then dedupe — and each branch is separate code in the
//! reference. What this port mirrors is those branches, in the reference's order.

use std::collections::BTreeMap;

use super::overlap::{resolve_overlaps, OverlapPolicy, ScoredSpan};
use crate::models::gliner::prompt::{word_spans, WordSplitter};

/// One window of the document, with offsets back into the original text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextChunk {
    pub text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub start_word: usize,
    pub end_word: usize,
}

/// `split_text_into_chunks` (`chunking.py:43`): overlapping **word** windows.
///
/// Three rules a plausible port gets wrong:
///
/// * `start_char` / `end_char` come from the word tokens, so `text` is a *slice of
///   the original* and keeps the document's casing. `iter_word_offsets` asks the
///   splitter for `lower=False`; the lower-casing happens later, in the model.
/// * The step is `chunk_size - chunk_overlap`, and the loop **breaks once
///   `end_word` reaches the end** rather than stepping again — otherwise the
///   final chunk is duplicated by a trailing empty window.
/// * A document with no words still yields **one** chunk spanning the whole text.
///   Zero chunks would make `merge_chunk_results`' length check fire for every
///   empty document.
pub fn split_text_into_chunks(
    text: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    splitter: WordSplitter,
) -> Result<Vec<TextChunk>, String> {
    if chunk_size == 0 {
        return Err("chunk_size must be greater than 0".into());
    }
    if chunk_overlap >= chunk_size {
        return Err("chunk_overlap must be smaller than chunk_size".into());
    }
    // Character offsets, so a chunk's `text` is a slice of the original and the
    // document's casing survives chunking.
    let spans = char_spans(text, splitter);
    if spans.is_empty() {
        return Ok(vec![TextChunk {
            text: text.to_string(),
            start_char: 0,
            end_char: text.chars().count(),
            start_word: 0,
            end_word: 0,
        }]);
    }
    let mut chunks = Vec::new();
    let step = chunk_size - chunk_overlap;
    let mut start_word = 0usize;
    while start_word < spans.len() {
        let end_word = (start_word + chunk_size).min(spans.len());
        let start_char = spans[start_word].0;
        let end_char = spans[end_word - 1].1;
        chunks.push(TextChunk {
            text: slice_chars(text, start_char, end_char),
            start_char,
            end_char,
            start_word,
            end_word,
        });
        if end_word == spans.len() {
            break;
        }
        start_word += step;
    }
    Ok(chunks)
}

/// `(start, end)` code point spans of each word, for slicing the original.
fn char_spans(text: &str, splitter: WordSplitter) -> Vec<(usize, usize)> {
    // `word_spans` reports code point offsets with the token lower-cased when
    // asked; here only the offsets matter, so the casing is irrelevant.
    word_spans(text, splitter, false)
        .into_iter()
        .map(|span| (span.start, span.end))
        .collect()
}

/// `text[start..end]` in **code points**, mirroring Python's `str` slicing.
fn slice_chars(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

/// `_is_span_dict` (`chunking.py:377`): a dict with text and integer offsets.
fn is_span_dict(value: &serde_json::Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.contains_key("text")
        && object.contains_key("start")
        && object.contains_key("end")
        && value["start"].is_i64()
        && value["end"].is_i64()
}

/// `_is_classification_dict` (`chunking.py:385`).
fn is_classification_dict(value: &serde_json::Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.contains_key("label") && object.contains_key("confidence")
}

/// A prediction under one key, in the shape the merge walks.
pub type ChunkResult = serde_json::Value;

/// `merge_chunk_results` (`chunking.py:110`): merge one document's chunk results.
///
/// `scalar_entity_labels` names entity types declared non-list; those collapse to
/// a single best value. `policy` is the overlap policy, defaulting to
/// `disallow` here — note this is a *different* default from the `allow`
/// `_dedupe_items` passes to the span resolver, and the reference keeps both.
pub fn merge_chunk_results(
    original_text: &str,
    chunks: &[TextChunk],
    chunk_results: &[ChunkResult],
    include_confidence: bool,
    include_spans: bool,
    scalar_entity_labels: &[String],
    policy: OverlapPolicy,
) -> Result<serde_json::Value, String> {
    if chunks.len() != chunk_results.len() {
        return Err("chunks and chunk_results must have the same length".into());
    }
    // Each chunk's local offsets shift by its start before merging, so a span
    // found in the second window is reported against the document.
    let remapped: Vec<ChunkResult> = chunk_results
        .iter()
        .zip(chunks)
        .map(|(result, chunk)| remap_result_spans(result, original_text, chunk))
        .collect();
    let merged = merge_result_dicts(&remapped, scalar_entity_labels, policy);
    Ok(strip_span_metadata(
        &merged,
        include_confidence,
        include_spans,
    ))
}

/// `remap_result_spans` (`chunking.py:89`): shift every span dict's offsets by the
/// chunk's `start_char`, then re-derive its `text` from the document.
fn remap_result_spans(value: &ChunkResult, original_text: &str, chunk: &TextChunk) -> ChunkResult {
    match value {
        serde_json::Value::Array(items) => serde_json::Value::Array(
            items
                .iter()
                .map(|item| remap_result_spans(item, original_text, chunk))
                .collect(),
        ),
        serde_json::Value::Object(object) => {
            let mut remapped = serde_json::Map::new();
            for (key, item) in object {
                remapped.insert(key.clone(), remap_result_spans(item, original_text, chunk));
            }
            let mut value = serde_json::Value::Object(remapped);
            if is_span_dict(&value) {
                let start = value["start"].as_i64().unwrap_or(0) as usize + chunk.start_char;
                let end = value["end"].as_i64().unwrap_or(0) as usize + chunk.start_char;
                if let Some(object) = value.as_object_mut() {
                    object.insert("start".into(), serde_json::json!(start));
                    object.insert("end".into(), serde_json::json!(end));
                    if let Some(surface) = char_slice(original_text, start, end) {
                        object.insert("text".into(), serde_json::json!(surface));
                    }
                }
            }
            value
        }
        other => other.clone(),
    }
}

/// `original_text[start..end]` in code points, or `None` when out of range.
fn char_slice(text: &str, start: usize, end: usize) -> Option<String> {
    if end > text.chars().count() {
        return None;
    }
    Some(slice_chars(text, start, end.max(start)))
}

/// `_merge_result_dicts` (`chunking.py:134`).
fn merge_result_dicts(
    results: &[ChunkResult],
    scalar_entity_labels: &[String],
    policy: OverlapPolicy,
) -> ChunkResult {
    let mut keys: Vec<&String> = Vec::new();
    let mut seen: std::collections::BTreeSet<&String> = std::collections::BTreeSet::new();
    for result in results {
        if let Some(object) = result.as_object() {
            for key in object.keys() {
                if seen.insert(key) {
                    keys.push(key);
                }
            }
        }
    }
    let scalar: std::collections::BTreeSet<&str> =
        scalar_entity_labels.iter().map(|s| s.as_str()).collect();
    let mut merged = serde_json::Map::new();
    for key in keys {
        let values: Vec<ChunkResult> = results
            .iter()
            .filter_map(|result| result.get(key))
            .cloned()
            .collect();
        let value = if key == "entities" {
            merge_entity_maps(&values, &scalar, policy)
        } else {
            merge_values(&values, policy)
        };
        merged.insert(key.clone(), value);
    }
    serde_json::Value::Object(merged)
}

/// `_merge_entity_maps` (`chunking.py:163`).
fn merge_entity_maps(
    values: &[ChunkResult],
    scalar_labels: &std::collections::BTreeSet<&str>,
    policy: OverlapPolicy,
) -> ChunkResult {
    let mut labels: Vec<&String> = Vec::new();
    let mut seen: std::collections::BTreeSet<&String> = std::collections::BTreeSet::new();
    for value in values {
        if let Some(object) = value.as_object() {
            for label in object.keys() {
                if seen.insert(label) {
                    labels.push(label);
                }
            }
        }
    }
    let mut merged = serde_json::Map::new();
    for label in labels {
        let mut items: Vec<ChunkResult> = Vec::new();
        for value in values {
            if let Some(field) = value.get(label) {
                items.extend(as_list(field));
            }
        }
        let deduped = dedupe_items(&items, policy);
        if scalar_labels.contains(label.as_str()) {
            merged.insert(
                label.clone(),
                deduped
                    .into_iter()
                    .next()
                    .unwrap_or(serde_json::Value::Null),
            );
        } else {
            merged.insert(label.clone(), serde_json::Value::Array(deduped));
        }
    }
    serde_json::Value::Object(merged)
}

/// `_merge_values` (`chunking.py:211`): the type-dependent core.
fn merge_values(values: &[ChunkResult], policy: OverlapPolicy) -> ChunkResult {
    // `non_empty = [v for v in values if v not in (None, {}, [])]` — excludes only
    // null, an empty object, and an empty list. A *non-empty* object is kept, which
    // is what lets the classification-dict branch below ever fire.
    let non_empty: Vec<&ChunkResult> = values
        .iter()
        .filter(|value| {
            !value.is_null()
                && !value.as_array().is_some_and(|items| items.is_empty())
                && !value.as_object().is_some_and(|object| object.is_empty())
        })
        .collect();
    if non_empty.is_empty() {
        return values.first().cloned().unwrap_or(serde_json::Value::Null);
    }
    // A classification dict merges to the max confidence — not the most common.
    if non_empty.iter().all(|value| is_classification_dict(value)) {
        let best = non_empty
            .iter()
            .max_by(|a, b| {
                confidence_of(a)
                    .partial_cmp(&confidence_of(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty");
        return (*best).clone();
    }
    // Bare strings merge by majority, ties to the *earliest* occurrence.
    if non_empty.iter().all(|value| value.is_string()) {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for value in &non_empty {
            if let Some(text) = value.as_str() {
                *counts.entry(text).or_insert(0) += 1;
            }
        }
        let mut best: Option<(&str, usize, usize)> = None;
        for (position, value) in non_empty.iter().enumerate() {
            let text = value.as_str().expect("string");
            let count = counts[text];
            let candidate = (text, count, position);
            // More votes wins; on a tie the earlier chunk wins, which is why the
            // key is (count, Reverse(position)) rather than just count.
            best = match best {
                // More votes wins; on a tie the earlier chunk wins, so the key is
                // (count, Reverse(position)) and `>=` keeps the incumbent.
                Some(current)
                    if (current.1, std::cmp::Reverse(current.2))
                        >= (candidate.1, std::cmp::Reverse(candidate.2)) =>
                {
                    Some(current)
                }
                _ => Some(candidate),
            };
        }
        let text = best.map(|(text, _, _)| text).unwrap_or_default();
        return serde_json::Value::String(text.to_string());
    }
    if non_empty.iter().all(|value| value.is_array()) {
        let mut items: Vec<ChunkResult> = Vec::new();
        for value in &non_empty {
            items.extend(value.as_array().expect("array").clone());
        }
        return serde_json::Value::Array(dedupe_items(&items, policy));
    }
    if non_empty.iter().all(|value| value.is_object()) {
        return merge_nested_dicts(&non_empty, policy);
    }
    (*non_empty[0]).clone()
}

/// `_merge_nested_dicts` (`chunking.py:239`).
fn merge_nested_dicts(values: &[&ChunkResult], policy: OverlapPolicy) -> ChunkResult {
    let mut keys: Vec<&String> = Vec::new();
    let mut seen: std::collections::BTreeSet<&String> = std::collections::BTreeSet::new();
    for value in values {
        if let Some(object) = value.as_object() {
            for key in object.keys() {
                if seen.insert(key) {
                    keys.push(key);
                }
            }
        }
    }
    let mut merged = serde_json::Map::new();
    for key in keys {
        let inner: Vec<ChunkResult> = values
            .iter()
            .filter_map(|value| value.get(key))
            .cloned()
            .collect();
        merged.insert(key.clone(), merge_values(&inner, policy));
    }
    serde_json::Value::Object(merged)
}

/// `_dedupe_items` (`chunking.py:267`): span items go through the overlap
/// resolver and are re-sorted by `(start, end, text)`; non-span items collapse on
/// a **confidence-insensitive** canonical key, keeping the higher-confidence one.
fn dedupe_items(items: &[ChunkResult], policy: OverlapPolicy) -> Vec<ChunkResult> {
    let (span_items, other_items): (Vec<_>, Vec<_>) =
        items.iter().partition(|item| is_span_dict(item));
    let mut deduped: Vec<ChunkResult> = Vec::new();
    if !span_items.is_empty() {
        let scored: Vec<ScoredSpan> = span_items
            .iter()
            .map(|item| ScoredSpan {
                score: confidence_of(item),
                start: item["start"].as_i64().unwrap_or(0) as usize,
                end: item["end"].as_i64().unwrap_or(0) as usize,
            })
            .collect();
        let mut selected: Vec<ChunkResult> = resolve_overlaps(&scored, policy)
            .into_iter()
            .map(|index| span_items[index].clone())
            .collect();
        selected.sort_by_key(|item| {
            (
                item["start"].as_i64().unwrap_or(0),
                item["end"].as_i64().unwrap_or(0),
                item["text"].as_str().unwrap_or_default().to_string(),
            )
        });
        deduped.extend(selected);
    }
    // Non-span items: same prediction in two overlapping chunks, slightly
    // different scores, collapses to the higher one.
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    let mut other_deduped: Vec<ChunkResult> = Vec::new();
    for item in other_items {
        let key = canonical_key(item);
        match seen.get(&key) {
            Some(&index) => {
                if representative_confidence(item)
                    > representative_confidence(&other_deduped[index])
                {
                    other_deduped[index] = (*item).clone();
                }
            }
            None => {
                seen.insert(key, other_deduped.len());
                other_deduped.push((*item).clone());
            }
        }
    }
    deduped.extend(other_deduped);
    deduped
}

/// `_canonical_key` (`chunking.py:388`): a confidence-**insensitive** identity, so
/// the same prediction with two scores is one prediction.
fn canonical_key(value: &ChunkResult) -> String {
    match value {
        serde_json::Value::Object(object) => {
            let mut pairs: Vec<String> = object
                .iter()
                .filter(|(key, _)| key.as_str() != "confidence")
                .map(|(key, item)| format!("{key}: {}", canonical_key(item)))
                .collect();
            pairs.sort();
            format!("{{{}}}", pairs.join(","))
        }
        serde_json::Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(canonical_key).collect();
            format!("[{}]", inner.join(","))
        }
        other => other.to_string(),
    }
}

/// `_representative_confidence` (`chunking.py:401`): the best confidence anywhere
/// in a (possibly nested) prediction.
fn representative_confidence(value: &ChunkResult) -> f32 {
    match value {
        serde_json::Value::Object(object) => {
            if let Some(confidence) = object.get("confidence").and_then(|c| c.as_f64()) {
                return confidence as f32;
            }
            object
                .values()
                .map(representative_confidence)
                .fold(f32::NEG_INFINITY, f32::max)
                .max(0.0)
        }
        serde_json::Value::Array(items) => items
            .iter()
            .map(representative_confidence)
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0),
        _ => 0.0,
    }
}

fn confidence_of(value: &ChunkResult) -> f32 {
    value
        .get("confidence")
        .and_then(|c| c.as_f64())
        .unwrap_or(0.0) as f32
}

/// `_as_list` (`chunking.py:372`).
fn as_list(value: &ChunkResult) -> Vec<ChunkResult> {
    match value {
        serde_json::Value::Null => Vec::new(),
        serde_json::Value::Array(items) => items.clone(),
        other => vec![other.clone()],
    }
}

/// `_strip_span_metadata` (`chunking.py:316`): re-shape each span for the flags the
/// caller passed. A span with **neither** flag collapses to its bare `text`; a
/// choice/enum field (`text` + `confidence`, no offsets) collapses to the bare
/// string when confidence is off — a shape that does not exist off this path.
fn strip_span_metadata(
    value: &ChunkResult,
    include_confidence: bool,
    include_spans: bool,
) -> ChunkResult {
    match value {
        serde_json::Value::Array(items) => serde_json::Value::Array(
            items
                .iter()
                .map(|item| strip_span_metadata(item, include_confidence, include_spans))
                .collect(),
        ),
        serde_json::Value::Object(object) => {
            let value = serde_json::Value::Object(object.clone());
            if is_span_dict(&value) {
                // Attribute payloads (extra keys on a span) survive regardless of
                // the flags, matching the non-long attribute API.
                let extras: serde_json::Map<String, serde_json::Value> = object
                    .iter()
                    .filter(|(key, _)| {
                        !matches!(key.as_str(), "text" | "confidence" | "start" | "end")
                    })
                    .map(|(key, item)| (key.clone(), item.clone()))
                    .collect();
                if !include_confidence && !include_spans && extras.is_empty() {
                    return serde_json::Value::String(
                        object
                            .get("text")
                            .and_then(|t| t.as_str())
                            .unwrap_or_default()
                            .to_string(),
                    );
                }
                let mut stripped = serde_json::Map::new();
                stripped.insert(
                    "text".into(),
                    serde_json::json!(object
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or_default()),
                );
                if include_confidence && object.contains_key("confidence") {
                    stripped.insert("confidence".into(), value["confidence"].clone());
                }
                if include_spans {
                    stripped.insert("start".into(), value["start"].clone());
                    stripped.insert("end".into(), value["end"].clone());
                }
                for (key, item) in extras {
                    stripped.insert(key, item);
                }
                return serde_json::Value::Object(stripped);
            }
            if is_classification_dict(&value) {
                if include_confidence {
                    return serde_json::json!({
                        "label": value["label"], "confidence": value["confidence"]
                    });
                }
                return value["label"].clone();
            }
            // An enum/choice field: text + confidence, no offsets.
            if object.contains_key("text")
                && object.contains_key("confidence")
                && !object.contains_key("start")
                && !object.contains_key("end")
            {
                if include_confidence {
                    return serde_json::json!({"text": value["text"], "confidence": value["confidence"]});
                }
                return value["text"].clone();
            }
            let mut out = serde_json::Map::new();
            for (key, item) in object {
                out.insert(
                    key.clone(),
                    strip_span_metadata(item, include_confidence, include_spans),
                );
            }
            serde_json::Value::Object(out)
        }
        other => other.clone(),
    }
}
