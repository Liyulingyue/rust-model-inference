//! `Extraction` -> the reference's raw results dict.
//!
//! [`format_results`](super::formatting::format_results) shapes a raw results
//! dict into the public payload, and the long-document merge walks a formatted
//! payload. The decode stages in this port produce a typed
//! [`Extraction`](super::extract::Extraction) instead, so this is the missing
//! middle: it turns one into the other without going through JSON text.
//!
//! Ground truth is `tools/oracle/gliner_boundary/dump_raw_results.py`, which
//! drives the real `batch_extract(format_results=False)` on the checkpoint.
//!
//! Three things this has to get right, each of which the fixture pins because
//! each is a plausible thing to get wrong:
//!
//! **Coordinates are in different units.** The port's spans and relations carry
//! **word** offsets into the normalized word list; the reference's raw dict
//! carries **character** offsets into the text, built as
//! `start_map[start], end_map[end - 1]` from the word splitter. The surface text
//! is then re-sliced from the original text and stripped, so a span whose word
//! range covers surrounding whitespace reports the trimmed text while keeping
//! the **untrimmed** offsets — the reference never adjusts them for the strip.
//!
//! **The key shape is per group type, and none of them is the obvious one.**
//! `entities` is a list holding one map; a structures group is keyed by its own
//! name; a relation type sits beside both with `{head, tail}` objects; a
//! classification is a bare `(label, score)` pair. Passing the wrong shape here
//! does not error — `format_results` sniffs types, so a mis-shaped value is
//! silently routed to a different branch.
//!
//! **A `choices` field has no location.** It is scored at the literal's own
//! prefix token rather than at a document span, so its dict carries `text` and
//! `confidence` and nothing else. Inventing `start`/`end` for it would put
//! offsets on a value that has none.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use super::extract::{ClassificationResult, ExtractedRecord, ExtractedSpan, Extraction};
use super::relations::ExtractedRelation;
use super::structure::{ChoiceValue, StructureField, StructureSpan};
use super::ExtractedStructure;

/// Per-word character spans, as the reference's `start_token_idx` /
/// `end_token_idx` carry them: index `i` is the half-open code point range of
/// word `i`.
///
/// Built with [`word_spans`](crate::models::gliner::prompt::word_spans) rather
/// than threaded through the prompt encoder, because the offsets describe the
/// text and not the tokenization. Code points, not bytes: the reference slices
/// Python strings, so `中华人民共和国`'s second word starts at code point 1.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WordCharSpans(pub Vec<(usize, usize)>);

impl WordCharSpans {
    /// The word map for `text`, built exactly as the prompt encoder builds the
    /// word list it indexes into.
    ///
    /// `build_with_child_marker_mixed` calls `split_words(&normalize_text(text))`,
    /// so the map has to be over the **normalized** text with the same splitter,
    /// or every offset is shifted — by the appended period at minimum, and by
    /// however much normalization changed the text. Constructing it here rather
    /// than taking a `&str` plus a splitter from each caller is what keeps that
    /// pairing from being gotten wrong at one call site.
    pub fn for_text(text: &str) -> Self {
        use crate::models::gliner::prompt::{normalize_text, word_spans, WordSplitter};
        WordCharSpans(
            word_spans(&normalize_text(text), WordSplitter::Whitespace, false)
                .into_iter()
                .map(|span| (span.start, span.end))
                .collect(),
        )
    }

    /// The character range covering words `[start, end)`, matching the
    /// reference's `start_map[start], end_map[end - 1]`.
    ///
    /// Returns `None` when either end is out of range, which the reference
    /// absorbs with an `IndexError` guard and skips the span — so an out-of-range
    /// span disappears rather than becoming a clamped one.
    pub fn char_range(&self, start: usize, end: usize) -> Option<(usize, usize)> {
        // An empty or reversed word range has no character range. Without this
        // guard a reversed pair indexes two *valid* words and produces a range
        // whose start is past its end — the slice then comes out empty and the
        // span is dropped by the surface check anyway, so the bug shows up as a
        // mysteriously absent span rather than as a panic.
        if end <= start {
            return None;
        }
        let (_, char_end) = self.0.get(end - 1)?;
        let (char_start, _) = self.0.get(start)?;
        Some((*char_start, *char_end))
    }

    /// How many words are covered.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Which entity labels are scalar, and in what order they are reported.
///
/// `dtype` lives in the schema's `entity_metadata`, keyed by **label**, and the
/// reference defaults it to `"list"`. A scalar label is reported as a single
/// object rather than a list — and as `null` when it found nothing — so this is
/// not a detail the caller can leave to the converter to infer.
#[derive(Clone, Debug, Default)]
pub struct EntityOrder {
    /// Label per entry, in the reference's report order.
    pub labels: Vec<String>,
    /// Labels declared `dtype: "str"`, which collapse to one value or `null`.
    pub scalar: Vec<String>,
}

impl EntityOrder {
    fn is_scalar(&self, label: &str) -> bool {
        self.scalar.iter().any(|name| name == label)
    }
}

/// The schema-level shape the converter cannot read off the [`Extraction`].
///
/// Everything here is metadata the reference threads in alongside the prompt
/// (`_build_schema_dicts_and_metadata`, `runtime.py:345`). It is passed rather
/// than re-parsed so that a caller who already resolved it — as the adapter
/// does, to build the prompt — does not resolve it twice.
#[derive(Clone, Debug, Default)]
pub struct RawShape {
    /// Entity labels and their dtypes.
    pub entities: EntityOrder,
    /// Classification task names, which `format_results` needs because a
    /// `(label, score)` pair is otherwise indistinguishable from a relation.
    pub classification_tasks: Vec<String>,
    /// Structure group names in report order. A group is keyed by its own name,
    /// not by `json_structures`, so the name has to survive to the output key.
    pub structure_groups: Vec<String>,
    /// Bare relation type names, passed to `format_results` as its
    /// `requested_relations`.
    ///
    /// This list does two things there: it lets a `{head, tail}` value be routed
    /// to `relation_extraction` (though the type sniff already does that), and it
    /// adds an **empty list** for every name that matched nothing. The second
    /// effect is why it holds the declared names rather than the ones that
    /// happened to match — reporting only what matched would silently drop the
    /// empty entries a caller indexing every requested relation relies on.
    pub relation_types: Vec<String>,
}

/// The two flags `batch_extract` threads through every shape.
///
/// The long-document path always passes both (`chunking.py` calls
/// `batch_extract` with `include_confidence=True, include_spans=True`), but the
/// single-document path does not, and a `choices` value has no `start`/`end` to
/// drop in the first place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeFlags {
    pub include_confidence: bool,
    pub include_spans: bool,
}

impl Default for ShapeFlags {
    fn default() -> Self {
        ShapeFlags {
            include_confidence: true,
            include_spans: true,
        }
    }
}

/// A span as the reference emits it: `text` always, then `confidence` and the
/// offsets in that order.
///
/// The order is not incidental for the relation path, which builds
/// `{text, start, end}` and then *assigns* `confidence`; matching it here keeps a
/// golden-file comparison from differing on key order alone.
fn span_value(
    text: &str,
    confidence: f32,
    char_start: usize,
    char_end: usize,
    flags: ShapeFlags,
) -> Value {
    // `_format_spans` has four branches and the last one returns the surface on
    // its own, so a span with neither flag is a bare **string** rather than an
    // object with one key. Three of the four are dicts, which makes the fourth
    // easy to miss — and a list of one-key objects is a shape no reference path
    // produces.
    if !flags.include_confidence && !flags.include_spans {
        return Value::String(text.to_string());
    }
    let mut object = Map::new();
    object.insert("text".to_string(), Value::String(text.to_string()));
    if flags.include_confidence {
        object.insert("confidence".to_string(), json_number(confidence));
    }
    if flags.include_spans {
        object.insert("start".to_string(), Value::from(char_start));
        object.insert("end".to_string(), Value::from(char_end));
    }
    Value::Object(object)
}

/// A `choices` value: `text` and `confidence`, and never offsets.
///
/// Separate from [`span_value`] on purpose. A choice is scored at its own prefix
/// token, so it has no document location, and a shared helper that always wrote
/// offsets would give every choice a location the reference does not have.
fn choice_value(choice: &ChoiceValue, flags: ShapeFlags) -> Value {
    let mut object = Map::new();
    object.insert("text".to_string(), Value::String(choice.text.clone()));
    if flags.include_confidence {
        object.insert("confidence".to_string(), json_number(choice.score));
    }
    Value::Object(object)
}

fn json_number(value: f32) -> Value {
    serde_json::Number::from_f64(value as f64)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// `text[char_start..char_end].strip()` in code points.
///
/// The reference slices the **original** text and strips, then reports the
/// *untrimmed* offsets. A span covering a trailing comma therefore reports the
/// comma-free text with offsets that still include the comma, and the two are
/// only reconcilable by the caller — matching it here means a port that trimmed
/// the offsets too would differ from the fixture.
fn surface(text: &str, char_start: usize, char_end: usize) -> String {
    slice_code_points(text, char_start, char_end)
        .trim()
        .to_string()
}

/// Python's `text[start:end]` over code points.
fn slice_code_points(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

/// Convert one [`Extraction`] into the reference's raw results dict.
///
/// `text` is the **normalized** text the spans were decoded against — the one
/// the word splitter saw, including the sentence-final `.` this port appends —
/// because that is what `char_range` indexes into and what the reference slices.
/// Passing the caller's original text instead puts every offset off by the
/// appended period and, for text needing normalization, by much more.
pub fn extraction_to_raw_results(
    extraction: &Extraction,
    spans: &WordCharSpans,
    text: &str,
    shape: &RawShape,
    flags: ShapeFlags,
) -> Map<String, Value> {
    let mut results = Map::new();

    // A classification group is a bare `(label, score)` pair for a single-label
    // group and a list of them for a multi-label one. Emitted first so the
    // reference's own ordering is not disturbed by the heavier groups.
    for group in &extraction.classifications {
        results.insert(group.task.clone(), classification_value(group, flags));
    }

    if let Some(entity) = entities_value(extraction, spans, text, shape, flags) {
        results.insert("entities".to_string(), entity);
    }
    for group in &extraction.structures {
        results.insert(
            group.task.clone(),
            structure_value(group, spans, text, flags),
        );
    }
    for record in &extraction.records {
        results.insert(
            record.task.clone(),
            record_value(record, spans, text, flags),
        );
    }
    // Relations are keyed by bare type name and carry `{head, tail}` objects.
    // Both sides report the edge score, so they are built together rather than
    // converted independently — a per-side conversion would have no score to
    // use and would have to invent one.
    let relations = relations_value(extraction, spans, text, flags);
    for (relation_type, value) in relations {
        results.insert(relation_type, value);
    }
    results
}

/// `entities`: one map from label to spans, or to a single value / `null` for a
/// scalar-dtype label.
///
/// Returns `None` when the schema declared no entity labels, so the key is
/// absent rather than present-and-empty — the reference only emits `entities`
/// for a group the schema actually routed.
fn entities_value(
    extraction: &Extraction,
    spans: &WordCharSpans,
    text: &str,
    shape: &RawShape,
    flags: ShapeFlags,
) -> Option<Value> {
    if shape.entities.labels.is_empty() {
        return None;
    }
    let mut by_label: BTreeMap<&str, Vec<&ExtractedSpan>> = BTreeMap::new();
    for span in &extraction.spans {
        by_label.entry(span.field.as_str()).or_default().push(span);
    }
    let mut map = Map::new();
    for label in &shape.entities.labels {
        let owned = by_label
            .get(label.as_str())
            .map(|list| list.as_slice())
            .unwrap_or(&[]);
        if shape.entities.is_scalar(label) {
            map.insert(
                label.clone(),
                match owned.first() {
                    Some(span) => span_value_for(span, spans, text, flags),
                    // A scalar label with nothing to report is `null`, which is a
                    // different JSON type from the list an empty label gets.
                    None => Value::Null,
                },
            );
        } else {
            map.insert(
                label.clone(),
                Value::Array(entity_spans(owned, spans, text, flags)),
            );
        }
    }
    Some(Value::Array(vec![Value::Object(map)]))
}

/// One legacy structure group: a list holding the single instance's field map.
///
/// A legacy group always emits **one** instance even when every field is empty —
/// dropping empty instances is the record path's rule, not this one's — so the
/// array has one element regardless.
fn structure_value(
    group: &ExtractedStructure,
    spans: &WordCharSpans,
    text: &str,
    flags: ShapeFlags,
) -> Value {
    let mut instance = Map::new();
    for (name, field) in &group.fields {
        instance.insert(
            name.clone(),
            structure_field_value(field, spans, text, flags),
        );
    }
    Value::Array(vec![Value::Object(instance)])
}

/// A record group: a list of instances, one per formed record.
fn record_value(
    record: &ExtractedRecord,
    spans: &WordCharSpans,
    text: &str,
    flags: ShapeFlags,
) -> Value {
    Value::Array(
        record
            .fields
            .iter()
            .map(|(query_id, bound)| {
                Value::Array(
                    bound
                        .iter()
                        .map(|(start, end)| match spans.char_range(*start, *end) {
                            Some((char_start, char_end)) => span_value(
                                &surface(text, char_start, char_end),
                                record_score_for(record, *query_id, *start, *end),
                                char_start,
                                char_end,
                                flags,
                            ),
                            None => Value::Null,
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

/// The score the record head assigned to one field of one instance.
///
/// `ExtractedRecord` keys its scores by `(field, span)` rather than nesting them
/// per instance, so the instance a `(query_id, span)` pair belongs to has to be
/// found rather than indexed.
fn record_score_for(record: &ExtractedRecord, query_id: usize, start: usize, end: usize) -> f32 {
    let Some(scores) = record.field_scores.get(&query_id) else {
        return record.score;
    };
    // The span order within a field is the resolution order, and the score list
    // is parallel to it, so the position is what identifies the score. Falling
    // back to the instance score keeps a mismatch from reading as `0.0`.
    record
        .fields
        .get(&query_id)
        .and_then(|bound| bound.iter().position(|(s, e)| *s == start && *e == end))
        .and_then(|index| scores.get(index).copied())
        .unwrap_or(record.score)
}

/// A structure field's value, which is one of four shapes depending on the
/// field's dtype and whether it declares `choices`.
fn structure_field_value(
    field: &StructureField,
    spans: &WordCharSpans,
    text: &str,
    flags: ShapeFlags,
) -> Value {
    match field {
        StructureField::Scalar(slot) => slot
            .as_ref()
            .map(|span| structure_span_value(span, spans, text, flags))
            .unwrap_or(Value::Null),
        StructureField::List(list) => Value::Array(
            list.iter()
                .map(|span| structure_span_value(span, spans, text, flags))
                .collect(),
        ),
        StructureField::ChoiceScalar(slot) => slot
            .as_ref()
            .map(|choice| choice_value(choice, flags))
            .unwrap_or(Value::Null),
        StructureField::ChoiceList(choices) => Value::Array(
            choices
                .iter()
                .map(|choice| choice_value(choice, flags))
                .collect(),
        ),
    }
}

fn structure_span_value(
    span: &StructureSpan,
    spans: &WordCharSpans,
    text: &str,
    flags: ShapeFlags,
) -> Value {
    match spans.char_range(span.start, span.end) {
        Some((char_start, char_end)) => span_value(
            &surface(text, char_start, char_end),
            span.score,
            char_start,
            char_end,
            flags,
        ),
        None => Value::Null,
    }
}

/// A classification: `(label, score)` for a single-label group, a list of pairs
/// for a multi-label one.
///
/// Only the labels the reference selected are emitted, and a single-label group
/// emits its winner even when nothing cleared the threshold — the argmax is the
/// reported value. A multi-label group with nothing selected emits an **empty
/// list**, which is why the two branches cannot share a helper: one falls back to
/// the argmax and the other does not.
fn classification_value(group: &ClassificationResult, _flags: ShapeFlags) -> Value {
    if group.multi_label {
        return Value::Array(
            group
                .selected
                .iter()
                .filter_map(|label| {
                    group
                        .labels
                        .iter()
                        .position(|name| name == label)
                        .and_then(|index| group.probabilities.get(index).copied())
                        .map(|probability| {
                            Value::Array(vec![
                                Value::String(label.clone()),
                                json_number(probability),
                            ])
                        })
                })
                .collect(),
        );
    }
    match (&group.choice_label, group.probabilities.first()) {
        (Some(label), Some(probability)) => Value::Array(vec![
            Value::String(label.clone()),
            json_number(*probability),
        ]),
        _ => Value::Array(Vec::new()),
    }
}

/// Relation types mapped to their edges, in the order the types first appear.
///
/// The reference emits one key per type and the port already dedupes **per
/// type**, so grouping here is just a partition — no cross-type merge, which
/// would drop edges of one type in favour of another.
fn relations_value(
    extraction: &Extraction,
    spans: &WordCharSpans,
    text: &str,
    flags: ShapeFlags,
) -> Vec<(String, Value)> {
    let mut order: Vec<String> = Vec::new();
    let mut grouped: BTreeMap<String, Vec<&ExtractedRelation>> = BTreeMap::new();
    for relation in &extraction.relations {
        if !grouped.contains_key(&relation.relation_type) {
            order.push(relation.relation_type.clone());
        }
        grouped
            .entry(relation.relation_type.clone())
            .or_default()
            .push(relation);
    }
    order
        .into_iter()
        .map(|relation_type| {
            let edges = grouped
                .get(&relation_type)
                .map(|list| list.as_slice())
                .unwrap_or(&[])
                .iter()
                .map(|relation| relation_value(relation, spans, text, flags))
                .collect();
            (relation_type, Value::Array(edges))
        })
        .collect()
}

/// One relation edge.
///
/// Both sides carry the **same** score, because the edge is scored once and the
/// confidence is attached to each side afterwards. Building the sides
/// independently would leave each without a score to report.
fn relation_value(
    relation: &ExtractedRelation,
    spans: &WordCharSpans,
    text: &str,
    flags: ShapeFlags,
) -> Value {
    if !flags.include_confidence && !flags.include_spans {
        // `_decode_relations`'s final `else` builds a bare `(head, tail)` pair,
        // so an edge with neither flag is a two-element array of surfaces. It
        // still needs the word-to-character translation, because the surfaces are
        // sliced from the text just as the dicts are.
        return Value::Array(vec![
            side_surface(relation.head_start, relation.head_end, spans, text),
            side_surface(relation.tail_start, relation.tail_end, spans, text),
        ]);
    }
    let mut edge = Map::new();
    edge.insert(
        "head".to_string(),
        side_value(
            relation.head_start,
            relation.head_end,
            relation.score,
            spans,
            text,
            flags,
        ),
    );
    edge.insert(
        "tail".to_string(),
        side_value(
            relation.tail_start,
            relation.tail_end,
            relation.score,
            spans,
            text,
            flags,
        ),
    );
    Value::Object(edge)
}

/// The flags-off relation side: the surface string, or `null` when the word
/// range has no character range.
///
/// Deliberately the same `null` the flags-on path returns for a missing range.
/// The word-joined text the decode stage already holds would be a plausible
/// fallback and it is *wrong*: those words were joined with single spaces while
/// the reference's surface is a slice of the original text, so the two disagree
/// wherever the document has unusual spacing. Inventing one in this branch and
/// nulling in the other made the two flag states treat the same impossible
/// input differently.
fn side_surface(word_start: usize, word_end: usize, spans: &WordCharSpans, text: &str) -> Value {
    match spans.char_range(word_start, word_end) {
        Some((char_start, char_end)) => Value::String(surface(text, char_start, char_end)),
        None => Value::Null,
    }
}

fn side_value(
    word_start: usize,
    word_end: usize,
    score: f32,
    spans: &WordCharSpans,
    text: &str,
    flags: ShapeFlags,
) -> Value {
    match spans.char_range(word_start, word_end) {
        Some((char_start, char_end)) => span_value(
            &surface(text, char_start, char_end),
            score,
            char_start,
            char_end,
            flags,
        ),
        None => Value::Null,
    }
}

fn entity_spans(
    spans: &[&ExtractedSpan],
    word_spans: &WordCharSpans,
    text: &str,
    flags: ShapeFlags,
) -> Vec<Value> {
    spans
        .iter()
        .map(|span| span_value_for(span, word_spans, text, flags))
        .collect()
}

fn span_value_for(
    span: &ExtractedSpan,
    word_spans: &WordCharSpans,
    text: &str,
    flags: ShapeFlags,
) -> Value {
    match word_spans.char_range(span.start, span.end) {
        Some((char_start, char_end)) => span_value(
            &surface(text, char_start, char_end),
            span.score,
            char_start,
            char_end,
            flags,
        ),
        None => Value::Null,
    }
}
