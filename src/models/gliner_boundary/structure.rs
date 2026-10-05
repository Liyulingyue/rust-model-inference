//! Legacy `json_structures` decode — the path taken by a `[C]` group with **no**
// `record_metadata`.
//!
//! Mirrors `BoundaryExtractor._decode_json_structures`
//! (`target/gliner2-oracle/gliner2/models/boundary/engine.py:506-588`) and
//! `_format_structure_field` (`engine.py:762`).
//!
//! # This is a different feature from records, not a worse one
//!
//! A structure group is a record only when the schema annotates it with a `mode`
//! in `record_metadata`. Without that it decodes here, and the two paths are not
//! two implementations of the same thing:
//!
//! * **Records** form *instances*. Several `person` records can come out of one
//!   schema, each an anchor plus an assignment, and exclusive fields are solved
//!   jointly so a mention cannot bind twice.
//! * **Legacy structures** emit exactly **one** instance per group. The
//!   reference's docstring says why: "Boundary checkpoints do not have the span
//!   architecture's count-slot axis, so legacy structures are emitted as one
//!   instance containing all list-valued fields and the best scalar value for
//!   each scalar field." There is nothing to form multiple instances *from*.
//!
//! # Two orderings that decide the answer
//!
//! * **Field order is the schema's**, not the routed query order. The reference
//!   reads `metadata["field_orders"]` precisely to pin it, and emits the instance
//!   as an `OrderedDict` in that order.
//! * **A scalar field binds `spans[0]`** — the *first* resolved span, and the
//!   rest are discarded. The resolver returns `(-score, start, end)`, so the
//!   ordering of the candidates decides which span a scalar takes. This is why
//!   `scored` must arrive already thresholded and sorted: re-sorting here would
//!   silently change which value a `str` field reports.

use super::overlap::{resolve_overlaps, OverlapPolicy, ScoredSpan};

// ---------------------------------------------------------------------------
// Legacy `json_structures` decode (no `record_metadata`)
// ---------------------------------------------------------------------------

/// One resolved span of a legacy structure field.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureSpan {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub score: f32,
}

/// One literal-enum value: the declared choice and its score.
///
/// A `choices` field has no document location. It is scored at the choice's own
/// token inside the **prefix** on the text stream (`processor.py:645`), not at a
/// span of the input, so there is no `start`/`end` to report and `StructureSpan`
/// would be a lie.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceValue {
    /// The literal exactly as the schema declared it — the reference does not
    /// lower-case the reported value, only the lookup key.
    pub text: String,
    pub score: f32,
}

/// One field of a legacy structure instance.
#[derive(Clone, Debug, PartialEq)]
pub enum StructureField {
    /// A `dtype: "str"` field: the **best** span, or absent.
    ///
    /// The reference takes `spans[0]` and ignores the rest, so a scalar field
    /// is "the best one" rather than "all of them" — which is what makes this
    /// different from a record's `required_one` field, where every instance must
    /// bind one and the assignment decides which.
    Scalar(Option<StructureSpan>),
    /// Every surviving span, in resolution order.
    List(Vec<StructureSpan>),
    /// A `choices` field under `dtype: "str"`: the argmax, or absent.
    ///
    /// `None` when even the best choice is below the threshold. A scalar choice
    /// field never falls back to the first choice — it reports nothing, which is
    /// the behaviour `unreachable_threshold_scalar_reports_nothing` pins.
    ChoiceScalar(Option<ChoiceValue>),
    /// A `choices` field under `dtype: "list"`: every choice at or above the
    /// threshold, in **declaration** order.
    ///
    /// Declaration order, not score order: the reference iterates `present`,
    /// which follows the schema. `uppercase_choices` pins it, where the second
    /// choice scores higher yet is reported second.
    ChoiceList(Vec<ChoiceValue>),
}

/// One legacy `json_structures` instance.
///
/// There is exactly one per structure group: the docstring on the reference's
/// decoder says "legacy structures are emitted as one instance containing all
/// list-valued fields and the best scalar value for each scalar field". Boundary
/// checkpoints have no count-slot axis, so there is nothing to form multiple
/// instances from — that is exactly the gap the record head fills.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureInstance {
    pub task: String,
    /// Field name -> value, in the schema's field order.
    pub fields: Vec<(String, StructureField)>,
}

/// Decode the `json_structures` groups that are **not** records.
///
/// `candidates` is the `[B, Q, C]` pool contract; `scored[i]` is one query's
/// threshold-and-sort survivors as `(score, start, end)`, and `is_scalar[i]`
/// says whether query `i` is a `dtype: "str"` field.
///
/// Two ordering details are load-bearing. Fields follow the schema's declared
/// order, not the routed query order, because `field_orders` exists precisely to
/// pin it. And the span *order within a field* is the resolver's output order
/// (`(-score, start, end)`), because a scalar field takes `spans[0]` — so the
/// ordering of `scored` decides which span a scalar binds.
pub fn decode_legacy_structures(
    groups: &[LegacyStructureGroup<'_>],
    overlap_policy: OverlapPolicy,
    validators: &std::collections::BTreeMap<String, super::validator::CompiledValidators>,
) -> Vec<StructureInstance> {
    let mut out = Vec::new();
    for group in groups {
        let mut instance = StructureInstance {
            task: group.name.to_string(),
            fields: Vec::new(),
        };
        for (index, &field_name) in group.field_names.iter().enumerate() {
            let query = group.query_ids[index];
            let spans = &group.scored[query];
            let is_scalar = group.is_scalar.get(query).copied().unwrap_or(false);
            // `_resolve_spans` runs on the survivors, so the overlap policy still
            // applies here: two fields' spans never interact, but one field's
            // candidates do.
            // `resolve_overlaps` returns indices into its input, in resolved
            // order, so the scores come back off the original slice.
            let candidates: Vec<ScoredSpan> = spans
                .iter()
                .map(|&(score, start, end)| ScoredSpan { score, start, end })
                .collect();
            let kept: Vec<StructureSpan> = resolve_overlaps(&candidates, overlap_policy)
                .into_iter()
                .map(|index| candidates[index])
                // The reference's bounds check, minus the `offset` the word-routed
                // inference path does not have.
                .filter(|span| span.start < span.end && span.end <= group.words.len())
                .map(|span| StructureSpan {
                    start: span.start,
                    end: span.end,
                    text: group.words[span.start..span.end].join(" "),
                    score: span.score,
                })
                // `_decode_json_structures` filters on the derived surface
                // (`engine.py:576-580`), so a validator sees the same string the
                // output reports.
                //
                // The key is `<group>.<field>`, the shape `field_metadata` uses and
                // the same shape `_query_thresholds` looks its threshold up under.
                // A bare field name would collide across two groups declaring the
                // same field, and would never match what the caller built.
                .filter(
                    |span| match validators.get(&format!("{}.{}", group.name, field_name)) {
                        Some(rules) if !rules.is_empty() => rules.accepts(&span.text),
                        _ => true,
                    },
                )
                .collect();
            let value = if is_scalar {
                StructureField::Scalar(kept.into_iter().next())
            } else {
                StructureField::List(kept)
            };
            instance.fields.push((field_name.to_string(), value));
        }
        // `if any(value is not None and value != [] for value in instance.values())`:
        // a structure whose every field came back empty is dropped, not emitted
        // as an empty object.
        let has_content = instance.fields.iter().any(|(_, value)| match value {
            StructureField::Scalar(Some(_)) => true,
            StructureField::List(spans) => !spans.is_empty(),
            StructureField::Scalar(None) => false,
            // A choice field that resolved to a value is content, exactly like a
            // span. One that resolved to nothing — every choice below the
            // threshold, or none found in the prefix — is not.
            StructureField::ChoiceScalar(choice) => choice.is_some(),
            StructureField::ChoiceList(choices) => !choices.is_empty(),
        });
        if has_content {
            out.push(instance);
        }
    }
    out
}

/// One non-record `json_structures` group, in the shape [`decode_legacy_structures`] reads.
pub struct LegacyStructureGroup<'a> {
    /// The `json_structures` group name.
    pub name: &'a str,
    /// Field names in **schema** order. `fields[i]` indexes into `scored` /
    /// `is_scalar` for the query this field came from.
    pub field_names: Vec<&'a str>,
    /// Query id per field, in `field_names` order. This is how the schema order
    /// survives the routed query order.
    pub query_ids: Vec<usize>,
    /// `[query][candidate]` -> `(score, start, end)` after thresholding.
    pub scored: &'a [Vec<(f32, usize, usize)>],
    /// Per query: `dtype == "str"`.
    pub is_scalar: &'a [bool],
    /// The document's word list, for the surface text.
    pub words: &'a [String],
}

/// `_find_choice_idx` (`runtime.py:1206`): the index of `choice` among the
/// prefix tokens, or `None`.
///
/// Two details the reference gets right that are easy to get wrong:
///
/// * the comparison lower-cases **both** sides, so `"Happy"` is found in a prefix
///   rendered as `Happy` and a lowercase schema value is found in either;
/// * it returns the **first** match, and `_decode_choice_field` skips a repeated
///   literal before looking it up, so a choice declared twice is scored once at
///   its first occurrence.
///
/// The prefix entries are matched verbatim and never re-split, so a multi-word
/// literal is one entry and matches as one entry.
pub fn find_choice_idx(choice: &str, prefix_tokens: &[String]) -> Option<usize> {
    let wanted = choice.to_lowercase();
    prefix_tokens
        .iter()
        .position(|token| token.to_lowercase() == wanted)
}

/// The choices of one field, in the order the reference's `present` list holds
/// them: declared order, first occurrence of each, skipping any literal the
/// prefix does not contain.
pub fn present_choices(
    choice_literals: &[String],
    prefix_tokens: &[String],
) -> Vec<(String, usize)> {
    let mut present: Vec<(String, usize)> = Vec::new();
    for choice in choice_literals {
        if present.iter().any(|(seen, _)| seen == choice) {
            continue;
        }
        if let Some(index) = find_choice_idx(choice, prefix_tokens) {
            present.push((choice.clone(), index));
        }
    }
    present
}
