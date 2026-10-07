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

use std::collections::BTreeMap;

use regex::Regex;
use std::sync::OnceLock;

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

/// `_record_local_choice_mentions` (`engine.py:595-654`): find the choice
/// literals that occur in the document text and assign each to a record.
///
/// Returns `(has_literal_choices, per-record mentions)`, keyed by record index.
/// `has_literal_choices` distinguishes "the document mentioned some choice" from
/// "it mentioned none", which is what the caller uses to decide between the
/// document-level answer and the schema-prefix fallback.
///
/// The reference calls this a *pure function* of `(text, choices,
/// anchor_char_spans)`, and it is: the regex find, the preceding-anchor
/// assignment, and the per-record dedup. Three rules are load-bearing and each
/// is a way a plausible port diverges:
///
/// 1. **A choice is a whole word**, `(?<!\w)…(?!\w)`, case-insensitively. So
///    `paris` is not reported inside `Parisian`, and a mention is a single
///    contiguous run — never a fuzzy match.
/// 2. **Ownership is by the *preceding* anchor, not the nearest.** A mention
///    between two anchors belongs to the earlier one; a mention before the first
///    anchor binds to the first. The reference's example is two `Amazon …`
///    sentences, and the trailing `(books)` goes to the second record.
/// 3. **Values are semantic sets.** Within one record a value is kept once, at
///    its first source occurrence, and the retained list is in source order. The
///    *declared* choice is reported, not the document's casing.
pub fn record_local_choice_mentions(
    text: &str,
    choices: &[String],
    anchor_char_spans: &[Option<(usize, usize)>],
) -> (bool, BTreeMap<usize, Vec<(String, usize, usize)>>) {
    // Byte offset -> char offset, so the reported spans index the string the way
    // Python's `str` slicing does.
    let mut char_of_byte = vec![0usize; text.len() + 1];
    let mut chars = 0usize;
    for (byte, _) in text.char_indices() {
        char_of_byte[byte] = chars;
        chars += 1;
    }
    char_of_byte[text.len()] = chars;

    let mut mentions: Vec<(String, usize, usize)> = Vec::new();
    for choice in choices {
        for (start, end) in find_word_occurrences(text, choice) {
            mentions.push((choice.clone(), char_of_byte[start], char_of_byte[end]));
        }
    }
    if mentions.is_empty() {
        return (false, BTreeMap::new());
    }
    let valid: Vec<(usize, (usize, usize))> = anchor_char_spans
        .iter()
        .enumerate()
        .filter_map(|(index, anchor)| anchor.map(|a| (index, a)))
        .collect();
    if valid.is_empty() {
        // Mentions exist but no record has an anchor to own them, so the caller
        // must not treat them as assigned to anything.
        return (true, BTreeMap::new());
    }
    let mut ordered = valid.clone();
    ordered.sort_by_key(|(index, anchor)| (anchor.0, *index));

    let mut assigned: BTreeMap<usize, Vec<(String, usize, usize)>> = BTreeMap::new();
    mentions.sort_by_key(|(_, start, _)| *start);
    for mention in mentions {
        let preceding: Vec<(usize, usize)> = ordered
            .iter()
            .filter(|(_, anchor)| anchor.0 <= mention.1)
            .map(|(index, anchor)| (anchor.0, *index))
            .collect();
        let owner = if preceding.is_empty() {
            ordered[0].0
        } else {
            preceding
                .iter()
                .max_by_key(|(anchor_start, index)| (*anchor_start, *index))
                .expect("preceding is non-empty")
                .1
        };
        assigned.entry(owner).or_default().push(mention);
    }

    for owned in assigned.values_mut() {
        let mut unique: Vec<(String, usize, usize)> = Vec::new();
        let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for mention in owned.iter() {
            if seen.insert(mention.0.as_str()) {
                unique.push(mention.clone());
            }
        }
        unique.sort_by_key(|(_, start, _)| *start);
        *owned = unique;
    }
    (true, assigned)
}

/// Byte spans of every `(?<!\w)choice(?!\w)` occurrence, case-insensitively.
///
/// The `regex` crate has **no look-around**, so the reference's look-behind and
/// look-ahead cannot be spelled directly and `\b` is not a substitute — `\b`
/// anchors a *word boundary*, which differs from "not a word character" on both
/// sides whenever the choice itself starts or ends with punctuation. Instead this
/// enumerates whole words and compares: matching `[\p{L}\p{N}_]+` and keeping
/// the ones equal to `choice` under case folding is exactly the reference's rule,
/// and it also makes the choice a literal so metacharacters cannot widen it.
///
/// `\w` is spelled out for the same reason the word splitter spells it — the
/// `regex` crate's `\w` also folds combining marks, which Python's does not.
fn find_word_occurrences(text: &str, choice: &str) -> Vec<(usize, usize)> {
    if choice.is_empty() {
        return Vec::new();
    }
    let Some(re) = literal_pattern(choice) else {
        return Vec::new();
    };
    let is_word = |c: char| {
        // `\w` spelled out: `\p{L}` or `\p{N}` or `_`. The `regex` crate's own `\w`
        // would also fold combining marks, which Python's does not.
        c.is_alphanumeric() || c == '_'
    };
    re.find_iter(text)
        .filter(|m| {
            let before_ok = text[..m.start()]
                .chars()
                .next_back()
                .is_none_or(|c| !is_word(c));
            let after_ok = text[m.end()..].chars().next().is_none_or(|c| !is_word(c));
            before_ok && after_ok
        })
        .map(|m| (m.start(), m.end()))
        .collect()
}

/// A case-insensitive literal match for `choice`, escaping it so a choice
/// containing regex metacharacters matches itself. This is `re.escape(choice)`
/// plus `re.IGNORECASE`, and it is what makes the reference's `a.b` a literal
/// `a\.b` rather than "any character" — the reference escapes before compiling.
///
/// Compiled per call. `find_word_occurrences` runs once per choice per decode,
/// not per span, so caching would add a lock for no measurable gain.
fn literal_pattern(choice: &str) -> Option<Regex> {
    Regex::new(&format!("(?i){}", regex::escape(choice))).ok()
}

/// `[\p{L}\p{N}_]+`, the word run the choice is matched against.
fn word_run_pattern() -> Option<&'static Regex> {
    static PATTERN: OnceLock<Option<Regex>> = OnceLock::new();
    PATTERN
        .get_or_init(|| Regex::new(r"[\p{L}\p{N}_]+").ok())
        .as_ref()
}
