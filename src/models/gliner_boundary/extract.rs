//! Real entry point: text + extractive schema -> spans.
//!
//! This is the module that makes the model usable. Everything else in
//! `gliner_boundary` starts from `text_states` / `query_states`; this one owns
//! the whole path a caller actually takes:
//!
//! 1. `SchemaTransformer`-shaped prompt assembly (`[E]` child markers, not
//!    Decide's `[L]`), recording the two routing index sets the boundary head
//!    needs.
//! 2. The DeBERTa-v3 encoder pass.
//! 3. Gathering `text_states` at each word's first subword and `query_states`
//!    at each `[E]` marker — `BoundaryExtractorModel._encode_core`'s
//!    `fast_routing` branch (`boundary/model.py:1300-1320`).
//! 4. `score_document_candidates` — the `candidate_pool = "shared"` mainline.
//! 5. `sigmoid(pair_logits / pair_temperature)`, threshold, and the reference's
//!    sort order.
//!
//! Span offsets are **word** indices into the normalized, lowercased word list,
//! which is what the reference reports; `ExtractedSpan::text` resolves them back
//! to a string.
//!
//! Scope: extractive (`[E]`) schemas only. Classification groups (`[C]`) are
//! scored by the shared `classifier.0`/`classifier.3` head, and relations
//! (`[R]`) by `relation_scorer`; both are separate heads and are not wired here
//! (see `glinerTODO.md` 5.2.3 / 5.2.4b). Passing a schema whose fields do not
//! all come from one group would silently mis-route, so the marker is chosen by
//! the caller rather than inferred.

use std::collections::BTreeMap;

use crate::core::tensor::TensorSource;
use crate::models::gliner::compute;
use crate::models::gliner::prompt::{self, BoundaryTaskKind, EncodedPrompt, Task, C_TOKEN};
use crate::models::gliner_boundary::tensor_util::apply_linear_full;

use super::loader::BoundaryModel;
use super::overlap::{
    normalize_overlap_policy, resolve_overlaps, OverlapPolicy, ScoredSpan as OverlapSpan,
};
use super::record_head::{
    decode_group, RecordCandidates, RecordDecodeSettings, RecordGroup, RecordHead,
};
use super::record_spec::{compile_record_specs, LayoutQuery, RecordSpec};
use super::relations::{
    self, ExtractedRelation, RelationCandidates, RelationDecodeSettings, RelationProposalSettings,
    RelationStates, RelationTypeSpec,
};
use super::spans::{
    group_scored_candidates, score_document_candidates, DocumentCandidateBatch, QueryThresholds,
};
use super::structure::{decode_legacy_structures, LegacyStructureGroup, StructureField};

/// The reference's default score threshold, used when the caller does not pass
/// one (`_group_scored_candidates`'s `threshold: float = 0.5`).
const DEFAULT_SCORE_THRESHOLD: f32 = 0.5;

/// One extracted span for one schema field.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtractedSpan {
    /// The schema field this span was scored against, e.g. `"person"`.
    pub field: String,
    /// Index of that field among the query markers.
    pub query_index: usize,
    /// `sigmoid(pair_logit / pair_temperature)`.
    pub score: f32,
    /// Half-open word offsets into the normalized word list.
    pub start: usize,
    pub end: usize,
    /// The spanned words joined by single spaces.
    pub text: String,
    /// The pair logit before the sigmoid, for callers that want a margin.
    pub logit: f32,
}

/// Assemble the boundary prompt for `tasks` and `text`.
///
/// `child_marker` is `[E]` for extractive groups. It is a parameter rather than
/// derived from the task because the reference picks it from the schema's task
/// type, which the Rust `Task` type does not carry.
pub fn encode_boundary_prompt(
    model: &BoundaryModel<'_>,
    tasks: &[Task],
    text: &str,
    child_marker: &str,
) -> Result<EncodedPrompt, String> {
    let kinds = vec![
        if child_marker == C_TOKEN {
            BoundaryTaskKind::JsonStructure
        } else {
            BoundaryTaskKind::Entities
        };
        tasks.len()
    ];
    // The single-marker convenience path takes no schema, so no `choices`.
    encode_mixed_boundary_prompt(model, tasks, &kinds, None, text)
}

/// Prompt assembly for a mix of group kinds. `kinds[i]` describes `tasks[i]`.
/// `schema` is the caller's raw schema, read only for `choices`; pass `None`
/// when there is none, which is every schema on the paths that predate it.
pub fn encode_mixed_boundary_prompt(
    model: &BoundaryModel<'_>,
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    schema: Option<&serde_json::Value>,
    text: &str,
) -> Result<EncodedPrompt, String> {
    let text_prefix = schema.map(prompt::render_choice_prefix).unwrap_or_default();
    match &model.tokenizer {
        crate::models::gliner::ModelTokenizer::SentencePiece(spm) => {
            prompt::build_mixed_boundary_prompt(tasks, kinds, &text_prefix, text, spm)
        }
        crate::models::gliner::ModelTokenizer::Json(fast) => {
            prompt::build_mixed_boundary_prompt_with(tasks, kinds, &text_prefix, text, |part| {
                let encoding = fast
                    .encode(part, false)
                    .map_err(|error| error.to_string())?;
                if encoding.get_ids().is_empty() {
                    return Err(format!("tokenizer returned no IDs for {part:?}"));
                }
                Ok(encoding.get_ids().to_vec())
            })
        }
    }
}

/// Gather `text_states` / `query_states` from the encoder's hidden states.
///
/// Word pooling is `first` (the checkpoint's `token_pooling`), so each text
/// state is the hidden state at that word's first subword. Mirrors
/// `gather_routed` in `_encode_core`: the gather is clamped and then multiplied
/// by the mask.
pub fn gather_states(hidden: &[f32], positions: &[usize], hidden_size: usize) -> Vec<f32> {
    let rows = hidden.len() / hidden_size.max(1);
    let mut out = vec![0.0f32; positions.len() * hidden_size];
    for (target, &position) in positions.iter().enumerate() {
        let source = position.min(rows.saturating_sub(1)) * hidden_size;
        out[target * hidden_size..][..hidden_size]
            .copy_from_slice(&hidden[source..][..hidden_size]);
    }
    out
}

/// Everything one inference pass produces.
pub struct Extraction {
    /// The span candidates, in the public `[B, Q, C]` order. Empty when the
    /// schema declared no extractive group.
    pub candidates: DocumentCandidateBatch,
    /// Decoded spans, sorted per field by `(-score, start, end)`.
    pub spans: Vec<ExtractedSpan>,
    /// One entry per classification group, in schema order.
    pub classifications: Vec<ClassificationResult>,
    /// Decoded relation edges, in pair order. Empty when the schema declared no
    /// relation group.
    pub relations: Vec<ExtractedRelation>,
    /// Decoded records, one entry per compiled record group in schema order.
    /// Empty unless a `[C]` group carried a `mode` in `record_metadata`.
    pub records: Vec<ExtractedRecord>,
    /// Decoded *legacy* `json_structures` instances — one per group that did
    /// **not** carry a `record_metadata` mode. Mutually exclusive with
    /// `records` for any given group: the annotation is what picks the path.
    pub structures: Vec<ExtractedStructure>,
    /// `null_projection` / `count_head` per extractive query.
    pub query_heads: QueryHeads,
    /// The normalized, lowercased word list the spans index into.
    pub words: Vec<String>,
    /// Field name per extractive query.
    pub query_names: Vec<String>,
}

/// The schema keys that select a decode path, as the reference reads them.
///
/// Grouped because they are two `Option<&Value>` that must not be transposed:
/// `record_metadata` decides record-vs-legacy, `field_metadata` decides
/// scalar-vs-list, and swapping them produces a different decode with no error.
#[derive(Clone, Copy, Debug, Default)]
pub struct SchemaOptions<'a> {
    pub record_metadata: Option<&'a serde_json::Value>,
    pub field_metadata: Option<&'a serde_json::Value>,
    /// Per-entity knobs, keyed by entity label. `_query_thresholds` reads the
    /// `threshold` here for `entities` queries — note that the entity side of
    /// the schema is keyed by *label*, not by `<group>.<field>` the way
    /// `field_metadata` is for `json_structures`.
    pub entity_metadata: Option<&'a serde_json::Value>,
    /// Per-relation-type knobs, keyed by the bare relation name. Not read by
    /// the candidate stage at all; see `resolve_relation_thresholds`.
    pub relation_metadata: Option<&'a serde_json::Value>,
    /// The caller's raw schema, read only for the `choices` prefix.
    ///
    /// Distinct from the metadata tables above because `choices` lives *inside*
    /// each `json_structures` field's value rather than in a side table, and
    /// because it is consumed during prompt assembly — before any of the
    /// decode-side metadata is read.
    pub schema: Option<&'a serde_json::Value>,
}

/// One query's identity, as `_query_thresholds` sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuerySpec {
    /// `"entities"` or `"json_structures"`. The reference's `if`/`elif` chain
    /// matches on exactly these two strings.
    pub task_type: String,
    /// The group's name. Only `json_structures` looks anything up by it, and
    /// only as the `<group>` half of `field_metadata`'s key.
    pub task_name: String,
    pub field_name: String,
}

/// Flatten `tasks` into the query layout, in the order the prompt routes them.
///
/// Classification groups contribute no boundary queries, and relation queries
/// are included because the reference's `_query_thresholds` walks every spec and
/// simply leaves the ones it does not recognise at the caller's default. The
/// order must match `encoded.query_names`, which is the same
/// `json_structures` → `entities` → `relations` order `parse_boundary_schema`
/// produces.
pub fn query_layout(tasks: &[Task], kinds: &[BoundaryTaskKind]) -> Vec<QuerySpec> {
    let mut specs = Vec::new();
    for (task, kind) in tasks.iter().zip(kinds) {
        let task_type = match kind {
            BoundaryTaskKind::Entities => "entities",
            BoundaryTaskKind::JsonStructure => "json_structures",
            BoundaryTaskKind::Relation => "relations",
            BoundaryTaskKind::Classification => continue,
        };
        for label in &task.labels {
            specs.push(QuerySpec {
                task_type: task_type.to_string(),
                task_name: task.name.clone(),
                field_name: label.name.clone(),
            });
        }
    }
    specs
}

/// [`query_layout`], checked against the field names the prompt actually routed.
///
/// A mismatch means the layout and the marker bookkeeping disagree about which
/// field each query scores, and a per-query threshold applied to the wrong query
/// is a wrong answer rather than an error — a schema that configures
/// `entity_metadata["city"]` at 0.9 would silently throttle `person` instead. So
/// this refuses rather than guesses.
fn query_layout_from_names(
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    query_names: &[String],
) -> Result<Vec<QuerySpec>, String> {
    let specs = query_layout(tasks, kinds);
    if specs.len() != query_names.len() {
        return Err(format!(
            "query layout has {} entries but {} queries were routed",
            specs.len(),
            query_names.len()
        ));
    }
    for (spec, name) in specs.iter().zip(query_names) {
        if spec.field_name != *name {
            return Err(format!(
                "query layout expects field {:?} at this position but the prompt routed {:?}",
                spec.field_name, name
            ));
        }
    }
    Ok(specs)
}

/// `_query_thresholds` (`engine.py:197-226`): per-query admission thresholds.
///
/// Two metadata keys, and the difference between them is a `None` check and a
/// key shape:
///
/// * `entities` queries read `entity_metadata[<field_name>]["threshold"]`,
///   because an entity query *is* its label;
/// * `json_structures` queries read
///   `field_metadata["<task_name>.<field_name>"]["threshold"]`.
///
/// Every other task type — relations — keeps `default`. That is not an
/// oversight in the port: a relation's configured threshold is applied once by
/// the relation scorer, and letting it reach the candidate stage would change
/// which pairs get *generated* rather than which are kept. See
/// `resolve_relation_thresholds`.
///
/// The tensor is dense and starts at `default`, so a label the schema never
/// mentions is indistinguishable from one that configured the default
/// explicitly.
pub fn resolve_query_thresholds(
    specs: &[QuerySpec],
    entity_metadata: Option<&serde_json::Value>,
    field_metadata: Option<&serde_json::Value>,
    default: f32,
) -> Vec<f32> {
    let entity = entity_metadata.and_then(|value| value.as_object());
    let field = field_metadata.and_then(|value| value.as_object());
    specs
        .iter()
        .map(|spec| {
            let configured = match spec.task_type.as_str() {
                "entities" => entity
                    .and_then(|table| table.get(&spec.field_name))
                    .and_then(|entry| entry.get("threshold")),
                "json_structures" => field
                    .and_then(|table| table.get(&format!("{}.{}", spec.task_name, spec.field_name)))
                    .and_then(|entry| entry.get("threshold")),
                _ => None,
            };
            configured
                .and_then(|value| value.as_f64())
                .map(|value| value as f32)
                .unwrap_or(default)
        })
        .collect()
}

/// Per-relation-type thresholds, as `_decode_relations` resolves them
/// (`engine.py:837-853`).
///
/// This is a **separate channel** from `resolve_query_thresholds`, and the two
/// thresholds for one relation are independent: the candidate stage admits pairs
/// at the caller's global threshold, and this value is then applied to the
/// scored pair. A port that folds the two together changes which pairs are
/// generated, which is observable in the pair count and not only in the edges.
///
/// Two details are load-bearing:
///
/// * The lookup key is the *resolved* relation name. A relation declared with a
///   description reaches the scorer as `"<name>: <description>"`, so the
///   reference inverts the description map first (`engine.py:837-843`). Using
///   the scorer's own string as the key silently misses.
/// * A present-but-`null` threshold falls back to `default`. `.get(key, default)`
///   does not cover that, which is why the reference re-checks for `None`.
pub fn resolve_relation_thresholds(
    relation_metadata: Option<&serde_json::Value>,
    default: f32,
) -> BTreeMap<String, f32> {
    let mut out = BTreeMap::new();
    let Some(table) = relation_metadata.and_then(|value| value.as_object()) else {
        return out;
    };
    for (name, config) in table {
        let configured = match config.get("threshold") {
            Some(value) => value.as_f64(),
            None => Some(default as f64),
        };
        match configured {
            Some(value) => {
                out.insert(name.clone(), value as f32);
            }
            // An explicit null carries no value, so the caller decides.
            None => {
                out.insert(name.clone(), default);
            }
        }
    }
    out
}

/// Run the whole pipeline.
///
/// `kinds[i]` says which head scores `tasks[i]`. Groups whose kind does not
/// yield boundary queries (`Classification` today) contribute no candidates, and
/// a schema with no extractive group produces no spans.
pub fn run_mixed_extraction(
    model: &BoundaryModel<'_>,
    text: &str,
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    n_threads_arg: usize,
    // The score threshold relations are decoded at, matching the reference,
    // where `_decode_relations` receives the same `threshold` as the span path.
    // `None` is the reference's 0.5 default.
    relation_threshold: Option<f32>,
    // `record_metadata` picks record-vs-legacy per `json_structures` group, and
    // `field_metadata` (`{"<group>.<field>": {"dtype": "str"}}`) picks
    // scalar-vs-list. Grouped so the two cannot be transposed: swapping them
    // changes the decode with no error.
    schema: SchemaOptions<'_>,
) -> Result<Extraction, String> {
    let SchemaOptions {
        record_metadata,
        field_metadata,
        entity_metadata: _,
        relation_metadata,
        schema,
    } = schema;
    if tasks.is_empty() {
        return Err("extraction needs at least one schema task".into());
    }
    let encoded = encode_mixed_boundary_prompt(model, tasks, kinds, schema, text)?;
    let hidden_size = model.config.n_embd;
    if hidden_size == 0 {
        return Err("encoder hidden size is zero".into());
    }
    let hidden = compute::encode(
        &model.encoder,
        &model.config,
        &encoded.input_ids,
        n_threads_arg,
    )?;
    if hidden.len() != encoded.input_ids.len() * hidden_size {
        return Err("encoder output does not match the encoded prompt".into());
    }
    if encoded.text_word_first_positions.len() != encoded.words.len() {
        return Err(format!(
            "word routing is not 1:1: {} words but {} positions",
            encoded.words.len(),
            encoded.text_word_first_positions.len()
        ));
    }
    if encoded.query_positions.len() != encoded.query_names.len() {
        return Err("query marker routing is not 1:1 with the field names".into());
    }
    if encoded.classification_positions.len() != encoded.classification_names.len() {
        return Err("classification routing is not 1:1 with its labels".into());
    }

    let mut extractions = Extraction {
        candidates: DocumentCandidateBatch {
            indices: Vec::new(),
            pair_logits: Vec::new(),
            valid_mask: Vec::new(),
            candidate_states: Vec::new(),
            pool_candidate_features: Vec::new(),
            pool_size: model.settings.pool_size,
        },
        spans: Vec::new(),
        classifications: Vec::new(),
        relations: Vec::new(),
        records: Vec::new(),
        structures: Vec::new(),
        query_heads: QueryHeads {
            null_logits: Vec::new(),
            count_log_rates: Vec::new(),
        },
        words: encoded.words.clone(),
        query_names: encoded.query_names.clone(),
    };

    if !encoded.query_positions.is_empty() {
        let text_states = gather_states(&hidden, &encoded.text_word_first_positions, hidden_size);
        let query_states = gather_states(&hidden, &encoded.query_positions, hidden_size);
        let text_mask = vec![vec![true; encoded.words.len()]];
        let query_mask = vec![vec![true; encoded.query_names.len()]];
        extractions.query_heads =
            query_heads(model, &query_states, encoded.query_names.len(), hidden_size)?;
        extractions.candidates =
            score_document_candidates(model, &text_states, &text_mask, &query_states, &query_mask);
    }

    // Relations reuse the mention candidates the pool already produced, so this
    // stage runs inside the `query_positions` branch: it needs the same
    // word-gathered text states and per-query states.
    if !encoded.query_positions.is_empty() {
        let text_states = gather_states(&hidden, &encoded.text_word_first_positions, hidden_size);
        let query_states = gather_states(&hidden, &encoded.query_positions, hidden_size);
        let (specs, spec_names) = relation_specs(tasks, kinds)?;
        if !specs.is_empty() {
            let scorer = model.relation_scorer.as_ref().ok_or_else(|| {
                format!(
                    "schema declares relation group(s) {spec_names:?} but the checkpoint \
                     sets enable_relations = false"
                )
            })?;
            if spec_names.len() != encoded.query_names.len()
                && specs
                    .iter()
                    .flat_map(|spec| spec.head_query_ids.iter().chain(&spec.tail_query_ids))
                    .any(|&q| q >= encoded.query_names.len())
            {
                return Err(format!(
                    "relation group(s) {spec_names:?} name queries past the {} routed",
                    encoded.query_names.len()
                ));
            }
            let states = RelationStates {
                text: &text_states,
                query: &query_states,
                query_count: encoded.query_names.len(),
                hidden_size,
            };
            let candidates = RelationCandidates {
                indices: &extractions.candidates.indices,
                pair_logits: &extractions.candidates.pair_logits,
                valid_mask: &extractions.candidates.valid_mask,
                q_count: encoded.query_names.len(),
                c_count: extractions.candidates.pool_size,
            };
            extractions.relations = relations::score_relations(
                scorer,
                &RelationProposalSettings::from_settings(&model.settings),
                &states,
                &candidates,
                &specs,
                &extractions.words,
                // `_decode_relations` falls back to the caller's threshold when
                // `relation_metadata` carries no per-type override.
                RelationDecodeSettings::from_settings(
                    &model.settings,
                    relation_threshold.unwrap_or(DEFAULT_SCORE_THRESHOLD),
                    resolve_relation_thresholds(
                        relation_metadata,
                        relation_threshold.unwrap_or(DEFAULT_SCORE_THRESHOLD),
                    ),
                ),
            );
        }
    }

    // Legacy structures: the `[C]` groups that did *not* take the record path.
    // Emitted alongside the spans rather than instead of them — the reference's
    // engine returns both the per-field spans and the structure instances.
    if !encoded.query_positions.is_empty() {
        extractions.structures = score_structures(
            model,
            tasks,
            kinds,
            &extractions.candidates,
            &extractions.words,
            &encoded.query_names,
            SchemaOptions {
                record_metadata,
                entity_metadata: None,
                relation_metadata: None,
                schema: None,
                field_metadata,
            },
            relation_threshold.unwrap_or(DEFAULT_SCORE_THRESHOLD),
        )?;
    }

    // Records: a `[C]` group carrying a `mode` in `record_metadata` compiles to a
    // record spec; one without keeps the legacy structure path and produces
    // nothing here. Records read the same pool candidates the span path does, plus
    // the `candidate_encoder` states, so this runs alongside the relation stage.
    if !encoded.query_positions.is_empty() {
        let query_states = gather_states(&hidden, &encoded.query_positions, hidden_size);
        extractions.records = score_records(
            model,
            &query_states,
            tasks,
            kinds,
            &encoded.query_names,
            &extractions.candidates,
            record_metadata,
        )?;
    }

    // Classification groups: score every `[L]` marker state with the shared
    // classifier, in schema order. `multi_label` comes from the task, and the
    // reference resolves each group's config by task name, so position and name
    // have to agree — hence the explicit check rather than a silent zip.
    if !encoded.classification_positions.is_empty() {
        let states = gather_states(&hidden, &encoded.classification_positions, hidden_size);
        let mut cursor = 0usize;
        for (task, kind) in tasks.iter().zip(kinds) {
            if !matches!(kind, BoundaryTaskKind::Classification) {
                continue;
            }
            let count = task.labels.len();
            if cursor + count > encoded.classification_names.len() {
                return Err(format!(
                    "task {:?}: classification routing ran past {} choices",
                    task.name,
                    encoded.classification_names.len()
                ));
            }
            let routed = &encoded.classification_names[cursor..cursor + count];
            let declared: Vec<&str> = task
                .labels
                .iter()
                .map(|label| label.name.as_str())
                .collect();
            if routed.iter().map(String::as_str).collect::<Vec<_>>() != declared {
                return Err(format!(
                    "task {:?}: classification routing is {:?}, schema says {:?}",
                    task.name, routed, declared
                ));
            }
            let slice = &states[cursor * hidden_size..(cursor + count) * hidden_size];
            extractions
                .classifications
                .push(classify_group(model, task, slice, task.multi_label)?);
            cursor += count;
        }
        if cursor != encoded.classification_names.len() {
            return Err(format!(
                "{} classification choices routed but {} consumed",
                encoded.classification_names.len(),
                cursor
            ));
        }
    }

    Ok(extractions)
}

/// Decode the `json_structures` groups that are **not** records.
///
/// A group takes the record path only when `record_metadata` gives it a `mode`
/// (`compile_record_specs` keys off that), so the legacy path is exactly the
/// complement. Both run over the same pool candidates; the difference is that
/// the record head forms instances and the legacy path does not.
///
/// `is_scalar` comes from the schema's optional `field_metadata`, read the
/// same way the reference reads `field_metadata["<parent>.<field>"]["dtype"]`
/// with a default of `"list"`. A schema that says nothing about dtypes therefore
/// yields all-list fields, which is the reference's default rather than a guess.
fn score_structures(
    model: &BoundaryModel<'_>,
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    candidates: &DocumentCandidateBatch,
    words: &[String],
    query_names: &[String],
    schema: SchemaOptions<'_>,
    threshold: f32,
) -> Result<Vec<ExtractedStructure>, String> {
    let SchemaOptions {
        record_metadata,
        field_metadata,
        entity_metadata,
        relation_metadata: _,
        schema: _,
    } = schema;
    if !tasks
        .iter()
        .zip(kinds)
        .any(|(_, kind)| *kind == BoundaryTaskKind::JsonStructure)
    {
        return Ok(Vec::new());
    }
    let metadata = record_metadata
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let annotated: std::collections::BTreeSet<String> = metadata
        .as_object()
        .map(|map| {
            map.iter()
                .filter(|(_, config)| {
                    config
                        .get("mode")
                        .and_then(|v| v.as_str())
                        .is_some_and(|mode| matches!(mode, "natural" | "latent" | "anchorless"))
                })
                .map(|(name, _)| name.clone())
                .collect()
        })
        .unwrap_or_default();
    let policy = boundary_overlap_policy(model)?;

    // Field order follows the schema's declaration, and `query_ids` maps each
    // declared field back to the query the prompt actually routed it to. Those two
    // orders differ whenever a non-extractive group is interleaved, which is why
    // the mapping is explicit rather than positional.
    let mut is_scalar: Vec<bool> = vec![false; query_names.len()];
    let temperature = if model.settings.pair_temperature > 0.0 {
        model.settings.pair_temperature
    } else {
        1.0
    };
    // One admission pass for the whole batch, shared with the span path, so the
    // per-query thresholds a schema configures apply here too. A structure
    // field is a `json_structures` query, which is one of the two task types
    // `_query_thresholds` reads `field_metadata` for.
    let probabilities: Vec<f32> = candidates
        .pair_logits
        .iter()
        .map(|logit| 1.0 / (1.0 + (-logit / temperature).exp()))
        .collect();
    let thresholds = resolve_query_thresholds(
        &query_layout_from_names(tasks, kinds, query_names)?,
        entity_metadata,
        field_metadata,
        threshold,
    );
    let grouped = group_scored_candidates(
        candidates,
        &probabilities,
        query_names.len(),
        &QueryThresholds::PerQuery {
            values: thresholds,
            queries: query_names.len(),
        },
        None,
        false,
    );
    // `_group_scored_candidates` emits candidates in ascending slot order; the
    // legacy decoder takes `spans[0]` for a scalar field, so ranking by
    // `(-score, start, end)` first is the value choice and must not be redone
    // afterwards.
    let scored: Vec<Vec<(f32, usize, usize)>> = grouped[0]
        .iter()
        .map(|hits| {
            let mut hits = hits.clone();
            hits.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            hits
        })
        .collect();

    // Resolve every field's dtype first: `LegacyStructureGroup` borrows
    // `is_scalar` immutably, so it cannot be filled while a group is alive.
    let dtypes = field_metadata
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let mut legacy: Vec<(usize, String, Vec<String>)> = Vec::new();
    let mut cursor = 0usize;
    for (task, kind) in tasks.iter().zip(kinds) {
        if *kind == BoundaryTaskKind::Classification {
            continue;
        }
        let field_count = task.labels.len();
        if *kind == BoundaryTaskKind::JsonStructure && !annotated.contains(&task.name) {
            for (index, label) in task.labels.iter().enumerate() {
                is_scalar[cursor + index] = dtypes
                    .get(&format!("{}.{}", task.name, label.name))
                    .and_then(|entry| entry.get("dtype"))
                    .and_then(|value| value.as_str())
                    == Some("str");
            }
            legacy.push((
                cursor,
                task.name.clone(),
                task.labels.iter().map(|l| l.name.clone()).collect(),
            ));
        }
        cursor += field_count;
    }
    let groups: Vec<LegacyStructureGroup<'_>> = legacy
        .iter()
        .map(|(start, name, fields)| LegacyStructureGroup {
            name,
            field_names: fields.iter().map(String::as_str).collect(),
            query_ids: (*start..start + fields.len()).collect(),
            scored: &scored,
            is_scalar: &is_scalar,
            words,
        })
        .collect();
    let validators = super::validator::parse_metadata_validators(
        entity_metadata,
        field_metadata,
        std::iter::empty(),
        legacy.iter().flat_map(|(_, name, fields)| {
            fields
                .iter()
                .map(move |field: &String| (name.clone(), field.clone()))
        }),
    )?;
    Ok(decode_legacy_structures(&groups, policy, &validators)
        .into_iter()
        .map(|instance| ExtractedStructure {
            task: instance.task,
            fields: instance.fields,
        })
        .collect())
}

/// Compile and decode every record group in the schema.
///
/// The query layout is rebuilt from `tasks` / `kinds` rather than carried in, for
/// the same reason the relation specs are: the group order fixes the query ids,
/// and deriving both from the same place is what keeps them agreeing. A
/// `json_structures` group without a `mode` compiles to no spec, so it silently
/// keeps the legacy structure path — see `record_spec.rs`.
fn score_records(
    model: &BoundaryModel<'_>,
    query_states: &[f32],
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    query_names: &[String],
    candidates: &DocumentCandidateBatch,
    record_metadata: Option<&serde_json::Value>,
) -> Result<Vec<ExtractedRecord>, String> {
    let metadata = record_metadata
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    if !tasks
        .iter()
        .zip(kinds)
        .any(|(_, kind)| *kind == BoundaryTaskKind::JsonStructure)
    {
        return Ok(Vec::new());
    }
    let Some(head) = model.record_head.as_ref() else {
        // `enable_records = false`: the spec compiler would have to be told the
        // head exists, so this is a schema/weights mismatch rather than a
        // silently empty result.
        return Err(
            "schema declares a json_structures group but the checkpoint sets \
                    enable_records = false"
                .into(),
        );
    };
    let layout = record_query_layout(tasks, kinds, query_names)?;
    let specs = compile_record_specs(&layout, &metadata, &BTreeMap::new())?;
    if specs.is_empty() {
        return Ok(Vec::new());
    }
    let c_count = candidates.pool_size;
    let decode = RecordDecodeSettings {
        anchor_threshold: model.settings.record_anchor_threshold,
        object_threshold: model.settings.record_anchor_proposal_threshold,
        field_threshold: model.settings.record_field_threshold,
        temperature: model.settings.record_temperature,
    };
    let mut out = Vec::new();
    for spec in specs.values() {
        let group = build_record_group(
            head,
            spec,
            query_states,
            query_names.len(),
            candidates,
            c_count,
        )?;
        for record in decode_group(&group, decode)? {
            out.push(ExtractedRecord {
                task: spec.task_name.clone(),
                mode: spec.mode.clone(),
                fields: record.fields,
                field_scores: record.field_scores,
                anchor_span: record.anchor_span,
                score: record.score,
            });
        }
    }
    Ok(out)
}

/// `RecordHead::forward_group` with the pool's candidate batch.
///
/// Split out so `score_records` stays readable; the `candidates` are the same
/// `[B, 1, ...]` slices the span path produced, with `q_count` queries.
fn build_record_group(
    head: &RecordHead<'_>,
    spec: &RecordSpec,
    query_states: &[f32],
    q_count: usize,
    candidates: &DocumentCandidateBatch,
    c_count: usize,
) -> Result<RecordGroup, String> {
    head.forward_group(
        spec,
        query_states,
        &RecordCandidates {
            indices: &candidates.indices,
            pair_logits: &candidates.pair_logits,
            valid_mask: &candidates.valid_mask,
            states: &candidates.candidate_states,
            q_count,
            c_count,
        },
    )
}

/// The query layout `compile_record_specs` binds field names to query ids.
///
/// Only `json_structures` groups become queries here, and each field's
/// `role_index` is its position within its group — which is what makes the
/// compiled specs independent of the caller's field order.
fn record_query_layout(
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    query_names: &[String],
) -> Result<Vec<LayoutQuery>, String> {
    let mut queries = Vec::new();
    let mut query_id = 0usize;
    for (task_index, (task, kind)) in tasks.iter().zip(kinds).enumerate() {
        if *kind == BoundaryTaskKind::Classification {
            continue;
        }
        for (role_index, field) in task.labels.iter().enumerate() {
            if query_id >= query_names.len() {
                return Err(format!(
                    "task {:?} field {:?} has no routed query; {} queries were routed",
                    task.name,
                    field.name,
                    query_names.len()
                ));
            }
            if query_names[query_id] != field.name {
                return Err(format!(
                    "task {:?} field {:?} routed to query {query_id} named {:?}",
                    task.name, field.name, query_names[query_id]
                ));
            }
            queries.push(LayoutQuery {
                query_id,
                task_index,
                task_type: match kind {
                    BoundaryTaskKind::Entities => "entities".to_string(),
                    BoundaryTaskKind::Relation => "relations".to_string(),
                    BoundaryTaskKind::JsonStructure => "json_structures".to_string(),
                    BoundaryTaskKind::Classification => "classifications".to_string(),
                },
                task_name: task.name.clone(),
                role_index,
                role_name: field.name.clone(),
            });
            query_id += 1;
        }
    }
    Ok(queries)
}

/// Build the relation specs, and the group names they came from.
///
/// The reference assigns extractive query ids in schema-group order, counting
/// each group's fields and skipping classification groups entirely
/// (`_build_rel_specs`, `model.py:1449-1502`). A relation group contributes a
/// spec only when it has at least two fields, and the first two are the head and
/// tail roles.
///
/// The `relation_type` is the *prompt-joined* group name — `"founded: who
/// founded what"` when the schema gives a description — because that is what
/// `_schema_group_name` recovers from the `[P]` token and what
/// `_decode_relations`'s alias table is keyed on. The bare name is recovered at
/// decode time, not here.
fn relation_specs(
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
) -> Result<(Vec<RelationTypeSpec>, Vec<String>), String> {
    let mut specs = Vec::new();
    let mut names = Vec::new();
    let mut query_cursor = 0usize;
    for (task, kind) in tasks.iter().zip(kinds) {
        if matches!(kind, BoundaryTaskKind::Classification) {
            // Classification groups emit no extractive query, so they consume no
            // ids — but they do consume positions in the prompt, which is why
            // this cannot be a plain enumerate over the task index.
            continue;
        }
        let field_count = task.labels.len();
        if *kind == BoundaryTaskKind::Relation {
            if field_count >= 2 {
                let prompt = match &task.prompt {
                    Some(text) => format!("{}: {text}", task.name),
                    None => task.name.clone(),
                };
                specs.push(RelationTypeSpec::two_role(
                    prompt,
                    query_cursor,
                    query_cursor + 1,
                ));
                names.push(task.name.clone());
            } else {
                return Err(format!(
                    "relation group {:?} declares {} field(s); the reference needs a head \
                     and a tail, so at least 2",
                    task.name, field_count
                ));
            }
        }
        query_cursor += field_count;
    }
    Ok((specs, names))
}

/// Run the whole pipeline for a single extractive group.
///
/// Kept as the common case's shorthand: one `[E]` group, no classification.
pub fn run_extraction(
    model: &BoundaryModel<'_>,
    text: &str,
    tasks: &[Task],
    child_marker: &str,
    n_threads_arg: usize,
) -> Result<(DocumentCandidateBatch, Vec<String>), String> {
    let kinds = vec![
        if child_marker == C_TOKEN {
            BoundaryTaskKind::JsonStructure
        } else {
            BoundaryTaskKind::Entities
        };
        tasks.len()
    ];
    let result = run_mixed_extraction(
        model,
        text,
        tasks,
        &kinds,
        n_threads_arg,
        None,
        SchemaOptions::default(),
    )?;
    Ok((result.candidates, result.words))
}

/// Threshold the candidate batch into spans.
///
/// `keep = valid_mask & (sigmoid(logit / pair_temperature) >= threshold)`, then
/// per field sorted by `(-score, start, end)` — `decode_candidates`' order.
/// Padded candidates carry `MASK_LOGIT`, so their probability is ~0 and the
/// threshold drops them; the `valid_mask` check is belt and braces.
///
/// `thresholds` is per query, because `_query_thresholds` hands the reference a
/// `[B, Q]` tensor rather than one scalar. A schema may configure one entity
/// label at 0.05 and its neighbour at 0.9, and a scalar cannot express that.
pub fn decode_spans(
    batch: &DocumentCandidateBatch,
    words: &[String],
    field_names: &[String],
    pair_temperature: f32,
    thresholds: &[f32],
    default_threshold: f32,
    overlap_policy: Option<OverlapPolicy>,
    validators: &BTreeMap<String, super::validator::CompiledValidators>,
) -> Vec<ExtractedSpan> {
    let temperature = if pair_temperature > 0.0 {
        pair_temperature
    } else {
        1.0
    };
    let q_count = field_names.len();
    let mut out = Vec::new();
    for q in 0..q_count {
        let threshold = thresholds.get(q).copied().unwrap_or(default_threshold);
        let mut hits: Vec<ExtractedSpan> = Vec::new();
        for slot in 0..batch.pool_size {
            let flat = q * batch.pool_size + slot;
            if !batch.valid_mask[flat] {
                continue;
            }
            let logit = batch.pair_logits[flat];
            let score = 1.0 / (1.0 + (-logit / temperature).exp());
            if score < threshold {
                continue;
            }
            let start = batch.indices[flat * 2];
            let end = batch.indices[flat * 2 + 1];
            if end > words.len() || start >= end {
                continue;
            }
            hits.push(ExtractedSpan {
                field: field_names[q].clone(),
                query_index: q,
                score,
                start,
                end,
                text: words[start..end].join(" "),
                logit,
            });
        }
        if let Some(policy) = overlap_policy {
            // Thresholding leaves overlapping candidates in; the reference then
            // resolves them (`engine.py:36` -> `resolve_overlaps`). Skipping
            // this reports "apple inc" and "apple" as two spans of the same
            // field, which reads as a bug in the model rather than in the
            // decoder.
            let scored: Vec<OverlapSpan> = hits
                .iter()
                .map(|hit| OverlapSpan {
                    score: hit.score,
                    start: hit.start,
                    end: hit.end,
                })
                .collect();
            let keep = resolve_overlaps(&scored, policy);
            let kept: Vec<ExtractedSpan> =
                keep.into_iter().map(|index| hits[index].clone()).collect();
            // `resolve_overlaps` returns ranked order already; re-sorting would
            // be a no-op but hides the contract, so assert it instead.
            debug_assert!(kept.windows(2).all(|pair| {
                pair[0].score > pair[1].score
                    || (pair[0].score == pair[1].score
                        && (pair[0].start, pair[0].end) <= (pair[1].start, pair[1].end))
            }));
            hits = kept;
        } else {
            hits.sort_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then(a.start.cmp(&b.start))
                    .then(a.end.cmp(&b.end))
            });
        }
        // `_decode_entities` filters after `_resolve_spans`, on the surface text
        // it derived (`engine.py:293`), so a validator sees the same string the
        // output reports and not a word join.
        if let Some(rules) = validators.get(&field_names[q]) {
            if !rules.is_empty() {
                hits.retain(|hit| rules.accepts(&hit.text));
            }
        }
        out.extend(hits);
    }
    out
}

/// Text + extractive schema -> spans above `threshold`.
///
/// The default threshold is the reference's (`_group_scored_candidates`).
pub fn extract_spans(
    model: &BoundaryModel<'_>,
    text: &str,
    tasks: &[Task],
    child_marker: &str,
    n_threads_arg: usize,
    threshold: Option<f32>,
) -> Result<Vec<ExtractedSpan>, String> {
    let (batch, words) = run_extraction(model, text, tasks, child_marker, n_threads_arg)?;
    let fields: Vec<String> = tasks
        .iter()
        .flat_map(|task| task.labels.iter().map(|label| label.name.clone()))
        .collect();
    let default = threshold.unwrap_or(DEFAULT_SCORE_THRESHOLD);
    Ok(decode_spans(
        &batch,
        &words,
        &fields,
        model.settings.pair_temperature,
        // This convenience entry point takes no schema, so every query shares
        // the caller's threshold. `run_mixed_extraction` plus
        // `resolve_query_thresholds` is the path that honours per-query values.
        &vec![default; fields.len()],
        default,
        Some(boundary_overlap_policy(model)?),
        &BTreeMap::new(),
    ))
}

/// The checkpoint's overlap policy, canonicalized.
///
/// `_resolved_overlap_policy` (`inference/runtime.py:400`) resolves an explicit
/// per-sample override through `normalize_overlap_policy`, and otherwise falls
/// back to the architecture default — `disallow` for the boundary variant. An
/// unknown name is an error rather than a silent "keep everything", so a
/// mis-transcribed setting cannot quietly double-report overlapping spans.
pub fn boundary_overlap_policy(model: &BoundaryModel<'_>) -> Result<OverlapPolicy, String> {
    normalize_overlap_policy(Some(model.settings.overlap_policy.as_str()), "disallow")
}

/// Drop a whole query's spans when its abstention logit clears the threshold.
///
/// The reference reads this as `sigmoid(null_logits[q]) >
/// abstention_threshold` (`engine.py:269`): a query the model expects to find
/// nothing in is emptied wholesale rather than returning low-scoring spans.
pub fn apply_abstention(
    spans: &mut Vec<ExtractedSpan>,
    query_heads: &QueryHeads,
    abstention_threshold: f32,
) {
    for span in spans.iter_mut() {
        let logit = query_heads
            .null_logits
            .get(span.query_index)
            .copied()
            .unwrap_or(f32::NEG_INFINITY);
        if 1.0 / (1.0 + (-logit).exp()) > abstention_threshold {
            span.text = String::new();
            span.score = 0.0;
        }
    }
    spans.retain(|span| !span.text.is_empty());
}

/// Load a boundary GGUF and check it is the boundary variant.
pub fn load(path: &std::path::Path) -> Result<Box<dyn TensorSource>, String> {
    let source =
        crate::format::ggufrs::open_model_source(path, crate::format::ggufrs::ComponentRole::Llm)
            .map_err(|error| format!("open {}: {error}", path.display()))?;
    if !super::is_boundary_gguf(source.as_ref()) {
        return Err(format!(
            "{} is not a gliner2 boundary variant (gliner2.variant = \"boundary\")",
            path.display()
        ));
    }
    Ok(source)
}

// ---------------------------------------------------------------------------
// Classification head
// ---------------------------------------------------------------------------

/// One decoded record group (`_decode_records` in `engine.py:1285-1320`).
#[derive(Clone, Debug, PartialEq)]
pub struct ExtractedRecord {
    /// The `json_structures` group name.
    pub task: String,
    /// `natural` | `latent` | `anchorless`.
    pub mode: String,
    /// Field query id -> the spans bound to it.
    pub fields: BTreeMap<usize, Vec<(usize, usize)>>,
    pub field_scores: BTreeMap<usize, Vec<f32>>,
    /// Set for `natural` mode: the span the instance seeded from.
    pub anchor_span: Option<(usize, usize)>,
    /// `sigmoid(object logit / record_temperature)`.
    pub score: f32,
}

/// One legacy `json_structures` instance, flattened onto the extraction result.
///
/// The reference's engine keys these by structure name and nests one instance per
/// group; a flat list keeps the group name on each entry, which is the same
/// information without the string keys.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtractedStructure {
    /// The `json_structures` group name.
    pub task: String,
    /// `(field name, value)` in the schema's declared field order.
    pub fields: Vec<(String, StructureField)>,
}

/// One classification group's decoded result.
///
/// The reference's `_extract_classification_result`
/// (`inference/runtime.py:562`) applies the shared classifier to the `[C]`
/// marker states, divides by `classification_temperature`, and picks softmax for
/// a single-label group or sigmoid for a multi-label one.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassificationResult {
    /// The group's prompt, i.e. its `Task::name`.
    pub task: String,
    /// `sigmoid` for a multi-label group, `softmax` otherwise.
    pub activation: &'static str,
    /// `classifier` logits, already divided by the temperature.
    pub logits: Vec<f32>,
    /// Per-label probability, normalized as `activation` says.
    pub probabilities: Vec<f32>,
    pub labels: Vec<String>,
    /// Labels at or above `threshold`. Empty for a single-label group, whose
    /// winner is reported in `choice_label` instead.
    pub selected: Vec<String>,
    /// The argmax label. Also the fallback the reference reports when nothing
    /// clears a multi-label group's threshold.
    pub choice_label: Option<String>,
    pub multi_label: bool,
}

/// `classifier.0` -> ReLU -> `classifier.3` for one hidden-state row.
///
/// The boundary classifier is `create_mlp(hidden, [2 * hidden], 1, dropout,
/// activation="relu", add_layer_norm=False)`, so the ReLU sits between the two
/// linears and the final linear is at index 3 — the layer index that
/// distinguishes the boundary GGUF from Decide's. The reference's decoder
/// slices `embs[1:]` before calling, dropping the group's `[P]` row, which is
/// not scored.
/// `classifier.0` -> ReLU -> `classifier.3` for one hidden-state row.
///
/// The boundary classifier is `create_mlp(hidden, [2 * hidden], 1, dropout,
/// activation="relu", add_layer_norm=False)`, so the ReLU sits between the two
/// linears and the final linear is at index 3 — the layer index that
/// distinguishes the boundary GGUF from Decide's. The reference's decoder
/// slices `embs[1:]` before calling, dropping the group's `[P]` row, which is
/// not scored.
///
/// No `ComputePool` here. The Decide path uses one because it classifies a whole
/// label set, but a boundary pass scores a handful of choices, and pool workers
/// busy-spin while idle — a second pool per extraction turned two concurrent
/// extractions into 48 spinning threads on 12 cores and made the test suite
/// ~12x slower. The work here is 1.2M MACs, so a plain row loop is both faster
/// and free of that contention.
fn classify_state(model: &BoundaryModel<'_>, state: &[f32]) -> Result<f32, String> {
    let hidden_size = model.config.n_embd;
    if state.len() != hidden_size {
        return Err(format!(
            "classifier input is {} wide, expected {hidden_size}",
            state.len()
        ));
    }
    let intermediate = model.classifier_0_bias.len();
    let mut hidden = vec![0.0f32; intermediate];
    apply_linear_full(
        state,
        &model.classifier_0,
        &model.classifier_0_bias,
        &mut hidden,
    );
    // The activation is the boundary classifier's own; the reference hardcodes
    // `activation="relu"` in `create_mlp` for this variant.
    for value in hidden.iter_mut() {
        *value = value.max(0.0);
    }
    let mut out = [0.0f32; 1];
    apply_linear_full(
        &hidden,
        &model.classifier_3,
        &model.classifier_3_bias,
        &mut out,
    );
    Ok(out[0])
}

/// Score a classification group from its `[C]` marker states.
///
/// `multi_label` selects sigmoid over softmax, matching the reference's
/// `class_act: "auto"` default. The reference raises for a non-positive
/// temperature rather than silently dividing.
pub fn classify_group(
    model: &BoundaryModel<'_>,
    task: &Task,
    choice_states: &[f32],
    multi_label: bool,
) -> Result<ClassificationResult, String> {
    let hidden_size = model.config.n_embd;
    if choice_states.len() != task.labels.len() * hidden_size {
        return Err(format!(
            "task {:?}: {} choice states for {} labels",
            task.name,
            choice_states.len() / hidden_size.max(1),
            task.labels.len()
        ));
    }
    let temperature = model.settings.classification_temperature;
    if temperature <= 0.0 {
        return Err("classification temperature must be > 0".into());
    }
    let mut logits = Vec::with_capacity(task.labels.len());
    for index in 0..task.labels.len() {
        let state = &choice_states[index * hidden_size..][..hidden_size];
        logits.push(classify_state(model, state)? / temperature);
    }
    let activation = if multi_label { "sigmoid" } else { "softmax" };
    let probabilities: Vec<f32> = if multi_label {
        logits
            .iter()
            .map(|logit| 1.0 / (1.0 + (-logit).exp()))
            .collect()
    } else {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|logit| (logit - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|value| value / sum).collect()
    };
    let labels: Vec<String> = task.labels.iter().map(|label| label.name.clone()).collect();
    let best = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index);
    // The reference thresholds at `cls_threshold`, defaulting to 0.5, and falls
    // back to the argmax when a multi-label group selects nothing.
    // `Task::cls_threshold` is a plain f32 that already defaults to 0.5.
    let threshold = task.cls_threshold;
    let selected: Vec<String> = if multi_label {
        let chosen: Vec<String> = labels
            .iter()
            .enumerate()
            .filter(|(index, _)| probabilities[*index] >= threshold)
            .map(|(_, label)| label.clone())
            .collect();
        if chosen.is_empty() {
            best.map(|index| labels[index].clone())
                .into_iter()
                .collect()
        } else {
            chosen
        }
    } else {
        Vec::new()
    };
    Ok(ClassificationResult {
        task: task.name.clone(),
        activation,
        logits,
        probabilities,
        choice_label: best.and_then(|index| labels.get(index).cloned()),
        labels,
        selected,
        multi_label,
    })
}

/// `count_head` / `null_projection`: one scalar per extractive query.
///
/// `null_logits` is the abstention gate — the reference drops a whole query's
/// spans when `sigmoid(null_logits[q]) > abstention_threshold`
/// (`engine.py:269`). `count_log_rates` feeds the adaptive-threshold path,
/// which base-v1 leaves off (`adaptive_threshold: false`), so it is reported
/// rather than applied.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryHeads {
    pub null_logits: Vec<f32>,
    pub count_log_rates: Vec<f32>,
}

/// Apply both scalar heads to `[B, Q, hidden]` query states.
pub fn query_heads(
    model: &BoundaryModel<'_>,
    query_states: &[f32],
    q_count: usize,
    hidden_size: usize,
) -> Result<QueryHeads, String> {
    let mut null_logits = Vec::with_capacity(q_count);
    let mut count_log_rates = Vec::with_capacity(q_count);
    for q in 0..q_count {
        let state = &query_states[q * hidden_size..][..hidden_size];
        null_logits.push(apply_row(model, "null_projection", state)?);
        count_log_rates.push(apply_row(model, "count_head", state)?);
    }
    Ok(QueryHeads {
        null_logits,
        count_log_rates,
    })
}

fn apply_row(model: &BoundaryModel<'_>, name: &str, state: &[f32]) -> Result<f32, String> {
    let (weight, bias) = match name {
        "null_projection" => (&model.null_projection, &model.null_projection_bias),
        "count_head" => (&model.count_head, &model.count_head_bias),
        other => return Err(format!("unknown scalar head {other}")),
    };
    let mut out = [0.0f32; 1];
    apply_linear_full(state, weight, bias, &mut out);
    Ok(out[0])
}
