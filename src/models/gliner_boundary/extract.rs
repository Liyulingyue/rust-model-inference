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

use crate::core::tensor::TensorSource;
use crate::models::gliner::compute;
use crate::models::gliner::prompt::{self, BoundaryTaskKind, EncodedPrompt, Task, C_TOKEN};
use crate::ops::kernel::Weight;

use super::loader::BoundaryModel;
use super::spans::{score_document_candidates, DocumentCandidateBatch};

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
    encode_mixed_boundary_prompt(model, tasks, &kinds, text)
}

/// Prompt assembly for a mix of group kinds. `kinds[i]` describes `tasks[i]`.
pub fn encode_mixed_boundary_prompt(
    model: &BoundaryModel<'_>,
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    text: &str,
) -> Result<EncodedPrompt, String> {
    match &model.tokenizer {
        crate::models::gliner::ModelTokenizer::SentencePiece(spm) => {
            prompt::build_mixed_boundary_prompt(tasks, kinds, text, spm)
        }
        crate::models::gliner::ModelTokenizer::Json(fast) => {
            prompt::build_mixed_boundary_prompt_with(tasks, kinds, text, |part| {
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
    /// `null_projection` / `count_head` per extractive query.
    pub query_heads: QueryHeads,
    /// The normalized, lowercased word list the spans index into.
    pub words: Vec<String>,
    /// Field name per extractive query.
    pub query_names: Vec<String>,
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
) -> Result<Extraction, String> {
    if tasks.is_empty() {
        return Err("extraction needs at least one schema task".into());
    }
    let encoded = encode_mixed_boundary_prompt(model, tasks, kinds, text)?;
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
            pool_size: model.settings.pool_size,
        },
        spans: Vec::new(),
        classifications: Vec::new(),
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
            extractions.classifications.push(classify_group(
                model,
                task,
                slice,
                task.multi_label,
                n_threads_arg,
            )?);
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
    let result = run_mixed_extraction(model, text, tasks, &kinds, n_threads_arg)?;
    Ok((result.candidates, result.words))
}

/// Threshold the candidate batch into spans.
///
/// `keep = valid_mask & (sigmoid(logit / pair_temperature) >= threshold)`, then
/// per field sorted by `(-score, start, end)` — `decode_candidates`' order.
/// Padded candidates carry `MASK_LOGIT`, so their probability is ~0 and the
/// threshold drops them; the `valid_mask` check is belt and braces.
pub fn decode_spans(
    batch: &DocumentCandidateBatch,
    words: &[String],
    field_names: &[String],
    pair_temperature: f32,
    threshold: f32,
) -> Vec<ExtractedSpan> {
    let temperature = if pair_temperature > 0.0 {
        pair_temperature
    } else {
        1.0
    };
    let q_count = field_names.len();
    let mut out = Vec::new();
    for q in 0..q_count {
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
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.start.cmp(&b.start))
                .then(a.end.cmp(&b.end))
        });
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
    Ok(decode_spans(
        &batch,
        &words,
        &fields,
        model.settings.pair_temperature,
        threshold.unwrap_or(0.5),
    ))
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
    n_threads_arg: usize,
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

fn apply_linear_full(input: &[f32], weight: &Weight<'_>, bias: &[f32], output: &mut [f32]) {
    if let Some(rows) = weight.kernel.f32_slice() {
        let n_in = input.len();
        let n_out = output.len();
        for (out_index, row) in rows.chunks_exact(n_in).take(n_out).enumerate() {
            output[out_index] = crate::ops::dot_f32(row, input, n_in) + bias[out_index];
        }
    } else {
        weight
            .kernel
            .forward(input, output, weight.n_in, weight.n_out);
        for (out, b) in output.iter_mut().zip(bias.iter()) {
            *out += *b;
        }
    }
}
