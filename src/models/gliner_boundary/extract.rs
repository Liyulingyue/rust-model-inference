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
use crate::models::gliner::prompt::{self, EncodedPrompt, Task};

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
    match &model.tokenizer {
        crate::models::gliner::ModelTokenizer::SentencePiece(spm) => {
            prompt::build_boundary_prompt(tasks, text, child_marker, spm)
        }
        crate::models::gliner::ModelTokenizer::Json(fast) => {
            prompt::build_boundary_prompt_with(tasks, text, child_marker, |part| {
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

/// Run the whole pipeline and return the candidate batch plus the word list.
///
/// Split out from [`extract_spans`] so a caller that wants raw logits (or a
/// different threshold policy, or the candidate states) can stop here without
/// re-running the encoder.
pub fn run_extraction(
    model: &BoundaryModel<'_>,
    text: &str,
    tasks: &[Task],
    child_marker: &str,
    n_threads_arg: usize,
) -> Result<(DocumentCandidateBatch, Vec<String>), String> {
    if tasks.is_empty() {
        return Err("extraction needs at least one schema task".into());
    }
    let encoded = encode_boundary_prompt(model, tasks, text, child_marker)?;
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
    if encoded.query_positions.is_empty() {
        return Err("schema produced no query markers".into());
    }

    let text_states = gather_states(&hidden, &encoded.text_word_first_positions, hidden_size);
    let query_states = gather_states(&hidden, &encoded.query_positions, hidden_size);
    let text_mask = vec![vec![true; encoded.words.len()]];
    let query_mask = vec![vec![true; encoded.query_names.len()]];
    let batch =
        score_document_candidates(model, &text_states, &text_mask, &query_states, &query_mask);
    Ok((batch, encoded.words))
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
