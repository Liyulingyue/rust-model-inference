//! `score_spans` — the top-level span-conditioned scoring API for
//! BoundaryExtractor.
//!
//! Chains the pieces that each got their own byte-exact oracle:
//!  1. `BoundaryEncoder.forward` — text → boundary states
//!  2. `BoundaryQueryHead.forward` — per-query marginals + inside prefix
//!     (mean-centered, with the mean carried separately)
//!  3. `BoundaryProposer::score_explicit_pairs` — marginal-free compat prior
//!  4. `PairScorer.forward` — per-candidate score
//!
//! [`score_document_candidates`] is the *other* entry point and the one
//! ordinary span extraction uses: `gliner2.5-base-v1` sets
//! `candidate_pool = "shared"`, so `BoundaryHead.forward` builds one
//! document-wide pool (`DocumentCandidatePool`) and scores it against every
//! query in one pass (`SharedPoolScorer`).
//!
//! This mirrors `BoundaryExtractor.score_explicit_spans`
//! (`boundary/model.py:274`), which is the span-conditioned entry point used
//! by the reference engine to score caller-supplied spans: the entity
//! classification step (`engine.py`, `choice_pairs = [(i, i + 1)]`), the
//! entity-attribute step, and joint-IE. For span extraction the pair score IS
//! the output — the caller thresholds or top-Ks the per-candidate logits.
//!
//! Two things are deliberately *not* here:
//!  - the `classifier.0` + `classifier.3` MLP, which consumes hidden-size
//!    `choice_states` / `group_embs` from the schema-prompt machinery
//!    (`models/gliner/`) rather than boundary states, and
//!  - the document-level candidate path. `gliner2.5-base-v1` sets
//!    `candidate_pool = "shared"`, so ordinary inference goes through
//!    `DocumentCandidatePool` + `SharedPoolScorer` (`boundary/pool.py`)
//!    rather than this pair scorer. See `glinerTODO.md`.

use super::loader::BoundaryModel;
use super::pair_scorer::PairScoreInputs;
use super::pool::SharedPoolInputs;

/// One scored candidate span.
#[derive(Clone, Debug)]
pub struct ScoredSpan {
    pub start: usize,
    pub end: usize,
    /// Pair-scorer logit for this (query, start, end) triple.
    pub logit: f32,
}

/// Score explicit `(start, end)` candidates for every query in the batch.
///
/// `indices` is `[B, Q, C, 2]` flat; `valid_mask` is `[B, Q, C]`. Invalid
/// candidates carry `MASK_LOGIT` from the pair scorer so downstream softmax /
/// top-K cannot pick them.
pub fn score_spans(
    model: &BoundaryModel<'_>,
    text_states: &[f32],
    text_mask: &[Vec<bool>],
    query_states: &[f32],
    query_mask: &[Vec<bool>],
    indices: &[usize],
    valid_mask: &[bool],
    batch: usize,
    q_count: usize,
    c: usize,
) -> Vec<ScoredSpan> {
    // 1. text → boundary states.
    let encoding = model.boundary.forward(text_states, text_mask);
    let boundary_len = text_mask.first().map_or(0, |row| row.len()) + 1;

    // 2. per-query marginals + inside evidence prefix.
    let marginals = model.query_head.forward(
        &encoding.states,
        &encoding.mask,
        text_states,
        text_mask,
        query_states,
        query_mask,
    );

    // 3. compatibility prior.
    let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    let compat = model.proposer.score_explicit_pairs(
        &encoding.states,
        boundary_len,
        query_states,
        batch,
        q_count,
        c,
        &indices_u32,
        valid_mask,
    );

    // 4. pair score.
    let text_lengths: Vec<usize> = text_mask
        .iter()
        .map(|row| row.iter().filter(|m| **m).count())
        .collect();
    let scores = model.pair_scorer.forward(&PairScoreInputs {
        boundary_states: &encoding.states,
        boundary_len,
        query_states,
        start_logits: &marginals.start_logits,
        end_logits: &marginals.end_logits,
        compat_logits: &compat,
        indices,
        valid_mask,
        inside_prefix: &marginals.inside_prefix,
        inside_prefix_mean: &marginals.inside_prefix_mean,
        text_states,
        text_mask,
        text_lengths: &text_lengths,
        batch,
        q_count,
        c,
    });

    (0..batch * q_count * c)
        .map(|idx| ScoredSpan {
            start: indices[idx * 2],
            end: indices[idx * 2 + 1],
            logit: scores[idx],
        })
        .collect()
}

/// One batch of document-level candidates, in the public per-query order.
///
/// `PooledCandidates::to_candidate_batch` (`pool.py:41`) is the reference's
/// adapter from its candidate-major internals to this shape; the fields here
/// are the result of that transpose.
pub struct DocumentCandidateBatch {
    /// `[B, Q, C, 2]` candidate `(start, end)` pairs, query-agnostic within a
    /// row of the batch.
    pub indices: Vec<usize>,
    /// `[B, Q, C]` final per-candidate logits. Invalid candidates carry
    /// `MASK_LOGIT`.
    pub pair_logits: Vec<f32>,
    /// `[B, Q, C]`. False for padding and for inactive queries.
    pub valid_mask: Vec<bool>,
    /// `[B, C, pair_dim]` contextual candidate states. Feeds the record head's
    /// `candidate_encoder`; `None` would be equivalent to
    /// `candidate_attention_layers = 0` plus no records.
    pub candidate_states: Vec<f32>,
    /// `C`, the padded pool width.
    pub pool_size: usize,
}

/// Score the shared document pool for every query.
///
/// This is the mainline path (`BoundaryHead.forward` with
/// `candidate_pool == "shared"`, `model.py:396-466`):
///  1. `BoundaryEncoder` — text → boundary states
///  2. `BoundaryQueryHead` — per-query marginals + inside prefix
///  3. `DocumentCandidatePool` — one deduplicated span pool per document
///  4. `SharedPoolScorer` — score the pool against all queries
///
/// `indices` / `valid_mask` / `pair_logits` are returned transposed to
/// `[B, Q, C]`; `candidate_states` stays `[B, C, pair_dim]` because the
/// reference keeps it candidate-major.
pub fn score_document_candidates(
    model: &BoundaryModel<'_>,
    text_states: &[f32],
    text_mask: &[Vec<bool>],
    query_states: &[f32],
    query_mask: &[Vec<bool>],
) -> DocumentCandidateBatch {
    let batch = text_mask.len();
    let q_count = query_mask.first().map_or(0, Vec::len);
    if batch == 0 || q_count == 0 {
        return DocumentCandidateBatch {
            indices: Vec::new(),
            pair_logits: Vec::new(),
            valid_mask: Vec::new(),
            candidate_states: Vec::new(),
            pool_size: model.settings.pool_size,
        };
    }

    let encoding = model.boundary.forward(text_states, text_mask);
    let marginals = model.query_head.forward(
        &encoding.states,
        &encoding.mask,
        text_states,
        text_mask,
        query_states,
        query_mask,
    );
    let pooled = model.pool_builder.build(
        &encoding.states,
        &encoding.mask,
        query_mask,
        &marginals.start_logits,
        &marginals.end_logits,
    );
    let text_lengths: Vec<usize> = text_mask
        .iter()
        .map(|row| row.iter().filter(|m| **m).count())
        .collect();
    let (pair_logits, candidate_states) = model.pool_scorer.forward(
        &SharedPoolInputs {
            boundary_states: &encoding.states,
            query_states,
            query_mask,
            inside_prefix: &marginals.inside_prefix,
            inside_prefix_mean: &marginals.inside_prefix_mean,
            text_states,
            text_mask,
            start_logits: &marginals.start_logits,
            end_logits: &marginals.end_logits,
            text_lengths: &text_lengths,
            q_count,
        },
        &pooled,
    );

    // `to_candidate_batch`: expand the query-agnostic pool across queries and
    // transpose the candidate-major scores.
    let c = pooled.pool_size;
    let mut indices = vec![0usize; batch * q_count * c * 2];
    let mut valid_mask = vec![false; batch * q_count * c];
    let mut transposed = vec![0.0f32; batch * q_count * c];
    for b in 0..batch {
        for q in 0..q_count {
            for slot in 0..c {
                let dst = b * q_count * c + q * c + slot;
                let src = b * c + slot;
                indices[dst * 2] = pooled.indices[src * 2];
                indices[dst * 2 + 1] = pooled.indices[src * 2 + 1];
                valid_mask[dst] = pooled.mask[src];
                transposed[dst] = pair_logits[src * q_count + q];
            }
        }
    }

    DocumentCandidateBatch {
        indices,
        pair_logits: transposed,
        valid_mask,
        candidate_states,
        pool_size: c,
    }
}
