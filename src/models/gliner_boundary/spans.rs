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
