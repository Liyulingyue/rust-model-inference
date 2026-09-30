//! Final classifier pass + top-level `score_spans` API for
//! BoundaryExtractor.
//!
//! Chains the pieces that each got their own byte-exact oracle:
//!  1. `BoundaryEncoder.forward`               — text → boundary states
//!  2. `BoundaryQueryHead.forward`             — per-query marginals
//!  3. `BoundaryProposer::score_explicit_pairs` — compatibility prior
//!  4. `PairScorer.forward`                    — per-candidate score
//!
//! For span extraction the pair score IS the final output — the caller
//! thresholds / top-Ks the per-candidate logits. The `classifier.0` +
//! `classifier.3` MLP belongs to the *classification* head (label-set
//! scoring), which shares the classifier with SpanExtractor and is
//! driven by the schema-prompt machinery (`prompt::encode_token` in
//! `models/gliner/`). That path is wired in the CLI/HTTP turn; this
//! module exposes the span-extraction surface.
//!
//! Known limitation: this chain uses the LIMITED pair scorer (no span
//! content, no inside evidence, no endpoint difference) — see
//! `pair_scorer.rs`. Gliner2.5-base-v1's published config enables all
//! three, so the span scores are missing those contributions until the
//! SpanContentPooler lands. Tracked in glinerTODO.md Phase 5.2.3.

use super::loader::BoundaryModel;

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
/// `indices` is `[B, Q, C, 2]` flat; returns `[B, Q, C]` final logits.
/// Invalid candidates (`valid_mask[b][q][c] == false`) carry
/// `MASK_LOGIT` from the pair scorer so downstream softmax / top-K
/// can't pick them.
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

    // 2. per-query marginals.
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

    // 4. pair score (limited: no span content / inside evidence /
    //    endpoint difference).
    let text_lengths: Vec<usize> = text_mask
        .iter()
        .map(|row| row.iter().filter(|m| **m).count())
        .collect();
    let pair_scores = model.pair_scorer.forward(
        &encoding.states,
        boundary_len,
        query_states,
        batch,
        q_count,
        c,
        &marginals.start_logits,
        &marginals.end_logits,
        &compat,
        indices,
        &text_lengths,
        valid_mask,
    );

    (0..batch * q_count * c)
        .map(|idx| ScoredSpan {
            start: indices[idx * 2],
            end: indices[idx * 2 + 1],
            logit: pair_scores[idx],
        })
        .collect()
}
