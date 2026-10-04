//! `DocumentCandidatePool` — the shared, query-agnostic document span pool.
//!
//! Mirrors `gliner2.models.boundary.pool.DocumentCandidatePool`
//! (`target/gliner2-oracle/gliner2/models/boundary/pool.py:107`).
//!
//! This is the mainline. `gliner2.5-base-v1` sets `candidate_pool = "shared"`,
//! so `BoundaryHead.forward` (`model.py:396`) routes ordinary inference here
//! instead of through `SparseBoundaryProposer` / `SparseBoundaryPairScorer`.
//! The scorer we already ported serves a different entry point
//! (`score_explicit_spans`: entity classification, entity attributes, joint-IE).
//!
//! The pool is built once per document and shared by every query:
//!  1. Union the per-query start / end marginals with `amax` over queries, so
//!     the pool stays query-agnostic while remaining evidence-driven.
//!  2. Keep the top `pool_boundary_top_k` starts and ends (32 for base-v1,
//!     clamped down to `n_boundaries`).
//!  3. Pair them as a Cartesian product, dropping `end <= start`.
//!  4. Score each pair as `(start_proj * end_proj).sum(-1) / sqrt(boundary_dim)`
//!     plus both union marginals.
//!  5. Reserve each query's best `min_pool_per_query` pairs (8) in a score band
//!     above every global score, so one query cannot crowd out another.
//!  6. Deduplicate on `start * n + end`, keeping the best-scoring occurrence,
//!     then keep the top `pool_size` (192) by score.
//!
//! Steps 2 and 6 sort with a *stable* comparator, so a tie broken differently
//! yields a different pool and a wrong score that looks like a numeric bug
//! rather than a logic bug. `stable_argsort` below pins the tie-break to the
//! original position, matching `torch.sort(stable=True)`.

use crate::core::tensor::TensorSource;
use crate::models::gliner_boundary::tensor_util::{
    apply_linear_full, apply_linear_rows, load_vec, load_weight,
};
use crate::ops::kernel::Weight;

use super::content_pooler::SpanContentPooler;
use super::settings::BoundarySettings;

/// `MASK_LOGIT` from `boundary/constants.py`.
const MASK_LOGIT: f32 = -1.0e4;

/// A padded, deduplicated span pool for one batch.
///
/// `indices` / `mask` / `proposal_logits` / `compat_logits` are all
/// query-agnostic: the pool is built per document, not per query.
pub struct PooledCandidates {
    /// `[B, C, 2]` candidate `(start, end)`. Padded slots are `[0, 0]`.
    pub indices: Vec<usize>,
    /// `[B, C]`. False for padding.
    pub mask: Vec<bool>,
    /// `[B, C]` `compat + union_start + union_end`, or `MASK_LOGIT` when the
    /// slot is padding.
    pub proposal_logits: Vec<f32>,
    /// `[B, C]` the marginal-free `(start_proj * end_proj).sum / sqrt(d)` term,
    /// zeroed for padding. This is what `SharedPoolScorer` feeds
    /// `prior_projection`.
    pub compat_logits: Vec<f32>,
    /// `C`, the padded pool width (`pool_size`).
    pub pool_size: usize,
}

pub struct DocumentCandidatePool<'a> {
    pub boundary_dim: usize,
    pub pool_boundary_top_k: usize,
    pub pool_size: usize,
    pub min_pool_per_query: usize,
    pub start_projection: Weight<'a>,
    pub start_projection_bias: Vec<f32>,
    pub end_projection: Weight<'a>,
    pub end_projection_bias: Vec<f32>,
}

impl<'a> DocumentCandidatePool<'a> {
    pub fn load(
        source: &'a dyn TensorSource,
        boundary_dim: usize,
        pool_boundary_top_k: usize,
        pool_size: usize,
        min_pool_per_query: usize,
    ) -> Result<Self, String> {
        let start_projection = load_weight(
            source,
            "boundary_head.shared_pool_builder.start_projection.weight",
            boundary_dim,
            boundary_dim,
        )?;
        let start_projection_bias = load_vec(
            source,
            "boundary_head.shared_pool_builder.start_projection.bias",
            boundary_dim,
        )?;
        let end_projection = load_weight(
            source,
            "boundary_head.shared_pool_builder.end_projection.weight",
            boundary_dim,
            boundary_dim,
        )?;
        let end_projection_bias = load_vec(
            source,
            "boundary_head.shared_pool_builder.end_projection.bias",
            boundary_dim,
        )?;
        Ok(Self {
            boundary_dim,
            pool_boundary_top_k,
            pool_size,
            min_pool_per_query,
            start_projection,
            start_projection_bias,
            end_projection,
            end_projection_bias,
        })
    }

    /// Build the pool.
    ///
    ///  - `boundary_states` `[B, N, boundary_dim]`
    ///  - `boundary_mask` `[B][N]`
    ///  - `query_mask` `[B][Q]`
    ///  - `start_logits` / `end_logits` `[B, Q, N]`
    pub fn build(
        &self,
        boundary_states: &[f32],
        boundary_mask: &[Vec<bool>],
        query_mask: &[Vec<bool>],
        start_logits: &[f32],
        end_logits: &[f32],
    ) -> PooledCandidates {
        let batch = boundary_mask.len();
        let n = boundary_mask.first().map_or(0, Vec::len);
        let q = query_mask.first().map_or(0, Vec::len);
        let c = self.pool_size;
        let d = self.boundary_dim;
        if batch == 0 || n == 0 {
            return PooledCandidates {
                indices: Vec::new(),
                mask: Vec::new(),
                proposal_logits: Vec::new(),
                compat_logits: Vec::new(),
                pool_size: c,
            };
        }
        // `select_top_boundaries` clamps k to the boundary count.
        let k = self.pool_boundary_top_k.min(n);
        let pair_count = k * k;
        let quota = self.min_pool_per_query.min(pair_count);
        let scale = 1.0 / (d as f32).sqrt();

        let mut indices = vec![0usize; batch * c * 2];
        let mut mask = vec![false; batch * c];
        let mut proposal_logits = vec![0.0f32; batch * c];
        let mut compat_logits = vec![0.0f32; batch * c];

        for b in 0..batch {
            // Projected endpoints, shared by every pair in this document.
            let mut start_all = vec![0.0f32; n * d];
            let mut end_all = vec![0.0f32; n * d];
            apply_linear_rows(
                &boundary_states[b * n * d..][..n * d],
                &self.start_projection,
                &self.start_projection_bias,
                &mut start_all,
                d,
            );
            apply_linear_rows(
                &boundary_states[b * n * d..][..n * d],
                &self.end_projection,
                &self.end_projection_bias,
                &mut end_all,
                d,
            );

            let any_query = query_mask[b].iter().any(|m| *m);
            // 1. Query union of the marginals.
            let mut union_start = vec![MASK_LOGIT; n];
            let mut union_end = vec![MASK_LOGIT; n];
            let mut union_valid = vec![false; n];
            for j in 0..n {
                union_valid[j] = boundary_mask[b][j] && any_query;
                if !boundary_mask[b][j] {
                    continue;
                }
                for qi in 0..q {
                    if !query_mask[b][qi] {
                        continue;
                    }
                    let s = start_logits[b * q * n + qi * n + j];
                    let e = end_logits[b * q * n + qi * n + j];
                    if s > union_start[j] {
                        union_start[j] = s;
                    }
                    if e > union_end[j] {
                        union_end[j] = e;
                    }
                }
            }

            // 2. Top-k boundaries.
            let (starts, starts_valid) = select_top_boundaries(&union_start, &union_valid, k);
            let (ends, ends_valid) = select_top_boundaries(&union_end, &union_valid, k);

            // 3. Cartesian pairing: pair p = s_slot * k + e_slot.
            let mut pair_s = vec![0usize; pair_count];
            let mut pair_e = vec![0usize; pair_count];
            let mut pair_valid = vec![false; pair_count];
            let mut compat = vec![0.0f32; pair_count];
            let mut union_pair_score = vec![0.0f32; pair_count];
            for s_slot in 0..k {
                for e_slot in 0..k {
                    let p = s_slot * k + e_slot;
                    let s = starts[s_slot];
                    let e = ends[e_slot];
                    pair_s[p] = s;
                    pair_e[p] = e;
                    pair_valid[p] = starts_valid[s_slot] && ends_valid[e_slot] && e > s;
                    let mut dot = 0.0f32;
                    for dd in 0..d {
                        dot += start_all[s * d + dd] * end_all[e * d + dd];
                    }
                    let term = dot * scale;
                    compat[p] = term;
                    union_pair_score[p] =
                        term + union_start[s.min(n - 1)] + union_end[e.min(n - 1)];
                }
            }

            // 5. Per-query quota. Scores sit in a band above every global
            //    score, with a rank bonus so the reservation order survives
            //    deduplication: rank r gets `-MASK_LOGIT/2 + (quota - r)`.
            let quota_len = q * quota;
            let mut quota_keys = vec![0i64; quota_len];
            let mut quota_scores = vec![0.0f32; quota_len];
            let mut quota_valid = vec![false; quota_len];
            for qi in 0..q {
                let mut per_query = vec![0.0f32; pair_count];
                let mut per_query_valid = vec![false; pair_count];
                for p in 0..pair_count {
                    per_query_valid[p] = pair_valid[p] && query_mask[b][qi];
                    if per_query_valid[p] {
                        let s = pair_s[p].min(n - 1);
                        let e = pair_e[p].min(n - 1);
                        per_query[p] = start_logits[b * q * n + qi * n + s]
                            + end_logits[b * q * n + qi * n + e]
                            + compat[p];
                    } else {
                        per_query[p] = MASK_LOGIT;
                    }
                }
                let ranked = stable_argsort_desc(&per_query);
                for r in 0..quota {
                    let p = ranked[r];
                    let slot = qi * quota + r;
                    quota_valid[slot] = per_query_valid[p];
                    quota_keys[slot] =
                        (pair_s[p].min(n - 1) as i64) * n as i64 + pair_e[p].min(n - 1) as i64;
                    quota_scores[slot] = -MASK_LOGIT * 0.5 + (quota - r) as f32;
                }
            }

            // 6. Deduplicate + truncate.
            let mut all_keys = vec![0i64; quota_len + pair_count];
            let mut all_scores = vec![0.0f32; quota_len + pair_count];
            let mut all_valid = vec![false; quota_len + pair_count];
            all_keys[..quota_len].copy_from_slice(&quota_keys);
            all_scores[..quota_len].copy_from_slice(&quota_scores);
            all_valid[..quota_len].copy_from_slice(&quota_valid);
            for p in 0..pair_count {
                all_keys[quota_len + p] = pair_s[p] as i64 * n as i64 + pair_e[p] as i64;
                all_scores[quota_len + p] = union_pair_score[p];
                all_valid[quota_len + p] = pair_valid[p];
            }
            let (selected_keys, selected_valid) =
                deduplicate_pool(&all_keys, &all_scores, &all_valid, c, n);

            // 7. Recompute the differentiable scores for retained candidates.
            for slot in 0..c {
                let valid = selected_valid[slot];
                let key = if valid { selected_keys[slot] } else { 0 };
                let s = key / n as i64;
                let e = key - s * n as i64;
                indices[(b * c + slot) * 2] = s.max(0) as usize;
                indices[(b * c + slot) * 2 + 1] = e.max(0) as usize;
                mask[b * c + slot] = valid;
                if !valid {
                    // `selected_score.masked_fill(~selected_valid, MASK_LOGIT)`
                    // and `selected_compat = where(valid, ..., 0)`.
                    proposal_logits[b * c + slot] = MASK_LOGIT;
                    continue;
                }
                let s = indices[(b * c + slot) * 2];
                let e = indices[(b * c + slot) * 2 + 1];
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += start_all[s * d + dd] * end_all[e * d + dd];
                }
                let term = dot * scale;
                compat_logits[b * c + slot] = term;
                proposal_logits[b * c + slot] =
                    term + union_start[s.min(n - 1)] + union_end[e.min(n - 1)];
            }
        }

        PooledCandidates {
            indices,
            mask,
            proposal_logits,
            compat_logits,
            pool_size: c,
        }
    }
}

// ---------------------------------------------------------------------------
// SharedPoolScorer
// ---------------------------------------------------------------------------

/// Inputs for [`SharedPoolScorer::forward`].
pub struct SharedPoolInputs<'t> {
    /// `[B, N, boundary_dim]`.
    pub boundary_states: &'t [f32],
    /// `[B, Q, query_dim]`.
    pub query_states: &'t [f32],
    /// `[B][Q]`.
    pub query_mask: &'t [Vec<bool>],
    /// `[B, L+1]` mean-centered inside prefix; only read when
    /// `use_inside_evidence` is on.
    pub inside_prefix: &'t [f32],
    /// `[B, Q]` the mean subtracted before the cumulative sum.
    pub inside_prefix_mean: &'t [f32],
    /// `[B, L, hidden]` encoder token states; only read when span content is on.
    pub text_states: &'t [f32],
    /// `[B][L]` encoder token mask. Also fixes `L` for the content prefix.
    pub text_mask: &'t [Vec<bool>],
    /// `[B, Q, N]` start marginals.
    pub start_logits: &'t [f32],
    /// `[B, Q, N]` end marginals.
    pub end_logits: &'t [f32],
    /// `[B]` real token counts.
    pub text_lengths: &'t [usize],
    /// `Q`, the number of queries.
    pub q_count: usize,
}

/// `SharedPoolScorer` (`pool.py:446`): score the whole document pool against
/// every query in one pass.
///
/// Unlike `SparseBoundaryPairScorer`, which builds a per-query score by adding
/// independent scalar terms, this scores each candidate once as a
/// `pair_dim` vector and then dot-products it with a per-query vector. The
/// marginals and inside evidence are still added afterwards, per query.
///
/// The two optional attention stacks are not implemented: base-v1 sets
/// `candidate_attention_layers = 0` and `query_attention_layers = 0`, so it
/// never instantiates them. `load` refuses a configuration that does, rather
/// than quietly skipping layers whose weights are in the checkpoint.
pub struct SharedPoolScorer<'a> {
    pub boundary_dim: usize,
    pub pair_dim: usize,
    pub query_dim: usize,
    pub content_dim: usize,
    pub start_projection: Weight<'a>,
    pub start_projection_bias: Vec<f32>,
    pub end_projection: Weight<'a>,
    pub end_projection_bias: Vec<f32>,
    /// `nn.Linear(3, pair_dim)`.
    pub length_projection: Weight<'a>,
    pub length_projection_bias: Vec<f32>,
    /// `nn.Linear(1, pair_dim)`.
    pub prior_projection: Weight<'a>,
    pub prior_projection_bias: Vec<f32>,
    pub content_pooler: Option<SpanContentPooler<'a>>,
    /// `nn.Linear(content_dim, pair_dim)`.
    pub content_projection: Option<(Weight<'a>, Vec<f32>)>,
    pub candidate_norm_weight: Vec<f32>,
    pub candidate_norm_bias: Vec<f32>,
    /// `nn.Linear(query_dim, pair_dim)`.
    pub query_projection: Weight<'a>,
    pub query_projection_bias: Vec<f32>,
    /// `nn.Linear(pair_dim, 2 * pair_dim)`, split into gamma and beta.
    pub film: Weight<'a>,
    pub film_bias: Vec<f32>,
    /// `nn.Linear(pair_dim, 64)` — the FiLM-conditioned MLP's first layer.
    pub film_hidden: Weight<'a>,
    pub film_hidden_bias: Vec<f32>,
    /// `nn.Linear(64, 1)` — its output layer.
    pub film_output: Weight<'a>,
    pub film_output_bias: Vec<f32>,
    use_inside_evidence: bool,
}

impl<'a> SharedPoolScorer<'a> {
    pub fn load(
        source: &'a dyn TensorSource,
        hidden_size: usize,
        settings: &BoundarySettings,
    ) -> Result<Self, String> {
        if settings.candidate_attention_layers != 0 || settings.query_attention_layers != 0 {
            return Err(format!(
                "candidate_attention_layers = {} and query_attention_layers = {} are not \
                 implemented; base-v1 sets both to 0. A checkpoint that enables them needs \
                 OverlapBiasedCandidateAttention (pool.py:371) and \
                 EvidenceConditionedQueryAttention (pool.py:410) first.",
                settings.candidate_attention_layers, settings.query_attention_layers
            ));
        }
        let d = settings.boundary_dim;
        let pair = settings.pair_dim;
        let prefix = "boundary_head.shared_pool_scorer";
        let load =
            |name: &str, n_in, n_out| load_weight(source, &format!("{prefix}.{name}"), n_in, n_out);
        let bias = |name: &str, len| load_vec(source, &format!("{prefix}.{name}"), len);

        let content_pooler = if settings.enable_span_content {
            Some(SpanContentPooler::load(
                source,
                &format!("{prefix}.content_pooler"),
                hidden_size,
                settings.content_dim,
                settings.content_soft_max_pool,
            )?)
        } else {
            None
        };
        let content_dim = content_pooler.as_ref().map_or(0, |p| p.output_dim);
        let content_projection = if settings.enable_span_content {
            Some((
                load("content_projection.weight", content_dim, pair)?,
                bias("content_projection.bias", pair)?,
            ))
        } else {
            None
        };

        Ok(Self {
            boundary_dim: d,
            pair_dim: pair,
            query_dim: query_width(source, prefix)?,
            content_dim,
            start_projection: load("start_projection.weight", d, pair)?,
            start_projection_bias: bias("start_projection.bias", pair)?,
            end_projection: load("end_projection.weight", d, pair)?,
            end_projection_bias: bias("end_projection.bias", pair)?,
            length_projection: load("length_projection.weight", 3, pair)?,
            length_projection_bias: bias("length_projection.bias", pair)?,
            prior_projection: load("prior_projection.weight", 1, pair)?,
            prior_projection_bias: bias("prior_projection.bias", pair)?,
            content_pooler,
            content_projection,
            candidate_norm_weight: bias("candidate_norm.weight", pair)?,
            candidate_norm_bias: bias("candidate_norm.bias", pair)?,
            query_projection: load("query_projection.weight", hidden_size, pair)?,
            query_projection_bias: bias("query_projection.bias", pair)?,
            film: load("film.weight", pair, 2 * pair)?,
            film_bias: bias("film.bias", 2 * pair)?,
            film_hidden: load("film_output.0.weight", pair, 64)?,
            film_hidden_bias: bias("film_output.0.bias", 64)?,
            film_output: load("film_output.3.weight", 64, 1)?,
            film_output_bias: bias("film_output.3.bias", 1)?,
            use_inside_evidence: settings.use_inside_evidence,
        })
    }

    /// Score the pool. Returns `(pair_logits, candidate_states)` in
    /// `[B, C, Q]` and `[B, C, pair_dim]` order — the reference's internal
    /// candidate-major order, which `PooledCandidates::to_candidate_batch`
    /// transposes to the public `[B, Q, C]` contract.
    pub fn forward(
        &self,
        input: &SharedPoolInputs<'_>,
        pooled: &PooledCandidates,
    ) -> (Vec<f32>, Vec<f32>) {
        let SharedPoolInputs {
            boundary_states,
            query_states,
            query_mask,
            inside_prefix,
            inside_prefix_mean,
            text_states,
            text_mask,
            start_logits,
            end_logits,
            text_lengths,
            q_count,
        } = *input;
        let batch = query_mask.len();
        let q = q_count;
        if batch == 0 || q == 0 || pooled.pool_size == 0 {
            return (Vec::new(), Vec::new());
        }
        let c = pooled.pool_size;
        let d = self.boundary_dim;
        let pair = self.pair_dim;
        let query_dim = self.query_dim;
        let n = boundary_states.len() / batch / d.max(1);
        let seq_len = text_mask.first().map_or(0, Vec::len);
        let content_scale = 1.0 / (pair as f32).sqrt();
        // Both survive the per-document loop: the reference returns the
        // candidate states alongside the scores.
        let mut candidate = vec![0.0f32; batch * c * pair];
        let mut score = vec![0.0f32; batch * c * q];

        for b in 0..batch {
            // start / end representations for the retained candidates.
            let mut start_all = vec![0.0f32; n * pair];
            apply_linear_rows(
                &boundary_states[b * n * d..][..n * d],
                &self.start_projection,
                &self.start_projection_bias,
                &mut start_all,
                pair,
            );
            let mut end_all = vec![0.0f32; n * pair];
            apply_linear_rows(
                &boundary_states[b * n * d..][..n * d],
                &self.end_projection,
                &self.end_projection_bias,
                &mut end_all,
                pair,
            );
            // Gather both at the candidate endpoints.
            let mut start_rep = vec![0.0f32; c * pair];
            let mut end_rep = vec![0.0f32; c * pair];
            for slot in 0..c {
                let s = pooled.indices[(b * c + slot) * 2].min(n - 1);
                let e = pooled.indices[(b * c + slot) * 2 + 1].min(n - 1);
                start_rep[slot * pair..][..pair].copy_from_slice(&start_all[s * pair..][..pair]);
                end_rep[slot * pair..][..pair].copy_from_slice(&end_all[e * pair..][..pair]);
            }

            // Length features + learned projections, then the prior.
            let tl = text_lengths.get(b).copied().unwrap_or(1).max(1) as f32;
            let mut length_features = [0.0f32; 3];
            let mut prior_row = [0.0f32; 1];
            let mut projected = vec![0.0f32; pair];
            let mut prior = vec![0.0f32; pair];
            for slot in 0..c {
                let s = pooled.indices[(b * c + slot) * 2];
                let e = pooled.indices[(b * c + slot) * 2 + 1];
                let length = (e as f32 - s as f32).max(1.0);
                length_features[0] = (length + 1.0).ln();
                length_features[1] = length / tl;
                length_features[2] = 1.0 / length.sqrt();
                prior_row[0] = pooled.compat_logits[b * c + slot];
                apply_linear_full(
                    &length_features,
                    &self.length_projection,
                    &self.length_projection_bias,
                    &mut projected,
                );
                apply_linear_full(
                    &prior_row,
                    &self.prior_projection,
                    &self.prior_projection_bias,
                    &mut prior,
                );
                let base = b * c * pair + slot * pair;
                for dd in 0..pair {
                    candidate[base + dd] = start_rep[slot * pair + dd]
                        + end_rep[slot * pair + dd]
                        + projected[dd]
                        + prior[dd];
                }
            }

            // Span content.
            if let (Some(pooler), Some((content_projection, content_bias))) =
                (&self.content_pooler, &self.content_projection)
            {
                let width = self.content_dim;
                let flat_mask: Vec<bool> = text_mask.iter().flatten().copied().collect();
                let mean_prefix = pooler.build_prefix(text_states, &flat_mask, batch, seq_len);
                let starts: Vec<usize> =
                    (0..batch * c).map(|idx| pooled.indices[idx * 2]).collect();
                let ends: Vec<usize> = (0..batch * c)
                    .map(|idx| pooled.indices[idx * 2 + 1])
                    .collect();
                let content = pooler.pool(&mean_prefix, &starts, &ends, c, seq_len);
                for slot in 0..c {
                    apply_linear_full(
                        &content[slot * width..][..width],
                        content_projection,
                        content_bias,
                        &mut projected,
                    );
                    let base = b * c * pair + slot * pair;
                    for dd in 0..pair {
                        candidate[base + dd] += projected[dd];
                    }
                }
            }

            // LayerNorm, then zero the padding slots.
            for slot in 0..c {
                let base = b * c * pair + slot * pair;
                let mut source = vec![0.0f32; pair];
                source.copy_from_slice(&candidate[base..][..pair]);
                let mut normalized = vec![0.0f32; pair];
                crate::ops::layer_norm(
                    &source,
                    &self.candidate_norm_weight,
                    &self.candidate_norm_bias,
                    1e-5,
                    &mut normalized,
                );
                let keep = pooled.mask[b * c + slot];
                for dd in 0..pair {
                    candidate[base + dd] = if keep { normalized[dd] } else { 0.0 };
                }
            }

            // Per-query vectors, then the candidate-query dot products.
            let mut query_all = vec![0.0f32; q * pair];
            apply_linear_rows(
                &query_states[b * q * query_dim..][..q * query_dim],
                &self.query_projection,
                &self.query_projection_bias,
                &mut query_all,
                pair,
            );
            let mut film_all = vec![0.0f32; q * 2 * pair];
            apply_linear_rows(
                &query_all,
                &self.film,
                &self.film_bias,
                &mut film_all,
                2 * pair,
            );

            let mut hidden = vec![0.0f32; 64];
            for qi in 0..q {
                let qrow = &query_all[qi * pair..][..pair];
                // `film(query).chunk(2, -1)`: gamma is the first `pair_dim`
                // slice of each row, beta the second. The row stride is
                // `2 * pair_dim`, not `3 * pair_dim`.
                let gamma = &film_all[qi * 2 * pair..][..pair];
                let beta = &film_all[qi * 2 * pair + pair..][..pair];
                for slot in 0..c {
                    let cand = &candidate[b * c * pair + slot * pair..][..pair];
                    let mut dot = 0.0f32;
                    let mut conditioned = vec![0.0f32; pair];
                    for dd in 0..pair {
                        dot += cand[dd] * qrow[dd];
                        conditioned[dd] = cand[dd] * (1.0 + gamma[dd]) + beta[dd];
                    }
                    let base = dot * content_scale;
                    apply_linear_full(
                        &conditioned,
                        &self.film_hidden,
                        &self.film_hidden_bias,
                        &mut hidden,
                    );
                    for v in hidden.iter_mut() {
                        *v = crate::ops::gelu_erf(*v);
                    }
                    let mut out = [0.0f32; 1];
                    apply_linear_full(&hidden, &self.film_output, &self.film_output_bias, &mut out);
                    score[b * c * q + slot * q + qi] = base + out[0];
                }
            }

            // Marginals, then inside evidence.
            for qi in 0..q {
                for slot in 0..c {
                    let s = pooled.indices[(b * c + slot) * 2].min(n - 1);
                    let e = pooled.indices[(b * c + slot) * 2 + 1].min(n - 1);
                    score[b * c * q + slot * q + qi] +=
                        start_logits[b * q * n + qi * n + s] + end_logits[b * q * n + qi * n + e];
                }
            }
            if self.use_inside_evidence && !inside_prefix.is_empty() {
                for qi in 0..q {
                    let prefix_base = b * q * (seq_len + 1) + qi * (seq_len + 1);
                    let mean = inside_prefix_mean.get(b * q + qi).copied().unwrap_or(0.0);
                    for slot in 0..c {
                        let s = pooled.indices[(b * c + slot) * 2].min(seq_len);
                        let e = pooled.indices[(b * c + slot) * 2 + 1].min(seq_len);
                        let width = e as f32 - s as f32;
                        let interval = inside_prefix[prefix_base + e]
                            - inside_prefix[prefix_base + s]
                            + mean * width;
                        score[b * c * q + slot * q + qi] += interval / width.max(1.0).sqrt();
                    }
                }
            }

            // Padding slots and inactive queries.
            for qi in 0..q {
                for slot in 0..c {
                    if !pooled.mask[b * c + slot] || !query_mask[b][qi] {
                        score[b * c * q + slot * q + qi] = MASK_LOGIT;
                    }
                }
            }
        }

        (score, candidate)
    }
}

fn query_width(source: &dyn TensorSource, prefix: &str) -> Result<usize, String> {
    source
        .tensor_info(&format!("{prefix}.query_projection.weight"))
        .map(|info| info.dims[0] as usize)
        .ok_or_else(|| format!("missing tensor {prefix}.query_projection.weight"))
}

// ---------------------------------------------------------------------------
// Selection helpers
// ---------------------------------------------------------------------------

/// `select_top_boundaries` (`proposal.py:82`): stable descending top-k with
/// invalid positions floored to `MASK_LOGIT`. Returns `(indices, valid)`.
fn select_top_boundaries(logits: &[f32], valid_mask: &[bool], k: usize) -> (Vec<usize>, Vec<bool>) {
    let n = logits.len();
    let floored: Vec<f32> = (0..n)
        .map(|i| if valid_mask[i] { logits[i] } else { MASK_LOGIT })
        .collect();
    let order = stable_argsort_desc(&floored);
    let mut indices = vec![0usize; k];
    let mut valid = vec![false; k];
    for (rank, slot) in order.iter().take(k).enumerate() {
        valid[rank] = valid_mask[*slot];
        indices[rank] = if valid[rank] { *slot } else { 0 };
    }
    (indices, valid)
}

/// `_deduplicate_pool` (`pool.py:70`): keep the highest-priority occurrence of
/// each key, then the top `capacity` by score.
///
/// The two stable sorts are the whole point. Invalid entries are pushed to key
/// `n * n` and score `MASK_LOGIT` *before* the first sort, so an invalid entry
/// can never outrank a real one; the key sort then groups duplicates while
/// preserving score order inside a group, and `first` keeps the head of each
/// group. `capacity` may exceed the number of survivors, in which case the tail
/// is padded with `(key = 0, valid = false)`.
fn deduplicate_pool(
    keys: &[i64],
    scores: &[f32],
    valid: &[bool],
    capacity: usize,
    n_boundaries: usize,
) -> (Vec<i64>, Vec<bool>) {
    let invalid_key = (n_boundaries * n_boundaries) as i64;
    let mut keys: Vec<i64> = (0..keys.len())
        .map(|i| if valid[i] { keys[i] } else { invalid_key })
        .collect();
    let mut scores: Vec<f32> = (0..scores.len())
        .map(|i| if valid[i] { scores[i] } else { MASK_LOGIT })
        .collect();
    let mut valid = valid.to_vec();

    let by_score = stable_argsort_desc(&scores);
    permute(&mut keys, &by_score);
    permute(&mut scores, &by_score);
    permute(&mut valid, &by_score);

    let by_key = stable_argsort_asc_i64(&keys);
    permute(&mut keys, &by_key);
    permute(&mut scores, &by_key);
    permute(&mut valid, &by_key);

    let mut keep = vec![false; valid.len()];
    for i in 0..valid.len() {
        let first = i == 0 || keys[i] != keys[i - 1];
        keep[i] = valid[i] && first;
    }
    let ranked: Vec<f32> = (0..scores.len())
        .map(|i| if keep[i] { scores[i] } else { MASK_LOGIT })
        .collect();
    let order = stable_argsort_desc(&ranked);

    let mut selected_keys = vec![0i64; capacity];
    let mut selected_valid = vec![false; capacity];
    for (slot, source) in order.iter().take(capacity).enumerate() {
        selected_keys[slot] = keys[*source];
        selected_valid[slot] = keep[*source];
    }
    (selected_keys, selected_valid)
}

fn permute<T: Clone>(values: &mut [T], order: &[usize]) {
    let source = values.to_vec();
    for (target, from) in values.iter_mut().zip(order) {
        *target = source[*from].clone();
    }
}

/// `torch.sort(descending=True, stable=True)`'s index permutation. Stability
/// means equal values keep their original relative order, which is the same as
/// ordering by `(value desc, position asc)`.
fn stable_argsort_desc(values: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| f32_order(values[b], values[a]).then(a.cmp(&b)));
    order
}

fn stable_argsort_asc_i64(values: &[i64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].cmp(&values[b]).then(a.cmp(&b)));
    order
}

fn f32_order(a: f32, b: f32) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------
