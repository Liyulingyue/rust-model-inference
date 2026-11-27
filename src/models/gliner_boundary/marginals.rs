//! `BoundaryQueryHead` — per-query marginals over boundary positions.
//!
//! Mirrors `gliner2.models.boundary.heads.BoundaryQueryHead.forward`
//! (`target/gliner2-oracle/gliner2/models/boundary/heads.py`). Produces:
//!  - per-(B, Q, L+1) start and end logits via dot product of projected
//!    boundary states and projected query states, scaled by 1/sqrt(d),
//!  - per-(B, Q, L) inside logits over text tokens (not boundaries),
//!  - cumulative-sum prefix for O(1) inside scores over any [i, j).
//!
//! Mask invalid positions with a finite sentinel (`-1e9` in the reference)
//! so downstream softmax / select-top-K does not NaN. Masked positions
//! contribute zero to the inside prefix so range sums are correct.
//!
//! Byte-exact oracle: see `tools/oracle/gliner_boundary/
//! dump_boundary_query_head.py` and
//! `tests/gliner2_5_base_v1_boundary_query_head_parity.rs`.

use crate::core::tensor::TensorSource;
use crate::models::gliner_boundary::tensor_util::{apply_linear_full, load_vec, load_weight};
use crate::ops::kernel::Weight;

/// Per-(B, Q) marginals + cumulative inside prefix.
#[derive(Clone)]
pub struct BoundaryMarginals<'a> {
    /// `[B, Q, L + 1]` per-query start logits over boundaries.
    pub start_logits: Vec<f32>,
    /// `[B, Q, L + 1]` per-query end logits over boundaries.
    pub end_logits: Vec<f32>,
    /// `[B, Q, L + 1]` cumulative sum of mean-centered inside logits.
    /// `prefix[j] - prefix[i]` is the centered sum over `[i, j)`.
    pub inside_prefix: Vec<f32>,
    /// `[B, Q]` mean that was subtracted before the cumulative sum. The
    /// reference carries it separately and restores it in interval scoring
    /// (``scoring.interval_prefix_score``'s ``mean`` argument), which
    /// recovers the raw interval sum without letting the running cumsum
    /// drift on long sequences.
    pub inside_prefix_mean: Vec<f32>,
    _marker: std::marker::PhantomData<&'a ()>,
}

/// `BoundaryQueryHead` weights. Five projections, each (in_dim ->
/// boundary_dim) plus bias.
pub struct BoundaryQueryHead<'a> {
    pub boundary_dim: usize,
    pub start_boundary: Weight<'a>,
    pub start_boundary_bias: Vec<f32>,
    pub start_query: Weight<'a>,
    pub start_query_bias: Vec<f32>,
    pub end_boundary: Weight<'a>,
    pub end_boundary_bias: Vec<f32>,
    pub end_query: Weight<'a>,
    pub end_query_bias: Vec<f32>,
    pub inside_text: Weight<'a>,
    pub inside_text_bias: Vec<f32>,
    pub inside_query: Weight<'a>,
    pub inside_query_bias: Vec<f32>,
}

impl<'a> BoundaryQueryHead<'a> {
    pub fn load(source: &'a dyn TensorSource, hidden_size: usize) -> Result<Self, String> {
        // boundary_dim is read from the start_boundary output channel.
        let boundary_dim = source
            .tensor_info("boundary_head.boundary_query_head.start_boundary_projection.weight")
            .ok_or("missing boundary_head.boundary_query_head.start_boundary_projection.weight")?
            .dims[0] as usize;
        // q_dim is read from the start_query input channel.
        let q_dim = source
            .tensor_info("boundary_head.boundary_query_head.start_query_projection.weight")
            .ok_or("missing boundary_head.boundary_query_head.start_query_projection.weight")?
            .dims[0] as usize;

        let start_boundary = load_weight(
            source,
            "boundary_head.boundary_query_head.start_boundary_projection.weight",
            boundary_dim,
            boundary_dim,
        )?;
        let start_boundary_bias = load_vec(
            source,
            "boundary_head.boundary_query_head.start_boundary_projection.bias",
            boundary_dim,
        )?;
        let start_query = load_weight(
            source,
            "boundary_head.boundary_query_head.start_query_projection.weight",
            q_dim,
            boundary_dim,
        )?;
        let start_query_bias = load_vec(
            source,
            "boundary_head.boundary_query_head.start_query_projection.bias",
            boundary_dim,
        )?;
        let end_boundary = load_weight(
            source,
            "boundary_head.boundary_query_head.end_boundary_projection.weight",
            boundary_dim,
            boundary_dim,
        )?;
        let end_boundary_bias = load_vec(
            source,
            "boundary_head.boundary_query_head.end_boundary_projection.bias",
            boundary_dim,
        )?;
        let end_query = load_weight(
            source,
            "boundary_head.boundary_query_head.end_query_projection.weight",
            q_dim,
            boundary_dim,
        )?;
        let end_query_bias = load_vec(
            source,
            "boundary_head.boundary_query_head.end_query_projection.bias",
            boundary_dim,
        )?;
        let inside_text = load_weight(
            source,
            "boundary_head.boundary_query_head.inside_text_projection.weight",
            hidden_size,
            boundary_dim,
        )?;
        let inside_text_bias = load_vec(
            source,
            "boundary_head.boundary_query_head.inside_text_projection.bias",
            boundary_dim,
        )?;
        let inside_query = load_weight(
            source,
            "boundary_head.boundary_query_head.inside_query_projection.weight",
            q_dim,
            boundary_dim,
        )?;
        let inside_query_bias = load_vec(
            source,
            "boundary_head.boundary_query_head.inside_query_projection.bias",
            boundary_dim,
        )?;

        Ok(Self {
            boundary_dim,
            start_boundary,
            start_boundary_bias,
            start_query,
            start_query_bias,
            end_boundary,
            end_boundary_bias,
            end_query,
            end_query_bias,
            inside_text,
            inside_text_bias,
            inside_query,
            inside_query_bias,
        })
    }

    /// Forward pass. Inputs:
    ///  - `boundary_states` `[B, L+1, boundary_dim]`
    ///  - `boundary_mask` `[B][L+1]` (each row is a `Vec<bool>`)
    ///  - `text_states` `[B, L, q_dim]` — the head reads text_states as
    ///    if they were queries (both feed the same-size projection)
    ///  - `text_mask` `[B][L]`
    ///  - `query_states` `[B, Q, q_dim]`
    ///  - `query_mask` `[B][Q]`
    pub fn forward(
        &self,
        boundary_states: &[f32],
        boundary_mask: &[Vec<bool>],
        text_states: &[f32],
        text_mask: &[Vec<bool>],
        query_states: &[f32],
        query_mask: &[Vec<bool>],
    ) -> BoundaryMarginals<'a> {
        let batch = boundary_mask.len();
        let boundary_len = if batch == 0 {
            0
        } else {
            boundary_mask[0].len()
        };
        let seq_len = if batch == 0 { 0 } else { text_mask[0].len() };
        let q_count = if batch == 0 { 0 } else { query_mask[0].len() };
        let q_dim = if batch * q_count == 0 {
            0
        } else {
            query_states.len() / (batch * q_count)
        };
        let boundary_dim = self.boundary_dim;
        let scale = 1.0 / (boundary_dim as f32).sqrt();

        // 1. Project boundary + query + query (start/end/inside).
        let mut start_b = vec![0.0f32; batch * boundary_len * boundary_dim];
        let mut end_b = vec![0.0f32; batch * boundary_len * boundary_dim];
        let mut start_q = vec![0.0f32; batch * q_count * boundary_dim];
        let mut end_q = vec![0.0f32; batch * q_count * boundary_dim];
        let mut inside_t = vec![0.0f32; batch * seq_len * boundary_dim];
        let mut inside_q = vec![0.0f32; batch * q_count * boundary_dim];
        for b in 0..batch {
            for i in 0..boundary_len {
                apply_linear_full(
                    &boundary_states[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim],
                    &self.start_boundary,
                    &self.start_boundary_bias,
                    &mut start_b[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim],
                );
                apply_linear_full(
                    &boundary_states[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim],
                    &self.end_boundary,
                    &self.end_boundary_bias,
                    &mut end_b[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim],
                );
            }
            for i in 0..seq_len {
                apply_linear_full(
                    &text_states[b * seq_len * q_dim + i * q_dim..][..q_dim],
                    &self.inside_text,
                    &self.inside_text_bias,
                    &mut inside_t[b * seq_len * boundary_dim + i * boundary_dim..][..boundary_dim],
                );
            }
            for q in 0..q_count {
                let q_row = &query_states[b * q_count * q_dim + q * q_dim..][..q_dim];
                apply_linear_full(
                    q_row,
                    &self.start_query,
                    &self.start_query_bias,
                    &mut start_q[b * q_count * boundary_dim + q * boundary_dim..][..boundary_dim],
                );
                apply_linear_full(
                    q_row,
                    &self.end_query,
                    &self.end_query_bias,
                    &mut end_q[b * q_count * boundary_dim + q * boundary_dim..][..boundary_dim],
                );
                apply_linear_full(
                    q_row,
                    &self.inside_query,
                    &self.inside_query_bias,
                    &mut inside_q[b * q_count * boundary_dim + q * boundary_dim..][..boundary_dim],
                );
            }
        }

        // 2. Einsum: start/end logits are dot(start_b, start_q) * scale;
        //    inside_logits are dot(inside_t, inside_q) * scale over text tokens.
        let mut start_logits = vec![0.0f32; batch * q_count * boundary_len];
        let mut end_logits = vec![0.0f32; batch * q_count * boundary_len];
        let mut inside_logits = vec![0.0f32; batch * q_count * seq_len];
        for b in 0..batch {
            for q in 0..q_count {
                let sb = &start_q[b * q_count * boundary_dim + q * boundary_dim..][..boundary_dim];
                let eb = &end_q[b * q_count * boundary_dim + q * boundary_dim..][..boundary_dim];
                let ib = &inside_q[b * q_count * boundary_dim + q * boundary_dim..][..boundary_dim];
                for i in 0..boundary_len {
                    let br = &start_b[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim];
                    let mut s = 0.0f32;
                    for kk in 0..boundary_dim {
                        s += br[kk] * sb[kk];
                    }
                    start_logits[b * q_count * boundary_len + q * boundary_len + i] = s * scale;
                }
                for i in 0..boundary_len {
                    let br = &end_b[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim];
                    let mut s = 0.0f32;
                    for kk in 0..boundary_dim {
                        s += br[kk] * eb[kk];
                    }
                    end_logits[b * q_count * boundary_len + q * boundary_len + i] = s * scale;
                }
                for i in 0..seq_len {
                    let br =
                        &inside_t[b * seq_len * boundary_dim + i * boundary_dim..][..boundary_dim];
                    let mut s = 0.0f32;
                    for kk in 0..boundary_dim {
                        s += br[kk] * ib[kk];
                    }
                    inside_logits[b * q_count * seq_len + q * seq_len + i] = s * scale;
                }
            }
        }

        // 3. Mask invalid positions with a finite sentinel so downstream
        // softmax / select-top-K does not NaN. Matches `MASK_LOGIT =
        // -1e4` in `target/.../boundary/constants.py`.
        const MASK_LOGIT: f32 = -1.0e4;
        for b in 0..batch {
            for q in 0..q_count {
                let q_valid = query_mask[b][q];
                for i in 0..boundary_len {
                    let idx = b * q_count * boundary_len + q * boundary_len + i;
                    let b_valid = boundary_mask[b][i];
                    if !q_valid || !b_valid {
                        start_logits[idx] = MASK_LOGIT;
                        end_logits[idx] = MASK_LOGIT;
                    }
                }
                for i in 0..seq_len {
                    let idx = b * q_count * seq_len + q * seq_len + i;
                    let t_valid = text_mask[b][i];
                    if !q_valid || !t_valid {
                        inside_logits[idx] = MASK_LOGIT;
                    }
                }
            }
        }

        // 4. Inside prefix: cumulative sum over tokens after mean-centering.
        //    Matches ``heads.BoundaryQueryHead.forward``: masked positions
        //    contribute zero, the mean is taken over the positions that
        //    survive ``text_mask & query_mask`` and is returned separately so
        //    interval scoring can restore the raw sum. Keeping the mean out of
        //    the cumsum is what stops a long document from accumulating a
        //    large constant offset.
        let mut inside_prefix = vec![0.0f32; batch * q_count * (seq_len + 1)];
        let mut inside_prefix_mean = vec![0.0f32; batch * q_count];
        for b in 0..batch {
            for q in 0..q_count {
                let prefix_base = b * q_count * (seq_len + 1) + q * (seq_len + 1);
                let logit_base = b * q_count * seq_len + q * seq_len;
                if !query_mask[b][q] {
                    inside_prefix_mean[b * q_count + q] = 0.0;
                    continue;
                }
                let mut sum = 0.0f32;
                let mut count = 0usize;
                for i in 0..seq_len {
                    if text_mask[b][i] {
                        sum += inside_logits[logit_base + i];
                        count += 1;
                    }
                }
                let mean = if count == 0 { 0.0 } else { sum / count as f32 };
                inside_prefix_mean[b * q_count + q] = mean;

                let mut running = 0.0f32;
                inside_prefix[prefix_base] = 0.0;
                for i in 0..seq_len {
                    if text_mask[b][i] {
                        running += inside_logits[logit_base + i] - mean;
                    }
                    inside_prefix[prefix_base + i + 1] = running;
                }
            }
        }

        BoundaryMarginals {
            start_logits,
            end_logits,
            inside_prefix,
            inside_prefix_mean,
            _marker: std::marker::PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
