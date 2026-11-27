//! `SpanContentPooler` — prefix-sum span content pooling.
//!
//! Mirrors `gliner2.models.boundary.content.SpanContentPooler`
//! (`target/gliner2-oracle/gliner2/models/boundary/content.py`).
//!
//! Every token state is projected to `content_dim`, masked, and turned into a
//! cumulative sum with a leading zero row, so pooling any `[start, end)` is
//! two gathers plus a divide — no per-candidate scan. The reference centers
//! nothing here; the mean is over the span only, taken as
//! `span_sum / max(end - start, 1)`.
//!
//! For base-v1: `content_dim = 64`, `content_soft_max_pool = false`,
//! `text_hidden_size = 768`. The mean channel is therefore the whole
//! `output_dim` and `layer_norm` runs over 64 features.
//!
//! `content_soft_max_pool = true` needs a second, log-cumsum-exp prefix and
//! is not implemented; [`SpanContentPooler::load`] refuses that
//! configuration rather than returning zeros for the soft-max half.

use crate::core::tensor::TensorSource;
use crate::models::gliner_boundary::tensor_util::{apply_linear_full, load_vec, load_weight};
use crate::ops::kernel::Weight;

/// Prefix-based span content pooling.
pub struct SpanContentPooler<'a> {
    /// Width of the projected token states.
    pub content_dim: usize,
    /// Width of the per-candidate pooled vector (2x `content_dim` when the
    /// soft-max channel is enabled, which this port does not support).
    pub output_dim: usize,
    /// Input width of `value_projection` (the encoder hidden size).
    pub text_hidden_size: usize,
    pub value_projection: Weight<'a>,
    pub value_bias: Vec<f32>,
    pub layer_norm_weight: Vec<f32>,
    pub layer_norm_bias: Vec<f32>,
}

impl<'a> SpanContentPooler<'a> {
    /// Load the pooler from `source`.
    ///
    /// `prefix` is the state-dict prefix without the `value_projection` /
    /// `layer_norm` leaf names, i.e. `boundary_head.pair_scorer.content_pooler`
    /// or `boundary_head.shared_pool_scorer.content_pooler`. base-v1 ships two
    /// independent poolers with the same shapes and different values, so the
    /// prefix is not cosmetic.
    pub fn load(
        source: &'a dyn TensorSource,
        prefix: &str,
        text_hidden_size: usize,
        content_dim: usize,
        content_soft_max_pool: bool,
    ) -> Result<Self, String> {
        if content_soft_max_pool {
            return Err(
                "content_soft_max_pool = true needs the log-cumsum-exp prefix from \
                 content.py:50-58, which the Rust SpanContentPooler does not implement"
                    .to_string(),
            );
        }
        let value_projection = load_weight(
            source,
            &format!("{prefix}.value_projection.weight"),
            text_hidden_size,
            content_dim,
        )?;
        let value_bias = load_vec(
            source,
            &format!("{prefix}.value_projection.bias"),
            content_dim,
        )?;
        let layer_norm_weight =
            load_vec(source, &format!("{prefix}.layer_norm.weight"), content_dim)?;
        let layer_norm_bias = load_vec(source, &format!("{prefix}.layer_norm.bias"), content_dim)?;
        Ok(Self {
            content_dim,
            output_dim: content_dim,
            text_hidden_size,
            value_projection,
            value_bias,
            layer_norm_weight,
            layer_norm_bias,
        })
    }

    /// Cumulative sum of the masked projected token states.
    ///
    /// `text_states` is `[B, L, text_hidden_size]` and `text_mask` is
    /// `[B, L]` flattened row-major (position `b * L + i`). Returns
    /// `[B, L + 1, content_dim]` where row 0 is zeros, so
    /// `prefix[b][end] - prefix[b][start]` is the span sum.
    pub fn build_prefix(
        &self,
        text_states: &[f32],
        text_mask: &[bool],
        batch: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let content_dim = self.content_dim;
        let hidden = self.text_hidden_size;
        debug_assert_eq!(text_mask.len(), batch * seq_len);
        let mut prefix = vec![0.0f32; batch * (seq_len + 1) * content_dim];
        let mut values = vec![0.0f32; content_dim];
        for b in 0..batch {
            let state_base = b * seq_len * hidden;
            let prefix_base = b * (seq_len + 1) * content_dim;
            for i in 0..seq_len {
                let slot = &mut prefix
                    [prefix_base + (i + 1) * content_dim..prefix_base + (i + 2) * content_dim];
                if !text_mask[b * seq_len + i] {
                    // `values * text_mask` in the reference: masked tokens add
                    // nothing to the running sum.
                    slot.fill(0.0);
                    continue;
                }
                let state = &text_states[state_base + i * hidden..][..hidden];
                apply_linear_full(state, &self.value_projection, &self.value_bias, &mut values);
                for (dst, v) in slot.iter_mut().zip(values.iter()) {
                    *dst = *v;
                }
            }
            // prefix[j + 1] = prefix[j] + values[j].
            //
            // This has to run *forward*. A backward pass would fold the raw
            // value of row j-1 into row j before row j-1 itself had been
            // accumulated, so every row would end up holding just the last two
            // terms — and it fails silently, only showing up as a smallish
            // score delta on spans that do not start at 0.
            for i in 0..seq_len {
                let (lower, upper) = prefix
                    [prefix_base + i * content_dim..prefix_base + (i + 2) * content_dim]
                    .split_at_mut(content_dim);
                for d in 0..content_dim {
                    upper[d] += lower[d];
                }
            }
        }
        prefix
    }

    /// Pool `[start, end)` content for each candidate and LayerNorm it.
    ///
    /// `starts` / `ends` are the candidate index pairs flattened row-major
    /// over `[B, Q, C]`, so `per_batch` (i.e. `q_count * c`) candidates
    /// belong to each row of the batch. Indices are clamped to `seq_len`,
    /// matching `gather_rows`' `clamp(0, n - 1)`. Returns
    /// `[B * Q * C, output_dim]`.
    pub fn pool(
        &self,
        mean_prefix: &[f32],
        starts: &[usize],
        ends: &[usize],
        per_batch: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let content_dim = self.content_dim;
        let output_dim = self.output_dim;
        debug_assert_eq!(starts.len(), ends.len());
        let mut pooled = vec![0.0f32; starts.len() * output_dim];
        for (idx, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
            let b = idx / per_batch.max(1);
            let start = start.min(seq_len);
            let end = end.min(seq_len);
            let length = end.saturating_sub(start).max(1) as f32;
            let prefix_base = b * (seq_len + 1) * content_dim;
            let row = &mut pooled[idx * output_dim..][..output_dim];
            for d in 0..content_dim {
                let p_end = mean_prefix[prefix_base + end * content_dim + d];
                let p_start = mean_prefix[prefix_base + start * content_dim + d];
                row[d] = (p_end - p_start) / length;
            }
            // The reference casts to the feature dtype before LayerNorm
            // (`pooled.to(out_dtype)`); everything here is already F32.
            let mut normalized = vec![0.0f32; output_dim];
            crate::ops::layer_norm(
                row,
                &self.layer_norm_weight,
                &self.layer_norm_bias,
                1e-5,
                &mut normalized,
            );
            row.copy_from_slice(&normalized);
        }
        pooled
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
