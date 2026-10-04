//! `SparseBoundaryProposer::score_explicit_pairs` + `RotaryBoundaryEmbedding`.
//!
//! Mirrors `gliner2.models.boundary.proposal.SparseBoundaryProposer.
//! score_explicit_pairs` (`target/gliner2-oracle/gliner2/models/boundary/
//! proposal.py:446`) and `gliner2.models.boundary.rotary.
//! RotaryBoundaryEmbedding.forward`. This path bypasses the sparse
//! top-K proposer (which would require ~700 lines) and directly scores
//! caller-provided `[B, Q, C, 2]` (start, end) indices. Same projection
//! layers and same compatibility math as the regular candidate pass;
//! only the selection step is replaced.
//!
//! Configuration baked into the base-v1 GGUF:
//!  - `enable_rotary_endpoints: true` — endpoint vectors are rotated
//!    before the dot product, so query gate `repeat_interleave(2, dim=-1)`
//!    matches the even/odd interleave of the rotary.
//!  - `boundary_dim: 128` (no multihead pair-compat in this path).
//!  - `query_dim: 768` (== hidden_size for base-v1).
//!
//! Byte-exact oracle: `tools/oracle/gliner_boundary/
//! dump_score_explicit_pairs.py` + `tests/gliner2_5_base_v1_
//! score_explicit_pairs_parity.rs`.

use crate::core::tensor::TensorSource;
use crate::models::gliner_boundary::tensor_util::{apply_linear_full, load_vec, load_weight};
use crate::ops::kernel::Weight;

/// `SparseBoundaryProposer` weights used by the explicit-span scorer.
pub struct BoundaryProposer<'a> {
    pub start_pair: Weight<'a>,
    pub start_pair_bias: Vec<f32>,
    pub end_key: Weight<'a>,
    pub end_key_bias: Vec<f32>,
    pub start_query: Weight<'a>,
    pub start_query_bias: Vec<f32>,
    /// Optional rotary embedding. Only constructed if
    /// `enable_rotary_endpoints` is set in the boundary config (base-v1
    /// has it enabled; base variants without it pass `None` here and the
    /// rotary step is skipped).
    pub rotary: Option<RotaryBoundaryEmbedding>,
    pub boundary_dim: usize,
    pub query_dim: usize,
    pub enable_rotary_endpoints: bool,
}

impl<'a> BoundaryProposer<'a> {
    pub fn load(source: &'a dyn TensorSource, hidden_size: usize) -> Result<Self, String> {
        let boundary_dim = source
            .tensor_info("boundary_head.boundary_proposer.start_pair_projection.weight")
            .ok_or("missing boundary_head.boundary_proposer.start_pair_projection.weight")?
            .dims[0] as usize;
        let q_dim = source
            .tensor_info("boundary_head.boundary_proposer.start_query_projection.weight")
            .ok_or("missing boundary_head.boundary_proposer.start_query_projection.weight")?
            .dims[0] as usize;
        let _ = hidden_size;

        let start_pair = load_weight(
            source,
            "boundary_head.boundary_proposer.start_pair_projection.weight",
            boundary_dim,
            boundary_dim,
        )?;
        let start_pair_bias = load_vec(
            source,
            "boundary_head.boundary_proposer.start_pair_projection.bias",
            boundary_dim,
        )?;
        let end_key = load_weight(
            source,
            "boundary_head.boundary_proposer.end_key_projection.weight",
            boundary_dim,
            boundary_dim,
        )?;
        let end_key_bias = load_vec(
            source,
            "boundary_head.boundary_proposer.end_key_projection.bias",
            boundary_dim,
        )?;

        // Base-v1 has `enable_rotary_endpoints: true`. The Python GGUF
        // doesn't carry an explicit toggle; we read it via a GGUF
        // metadata field added by the converter. If missing, default to
        // true (matches base-v1's published config).
        let enable_rotary_endpoints = source
            .metadata("gliner2.boundary.enable_rotary_endpoints")
            .and_then(|v| v.to_f64())
            .map(|v| v != 0.0)
            .unwrap_or(true);
        // Read start_query_projection with the right output shape (rotary
        // toggles the output dim between `boundary_dim` and
        // `boundary_dim // 2`).
        let start_query = load_weight(
            source,
            "boundary_head.boundary_proposer.start_query_projection.weight",
            q_dim,
            if enable_rotary_endpoints {
                boundary_dim / 2
            } else {
                boundary_dim
            },
        )?;
        let start_query_bias = load_vec(
            source,
            "boundary_head.boundary_proposer.start_query_projection.bias",
            if enable_rotary_endpoints {
                boundary_dim / 2
            } else {
                boundary_dim
            },
        )?;
        let rotary = if enable_rotary_endpoints {
            let rotary_base = source
                .metadata("gliner2.boundary.rotary_base")
                .and_then(|v| v.to_f64())
                .map(|v| v as f32)
                .unwrap_or(10000.0);
            Some(RotaryBoundaryEmbedding::new(boundary_dim, rotary_base))
        } else {
            None
        };

        Ok(Self {
            start_pair,
            start_pair_bias,
            end_key,
            end_key_bias,
            start_query,
            start_query_bias,
            rotary,
            boundary_dim,
            query_dim: q_dim,
            enable_rotary_endpoints,
        })
    }

    /// Score `[B, Q, C, 2]` half-open (start, end) indices against
    /// boundary states. Returns per-(B, Q, C) compatibility logit.
    /// `valid_mask` is `[B, Q, C]` and zeros out invalid candidates.
    ///
    /// Shapes (all flat row-major):
    ///  - `boundary_states`: `[B, boundary_len, boundary_dim]`
    ///  - `query_states`: `[B, Q, query_dim]`
    ///  - `indices`: `[B, Q, C, 2]` (start, end pairs concatenated)
    ///  - `valid_mask`: `[B, Q, C]`
    pub fn score_explicit_pairs(
        &self,
        boundary_states: &[f32],
        boundary_len: usize,
        query_states: &[f32],
        batch: usize,
        q_count: usize,
        c: usize,
        indices: &[u32],
        valid_mask: &[bool],
    ) -> Vec<f32> {
        let boundary_dim = self.boundary_dim;
        let _ = c;

        // 1. Project boundary states: start_pair (boundary_dim -> boundary_dim),
        //    end_key (boundary_dim -> boundary_dim).
        let mut start_all = vec![0.0f32; batch * boundary_len * boundary_dim];
        let mut end_all = vec![0.0f32; batch * boundary_len * boundary_dim];
        for b in 0..batch {
            for i in 0..boundary_len {
                apply_linear_full(
                    &boundary_states[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim],
                    &self.start_pair,
                    &self.start_pair_bias,
                    &mut start_all[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim],
                );
                apply_linear_full(
                    &boundary_states[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim],
                    &self.end_key,
                    &self.end_key_bias,
                    &mut end_all[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim],
                );
            }
        }

        // 2. Optional rotary embeddings on the boundary projections.
        if let Some(rotary) = &self.rotary {
            rotary.apply(&mut start_all, boundary_len, boundary_dim);
            rotary.apply(&mut end_all, boundary_len, boundary_dim);
        }

        // 3. Project query states: start_query (q_dim -> boundary_dim).
        //    Apply sigmoid to get the gate.
        let mut gate = vec![0.0f32; batch * q_count * boundary_dim];
        for b in 0..batch {
            for q in 0..q_count {
                apply_linear_full(
                    &query_states[b * q_count * self.query_dim + q * self.query_dim..]
                        [..self.query_dim],
                    &self.start_query,
                    &self.start_query_bias,
                    &mut gate[b * q_count * boundary_dim + q * boundary_dim..][..boundary_dim],
                );
                for v in &mut gate[b * q_count * boundary_dim + q * boundary_dim..][..boundary_dim]
                {
                    *v = 1.0 / (1.0 + (-*v).exp()); // sigmoid
                }
            }
        }
        if self.rotary.is_some() {
            // repeat_interleave(2, dim=-1): [B, Q, d] -> [B, Q, 2*d]
            let mut expanded = vec![0.0f32; batch * q_count * 2 * boundary_dim];
            for b in 0..batch {
                for q in 0..q_count {
                    for k in 0..boundary_dim {
                        expanded[b * q_count * 2 * boundary_dim + q * 2 * boundary_dim + 2 * k] =
                            gate[b * q_count * boundary_dim + q * boundary_dim + k];
                        expanded
                            [b * q_count * 2 * boundary_dim + q * 2 * boundary_dim + 2 * k + 1] =
                            gate[b * q_count * boundary_dim + q * boundary_dim + k];
                    }
                }
            }
            gate = expanded;
        }

        // 4. Gather start/end at indices, multiply by gate (start only),
        //    take dot product, scale by 1/sqrt(d).
        let scale = 1.0 / (boundary_dim as f32).sqrt();
        let mut compatibility = vec![0.0f32; batch * q_count * c];
        for b in 0..batch {
            for q in 0..q_count {
                for ci in 0..c {
                    let idx_base = ((b * q_count + q) * c + ci) * 2;
                    let start_idx = indices[idx_base] as usize;
                    let end_idx = indices[idx_base + 1] as usize;
                    if start_idx >= boundary_len || end_idx >= boundary_len {
                        // clamp will saturate the dot product to a finite
                        // value; we'll mask it out via valid_mask below.
                        continue;
                    }
                    let mut dot = 0.0f32;
                    let gate_stride = if self.rotary.is_some() {
                        2 * boundary_dim
                    } else {
                        boundary_dim
                    };
                    let gate_base = b * q_count * gate_stride + q * gate_stride;
                    for k in 0..boundary_dim {
                        let s = start_all
                            [b * boundary_len * boundary_dim + start_idx * boundary_dim + k];
                        let e =
                            end_all[b * boundary_len * boundary_dim + end_idx * boundary_dim + k];
                        let g = gate[gate_base + k];
                        dot += s * e * g;
                    }
                    compatibility[b * q_count * c + q * c + ci] = dot * scale;
                }
            }
        }

        // 5. Mask invalid candidates with zero (matches the reference's
        //    `torch.where(valid_mask, compatibility, torch.zeros_like(...))`).
        for b in 0..batch {
            for q in 0..q_count {
                for ci in 0..c {
                    if !valid_mask[b * q_count * c + q * c + ci] {
                        compatibility[b * q_count * c + q * c + ci] = 0.0;
                    }
                }
            }
        }

        compatibility
    }
}

/// Rotary embeddings over boundary positions: rotates even/odd halves
/// of the endpoint projection by `position * inv_freq[k]`.
pub struct RotaryBoundaryEmbedding {
    inv_freq: Vec<f32>,
}

impl RotaryBoundaryEmbedding {
    pub fn new(dim: usize, base: f32) -> Self {
        debug_assert_eq!(dim % 2, 0);
        let mut inv_freq = Vec::with_capacity(dim / 2);
        for k in 0..(dim / 2) {
            inv_freq.push(1.0 / base.powf((2 * k) as f32 / dim as f32));
        }
        Self { inv_freq }
    }

    /// In-place rotate `states[b, i, d]` for every batch and position.
    /// `dim` must equal `states.len() / (batch * seq_len)`.
    pub fn apply(&self, states: &mut [f32], seq_len: usize, dim: usize) {
        let batch = states.len() / (seq_len * dim);
        for b in 0..batch {
            for i in 0..seq_len {
                for k in 0..(dim / 2) {
                    let angle = (i as f32) * self.inv_freq[k];
                    let cos = angle.cos();
                    let sin = angle.sin();
                    let even_idx = b * seq_len * dim + i * dim + 2 * k;
                    let odd_idx = even_idx + 1;
                    let even = states[even_idx];
                    let odd = states[odd_idx];
                    states[even_idx] = even * cos - odd * sin;
                    states[odd_idx] = even * sin + odd * cos;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
