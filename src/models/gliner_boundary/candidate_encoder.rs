//! `candidate_encoder` — the record head's candidate input projection.
//!
//! Mirrors `BoundaryHead.candidate_encoder` (`model.py:183-185`, built at
//! `model.py:445`).
//!
//! This exists because `candidate_states` means two different things in the
//! reference and conflating them was a live bug here:
//!
//! * `SharedPoolScorer`'s **internal** `candidate` vector is
//!   `start_rep + end_rep + length_proj + prior`, `pair_dim` (128) wide. It
//!   exists only to produce `pair_logits`.
//! * `CandidateTensorBatch.candidate_states` is the **public** one the record
//!   head consumes: `candidate_encoder(cat(start_boundary, end_boundary))`,
//!   `hidden_size` (768) wide, zeroed on invalid slots.
//!
//! `DocumentCandidateBatch` used to carry the first under the second's name, so
//! the width was off by 6x and nothing that ran so far noticed, because spans
//! read `pair_logits` and no consumer of the public tensor existed yet. The two
//! are now separate fields with the names they have in the reference.

use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};

/// `nn.Linear(2 * boundary_dim, hidden_size)`, no activation.
pub struct CandidateEncoder<'a> {
    pub hidden_size: usize,
    pub boundary_dim: usize,
    weight: Weight<'a>,
    bias: Vec<f32>,
}

impl<'a> CandidateEncoder<'a> {
    pub fn load(
        source: &'a dyn TensorSource,
        hidden_size: usize,
        boundary_dim: usize,
    ) -> Result<Self, String> {
        let in_dim = 2 * boundary_dim;
        let info = source
            .tensor_info("boundary_head.candidate_encoder.weight")
            .ok_or("missing tensor boundary_head.candidate_encoder.weight")?;
        if info.dims != [in_dim as u64, hidden_size as u64] {
            return Err(format!(
                "tensor boundary_head.candidate_encoder.weight has dims {:?}, expected [{in_dim}, {hidden_size}]",
                info.dims
            ));
        }
        let bytes = source
            .tensor_slice("boundary_head.candidate_encoder.weight")
            .ok_or("missing tensor data boundary_head.candidate_encoder.weight")?;
        let bias = crate::core::tensor::load_f32_tensor(
            source,
            "boundary_head.candidate_encoder.bias",
            &[hidden_size as u64],
        )
        .map_err(|e| format!("boundary_head.candidate_encoder.bias: {e}"))?;
        Ok(Self {
            hidden_size,
            boundary_dim,
            weight: Weight::from_quantized(QuantizedTensor::from_bytes(
                bytes,
                info.ggml_type,
                in_dim,
                hidden_size,
            )),
            bias,
        })
    }

    /// `hidden[positions, start, :]` and `hidden[positions, end, :]` are
    /// concatenated along the last axis and projected. `start` / `end` are the
    /// candidate's half-open offsets; the *end* is used as-is, not decremented —
    /// the reference gathers `pooled.indices[..., 1]` directly, so the last
    /// covered token is `end - 1` and the gathered row is one past it. That is
    /// what the reference does, so it is what this does.
    ///
    /// `valid` zeroes the row, matching
    /// `.masked_fill(~pooled.mask.unsqueeze(-1), 0.0)`. Zeroing rather than
    /// leaving the projection is load-bearing: an invalid slot's endpoints are
    /// whatever the pool padding happened to be, and a real state there would
    /// leak into the record head's assignment scores.
    pub fn forward(
        &self,
        hidden: &[f32],
        positions: usize,
        start: &[usize],
        end: &[usize],
        valid: &[bool],
    ) -> Vec<f32> {
        let candidates = start.len();
        let mut out = vec![0.0f32; candidates * self.hidden_size];
        let mut features = vec![0.0f32; 2 * self.boundary_dim];
        for slot in 0..candidates {
            if !valid.get(slot).copied().unwrap_or(false) {
                continue;
            }
            let s = start[slot].min(positions.saturating_sub(1));
            let e = end[slot].min(positions.saturating_sub(1));
            let s_base = s * self.boundary_dim;
            let e_base = e * self.boundary_dim;
            features[..self.boundary_dim].copy_from_slice(&hidden[s_base..][..self.boundary_dim]);
            features[self.boundary_dim..].copy_from_slice(&hidden[e_base..][..self.boundary_dim]);
            apply_linear_full(
                &features,
                &self.weight,
                &self.bias,
                &mut out[slot * self.hidden_size..][..self.hidden_size],
            );
        }
        out
    }
}

fn apply_linear_full(input: &[f32], weight: &Weight<'_>, bias: &[f32], output: &mut [f32]) {
    if let Some(rows) = weight.kernel.f32_slice() {
        let n_in = input.len();
        let n_out = output.len();
        debug_assert_eq!(bias.len(), n_out);
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
