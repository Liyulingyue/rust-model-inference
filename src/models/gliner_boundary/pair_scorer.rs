//! `SparseBoundaryPairScorer.forward` — combine marginals + endpoint
//! compatibility + length features into a per-candidate logit.
//!
//! Mirrors `gliner2.models.boundary.scoring.SparseBoundaryPairScorer.forward`
//! (`target/gliner2-oracle/gliner2/models/boundary/scoring.py:177`)
//! with three feature sources DISABLED to keep this commit small:
//!  - `enable_span_content` (`SpanContentPooler`, ~600 lines of Python)
//!  - `use_inside_evidence` (`inside_prefix` interval scoring)
//!  - `endpoint_difference_features` (`(s - e) * (s - e).abs()` projection)
//!
//! These are flagged in the GGUF metadata so the omitted features can
//! be re-introduced as separate commits once the underlying primitives
//! are ported. For base-v1's published config they contribute non-trivial
//! signal, so the byte-exact oracle *cannot* match the upstream
//! reference until those come back; the oracle instead compares against
//! a Python reference that has the same features turned off
//! (`tools/oracle/gliner_boundary/dump_pair_scorer_limited.py`).
//!
//! Pipeline (limited):
//!  1. `project_endpoints(boundary_states)` -> (start_proj, end_proj) in
//!     `pair_dim`, rotary-rotated if `enable_rotary_endpoints`.
//!  2. `query_gate * s_proj * e_proj`, summed over `multihead_pair_compat_heads`
//!     pair-compat heads and reduced by `compat_mix` (init 1/heads each).
//!  3. Gather `start_logits` / `end_logits` at the candidate indices, add
//!     to the score (one-time marginals).
//!  4. Add the proposer's `compat_logits` (the marginal-free prior).
//!  5. Continuous length features `[log1p(end-start), (end-start)/len,
//!     1/sqrt(end-start)]`, projected through `length_query_projection`
//!     and dot-producted.
//!  6. Mask invalid candidates with `MASK_LOGIT`.
//!
//! After this scorer returns, the higher-level pipeline runs each
//! candidate's `(boundary_states[start], boundary_states[end])` concat
//! through `classifier.0` (GeLU) + `classifier.3` to produce the final
//! logit.

use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};

use super::proposer::RotaryBoundaryEmbedding;

/// Sparse pair-scoring math (limited).
pub struct PairScorer<'a> {
    pub boundary_dim: usize,
    pub pair_dim: usize,
    pub query_dim: usize,
    pub multihead_pair_compat_heads: usize,
    pub start_endpoint: Weight<'a>,
    pub start_endpoint_bias: Vec<f32>,
    pub end_endpoint: Weight<'a>,
    pub end_endpoint_bias: Vec<f32>,
    pub query_gate: Weight<'a>,
    pub query_gate_bias: Vec<f32>,
    pub compat_mix: Weight<'a>,
    pub compat_mix_bias: Vec<f32>,
    pub length_query: Weight<'a>,
    pub length_query_bias: Vec<f32>,
    pub rotary: Option<RotaryBoundaryEmbedding>,
}

impl<'a> PairScorer<'a> {
    pub fn load(source: &'a dyn TensorSource, hidden_size: usize) -> Result<Self, String> {
        let pair_dim = source
            .tensor_info("boundary_head.pair_scorer.start_endpoint_projection.weight")
            .ok_or("missing boundary_head.pair_scorer.start_endpoint_projection.weight")?
            .dims[0] as usize;
        let q_dim = source
            .tensor_info("boundary_head.pair_scorer.query_gate.weight")
            .ok_or("missing boundary_head.pair_scorer.query_gate.weight")?
            .dims[0] as usize;

        // Boundary dim from start_endpoint_projection output channel.
        let boundary_dim = source
            .tensor_info("boundary_head.pair_scorer.start_endpoint_projection.weight")
            .ok_or("missing start_endpoint_projection")?
            .dims[1] as usize;
        // If start_pair_projection disagrees, the boundary_dim is the
        // boundary encoder's, which is `start_pair_projection`'s input.
        // Use the pair_dim if they differ (rare).

        let enable_rotary_endpoints = source
            .metadata("gliner2.boundary.enable_rotary_endpoints")
            .and_then(|v| v.to_f64())
            .map(|v| v != 0.0)
            .unwrap_or(true);

        // Infer multihead_pair_compat_heads from compat_mix.weight's
        // input channel (the reference stores it as
        // `nn.Linear(multihead_pair_compat_heads, 1)`). Metadata is
        // preferred when present.
        let multihead_pair_compat_heads = source
            .tensor_info("boundary_head.pair_scorer.compat_mix.weight")
            .map(|info| info.dims[0] as usize)
            .or_else(|| {
                source
                    .metadata("gliner2.boundary.multihead_pair_compat_heads")
                    .and_then(|v| v.to_u64())
                    .map(|v| v as usize)
            })
            .unwrap_or(1);
        if pair_dim % multihead_pair_compat_heads != 0 {
            return Err(format!(
                "pair_dim {pair_dim} not divisible by multihead_pair_compat_heads {multihead_pair_compat_heads}"
            ));
        }
        let compat_head_dim = pair_dim / multihead_pair_compat_heads;

        let start_endpoint = load_weight(
            source,
            "boundary_head.pair_scorer.start_endpoint_projection.weight",
            boundary_dim,
            pair_dim,
        )?;
        let start_endpoint_bias = load_vec(
            source,
            "boundary_head.pair_scorer.start_endpoint_projection.bias",
            pair_dim,
        )?;
        let end_endpoint = load_weight(
            source,
            "boundary_head.pair_scorer.end_endpoint_projection.weight",
            boundary_dim,
            pair_dim,
        )?;
        let end_endpoint_bias = load_vec(
            source,
            "boundary_head.pair_scorer.end_endpoint_projection.bias",
            pair_dim,
        )?;
        let query_gate = load_weight(
            source,
            "boundary_head.pair_scorer.query_gate.weight",
            q_dim,
            if enable_rotary_endpoints {
                pair_dim / 2
            } else {
                pair_dim
            },
        )?;
        let query_gate_bias = load_vec(
            source,
            "boundary_head.pair_scorer.query_gate.bias",
            if enable_rotary_endpoints {
                pair_dim / 2
            } else {
                pair_dim
            },
        )?;
        let compat_mix = load_weight(
            source,
            "boundary_head.pair_scorer.compat_mix.weight",
            multihead_pair_compat_heads,
            1,
        )?;
        let compat_mix_bias = load_vec(source, "boundary_head.pair_scorer.compat_mix.bias", 1)?;
        let length_query = load_weight(
            source,
            "boundary_head.pair_scorer.length_query_projection.weight",
            q_dim,
            3,
        )?;
        let length_query_bias = load_vec(
            source,
            "boundary_head.pair_scorer.length_query_projection.bias",
            3,
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

        let _ = compat_head_dim;
        let _ = hidden_size;

        Ok(Self {
            boundary_dim,
            pair_dim,
            query_dim: q_dim,
            multihead_pair_compat_heads,
            start_endpoint,
            start_endpoint_bias,
            end_endpoint,
            end_endpoint_bias,
            query_gate,
            query_gate_bias,
            compat_mix,
            compat_mix_bias,
            length_query,
            length_query_bias,
            rotary,
        })
    }

    /// Project boundary states to start/end endpoint vectors in pair_dim
    /// space, with optional rotary embedding.
    fn project_endpoints(
        &self,
        boundary_states: &[f32],
        boundary_len: usize,
        batch: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut start_all = vec![0.0f32; batch * boundary_len * self.pair_dim];
        let mut end_all = vec![0.0f32; batch * boundary_len * self.pair_dim];
        for b in 0..batch {
            for i in 0..boundary_len {
                apply_linear_full(
                    &boundary_states
                        [b * boundary_len * self.boundary_dim + i * self.boundary_dim..]
                        [..self.boundary_dim],
                    &self.start_endpoint,
                    &self.start_endpoint_bias,
                    &mut start_all[b * boundary_len * self.pair_dim + i * self.pair_dim..]
                        [..self.pair_dim],
                );
                apply_linear_full(
                    &boundary_states
                        [b * boundary_len * self.boundary_dim + i * self.boundary_dim..]
                        [..self.boundary_dim],
                    &self.end_endpoint,
                    &self.end_endpoint_bias,
                    &mut end_all[b * boundary_len * self.pair_dim + i * self.pair_dim..]
                        [..self.pair_dim],
                );
            }
        }
        if let Some(rotary) = &self.rotary {
            rotary.apply(&mut start_all, boundary_len, self.pair_dim);
            rotary.apply(&mut end_all, boundary_len, self.pair_dim);
        }
        (start_all, end_all)
    }

    /// Compute per-(B, Q, C) pair score.
    ///
    /// Inputs:
    ///  - `boundary_states` [B, boundary_len, boundary_dim]
    ///  - `query_states` [B, Q, query_dim]
    ///  - `start_logits` / `end_logits` [B, Q, boundary_len] (from
    ///    `BoundaryQueryHead`)
    ///  - `compat_logits` [B, Q, C] (the marginal-free prior from
    ///    `score_explicit_pairs`)
    ///  - `indices` [B, Q, C, 2] (start, end pairs flat)
    ///  - `text_lengths` [B]
    ///  - `valid_mask` [B, Q, C]
    pub fn forward(
        &self,
        boundary_states: &[f32],
        boundary_len: usize,
        query_states: &[f32],
        batch: usize,
        q_count: usize,
        c: usize,
        start_logits: &[f32],
        end_logits: &[f32],
        compat_logits: &[f32],
        indices: &[usize],
        text_lengths: &[usize],
        valid_mask: &[bool],
    ) -> Vec<f32> {
        // 1. Project endpoints.
        let (start_all, end_all) = self.project_endpoints(boundary_states, boundary_len, batch);

        // 2. Query gate (sigmoid) and gate-scaled start/end dot products
        //    reduced by multihead pair compat + compat_mix.
        let compat_head_dim = self.pair_dim / self.multihead_pair_compat_heads;
        let mut compat_per_cand = vec![0.0f32; batch * q_count * c];
        for b in 0..batch {
            for q in 0..q_count {
                // query gate projected at this query
                let qg_bias_stride = if self.rotary.is_some() {
                    self.pair_dim / 2
                } else {
                    self.pair_dim
                };
                let qg_offset = b * q_count * qg_bias_stride + q * qg_bias_stride;
                let gate_dim = qg_bias_stride;
                // gate[p] = sigmoid(query_gate_weight @ q_row + bias)
                // stored in a temp
                let mut gate = vec![0.0f32; gate_dim];
                apply_linear_full(
                    &query_states[b * q_count * self.query_dim + q * self.query_dim..]
                        [..self.query_dim],
                    &self.query_gate,
                    &self.query_gate_bias,
                    &mut gate,
                );
                for v in gate.iter_mut() {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
                // repeat_interleave(2, dim=-1) when rotary is on
                let gate_full = if self.rotary.is_some() {
                    let mut g = vec![0.0f32; 2 * gate_dim];
                    for k in 0..gate_dim {
                        g[2 * k] = gate[k];
                        g[2 * k + 1] = gate[k];
                    }
                    g
                } else {
                    gate
                };
                for ci in 0..c {
                    let idx_base = (b * q_count * c + q * c + ci) * 2;
                    let start_idx = indices[idx_base];
                    let end_idx = indices[idx_base + 1];
                    if start_idx >= boundary_len || end_idx >= boundary_len {
                        continue;
                    }
                    let s_off = b * boundary_len * self.pair_dim + start_idx * self.pair_dim;
                    let e_off = b * boundary_len * self.pair_dim + end_idx * self.pair_dim;
                    let mut per_head = vec![0.0f32; self.multihead_pair_compat_heads];
                    // Reference eval order: (s_proj * gate) * e_proj, then
                    // sum over the head's last dim. Match the multiply
                    // order so F32 rounding lands on the same bits.
                    for h in 0..self.multihead_pair_compat_heads {
                        let mut sum = 0.0f32;
                        for k in 0..compat_head_dim {
                            let sv = start_all[s_off + h * compat_head_dim + k];
                            let ev = end_all[e_off + h * compat_head_dim + k];
                            let gv = gate_full[h * compat_head_dim + k];
                            sum += (sv * gv) * ev;
                        }
                        per_head[h] = sum;
                    }
                    // compat = compat_mix(per_head).squeeze(-1) * scale
                    // where scale = 1 / sqrt(pair_dim).
                    let mut compat_logit = 0.0f32;
                    for h in 0..self.multihead_pair_compat_heads {
                        compat_logit += self.compat_mix_at(h) * per_head[h];
                    }
                    compat_logit = compat_logit * (1.0 / (self.pair_dim as f32).sqrt())
                        + self.compat_mix_bias[0];
                    compat_per_cand[b * q_count * c + q * c + ci] = compat_logit;
                }
            }
        }

        // 3. Gather start/end marginals at indices.
        let mut score = vec![0.0f32; batch * q_count * c];
        for b in 0..batch {
            for q in 0..q_count {
                for ci in 0..c {
                    let idx_base = (b * q_count * c + q * c + ci) * 2;
                    let start_idx = indices[idx_base];
                    let end_idx = indices[idx_base + 1];
                    if start_idx >= boundary_len || end_idx >= boundary_len {
                        continue;
                    }
                    let s_logit =
                        start_logits[b * q_count * boundary_len + q * boundary_len + start_idx];
                    let e_logit =
                        end_logits[b * q_count * boundary_len + q * boundary_len + end_idx];
                    score[b * q_count * c + q * c + ci] =
                        compat_per_cand[b * q_count * c + q * c + ci] + s_logit + e_logit;
                }
            }
        }

        // 4. Add the proposer's compat_logits prior.
        for i in 0..compat_logits.len().min(score.len()) {
            score[i] += compat_logits[i];
        }

        // 5. Length features.
        // feats = stack([log1p(end-start), (end-start)/text_length, rsqrt(end-start)])
        // length_coeff = length_query_projection(query_states) -> [B, Q, 3]
        // score += sum(feats * length_coeff, dim=-1)
        for b in 0..batch {
            for q in 0..q_count {
                let length_coeff = self.length_query_at(query_states, q_count, q, b);
                for ci in 0..c {
                    let idx_base = (b * q_count * c + q * c + ci) * 2;
                    let start_idx = indices[idx_base];
                    let end_idx = indices[idx_base + 1];
                    if start_idx >= boundary_len || end_idx >= boundary_len {
                        continue;
                    }
                    let length = end_idx as f32 - start_idx as f32;
                    let safe_length = length.max(1.0);
                    let f1 = (safe_length + 1.0).ln(); // log1p
                    let tl = text_lengths.get(b).copied().unwrap_or(1).max(1) as f32;
                    let f2 = safe_length / tl;
                    let f3 = 1.0 / safe_length.sqrt();
                    let contrib =
                        f1 * length_coeff[0] + f2 * length_coeff[1] + f3 * length_coeff[2];
                    score[b * q_count * c + q * c + ci] += contrib;
                }
            }
        }

        // 6. Mask invalid candidates with MASK_LOGIT.
        const MASK_LOGIT: f32 = -1.0e4;
        for i in 0..valid_mask.len() {
            if !valid_mask[i] {
                score[i] = MASK_LOGIT;
            }
        }

        score
    }

    /// Read `compat_mix.weight[h]` (init = 1 / multihead_pair_compat_heads).
    fn compat_mix_at(&self, head: usize) -> f32 {
        // compat_mix is shape [multihead_pair_compat_heads, 1] stored
        // row-major; we index as compat_mix_at * 1 + 0.
        if let Some(rows) = self.compat_mix.kernel.f32_slice() {
            rows[head]
        } else {
            0.0
        }
    }

    /// Read `length_query_projection.weight @ query_state + bias` for
    /// the (b, q) query as a [3] Vec.
    fn length_query_at(
        &self,
        query_states: &[f32],
        q_count: usize,
        q: usize,
        batch_offset: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; 3];
        let start = batch_offset * q_count * self.query_dim + q * self.query_dim;
        apply_linear_full(
            &query_states[start..][..self.query_dim],
            &self.length_query,
            &self.length_query_bias,
            &mut out,
        );
        out
    }
}

// (q_count_for_shape removed: unused.)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_vec(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    crate::core::tensor::load_f32_tensor(source, name, &[len as u64])
        .map_err(|e| format!("{name}: {e}"))
}

fn load_weight<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] {
        return Err(format!(
            "tensor {name} has dims {:?}, expected [{n_in}, {n_out}]",
            info.dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("missing tensor data {name}"))?;
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in,
        n_out,
    )))
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
