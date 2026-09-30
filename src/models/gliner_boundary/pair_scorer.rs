//! `SparseBoundaryPairScorer.forward` — combine marginals + endpoint
//! compatibility + span content + inside evidence + length features into a
//! per-candidate logit.
//!
//! Mirrors `gliner2.models.boundary.scoring.SparseBoundaryPairScorer.forward`
//! (`target/gliner2-oracle/gliner2/models/boundary/scoring.py:177`).
//!
//! Pipeline, in the reference's order (the order matters: F32 rounding is
//! only reproducible if the adds happen the same way):
//!  1. `project_endpoints(boundary_states)` -> (start_proj, end_proj) in
//!     `pair_dim`, rotary-rotated if `enable_rotary_endpoints`.
//!  2. `query_gate * s_proj * e_proj`, summed over `multihead_pair_compat_heads`
//!     pair-compat heads and reduced by `compat_mix` (init 1/heads each).
//!  3. `endpoint_difference_projection(cat(s - e, |s - e|))` if enabled.
//!  4. Gather `start_logits` / `end_logits` at the candidate indices and add
//!     them (the marginals enter exactly once; the prior is marginal-free).
//!  5. Add the proposer's `compat_logits` prior, zeroed where invalid.
//!  6. Span content: `(span_content * coeff).sum() * 1/sqrt(content_dim)` plus
//!     `content_bias(span_content)` if `enable_span_content`.
//!  7. Inside evidence: `inside_weight * (interval / sqrt(width))` if
//!     `use_inside_evidence`. `interval` is the *raw* inside-logit sum, i.e.
//!     the mean-centered prefix difference with `inside_prefix_mean` added
//!     back over the span width.
//!  8. Continuous length features `[log1p(end-start), (end-start)/len,
//!     rsqrt(end-start)]` projected through `length_query_projection`.
//!  9. Mask invalid candidates with `MASK_LOGIT`.
//!
//! Which optional feature sources exist is decided by the
//! `gliner2.boundary.*` metadata that
//! `tools/converter/gliner/convert_boundary.py` transcribes out of the
//! checkpoint's `boundary_head` block. The loader refuses to guess: a
//! checkpoint whose config says `enable_span_content` but whose GGUF lacks
//! the metadata is a conversion bug, not a model to silently down-grade.
//!
//! This scorer is the *span-conditioned* path
//! (`BoundaryExtractor.score_explicit_spans`). The document-level inference
//! path is `DocumentCandidatePool` + `SharedPoolScorer` whenever
//! `candidate_pool = "shared"` (base-v1's setting), which is tracked
//! separately in `glinerTODO.md`.

use crate::core::tensor::{MetaValue, TensorSource};
use crate::ops::kernel::{QuantizedTensor, Weight};

use super::content_pooler::SpanContentPooler;
use super::proposer::RotaryBoundaryEmbedding;

/// `MASK_LOGIT` from `boundary/constants.py`. A finite sentinel, not `-inf`,
/// so sums and softmaxes downstream stay finite.
const MASK_LOGIT: f32 = -1.0e4;

/// Optional feature sources, transcribed from `boundary_head` in the
/// checkpoint config. Every field changes the score, so a mismatch is a
/// wrong-answer bug rather than a perf knob.
#[derive(Clone, Copy, Debug)]
pub struct PairScorerFeatures {
    pub use_inside_evidence: bool,
    pub enable_span_content: bool,
    pub content_soft_max_pool: bool,
    pub query_conditioned_inside_weight: bool,
    pub endpoint_difference_features: bool,
    pub enable_rotary_endpoints: bool,
    pub reranker_endpoint_compat: bool,
}

/// Everything [`PairScorer::forward`] needs.
///
/// The flat tensors are all row-major with the shapes noted per field;
/// `batch` / `q_count` / `c` / `boundary_len` describe how to walk them.
pub struct PairScoreInputs<'t> {
    /// `[B, L+1, boundary_dim]` from `BoundaryEncoder`.
    pub boundary_states: &'t [f32],
    /// `L + 1` — the boundary positions per row.
    pub boundary_len: usize,
    /// `[B, Q, query_dim]`.
    pub query_states: &'t [f32],
    /// `[B, Q, L+1]` from `BoundaryQueryHead`.
    pub start_logits: &'t [f32],
    /// `[B, Q, L+1]` from `BoundaryQueryHead`.
    pub end_logits: &'t [f32],
    /// `[B, Q, C]` marginal-free prior from `score_explicit_pairs`.
    pub compat_logits: &'t [f32],
    /// `[B, Q, C, 2]` candidate `(start, end)` pairs.
    pub indices: &'t [usize],
    /// `[B, Q, C]`.
    pub valid_mask: &'t [bool],
    /// `[B, Q, L+1]` mean-centered inside prefix from `BoundaryQueryHead`.
    /// Unused when `use_inside_evidence` is false.
    pub inside_prefix: &'t [f32],
    /// `[B, Q]` the mean that was subtracted before the cumulative sum.
    pub inside_prefix_mean: &'t [f32],
    /// `[B, L, hidden]` encoder token states. Only read when
    /// `enable_span_content` is true.
    pub text_states: &'t [f32],
    /// `[B][L]` encoder token mask. Also fixes the sequence length for the
    /// content prefix.
    pub text_mask: &'t [Vec<bool>],
    /// `[B]` number of real tokens per row.
    pub text_lengths: &'t [usize],
    pub batch: usize,
    pub q_count: usize,
    pub c: usize,
}

/// How the inside-evidence term is weighted.
///
/// `query_conditioned_inside_weight` decides between a learned
/// `nn.Linear(query_dim, 1)` and a bare scalar parameter
/// (`scoring.py:121-125`); the two have different state dicts.
enum InsideWeight<'a> {
    Scalar(f32),
    QueryConditioned { weight: Weight<'a>, bias: Vec<f32> },
}

/// Sparse pair-scoring math.
pub struct PairScorer<'a> {
    pub features: PairScorerFeatures,
    pub boundary_dim: usize,
    pub pair_dim: usize,
    pub query_dim: usize,
    /// Width of the pooled span-content vector (`content_dim` for base-v1;
    /// zero when `enable_span_content` is false).
    pub content_output_dim: usize,
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
    /// `nn.Linear(2 * pair_dim, 1)`; `None` when the feature is off.
    endpoint_difference: Option<(Weight<'a>, Vec<f32>)>,
    inside_weight: InsideWeight<'a>,
    content_pooler: Option<SpanContentPooler<'a>>,
    /// `nn.Linear(query_dim, content_output_dim)`.
    content_query: Option<(Weight<'a>, Vec<f32>)>,
    /// `nn.Linear(content_output_dim, 1)`.
    content_bias: Option<(Weight<'a>, Vec<f32>)>,
    rotary: Option<RotaryBoundaryEmbedding>,
}

impl<'a> PairScorer<'a> {
    /// Load the pair scorer. `hidden_size` is the encoder width, which is both
    /// the `content_pooler.value_projection` input and (for base-v1) the
    /// query width.
    pub fn load(source: &'a dyn TensorSource, hidden_size: usize) -> Result<Self, String> {
        let settings = load_settings(source)?.ok_or_else(|| {
            "missing gliner2.boundary.* settings metadata; re-convert the checkpoint with \
             tools/converter/gliner/convert_boundary.py so the pair scorer's feature flags \
             come from the config instead of defaults"
                .to_string()
        })?;
        let features = settings.features;
        let pair_dim = source
            .metadata("gliner2.boundary.pair_dim")
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .ok_or("missing metadata gliner2.boundary.pair_dim")?;
        let q_dim = source
            .tensor_info("boundary_head.pair_scorer.query_gate.weight")
            .ok_or("missing boundary_head.pair_scorer.query_gate.weight")?
            .dims[0] as usize;
        let boundary_dim = source
            .tensor_info("boundary_head.pair_scorer.start_endpoint_projection.weight")
            .ok_or("missing boundary_head.pair_scorer.start_endpoint_projection.weight")?
            .dims[1] as usize;
        // `compat_mix` is `nn.Linear(multihead_pair_compat_heads, 1)`, so its
        // input width is the authoritative head count.
        let multihead_pair_compat_heads = source
            .tensor_info("boundary_head.pair_scorer.compat_mix.weight")
            .map(|info| info.dims[0] as usize)
            .unwrap_or(settings.multihead_pair_compat_heads);
        if pair_dim % multihead_pair_compat_heads != 0 {
            return Err(format!(
                "pair_dim {pair_dim} not divisible by multihead_pair_compat_heads \
                 {multihead_pair_compat_heads}"
            ));
        }

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
        // With rotary endpoints the gate is half-width and gets
        // `repeat_interleave(2)` before use (scoring.py:208-209).
        let gate_dim = if features.enable_rotary_endpoints {
            pair_dim / 2
        } else {
            pair_dim
        };
        let query_gate = load_weight(
            source,
            "boundary_head.pair_scorer.query_gate.weight",
            q_dim,
            gate_dim,
        )?;
        let query_gate_bias = load_vec(
            source,
            "boundary_head.pair_scorer.query_gate.bias",
            gate_dim,
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

        let endpoint_difference = if features.endpoint_difference_features {
            Some((
                load_weight(
                    source,
                    "boundary_head.pair_scorer.endpoint_difference_projection.weight",
                    2 * pair_dim,
                    1,
                )?,
                load_vec(
                    source,
                    "boundary_head.pair_scorer.endpoint_difference_projection.bias",
                    1,
                )?,
            ))
        } else {
            None
        };

        let inside_weight = if features.query_conditioned_inside_weight {
            InsideWeight::QueryConditioned {
                weight: load_weight(
                    source,
                    "boundary_head.pair_scorer.inside_weight.weight",
                    q_dim,
                    1,
                )?,
                bias: load_vec(source, "boundary_head.pair_scorer.inside_weight.bias", 1)?,
            }
        } else {
            // `nn.Parameter(torch.tensor(1.0))` in the reference, so the
            // state dict holds a bare 1-element tensor.
            InsideWeight::Scalar(load_vec(source, "boundary_head.pair_scorer.inside_weight", 1)?[0])
        };

        let content_pooler = if features.enable_span_content {
            Some(SpanContentPooler::load(
                source,
                hidden_size,
                settings.content_dim,
                features.content_soft_max_pool,
            )?)
        } else {
            None
        };
        let content_output_dim = content_pooler
            .as_ref()
            .map_or(0, |pooler| pooler.output_dim);
        let content_query = if features.enable_span_content {
            Some((
                load_weight(
                    source,
                    "boundary_head.pair_scorer.content_query_projection.weight",
                    q_dim,
                    content_output_dim,
                )?,
                load_vec(
                    source,
                    "boundary_head.pair_scorer.content_query_projection.bias",
                    content_output_dim,
                )?,
            ))
        } else {
            None
        };
        let content_bias = if features.enable_span_content {
            Some((
                load_weight(
                    source,
                    "boundary_head.pair_scorer.content_bias.weight",
                    content_output_dim,
                    1,
                )?,
                load_vec(source, "boundary_head.pair_scorer.content_bias.bias", 1)?,
            ))
        } else {
            None
        };

        let rotary = if features.enable_rotary_endpoints {
            Some(RotaryBoundaryEmbedding::new(
                boundary_dim,
                settings.rotary_base,
            ))
        } else {
            None
        };

        Ok(Self {
            features,
            boundary_dim,
            pair_dim,
            query_dim: q_dim,
            content_output_dim,
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
            endpoint_difference,
            inside_weight,
            content_pooler,
            content_query,
            content_bias,
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
                let state = &boundary_states
                    [b * boundary_len * self.boundary_dim + i * self.boundary_dim..]
                    [..self.boundary_dim];
                apply_linear_full(
                    state,
                    &self.start_endpoint,
                    &self.start_endpoint_bias,
                    &mut start_all[b * boundary_len * self.pair_dim + i * self.pair_dim..]
                        [..self.pair_dim],
                );
                apply_linear_full(
                    state,
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

    /// Compute the per-(B, Q, C) pair logit.
    pub fn forward(&self, input: &PairScoreInputs<'_>) -> Vec<f32> {
        let PairScoreInputs {
            boundary_states,
            boundary_len,
            query_states,
            start_logits,
            end_logits,
            compat_logits,
            indices,
            valid_mask,
            inside_prefix,
            inside_prefix_mean,
            text_states,
            text_mask,
            text_lengths,
            batch,
            q_count,
            c,
        } = *input;
        if batch == 0 || q_count == 0 || c == 0 || boundary_len == 0 {
            return Vec::new();
        }
        let seq_len = text_mask.first().map_or(0, Vec::len);
        let total = batch * q_count * c;
        let compat_head_dim = self.pair_dim / self.multihead_pair_compat_heads;
        let scale = 1.0 / (self.pair_dim as f32).sqrt();
        // `gather_*` clamps candidate indices into range; the reference relies
        // on that instead of erroring on an out-of-range candidate.
        let last_boundary = boundary_len - 1;
        // Split the interleaved `[B, Q, C, 2]` index tensor once.
        let starts: Vec<usize> = indices.iter().step_by(2).copied().take(total).collect();
        let ends: Vec<usize> = indices
            .iter()
            .skip(1)
            .step_by(2)
            .copied()
            .take(total)
            .collect();

        // 1. Project every boundary position to pair_dim, then gather the
        //    candidate endpoints. The reference always takes this path for
        //    explicit spans: `proposals.score_start_states` is only populated
        //    by the document-level proposer, never by `score_explicit_spans`.
        let (start_all, end_all) = self.project_endpoints(boundary_states, boundary_len, batch);
        let mut s_proj = vec![0.0f32; total * self.pair_dim];
        let mut e_proj = vec![0.0f32; total * self.pair_dim];
        for idx in 0..total {
            let b = idx / (q_count * c);
            let s_off =
                b * boundary_len * self.pair_dim + starts[idx].min(last_boundary) * self.pair_dim;
            let e_off =
                b * boundary_len * self.pair_dim + ends[idx].min(last_boundary) * self.pair_dim;
            s_proj[idx * self.pair_dim..][..self.pair_dim]
                .copy_from_slice(&start_all[s_off..][..self.pair_dim]);
            e_proj[idx * self.pair_dim..][..self.pair_dim]
                .copy_from_slice(&end_all[e_off..][..self.pair_dim]);
        }

        // 2. Endpoint compatibility: per-head sum of (s * gate) * e, mixed
        //    across heads and scaled by 1/sqrt(pair_dim).
        let mut score = vec![0.0f32; total];
        let mut gate = vec![0.0f32; self.pair_dim];
        let mut difference = vec![0.0f32; 2 * self.pair_dim];
        for b in 0..batch {
            for q in 0..q_count {
                let qi = b * q_count + q;
                let mut gate_half = vec![0.0f32; self.query_gate_bias.len()];
                apply_linear_full(
                    &query_states[qi * self.query_dim..][..self.query_dim],
                    &self.query_gate,
                    &self.query_gate_bias,
                    &mut gate_half,
                );
                for v in gate_half.iter_mut() {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
                if self.rotary.is_some() {
                    // `repeat_interleave(2, dim=-1)`.
                    for (k, v) in gate_half.iter().enumerate() {
                        gate[2 * k] = *v;
                        gate[2 * k + 1] = *v;
                    }
                } else {
                    gate.copy_from_slice(&gate_half);
                }

                for ci in 0..c {
                    let idx = qi * c + ci;
                    if self.features.reranker_endpoint_compat {
                        let mut compat_logit = 0.0f32;
                        for h in 0..self.multihead_pair_compat_heads {
                            let mut sum = 0.0f32;
                            for k in 0..compat_head_dim {
                                let d = h * compat_head_dim + k;
                                let sv = s_proj[idx * self.pair_dim + d];
                                let ev = e_proj[idx * self.pair_dim + d];
                                // Reference eval order: (s * gate) * e.
                                sum += (sv * gate[d]) * ev;
                            }
                            compat_logit += self.compat_mix_at(h) * sum;
                        }
                        score[idx] = compat_logit * scale + self.compat_mix_bias[0];
                    }

                    // 3. Endpoint difference: Linear(cat(s - e, |s - e|)),
                    //    so the projection's input width is 2 * pair_dim.
                    if let Some((weight, bias)) = &self.endpoint_difference {
                        for d in 0..self.pair_dim {
                            let delta =
                                s_proj[idx * self.pair_dim + d] - e_proj[idx * self.pair_dim + d];
                            difference[d] = delta;
                            difference[self.pair_dim + d] = delta.abs();
                        }
                        score[idx] += dot_row(weight, 2 * self.pair_dim, &difference, bias, 0);
                    }
                }
            }
        }

        // 4. Marginals, gathered once at the candidate endpoints.
        for idx in 0..total {
            let q = (idx / c) % q_count;
            let b = idx / (q_count * c);
            let a = start_logits
                [b * q_count * boundary_len + q * boundary_len + starts[idx].min(last_boundary)];
            let bmarg = end_logits
                [b * q_count * boundary_len + q * boundary_len + ends[idx].min(last_boundary)];
            score[idx] += a + bmarg;
        }

        // 5. Prior: `torch.where(valid, compat_logits, 0)`.
        for idx in 0..total {
            if valid_mask.get(idx).copied().unwrap_or(false) {
                score[idx] += compat_logits.get(idx).copied().unwrap_or(0.0);
            }
        }

        // 6. Span content.
        if let (Some(pooler), Some((content_query, content_query_bias))) =
            (&self.content_pooler, &self.content_query)
        {
            let width = self.content_output_dim;
            let flat_mask: Vec<bool> = text_mask.iter().flatten().copied().collect();
            let mean_prefix = pooler.build_prefix(text_states, &flat_mask, batch, seq_len);
            let span_content = pooler.pool(&mean_prefix, &starts, &ends, q_count * c, seq_len);
            let content_scale = 1.0 / (width as f32).sqrt();
            for b in 0..batch {
                for q in 0..q_count {
                    let qi = b * q_count + q;
                    let mut coefficient = vec![0.0f32; width];
                    apply_linear_full(
                        &query_states[qi * self.query_dim..][..self.query_dim],
                        content_query,
                        content_query_bias,
                        &mut coefficient,
                    );
                    for ci in 0..c {
                        let idx = qi * c + ci;
                        let row = &span_content[idx * width..][..width];
                        let mut dot = 0.0f32;
                        for (value, coefficient_value) in row.iter().zip(coefficient.iter()) {
                            dot += value * coefficient_value;
                        }
                        score[idx] += dot * content_scale;
                    }
                }
            }
            if let Some((content_bias, content_bias_vec)) = &self.content_bias {
                for idx in 0..total {
                    let row = &span_content[idx * width..][..width];
                    score[idx] += dot_row(content_bias, width, row, content_bias_vec, 0);
                }
            }
        }

        // 7. Inside evidence. `interval` restores the raw inside-logit sum:
        // the prefix is mean-centered, and `inside_prefix_mean * width` adds
        // the mean back (scoring.py:36-54).
        if self.features.use_inside_evidence && !inside_prefix.is_empty() {
            for b in 0..batch {
                for q in 0..q_count {
                    let qi = b * q_count + q;
                    let prefix_base = qi * (seq_len + 1);
                    let mean = inside_prefix_mean.get(qi).copied().unwrap_or(0.0);
                    let weight = match &self.inside_weight {
                        InsideWeight::Scalar(value) => *value,
                        InsideWeight::QueryConditioned { weight, bias } => {
                            let mut out = [0.0f32; 1];
                            apply_linear_full(
                                &query_states[qi * self.query_dim..][..self.query_dim],
                                weight,
                                bias,
                                &mut out,
                            );
                            out[0]
                        }
                    };
                    for ci in 0..c {
                        let idx = qi * c + ci;
                        let start_idx = starts[idx].min(seq_len);
                        let end_idx = ends[idx].min(seq_len);
                        let p_end = inside_prefix[prefix_base + end_idx];
                        let p_start = inside_prefix[prefix_base + start_idx];
                        let width = end_idx as f32 - start_idx as f32;
                        let interval = p_end - p_start + mean * width;
                        score[idx] += weight * (interval / width.max(1.0).sqrt());
                    }
                }
            }
        }

        // 8. Continuous length features.
        for b in 0..batch {
            let tl = text_lengths.get(b).copied().unwrap_or(1).max(1) as f32;
            for q in 0..q_count {
                let length_coeff = self.length_query_at(query_states, q_count, q, b);
                for ci in 0..c {
                    let idx = b * q_count * c + q * c + ci;
                    let length = (ends[idx] as f32 - starts[idx] as f32).max(1.0);
                    let f1 = (length + 1.0).ln(); // log1p
                    let f2 = length / tl;
                    let f3 = 1.0 / length.sqrt();
                    score[idx] +=
                        f1 * length_coeff[0] + f2 * length_coeff[1] + f3 * length_coeff[2];
                }
            }
        }

        // 9. Mask invalid candidates.
        for (idx, slot) in score.iter_mut().enumerate() {
            if !valid_mask.get(idx).copied().unwrap_or(false) {
                *slot = MASK_LOGIT;
            }
        }

        score
    }

    /// `compat_mix.weight[h]` (init = 1 / multihead_pair_compat_heads).
    fn compat_mix_at(&self, head: usize) -> f32 {
        match self.compat_mix.kernel.f32_slice() {
            Some(rows) => rows[head],
            None => 0.0,
        }
    }

    /// `length_query_projection.weight @ query_state + bias` for (b, q).
    fn length_query_at(
        &self,
        query_states: &[f32],
        q_count: usize,
        q: usize,
        b: usize,
    ) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        apply_linear_full(
            &query_states[b * q_count * self.query_dim + q * self.query_dim..][..self.query_dim],
            &self.length_query,
            &self.length_query_bias,
            &mut out,
        );
        out
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Settings the loader needs to build [`PairScorerFeatures`], plus the dims
/// and rotary base it validates tensors against.
struct ScorerSettings {
    features: PairScorerFeatures,
    content_dim: usize,
    multihead_pair_compat_heads: usize,
    rotary_base: f32,
}

/// Read the transcoded `boundary_head` settings. `Ok(None)` means the GGUF
/// predates the settings metadata, which the caller turns into a
/// re-conversion hint rather than a silent default.
fn load_settings(source: &dyn TensorSource) -> Result<Option<ScorerSettings>, String> {
    let flag = |name: &str| -> Result<Option<bool>, String> {
        match source.metadata(&format!("gliner2.boundary.{name}")) {
            None => Ok(None),
            Some(MetaValue::Bool(value)) => Ok(Some(*value)),
            Some(MetaValue::Uint32(value)) => Ok(Some(*value != 0)),
            Some(MetaValue::Int32(value)) => Ok(Some(*value != 0)),
            Some(MetaValue::Uint64(value)) => Ok(Some(*value != 0)),
            Some(MetaValue::Int64(value)) => Ok(Some(*value != 0)),
            Some(other) => Err(format!("gliner2.boundary.{name} is not a bool: {other:?}")),
        }
    };
    let number = |name: &str| -> Result<Option<f64>, String> {
        match source.metadata(&format!("gliner2.boundary.{name}")) {
            None => Ok(None),
            Some(value) => value
                .to_f64()
                .map(Some)
                .ok_or_else(|| format!("gliner2.boundary.{name} is not a number: {value:?}")),
        }
    };
    if source
        .metadata("gliner2.boundary.use_inside_evidence")
        .is_none()
    {
        return Ok(None);
    }
    let required_flag = |name: &str| -> Result<bool, String> {
        flag(name)?.ok_or_else(|| format!("missing metadata gliner2.boundary.{name}"))
    };
    let required_usize = |name: &str| -> Result<usize, String> {
        number(name)?
            .map(|value| value as usize)
            .ok_or_else(|| format!("missing metadata gliner2.boundary.{name}"))
    };
    Ok(Some(ScorerSettings {
        features: PairScorerFeatures {
            use_inside_evidence: required_flag("use_inside_evidence")?,
            enable_span_content: required_flag("enable_span_content")?,
            content_soft_max_pool: required_flag("content_soft_max_pool")?,
            query_conditioned_inside_weight: required_flag("query_conditioned_inside_weight")?,
            endpoint_difference_features: required_flag("endpoint_difference_features")?,
            enable_rotary_endpoints: required_flag("enable_rotary_endpoints")?,
            reranker_endpoint_compat: required_flag("reranker_endpoint_compat")?,
        },
        content_dim: required_usize("content_dim")?,
        multihead_pair_compat_heads: required_usize("multihead_pair_compat_heads")?,
        rotary_base: number("rotary_base")?.unwrap_or(10000.0) as f32,
    }))
}

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

/// `weight @ input + bias[out]` for the `out`-th output row. Single-row
/// projections (`content_bias`, `endpoint_difference_projection`,
/// `inside_weight`) reuse the same F32 dot as the wide ones so the rounding
/// matches the reference's `nn.Linear`.
///
/// `n_in` is the projection's *declared* input width, which is not always
/// `input.len()`: `endpoint_difference_projection` takes the concatenated
/// `2 * pair_dim` difference vector.
fn dot_row(weight: &Weight<'_>, n_in: usize, input: &[f32], bias: &[f32], out: usize) -> f32 {
    if let Some(rows) = weight.kernel.f32_slice() {
        crate::ops::dot_f32(&rows[out * n_in..][..n_in], input, n_in) + bias[out]
    } else {
        let mut tmp = vec![0.0f32; weight.n_out];
        weight
            .kernel
            .forward(input, &mut tmp, weight.n_in, weight.n_out);
        tmp[out] + bias[out]
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
