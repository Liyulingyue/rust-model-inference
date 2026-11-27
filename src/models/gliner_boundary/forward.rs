//! `BoundaryEncoder.forward`: project text states into per-boundary states.
//!
//! Mirrors `gliner2.models.boundary.encoding.BoundaryEncoder.forward`
//! (`target/gliner2-oracle/gliner2/models/boundary/encoding.py:185-205`)
//! minus dropout (no inference-time dropout, matching the reference's
//! `model.eval()` path).
//!
//! Pipeline:
//!  1. shift text states left with BOS state, right with EOS state
//!  2. left/right projection into `boundary_dim`
//!  3. concat + output projection + LayerNorm
//!  4. optional boundary self-attention blocks (pre-norm + window)
//!  5. optional refinement blocks (SwiGLU residual)
//!  6. zero out padding boundary rows so they can't leak via numerical noise

use crate::core::tensor::TensorSource;
use crate::models::gliner_boundary::tensor_util::{apply_linear_full, load_vec, load_weight};
use crate::ops::kernel::Weight;

/// Output of `BoundaryEncoder.forward`: per-boundary states + a validity mask
/// so downstream code can mask out padding-boundary rows cheaply.
pub struct BoundaryEncoding {
    /// `[B, L + 1, boundary_dim]` row-major.
    pub states: Vec<f32>,
    /// `[B][L + 1]` boundary validity mask (per-batch row).
    pub mask: Vec<Vec<bool>>,
    pub boundary_dim: usize,
    pub seq_len: usize,
    pub batch: usize,
}

#[derive(Debug, Clone)]
pub struct Norm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// BoundaryEncoder weights + config derived from the GGUF. Stored by
/// `BoundaryModel` and run by `boundary_encoder::forward`.
pub struct BoundaryEncoder<'a> {
    pub left_projection: Weight<'a>,
    pub left_bias: Vec<f32>,
    pub right_projection: Weight<'a>,
    pub right_bias: Vec<f32>,
    pub output_projection: Weight<'a>,
    pub output_bias: Vec<f32>,
    pub layer_norm: Norm,
    pub bos_state: Vec<f32>,
    pub eos_state: Vec<f32>,
    pub attention_blocks: Vec<BoundaryAttentionBlock<'a>>,
    pub refinement_blocks: Vec<ResidualSwiGLU<'a>>,
    pub hidden_size: usize,
    pub boundary_dim: usize,
    pub ffn_multiplier: f32,
}

pub struct BoundaryAttentionBlock<'a> {
    pub norm: Norm,
    pub qkv_projection: Weight<'a>,
    pub qkv_bias: Vec<f32>,
    pub output_projection: Weight<'a>,
    pub output_bias: Vec<f32>,
    pub num_heads: usize,
    pub window: usize,
}

pub struct ResidualSwiGLU<'a> {
    pub norm: Norm,
    pub input_projection: Weight<'a>,
    pub input_bias: Vec<f32>,
    pub output_projection: Weight<'a>,
    pub output_bias: Vec<f32>,
    pub hidden_dim: usize,
    pub boundary_dim: usize,
}

impl<'a> BoundaryEncoder<'a> {
    /// `attention_window` is `boundary_head.boundary_attention_window`: the
    /// local attention band `|i - j| <= window` applied to every attention
    /// block. base-v1 uses 128, which is a no-op for documents shorter than
    /// 257 boundary positions — i.e. it only binds on long documents, so a
    /// short fixture will not catch a missing window.
    pub fn load(
        source: &'a dyn TensorSource,
        hidden_size: usize,
        attention_window: usize,
    ) -> Result<Self, String> {
        let boundary_dim = source
            .tensor_info("boundary_head.boundary_encoder.layer_norm.weight")
            .ok_or("missing boundary_head.boundary_encoder.layer_norm.weight")?
            .dims[0] as usize;

        let left_projection = load_weight(
            source,
            "boundary_head.boundary_encoder.left_projection.weight",
            hidden_size,
            boundary_dim,
        )?;
        let left_bias = load_vec(
            source,
            "boundary_head.boundary_encoder.left_projection.bias",
            boundary_dim,
        )?;
        let right_projection = load_weight(
            source,
            "boundary_head.boundary_encoder.right_projection.weight",
            hidden_size,
            boundary_dim,
        )?;
        let right_bias = load_vec(
            source,
            "boundary_head.boundary_encoder.right_projection.bias",
            boundary_dim,
        )?;
        let output_projection = load_weight(
            source,
            "boundary_head.boundary_encoder.output_projection.weight",
            2 * boundary_dim,
            boundary_dim,
        )?;
        let output_bias = load_vec(
            source,
            "boundary_head.boundary_encoder.output_projection.bias",
            boundary_dim,
        )?;

        let layer_norm = Norm {
            weight: load_vec(
                source,
                "boundary_head.boundary_encoder.layer_norm.weight",
                boundary_dim,
            )?,
            bias: load_vec(
                source,
                "boundary_head.boundary_encoder.layer_norm.bias",
                boundary_dim,
            )?,
        };

        let bos_state = load_vec(
            source,
            "boundary_head.boundary_encoder.bos_state",
            hidden_size,
        )?;
        let eos_state = load_vec(
            source,
            "boundary_head.boundary_encoder.eos_state",
            hidden_size,
        )?;

        // Attention blocks: scan ``boundary_head.boundary_encoder.attention_blocks.{i}.*``
        let mut attention_blocks = Vec::new();
        let mut attn_index = 0usize;
        loop {
            let qkv_name = format!(
                "boundary_head.boundary_encoder.attention_blocks.{attn_index}.qkv_projection.weight"
            );
            if source.tensor_info(&qkv_name).is_none() {
                break;
            }
            let head_dim = boundary_dim / attention_num_heads(source, attn_index).unwrap_or(4);
            // The qkv projection is `(3 * boundary_dim, boundary_dim)`.
            // We don't actually need head_dim here — the loader just reads
            // the weight matrix. The forward picks num_heads from config
            // below; the head_dim derivation lives in `forward_attention`.
            let _ = head_dim;
            let num_heads = attention_num_heads(source, attn_index)
                .ok_or_else(|| format!("missing attention_blocks.{attn_index}.qkv.bias"))?;
            attention_blocks.push(BoundaryAttentionBlock {
                norm: Norm {
                    weight: load_vec(
                        source,
                        &format!(
                            "boundary_head.boundary_encoder.attention_blocks.{attn_index}.norm.weight"
                        ),
                        boundary_dim,
                    )?,
                    bias: load_vec(
                        source,
                        &format!(
                            "boundary_head.boundary_encoder.attention_blocks.{attn_index}.norm.bias"
                        ),
                        boundary_dim,
                    )?,
                },
                qkv_projection: load_weight(
                    source,
                    &format!(
                        "boundary_head.boundary_encoder.attention_blocks.{attn_index}.qkv_projection.weight"
                    ),
                    boundary_dim,
                    3 * boundary_dim,
                )?,
                qkv_bias: load_vec(
                    source,
                    &format!(
                        "boundary_head.boundary_encoder.attention_blocks.{attn_index}.qkv_projection.bias"
                    ),
                    3 * boundary_dim,
                )?,
                output_projection: load_weight(
                    source,
                    &format!(
                        "boundary_head.boundary_encoder.attention_blocks.{attn_index}.output_projection.weight"
                    ),
                    boundary_dim,
                    boundary_dim,
                )?,
                output_bias: load_vec(
                    source,
                    &format!(
                        "boundary_head.boundary_encoder.attention_blocks.{attn_index}.output_projection.bias"
                    ),
                    boundary_dim,
                )?,
                num_heads,
                window: attention_window,
            });
            attn_index += 1;
        }

        // Refinement blocks: SwiGLU residual with `boundary_ffn_multiplier`
        // hidden_dim. base-v1 has multiplier=2.0 and 1 refinement block, so
        // hidden_dim = 128 * 2 = 256.
        let ffn_multiplier = source
            .metadata("gliner2.boundary.ffn_multiplier")
            .and_then(|v| v.to_f64())
            .map(|v| v as f32)
            .unwrap_or(2.0);
        let hidden_dim = ((boundary_dim as f32) * ffn_multiplier).round() as usize;

        let mut refinement_blocks = Vec::new();
        let mut ref_index = 0usize;
        loop {
            let input_name = format!(
                "boundary_head.boundary_encoder.refinement_blocks.{ref_index}.input_projection.weight"
            );
            if source.tensor_info(&input_name).is_none() {
                break;
            }
            refinement_blocks.push(ResidualSwiGLU {
                norm: Norm {
                    weight: load_vec(
                        source,
                        &format!(
                            "boundary_head.boundary_encoder.refinement_blocks.{ref_index}.norm.weight"
                        ),
                        boundary_dim,
                    )?,
                    bias: load_vec(
                        source,
                        &format!(
                            "boundary_head.boundary_encoder.refinement_blocks.{ref_index}.norm.bias"
                        ),
                        boundary_dim,
                    )?,
                },
                input_projection: load_weight(
                    source,
                    &format!(
                        "boundary_head.boundary_encoder.refinement_blocks.{ref_index}.input_projection.weight"
                    ),
                    boundary_dim,
                    2 * hidden_dim,
                )?,
                input_bias: load_vec(
                    source,
                    &format!(
                        "boundary_head.boundary_encoder.refinement_blocks.{ref_index}.input_projection.bias"
                    ),
                    2 * hidden_dim,
                )?,
                output_projection: load_weight(
                    source,
                    &format!(
                        "boundary_head.boundary_encoder.refinement_blocks.{ref_index}.output_projection.weight"
                    ),
                    hidden_dim,
                    boundary_dim,
                )?,
                output_bias: load_vec(
                    source,
                    &format!(
                        "boundary_head.boundary_encoder.refinement_blocks.{ref_index}.output_projection.bias"
                    ),
                    boundary_dim,
                )?,
                hidden_dim,
                boundary_dim,
            });
            ref_index += 1;
        }

        Ok(Self {
            left_projection,
            left_bias,
            right_projection,
            right_bias,
            output_projection,
            output_bias,
            layer_norm,
            bos_state,
            eos_state,
            attention_blocks,
            refinement_blocks,
            hidden_size,
            boundary_dim,
            ffn_multiplier,
        })
    }

    /// Forward pass on a batch of text states `[B, L, hidden_size]`.
    /// `text_mask[b][i] = true` for valid text tokens.
    pub fn forward(&self, text_states: &[f32], text_mask: &[Vec<bool>]) -> BoundaryEncoding {
        let batch = text_mask.len();
        let seq_len = if batch == 0 { 0 } else { text_mask[0].len() };
        let boundary_len = seq_len + 1;
        let hidden_size = self.hidden_size;
        let boundary_dim = self.boundary_dim;

        // 1. shift left with BOS, shift right with EOS in *hidden_size*
        // space (matches the reference: bos_state / eos_state are stored
        // at hidden_size and only enter boundary_dim via the projection).
        let mut left = vec![0.0f32; batch * boundary_len * hidden_size];
        let mut right = vec![0.0f32; batch * boundary_len * hidden_size];
        let mut text_lengths: Vec<usize> = vec![0; batch];
        for b in 0..batch {
            let mut len = 0usize;
            for (i, &m) in text_mask[b].iter().enumerate() {
                if m {
                    len = i + 1;
                }
            }
            text_lengths[b] = len;
            // left[b, 0] = bos_state (hidden_size)
            let left_row_0 = &mut left[b * boundary_len * hidden_size..][..hidden_size];
            left_row_0.copy_from_slice(&self.bos_state);
            // left[b, 1..len+1] = text_states[b, 0..len]
            let left_body = &mut left[b * boundary_len * hidden_size + hidden_size..];
            for i in 0..len {
                left_body[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(
                    &text_states[b * seq_len * hidden_size + i * hidden_size..][..hidden_size],
                );
            }
            // right[b, 0..len] = text_states[b, 0..len]
            let right_body = &mut right[b * boundary_len * hidden_size..];
            for i in 0..len {
                right_body[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(
                    &text_states[b * seq_len * hidden_size + i * hidden_size..][..hidden_size],
                );
            }
            // right[b, len] = eos_state (per-sample at the sample's final
            // valid boundary index, mirroring `shift_right_with_eos`).
            let right_eos_at = len.min(boundary_len - 1);
            let right_eos = &mut right_body[right_eos_at * hidden_size..][..hidden_size];
            right_eos.copy_from_slice(&self.eos_state);
        }

        // 2. left / right projection (linear: hidden_size -> boundary_dim)
        let mut left_p = vec![0.0f32; batch * boundary_len * boundary_dim];
        let mut right_p = vec![0.0f32; batch * boundary_len * boundary_dim];
        for b in 0..batch {
            for i in 0..boundary_len {
                apply_linear_full(
                    &left[b * boundary_len * hidden_size + i * hidden_size..][..hidden_size],
                    &self.left_projection,
                    &self.left_bias,
                    &mut left_p[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim],
                );
                apply_linear_full(
                    &right[b * boundary_len * hidden_size + i * hidden_size..][..hidden_size],
                    &self.right_projection,
                    &self.right_bias,
                    &mut right_p[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim],
                );
            }
        }

        // 3. concat + output projection + LayerNorm
        let mut states = vec![0.0f32; batch * boundary_len * boundary_dim];
        for b in 0..batch {
            for i in 0..boundary_len {
                let mut combined = vec![0.0f32; 2 * boundary_dim];
                combined[..boundary_dim].copy_from_slice(
                    &left_p[b * boundary_len * boundary_dim + i * boundary_dim..][..boundary_dim],
                );
                combined[boundary_dim..].copy_from_slice(
                    &right_p[b * boundary_len * boundary_dim + i * boundary_dim..][..boundary_dim],
                );
                apply_linear_full(
                    &combined,
                    &self.output_projection,
                    &self.output_bias,
                    &mut states[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim],
                );
            }
        }

        // LayerNorm per-row (post-attention norm is part of the attention
        // block, so the encoder's layer_norm is here).
        for b in 0..batch {
            for i in 0..boundary_len {
                let base = b * boundary_len * boundary_dim + i * boundary_dim;
                let row = states[base..base + boundary_dim].to_vec();
                let mut out = vec![0.0f32; boundary_dim];
                crate::ops::layer_norm(
                    &row,
                    &self.layer_norm.weight,
                    &self.layer_norm.bias,
                    1e-5,
                    &mut out,
                );
                states[base..base + boundary_dim].copy_from_slice(&out);
            }
        }

        // 4. attention blocks
        for block in &self.attention_blocks {
            run_attention_block(
                &mut states,
                block,
                batch,
                boundary_len,
                boundary_dim,
                &text_lengths,
            );
        }

        // 5. refinement blocks (SwiGLU)
        for block in &self.refinement_blocks {
            run_refinement_block(&mut states, block, batch, boundary_len);
        }

        // 6. mask: zero out padding boundary rows
        let mut mask: Vec<Vec<bool>> = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut row = vec![false; boundary_len];
            for i in 0..boundary_len {
                if i <= text_lengths[b] {
                    row[i] = true;
                }
            }
            mask.push(row);
        }
        for b in 0..batch {
            for i in 0..boundary_len {
                if i > text_lengths[b] {
                    let row = &mut states[b * boundary_len * boundary_dim + i * boundary_dim..]
                        [..boundary_dim];
                    for v in row.iter_mut() {
                        *v = 0.0;
                    }
                }
            }
        }

        BoundaryEncoding {
            states,
            mask,
            boundary_dim,
            seq_len,
            batch,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The `BoundaryAttentionBlock` attention mask (`encoding.py:122-131`):
///
/// ```text
/// allowed[i][j] = (mask[j] && (window == 0 || |i - j| <= window)) || (i == j)
/// ```
///
/// The diagonal OR is unconditional — the reference does *not* gate it on
/// `mask[i]`, which is what keeps a padding query row from having an entirely
/// masked row (and therefore a NaN softmax). Exposed so the mask can be
/// tested directly: `boundary_attention_window` is 128 for base-v1, so it only
/// starts excluding keys past 257 boundary positions, far beyond the
/// document lengths the end-to-end fixtures use.
pub fn attention_allowed(mask_row: &[bool], i: usize, j: usize, window: usize) -> bool {
    let in_window = window == 0 || i.abs_diff(j) <= window;
    (mask_row.get(j).copied().unwrap_or(false) && in_window) || i == j
}

/// `head_dim` for attention blocks is `boundary_dim / num_heads`. The
/// reference uses `boundary_attention_heads` from metadata (default 4).
///
/// This is an existence check, not a derivation: the QKV bias must be present
/// for the block to load at all, but its `(3 * boundary_dim,)` shape carries no
/// head count, so the answer is always the config default. The doc previously
/// claimed this "requires the QKV bias tensor to declare `3 * boundary_dim`",
/// which it never did — `run_attention_block` is what trips on a bad shape.
fn attention_num_heads(source: &dyn TensorSource, attn_index: usize) -> Option<usize> {
    let qkv_bias_name =
        format!("boundary_head.boundary_encoder.attention_blocks.{attn_index}.qkv_projection.bias");
    source.tensor_info(&qkv_bias_name).map(|_| 4)
}

/// Scalar `y = W @ x + b` for an F32 weight. The boundary projections
/// are stored as F32, so we use the same scalar fallback the Decide
/// encoder uses when the weight has an F32 kernel slice.
fn run_attention_block(
    states: &mut [f32],
    block: &BoundaryAttentionBlock<'_>,
    batch: usize,
    boundary_len: usize,
    boundary_dim: usize,
    text_lengths: &[usize],
) {
    // Pre-compute the validity mask for this batch so the softmax can
    // exclude invalid keys.
    let mask: Vec<bool> = (0..batch * boundary_len)
        .map(|idx| {
            let b = idx / boundary_len;
            let i = idx % boundary_len;
            i <= text_lengths[b]
        })
        .collect();
    let num_heads = block.num_heads;
    let head_dim = boundary_dim / num_heads;
    let total = batch * boundary_len * boundary_dim;

    // Pre-norm -> QKV
    let mut qkv = vec![0.0f32; batch * boundary_len * 3 * boundary_dim];
    for b in 0..batch {
        for i in 0..boundary_len {
            let row_in =
                &states[b * boundary_len * boundary_dim + i * boundary_dim..][..boundary_dim];
            let normed = apply_norm_static(row_in, &block.norm.weight, &block.norm.bias, 1e-5);
            // qkv_projection: (3 * boundary_dim, boundary_dim), bias: (3 * boundary_dim,)
            apply_linear_full(
                &normed,
                &block.qkv_projection,
                &block.qkv_bias,
                &mut qkv[b * boundary_len * 3 * boundary_dim + i * 3 * boundary_dim..]
                    [..3 * boundary_dim],
            );
        }
    }

    // Self-attention per (batch, head). Not causal (boundary attention is
    // bidirectional), and restricted to the local band `|i - j| <= window`
    // when `window > 0` (`BoundaryAttentionBlock.forward`, encoding.py:122).
    let mut output = vec![0.0f32; total];
    for b in 0..batch {
        for head in 0..num_heads {
            // gather query / key / value for this (batch, head)
            let mut q = vec![0.0f32; boundary_len * head_dim];
            let mut k = vec![0.0f32; boundary_len * head_dim];
            let mut v = vec![0.0f32; boundary_len * head_dim];
            for i in 0..boundary_len {
                let qkv_base = b * boundary_len * 3 * boundary_dim + i * 3 * boundary_dim;
                // q at index 0, k at boundary_dim, v at 2 * boundary_dim
                let head_offset = head * head_dim;
                q[i * head_dim..(i + 1) * head_dim]
                    .copy_from_slice(&qkv[qkv_base + head_offset..][..head_dim]);
                k[i * head_dim..(i + 1) * head_dim]
                    .copy_from_slice(&qkv[qkv_base + boundary_dim + head_offset..][..head_dim]);
                v[i * head_dim..(i + 1) * head_dim]
                    .copy_from_slice(&qkv[qkv_base + 2 * boundary_dim + head_offset..][..head_dim]);
            }
            // attention scores (boundary_len, boundary_len) — masked with
            // the boundary validity mask + a diagonal self-attend fallback
            // so padding rows still have at least one legal key.
            let mut scores = vec![0.0f32; boundary_len * boundary_len];
            let scale = 1.0 / (head_dim as f32).sqrt();
            for i in 0..boundary_len {
                for j in 0..boundary_len {
                    let mut s = 0.0f32;
                    for kk in 0..head_dim {
                        s += q[i * head_dim + kk] * k[j * head_dim + kk];
                    }
                    scores[i * boundary_len + j] = s * scale;
                }
            }
            // allowed[j] = mask[j] && |i - j| <= window, then OR the diagonal
            // unconditionally so a padding query row still has one legal key
            // (encoding.py:122-131). The unconditional OR matters: the
            // reference does not gate it on mask[i].
            let window = block.window;
            for i in 0..boundary_len {
                for j in 0..boundary_len {
                    let valid = attention_allowed(
                        &mask[b * boundary_len..(b + 1) * boundary_len],
                        i,
                        j,
                        window,
                    );
                    if !valid {
                        scores[i * boundary_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            // softmax per row
            for i in 0..boundary_len {
                let row = &mut scores[i * boundary_len..(i + 1) * boundary_len];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
            // attention output
            let mut head_out = vec![0.0f32; boundary_len * head_dim];
            for i in 0..boundary_len {
                for kk in 0..head_dim {
                    let mut s = 0.0f32;
                    for j in 0..boundary_len {
                        s += scores[i * boundary_len + j] * v[j * head_dim + kk];
                    }
                    head_out[i * head_dim + kk] = s;
                }
            }
            // scatter back to output [B, boundary_len, boundary_dim]
            for i in 0..boundary_len {
                output[b * boundary_len * boundary_dim + i * boundary_dim + head * head_dim..]
                    [..head_dim]
                    .copy_from_slice(&head_out[i * head_dim..(i + 1) * head_dim]);
            }
        }
    }

    // Output projection + residual
    let mut updated = vec![0.0f32; total];
    for b in 0..batch {
        for i in 0..boundary_len {
            let row_in =
                &output[b * boundary_len * boundary_dim + i * boundary_dim..][..boundary_dim];
            apply_linear_full(
                row_in,
                &block.output_projection,
                &block.output_bias,
                &mut updated[b * boundary_len * boundary_dim + i * boundary_dim..][..boundary_dim],
            );
        }
    }
    for b in 0..batch {
        for i in 0..boundary_len {
            for kk in 0..boundary_dim {
                let v = states[b * boundary_len * boundary_dim + i * boundary_dim + kk];
                let u = updated[b * boundary_len * boundary_dim + i * boundary_dim + kk];
                states[b * boundary_len * boundary_dim + i * boundary_dim + kk] = v + u;
            }
        }
    }
}

fn run_refinement_block(
    states: &mut [f32],
    block: &ResidualSwiGLU<'_>,
    batch: usize,
    boundary_len: usize,
) {
    let boundary_dim = block.boundary_dim;
    let total = batch * boundary_len * boundary_dim;
    let mut updated = vec![0.0f32; total];
    for b in 0..batch {
        for i in 0..boundary_len {
            let row_in =
                &states[b * boundary_len * boundary_dim + i * boundary_dim..][..boundary_dim];
            let normed = apply_norm_static(row_in, &block.norm.weight, &block.norm.bias, 1e-5);
            // input_projection: (2 * hidden_dim, boundary_dim)
            let mut gate_value = vec![0.0f32; 2 * block.hidden_dim];
            apply_linear_full(
                &normed,
                &block.input_projection,
                &block.input_bias,
                &mut gate_value,
            );
            let hidden_dim = block.hidden_dim;
            for h in 0..hidden_dim {
                let value = gate_value[h];
                let gate = gate_value[hidden_dim + h];
                gate_value[h] = value * crate::ops::silu(gate);
            }
            // output_projection: (boundary_dim, hidden_dim)
            let mut out = vec![0.0f32; boundary_dim];
            apply_linear_full(
                &gate_value[..hidden_dim],
                &block.output_projection,
                &block.output_bias,
                &mut out,
            );
            for kk in 0..boundary_dim {
                updated[b * boundary_len * boundary_dim + i * boundary_dim + kk] = out[kk];
            }
        }
    }
    for b in 0..batch {
        for i in 0..boundary_len {
            for kk in 0..boundary_dim {
                states[b * boundary_len * boundary_dim + i * boundary_dim + kk] +=
                    updated[b * boundary_len * boundary_dim + i * boundary_dim + kk];
            }
        }
    }
}

fn apply_norm_static(input: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    crate::ops::layer_norm(input, weight, bias, eps, &mut out);
    out
}
