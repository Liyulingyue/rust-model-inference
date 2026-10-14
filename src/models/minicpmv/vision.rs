//! Forward pass for the `minicpmv4_7` projector.
//!
//! Port of `clip_graph_minicpmv4_6::build` (shared by 4.6 and 4.7) at the
//! pinned oracle `69f201a2051ea9b9e9b50c3cc56afd9e2ae64414`,
//! `tools/mtmd/models/minicpmv.cpp`.
//!
//! Per tile of `nx * ny` patches:
//!
//! ```text
//! patch_embd -> + position_embd
//!   ViT blocks [0, insert_layer_id]
//!   vit_merger window attention (4x4 blocks) + residual
//!   vit_merger 2x2 downsample: mean -> concat -> ds_ln -> GELU MLP -> + mean
//!   ViT blocks [insert_layer_id + 1, n_layer)
//!   post_ln
//!   final merger: 2x2 concat -> mm.input_norm -> GELU_erf MLP
//! ```
//!
//! Numerics worth calling out, because they are what the oracle pins down:
//!
//! * The tower uses **LayerNorm** (`NORM_TYPE_NORMAL`), not RMSNorm.
//! * `build_vit_layers` is *pre*-norm with the attention output added back
//!   onto `input` (not onto the normed copy).
//! * The vit_merger's 2x2 downsample keeps a `mean` residual across the
//!   concat-MLP: `((p0+p1+p2+p3) * 0.25)` is added to the MLP output.
//! * The vit_merger MLP uses tanh GELU; only the final projector MLP uses
//!   erf GELU.
//! * The 2x2 gather order is `p0 p1 p2 p3` concatenated on the feature axis,
//!   which is row-major over `(row, col, quad)`.

use super::config::MiniCpmV47Config;
use super::weights::{MiniCpmV47Weights, ViTLayer};
use crate::core::thread_pool::ComputePool;
use crate::ops::activation::{gelu_erf, gelu_ggml_f32};
use std::sync::Arc;

/// Reusable buffers so `encode_tile` does not reallocate per call.
#[derive(Default)]
pub struct MiniCpmV47Scratch {
    hidden: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    scores: Vec<f32>,
    ff: Vec<f32>,
    merged: Vec<f32>,
    mean: Vec<f32>,
}

impl MiniCpmV47Scratch {
    fn fit(&mut self, tokens: usize, width: usize, ffn: usize) {
        let n = tokens * width;
        self.hidden.resize(n, 0.0);
        self.normed.resize(n, 0.0);
        self.q.resize(n, 0.0);
        self.k.resize(n, 0.0);
        self.v.resize(n, 0.0);
        self.attn.resize(n, 0.0);
        self.scores.resize(tokens * tokens, 0.0);
        self.ff.resize(tokens * ffn, 0.0);
        self.merged.resize(tokens * width * 4, 0.0);
        self.mean.resize(tokens * width * 4, 0.0);
    }
}

/// `W: [n_out][n_in]` times `input: [tokens][n_in]`, plus optional bias.
fn linear(
    pool: &ComputePool,
    w: &[f32],
    bias: Option<&[f32]>,
    input: &[f32],
    output: &mut [f32],
    tokens: usize,
    n_in: usize,
    n_out: usize,
) {
    let b = bias.unwrap_or(&[]);
    if tokens * n_out < 256 {
        for token in 0..tokens {
            let src = &input[token * n_in..(token + 1) * n_in];
            for out in 0..n_out {
                let value = crate::ops::dot_f32(&w[out * n_in..(out + 1) * n_in], src, n_in);
                output[token * n_out + out] = if b.is_empty() { value } else { value + b[out] };
            }
        }
        return;
    }
    let w_ptr = w.as_ptr();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let bias_ptr = b.as_ptr();
    let pool = pool.clone();
    pool.compute(move |ith, nth| {
        let (start, end) = crate::ops::kernel::f32::scalar::row_range(tokens, ith, nth);
        for token in start..end {
            let src = unsafe { std::slice::from_raw_parts(in_ptr.add(token * n_in), n_in) };
            for out in 0..n_out {
                let row = unsafe { std::slice::from_raw_parts(w_ptr.add(out * n_in), n_in) };
                let value = crate::ops::dot_f32(row, src, n_in);
                let bias = if bias_ptr.is_null() {
                    0.0
                } else {
                    unsafe { *bias_ptr.add(out) }
                };
                unsafe {
                    *out_ptr.add(token * n_out + out) = value + bias;
                }
            }
        }
    });
}

/// In-place softmax over one row.
fn softmax(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in row.iter() {
        max = max.max(v);
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// Single-head-group attention over `tokens` positions.
///
/// `window` selects the vit_merger pattern: when true, position `p` may only
/// attend to the 15 other positions in its own 4x4 block of the patch grid.
#[allow(clippy::too_many_arguments)]
fn attention(
    cfg: &MiniCpmV47Config,
    scores: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    tokens: usize,
    nx: usize,
    window: bool,
) {
    let (heads, d_head, dim) = (cfg.n_head, cfg.d_head, cfg.n_embd);
    // ggml builds `softmax(QK^T * kq_scale + mask)`.
    for token in 0..tokens {
        let q_row = &q[token * dim..(token + 1) * dim];
        for other in 0..tokens {
            let k_row = &k[other * dim..(other + 1) * dim];
            let mut dot = 0.0f32;
            for head in 0..heads {
                let qh = &q_row[head * d_head..(head + 1) * d_head];
                let kh = &k_row[head * d_head..(head + 1) * d_head];
                let mut acc = 0.0f32;
                for i in 0..d_head {
                    acc += qh[i] * kh[i];
                }
                dot += acc;
            }
            scores[token * tokens + other] = dot * cfg.kq_scale;
        }
    }
    if window && nx > 0 {
        // 2x2 blocks of the patch grid — the oracle reorders to window-major
        // with four tokens per window (`window_idx[k] = (2*wi)*pos_w + 2*wj + {0,1}`),
        // then masks everything outside the block diagonal.
        let win = 2usize;
        let ny = tokens.div_ceil(nx);
        for token in 0..tokens {
            let (ty, tx) = (token / nx, token % nx);
            let (wy, wx) = ((ty / win) * win, (tx / win) * win);
            for other in 0..tokens {
                let (oy, ox) = (other / nx, other % nx);
                let in_block = oy / win == wy / win && ox / win == wx / win;
                if !in_block {
                    scores[token * tokens + other] = f32::NEG_INFINITY;
                }
            }
        }
        let _ = ny;
    }
    for token in 0..tokens {
        softmax(&mut scores[token * tokens..(token + 1) * tokens]);
    }
    out[..tokens * dim].fill(0.0);
    for token in 0..tokens {
        let weights = &scores[token * tokens..(token + 1) * tokens];
        for other in 0..tokens {
            let w = weights[other];
            if w == 0.0 {
                continue;
            }
            let v_row = &v[other * dim..(other + 1) * dim];
            for i in 0..dim {
                out[token * dim + i] += w * v_row[i];
            }
        }
    }
}

fn vit_block(
    cfg: &MiniCpmV47Config,
    scratch: &mut MiniCpmV47Scratch,
    pool: &ComputePool,
    layer: &ViTLayer,
    hidden: &mut [f32],
    nx: usize,
) {
    let tokens = hidden.len() / cfg.n_embd;
    let dim = cfg.n_embd;
    scratch.fit(tokens, dim, cfg.n_ff);

    // attention, pre-norm
    let normed = &mut scratch.normed[..tokens * dim];
    for token in 0..tokens {
        crate::ops::norm::layer_norm(
            &hidden[token * dim..(token + 1) * dim],
            &layer.ln1_w,
            &layer.ln1_b,
            cfg.eps,
            &mut normed[token * dim..(token + 1) * dim],
        );
    }
    let (q, k, v, attn, scores) = (
        &mut scratch.q[..tokens * dim],
        &mut scratch.k[..tokens * dim],
        &mut scratch.v[..tokens * dim],
        &mut scratch.attn[..tokens * dim],
        &mut scratch.scores[..tokens * tokens],
    );
    linear(
        pool,
        &layer.q_w,
        Some(&layer.q_b),
        normed,
        q,
        tokens,
        dim,
        dim,
    );
    linear(
        pool,
        &layer.k_w,
        Some(&layer.k_b),
        normed,
        k,
        tokens,
        dim,
        dim,
    );
    linear(
        pool,
        &layer.v_w,
        Some(&layer.v_b),
        normed,
        v,
        tokens,
        dim,
        dim,
    );
    attention(cfg, scores, q, k, v, attn, tokens, nx, false);
    let mut proj = vec![0.0f32; tokens * dim];
    linear(
        pool,
        &layer.o_w,
        Some(&layer.o_b),
        attn,
        &mut proj,
        tokens,
        dim,
        dim,
    );
    // residual is added onto `input`, not the normed copy
    for i in 0..tokens * dim {
        hidden[i] += proj[i];
    }

    // ffn, pre-norm
    let normed = &mut scratch.normed[..tokens * dim];
    for token in 0..tokens {
        crate::ops::norm::layer_norm(
            &hidden[token * dim..(token + 1) * dim],
            &layer.ln2_w,
            &layer.ln2_b,
            cfg.eps,
            &mut normed[token * dim..(token + 1) * dim],
        );
    }
    let ff = &mut scratch.ff[..tokens * cfg.n_ff];
    linear(
        pool,
        &layer.ff_up_w,
        Some(&layer.ff_up_b),
        normed,
        ff,
        tokens,
        dim,
        cfg.n_ff,
    );
    for value in ff.iter_mut() {
        *value = gelu_ggml_f32(*value);
    }
    let mut out = vec![0.0f32; tokens * dim];
    linear(
        pool,
        &layer.ff_down_w,
        Some(&layer.ff_down_b),
        ff,
        &mut out,
        tokens,
        cfg.n_ff,
        dim,
    );
    for i in 0..tokens * dim {
        hidden[i] += out[i];
    }
}

/// Gather indices for a 2x2 downsample over a `nx * ny` grid.
fn ds_indices(nx: usize, ny: usize) -> Vec<[usize; 4]> {
    let (hx, hy) = (nx / 2, ny / 2);
    let mut out = Vec::with_capacity(hx * hy);
    for row in 0..hy {
        for col in 0..hx {
            let base = (row * 2) * nx + col * 2;
            out.push([base, base + 1, base + nx, base + nx + 1]);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn encode_tile(
    weights: &MiniCpmV47Weights,
    scratch: &mut MiniCpmV47Scratch,
    pool: &Arc<ComputePool>,
    pixels: &[f32],
    nx: usize,
    ny: usize,
) -> Result<Vec<f32>, String> {
    let cfg = &weights.config;
    let dim = cfg.n_embd;
    let tokens = nx * ny;
    scratch.fit(tokens, dim, cfg.n_ff);

    // patch embedding: gather 3*ps*ps patches, project, add bias
    let ps = cfg.patch_size;
    let patch_dim = 3 * ps * ps;
    let mut hidden = vec![0.0f32; tokens * dim];
    for ty in 0..ny {
        for tx in 0..nx {
            let token = ty * nx + tx;
            let mut patch = Vec::with_capacity(patch_dim);
            for c in 0..3 {
                for y in 0..ps {
                    for x in 0..ps {
                        let px = tx * ps + x;
                        let py = ty * ps + y;
                        let idx = (py * nx * ps + px) * 3 + c;
                        patch.push(pixels.get(idx).copied().unwrap_or(0.0));
                    }
                }
            }
            for (out, slot) in hidden[token * dim..(token + 1) * dim]
                .iter_mut()
                .enumerate()
            {
                *slot = crate::ops::dot_f32(
                    &weights.patch_embd_w[out * patch_dim..(out + 1) * patch_dim],
                    &patch,
                    patch_dim,
                ) + weights.patch_embd_b[out];
            }
        }
    }
    // + learned positional embedding. The table is 70x70, so each axis is
    // bucketed into 70 slots with `floor(70 * i / pos)` before the row index is
    // formed — this is not a raster walk of the table.
    const POS_GRID: usize = 70;
    let bucket = |i: usize, n: usize| -> usize {
        if n == 0 {
            return 0;
        }
        ((i * POS_GRID) / n).min(POS_GRID - 1)
    };
    let table = weights.n_positions.min(POS_GRID * POS_GRID);
    for ty in 0..ny {
        for tx in 0..nx {
            let pos = (bucket(ty, ny) * POS_GRID + bucket(tx, nx)).min(table.saturating_sub(1));
            let token = ty * nx + tx;
            let base = pos * dim;
            for i in 0..dim {
                hidden[token * dim + i] += weights.position_embd[base + i];
            }
        }
    }

    let insert = cfg.insert_layer_id;
    for il in 0..=insert.min(cfg.n_layer - 1) {
        vit_block(cfg, scratch, pool, &weights.layers[il], &mut hidden, nx);
    }

    // --- vit_merger: window attention + residual
    let merger = &weights.merger;
    for token in 0..tokens {
        crate::ops::norm::layer_norm(
            &hidden[token * dim..(token + 1) * dim],
            &merger.ln1_w,
            &merger.ln1_b,
            cfg.eps,
            &mut scratch.normed[token * dim..(token + 1) * dim],
        );
    }
    // ggml reorders into window-major order before the attention; the mask
    // makes that a pure permutation, so attention runs on the identity order
    // and the 4x4 block mask does the restricting.
    {
        let (q, k, v, attn, scores) = (
            &mut scratch.q[..tokens * dim],
            &mut scratch.k[..tokens * dim],
            &mut scratch.v[..tokens * dim],
            &mut scratch.attn[..tokens * dim],
            &mut scratch.scores[..tokens * tokens],
        );
        linear(
            pool,
            &merger.q_w,
            Some(&merger.q_b),
            &scratch.normed[..tokens * dim],
            q,
            tokens,
            dim,
            dim,
        );
        linear(
            pool,
            &merger.k_w,
            Some(&merger.k_b),
            &scratch.normed[..tokens * dim],
            k,
            tokens,
            dim,
            dim,
        );
        linear(
            pool,
            &merger.v_w,
            Some(&merger.v_b),
            &scratch.normed[..tokens * dim],
            v,
            tokens,
            dim,
            dim,
        );
        attention(cfg, scores, q, k, v, attn, tokens, nx, true);
    }
    let mut proj = vec![0.0f32; tokens * dim];
    linear(
        pool,
        &merger.o_w,
        Some(&merger.o_b),
        &scratch.attn[..tokens * dim],
        &mut proj,
        tokens,
        dim,
        dim,
    );
    for i in 0..tokens * dim {
        hidden[i] += proj[i];
    }

    // --- vit_merger: 2x2 downsample with mean residual
    let quads = ds_indices(nx, ny);
    let merged = quads.len();
    let merged_width = dim * 4;
    scratch.merged.resize(merged * merged_width, 0.0);
    scratch.mean.resize(merged * merged_width, 0.0);
    for (qi, [a, b, c, d]) in quads.iter().enumerate() {
        for i in 0..dim {
            let p0 = hidden[a * dim + i];
            let p1 = hidden[b * dim + i];
            let p2 = hidden[c * dim + i];
            let p3 = hidden[d * dim + i];
            let base = qi * merged_width;
            scratch.merged[base + i] = p0;
            scratch.merged[base + dim + i] = p1;
            scratch.merged[base + 2 * dim + i] = p2;
            scratch.merged[base + 3 * dim + i] = p3;
            scratch.mean[base + i] = (p0 + p1 + p2 + p3) * 0.25;
            scratch.mean[base + dim + i] = scratch.mean[base + i];
            scratch.mean[base + 2 * dim + i] = scratch.mean[base + i];
            scratch.mean[base + 3 * dim + i] = scratch.mean[base + i];
        }
    }
    let ds_hidden = merger.ds_up_w.len() / merged_width;
    for qi in 0..merged {
        crate::ops::norm::layer_norm(
            &scratch.merged[qi * merged_width..(qi + 1) * merged_width],
            &merger.ds_ln_w,
            &merger.ds_ln_b,
            cfg.eps,
            &mut scratch.normed[qi * merged_width..(qi + 1) * merged_width],
        );
    }
    scratch.ff.resize(merged * ds_hidden, 0.0);
    linear(
        pool,
        &merger.ds_up_w,
        Some(&merger.ds_up_b),
        &scratch.normed[..merged * merged_width],
        &mut scratch.ff[..merged * ds_hidden],
        merged,
        merged_width,
        ds_hidden,
    );
    for value in scratch.ff[..merged * ds_hidden].iter_mut() {
        *value = gelu_ggml_f32(*value);
    }
    let mut merged_out = vec![0.0f32; merged * dim];
    linear(
        pool,
        &merger.ds_down_w,
        Some(&merger.ds_down_b),
        &scratch.ff[..merged * ds_hidden],
        &mut merged_out,
        merged,
        ds_hidden,
        dim,
    );
    for i in 0..merged * dim {
        merged_out[i] += scratch.mean[i];
    }

    // --- remaining ViT blocks over the merged grid
    let (mnx, mny) = (nx / 2, ny / 2);
    for il in (insert + 1)..cfg.n_layer {
        vit_block(
            cfg,
            scratch,
            pool,
            &weights.layers[il],
            &mut merged_out,
            mnx,
        );
    }

    // --- post_ln
    for token in 0..merged {
        crate::ops::norm::layer_norm(
            &merged_out[token * dim..(token + 1) * dim],
            &weights.post_ln_w,
            &weights.post_ln_b,
            cfg.eps,
            &mut scratch.normed[token * dim..(token + 1) * dim],
        );
    }

    // --- final merger: 2x2 concat -> input_norm -> GELU_erf MLP
    let quads = ds_indices(mnx, mny);
    let n_out = quads.len();
    scratch.merged.resize(n_out * merged_width, 0.0);
    for (qi, [a, b, c, d]) in quads.iter().enumerate() {
        for i in 0..dim {
            let base = qi * merged_width;
            scratch.merged[base + i] = merged_out[a * dim + i];
            scratch.merged[base + dim + i] = merged_out[b * dim + i];
            scratch.merged[base + 2 * dim + i] = merged_out[c * dim + i];
            scratch.merged[base + 3 * dim + i] = merged_out[d * dim + i];
        }
    }
    for qi in 0..n_out {
        crate::ops::norm::layer_norm(
            &scratch.merged[qi * merged_width..(qi + 1) * merged_width],
            &weights.projector.input_norm_w,
            &weights.projector.input_norm_b,
            cfg.eps,
            &mut scratch.normed[qi * merged_width..(qi + 1) * merged_width],
        );
    }
    let proj = &weights.projector;
    let up_hidden = proj.up_w.len() / merged_width;
    scratch.ff.resize(n_out * up_hidden, 0.0);
    linear(
        pool,
        &proj.up_w,
        Some(&proj.up_b),
        &scratch.normed[..n_out * merged_width],
        &mut scratch.ff[..n_out * up_hidden],
        n_out,
        merged_width,
        up_hidden,
    );
    for value in scratch.ff[..n_out * up_hidden].iter_mut() {
        *value = gelu_erf(*value);
    }
    let mut out = vec![0.0f32; n_out * cfg.n_embd_text];
    linear(
        pool,
        &proj.down_w,
        Some(&proj.down_b),
        &scratch.ff[..n_out * up_hidden],
        &mut out,
        n_out,
        up_hidden,
        cfg.n_embd_text,
    );
    Ok(out)
}

/// Append the separator rows for `suffix` to `embeddings`, in place.
pub fn append_suffix(weights: &MiniCpmV47Weights, suffix: &[usize], embeddings: &mut Vec<f32>) {
    let dim = weights.config.n_embd_text;
    for &row in suffix {
        if row >= weights.n_sep_rows {
            continue;
        }
        embeddings.extend_from_slice(&weights.tok_embd_sep[row * dim..(row + 1) * dim]);
    }
}

/// Rows of `v.tok_embd_sep` a tile appends, matching
/// `clip_suffix_rows()` at the pinned oracle:
///
/// | suffix type | rows |
/// |---|---|
/// | `OV` (no slices follow) | `{0}` |
/// | `OV_SLICE` | `{0, 1}` |
/// | `SLICE` (mid row) | `{2, 1}` |
/// | `ROW_END` (last column) | `{2, 3, 1}` |
/// | `LAST` (bottom right) | `{2}` |
pub const SUFFIX_OV: &[usize] = &[0];
pub const SUFFIX_OV_SLICE: &[usize] = &[0, 1];
pub const SUFFIX_SLICE: &[usize] = &[2, 1];
pub const SUFFIX_ROW_END: &[usize] = &[2, 3, 1];
pub const SUFFIX_LAST: &[usize] = &[2];

/// Suffix rows for a `grid_x * grid_y` slice grid, in row-major order.
pub fn suffix_plan(grid_x: usize, grid_y: usize) -> Vec<&'static [usize]> {
    let mut plan = Vec::with_capacity(1 + grid_x * grid_y);
    plan.push(if grid_x * grid_y == 0 {
        SUFFIX_OV
    } else {
        SUFFIX_OV_SLICE
    });
    for y in 0..grid_y {
        for x in 0..grid_x {
            plan.push(if y == grid_y - 1 && x == grid_x - 1 {
                SUFFIX_LAST
            } else if x == grid_x - 1 {
                SUFFIX_ROW_END
            } else {
                SUFFIX_SLICE
            });
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_plan_matches_the_reference_layout() {
        // [ov] </image><slice> [S00] </slice><slice> [S01] </slice>\n<slice>
        // [S10] </slice><slice> [S11] </slice>
        let plan = suffix_plan(2, 2);
        assert_eq!(plan[0], SUFFIX_OV_SLICE);
        assert_eq!(plan[1], SUFFIX_SLICE);
        assert_eq!(plan[2], SUFFIX_ROW_END);
        assert_eq!(plan[3], SUFFIX_SLICE);
        assert_eq!(plan[4], SUFFIX_LAST);
    }

    #[test]
    fn overview_only_uses_a_single_close_tag() {
        assert_eq!(suffix_plan(0, 0), vec![SUFFIX_OV]);
    }

    #[test]
    fn single_slice_grid_closes_immediately() {
        let plan = suffix_plan(1, 1);
        assert_eq!(plan[0], SUFFIX_OV_SLICE);
        assert_eq!(plan[1], SUFFIX_LAST);
    }

    #[test]
    fn ds_indices_walk_the_grid_row_major() {
        let idx = ds_indices(4, 4);
        assert_eq!(idx.len(), 4);
        assert_eq!(idx[0], [0, 1, 4, 5]);
        assert_eq!(idx[1], [2, 3, 6, 7]);
        assert_eq!(idx[2], [8, 9, 12, 13]);
        assert_eq!(idx[3], [10, 11, 14, 15]);
    }
}
/// One `[t, x, y, z]` M-RoPE position per image token.
pub type CanvasPosition = [usize; 4];

/// Per-tile output grid of a canvas chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGrid {
    pub grid_w: usize,
    pub grid_h: usize,
}

/// `mtmd_canvas_scale` — map a tile-local coordinate onto the canvas.
///
/// `round(linspace(0, canvas - 1, grid))` with `round` breaking ties to even,
/// which is `nearbyint` under the default rounding mode.
pub fn canvas_scale(coord: usize, grid: usize, canvas: usize) -> usize {
    if grid <= 1 || canvas <= 1 {
        return 0;
    }
    let v = coord as f64 * (canvas as f64 - 1.0) / (grid as f64 - 1.0);
    v.round().min((canvas - 1) as f64) as usize
}

/// Build the M-RoPE positions for a canvas chunk.
///
/// Port of `mtmd_canvas_decoder_pos` at the pinned oracle: every tile shares
/// the `<image>` token as its origin (`base = pos_0 - 1`), the overview is
/// stretched across the whole canvas, each slice fills its own cell, and the
/// separators after a tile take consecutive positions of their own.
///
/// `tiles` is `[overview, S00, S01, ...]` with `suffixes` the matching
/// [`suffix_plan`]. `token_grids` is each tile's `[grid_w, grid_h]`, scaled
/// from the last tile the way `canvas_tile_grid` does.
#[allow(clippy::too_many_arguments)]
pub fn canvas_positions(
    token_grids: &[TileGrid],
    suffixes: &[&[usize]],
    slice_grid_x: usize,
    slice_grid_y: usize,
    pos_0: usize,
) -> Result<Vec<CanvasPosition>, String> {
    if token_grids.len() != suffixes.len() {
        return Err(format!(
            "canvas: {} tile grids but {} suffix plans",
            token_grids.len(),
            suffixes.len()
        ));
    }
    let reference = *token_grids.last().ok_or("canvas: no tiles")?;
    if reference.grid_w == 0 || reference.grid_h == 0 {
        return Err("canvas: reference tile has an empty grid".into());
    }
    let (nx, ny) = (reference.grid_w, reference.grid_h);
    let tiled = slice_grid_x > 0 && slice_grid_y > 0;
    let (canvas_w, canvas_h) = if tiled {
        (slice_grid_x * nx, slice_grid_y * ny)
    } else {
        (nx, ny)
    };
    let base = pos_0.saturating_sub(1);

    // Every tile's grid is scaled from the reference's.
    let grids: Vec<TileGrid> = token_grids
        .iter()
        .map(|g| TileGrid {
            grid_w: g.grid_w * nx / reference.grid_w,
            grid_h: g.grid_h * ny / reference.grid_h,
        })
        .collect();

    let mut out = Vec::new();
    let mut t = pos_0;
    for (k, grid) in grids.iter().enumerate() {
        for i in 0..(grid.grid_w * grid.grid_h) {
            let row = i / grid.grid_w;
            let col = i % grid.grid_w;
            let (h, w) = if k == 0 {
                (
                    canvas_scale(row, grid.grid_h, canvas_h),
                    canvas_scale(col, grid.grid_w, canvas_w),
                )
            } else {
                let s = k - 1;
                (
                    (s / slice_grid_x.max(1)) * ny + row,
                    (s % slice_grid_x.max(1)) * nx + col,
                )
            };
            out.push([t, base + w, base + h, base]);
        }
        // separators after the tile
        for i in 0..suffixes[k].len() {
            let p = t + 1 + i;
            out.push([p, p, p, p]);
        }
        t += 1 + suffixes[k].len();
    }
    Ok(out)
}

#[cfg(test)]
mod canvas_tests {
    use super::*;

    #[test]
    fn canvas_scale_is_endpoints_inclusive() {
        assert_eq!(canvas_scale(0, 8, 8), 0);
        assert_eq!(canvas_scale(7, 8, 8), 7);
        // single-position grid collapses to the origin
        assert_eq!(canvas_scale(0, 1, 8), 0);
    }

    #[test]
    fn overview_only_chunk_takes_one_position_per_token() {
        let positions = canvas_positions(
            &[TileGrid {
                grid_w: 8,
                grid_h: 8,
            }],
            &[SUFFIX_OV],
            0,
            0,
            5,
        )
        .expect("canvas");
        // 64 tile tokens + 1 separator
        assert_eq!(positions.len(), 65);
        assert_eq!(positions[0], [5, 4, 4, 4]);
        assert_eq!(positions[64], [6, 6, 6, 6]);
    }

    #[test]
    fn tiles_share_the_image_token_as_origin() {
        let plan = suffix_plan(1, 1);
        let grids = [
            TileGrid {
                grid_w: 8,
                grid_h: 8,
            },
            TileGrid {
                grid_w: 8,
                grid_h: 8,
            },
        ];
        let positions = canvas_positions(&grids, &plan, 1, 1, 5).expect("canvas");
        // overview 64 + 2 seps, slice 64 + 1 sep
        assert_eq!(positions.len(), 64 + 2 + 64 + 1);
        // every position t only ever advances, and the slice starts after the
        // overview's own separators
        // overview tokens all sit at t=pos_0; its two separators take t+1, t+2
        assert_eq!(positions[0][0], 5);
        assert_eq!(positions[63][0], 5);
        assert_eq!(positions[64][0], 6);
        assert_eq!(positions[65][0], 7);
        // the slice resumes at t = pos_0 + 1 + n_sep
        assert_eq!(positions[66][0], 8);
    }

    #[test]
    fn slice_cells_tile_the_canvas() {
        let plan = suffix_plan(2, 2);
        let grids = [
            TileGrid {
                grid_w: 8,
                grid_h: 8,
            },
            TileGrid {
                grid_w: 8,
                grid_h: 8,
            },
            TileGrid {
                grid_w: 8,
                grid_h: 8,
            },
            TileGrid {
                grid_w: 8,
                grid_h: 8,
            },
            TileGrid {
                grid_w: 8,
                grid_h: 8,
            },
        ];
        let positions = canvas_positions(&grids, &plan, 2, 2, 5).expect("canvas");
        let expected = 5 * 64 + 2 + 2 + 3 + 2 + 1;
        assert_eq!(positions.len(), expected);
        // canvas is 2*8 x 2*8; the last slice ends at the canvas corner
        let last = positions[positions.len() - 2];
        assert_eq!((last[1] - 4, last[2] - 4), (15, 15));
    }

    #[test]
    fn mismatched_tile_and_suffix_counts_are_rejected() {
        assert!(canvas_positions(
            &[TileGrid {
                grid_w: 4,
                grid_h: 4
            }],
            &[SUFFIX_OV, SUFFIX_LAST],
            1,
            1,
            1,
        )
        .is_err());
    }
}
