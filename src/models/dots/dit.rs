//! DiT velocity-field predictor + flow-matching solver for dots.tts.
//!
//! Port of `dit_inference.py`'s `EagerDiTRunner._decode_flow_matching` and
//! `core.py::fm_solver_step`: for each 4-frame latent patch, integrate
//! dz/dt = v_cond + guidance·(v_cond − v_uncond) from t=0 to 1 with Euler
//! (default NFE=10). The DiT is conditioned on time + speaker and attends to
//! the accumulated FM sequence with the reference mask/positions.

use crate::core::tensor::TensorSource;
use crate::models::dots::config::DotsTtsConfig;
use crate::models::dots::patch_encoder::{dots_rotary, linear_forward, load_f16_f32};
use crate::ops::{dot_f32, silu};

const DIT_HEADS: usize = 16;
const DIT_HEAD_DIM: usize = 64;
const DIT_HIDDEN: usize = 1024;
const DIT_FFN: usize = 4096;
const DIT_ROPE_THETA: f32 = 10_000.0;
const DIT_NORM_EPS: f32 = 1e-5;
const TIME_EMBED_DIM: usize = 256;

pub(crate) struct DitBlockWeights {
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) o: Vec<f32>,
    pub(crate) o_bias: Vec<f32>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) fc1: Vec<f32>,
    pub(crate) fc1_bias: Vec<f32>,
    pub(crate) fc2: Vec<f32>,
    pub(crate) fc2_bias: Vec<f32>,
    pub(crate) adaln: Vec<f32>,
    pub(crate) adaln_bias: Vec<f32>,
}

pub struct DiT {
    pub n_latent: usize,
    pub input_w: Vec<f32>,
    pub input_b: Vec<f32>,
    pub time_w0: Vec<f32>,
    pub time_b0: Vec<f32>,
    pub time_w2: Vec<f32>,
    pub time_b2: Vec<f32>,
    pub(crate) blocks: Vec<DitBlockWeights>,
    pub out_adaln_w: Vec<f32>,
    pub out_adaln_b: Vec<f32>,
    pub out_linear_w: Vec<f32>,
    pub out_linear_b: Vec<f32>,
}

impl DiT {
    pub fn from_source(source: &dyn TensorSource, config: DotsTtsConfig) -> Result<Self, String> {
        let w = |name: &str, dims: &[u64]| -> Result<Vec<f32>, String> { load_f16_f32(source, name, dims) };
        let input_w = w("dotstts.dit.input_layer.weight", &[DIT_HIDDEN as u64; 2])?;
        let input_b = w("dotstts.dit.input_layer.bias", &[DIT_HIDDEN as u64])?;
        let time_w0 = w(
            "dotstts.dit.time_embedder.mlp.0.weight",
            &[TIME_EMBED_DIM as u64, DIT_HIDDEN as u64],
        )?;
        let time_b0 = w("dotstts.dit.time_embedder.mlp.0.bias", &[DIT_HIDDEN as u64])?;
        let time_w2 = w("dotstts.dit.time_embedder.mlp.2.weight", &[DIT_HIDDEN as u64; 2])?;
        let time_b2 = w("dotstts.dit.time_embedder.mlp.2.bias", &[DIT_HIDDEN as u64])?;
        let mut blocks = Vec::with_capacity(config.dit_layers);
        for layer in 0..config.dit_layers {
            let name = |suffix: &str| format!("dotstts.dit.blocks.{layer}.{suffix}");
            let hid2 = [DIT_HIDDEN as u64; 2];
            blocks.push(DitBlockWeights {
                q: w(&name("attn.q.weight"), &hid2)?,
                k: w(&name("attn.k.weight"), &hid2)?,
                v: w(&name("attn.v.weight"), &hid2)?,
                o: w(&name("attn.o.weight"), &hid2)?,
                o_bias: w(&name("attn.o.bias"), &[DIT_HIDDEN as u64])?,
                q_norm: w(&name("attn.q_norm.weight"), &[DIT_HEAD_DIM as u64])?,
                k_norm: w(&name("attn.k_norm.weight"), &[DIT_HEAD_DIM as u64])?,
                fc1: w(&name("ffn.fc1.weight"), &[DIT_HIDDEN as u64, DIT_FFN as u64])?,
                fc1_bias: w(&name("ffn.fc1.bias"), &[DIT_FFN as u64])?,
                fc2: w(&name("ffn.fc2.weight"), &[DIT_FFN as u64, DIT_HIDDEN as u64])?,
                fc2_bias: w(&name("ffn.fc2.bias"), &[DIT_HIDDEN as u64])?,
                adaln: w(
                    &name("adaLN_modulation.1.weight"),
                    &[DIT_HIDDEN as u64, (6 * DIT_HIDDEN) as u64],
                )?,
                adaln_bias: w(&name("adaLN_modulation.1.bias"), &[(6 * DIT_HIDDEN) as u64])?,
            });
        }
        let out_adaln_w = w(
            "dotstts.dit.output_layer.adaLN_modulation.1.weight",
            &[DIT_HIDDEN as u64, (2 * DIT_HIDDEN) as u64],
        )?;
        let out_adaln_b = w(
            "dotstts.dit.output_layer.adaLN_modulation.1.bias",
            &[(2 * DIT_HIDDEN) as u64],
        )?;
        let out_linear_w = w(
            "dotstts.dit.output_layer.linear.weight",
            &[DIT_HIDDEN as u64, config.latent_dim as u64],
        )?;
        let out_linear_b = w(
            "dotstts.dit.output_layer.linear.bias",
            &[config.latent_dim as u64],
        )?;
        Ok(Self {
            n_latent: config.latent_dim,
            input_w,
            input_b,
            time_w0,
            time_b0,
            time_w2,
            time_b2,
            blocks,
            out_adaln_w,
            out_adaln_b,
            out_linear_w,
            out_linear_b,
        })
    }

    /// Reference `TimestepEmbedder.timestep_embedding`: cos/sin over 256 dims.
    pub fn time_embedding(t: f32) -> Vec<f32> {
        let half = TIME_EMBED_DIM / 2;
        let mut embedding = vec![0.0f32; TIME_EMBED_DIM];
        for k in 0..half {
            let freq = (-(10_000.0f32).ln() * k as f32 / half as f32).exp();
            let arg = t * freq;
            embedding[k] = arg.cos();
            embedding[half + k] = arg.sin();
        }
        embedding
    }

    fn time_mlp(&self, t: f32) -> Vec<f32> {
        let c = Self::time_embedding(t);
        let mut hidden = vec![0.0f32; DIT_HIDDEN];
        linear_forward(&self.time_w0, Some(&self.time_b0), &c, TIME_EMBED_DIM, DIT_HIDDEN, &mut hidden);
        for value in hidden.iter_mut() {
            *value = silu(*value);
        }
        let mut out = vec![0.0f32; DIT_HIDDEN];
        linear_forward(&self.time_w2, Some(&self.time_b2), &hidden, DIT_HIDDEN, DIT_HIDDEN, &mut out);
        out
    }

    /// Full DiT forward. `x` is `[rows, DIT_HIDDEN]` with both CFG branches
    /// already stacked (same mask/positions apply per branch), `g_cond` is one
    /// 1024-dim vector per row, `mask` is `[branch_len, branch_len]` bool,
    /// `positions` is `[branch_len]`. Output `[rows, latent_dim]`.
    fn forward(
        &self,
        x_in: &[f32],
        t: f32,
        g_cond: &[f32],
        mask: &[bool],
        positions: &[usize],
        out: &mut [f32],
    ) {
        let branch_len = positions.len();
        let rows = x_in.len() / DIT_HIDDEN;
        let mut x = vec![0.0f32; rows * DIT_HIDDEN];
        linear_forward(&self.input_w, Some(&self.input_b), x_in, DIT_HIDDEN, DIT_HIDDEN, &mut x);

        let time_c = self.time_mlp(t);
        let mut cond = vec![0.0f32; rows * DIT_HIDDEN];
        for row in 0..rows {
            let dst = &mut cond[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
            dst.copy_from_slice(&time_c);
            for (d, &g) in dst
                .iter_mut()
                .zip(&g_cond[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN])
            {
                *d += g;
            }
        }

        // Per-block scratch reused across layers.
        let mut attn_in = vec![0.0f32; rows * DIT_HIDDEN];
        let mut q = vec![0.0f32; rows * DIT_HIDDEN];
        let mut k = vec![0.0f32; rows * DIT_HIDDEN];
        let mut v = vec![0.0f32; rows * DIT_HIDDEN];
        let mut attn_out = vec![0.0f32; rows * DIT_HIDDEN];
        let mut h = vec![0.0f32; DIT_HIDDEN];
        let mut normed = vec![0.0f32; DIT_HIDDEN];
        let mut ffn_buf = vec![0.0f32; DIT_FFN];
        let mut ffn_out = vec![0.0f32; DIT_HIDDEN];

        for block in &self.blocks {
            // 1. per-row adaLN mods + modulated attention input
            let mut mods = vec![0.0f32; rows * 6 * DIT_HIDDEN];
            let mut silu_row = vec![0.0f32; DIT_HIDDEN];
            for row in 0..rows {
                let c = &cond[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
                for (m, &cv) in silu_row.iter_mut().zip(c.iter()) {
                    *m = silu(cv);
                }
                let mrow = row * 6 * DIT_HIDDEN;
                let adaln_in = silu_row.clone();
                linear_forward(
                    &block.adaln,
                    Some(&block.adaln_bias),
                    &adaln_in,
                    DIT_HIDDEN,
                    6 * DIT_HIDDEN,
                    &mut mods[mrow..mrow + 6 * DIT_HIDDEN],
                );
                let xrow = &x[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
                layernorm_no_affine(xrow, &mut h, DIT_NORM_EPS);
                let (sa, sca) = (
                    &mods[mrow..mrow + DIT_HIDDEN],
                    &mods[mrow + DIT_HIDDEN..mrow + 2 * DIT_HIDDEN],
                );
                let dst = &mut attn_in[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
                for d in 0..DIT_HIDDEN {
                    dst[d] = h[d] * (1.0 + sca[d]) + sa[d];
                }
                let _ = sa;
            }
            // 2. QKV projections (all rows at once)
            linear_forward(&block.q, None, &attn_in, DIT_HIDDEN, DIT_HIDDEN, &mut q);
            linear_forward(&block.k, None, &attn_in, DIT_HIDDEN, DIT_HIDDEN, &mut k);
            linear_forward(&block.v, None, &attn_in, DIT_HIDDEN, DIT_HIDDEN, &mut v);
            // 3. q/k RMSNorm (learned) + rotary at absolute positions
            for row in 0..rows {
                for head in 0..DIT_HEADS {
                    let qh = &mut q[row * DIT_HIDDEN + head * DIT_HEAD_DIM
                        ..row * DIT_HIDDEN + (head + 1) * DIT_HEAD_DIM];
                    let kh = &mut k[row * DIT_HIDDEN + head * DIT_HEAD_DIM
                        ..row * DIT_HIDDEN + (head + 1) * DIT_HEAD_DIM];
                    rms_norm_weighted(qh, &block.q_norm, DIT_NORM_EPS);
                    rms_norm_weighted(kh, &block.k_norm, DIT_NORM_EPS);
                    let pos = positions[row % branch_len];
                    dots_rotary(qh, pos, DIT_HEAD_DIM, DIT_ROPE_THETA);
                    dots_rotary(kh, pos, DIT_HEAD_DIM, DIT_ROPE_THETA);
                }
            }
            // 4. attention with the reference mask
            self.attention(&q, &k, &v, &mut attn_out, rows, branch_len, mask);
            // 5. o_proj + residual + FFN
            for row in 0..rows {
                linear_forward(
                    &block.o,
                    Some(&block.o_bias),
                    &attn_out[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN],
                    DIT_HIDDEN,
                    DIT_HIDDEN,
                    &mut normed,
                );
                let mrow = row * 6 * DIT_HIDDEN;
                let ga = &mods[mrow + 2 * DIT_HIDDEN..mrow + 3 * DIT_HIDDEN];
                let xrow = &mut x[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
                for d in 0..DIT_HIDDEN {
                    xrow[d] += ga[d] * normed[d];
                }
                layernorm_no_affine(xrow, &mut h, DIT_NORM_EPS);
                let (sf, scf, gf) = (
                    &mods[mrow + 3 * DIT_HIDDEN..mrow + 4 * DIT_HIDDEN],
                    &mods[mrow + 4 * DIT_HIDDEN..mrow + 5 * DIT_HIDDEN],
                    &mods[mrow + 5 * DIT_HIDDEN..mrow + 6 * DIT_HIDDEN],
                );
                for d in 0..DIT_HIDDEN {
                    normed[d] = h[d] * (1.0 + scf[d]) + sf[d];
                }
                linear_forward(&block.fc1, Some(&block.fc1_bias), &normed, DIT_HIDDEN, DIT_FFN, &mut ffn_buf);
                for value in ffn_buf.iter_mut() {
                    *value = gelu_tanh(*value);
                }
                linear_forward(&block.fc2, Some(&block.fc2_bias), &ffn_buf, DIT_FFN, DIT_HIDDEN, &mut ffn_out);
                for d in 0..DIT_HIDDEN {
                    xrow[d] += gf[d] * ffn_out[d];
                }
            }
        }

        // output layer: adaLN shift/scale on a no-affine LayerNorm, then linear
        let mut mods = vec![0.0f32; 2 * DIT_HIDDEN];
        let mut silu_row = vec![0.0f32; DIT_HIDDEN];
        for row in 0..rows {
            let c = &cond[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
            for (m, &cv) in silu_row.iter_mut().zip(c.iter()) {
                *m = silu(cv);
            }
            let adaln_in = silu_row.clone();
            linear_forward(
                &self.out_adaln_w,
                Some(&self.out_adaln_b),
                &adaln_in,
                DIT_HIDDEN,
                2 * DIT_HIDDEN,
                &mut mods,
            );
            let shift = &mods[..DIT_HIDDEN];
            let scale = &mods[DIT_HIDDEN..];
            let xrow = &x[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
            layernorm_no_affine(xrow, &mut h, DIT_NORM_EPS);
            for d in 0..DIT_HIDDEN {
                normed[d] = h[d] * (1.0 + scale[d]) + shift[d];
            }
            linear_forward(
                &self.out_linear_w,
                Some(&self.out_linear_b),
                &normed,
                DIT_HIDDEN,
                self.n_latent,
                &mut out[row * self.n_latent..(row + 1) * self.n_latent],
            );
        }
    }

    /// Masked multi-head attention. Rows are `[batch*branch_len, hidden]`;
    /// queries within the same branch share the `mask` (`[branch_len, branch_len]`).
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        rows: usize,
        branch_len: usize,
        mask: &[bool],
    ) {
        let scale = 1.0 / (DIT_HEAD_DIM as f32).sqrt();
        out.fill(0.0);
        let mut acc = [0.0f32; DIT_HEAD_DIM];
        for row in 0..rows {
            let qb = row / branch_len;
            let qr = row % branch_len;
            for head in 0..DIT_HEADS {
                let qh = &q[row * DIT_HIDDEN + head * DIT_HEAD_DIM
                    ..row * DIT_HIDDEN + (head + 1) * DIT_HEAD_DIM];
                acc.fill(0.0);
                let mut sum = 0.0f32;
                let mut max = f32::NEG_INFINITY;
                for key in 0..branch_len {
                    if !mask[qr * branch_len + key] {
                        continue;
                    }
                    let key_row = qb * branch_len + key;
                    let koff = key_row * DIT_HIDDEN + head * DIT_HEAD_DIM;
                    let score = dot_f32(
                        qh,
                        &k[koff..koff + DIT_HEAD_DIM],
                        DIT_HEAD_DIM,
                    ) * scale;
                    let voff = key_row * DIT_HIDDEN + head * DIT_HEAD_DIM;
                    let vrow = &v[voff..voff + DIT_HEAD_DIM];
                    if score > max {
                        let rescale = (max - score).exp();
                        max = score;
                        for value in acc.iter_mut() {
                            *value *= rescale;
                        }
                        sum = sum.mul_add(rescale, 1.0);
                        for (value, &vv) in acc.iter_mut().zip(vrow.iter()) {
                            *value += vv;
                        }
                    } else {
                        let weight = (score - max).exp();
                        sum += weight;
                        for (value, &vv) in acc.iter_mut().zip(vrow.iter()) {
                            *value += vv * weight;
                        }
                    }
                }
                let recip = if sum == 0.0 { 0.0 } else { sum.recip() };
                let dst = row * DIT_HIDDEN + head * DIT_HEAD_DIM;
                for (slot, &value) in out[dst..dst + DIT_HEAD_DIM].iter_mut().zip(acc.iter()) {
                    *slot = value * recip;
                }
            }
        }
    }

    /// Decode one latent patch (reference `_decode_flow_matching`):
    /// euler-integrate CFG-guided velocity over `nfe` steps.
    ///
    /// `sequence` and `cfg_sequence` are `[fm_seq_len, DIT_HIDDEN]` prefixes;
    /// the latent region (4 rows) is replaced by `coordinate_proj(z)` at each
    /// step and carries positions `fm_seq_len..fm_seq_len+4`.
    pub fn solve_patch(
        &self,
        sequence: &[f32],
        cfg_sequence: &[f32],
        fm_seq_len: usize,
        g_cond: &[f32],
        coordinate_proj: &[f32],
        coordinate_bias: &[f32],
        guidance: f32,
        nfe: usize,
        z0: &[f32],
        out: &mut [f32],
    ) -> Result<(), String> {
        let patch = self.n_latent_slots();
        let total = fm_seq_len + patch;
        if sequence.len() != fm_seq_len * DIT_HIDDEN || cfg_sequence.len() != fm_seq_len * DIT_HIDDEN {
            return Err("solve_patch: sequence width mismatch".into());
        }
        if z0.len() != patch * self.n_latent {
            return Err("solve_patch: noise width mismatch".into());
        }
        if g_cond.len() != DIT_HIDDEN {
            return Err("solve_patch: g_cond width mismatch".into());
        }

        // reference mask (EagerDiTRunner._build_decode_mask)
        let block_start = fm_seq_len.saturating_sub(1);
        let latent_start = total - patch;
        let mut mask = vec![false; total * total];
        for qr in 0..total {
            for key in 0..total {
                let allowed = if qr < block_start {
                    key <= qr
                } else if qr < fm_seq_len {
                    true
                } else {
                    key < fm_seq_len || key >= latent_start
                };
                mask[qr * total + key] = allowed;
            }
        }
        // positions: 0..fm_seq_len, then fm_seq_len..fm_seq_len+patch
        let positions: Vec<usize> = (0..total).map(|i| if i < fm_seq_len { i } else { fm_seq_len + (i - latent_start) }).collect();

        // input tensors: [cond seq; uncond seq] + z region
        let mut x = vec![0.0f32; 2 * total * DIT_HIDDEN];
        for (branch, src) in [(0usize, sequence), (1, cfg_sequence)] {
            let base = branch * total * DIT_HIDDEN;
            x[base..base + fm_seq_len * DIT_HIDDEN]
                .copy_from_slice(&src[..fm_seq_len * DIT_HIDDEN]);
        }
        let mut g_branches = vec![0.0f32; 2 * DIT_HIDDEN];
        g_branches[..DIT_HIDDEN].copy_from_slice(g_cond);
        // g_cond repeated per row
        let mut g_repeated = vec![0.0f32; 2 * total * DIT_HIDDEN];
        for row in 0..2 * total {
            let branch = row / total;
            let src = &g_branches[branch * DIT_HIDDEN..(branch + 1) * DIT_HIDDEN];
            g_repeated[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN].copy_from_slice(src);
        }

        let mut z = z0.to_vec();
        let mut z_proj = vec![0.0f32; patch * DIT_HIDDEN];
        let mut velocity = vec![0.0f32; 2 * patch * self.n_latent];
        let mut diag = vec![0.0f32; 2 * total * self.n_latent];
        let dt = 1.0 / nfe as f32;
        for step in 0..nfe {
            let t = step as f32 / nfe as f32;
            // z_proj = coordinate_proj(z)
            linear_forward(coordinate_proj, Some(coordinate_bias), &z, self.n_latent, DIT_HIDDEN, &mut z_proj);
            // splat into the latent region of both branches
            for branch in 0..2 {
                let base = branch * total * DIT_HIDDEN + latent_start * DIT_HIDDEN;
                x[base..base + patch * DIT_HIDDEN].copy_from_slice(&z_proj);
            }
            self.forward(&x, t, &g_repeated, &mask, &positions, &mut diag);
            // extract the latent rows of each branch
            for branch in 0..2 {
                let src = &diag[branch * total * self.n_latent + latent_start * self.n_latent
                    ..branch * total * self.n_latent + (latent_start + patch) * self.n_latent];
                velocity[branch * patch * self.n_latent..(branch + 1) * patch * self.n_latent]
                    .copy_from_slice(src);
            }
            // CFG
            for i in 0..patch * self.n_latent {
                let cond_v = velocity[i];
                let uncond_v = velocity[patch * self.n_latent + i];
                z[i] += dt * (cond_v + guidance * (cond_v - uncond_v));
            }
        }
        out.copy_from_slice(&z);
        Ok(())
    }

    pub fn n_latent_slots(&self) -> usize {
        4 // patch_size is fixed at 4 for all dots.tts artifacts
    }
}

fn layernorm_no_affine(x: &[f32], out: &mut [f32], eps: f32) {
    let mut mean = 0.0f64;
    for &value in x.iter() {
        mean += value as f64;
    }
    mean /= x.len() as f64;
    let mut var = 0.0f64;
    for &value in x.iter() {
        let d = value as f64 - mean;
        var += d * d;
    }
    var /= x.len() as f64;
    let inv = 1.0 / (var + eps as f64).sqrt();
    for (o, &value) in out.iter_mut().zip(x.iter()) {
        *o = ((value as f64 - mean) * inv) as f32;
    }
}

fn rms_norm_weighted(x: &mut [f32], weight: &[f32], eps: f32) {
    let mut mean_sq = 0.0f64;
    for &value in x.iter() {
        mean_sq += (value as f64) * (value as f64);
    }
    mean_sq /= x.len() as f64;
    let inv = 1.0 / (mean_sq + eps as f64).sqrt();
    for (value, &w) in x.iter_mut().zip(weight.iter()) {
        *value = (*value as f64 * inv) as f32 * w;
    }
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_embedding_has_reference_shape_and_finiteness() {
        let emb = DiT::time_embedding(0.0);
        assert_eq!(emb.len(), 256);
        assert!(emb.iter().all(|v| v.is_finite()));
        assert!((emb[0] - 1.0).abs() < 1e-6); // cos(0)
        assert!(emb[128].abs() < 1e-6); // sin(0)
    }

    #[test]
    fn gelu_tanh_matches_known_values() {
        assert!((gelu_tanh(0.0) - 0.0).abs() < 1e-6);
        assert!((gelu_tanh(1.0) - 0.841192).abs() < 1e-4);
        assert!((gelu_tanh(-1.0) + 0.158808).abs() < 1e-4);
    }

    #[test]
    fn rms_norm_weighted_matches_definition() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![2.0f32; 4];
        rms_norm_weighted(&mut x, &w, 1e-5);
        let mean_sq = (1.0f64 + 4.0 + 9.0 + 16.0) / 4.0;
        let inv = 1.0 / (mean_sq + 1e-5).sqrt();
        assert!((x[0] - (inv as f32) * 2.0).abs() < 1e-5);
    }
}