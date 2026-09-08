//! DiT velocity-field predictor + flow-matching solver for dots.tts.
//!
//! Port of `dit_inference.py`'s `EagerDiTRunner._decode_flow_matching` and
//! `core.py::fm_solver_step`: for each 4-frame latent patch, integrate
//! dz/dt = v_cond + guidance·(v_cond − v_uncond) from t=0 to 1 with Euler
//! (default NFE=10). The DiT is conditioned on time + speaker and attends to
//! the accumulated FM sequence with the reference mask/positions.

use super::blas::sys;

use crate::core::tensor::TensorSource;
use crate::models::dots::config::DotsTtsConfig;
use crate::models::dots::patch_encoder::{dots_rotary, linear_forward, load_f16_f32};
use crate::models::dots::speaker::exp::{torch28_exp, torch28_tanh};
use crate::ops::{dot_f32, rope_sin_cos_sleef};

#[cfg(any(
    target_os = "macos",
    all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
))]
#[inline(always)]
fn torch28_sum4(values: &[f32; 4]) -> f32 {
    (values[0] + values[2]) + (values[1] + values[3])
}

#[cfg(feature = "parity-trace")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "parity-trace")]
static TRACE_DIT_INTERNAL: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "parity-trace")]
pub(crate) fn reset_internal_trace() {
    TRACE_DIT_INTERNAL.store(false, Ordering::Relaxed);
}

const DIT_HEADS: usize = 16;
const DIT_HEAD_DIM: usize = 64;
const DIT_HIDDEN: usize = 1024;
const DIT_FFN: usize = 4096;
const DIT_ROPE_THETA: f32 = 10_000.0;
const DIT_NORM_EPS: f32 = 1e-5;
const TIME_EMBED_DIM: usize = 256;

#[inline(always)]
fn torch28_silu(value: f32) -> f32 {
    value / (1.0 + torch28_exp(-value))
}

#[inline(always)]
fn torch28_euler_step(z: f32, velocity: f32, nfe: usize) -> f32 {
    z + velocity / nfe as f32
}

pub(crate) fn build_decode_mask_positions(
    fm_seq_len: usize,
    patch_size: usize,
) -> Result<(Vec<bool>, Vec<usize>), String> {
    if fm_seq_len == 0 || patch_size == 0 {
        return Err("decode mask requires non-empty prefix and latent patch".into());
    }
    let total = fm_seq_len
        .checked_add(patch_size)
        .ok_or_else(|| "decode mask length overflow".to_string())?;
    let mask_len = total
        .checked_mul(total)
        .ok_or_else(|| "decode mask shape overflow".to_string())?;
    let block_start = fm_seq_len - 1;
    let mut mask = vec![false; mask_len];
    for query in 0..total {
        for key in 0..total {
            mask[query * total + key] = if query < block_start {
                key <= query
            } else if query < fm_seq_len {
                true
            } else {
                key < fm_seq_len || key >= fm_seq_len
            };
        }
    }
    Ok((mask, (0..total).collect()))
}

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
    fused_adaln_w: Vec<f32>,
    fused_adaln_b: Vec<f32>,
    pub out_linear_w: Vec<f32>,
    pub out_linear_b: Vec<f32>,
}

impl DiT {
    pub fn from_source(source: &dyn TensorSource, config: DotsTtsConfig) -> Result<Self, String> {
        let w = |name: &str, dims: &[u64]| -> Result<Vec<f32>, String> {
            load_f16_f32(source, name, dims)
        };
        let input_w = w("dotstts.dit.input_layer.weight", &[DIT_HIDDEN as u64; 2])?;
        let input_b = w("dotstts.dit.input_layer.bias", &[DIT_HIDDEN as u64])?;
        let time_w0 = w(
            "dotstts.dit.time_embedder.mlp.0.weight",
            &[TIME_EMBED_DIM as u64, DIT_HIDDEN as u64],
        )?;
        let time_b0 = w("dotstts.dit.time_embedder.mlp.0.bias", &[DIT_HIDDEN as u64])?;
        let time_w2 = w(
            "dotstts.dit.time_embedder.mlp.2.weight",
            &[DIT_HIDDEN as u64; 2],
        )?;
        let time_b2 = w("dotstts.dit.time_embedder.mlp.2.bias", &[DIT_HIDDEN as u64])?;
        let fused_adaln_dim = (6 * config.dit_layers + 2) * DIT_HIDDEN;
        let mut fused_adaln_w = Vec::with_capacity(DIT_HIDDEN * fused_adaln_dim);
        let mut fused_adaln_b = Vec::with_capacity(fused_adaln_dim);
        let mut blocks = Vec::with_capacity(config.dit_layers);
        for layer in 0..config.dit_layers {
            let name = |suffix: &str| format!("dotstts.dit.blocks.{layer}.{suffix}");
            let hid2 = [DIT_HIDDEN as u64; 2];
            fused_adaln_w.extend(w(
                &name("adaLN_modulation.1.weight"),
                &[DIT_HIDDEN as u64, (6 * DIT_HIDDEN) as u64],
            )?);
            fused_adaln_b.extend(w(
                &name("adaLN_modulation.1.bias"),
                &[(6 * DIT_HIDDEN) as u64],
            )?);
            blocks.push(DitBlockWeights {
                q: w(&name("attn.q.weight"), &hid2)?,
                k: w(&name("attn.k.weight"), &hid2)?,
                v: w(&name("attn.v.weight"), &hid2)?,
                o: w(&name("attn.o.weight"), &hid2)?,
                o_bias: w(&name("attn.o.bias"), &[DIT_HIDDEN as u64])?,
                q_norm: w(&name("attn.q_norm.weight"), &[DIT_HEAD_DIM as u64])?,
                k_norm: w(&name("attn.k_norm.weight"), &[DIT_HEAD_DIM as u64])?,
                fc1: w(
                    &name("ffn.fc1.weight"),
                    &[DIT_HIDDEN as u64, DIT_FFN as u64],
                )?,
                fc1_bias: w(&name("ffn.fc1.bias"), &[DIT_FFN as u64])?,
                fc2: w(
                    &name("ffn.fc2.weight"),
                    &[DIT_FFN as u64, DIT_HIDDEN as u64],
                )?,
                fc2_bias: w(&name("ffn.fc2.bias"), &[DIT_HIDDEN as u64])?,
            });
        }
        fused_adaln_w.extend(w(
            "dotstts.dit.output_layer.adaLN_modulation.1.weight",
            &[DIT_HIDDEN as u64, (2 * DIT_HIDDEN) as u64],
        )?);
        fused_adaln_b.extend(w(
            "dotstts.dit.output_layer.adaLN_modulation.1.bias",
            &[(2 * DIT_HIDDEN) as u64],
        )?);
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
            fused_adaln_w,
            fused_adaln_b,
            out_linear_w,
            out_linear_b,
        })
    }

    /// Reference `TimestepEmbedder.timestep_embedding`: cos/sin over 256 dims.
    pub fn time_embedding(t: f32) -> Vec<f32> {
        let half = TIME_EMBED_DIM / 2;
        let mut embedding = vec![0.0f32; TIME_EMBED_DIM];
        for k in 0..half {
            let freq = torch28_exp(-(10_000.0f32).ln() * k as f32 / half as f32);
            let arg = t * freq;
            let (cos, sin) = rope_sin_cos_sleef(arg);
            embedding[k] = cos;
            embedding[half + k] = sin;
        }
        embedding
    }

    fn time_mlp_batch(&self, times: &[f32], trace_internal: bool) -> Vec<f32> {
        let mut embeddings = Vec::with_capacity(times.len() * TIME_EMBED_DIM);
        for &time in times {
            embeddings.extend(Self::time_embedding(time));
        }
        #[cfg(feature = "parity-trace")]
        if trace_internal {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.dit.time_embedding",
                None,
                &[times.len(), TIME_EMBED_DIM],
                &embeddings,
            ));
        }
        let mut hidden = vec![0.0f32; times.len() * DIT_HIDDEN];
        linear_forward(
            &self.time_w0,
            Some(&self.time_b0),
            &embeddings,
            TIME_EMBED_DIM,
            DIT_HIDDEN,
            &mut hidden,
        );
        #[cfg(feature = "parity-trace")]
        if trace_internal {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.dit.time_linear0",
                None,
                &[times.len(), DIT_HIDDEN],
                &hidden,
            ));
        }
        for value in hidden.iter_mut() {
            *value = torch28_silu(*value);
        }
        #[cfg(feature = "parity-trace")]
        if trace_internal {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.dit.time_silu",
                None,
                &[times.len(), DIT_HIDDEN],
                &hidden,
            ));
        }
        let mut out = vec![0.0f32; times.len() * DIT_HIDDEN];
        linear_forward(
            &self.time_w2,
            Some(&self.time_b2),
            &hidden,
            DIT_HIDDEN,
            DIT_HIDDEN,
            &mut out,
        );
        #[cfg(feature = "parity-trace")]
        if trace_internal {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.dit.time",
                None,
                &[times.len(), DIT_HIDDEN],
                &out,
            ));
        }
        out
    }

    fn prepare_flow_matching_mods(
        &self,
        g_cond: &[f32],
        step: usize,
        nfe: usize,
        trace_internal: bool,
    ) -> Vec<f32> {
        let times = [step as f32 / nfe as f32; 2];
        let mut condition = self.time_mlp_batch(&times, trace_internal);
        for (row, values) in condition.chunks_exact_mut(DIT_HIDDEN).enumerate() {
            if row == 0 {
                for (value, &g) in values.iter_mut().zip(g_cond) {
                    *value += g;
                }
            }
            for value in values {
                *value = torch28_silu(*value);
            }
        }
        #[cfg(feature = "parity-trace")]
        if trace_internal {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.dit.mods_input",
                None,
                &[2, DIT_HIDDEN],
                &condition,
            ));
        }
        let mut mods = vec![0.0; 2 * self.fused_adaln_b.len()];
        linear_forward(
            &self.fused_adaln_w,
            Some(&self.fused_adaln_b),
            &condition,
            DIT_HIDDEN,
            self.fused_adaln_b.len(),
            &mut mods,
        );
        mods
    }

    /// Full DiT forward. `x` is `[rows, DIT_HIDDEN]` with both CFG branches
    /// already stacked (same mask/positions apply per branch), `g_cond` is one
    /// 1024-dim vector per row, `mask` is `[branch_len, branch_len]` bool,
    /// `positions` is `[branch_len]`. Output `[rows, latent_dim]`.
    fn forward(
        &self,
        x_in: &[f32],
        all_mods: &[f32],
        mask: &[bool],
        positions: &[usize],
        out: &mut [f32],
        trace_internal: bool,
    ) {
        let branch_len = positions.len();
        let rows = x_in.len() / DIT_HIDDEN;
        let mods_per_branch = self.fused_adaln_b.len();
        debug_assert_eq!(all_mods.len(), 2 * mods_per_branch);
        let mut x = vec![0.0f32; rows * DIT_HIDDEN];
        linear_forward(
            &self.input_w,
            Some(&self.input_b),
            x_in,
            DIT_HIDDEN,
            DIT_HIDDEN,
            &mut x,
        );
        #[cfg(feature = "parity-trace")]
        if trace_internal {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.dit.projected_input",
                None,
                &[2, branch_len, DIT_HIDDEN],
                &x,
            ));
        }

        // Per-block scratch reused across layers.
        let mut attn_in = vec![0.0f32; rows * DIT_HIDDEN];
        let mut q = vec![0.0f32; rows * DIT_HIDDEN];
        let mut k = vec![0.0f32; rows * DIT_HIDDEN];
        let mut v = vec![0.0f32; rows * DIT_HIDDEN];
        let mut attn_out = vec![0.0f32; rows * DIT_HIDDEN];
        let mut attn_proj = vec![0.0f32; rows * DIT_HIDDEN];
        let mut h = vec![0.0f32; DIT_HIDDEN];
        let mut ffn_in = vec![0.0f32; rows * DIT_HIDDEN];
        let mut ffn_buf = vec![0.0f32; rows * DIT_FFN];
        let mut ffn_out = vec![0.0f32; rows * DIT_HIDDEN];

        for (block_index, block) in self.blocks.iter().enumerate() {
            // 1. broadcast the fused per-branch adaLN mods over sequence rows.
            let block_mod_start = block_index * 6 * DIT_HIDDEN;
            #[cfg(feature = "parity-trace")]
            let trace_block0 = trace_internal && block_index == 0;
            #[cfg(feature = "parity-trace")]
            if trace_block0 {
                let mut branch_mods = Vec::with_capacity(2 * 6 * DIT_HIDDEN);
                for branch in 0..2 {
                    let start = branch * mods_per_branch + block_mod_start;
                    branch_mods.extend_from_slice(&all_mods[start..start + 6 * DIT_HIDDEN]);
                }
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.dit.block_mods",
                    Some(block_index),
                    &[2, 6, DIT_HIDDEN],
                    &branch_mods,
                ));
            }
            #[cfg(feature = "parity-trace")]
            let mut traced_norm1 = trace_block0.then(|| vec![0.0f32; rows * DIT_HIDDEN]);
            for row in 0..rows {
                let branch = row / branch_len;
                let mrow = branch * mods_per_branch + block_mod_start;
                let xrow = &x[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
                layernorm_no_affine(xrow, &mut h, DIT_NORM_EPS);
                #[cfg(feature = "parity-trace")]
                if let Some(values) = traced_norm1.as_mut() {
                    values[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN].copy_from_slice(&h);
                }
                let (sa, sca) = (
                    &all_mods[mrow..mrow + DIT_HIDDEN],
                    &all_mods[mrow + DIT_HIDDEN..mrow + 2 * DIT_HIDDEN],
                );
                let dst = &mut attn_in[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
                for d in 0..DIT_HIDDEN {
                    dst[d] = h[d] * (1.0 + sca[d]) + sa[d];
                }
                let _ = sa;
            }
            #[cfg(feature = "parity-trace")]
            if let Some(values) = traced_norm1.as_ref() {
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.dit.block0.norm1",
                    Some(block_index),
                    &[2, branch_len, DIT_HIDDEN],
                    values,
                ));
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.dit.block0.attn_in",
                    Some(block_index),
                    &[2, branch_len, DIT_HIDDEN],
                    &attn_in,
                ));
            }
            // 2. QKV projections (all rows at once)
            linear_forward(&block.q, None, &attn_in, DIT_HIDDEN, DIT_HIDDEN, &mut q);
            linear_forward(&block.k, None, &attn_in, DIT_HIDDEN, DIT_HIDDEN, &mut k);
            linear_forward(&block.v, None, &attn_in, DIT_HIDDEN, DIT_HIDDEN, &mut v);
            #[cfg(feature = "parity-trace")]
            if trace_block0 {
                let mut qkv = Vec::with_capacity(rows * 3 * DIT_HIDDEN);
                for row in 0..rows {
                    let range = row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN;
                    qkv.extend_from_slice(&q[range.clone()]);
                    qkv.extend_from_slice(&k[range.clone()]);
                    qkv.extend_from_slice(&v[range]);
                }
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.dit.block0.qkv",
                    Some(block_index),
                    &[2, branch_len, 3 * DIT_HIDDEN],
                    &qkv,
                ));
            }
            // 3. q/k RMSNorm (learned) + rotary at absolute positions
            #[cfg(feature = "parity-trace")]
            let mut traced_q_norm = trace_block0.then(|| vec![0.0f32; rows * DIT_HIDDEN]);
            #[cfg(feature = "parity-trace")]
            let mut traced_k_norm = trace_block0.then(|| vec![0.0f32; rows * DIT_HIDDEN]);
            for row in 0..rows {
                for head in 0..DIT_HEADS {
                    let start = row * DIT_HIDDEN + head * DIT_HEAD_DIM;
                    let end = start + DIT_HEAD_DIM;
                    let qh = &mut q[start..end];
                    let kh = &mut k[start..end];
                    rms_norm_weighted(qh, &block.q_norm, f32::EPSILON);
                    rms_norm_weighted(kh, &block.k_norm, f32::EPSILON);
                    #[cfg(feature = "parity-trace")]
                    if let Some(values) = traced_q_norm.as_mut() {
                        values[start..end].copy_from_slice(qh);
                    }
                    #[cfg(feature = "parity-trace")]
                    if let Some(values) = traced_k_norm.as_mut() {
                        values[start..end].copy_from_slice(kh);
                    }
                }
            }
            dots_rotary(&mut q, positions, DIT_HEADS, DIT_HEAD_DIM, DIT_ROPE_THETA);
            dots_rotary(&mut k, positions, DIT_HEADS, DIT_HEAD_DIM, DIT_ROPE_THETA);
            #[cfg(feature = "parity-trace")]
            if let (Some(q_values), Some(k_values)) =
                (traced_q_norm.as_ref(), traced_k_norm.as_ref())
            {
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.dit.block0.q_norm",
                    Some(block_index),
                    &[2, branch_len, DIT_HEADS, DIT_HEAD_DIM],
                    q_values,
                ));
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.dit.block0.k_norm",
                    Some(block_index),
                    &[2, branch_len, DIT_HEADS, DIT_HEAD_DIM],
                    k_values,
                ));
            }
            // 4. attention with the reference mask
            self.attention(&q, &k, &v, &mut attn_out, rows, branch_len, mask);
            #[cfg(feature = "parity-trace")]
            if trace_block0 {
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.dit.block0.attn_out",
                    Some(block_index),
                    &[2, branch_len, DIT_HIDDEN],
                    &attn_out,
                ));
            }
            // 5. o_proj + residual + FFN
            linear_forward(
                &block.o,
                Some(&block.o_bias),
                &attn_out,
                DIT_HIDDEN,
                DIT_HIDDEN,
                &mut attn_proj,
            );
            #[cfg(feature = "parity-trace")]
            let traced_attn_proj = trace_block0.then(|| attn_proj.clone());
            #[cfg(feature = "parity-trace")]
            let mut traced_norm2 = trace_block0.then(|| vec![0.0f32; rows * DIT_HIDDEN]);
            for row in 0..rows {
                let projection = &attn_proj[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
                let branch = row / branch_len;
                let mrow = branch * mods_per_branch + block_mod_start;
                let ga = &all_mods[mrow + 2 * DIT_HIDDEN..mrow + 3 * DIT_HIDDEN];
                let xrow = &mut x[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
                for d in 0..DIT_HIDDEN {
                    xrow[d] += ga[d] * projection[d];
                }
                layernorm_no_affine(xrow, &mut h, DIT_NORM_EPS);
                #[cfg(feature = "parity-trace")]
                if let Some(values) = traced_norm2.as_mut() {
                    values[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN].copy_from_slice(&h);
                }
                let (sf, scf) = (
                    &all_mods[mrow + 3 * DIT_HIDDEN..mrow + 4 * DIT_HIDDEN],
                    &all_mods[mrow + 4 * DIT_HIDDEN..mrow + 5 * DIT_HIDDEN],
                );
                let ffn_row = &mut ffn_in[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
                for d in 0..DIT_HIDDEN {
                    ffn_row[d] = h[d] * (1.0 + scf[d]) + sf[d];
                }
            }
            linear_forward(
                &block.fc1,
                Some(&block.fc1_bias),
                &ffn_in,
                DIT_HIDDEN,
                DIT_FFN,
                &mut ffn_buf,
            );
            #[cfg(feature = "parity-trace")]
            let traced_ffn_in = trace_block0.then(|| ffn_in.clone());
            #[cfg(feature = "parity-trace")]
            let traced_fc1 = trace_block0.then(|| ffn_buf.clone());
            for value in &mut ffn_buf {
                *value = gelu_tanh(*value);
            }
            #[cfg(feature = "parity-trace")]
            let traced_gelu = trace_block0.then(|| ffn_buf.clone());
            linear_forward(
                &block.fc2,
                Some(&block.fc2_bias),
                &ffn_buf,
                DIT_FFN,
                DIT_HIDDEN,
                &mut ffn_out,
            );
            #[cfg(feature = "parity-trace")]
            let traced_fc2 = trace_block0.then(|| ffn_out.clone());
            for row in 0..rows {
                let branch = row / branch_len;
                let mrow = branch * mods_per_branch + block_mod_start;
                let gf = &all_mods[mrow + 5 * DIT_HIDDEN..mrow + 6 * DIT_HIDDEN];
                let xrow = &mut x[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
                let ffn_row = &ffn_out[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
                for d in 0..DIT_HIDDEN {
                    xrow[d] += gf[d] * ffn_row[d];
                }
            }
            #[cfg(feature = "parity-trace")]
            if let (Some(attn_proj), Some(norm2), Some(ffn_in), Some(fc1), Some(gelu), Some(fc2)) = (
                traced_attn_proj.as_ref(),
                traced_norm2.as_ref(),
                traced_ffn_in.as_ref(),
                traced_fc1.as_ref(),
                traced_gelu.as_ref(),
                traced_fc2.as_ref(),
            ) {
                for (name, shape, values) in [
                    (
                        "dots.dit.block0.attn_proj",
                        [2, branch_len, DIT_HIDDEN],
                        attn_proj.as_slice(),
                    ),
                    (
                        "dots.dit.block0.norm2",
                        [2, branch_len, DIT_HIDDEN],
                        norm2.as_slice(),
                    ),
                    (
                        "dots.dit.block0.ffn_in",
                        [2, branch_len, DIT_HIDDEN],
                        ffn_in.as_slice(),
                    ),
                ] {
                    crate::parity_trace::report(crate::parity_trace::checkpoint(
                        name,
                        Some(block_index),
                        &shape,
                        values,
                    ));
                }
                for (name, values) in [
                    ("dots.dit.block0.fc1", fc1.as_slice()),
                    ("dots.dit.block0.gelu", gelu.as_slice()),
                ] {
                    crate::parity_trace::report(crate::parity_trace::checkpoint(
                        name,
                        Some(block_index),
                        &[2, branch_len, DIT_FFN],
                        values,
                    ));
                }
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.dit.block0.fc2",
                    Some(block_index),
                    &[2, branch_len, DIT_HIDDEN],
                    fc2,
                ));
            }
            #[cfg(feature = "parity-trace")]
            if trace_internal {
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.dit.block",
                    Some(block_index),
                    &[2, branch_len, DIT_HIDDEN],
                    &x,
                ));
            }
        }

        // output layer: adaLN shift/scale on a no-affine LayerNorm, then linear
        let final_mod_start = self.blocks.len() * 6 * DIT_HIDDEN;
        #[cfg(feature = "parity-trace")]
        let mut traced_final_norm = trace_internal.then(|| vec![0.0f32; rows * DIT_HIDDEN]);
        #[cfg(feature = "parity-trace")]
        let mut traced_final_input = trace_internal.then(|| vec![0.0f32; rows * DIT_HIDDEN]);
        for row in 0..rows {
            let branch = row / branch_len;
            let mod_start = branch * mods_per_branch + final_mod_start;
            let shift = &all_mods[mod_start..mod_start + DIT_HIDDEN];
            let scale = &all_mods[mod_start + DIT_HIDDEN..mod_start + 2 * DIT_HIDDEN];
            let xrow = &x[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
            layernorm_no_affine(xrow, &mut h, DIT_NORM_EPS);
            #[cfg(feature = "parity-trace")]
            if let Some(values) = traced_final_norm.as_mut() {
                values[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN].copy_from_slice(&h);
            }
            let final_row = &mut ffn_in[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN];
            for d in 0..DIT_HIDDEN {
                final_row[d] = h[d] * (1.0 + scale[d]) + shift[d];
            }
            #[cfg(feature = "parity-trace")]
            if let Some(values) = traced_final_input.as_mut() {
                values[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN].copy_from_slice(final_row);
            }
        }
        linear_forward(
            &self.out_linear_w,
            Some(&self.out_linear_b),
            &ffn_in,
            DIT_HIDDEN,
            self.n_latent,
            out,
        );
        #[cfg(feature = "parity-trace")]
        if let (Some(norm), Some(input)) = (traced_final_norm.as_ref(), traced_final_input.as_ref())
        {
            for (name, shape, values) in [
                (
                    "dots.dit.final.norm",
                    [2, branch_len, DIT_HIDDEN],
                    norm.as_slice(),
                ),
                (
                    "dots.dit.final.input",
                    [2, branch_len, DIT_HIDDEN],
                    input.as_slice(),
                ),
            ] {
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    name, None, &shape, values,
                ));
            }
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.dit.final.output",
                None,
                &[2, branch_len, self.n_latent],
                out,
            ));
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
        #[cfg(any(
            target_os = "macos",
            all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        ))]
        {
            const QUERY_BLOCK: usize = 32;
            let scale = 1.0 / (DIT_HEAD_DIM as f32).sqrt();
            let mut qk = vec![0.0f32; QUERY_BLOCK * branch_len];
            let mut dst = vec![0.0f32; QUERY_BLOCK * DIT_HEAD_DIM];
            let mut sums = [0.0f32; QUERY_BLOCK];
            out.fill(0.0);

            for branch in 0..rows / branch_len {
                let branch_offset = branch * branch_len * DIT_HIDDEN;
                for head in 0..DIT_HEADS {
                    let head_offset = branch_offset + head * DIT_HEAD_DIM;
                    for query_start in (0..branch_len).step_by(QUERY_BLOCK) {
                        let queries = QUERY_BLOCK.min(branch_len - query_start);
                        unsafe {
                            sys::cblas_sgemm(
                                102,
                                112,
                                111,
                                branch_len as i32,
                                queries as i32,
                                DIT_HEAD_DIM as i32,
                                1.0,
                                k.as_ptr().add(head_offset),
                                DIT_HIDDEN as i32,
                                q.as_ptr().add(head_offset + query_start * DIT_HIDDEN),
                                DIT_HIDDEN as i32,
                                0.0,
                                qk.as_mut_ptr(),
                                branch_len as i32,
                            );
                        }

                        for query in 0..queries {
                            let row = &mut qk[query * branch_len..(query + 1) * branch_len];
                            let mask_row = &mask[(query_start + query) * branch_len
                                ..(query_start + query + 1) * branch_len];
                            let mut max = f32::NEG_INFINITY;
                            for (score, &allowed) in row.iter_mut().zip(mask_row) {
                                *score = if allowed {
                                    *score * scale
                                } else {
                                    f32::NEG_INFINITY
                                };
                                max = max.max(*score);
                            }

                            let mut lanes = [0.0f32; 4];
                            let vector_end = branch_len - branch_len % 4;
                            for key in 0..vector_end {
                                let weight = torch28_exp(row[key] - max);
                                row[key] = weight;
                                lanes[key % 4] += weight;
                            }
                            let mut sum = torch28_sum4(&lanes);
                            for score in &mut row[vector_end..] {
                                *score = (*score - max).exp();
                                sum += *score;
                            }
                            sums[query] = sum;
                        }

                        unsafe {
                            sys::cblas_sgemm(
                                102,
                                111,
                                111,
                                DIT_HEAD_DIM as i32,
                                queries as i32,
                                branch_len as i32,
                                1.0,
                                v.as_ptr().add(head_offset),
                                DIT_HIDDEN as i32,
                                qk.as_ptr(),
                                branch_len as i32,
                                0.0,
                                dst.as_mut_ptr(),
                                DIT_HEAD_DIM as i32,
                            );
                        }
                        for query in 0..queries {
                            let reciprocal = sums[query].recip();
                            let source = &dst[query * DIT_HEAD_DIM..(query + 1) * DIT_HEAD_DIM];
                            let output_start = head_offset + (query_start + query) * DIT_HIDDEN;
                            for (output, &value) in out[output_start..output_start + DIT_HEAD_DIM]
                                .iter_mut()
                                .zip(source)
                            {
                                *output = value * reciprocal;
                            }
                        }
                    }
                }
            }
            return;
        }

        #[cfg(not(any(
            target_os = "macos",
            all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        )))]
        {
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
                        let score =
                            dot_f32(qh, &k[koff..koff + DIT_HEAD_DIM], DIT_HEAD_DIM) * scale;
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
        let total = fm_seq_len
            .checked_add(patch)
            .ok_or_else(|| "solve_patch: sequence length overflow".to_string())?;
        if sequence.len() != fm_seq_len * DIT_HIDDEN
            || cfg_sequence.len() != fm_seq_len * DIT_HIDDEN
        {
            return Err("solve_patch: sequence width mismatch".into());
        }
        if z0.len() != patch * self.n_latent {
            return Err("solve_patch: noise width mismatch".into());
        }
        if g_cond.len() != DIT_HIDDEN {
            return Err("solve_patch: g_cond width mismatch".into());
        }

        // reference mask (EagerDiTRunner._build_decode_mask)
        let latent_start = total - patch;
        let (mask, positions) = build_decode_mask_positions(fm_seq_len, patch)?;

        // input tensors: [cond seq; uncond seq] + z region
        let mut x = vec![0.0f32; 2 * total * DIT_HIDDEN];
        for (branch, src) in [(0usize, sequence), (1, cfg_sequence)] {
            let base = branch * total * DIT_HIDDEN;
            x[base..base + fm_seq_len * DIT_HIDDEN]
                .copy_from_slice(&src[..fm_seq_len * DIT_HIDDEN]);
        }
        #[cfg(feature = "parity-trace")]
        let trace_internal = !TRACE_DIT_INTERNAL.swap(true, Ordering::Relaxed);
        #[cfg(not(feature = "parity-trace"))]
        let trace_internal = false;
        let mut z = z0.to_vec();
        let mut z_proj = vec![0.0f32; patch * DIT_HIDDEN];
        let mut velocity = vec![0.0f32; 2 * patch * self.n_latent];
        let mut diag = vec![0.0f32; 2 * total * self.n_latent];
        for step in 0..nfe {
            // z_proj = coordinate_proj(z)
            linear_forward(
                coordinate_proj,
                Some(coordinate_bias),
                &z,
                self.n_latent,
                DIT_HIDDEN,
                &mut z_proj,
            );
            // splat into the latent region of both branches
            for branch in 0..2 {
                let base = branch * total * DIT_HIDDEN + latent_start * DIT_HIDDEN;
                x[base..base + patch * DIT_HIDDEN].copy_from_slice(&z_proj);
            }
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint_at(
                "dots.dit.input",
                None,
                Some(step),
                &[2, total, DIT_HIDDEN],
                &x,
            ));
            let trace_step = trace_internal && step <= 1;
            let mods = self.prepare_flow_matching_mods(g_cond, step, nfe, trace_step);
            self.forward(&x, &mods, &mask, &positions, &mut diag, trace_step);
            // extract the latent rows of each branch
            for branch in 0..2 {
                let src = &diag[branch * total * self.n_latent + latent_start * self.n_latent
                    ..branch * total * self.n_latent + (latent_start + patch) * self.n_latent];
                velocity[branch * patch * self.n_latent..(branch + 1) * patch * self.n_latent]
                    .copy_from_slice(src);
            }
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint_at(
                "dots.dit.raw_velocity",
                None,
                Some(step),
                &[2, patch, self.n_latent],
                &velocity,
            ));
            // CFG
            let mut guided = vec![0.0f32; patch * self.n_latent];
            for i in 0..patch * self.n_latent {
                let cond_v = velocity[i];
                let uncond_v = velocity[patch * self.n_latent + i];
                guided[i] = cond_v + guidance * (cond_v - uncond_v);
                z[i] = torch28_euler_step(z[i], guided[i], nfe);
            }
            #[cfg(feature = "parity-trace")]
            {
                crate::parity_trace::report(crate::parity_trace::checkpoint_at(
                    "dots.dit.velocity",
                    None,
                    Some(step),
                    &[1, patch, self.n_latent],
                    &guided,
                ));
                crate::parity_trace::report(crate::parity_trace::checkpoint_at(
                    "dots.dit.z",
                    None,
                    Some(step),
                    &[1, patch, self.n_latent],
                    &z,
                ));
            }
        }
        out.copy_from_slice(&z);
        Ok(())
    }

    pub fn n_latent_slots(&self) -> usize {
        4 // patch_size is fixed at 4 for all dots.tts artifacts
    }
}

fn add_torch28_moments(
    count_add: usize,
    mean_add: f32,
    m2_add: f32,
    count: &mut usize,
    mean: &mut f32,
    m2: &mut f32,
) {
    let total = *count + count_add;
    let weight = if total == 0 {
        0.0
    } else {
        count_add as f32 / total as f32
    };
    let delta = mean_add - *mean;
    *mean = weight.mul_add(delta, *mean);
    *m2 += (delta * delta * weight).mul_add(*count as f32, m2_add);
    *count = total;
}

fn add_torch28_moments4(
    count_add: usize,
    mean_add: &[f32; 4],
    m2_add: &[f32; 4],
    count: &mut usize,
    mean: &mut [f32; 4],
    m2: &mut [f32; 4],
) {
    let total = *count + count_add;
    let weight = if total == 0 {
        0.0
    } else {
        count_add as f32 / total as f32
    };
    for lane in 0..4 {
        let delta = mean_add[lane] - mean[lane];
        mean[lane] += weight * delta;
        m2[lane] += m2_add[lane] + delta * delta * weight * *count as f32;
    }
    *count = total;
}

fn layernorm_no_affine(x: &[f32], out: &mut [f32], eps: f32) {
    debug_assert_eq!(x.len(), DIT_HIDDEN);
    debug_assert_eq!(out.len(), DIT_HIDDEN);

    let mut counts = [0usize; 4];
    let mut means = [[0.0f32; 4]; 4];
    let mut m2s = [[0.0f32; 4]; 4];
    for chunk in 0..16 {
        let mut chunk_mean = [0.0f32; 4];
        let mut chunk_m2 = [0.0f32; 4];
        for vector in 0..16 {
            let weight = 1.0 / (vector + 1) as f32;
            let start = (chunk * 16 + vector) * 4;
            for lane in 0..4 {
                let value = x[start + lane];
                let delta = value - chunk_mean[lane];
                chunk_mean[lane] += delta * weight;
                chunk_m2[lane] += delta * (value - chunk_mean[lane]);
            }
        }
        add_torch28_moments4(
            16,
            &chunk_mean,
            &chunk_m2,
            &mut counts[0],
            &mut means[0],
            &mut m2s[0],
        );
        let mut mask = chunk + 1;
        let mut depth = 1;
        while depth < 4 && mask & 1 == 0 {
            let count_add = counts[depth - 1];
            let mean_add = means[depth - 1];
            let m2_add = m2s[depth - 1];
            add_torch28_moments4(
                count_add,
                &mean_add,
                &m2_add,
                &mut counts[depth],
                &mut means[depth],
                &mut m2s[depth],
            );
            counts[depth - 1] = 0;
            means[depth - 1] = [0.0; 4];
            m2s[depth - 1] = [0.0; 4];
            mask >>= 1;
            depth += 1;
        }
    }
    for depth in 1..4 {
        let count_add = counts[depth];
        let mean_add = means[depth];
        let m2_add = m2s[depth];
        add_torch28_moments4(
            count_add,
            &mean_add,
            &m2_add,
            &mut counts[0],
            &mut means[0],
            &mut m2s[0],
        );
    }
    let mut count = 0;
    let mut mean = 0.0f32;
    let mut var = 0.0f32;
    for lane in 0..4 {
        add_torch28_moments(
            DIT_HIDDEN / 4,
            means[0][lane],
            m2s[0][lane],
            &mut count,
            &mut mean,
            &mut var,
        );
    }
    var /= DIT_HIDDEN as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for (o, &value) in out.iter_mut().zip(x.iter()) {
        *o = (value + -mean) * inv;
    }
}

fn rms_norm_weighted(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), DIT_HEAD_DIM);
    debug_assert_eq!(weight.len(), DIT_HEAD_DIM);

    let mut partials = [[0.0f32; 4]; 4];
    for row in 0..4 {
        for partial in 0..4 {
            let start = (row * 4 + partial) * 4;
            for lane in 0..4 {
                let value = x[start + lane];
                partials[partial][lane] += value * value;
            }
        }
    }
    for partial in 1..4 {
        for lane in 0..4 {
            partials[0][lane] += partials[partial][lane];
        }
    }
    let mut mean_sq = 0.0f32;
    for value in partials[0] {
        mean_sq += value;
    }
    mean_sq /= DIT_HEAD_DIM as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for (value, &w) in x.iter_mut().zip(weight.iter()) {
        *value *= inv;
        *value *= w;
    }
}

fn gelu_tanh(x: f32) -> f32 {
    let x_cube = x * x * x;
    0.5 * x * (1.0 + torch28_tanh(f32::from_bits(0x3f4c_422a) * (x + 0.044715 * x_cube)))
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
    fn time_embedding_matches_torch28_arm_bits_at_second_ode_step() {
        let emb = DiT::time_embedding(0.1);
        assert_eq!(emb[27].to_bits(), 0x3f7f_f946);
        assert_eq!(emb[128 + 13].to_bits(), 0x3d20_b18e);
    }

    #[cfg(any(
        target_os = "macos",
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
    #[test]
    #[ignore = "requires fixed dots mmproj and Torch time-embedding sidecar"]
    fn production_time_mlp_matches_pinned_oracle_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let source = open_model_source(
            std::path::Path::new(&std::env::var_os("DOTS_DIT_MMPROJ").unwrap()),
            ComponentRole::Mmproj,
        )
        .unwrap();
        let config = DotsTtsConfig::from_source(source.as_ref()).unwrap();
        let dit = DiT::from_source(source.as_ref(), config).unwrap();
        let expected = std::fs::read(std::env::var_os("DOTS_DIT_TIME").unwrap()).unwrap();
        let expected = expected
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        let rows = expected.len() / DIT_HIDDEN;
        let nfe = rows / 2;
        let times = (0..rows)
            .map(|row| (row / 2) as f32 / nfe as f32)
            .collect::<Vec<_>>();
        let actual = dit.time_mlp_batch(&times, false);
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "time[{index}] rust={:08x} oracle={:08x}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }

    #[cfg(any(
        target_os = "macos",
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
    #[test]
    #[ignore = "requires fixed dots mmproj and Torch DiT sidecars"]
    fn production_block_mods_match_pinned_oracle_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let source = open_model_source(
            std::path::Path::new(&std::env::var_os("DOTS_DIT_MMPROJ").unwrap()),
            ComponentRole::Mmproj,
        )
        .unwrap();
        let config = DotsTtsConfig::from_source(source.as_ref()).unwrap();
        let dit = DiT::from_source(source.as_ref(), config).unwrap();
        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let time = read("DOTS_DIT_TIME");
        let g_cond = read("DOTS_DIT_G_COND");
        let expected = read("DOTS_DIT_BLOCK_MODS");
        let nfe = time.len() / (2 * DIT_HIDDEN);
        let all_mods = dit.prepare_flow_matching_mods(&g_cond, 0, nfe, false);
        let mods_per_branch = dit.fused_adaln_b.len();
        let mut actual = Vec::with_capacity(2 * 6 * DIT_HIDDEN);
        for branch in 0..2 {
            let start = branch * mods_per_branch;
            actual.extend_from_slice(&all_mods[start..start + 6 * DIT_HIDDEN]);
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "block_mods[{index}] rust={:08x} oracle={:08x}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }

    #[cfg(any(
        target_os = "macos",
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
    #[test]
    #[ignore = "requires fixed Torch DiT LayerNorm sidecars"]
    fn production_layernorm_matches_pinned_oracle_bitwise() {
        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let input = read("DOTS_DIT_LAYERNORM_INPUT");
        let expected = read("DOTS_DIT_LAYERNORM_EXPECTED");
        assert_eq!(input.len(), expected.len());
        assert_eq!(input.len() % DIT_HIDDEN, 0);

        let mut actual = vec![0.0f32; DIT_HIDDEN];
        for (row, (input, expected)) in input
            .chunks_exact(DIT_HIDDEN)
            .zip(expected.chunks_exact(DIT_HIDDEN))
            .enumerate()
        {
            layernorm_no_affine(input, &mut actual, DIT_NORM_EPS);
            for (column, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "layernorm[{row},{column}] rust={:08x} oracle={:08x}",
                    actual.to_bits(),
                    expected.to_bits()
                );
            }
        }
    }

    #[cfg(any(
        target_os = "macos",
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
    #[test]
    #[ignore = "requires fixed dots mmproj and Torch DiT RMSNorm sidecars"]
    fn production_rms_norm_matches_pinned_oracle_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let source = open_model_source(
            std::path::Path::new(&std::env::var_os("DOTS_DIT_MMPROJ").unwrap()),
            ComponentRole::Mmproj,
        )
        .unwrap();
        let config = DotsTtsConfig::from_source(source.as_ref()).unwrap();
        let dit = DiT::from_source(source.as_ref(), config).unwrap();
        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let qkv = read("DOTS_DIT_QKV");
        let expected = read("DOTS_DIT_Q_NORM");
        let rows = qkv.len() / (3 * DIT_HIDDEN);
        assert_eq!(qkv.len(), rows * 3 * DIT_HIDDEN);
        assert_eq!(expected.len(), rows * DIT_HIDDEN);

        for row in 0..rows {
            for head in 0..DIT_HEADS {
                let input_start = row * 3 * DIT_HIDDEN + head * DIT_HEAD_DIM;
                let output_start = row * DIT_HIDDEN + head * DIT_HEAD_DIM;
                let mut actual = qkv[input_start..input_start + DIT_HEAD_DIM].to_vec();
                rms_norm_weighted(&mut actual, &dit.blocks[0].q_norm, f32::EPSILON);
                for (column, (&actual, &expected)) in actual
                    .iter()
                    .zip(&expected[output_start..output_start + DIT_HEAD_DIM])
                    .enumerate()
                {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "rms_norm[{row},{head},{column}] rust={:08x} oracle={:08x}",
                        actual.to_bits(),
                        expected.to_bits()
                    );
                }
            }
        }
    }

    #[cfg(any(
        target_os = "macos",
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
    #[test]
    #[ignore = "requires fixed Torch DiT RoPE sidecars"]
    fn production_dots_rotary_matches_pinned_oracle_bitwise() {
        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let input = read("DOTS_DIT_Q_NORM");
        let expected = read("DOTS_DIT_Q_ROPE");
        assert_eq!(input.len(), expected.len());
        assert_eq!(input.len() % DIT_HIDDEN, 0);
        let rows = input.len() / DIT_HIDDEN;
        assert_eq!(rows % 2, 0);
        let branch_len = rows / 2;

        let positions = (0..branch_len).collect::<Vec<_>>();
        let mut actual = input;
        dots_rotary(
            &mut actual,
            &positions,
            DIT_HEADS,
            DIT_HEAD_DIM,
            DIT_ROPE_THETA,
        );
        for row in 0..rows {
            for head in 0..DIT_HEADS {
                let start = row * DIT_HIDDEN + head * DIT_HEAD_DIM;
                for (column, (&actual, &expected)) in actual[start..start + DIT_HEAD_DIM]
                    .iter()
                    .zip(&expected[start..start + DIT_HEAD_DIM])
                    .enumerate()
                {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "rope[{row},{head},{column}] rust={:08x} oracle={:08x}",
                        actual.to_bits(),
                        expected.to_bits()
                    );
                }
            }
        }
    }

    #[cfg(any(
        target_os = "macos",
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
    #[test]
    #[ignore = "requires fixed dots mmproj and Torch DiT attention sidecars"]
    fn production_attention_matches_pinned_oracle_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let source = open_model_source(
            std::path::Path::new(&std::env::var_os("DOTS_DIT_MMPROJ").unwrap()),
            ComponentRole::Mmproj,
        )
        .unwrap();
        let config = DotsTtsConfig::from_source(source.as_ref()).unwrap();
        let dit = DiT::from_source(source.as_ref(), config).unwrap();
        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let qkv = read("DOTS_DIT_QKV");
        let mut q = read("DOTS_DIT_Q_NORM");
        let mut k = read("DOTS_DIT_K_NORM");
        let expected = read("DOTS_DIT_ATTN_OUT");
        let rows = q.len() / DIT_HIDDEN;
        let branch_len = rows / 2;
        let positions = (0..branch_len).collect::<Vec<_>>();
        dots_rotary(&mut q, &positions, DIT_HEADS, DIT_HEAD_DIM, DIT_ROPE_THETA);
        dots_rotary(&mut k, &positions, DIT_HEADS, DIT_HEAD_DIM, DIT_ROPE_THETA);
        let mut v = vec![0.0f32; rows * DIT_HIDDEN];
        for row in 0..rows {
            let src = row * 3 * DIT_HIDDEN + 2 * DIT_HIDDEN;
            v[row * DIT_HIDDEN..(row + 1) * DIT_HIDDEN]
                .copy_from_slice(&qkv[src..src + DIT_HIDDEN]);
        }
        let (mask, _) = build_decode_mask_positions(branch_len - 4, 4).unwrap();
        let mut actual = vec![0.0f32; expected.len()];
        dit.attention(&q, &k, &v, &mut actual, rows, branch_len, &mask);
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "attention[{index}] rust={:08x} oracle={:08x}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    fn gelu_tanh_matches_known_values() {
        assert!((gelu_tanh(0.0) - 0.0).abs() < 1e-6);
        assert!((gelu_tanh(1.0) - 0.841192).abs() < 1e-4);
        assert!((gelu_tanh(-1.0) + 0.158808).abs() < 1e-4);
    }

    #[test]
    fn gelu_tanh_matches_pinned_torch_arm_bits() {
        let cases = [
            (0xbf1d_2d7c, 0xbe29_8bff),
            (0xbc46_207d, 0xbbc4_371f),
            (0xbfb9_7bd5, 0xbddb_0ecd),
            (0xbee8_1b30, 0xbe16_f401),
            (0xbf65_d2e6, 0xbe29_e048),
            (0x3f49_a42e, 0x3f1e_2d7f),
            (0xbec2_337a, 0xbe08_d078),
            (0xbf1b_06af, 0xbe28_f382),
            (0xc022_0bda, 0xbc63_e384),
        ];
        for (input_bits, expected_bits) in cases {
            assert_eq!(
                gelu_tanh(f32::from_bits(input_bits)).to_bits(),
                expected_bits,
                "input={input_bits:08x}"
            );
        }
    }

    #[test]
    fn euler_step_matches_pinned_torch_arm_bits() {
        assert_eq!(
            torch28_euler_step(f32::from_bits(0xbd7d_5c5f), f32::from_bits(0x3ed4_96f1), 10,)
                .to_bits(),
            0xbca6_940a,
        );
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

    #[test]
    fn decode_mask_and_positions_match_reference_layout() {
        let (mask, positions) = build_decode_mask_positions(6, 4).unwrap();
        assert_eq!(positions, (0..10).collect::<Vec<_>>());
        for query in 0..10 {
            for key in 0..10 {
                let expected = if query < 5 {
                    key <= query
                } else if query < 6 {
                    true
                } else {
                    key < 6 || key >= 6
                };
                assert_eq!(mask[query * 10 + key], expected, "q={query} k={key}");
            }
        }
    }
}
