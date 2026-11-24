//! Tessera-small hourglass transformer, with pixel and patch tokens in a common RoPE frame.
mod weights;
pub use weights::Weights;

use self::weights::Projection;
use super::{COLORS as VOCAB, GRID, PIXELS};
use crate::compute::{linear::LinearExecutor, ComputePolicy, UsedBackend};
use crate::core::thread_pool::ComputePool;
use crate::ops::{
    nn::{attention_scalar, AttentionSpec, Linear},
    rms_norm_inplace, rms_unit_inplace, silu_inplace, vec_add_into,
};
use std::sync::Arc;

const D: usize = 512;
const FF: usize = 1536;
const HEADS: usize = 8;
const HEAD_DIM: usize = D / HEADS;
const TEXT_LEN: usize = 24;
const REGISTERS: usize = 4;
const PREFIX: usize = REGISTERS + TEXT_LEN;
const EPS: f32 = 1e-6;

struct Block {
    qkv: Projection,
    proj: Projection,
    w12: Projection,
    w3: Projection,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    kernel: Vec<f32>,
    bias: Vec<f32>,
    mod_bias: Vec<f32>,
}

struct Rope {
    grid: usize,
    // Angles occur only on image tokens, with row frequencies then column frequencies.
    angles: Vec<(f32, f32)>,
}

impl Rope {
    fn new(grid: usize, stride: usize) -> Self {
        let mut angles = Vec::with_capacity(grid * grid * (HEAD_DIM / 2));
        for token in 0..grid * grid {
            for coordinate in [token / grid, token % grid] {
                let position = (coordinate * stride) as f64 + (stride - 1) as f64 * 0.5;
                for i in 0..HEAD_DIM / 4 {
                    let frequency = 100_f64.powf(-(i as f64) / (HEAD_DIM / 4) as f64);
                    let (sin, cos) = (position * frequency).sin_cos();
                    angles.push((sin as f32, cos as f32));
                }
            }
        }
        Self { grid, angles }
    }

    fn normalize_and_rotate(&self, values: &mut [f32], weight: &[f32]) {
        for head in values.chunks_exact_mut(HEAD_DIM) {
            rms_norm_inplace(head, weight, EPS);
        }
        for (token, angles) in values[PREFIX * D..]
            .chunks_exact_mut(D)
            .zip(self.angles.chunks_exact(HEAD_DIM / 2))
        {
            for head in token.chunks_exact_mut(HEAD_DIM) {
                let (lo, hi) = head.split_at_mut(HEAD_DIM / 2);
                for ((a, b), &(sin, cos)) in lo.iter_mut().zip(hi).zip(angles) {
                    (*a, *b) = (*a * cos - *b * sin, *a * sin + *b * cos);
                }
            }
        }
    }
}

fn fourier(x: f32, frequencies: usize, out: &mut Vec<f32>) {
    for f in [f32::sin as fn(f32) -> f32, f32::cos] {
        for i in 0..frequencies {
            out.push(f(x * (std::f32::consts::PI * (1 << i) as f32)));
        }
    }
}

fn color_features(lab: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(lab.len() / 3 * 40);
    for (index, lab) in lab.chunks_exact(3).enumerate() {
        let alpha = if index == 0 { 0.0 } else { 1.0 };
        let raw = [
            (2. * lab[0] - 1.) * alpha,
            lab[1] / 0.4 * alpha,
            lab[2] / 0.4 * alpha,
        ];
        out.extend_from_slice(&raw);
        for x in raw {
            fourier(x, 6, &mut out);
        }
        out.push(alpha);
    }
    out
}

struct Projections<'m, 'w> {
    linears: &'m [Linear<'w>],
    executor: LinearExecutor<'m, 'w>,
    cpu_calls: usize,
    gpu_calls: usize,
}

impl Projections<'_, '_> {
    fn apply(&mut self, Projection(slot): Projection, x: &[f32]) -> Result<Vec<f32>, String> {
        let linear = &self.linears[slot];
        let rows = x.len() / linear.n_in();
        let mut out = vec![0.; rows * linear.n_out()];
        let id = self.executor.id(slot).map_err(|e| e.to_string())?;
        match self
            .executor
            .run(id, x, rows, &mut out)
            .map_err(|e| e.to_string())?
        {
            UsedBackend::Cpu => self.cpu_calls += 1,
            UsedBackend::Vulkan => self.gpu_calls += 1,
        }
        if let Some(bias) = linear.bias() {
            for row in out.chunks_exact_mut(linear.n_out()) {
                vec_add_into(bias, row);
            }
        }
        Ok(out)
    }

    fn mlp(&mut self, layers: [Projection; 2], x: &[f32]) -> Result<Vec<f32>, String> {
        let mut hidden = self.apply(layers[0], x)?;
        silu_inplace(&mut hidden);
        self.apply(layers[1], &hidden)
    }
}

pub struct Denoiser<'m, 'w> {
    weights: &'m Weights<'w>,
    linear: Projections<'m, 'w>,
    pixel_rope: Rope,
    patch_rope: Rope,
    embedding: Vec<f32>,
    policy: ComputePolicy,
}

impl<'m, 'w> Denoiser<'m, 'w> {
    pub fn new(
        weights: &'m Weights<'w>,
        pool: Arc<ComputePool>,
        policy: ComputePolicy,
    ) -> Result<Self, String> {
        let bindings = weights.linears.iter().map(Linear::binding).collect();
        let executor =
            LinearExecutor::new(policy, bindings, VOCAB, pool).map_err(|e| e.to_string())?;
        let mut linear = Projections {
            linears: &weights.linears,
            executor,
            cpu_calls: 0,
            gpu_calls: 0,
        };
        let mut embedding = weights.embedding.clone();
        let features = color_features(&weights.lab);
        let colors = linear.mlp(weights.color_mlp, &features)?;
        vec_add_into(&colors, &mut embedding[..VOCAB * D]);

        Ok(Self {
            weights,
            linear,
            pixel_rope: Rope::new(GRID, 1),
            patch_rope: Rope::new(GRID / 2, 2),
            embedding,
            policy,
        })
    }

    pub fn counts(&self) -> (usize, usize) {
        (self.linear.cpu_calls, self.linear.gpu_calls)
    }

    pub fn forward(
        &mut self,
        ids: &[u16],
        text_states: &[f32],
        pooled: &[f32],
        sc_probs: Option<&[f32]>,
        unconditional: bool,
    ) -> Result<Vec<f32>, String> {
        if ids.len() != PIXELS
            || ids.iter().any(|&x| usize::from(x) > VOCAB)
            || text_states.len() != TEXT_LEN * D
            || pooled.len() != D
            || sc_probs.is_some_and(|s| s.len() != PIXELS * VOCAB)
        {
            return Err("Invalid Tessera denoiser inputs".into());
        }
        let _scope = self.policy.enter_legacy_scope();
        let w = self.weights;
        let mut h = w.positions.clone();
        for (&id, row) in ids.iter().zip(h.chunks_exact_mut(D)) {
            let start = usize::from(id) * D;
            vec_add_into(&self.embedding[start..start + D], row);
        }
        if let Some(probs) = sc_probs {
            let sc = self.linear.apply(w.self_cond, probs)?;
            for ((&id, row), extra) in ids
                .iter()
                .zip(h.chunks_exact_mut(D))
                .zip(sc.chunks_exact(D))
            {
                if usize::from(id) == VOCAB {
                    vec_add_into(extra, row);
                }
            }
        }
        let (t, c) = if unconditional {
            (w.null_text.clone(), w.null_cond.clone())
        } else {
            let mut tokens = text_states.to_vec();
            for row in tokens.chunks_exact_mut(D) {
                rms_norm_inplace(row, &w.text_norm, EPS);
            }
            let mut pool = pooled.to_vec();
            rms_norm_inplace(&mut pool, &w.pool_norm, EPS);
            (
                self.linear.apply(w.text_proj, &tokens)?,
                self.linear.mlp(w.pool_mlp, &pool)?,
            )
        };
        let ratio =
            ids.iter().filter(|&&id| usize::from(id) == VOCAB).count() as f32 / PIXELS as f32;
        let mut features = Vec::with_capacity(16);
        fourier(2. * ratio - 1., 8, &mut features);
        let mut cond = self.linear.mlp(w.ratio_mlp, &features)?;
        vec_add_into(&c, &mut cond);
        silu_inplace(&mut cond);
        let modulations = self.linear.apply(w.modulation, &cond)?;
        let mut xs = [w.registers.as_slice(), &t, &h].concat();
        for blk in &w.stages[0] {
            blk.forward(&mut self.linear, &mut xs, &modulations, &self.pixel_rope)?;
        }
        let skip = xs.split_off(PREFIX * D);
        let mut merged = pack(&skip);
        for row in merged.chunks_exact_mut(4 * D) {
            rms_norm_inplace(row, &w.down_norm, EPS);
        }
        xs.extend(self.linear.apply(w.down, &merged)?);
        for blk in &w.stages[1] {
            blk.forward(&mut self.linear, &mut xs, &modulations, &self.patch_rope)?;
        }
        let core = xs.split_off(PREFIX * D);
        let mut expanded = unpack(&self.linear.apply(w.up, &core)?);
        vec_add_into(&skip, &mut expanded);
        xs.extend(expanded);
        for blk in &w.stages[2] {
            blk.forward(&mut self.linear, &mut xs, &modulations, &self.pixel_rope)?;
        }
        let params = self.linear.apply(w.final_mod, &cond)?;
        let img = &mut xs[PREFIX * D..];
        modulate(img, &params[..D], &params[D..]);
        self.linear.apply(w.head, img)
    }
}

fn modulate(values: &mut [f32], shift: &[f32], scale: &[f32]) {
    for row in values.chunks_exact_mut(D) {
        rms_unit_inplace(row, EPS);
        for i in 0..D {
            row[i] = row[i] * (1. + scale[i]) + shift[i];
        }
    }
}

fn add_gated(residual: &mut [f32], delta: &[f32], gate: &[f32]) {
    for (x, y) in residual.chunks_exact_mut(D).zip(delta.chunks_exact(D)) {
        for i in 0..D {
            x[i] += y[i] * gate[i];
        }
    }
}

impl Block {
    fn forward(
        &self,
        linear: &mut Projections<'_, '_>,
        xs: &mut [f32],
        shared_mod: &[f32],
        rope: &Rope,
    ) -> Result<(), String> {
        let mut params = self.mod_bias.clone();
        vec_add_into(shared_mod, &mut params);
        let mut normed = xs.to_vec();
        modulate(&mut normed, &params[..D], &params[D..2 * D]);
        let qkv = linear.apply(self.qkv, &normed)?;
        let (mut q, mut k, mut v) = (
            Vec::with_capacity(xs.len()),
            Vec::with_capacity(xs.len()),
            Vec::with_capacity(xs.len()),
        );
        for row in qkv.chunks_exact(3 * D) {
            q.extend_from_slice(&row[..D]);
            k.extend_from_slice(&row[D..2 * D]);
            v.extend_from_slice(&row[2 * D..]);
        }
        rope.normalize_and_rotate(&mut q, &self.q_norm);
        rope.normalize_and_rotate(&mut k, &self.k_norm);
        let tokens = xs.len() / D;
        let attn = attention_scalar(
            &q,
            &k,
            &v,
            AttentionSpec {
                query_tokens: tokens,
                key_tokens: tokens,
                query_heads: HEADS,
                key_value_heads: HEADS,
                head_dim: HEAD_DIM,
                causal: false,
                scale: 1. / (HEAD_DIM as f32).sqrt(),
            },
        )?;
        add_gated(xs, &linear.apply(self.proj, &attn)?, &params[2 * D..3 * D]);
        normed.copy_from_slice(xs);
        modulate(&mut normed, &params[3 * D..4 * D], &params[4 * D..5 * D]);
        let w12 = linear.apply(self.w12, &normed)?;
        let mut gate = Vec::with_capacity(tokens * FF);
        let mut up = Vec::with_capacity(tokens * FF);
        for row in w12.chunks_exact(2 * FF) {
            up.extend_from_slice(&row[..FF]);
            gate.extend_from_slice(&row[FF..]);
        }
        self.depthwise(&mut gate[PREFIX * FF..], rope.grid);
        silu_inplace(&mut gate);
        for (v, g) in up.iter_mut().zip(gate) {
            *v *= g;
        }
        add_gated(xs, &linear.apply(self.w3, &up)?, &params[5 * D..]);
        Ok(())
    }

    fn depthwise(&self, gate: &mut [f32], grid: usize) {
        let input = gate.to_vec();
        for (pixel, out) in gate.chunks_exact_mut(FF).enumerate() {
            vec_add_into(&self.bias, out);
            for ky in 0..3 {
                let y = pixel / grid + ky;
                if y == 0 || y > grid {
                    continue;
                }
                for kx in 0..3 {
                    let x = pixel % grid + kx;
                    if x == 0 || x > grid {
                        continue;
                    }
                    let at = ((y - 1) * grid + x - 1) * FF;
                    let k = (ky * 3 + kx) * FF;
                    for ((v, &s), &w) in out
                        .iter_mut()
                        .zip(&input[at..at + FF])
                        .zip(&self.kernel[k..k + FF])
                    {
                        *v += w * s;
                    }
                }
            }
        }
    }
}

// Row-major patches; within each 2x2 patch the offsets are TL, TR, BL, BR.
fn packed_index(pixel: usize) -> usize {
    let (y, x) = (pixel / GRID, pixel % GRID);
    (y / 2 * (GRID / 2) + x / 2) * 4 + y % 2 * 2 + x % 2
}

fn pack(image: &[f32]) -> Vec<f32> {
    let mut out = vec![0.; image.len()];
    for (i, row) in image.chunks_exact(D).enumerate() {
        let at = packed_index(i) * D;
        out[at..at + D].copy_from_slice(row);
    }
    out
}

fn unpack(patches: &[f32]) -> Vec<f32> {
    let mut out = vec![0.; patches.len()];
    for (i, row) in out.chunks_exact_mut(D).enumerate() {
        let at = packed_index(i) * D;
        row.copy_from_slice(&patches[at..at + D]);
    }
    out
}

#[cfg(test)]
mod tests;
