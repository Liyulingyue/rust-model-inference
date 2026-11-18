use super::{add, rng::Noise, trace, Runner};
use crate::ops::{
    nn::{attention_scalar, layer_norm_rows, rms_norm_rows, AttentionSpec},
    silu_inplace,
};

const WIDTH: usize = 448;
const HEADS: usize = 7;
const HEAD_DIM: usize = 64;
const FF: usize = 1216;
const HEAD_WIDTH: usize = 320;

#[derive(Default)]
struct Kv {
    key: Vec<f32>,
    value: Vec<f32>,
}

fn rms(x: &[f32]) -> Result<Vec<f32>, String> {
    rms_norm_rows(x, x.len() / WIDTH, &[1.0; WIDTH], f32::EPSILON)
}

pub(super) fn norm(
    x: &[f32],
    width: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<Vec<f32>, String> {
    let unit;
    let weight = match weight {
        Some(w) => w,
        None => {
            unit = vec![1.0; width];
            &unit
        }
    };
    layer_norm_rows(x, x.len() / width, Some(weight), bias, eps)
}

pub(super) fn rope(values: &mut [f32], start: usize, heads: usize, inv: &[f32]) {
    let dim = inv.len() * 2;
    for (token, row) in values.chunks_exact_mut(heads * dim).enumerate() {
        for head in row.chunks_exact_mut(dim) {
            for i in 0..dim / 2 {
                let (sin, cos) = ((start + token) as f32 * inv[i]).sin_cos();
                let (a, b) = (head[i], head[i + dim / 2]);
                head[i] = a * cos - b * sin;
                head[i + dim / 2] = b * cos + a * sin;
            }
        }
    }
}

pub(super) fn qkv_parts(values: &[f32], width: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut q, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for row in values.chunks_exact(3 * width) {
        q.extend_from_slice(&row[..width]);
        k.extend_from_slice(&row[width..2 * width]);
        v.extend_from_slice(&row[2 * width..]);
    }
    (q, k, v)
}

impl Runner<'_, '_> {
    fn speech_block(
        &mut self,
        layer: usize,
        mut x: Vec<f32>,
        start: usize,
        cache: &mut Kv,
    ) -> Result<Vec<f32>, String> {
        let prefix = format!("speech.layers.{layer}");
        let rows = x.len() / WIDTH;
        let qkv = self.linear(&format!("{prefix}.qkv"), &rms(&x)?)?;
        let (mut q, mut k, v) = qkv_parts(&qkv, WIDTH);
        let inv: Vec<f32> = (0..HEAD_DIM / 2)
            .map(|i| 1.0 / 10000.0f32.powf((2 * i) as f32 / HEAD_DIM as f32))
            .collect();
        rope(&mut q, start, HEADS, &inv);
        rope(&mut k, start, HEADS, &inv);
        cache.key.extend(k);
        cache.value.extend(v);
        let a = attention_scalar(
            &q,
            &cache.key,
            &cache.value,
            AttentionSpec {
                query_tokens: rows,
                key_tokens: cache.key.len() / WIDTH,
                query_heads: HEADS,
                key_value_heads: HEADS,
                head_dim: HEAD_DIM,
                causal: true,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            },
        )?;
        add(&mut x, &self.linear(&format!("{prefix}.o"), &a)?);
        let gu = self.linear(&format!("{prefix}.fc1"), &rms(&x)?)?;
        let mut hidden = Vec::with_capacity(rows * FF);
        for row in gu.chunks_exact(2 * FF) {
            let mut gate = row[..FF].to_vec();
            silu_inplace(&mut gate);
            hidden.extend(gate.iter().zip(&row[FF..]).map(|(g, u)| g * u));
        }
        add(&mut x, &self.linear(&format!("{prefix}.fc2"), &hidden)?);
        Ok(x)
    }

    fn condition(&mut self, x: &[f32], step: usize) -> Result<(Vec<f32>, f32), String> {
        let mut ce = self.linear("speech.ce", &rms(x)?)?;
        silu_inplace(&mut ce[..HEAD_WIDTH]);
        trace(
            &format!("speech.condition.{step}"),
            &[HEAD_WIDTH],
            &ce[..HEAD_WIDTH],
        );
        trace(&format!("speech.eos.{step}"), &[1], &ce[HEAD_WIDTH..]);
        Ok((ce[..HEAD_WIDTH].to_vec(), ce[HEAD_WIDTH]))
    }

    fn flow_head(&mut self, condition: &[f32], noise: &[f32]) -> Result<Vec<f32>, String> {
        let ada = self.linear("speech.ada", condition)?;
        let mut hidden = self.linear("speech.h_inp", noise)?;
        for block in 0..4 {
            let offset = 3 * HEAD_WIDTH * block;
            let mut y = norm(&hidden, HEAD_WIDTH, None, None, 1e-5)?;
            for (i, y) in y.iter_mut().enumerate() {
                *y = ada[offset + i] + *y * ada[offset + HEAD_WIDTH + i];
            }
            let mut y = self.linear(&format!("speech.hblocks.{block}.fc1"), &y)?;
            silu_inplace(&mut y);
            let y = self.linear(&format!("speech.hblocks.{block}.fc2"), &y)?;
            for (i, h) in hidden.iter_mut().enumerate() {
                *h += ada[offset + 2 * HEAD_WIDTH + i] * y[i];
            }
        }
        let offset = 12 * HEAD_WIDTH;
        let mut y = norm(&hidden, HEAD_WIDTH, None, None, 1e-5)?;
        for (i, y) in y.iter_mut().enumerate() {
            *y = ada[offset + i] + *y * ada[offset + HEAD_WIDTH + i];
        }
        self.linear("speech.h_out", &y)
    }

    pub(super) fn generate(
        &mut self,
        voice: &str,
        ids: &[u32],
        noise: &mut Noise,
        temperature: f32,
        force: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        if ids.is_empty() || ids.iter().any(|id| *id >= 95) {
            return Err("Invalid Underdog Woof character IDs".into());
        }
        let mut x = self.weights.values(&format!("voices.{voice}")).to_vec();
        let embedding = self.weights.values("speech.text_emb.weight");
        for &id in ids {
            x.extend_from_slice(&embedding[id as usize * WIDTH..(id as usize + 1) * WIDTH]);
        }
        x.extend_from_slice(self.weights.values("speech.bos"));
        let prefix_len = x.len() / WIDTH;
        if prefix_len >= 1100 {
            return Err("Underdog Woof prefix exceeds its context".into());
        }
        let max_frames = force.map_or(750, |v| v.len() / 32).min(1100 - prefix_len);
        if force
            .is_some_and(|v| v.is_empty() || v.len() % 32 != 0 || v.iter().any(|x| !x.is_finite()))
        {
            return Err("Invalid Underdog Woof forced latent fixture".into());
        }
        trace("speech.prefix_input", &[prefix_len, WIDTH], &x);
        let mut caches: Vec<Kv> = (0..8).map(|_| Kv::default()).collect();
        for (layer, cache) in caches.iter_mut().enumerate() {
            x = self.speech_block(layer, x, 0, cache)?;
            trace(&format!("speech.prefix.{layer}"), &[prefix_len, WIDTH], &x);
        }
        x = x[x.len() - WIDTH..].to_vec();
        let (mut condition, mut eos) = self.condition(&x, 0)?;
        let mut output = Vec::with_capacity(max_frames * 32);
        for frame in 0..max_frames {
            if force.is_none() && frame > 3 && f64::from(eos) > (0.9f64 / 0.1).ln() {
                break;
            }
            let z = noise.frame().map(|v| v * temperature);
            trace(&format!("speech.noise.{frame}"), &[32], &z);
            let delta = self.flow_head(&condition, &z)?;
            trace(&format!("speech.head.{frame}"), &[32], &delta);
            let latent: Vec<f32> = z.iter().zip(delta).map(|(z, d)| z + d).collect();
            output.extend_from_slice(&latent);
            let feedback = force.map_or(latent.as_slice(), |v| &v[frame * 32..(frame + 1) * 32]);
            x = self.linear("speech.lat_in", feedback)?;
            for (layer, cache) in caches.iter_mut().enumerate() {
                x = self.speech_block(layer, x, prefix_len + frame, cache)?;
                trace(&format!("speech.block.{frame}.{layer}"), &[WIDTH], &x);
            }
            (condition, eos) = self.condition(&x, frame + 1)?;
        }
        trace("latents", &[output.len() / 32, 32], &output);
        Ok(output)
    }
}
