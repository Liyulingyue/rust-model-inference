use super::{trace, Matrix};
use crate::compute::linear::LinearExecutor;
use crate::core::tensor::TensorSource;
use rayon::prelude::*;

#[cfg(feature = "vulkan")]
pub(super) mod device;

pub(super) struct Layer<'a> {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    qkv: Matrix<'a>,
    output: Matrix<'a>,
    gate: Matrix<'a>,
    up: Matrix<'a>,
    down: Matrix<'a>,
    attn_scale: Vec<f32>,
    ffn_scale: Vec<f32>,
}

#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub width: usize,
    pub heads: usize,
    pub kv: usize,
    pub intermediate: usize,
    pub epsilon: f32,
    pub rope_base: f32,
    pub window: Option<usize>,
}

pub(super) struct Transformer<'a> {
    pub shape: Shape,
    layers: Vec<Layer<'a>>,
    norm: Vec<f32>,
    name: String,
}

#[derive(Default)]
pub(super) struct Cache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    pub position: usize,
}

impl<'a> Transformer<'a> {
    pub fn load(
        source: &'a dyn TensorSource,
        prefix: &str,
        norm_prefix: &str,
        count: usize,
        shape: Shape,
        qkv_bias: bool,
        layer_scale: bool,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(count);
        for index in 0..count {
            let p = format!("{prefix}.{index}");
            let norm = |suffix: &str| super::vector(source, &format!("{p}.{suffix}"), shape.width);
            layers.push(Layer {
                norm1: norm("attention_norm.weight")?,
                norm2: norm("ffn_norm.weight")?,
                qkv: Matrix::load(
                    source,
                    &format!("{p}.attention.wqkv"),
                    shape.width,
                    (shape.heads + 2 * shape.kv) * 64,
                    qkv_bias,
                )?,
                output: Matrix::load(
                    source,
                    &format!("{p}.attention.wo"),
                    shape.heads * 64,
                    shape.width,
                    false,
                )?,
                gate: Matrix::load(
                    source,
                    &format!("{p}.feed_forward.w1"),
                    shape.width,
                    shape.intermediate,
                    false,
                )?,
                up: Matrix::load(
                    source,
                    &format!("{p}.feed_forward.w3"),
                    shape.width,
                    shape.intermediate,
                    false,
                )?,
                down: Matrix::load(
                    source,
                    &format!("{p}.feed_forward.w2"),
                    shape.intermediate,
                    shape.width,
                    false,
                )?,
                attn_scale: if layer_scale {
                    norm("attention_layer_scale.gamma")?
                } else {
                    vec![1.0; shape.width]
                },
                ffn_scale: if layer_scale {
                    norm("ffn_layer_scale.gamma")?
                } else {
                    vec![1.0; shape.width]
                },
            });
        }
        Ok(Self {
            shape,
            layers,
            norm: super::vector(source, &format!("{norm_prefix}.weight"), shape.width)?,
            name: prefix.into(),
        })
    }

    pub fn normalize(&self, hidden: &[f32]) -> Vec<f32> {
        normalize(hidden, &self.norm, self.shape.epsilon)
    }

    pub fn matrices(&self) -> impl Iterator<Item = &Matrix<'a>> {
        self.layers
            .iter()
            .flat_map(|l| [&l.qkv, &l.output, &l.gate, &l.up, &l.down])
    }

    pub fn forward(
        &self,
        mut hidden: Vec<f32>,
        cache: &mut Cache,
        linear: &mut LinearExecutor<'_, '_>,
    ) -> Result<Vec<f32>, String> {
        let s = self.shape;
        if hidden.is_empty() || hidden.len() % s.width != 0 {
            return Err("invalid ArkTTS transformer input".into());
        }
        let frames = hidden.len() / s.width;
        let start = cache.position;
        let end = start
            .checked_add(frames)
            .ok_or("ArkTTS position overflow")?;
        if cache.keys.is_empty() {
            cache.keys = vec![Vec::new(); self.layers.len()];
            cache.values = vec![Vec::new(); self.layers.len()];
        }
        trace(
            &format!("{}.input", self.name),
            None,
            Some(start),
            &[frames, s.width],
            &hidden,
        )?;
        for (index, layer) in self.layers.iter().enumerate() {
            let normalized = normalize(&hidden, &layer.norm1, s.epsilon);
            let qkv = layer.qkv.forward(&normalized, linear)?;
            let q_width = s.heads * 64;
            let kv_width = s.kv * 64;
            let mut query = Vec::with_capacity(frames * q_width);
            let mut key = Vec::with_capacity(frames * kv_width);
            let mut value = Vec::with_capacity(frames * kv_width);
            for (time, row) in qkv.chunks_exact(q_width + 2 * kv_width).enumerate() {
                let mut q = row[..q_width].to_vec();
                let mut k = row[q_width..q_width + kv_width].to_vec();
                rope(&mut q, start + time, s.rope_base);
                rope(&mut k, start + time, s.rope_base);
                query.extend(q);
                key.extend(k);
                value.extend_from_slice(&row[q_width + kv_width..]);
            }
            cache.keys[index].extend(key);
            cache.values[index].extend(value);
            let keys = &cache.keys[index];
            let values = &cache.values[index];
            let mut attention = vec![0.0; frames * q_width];
            attention
                .par_chunks_mut(64)
                .enumerate()
                .for_each(|(row, output)| {
                    let time = start + row / s.heads;
                    let head = row % s.heads;
                    let kv_head = head / (s.heads / s.kv);
                    let first = s
                        .window
                        .map_or(0, |window| (time + 1).saturating_sub(window));
                    let q = &query[row * 64..(row + 1) * 64];
                    let mut scores: Vec<f32> = (first..=time)
                        .map(|position| {
                            let offset = position * kv_width + kv_head * 64;
                            crate::ops::dot_f32(q, &keys[offset..offset + 64], 64) * 0.125
                        })
                        .collect();
                    crate::ops::softmax_inplace(&mut scores);
                    for (position, probability) in (first..=time).zip(scores) {
                        let offset = position * kv_width + kv_head * 64;
                        for (out, &v) in output.iter_mut().zip(&values[offset..offset + 64]) {
                            *out += probability * v;
                        }
                    }
                });
            let update = layer.output.forward(&attention, linear)?;
            for (i, (out, update)) in hidden.iter_mut().zip(update).enumerate() {
                *out += update * layer.attn_scale[i % s.width];
            }
            let normalized = normalize(&hidden, &layer.norm2, s.epsilon);
            let mut gate = layer.gate.forward(&normalized, linear)?;
            let up = layer.up.forward(&normalized, linear)?;
            for (gate, up) in gate.iter_mut().zip(up) {
                *gate = crate::ops::silu(*gate) * up;
            }
            let update = layer.down.forward(&gate, linear)?;
            for (i, (out, update)) in hidden.iter_mut().zip(update).enumerate() {
                *out += update * layer.ffn_scale[i % s.width];
            }
            trace(
                &self.name,
                Some(index),
                Some(start),
                &[frames, s.width],
                &hidden,
            )?;
        }
        cache.position = end;
        Ok(hidden)
    }
}

fn normalize(hidden: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    let mut output = vec![0.0; hidden.len()];
    output
        .par_chunks_mut(weight.len())
        .zip(hidden.par_chunks(weight.len()))
        .for_each(|(out, row)| {
            crate::ops::rms_norm(row, weight, out, epsilon);
        });
    output
}

fn rope(values: &mut [f32], position: usize, base: f32) {
    for head in values.chunks_exact_mut(64) {
        for (index, pair) in head.chunks_exact_mut(2).enumerate() {
            let phase = position as f32 * (1.0 / base.powf((2 * index) as f32 / 64.0));
            // The official FP32 CPU model retains its BF16-precomputed RoPE table.
            let c = half::bf16::from_f32(phase.cos()).to_f32();
            let s = half::bf16::from_f32(phase.sin()).to_f32();
            let (x, y) = (pair[0], pair[1]);
            pair[0] = x * c - y * s;
            pair[1] = y * c + x * s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotary_pairs_use_bf16_coefficients_and_interleaved_coordinates() {
        let mut values = vec![0.0; 64];
        values[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        rope(&mut values, 1, 1_000_000.0);
        assert_eq!(&values[..2], &[-1.140625, 1.91796875]);
        assert_eq!(values[2], -0.03125);
    }
}
