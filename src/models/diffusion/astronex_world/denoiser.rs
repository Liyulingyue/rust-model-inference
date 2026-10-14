//! Astronex-World base Wan denoiser.

use super::super::dreamx::kernels::{
    attention_scalar_f32, checked_len, conv3d_with_options, gelu_approx_inplace, load_float_values,
    AttentionSpec, Conv3dSpec, Linear,
};
use super::{
    DIM, FFN, FREQ_DIM, HEADS, HEAD_DIM, LATENT_CHANNELS, LAYERS, PATCH, TEXT_DIM, TEXT_LEN,
};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use std::sync::Arc;

struct Block {
    norm3_weight: Vec<f32>,
    norm3_bias: Vec<f32>,
    modulation: Vec<f32>,
    self_q: Linear<'static>,
    self_k: Linear<'static>,
    self_v: Linear<'static>,
    self_o: Linear<'static>,
    self_norm_q: Vec<f32>,
    self_norm_k: Vec<f32>,
    cross_q: Linear<'static>,
    cross_k: Linear<'static>,
    cross_v: Linear<'static>,
    cross_o: Linear<'static>,
    cross_norm_q: Vec<f32>,
    cross_norm_k: Vec<f32>,
    ffn_up: Linear<'static>,
    ffn_down: Linear<'static>,
}

pub(super) struct Denoiser {
    _source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    patch_weight: Vec<f32>,
    patch_bias: Vec<f32>,
    text_0: Linear<'static>,
    text_2: Linear<'static>,
    time_0: Linear<'static>,
    time_2: Linear<'static>,
    time_projection: Linear<'static>,
    blocks: Vec<Block>,
    head: Linear<'static>,
    head_modulation: Vec<f32>,
}

fn original_dims(dims: &[usize]) -> Vec<u64> {
    dims.iter().rev().map(|&v| v as u64).collect()
}

fn values(source: &dyn TensorSource, name: &str, dims: &[usize]) -> Result<Vec<f32>, String> {
    load_float_values(source, name, &original_dims(dims))
}

fn linear(
    source: &'static dyn TensorSource,
    prefix: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Linear<'static>, String> {
    Linear::from_source(
        source,
        &format!("{prefix}.weight"),
        Some(&format!("{prefix}.bias")),
        n_in,
        n_out,
    )
}

impl Denoiser {
    pub(super) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let source_ref: &'static dyn TensorSource = unsafe { &*Arc::as_ptr(&source) };
        let patch_weight = values(
            source_ref,
            "astronex.patch_embedding.weight",
            &[DIM, LATENT_CHANNELS, 1, 2, 2],
        )?;
        let patch_bias = values(source_ref, "astronex.patch_embedding.bias", &[DIM])?;
        let text_0 = linear(source_ref, "astronex.text_embedding.0", TEXT_DIM, DIM)?;
        let text_2 = linear(source_ref, "astronex.text_embedding.2", DIM, DIM)?;
        let time_0 = linear(source_ref, "astronex.time_embedding.0", FREQ_DIM, DIM)?;
        let time_2 = linear(source_ref, "astronex.time_embedding.2", DIM, DIM)?;
        let time_projection = linear(source_ref, "astronex.time_projection.1", DIM, DIM * 6)?;
        let head = linear(source_ref, "astronex.head.head", DIM, LATENT_CHANNELS * 4)?;
        let head_modulation = values(source_ref, "astronex.head.modulation", &[1, 2, DIM])?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            let prefix = format!("astronex.blocks.{index}");
            blocks.push(Block {
                norm3_weight: values(source_ref, &format!("{prefix}.norm3.weight"), &[DIM])?,
                norm3_bias: values(source_ref, &format!("{prefix}.norm3.bias"), &[DIM])?,
                modulation: values(source_ref, &format!("{prefix}.modulation"), &[1, 6, DIM])?,
                self_q: linear(source_ref, &format!("{prefix}.self_attn.q"), DIM, DIM)?,
                self_k: linear(source_ref, &format!("{prefix}.self_attn.k"), DIM, DIM)?,
                self_v: linear(source_ref, &format!("{prefix}.self_attn.v"), DIM, DIM)?,
                self_o: linear(source_ref, &format!("{prefix}.self_attn.o"), DIM, DIM)?,
                self_norm_q: values(
                    source_ref,
                    &format!("{prefix}.self_attn.norm_q.weight"),
                    &[DIM],
                )?,
                self_norm_k: values(
                    source_ref,
                    &format!("{prefix}.self_attn.norm_k.weight"),
                    &[DIM],
                )?,
                cross_q: linear(source_ref, &format!("{prefix}.cross_attn.q"), DIM, DIM)?,
                cross_k: linear(source_ref, &format!("{prefix}.cross_attn.k"), DIM, DIM)?,
                cross_v: linear(source_ref, &format!("{prefix}.cross_attn.v"), DIM, DIM)?,
                cross_o: linear(source_ref, &format!("{prefix}.cross_attn.o"), DIM, DIM)?,
                cross_norm_q: values(
                    source_ref,
                    &format!("{prefix}.cross_attn.norm_q.weight"),
                    &[DIM],
                )?,
                cross_norm_k: values(
                    source_ref,
                    &format!("{prefix}.cross_attn.norm_k.weight"),
                    &[DIM],
                )?,
                ffn_up: linear(source_ref, &format!("{prefix}.ffn.0"), DIM, FFN)?,
                ffn_down: linear(source_ref, &format!("{prefix}.ffn.2"), FFN, DIM)?,
            });
        }
        Ok(Self {
            _source: source,
            pool,
            patch_weight,
            patch_bias,
            text_0,
            text_2,
            time_0,
            time_2,
            time_projection,
            blocks,
            head,
            head_modulation,
        })
    }

    fn patch(&self, latent: &[f32], shape: [usize; 4]) -> Result<(Vec<f32>, [usize; 3]), String> {
        let (data, out) = conv3d_with_options(
            &self.pool,
            latent,
            shape,
            &self.patch_weight,
            [DIM, LATENT_CHANNELS, 1, 2, 2],
            Some(&self.patch_bias),
            Conv3dSpec {
                stride: PATCH,
                ..Conv3dSpec::default()
            },
        )?;
        let [_, frames, height, width] = out;
        let tokens = frames * height * width;
        let mut hidden = vec![0.0; tokens * DIM];
        for f in 0..frames {
            for h in 0..height {
                for w in 0..width {
                    let row = (f * height + h) * width + w;
                    for c in 0..DIM {
                        hidden[row * DIM + c] =
                            data[(c * frames + f) * height * width + h * width + w];
                    }
                }
            }
        }
        Ok((hidden, [frames, height, width]))
    }

    fn embed_text(&self, pool: &ComputePool, context: &[f32]) -> Result<Vec<f32>, String> {
        if context.len() != TEXT_LEN * TEXT_DIM {
            return Err(format!(
                "Astronex context must have {} values",
                TEXT_LEN * TEXT_DIM
            ));
        }
        let mut hidden = self.text_0.forward(pool, context, TEXT_LEN)?;
        gelu_approx_inplace(&mut hidden);
        self.text_2.forward(pool, &hidden, TEXT_LEN)
    }

    fn embed_time(
        &self,
        pool: &ComputePool,
        timestep: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut sinusoid = vec![0.0; FREQ_DIM];
        let half = FREQ_DIM / 2;
        for i in 0..half {
            let frequency = 10000.0f64.powf(-(i as f64) / half as f64);
            let value = timestep as f64 * frequency;
            sinusoid[i] = value.cos() as f32;
            sinusoid[half + i] = value.sin() as f32;
        }
        let mut time = self.time_0.forward(pool, &sinusoid, 1)?;
        for value in &mut time {
            *value = *value / (1.0 + (-*value).exp());
        }
        let mut time = self.time_2.forward(pool, &time, 1)?;
        for value in &mut time {
            *value = *value / (1.0 + (-*value).exp());
        }
        let projected = self.time_projection.forward(pool, &time, 1)?;
        Ok((time, projected))
    }

    pub(super) fn forward(
        &self,
        latent: &[f32],
        shape: [usize; 4],
        timestep: f32,
        context: &[f32],
        context_tokens: usize,
    ) -> Result<Vec<f32>, String> {
        let (mut x, grid) = self.patch(latent, shape)?;
        let tokens = x.len() / DIM;
        let context = self.embed_text(&self.pool, context)?;
        if context_tokens == 0 || context_tokens > TEXT_LEN {
            return Err("Astronex text context length is out of range".into());
        }
        // The official Wan T2V graph pads UMT5 to all 512 positions and calls
        // cross-attention without a key-length mask. Padding embeddings are
        // zeroed by the text encoder, but those zero keys still participate in
        // the softmax denominator. Keep the full context here for bit-level
        // graph parity; `context_tokens` remains a contract check.
        let context = &context[..TEXT_LEN * DIM];
        let (time, time_mod) = self.embed_time(&self.pool, timestep)?;
        let (frames, height, width) = (grid[0], grid[1], grid[2]);
        for block in &self.blocks {
            x = block.forward(&self.pool, x, tokens, grid, &time_mod, context, TEXT_LEN)?;
        }
        let mut modulated = x.clone();
        // Head::forward consumes the unprojected time embedding `e`, while
        // blocks consume the six-way `time_projection(e)` modulation.
        let hmod = &self.head_modulation[..DIM];
        let smod = &self.head_modulation[DIM..2 * DIM];
        let normed = norm_no_affine(&x, tokens, 1e-6)?;
        for row in 0..tokens {
            for c in 0..DIM {
                modulated[row * DIM + c] =
                    normed[row * DIM + c] * (1.0 + smod[c]) + hmod[c] + time[c];
            }
        }
        let output = self.head.forward(&self.pool, &modulated, tokens)?;
        let mut video = vec![0.0; checked_len("Astronex denoiser output", &shape)?];
        for f in 0..frames {
            for h in 0..height {
                for w in 0..width {
                    let row = (f * height + h) * width + w;
                    for pf in 0..PATCH[0] {
                        for ph in 0..PATCH[1] {
                            for pw in 0..PATCH[2] {
                                let patch_index = (pf * PATCH[1] + ph) * PATCH[2] + pw;
                                let ff = f * PATCH[0] + pf;
                                let hh = h * PATCH[1] + ph;
                                let ww = w * PATCH[2] + pw;
                                for channel in 0..LATENT_CHANNELS {
                                    video[(channel * shape[1] + ff) * shape[2] * shape[3]
                                        + hh * shape[3]
                                        + ww] = output[row * (LATENT_CHANNELS * 4)
                                        + patch_index * LATENT_CHANNELS
                                        + channel];
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(video)
    }
}

impl Block {
    fn forward(
        &self,
        pool: &ComputePool,
        mut x: Vec<f32>,
        tokens: usize,
        grid: [usize; 3],
        time: &[f32],
        context: &[f32],
        context_tokens: usize,
    ) -> Result<Vec<f32>, String> {
        let mut norm = norm_no_affine(&x, tokens, 1e-6)?;
        let mut e = self.modulation.clone();
        for i in 0..6 * DIM {
            e[i] += time[i];
        }
        let mut q_in = norm.clone();
        for row in 0..tokens {
            for c in 0..DIM {
                q_in[row * DIM + c] = norm[row * DIM + c] * (1.0 + e[DIM + c]) + e[c];
            }
        }
        let mut q = self.self_q.forward(pool, &q_in, tokens)?;
        let mut k = self.self_k.forward(pool, &q_in, tokens)?;
        let v = self.self_v.forward(pool, &q_in, tokens)?;
        rms_norm_in_rows(&mut q, tokens, &self.self_norm_q, 1e-6);
        rms_norm_in_rows(&mut k, tokens, &self.self_norm_k, 1e-6);
        apply_rope(&mut q, &mut k, grid);
        // The bidirectional Wan graph uses the complete clip as its key/value
        // set (`window_size=(-1, -1)`).  A local temporal window belongs to the
        // separate causal pipeline and silently changes the released graph.
        let attended = attention_scalar_f32(
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
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            },
        )?;
        let attn = self.self_o.forward(pool, &attended, tokens)?;
        for row in 0..tokens {
            for c in 0..DIM {
                x[row * DIM + c] += attn[row * DIM + c] * e[2 * DIM + c];
            }
        }

        norm = layer_norm_affine(&x, tokens, &self.norm3_weight, &self.norm3_bias, 1e-6)?;
        let q = self.cross_q.forward(pool, &norm, tokens)?;
        let mut k = self.cross_k.forward(pool, context, context_tokens)?;
        let v = self.cross_v.forward(pool, context, context_tokens)?;
        let mut q = q;
        rms_norm_in_rows(&mut q, tokens, &self.cross_norm_q, 1e-6);
        rms_norm_in_rows(&mut k, context_tokens, &self.cross_norm_k, 1e-6);
        let cross = attention_scalar_f32(
            &q,
            &k,
            &v,
            AttentionSpec {
                query_tokens: tokens,
                key_tokens: context_tokens,
                query_heads: HEADS,
                key_value_heads: HEADS,
                head_dim: HEAD_DIM,
                causal: false,
                scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            },
        )?;
        let cross = self.cross_o.forward(pool, &cross, tokens)?;
        for row in 0..tokens {
            for c in 0..DIM {
                x[row * DIM + c] += cross[row * DIM + c];
            }
        }

        norm = norm_no_affine(&x, tokens, 1e-6)?;
        for row in 0..tokens {
            for c in 0..DIM {
                norm[row * DIM + c] = norm[row * DIM + c] * (1.0 + e[4 * DIM + c]) + e[3 * DIM + c];
            }
        }
        let mut ff = self.ffn_up.forward(pool, &norm, tokens)?;
        gelu_approx_inplace(&mut ff);
        let ff = self.ffn_down.forward(pool, &ff, tokens)?;
        for row in 0..tokens {
            for c in 0..DIM {
                x[row * DIM + c] += ff[row * DIM + c] * e[5 * DIM + c];
            }
        }
        Ok(x)
    }
}

fn norm_no_affine(input: &[f32], rows: usize, eps: f32) -> Result<Vec<f32>, String> {
    if rows == 0 || input.len() != checked_len("Astronex LayerNorm input", &[rows, DIM])? {
        return Err("Invalid Astronex LayerNorm tensors".into());
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let slice = &input[row * DIM..(row + 1) * DIM];
        let mean = slice.iter().copied().sum::<f32>() / DIM as f32;
        let variance = slice
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / DIM as f32;
        let inv = 1.0 / (variance + eps).sqrt();
        for c in 0..DIM {
            output[row * DIM + c] = (slice[c] - mean) * inv;
        }
    }
    Ok(output)
}

fn layer_norm_affine(
    input: &[f32],
    rows: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<Vec<f32>, String> {
    let mut output = norm_no_affine(input, rows, eps)?;
    for row in 0..rows {
        for c in 0..DIM {
            output[row * DIM + c] = output[row * DIM + c] * weight[c] + bias[c];
        }
    }
    Ok(output)
}

fn rms_scalar(values: &[f32], eps: f32) -> f32 {
    (values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32 + eps).sqrt()
}

fn rms_norm_in_rows(values: &mut [f32], rows: usize, weight: &[f32], eps: f32) {
    for row in 0..rows {
        let start = row * DIM;
        let scale = 1.0 / rms_scalar(&values[start..start + DIM], eps);
        for c in 0..DIM {
            values[start + c] *= scale * weight[c];
        }
    }
}

fn apply_rope(q: &mut [f32], k: &mut [f32], grid: [usize; 3]) {
    let (frames, height, width) = (grid[0], grid[1], grid[2]);
    let splits = [22usize, 21, 21];
    for f in 0..frames {
        for h in 0..height {
            for w in 0..width {
                let token = (f * height + h) * width + w;
                let positions = [f, h, w];
                let mut offset = 0;
                for axis in 0..3 {
                    for i in 0..splits[axis] {
                        let dim = offset + i;
                        // Wan computes the complex RoPE multiplier in F64 and
                        // casts back to the activation dtype. Keep this
                        // separate from the intentionally approximate F32
                        // generation kernels; it is the only RoPE path used
                        // by the scalar parity trace.
                        let inv = 10000.0f64.powf(-(i as f64) / splits[axis] as f64);
                        let angle = positions[axis] as f64 * inv;
                        let (sin, cos) = angle.sin_cos();
                        for head in 0..HEADS {
                            let base = token * DIM + head * HEAD_DIM + 2 * dim;
                            for data in [&mut *q, &mut *k] {
                                let a = data[base] as f64;
                                let b = data[base + 1] as f64;
                                data[base] = (a * cos - b * sin) as f32;
                                data[base + 1] = (a * sin + b * cos) as f32;
                            }
                        }
                    }
                    offset += splits[axis];
                }
            }
        }
    }
}
