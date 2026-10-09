//! Qwen-Image-2.1 DiT forward pass, numerically aligned with the pinned
//! stable-diffusion.cpp oracle's ggml CPU kernels (see tools/oracle/qwen_image_2_1).

use super::{Condition, QwenImage21Config, ReferenceLatent, PREFIX};
use crate::core::tensor::TensorSource;
use rayon::prelude::*;
use std::sync::Arc;

const RMS_EPS: f32 = 1e-6;
const LAYER_NORM_EPS: f32 = 1e-6;
const ATTENTION_OUT_SCALE: f32 = 1.0 / 32.0;

fn rms_norm_mul_inplace(x: &mut [f32], w: &[f32]) {
    crate::ops::rms_norm_inplace(x, w, RMS_EPS);
}

/// ggml LayerNorm without affine, using the shared scalar/SIMD reductions.
fn layer_norm_row(x: &[f32], y: &mut [f32]) {
    let mean = crate::ops::sum_f32(x) as f32 / x.len() as f32;
    for (value, output) in x.iter().zip(y.iter_mut()) {
        *output = *value - mean;
    }
    let variance = (crate::ops::sum_sq_f32(y) / x.len() as f64) as f32;
    let scale = 1.0f32 / (variance + LAYER_NORM_EPS).sqrt();
    for value in y.iter_mut() {
        *value *= scale;
    }
}

/// ggml `ggml_compute_forward_timestep_embedding_f32` with dim 256.
fn timestep_embedding_row(timestep: f32, output: &mut [f32]) {
    let half = output.len() / 2;
    let neg_log_period = -(10_000.0f32).ln();
    for j in 0..half {
        let freq = (neg_log_period * j as f32 / half as f32).exp();
        let arg = timestep * freq;
        let (sin_value, cos_value) = crate::ops::rope::ggml_sin_cos(arg);
        output[j] = cos_value;
        output[j + half] = sin_value;
    }
}

/// `Rope::rope_frequencies(dim, theta)` including its linspace rounding.
fn rope_frequencies(dim: usize, theta: f32) -> Vec<f32> {
    let half_dim = dim / 2;
    let end = (dim as f32 - 2.0) / dim as f32;
    let step = end / (half_dim as f32 - 1.0);
    (0..half_dim)
        .map(|i| 1.0f32 / theta.powf(i as f32 * step))
        .collect()
}

/// One axis block of `Rope::embed_nd`: [cos, -sin, sin, cos] per frequency.
fn rope_axis_block(position: f32, omega: &[f32]) -> Vec<f32> {
    let mut block = Vec::with_capacity(omega.len() * 4);
    for frequency in omega {
        let angle = position * frequency;
        let (sin_value, cos_value) = angle.sin_cos();
        block.push(cos_value);
        block.push(-sin_value);
        block.push(sin_value);
        block.push(cos_value);
    }
    block
}

/// `Rope::apply_rope` interleaved, standard rotation per frequency pair:
/// out = [a*cos - b*sin, a*sin + b*cos], evaluated as elementwise mul nodes
/// and one add, with the pe quad [cos, -sin, sin, cos].
fn apply_rope_row(values: &mut [f32], pe: &[f32]) {
    for pair_index in 0..values.len() / 2 {
        let a = values[2 * pair_index];
        let b = values[2 * pair_index + 1];
        let cos = pe[4 * pair_index];
        let neg_sin = pe[4 * pair_index + 1];
        let sin = pe[4 * pair_index + 2];
        let cos_tail = pe[4 * pair_index + 3];
        let m0 = a * cos;
        let m1 = b * neg_sin;
        let m2 = a * sin;
        let m3 = b * cos_tail;
        values[2 * pair_index] = m0 + m1;
        values[2 * pair_index + 1] = m2 + m3;
    }
}

pub(crate) struct QwenImage21Dit {
    pub(crate) config: QwenImage21Config,
    source: Arc<dyn TensorSource>,
    q8_values: Vec<u8>,
    q8_scales: Vec<f32>,
    thread_pool: rayon::ThreadPool,
}

impl QwenImage21Dit {
    pub(crate) fn load(source: Arc<dyn TensorSource>, n_threads: usize) -> Result<Self, String> {
        let config = QwenImage21Config::detect_from_source(source.as_ref())?;
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads.max(1))
            .build()
            .map_err(|error| format!("Create Qwen-Image-2.1 thread pool: {error}"))?;
        Ok(Self {
            config,
            source,
            q8_values: Vec::new(),
            q8_scales: Vec::new(),
            thread_pool,
        })
    }

    fn weight_bytes(&self, name: &str) -> Result<&[u8], String> {
        self.source
            .tensor_slice(name)
            .ok_or_else(|| format!("Missing tensor data: {name}"))
    }

    fn q8_linear(
        &mut self,
        name: &str,
        n_in: usize,
        n_out: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        if n_in == 0 || n_in % 32 != 0 || n_out == 0 {
            return Err(format!("Invalid {name} input dimensions"));
        }
        let tokens = input.len() / n_in;
        if tokens.checked_mul(n_in) != Some(input.len()) {
            return Err(format!("Invalid {name} input length"));
        }
        if tokens.checked_mul(n_out) != Some(output.len()) {
            return Err(format!("Invalid {name} output length"));
        }
        let weight_bytes = n_out
            .checked_mul(n_in)
            .and_then(|elements| elements.checked_mul(34))
            .and_then(|bytes| bytes.checked_div(32))
            .ok_or_else(|| format!("Invalid {name} dimensions"))?;
        let source = Arc::clone(&self.source);
        let weight = source
            .tensor_slice(name)
            .ok_or_else(|| format!("Missing tensor data: {name}"))?;
        if weight.len() != weight_bytes {
            return Err(format!("Invalid {name} weight length"));
        }
        let blocks = n_in / 32;
        let q8_len = tokens
            .checked_mul(n_in)
            .ok_or_else(|| format!("Invalid {name} dimensions"))?;
        let scales_len = tokens
            .checked_mul(blocks)
            .ok_or_else(|| format!("Invalid {name} dimensions"))?;
        self.q8_values.resize(q8_len, 0);
        self.q8_scales.resize(scales_len, 0.0);
        for token in 0..tokens {
            let row = &input[token * n_in..(token + 1) * n_in];
            let q_start = token * n_in;
            let scale_start = token * blocks;
            crate::ops::quantize_q8_0_into(
                row,
                n_in,
                &mut self.q8_values[q_start..q_start + n_in],
                &mut self.q8_scales[scale_start..scale_start + blocks],
            );
        }
        let q8_values = &self.q8_values;
        let q8_scales = &self.q8_scales;
        self.thread_pool.install(|| {
            output
                .par_chunks_mut(n_out)
                .zip(q8_values.par_chunks_exact(n_in))
                .zip(q8_scales.par_chunks_exact(blocks))
                .for_each(|((out, q8), scales)| {
                    crate::ops::kernel::q8_0::dispatch::matmul_q8_0_quantized_range(
                        weight, q8, scales, out, n_in, 0, n_out,
                    );
                });
        });
        Ok(())
    }

    /// Linear with BF16 weights: activations are rounded to BF16 once, then
    /// every output element is a scalar f64-accumulated dot (the aarch64
    /// ggml_vec_dot_bf16 fallback path).
    fn bf16_linear(
        &self,
        name: &str,
        n_in: usize,
        n_out: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        let weight = self.weight_bytes(name)?;
        let tokens = input.len() / n_in;
        if tokens * n_in != input.len() {
            return Err(format!("Invalid {name} input length"));
        }
        if tokens * n_out != output.len() {
            return Err(format!("Invalid {name} output length"));
        }
        let weight: Vec<f32> = weight
            .chunks_exact(2)
            .map(|pair| crate::ops::bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
            .collect();
        let act: Vec<_> = input
            .iter()
            .map(|&v| crate::ops::bf16_to_f32(crate::ops::f32_to_bf16(v)))
            .collect();
        self.thread_pool.install(|| {
            output.par_iter_mut().enumerate().for_each(|(i, value)| {
                *value = crate::ops::dot_f32(
                    &weight[i % n_out * n_in..],
                    &act[i / n_out * n_in..],
                    n_in,
                );
            })
        });
        Ok(())
    }

    /// Linear with F32 weights, one ggml-order dot per output element.
    fn f32_linear(
        &self,
        name: &str,
        n_in: usize,
        n_out: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        let weight = self.weight_bytes(name)?;
        let tokens = input.len() / n_in;
        if tokens * n_in != input.len() {
            return Err(format!("Invalid {name} input length"));
        }
        if tokens * n_out != output.len() {
            return Err(format!("Invalid {name} output length"));
        }
        let weight: Vec<f32> = weight
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        self.thread_pool.install(|| {
            output.par_iter_mut().enumerate().for_each(|(i, value)| {
                *value = crate::ops::dot_f32(
                    &weight[i % n_out * n_in..],
                    &input[i / n_out * n_in..],
                    n_in,
                );
            })
        });
        Ok(())
    }

    fn f32_vector(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let bytes = self.weight_bytes(name)?;
        if bytes.len() != len * 4 {
            return Err(format!("Invalid {name} length"));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn bf16_vector_widened(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let bytes = self.weight_bytes(name)?;
        if bytes.len() != len * 2 {
            return Err(format!("Invalid {name} length"));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|pair| crate::ops::bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
            .collect())
    }

    /// Single forward: velocity prediction for one latent, traced at the same
    /// checkpoints as the oracle harness.
    pub(crate) fn forward(
        &mut self,
        latent: &[f32],
        width: usize,
        height: usize,
        context: &[f32],
        context_len: usize,
        timestep: f32,
    ) -> Result<Vec<f32>, String> {
        if context_len.checked_mul(4096) != Some(context.len()) {
            return Err("Invalid context length".into());
        }
        self.forward_conditioned(
            latent,
            width,
            height,
            &Condition {
                values: context.to_vec(),
                image_slots: Vec::new(),
            },
            &[],
            timestep,
        )
    }

    pub(crate) fn forward_conditioned(
        &mut self,
        latent: &[f32],
        width: usize,
        height: usize,
        condition: &Condition,
        references: &[ReferenceLatent],
        timestep: f32,
    ) -> Result<Vec<f32>, String> {
        let context = &condition.values;
        let context_len = context.len() / 4096;
        let config = self.config.clone();
        let hidden = config.hidden_size;
        let heads = hidden / config.head_dim;
        if width == 0 || height == 0 {
            return Err("Qwen-Image-2.1 latent width and height must be positive".into());
        }
        let image_tokens = width
            .checked_mul(height)
            .ok_or("Qwen-Image-2.1 latent dimensions overflow")?;
        let layout = Layout::build(
            context_len,
            &condition.image_slots,
            references,
            width,
            height,
        )?;
        let seq = layout.positions.len();
        let prefix_length = layout.prefix_length;
        if context_len == 0 {
            return Err("Qwen-Image-2.1 context must not be empty".into());
        }
        if !timestep.is_finite()
            || !latent.iter().all(|value| value.is_finite())
            || !context.iter().all(|value| value.is_finite())
        {
            return Err("Qwen-Image-2.1 inputs must contain only finite values".into());
        }
        let expected_latent = config
            .in_channels
            .checked_mul(image_tokens)
            .ok_or("Qwen-Image-2.1 latent dimensions overflow")?;
        let expected_context = config
            .context_dim
            .checked_mul(context_len)
            .ok_or("Qwen-Image-2.1 context dimensions overflow")?;
        if latent.len() != expected_latent {
            return Err(format!(
                "Invalid latent length: expected {}, got {}",
                expected_latent,
                latent.len()
            ));
        }
        if context.len() != expected_context {
            return Err(format!(
                "Invalid context length: expected {}, got {}",
                expected_context,
                context.len()
            ));
        }
        let report = |name: &str, shape: &[usize], values: &[f32]| {
            let _ = (name, shape, values);
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(name, None, shape, values));
        };
        // The RoPE table depends only on dimensions and precedes the model
        // projections in the oracle trace.
        let pe = self.build_pe(&layout.positions)?;
        report("qwen.pe", &[2, 2, config.head_dim / 2, seq], &pe);

        // --- timestep embedding: [cos; sin] concat, silu, then modulation. ---
        let mut time = vec![0.0f32; 256 * 2];
        timestep_embedding_row(timestep, &mut time[..256]);
        timestep_embedding_row(0.0, &mut time[256..]);
        report("qwen.debug.time", &[256, 2], &time);
        let mut time_embed = vec![0.0f32; hidden * 2];
        self.q8_linear(
            &format!("{PREFIX}.time_text_embed.timestep_embedder.linear_1.weight"),
            256,
            hidden,
            &time,
            &mut time_embed,
        )?;
        report("qwen.debug.temb_linear1", &[hidden, 2], &time_embed);
        crate::ops::silu_approx_inplace(&mut time_embed);
        report("qwen.debug.temb_silu", &[hidden, 2], &time_embed);
        let mut time_hidden = vec![0.0f32; hidden * 2];
        self.q8_linear(
            &format!("{PREFIX}.time_text_embed.timestep_embedder.linear_2.weight"),
            hidden,
            hidden,
            &time_embed,
            &mut time_hidden,
        )?;
        time_embed = time_hidden;
        report("qwen.time_embed", &[hidden, 2], &time_embed);
        // The model forward applies a second silu to the embedding; both the
        // modulation projection and the final scale consume the silu'd values.
        crate::ops::silu_approx_inplace(&mut time_embed);

        let mut modulation = vec![0.0f32; 4 * hidden * 2];
        self.q8_linear(
            &format!("{PREFIX}.modulation.1.weight"),
            hidden,
            4 * hidden,
            &time_embed,
            &mut modulation,
        )?;
        report("qwen.modulation", &[4 * hidden, 2], &modulation);

        // --- text projection. ---
        let mut text = vec![0.0f32; hidden * context_len];
        self.txt_in(context, &mut text)?;
        report("qwen.txt_in", &[hidden, context_len], &text);

        // --- joint sequence: text rows then image rows. ---
        let mut joint = vec![0.0f32; hidden * seq];
        for segment in &layout.segments {
            let target = &mut joint[segment.start * hidden..segment.end * hidden];
            if let Some(index) = segment.image {
                let (data, w, h) = if index == references.len() {
                    (latent, width, height)
                } else {
                    let r = &references[index];
                    (r.values.as_slice(), r.width, r.height)
                };
                let count = w * h;
                let mut rows = vec![0.0; 64 * count];
                for token in 0..count {
                    for c in 0..64 {
                        rows[token * 64 + c] = data[c * count + token];
                    }
                }
                self.bf16_linear(
                    &format!("{PREFIX}.img_in.weight"),
                    64,
                    hidden,
                    &rows,
                    target,
                )?;
            } else {
                target.copy_from_slice(
                    &text[segment.context_start * hidden
                        ..(segment.context_start + segment.end - segment.start) * hidden],
                );
            }
        }
        report("qwen.joint", &[hidden, seq], &joint);

        // --- transformer blocks. ---
        for layer in 0..config.num_layers {
            let prefix = format!("{PREFIX}.transformer_blocks.{layer}");
            self.block(
                &prefix,
                &mut joint,
                &modulation,
                &pe,
                hidden,
                heads,
                config.head_dim,
                seq,
                prefix_length,
                &layout.segments,
            )?;
            report(&format!("qwen.block.{layer}"), &[hidden, seq], &joint);
        }

        // --- final layer over the image rows only. ---
        let joint_final = &joint[hidden * prefix_length..];
        let image_rows = image_tokens;
        report("qwen.joint_final", &[hidden, image_rows], joint_final);

        let mut scale = vec![0.0f32; hidden];
        self.f32_linear(
            &format!("{PREFIX}.norm_out.linear.weight"),
            hidden,
            hidden,
            &time_embed[..hidden],
            &mut scale,
        )?;
        report("qwen.scale", &[hidden], &scale);

        let mut normed = vec![0.0f32; hidden * image_rows];
        for token in 0..image_rows {
            layer_norm_row(
                &joint_final[token * hidden..(token + 1) * hidden],
                &mut normed[token * hidden..(token + 1) * hidden],
            );
        }
        for (i, value) in normed.iter_mut().enumerate() {
            *value *= scale[i % hidden] + 1.0f32;
        }
        report("qwen.norm_out", &[hidden, image_rows], &normed);

        let mut out = vec![0.0f32; config.out_channels * image_rows];
        self.q8_linear(
            &format!("{PREFIX}.proj_out.weight"),
            hidden,
            config.out_channels,
            &normed,
            &mut out,
        )?;
        report("qwen.out", &[config.out_channels, image_rows], &out);
        // Keep input records adjacent to the oracle's post-graph input trace.
        report("qwen.input.x", &[width, height, config.in_channels], latent);
        report(
            "qwen.input.context",
            &[config.context_dim, context_len],
            context,
        );
        report("qwen.input.timesteps", &[1], &[timestep]);

        // --- unpatchify back to [W, H, C]. ---
        let mut output = vec![0.0f32; config.out_channels * image_tokens];
        for h in 0..height {
            for w in 0..width {
                let token = h * width + w;
                for c in 0..config.out_channels {
                    output[w + width * h + width * height * c] =
                        out[token * config.out_channels + c];
                }
            }
        }
        report(
            "qwen.output",
            &[width, height, config.out_channels],
            &output,
        );
        Ok(output)
    }

    fn txt_in(&mut self, context: &[f32], output: &mut [f32]) -> Result<(), String> {
        let report = |name: &str, shape: &[usize], values: &[f32]| {
            let _ = (name, shape, values);
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(name, None, shape, values));
        };
        let hidden = self.config.hidden_size;
        let tokens = context.len() / hidden;
        // Zero-center RMSNorm: the weight participates as (w + 1).
        let norm_weight: Vec<f32> = self
            .bf16_vector_widened(&format!("{PREFIX}.txt_in.text_norm.weight"), hidden)?
            .iter()
            .map(|value| value + 1.0)
            .collect();
        let mut normed = context.to_vec();
        for token in 0..tokens {
            rms_norm_mul_inplace(
                &mut normed[token * hidden..(token + 1) * hidden],
                &norm_weight,
            );
        }
        report("qwen.debug.txt_norm", &[hidden, tokens], &normed);
        let mut projected = vec![0.0f32; hidden * tokens];
        self.bf16_linear(
            &format!("{PREFIX}.txt_in.in_layer.weight"),
            hidden,
            hidden,
            &normed,
            &mut projected,
        )?;
        report("qwen.debug.txt_in_layer", &[hidden, tokens], &projected);
        for value in projected.iter_mut() {
            *value = crate::ops::gelu_ggml_f16(*value);
        }
        report("qwen.debug.txt_gelu", &[hidden, tokens], &projected);
        self.bf16_linear(
            &format!("{PREFIX}.txt_in.out_layer.weight"),
            hidden,
            hidden,
            &projected,
            output,
        )
    }

    /// Builds the [2, 2, head_dim/2, seq] RoPE table row by row.
    fn build_pe(&self, positions: &[[f32; 3]]) -> Result<Vec<f32>, String> {
        let omega = [
            rope_frequencies(16, 10_000.0),
            rope_frequencies(56, 10_000.0),
            rope_frequencies(56, 10_000.0),
        ];
        let mut pe = Vec::with_capacity(positions.len() * 256);
        for position in positions {
            for axis in 0..3 {
                pe.extend(rope_axis_block(position[axis], &omega[axis]));
            }
        }
        Ok(pe)
    }

    #[allow(clippy::too_many_arguments)]
    fn block(
        &mut self,
        prefix: &str,
        joint: &mut [f32],
        modulation: &[f32],
        pe: &[f32],
        hidden: usize,
        heads: usize,
        head_dim: usize,
        seq: usize,
        prefix_length: usize,
        segments: &[Segment],
    ) -> Result<(), String> {
        let config = self.config.clone();

        // img_norm1 (no affine) then modulate: image rows use the timestep row
        // (column 0), text rows the zero-timestep row (column 1).
        let report = |name: &str, values: &[f32]| {
            let _ = (name, values);
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                name,
                None,
                &[hidden, seq],
                values,
            ));
        };
        let mut h = vec![0.0f32; hidden * seq];
        for token in 0..seq {
            layer_norm_row(
                &joint[token * hidden..(token + 1) * hidden],
                &mut h[token * hidden..(token + 1) * hidden],
            );
        }
        report("qwen.debug.b.norm1", &h);
        for token in 0..seq {
            let row = modulation_row(modulation, 0, token < prefix_length, hidden);
            for j in 0..hidden {
                h[token * hidden + j] *= row[j] + 1.0f32;
            }
        }
        report("qwen.debug.b.mod1", &h);

        // --- joint attention. ---
        let mut q = vec![0.0f32; hidden * seq];
        let mut k = vec![0.0f32; hidden * seq];
        let mut v = vec![0.0f32; hidden * seq];
        self.q8_linear(
            &format!("{prefix}.attn.to_q.weight"),
            hidden,
            hidden,
            &h,
            &mut q,
        )?;
        self.q8_linear(
            &format!("{prefix}.attn.to_k.weight"),
            hidden,
            hidden,
            &h,
            &mut k,
        )?;
        self.q8_linear(
            &format!("{prefix}.attn.to_v.weight"),
            hidden,
            hidden,
            &h,
            &mut v,
        )?;

        let norm_q = self.f32_vector(&format!("{prefix}.attn.norm_q.weight"), head_dim)?;
        let norm_k = self.f32_vector(&format!("{prefix}.attn.norm_k.weight"), head_dim)?;
        for token in 0..seq {
            for head in 0..heads {
                let base = token * hidden + head * head_dim;
                rms_norm_mul_inplace(&mut q[base..base + head_dim], &norm_q);
                rms_norm_mul_inplace(&mut k[base..base + head_dim], &norm_k);
                apply_rope_row(
                    &mut q[base..base + head_dim],
                    &pe[token * 256..(token + 1) * 256],
                );
                apply_rope_row(
                    &mut k[base..base + head_dim],
                    &pe[token * 256..(token + 1) * 256],
                );
            }
        }

        let scale = (1.0f64 / (head_dim as f64).sqrt()) as f32;
        let mut v_transposed = vec![0.0f32; hidden * seq];
        for channel in 0..hidden {
            for token in 0..seq {
                v_transposed[channel * seq + token] = v[token * hidden + channel];
            }
        }
        let mut attn_out = vec![0.0f32; hidden * seq];
        let mut scores = vec![0.0f32; seq];
        let mut probs = vec![0.0f32; seq];
        #[cfg(feature = "parity-trace")]
        for (name, values) in [("qwen.debug.attn.q", &q), ("qwen.debug.attn.k", &k)] {
            let transposed: Vec<_> = (0..hidden * seq)
                .map(|i| {
                    values[(i / head_dim % seq) * hidden
                        + i / (seq * head_dim) * head_dim
                        + i % head_dim]
                })
                .collect();
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                name,
                None,
                &[head_dim, seq, heads],
                &transposed,
            ));
        }
        // Text queries attend causally to the text keys; image queries attend
        // to the whole sequence.
        for (index, segment) in segments.iter().enumerate() {
            let _ = index;
            let (query_start, query_end, key_end, causal) = (
                segment.start,
                segment.end,
                segment.end,
                segment.image.is_none(),
            );
            for token in query_start..query_end {
                for head in 0..heads {
                    let q_base = token * hidden + head * head_dim;
                    for key in 0..key_end {
                        let k_base = key * hidden + head * head_dim;
                        scores[key] =
                            crate::ops::dot_f32(&q[q_base..], &k[k_base..], head_dim) * scale;
                    }
                    if causal {
                        for (key, score) in scores[..key_end].iter_mut().enumerate() {
                            if key > token {
                                *score = f32::NEG_INFINITY;
                            }
                        }
                    }
                    probs[..key_end].copy_from_slice(&scores[..key_end]);
                    crate::ops::softmax_approx_inplace(&mut probs[..key_end]);
                    let out_base = token * hidden + head * head_dim;
                    for d in 0..head_dim {
                        attn_out[out_base + d] = crate::ops::dot_f32(
                            &v_transposed[(head * head_dim + d) * seq..],
                            &probs,
                            key_end,
                        );
                    }
                }
            }
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                &format!("qwen.debug.attn.seg{index}"),
                None,
                &[hidden, query_end - query_start],
                &attn_out[query_start * hidden..query_end * hidden],
            ));
        }

        // to_out.0 carries the ggml Linear scale workaround: input * 1/32
        // before quantization, output * 32 after.
        let mut scaled = vec![0.0f32; hidden * seq];
        for (value, target) in attn_out.iter().zip(scaled.iter_mut()) {
            *target = value * ATTENTION_OUT_SCALE;
        }
        let mut attn_projected = vec![0.0f32; hidden * seq];
        self.q8_linear(
            &format!("{prefix}.attn.to_out.0.weight"),
            hidden,
            hidden,
            &scaled,
            &mut attn_projected,
        )?;
        for value in attn_projected.iter_mut() {
            *value *= 32.0f32;
        }
        report("qwen.debug.b.attn", &attn_projected);

        // x += tanh-gated attention output.
        for token in 0..seq {
            let row = modulation_row(modulation, 1, token < prefix_length, hidden);
            for j in 0..hidden {
                joint[token * hidden + j] += row[j].tanh() * attn_projected[token * hidden + j];
            }
        }
        report("qwen.debug.b.res1", joint);

        // img_norm2 + modulate + gated MLP.
        for token in 0..seq {
            layer_norm_row(
                &joint[token * hidden..(token + 1) * hidden],
                &mut h[token * hidden..(token + 1) * hidden],
            );
        }
        for token in 0..seq {
            let row = modulation_row(modulation, 2, token < prefix_length, hidden);
            for j in 0..hidden {
                h[token * hidden + j] *= row[j] + 1.0f32;
            }
        }
        let mut gate_up = vec![0.0f32; 2 * config.intermediate_size * seq];
        report("qwen.debug.b.mod2", &h);
        self.q8_linear(
            &format!("{prefix}.img_mlp.gate_up.weight"),
            hidden,
            2 * config.intermediate_size,
            &h,
            &mut gate_up,
        )?;
        let mut mlp_hidden = vec![0.0f32; config.intermediate_size * seq];
        for token in 0..seq {
            let base = token * 2 * config.intermediate_size;
            let (gate, up) = gate_up[base..base + 2 * config.intermediate_size]
                .split_at_mut(config.intermediate_size);
            crate::ops::silu_approx_inplace(gate);
            for i in 0..config.intermediate_size {
                mlp_hidden[token * config.intermediate_size + i] = up[i] * gate[i];
            }
        }
        let mut mlp_out = vec![0.0f32; hidden * seq];
        self.q8_linear(
            &format!("{prefix}.img_mlp.out.weight"),
            config.intermediate_size,
            hidden,
            &mlp_hidden,
            &mut mlp_out,
        )?;
        for token in 0..seq {
            let row = modulation_row(modulation, 3, token < prefix_length, hidden);
            for j in 0..hidden {
                joint[token * hidden + j] += row[j].tanh() * mlp_out[token * hidden + j];
            }
        }
        Ok(())
    }
}

/// `ggml_ext_chunk(modulation, 4, 0)` then `chunk(.., 2, 1)`: part p selects
/// the feature quarter, row selects the timestep (false) or zero (true)
/// modulation row. Text tokens take the zero row, image tokens the real one.
fn modulation_row<'a>(
    modulation: &'a [f32],
    part: usize,
    zero_row: bool,
    hidden: usize,
) -> &'a [f32] {
    let total = 4 * hidden;
    let row = if zero_row { 1 } else { 0 };
    &modulation[row * total + part * hidden..row * total + (part + 1) * hidden]
}

struct Segment {
    start: usize,
    end: usize,
    context_start: usize,
    image: Option<usize>,
}
struct Layout {
    segments: Vec<Segment>,
    positions: Vec<[f32; 3]>,
    prefix_length: usize,
}
impl Layout {
    fn build(
        text_len: usize,
        slots: &[usize],
        references: &[ReferenceLatent],
        width: usize,
        height: usize,
    ) -> Result<Self, String> {
        if !slots.is_empty() && slots.len() != text_len {
            return Err("Invalid Qwen image slot count".into());
        }
        let mut layout = Self {
            segments: Vec::new(),
            positions: Vec::new(),
            prefix_length: 0,
        };
        let mut position = 0usize;
        let mut image = 0;
        let mut i = 0;
        while i < text_len {
            let tag = slots.get(i).copied().unwrap_or(0);
            let begin = i;
            i += 1;
            while i < text_len && slots.get(i).copied().unwrap_or(0) == tag {
                i += 1;
            }
            if tag == 0 {
                let start = layout.positions.len();
                layout.segments.push(Segment {
                    start,
                    end: start + i - begin,
                    context_start: begin,
                    image: None,
                });
                for _ in begin..i {
                    let p = position as f32;
                    layout.positions.push([p, p, p]);
                    position += 1;
                }
            } else {
                let r = references
                    .get(image)
                    .ok_or("Qwen image slot has no reference")?;
                let count = r
                    .width
                    .checked_mul(r.height)
                    .ok_or("Reference size overflow")?;
                if tag != image + 1
                    || (i - begin).checked_mul(4) != Some(count)
                    || count.checked_mul(64) != Some(r.values.len())
                    || r.values.iter().any(|v| !v.is_finite())
                {
                    return Err("Qwen vision slots and reference latents must match".into());
                }
                layout.append_image(image, begin, r.width, r.height, &mut position)?;
                image += 1;
            }
        }
        if image != references.len() {
            return Err("Missing Qwen reference image slots".into());
        }
        layout.prefix_length = layout.positions.len();
        layout.append_image(image, text_len, width, height, &mut position)?;
        Ok(layout)
    }
    fn append_image(
        &mut self,
        image: usize,
        context_start: usize,
        w: usize,
        h: usize,
        position: &mut usize,
    ) -> Result<(), String> {
        let count = w
            .checked_mul(h)
            .filter(|&n| n > 0)
            .ok_or("Qwen image size overflow")?;
        let start = self.positions.len();
        let end = start.checked_add(count).ok_or("Qwen sequence overflow")?;
        self.segments.push(Segment {
            start,
            end,
            context_start,
            image: Some(image),
        });
        for y in 0..h {
            for x in 0..w {
                self.positions.push([
                    *position as f32,
                    y as f32 - (h - h / 2) as f32,
                    x as f32 - (w - w / 2) as f32,
                ]);
            }
        }
        *position = position
            .checked_add(h.max(w))
            .ok_or("Qwen position overflow")?;
        Ok(())
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    #[test]
    fn references_replace_vision_slots_and_bound_attention_segments() {
        let refs = [ReferenceLatent {
            values: vec![0.; 64 * 8],
            width: 4,
            height: 2,
        }];
        let layout = Layout::build(5, &[0, 1, 1, 0, 0], &refs, 2, 2).unwrap();
        assert_eq!(layout.prefix_length, 11);
        assert_eq!(
            layout.segments.iter().map(|s| s.end).collect::<Vec<_>>(),
            [1, 9, 11, 15]
        );
        assert_eq!(layout.positions[9], [5., 5., 5.]);
        assert!(Layout::build(5, &[0, 1, 0, 0, 0], &refs, 2, 2).is_err());
    }
}
