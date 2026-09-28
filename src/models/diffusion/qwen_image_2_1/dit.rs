//! Qwen-Image-2.1 DiT forward pass, numerically aligned with the pinned
//! stable-diffusion.cpp oracle's ggml CPU kernels (see tools/oracle/qwen_image_2_1).

use super::{QwenImage21Config, PREFIX};
use crate::core::tensor::TensorSource;
use rayon::prelude::*;
use std::sync::Arc;

const RMS_EPS: f32 = 1e-6;
const LAYER_NORM_EPS: f32 = 1e-6;
const ATTENTION_OUT_SCALE: f32 = 1.0 / 32.0;

/// ggml `ggml_compute_fp32_to_bf16`: round-to-nearest-even with carry.
fn fp32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) | 64) as u16;
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// ggml `ggml_vec_dot_bf16` on aarch64: no NEON branch, scalar f64
/// accumulation of single-precision products. Both operands are BF16; the
/// activation row is rounded once by the mul_mat `from_float` step.
fn bf16_dot(w_bf16: &[u16], act_bf16: &[u16], n: usize) -> f32 {
    let mut sumf = 0.0f64;
    for i in 0..n {
        let product = crate::ops::bf16_to_f32(w_bf16[i]) * crate::ops::bf16_to_f32(act_bf16[i]);
        sumf += f64::from(product);
    }
    sumf as f32
}

/// Lane-wise replica of ggml's NEON `ggml_v_expf` polynomial approximation.
fn rms_norm_mul_inplace(x: &mut [f32], w: &[f32]) {
    crate::ops::rms_norm_inplace(x, w, RMS_EPS);
}

/// ggml `ggml_compute_forward_norm_f32` (LayerNorm, no affine): f64 mean,
/// 4-wide centered variance summed pairwise into f64, then one scale multiply.
fn layer_norm_row(x: &[f32], y: &mut [f32]) {
    let mut sum = 0.0f64;
    for value in x.iter() {
        sum += f64::from(*value);
    }
    let mean = (sum as f32) / x.len() as f32;
    let mut variance_sum = 0.0f64;
    let mut quad = [0.0f32; 4];
    let mut i = 0;
    while i + 4 <= x.len() {
        for lane in 0..4 {
            quad[lane] = x[i + lane] - mean;
        }
        y[i..i + 4].copy_from_slice(&quad);
        quad[0] *= quad[0];
        quad[1] *= quad[1];
        quad[2] *= quad[2];
        quad[3] *= quad[3];
        variance_sum += f64::from((quad[0] + quad[1]) + (quad[2] + quad[3]));
        i += 4;
    }
    while i < x.len() {
        let centered = x[i] - mean;
        y[i] = centered;
        variance_sum += f64::from(centered * centered);
        i += 1;
    }
    let variance = (variance_sum / x.len() as f64) as f32;
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
        let (sin_value, cos_value) = arg.sin_cos();
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
        let weight: Vec<u16> = weight
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        let mut act = Vec::with_capacity(input.len());
        for value in input {
            act.push(fp32_to_bf16(*value));
        }
        for token in 0..tokens {
            let row = &act[token * n_in..(token + 1) * n_in];
            for o in 0..n_out {
                output[token * n_out + o] = bf16_dot(&weight[o * n_in..(o + 1) * n_in], row, n_in);
            }
        }
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
        for token in 0..tokens {
            let row = &input[token * n_in..(token + 1) * n_in];
            for o in 0..n_out {
                output[token * n_out + o] =
                    crate::ops::dot_f32(&weight[o * n_in..(o + 1) * n_in], row, n_in);
            }
        }
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
        let config = self.config.clone();
        let hidden = config.hidden_size;
        let heads = hidden / config.head_dim;
        if width == 0 || height == 0 {
            return Err("Qwen-Image-2.1 latent width and height must be positive".into());
        }
        let image_tokens = width
            .checked_mul(height)
            .ok_or("Qwen-Image-2.1 latent dimensions overflow")?;
        let seq = context_len
            .checked_add(image_tokens)
            .ok_or("Qwen-Image-2.1 sequence length overflow")?;
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
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(name, None, shape, values));
        };
        // The RoPE table depends only on dimensions and precedes the model
        // projections in the oracle trace.
        let pe = self.build_pe(width, height, context_len, seq)?;
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
        joint[..hidden * context_len].copy_from_slice(&text);
        let mut patchified = vec![0.0f32; config.in_channels * image_tokens];
        for h in 0..height {
            for w in 0..width {
                let token = h * width + w;
                for c in 0..config.in_channels {
                    patchified[token * config.in_channels + c] =
                        latent[w + width * h + width * height * c];
                }
            }
        }
        let mut img_in_out = vec![0.0f32; hidden * image_tokens];
        self.bf16_linear(
            &format!("{PREFIX}.img_in.weight"),
            config.in_channels,
            hidden,
            &patchified,
            &mut img_in_out,
        )?;
        joint[hidden * context_len..].copy_from_slice(&img_in_out);
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
                context_len,
            )?;
            report(&format!("qwen.block.{layer}"), &[hidden, seq], &joint);
        }

        // --- final layer over the image rows only. ---
        let joint_final = &joint[hidden * context_len..];
        let image_rows = seq - context_len;
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
    fn build_pe(
        &self,
        width: usize,
        height: usize,
        context_len: usize,
        seq: usize,
    ) -> Result<Vec<f32>, String> {
        let omega0 = rope_frequencies(16, 10_000.0);
        let omega1 = rope_frequencies(56, 10_000.0);
        let omega2 = rope_frequencies(56, 10_000.0);
        let mut pe = vec![0.0f32; seq * 256];
        for token in 0..seq {
            let (axis0, axis1, axis2) = if token < context_len {
                let position = token as f32;
                (position, position, position)
            } else {
                let image_token = token - context_len;
                let h = (image_token / width) as f32;
                let w = (image_token % width) as f32;
                (
                    context_len as f32,
                    h - (height - height / 2) as f32,
                    w - (width - width / 2) as f32,
                )
            };
            let mut row = Vec::with_capacity(256);
            row.extend(rope_axis_block(axis0, &omega0));
            row.extend(rope_axis_block(axis1, &omega1));
            row.extend(rope_axis_block(axis2, &omega2));
            pe[token * 256..(token + 1) * 256].copy_from_slice(&row);
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
    ) -> Result<(), String> {
        let config = self.config.clone();

        // img_norm1 (no affine) then modulate: image rows use the timestep row
        // (column 0), text rows the zero-timestep row (column 1).
        let report = |name: &str, values: &[f32]| {
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
        report("qwen.debug.attn.q", &q);
        report("qwen.debug.attn.k", &k);
        // Text queries attend causally to the text keys; image queries attend
        // to the whole sequence.
        for (query_start, query_end, key_end, causal) in [
            (0usize, prefix_length, prefix_length, true),
            (prefix_length, seq, seq, false),
        ]
        .into_iter()
        {
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
