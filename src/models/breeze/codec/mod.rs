//! Native F32 Qwen3 12 Hz tokenizer shipped with Breeze TTS 2.
//!
//! Tensor names and operations follow qwen-tts 0.1.1 / transformers 4.57.3.
//! Buffers inside this module are time-major [frames, channels].

use crate::core::tensor::{load_f32_tensor, GGMLType, MetaValue, TensorSource};
use crate::ops::dot_f32;
use rayon::prelude::*;

mod decoder;
mod encoder;
#[cfg(test)]
mod tests;

pub const SAMPLE_RATE: u32 = 24_000;
pub const SAMPLES_PER_FRAME: usize = 1920;
const DIM: usize = 512;
const CODE_DIM: usize = 256;
const CODE_SIZE: usize = 2048;

pub struct BreezeCodec {
    encoder: encoder::Encoder,
    decoder: decoder::Decoder,
}

impl BreezeCodec {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        validate_config(source)?;
        Ok(Self {
            encoder: encoder::Encoder::load(source)?,
            decoder: decoder::Decoder::load(source)?,
        })
    }

    /// Encode finite mono PCM sampled at 24 kHz.
    pub fn encode(&self, audio: &[f32]) -> Result<Vec<[u32; 16]>, String> {
        let frames = validate_audio(audio)?;
        let codes = self.encoder.forward(audio)?;
        if codes.len() != frames {
            return Err(format!(
                "Breeze codec encoder produced {} frames, expected {frames}",
                codes.len()
            ));
        }
        Ok(codes)
    }

    /// Match the official public decoder's 300-frame chunks / 25-frame context.
    pub fn decode(&self, codes: &[[u32; 16]]) -> Result<Vec<f32>, String> {
        let samples = validate_frames(codes)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(samples)
            .map_err(|e| format!("Breeze codec PCM allocation: {e}"))?;
        for start in (0..codes.len()).step_by(300) {
            let context = start.min(25);
            let end = (start + 300).min(codes.len());
            let chunk = self.decoder.forward(&codes[start - context..end])?;
            output.extend_from_slice(&chunk[context * SAMPLES_PER_FRAME..]);
        }
        if output.len() != samples || output.iter().any(|v| !v.is_finite()) {
            return Err("Breeze codec produced invalid PCM".into());
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "breeze.codec.audio",
            None,
            &[output.len()],
            &output,
        ));
        Ok(output)
    }
}

fn validate_audio(audio: &[f32]) -> Result<usize, String> {
    if audio.is_empty() || audio.iter().any(|v| !v.is_finite()) {
        return Err("Breeze codec requires nonempty, finite mono 24 kHz audio".into());
    }
    Ok(audio.len().div_ceil(SAMPLES_PER_FRAME))
}

fn validate_frames(codes: &[[u32; 16]]) -> Result<usize, String> {
    if codes.is_empty() || codes.iter().flatten().any(|&id| id >= CODE_SIZE as u32) {
        return Err("Breeze codec requires nonempty frames with 16 code ids in 0..2048".into());
    }
    codes
        .len()
        .checked_mul(SAMPLES_PER_FRAME)
        .ok_or_else(|| "Breeze codec PCM size overflow".into())
}

fn validate_config(source: &dyn TensorSource) -> Result<(), String> {
    if !matches!(source.metadata("general.architecture"), Some(MetaValue::String(s)) if s == "breeze_audio")
    {
        return Err("Breeze codec requires general.architecture=breeze_audio".into());
    }
    let Some(MetaValue::String(json)) = source.metadata("breeze_audio.config") else {
        return Err("Breeze codec is missing breeze_audio.config".into());
    };
    let config: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("Invalid Breeze audio config: {e}"))?;
    for (path, expected) in [
        ("/model_type", serde_json::json!("qwen3_tts_tokenizer_12hz")),
        ("/encoder_valid_num_quantizers", serde_json::json!(16)),
        ("/input_sample_rate", serde_json::json!(24000)),
        ("/output_sample_rate", serde_json::json!(24000)),
        ("/decode_upsample_rate", serde_json::json!(1920)),
        ("/encode_downsample_rate", serde_json::json!(1920)),
        ("/encoder_config/audio_channels", serde_json::json!(1)),
        ("/encoder_config/hidden_size", serde_json::json!(512)),
        ("/encoder_config/num_filters", serde_json::json!(64)),
        ("/encoder_config/num_hidden_layers", serde_json::json!(8)),
        ("/encoder_config/num_attention_heads", serde_json::json!(8)),
        ("/encoder_config/num_key_value_heads", serde_json::json!(8)),
        ("/encoder_config/head_dim", serde_json::json!(64)),
        ("/encoder_config/intermediate_size", serde_json::json!(2048)),
        ("/encoder_config/hidden_act", serde_json::json!("gelu")),
        (
            "/encoder_config/upsampling_ratios",
            serde_json::json!([8, 6, 5, 4]),
        ),
        ("/encoder_config/kernel_size", serde_json::json!(7)),
        ("/encoder_config/last_kernel_size", serde_json::json!(3)),
        ("/encoder_config/residual_kernel_size", serde_json::json!(3)),
        ("/encoder_config/num_residual_layers", serde_json::json!(1)),
        ("/encoder_config/compress", serde_json::json!(2)),
        ("/encoder_config/use_causal_conv", serde_json::json!(true)),
        (
            "/encoder_config/use_conv_shortcut",
            serde_json::json!(false),
        ),
        ("/encoder_config/normalize", serde_json::json!(false)),
        ("/encoder_config/pad_mode", serde_json::json!("constant")),
        ("/encoder_config/attention_bias", serde_json::json!(false)),
        ("/encoder_config/codebook_size", serde_json::json!(2048)),
        ("/encoder_config/codebook_dim", serde_json::json!(256)),
        (
            "/encoder_config/vector_quantization_hidden_dimension",
            serde_json::json!(256),
        ),
        (
            "/encoder_config/num_semantic_quantizers",
            serde_json::json!(1),
        ),
        ("/encoder_config/num_quantizers", serde_json::json!(32)),
        ("/decoder_config/hidden_size", serde_json::json!(512)),
        ("/decoder_config/latent_dim", serde_json::json!(1024)),
        ("/decoder_config/decoder_dim", serde_json::json!(1536)),
        ("/decoder_config/num_hidden_layers", serde_json::json!(8)),
        ("/decoder_config/num_attention_heads", serde_json::json!(16)),
        ("/decoder_config/num_key_value_heads", serde_json::json!(16)),
        ("/decoder_config/head_dim", serde_json::json!(64)),
        ("/decoder_config/intermediate_size", serde_json::json!(1024)),
        ("/decoder_config/hidden_act", serde_json::json!("silu")),
        ("/decoder_config/attention_bias", serde_json::json!(false)),
        ("/decoder_config/sliding_window", serde_json::json!(72)),
        (
            "/decoder_config/upsampling_ratios",
            serde_json::json!([2, 2]),
        ),
        (
            "/decoder_config/upsample_rates",
            serde_json::json!([8, 5, 4, 3]),
        ),
        ("/decoder_config/codebook_size", serde_json::json!(2048)),
        ("/decoder_config/codebook_dim", serde_json::json!(512)),
        ("/decoder_config/num_quantizers", serde_json::json!(16)),
        (
            "/decoder_config/num_semantic_quantizers",
            serde_json::json!(1),
        ),
    ] {
        if config.pointer(path) != Some(&expected) {
            return Err(format!(
                "Unsupported Breeze audio config {path}: expected {expected}"
            ));
        }
    }
    for (path, expected) in [
        ("/encoder_config/norm_eps", 1e-5),
        ("/decoder_config/rms_norm_eps", 1e-5),
        ("/encoder_config/rope_theta", 10000.),
        ("/decoder_config/rope_theta", 10000.),
        ("/encoder_config/_frame_rate", 12.5),
    ] {
        if config.pointer(path).and_then(|v| v.as_f64()) != Some(expected) {
            return Err(format!(
                "Unsupported Breeze audio config {path}: expected {expected}"
            ));
        }
    }
    Ok(())
}

fn tensor(source: &dyn TensorSource, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    let dims: Vec<u64> = shape.iter().rev().map(|&v| v as u64).collect();
    if source
        .tensor_info(name)
        .is_none_or(|i| i.ggml_type != GGMLType::F32)
    {
        return Err(format!("Breeze codec requires original F32 tensor {name}"));
    }
    let values = load_f32_tensor(source, name, &dims)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(format!("Non-finite Breeze codec tensor {name}"));
    }
    Ok(values)
}

fn trace(name: &str, layer: Option<usize>, values: &[f32], channels: usize) {
    #[cfg(feature = "parity-trace")]
    if std::env::var_os("RMI_BREEZE_FINE_CODEC").is_some() {
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            name,
            layer,
            &[values.len() / channels, channels],
            values,
        ));
    }
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, values, channels);
}

struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}

impl Linear {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: tensor(source, &format!("{prefix}.weight"), &[output, input])?,
            bias: if bias {
                tensor(source, &format!("{prefix}.bias"), &[output])?
            } else {
                vec![0.; output]
            },
            input,
            output,
        })
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len() % self.input, 0);
        let mut output = vec![0.; input.len() / self.input * self.output];
        output
            .par_chunks_mut(self.output)
            .zip(input.par_chunks(self.input))
            .for_each(|(out, row)| {
                for (channel, value) in out.iter_mut().enumerate() {
                    *value = dot_f32(
                        row,
                        &self.weight[channel * self.input..(channel + 1) * self.input],
                        self.input,
                    ) + self.bias[channel];
                }
            });
        output
    }
}

struct Conv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    replicate: bool,
}

impl Conv {
    #[allow(clippy::too_many_arguments)]
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        replicate: bool,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: tensor(
                source,
                &format!("{prefix}.weight"),
                &[output, input / groups, kernel],
            )?,
            bias: if bias {
                tensor(source, &format!("{prefix}.bias"), &[output])?
            } else {
                vec![0.; output]
            },
            input,
            output,
            kernel,
            stride,
            dilation,
            groups,
            replicate,
        })
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.is_empty() || input.len() % self.input != 0 {
            return Err("Invalid Breeze codec convolution input".into());
        }
        let length = input.len() / self.input;
        let output_length = length.div_ceil(self.stride);
        let mut output = vec![
            0.;
            output_length
                .checked_mul(self.output)
                .ok_or("Breeze convolution size overflow")?
        ];
        let padding = (self.kernel - 1) * self.dilation + 1 - self.stride;
        let in_group = self.input / self.groups;
        let out_group = self.output / self.groups;
        let reduction = in_group * self.kernel;
        output
            .par_chunks_mut(self.output)
            .enumerate()
            .for_each_init(
                || vec![0.; reduction],
                |patch, (time, out)| {
                    for group in 0..self.groups {
                        for ic in 0..in_group {
                            for tap in 0..self.kernel {
                                let index = (time * self.stride + tap * self.dilation) as isize
                                    - padding as isize;
                                patch[ic * self.kernel + tap] = if self.replicate {
                                    input[index.clamp(0, length as isize - 1) as usize * self.input
                                        + group * in_group
                                        + ic]
                                } else if index >= 0 && index < length as isize {
                                    input[index as usize * self.input + group * in_group + ic]
                                } else {
                                    0.
                                };
                            }
                        }
                        for oc in group * out_group..(group + 1) * out_group {
                            out[oc] = dot_f32(
                                patch,
                                &self.weight[oc * reduction..(oc + 1) * reduction],
                                reduction,
                            ) + self.bias[oc];
                        }
                    }
                },
            );
        Ok(output)
    }
}

struct Norm {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    epsilon: f32,
}

impl Norm {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        channels: usize,
        layer_norm: bool,
        epsilon: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: tensor(source, &format!("{prefix}.weight"), &[channels])?,
            bias: if layer_norm {
                Some(tensor(source, &format!("{prefix}.bias"), &[channels])?)
            } else {
                None
            },
            epsilon,
        })
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let dim = self.weight.len();
        let mut output = vec![0.; input.len()];
        output
            .par_chunks_mut(dim)
            .zip(input.par_chunks(dim))
            .for_each(|(out, row)| {
                if let Some(bias) = &self.bias {
                    let mean = (crate::ops::sum_f32(row) / row.len() as f64) as f32;
                    let var =
                        (crate::ops::sum_sq_centered_f32(row, mean) / row.len() as f64) as f32;
                    let scale = 1. / (var + self.epsilon).sqrt();
                    for i in 0..dim {
                        out[i] = (row[i] - mean) * scale * self.weight[i] + bias[i];
                    }
                } else {
                    crate::ops::rms_norm(row, &self.weight, out, self.epsilon);
                }
            });
        output
    }
}

fn gelu(values: &mut [f32]) {
    for v in values {
        *v = crate::ops::gelu_erf(*v);
    }
}

fn elu(values: &mut [f32]) {
    for v in values {
        if *v < 0. {
            *v = v.exp_m1();
        }
    }
}

fn add_scaled(values: &mut [f32], branch: &[f32], scale: &[f32]) {
    for (i, (out, &add)) in values.iter_mut().zip(branch).enumerate() {
        *out += add * scale[i % scale.len()];
    }
}

struct TransformerLayer {
    norm1: Norm,
    norm2: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    fc1: Linear,
    gate: Option<Linear>,
    fc2: Linear,
    attn_scale: Vec<f32>,
    mlp_scale: Vec<f32>,
    heads: usize,
    window: Option<usize>,
}

impl TransformerLayer {
    fn load(source: &dyn TensorSource, prefix: &str, decoder: bool) -> Result<Self, String> {
        let heads = if decoder { 16 } else { 8 };
        let intermediate = if decoder { 1024 } else { 2048 };
        Ok(Self {
            norm1: Norm::load(
                source,
                &format!("{prefix}.input_layernorm"),
                DIM,
                !decoder,
                1e-5,
            )?,
            norm2: Norm::load(
                source,
                &format!("{prefix}.post_attention_layernorm"),
                DIM,
                !decoder,
                1e-5,
            )?,
            q: Linear::load(
                source,
                &format!("{prefix}.self_attn.q_proj"),
                DIM,
                heads * 64,
                false,
            )?,
            k: Linear::load(
                source,
                &format!("{prefix}.self_attn.k_proj"),
                DIM,
                heads * 64,
                false,
            )?,
            v: Linear::load(
                source,
                &format!("{prefix}.self_attn.v_proj"),
                DIM,
                heads * 64,
                false,
            )?,
            o: Linear::load(
                source,
                &format!("{prefix}.self_attn.o_proj"),
                heads * 64,
                DIM,
                false,
            )?,
            fc1: Linear::load(
                source,
                &format!("{prefix}.mlp.{}", if decoder { "up_proj" } else { "fc1" }),
                DIM,
                intermediate,
                false,
            )?,
            gate: if decoder {
                Some(Linear::load(
                    source,
                    &format!("{prefix}.mlp.gate_proj"),
                    DIM,
                    intermediate,
                    false,
                )?)
            } else {
                None
            },
            fc2: Linear::load(
                source,
                &format!("{prefix}.mlp.{}", if decoder { "down_proj" } else { "fc2" }),
                intermediate,
                DIM,
                false,
            )?,
            attn_scale: tensor(
                source,
                &format!("{prefix}.self_attn_layer_scale.scale"),
                &[DIM],
            )?,
            mlp_scale: tensor(source, &format!("{prefix}.mlp_layer_scale.scale"), &[DIM])?,
            heads,
            window: if decoder { Some(72) } else { None },
        })
    }

    fn forward(&self, mut x: Vec<f32>, index: usize) -> Vec<f32> {
        let prefix = if self.gate.is_some() {
            "breeze.codec.decoder"
        } else {
            "breeze.codec.encoder"
        };
        let stage = |name: &str, values: &[f32], channels| {
            trace(
                &format!("{prefix}.transformer.{name}"),
                Some(index),
                values,
                channels,
            );
        };
        let normalized = self.norm1.forward(&x);
        stage("norm1", &normalized, DIM);
        let mut q = self.q.forward(&normalized);
        let mut k = self.k.forward(&normalized);
        let v = self.v.forward(&normalized);
        stage("q", &q, self.heads * 64);
        stage("k", &k, self.heads * 64);
        stage("v", &v, self.heads * 64);
        rope(&mut q, self.heads);
        rope(&mut k, self.heads);
        stage("q_rope", &q, self.heads * 64);
        stage("k_rope", &k, self.heads * 64);
        let channels = self.heads * 64;
        let frames = q.len() / channels;
        let mut attention = vec![0.; q.len()];
        attention
            .par_chunks_mut(channels)
            .enumerate()
            .for_each(|(time, row)| {
                let first = self.window.map_or(0, |w| (time + 1).saturating_sub(w));
                let mut scores = vec![f32::NEG_INFINITY; frames];
                for head in 0..self.heads {
                    scores.fill(f32::NEG_INFINITY);
                    let qr = &q[time * channels + head * 64..time * channels + (head + 1) * 64];
                    for (offset, score) in scores.iter_mut().enumerate().take(time + 1).skip(first)
                    {
                        let kr =
                            &k[offset * channels + head * 64..offset * channels + (head + 1) * 64];
                        *score = dot_f32(qr, kr, 64) * 0.125;
                    }
                    crate::ops::softmax_inplace(&mut scores);
                    for (offset, &probability) in
                        scores.iter().enumerate().take(time + 1).skip(first)
                    {
                        let vr =
                            &v[offset * channels + head * 64..offset * channels + (head + 1) * 64];
                        for i in 0..64 {
                            row[head * 64 + i] = probability.mul_add(vr[i], row[head * 64 + i]);
                        }
                    }
                }
            });
        stage("attention", &attention, channels);
        let projected = self.o.forward(&attention);
        stage("o", &projected, DIM);
        add_scaled(&mut x, &projected, &self.attn_scale);
        stage("after_attention", &x, DIM);
        let normalized = self.norm2.forward(&x);
        stage("norm2", &normalized, DIM);
        let mut hidden = self.fc1.forward(&normalized);
        stage("up", &hidden, self.fc1.output);
        if let Some(gate) = &self.gate {
            let gated = gate.forward(&normalized);
            stage("gate", &gated, self.fc1.output);
            for (h, g) in hidden.iter_mut().zip(gated) {
                *h *= crate::ops::silu(g);
            }
        } else {
            gelu(&mut hidden);
        }
        stage("activation", &hidden, self.fc1.output);
        let projected = self.fc2.forward(&hidden);
        stage("down", &projected, DIM);
        add_scaled(&mut x, &projected, &self.mlp_scale);
        x
    }
}

fn rope(values: &mut [f32], heads: usize) {
    for (position, row) in values.chunks_exact_mut(heads * 64).enumerate() {
        for head in row.chunks_exact_mut(64) {
            for i in 0..32 {
                let angle = position as f32 / 10_000_f32.powf((2 * i) as f32 / 64.0);
                let (sin, cos) = crate::ops::rope_sin_cos(angle);
                let first = head[i];
                let second = head[i + 32];
                head[i] = first * cos - second * sin;
                head[i + 32] = second * cos + first * sin;
            }
        }
    }
}

struct Codebook {
    values: Vec<f32>,
    norms: Vec<f32>,
    dim: usize,
}

impl Codebook {
    fn load(source: &dyn TensorSource, prefix: &str, decoder: bool) -> Result<Self, String> {
        let usage = tensor(source, &format!("{prefix}.cluster_usage"), &[CODE_SIZE])?;
        let mut values = tensor(
            source,
            &format!(
                "{prefix}.{}",
                if decoder {
                    "embedding_sum"
                } else {
                    "embed_sum"
                }
            ),
            &[CODE_SIZE, CODE_DIM],
        )?;
        for (row, &count) in values.chunks_exact_mut(CODE_DIM).zip(&usage) {
            for value in row {
                *value /= count.max(1e-5);
            }
        }
        let norms = values
            .chunks_exact(CODE_DIM)
            .map(|row| crate::ops::sum_sq_f32(row) as f32)
            .collect();
        Ok(Self {
            values,
            norms,
            dim: CODE_DIM,
        })
    }

    fn encode(&self, residual: &mut [f32]) -> Vec<u32> {
        residual
            .par_chunks_mut(self.dim)
            .map(|row| {
                let norm = crate::ops::sum_sq_f32(row) as f32;
                let mut best = 0;
                let mut distance = f32::INFINITY;
                for (id, centroid) in self.values.chunks_exact(self.dim).enumerate() {
                    let candidate = ((-2. * dot_f32(row, centroid, self.dim) + norm)
                        + self.norms[id])
                        .max(0.)
                        .sqrt();
                    if candidate < distance {
                        distance = candidate;
                        best = id;
                    }
                }
                for (v, &center) in row
                    .iter_mut()
                    .zip(&self.values[best * self.dim..(best + 1) * self.dim])
                {
                    *v -= center;
                }
                best as u32
            })
            .collect()
    }

    fn add_codes(&self, output: &mut [f32], codes: &[[u32; 16]], level: usize) {
        for (row, frame) in output.chunks_exact_mut(self.dim).zip(codes) {
            let start = frame[level] as usize * self.dim;
            for (v, &center) in row.iter_mut().zip(&self.values[start..start + self.dim]) {
                *v += center;
            }
        }
    }
}
