use super::Gemma4AudioConfig;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::qwen3::asr::audio_processor::{decode_pcm16_wav_any, RealFft};
use crate::ops::{
    dot_f16_f16_bytes, dot_f16_f32, dot_f32, f32_to_f16, rms_norm_inplace, silu,
    softmax_ggml_inplace, sum_sq_f32,
};
use std::path::Path;

const SAMPLE_RATE: u32 = 16_000;
const FFT_SIZE: usize = 512;
const WINDOW_SIZE: usize = 320;
const HOP: usize = 160;
const MEL_BINS: usize = 128;
const CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
const EMBED: usize = 1024;
const FFN: usize = 4096;
const HEADS: usize = 8;
const HEAD_DIM: usize = EMBED / HEADS;
const PROJECTION: usize = 1536;
const LAYERS: usize = 12;
const LOCAL_CONTEXT: usize = 12;
const RPE_LEN: usize = LOCAL_CONTEXT + 1;
const CONV_KERNEL: usize = 5;
const EPS: f32 = 1e-6;
const SOFTCAP: f32 = 50.0;

#[derive(Clone, Copy)]
struct Clamp {
    input_min: f32,
    input_max: f32,
    output_min: f32,
    output_max: f32,
}

struct F16Linear<'a> {
    weight: &'a [u8],
    input: usize,
    output: usize,
    clamp: Option<Clamp>,
}

struct F32Linear<'a> {
    weight: &'a [f32],
    input: usize,
    output: usize,
}

struct FrontendConv<'a> {
    weight: &'a [f32],
    norm: &'a [f32],
    input_channels: usize,
    output_channels: usize,
}

struct AudioLayer<'a> {
    ffn_norm: &'a [f32],
    ffn_up: F16Linear<'a>,
    ffn_down: F16Linear<'a>,
    ffn_post_norm: &'a [f32],
    attn_pre_norm: &'a [f32],
    q: F16Linear<'a>,
    k: F16Linear<'a>,
    v: F16Linear<'a>,
    per_dim_scale: &'a [f32],
    relative: F16Linear<'a>,
    output: F16Linear<'a>,
    attn_post_norm: &'a [f32],
    pre_conv_norm: &'a [f32],
    conv_pw1: F16Linear<'a>,
    conv_dw: &'a [f32],
    post_conv_norm: &'a [f32],
    conv_pw2: F16Linear<'a>,
    ffn_norm_1: &'a [f32],
    ffn_up_1: F16Linear<'a>,
    ffn_down_1: F16Linear<'a>,
    ffn_post_norm_1: &'a [f32],
    output_norm: &'a [f32],
}

pub struct Gemma4AudioModel<'a> {
    pub config: Gemma4AudioConfig,
    pool: ComputePool,
    conv0: FrontendConv<'a>,
    conv1: FrontendConv<'a>,
    input_projection: F32Linear<'a>,
    layers: Vec<AudioLayer<'a>>,
    output_projection: F16Linear<'a>,
    output_bias: &'a [f32],
    multimodal_projection: F16Linear<'a>,
}

struct AudioScratch {
    activation_f16: Vec<u16>,
    x: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention: Vec<f32>,
    projected: Vec<f32>,
    expanded: Vec<f32>,
    convolution: Vec<f32>,
    relative: Vec<f32>,
}

#[derive(Clone, Copy)]
struct SharedMut<T>(*mut T);

unsafe impl<T> Send for SharedMut<T> {}
unsafe impl<T> Sync for SharedMut<T> {}

impl<T> SharedMut<T> {
    unsafe fn write(&self, index: usize, value: T) {
        self.0.add(index).write(value);
    }
}

/// Gemma4A log-mel data in mel-major `[128, frames]` layout.
pub struct Gemma4AudioFeatures {
    pub values: Vec<f32>,
    pub frames: usize,
}

impl<'a> Gemma4AudioModel<'a> {
    pub fn from_source(source: &'a dyn TensorSource, threads: usize) -> Result<Self, String> {
        let config = Gemma4AudioConfig::from_source(source)?;
        let conv0 = FrontendConv {
            weight: f32_tensor(source, "a.conv1d.0.weight", &[3, 3, 1, 128])?,
            norm: f32_tensor(source, "a.conv1d.0.norm.weight", &[128])?,
            input_channels: 1,
            output_channels: 128,
        };
        let conv1 = FrontendConv {
            weight: f32_tensor(source, "a.conv1d.1.weight", &[3, 3, 128, 32])?,
            norm: f32_tensor(source, "a.conv1d.1.norm.weight", &[32])?,
            input_channels: 128,
            output_channels: 32,
        };
        let input_projection = F32Linear::plain(source, "a.input_projection.weight", EMBED, EMBED)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(LAYERS)
            .map_err(|_| "Gemma4 audio layer allocation failed")?;
        for index in 0..LAYERS {
            let prefix = format!("a.blk.{index}");
            layers.push(AudioLayer {
                ffn_norm: f32_tensor(
                    source,
                    &format!("{prefix}.ffn_norm.weight"),
                    &[EMBED as u64],
                )?,
                ffn_up: F16Linear::clippable(source, &format!("{prefix}.ffn_up"), EMBED, FFN)?,
                ffn_down: F16Linear::clippable(source, &format!("{prefix}.ffn_down"), FFN, EMBED)?,
                ffn_post_norm: f32_tensor(
                    source,
                    &format!("{prefix}.ffn_post_norm.weight"),
                    &[EMBED as u64],
                )?,
                attn_pre_norm: f32_tensor(
                    source,
                    &format!("{prefix}.attn_pre_norm.weight"),
                    &[EMBED as u64],
                )?,
                q: F16Linear::clippable(source, &format!("{prefix}.attn_q"), EMBED, EMBED)?,
                k: F16Linear::clippable(source, &format!("{prefix}.attn_k"), EMBED, EMBED)?,
                v: F16Linear::clippable(source, &format!("{prefix}.attn_v"), EMBED, EMBED)?,
                per_dim_scale: f32_tensor(
                    source,
                    &format!("{prefix}.per_dim_scale.weight"),
                    &[HEAD_DIM as u64],
                )?,
                relative: F16Linear::plain(
                    source,
                    &format!("{prefix}.attn_k_rel.weight"),
                    EMBED,
                    EMBED,
                )?,
                output: F16Linear::clippable(source, &format!("{prefix}.attn_out"), EMBED, EMBED)?,
                attn_post_norm: f32_tensor(
                    source,
                    &format!("{prefix}.attn_post_norm.weight"),
                    &[EMBED as u64],
                )?,
                // The pinned converter swaps these two GGUF names.
                pre_conv_norm: f32_tensor(
                    source,
                    &format!("{prefix}.conv_norm.weight"),
                    &[EMBED as u64],
                )?,
                conv_pw1: F16Linear::clippable(
                    source,
                    &format!("{prefix}.conv_pw1"),
                    EMBED,
                    EMBED * 2,
                )?,
                conv_dw: f32_tensor(
                    source,
                    &format!("{prefix}.conv_dw.weight"),
                    &[CONV_KERNEL as u64, EMBED as u64],
                )?,
                post_conv_norm: f32_tensor(
                    source,
                    &format!("{prefix}.norm_conv.weight"),
                    &[EMBED as u64],
                )?,
                conv_pw2: F16Linear::clippable(
                    source,
                    &format!("{prefix}.conv_pw2"),
                    EMBED,
                    EMBED,
                )?,
                ffn_norm_1: f32_tensor(
                    source,
                    &format!("{prefix}.ffn_norm_1.weight"),
                    &[EMBED as u64],
                )?,
                ffn_up_1: F16Linear::clippable(source, &format!("{prefix}.ffn_up_1"), EMBED, FFN)?,
                ffn_down_1: F16Linear::clippable(
                    source,
                    &format!("{prefix}.ffn_down_1"),
                    FFN,
                    EMBED,
                )?,
                ffn_post_norm_1: f32_tensor(
                    source,
                    &format!("{prefix}.ffn_post_norm_1.weight"),
                    &[EMBED as u64],
                )?,
                output_norm: f32_tensor(source, &format!("{prefix}.ln2.weight"), &[EMBED as u64])?,
            });
        }
        Ok(Self {
            config,
            pool: ComputePool::new(threads.max(1)),
            conv0,
            conv1,
            input_projection,
            layers,
            output_projection: F16Linear::plain(
                source,
                "a.pre_encode.out.weight",
                EMBED,
                PROJECTION,
            )?,
            output_bias: f32_tensor(source, "a.pre_encode.out.bias", &[PROJECTION as u64])?,
            multimodal_projection: F16Linear::plain(
                source,
                "mm.a.input_projection.weight",
                PROJECTION,
                PROJECTION,
            )?,
        })
    }

    pub fn encode_wav_path(&self, path: &Path) -> Result<Vec<f32>, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("Failed to read Gemma4 audio {}: {error}", path.display()))?;
        self.encode_features(&gemma4_audio_features(&bytes)?)
    }

    fn encode_features(&self, features: &Gemma4AudioFeatures) -> Result<Vec<f32>, String> {
        let expected = checked_len("Gemma4 Mel input", &[MEL_BINS, features.frames])?;
        if features.frames == 0
            || features.values.len() != expected
            || features.values.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid Gemma4 Mel input".into());
        }
        let (mut x, rows) = self.frontend(features)?;
        let hidden_len = checked_len("Gemma4 audio hidden", &[rows, EMBED])?;
        let expanded_len = checked_len("Gemma4 audio FFN", &[rows, FFN])?;
        let mut scratch = AudioScratch {
            activation_f16: Vec::new(),
            x: Vec::new(),
            normed: zeroed_f32("Gemma4 audio norm", hidden_len)?,
            q: zeroed_f32("Gemma4 audio query", hidden_len)?,
            k: zeroed_f32("Gemma4 audio key", hidden_len)?,
            v: zeroed_f32("Gemma4 audio value", hidden_len)?,
            attention: zeroed_f32("Gemma4 audio attention", hidden_len)?,
            projected: zeroed_f32("Gemma4 audio projection", hidden_len)?,
            expanded: zeroed_f32("Gemma4 audio FFN", expanded_len)?,
            convolution: zeroed_f32("Gemma4 audio convolution", hidden_len)?,
            relative: zeroed_f32("Gemma4 audio relative position", RPE_LEN * EMBED)?,
        };
        std::mem::swap(&mut x, &mut scratch.x);
        let positions = sinusoidal_positions()?;

        for layer in &self.layers {
            scratch.normed.copy_from_slice(&scratch.x);
            weighted_rms_rows(&mut scratch.normed, layer.ffn_norm, EMBED)?;
            layer.ffn_up.forward(
                &self.pool,
                &scratch.normed,
                rows,
                &mut scratch.expanded,
                &mut scratch.activation_f16,
            )?;
            scratch
                .expanded
                .iter_mut()
                .for_each(|value| *value = silu(*value));
            layer.ffn_down.forward(
                &self.pool,
                &scratch.expanded,
                rows,
                &mut scratch.projected,
                &mut scratch.activation_f16,
            )?;
            weighted_rms_rows(&mut scratch.projected, layer.ffn_post_norm, EMBED)?;
            add_scaled(&mut scratch.x, &scratch.projected, 0.5)?;

            scratch.normed.copy_from_slice(&scratch.x);
            weighted_rms_rows(&mut scratch.normed, layer.attn_pre_norm, EMBED)?;
            layer.q.forward(
                &self.pool,
                &scratch.normed,
                rows,
                &mut scratch.q,
                &mut scratch.activation_f16,
            )?;
            layer.k.forward(
                &self.pool,
                &scratch.normed,
                rows,
                &mut scratch.k,
                &mut scratch.activation_f16,
            )?;
            layer.v.forward(
                &self.pool,
                &scratch.normed,
                rows,
                &mut scratch.v,
                &mut scratch.activation_f16,
            )?;
            scale_queries_and_keys(&mut scratch.q, &mut scratch.k, layer.per_dim_scale)?;
            layer.relative.forward(
                &self.pool,
                &positions,
                RPE_LEN,
                &mut scratch.relative,
                &mut scratch.activation_f16,
            )?;
            local_attention(
                &self.pool,
                &scratch.q,
                &scratch.k,
                &scratch.v,
                &scratch.relative,
                rows,
                &mut scratch.attention,
            )?;
            layer.output.forward(
                &self.pool,
                &scratch.attention,
                rows,
                &mut scratch.projected,
                &mut scratch.activation_f16,
            )?;
            weighted_rms_rows(&mut scratch.projected, layer.attn_post_norm, EMBED)?;
            add_scaled(&mut scratch.x, &scratch.projected, 1.0)?;

            scratch.normed.copy_from_slice(&scratch.x);
            weighted_rms_rows(&mut scratch.normed, layer.pre_conv_norm, EMBED)?;
            let pointwise_len = checked_len("Gemma4 pointwise convolution", &[rows, EMBED * 2])?;
            layer.conv_pw1.forward(
                &self.pool,
                &scratch.normed,
                rows,
                &mut scratch.expanded[..pointwise_len],
                &mut scratch.activation_f16,
            )?;
            for row in 0..rows {
                for feature in 0..EMBED {
                    let base = row * EMBED * 2;
                    scratch.convolution[row * EMBED + feature] = scratch.expanded[base + feature]
                        * (1.0 / (1.0 + (-scratch.expanded[base + EMBED + feature]).exp()));
                }
            }
            causal_depthwise_conv(
                &self.pool,
                &scratch.convolution,
                layer.conv_dw,
                rows,
                &mut scratch.projected,
            )?;
            weighted_rms_rows(&mut scratch.projected, layer.post_conv_norm, EMBED)?;
            scratch
                .projected
                .iter_mut()
                .for_each(|value| *value = silu(*value));
            layer.conv_pw2.forward(
                &self.pool,
                &scratch.projected,
                rows,
                &mut scratch.attention,
                &mut scratch.activation_f16,
            )?;
            add_scaled(&mut scratch.x, &scratch.attention, 1.0)?;

            scratch.normed.copy_from_slice(&scratch.x);
            weighted_rms_rows(&mut scratch.normed, layer.ffn_norm_1, EMBED)?;
            layer.ffn_up_1.forward(
                &self.pool,
                &scratch.normed,
                rows,
                &mut scratch.expanded,
                &mut scratch.activation_f16,
            )?;
            scratch
                .expanded
                .iter_mut()
                .for_each(|value| *value = silu(*value));
            layer.ffn_down_1.forward(
                &self.pool,
                &scratch.expanded,
                rows,
                &mut scratch.projected,
                &mut scratch.activation_f16,
            )?;
            weighted_rms_rows(&mut scratch.projected, layer.ffn_post_norm_1, EMBED)?;
            add_scaled(&mut scratch.x, &scratch.projected, 0.5)?;
            weighted_rms_rows(&mut scratch.x, layer.output_norm, EMBED)?;
        }

        let projected_len = checked_len("Gemma4 projected audio", &[rows, PROJECTION])?;
        let mut projected = zeroed_f32("Gemma4 projected audio", projected_len)?;
        self.output_projection.forward(
            &self.pool,
            &scratch.x,
            rows,
            &mut projected,
            &mut scratch.activation_f16,
        )?;
        for row in projected.chunks_exact_mut(PROJECTION) {
            for (value, bias) in row.iter_mut().zip(self.output_bias) {
                *value += *bias;
            }
        }
        unweighted_rms_rows(&mut projected, PROJECTION)?;
        let mut output = zeroed_f32("Gemma4 multimodal audio", projected_len)?;
        self.multimodal_projection.forward(
            &self.pool,
            &projected,
            rows,
            &mut output,
            &mut scratch.activation_f16,
        )?;
        validate_projected_rows(&output)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "gemma4.audio.projected",
            None,
            &[PROJECTION, rows, 1, 1],
            &output,
        ));
        Ok(output)
    }

    fn frontend(&self, features: &Gemma4AudioFeatures) -> Result<(Vec<f32>, usize), String> {
        let mut input = zeroed_f32("Gemma4 transposed Mel", features.values.len())?;
        for frame in 0..features.frames {
            for mel in 0..MEL_BINS {
                input[frame * MEL_BINS + mel] = features.values[mel * features.frames + frame];
            }
        }
        let (stage0, freq0, time0) =
            conv2d_stride2(&self.pool, &input, MEL_BINS, features.frames, &self.conv0)?;
        let (stage1, freq1, time1) =
            conv2d_stride2(&self.pool, &stage0, freq0, time0, &self.conv1)?;
        if freq1.checked_mul(self.conv1.output_channels) != Some(EMBED) || time1 == 0 {
            return Err("Invalid Gemma4 convolution output shape".into());
        }
        let mut output = zeroed_f32("Gemma4 input projection", time1 * EMBED)?;
        self.input_projection
            .forward(&self.pool, &stage1, time1, &mut output)?;
        Ok((output, time1))
    }
}

impl<'a> F16Linear<'a> {
    fn plain(
        source: &'a dyn TensorSource,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: f16_tensor(source, name, &[input as u64, output as u64])?,
            input,
            output,
            clamp: None,
        })
    }

    fn clippable(
        source: &'a dyn TensorSource,
        prefix: &str,
        input: usize,
        output: usize,
    ) -> Result<Self, String> {
        let clamp = Clamp {
            input_min: f32_scalar(source, &format!("{prefix}.input_min"))?,
            input_max: f32_scalar(source, &format!("{prefix}.input_max"))?,
            output_min: f32_scalar(source, &format!("{prefix}.output_min"))?,
            output_max: f32_scalar(source, &format!("{prefix}.output_max"))?,
        };
        if clamp.input_min > clamp.input_max || clamp.output_min > clamp.output_max {
            return Err(format!("Invalid clamp bounds: {prefix}"));
        }
        Ok(Self {
            weight: f16_tensor(
                source,
                &format!("{prefix}.weight"),
                &[input as u64, output as u64],
            )?,
            input,
            output,
            clamp: Some(clamp),
        })
    }

    fn forward(
        &self,
        pool: &ComputePool,
        input: &[f32],
        rows: usize,
        output: &mut [f32],
        activation: &mut Vec<u16>,
    ) -> Result<(), String> {
        let input_len = checked_len("Gemma4 F16 input", &[rows, self.input])?;
        let output_len = checked_len("Gemma4 F16 output", &[rows, self.output])?;
        if rows == 0
            || input.len() != input_len
            || output.len() != output_len
            || input.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid Gemma4 F16 linear shape".into());
        }
        resize_u16(activation, input_len, "Gemma4 F16 activation")?;
        for (source, target) in input.iter().zip(activation.iter_mut()) {
            *target = f32_to_f16(
                self.clamp
                    .map(|clamp| source.clamp(clamp.input_min, clamp.input_max))
                    .unwrap_or(*source),
            );
        }
        let output_ptr = SharedMut(output.as_mut_ptr());
        pool.compute(|thread, threads| {
            for index in (thread..output_len).step_by(threads) {
                let row = index / self.output;
                let column = index % self.output;
                let value = dot_f16_f16_bytes(
                    &activation[row * self.input..(row + 1) * self.input],
                    &self.weight[column * self.input * 2..(column + 1) * self.input * 2],
                    self.input,
                );
                let value = self
                    .clamp
                    .map(|clamp| value.clamp(clamp.output_min, clamp.output_max))
                    .unwrap_or(value);
                unsafe { output_ptr.write(index, value) };
            }
        });
        validate_finite("Gemma4 F16 linear output", output)
    }
}

impl<'a> F32Linear<'a> {
    fn plain(
        source: &'a dyn TensorSource,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: f32_tensor(source, name, &[input as u64, output as u64])?,
            input,
            output,
        })
    }

    fn forward(
        &self,
        pool: &ComputePool,
        input: &[f32],
        rows: usize,
        output: &mut [f32],
    ) -> Result<(), String> {
        let input_len = checked_len("Gemma4 F32 input", &[rows, self.input])?;
        let output_len = checked_len("Gemma4 F32 output", &[rows, self.output])?;
        if rows == 0
            || input.len() != input_len
            || output.len() != output_len
            || input.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid Gemma4 F32 linear shape".into());
        }
        let output_ptr = SharedMut(output.as_mut_ptr());
        pool.compute(|thread, threads| {
            for index in (thread..output_len).step_by(threads) {
                let row = index / self.output;
                let column = index % self.output;
                let value = dot_f32(
                    &input[row * self.input..(row + 1) * self.input],
                    &self.weight[column * self.input..(column + 1) * self.input],
                    self.input,
                );
                unsafe { output_ptr.write(index, value) };
            }
        });
        validate_finite("Gemma4 F32 linear output", output)
    }
}

fn conv2d_stride2(
    pool: &ComputePool,
    input: &[f32],
    frequency: usize,
    time: usize,
    convolution: &FrontendConv<'_>,
) -> Result<(Vec<f32>, usize, usize), String> {
    let input_len = checked_len(
        "Gemma4 convolution input",
        &[time, frequency, convolution.input_channels],
    )?;
    let patch_len = checked_len(
        "Gemma4 convolution patch",
        &[convolution.input_channels, 3, 3],
    )?;
    if frequency == 0
        || time == 0
        || input.len() != input_len
        || convolution.weight.len() != patch_len * convolution.output_channels
        || convolution.norm.len() != convolution.output_channels
        || input.iter().any(|value| !value.is_finite())
    {
        return Err("Invalid Gemma4 convolution input".into());
    }
    let output_frequency = frequency
        .checked_add(1)
        .ok_or("Gemma4 convolution frequency overflow")?
        / 2;
    let output_time = time
        .checked_add(1)
        .ok_or("Gemma4 convolution time overflow")?
        / 2;
    let positions = checked_len(
        "Gemma4 convolution positions",
        &[output_time, output_frequency],
    )?;
    let mut patches = zeroed_u16(
        "Gemma4 convolution patches",
        checked_len("Gemma4 convolution patches", &[positions, patch_len])?,
    )?;
    for output_t in 0..output_time {
        for output_f in 0..output_frequency {
            let patch = &mut patches[(output_t * output_frequency + output_f) * patch_len
                ..(output_t * output_frequency + output_f + 1) * patch_len];
            for channel in 0..convolution.input_channels {
                for kernel_t in 0..3 {
                    let padded_t = output_t * 2 + kernel_t;
                    if padded_t == 0 || padded_t > time {
                        continue;
                    }
                    for kernel_f in 0..3 {
                        let padded_f = output_f * 2 + kernel_f;
                        if padded_f == 0 || padded_f > frequency {
                            continue;
                        }
                        let input_index = (((padded_t - 1) * frequency + padded_f - 1)
                            * convolution.input_channels)
                            + channel;
                        patch[(channel * 3 + kernel_t) * 3 + kernel_f] =
                            f32_to_f16(input[input_index]);
                    }
                }
            }
        }
    }
    let output_len = checked_len(
        "Gemma4 convolution output",
        &[positions, convolution.output_channels],
    )?;
    let mut output = zeroed_f32("Gemma4 convolution output", output_len)?;
    let output_ptr = SharedMut(output.as_mut_ptr());
    pool.compute(|thread, threads| {
        for index in (thread..output_len).step_by(threads) {
            let position = index / convolution.output_channels;
            let channel = index % convolution.output_channels;
            let value = dot_f16_f32(
                &convolution.weight[channel * patch_len..(channel + 1) * patch_len],
                &patches[position * patch_len..(position + 1) * patch_len],
                patch_len,
            );
            unsafe { output_ptr.write(index, value) };
        }
    });
    for row in output.chunks_exact_mut(convolution.output_channels) {
        layer_norm_row(row, convolution.norm)?;
        row.iter_mut().for_each(|value| *value = value.max(0.0));
    }
    validate_finite("Gemma4 convolution output", &output)?;
    Ok((output, output_frequency, output_time))
}

fn layer_norm_row(row: &mut [f32], weight: &[f32]) -> Result<(), String> {
    if row.is_empty() || row.len() != weight.len() {
        return Err("Invalid Gemma4 convolution norm shape".into());
    }
    let mean = row.iter().map(|value| f64::from(*value)).sum::<f64>() / row.len() as f64;
    let variance = row
        .iter()
        .map(|value| {
            let centered = f64::from(*value) - mean;
            centered * centered
        })
        .sum::<f64>()
        / row.len() as f64;
    let mean = mean as f32;
    let inverse = (1.0 / (variance + f64::from(EPS)).sqrt()) as f32;
    for (value, weight) in row.iter_mut().zip(weight) {
        *value = (*value - mean) * inverse * *weight;
    }
    validate_finite("Gemma4 convolution norm", row)
}

fn scale_queries_and_keys(q: &mut [f32], k: &mut [f32], scale: &[f32]) -> Result<(), String> {
    if q.len() != k.len() || q.len() % EMBED != 0 || scale.len() != HEAD_DIM {
        return Err("Invalid Gemma4 attention scaling shape".into());
    }
    let q_scale = (1.0 / (HEAD_DIM as f32).sqrt()) / 2.0f32.ln();
    let k_scale = (1.0 + 1.0f32.exp()).ln() / 2.0f32.ln();
    for (index, (query, key)) in q.iter_mut().zip(k.iter_mut()).enumerate() {
        *query *= q_scale;
        *query *= scale[index % HEAD_DIM];
        *key *= k_scale;
    }
    validate_finite("Gemma4 scaled query", q)?;
    validate_finite("Gemma4 scaled key", k)
}

fn local_attention(
    pool: &ComputePool,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    relative: &[f32],
    rows: usize,
    output: &mut [f32],
) -> Result<(), String> {
    let hidden_len = checked_len("Gemma4 local attention", &[rows, EMBED])?;
    if rows == 0
        || q.len() != hidden_len
        || k.len() != hidden_len
        || v.len() != hidden_len
        || output.len() != hidden_len
        || relative.len() != RPE_LEN * EMBED
    {
        return Err("Invalid Gemma4 local attention shape".into());
    }
    let output_ptr = SharedMut(output.as_mut_ptr());
    pool.compute(|thread, threads| {
        for task in (thread..rows * HEADS).step_by(threads) {
            let query = task / HEADS;
            let head = task % HEADS;
            let start = query.saturating_sub(LOCAL_CONTEXT - 1);
            let count = query - start + 1;
            let q_row = &q[query * EMBED + head * HEAD_DIM..query * EMBED + (head + 1) * HEAD_DIM];
            let mut scores = [0.0f32; LOCAL_CONTEXT];
            for (offset, key) in (start..=query).enumerate() {
                let distance = query - key;
                let relative_row = RPE_LEN - 1 - distance;
                let k_row = &k[key * EMBED + head * HEAD_DIM..key * EMBED + (head + 1) * HEAD_DIM];
                let r_row = &relative[relative_row * EMBED + head * HEAD_DIM
                    ..relative_row * EMBED + (head + 1) * HEAD_DIM];
                let score = dot_f32(q_row, k_row, HEAD_DIM) + dot_f32(q_row, r_row, HEAD_DIM);
                scores[offset] = (score / SOFTCAP).tanh() * SOFTCAP;
            }
            softmax_ggml_inplace(&mut scores[..count]);
            for dimension in 0..HEAD_DIM {
                let mut value = 0.0f32;
                for (offset, key) in (start..=query).enumerate() {
                    value += v[key * EMBED + head * HEAD_DIM + dimension] * scores[offset];
                }
                unsafe { output_ptr.write(query * EMBED + head * HEAD_DIM + dimension, value) };
            }
        }
    });
    validate_finite("Gemma4 local attention", output)
}

fn causal_depthwise_conv(
    pool: &ComputePool,
    input: &[f32],
    weight: &[f32],
    rows: usize,
    output: &mut [f32],
) -> Result<(), String> {
    let hidden_len = checked_len("Gemma4 depthwise convolution", &[rows, EMBED])?;
    if rows == 0
        || input.len() != hidden_len
        || output.len() != hidden_len
        || weight.len() != CONV_KERNEL * EMBED
    {
        return Err("Invalid Gemma4 depthwise convolution sequence".into());
    }
    let padded = rows
        .checked_add(CONV_KERNEL - 1)
        .ok_or("Gemma4 depthwise convolution sequence overflow")?;
    if padded < CONV_KERNEL {
        return Err("Invalid Gemma4 depthwise convolution sequence".into());
    }
    let output_ptr = SharedMut(output.as_mut_ptr());
    pool.compute(|thread, threads| {
        for index in (thread..hidden_len).step_by(threads) {
            let row = index / EMBED;
            let channel = index % EMBED;
            let mut sum = 0.0f32;
            for kernel in (CONV_KERNEL - 1).saturating_sub(row)..CONV_KERNEL {
                let input_row = row + kernel - (CONV_KERNEL - 1);
                sum += input[input_row * EMBED + channel] * weight[channel * CONV_KERNEL + kernel];
            }
            unsafe { output_ptr.write(index, sum) };
        }
    });
    validate_finite("Gemma4 depthwise convolution", output)
}

fn sinusoidal_positions() -> Result<Vec<f32>, String> {
    let mut positions = zeroed_f32("Gemma4 relative positions", RPE_LEN * EMBED)?;
    let timescales = EMBED / 2;
    let increment = 10_000.0f32.ln() / (timescales - 1) as f32;
    for position in 0..RPE_LEN {
        let value = (RPE_LEN - 1 - position) as f32;
        for scale in 0..timescales {
            let angle = value * (-(scale as f32) * increment).exp();
            positions[position * EMBED + scale] = angle.sin();
            positions[position * EMBED + timescales + scale] = angle.cos();
        }
    }
    validate_finite("Gemma4 relative positions", &positions)?;
    Ok(positions)
}

fn weighted_rms_rows(values: &mut [f32], weight: &[f32], width: usize) -> Result<(), String> {
    if width == 0 || values.len() % width != 0 || weight.len() != width {
        return Err("Invalid Gemma4 weighted RMS shape".into());
    }
    for row in values.chunks_exact_mut(width) {
        rms_norm_inplace(row, weight, EPS);
    }
    validate_finite("Gemma4 weighted RMS", values)
}

fn unweighted_rms_rows(values: &mut [f32], width: usize) -> Result<(), String> {
    if width == 0 || values.len() % width != 0 {
        return Err("Invalid Gemma4 unweighted RMS shape".into());
    }
    for row in values.chunks_exact_mut(width) {
        let mean = (sum_sq_f32(row) / width as f64) as f32;
        let scale = 1.0 / (mean + EPS).sqrt();
        row.iter_mut().for_each(|value| *value *= scale);
    }
    validate_finite("Gemma4 unweighted RMS", values)
}

fn add_scaled(target: &mut [f32], source: &[f32], scale: f32) -> Result<(), String> {
    if target.len() != source.len() || !scale.is_finite() {
        return Err("Invalid Gemma4 residual shape".into());
    }
    for (target, source) in target.iter_mut().zip(source) {
        *target += *source * scale;
    }
    validate_finite("Gemma4 residual", target)
}

fn validate_projected_rows(values: &[f32]) -> Result<(), String> {
    if values.is_empty()
        || values.len() % PROJECTION != 0
        || values.iter().any(|value| !value.is_finite())
    {
        return Err("Invalid Gemma4 projected audio rows".into());
    }
    Ok(())
}

fn f16_tensor<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<&'a [u8], String> {
    let bytes = checked_tensor(source, name, dims, GGMLType::F16)?;
    if bytes.chunks_exact(2).any(|chunk| {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        bits & 0x7c00 == 0x7c00
    }) {
        return Err(format!("Non-finite F16 tensor: {name}"));
    }
    Ok(bytes)
}

fn f32_tensor<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<&'a [f32], String> {
    if cfg!(target_endian = "big") {
        return Err("Gemma4 borrowed F32 tensors require little-endian input".into());
    }
    let bytes = checked_tensor(source, name, dims, GGMLType::F32)?;
    if bytes.as_ptr().align_offset(std::mem::align_of::<f32>()) != 0 {
        return Err(format!("Unaligned F32 tensor: {name}"));
    }
    let values =
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("Non-finite F32 tensor: {name}"));
    }
    Ok(values)
}

fn f32_scalar(source: &dyn TensorSource, name: &str) -> Result<f32, String> {
    Ok(f32_tensor(source, name, &[1])?[0])
}

fn checked_tensor<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    dims: &[u64],
    kind: GGMLType,
) -> Result<&'a [u8], String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if dims.is_empty() || dims.contains(&0) || info.dims != dims || info.ggml_type != kind {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} {:?}",
            info.dims, info.ggml_type, dims, kind
        ));
    }
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?,
    )
    .map_err(|_| format!("Tensor byte size does not fit usize: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn checked_len(label: &str, factors: &[usize]) -> Result<usize, String> {
    factors.iter().try_fold(1usize, |length, factor| {
        length
            .checked_mul(*factor)
            .ok_or_else(|| format!("{label} length overflow"))
    })
}

fn zeroed_f32(label: &str, len: usize) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| format!("{label} allocation failed"))?;
    values.resize(len, 0.0);
    Ok(values)
}

fn zeroed_u16(label: &str, len: usize) -> Result<Vec<u16>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| format!("{label} allocation failed"))?;
    values.resize(len, f32_to_f16(0.0));
    Ok(values)
}

fn resize_u16(values: &mut Vec<u16>, len: usize, label: &str) -> Result<(), String> {
    if values.len() < len {
        values
            .try_reserve_exact(len - values.len())
            .map_err(|_| format!("{label} allocation failed"))?;
        values.resize(len, 0);
    } else {
        values.truncate(len);
    }
    Ok(())
}

fn validate_finite(label: &str, values: &[f32]) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("Non-finite {label}"));
    }
    Ok(())
}

pub fn gemma4_audio_features(bytes: &[u8]) -> Result<Gemma4AudioFeatures, String> {
    let decoded = decode_pcm16_wav_any(bytes).map_err(|error| format!("{error:?}"))?;
    let channels = usize::from(decoded.channels);
    let mut mono = Vec::new();
    mono.try_reserve_exact(decoded.samples.len() / channels)
        .map_err(|_| "Gemma4 audio allocation failed")?;
    for frame in decoded.samples.chunks_exact(channels) {
        let sample = frame.iter().map(|value| f64::from(*value)).sum::<f64>() / channels as f64;
        if !sample.is_finite() {
            return Err("non-finite mixed audio sample".into());
        }
        mono.push(sample as f32);
    }
    let samples = resample_linear(&mono, decoded.sample_rate)?;
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("non-finite resampled audio sample".into());
    }

    let mut chunks = Vec::new();
    let mut frames = 0usize;
    for chunk in samples.chunks(CHUNK_SAMPLES) {
        let (values, chunk_frames) = gemma4_chunk_mel(chunk)?;
        frames = frames
            .checked_add(chunk_frames)
            .ok_or("Gemma4 audio frame count overflow")?;
        chunks.push((values, chunk_frames));
    }
    if frames == 0 {
        return Err("Gemma4 audio produced no frames".into());
    }
    let mut values = zeroed(
        MEL_BINS
            .checked_mul(frames)
            .ok_or("Gemma4 Mel size overflow")?,
    )?;
    let mut frame_offset = 0usize;
    for (chunk, chunk_frames) in chunks {
        for mel in 0..MEL_BINS {
            let source = &chunk[mel * chunk_frames..(mel + 1) * chunk_frames];
            let destination = &mut values
                [mel * frames + frame_offset..mel * frames + frame_offset + chunk_frames];
            destination.copy_from_slice(source);
        }
        frame_offset += chunk_frames;
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("non-finite Gemma4 Mel value".into());
    }

    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(
        "gemma4.audio.mel",
        None,
        &[MEL_BINS, frames],
        &values,
    ));
    Ok(Gemma4AudioFeatures { values, frames })
}

fn resample_linear(samples: &[f32], source_rate: u32) -> Result<Vec<f32>, String> {
    if samples.is_empty() || source_rate == 0 {
        return Err("invalid Gemma4 audio sample rate or data".into());
    }
    if source_rate == SAMPLE_RATE {
        return Ok(samples.to_vec());
    }
    let input_last = u64::try_from(samples.len() - 1).map_err(|_| "Gemma4 audio is too large")?;
    let output_last = input_last
        .checked_mul(u64::from(SAMPLE_RATE))
        .ok_or("Gemma4 resampled length overflow")?
        / u64::from(source_rate);
    let output_len = usize::try_from(
        output_last
            .checked_add(1)
            .ok_or("Gemma4 resampled length overflow")?,
    )
    .map_err(|_| "Gemma4 resampled length overflow")?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| "Gemma4 resample allocation failed")?;
    for output_index in 0..output_len {
        let position = u64::try_from(output_index)
            .map_err(|_| "Gemma4 resample position overflow")?
            .checked_mul(u64::from(source_rate))
            .ok_or("Gemma4 resample position overflow")?;
        let input_index = usize::try_from(position / u64::from(SAMPLE_RATE))
            .map_err(|_| "Gemma4 resample index overflow")?;
        let current = *samples
            .get(input_index)
            .ok_or("Gemma4 resample exceeded input")?;
        let next = samples.get(input_index + 1).copied().unwrap_or(current);
        let fraction = (position % u64::from(SAMPLE_RATE)) as f32 / SAMPLE_RATE as f32;
        let value = current + (next - current) * fraction;
        if !value.is_finite() {
            return Err("non-finite resampled audio sample".into());
        }
        output.push(value);
    }
    Ok(output)
}

fn gemma4_chunk_mel(samples: &[f32]) -> Result<(Vec<f32>, usize), String> {
    let left_pad = WINDOW_SIZE / 2;
    let with_left = i64::try_from(samples.len())
        .map_err(|_| "Gemma4 audio chunk is too large")?
        .checked_add(i64::try_from(left_pad).unwrap())
        .ok_or("Gemma4 audio frame count overflow")?;
    let frames =
        (with_left - i64::try_from(WINDOW_SIZE + 1).unwrap()) / i64::try_from(HOP).unwrap() + 1;
    let frames = usize::try_from(frames).map_err(|_| "Gemma4 audio is too short for a frame")?;
    if frames == 0 {
        return Err("Gemma4 audio is too short for a frame".into());
    }
    let padded_len = frames
        .checked_sub(1)
        .and_then(|value| value.checked_mul(HOP))
        .and_then(|value| value.checked_add(FFT_SIZE))
        .ok_or("Gemma4 padding size overflow")?;
    let total_pad = padded_len
        .checked_sub(samples.len())
        .ok_or("Gemma4 padding size overflow")?
        .max(left_pad);
    let mut padded = zeroed(
        total_pad
            .checked_add(samples.len())
            .ok_or("Gemma4 padding size overflow")?,
    )?;
    padded[left_pad..left_pad + samples.len()].copy_from_slice(samples);
    let hann = periodic_hann_window();
    let filters = htk_mel_filters()?;
    let fft_bins = FFT_SIZE / 2 + 1;
    let mut values = zeroed(
        MEL_BINS
            .checked_mul(frames)
            .ok_or("Gemma4 Mel size overflow")?,
    )?;
    let mut input = zeroed(FFT_SIZE)?;
    let mut magnitude = zeroed(fft_bins)?;
    let mut fft = RealFft::new(FFT_SIZE).map_err(|error| format!("{error:?}"))?;
    for frame in 0..frames {
        let start = frame.checked_mul(HOP).ok_or("Gemma4 FFT offset overflow")?;
        let source = padded
            .get(start..start + FFT_SIZE)
            .ok_or("Gemma4 padded FFT frame is truncated")?;
        for index in 0..FFT_SIZE {
            input[index] = source[index] * hann[index];
        }
        fft.magnitude(&input, &mut magnitude);
        if magnitude.iter().any(|value| !value.is_finite()) {
            return Err("non-finite Gemma4 FFT output".into());
        }
        for mel in 0..MEL_BINS {
            let filter = &filters[mel * fft_bins..(mel + 1) * fft_bins];
            let mut sum = 0.0f64;
            let mut bin = 0usize;
            while bin + 3 < fft_bins {
                sum += f64::from(
                    magnitude[bin] * filter[bin]
                        + magnitude[bin + 1] * filter[bin + 1]
                        + magnitude[bin + 2] * filter[bin + 2]
                        + magnitude[bin + 3] * filter[bin + 3],
                );
                bin += 4;
            }
            while bin < fft_bins {
                sum += f64::from(magnitude[bin] * filter[bin]);
                bin += 1;
            }
            let value = sum.max(0.001).ln() as f32;
            if !value.is_finite() {
                return Err("non-finite Gemma4 Mel value".into());
            }
            values[mel * frames + frame] = value;
        }
    }
    Ok((values, frames))
}

fn periodic_hann_window() -> Vec<f32> {
    (0..FFT_SIZE)
        .map(|index| {
            if index < WINDOW_SIZE {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / WINDOW_SIZE as f32).cos()
            } else {
                0.0
            }
        })
        .collect()
}

fn htk_mel_filters() -> Result<Vec<f32>, String> {
    let fft_bins = FFT_SIZE / 2 + 1;
    let max_mel = htk_hz_to_mel(SAMPLE_RATE as f64 / 2.0);
    let points: Vec<f64> = (0..MEL_BINS + 2)
        .map(|index| htk_mel_to_hz(max_mel * index as f64 / (MEL_BINS + 1) as f64))
        .collect();
    let mut filters = zeroed(
        MEL_BINS
            .checked_mul(fft_bins)
            .ok_or("Gemma4 Mel filter size overflow")?,
    )?;
    for mel in 0..MEL_BINS {
        let left = points[mel];
        let center = points[mel + 1];
        let right = points[mel + 2];
        for bin in 0..fft_bins {
            let hz = bin as f64 * SAMPLE_RATE as f64 / FFT_SIZE as f64;
            let weight = if hz >= left && hz <= center {
                (hz - left) / (center - left).max(1e-30)
            } else if hz > center && hz <= right {
                (right - hz) / (right - center).max(1e-30)
            } else {
                0.0
            };
            if !weight.is_finite() {
                return Err("non-finite Gemma4 Mel filter".into());
            }
            filters[mel * fft_bins + bin] = weight as f32;
        }
    }
    Ok(filters)
}

fn htk_hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn htk_mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

fn zeroed(len: usize) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| "Gemma4 audio allocation failed")?;
    values.resize(len, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_sized_test_audio_model() -> Gemma4AudioModel<'static> {
        let empty_f16 = |input, output| F16Linear {
            weight: &[],
            input,
            output,
            clamp: None,
        };
        Gemma4AudioModel {
            config: Gemma4AudioConfig {
                layers: LAYERS,
                embd: EMBED,
                heads: HEADS,
                mel_bins: MEL_BINS,
                projection: PROJECTION,
            },
            pool: ComputePool::new(1),
            conv0: FrontendConv {
                weight: &[],
                norm: &[],
                input_channels: 1,
                output_channels: 128,
            },
            conv1: FrontendConv {
                weight: &[],
                norm: &[],
                input_channels: 128,
                output_channels: 32,
            },
            input_projection: F32Linear {
                weight: &[],
                input: EMBED,
                output: EMBED,
            },
            layers: Vec::new(),
            output_projection: empty_f16(EMBED, PROJECTION),
            output_bias: &[],
            multimodal_projection: empty_f16(PROJECTION, PROJECTION),
        }
    }

    fn pcm16_wav(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data_len = u32::try_from(samples.len() * 2).unwrap();
        let block_align = channels * 2;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn gemma4_frontend_returns_one_finite_128_bin_frame_for_20ms_stereo() {
        let samples: Vec<i16> = (0..960)
            .flat_map(|i| {
                let sample = (i as f32 * 0.03125).sin() * 16_384.0;
                [sample as i16, sample as i16]
            })
            .collect();
        let mel = gemma4_audio_features(&pcm16_wav(48_000, 2, &samples)).unwrap();
        assert_eq!(mel.frames, 1);
        assert_eq!(mel.values.len(), 128);
        assert!(mel.values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn frontend_convolution_preserves_f32_weights() {
        let mut weight = vec![0.0; 3 * 3 * 3];
        weight[4] = 1.0006;
        weight[9 + 4] = 0.2003;
        weight[18 + 4] = -0.1003;
        let convolution = FrontendConv {
            weight: &weight,
            norm: &[1.0; 3],
            input_channels: 1,
            output_channels: 3,
        };
        let (output, frequency, time) =
            conv2d_stride2(&ComputePool::new(1), &[1.0], 1, 1, &convolution).unwrap();

        assert_eq!((frequency, time), (1, 1));
        assert_eq!(output[0].to_bits(), 0x3fae_9724);
        assert_eq!(&output[1..], &[0.0, 0.0]);
    }

    #[test]
    fn conformer_rejects_wrong_mel_width_and_nonfinite_values() {
        let model = zero_sized_test_audio_model();
        assert!(model
            .encode_features(&Gemma4AudioFeatures {
                values: vec![0.0; 127],
                frames: 1,
            })
            .is_err());
        assert!(model
            .encode_features(&Gemma4AudioFeatures {
                values: vec![f32::NAN; 128],
                frames: 1,
            })
            .is_err());
    }

    #[test]
    fn projected_audio_requires_complete_1536_rows() {
        assert!(validate_projected_rows(&vec![0.0; 1535]).is_err());
        assert!(validate_projected_rows(&vec![0.0; 1536]).is_ok());
    }
}
