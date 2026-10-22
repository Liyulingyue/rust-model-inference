//! Audio forward path for EmbeddingGemma-2 (Conformer-style, 12 layers).
//!
//! Architecture (per HF safetensors inspection):
//! - input: 16 kHz mono audio, log-mel spectrogram
//! - subsample_conv_projection: Conv2d (1, 128, 3, 3) → Conv2d (128, 32, 3, 3) →
//!   BatchNorm → Linear (32 → 1024 = hidden_size)
//! - 12 × Conformer layer:
//!   - lconv1d: depthwise Conv1d (1024, 1, kernel=5) + linear_start (2048 → 1024)
//!     + linear_end (1024 → 1024) + conv_norm
//!   - self_attn: q/k/v (each 1024 → 1024) + post (1024 → 1024) +
//!     per_dim_scale (128,) + relative_k_proj (1024 → 1024)
//!   - feed_forward1: pre_layer_norm + ffw_layer_1 (1024 → 4096) + ffw_layer_2 (4096 → 1024)
//!     + post_layer_norm
//!   - feed_forward2: same structure as ffn1
//!   - norm_pre_attn, norm_post_attn, norm_out: layer norms
//! - output_proj: 1024 → 1536 (projection_dim)
//!
//! Forward flow: audio → log-mel → subsample → 12 Conformer layers → output_proj.

use crate::core::tensor::TensorSource;
use std::sync::Arc;

pub const RMS_NORM_EPS: f32 = 1e-6;
pub const SAMPLE_RATE: usize = 16000;
pub const MAX_AUDIO_SECONDS: usize = 30;
pub const ENCODER_FRAME_MS: usize = 80;

#[derive(Debug, Clone)]
pub struct EmbeddingGemmaAudioConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub projection_dim: usize,
}

impl EmbeddingGemmaAudioConfig {
    pub fn from_metadata<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let get = |key: &str| -> Result<usize, String> {
            source
                .metadata(key)
                .and_then(|v| v.to_u64())
                .and_then(|v| v.try_into().ok())
                .ok_or_else(|| format!("Missing metadata: {key}"))
        };
        let depth = get("gemma4-embedding.audio.depth")?;
        let hidden_size = get("gemma4-embedding.audio.hidden_size")?;
        let ffn_hidden_size = get("gemma4-embedding.audio.ffn_hidden_size")?;
        let projection_dim = get("gemma4-embedding.audio.projection_dim")?;
        Ok(Self {
            depth,
            hidden_size,
            ffn_hidden_size,
            projection_dim,
        })
    }
}

fn decode_f32(data: &[u8], expected: usize) -> Result<Vec<f32>, String> {
    let n = data.len() / 4;
    if n != expected {
        return Err(format!(
            "f32 payload length mismatch: {} != {expected}",
            data.len() / 4
        ));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u32::from_le_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]);
        out.push(f32::from_bits(bits));
    }
    Ok(out)
}

fn dequant_bf16_to_f32(raw: &[u8], n: usize) -> Vec<f32> {
    // The converter writes PyTorch [n_out, n_in] bytes unchanged; GGUF dims
    // describe the reversed storage axes but do not reorder the payload.
    let mut out = Vec::with_capacity(n);
    for idx in 0..n {
        let bits = u16::from_le_bytes([raw[idx * 2], raw[idx * 2 + 1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    out
}

/// 1-D BF16 dequant (for RMSNorm/affine weight tensors).
fn dequant_bf16_to_f32_norm(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    out
}

/// N-D BF16 dequant with row-major layout preserved (for conv weights and per-tensor scalars).
fn dequant_bf16_to_f32_flat(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    out
}

#[derive(Clone, Copy)]
struct LinearClamp {
    input_min: f32,
    input_max: f32,
    output_min: f32,
    output_max: f32,
}

struct Linear {
    matrix: Vec<f32>,
    n_in: usize,
    n_out: usize,
    clamp: Option<LinearClamp>,
}

impl Linear {
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.n_in);
        assert_eq!(output.len(), self.n_out);
        for i in 0..self.n_out {
            let row = &self.matrix[i * self.n_in..(i + 1) * self.n_in];
            let mut s = 0.0f32;
            for k in 0..self.n_in {
                let value = self
                    .clamp
                    .map(|clamp| input[k].clamp(clamp.input_min, clamp.input_max))
                    .unwrap_or(input[k]);
                s += row[k] * value;
            }
            output[i] = self
                .clamp
                .map(|clamp| s.clamp(clamp.output_min, clamp.output_max))
                .unwrap_or(s);
        }
    }
}

fn linear_from_gguf_bf16(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Linear, String> {
    let raw = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor: {name}"));
    Ok(Linear {
        matrix: dequant_bf16_to_f32(raw?, n_in * n_out),
        n_in,
        n_out,
        clamp: None,
    })
}

fn bf16_scalar(source: &dyn TensorSource, name: &str) -> Result<f32, String> {
    let raw = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let values = dequant_bf16_to_f32_norm(raw, 1);
    Ok(values[0])
}

fn linear_from_gguf_bf16_clipped(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
    prefix: &str,
) -> Result<Linear, String> {
    let mut linear = linear_from_gguf_bf16(source, name, n_in, n_out)?;
    linear.clamp = Some(LinearClamp {
        input_min: bf16_scalar(source, &format!("{prefix}.input_min"))?,
        input_max: bf16_scalar(source, &format!("{prefix}.input_max"))?,
        output_min: bf16_scalar(source, &format!("{prefix}.output_min"))?,
        output_max: bf16_scalar(source, &format!("{prefix}.output_max"))?,
    });
    Ok(linear)
}

/// Compute log-mel spectrogram features for 16 kHz mono audio.
/// Returns shape `[n_frames, 128]` (128 mel bins).
///
/// The Gemma4 feature extractor uses 20 ms windows with a 10 ms hop, 128 HTK
/// mel bins spanning 0–8000 Hz, and applies a periodic Hann window followed by
/// magnitude, mel projection, and log compression.
fn log_mel_features(pcm16: &[i16]) -> Vec<f32> {
    // Gemma4AudioFeatureExtractor: semicausal 160-sample padding, 320-sample
    // periodic-Hann frames, 512-point RFFT, HTK mel bank, log(mel + 1e-3).
    const FRAME: usize = 320;
    const HOP: usize = 160;
    const FFT: usize = 512;
    const MELS: usize = 128;
    if pcm16.is_empty() {
        return vec![];
    }
    let pad_left = FRAME / 2;
    let padded_len = pcm16.len() + pad_left;
    let n_frames = padded_len.saturating_sub(FRAME + 1) / HOP + 1;
    if n_frames == 0 {
        return vec![];
    }

    // The extractor applies its f32 waveform/window product before NumPy's
    // f64 FFT and mel projection, then casts to f32 after the log. Preserve
    // that boundary so the model sees the same input values.
    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0);
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel((SAMPLE_RATE / 2) as f64);
    let filter_freqs: Vec<f64> = (0..MELS + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (MELS + 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..=FFT / 2)
        .map(|i| SAMPLE_RATE as f64 * i as f64 / FFT as f64)
        .collect();
    let mut filters = vec![0.0f64; (FFT / 2 + 1) * MELS];
    for k in 0..=FFT / 2 {
        for m in 0..MELS {
            let rising = (fft_freqs[k] - filter_freqs[m]) / (filter_freqs[m + 1] - filter_freqs[m]);
            let falling =
                (filter_freqs[m + 2] - fft_freqs[k]) / (filter_freqs[m + 2] - filter_freqs[m + 1]);
            filters[k * MELS + m] = rising.min(falling).max(0.0);
        }
    }

    let mut features = vec![0.0f32; n_frames * MELS];
    let mut frame = vec![0.0f64; FRAME];
    let mut power = vec![0.0f64; FFT / 2 + 1];
    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP;
        for i in 0..FRAME {
            let sample_idx = start + i;
            let sample = if sample_idx < pad_left {
                0.0
            } else if sample_idx - pad_left < pcm16.len() {
                pcm16[sample_idx - pad_left] as f32 / 32768.0
            } else {
                0.0
            };
            // NumPy builds the periodic Hann window in f64, then stores it as
            // f32 before multiplying the f32 waveform.
            let window = (0.5f64
                - 0.5f64 * (2.0f64 * std::f64::consts::PI * i as f64 / FRAME as f64).cos())
                as f32;
            frame[i] = f64::from(sample * window);
        }
        rfft_magnitude(&frame, &mut power);
        for m in 0..MELS {
            let mut value = 0.0f64;
            for k in 0..=FFT / 2 {
                value += power[k] * filters[k * MELS + m];
            }
            features[frame_idx * MELS + m] = (value + 1.0e-3).ln() as f32;
        }
    }
    features
}

fn rms_norm_rows(x: &mut [f32], row_len: usize, weight: &[f32]) {
    assert_eq!(weight.len(), row_len);
    for row in x.chunks_exact_mut(row_len) {
        crate::ops::norm::rms_norm_inplace(row, weight, RMS_NORM_EPS);
    }
}

fn layer_norm_channels(x: &mut [f32], frames: usize, bins: usize, channels: usize, weight: &[f32]) {
    assert_eq!(x.len(), frames * bins * channels);
    assert_eq!(weight.len(), channels);
    for frame in 0..frames {
        for bin in 0..bins {
            let offset = (frame * bins + bin) * channels;
            let row = &mut x[offset..offset + channels];
            let mean = row.iter().copied().sum::<f32>() / channels as f32;
            let variance = row
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f32>()
                / channels as f32;
            let scale = (variance + RMS_NORM_EPS).sqrt().recip();
            for ch in 0..channels {
                row[ch] = (row[ch] - mean) * scale * weight[ch];
            }
        }
    }
}

fn rfft_magnitude(frame: &[f64], output: &mut [f64]) {
    let n = output.len().saturating_sub(1) * 2;
    debug_assert!(frame.len() <= n && n.is_power_of_two());
    let mut real = vec![0.0f64; n];
    real[..frame.len()].copy_from_slice(frame);
    let mut imag = vec![0.0f64; n];

    let mut reversed = 0usize;
    for index in 1..n {
        let mut bit = n >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            real.swap(index, reversed);
        }
    }

    let mut width = 2usize;
    while width <= n {
        let half = width / 2;
        for start in (0..n).step_by(width) {
            for offset in 0..half {
                let angle = -2.0 * std::f64::consts::PI * offset as f64 / width as f64;
                let wr = angle.cos();
                let wi = angle.sin();
                let even = start + offset;
                let odd = even + half;
                let tr = wr * real[odd] - wi * imag[odd];
                let ti = wr * imag[odd] + wi * real[odd];
                let ur = real[even];
                let ui = imag[even];
                real[even] = ur + tr;
                imag[even] = ui + ti;
                real[odd] = ur - tr;
                imag[odd] = ui - ti;
            }
        }
        width <<= 1;
    }

    for (index, value) in output.iter_mut().enumerate() {
        *value = real[index]
            .mul_add(real[index], imag[index] * imag[index])
            .sqrt();
    }
}

fn conv2d_stride2(
    input: &[f32],
    in_frames: usize,
    in_bins: usize,
    in_channels: usize,
    weight: &[f32],
    out_channels: usize,
) -> (Vec<f32>, usize, usize) {
    let out_frames = (in_frames + 1) / 2;
    let out_bins = (in_bins + 1) / 2;
    let mut output = vec![0.0f32; out_frames * out_bins * out_channels];
    for ot in 0..out_frames {
        for ob in 0..out_bins {
            for oc in 0..out_channels {
                let mut sum = 0.0f32;
                for kh in 0..3 {
                    let it = ot * 2 + kh;
                    if it == 0 || it > in_frames {
                        continue;
                    }
                    let it = it - 1;
                    for kw in 0..3 {
                        let ib = ob * 2 + kw;
                        if ib == 0 || ib > in_bins {
                            continue;
                        }
                        let ib = ib - 1;
                        for ic in 0..in_channels {
                            let input_idx = (it * in_bins + ib) * in_channels + ic;
                            let weight_idx = ((oc * in_channels + ic) * 3 + kh) * 3 + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
                output[(ot * out_bins + ob) * out_channels + oc] = sum;
            }
        }
    }
    (output, out_frames, out_bins)
}

fn feed_forward(
    input: &[f32],
    hidden_size: usize,
    ffn_hidden_size: usize,
    pre_norm: &[f32],
    ffw1: &Linear,
    ffw2: &Linear,
    post_norm: &[f32],
) -> Vec<f32> {
    let mut output = input.to_vec();
    let mut normalized = vec![0.0f32; hidden_size];
    let mut intermediate = vec![0.0f32; ffn_hidden_size];
    let mut projected = vec![0.0f32; hidden_size];
    for (row_idx, row) in input.chunks_exact(hidden_size).enumerate() {
        normalized.copy_from_slice(row);
        crate::ops::norm::rms_norm_inplace(&mut normalized, pre_norm, RMS_NORM_EPS);
        ffw1.forward(&normalized, &mut intermediate);
        for value in &mut intermediate {
            *value = crate::ops::silu(*value);
        }
        ffw2.forward(&intermediate, &mut projected);
        crate::ops::norm::rms_norm_inplace(&mut projected, post_norm, RMS_NORM_EPS);
        let offset = row_idx * hidden_size;
        for j in 0..hidden_size {
            output[offset + j] = row[j] + 0.5 * projected[j];
        }
    }
    output
}

struct ConformerLayer {
    lconv1d_conv_norm: Vec<f32>,
    /// `[hidden, 1, 5]` Conv1d weight (stored as BF16).
    lconv1d_depthwise_weight: Vec<f32>,
    lconv1d_pre_layer_norm: Vec<f32>,
    lconv1d_linear_start: Linear,
    lconv1d_linear_end: Linear,
    norm_pre_attn: Vec<f32>,
    norm_post_attn: Vec<f32>,
    norm_out: Vec<f32>,
    self_attn_q_proj: Linear,
    self_attn_k_proj: Linear,
    self_attn_v_proj: Linear,
    self_attn_post: Linear,
    self_attn_relative_k_proj: Linear,
    /// 128-element per-dim scaling for Q·K attention scores.
    self_attn_per_dim_scale: Vec<f32>,
    ff1_pre_layer_norm: Vec<f32>,
    ff1_ffw_layer_1: Linear,
    ff1_ffw_layer_2: Linear,
    ff1_post_layer_norm: Vec<f32>,
    ff2_pre_layer_norm: Vec<f32>,
    ff2_ffw_layer_1: Linear,
    ff2_ffw_layer_2: Linear,
    ff2_post_layer_norm: Vec<f32>,
}

pub struct EmbeddingGemmaAudioModel {
    pub config: EmbeddingGemmaAudioConfig,
    pub(super) source: Arc<dyn TensorSource>,
    pub(super) subsample_input_proj_linear: Linear,
    /// Subsample conv2d: `[128, 1, 3, 3]` (in=1, out=128, kernel=3, stride=2).
    pub(super) subsample_layer0_conv: Vec<f32>,
    pub(super) subsample_layer0_norm: Vec<f32>,
    pub(super) subsample_layer1_conv: Vec<f32>,
    pub(super) subsample_layer1_norm: Vec<f32>,
    pub(super) output_proj: Linear,
    pub(super) output_proj_bias: Vec<f32>,
    pub(super) audio_embed: Linear,
    pub(super) layers: Vec<ConformerLayer>,
}

#[cfg(test)]
mod tests {
    use super::{dequant_bf16_to_f32, rfft_magnitude};

    #[test]
    fn linear_bf16_payload_keeps_pytorch_row_order() {
        // The converter writes a PyTorch [out, in] tensor unchanged; GGUF
        // dims are reversed metadata only. A 2x3 matrix must therefore stay
        // [1, 2, 3, 4, 5, 6] when decoded for Linear::forward.
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let raw: Vec<u8> = values
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        assert_eq!(dequant_bf16_to_f32(&raw, 6), values);
    }

    #[test]
    fn rfft_magnitude_zero_pads_and_preserves_impulse() {
        let mut frame = vec![0.0f64; 320];
        frame[0] = 1.0;
        let mut spectrum = vec![0.0f64; 257];
        rfft_magnitude(&frame, &mut spectrum);
        assert!(spectrum.iter().all(|value| (*value - 1.0).abs() < 1e-12));
    }
}

impl EmbeddingGemmaAudioModel {
    pub fn from_source(source: &Arc<dyn TensorSource>) -> Result<Self, String> {
        let config = EmbeddingGemmaAudioConfig::from_metadata(source.as_ref())?;

        let subsample_input_proj_linear = linear_from_gguf_bf16(
            source.as_ref(),
            "audio.subsample_conv_projection.input_proj_linear.weight",
            (128 / 4) * 32,
            config.hidden_size,
        )?;

        let layer0_conv = dequant_bf16_to_f32_flat(
            source
                .tensor_slice("audio.subsample_conv_projection.layer0.conv.weight")
                .ok_or("Missing audio.subsample_conv_projection.layer0.conv.weight")?,
            128 * 1 * 3 * 3,
        );
        let layer0_norm = dequant_bf16_to_f32_norm(
            source
                .tensor_slice("audio.subsample_conv_projection.layer0.norm.weight")
                .ok_or("Missing audio.subsample_conv_projection.layer0.norm.weight")?,
            128,
        );
        let layer1_conv = dequant_bf16_to_f32_flat(
            source
                .tensor_slice("audio.subsample_conv_projection.layer1.conv.weight")
                .ok_or("Missing audio.subsample_conv_projection.layer1.conv.weight")?,
            32 * 128 * 3 * 3,
        );
        let layer1_norm = dequant_bf16_to_f32_norm(
            source
                .tensor_slice("audio.subsample_conv_projection.layer1.norm.weight")
                .ok_or("Missing audio.subsample_conv_projection.layer1.norm.weight")?,
            32,
        );

        let output_proj = linear_from_gguf_bf16(
            source.as_ref(),
            "audio.output_proj.weight",
            config.hidden_size,
            config.projection_dim,
        )?;
        let output_proj_bias = dequant_bf16_to_f32_norm(
            source
                .tensor_slice("audio.output_proj.bias")
                .ok_or("Missing audio.output_proj.bias")?,
            config.projection_dim,
        );
        let text_dim = source
            .metadata("gemma4-embedding.embedding_length")
            .and_then(|value| value.to_u64())
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("Missing metadata: gemma4-embedding.embedding_length")?;
        let audio_embed = linear_from_gguf_bf16(
            source.as_ref(),
            "audio_embed.weight",
            config.projection_dim,
            text_dim,
        )?;

        let mut layers = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let blk = |suffix: &str| format!("audio.layers.{i}.{suffix}");
            let lconv1d_conv_norm = dequant_bf16_to_f32_norm(
                source
                    .tensor_slice(&blk("lconv1d.conv_norm.weight"))
                    .ok_or_else(|| format!("Missing {}.weight", blk("lconv1d.conv_norm")))?,
                config.hidden_size,
            );
            // Conv1d stores depthwise weights as [channels, 1, kernel].
            let dw_raw = source
                .tensor_slice(&blk("lconv1d.depthwise_conv1d.weight"))
                .ok_or_else(|| format!("Missing {}.weight", blk("lconv1d.depthwise_conv1d")))?;
            let n_dw = 5 * 1 * config.hidden_size;
            let lconv1d_depthwise_weight = dequant_bf16_to_f32_flat(dw_raw, n_dw);
            let lconv1d_pre_layer_norm = dequant_bf16_to_f32_norm(
                source
                    .tensor_slice(&blk("lconv1d.pre_layer_norm.weight"))
                    .ok_or_else(|| format!("Missing {}.weight", blk("lconv1d.pre_layer_norm")))?,
                config.hidden_size,
            );
            let lconv1d_linear_start = linear_from_gguf_bf16_clipped(
                source.as_ref(),
                &blk("lconv1d.linear_start.linear.weight"),
                config.hidden_size,
                2 * config.hidden_size,
                &blk("lconv1d.linear_start"),
            )?;
            let lconv1d_linear_end = linear_from_gguf_bf16_clipped(
                source.as_ref(),
                &blk("lconv1d.linear_end.linear.weight"),
                config.hidden_size,
                config.hidden_size,
                &blk("lconv1d.linear_end"),
            )?;

            let norm_pre_attn = dequant_bf16_to_f32_norm(
                source
                    .tensor_slice(&blk("norm_pre_attn.weight"))
                    .ok_or_else(|| format!("Missing {}.weight", blk("norm_pre_attn")))?,
                config.hidden_size,
            );
            let norm_post_attn = dequant_bf16_to_f32_norm(
                source
                    .tensor_slice(&blk("norm_post_attn.weight"))
                    .ok_or_else(|| format!("Missing {}.weight", blk("norm_post_attn")))?,
                config.hidden_size,
            );
            let norm_out = dequant_bf16_to_f32_norm(
                source
                    .tensor_slice(&blk("norm_out.weight"))
                    .ok_or_else(|| format!("Missing {}.weight", blk("norm_out")))?,
                config.hidden_size,
            );

            let self_attn_q_proj = linear_from_gguf_bf16_clipped(
                source.as_ref(),
                &blk("self_attn.q_proj.linear.weight"),
                config.hidden_size,
                config.hidden_size,
                &blk("self_attn.q_proj"),
            )?;
            let self_attn_k_proj = linear_from_gguf_bf16_clipped(
                source.as_ref(),
                &blk("self_attn.k_proj.linear.weight"),
                config.hidden_size,
                config.hidden_size,
                &blk("self_attn.k_proj"),
            )?;
            let self_attn_v_proj = linear_from_gguf_bf16_clipped(
                source.as_ref(),
                &blk("self_attn.v_proj.linear.weight"),
                config.hidden_size,
                config.hidden_size,
                &blk("self_attn.v_proj"),
            )?;
            let self_attn_post = linear_from_gguf_bf16_clipped(
                source.as_ref(),
                &blk("self_attn.post.linear.weight"),
                config.hidden_size,
                config.hidden_size,
                &blk("self_attn.post"),
            )?;
            let self_attn_relative_k_proj = linear_from_gguf_bf16(
                source.as_ref(),
                &blk("self_attn.relative_k_proj.weight"),
                config.hidden_size,
                config.hidden_size,
            )?;
            // per_dim_scale: shape (128,) — stored without `.weight` suffix.
            let self_attn_per_dim_scale = dequant_bf16_to_f32_norm(
                source
                    .tensor_slice(&blk("self_attn.per_dim_scale"))
                    .ok_or_else(|| format!("Missing {}", blk("self_attn.per_dim_scale")))?,
                128,
            );

            // feed_forward1 + feed_forward2 — same structure (Conformer FFN).
            let mut ff_layers = Vec::new();
            for ff_idx in 1..=2 {
                let ff = |suffix: &str| format!("audio.layers.{i}.feed_forward{ff_idx}.{suffix}");
                let pre_ln = dequant_bf16_to_f32_norm(
                    source
                        .tensor_slice(&ff("pre_layer_norm.weight"))
                        .ok_or_else(|| format!("Missing {}.weight", ff("pre_layer_norm")))?,
                    config.hidden_size,
                );
                let ffw_1 = linear_from_gguf_bf16_clipped(
                    source.as_ref(),
                    &ff("ffw_layer_1.linear.weight"),
                    config.hidden_size,
                    config.ffn_hidden_size,
                    &ff("ffw_layer_1"),
                )?;
                let ffw_2 = linear_from_gguf_bf16_clipped(
                    source.as_ref(),
                    &ff("ffw_layer_2.linear.weight"),
                    config.ffn_hidden_size,
                    config.hidden_size,
                    &ff("ffw_layer_2"),
                )?;
                let post_ln = dequant_bf16_to_f32_norm(
                    source
                        .tensor_slice(&ff("post_layer_norm.weight"))
                        .ok_or_else(|| format!("Missing {}.weight", ff("post_layer_norm")))?,
                    config.hidden_size,
                );
                ff_layers.push((pre_ln, ffw_1, ffw_2, post_ln));
            }
            let (ff2_pre_layer_norm, ff2_ffw_layer_1, ff2_ffw_layer_2, ff2_post_layer_norm) =
                ff_layers.pop().expect("ff_layers[1]");
            let (ff1_pre_layer_norm, ff1_ffw_layer_1, ff1_ffw_layer_2, ff1_post_layer_norm) =
                ff_layers.pop().expect("ff_layers[0]");

            layers.push(ConformerLayer {
                lconv1d_conv_norm,
                lconv1d_depthwise_weight,
                lconv1d_pre_layer_norm,
                lconv1d_linear_start,
                lconv1d_linear_end,
                norm_pre_attn,
                norm_post_attn,
                norm_out,
                self_attn_q_proj,
                self_attn_k_proj,
                self_attn_v_proj,
                self_attn_post,
                self_attn_relative_k_proj,
                self_attn_per_dim_scale,
                ff1_pre_layer_norm,
                ff1_ffw_layer_1,
                ff1_ffw_layer_2,
                ff1_post_layer_norm,
                ff2_pre_layer_norm,
                ff2_ffw_layer_1,
                ff2_ffw_layer_2,
                ff2_post_layer_norm,
            });
        }

        Ok(Self {
            config,
            source: Arc::clone(source),
            subsample_input_proj_linear,
            subsample_layer0_conv: layer0_conv,
            subsample_layer0_norm: layer0_norm,
            subsample_layer1_conv: layer1_conv,
            subsample_layer1_norm: layer1_norm,
            output_proj,
            output_proj_bias,
            audio_embed,
            layers,
        })
    }

    /// Encode PCM16 mono audio at 16 kHz to raw audio-tower rows of
    /// `projection_dim` (1536). Returns `[n_frames, proj_dim]`.
    pub fn encode_tower(&self, pcm16: &[i16]) -> Result<Vec<f32>, String> {
        if pcm16.is_empty() {
            return Err("empty audio".to_string());
        }
        let mel = log_mel_features(pcm16);
        let n_frames = mel.len() / 128;
        if n_frames == 0 {
            return Err("audio too short for a mel frame".to_string());
        }
        let cfg = &self.config;
        // SSCP: [T, 128] -> [ceil(T/2), ceil(64/2), 128] ->
        // [ceil(T/4), 32, 32], then flatten frequency × channel.
        let mut conv_input = vec![0.0f32; n_frames * 128];
        for t in 0..n_frames {
            conv_input[t * 128..(t + 1) * 128].copy_from_slice(&mel[t * 128..(t + 1) * 128]);
        }
        let (mut conv0, frames0, bins0) = conv2d_stride2(
            &conv_input,
            n_frames,
            128,
            1,
            &self.subsample_layer0_conv,
            128,
        );
        layer_norm_channels(&mut conv0, frames0, bins0, 128, &self.subsample_layer0_norm);
        for value in &mut conv0 {
            *value = value.max(0.0);
        }
        let (mut conv1, frames1, bins1) =
            conv2d_stride2(&conv0, frames0, bins0, 128, &self.subsample_layer1_conv, 32);
        layer_norm_channels(&mut conv1, frames1, bins1, 32, &self.subsample_layer1_norm);
        for value in &mut conv1 {
            *value = value.max(0.0);
        }
        let proj_input = bins1 * 32;
        let mut x = vec![0.0f32; frames1 * cfg.hidden_size];
        for t in 0..frames1 {
            let mut input = vec![0.0f32; proj_input];
            for b in 0..bins1 {
                for c in 0..32 {
                    input[b * 32 + c] = conv1[(t * bins1 + b) * 32 + c];
                }
            }
            self.subsample_input_proj_linear.forward(
                &input,
                &mut x[t * cfg.hidden_size..(t + 1) * cfg.hidden_size],
            );
        }
        for il in 0..cfg.depth {
            self.forward_conformer_layer(il, &mut x, frames1)?;
        }

        let mut proj = vec![0.0f32; frames1 * cfg.projection_dim];
        let mut tmp = vec![0.0f32; cfg.projection_dim];
        for t in 0..frames1 {
            let src = &x[t * cfg.hidden_size..(t + 1) * cfg.hidden_size];
            self.output_proj.forward(src, &mut tmp);
            for j in 0..cfg.projection_dim {
                proj[t * cfg.projection_dim + j] = tmp[j] + self.output_proj_bias[j];
            }
        }
        Ok(proj)
    }

    /// Encode PCM16 mono audio to the text hidden space used by multimodal
    /// EmbeddingGemma soft tokens.
    pub fn encode(&self, pcm16: &[i16]) -> Result<Vec<f32>, String> {
        let tower = self.encode_tower(pcm16)?;
        let mut output =
            vec![0.0f32; tower.len() / self.config.projection_dim * self.audio_embed.n_out];
        for (input, out) in tower
            .chunks_exact(self.config.projection_dim)
            .zip(output.chunks_exact_mut(self.audio_embed.n_out))
        {
            let mut normalized = input.to_vec();
            crate::ops::norm::rms_unit_inplace(&mut normalized, RMS_NORM_EPS);
            self.audio_embed.forward(&normalized, out);
        }
        Ok(output)
    }

    fn forward_conformer_layer(
        &self,
        il: usize,
        x: &mut [f32],
        n_frames: usize,
    ) -> Result<(), String> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let layer = &self.layers[il];
        // Gemma4's Conformer order is FFN1 -> attention -> light convolution ->
        // FFN2 -> output norm. Each feed-forward block has a 0.5 residual.
        let mut hidden = feed_forward(
            x,
            h,
            cfg.ffn_hidden_size,
            &layer.ff1_pre_layer_norm,
            &layer.ff1_ffw_layer_1,
            &layer.ff1_ffw_layer_2,
            &layer.ff1_post_layer_norm,
        );

        let residual = hidden.clone();
        rms_norm_rows(&mut hidden, h, &layer.norm_pre_attn);
        let mut q = vec![0.0f32; n_frames * h];
        let mut k = vec![0.0f32; n_frames * h];
        let mut v = vec![0.0f32; n_frames * h];
        let mut tmp = vec![0.0f32; h];
        for t in 0..n_frames {
            let row = &hidden[t * h..(t + 1) * h];
            layer.self_attn_q_proj.forward(row, &mut tmp);
            q[t * h..(t + 1) * h].copy_from_slice(&tmp);
            layer.self_attn_k_proj.forward(row, &mut tmp);
            k[t * h..(t + 1) * h].copy_from_slice(&tmp);
            layer.self_attn_v_proj.forward(row, &mut tmp);
            v[t * h..(t + 1) * h].copy_from_slice(&tmp);
        }
        let head_dim = 128;
        let n_heads = h / head_dim;
        let q_scale = (head_dim as f32).sqrt().recip() / std::f32::consts::LN_2;
        let k_scale = (1.0 + std::f32::consts::E).ln() / std::f32::consts::LN_2;
        for t in 0..n_frames {
            for head in 0..n_heads {
                for d in 0..head_dim {
                    let q_idx = t * h + head * head_dim + d;
                    let k_idx = q_idx;
                    let sp = if layer.self_attn_per_dim_scale[d] > 20.0 {
                        layer.self_attn_per_dim_scale[d]
                    } else {
                        (1.0 + layer.self_attn_per_dim_scale[d].exp()).ln()
                    };
                    q[q_idx] *= q_scale * sp;
                    k[k_idx] *= k_scale;
                }
            }
        }

        const CHUNK: usize = 12;
        const PAST: usize = 12;
        const CONTEXT: usize = CHUNK + PAST;
        const REL_POSITIONS: usize = CONTEXT / 2 + 1;
        let num_blocks = n_frames.div_ceil(CHUNK);
        let mut rel_input = vec![0.0f32; REL_POSITIONS * h];
        let log_increment = (10_000.0f32).ln() / (h / 2 - 1) as f32;
        for p in 0..REL_POSITIONS {
            let position = (CONTEXT / 2 - p) as f32;
            for d in 0..h / 2 {
                let value = position * (-log_increment * d as f32).exp();
                rel_input[p * h + d] = value.sin();
                rel_input[p * h + h / 2 + d] = value.cos();
            }
        }
        let mut relative_keys = vec![0.0f32; REL_POSITIONS * h];
        for p in 0..REL_POSITIONS {
            layer.self_attn_relative_k_proj.forward(
                &rel_input[p * h..(p + 1) * h],
                &mut relative_keys[p * h..(p + 1) * h],
            );
        }

        let mut attn_out = vec![0.0f32; n_frames * h];
        for block in 0..num_blocks {
            let block_start = block * CHUNK;
            for query in 0..CHUNK {
                let query_pos = block_start + query;
                if query_pos >= n_frames {
                    continue;
                }
                for head in 0..n_heads {
                    let q_offset = query_pos * h + head * head_dim;
                    let mut scores = vec![0.0f32; CONTEXT];
                    for slot in 0..CONTEXT {
                        let Some(key_pos) = block_start
                            .checked_add(slot)
                            .and_then(|position| position.checked_sub(PAST))
                        else {
                            scores[slot] = -1.0e9;
                            continue;
                        };
                        let Some(distance) = query_pos
                            .checked_sub(key_pos)
                            .filter(|distance| *distance < PAST)
                        else {
                            scores[slot] = -1.0e9;
                            continue;
                        };
                        if key_pos >= n_frames {
                            scores[slot] = -1.0e9;
                            continue;
                        }
                        let k_offset = key_pos * h + head * head_dim;
                        let relative_row = REL_POSITIONS - 1 - distance;
                        let r_offset = relative_row * h + head * head_dim;
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q[q_offset + d] * k[k_offset + d];
                            score += q[q_offset + d] * relative_keys[r_offset + d];
                        }
                        scores[slot] = (score / 50.0).tanh() * 50.0;
                    }
                    crate::ops::softmax::softmax_inplace(&mut scores);
                    let dst = query_pos * h + head * head_dim;
                    for slot in 0..CONTEXT {
                        let Some(key_pos) = block_start
                            .checked_add(slot)
                            .and_then(|position| position.checked_sub(PAST))
                        else {
                            continue;
                        };
                        if key_pos >= n_frames {
                            continue;
                        }
                        let src = key_pos * h + head * head_dim;
                        for d in 0..head_dim {
                            attn_out[dst + d] += scores[slot] * v[src + d];
                        }
                    }
                }
            }
        }
        let mut projected = vec![0.0f32; n_frames * h];
        for t in 0..n_frames {
            layer.self_attn_post.forward(
                &attn_out[t * h..(t + 1) * h],
                &mut projected[t * h..(t + 1) * h],
            );
        }
        rms_norm_rows(&mut projected, h, &layer.norm_post_attn);
        for (value, &base) in projected.iter_mut().zip(&residual) {
            *value += base;
        }
        hidden = projected.clone();

        let residual = hidden.clone();
        rms_norm_rows(&mut hidden, h, &layer.lconv1d_pre_layer_norm);
        let mut conv_input = vec![0.0f32; n_frames * h];
        let mut linear_start = vec![0.0f32; 2 * h];
        for t in 0..n_frames {
            layer
                .lconv1d_linear_start
                .forward(&hidden[t * h..(t + 1) * h], &mut linear_start);
            for d in 0..h {
                conv_input[t * h + d] =
                    linear_start[d] * (1.0 + (-linear_start[h + d]).exp()).recip();
            }
        }
        let mut conv_output = vec![0.0f32; n_frames * h];
        for t in 0..n_frames {
            for ch in 0..h {
                let mut value = 0.0f32;
                for kernel in 0..5 {
                    let src_t = t as isize + kernel as isize - 4;
                    if src_t >= 0 {
                        value += conv_input[src_t as usize * h + ch]
                            * layer.lconv1d_depthwise_weight[ch * 5 + kernel];
                    }
                }
                conv_output[t * h + ch] = value;
            }
        }
        rms_norm_rows(&mut conv_output, h, &layer.lconv1d_conv_norm);
        for value in &mut conv_output {
            *value = crate::ops::silu(*value);
        }
        for t in 0..n_frames {
            layer.lconv1d_linear_end.forward(
                &conv_output[t * h..(t + 1) * h],
                &mut projected[t * h..(t + 1) * h],
            );
        }
        for (value, &base) in projected.iter_mut().zip(&residual) {
            *value += base;
        }
        hidden = feed_forward(
            &projected,
            h,
            cfg.ffn_hidden_size,
            &layer.ff2_pre_layer_norm,
            &layer.ff2_ffw_layer_1,
            &layer.ff2_ffw_layer_2,
            &layer.ff2_post_layer_norm,
        );
        rms_norm_rows(&mut hidden, h, &layer.norm_out);
        x.copy_from_slice(&hidden);

        Ok(())
    }
}
