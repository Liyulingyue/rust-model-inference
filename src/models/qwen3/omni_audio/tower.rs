//! Qwen2.5-Omni audio tower encoder.
//!
//! Implements the full path from a mel-spectrogram (128 mel bins, time-major)
//! through conv1/conv2 + 32 transformer encoder layers + ln_post + proj
//! to per-frame 2048-dim hidden states in Qwen2.5-Omni's text space.
//!
//! Reference: `references/audio.cpp/src/community_models/auk/audio_conditioning.cpp`
//! (C++ implementation from 0xShug0, ~227 LoC).
//!
//! Processing model: 200-frame chunks (after Whisper Mel extraction); for
//! each chunk, conv1 + conv2 + 32 transformer layers; chunks concatenate
//! along the time axis.
//!
//! Weight handling: F16 / BF16 GGUF tensors are dequantized to F32 at load
//! time. The audio tower is small (~1.5 GB of weights) so the F32 expansion
//! is acceptable; subsequent matmul is on F32xF32 which can use SIMD
//! acceleration (we don't use the Q8_0 GPU path here -- that path requires
//! the F16/BF16 kernel to go through `matmul_rows` with a fixed `format`,
//! while our F32 paths use a different code path).

use std::sync::Arc;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::qwen3::asr::audio_processor::{compute_log_mel, LogMel, SAMPLE_RATE};
use crate::models::qwen3::asr::mel_encoder::{
    add_residual, apply_gelu_erf, full_attention_into, layer_norm_rows, LayerNormWeights,
};
use crate::ops::activation::gelu_erf;
use crate::ops::bf16_to_f32;
use crate::ops::f16_to_f32;

/// GGUF tensor prefix for the Qwen2.5-Omni audio tower (audio.cpp format).
pub const QWEN25_OMNI_AUDIO_TOWER_PREFIX: &str = "thinker.audio_tower.";

/// Configuration for the Qwen2.5-Omni audio tower.
///
/// Verified against `models/Qwen2.5-Omni-3B-bf16-GGUF/qwen2.5-omni-3b-bf16.gguf`:
/// - conv1: [3, 128, 1280] (kernel 3, in 128, out 1280)
/// - conv2: [3, 1280, 1280] (kernel 3, stride 2, in 1280, out 1280)
/// - 32 layers, hidden 1280, ffn 5120, heads 20, head_dim 64
/// - ln_post 1280, proj 1280 -> 2048
/// - 100 sin/cos position frames (max), 1280-dim
pub const AUDIO_TOWER_HIDDEN: usize = 1280;
pub const AUDIO_TOWER_HEADS: usize = 20;
pub const AUDIO_TOWER_HEAD_DIM: usize = 64;
pub const AUDIO_TOWER_FFN: usize = 5120;
pub const AUDIO_TOWER_LAYERS: usize = 32;
pub const AUDIO_TOWER_PROJ_DIM: usize = 2048;
pub const AUDIO_TOWER_MEL_BINS: usize = 128;
pub const AUDIO_TOWER_POS_FRAMES: usize = 100;
pub const AUDIO_TOWER_CHUNK_FRAMES: usize = 200;
pub const AUDIO_TOWER_LN_EPS: f32 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioTowerConfig {
    pub hidden: usize,
    pub ffn: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub mel_bins: usize,
    pub proj_dim: usize,
    pub chunk_frames: usize,
    pub pos_frames: usize,
    pub ln_eps_millionths: u32,
}

impl AudioTowerConfig {
    pub fn from_source(_source: &dyn TensorSource) -> Result<Self, String> {
        Ok(AudioTowerConfig {
            hidden: AUDIO_TOWER_HIDDEN,
            ffn: AUDIO_TOWER_FFN,
            layers: AUDIO_TOWER_LAYERS,
            heads: AUDIO_TOWER_HEADS,
            head_dim: AUDIO_TOWER_HEAD_DIM,
            mel_bins: AUDIO_TOWER_MEL_BINS,
            proj_dim: AUDIO_TOWER_PROJ_DIM,
            chunk_frames: AUDIO_TOWER_CHUNK_FRAMES,
            pos_frames: AUDIO_TOWER_POS_FRAMES,
            ln_eps_millionths: 10,
        })
    }
}

/// Linear weight in F32 layout `[in, out]`. We dequantize from F16/BF16
/// at load time and run F32xF32 matmul on CPU.
struct LinearF32 {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}

impl LinearF32 {
    /// Load a linear tensor by name, dequantizing F16/BF16 to F32.
    fn load(
        source: &dyn TensorSource,
        weight_name: &str,
        bias_name: Option<&str>,
        input: usize,
        output: usize,
    ) -> Result<Self, String> {
        let info = source
            .tensor_info(weight_name)
            .ok_or_else(|| format!("Missing tensor: {weight_name}"))?;
        let bytes = source
            .tensor_slice(weight_name)
            .ok_or_else(|| format!("Missing tensor data: {weight_name}"))?;
        let expected_len = input * output;
        let mut weight = vec![0.0f32; expected_len];
        match info.ggml_type {
            GGMLType::F16 => {
                if bytes.len() != expected_len * 2 {
                    return Err(format!(
                        "Tensor {weight_name} expected {} bytes, got {}",
                        expected_len * 2,
                        bytes.len()
                    ));
                }
                for i in 0..expected_len {
                    let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                    weight[i] = f16_to_f32(bits);
                }
            }
            GGMLType::BF16 => {
                if bytes.len() != expected_len * 2 {
                    return Err(format!(
                        "Tensor {weight_name} expected {} bytes, got {}",
                        expected_len * 2,
                        bytes.len()
                    ));
                }
                for i in 0..expected_len {
                    let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                    weight[i] = bf16_to_f32(bits);
                }
            }
            other => {
                return Err(format!(
                    "Tensor {weight_name} expected F16/BF16, got {other:?}"
                ));
            }
        }
        let bias = match bias_name {
            Some(name) => load_f32_tensor_named(source, name, &[output as u64])?,
            None => Vec::new(),
        };
        Ok(Self {
            weight,
            bias,
            input,
            output,
        })
    }

    /// F32xF32 matmul: out[r, c] = sum_k weight[k, c] * input[r, k] + bias[c].
    /// `rows` is the batch size of the input.
    fn project(&self, input: &[f32], rows: usize, output: &mut [f32]) {
        debug_assert_eq!(input.len(), rows * self.input);
        debug_assert_eq!(output.len(), rows * self.output);
        for r in 0..rows {
            let in_row = &input[r * self.input..(r + 1) * self.input];
            let out_row = &mut output[r * self.output..(r + 1) * self.output];
            for c in 0..self.output {
                let mut acc = if c < self.bias.len() {
                    self.bias[c]
                } else {
                    0.0
                };
                let w_col = &self.weight[c..]; // stride = output
                for k in 0..self.input {
                    acc += w_col[k * self.output] * in_row[k];
                }
                out_row[c] = acc;
            }
        }
    }
}

/// Conv1d weights dequantized to F32 layout `[kernel, in, out]`.
struct Conv1dF32 {
    weight: Vec<f32>, // len = kernel * in * out
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    bias: Vec<f32>,
}

impl Conv1dF32 {
    fn load(
        source: &dyn TensorSource,
        weight_name: &str,
        bias_name: &str,
        kernel: usize,
        input: usize,
        output: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self, String> {
        let info = source
            .tensor_info(weight_name)
            .ok_or_else(|| format!("Missing tensor: {weight_name}"))?;
        let bytes = source
            .tensor_slice(weight_name)
            .ok_or_else(|| format!("Missing tensor data: {weight_name}"))?;
        let expected = kernel * input * output;
        let mut weight = vec![0.0f32; expected];
        match info.ggml_type {
            GGMLType::F16 => {
                if bytes.len() != expected * 2 {
                    return Err(format!(
                        "Tensor {weight_name} expected {} bytes, got {}",
                        expected * 2,
                        bytes.len()
                    ));
                }
                for i in 0..expected {
                    let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                    weight[i] = f16_to_f32(bits);
                }
            }
            GGMLType::BF16 => {
                if bytes.len() != expected * 2 {
                    return Err(format!(
                        "Tensor {weight_name} expected {} bytes, got {}",
                        expected * 2,
                        bytes.len()
                    ));
                }
                for i in 0..expected {
                    let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                    weight[i] = bf16_to_f32(bits);
                }
            }
            other => {
                return Err(format!(
                    "Tensor {weight_name} expected F16/BF16, got {other:?}"
                ));
            }
        }
        let bias = load_f32_tensor_named(source, bias_name, &[output as u64])?;
        Ok(Self {
            weight,
            input,
            output,
            kernel,
            stride,
            padding,
            bias,
        })
    }
}

struct AudioLayer {
    ln1: LayerNormWeights,
    q: LinearF32,
    k: LinearF32,
    v: LinearF32,
    output: LinearF32,
    ln2: LayerNormWeights,
    fc1: LinearF32,
    fc2: LinearF32,
}

pub struct AudioTowerModel {
    config: AudioTowerConfig,
    conv1: Conv1dF32,
    conv2: Conv1dF32,
    positions: Vec<f32>,
    layers: Vec<AudioLayer>,
    post_ln: LayerNormWeights,
    proj: LinearF32,
}

impl AudioTowerModel {
    /// Load the audio tower from a Qwen2.5-Omni GGUF (must include the audio
    /// tower, i.e. the BF16 variant from audio-cpp's distribution).
    pub fn from_source(source: Arc<dyn TensorSource>) -> Result<Self, String> {
        let config = AudioTowerConfig::from_source(source.as_ref())?;
        let root = QWEN25_OMNI_AUDIO_TOWER_PREFIX;

        // conv1: in=128 (mel bins), out=hidden, kernel=3, stride=1, padding=1
        let conv1 = Conv1dF32::load(
            source.as_ref(),
            &format!("{root}conv1.weight"),
            &format!("{root}conv1.bias"),
            3,
            config.mel_bins,
            config.hidden,
            1,
            1,
        )?;
        // conv2: in=hidden, out=hidden, kernel=3, stride=2, padding=1
        let conv2 = Conv1dF32::load(
            source.as_ref(),
            &format!("{root}conv2.weight"),
            &format!("{root}conv2.bias"),
            3,
            config.hidden,
            config.hidden,
            2,
            1,
        )?;

        // Sinusoidal position encoding: 100 frames, 1280 dim.
        // Sin in [0, 640), cos in [640, 1280).
        let mut positions = vec![0.0f32; config.pos_frames * config.hidden];
        let increment = (10000.0f32).ln() / 639.0f32;
        for channel in 0..640usize {
            let frequency = (-increment * channel as f32).exp();
            for frame in 0..config.pos_frames {
                let phase = frame as f32 * frequency;
                positions[frame * config.hidden + channel] = phase.sin();
                positions[frame * config.hidden + channel + 640] = phase.cos();
            }
        }

        let hidden_dim = [config.hidden as u64];
        let mut layers = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let prefix = format!("{root}layers.{layer}.");
            let ln1 = LayerNormWeights {
                weight: load_f32_tensor_named(
                    source.as_ref(),
                    &format!("{prefix}self_attn_layer_norm.weight"),
                    &hidden_dim,
                )?,
                bias: load_f32_tensor_named(
                    source.as_ref(),
                    &format!("{prefix}self_attn_layer_norm.bias"),
                    &hidden_dim,
                )?,
            };
            layers.push(AudioLayer {
                ln1,
                q: LinearF32::load(
                    source.as_ref(),
                    &format!("{prefix}self_attn.q_proj.weight"),
                    Some(&format!("{prefix}self_attn.q_proj.bias")),
                    config.hidden,
                    config.hidden,
                )?,
                k: LinearF32::load(
                    source.as_ref(),
                    &format!("{prefix}self_attn.k_proj.weight"),
                    None, // k_proj has no bias in Qwen2.5-Omni
                    config.hidden,
                    config.hidden,
                )?,
                v: LinearF32::load(
                    source.as_ref(),
                    &format!("{prefix}self_attn.v_proj.weight"),
                    Some(&format!("{prefix}self_attn.v_proj.bias")),
                    config.hidden,
                    config.hidden,
                )?,
                output: LinearF32::load(
                    source.as_ref(),
                    &format!("{prefix}self_attn.out_proj.weight"),
                    Some(&format!("{prefix}self_attn.out_proj.bias")),
                    config.hidden,
                    config.hidden,
                )?,
                ln2: LayerNormWeights {
                    weight: load_f32_tensor_named(
                        source.as_ref(),
                        &format!("{prefix}final_layer_norm.weight"),
                        &hidden_dim,
                    )?,
                    bias: load_f32_tensor_named(
                        source.as_ref(),
                        &format!("{prefix}final_layer_norm.bias"),
                        &hidden_dim,
                    )?,
                },
                fc1: LinearF32::load(
                    source.as_ref(),
                    &format!("{prefix}fc1.weight"),
                    Some(&format!("{prefix}fc1.bias")),
                    config.hidden,
                    config.ffn,
                )?,
                fc2: LinearF32::load(
                    source.as_ref(),
                    &format!("{prefix}fc2.weight"),
                    Some(&format!("{prefix}fc2.bias")),
                    config.ffn,
                    config.hidden,
                )?,
            });
        }
        let post_ln = LayerNormWeights {
            weight: load_f32_tensor_named(
                source.as_ref(),
                &format!("{root}ln_post.weight"),
                &hidden_dim,
            )?,
            bias: load_f32_tensor_named(
                source.as_ref(),
                &format!("{root}ln_post.bias"),
                &hidden_dim,
            )?,
        };
        let proj = LinearF32::load(
            source.as_ref(),
            &format!("{root}proj.weight"),
            Some(&format!("{root}proj.bias")),
            config.hidden,
            config.proj_dim,
        )?;

        // Hold the source alive for the lifetime of the model (in case any
        // tensor data references it; we dequantize all weights at load so
        // this is a precaution, not a requirement).
        let _ = source;
        Ok(Self {
            config,
            conv1,
            conv2,
            positions,
            layers,
            post_ln,
            proj,
        })
    }

    pub fn config(&self) -> AudioTowerConfig {
        self.config
    }

    /// Run the full forward: mel-spectrogram (already Whisper-normalized) ->
    /// chunked conv1/conv2 + transformer stack + ln_post + proj.
    pub fn encode_mel(&self, mel: &LogMel) -> Result<(Vec<f32>, usize), String> {
        if mel.frames == 0 {
            return Err("Cannot encode empty mel-spectrogram".into());
        }
        if mel.normalized.len() != self.config.mel_bins * mel.frames {
            return Err(format!(
                "Invalid mel-spectrogram shape: expected {}*{}={}, got {}",
                self.config.mel_bins,
                mel.frames,
                self.config.mel_bins * mel.frames,
                mel.normalized.len()
            ));
        }
        let chunk_in_frames = self.config.chunk_frames;
        let mut all_output = Vec::with_capacity((mel.frames + 1) / 2 * self.config.proj_dim);
        let mut total_tokens = 0usize;

        let mut start = 0usize;
        while start < mel.frames {
            let length = chunk_in_frames.min(mel.frames - start);
            let chunk_tokens = self.encode_chunk(&mel.normalized, start, length)?;
            all_output.extend_from_slice(&chunk_tokens.0);
            total_tokens += chunk_tokens.1;
            start += length;
        }

        Ok((all_output, total_tokens))
    }

    /// Convenience: mel-extract + encode. Input must be 16 kHz mono PCM f32.
    pub fn encode_pcm(&self, samples_16k_mono: &[f32]) -> Result<(Vec<f32>, usize), String> {
        let mel = compute_log_mel(samples_16k_mono)
            .map_err(|e| format!("Qwen2.5-Omni mel-spectrogram error: {:?}", e))?;
        self.encode_mel(&mel)
    }

    /// Process a single chunk: conv1 (GELU) + conv2 (GELU) + 32 transformer
    /// layers + ln_post + proj. Returns (values, n_tokens).
    fn encode_chunk(
        &self,
        mel_normalized: &[f32],
        start: usize,
        length: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        if length == 0 {
            return Err("Cannot encode empty chunk".into());
        }
        if length > self.config.chunk_frames {
            return Err(format!(
                "Chunk length {} exceeds max {}",
                length, self.config.chunk_frames
            ));
        }
        if start + length > mel_normalized.len() / self.config.mel_bins {
            return Err(format!(
                "Mel chunk out of bounds: start={} length={} mel frames={}",
                start,
                length,
                mel_normalized.len() / self.config.mel_bins
            ));
        }

        // 1. Conv1: mel [128, length] -> hidden [1280, length]
        let mut stage1 = vec![0.0f32; self.config.hidden * length];
        conv1d_into(
            mel_normalized,
            start,
            self.config.mel_bins,
            length,
            &self.conv1,
            &mut stage1,
        )?;
        apply_gelu_erf_inplace(&mut stage1)?;

        // 2. Conv2: [1280, length] -> [1280, (length+1)/2]
        let conv2_out_len = (length + 1) / 2;
        let mut stage2 = vec![0.0f32; self.config.hidden * conv2_out_len];
        conv1d_into(
            &stage1,
            0,
            self.config.hidden,
            length,
            &self.conv2,
            &mut stage2,
        )?;
        apply_gelu_erf_inplace(&mut stage2)?;
        let tokens = conv2_out_len;

        // 3. Add position embedding: transpose stage2 to [tokens, hidden] then add
        let mut hidden = vec![0.0f32; tokens * self.config.hidden];
        for t in 0..tokens {
            for h in 0..self.config.hidden {
                hidden[t * self.config.hidden + h] =
                    stage2[h * tokens + t] + self.positions[t * self.config.hidden + h];
            }
        }
        require_finite(&hidden, "audio tower after conv+position")?;

        // 4. 32 transformer encoder layers
        let epsilon = self.config.ln_eps_millionths as f32 * 1.0e-6;
        for (i, layer) in self.layers.iter().enumerate() {
            run_encoder_layer(&mut hidden, layer, &self.config, epsilon)
                .map_err(|e| format!("audio tower layer {i}: {e}"))?;
        }
        require_finite(&hidden, "audio tower after transformer")?;

        // 5. ln_post
        let mut post = Vec::with_capacity(hidden.len());
        layer_norm_rows(&hidden, tokens, &self.post_ln, epsilon, &mut post)?;
        require_finite(&post, "audio tower after ln_post")?;

        // 6. proj 1280 -> 2048
        let mut out = vec![0.0f32; tokens * self.config.proj_dim];
        self.proj.project(&post, tokens, &mut out);
        require_finite(&out, "audio tower after proj")?;

        Ok((out, tokens))
    }
}

/// Run a single transformer encoder layer (pre-norm).
fn run_encoder_layer(
    hidden: &mut [f32],
    layer: &AudioLayer,
    config: &AudioTowerConfig,
    epsilon: f32,
) -> Result<(), String> {
    let tokens = hidden.len() / config.hidden;
    // 1. ln1 -> self-attention
    let mut normed = Vec::with_capacity(hidden.len());
    layer_norm_rows(hidden, tokens, &layer.ln1, epsilon, &mut normed)?;

    let mut q = vec![0.0f32; tokens * config.hidden];
    let mut k = vec![0.0f32; tokens * config.hidden];
    let mut v = vec![0.0f32; tokens * config.hidden];
    layer.q.project(&normed, tokens, &mut q);
    layer.k.project(&normed, tokens, &mut k);
    layer.v.project(&normed, tokens, &mut v);

    let mut scores = Vec::new();
    let mut attn = Vec::with_capacity(tokens * config.hidden);
    full_attention_into(
        &q,
        &k,
        &v,
        tokens,
        config.heads,
        config.head_dim,
        &mut scores,
        &mut attn,
    )?;

    let mut proj_out = vec![0.0f32; tokens * config.hidden];
    layer.output.project(&attn, tokens, &mut proj_out);
    add_residual(hidden, &proj_out)?;

    // 2. ln2 -> FFN (fc1 -> GELU -> fc2)
    let mut normed2 = Vec::with_capacity(hidden.len());
    layer_norm_rows(hidden, tokens, &layer.ln2, epsilon, &mut normed2)?;

    let mut ffn1 = vec![0.0f32; tokens * config.ffn];
    layer.fc1.project(&normed2, tokens, &mut ffn1);
    apply_gelu_erf(&mut ffn1)?;

    let mut ffn2 = vec![0.0f32; tokens * config.hidden];
    layer.fc2.project(&ffn1, tokens, &mut ffn2);

    add_residual(hidden, &ffn2)?;
    Ok(())
}

/// 1-D convolution forward (single batch, no dilation). Input is
/// `[input_channels, length]`; output is `[output_channels, out_len]`.
/// Weight layout: `[kernel, input_channels, output_channels]`.
fn conv1d_into(
    input: &[f32],
    input_offset_frames: usize,
    input_channels: usize,
    input_length: usize,
    conv: &Conv1dF32,
    output: &mut [f32],
) -> Result<(), String> {
    if conv.input != input_channels {
        return Err(format!(
            "Conv1d input channel mismatch: weight expects {}, got {}",
            conv.input, input_channels
        ));
    }
    let out_len = (input_length + 2 * conv.padding - conv.kernel) / conv.stride + 1;
    if output.len() != conv.output * out_len {
        return Err(format!(
            "Conv1d output buffer size mismatch: expected {}*{}={}, got {}",
            conv.output,
            out_len,
            conv.output * out_len,
            output.len()
        ));
    }
    if input.len() < input_channels * (input_offset_frames + input_length) {
        return Err(format!(
            "Conv1d input buffer too small: have {}, need {}",
            input.len(),
            input_channels * (input_offset_frames + input_length)
        ));
    }
    for c_out in 0..conv.output {
        for t_out in 0..out_len {
            let mut acc = 0.0f32;
            for k in 0..conv.kernel {
                let t_in = (t_out * conv.stride + k) as isize - conv.padding as isize;
                if t_in < 0 || t_in >= input_length as isize {
                    continue;
                }
                for c_in in 0..input_channels {
                    let input_idx = (input_offset_frames + t_in as usize) * input_channels + c_in;
                    // Weight index: [k, c_in, c_out]
                    let w_idx = k * conv.input * conv.output + c_in * conv.output + c_out;
                    acc += input[input_idx] * conv.weight[w_idx];
                }
            }
            output[c_out * out_len + t_out] = acc + conv.bias[c_out];
        }
    }
    Ok(())
}

fn apply_gelu_erf_inplace(values: &mut [f32]) -> Result<(), String> {
    if values.is_empty() {
        return Ok(());
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err("Invalid audio conv output (non-finite)".into());
    }
    for v in values.iter_mut() {
        *v = gelu_erf(*v);
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err("Non-finite audio GELU output".into());
    }
    Ok(())
}

fn require_finite(values: &[f32], context: &str) -> Result<(), String> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(format!("{context} contains non-finite values"));
    }
    Ok(())
}

fn load_f32_tensor_named(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims {
        return Err(format!(
            "Tensor {name} shape mismatch: expected {dims:?}, got {:?}",
            info.dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let expected_elements: usize = dims.iter().map(|&d| d as usize).product();
    match info.ggml_type {
        GGMLType::F32 => {
            if bytes.len() != expected_elements * 4 {
                return Err(format!(
                    "Tensor {name} expected {} bytes, got {}",
                    expected_elements * 4,
                    bytes.len()
                ));
            }
            let mut values = Vec::with_capacity(expected_elements);
            for chunk in bytes.chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(values)
        }
        GGMLType::F16 => {
            if bytes.len() != expected_elements * 2 {
                return Err(format!(
                    "Tensor {name} expected {} bytes, got {}",
                    expected_elements * 2,
                    bytes.len()
                ));
            }
            let mut values = Vec::with_capacity(expected_elements);
            for i in 0..expected_elements {
                let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                values.push(f16_to_f32(bits));
            }
            Ok(values)
        }
        GGMLType::BF16 => {
            if bytes.len() != expected_elements * 2 {
                return Err(format!(
                    "Tensor {name} expected {} bytes, got {}",
                    expected_elements * 2,
                    bytes.len()
                ));
            }
            let mut values = Vec::with_capacity(expected_elements);
            for i in 0..expected_elements {
                let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                values.push(bf16_to_f32(bits));
            }
            Ok(values)
        }
        other => Err(format!(
            "Tensor {name} expected F32/F16/BF16, got {other:?}"
        )),
    }
}

pub fn audio_tower_sample_rate() -> usize {
    SAMPLE_RATE
}
