use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::qwen3::asr::audio_processor::log_mel_windows;
use crate::models::qwen3::asr::audio_processor::{compute_log_mel, HOP};
use crate::models::qwen3::asr::mel_encoder::Qwen3AudioModel;
use crate::models::qwen3::asr::mel_encoder::{
    add_residual, apply_gelu_erf, checked_product, full_attention_into, layer_norm_rows_qwen25,
    load_f32_tensor, reserved_f32, resize_f32, static_tensor, AudioLinear, LayerNormWeights,
};
use crate::ops::dot_f16_f16_bytes;
use std::sync::Arc;

const WHISPER_CHUNK: usize = 3000;

struct Conv1dWeights {
    weight: &'static [u8],
    bias: Vec<f32>,
    input: usize,
    output: usize,
}

struct AudioLayer {
    ln1: LayerNormWeights,
    q: AudioLinear,
    k: AudioLinear,
    v: AudioLinear,
    output: AudioLinear,
    ln2: LayerNormWeights,
    up: AudioLinear,
    down: AudioLinear,
}

pub struct Qwen25OmniAudioModel {
    _source: Arc<dyn TensorSource>,
    config: Qwen25OmniAudioConfig,
    conv1: Conv1dWeights,
    conv2: Conv1dWeights,
    positions: Vec<f32>,
    layers: Vec<AudioLayer>,
    post_ln: LayerNormWeights,
    projector: AudioLinear,
}

impl Qwen25OmniAudioModel {
    pub fn from_source(source: Arc<dyn TensorSource>) -> Result<Self, String> {
        let config = Qwen25OmniAudioConfig::from_source(source.as_ref())?;
        let conv1 = load_conv1d(&source, "a.conv1d.1", config.mel_bins, config.hidden)?;
        let conv2 = load_conv1d(&source, "a.conv1d.2", config.hidden, config.hidden)?;
        let positions = load_f32_tensor(
            source.as_ref(),
            "a.position_embd.weight",
            &[config.hidden as u64, 1500],
        )?;
        let hidden = [config.hidden as u64];
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.layers)
            .map_err(|_| "Failed to allocate Qwen2.5-Omni audio layers".to_string())?;
        for layer in 0..config.layers {
            let prefix = format!("a.blk.{layer}");
            layers.push(AudioLayer {
                ln1: LayerNormWeights::load(source.as_ref(), &format!("{prefix}.ln1"), &hidden)?,
                q: AudioLinear::load(
                    &source,
                    &format!("{prefix}.attn_q.weight"),
                    Some(&format!("{prefix}.attn_q.bias")),
                    config.hidden,
                    config.hidden,
                    GGMLType::F16,
                )?,
                k: AudioLinear::load(
                    &source,
                    &format!("{prefix}.attn_k.weight"),
                    None,
                    config.hidden,
                    config.hidden,
                    GGMLType::F16,
                )?,
                v: AudioLinear::load(
                    &source,
                    &format!("{prefix}.attn_v.weight"),
                    Some(&format!("{prefix}.attn_v.bias")),
                    config.hidden,
                    config.hidden,
                    GGMLType::F16,
                )?,
                output: AudioLinear::load(
                    &source,
                    &format!("{prefix}.attn_out.weight"),
                    Some(&format!("{prefix}.attn_out.bias")),
                    config.hidden,
                    config.hidden,
                    GGMLType::F16,
                )?,
                ln2: LayerNormWeights::load(source.as_ref(), &format!("{prefix}.ln2"), &hidden)?,
                up: AudioLinear::load(
                    &source,
                    &format!("{prefix}.ffn_up.weight"),
                    Some(&format!("{prefix}.ffn_up.bias")),
                    config.hidden,
                    config.ffn,
                    GGMLType::F16,
                )?,
                down: AudioLinear::load(
                    &source,
                    &format!("{prefix}.ffn_down.weight"),
                    Some(&format!("{prefix}.ffn_down.bias")),
                    config.ffn,
                    config.hidden,
                    GGMLType::F16,
                )?,
            });
        }
        let post_ln = LayerNormWeights::load(source.as_ref(), "a.post_ln", &hidden)?;
        let projector = AudioLinear::load(
            &source,
            "mm.a.fc.weight",
            Some("mm.a.fc.bias"),
            config.hidden,
            config.projection,
            GGMLType::F16,
        )?;
        Ok(Self {
            _source: source,
            config,
            conv1,
            conv2,
            positions,
            layers,
            post_ln,
            projector,
        })
    }

    pub fn encode(&self, samples: &[f32]) -> Result<Vec<f32>, String> {
        let (layout, input) = prepare_whisper_mel(samples, self.config.mel_bins)?;

        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "omni.audio.mel",
            None,
            &[layout.padded_mel_frames, self.config.mel_bins],
            &input,
        ));

        let mut conv1 = Vec::new();
        conv1d_same_f16(
            &input,
            layout.padded_mel_frames,
            self.conv1.input,
            self.conv1.output,
            self.conv1.weight,
            &self.conv1.bias,
            1,
            &mut conv1,
        )?;
        apply_gelu_erf(&mut conv1)?;
        let mut hidden = Vec::new();
        conv1d_same_f16(
            &conv1,
            layout.padded_mel_frames,
            self.conv2.input,
            self.conv2.output,
            self.conv2.weight,
            &self.conv2.bias,
            2,
            &mut hidden,
        )?;
        apply_gelu_erf(&mut hidden)?;

        for token in 0..layout.post_conv_tokens {
            let position = token % (self.positions.len() / self.config.hidden);
            let position_row =
                &self.positions[position * self.config.hidden..(position + 1) * self.config.hidden];
            let hidden_row =
                &mut hidden[token * self.config.hidden..(token + 1) * self.config.hidden];
            for (value, position) in hidden_row.iter_mut().zip(position_row) {
                *value += *position;
            }
        }

        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "omni.audio.after_conv1d_chunked",
            None,
            &[layout.post_conv_tokens, self.config.hidden],
            &hidden,
        ));

        let values = layout.post_conv_tokens * self.config.hidden;
        let ffn_values = layout.post_conv_tokens * self.config.ffn;
        let mut normed = reserved_f32("Qwen2.5-Omni normalized", values)?;
        let mut q = reserved_f32("Qwen2.5-Omni queries", values)?;
        let mut k = reserved_f32("Qwen2.5-Omni keys", values)?;
        let mut v = reserved_f32("Qwen2.5-Omni values", values)?;
        let mut attention = reserved_f32("Qwen2.5-Omni attention", values)?;
        let mut update = reserved_f32("Qwen2.5-Omni update", values)?;
        let mut ffn_up = reserved_f32("Qwen2.5-Omni FFN up", ffn_values)?;
        let mut ffn_down = reserved_f32("Qwen2.5-Omni FFN down", values)?;
        let mut scores = reserved_f32("Qwen2.5-Omni scores", self.config.window)?;
        let head_dim = self.config.hidden / self.config.heads;
        for layer in &self.layers {
            layer_norm_rows_qwen25(
                &hidden,
                layout.post_conv_tokens,
                &layer.ln1,
                self.config.epsilon,
                &mut normed,
            )?;
            layer
                .q
                .project_f16_ggml(&normed, layout.post_conv_tokens, &mut q)?;
            layer
                .k
                .project_f16_ggml(&normed, layout.post_conv_tokens, &mut k)?;
            layer
                .v
                .project_f16_ggml(&normed, layout.post_conv_tokens, &mut v)?;
            full_attention_into(
                &q,
                &k,
                &v,
                layout.post_conv_tokens,
                self.config.heads,
                head_dim,
                &mut scores,
                &mut attention,
            )?;
            layer
                .output
                .project_f16_ggml(&attention, layout.post_conv_tokens, &mut update)?;
            add_residual(&mut hidden, &update)?;
            layer_norm_rows_qwen25(
                &hidden,
                layout.post_conv_tokens,
                &layer.ln2,
                self.config.epsilon,
                &mut normed,
            )?;
            layer
                .up
                .project_f16_ggml(&normed, layout.post_conv_tokens, &mut ffn_up)?;
            apply_gelu_erf(&mut ffn_up)?;
            layer
                .down
                .project_f16_ggml(&ffn_up, layout.post_conv_tokens, &mut ffn_down)?;
            add_residual(&mut hidden, &ffn_down)?;
        }

        let mut pooled = Vec::new();
        average_pool_pairs(&hidden, self.config.hidden, &mut pooled)?;
        pooled.truncate(layout.output_rows * self.config.hidden);
        layer_norm_rows_qwen25(
            &pooled,
            layout.output_rows,
            &self.post_ln,
            self.config.epsilon,
            &mut normed,
        )?;

        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "omni.audio.after_transformer",
            None,
            &[layout.output_rows, self.config.hidden],
            &normed,
        ));

        let mut projected = Vec::new();
        self.projector
            .project_f16_ggml(&normed, layout.output_rows, &mut projected)?;
        if projected.iter().any(|value| !value.is_finite()) {
            return Err("Non-finite Qwen2.5-Omni audio projection".into());
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "omni.audio.projected",
            None,
            &[layout.output_rows, self.config.projection],
            &projected,
        ));
        Ok(projected)
    }
}

pub fn encode_audio(
    source: Arc<dyn TensorSource>,
    samples: &[f32],
    threads: usize,
) -> Result<Vec<f32>, String> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err("Audio samples must be non-empty and finite".into());
    }
    if source
        .metadata("clip.audio.projector_type")
        .and_then(MetaValue::to_string_val)
        .is_some_and(|value| value == "qwen3a")
    {
        let n_threads = if threads == 0 {
            std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1)
        } else {
            threads
        };
        let model = Qwen3AudioModel::from_source(
            Arc::clone(&source),
            Arc::new(ComputePool::new(n_threads)),
        )?;
        let windows =
            log_mel_windows(samples).map_err(|error| format!("Audio Mel error: {error:?}"))?;
        return Ok(model.encode(&windows)?.values);
    }
    Qwen25OmniAudioModel::from_source(source)?.encode(samples)
}

fn load_conv1d(
    source: &Arc<dyn TensorSource>,
    prefix: &str,
    input: usize,
    output: usize,
) -> Result<Conv1dWeights, String> {
    let dims = [3, input as u64, output as u64];
    Ok(Conv1dWeights {
        weight: static_tensor(source, &format!("{prefix}.weight"), &dims, GGMLType::F16)?,
        bias: load_f32_tensor(
            source.as_ref(),
            &format!("{prefix}.bias"),
            &[1, output as u64],
        )?,
        input,
        output,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen25OmniAudioConfig {
    pub hidden: usize,
    pub ffn: usize,
    pub layers: usize,
    pub heads: usize,
    pub mel_bins: usize,
    pub window: usize,
    pub projection: usize,
    pub epsilon: f32,
}

impl Qwen25OmniAudioConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        require_string(source, "general.architecture", "clip")?;
        require_string(source, "general.type", "mmproj")?;
        require_bool(source, "clip.has_audio_encoder", true)?;
        require_string(source, "clip.projector_type", "qwen2.5o")?;
        let hidden = require_u32(source, "clip.audio.embedding_length", 1280)? as usize;
        let ffn = require_u32(source, "clip.audio.feed_forward_length", 5120)? as usize;
        let layers = require_u32(source, "clip.audio.block_count", 32)? as usize;
        let heads = require_u32(source, "clip.audio.attention.head_count", 20)? as usize;
        let mel_bins = require_u32(source, "clip.audio.num_mel_bins", 128)? as usize;
        // `clip.audio.n_window` (audio encoder STFT window) is not always
        // shipped in mmproj metadata; fall back to the documented
        // 100-tap default rather than rejecting the load.
        let window = source
            .metadata("clip.audio.n_window")
            .and_then(MetaValue::to_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "clip.audio.n_window does not fit usize")?
            .unwrap_or(100);
        let epsilon = require_f32(source, "clip.audio.attention.layer_norm_epsilon", 1e-5)?;

        require_tensor(
            source,
            "a.position_embd.weight",
            &[1280, 1500],
            GGMLType::F32,
        )?;
        require_tensor(source, "a.conv1d.1.weight", &[3, 128, 1280], GGMLType::F16)?;
        require_tensor(source, "a.conv1d.1.bias", &[1, 1280], GGMLType::F32)?;
        require_tensor(source, "a.conv1d.2.weight", &[3, 1280, 1280], GGMLType::F16)?;
        require_tensor(source, "a.conv1d.2.bias", &[1, 1280], GGMLType::F32)?;
        for layer in 0..layers {
            let prefix = format!("a.blk.{layer}");
            for name in ["attn_q", "attn_k", "attn_v", "attn_out"] {
                require_tensor(
                    source,
                    &format!("{prefix}.{name}.weight"),
                    &[1280, 1280],
                    GGMLType::F16,
                )?;
            }
            if source
                .tensor_info(&format!("{prefix}.attn_k.bias"))
                .is_some()
            {
                return Err(format!(
                    "Qwen2.5-Omni audio K bias must be absent in {prefix}"
                ));
            }
            for name in ["attn_q", "attn_v", "attn_out"] {
                require_tensor(
                    source,
                    &format!("{prefix}.{name}.bias"),
                    &[1280],
                    GGMLType::F32,
                )?;
            }
            for name in ["ln1", "ln2"] {
                require_tensor(
                    source,
                    &format!("{prefix}.{name}.weight"),
                    &[1280],
                    GGMLType::F32,
                )?;
                require_tensor(
                    source,
                    &format!("{prefix}.{name}.bias"),
                    &[1280],
                    GGMLType::F32,
                )?;
            }
            require_tensor(
                source,
                &format!("{prefix}.ffn_up.weight"),
                &[1280, 5120],
                GGMLType::F16,
            )?;
            require_tensor(
                source,
                &format!("{prefix}.ffn_up.bias"),
                &[5120],
                GGMLType::F32,
            )?;
            require_tensor(
                source,
                &format!("{prefix}.ffn_down.weight"),
                &[5120, 1280],
                GGMLType::F16,
            )?;
            require_tensor(
                source,
                &format!("{prefix}.ffn_down.bias"),
                &[1280],
                GGMLType::F32,
            )?;
        }
        for name in ["weight", "bias"] {
            require_tensor(source, &format!("a.post_ln.{name}"), &[1280], GGMLType::F32)?;
        }
        let projector = source
            .tensor_info("mm.a.fc.weight")
            .ok_or("Missing Qwen2.5-Omni audio tensor: mm.a.fc.weight")?;
        // F16 / BF16 are interchangeable for this projection layer (matmul
        // contract is identical); accept either to match Unsloth's BF16
        // re-quantization.
        let type_ok = matches!(projector.ggml_type, GGMLType::F16 | GGMLType::BF16);
        if projector.dims.first() != Some(&(hidden as u64)) || projector.dims.len() != 2 || !type_ok
        {
            return Err(format!(
                "Invalid Qwen2.5-Omni audio tensor: mm.a.fc.weight \
                 (shape {:?} type {:?}; expected [{}, ?] F16/BF16)",
                projector.dims, projector.ggml_type, hidden
            ));
        }
        let projection = projector.dims[1] as usize;
        require_tensor(source, "mm.a.fc.bias", &[projection as u64], GGMLType::F32)?;

        Ok(Self {
            hidden,
            ffn,
            layers,
            heads,
            mel_bins,
            window,
            projection,
            epsilon,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AudioLayout {
    padded_mel_frames: usize,
    post_conv_tokens: usize,
    output_rows: usize,
}

impl AudioLayout {
    fn for_real_frames(real_mel_frames: usize) -> Result<Self, String> {
        if real_mel_frames == 0 {
            return Err("Audio must contain at least one Mel frame".into());
        }
        if real_mel_frames > WHISPER_CHUNK {
            return Err("Audio exceeds the supported 30-second Whisper chunk".into());
        }
        let padded_mel_frames = WHISPER_CHUNK;
        Ok(Self {
            padded_mel_frames,
            post_conv_tokens: padded_mel_frames / 2,
            output_rows: padded_mel_frames / 4,
        })
    }
}

fn prepare_whisper_mel(
    samples: &[f32],
    mel_bins: usize,
) -> Result<(AudioLayout, Vec<f32>), String> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err("Audio samples must be non-empty and finite".into());
    }
    let layout = AudioLayout::for_real_frames(samples.len().div_ceil(HOP))?;
    // llama.cpp's Whisper preprocessor appends silence before computing Mel
    // frames. Leave enough zeros for the centered final FFT window.
    let mut padded_samples = samples.to_vec();
    padded_samples.extend([0.0; 400]);
    let mel = compute_log_mel(&padded_samples)
        .map_err(|error| format!("Audio Mel error: {error:?}"))?;
    if mel.normalized.len() != mel.frames * mel_bins {
        return Err("Audio Mel output shape does not match the projector".into());
    }
    let mut input = reserved_f32(
        "Qwen2.5-Omni Mel input",
        checked_product("Qwen2.5-Omni Mel input", layout.padded_mel_frames, mel_bins)?,
    )?;
    for frame in 0..layout.padded_mel_frames {
        let source_frame = frame.min(mel.frames - 1);
        for mel_bin in 0..mel_bins {
            input[frame * mel_bins + mel_bin] = mel.normalized[mel_bin * mel.frames + source_frame];
        }
    }
    Ok((layout, input))
}

#[allow(clippy::too_many_arguments)]
fn conv1d_same_f16(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    weights: &[u8],
    bias: &[f32],
    stride: usize,
    output: &mut Vec<f32>,
) -> Result<(), String> {
    if rows == 0 || input_dim == 0 || output_dim == 0 || stride == 0 {
        return Err("Qwen2.5-Omni Conv1D dimensions must be nonzero".into());
    }
    let input_len = checked_product("Qwen2.5-Omni Conv1D input", rows, input_dim)?;
    let kernel_values = checked_product(
        "Qwen2.5-Omni Conv1D weights",
        checked_product("Qwen2.5-Omni Conv1D kernel", 3, input_dim)?,
        output_dim,
    )?;
    if input.len() != input_len
        || weights.len() != checked_product("Qwen2.5-Omni Conv1D bytes", kernel_values, 2)?
        || bias.len() != output_dim
        || input.iter().chain(bias).any(|value| !value.is_finite())
    {
        return Err("Invalid Qwen2.5-Omni Conv1D tensors".into());
    }
    let output_rows = rows.div_ceil(stride);
    resize_f32(
        output,
        "Qwen2.5-Omni Conv1D output",
        checked_product("Qwen2.5-Omni Conv1D output", output_rows, output_dim)?,
    )?;
    let patch_len = checked_product("Qwen2.5-Omni Conv1D patch", input_dim, 3)?;
    let mut patch = vec![crate::ops::f32_to_f16(0.0); patch_len];
    for output_row in 0..output_rows {
        patch.fill(crate::ops::f32_to_f16(0.0));
        let center = output_row * stride;
        for input_channel in 0..input_dim {
            for kernel in 0..3 {
                let Some(input_row) = center
                    .checked_add(kernel)
                    .and_then(|row| row.checked_sub(1))
                else {
                    continue;
                };
                if input_row < rows {
                    patch[input_channel * 3 + kernel] =
                        crate::ops::f32_to_f16(input[input_row * input_dim + input_channel]);
                }
            }
        }
        for output_channel in 0..output_dim {
            let weight_start = output_channel * patch_len * 2;
            let value = bias[output_channel]
                + dot_f16_f16_bytes(
                    &patch,
                    &weights[weight_start..weight_start + patch_len * 2],
                    patch_len,
                );
            if !value.is_finite() {
                return Err("Non-finite Qwen2.5-Omni Conv1D output".into());
            }
            output[output_row * output_dim + output_channel] = value;
        }
    }
    Ok(())
}

fn average_pool_pairs(input: &[f32], width: usize, output: &mut Vec<f32>) -> Result<(), String> {
    if width == 0 || input.is_empty() || input.len() % (width * 2) != 0 {
        return Err("Invalid Qwen2.5-Omni average-pool shape".into());
    }
    resize_f32(output, "Qwen2.5-Omni average-pool output", input.len() / 2)?;
    for (row, pair) in input.chunks_exact(width * 2).enumerate() {
        for lane in 0..width {
            output[row * width + lane] = (pair[lane] + pair[width + lane]) * 0.5;
        }
    }
    Ok(())
}

fn require_string(source: &dyn TensorSource, key: &str, expected: &str) -> Result<(), String> {
    match source.metadata(key) {
        Some(MetaValue::String(value)) if value == expected => Ok(()),
        _ => Err(format!(
            "Invalid Qwen2.5-Omni audio metadata {key}: expected {expected}"
        )),
    }
}

fn require_bool(source: &dyn TensorSource, key: &str, expected: bool) -> Result<(), String> {
    match source.metadata(key) {
        Some(MetaValue::Bool(value)) if *value == expected => Ok(()),
        _ => Err(format!(
            "Invalid Qwen2.5-Omni audio metadata {key}: expected {expected}"
        )),
    }
}

fn require_u32(source: &dyn TensorSource, key: &str, expected: u32) -> Result<u32, String> {
    match source.metadata(key) {
        Some(MetaValue::Uint32(value)) if *value == expected => Ok(*value),
        _ => Err(format!(
            "Invalid Qwen2.5-Omni audio metadata {key}: expected {expected}"
        )),
    }
}

fn require_f32(source: &dyn TensorSource, key: &str, expected: f32) -> Result<f32, String> {
    match source.metadata(key) {
        Some(MetaValue::Float32(value)) if *value == expected => Ok(*value),
        _ => Err(format!(
            "Invalid Qwen2.5-Omni audio metadata {key}: expected {expected}"
        )),
    }
}

fn require_tensor(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
    kind: GGMLType,
) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing Qwen2.5-Omni audio tensor: {name}"))?;
    // Shape is strict (architectural mismatch). Type is F16-or-BF16 tolerant:
    // some Unsloth re-quantization rounds trip BF16→F16→BF16 and the upstream
    // GGUF uses F16 while our tools/tests produce BF16; both pass the same
    // matmul contract.
    if info.dims != dims {
        return Err(format!(
            "Invalid Qwen2.5-Omni audio tensor {name}: shape {:?}; expected {:?}",
            info.dims, dims
        ));
    }
    if info.ggml_type != kind
        && !(matches!(kind, GGMLType::F16) && matches!(info.ggml_type, GGMLType::BF16))
        && !(matches!(kind, GGMLType::BF16) && matches!(info.ggml_type, GGMLType::F16))
    {
        return Err(format!(
            "Invalid Qwen2.5-Omni audio tensor {name}: type {:?}; expected {:?} (or its BF16/F16 sibling)",
            info.ggml_type, kind
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo, TensorSource};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[derive(Default)]
    struct MapTensorSource {
        metadata: HashMap<String, MetaValue>,
        tensors: HashMap<String, TensorInfo>,
    }

    impl TensorSource for MapTensorSource {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.metadata.get(key)
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.tensors.get(name)
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    fn add_tensor(
        source: &mut MapTensorSource,
        name: impl Into<String>,
        dims: &[u64],
        kind: GGMLType,
    ) {
        let name = name.into();
        source.tensors.insert(
            name.clone(),
            TensorInfo {
                name,
                dims: dims.to_vec(),
                ggml_type: kind,
                offset: 0,
            },
        );
    }

    fn valid_source() -> MapTensorSource {
        let mut source = MapTensorSource {
            metadata: HashMap::from([
                (
                    "general.architecture".into(),
                    MetaValue::String("clip".into()),
                ),
                ("general.type".into(), MetaValue::String("mmproj".into())),
                ("clip.has_audio_encoder".into(), MetaValue::Bool(true)),
                (
                    "clip.projector_type".into(),
                    MetaValue::String("qwen2.5o".into()),
                ),
                (
                    "clip.audio.embedding_length".into(),
                    MetaValue::Uint32(1280),
                ),
                (
                    "clip.audio.feed_forward_length".into(),
                    MetaValue::Uint32(5120),
                ),
                ("clip.audio.block_count".into(), MetaValue::Uint32(32)),
                (
                    "clip.audio.attention.head_count".into(),
                    MetaValue::Uint32(20),
                ),
                ("clip.audio.num_mel_bins".into(), MetaValue::Uint32(128)),
                (
                    "clip.audio.attention.layer_norm_epsilon".into(),
                    MetaValue::Float32(1e-5),
                ),
                ("clip.audio.n_window".into(), MetaValue::Uint32(100)),
            ]),
            tensors: HashMap::new(),
        };
        add_tensor(
            &mut source,
            "a.position_embd.weight",
            &[1280, 1500],
            GGMLType::F32,
        );
        add_tensor(
            &mut source,
            "a.conv1d.1.weight",
            &[3, 128, 1280],
            GGMLType::F16,
        );
        add_tensor(&mut source, "a.conv1d.1.bias", &[1, 1280], GGMLType::F32);
        add_tensor(
            &mut source,
            "a.conv1d.2.weight",
            &[3, 1280, 1280],
            GGMLType::F16,
        );
        add_tensor(&mut source, "a.conv1d.2.bias", &[1, 1280], GGMLType::F32);
        for layer in 0..32 {
            let prefix = format!("a.blk.{layer}");
            for name in ["attn_q", "attn_k", "attn_v", "attn_out"] {
                add_tensor(
                    &mut source,
                    format!("{prefix}.{name}.weight"),
                    &[1280, 1280],
                    GGMLType::F16,
                );
            }
            for name in ["attn_q", "attn_v", "attn_out"] {
                add_tensor(
                    &mut source,
                    format!("{prefix}.{name}.bias"),
                    &[1280],
                    GGMLType::F32,
                );
            }
            for name in ["ln1", "ln2"] {
                add_tensor(
                    &mut source,
                    format!("{prefix}.{name}.weight"),
                    &[1280],
                    GGMLType::F32,
                );
                add_tensor(
                    &mut source,
                    format!("{prefix}.{name}.bias"),
                    &[1280],
                    GGMLType::F32,
                );
            }
            add_tensor(
                &mut source,
                format!("{prefix}.ffn_up.weight"),
                &[1280, 5120],
                GGMLType::F16,
            );
            add_tensor(
                &mut source,
                format!("{prefix}.ffn_up.bias"),
                &[5120],
                GGMLType::F32,
            );
            add_tensor(
                &mut source,
                format!("{prefix}.ffn_down.weight"),
                &[5120, 1280],
                GGMLType::F16,
            );
            add_tensor(
                &mut source,
                format!("{prefix}.ffn_down.bias"),
                &[1280],
                GGMLType::F32,
            );
        }
        for name in ["weight", "bias"] {
            add_tensor(
                &mut source,
                format!("a.post_ln.{name}"),
                &[1280],
                GGMLType::F32,
            );
        }
        add_tensor(&mut source, "mm.a.fc.weight", &[1280, 1024], GGMLType::F16);
        add_tensor(&mut source, "mm.a.fc.bias", &[1024], GGMLType::F32);
        source
    }

    #[test]
    fn config_matches_qwen25_omni_and_accepts_bias_free_keys() {
        let config = Qwen25OmniAudioConfig::from_source(&valid_source()).unwrap();
        assert_eq!(config.hidden, 1280);
        assert_eq!(config.ffn, 5120);
        assert_eq!(config.layers, 32);
        assert_eq!(config.heads, 20);
        assert_eq!(config.mel_bins, 128);
        assert_eq!(config.window, 100);
        assert_eq!(config.projection, 1024);
    }

    #[test]
    fn audio_layout_uses_whisper_30_second_chunks() {
        let layout = AudioLayout::for_real_frames(201).unwrap();
        assert_eq!(layout.padded_mel_frames, 3000);
        assert_eq!(layout.post_conv_tokens, 1500);
        assert_eq!(layout.output_rows, 750);
    }

    #[test]
    fn audio_layout_rejects_a_second_whisper_chunk() {
        assert_eq!(AudioLayout::for_real_frames(1).unwrap().output_rows, 750);
        assert_eq!(AudioLayout::for_real_frames(3000).unwrap().output_rows, 750);
        assert!(AudioLayout::for_real_frames(3001).is_err());
    }

    #[test]
    #[ignore = "requires Jina audio mmproj, WAV, and pinned llama.cpp projected sidecar"]
    fn jina_audio_projection_matches_llama_cpp_bits() {
        let read_words = |name: &str| {
            let bytes = std::fs::read(std::env::var(name).expect(name)).unwrap();
            assert_eq!(bytes.len() % 4, 0);
            bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let wav = std::fs::read(std::env::var("RMI_JINA_AUDIO_WAV").unwrap()).unwrap();
        let pcm = crate::models::qwen3::asr::audio_processor::decode_pcm16_wav(&wav).unwrap();
        let mmproj = std::env::var("RMI_JINA_AUDIO_MMPROJ").unwrap();
        let source: Arc<dyn TensorSource> =
            Arc::new(crate::GGUFLoader::from_file(std::path::Path::new(&mmproj)).unwrap());
        let actual = encode_audio(source, &pcm, 1).unwrap();
        let oracle = read_words("RMI_JINA_AUDIO_ORACLE_PROJECTED");
        assert_eq!(actual.len(), 750 * 1024);
        assert_eq!(actual.len(), oracle.len());
        for (index, (actual, oracle)) in actual.iter().zip(oracle).enumerate() {
            assert_eq!(actual.to_bits(), oracle, "projected value {index}");
        }
    }

    #[test]
    fn conv1d_uses_same_padding_and_requested_stride() {
        let weights = [1.0f32, 10.0, 100.0]
            .map(crate::ops::f32_to_f16)
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let mut output = Vec::new();

        conv1d_same_f16(&[1.0, 2.0, 3.0], 3, 1, 1, &weights, &[0.0], 1, &mut output).unwrap();
        assert_eq!(output, [210.0, 321.0, 32.0]);

        conv1d_same_f16(&[1.0, 2.0, 3.0], 3, 1, 1, &weights, &[0.0], 2, &mut output).unwrap();
        assert_eq!(output, [210.0, 32.0]);
    }

    #[test]
    fn average_pool_reduces_adjacent_audio_rows() {
        let mut output = Vec::new();
        average_pool_pairs(
            &[1.0, 10.0, 3.0, 20.0, 5.0, 30.0, 9.0, 50.0],
            2,
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [2.0, 15.0, 7.0, 40.0]);
    }

    #[test]
    fn encode_audio_rejects_empty_samples_before_loading_weights() {
        let error = encode_audio(Arc::new(valid_source()), &[], 1).unwrap_err();
        assert!(error.contains("non-empty"), "{error}");
    }

    #[test]
    fn encode_audio_dispatches_qwen3a_to_the_qwen3_audio_tower() {
        let mut source = valid_source();
        source.metadata.insert(
            "clip.projector_type".into(),
            MetaValue::String("qwen3vl_merger".into()),
        );
        source.metadata.insert(
            "clip.audio.projector_type".into(),
            MetaValue::String("qwen3a".into()),
        );
        source
            .metadata
            .insert("clip.audio.embedding_length".into(), MetaValue::Uint32(896));
        source.metadata.insert(
            "clip.audio.feed_forward_length".into(),
            MetaValue::Uint32(3584),
        );
        source
            .metadata
            .insert("clip.audio.block_count".into(), MetaValue::Uint32(18));
        source.metadata.insert(
            "clip.audio.attention.head_count".into(),
            MetaValue::Uint32(14),
        );
        source
            .metadata
            .insert("clip.audio.projection_dim".into(), MetaValue::Uint32(1024));
        let error = encode_audio(Arc::new(source), &[0.0, 0.0, 0.0], 1).unwrap_err();
        assert!(
            error.contains("Qwen3A") || error.contains("Qwen3Audio"),
            "{error}"
        );
    }
}
