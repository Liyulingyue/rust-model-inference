use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use crate::models::qwen3::asr::audio_processor::{compute_log_mel, HOP};
use crate::models::qwen3::asr::mel_encoder::{
    add_residual, apply_gelu_erf, checked_product, full_attention_into, layer_norm_rows,
    load_f32_tensor, reserved_f32, resize_f32, static_tensor, AudioLinear, LayerNormWeights,
};
use crate::ops::dot_f16_f16_bytes;
use std::sync::Arc;

const MEL_CHUNK: usize = 200;

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
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err("Audio samples must be non-empty and finite".into());
        }
        let mel =
            compute_log_mel(samples).map_err(|error| format!("Audio Mel error: {error:?}"))?;
        let real_frames = samples.len().div_ceil(HOP);
        if mel.frames < real_frames || mel.normalized.len() != mel.frames * self.config.mel_bins {
            return Err("Audio Mel output is shorter than the real sample duration".into());
        }
        let layout = AudioLayout::for_real_frames(real_frames)?;
        if layout.output_rows == 0 {
            return Err("Audio is too short to produce an embedding row".into());
        }
        let mut input = reserved_f32(
            "Qwen2.5-Omni Mel input",
            checked_product(
                "Qwen2.5-Omni Mel input",
                layout.padded_mel_frames,
                self.config.mel_bins,
            )?,
        )?;
        for frame in 0..real_frames {
            for mel_bin in 0..self.config.mel_bins {
                input[frame * self.config.mel_bins + mel_bin] =
                    mel.normalized[mel_bin * mel.frames + frame];
            }
        }

        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "omni.audio.mel",
            None,
            &[layout.padded_mel_frames, self.config.mel_bins],
            &input,
        ));

        let mut hidden = reserved_f32(
            "Qwen2.5-Omni convolution output",
            checked_product(
                "Qwen2.5-Omni convolution output",
                layout.post_conv_tokens,
                self.config.hidden,
            )?,
        )?;
        let mut conv1 = Vec::new();
        let mut conv2 = Vec::new();
        let mel_chunk_values = MEL_CHUNK * self.config.mel_bins;
        let hidden_chunk_values = self.config.window * self.config.hidden;
        for chunk in 0..layout.padded_mel_frames / MEL_CHUNK {
            let input_start = chunk * mel_chunk_values;
            conv1d_same_f16(
                &input[input_start..input_start + mel_chunk_values],
                MEL_CHUNK,
                self.conv1.input,
                self.conv1.output,
                self.conv1.weight,
                &self.conv1.bias,
                1,
                &mut conv1,
            )?;
            apply_gelu_erf(&mut conv1)?;
            let chunk_start = chunk * MEL_CHUNK;
            let valid_rows = real_frames.saturating_sub(chunk_start).min(MEL_CHUNK);
            conv1[valid_rows * self.config.hidden..].fill(0.0);
            conv1d_same_f16(
                &conv1,
                MEL_CHUNK,
                self.conv2.input,
                self.conv2.output,
                self.conv2.weight,
                &self.conv2.bias,
                2,
                &mut conv2,
            )?;
            apply_gelu_erf(&mut conv2)?;
            let output_start = chunk * hidden_chunk_values;
            hidden[output_start..output_start + hidden_chunk_values].copy_from_slice(&conv2);
        }

        for token in 0..layout.post_conv_tokens {
            let position = token % self.config.window;
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
            layer_norm_rows(
                &hidden,
                layout.post_conv_tokens,
                &layer.ln1,
                self.config.epsilon,
                &mut normed,
            )?;
            layer
                .q
                .project_f16(&normed, layout.post_conv_tokens, &mut q)?;
            layer
                .k
                .project_f16(&normed, layout.post_conv_tokens, &mut k)?;
            layer
                .v
                .project_f16(&normed, layout.post_conv_tokens, &mut v)?;
            block_attention_into(
                &q,
                &k,
                &v,
                layout.post_conv_tokens,
                real_frames.div_ceil(2),
                self.config.heads,
                head_dim,
                self.config.window,
                &mut scores,
                &mut attention,
            )?;
            layer
                .output
                .project_f16(&attention, layout.post_conv_tokens, &mut update)?;
            add_residual(&mut hidden, &update)?;
            layer_norm_rows(
                &hidden,
                layout.post_conv_tokens,
                &layer.ln2,
                self.config.epsilon,
                &mut normed,
            )?;
            layer
                .up
                .project_f16(&normed, layout.post_conv_tokens, &mut ffn_up)?;
            apply_gelu_erf(&mut ffn_up)?;
            layer
                .down
                .project_f16(&ffn_up, layout.post_conv_tokens, &mut ffn_down)?;
            add_residual(&mut hidden, &ffn_down)?;
        }

        let mut pooled = Vec::new();
        average_pool_pairs(&hidden, self.config.hidden, &mut pooled)?;
        pooled.truncate(layout.output_rows * self.config.hidden);
        layer_norm_rows(
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
            .project_f16(&normed, layout.output_rows, &mut projected)?;
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
    _threads: usize,
) -> Result<Vec<f32>, String> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err("Audio samples must be non-empty and finite".into());
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
        let window = require_u32(source, "clip.audio.n_window", 100)? as usize;
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
        if projector.dims.first() != Some(&(hidden as u64))
            || projector.dims.len() != 2
            || projector.ggml_type != GGMLType::F16
        {
            return Err("Invalid Qwen2.5-Omni audio tensor: mm.a.fc.weight".into());
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
        let padded_mel_frames = real_mel_frames
            .checked_add(MEL_CHUNK - 1)
            .ok_or("Audio Mel frame count overflow")?
            / MEL_CHUNK
            * MEL_CHUNK;
        Ok(Self {
            padded_mel_frames,
            post_conv_tokens: padded_mel_frames / 2,
            output_rows: real_mel_frames
                .checked_add(1)
                .ok_or("Audio output row count overflow")?
                / 4,
        })
    }
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

#[allow(clippy::too_many_arguments)]
fn block_attention_into(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    tokens: usize,
    valid_tokens: usize,
    heads: usize,
    head_dim: usize,
    window: usize,
    scores: &mut Vec<f32>,
    output: &mut Vec<f32>,
) -> Result<(), String> {
    if window == 0
        || tokens == 0
        || tokens % window != 0
        || valid_tokens == 0
        || valid_tokens > tokens
    {
        return Err("Invalid Qwen2.5-Omni block-attention shape".into());
    }
    let width = checked_product("Qwen2.5-Omni attention width", heads, head_dim)?;
    let len = checked_product("Qwen2.5-Omni attention values", tokens, width)?;
    if query.len() != len || key.len() != len || value.len() != len {
        return Err("Invalid Qwen2.5-Omni block-attention tensors".into());
    }
    resize_f32(output, "Qwen2.5-Omni attention output", len)?;
    output.fill(0.0);
    let mut chunk_output = Vec::new();
    for start in (0..tokens).step_by(window) {
        let chunk_rows = valid_tokens.saturating_sub(start).min(window);
        if chunk_rows == 0 {
            break;
        }
        let chunk_len = checked_product("Qwen2.5-Omni attention chunk", chunk_rows, width)?;
        let offset = start * width;
        full_attention_into(
            &query[offset..offset + chunk_len],
            &key[offset..offset + chunk_len],
            &value[offset..offset + chunk_len],
            chunk_rows,
            heads,
            head_dim,
            scores,
            &mut chunk_output,
        )?;
        output[offset..offset + chunk_len].copy_from_slice(&chunk_output);
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
    if info.dims != dims || info.ggml_type != kind {
        return Err(format!(
            "Invalid Qwen2.5-Omni audio tensor {name}: shape {:?} type {:?}; expected {:?} {:?}",
            info.dims, info.ggml_type, dims, kind
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
    fn audio_layout_chunks_200_mel_frames() {
        let layout = AudioLayout::for_real_frames(201).unwrap();
        assert_eq!(layout.padded_mel_frames, 400);
        assert_eq!(layout.post_conv_tokens, 200);
        assert_eq!(layout.output_rows, 50);
    }

    #[test]
    fn output_rows_follow_real_mel_frames_after_stride_and_avg_pool() {
        assert_eq!(AudioLayout::for_real_frames(1).unwrap().output_rows, 0);
        assert_eq!(AudioLayout::for_real_frames(2).unwrap().output_rows, 0);
        assert_eq!(AudioLayout::for_real_frames(3).unwrap().output_rows, 1);
        assert_eq!(AudioLayout::for_real_frames(1100).unwrap().output_rows, 275);
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
    fn block_attention_never_reads_an_adjacent_chunk() {
        let query = [1.0, 1.0, 1.0, 1.0];
        let key = [1.0, 1.0, 1.0, 1.0];
        let value = [1.0, 3.0, 100.0, 300.0];
        let mut scores = Vec::new();
        let mut output = Vec::new();

        block_attention_into(
            &query,
            &key,
            &value,
            4,
            4,
            1,
            1,
            2,
            &mut scores,
            &mut output,
        )
        .unwrap();

        assert_eq!(output, [2.0, 2.0, 200.0, 200.0]);
    }

    #[test]
    fn partial_audio_chunk_does_not_attend_to_padding() {
        let query = [1.0, 1.0, 1.0, 1.0];
        let key = [1.0, 1.0, 1.0, 1.0];
        let value = [1.0, 3.0, 100.0, 300.0];
        let mut scores = Vec::new();
        let mut output = Vec::new();

        block_attention_into(
            &query,
            &key,
            &value,
            4,
            2,
            1,
            1,
            4,
            &mut scores,
            &mut output,
        )
        .unwrap();

        assert_eq!(&output[..2], [2.0, 2.0]);
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
}
