//! Audio8 ASR Infinite: Voxtral Realtime audio tower and frame projector.

pub mod mel;
pub mod streaming;
pub mod text;

use crate::core::tensor::{load_f32_tensor, GGMLType, MetaValue, TensorSource};
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::{gelu_erf, silu};
use std::sync::Arc;

unsafe extern "C" {
    fn powf(base: f32, exponent: f32) -> f32;
    fn cosf(value: f32) -> f32;
    fn sinf(value: f32) -> f32;
}

const AUDIO_WIDTH: usize = 1280;
const AUDIO_HEADS: usize = 32;
const HEAD_WIDTH: usize = 64;
const AUDIO_Q_WIDTH: usize = AUDIO_HEADS * HEAD_WIDTH;
const AUDIO_FFN: usize = 5120;
const TEXT_WIDTH: usize = 2048;
const AUDIO_LAYERS: usize = 32;
const CACHE_FRAMES: usize = 750;

fn bf16(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 2;
    f32::from_bits(u32::from(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])) << 16)
}

pub(crate) fn audio_rms_norm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let mut sum = 0.0f32;
    for &value in input {
        sum += value * value;
    }
    let scale = 1.0 / (sum / input.len() as f32 + eps).sqrt();
    for ((output, &value), &weight) in output.iter_mut().zip(input).zip(weight) {
        *output = weight * (value * scale);
    }
}

pub(crate) fn audio8_rope(values: &mut [f32], head_width: usize, position: usize) {
    debug_assert_eq!(values.len() % head_width, 0);
    for i in 0..head_width / 2 {
        let frequency = 1.0 / unsafe { powf(1_000_000.0, (2 * i) as f32 / head_width as f32) };
        let theta = position as f32 * frequency;
        let cosine = unsafe { cosf(theta) };
        let sine = unsafe { sinf(theta) };
        for head in values.chunks_exact_mut(head_width) {
            let a = head[i];
            let b = head[i + head_width / 2];
            head[i] = a * cosine + (-b) * sine;
            head[i + head_width / 2] = b * cosine + a * sine;
        }
    }
}

pub(crate) fn audio_dot(left: &[f32], right: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&left, &right) in left.iter().zip(right) {
        sum += left * right;
    }
    sum
}

/// Weight formats the runtime can execute for an Audio8 projection.
///
/// The converter emits BF16, and `tools/converter/utils/quantize_gguf.py`
/// derives the quantized variants from that file. The kernels behind these
/// types decode a block row the way llama.cpp does, so a quantized file
/// runs on the same code path as the BF16 one, only with block rounding.
const SUPPORTED_WEIGHT_TYPES: &[GGMLType] = &[
    GGMLType::F32,
    GGMLType::F16,
    GGMLType::BF16,
    GGMLType::Q8_0,
    GGMLType::Q4K,
    GGMLType::Q5K,
    GGMLType::Q6K,
];

fn check_tensor(source: &dyn TensorSource, name: &str, shape: &[u64]) -> Result<GGMLType, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("missing Audio8 tensor {name}"))?;
    if info.dims != shape {
        return Err(format!(
            "invalid Audio8 tensor {name}: {:?} {:?}; expected {shape:?}",
            info.ggml_type, info.dims
        ));
    }
    if !SUPPORTED_WEIGHT_TYPES.contains(&info.ggml_type) {
        return Err(format!(
            "unsupported Audio8 tensor type {name}: {:?}; expected one of {SUPPORTED_WEIGHT_TYPES:?}",
            info.ggml_type
        ));
    }
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("invalid Audio8 tensor shape: {name}"))?,
    )
    .map_err(|_| format!("Audio8 tensor byte size does not fit usize: {name}"))?;
    if source
        .tensor_slice(name)
        .is_none_or(|bytes| bytes.len() != expected)
    {
        return Err(format!("invalid Audio8 tensor bytes: {name}"));
    }
    Ok(info.ggml_type)
}

fn vector(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    // Norms and biases are read as float vectors; the runtime decodes F32,
    // F16 and BF16 into them, and the converter leaves every 1-D tensor in
    // its source type.
    load_f32_tensor(source, name, &[len as u64])
}

fn matrix(
    source: &dyn TensorSource,
    name: &str,
    input: usize,
    output: usize,
) -> Result<Weight<'static>, String> {
    let ggml_type = check_tensor(source, name, &[input as u64, output as u64])?;
    let bytes = source.tensor_slice(name).unwrap();
    // SAFETY: Audio8 owners retain the source Arc for the lifetime of every weight.
    let bytes: &'static [u8] = unsafe { std::mem::transmute(bytes) };
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes, ggml_type, input, output,
    )))
}

fn linear(weight: &Weight<'_>, input: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let mut output = vec![0.0; weight.n_out];
    weight
        .kernel
        .forward(input, &mut output, weight.n_in, weight.n_out);
    if let Some(bias) = bias {
        for (value, bias) in output.iter_mut().zip(bias) {
            *value += *bias;
        }
    }
    output
}

struct Conv {
    weight: &'static [u8],
    bias: Vec<f32>,
    input: usize,
    output: usize,
    stride: usize,
    left_pad: usize,
}

impl Conv {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        input: usize,
        output: usize,
        stride: usize,
    ) -> Result<Self, String> {
        let name = format!("{prefix}.weight");
        // The convolution decoder reads the weight as raw BF16 pairs, so this
        // tensor stays BF16 even in a quantized file. It is a rounding kernel
        // over 128 mel bins, not a projection: the converter keeps it and
        // every other tensor whose row width is not block-aligned.
        let ggml_type = check_tensor(source, &name, &[3, input as u64, output as u64])?;
        if ggml_type != GGMLType::BF16 {
            return Err(format!(
                "unsupported Audio8 convolution type {name}: {ggml_type:?}; expected BF16"
            ));
        }
        let weight: &'static [u8] =
            unsafe { std::mem::transmute(source.tensor_slice(&name).unwrap()) };
        Ok(Self {
            weight,
            bias: vector(source, &format!("{prefix}.bias"), output)?,
            input,
            output,
            stride,
            left_pad: 3 - stride,
        })
    }

    fn forward(&self, input: &[f32], frames: usize) -> Result<(Vec<f32>, usize), String> {
        if input.len() != frames * self.input || frames + self.left_pad < 3 {
            return Err("invalid Audio8 convolution input".into());
        }
        let out_frames = (frames + self.left_pad - 3) / self.stride + 1;
        let mut output = vec![0.0; out_frames * self.output];
        for time in 0..out_frames {
            for channel in 0..self.output {
                let mut value = self.bias[channel];
                for feature in 0..self.input {
                    for kernel in 0..3 {
                        let pos = time * self.stride + kernel;
                        if pos >= self.left_pad {
                            let source_time = pos - self.left_pad;
                            if source_time < frames {
                                value += input[source_time * self.input + feature]
                                    * bf16(
                                        self.weight,
                                        (channel * self.input + feature) * 3 + kernel,
                                    );
                            }
                        }
                    }
                }
                output[time * self.output + channel] = gelu_erf(value);
            }
        }
        Ok((output, out_frames))
    }
}

struct AudioLayer {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    q: Weight<'static>,
    k: Weight<'static>,
    v: Weight<'static>,
    o: Weight<'static>,
    gate: Weight<'static>,
    up: Weight<'static>,
    down: Weight<'static>,
    q_bias: Vec<f32>,
    v_bias: Vec<f32>,
    o_bias: Vec<f32>,
    down_bias: Vec<f32>,
}

pub struct Audio8Encoder {
    _source: Arc<dyn TensorSource>,
    conv1: Conv,
    conv2: Conv,
    layers: Vec<AudioLayer>,
    norm: Vec<f32>,
    project1: Weight<'static>,
    project2: Weight<'static>,
    pub frame_embedding: Vec<f32>,
}

pub struct Audio8State {
    key: Vec<Vec<f32>>,
    value: Vec<Vec<f32>>,
    position: usize,
}

impl Audio8State {
    pub fn new() -> Self {
        Self {
            key: vec![Vec::new(); AUDIO_LAYERS],
            value: vec![Vec::new(); AUDIO_LAYERS],
            position: 0,
        }
    }

    fn check_frames(&self, frames: usize) -> Result<(), String> {
        if self
            .position
            .checked_add(frames)
            .is_some_and(|end| end <= 1500)
        {
            Ok(())
        } else {
            Err(
                "Audio8 audio tower exceeds 1500 positions; rolling RoPE rebase is not implemented"
                    .into(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Audio8State;

    #[test]
    fn audio_tower_rejects_positions_beyond_its_context() {
        let mut state = Audio8State::new();
        state.position = 1496;
        assert!(state.check_frames(4).is_ok());
        assert!(state.check_frames(8).is_err());
    }

    /// A projection must load from every weight format the converter can
    /// emit and must compute the same product the float weights do, up to
    /// the block rounding the format introduces.
    #[test]
    fn quantized_projections_load_and_match_the_float_weights() {
        use crate::core::tensor::{GGMLType, MetaValue, TensorInfo, TensorSource};

        const N_IN: usize = 256;
        const N_OUT: usize = 64;

        struct Source {
            info: TensorInfo,
            bytes: Vec<u8>,
        }
        impl TensorSource for Source {
            fn metadata(&self, _key: &str) -> Option<&MetaValue> {
                None
            }
            fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
                Some(&self.info)
            }
            fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
                Some(&self.bytes)
            }
        }

        // One float row per output, `value = (row + column) / 8`.
        let weights: Vec<f32> = (0..N_OUT)
            .flat_map(|row| (0..N_IN).map(move |column| (row + column) as f32 / 8.0))
            .collect();
        let input: Vec<f32> = (0..N_IN).map(|index| (index % 7) as f32 / 4.0).collect();
        let expected: Vec<f32> = (0..N_OUT)
            .map(|row| {
                weights[row * N_IN..(row + 1) * N_IN]
                    .iter()
                    .zip(&input)
                    .map(|(weight, value)| weight * value)
                    .sum()
            })
            .collect();

        for (ggml_type, payload, tolerance) in [
            (GGMLType::BF16, encode::bf16(&weights), 1e-3),
            (GGMLType::F32, encode::f32(&weights), 1e-3),
            (GGMLType::Q8_0, encode::q8_0(&weights), 1e-2),
        ] {
            let source = Source {
                info: TensorInfo {
                    name: "projection".into(),
                    dims: vec![N_IN as u64, N_OUT as u64],
                    ggml_type,
                    offset: 0,
                },
                bytes: payload,
            };
            let weight = super::matrix(&source, "projection", N_IN, N_OUT).unwrap();
            let output = super::linear(&weight, &input, None);
            for (index, (got, want)) in output.iter().zip(&expected).enumerate() {
                let error = (got - want).abs();
                assert!(
                    error <= tolerance * want.abs().max(1.0),
                    "{ggml_type:?} row {index}: {got} vs {want} (error {error})"
                );
            }
        }
    }

    /// The k-quant formats are encoded by the converter, not by the runtime,
    /// so the runtime side only has to accept the payload and decode it.
    /// An all-zero super-block is a valid one: it dequantizes to zeros.
    #[test]
    fn k_quant_projections_load_and_decode() {
        use crate::core::tensor::{GGMLType, MetaValue, TensorInfo, TensorSource};

        const N_IN: usize = 256;
        const N_OUT: usize = 8;

        struct Source {
            info: TensorInfo,
            bytes: Vec<u8>,
        }
        impl TensorSource for Source {
            fn metadata(&self, _key: &str) -> Option<&MetaValue> {
                None
            }
            fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
                Some(&self.info)
            }
            fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
                Some(&self.bytes)
            }
        }

        for (ggml_type, block_bytes) in [
            (GGMLType::Q4K, crate::ops::quant::BLOCK_Q4K_SIZE),
            (GGMLType::Q5K, crate::ops::quant::BLOCK_Q5K_SIZE),
            (GGMLType::Q6K, crate::ops::quant::BLOCK_Q6K_SIZE),
        ] {
            let source = Source {
                info: TensorInfo {
                    name: "projection".into(),
                    dims: vec![N_IN as u64, N_OUT as u64],
                    ggml_type,
                    offset: 0,
                },
                bytes: vec![0; block_bytes * N_OUT],
            };
            let weight = super::matrix(&source, "projection", N_IN, N_OUT).unwrap();
            let output = super::linear(&weight, &vec![1.0; N_IN], None);
            assert_eq!(output.len(), N_OUT);
            assert!(output.iter().all(|value| value.is_finite()));
        }
    }

    /// Minimal encoders for the test above. They follow the same layouts
    /// the converter writes, so the loader sees a real payload.
    mod encode {
        pub fn f32(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }

        pub fn bf16(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        }

        /// GGML Q8_0: per 32-element block, an f16 scale and 32 int8
        /// values.
        pub fn q8_0(values: &[f32]) -> Vec<u8> {
            let mut out = Vec::new();
            for block in values.chunks(32) {
                let amax = block.iter().fold(0.0f32, |max, value| max.max(value.abs()));
                let scale = crate::ops::f32_to_f16(amax / 127.0);
                out.extend_from_slice(&scale.to_le_bytes());
                let scale = half::f16::from_bits(scale).to_f32();
                for value in block {
                    let quantized = (value / scale).round().clamp(-128.0, 127.0) as i8;
                    out.push(quantized as u8);
                }
            }
            out
        }
    }

    #[cfg(feature = "parity-trace")]
    #[test]
    fn trace_real_audio_group() {
        let (Some(gguf), Some(mel)) = (
            std::env::var_os("RMI_AUDIO8_GGUF"),
            std::env::var_os("RMI_AUDIO8_MEL"),
        ) else {
            return;
        };
        use crate::core::loader::GGUFLoader;
        use crate::core::tensor::TensorSource;
        use crate::core::tokenizer::BPETokenizer;
        use crate::models::audio8::text::{Audio8TextDecoder, Audio8TextSession};
        use std::sync::Arc;
        let bytes = std::fs::read(mel).unwrap();
        assert!(bytes.len() >= 128 * 8 * 4 && bytes.len() % (128 * 8 * 4) == 0);
        let frames = bytes.len() / (128 * 4);
        let groups = frames / 8;
        assert!(groups <= 4);
        let input: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let source: Arc<dyn TensorSource> = Arc::new(GGUFLoader::from_file(gguf).unwrap());
        let encoder = super::Audio8Encoder::from_source(Arc::clone(&source)).unwrap();
        let result = encoder
            .encode_window(&input, frames, groups * 4, &mut Audio8State::new())
            .unwrap();
        assert_eq!(result.len(), groups * super::TEXT_WIDTH);
        assert!(result.iter().all(|value| value.is_finite()));
        let tokenizer =
            BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
        let mut token_ids = vec![
            tokenizer.token_id("<|im_start|>").unwrap(),
            tokenizer.token_id("[LANGUAGE_ZH]").unwrap(),
        ];
        token_ids.resize(groups, tokenizer.token_id("[STREAMING_PAD]").unwrap());
        let decoder = Audio8TextDecoder::from_source(source).unwrap();
        let condition = super::time_condition(6, &encoder.frame_embedding).unwrap();
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "audio8.time_condition",
            None,
            &[super::TEXT_WIDTH],
            &condition,
        ));
        let mut session = Audio8TextSession::new(&decoder, groups, &condition).unwrap();
        let embedding = decoder.embed_audio_tokens(&token_ids, &result).unwrap();
        let logits = session.forward_logits(&embedding).unwrap();
        assert!(logits.iter().all(|value| value.is_finite()));
    }
}

impl Audio8Encoder {
    pub fn from_source(source: Arc<dyn TensorSource>) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val);
        if arch != Some("audio8_asr_infinite") {
            return Err(format!("expected audio8_asr_infinite GGUF, got {arch:?}"));
        }
        let raw = source
            .metadata("audio8_asr_infinite.config_json")
            .and_then(MetaValue::to_string_val)
            .ok_or("missing Audio8 config JSON")?;
        let cfg: serde_json::Value =
            serde_json::from_str(raw).map_err(|e| format!("invalid Audio8 config: {e}"))?;
        for (path, expected) in [
            ("/weight_format_version", 2),
            ("/audio_config/hidden_size", AUDIO_WIDTH),
            ("/audio_config/num_hidden_layers", AUDIO_LAYERS),
            ("/audio_config/num_attention_heads", AUDIO_HEADS),
            ("/audio_config/head_dim", HEAD_WIDTH),
            ("/audio_config/num_mel_bins", 128),
            ("/text_config/hidden_size", TEXT_WIDTH),
            ("/text_config/num_hidden_layers", 36),
            ("/max_frame_len", 8),
        ] {
            if cfg.pointer(path).and_then(serde_json::Value::as_u64) != Some(expected as u64) {
                return Err(format!("unsupported Audio8 config {path}"));
            }
        }
        if cfg
            .pointer("/text_config/model_type")
            .and_then(serde_json::Value::as_str)
            != Some("qwen2")
        {
            return Err("Audio8 requires its Qwen2 text decoder".into());
        }
        let conv1 = Conv::load(
            source.as_ref(),
            "audio_tower.embedder.conv1",
            128,
            AUDIO_WIDTH,
            1,
        )?;
        let conv2 = Conv::load(
            source.as_ref(),
            "audio_tower.embedder.conv2",
            AUDIO_WIDTH,
            AUDIO_WIDTH,
            2,
        )?;
        let mut layers = Vec::with_capacity(AUDIO_LAYERS);
        for layer in 0..AUDIO_LAYERS {
            let prefix = format!("audio_tower.layers.{layer}");
            let attn = format!("{prefix}.self_attn");
            let mlp = format!("{prefix}.mlp");
            layers.push(AudioLayer {
                norm1: vector(
                    source.as_ref(),
                    &format!("{prefix}.self_attn_layer_norm.weight"),
                    AUDIO_WIDTH,
                )?,
                norm2: vector(
                    source.as_ref(),
                    &format!("{prefix}.final_layer_norm.weight"),
                    AUDIO_WIDTH,
                )?,
                q: matrix(
                    source.as_ref(),
                    &format!("{attn}.q_proj.weight"),
                    AUDIO_WIDTH,
                    AUDIO_Q_WIDTH,
                )?,
                k: matrix(
                    source.as_ref(),
                    &format!("{attn}.k_proj.weight"),
                    AUDIO_WIDTH,
                    AUDIO_Q_WIDTH,
                )?,
                v: matrix(
                    source.as_ref(),
                    &format!("{attn}.v_proj.weight"),
                    AUDIO_WIDTH,
                    AUDIO_Q_WIDTH,
                )?,
                o: matrix(
                    source.as_ref(),
                    &format!("{attn}.o_proj.weight"),
                    AUDIO_Q_WIDTH,
                    AUDIO_WIDTH,
                )?,
                gate: matrix(
                    source.as_ref(),
                    &format!("{mlp}.gate_proj.weight"),
                    AUDIO_WIDTH,
                    AUDIO_FFN,
                )?,
                up: matrix(
                    source.as_ref(),
                    &format!("{mlp}.up_proj.weight"),
                    AUDIO_WIDTH,
                    AUDIO_FFN,
                )?,
                down: matrix(
                    source.as_ref(),
                    &format!("{mlp}.down_proj.weight"),
                    AUDIO_FFN,
                    AUDIO_WIDTH,
                )?,
                q_bias: vector(
                    source.as_ref(),
                    &format!("{attn}.q_proj.bias"),
                    AUDIO_Q_WIDTH,
                )?,
                v_bias: vector(
                    source.as_ref(),
                    &format!("{attn}.v_proj.bias"),
                    AUDIO_Q_WIDTH,
                )?,
                o_bias: vector(source.as_ref(), &format!("{attn}.o_proj.bias"), AUDIO_WIDTH)?,
                down_bias: vector(
                    source.as_ref(),
                    &format!("{mlp}.down_proj.bias"),
                    AUDIO_WIDTH,
                )?,
            });
        }
        Ok(Self {
            norm: vector(source.as_ref(), "audio_tower.norm.weight", AUDIO_WIDTH)?,
            project1: matrix(
                source.as_ref(),
                "multi_modal_projector.linear_1.weight",
                8 * AUDIO_WIDTH,
                TEXT_WIDTH,
            )?,
            project2: matrix(
                source.as_ref(),
                "multi_modal_projector.linear_2.weight",
                TEXT_WIDTH,
                TEXT_WIDTH,
            )?,
            frame_embedding: load_f32_tensor(
                source.as_ref(),
                "frame_len_embedding.weight",
                &[TEXT_WIDTH as u64, 3],
            )?,
            _source: source,
            conv1,
            conv2,
            layers,
        })
    }

    pub fn encode_window(
        &self,
        mel: &[f32],
        mel_frames: usize,
        expected_frames: usize,
        state: &mut Audio8State,
    ) -> Result<Vec<f32>, String> {
        if mel.len() != 128 * mel_frames || expected_frames == 0 || expected_frames % 4 != 0 {
            return Err("invalid Audio8 mel window".into());
        }
        state.check_frames(expected_frames)?;
        let mut transposed = vec![0.0; mel.len()];
        for bin in 0..128 {
            for time in 0..mel_frames {
                transposed[time * 128 + bin] = mel[bin * mel_frames + time];
            }
        }
        let (conv1, frames1) = self.conv1.forward(&transposed, mel_frames)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "audio8.conv1",
            None,
            &[frames1, AUDIO_WIDTH],
            &conv1,
        ));
        let (conv2, frames2) = self.conv2.forward(&conv1, frames1)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "audio8.conv2",
            None,
            &[frames2, AUDIO_WIDTH],
            &conv2,
        ));
        if frames2 < expected_frames {
            return Err(format!(
                "Audio8 convolution returned {frames2} frames; expected {expected_frames}"
            ));
        }
        let mut hidden = Vec::with_capacity(expected_frames * AUDIO_WIDTH);
        for frame in conv2[(frames2 - expected_frames) * AUDIO_WIDTH..].chunks_exact(AUDIO_WIDTH) {
            hidden.extend_from_slice(&self.encode_frame(frame, state)?);
        }
        let mut projected = Vec::with_capacity(expected_frames / 4 * TEXT_WIDTH);
        for group in hidden.chunks_exact(4 * AUDIO_WIDTH) {
            let mut padded = vec![0.0; 8 * AUDIO_WIDTH];
            padded[..group.len()].copy_from_slice(group);
            let mut value = linear(&self.project1, &padded, None);
            for item in &mut value {
                *item = gelu_erf(*item);
            }
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "audio8.project1",
                None,
                &[TEXT_WIDTH],
                &value,
            ));
            let output = linear(&self.project2, &value, None);
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "audio8.project2",
                None,
                &[TEXT_WIDTH],
                &output,
            ));
            projected.extend(output);
        }
        Ok(projected)
    }

    fn encode_frame(&self, frame: &[f32], state: &mut Audio8State) -> Result<Vec<f32>, String> {
        let position = state.position;
        let mut hidden = frame.to_vec();
        for (index, layer) in self.layers.iter().enumerate() {
            let mut normed = vec![0.0; AUDIO_WIDTH];
            audio_rms_norm(&hidden, &layer.norm1, &mut normed, 1e-5);
            #[cfg(feature = "parity-trace")]
            if position <= 1 && index == 0 {
                crate::parity_trace::report(crate::parity_trace::checkpoint_at(
                    "audio8.norm1",
                    Some(index),
                    Some(position),
                    &[AUDIO_WIDTH],
                    &normed,
                ));
            }
            let mut q = linear(&layer.q, &normed, Some(&layer.q_bias));
            let mut k = linear(&layer.k, &normed, None);
            let v = linear(&layer.v, &normed, Some(&layer.v_bias));
            audio8_rope(&mut q, HEAD_WIDTH, position);
            audio8_rope(&mut k, HEAD_WIDTH, position);
            #[cfg(feature = "parity-trace")]
            if position <= 1 && index == 0 {
                for (name, values) in [("audio8.query", &q), ("audio8.key", &k)] {
                    crate::parity_trace::report(crate::parity_trace::checkpoint_at(
                        name,
                        Some(index),
                        Some(position),
                        &[AUDIO_Q_WIDTH],
                        values,
                    ));
                }
            }
            let slot = position % CACHE_FRAMES;
            let end = (slot + 1) * AUDIO_Q_WIDTH;
            if state.key[index].len() < end {
                state.key[index].resize(end, 0.0);
                state.value[index].resize(end, 0.0);
            }
            state.key[index][slot * AUDIO_Q_WIDTH..end].copy_from_slice(&k);
            state.value[index][slot * AUDIO_Q_WIDTH..end].copy_from_slice(&v);
            let start_pos = (position + 1).saturating_sub(CACHE_FRAMES);
            let mut attention = vec![0.0; AUDIO_Q_WIDTH];
            for head in 0..AUDIO_HEADS {
                let head_start = head * HEAD_WIDTH;
                let query = &q[head_start..head_start + HEAD_WIDTH];
                let mut scores = Vec::with_capacity(position + 1 - start_pos);
                let mut max = f32::NEG_INFINITY;
                for past in start_pos..=position {
                    let base = (past % CACHE_FRAMES) * AUDIO_Q_WIDTH + head_start;
                    let score =
                        audio_dot(query, &state.key[index][base..base + HEAD_WIDTH]) * 0.125;
                    max = max.max(score);
                    scores.push(score);
                }
                let mut sum = 0.0f32;
                for score in &mut scores {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                for (offset, score) in scores.iter().enumerate() {
                    let past = start_pos + offset;
                    let base = (past % CACHE_FRAMES) * AUDIO_Q_WIDTH + head_start;
                    for dim in 0..HEAD_WIDTH {
                        attention[head_start + dim] +=
                            (*score / sum) * state.value[index][base + dim];
                    }
                }
            }
            #[cfg(feature = "parity-trace")]
            if position <= 1 && index == 0 {
                crate::parity_trace::report(crate::parity_trace::checkpoint_at(
                    "audio8.attention",
                    Some(index),
                    Some(position),
                    &[AUDIO_Q_WIDTH],
                    &attention,
                ));
            }
            let projected = linear(&layer.o, &attention, Some(&layer.o_bias));
            for (item, update) in hidden.iter_mut().zip(projected) {
                *item += update;
            }
            audio_rms_norm(&hidden, &layer.norm2, &mut normed, 1e-5);
            let mut gate = linear(&layer.gate, &normed, None);
            let up = linear(&layer.up, &normed, None);
            for (gate, up) in gate.iter_mut().zip(up) {
                *gate = silu(*gate) * up;
            }
            let down = linear(&layer.down, &gate, Some(&layer.down_bias));
            for (item, update) in hidden.iter_mut().zip(down) {
                *item += update;
            }
            #[cfg(feature = "parity-trace")]
            if position <= 1 {
                crate::parity_trace::report(crate::parity_trace::checkpoint_at(
                    "audio8.layer_output",
                    Some(index),
                    Some(position),
                    &[AUDIO_WIDTH],
                    &hidden,
                ));
            }
        }
        state.position += 1;
        let mut output = vec![0.0; AUDIO_WIDTH];
        audio_rms_norm(&hidden, &self.norm, &mut output, 1e-5);
        #[cfg(feature = "parity-trace")]
        if position <= 1 {
            crate::parity_trace::report(crate::parity_trace::checkpoint_at(
                "audio8.encoder_norm",
                None,
                Some(position),
                &[AUDIO_WIDTH],
                &output,
            ));
        }
        Ok(output)
    }
}

pub fn time_condition(delay_tokens: usize, frame_embedding: &[f32]) -> Result<Vec<f32>, String> {
    if frame_embedding.len() != 3 * TEXT_WIDTH {
        return Err("invalid Audio8 frame embedding".into());
    }
    let mut condition = vec![0.0; TEXT_WIDTH];
    for i in 0..TEXT_WIDTH / 2 {
        let phase =
            delay_tokens as f32 * (-10_000.0f32.ln() * i as f32 / (TEXT_WIDTH / 2) as f32).exp();
        condition[i] = unsafe { cosf(phase) } + frame_embedding[i];
        condition[TEXT_WIDTH / 2 + i] =
            unsafe { sinf(phase) } + frame_embedding[TEXT_WIDTH / 2 + i];
    }
    Ok(condition)
}
