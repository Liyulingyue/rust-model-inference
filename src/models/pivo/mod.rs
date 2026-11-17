//! PiVo-1.4M: English frontend, acoustic network and FARGAN vocoder.
mod acoustic;
#[cfg(feature = "vulkan")]
mod gpu;
#[cfg(test)]
mod tests;
mod text;
mod vocoder;

use crate::compute::ComputePolicy;
use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

pub const SAMPLE_RATE: u32 = 16000;
/// The `general.architecture` value the PiVo converter writes.
pub const ARCHITECTURE: &str = "pivo";
const CONTRACT: &str = include_str!("contract.json");

/// The weight storage a PiVo GGUF carries.
///
/// PiVo always computes in F32. The quantized export changes how the
/// weights are stored and how large the file is, not the arithmetic, so the
/// file size is the only thing it buys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    F32,
    Q8_0,
}

impl WeightFormat {
    /// The `pivo.format` metadata value that names this storage.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "pivo-f32-v1",
            Self::Q8_0 => "pivo-q8_0-v1",
        }
    }

    fn from_metadata(value: Option<&str>) -> Result<Self, String> {
        match value {
            Some("pivo-f32-v1") => Ok(Self::F32),
            Some("pivo-q8_0-v1") => Ok(Self::Q8_0),
            other => Err(format!(
                "Unsupported PiVo format {other:?}; expected pivo-f32-v1 or pivo-q8_0-v1"
            )),
        }
    }
}

/// Whether a path holds a PiVo model.
///
/// The app layer asks this before it decides which TTS pipeline to run, so
/// it lives next to the contract it checks against.
pub fn matches_model(path: &Path) -> bool {
    path.is_file()
        && crate::format::ggufrs::open_model_source(path, crate::format::ggufrs::ComponentRole::Llm)
            .ok()
            .is_some_and(|source| {
                source
                    .metadata("general.architecture")
                    .and_then(|value| value.to_string_val())
                    .as_deref()
                    == Some(ARCHITECTURE)
            })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechRequest {
    pub input: String,
    #[serde(default = "default_voice")]
    pub voice: String,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default)]
    pub overrides: HashMap<String, Vec<String>>,
}

fn default_voice() -> String {
    "Pi".into()
}
fn default_speed() -> f32 {
    1.0
}

#[derive(Debug)]
pub enum SynthesisError {
    InvalidInput(String),
    Numerical(String),
}

impl std::fmt::Display for SynthesisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) | Self::Numerical(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for SynthesisError {}

struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

pub struct Model {
    weights: HashMap<String, Tensor>,
    frontend: text::Frontend,
    /// How the GGUF stored the weights. The arithmetic is F32 either way.
    pub format: WeightFormat,
    /// Whether the acoustic encoder runs on the Vulkan device.
    device: bool,
    /// The policy the model was loaded with.
    policy: ComputePolicy,
}

/// Whether the acoustic encoder runs on the device.
///
/// The encoder is the only part of PiVo that can: the vocoder is a strictly
/// sequential recurrence, and the decoder feeds a pitch period whose
/// rounding margin is smaller than the difference between the host and the
/// device. The device path therefore runs the encoder on the device and
/// keeps the duration head, the decoder and the vocoder on the host, which
/// leaves the frame durations the host's.
///
/// `Auto` stays on the host. Two reasons, and both of them hold:
/// the device path is measured slower than the host's SIMD path at PiVo's
/// sizes -- the encoder is a small part of an utterance and the device's
/// per-dispatch cost dominates it -- and the host's frame durations are the
/// reference the oracle checks the WAV bytes against. The strict policy
/// still runs the encoder on the device: the flag asks for a device, and the
/// device path is there to be used.
fn device_enabled(compute: ComputePolicy) -> bool {
    match compute {
        ComputePolicy::Cpu => false,
        #[cfg(feature = "vulkan")]
        ComputePolicy::Vulkan => crate::ops::get_vulkan_context().is_some(),
        #[cfg(not(feature = "vulkan"))]
        ComputePolicy::Vulkan => false,
        ComputePolicy::Auto => false,
    }
}

/// Reads one contract tensor and returns its F32 values.
///
/// A quantized export is dequantized once here, so the compute path stays
/// F32 whatever the file stores. Norms, biases and per-channel statistics
/// stay F32 in the file: they are one-dimensional, small and precision
/// sensitive, and the Q8_0 block size does not divide them.
fn load_tensor(source: &dyn TensorSource, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    let dims: Vec<u64> = shape.iter().rev().map(|&n| n as u64).collect();
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing PiVo tensor: {name}"))?;
    if info.dims != dims {
        return Err(format!(
            "Invalid PiVo tensor {name}: expected shape {dims:?}"
        ));
    }
    let count = shape.iter().product::<usize>();
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing PiVo data: {name}"))?;
    let data = match info.ggml_type {
        GGMLType::F32 => {
            if bytes.len() != count * 4 {
                return Err(format!("Invalid PiVo tensor length: {name}"));
            }
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        GGMLType::Q8_0 => {
            if shape.len() < 2 {
                return Err(format!(
                    "PiVo tensor {name} is one-dimensional; norms, biases and \
                     per-channel statistics must stay F32"
                ));
            }
            // GGML quantizes a row at a time, and a Q8_0 block holds 32
            // values, so the row length — the fastest varying dimension —
            // has to be a whole number of blocks. The converter applies the
            // same rule in `quantizable` in
            // `tools/converter/pivo/convert_pivo.py`.
            let row = shape[shape.len() - 1];
            if row % 32 != 0 {
                return Err(format!(
                    "PiVo tensor {name} has a row length of {row}, which is not \
                     a multiple of the Q8_0 block size 32"
                ));
            }
            if bytes.len() != count / 32 * crate::ops::quant::BLOCK_Q80_SIZE {
                return Err(format!("Invalid PiVo tensor length: {name}"));
            }
            crate::ops::quant::dequant_q80_weight(bytes, count, 1)
        }
        other => {
            return Err(format!(
                "Unsupported PiVo tensor type for {name}: {other:?}; expected F32 or Q8_0"
            ))
        }
    };
    if data.iter().any(|value| !value.is_finite()) {
        return Err(format!("Non-finite PiVo tensor: {name}"));
    }
    Ok(data)
}

impl Model {
    pub fn from_source(source: &dyn TensorSource, compute: ComputePolicy) -> Result<Self, String> {
        let device = device_enabled(compute);
        if compute == ComputePolicy::Vulkan && !device {
            return Err("PiVo needs a Vulkan device for --compute vulkan".into());
        }
        let meta = |key| source.metadata(key).and_then(MetaValue::to_string_val);
        if meta("general.architecture").as_deref() != Some(ARCHITECTURE) {
            return Err("Expected the pivo architecture".into());
        }
        let format = WeightFormat::from_metadata(meta("pivo.format").as_deref())?;
        let contract: serde_json::Value = serde_json::from_str(CONTRACT).unwrap();
        let config: serde_json::Value =
            serde_json::from_str(meta("pivo.config").ok_or("Missing pivo.config")?)
                .map_err(|e| format!("Invalid PiVo config: {e}"))?;
        if config != contract["config"] {
            return Err("Unsupported PiVo configuration".into());
        }
        let mut weights = HashMap::new();
        for (name, shape) in contract["tensors"].as_object().unwrap() {
            let shape: Vec<usize> = serde_json::from_value(shape.clone()).unwrap();
            let data = load_tensor(source, name, &shape)?;
            weights.insert(name.clone(), Tensor { shape, data });
        }
        let frontend = text::Frontend::new(meta("pivo.lexicon").ok_or("Missing pivo.lexicon")?)?;
        Ok(Self {
            weights,
            frontend,
            format,
            device,
            policy: compute,
        })
    }

    pub fn synthesize(&self, request: &SpeechRequest) -> Result<Vec<f32>, SynthesisError> {
        use SynthesisError::{InvalidInput, Numerical};
        let voice = match request.voice.as_str() {
            "Pi" => 0,
            "Vo" => 1,
            _ => return Err(InvalidInput("PiVo voice must be Pi or Vo".into())),
        };
        if !request.speed.is_finite() || !(0.5..=2.0).contains(&request.speed) {
            return Err(InvalidInput(
                "PiVo speed must be finite and between 0.5 and 2.0".into(),
            ));
        }
        let front = self
            .frontend
            .phonemize(&request.input, &request.overrides)
            .map_err(InvalidInput)?;
        #[cfg(feature = "vulkan")]
        let (encoded, log_durations) = if self.device {
            let encoded = self
                .encode_on_device(&front.tokens, voice)
                .map_err(Numerical)?;
            let log_durations = self.duration_head(&encoded);
            (encoded, log_durations)
        } else {
            self.encode(&front.tokens, voice)
        };
        #[cfg(not(feature = "vulkan"))]
        let (encoded, log_durations) = self.encode(&front.tokens, voice);
        // The encoder is the only part that can be on the device, so the
        // trace reports the encoder's backend and not the utterance's.
        self.policy.trace(
            "pivo_encoder",
            if self.device {
                crate::compute::UsedBackend::Vulkan
            } else {
                crate::compute::UsedBackend::Cpu
            },
            front.tokens.len(),
        );
        if encoded.iter().chain(&log_durations).any(|v| !v.is_finite()) {
            return Err(Numerical("PiVo produced non-finite encoder output".into()));
        }
        let durations = acoustic::durations(&front.tokens, &log_durations, request.speed)
            .map_err(InvalidInput)?;
        let expanded = acoustic::expand(&encoded, &durations);
        let features = self.decode(&expanded);
        if features.iter().any(|v| !v.is_finite()) {
            return Err(Numerical(
                "PiVo produced non-finite acoustic features".into(),
            ));
        }
        let pcm = self.vocode(&features);
        if pcm.iter().any(|v| !v.is_finite()) {
            return Err(Numerical("PiVo produced non-finite audio".into()));
        }
        Ok(pcm)
    }

    fn w(&self, name: &str) -> &[f32] {
        &self.weights[name].data
    }

    fn linear(&self, name: &str, input: &[f32]) -> Vec<f32> {
        let weight = &self.weights[&format!("{name}.weight")];
        let out = weight.shape[0];
        let width = weight.data.len() / out;
        let bias = self.weights.get(&format!("{name}.bias"));
        let mut result = vec![0.; input.len() / width * out];
        let row = |(x, y): (&[f32], &mut [f32])| {
            for (i, (w, v)) in weight.data.chunks_exact(width).zip(y).enumerate() {
                *v = crate::ops::dot_f32(x, w, width) + bias.map_or(0., |b| b.data[i]);
            }
        };
        if input.len() / width >= 16 {
            input
                .par_chunks_exact(width)
                .zip(result.par_chunks_exact_mut(out))
                .for_each(row);
        } else {
            input
                .chunks_exact(width)
                .zip(result.chunks_exact_mut(out))
                .for_each(row);
        }
        result
    }

    fn norm(&self, name: &str, input: &[f32]) -> Vec<f32> {
        let weight = self.w(&format!("{name}.weight"));
        let mut result = vec![0.; input.len()];
        for (x, y) in input
            .chunks_exact(weight.len())
            .zip(result.chunks_exact_mut(weight.len()))
        {
            crate::ops::rms_norm(x, weight, y, 1e-6);
        }
        result
    }
}

fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(a, b)| a + b).collect()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn silu(values: &mut [f32]) {
    values.iter_mut().for_each(|v| *v = crate::ops::silu(*v));
}
