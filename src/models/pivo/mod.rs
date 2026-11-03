//! PiVo-1.4M: English frontend, acoustic network and FARGAN vocoder.
mod acoustic;
#[cfg(test)]
mod tests;
mod text;
mod vocoder;

use crate::compute::ComputePolicy;
use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;

pub const SAMPLE_RATE: u32 = 16000;
const CONTRACT: &str = include_str!("contract.json");

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
}

impl Model {
    pub fn from_source(source: &dyn TensorSource, compute: ComputePolicy) -> Result<Self, String> {
        if compute == ComputePolicy::Vulkan {
            return Err("PiVo has no Vulkan path; use --compute cpu or auto".into());
        }
        let meta = |key| source.metadata(key).and_then(MetaValue::to_string_val);
        if meta("general.architecture") != Some("pivo")
            || meta("pivo.format") != Some("pivo-f32-v1")
        {
            return Err("Expected pivo architecture with pivo-f32-v1 format".into());
        }
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
            let dims: Vec<u64> = shape.iter().rev().map(|&n| n as u64).collect();
            let info = source
                .tensor_info(name)
                .ok_or_else(|| format!("Missing PiVo tensor: {name}"))?;
            if info.dims != dims || info.ggml_type != GGMLType::F32 {
                return Err(format!(
                    "Invalid PiVo tensor {name}: expected shape {dims:?} F32"
                ));
            }
            let bytes = source
                .tensor_slice(name)
                .ok_or_else(|| format!("Missing PiVo data: {name}"))?;
            if bytes.len() != shape.iter().product::<usize>() * 4 {
                return Err(format!("Invalid PiVo tensor length: {name}"));
            }
            let data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            if data.iter().any(|v| !v.is_finite()) {
                return Err(format!("Non-finite PiVo tensor: {name}"));
            }
            weights.insert(name.clone(), Tensor { shape, data });
        }
        let frontend = text::Frontend::new(meta("pivo.lexicon").ok_or("Missing pivo.lexicon")?)?;
        compute.trace("pivo", crate::compute::UsedBackend::Cpu, 1);
        Ok(Self { weights, frontend })
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
        let (encoded, log_durations) = self.encode(&front.tokens, voice);
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
