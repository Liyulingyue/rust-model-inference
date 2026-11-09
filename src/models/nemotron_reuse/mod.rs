//! NVIDIA NemotronLabs RE-USE: native F32 SEMamba speech enhancement.
mod network;
mod signal;

use crate::compute::ComputePolicy;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::format::safetensors::SafetensorSource;
use std::collections::HashMap;
use std::path::Path;

const CONFIG: &str = include_str!("config.json");
const CHANNELS: usize = 64;
const INNER: usize = 256;
const STATE: usize = 16;
const LAYERS: usize = 30;

struct Tensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

pub struct Model {
    tensors: HashMap<String, Tensor>,
}

pub fn validate_config(config: &serde_json::Value) -> Result<(), String> {
    let expected: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    if config.get("model_cfg") != expected.get("model_cfg")
        || config.get("stft_cfg") != expected.get("stft_cfg")
    {
        return Err("RE-USE requires the published SEMamba model_cfg and stft_cfg".into());
    }
    Ok(())
}

fn contracts() -> Vec<(String, Vec<usize>)> {
    let mut shapes = Vec::new();
    let mut add = |name: String, shape: &[usize]| shapes.push((name, shape.to_vec()));
    for root in ["dense_encoder", "mask_decoder", "phase_decoder"] {
        if root == "dense_encoder" {
            add(format!("{root}.dense_conv_1.0.weight"), &[64, 2, 1, 1]);
            add(format!("{root}.dense_conv_1.0.bias"), &[64]);
            for suffix in ["1.weight", "1.bias", "2.weight"] {
                add(format!("{root}.dense_conv_1.{suffix}"), &[64]);
            }
        }
        for i in 0..4 {
            let p = format!("{root}.dense_block.dense_block.{i}");
            add(format!("{p}.0.weight"), &[64, 64 * (i + 1), 3, 3]);
            add(format!("{p}.0.bias"), &[64]);
            for suffix in ["1.weight", "1.bias", "2.weight"] {
                add(format!("{p}.{suffix}"), &[64]);
            }
        }
        if root == "dense_encoder" {
            add(format!("{root}.dense_conv_2.0.weight"), &[64, 64, 1, 3]);
            add(format!("{root}.dense_conv_2.0.bias"), &[64]);
            for suffix in ["1.weight", "1.bias", "2.weight"] {
                add(format!("{root}.dense_conv_2.{suffix}"), &[64]);
            }
        } else {
            for (index, scale) in [(1, 2), (2, 4)] {
                let p = format!("{root}.up_conv{index}");
                add(format!("{p}.0.conv.weight"), &[64 * scale, 64, 1, 3]);
                add(format!("{p}.0.conv.bias"), &[64 * scale]);
                for suffix in ["1.weight", "1.bias", "2.weight"] {
                    add(format!("{p}.{suffix}"), &[64]);
                }
            }
            for name in if root == "mask_decoder" {
                vec!["final_conv"]
            } else {
                vec!["phase_conv_r", "phase_conv_i"]
            } {
                add(format!("{root}.{name}.weight"), &[1, 64, 1, 1]);
                add(format!("{root}.{name}.bias"), &[1]);
            }
        }
    }
    for i in 0..LAYERS {
        for axis in ["time_mamba", "freq_mamba"] {
            let p = format!("TSMamba.{i}.{axis}");
            add(format!("{p}.output_proj.weight"), &[64, 128]);
            for name in ["output_proj.bias", "norm.weight", "norm.bias"] {
                add(format!("{p}.{name}"), &[64]);
            }
            for direction in ["forward_blocks", "backward_blocks"] {
                let p = format!("{p}.{direction}");
                for (name, shape) in [
                    ("in_proj.weight", vec![512, 64]),
                    ("conv1d.weight", vec![256, 1, 4]),
                    ("conv1d.bias", vec![256]),
                    ("x_proj.weight", vec![36, 256]),
                    ("dt_proj.weight", vec![256, 4]),
                    ("dt_proj.bias", vec![256]),
                    ("A_log", vec![256, 16]),
                    ("D", vec![256]),
                    ("out_proj.weight", vec![64, 256]),
                ] {
                    add(format!("{p}.{name}"), &shape);
                }
            }
        }
    }
    shapes
}

impl Model {
    pub fn load(path: &Path, compute: ComputePolicy) -> Result<Self, String> {
        Self::validate_compute(compute)?;
        let weights = if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.to_path_buf()
        };
        if weights.extension().and_then(|v| v.to_str()) != Some("safetensors") {
            return Err("RE-USE expects a model directory or model.safetensors".into());
        }
        let config_path = weights
            .parent()
            .unwrap_or(Path::new("."))
            .join("config.json");
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&config_path).map_err(|e| format!("{}: {e}", config_path.display()))?,
        )
        .map_err(|e| format!("RE-USE config.json: {e}"))?;
        validate_config(&config)?;
        let source = SafetensorSource::open(&[weights])?;
        if source.len() != contracts().len() {
            return Err(format!(
                "RE-USE tensor count: expected {}, got {}",
                contracts().len(),
                source.len()
            ));
        }
        Self::from_source(&source, &config, compute)
    }

    pub fn validate_compute(compute: ComputePolicy) -> Result<(), String> {
        if compute == ComputePolicy::Vulkan {
            return Err("RE-USE has no Vulkan implementation; use --compute cpu or auto".into());
        }
        Ok(())
    }

    pub fn from_source(
        source: &dyn TensorSource,
        config: &serde_json::Value,
        compute: ComputePolicy,
    ) -> Result<Self, String> {
        Self::validate_compute(compute)?;
        validate_config(config)?;
        let mut tensors = HashMap::new();
        for (name, shape) in contracts() {
            let info = source
                .tensor_info(&name)
                .ok_or_else(|| format!("RE-USE missing tensor {name}"))?;
            let dims: Vec<_> = shape.iter().rev().map(|&v| v as u64).collect();
            if info.dims != dims || info.ggml_type != GGMLType::F32 {
                return Err(format!(
                    "RE-USE invalid tensor {name}: expected F32 {shape:?}, got {:?} {:?}",
                    info.ggml_type, info.dims
                ));
            }
            let bytes = source
                .tensor_slice(&name)
                .ok_or_else(|| format!("RE-USE missing data {name}"))?;
            if info.checked_nbytes() != Some(bytes.len() as u64) {
                return Err(format!("RE-USE invalid storage {name}"));
            }
            let values: Vec<_> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(format!("RE-USE non-finite tensor {name}"));
            }
            tensors.insert(name, Tensor { shape, values });
        }
        compute.trace("nemotron_reuse", crate::compute::UsedBackend::Cpu, 0);
        Ok(Self { tensors })
    }

    fn values(&self, name: &str) -> &[f32] {
        &self.tensors[name].values
    }

    pub fn enhance_wav(&self, bytes: &[u8]) -> Result<Vec<u8>, String> {
        let audio = crate::ops::audio::frontend::decode_pcm16_wav_any(bytes)
            .map_err(|e| format!("RE-USE WAV: {e:?}"))?;
        let output = self.enhance(&audio.samples, audio.sample_rate, audio.channels)?;
        crate::format::wav::encode_wav_pcm16_channels(&output, audio.sample_rate, audio.channels)
    }

    /// Input and output are interleaved PCM samples; channels are enhanced independently.
    pub fn enhance(
        &self,
        samples: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> Result<Vec<f32>, String> {
        let (fft, _) = signal::parameters(sample_rate)?;
        let channels = usize::from(channels);
        if channels == 0
            || channels > 8
            || samples.len() % channels != 0
            || samples.len() / channels <= fft / 2
            || samples.len() / channels > sample_rate as usize * 600
            || samples.iter().any(|v| !v.is_finite())
        {
            return Err(
                "RE-USE expects finite PCM, 1–8 channels, more than 20 ms and at most 10 minutes"
                    .into(),
            );
        }
        let mut result = vec![0.0; samples.len()];
        for channel in 0..channels {
            let mono: Vec<_> = samples
                .iter()
                .skip(channel)
                .step_by(channels)
                .copied()
                .collect();
            let enhanced = self.enhance_channel(&mono, sample_rate)?;
            for (out, value) in result
                .iter_mut()
                .skip(channel)
                .step_by(channels)
                .zip(enhanced)
            {
                *out = value;
            }
        }
        Ok(result)
    }

    fn enhance_channel(&self, samples: &[f32], rate: u32) -> Result<Vec<f32>, String> {
        let chunk = 2 * rate as usize;
        if samples.len() <= chunk {
            return self.enhance_chunk(samples, rate);
        }
        let hop = chunk / 2;
        let count = (samples.len() - chunk).div_ceil(hop) + 1;
        let window: Vec<_> = signal::hann(chunk)
            .into_iter()
            .map(|v| v.max(1e-7))
            .collect();
        let mut output = vec![0.0; samples.len()];
        let mut window_sum = vec![0.0; samples.len()];
        for i in 0..count {
            let start = if i + 1 == count {
                samples.len() - chunk
            } else {
                i * hop
            };
            let enhanced = self.enhance_chunk(&samples[start..start + chunk], rate)?;
            for j in 0..chunk {
                output[start + j] += enhanced[j] * window[j];
                window_sum[start + j] += window[j];
            }
        }
        for (out, norm) in output.iter_mut().zip(window_sum) {
            *out /= norm;
        }
        Ok(output)
    }

    fn enhance_chunk(&self, samples: &[f32], rate: u32) -> Result<Vec<f32>, String> {
        let (mag, phase, t, f) = signal::stft(samples, rate)?;
        trace("reuse.stft.mag", &[t, f], &mag);
        trace("reuse.stft.phase", &[t, f], &phase);
        let (mut mag, phase) = self.forward(&mag, &phase, t, f);
        if mag.iter().chain(&phase).any(|v| !v.is_finite()) {
            return Err("RE-USE produced non-finite spectrum".into());
        }
        suppress_sweep_frames(&mut mag, f);
        let output = signal::istft(&mag, &phase, t, f, samples.len(), rate)?;
        trace("reuse.waveform", &[output.len()], &output);
        Ok(output)
    }
}

// Published inference.py/inference_chunk.py remove sweep artifacts before ISTFT.
fn suppress_sweep_frames(mag: &mut [f32], bins: usize) {
    for frame in mag.chunks_exact_mut(bins) {
        if frame.iter().filter(|&&v| v <= 0.0).count() * 2 > bins {
            frame.fill(0.0);
        }
    }
}

fn trace(name: &str, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(name, None, shape, values));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, shape, values);
}

#[cfg(test)]
mod tests;
