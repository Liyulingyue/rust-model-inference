//! Underdog Woof: character-conditioned autoregressive flow TTS and a Mimi decoder.
//!
//! Derived from Underdog Woof, Copyright 2026 Conway. Sources in this directory
//! are subject to the upstream Underdog Non-Commercial License 1.0; see
//! tools/oracle/underdog_woof/licenses/{LICENSE,NOTICE} and README.md for changes.
mod acoustic;
mod decoder;
mod rng;
pub mod text;

use crate::compute::{
    linear::{LinearExecutor, LinearId},
    ComputePolicy, UsedBackend,
};
use crate::core::{
    tensor::{require_tensor_meta, TensorSource},
    thread_pool::ComputePool,
};
use crate::ops::nn::{load_float_values, Linear};
use crate::GGMLType;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

pub const ARCHITECTURE: &str = "underdog_woof";
pub const SAMPLE_RATE: u32 = 24_000;
const FORMATS: &[&str] = &[
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q4_k", "q5_k", "q6_k",
];
const MAX_ROWS: usize = 1024;

fn contract() -> &'static Value {
    static CONTRACT: OnceLock<Value> = OnceLock::new();
    CONTRACT.get_or_init(|| serde_json::from_str(include_str!("contract.json")).unwrap())
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechRequest {
    pub input: String,
    #[serde(default = "default_voice")]
    pub voice: String,
    #[serde(default)]
    pub seed: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_normalize")]
    pub normalize: bool,
}
fn default_voice() -> String {
    "juno".into()
}
fn default_temperature() -> f32 {
    0.5
}
fn default_normalize() -> bool {
    true
}

impl SpeechRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.input.trim().is_empty() || self.input.chars().count() > 4096 {
            return Err("Underdog Woof input must contain 1..=4096 characters".into());
        }
        if !matches!(self.voice.as_str(), "juno" | "mira") {
            return Err("Underdog Woof voice must be juno or mira".into());
        }
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err("Underdog Woof temperature must be finite and in 0..=2".into());
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum SynthesisError {
    InvalidInput(String),
    Numerical(String),
}
impl std::fmt::Display for SynthesisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(e) | Self::Numerical(e) => f.write_str(e),
        }
    }
}
impl std::error::Error for SynthesisError {}

pub struct Model {
    source: Arc<dyn TensorSource>,
    frontend: text::Frontend,
    pool: Arc<ComputePool>,
    policy: ComputePolicy,
    pub format: String,
}

fn metadata<'a>(source: &'a dyn TensorSource, key: &str) -> Result<&'a str, String> {
    source
        .metadata(key)
        .and_then(|v| v.to_string_val())
        .ok_or_else(|| format!("Missing string metadata: {key}"))
}

fn validate_config(source: &dyn TensorSource) -> Result<String, String> {
    if metadata(source, "general.architecture")? != ARCHITECTURE {
        return Err("Expected explicit underdog_woof GGUF architecture".into());
    }
    let config: Value = serde_json::from_str(metadata(source, "underdog_woof.config")?)
        .map_err(|e| format!("Invalid Underdog Woof config: {e}"))?;
    if config != contract()["config"] {
        return Err("Unsupported Underdog Woof speech/decoder configuration".into());
    }
    let format = metadata(source, "underdog_woof.format")?;
    if !FORMATS.contains(&format) {
        return Err(format!("Unsupported Underdog Woof weight format: {format}"));
    }
    Ok(format.to_string())
}

impl Model {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        threads: usize,
        policy: ComputePolicy,
    ) -> Result<Self, String> {
        policy.check_build()?;
        let format = validate_config(source.as_ref())?;
        let lexicon = metadata(source.as_ref(), "underdog_woof.lexicon")?;
        let actual = format!("{:x}", Sha256::digest(lexicon.as_bytes()));
        if actual
            != contract()["source"]["hashes"]["underdog_woof/respell.tsv"]
                .as_str()
                .unwrap()
        {
            return Err("Underdog Woof respelling lexicon hash mismatch".into());
        }
        let frontend = text::Frontend::new(lexicon)?;
        Weights::load(source.as_ref())?;
        if threads == 0 {
            return Err("Underdog Woof requires at least one compute thread".into());
        }
        Ok(Self {
            source,
            frontend,
            pool: Arc::new(ComputePool::new(threads)),
            policy,
            format,
        })
    }

    pub fn synthesize(&self, request: &SpeechRequest) -> Result<Vec<f32>, SynthesisError> {
        use SynthesisError::{InvalidInput, Numerical};
        request.validate().map_err(InvalidInput)?;
        let text = if request.normalize {
            self.frontend
                .normalize(&request.input)
                .map_err(InvalidInput)?
        } else {
            request.input.clone()
        };
        if text.trim().is_empty() || text.chars().count() > 16384 {
            return Err(InvalidInput(
                "Underdog Woof normalized input is empty or too long".into(),
            ));
        }
        let pieces = text::split_text(&text);
        if pieces.is_empty()
            || pieces.len() > 128
            || pieces.iter().any(|p| p.chars().count() + 18 >= 1100)
        {
            return Err(InvalidInput("Underdog Woof text must split into 1..=128 utterances fitting its 1100-position context".into()));
        }
        let _scope = self.policy.enter_legacy_scope();
        let weights = Weights::load(self.source.as_ref()).map_err(Numerical)?;
        let mut run = Runner::new(&weights, self.pool.clone(), self.policy).map_err(Numerical)?;
        let mut noise = rng::Noise::new(request.seed);
        let mut output = Vec::new();
        for piece in &pieces {
            let ids = text::encode_text(piece);
            let mut best: Option<((u8, f64, f64), Vec<f32>)> = None;
            for _ in 0..4 {
                let latents = run
                    .generate(&request.voice, &ids, &mut noise, request.temperature, None)
                    .map_err(Numerical)?;
                let frames = latents.len() / 32;
                let waveform = run.decode(&latents).map_err(Numerical)?;
                if waveform.is_empty() || waveform.iter().any(|v| !v.is_finite()) {
                    return Err(Numerical(
                        "Underdog Woof produced empty or non-finite audio".into(),
                    ));
                }
                let pause = stall(&waveform);
                let seconds_per_char = waveform.len() as f64
                    / SAMPLE_RATE as f64
                    / piece.chars().count().max(1) as f64;
                let bad =
                    u8::from(pause >= 1.0) + u8::from(seconds_per_char > 0.095 || frames >= 750);
                let score = (bad, pause, seconds_per_char);
                if best.as_ref().is_none_or(|(previous, _)| score < *previous) {
                    best = Some((score, waveform));
                }
                if bad == 0 {
                    break;
                }
            }
            output.extend(best.unwrap().1);
        }
        eprintln!("Underdog Woof: format={} requested={:?} backend={} gpu_projections={} cpu_projections={} utterances={} samples={}",
            self.format, self.policy, if run.gpu_calls > 0 { "Vulkan+Cpu" } else { "Cpu" },
            run.gpu_calls, run.cpu_calls, pieces.len(), output.len());
        Ok(output)
    }
}

pub(super) struct Weights<'a> {
    linears: BTreeMap<String, Linear<'a>>,
    floats: BTreeMap<String, Vec<f32>>,
}
impl<'a> Weights<'a> {
    fn load(source: &'a dyn TensorSource) -> Result<Self, String> {
        let mut linears = BTreeMap::new();
        let mut floats = BTreeMap::new();
        for (name, shape) in contract()["tensors"].as_object().unwrap() {
            let dims: Vec<u64> = shape
                .as_array()
                .unwrap()
                .iter()
                .rev()
                .map(|v| v.as_u64().unwrap())
                .collect();
            if dims.len() == 2 && name.ends_with(".weight") && name != "speech.text_emb.weight" {
                let prefix = name.strip_suffix(".weight").unwrap().to_string();
                linears.insert(
                    prefix,
                    Linear::from_source(source, name, None, dims[0] as usize, dims[1] as usize)?,
                );
            } else {
                require_tensor_meta(source, name, &dims, &[GGMLType::F32])?;
                let values = load_float_values(source, name, &dims)?;
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(format!("Non-finite Underdog Woof tensor: {name}"));
                }
                floats.insert(name.to_string(), values);
            }
        }
        Ok(Self { linears, floats })
    }
    fn values(&self, name: &str) -> &[f32] {
        &self.floats[name]
    }
}

pub(super) struct Runner<'m, 'w> {
    weights: &'m Weights<'w>,
    executor: LinearExecutor<'m, 'w>,
    ids: BTreeMap<&'m str, LinearId>,
    cpu_calls: usize,
    gpu_calls: usize,
}
impl<'m, 'w> Runner<'m, 'w> {
    fn new(
        weights: &'m Weights<'w>,
        pool: Arc<ComputePool>,
        policy: ComputePolicy,
    ) -> Result<Self, String> {
        let bindings = weights.linears.values().map(Linear::binding).collect();
        let executor =
            LinearExecutor::new(policy, bindings, MAX_ROWS, pool).map_err(|e| e.to_string())?;
        let ids = weights
            .linears
            .keys()
            .enumerate()
            .map(|(i, name)| Ok((name.as_str(), executor.id(i).map_err(|e| e.to_string())?)))
            .collect::<Result<_, String>>()?;
        Ok(Self {
            weights,
            executor,
            ids,
            cpu_calls: 0,
            gpu_calls: 0,
        })
    }
    fn matmul(&mut self, name: &str, input: &[f32]) -> Result<Vec<f32>, String> {
        let linear = &self.weights.linears[name];
        let (width, out) = (linear.n_in(), linear.n_out());
        if input.is_empty() || input.len() % width != 0 {
            return Err(format!("Invalid Underdog Woof linear input: {name}"));
        }
        let rows = input.len() / width;
        let mut output = vec![0.0; rows * out];
        for (x, y) in input
            .chunks(MAX_ROWS * width)
            .zip(output.chunks_mut(MAX_ROWS * out))
        {
            match self
                .executor
                .run(self.ids[name], x, x.len() / width, y)
                .map_err(|e| e.to_string())?
            {
                UsedBackend::Cpu => self.cpu_calls += 1,
                UsedBackend::Vulkan => self.gpu_calls += 1,
            }
        }
        Ok(output)
    }
    fn linear(&mut self, name: &str, input: &[f32]) -> Result<Vec<f32>, String> {
        let mut output = self.matmul(name, input)?;
        let out = self.weights.linears[name].n_out();
        if let Some(bias) = self.weights.floats.get(&format!("{name}.bias")) {
            for row in output.chunks_exact_mut(out) {
                add(row, bias);
            }
        }
        Ok(output)
    }
}

fn add(x: &mut [f32], y: &[f32]) {
    debug_assert_eq!(x.len(), y.len());
    for (x, y) in x.iter_mut().zip(y) {
        *x += y;
    }
}
fn stall(waveform: &[f32]) -> f64 {
    let quiet: Vec<bool> = waveform
        .chunks_exact(960)
        .map(|frame| {
            let rms = (frame.iter().map(|v| v * v).sum::<f32>() / 960.0).sqrt();
            20.0 * (rms + 1e-9).log10() < -50.0
        })
        .collect();
    let Some(first) = quiet.iter().position(|q| !q) else {
        return 0.0;
    };
    let last = quiet.iter().rposition(|q| !q).unwrap();
    let (mut current, mut best) = (0, 0);
    for q in &quiet[first..=last] {
        current = if *q { current + 1 } else { 0 };
        best = best.max(current);
    }
    best as f64 * 0.04
}

fn trace(name: &str, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    if crate::parity_trace::enabled(name) {
        crate::parity_trace::report(crate::parity_trace::checkpoint(name, None, shape, values));
    }
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, shape, values);
}

#[cfg(test)]
mod tests;
