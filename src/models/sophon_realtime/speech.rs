//! Tag speech synthesis shared by the CLI and HTTP speech routes.

use super::{
    flow::{Flow, FlowRequest},
    hift::HiFT,
    talker::Talker,
};
use crate::compute::ComputePolicy;
use crate::core::{
    tensor::{MetaValue, TensorInfo, TensorSource},
    thread_pool::ComputePool,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

pub struct SpeechEngine {
    talker: Talker,
    code2wav: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    compute: ComputePolicy,
    voices: Vec<String>,
}

pub struct SpeechRequest<'a> {
    pub text: &'a str,
    pub voice: &'a str,
    pub max_tokens: usize,
    pub temperature: f32,
    pub seed: u64,
    pub stream: bool,
}

pub struct SpeechOutput {
    pub waveform: Vec<f32>,
    pub speech_tokens: Vec<u32>,
    pub terminated: bool,
    pub cancelled: bool,
}

impl SpeechEngine {
    pub fn new(
        talker: Arc<dyn TensorSource>,
        code2wav: Arc<dyn TensorSource>,
        threads: usize,
        compute: ComputePolicy,
    ) -> Result<Self, String> {
        if threads == 0 {
            return Err("speech requires at least one thread".into());
        }
        if compute == ComputePolicy::Vulkan {
            return Err("strict Vulkan speech awaits shared causal convolution and HiFT spectral device kernels".into());
        }
        super::require_string(
            code2wav.as_ref(),
            "general.architecture",
            "sophon_realtime_code2wav",
        )?;
        super::require_string(code2wav.as_ref(), "sophon_realtime.component", "code2wav")?;
        super::require_string(
            code2wav.as_ref(),
            "sophon_realtime.source_revision",
            super::SOURCE_REVISION,
        )?;
        let voices: Vec<String> = serde_json::from_str(
            code2wav
                .metadata("sophon_realtime.voices_json")
                .and_then(MetaValue::to_string_val)
                .ok_or("missing code2wav voices")?,
        )
        .map_err(|e| format!("invalid voice inventory: {e}"))?;
        if voices.is_empty()
            || voices
                .iter()
                .any(|v| v.is_empty() || !v.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'))
        {
            return Err("invalid code2wav voice names".into());
        }
        let _scope = compute.enter_legacy_scope();
        Flow::load(&Component {
            source: code2wav.as_ref(),
            prefix: "flow",
        })?;
        HiFT::load(&Component {
            source: code2wav.as_ref(),
            prefix: "hift",
        })?;
        for voice in &voices {
            Voice::load(code2wav.as_ref(), voice)?;
        }
        let pool = Arc::new(ComputePool::new(threads));
        let talker = Talker::from_source(talker, Arc::clone(&pool))?;
        Ok(Self {
            talker,
            code2wav,
            pool,
            compute,
            voices,
        })
    }

    pub fn voices(&self) -> &[String] {
        &self.voices
    }

    pub fn validate(&self, request: &SpeechRequest<'_>) -> Result<(), String> {
        if request.text.trim().is_empty()
            || request.text.len() > 65_536
            || request.max_tokens == 0
            || request.max_tokens > 1250
            || !request.temperature.is_finite()
            || !(0.0..=2.0).contains(&request.temperature)
        {
            return Err("speech requires nonempty text up to 64 KiB, 1..=1250 tokens and finite temperature in [0,2]".into());
        }
        if !self.voices.iter().any(|voice| voice == request.voice) {
            return Err(format!(
                "unknown voice {:?}; available: {}",
                request.voice,
                self.voices.join(", ")
            ));
        }
        Ok(())
    }

    /// The mutable receiver serializes the shared compute pool across requests.
    pub fn synthesize(
        &mut self,
        request: &SpeechRequest<'_>,
        emit: impl FnMut(&[f32]) -> bool,
    ) -> Result<SpeechOutput, String> {
        self.synthesize_until(request, emit, &|| false)
    }

    pub fn synthesize_until(
        &mut self,
        request: &SpeechRequest<'_>,
        mut emit: impl FnMut(&[f32]) -> bool,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<SpeechOutput, String> {
        self.validate(request)?;
        super::check_cancelled(is_cancelled)?;
        let _scope = self.compute.enter_legacy_scope();
        let voice = Voice::load(self.code2wav.as_ref(), request.voice)?;
        let flow_source = Component {
            source: self.code2wav.as_ref(),
            prefix: "flow",
        };
        let hift_source = Component {
            source: self.code2wav.as_ref(),
            prefix: "hift",
        };
        let flow = Flow::load(&flow_source)?;
        let hift = HiFT::load(&hift_source)?;
        let noise = Noise::new(request.seed, request.max_tokens)?;
        let mut tokens = Vec::new();
        let mut waveform = Vec::new();
        let mut mel_history = MelHistory::default();
        let mut emitted = 0;
        let mut cancelled = false;
        let mut error = None;
        let mut emit_prefix = |tokens: &[u32], finalize: bool| -> Result<bool, String> {
            super::check_cancelled(is_cancelled)?;
            let prefix = render(
                &flow,
                &hift,
                &voice,
                tokens,
                finalize,
                &noise,
                &mut mel_history,
                &self.pool,
                is_cancelled,
            )?;
            super::check_cancelled(is_cancelled)?;
            if prefix.len() < emitted {
                return Err("streamed speech prefix became shorter".into());
            }
            let chunk = &prefix[emitted..];
            let accepted = chunk.is_empty() || emit(chunk);
            if accepted {
                waveform.extend_from_slice(chunk);
                emitted = prefix.len();
            }
            Ok(accepted)
        };
        let result = self.talker.generate_tokens(
            request.text,
            request.max_tokens,
            request.temperature,
            request.seed,
            self.compute,
            |token| {
                if is_cancelled() {
                    cancelled = true;
                    return false;
                }
                tokens.push(token);
                if request.stream && tokens.len() % voice.hop == 0 {
                    match emit_prefix(&tokens, false) {
                        Ok(true) => (),
                        Ok(false) => {
                            cancelled = true;
                            return false;
                        }
                        Err(e) => {
                            if is_cancelled() {
                                cancelled = true;
                            } else {
                                error = Some(e);
                            }
                            return false;
                        }
                    }
                }
                true
            },
        )?;
        if let Some(error) = error {
            return Err(error);
        }
        cancelled |= is_cancelled();
        if !cancelled {
            if tokens.is_empty() {
                return Err("talker generated no speech tokens".into());
            }
            match emit_prefix(&tokens, true) {
                Ok(accepted) => cancelled = !accepted,
                Err(_) if is_cancelled() => cancelled = true,
                Err(error) => return Err(error),
            }
        }
        Ok(SpeechOutput {
            waveform,
            speech_tokens: result.tokens,
            terminated: result.terminated,
            cancelled,
        })
    }
}

struct Component<'a> {
    source: &'a dyn TensorSource,
    prefix: &'static str,
}
impl TensorSource for Component<'_> {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.source.metadata(&format!("{}.{key}", self.prefix))
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.source.tensor_info(&format!("{}.{name}", self.prefix))
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.source.tensor_slice(&format!("{}.{name}", self.prefix))
    }
}

struct Voice {
    tokens: Vec<u32>,
    mel: Vec<f32>,
    speaker: Vec<f32>,
    hop: usize,
}
impl Voice {
    fn load(source: &dyn TensorSource, name: &str) -> Result<Self, String> {
        let prefix = format!("voices.{name}");
        super::require_u64(source, &format!("{prefix}.hop"), 10)?;
        let token_key = format!("{prefix}.tokens");
        let info = source
            .tensor_info(&token_key)
            .ok_or("missing voice prompt tokens")?;
        if info.dims.len() != 1 || info.dims[0] == 0 || info.dims[0] > 7500 {
            return Err("invalid voice prompt length".into());
        }
        let length = info.dims[0] as usize;
        let values = super::flow::load_f32(source, &token_key, &[length as u64])?;
        if values
            .iter()
            .any(|&v| !(0.0..6561.0).contains(&v) || v.fract() != 0.0)
        {
            return Err("invalid discrete voice prompt token".into());
        }
        let mel =
            super::flow::load_f32(source, &format!("{prefix}.mel"), &[(length * 2) as u64, 80])?;
        let speaker = super::flow::load_f32(source, &format!("{prefix}.speaker"), &[192])?;
        if crate::ops::dot_f32(&speaker, &speaker, 192) == 0.0 {
            return Err("zero voice speaker embedding".into());
        }
        Ok(Self {
            tokens: values.iter().map(|v| *v as u32).collect(),
            mel,
            speaker,
            hop: 10,
        })
    }
}

struct Noise {
    flow: Vec<f32>,
    excitation: Vec<f32>,
}

#[derive(Default)]
struct MelHistory {
    mel: Vec<f32>,
    frames: usize,
}

impl MelHistory {
    fn append_prefix(&mut self, prefix: &[f32], frames: usize) -> Result<(), String> {
        if frames < self.frames || prefix.len() != 80 * frames {
            return Err("invalid streamed mel prefix".into());
        }
        let mut mel = Vec::with_capacity(prefix.len());
        for channel in 0..80 {
            mel.extend_from_slice(&self.mel[channel * self.frames..(channel + 1) * self.frames]);
            mel.extend_from_slice(&prefix[channel * frames + self.frames..(channel + 1) * frames]);
        }
        self.mel = mel;
        self.frames = frames;
        Ok(())
    }
}
impl Noise {
    fn new(seed: u64, max_tokens: usize) -> Result<Self, String> {
        if max_tokens == 0 || max_tokens > 1250 {
            return Err("invalid speech noise budget".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut flow = Vec::with_capacity(80 * 15000);
        for _ in 0..80 * 7500 {
            let radius = (-2.0 * (1.0 - rng.gen::<f32>()).ln()).sqrt();
            let (sin, cos) = (2.0 * std::f32::consts::PI * rng.gen::<f32>()).sin_cos();
            flow.extend([radius * cos, radius * sin]);
        }
        let excitation = (0..max_tokens * 2 * 480 * 9)
            .map(|_| rng.gen::<f32>())
            .collect();
        Ok(Self { flow, excitation })
    }
    fn flow_prefix(&self, frames: usize) -> Result<Vec<f32>, String> {
        if frames == 0 || frames > 15000 {
            return Err("speech exceeds the flow noise window".into());
        }
        Ok(self
            .flow
            .chunks_exact(15000)
            .flat_map(|row| row[..frames].iter().copied())
            .collect())
    }
}

fn render(
    flow: &Flow<'_>,
    hift: &HiFT<'_>,
    voice: &Voice,
    tokens: &[u32],
    finalize: bool,
    noise: &Noise,
    history: &mut MelHistory,
    pool: &ComputePool,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<f32>, String> {
    let frames = (voice.tokens.len() + tokens.len())
        .checked_sub(if finalize { 0 } else { 3 })
        .ok_or("not enough speech lookahead")?
        * 2;
    let initial_noise = noise.flow_prefix(frames)?;
    let mut request = FlowRequest::new(tokens, &voice.speaker, &initial_noise);
    request.prompt_tokens = &voice.tokens;
    request.prompt_mel = &voice.mel;
    request.finalize = finalize;
    let mel = flow.infer_until(&request, pool, cancelled)?;
    // Flow recomputes the prefix; the reference only commits newly produced mel frames.
    history.append_prefix(&mel.mel, mel.frames)?;
    let f0_frames = history
        .frames
        .checked_sub(if finalize { 0 } else { 3 })
        .ok_or("not enough vocoder lookahead")?;
    let excitation = noise
        .excitation
        .get(..f0_frames * 480 * 9)
        .ok_or("speech exceeds the excitation budget")?;
    // shortcut: recompute HiFT history; use a bounded phase-carry window for long-stream performance.
    hift.synthesize_until(
        &history.mel,
        history.frames,
        finalize,
        excitation,
        pool,
        cancelled,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_mel_keeps_committed_frames_when_flow_recomputes_them() {
        let mut history = MelHistory::default();
        let first: Vec<_> = (0..80).flat_map(|c| [c as f32, 1.]).collect();
        history.append_prefix(&first, 2).unwrap();
        let next: Vec<_> = (0..80).flat_map(|_| [-1., -2., 3., 4.]).collect();
        history.append_prefix(&next, 4).unwrap();
        for (c, row) in history.mel.chunks_exact(4).enumerate() {
            assert_eq!(row, [c as f32, 1., 3., 4.]);
        }
        assert!(history.append_prefix(&first, 2).is_err());
        assert!(history.append_prefix(&[], 5).is_err());
    }
    #[test]
    fn noise_prefix_is_stable_when_streams_grow() {
        let noise = Noise::new(42, 1).unwrap();
        let small = noise.flow_prefix(2).unwrap();
        let large = noise.flow_prefix(3).unwrap();
        for c in 0..80 {
            assert_eq!(&small[c * 2..c * 2 + 2], &large[c * 3..c * 3 + 2]);
        }
        assert!(noise.excitation.iter().all(|v| (0.0..1.0).contains(v)));
        assert!(noise.flow.iter().all(|v| v.is_finite()));
        assert!(Noise::new(0, usize::MAX).is_err());
        assert!(noise.flow_prefix(15001).is_err());
    }
}
