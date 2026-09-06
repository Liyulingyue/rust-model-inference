//! VibeVoice ASR model assembly and the streaming transcription loop.
//!
//! Protocol (mirrors `VibeVoiceASRForConditionalGeneration.streaming_generate`
//! with `encode_mode="split_then_encode"`): the system prompt is prefilled
//! once; every audio window of `chunk + lookahead` latent frames is encoded
//! independently (last window zero-padded), projected as
//! `acoustic_connector(noisy_acoustic) + semantic_connector(semantic)`, fed to
//! the decoder as `[speech_start, features…, speech_end]`, and greedily
//! decoded until `<|text_chunk_end|>` (or EOS), after which the chunk-end
//! token is appended to the cache and the next window begins.

use std::sync::Arc;

use rand::{Rng, SeedableRng};

use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::vibevoice_asr::config::VibeVoiceAsrConfig;
use crate::models::vibevoice_asr::encoder::{SpeechConnector, TokenizerEncoder};
use crate::models::vibevoice_asr::llm::{AsrInputRow, AsrLlmSession, VibeVoiceAsrLlm};

pub const DEFAULT_MAX_NEW_TOKENS_PER_CHUNK: usize = 256;

#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// Greedy decode cap per chunk (official demo default: 256).
    pub max_new_tokens_per_chunk: usize,
    /// Optional hotwords line appended to the prompt.
    pub context_info: Option<String>,
    /// Feed acoustic means without the gaussian latent noise (deterministic).
    pub deterministic_latents: bool,
    /// Seed for the latent-noise RNG (None = entropy).
    pub seed: Option<u64>,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            max_new_tokens_per_chunk: DEFAULT_MAX_NEW_TOKENS_PER_CHUNK,
            context_info: None,
            deterministic_latents: false,
            seed: None,
        }
    }
}

pub struct VibeVoiceAsrModel {
    pub config: VibeVoiceAsrConfig,
    pub llm: VibeVoiceAsrLlm,
    pub tokenizer: Arc<BPETokenizer>,
    acoustic_encoder: TokenizerEncoder,
    semantic_encoder: TokenizerEncoder,
    acoustic_connector: SpeechConnector,
    semantic_connector: SpeechConnector,
}

impl VibeVoiceAsrModel {
    pub fn from_sources(
        llm_source: Arc<dyn TensorSource>,
        mmproj_source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
        tokenizer: Arc<BPETokenizer>,
    ) -> Result<Self, String> {
        let config = VibeVoiceAsrConfig::from_source(mmproj_source.as_ref())?;
        let llm = VibeVoiceAsrLlm::from_source(llm_source, Arc::clone(&pool))?;
        if llm.config.n_embd != config.llm_hidden_size {
            return Err(format!(
                "mmproj llm_hidden_size {} does not match LLM embedding width {}",
                config.llm_hidden_size, llm.config.n_embd
            ));
        }
        let acoustic_encoder =
            TokenizerEncoder::from_source(mmproj_source.as_ref(), "acoustic", &config)?;
        let semantic_encoder =
            TokenizerEncoder::from_source(mmproj_source.as_ref(), "semantic", &config)?;
        let acoustic_connector = SpeechConnector::from_source(
            mmproj_source.as_ref(),
            "acoustic",
            config.acoustic_vae_dim,
            config.llm_hidden_size,
            config.connector_eps,
        )?;
        let semantic_connector = SpeechConnector::from_source(
            mmproj_source.as_ref(),
            "semantic",
            config.semantic_vae_dim,
            config.llm_hidden_size,
            config.connector_eps,
        )?;
        Ok(Self {
            config,
            llm,
            tokenizer,
            acoustic_encoder,
            semantic_encoder,
            acoustic_connector,
            semantic_connector,
        })
    }

    /// Encode one audio window into token-major LLM feature rows
    /// ([frames][hidden]). `frame_noise` is the per-window gaussian scale
    /// sample; pass `None` for the deterministic mean path.
    fn encode_window<R: Rng + ?Sized>(
        &self,
        audio: &[f32],
        frame_noise: Option<f32>,
        rng: &mut R,
    ) -> Result<Vec<f32>, String> {
        let acoustic_mean = self.acoustic_encoder.forward(audio)?;
        let frames = acoustic_mean.len() / self.config.acoustic_vae_dim;
        if frames == 0 {
            return Err("audio window encoded to zero latent frames".into());
        }
        let acoustic_latents: Vec<f32> = match frame_noise {
            None => acoustic_mean,
            Some(noise_scale) => acoustic_mean
                .iter()
                .map(|&mean| mean + noise_scale * standard_normal(rng))
                .collect(),
        };
        let semantic_mean = self.semantic_encoder.forward(audio)?;

        let mut connector_scratch = Vec::new();
        let mut acoustic_rows = Vec::new();
        self.acoustic_connector.forward(
            &acoustic_latents,
            frames,
            &mut connector_scratch,
            &mut acoustic_rows,
        )?;
        let mut semantic_rows = Vec::new();
        self.semantic_connector.forward(
            &semantic_mean,
            frames,
            &mut connector_scratch,
            &mut semantic_rows,
        )?;
        let combined: Vec<f32> = acoustic_rows
            .iter()
            .zip(semantic_rows.iter())
            .map(|(a, s)| a + s)
            .collect();
        Ok(combined)
    }
}

/// Standard-normal sample via Box–Muller (the latent noise is statistical
/// augmentation, not a security primitive; only its distribution matters).
fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u1: f64 = f64::from(rng.gen::<f32>()).max(1e-9);
    let u2: f32 = rng.gen();
    let magnitude = (-2.0 * u1.ln()).sqrt();
    let angle = std::f64::consts::TAU * f64::from(u2);
    (magnitude * angle.sin()) as f32
}

fn resolve_special_token(tokenizer: &BPETokenizer, literal: &str) -> Result<u32, String> {
    tokenizer
        .token_id(literal)
        .ok_or_else(|| format!("tokenizer is missing required token {literal}"))
}

fn build_prompt_text(context_info: Option<&str>) -> String {
    let keys = "speaker, content";
    match context_info
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(info) => format!(
            "You are a helpful assistant that transcribes audio input into text output. \
Please transcribe the following audios streamingly with these keys: {keys} \
and extra info: {info}\n"
        ),
        None => format!(
            "You are a helpful assistant that transcribes audio input into text output. \
Please transcribe the following audios streamingly with these keys: {keys}\n"
        ),
    }
}

/// Resample `samples` to the tokenizer sample rate when needed.
fn resample_to_model_rate(samples: &[f32], input_rate: usize, target_rate: usize) -> Vec<f32> {
    if input_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }
    crate::models::dots::speaker::Resampler::new(input_rate as u32, target_rate as u32)
        .resample(samples)
}

pub(crate) fn validate_audio_samples(samples: &[f32]) -> Result<(), String> {
    if samples.is_empty() {
        return Err("audio contains no samples".into());
    }
    if samples.iter().any(|value| !value.is_finite()) {
        return Err("audio contains non-finite samples".into());
    }
    Ok(())
}

fn streaming_windows(
    total_samples: usize,
    chunk_samples: usize,
    window_samples: usize,
) -> Result<Vec<(usize, usize)>, String> {
    if total_samples == 0 || chunk_samples == 0 || window_samples < chunk_samples {
        return Err("invalid VibeVoice streaming window dimensions".into());
    }
    let count = total_samples.div_ceil(chunk_samples);
    let mut windows = Vec::new();
    windows
        .try_reserve_exact(count)
        .map_err(|_| "failed to allocate VibeVoice streaming windows".to_string())?;
    for index in 0..count {
        let start = index
            .checked_mul(chunk_samples)
            .ok_or_else(|| "VibeVoice streaming window offset overflow".to_string())?;
        let end = start.saturating_add(window_samples).min(total_samples);
        windows.push((start, end));
    }
    Ok(windows)
}

fn required_session_capacity(
    prompt_tokens: usize,
    windows: usize,
    audio_rows: usize,
    max_new_tokens_per_chunk: usize,
) -> Result<usize, String> {
    let per_window = audio_rows
        .checked_add(max_new_tokens_per_chunk)
        .and_then(|rows| rows.checked_add(3))
        .ok_or_else(|| "VibeVoice ASR session capacity overflow".to_string())?;
    windows
        .checked_mul(per_window)
        .and_then(|rows| prompt_tokens.checked_add(rows))
        .ok_or_else(|| "VibeVoice ASR session capacity overflow".to_string())
}

/// Streaming transcription over 24 kHz mono samples.
///
/// `on_chunk` fires as `(chunk_index, total_chunks, chunk_text)` as soon as a
/// chunk's text is decoded, mirroring the official streaming demo.
pub fn transcribe_streaming(
    model: &VibeVoiceAsrModel,
    samples: &[f32],
    input_sample_rate: usize,
    options: &TranscribeOptions,
    mut on_chunk: impl FnMut(usize, usize, &str),
) -> Result<String, String> {
    let config = &model.config;
    if options.max_new_tokens_per_chunk == 0 {
        return Err("max_new_tokens_per_chunk must be greater than zero".into());
    }
    if input_sample_rate == 0 {
        return Err("input_sample_rate must be greater than zero".into());
    }
    let samples = resample_to_model_rate(samples, input_sample_rate, config.sample_rate);
    validate_audio_samples(&samples)?;

    let speech_start = resolve_special_token(&model.tokenizer, "<|object_ref_start|>")?;
    let speech_end = resolve_special_token(&model.tokenizer, "<|object_ref_end|>")?;
    let text_chunk_end = resolve_special_token(&model.tokenizer, "<|text_chunk_end|>")?;
    let eos = model
        .tokenizer
        .eos_id()
        .ok_or("tokenizer metadata is missing eos_token_id")?;

    let encode_options = EncodeOptions {
        add_special: false,
        parse_special: true,
    };
    let prompt_ids = model.tokenizer.encode(
        &build_prompt_text(options.context_info.as_deref()),
        encode_options,
    );
    if prompt_ids.is_empty() {
        return Err("prompt tokenized to zero tokens".into());
    }

    let chunk_samples = config.chunk_samples();
    let window_samples = config.window_samples();
    let windows = streaming_windows(samples.len(), chunk_samples, window_samples)?;
    let audio_rows = config.latent_frames(window_samples);
    let session_capacity = required_session_capacity(
        prompt_ids.len(),
        windows.len(),
        audio_rows,
        options.max_new_tokens_per_chunk,
    )?;
    if session_capacity > model.llm.config.n_ctx {
        return Err(format!(
            "VibeVoice ASR requires {session_capacity} context rows, model limit is {}",
            model.llm.config.n_ctx
        ));
    }
    if std::env::var_os("VIBEVOICE_ASR_DEBUG").is_some() {
        eprintln!("[debug] prompt ids: {prompt_ids:?}");
    }
    let mut session = AsrLlmSession::new(&model.llm, session_capacity)?;
    for &token in &prompt_ids {
        session.forward_step(AsrInputRow::Token(token))?;
    }
    session.debug_summary("after prompt prefill");
    if std::env::var_os("VIBEVOICE_ASR_DEBUG").is_some() {
        let logits = session.logits()?;
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
        let top: Vec<String> = order
            .iter()
            .take(5)
            .map(|&i| format!("{}:{:.3}", i, logits[i]))
            .collect();
        eprintln!("[debug] prefill logits top={top:?}");
    }

    let mut rng = match options.seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::from_entropy(),
    };
    let noise_scale = if options.deterministic_latents || config.acoustic_std_dist_type == "none" {
        None
    } else {
        Some(config.acoustic_fix_std / 0.8)
    };

    let total_windows = windows.len();
    let mut transcript = String::new();
    for (window_index, (text_start, audio_end)) in windows.into_iter().enumerate() {
        let mut window = samples[text_start..audio_end].to_vec();
        if window.len() < window_samples {
            // the official path zero-pads the final window to full length
            window.resize(window_samples, 0.0);
        }

        let frame_noise = noise_scale.map(|scale| {
            // per-window scalar: randn·(fix_std/0.8)
            let u1: f64 = f64::from(rng.gen::<f32>()).max(1e-9);
            let u2: f32 = rng.gen();
            let magnitude = (-2.0 * u1.ln()).sqrt();
            let angle = std::f64::consts::TAU * f64::from(u2);
            (magnitude * angle.sin() * scale as f64) as f32
        });
        let features = model.encode_window(&window, frame_noise, &mut rng)?;
        let frames = features.len() / config.llm_hidden_size;
        if frames != audio_rows || features.len() != frames * config.llm_hidden_size {
            return Err(format!(
                "VibeVoice encoder produced {frames} rows; expected {audio_rows}"
            ));
        }

        session.forward_step(AsrInputRow::Token(speech_start))?;
        for row in features.chunks_exact(config.llm_hidden_size) {
            session.forward_step(AsrInputRow::Embedding(row))?;
        }
        session.forward_step(AsrInputRow::Token(speech_end))?;

        let mut chunk_tokens: Vec<u32> = Vec::new();
        for _ in 0..options.max_new_tokens_per_chunk {
            let next = session.sample_argmax()?;
            if next == text_chunk_end || next == eos {
                break;
            }
            chunk_tokens.push(next);
            session.forward_step(AsrInputRow::Token(next))?;
        }
        // the official loop always feeds the chunk-end embedding afterwards
        session.forward_step(AsrInputRow::Token(text_chunk_end))?;

        let chunk_text = model.tokenizer.decode(&chunk_tokens, false);
        on_chunk(window_index, total_windows, &chunk_text);
        transcript.push_str(&chunk_text);
    }
    Ok(transcript)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_windows_keep_lookahead_and_partial_tail() {
        assert_eq!(
            streaming_windows(140_801, 70_400, 83_200).unwrap(),
            vec![(0, 83_200), (70_400, 140_801), (140_800, 140_801)]
        );
        assert_eq!(
            streaming_windows(83_200, 70_400, 83_200).unwrap(),
            vec![(0, 83_200), (70_400, 83_200)]
        );
        assert!(streaming_windows(0, 70_400, 83_200).is_err());
    }

    #[test]
    fn required_capacity_is_checked_and_includes_chunk_end() {
        assert_eq!(required_session_capacity(10, 2, 26, 4).unwrap(), 76);
        assert!(required_session_capacity(usize::MAX, 1, 26, 4).is_err());
        assert!(required_session_capacity(10, usize::MAX, 26, 4).is_err());
    }

    #[test]
    fn audio_validation_rejects_any_non_finite_value() {
        assert!(validate_audio_samples(&[]).is_err());
        assert!(validate_audio_samples(&[0.0, f32::NAN]).is_err());
        assert!(validate_audio_samples(&[0.0, f32::INFINITY]).is_err());
        assert!(validate_audio_samples(&[0.0, -0.5, 1.0]).is_ok());
    }
}
