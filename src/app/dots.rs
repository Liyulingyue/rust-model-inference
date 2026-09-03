//! dots.tts CLI pipeline: `--tts` with a `dotstts` mmproj.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rand::SeedableRng;

use crate::app::cli::{normalize_tts_language, resolve_thread_count, CliOptions};
use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::models::dots::generate::{
    synthesize_request, GenerateOptions, GenerationRequest, PromptConditioning,
};
use crate::models::dots::{resolve_edit_request, DotsTtsModel};

/// Run the dots.tts base or edit pipeline through the shared runtime loop.
pub fn run_dots_tts_cli(options: &CliOptions) -> Result<(), String> {
    let started = Instant::now();
    let mmproj_path = options
        .mmproj
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| "--tts requires --mmproj".to_string())?;
    let out_path = options
        .out
        .as_deref()
        .ok_or_else(|| "--tts requires --out".to_string())?;

    eprintln!("Loading dots.tts LLM from {}", options.model.display());
    let llm_source: Arc<dyn TensorSource> =
        Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
        llm_source.metadata(key).cloned()
    })?);
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let pool = Arc::new(ComputePool::new(resolve_thread_count(
        options.threads,
        available,
    )));

    eprintln!("Loading dots.tts mmproj from {}", mmproj_path.display());
    let mmproj_source: Arc<dyn TensorSource> =
        Arc::from(open_or_exit(mmproj_path, ComponentRole::Mmproj));
    let model = DotsTtsModel::from_sources(llm_source, mmproj_source, pool)?;
    eprintln!(
        "dots.tts: sr={} hop={} patch={} latent={} fm={} patches_llm={}",
        model.config.sample_rate,
        model.config.hop_size,
        model.config.patch_size,
        model.config.latent_dim,
        model.config.fm_hidden_size,
        model.config.llm_hidden_size,
    );
    let mut options_out = GenerateOptions::for_model(&model.config);
    if let Some(max_patches) = options.max_tokens {
        options_out.max_patches = max_patches;
    }
    if let Some(temperature) = options.temperature {
        options_out.temperature = temperature;
    }
    if let Some(steps) = options.steps {
        if steps > 0 {
            options_out.nfe = steps;
        }
    }
    if options.seed.is_some() {
        options_out.temperature = 0.9; // keep sampling; seed only fixes the RNG
    }

    let mut rng = make_rng(options.seed);
    let waveform = if options.edit {
        let source_path = options
            .source_audio
            .as_deref()
            .ok_or_else(|| "--tts --edit requires --source-audio".to_string())?;
        let instruction = options
            .instruction
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "--tts --edit requires a non-empty --instruction".to_string())?;
        let edit = resolve_edit_request(
            instruction,
            options.source_text.as_deref(),
            options.target_text.as_deref(),
            options.use_xvector,
        )?;
        eprintln!("Encoding edit source from {}", source_path.display());
        let source_wav = read_dots_wav(
            source_path,
            DotsWavMode::Edit {
                samples_per_patch: model.config.samples_per_patch(),
            },
        )?;
        let source = model.prepare_prompt_conditioning(
            &source_wav,
            options_out.speaker_scale,
            edit.use_xvector,
            0,
            &mut rng,
        )?;
        eprintln!(
            "Editing {:?} -> {:?} (max {} target patches, nfe {}, xvector {})",
            edit.source_text,
            edit.target_text,
            options_out.max_patches,
            options_out.nfe,
            edit.use_xvector,
        );
        synthesize_request(
            &model,
            &tokenizer,
            GenerationRequest::Edit {
                source_text: &edit.source_text,
                instruction,
                target_text: &edit.target_text,
                source: &source,
            },
            &options_out,
            &mut rng,
        )?
    } else {
        let target = options
            .prompt
            .as_deref()
            .filter(|prompt| !prompt.trim().is_empty())
            .ok_or_else(|| "--tts requires --prompt".to_string())?;
        let language = normalize_tts_language(options.language.as_deref())?;
        let reference_text = options.ref_text.as_deref().map(str::trim).filter(|v| !v.is_empty());
        let prompt = if let Some(ref_audio) = options.ref_audio.as_deref() {
            eprintln!("Encoding prompt reference from {}", ref_audio.display());
            let wav = read_dots_wav(ref_audio, DotsWavMode::Base)?;
            if reference_text.is_some() {
                Some(model.prepare_prompt_conditioning(
                    &wav,
                    options_out.speaker_scale,
                    true,
                    1,
                    &mut rng,
                )?)
            } else {
                let xvec = model.encode_speaker(&wav)?;
                let g_cond = model.speaker_condition(&xvec, options_out.speaker_scale)?;
                eprintln!(
                    "note: no --ref-text; using reference audio for speaker only (no patch prefill)"
                );
                Some(PromptConditioning {
                    patches: Vec::new(),
                    g_cond,
                })
            }
        } else {
            None
        };
        let schedule_text = base_schedule_text(target, reference_text, language)?;
        eprintln!(
            "Synthesizing {schedule_text:?} (max {} target patches, nfe {})",
            options_out.max_patches, options_out.nfe
        );
        synthesize_request(
            &model,
            &tokenizer,
            GenerationRequest::Base {
                text: &schedule_text,
                prompt: prompt.as_ref(),
            },
            &options_out,
            &mut rng,
        )?
    };
    let sample_rate = model.config.sample_rate as u32;
    crate::models::qwen3::tts::codec::write_wav_f32(out_path, &waveform, sample_rate)
        .map_err(|error| format!("WAV write failed: {error}"))?;
    eprintln!(
        "dots.tts: {} samples ({} s @ {} Hz) written to {} in {:.2}s",
        waveform.len(),
        waveform.len() as f64 / f64::from(sample_rate),
        sample_rate,
        out_path.display(),
        started.elapsed().as_secs_f64(),
    );
    Ok(())
}

fn make_rng(seed: Option<i64>) -> rand::rngs::StdRng {
    match seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed as u64),
        None => rand::rngs::StdRng::from_entropy(),
    }
}

fn base_schedule_text(
    target: &str,
    reference: Option<&str>,
    language: &str,
) -> Result<String, String> {
    let target = target.trim();
    if target.is_empty() {
        return Err("dots TTS target text must not be empty".into());
    }
    let tag = match language {
        "chinese" => "[ZH]",
        "english" => "[EN]",
        "german" => "[DE]",
        "italian" => "[IT]",
        "portuguese" => "[PT]",
        "spanish" => "[ES]",
        "japanese" => "[JA]",
        "korean" => "[KO]",
        "french" => "[FR]",
        "russian" => "[RU]",
        value => return Err(format!("Unsupported normalized TTS language {value:?}")),
    };
    match reference.map(str::trim).filter(|text| !text.is_empty()) {
        Some(reference) => Ok(format!("{tag}{reference}\n{target}")),
        None => Ok(format!("{tag}{target}")),
    }
}

fn normalize_edge_silence(
    wav: &[f32],
    sample_rate: usize,
    top_db: f32,
    target_ms: usize,
) -> Result<Vec<f32>, String> {
    if wav.is_empty() || sample_rate == 0 || !top_db.is_finite() || top_db < 0.0 {
        return Err("invalid edge-silence normalization input".into());
    }
    let finite_peak = wav
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(f32::abs)
        .reduce(f32::max)
        .ok_or_else(|| "waveform contains no finite samples".to_string())?;
    let target = sample_rate
        .checked_mul(target_ms)
        .and_then(|samples| samples.checked_add(500))
        .map(|samples| samples / 1000)
        .ok_or_else(|| "edge-silence target overflow".to_string())?;
    if finite_peak == 0.0 {
        return Ok(vec![0.0; target]);
    }
    let threshold = finite_peak * 10.0f32.powf(-top_db / 20.0);
    let first = wav
        .iter()
        .position(|value| value.is_finite() && value.abs() > threshold)
        .ok_or_else(|| "waveform has no samples above the silence threshold".to_string())?;
    let last = wav
        .iter()
        .rposition(|value| value.is_finite() && value.abs() > threshold)
        .ok_or_else(|| "waveform has no samples above the silence threshold".to_string())?;
    let leading = first;
    let trailing = wav.len() - last - 1;
    let mut normalized = wav.to_vec();
    if leading < target {
        let mut padded = vec![0.0; target - leading];
        padded.extend_from_slice(&normalized);
        normalized = padded;
    } else {
        normalized.drain(..leading - target);
    }
    if trailing < target {
        normalized.resize(normalized.len() + target - trailing, 0.0);
    } else {
        normalized.truncate(normalized.len() - (trailing - target));
    }
    Ok(normalized)
}

#[derive(Clone, Copy)]
enum DotsWavMode {
    Base,
    Edit { samples_per_patch: usize },
}

fn trim_edge_silence_frame_rms(wav: &[f32], top_db: f32) -> Result<Vec<f32>, String> {
    const FRAME: usize = 2048;
    const HOP: usize = 512;
    let mut padded = vec![0.0f32; wav.len() + FRAME];
    padded[FRAME / 2..FRAME / 2 + wav.len()].copy_from_slice(wav);
    let frame_count = (padded.len() - FRAME) / HOP + 1;
    let mut rms = Vec::with_capacity(frame_count);
    for start in (0..frame_count).map(|frame| frame * HOP) {
        let sum = padded[start..start + FRAME]
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        rms.push((sum / FRAME as f64).sqrt());
    }
    let peak = rms.iter().copied().fold(0.0f64, f64::max).max(1e-5);
    let threshold = 20.0 * peak.log10() - f64::from(top_db);
    let first = rms
        .iter()
        .position(|value| 20.0 * value.max(1e-5).log10() > threshold)
        .unwrap_or(0);
    let last = rms
        .iter()
        .rposition(|value| 20.0 * value.max(1e-5).log10() > threshold)
        .unwrap_or(frame_count - 1);
    let start = (first * HOP).min(wav.len());
    let end = ((last + 1) * HOP).min(wav.len());
    Ok(wav[start..end].to_vec())
}

/// Decode PCM16, mix to mono, and apply the phase-specific official preprocessing.
fn read_dots_wav(path: &Path, mode: DotsWavMode) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("Failed to read WAV {}: {error}", path.display()))?;
    let decoded =
        crate::models::qwen3::asr::audio_processor::decode_pcm16_wav_any(&bytes).map_err(|e| {
            format!("Failed to decode WAV {}: {e:?}", path.display())
        })?;
    let mut mono: Vec<f32> = if decoded.channels == 1 {
        decoded.samples
    } else {
        decoded
            .samples
            .chunks_exact(decoded.channels as usize)
            .map(|frame| {
                frame.iter().map(|&s| s as f64).sum::<f64>() / frame.len() as f64
            })
            .map(|v| v as f32)
            .collect()
    };
    if mono.is_empty() {
        return Err(format!("WAV {} contains no samples", path.display()));
    }
    if !mono.iter().any(|value| value.is_finite()) {
        return Err(format!("WAV {} contains no finite samples", path.display()));
    }
    if matches!(mode, DotsWavMode::Base) {
        mono = trim_edge_silence_frame_rms(&mono, 30.0)?;
    }
    if decoded.sample_rate != 48_000 {
        let width = match mode {
            DotsWavMode::Base => 64,
            DotsWavMode::Edit { .. } => 128,
        };
        mono = crate::models::dots::speaker::Resampler::with_width(
            decoded.sample_rate,
            48_000,
            width,
        )
        .resample(&mono);
    }
    if let DotsWavMode::Edit { samples_per_patch } = mode {
        let mut normalized = normalize_edge_silence(&mono, 48_000, 30.0, 250)?;
        if samples_per_patch == 0 {
            return Err("dots samples per patch must be positive".into());
        }
        let remainder = normalized.len() % samples_per_patch;
        if remainder != 0 {
            normalized.resize(normalized.len() + samples_per_patch - remainder, 0.0);
        }
        return Ok(normalized);
    }
    Ok(mono)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_silence_normalization_keeps_exactly_250ms_per_side() {
        let mut wav = vec![0.0; 20_000];
        wav[2_000..8_000].fill(0.5);
        let normalized = normalize_edge_silence(&wav, 48_000, 30.0, 250).unwrap();
        assert_eq!(normalized.len(), 12_000 + 6_000 + 12_000);
        assert!(normalized[..12_000].iter().all(|v| *v == 0.0));
        assert!(normalized[18_000..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn edge_silence_normalization_preserves_quiet_existing_edges() {
        let wav = vec![0.001, 0.001, 0.001, 1.0, 1.0, 0.001, 0.001, 0.001];
        let normalized = normalize_edge_silence(&wav, 1_000, 30.0, 2).unwrap();
        assert_eq!(normalized, vec![0.001, 0.001, 1.0, 1.0, 0.001, 0.001]);
    }

    #[test]
    fn base_schedule_text_places_language_on_the_conditioned_surface() {
        assert_eq!(
            base_schedule_text("hello", None, "english").unwrap(),
            "[EN]hello"
        );
        assert_eq!(
            base_schedule_text("target", Some("reference"), "english").unwrap(),
            "[EN]reference\ntarget",
        );
        assert_eq!(
            base_schedule_text("  target  ", Some("  reference  "), "english").unwrap(),
            "[EN]reference\ntarget",
        );
    }
}
