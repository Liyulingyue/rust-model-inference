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
    synthesize, GenerateOptions, DEFAULT_EOS_THRESHOLD, DEFAULT_GUIDANCE, DEFAULT_SPEAKER_SCALE,
};
use crate::models::dots::DotsTtsModel;

/// Run the dots.tts pipeline (base + edit share the same core loop; edit adds
/// source-prefill spans, handled by a follow-up schedule builder — the current
/// stage synthesizes from text via the base template).
pub fn run_dots_tts_cli(options: &CliOptions) -> Result<(), String> {
    let started = Instant::now();
    let prompt_text = options
        .prompt
        .as_deref()
        .filter(|prompt| !prompt.trim().is_empty())
        .ok_or_else(|| "--tts requires --prompt".to_string())?;
    let language = normalize_tts_language(options.language.as_deref())?;
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
    let arch = mmproj_source
        .metadata("general.architecture")
        .and_then(|value| value.to_string_val())
        .unwrap_or_default();
    if arch != "dotstts" {
        return Err(format!(
            "--tts mmproj architecture must be dotstts, got {arch}"
        ));
    }

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
    let _ = language;

    // sampling options
    let mut options_out = GenerateOptions {
        max_patches: options.max_tokens.unwrap_or(64),
        temperature: options.temperature.unwrap_or(0.9),
        nfe: 10,
        guidance: DEFAULT_GUIDANCE,
        speaker_scale: DEFAULT_SPEAKER_SCALE,
        eos_threshold: DEFAULT_EOS_THRESHOLD,
    };
    if let Some(steps) = options.steps {
        if steps > 0 {
            options_out.nfe = steps;
        }
    }
    if options.seed.is_some() {
        options_out.temperature = 0.9; // keep sampling; seed only fixes the RNG
    }

    // prompt conditioning (single RNG for the whole request when seeded)
    let mut rng = make_rng(options.seed);
    let prompt = if let Some(ref_audio) = options.ref_audio.as_deref() {
        eprintln!("Encoding prompt reference from {}", ref_audio.display());
        let wav = read_pcm16_wav_48k(ref_audio)?;
        let conditioning =
            model.prepare_prompt_conditioning(&wav, options_out.speaker_scale, &mut rng)?;
        if options.ref_text.is_none() {
            eprintln!("note: no --ref-text; using reference audio for speaker only (no patch prefill)");
        }
        Some(conditioning)
    } else {
        None
    };
    // with --ref-text the schedule text is "ref text\n target text" (runtime
    // prompt_text + text concatenation), and the leading spans become the
    // prompt-prefill audio
    let full_text = match options.ref_text.as_deref().filter(|t| !t.trim().is_empty()) {
        Some(ref_text) => format!("{ref_text}\n{prompt_text}"),
        None => prompt_text.to_string(),
    };
    eprintln!(
        "Synthesizing {full_text:?} (max {} patches, nfe {})",
        options_out.max_patches, options_out.nfe
    );
    let waveform = synthesize(
        &model,
        &tokenizer,
        &full_text,
        prompt.as_ref(),
        &options_out,
        &mut rng,
    )?;
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

/// Read a PCM16 WAV at any sample rate, mix to mono, and resample to 48 kHz
/// (the runtime `high_quality_resample` contract).
fn read_pcm16_wav_48k(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("Failed to read WAV {}: {error}", path.display()))?;
    let decoded =
        crate::models::qwen3::asr::audio_processor::decode_pcm16_wav_any(&bytes).map_err(|e| {
            format!("Failed to decode WAV {}: {e:?}", path.display())
        })?;
    let mono: Vec<f32> = if decoded.channels == 1 {
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
    if decoded.sample_rate == 48_000 {
        return Ok(mono);
    }
    let resampler = crate::models::dots::speaker::Resampler::new(
        decoded.sample_rate,
        48_000,
    );
    Ok(resampler.resample(&mono))
}