//! VibeVoice ASR CLI pipeline: `--audio` with a `vibevoice_asr` mmproj.

use std::sync::Arc;
use std::time::Instant;

use crate::app::cli::resolve_thread_count;
use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::models::vibevoice_asr::generate::{
    transcribe_streaming, TranscribeOptions, VibeVoiceAsrModel,
};

/// Run the VibeVoice ASR streaming pipeline.
pub fn run_vibevoice_asr_cli(options: &crate::app::cli::CliOptions) -> Result<(), String> {
    let started = Instant::now();
    let mmproj_path = options
        .mmproj
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| "VibeVoice ASR requires --mmproj".to_string())?;
    let audio_path = options
        .audio
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| "VibeVoice ASR requires --audio".to_string())?;

    eprintln!("Loading VibeVoice ASR LLM from {}", options.model.display());
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

    eprintln!(
        "Loading VibeVoice ASR mmproj from {}",
        mmproj_path.display()
    );
    let mmproj_source: Arc<dyn TensorSource> =
        Arc::from(open_or_exit(mmproj_path, ComponentRole::Mmproj));
    let model = VibeVoiceAsrModel::from_sources(
        Arc::clone(&llm_source),
        mmproj_source,
        Arc::clone(&pool),
        Arc::clone(&tokenizer),
    )?;
    eprintln!(
        "VibeVoice ASR: sr={} frame={}ms chunk={}+{} frames, window={}",
        model.config.sample_rate,
        model.config.compress_ratio * 1000 / model.config.sample_rate,
        model.config.chunk_frames,
        model.config.lookahead_frames,
        model.config.window_samples(),
    );

    let wav_bytes = std::fs::read(audio_path)
        .map_err(|error| format!("Failed to read {}: {error}", audio_path.display()))?;
    let decoded = crate::models::qwen3::asr::audio_processor::decode_pcm16_wav_any(&wav_bytes)
        .map_err(|error| format!("Failed to decode WAV: {error:?}"))?;
    let mono: Vec<f32> = if decoded.channels == 1 {
        decoded.samples
    } else {
        decoded
            .samples
            .chunks_exact(decoded.channels as usize)
            .map(|frame| {
                frame.iter().map(|&sample| f64::from(sample)).sum::<f64>() / frame.len() as f64
            })
            .map(|value| value as f32)
            .collect()
    };
    crate::models::vibevoice_asr::generate::validate_audio_samples(&mono)
        .map_err(|error| format!("Invalid WAV {}: {error}", audio_path.display()))?;
    eprintln!(
        "Decoded {} samples ({} ch, {} Hz)",
        mono.len(),
        decoded.channels,
        decoded.sample_rate
    );
    let transcribe_options = TranscribeOptions {
        max_new_tokens_per_chunk: options
            .max_tokens
            .unwrap_or(crate::models::vibevoice_asr::generate::DEFAULT_MAX_NEW_TOKENS_PER_CHUNK),
        context_info: options.prompt.clone(),
        deterministic_latents: std::env::var_os("VIBEVOICE_ASR_DETERMINISTIC").is_some(),
        seed: options.seed.map(|seed| u64::try_from(seed).unwrap_or(0)),
    };

    let mut chunks: Vec<String> = Vec::new();
    let duration = f64::from(mono.len() as u32) / f64::from(decoded.sample_rate);
    let transcript_start = Instant::now();
    let transcript = transcribe_streaming(
        &model,
        &mono,
        decoded.sample_rate as usize,
        &transcribe_options,
        |index, total, text| {
            chunks.push(text.to_string());
            eprintln!("[{index}/{total}] {text}", index = index + 1);
        },
    )?;
    eprintln!(
        "VibeVoice ASR: {:.2}s audio, {} chunks in {:.2}s (transcribe {:.2}s, RTF {:.3})",
        duration,
        chunks.len(),
        started.elapsed().as_secs_f64(),
        transcript_start.elapsed().as_secs_f64(),
        transcript_start.elapsed().as_secs_f64() / duration.max(1e-9),
    );
    println!("{transcript}");
    if let Some(out_path) = options.out.as_deref() {
        std::fs::write(out_path, format!("{transcript}\n")).map_err(|error| {
            format!("Failed to write transcript {}: {error}", out_path.display())
        })?;
        eprintln!("Transcript written to {}", out_path.display());
    }
    Ok(())
}
