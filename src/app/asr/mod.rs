//! ASR CLI dispatcher and model-specific sub-pipelines.
//!
//! Dispatches based on mmproj architecture:
//! - `vibevoice_asr` → VibeVoice ASR streaming pipeline
//! - `funasr-sensevoice-encoder` → Fun-ASR-Nano pipeline
//! - default → Qwen3-VL ASR pipeline

mod funasr;
mod vibevoice;

use crate::app::cli::{resolve_thread_count, transcription_options};
use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::models::qwen3::asr::model::{open_bundled_audio_source, AsrRuntime};
use crate::models::qwen3::Qwen3Model;
use std::sync::Arc;
use std::time::Instant;

pub fn run_asr_cli(
    options: &crate::app::cli::CliOptions,
    prefill_batch_size: usize,
) -> Result<(), String> {
    let started = Instant::now();
    if let Some(mmproj_path) = options
        .mmproj
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())
    {
        let probe = open_or_exit(mmproj_path, ComponentRole::Mmproj);
        let is_vibevoice = crate::models::vibevoice_asr::is_vibevoice_asr_mmproj(probe.as_ref());
        let is_funasr = crate::models::funasr::is_funasr_encoder(probe.as_ref());
        drop(probe);
        if is_vibevoice {
            return vibevoice::run_vibevoice_asr_cli(options);
        }
        if is_funasr {
            return funasr::run_funasr_cli(options, prefill_batch_size);
        }
    }
    run_qwen3_asr_cli(options, prefill_batch_size, started)
}

/// Qwen3-VL ASR pipeline (default when no VibeVoice/FunASR mmproj detected).
fn run_qwen3_asr_cli(
    options: &crate::app::cli::CliOptions,
    prefill_batch_size: usize,
    started: Instant,
) -> Result<(), String> {
    let llm_source: Arc<dyn TensorSource> =
        Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
    let arch = llm_source
        .metadata("general.architecture")
        .and_then(|value| value.to_string_val())
        .unwrap_or_default();
    crate::app::reject_incomplete_z_image_architecture(arch)?;
    eprintln!("Loading ASR decoder from {}", options.model.display());
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
    let decoder = Arc::new(Qwen3Model::from_source(llm_source, tokenizer, pool)?);
    if decoder.config().architecture != "qwen3vl" && decoder.config().architecture != "qwen3" {
        return Err("--audio requires a qwen3vl or qwen3 decoder".into());
    }
    let load_decoder_done = started.elapsed();
    let audio_source: Arc<dyn TensorSource> = match options.mmproj.as_deref() {
        Some(path) => Arc::from(open_or_exit(path, ComponentRole::Mmproj)),
        None => {
            open_bundled_audio_source(&options.model)?.ok_or("raw GGUF ASR requires --mmproj")?
        }
    };
    let runtime = AsrRuntime::new(decoder, audio_source, prefill_batch_size)
        .map_err(|error| error.to_string())?;
    let load_runtime_done = started.elapsed();
    let audio = options.audio.as_ref().expect("validated audio option");
    let wav = std::fs::read(audio)
        .map_err(|error| format!("Failed to read {}: {error}", audio.display()))?;
    let result = runtime
        .transcribe_wav(&wav, &transcription_options(options))
        .map_err(|error| error.to_string())?;
    let total = started.elapsed();
    eprintln!(
        "ASR: {} prompt tokens, {} audio tokens, {} output tokens in {:.3}s",
        result.prompt_tokens,
        result.audio_tokens,
        result.token_ids.len(),
        total.as_secs_f64(),
    );
    eprintln!(
        "    load_decoder={:.3}s load_runtime={:.3}s transcribe={:.3}s",
        load_decoder_done.as_secs_f64(),
        (load_runtime_done - load_decoder_done).as_secs_f64(),
        (total - load_runtime_done).as_secs_f64(),
    );
    println!("{}", result.text);
    Ok(())
}
