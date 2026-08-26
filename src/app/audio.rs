use crate::app::cli::{resolve_thread_count, transcription_options};
use crate::app::open_or_exit;
use crate::models::qwen3::asr::model::{open_bundled_audio_source, AsrRuntime};
use crate::format::ggufrs::ComponentRole;
use crate::core::tensor::TensorSource;
use crate::models::qwen3::base::Qwen3Model;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use std::sync::Arc;
use std::time::Instant;

pub fn run_asr_cli(options: &crate::app::cli::CliOptions) -> Result<(), String> {
    let started = Instant::now();
    eprintln!("Loading ASR decoder from {}", options.model.display());
    let llm_source: Arc<dyn TensorSource> = Arc::from(
        open_or_exit(&options.model, ComponentRole::Llm),
    );
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
    if decoder.config().architecture != "qwen3vl" {
        return Err("--audio requires a qwen3vl decoder".into());
    }
    let load_decoder_done = started.elapsed();
    let audio_source: Arc<dyn TensorSource> = match options.mmproj.as_deref() {
        Some(path) => Arc::from(
            open_or_exit(path, ComponentRole::Mmproj),
        ),
        None => open_bundled_audio_source(&options.model)?
            .ok_or("raw GGUF ASR requires --mmproj")?,
    };
    let runtime = AsrRuntime::new(decoder, audio_source).map_err(|error| error.to_string())?;
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
