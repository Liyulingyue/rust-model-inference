use crate::app::cli::resolve_thread_count;
use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::models::tts::{Qwen3TtsTalker, TTS_DEFAULT_TEMP, AUDIO_CODEBOOK_SIZE};
use std::sync::Arc;
use std::time::Instant;

pub fn run_tts_cli(options: &crate::app::cli::CliOptions) -> Result<(), String> {
    let started = Instant::now();
    eprintln!("Loading TTS talker from {}", options.model.display());
    let source: Arc<dyn TensorSource> = Arc::from(
        open_or_exit(&options.model, ComponentRole::Llm),
    );
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
        source.metadata(key).cloned()
    })?);
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let pool = Arc::new(ComputePool::new(resolve_thread_count(
        options.threads,
        available,
    )));
    let talker = Qwen3TtsTalker::from_source(source, tokenizer, pool)?;

    let prompt = options.prompt.as_deref().unwrap_or_default();
    if prompt.is_empty() {
        return Err("--tts requires --prompt".into());
    }

    let max_tokens = options.max_tokens.unwrap_or(128);
    let temperature = options.temperature.unwrap_or(TTS_DEFAULT_TEMP);

    eprintln!(
        "TTS: architecture={} layers={} vocab={} audio_codebook={} eos={}",
        talker.config().architecture,
        talker.config().n_layer,
        talker.config().vocab_size,
        talker.config().audio_codebook_size,
        talker.config().eos_token_id,
    );

    let generation = talker.synthesize(prompt, None, max_tokens, temperature)?;
    let elapsed = started.elapsed();
    eprintln!(
        "TTS: {} prompt chars, {} audio tokens ({:?}) in {:.3}s",
        prompt.chars().count(),
        generation.audio_token_ids.len(),
        generation.finished_reason,
        elapsed.as_secs_f64(),
    );

    if let Some(path) = options.out.as_deref() {
        let mut bytes = Vec::with_capacity(generation.audio_token_ids.len() * 4);
        for &token in &generation.audio_token_ids {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        std::fs::write(path, &bytes)
            .map_err(|error| format!("Failed to write {}: {error}", path.display()))?;
        eprintln!(
            "wrote {} audio tokens ({} bytes) to {}",
            generation.audio_token_ids.len(),
            bytes.len(),
            path.display(),
        );
    } else {
        println!(
            "audio_codes[{}]: {:?}",
            generation.audio_token_ids.len(),
            &generation.audio_token_ids[..generation.audio_token_ids.len().min(64)],
        );
        if generation.audio_token_ids.len() > 64 {
            println!("... ({} more tokens)", generation.audio_token_ids.len() - 64);
        }
        println!(
            "codebook_size={} eos_token_id={} finished={:?}",
            AUDIO_CODEBOOK_SIZE, generation.prompt_token_ids.last().copied().unwrap_or(0), generation.finished_reason,
        );
    }
    Ok(())
}