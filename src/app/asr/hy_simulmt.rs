use crate::app::cli::{resolve_thread_count, CliOptions};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::format::ggufrs::ComponentRole;
use crate::models::hy_simulmt::HySimulMt;
use crate::models::qwen3::asr::audio_processor::decode_pcm16_wav;
use std::sync::Arc;

pub(super) fn run(options: &CliOptions, batch: usize) -> Result<(), String> {
    if options.mmproj.is_some() {
        return Err("Hy-SimulMT GGUF already contains the audio encoder; omit --mmproj".into());
    }
    if options.temperature.is_some_and(|t| t != 0.0) {
        return Err("Hy-SimulMT requires greedy decoding (--temp 0)".into());
    }
    let _cpu = crate::compute::ComputePolicy::Cpu.cpu_scope();
    let source: Arc<dyn TensorSource> =
        Arc::from(crate::app::open_or_exit(&options.model, ComponentRole::Llm));
    let available = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let pool = Arc::new(ComputePool::new(resolve_thread_count(
        options.threads,
        available,
    )));
    let model = HySimulMt::from_source(source, pool)?;
    let path = options
        .audio
        .as_ref()
        .ok_or("Hy-SimulMT requires --audio")?;
    let wav = std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let samples = decode_pcm16_wav(&wav).map_err(|e| format!("Hy-SimulMT WAV: {e:?}"))?;
    let started = std::time::Instant::now();
    let result = model.translate(
        &samples,
        options.language.as_deref().unwrap_or("zh"),
        options.prompt.as_deref(),
        options.max_tokens.unwrap_or(512),
        options.effective_max_context(),
        batch,
        options.kv_format,
    )?;
    println!("{}", result.text);
    eprintln!(
        "Hy-SimulMT CPU: {} turns, {} audio tokens, {} prompt tokens, {} output tokens, {:.3}s",
        result.turns.len(),
        result.audio_tokens,
        result.prompt_tokens,
        result.token_ids.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
