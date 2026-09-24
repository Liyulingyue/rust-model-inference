use std::sync::Arc;
use std::time::Instant;

use super::cli::YuE2CliOptions;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::models::qwen3::tts::codec::write_wav_f32_channels;
use crate::models::yue2::{
    interleave_stereo, SamplingConfig, YuE2GenerateOptions, YuE2Model, YuE2Request, YuE2Vae,
};

pub fn run_yue2_cli(options: YuE2CliOptions, n_threads: usize) -> Result<(), String> {
    let started = Instant::now();
    let main: Arc<dyn TensorSource> = Arc::from(
        open_model_source(&options.model, ComponentRole::Llm).map_err(|error| {
            format!(
                "Failed to load YuE2 main component from {}: {error}",
                options.model.display()
            )
        })?,
    );
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
        main.metadata(key).cloned()
    })?);
    let model = YuE2Model::from_source(
        Arc::clone(&main),
        tokenizer,
        Arc::new(ComputePool::new(n_threads)),
    )?;
    let vae_source: Arc<dyn TensorSource> = Arc::from(
        open_model_source(&options.vae, ComponentRole::Llm).map_err(|error| {
            format!(
                "Failed to load YuE2 VAE component from {}: {error}",
                options.vae.display()
            )
        })?,
    );
    let vae = YuE2Vae::from_source(vae_source)?;
    let request = YuE2Request::new(options.style, options.lyrics, options.seed)?;
    let generation = model.generate(
        &vae,
        &request,
        &YuE2GenerateOptions {
            abc: SamplingConfig::abc(),
            semantic: options.semantic,
            steps: options.steps,
        },
    )?;
    let interleaved = interleave_stereo(
        &generation.channel_major_audio,
        generation.samples_per_channel,
    )?;
    write_wav_f32_channels(&options.out, &interleaved, 48_000, 2)
        .map_err(|error| error.to_string())?;

    println!("YuE2 abc stage: {} tokens", generation.abc_ids.len());
    println!(
        "YuE2 semantic stage: {} tokens",
        generation.semantic_ids.len()
    );
    println!("YuE2 nar stage: {} latent frames", generation.latent_frames);
    println!(
        "YuE2 vae stage: {} stereo samples per channel",
        generation.samples_per_channel
    );
    println!(
        "YuE2 generation completed in {:.3}s: {}",
        started.elapsed().as_secs_f64(),
        options.out.display()
    );
    Ok(())
}
