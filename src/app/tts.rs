use crate::app::cli::resolve_thread_count;
use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::models::tts::codec::{
    write_wav_f32, CodePredictor, DacDecoder, RvqDecoder, WaveformTransformer, WAVEFORM_SAMPLE_RATE,
};
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
    let synth_elapsed = started.elapsed();
    eprintln!(
        "TTS: {} prompt chars, {} audio tokens ({:?}) in {:.3}s",
        prompt.chars().count(),
        generation.audio_token_ids.len(),
        generation.finished_reason,
        synth_elapsed.as_secs_f64(),
    );

    let mmproj_path = options
        .mmproj
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty());
    let out_path = options.out.as_deref();

    if mmproj_path.is_none() && out_path.is_none() {
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
            AUDIO_CODEBOOK_SIZE,
            generation.prompt_token_ids.last().copied().unwrap_or(0),
            generation.finished_reason,
        );
        return Ok(());
    }

    let mmproj_path = mmproj_path
        .ok_or_else(|| "Stage 2 (codec decoder) requires --mmproj".to_string())?;
    eprintln!("Loading TTS codec decoder from {}", mmproj_path.display());
    let mmproj_source: Arc<dyn TensorSource> = Arc::from(
        open_or_exit(mmproj_path, ComponentRole::Mmproj),
    );
    let predictor = CodePredictor::from_source(mmproj_source.as_ref())?;
    let rvq = RvqDecoder::from_source(mmproj_source.as_ref())?;
    let tfm = WaveformTransformer::from_source(mmproj_source.as_ref())?;
    let dac = DacDecoder::from_source(mmproj_source.as_ref())?;
    eprintln!(
        "RVQ: {} first + {} rest codebooks, {} dim each",
        rvq.first_codebook_size(),
        crate::models::tts::codec::RVQ_LEVELS - 1,
        crate::models::tts::codec::RVQ_CODE_DIM,
    );

    // Predict 15 residual RVQ indices per Talker audio token.
    let residual_codes = predictor.predict(&generation.audio_token_ids)?;
    let n_tokens = generation.audio_token_ids.len();
    // The Talker emits one audio-codebook id per frame; the first-level
    // codebook index is `id % RVQ_CODEBOOK_SIZE` (the model was trained with
    // first-code indices in [0, 2047] regardless of the 3072-dim id).
    let codes: Vec<u32> = (0..n_tokens)
        .flat_map(|t| {
            let first = generation.audio_token_ids[t]
                % crate::models::tts::codec::RVQ_CODEBOOK_SIZE as u32;
            std::iter::once(first).chain(
                residual_codes
                    [t * (crate::models::tts::codec::RVQ_LEVELS - 1)
                        ..(t + 1) * (crate::models::tts::codec::RVQ_LEVELS - 1)]
                    .iter()
                    .copied(),
            )
        })
        .collect();
    let continuous = rvq.decode(&codes)?;
    let timesteps = continuous.len() / crate::models::tts::codec::RVQ_CODE_DIM;
    eprintln!(
        "RVQ decoded {} timesteps × {} dim",
        timesteps,
        crate::models::tts::codec::RVQ_CODE_DIM
    );

    // Pad 256-dim RVQ embedding to 1024-dim by 4× repetition (channel tile),
    // run the waveform transformer, and feed its output to the DAC entry conv.
    let padded = tile_channels(&continuous, timesteps, 1024);
    let lifted = tfm.forward(&padded, timesteps)?;
    let waveform = dac.decode(&lifted, 1024, timesteps)?;
    eprintln!(
        "DAC decoded {} samples ({} Hz target)",
        waveform.len(),
        WAVEFORM_SAMPLE_RATE
    );

    if let Some(path) = out_path {
        write_wav_f32(path, &waveform, WAVEFORM_SAMPLE_RATE)
            .map_err(|error| format!("WAV write failed: {error}"))?;
        eprintln!(
            "wrote {} samples to {} in {:.3}s total",
            waveform.len(),
            path.display(),
            started.elapsed().as_secs_f64(),
        );
    } else {
        eprintln!(
            "(no --out specified; would have written {} samples)",
            waveform.len()
        );
    }
    Ok(())
}

fn tile_channels(continuous: &[f32], timesteps: usize, target_channels: usize) -> Vec<f32> {
    let src_dim = crate::models::tts::codec::RVQ_CODE_DIM;
    let mut out = vec![0.0f32; target_channels * timesteps];
    for t in 0..timesteps {
        for c in 0..target_channels {
            out[c * timesteps + t] = continuous[t * src_dim + (c % src_dim)];
        }
    }
    out
}