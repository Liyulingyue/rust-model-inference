use crate::app::cli::resolve_thread_count;
use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::models::tts::codec::{write_wav_f32, DacDecoder, RvqDecoder, WAVEFORM_SAMPLE_RATE};
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
    let rvq = RvqDecoder::from_source(mmproj_source.as_ref())?;
    let dac = DacDecoder::from_source(mmproj_source.as_ref())?;
    eprintln!(
        "RVQ: {} first + {} rest codebooks, {} dim each",
        rvq.first_codebook_size(),
        crate::models::tts::codec::RVQ_LEVELS - 1,
        crate::models::tts::codec::RVQ_CODE_DIM,
    );

    // Stage 1 only emits one token per frame; we currently lack a code predictor
    // and waveform transformer, so map each Talker audio token to a single
    // first-codebook index by taking `token_id % RVQ_CODEBOOK_SIZE` and
    // padding the remaining 15 residual levels with zeros.
    let codes: Vec<u32> = expand_talker_codes_to_rvq(&generation.audio_token_ids);
    let continuous = rvq.decode(&codes)?;
    let timesteps = continuous.len() / crate::models::tts::codec::RVQ_CODE_DIM;
    eprintln!("RVQ decoded {} timesteps × {} dim", timesteps, crate::models::tts::codec::RVQ_CODE_DIM);

    // Stage 2 placeholder: feed the RVQ-decoded [256, timesteps] directly into
    // the DAC entry conv. The waveform TFM is not yet wired in — for now we
    // tile the 256-dim code vectors to 1024 channels (4× repeat) so the DAC
    // entry conv has something to consume.
    let tiled = tile_to_dac_input(&continuous, timesteps, 1024);

    let waveform = dac.decode(&tiled, 1024, timesteps)?;
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

fn expand_talker_codes_to_rvq(talker_codes: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(talker_codes.len() * crate::models::tts::codec::RVQ_LEVELS);
    for &token in talker_codes {
        // First-level code: take the audio-codebook id modulo codebook vocab.
        let first = (token % crate::models::tts::codec::RVQ_CODEBOOK_SIZE as u32);
        out.push(first);
        // Remaining 15 residual levels default to zero (placeholder until the
        // code predictor is implemented).
        for _ in 1..crate::models::tts::codec::RVQ_LEVELS {
            out.push(0);
        }
    }
    out
}

fn tile_to_dac_input(continuous: &[f32], timesteps: usize, target_channels: usize) -> Vec<f32> {
    let src_dim = crate::models::tts::codec::RVQ_CODE_DIM;
    let mut out = vec![0.0f32; target_channels * timesteps];
    for t in 0..timesteps {
        for c in 0..target_channels {
            out[c * timesteps + t] = continuous[t * src_dim + (c % src_dim)];
        }
    }
    out
}