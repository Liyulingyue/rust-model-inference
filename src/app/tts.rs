use crate::app::cli::resolve_thread_count;
use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::models::qwen3::{qwen_text_positions, validate_token_ids};
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

    let mmproj_path = options
        .mmproj
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty());
    let out_path = options.out.as_deref();
    let only_codes = mmproj_path.is_none() && out_path.is_none();

    if !only_codes {
        let mp = mmproj_path
            .ok_or_else(|| "--out requires --mmproj for codec decoding".to_string())?;
        eprintln!("Loading TTS codec decoder from {}", mp.display());
        let mmproj_source: Arc<dyn TensorSource> = Arc::from(
            open_or_exit(mp, ComponentRole::Mmproj),
        );
        let predictor = CodePredictor::from_source(mmproj_source.as_ref())?;
        let rvq = RvqDecoder::from_source(mmproj_source.as_ref())?;
        let tfm = WaveformTransformer::from_source(mmproj_source.as_ref())?;
        let dac = DacDecoder::from_source(mmproj_source.as_ref())?;
        eprintln!(
            "RVQ: {} first + {} rest codebooks, {} dim each",
            rvq.first_codebook_size(),
            crate::models::tts::codec::RVQ_LEVELS - 1,
            crate::models::tts::codec::RVQ_CODE_DIM
        );

        let (audio_ids, rvq_codes) = run_per_frame_pipeline(
            &talker,
            &predictor,
            prompt,
            max_tokens,
            temperature,
        )?;
        // TEST: use zeros instead of real codes
        // let mut rvq_codes = vec![0u32; audio_ids.len() * 16];
        // for i in 0..rvq_codes.len() { rvq_codes[i] = (i % 2048) as u32; }
        eprintln!(
            "TTS: {} prompt chars, {} audio frames, {} RVQ codes in {:.3}s",
            prompt.chars().count(),
            audio_ids.len(),
            rvq_codes.len(),
            started.elapsed().as_secs_f64(),
        );

        let continuous = rvq.decode(&rvq_codes)?;
        let timesteps = continuous.len() / 512;
        eprintln!("RVQ decoded {} timesteps × 512 dim", timesteps);
        let preconv = dac.pre_conv(&continuous, timesteps)?;
        let lifted = tfm.forward(&preconv, timesteps)?;
        let waveform = dac.decode(&lifted, timesteps)?;

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
        return Ok(());
    }

    // No mmproj path: emit raw audio token ids only (Stage 1 mode).
    let generation = talker.synthesize(prompt, None, max_tokens, temperature)?;
    eprintln!(
        "TTS: {} prompt chars, {} audio tokens ({:?}) in {:.3}s",
        prompt.chars().count(),
        generation.audio_token_ids.len(),
        generation.finished_reason,
        started.elapsed().as_secs_f64(),
    );
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
    Ok(())
}

/// Per-frame autoregressive TTS loop: prefill the prompt through the talker,
/// then for each generated frame run the code predictor and feed `out_embd`
/// back into the talker for the next frame's input embedding.
///
/// Returns `(audio_token_ids, rvq_codes)` — the talker-sampled tokens and
/// the full RVQ sequence (16 codes per frame) for downstream decoding.
fn run_per_frame_pipeline(
    talker: &Qwen3TtsTalker,
    predictor: &CodePredictor,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
) -> Result<(Vec<u32>, Vec<u32>), String> {
    let cfg = talker.config();
    let token_ids: Vec<u32> = talker.tokenizer().encode(
        prompt,
        crate::core::tokenizer::EncodeOptions::default(),
    );
    validate_token_ids(&token_ids, cfg.vocab_size)?;

    let mut session = talker.new_session()?;
    let positions = qwen_text_positions(token_ids.len());
    for (&tok, &pos) in token_ids.iter().zip(positions.iter()) {
        session.forward_step(tok, pos)?;
    }
    let mut next_position: [usize; 4] = positions.last().copied().unwrap_or([0; 4]);

    // First sample: talker's logits over audio_codebook.
    let mut logits = session.compute_audio_logits()?;
    let temp = if temperature <= 0.0 { 0.0 } else { temperature };
    let mut sampled = session.sample_from_logits(&logits, temp)?;
    let eos = cfg.eos_token_id;
    let mut audio_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut all_codes: Vec<u32> = Vec::with_capacity(max_new_tokens * 16);
    let mut rng = rand::thread_rng();
    if sampled != eos {
        audio_ids.push(sampled);
    }

    for _step in 1..max_new_tokens {
        if sampled == eos {
            break;
        }
        let h_state = session.hidden_state().to_vec();
        let (codes, out_embd) = predictor.predict_frame(&h_state, sampled, 50, &mut rng)?;
        for &c in &codes {
            all_codes.push(c);
        }
        if _step <= 3 {
            eprintln!(
                "[dbg] frame {} h_state[:8]={:?} sampled={} codes={:?}",
                _step,
                &h_state[..8.min(h_state.len())],
                sampled,
                &codes
            );
        }
        // Advance talker using out_embd as the next frame's input.
        next_position[0] = next_position[0].saturating_add(1);
        session.forward_step_with_embedding(&out_embd, next_position)?;
        // Sample next talker audio token.
        logits = session.compute_audio_logits()?;
        sampled = session.sample_from_logits(&logits, temp)?;
        if sampled != eos {
            audio_ids.push(sampled);
        }
    }
    eprintln!("[dbg] audio_ids: {:?}", &audio_ids[..audio_ids.len().min(8)]);
    Ok((audio_ids, all_codes))
}
