//! Simulated-streaming 80 ms / 480 ms Audio8 ASR inference.

use crate::app::cli::CliOptions;
use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::models::audio8::mel::compute_voxtral_log_mel;
use crate::models::audio8::text::{Audio8TextDecoder, Audio8TextSession};
use crate::models::audio8::{time_condition, Audio8Encoder, Audio8State};
use crate::models::qwen3::asr::audio_processor::decode_pcm16_wav_any;
use std::collections::VecDeque;
use std::sync::Arc;

const SAMPLE_RATE: usize = 16_000;
const SAMPLES_PER_TOKEN: usize = 1280;
const LEFT_PAD_TOKENS: usize = 18;
const DELAY_TOKENS: usize = 6;
const RIGHT_TEXT_TOKENS: usize = 10;
const LOOK_BACK_SAMPLES: usize = 840;
const LOOK_AHEAD_SAMPLES: usize = 40;

fn load_samples(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let decoded =
        decode_pcm16_wav_any(&bytes).map_err(|error| format!("invalid Audio8 WAV: {error:?}"))?;
    let samples = if decoded.channels == 1 {
        decoded.samples
    } else {
        decoded
            .samples
            .chunks(decoded.channels as usize)
            .map(|channels| channels.iter().sum::<f32>() / channels.len() as f32)
            .collect()
    };
    if samples.is_empty() {
        return Err("Audio8 WAV contains no audio".into());
    }
    let samples = if decoded.sample_rate as usize == SAMPLE_RATE {
        samples
    } else {
        crate::models::funasr::model::linear_resample(
            &samples,
            decoded.sample_rate as usize,
            SAMPLE_RATE,
        )
    };
    Ok(samples)
}

fn token(tokenizer: &BPETokenizer, literal: &str) -> Result<u32, String> {
    tokenizer
        .token_id(literal)
        .ok_or_else(|| format!("Audio8 tokenizer is missing {literal}"))
}

pub fn run_audio8_cli(options: &CliOptions) -> Result<(), String> {
    if options.mmproj.is_some() {
        return Err("Audio8 carries its audio tower in --model; omit --mmproj".into());
    }
    let source: Arc<dyn TensorSource> = Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
    let encoder = Audio8Encoder::from_source(Arc::clone(&source))?;
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
        source.metadata(key).cloned()
    })?);
    let language = options.language.as_deref().unwrap_or("zh");
    let language_token = match language {
        "zh" => token(&tokenizer, "[LANGUAGE_ZH]")?,
        "en" => token(&tokenizer, "[LANGUAGE_EN]")?,
        _ => return Err("Audio8 --language must be zh or en".into()),
    };
    let bos = token(&tokenizer, "<|im_start|>")?;
    let eos = token(&tokenizer, "<|im_end|>")?;
    let stream_pad = token(&tokenizer, "[STREAMING_PAD]")?;
    let stream_word = token(&tokenizer, "[STREAMING_WORD]")?;
    let pad = token(&tokenizer, "<|endoftext|>")?;
    let audio_path = options.audio.as_ref().ok_or("Audio8 requires --audio")?;
    let samples = load_samples(audio_path)?;
    let model = Audio8TextDecoder::from_source(source)?;
    let initial_tokens = LEFT_PAD_TOKENS + DELAY_TOKENS + 1;
    let right_tokens = DELAY_TOKENS + 1 + RIGHT_TEXT_TOKENS;
    let mut stream = vec![0.0; LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN];
    stream.extend_from_slice(&samples);
    stream.resize(stream.len() + right_tokens * SAMPLES_PER_TOKEN, 0.0);
    let remaining_steps = stream
        .len()
        .saturating_sub(initial_tokens * SAMPLES_PER_TOKEN + LOOK_AHEAD_SAMPLES)
        / SAMPLES_PER_TOKEN
        + 1;
    let capacity = initial_tokens + remaining_steps;
    if capacity > model.context() {
        return Err("Audio8 input exceeds the decoder context".into());
    }
    let mut session = Audio8TextSession::new(
        &model,
        capacity,
        &time_condition(DELAY_TOKENS, &encoder.frame_embedding)?,
    )?;
    let mut audio_state = Audio8State::new();
    let mut queue = VecDeque::from([bos, language_token]);
    queue.extend(std::iter::repeat_n(stream_pad, initial_tokens - 2));
    let mut generated = Vec::new();
    let mut start = 0usize;
    let mut end = initial_tokens * SAMPLES_PER_TOKEN;
    let max_tokens = options.max_tokens.unwrap_or(512);
    while end + LOOK_AHEAD_SAMPLES <= stream.len() && generated.len() < max_tokens {
        let count = (end - start) / SAMPLES_PER_TOKEN;
        let window = &stream[start.saturating_sub(LOOK_BACK_SAMPLES)..end + LOOK_AHEAD_SAMPLES];
        let mel =
            compute_voxtral_log_mel(window).map_err(|error| format!("Audio8 mel: {error:?}"))?;
        let audio_embeddings =
            encoder.encode_window(&mel.normalized, mel.frames, count * 4, &mut audio_state)?;
        let input_ids: Vec<u32> = (0..count)
            .map(|_| queue.pop_front().unwrap_or(stream_pad))
            .collect();
        #[cfg(feature = "parity-trace")]
        {
            crate::parity_trace::report(crate::parity_trace::token_ids(
                "audio8.input_ids",
                &input_ids,
            ));
            crate::parity_trace::report(crate::parity_trace::checkpoint_at(
                "audio8.audio_embeddings",
                None,
                Some(generated.len()),
                &[count, 2048],
                &audio_embeddings,
            ));
        }
        let embeddings = model.embed_audio_tokens(&input_ids, &audio_embeddings)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint_at(
            "audio8.text_embeddings",
            None,
            Some(generated.len()),
            &[count, 2048],
            &embeddings,
        ));
        let mut logits = session.forward_logits(&embeddings)?;
        if logits.iter().any(|value| value.is_nan()) {
            return Err("Audio8 decoder produced NaN logits".into());
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint_at(
            "audio8.logits",
            None,
            Some(generated.len()),
            &[logits.len()],
            &logits,
        ));
        logits[eos as usize] = f32::NEG_INFINITY;
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(id, _)| id as u32)
            .ok_or("Audio8 decoder returned no logits")?;
        generated.push(next);
        queue.push_back(next);
        start = end;
        end += SAMPLES_PER_TOKEN;
    }
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::token_ids(
        "audio8.generated_ids",
        &generated,
    ));
    let visible: Vec<u32> = generated
        .into_iter()
        .take_while(|&id| id != eos)
        .filter(|&id| ![bos, stream_pad, stream_word, pad].contains(&id))
        .collect();
    println!("{}", tokenizer.decode(&visible, false));
    Ok(())
}
