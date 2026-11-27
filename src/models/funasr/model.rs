//! Fun-ASR-Nano inference logic: fbank → encoder → LLM → text.
//!
//! CLI orchestration lives in `app/funasr.rs`. This module exposes
//! `transcribe_segment` and helper functions used by the CLI layer.

use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::tokenizer::EncodeOptions;
use crate::models::funasr::encoder::FunAsrEncoder;
use crate::models::funasr::fbank;
use crate::models::qwen3::{Qwen3GenerateOptions, Qwen3Input, Qwen3Model};
use std::sync::Arc;
use std::time::Instant;

const PROMPT_PREFIX: &str =
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：";
const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
const MIN_FBANK_FRAMES: usize = 1;

/// Check whether a GGUF source is a Fun-ASR-Nano encoder.
pub fn is_funasr_encoder(source: &dyn TensorSource) -> bool {
    source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .is_some_and(|arch| arch == crate::models::funasr::ENCODER_ARCH)
}

/// Transcribe a single audio segment: fbank → encoder → LLM → text.
pub fn transcribe_segment(
    encoder: &FunAsrEncoder,
    decoder: &Arc<Qwen3Model>,
    samples: &[f32],
    n_embd: usize,
    max_tokens: usize,
    prefill_batch_size: usize,
    repetition_penalty: f32,
) -> Result<String, String> {
    let t0 = Instant::now();
    let (fbank_data, t_fbank) = fbank::compute_fbank(samples);
    let t1 = Instant::now();
    eprintln!(
        "  Fbank: {} frames, {:.3}s",
        t_fbank,
        (t1 - t0).as_secs_f64()
    );
    if t_fbank < MIN_FBANK_FRAMES {
        return Err("Audio segment too short for one fbank frame".into());
    }

    let d_model = encoder.config.output_size;
    let scale = (d_model as f32).sqrt();
    let mut fbank_scaled = fbank_data;
    for v in &mut fbank_scaled {
        *v *= scale;
    }
    fbank::add_position_encoding(&mut fbank_scaled, t_fbank, encoder.config.input_size);

    let t2 = Instant::now();
    let adp_out = encoder.encode(&fbank_scaled, t_fbank)?;
    let t3 = Instant::now();
    eprintln!(
        "  Encoder: {} frames → {}-dim, {:.3}s",
        t_fbank,
        encoder.config.adp_llm_dim,
        (t3 - t2).as_secs_f64()
    );

    let n_aud = fbank::lfr_token_count(t_fbank);
    eprintln!("  LFR truncation: {} audio tokens", n_aud);
    let audio_embeds = &adp_out[..n_aud * n_embd];

    let tokenizer = decoder.tokenizer();
    let pre_tokens = tokenizer.encode(
        PROMPT_PREFIX,
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    let suf_tokens = tokenizer.encode(
        PROMPT_SUFFIX,
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    eprintln!(
        "  Prompt: {} prefix + {} audio + {} suffix = {} total",
        pre_tokens.len(),
        n_aud,
        suf_tokens.len(),
        pre_tokens.len() + n_aud + suf_tokens.len()
    );

    let total_tokens = pre_tokens.len() + n_aud + suf_tokens.len();
    let mut token_ids = Vec::with_capacity(total_tokens);
    token_ids.extend_from_slice(&pre_tokens);
    token_ids.extend(std::iter::repeat_n(0u32, n_aud));
    token_ids.extend_from_slice(&suf_tokens);

    let pre_embeds = decoder.embed_tokens(&pre_tokens)?;
    let suf_embeds = decoder.embed_tokens(&suf_tokens)?;
    let mut embeddings = Vec::with_capacity(total_tokens * n_embd);
    embeddings.extend_from_slice(&pre_embeds);
    embeddings.extend_from_slice(audio_embeds);
    embeddings.extend_from_slice(&suf_embeds);

    let positions: Vec<[usize; 4]> = (0..total_tokens).map(|i| [i, i, i, i]).collect();

    let t4 = Instant::now();
    let generation = decoder.generate_asr(
        Qwen3Input {
            token_ids: &token_ids,
            positions: &positions,
            embeddings: Some(&embeddings),
            deepstack_embeddings: None,
        },
        Qwen3GenerateOptions {
            max_new_tokens: max_tokens,
            temperature: 0.0,
            prefill_batch_size,
        },
        repetition_penalty,
    )?;
    let t5 = Instant::now();
    eprintln!(
        "  Generate: {} tokens, {:.3}s",
        generation.token_ids.len(),
        (t5 - t4).as_secs_f64()
    );

    Ok(generation.text)
}

/// Format an SRT entry: index, timestamp range, text.
pub fn format_srt_entry(idx: usize, start_ms: usize, end_ms: usize, text: &str) -> String {
    format!(
        "{idx}\n{} --> {}\n{text}\n",
        format_srt_timestamp(start_ms),
        format_srt_timestamp(end_ms),
    )
}

fn format_srt_timestamp(ms: usize) -> String {
    let total_s = ms / 1000;
    let hours = total_s / 3600;
    let minutes = (total_s % 3600) / 60;
    let seconds = total_s % 60;
    let millis = ms % 1000;
    format!("{hours:02}:{minutes:02}:{seconds:02},{millis:03}")
}

/// Simple linear interpolation resampler.
pub fn linear_resample(input: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let out_len = ((input.len() as f64) * ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;
        if idx + 1 < input.len() {
            out.push(input[idx] * (1.0 - frac as f32) + input[idx + 1] * frac as f32);
        } else {
            out.push(input[input.len() - 1]);
        }
    }
    out
}
