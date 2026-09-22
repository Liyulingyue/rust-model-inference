//! Fun-ASR-Nano model: integrates SAN-M encoder with Qwen3 LLM trunk.
//!
//! The encoder GGUF (`funasr-encoder-f16.gguf`) has architecture
//! `funasr-sensevoice-encoder`. The LLM GGUF is a standard Qwen3-0.6B
//! (architecture `qwen3`). Audio embeddings are injected into the Qwen3
//! trunk via `Qwen3Input.embeddings`.

use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::core::thread_pool::ComputePool;
use crate::format::ggufrs::ComponentRole;
use crate::models::funasr::encoder::FunAsrEncoder;
use crate::models::funasr::fbank;
use crate::models::funasr::vad::{FsmnVad, VadSegment};
use crate::models::qwen3::{Qwen3GenerateOptions, Qwen3Input, Qwen3Model};
use crate::models::qwen3::asr::audio_processor::decode_pcm16_wav_any;
use std::sync::Arc;
use std::time::Instant;

const SAMPLE_RATE: usize = 16_000;
const PROMPT_PREFIX: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：";
const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
const MIN_FBANK_FRAMES: usize = 1;

/// Check whether a GGUF source is a Fun-ASR-Nano encoder.
pub fn is_funasr_encoder(source: &dyn TensorSource) -> bool {
    source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .is_some_and(|arch| arch == crate::models::funasr::ENCODER_ARCH)
}

/// Run the full Fun-ASR-Nano pipeline: WAV → fbank → encoder → LLM → text.
pub fn run_funasr_cli(
    options: &crate::app::cli::CliOptions,
    prefill_batch_size: usize,
) -> Result<(), String> {
    let started = Instant::now();

    let enc_path = options
        .mmproj
        .as_deref()
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or("Fun-ASR-Nano requires --mmproj (encoder GGUF)")?;

    // Load encoder
    let enc_source: Arc<dyn TensorSource> =
        Arc::from(crate::app::open_or_exit(enc_path, ComponentRole::Mmproj));
    if !is_funasr_encoder(enc_source.as_ref()) {
        return Err(format!(
            "Expected encoder architecture {:?}, got {:?}",
            crate::models::funasr::ENCODER_ARCH,
            enc_source
                .metadata("general.architecture")
                .and_then(MetaValue::to_string_val)
                .unwrap_or_default()
        ));
    }
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let pool = Arc::new(ComputePool::new(crate::app::cli::resolve_thread_count(
        options.threads,
        available,
    )));
    let encoder = FunAsrEncoder::new(Arc::clone(&enc_source), Arc::clone(&pool))?;
    eprintln!(
        "Fun-ASR-Nano encoder: {}+{} layers, d_model={}, adp_llm_dim={}",
        encoder.config.num_blocks,
        encoder.config.tp_blocks,
        encoder.config.output_size,
        encoder.config.adp_llm_dim,
    );

    // Load LLM (Qwen3-0.6B)
    let llm_source: Arc<dyn TensorSource> =
        Arc::from(crate::app::open_or_exit(&options.model, ComponentRole::Llm));
    let arch = llm_source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .unwrap_or_default();
    if arch != "qwen3" {
        return Err(format!(
            "Fun-ASR-Nano LLM must be architecture \"qwen3\", got {arch:?}"
        ));
    }
    let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
        llm_source.metadata(key).cloned()
    })?);
    let decoder = Arc::new(Qwen3Model::from_source(llm_source, tokenizer, Arc::clone(&pool))?);
    let n_embd = decoder.config().n_embd;
    if n_embd != encoder.config.adp_llm_dim as usize {
        return Err(format!(
            "LLM embedding dim {n_embd} != encoder adp_llm_dim {}",
            encoder.config.adp_llm_dim
        ));
    }
    let load_done = started.elapsed();
    eprintln!("Models loaded in {:.3}s", load_done.as_secs_f64());

    // Load audio
    let audio_path = options
        .audio
        .as_ref()
        .expect("validated audio option");
    let wav_bytes = std::fs::read(audio_path)
        .map_err(|e| format!("Failed to read {}: {e}", audio_path.display()))?;
    let decoded = decode_pcm16_wav_any(&wav_bytes)
        .map_err(|e| format!("WAV decode error: {e:?}"))?;
    let samples: Vec<f32> = if decoded.channels == 1 {
        decoded.samples
    } else {
        decoded
            .samples
            .chunks(decoded.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    };
    let samples = if decoded.sample_rate as usize == SAMPLE_RATE {
        samples
    } else {
        eprintln!(
            "Warning: audio sample rate {} != {SAMPLE_RATE}, resampling (basic linear)",
            decoded.sample_rate
        );
        linear_resample(&samples, decoded.sample_rate as usize, SAMPLE_RATE)
    };
    eprintln!(
        "Audio: {} samples ({:.1}s)",
        samples.len(),
        samples.len() as f64 / SAMPLE_RATE as f64
    );

    // Build chunk windows: --vad for FSMN-VAD segments, --chunk for fixed windows,
    // otherwise whole file.
    let wins: Vec<(usize, usize)> = if let Some(vad_path) = options.vad.as_deref() {
        let vad_source: Arc<dyn TensorSource> =
            Arc::from(crate::app::open_or_exit(vad_path, ComponentRole::Mmproj));
        let vad = FsmnVad::new(Arc::clone(&vad_source))?;
        let max_seg_ms = if options.vad_maxseg > 0 { options.vad_maxseg } else { 30000 };
        let segs = vad.segments(&samples, max_seg_ms);
        eprintln!("[vad] {} segments", segs.len());
        segs
            .into_iter()
            .map(|seg| {
                let off = seg.start_ms * SAMPLE_RATE / 1000;
                let end = seg.end_ms * SAMPLE_RATE / 1000;
                let end = end.min(samples.len());
                (off, end.saturating_sub(off))
            })
            .filter(|(_, len)| *len >= 400)
            .collect()
    } else {
        let chunk_samples = options
            .chunk_seconds
            .map(|sec| (sec * SAMPLE_RATE as f64).round() as usize)
            .unwrap_or(samples.len());
        (0..samples.len())
            .step_by(chunk_samples)
            .map(|off| {
                let end = (off + chunk_samples).min(samples.len());
                (off, end - off)
            })
            .filter(|(_, len)| *len >= 400)
            .collect()
    };
    eprintln!("Segments: {}", wins.len());

    let max_tokens = options.max_tokens.unwrap_or(512);
    let rep_penalty = options.effective_repetition_penalty();
    let srt_mode = options.srt;
    let mut srt_idx = 0usize;
    let mut full_text = String::new();

    for (win_idx, &(off, len)) in wins.iter().enumerate() {
        let seg = &samples[off..off + len];
        let seg_start_ms = (off * 1000) / SAMPLE_RATE;
        let seg_end_ms = ((off + len) * 1000) / SAMPLE_RATE;
        if wins.len() > 1 {
            eprintln!(
                "Chunk {}/{}: {:.1}s–{:.1}s",
                win_idx + 1,
                wins.len(),
                seg_start_ms as f64 / 1000.0,
                seg_end_ms as f64 / 1000.0
            );
        }

        let text = transcribe_segment(
            &encoder,
            &decoder,
            seg,
            n_embd,
            max_tokens,
            prefill_batch_size,
            rep_penalty,
        )?;

        if srt_mode {
            if !text.is_empty() && text != "/sil" {
                srt_idx += 1;
                println!("{}", format_srt_entry(srt_idx, seg_start_ms, seg_end_ms, &text));
            }
        } else {
            full_text.push_str(&text);
        }
    }

    let total = started.elapsed();
    eprintln!(
        "Total: {:.3}s (load={:.3}s transcribe={:.3}s)",
        total.as_secs_f64(),
        load_done.as_secs_f64(),
        (total - load_done).as_secs_f64(),
    );

    if !srt_mode {
        println!("{full_text}");
    }
    Ok(())
}

/// Transcribe a single audio segment: fbank → encoder → LLM → text.
fn transcribe_segment(
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

    // Pre-scale by sqrt(d_model) and add position encoding
    let d_model = encoder.config.output_size;
    let scale = (d_model as f32).sqrt();
    let mut fbank_scaled = fbank_data;
    for v in &mut fbank_scaled {
        *v *= scale;
    }
    fbank::add_position_encoding(&mut fbank_scaled, t_fbank, encoder.config.input_size);

    // Run encoder + adaptor
    let t2 = Instant::now();
    let adp_out = encoder.encode(&fbank_scaled, t_fbank)?;
    let t3 = Instant::now();
    eprintln!(
        "  Encoder: {} frames → {}-dim, {:.3}s",
        t_fbank,
        encoder.config.adp_llm_dim,
        (t3 - t2).as_secs_f64()
    );

    // LFR truncation
    let n_aud = fbank::lfr_token_count(t_fbank);
    eprintln!("  LFR truncation: {} audio tokens", n_aud);
    let audio_embeds = &adp_out[..n_aud * n_embd];

    // Tokenize prompt
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

    // Build combined token_ids (dummy 0 for audio) and embeddings
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
fn format_srt_entry(idx: usize, start_ms: usize, end_ms: usize, text: &str) -> String {
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
fn linear_resample(input: &[f32], from: usize, to: usize) -> Vec<f32> {
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
