//! Fun-ASR-Nano ASR CLI pipeline: `--audio` with a `funasr-sensevoice-encoder` mmproj.

use std::sync::Arc;
use std::time::Instant;

use crate::app::cli::{resolve_thread_count, CliOptions};
use crate::app::open_or_exit;
use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::format::ggufrs::ComponentRole;
use crate::models::funasr::encoder::FunAsrEncoder;
use crate::models::funasr::fbank;
use crate::models::funasr::model::{format_srt_entry, transcribe_segment};
use crate::models::funasr::vad::FsmnVad;
use crate::models::qwen3::asr::audio_processor::decode_pcm16_wav_any;
use crate::models::qwen3::Qwen3Model;

const SAMPLE_RATE: usize = 16_000;

/// Run the full Fun-ASR-Nano pipeline: WAV → fbank → encoder → LLM → text.
pub fn run_funasr_cli(options: &CliOptions, prefill_batch_size: usize) -> Result<(), String> {
    let started = Instant::now();

    let enc_path = options
        .mmproj
        .as_deref()
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or("Fun-ASR-Nano requires --mmproj (encoder GGUF)")?;

    // Load encoder
    let enc_source: Arc<dyn TensorSource> =
        Arc::from(open_or_exit(enc_path, ComponentRole::Mmproj));
    if !crate::models::funasr::is_funasr_encoder(enc_source.as_ref()) {
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
    let pool = Arc::new(ComputePool::new(resolve_thread_count(
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
        Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
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
    let decoder = Arc::new(Qwen3Model::from_source(
        llm_source,
        tokenizer,
        Arc::clone(&pool),
    )?);
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
    let audio_path = options.audio.as_ref().expect("validated audio option");
    let wav_bytes = std::fs::read(audio_path)
        .map_err(|e| format!("Failed to read {}: {e}", audio_path.display()))?;
    let decoded =
        decode_pcm16_wav_any(&wav_bytes).map_err(|e| format!("WAV decode error: {e:?}"))?;
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
        crate::models::funasr::model::linear_resample(
            &samples,
            decoded.sample_rate as usize,
            SAMPLE_RATE,
        )
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
            Arc::from(open_or_exit(vad_path, ComponentRole::Mmproj));
        let vad = FsmnVad::new(Arc::clone(&vad_source))?;
        let max_seg_ms = if options.vad_maxseg > 0 {
            options.vad_maxseg
        } else {
            30000
        };
        let segs = vad.segments(&samples, max_seg_ms);
        eprintln!("[vad] {} segments", segs.len());
        segs.into_iter()
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
                println!(
                    "{}",
                    format_srt_entry(srt_idx, seg_start_ms, seg_end_ms, &text)
                );
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
