//! Whistle CLI driver — two backends:
//!
//! 1. **Pure Rust port** (default). Runs the encoder + decoder logits
//!    that we have lifted to `models::whistle` (ZCRMSNorm + HadamardMLP
//!    + multi-lane MHC at 1 ULP parity with the in-tree Python Oracle).
//!    The conv stem is still a placeholder, so output is not a real
//!    transcript yet.
//!
//! 2. **Cactus needle runtime** (set `--needle-runtime`). Calls the
//!    closed `libneedle3.so` via the FFI bindings in `crate::needle_ffi`
//!    to produce real transcripts through the .cact binary. The Rust
//!    CLI orchestrates WAV decode and language selection; the runtime
//!    runs the actual audio encoder + conv stem in C++.
//!
//! Both paths are exercised through `--audio whistle` so the user can
//! compare and decide which to ship.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use crate::app::open_or_exit;
use crate::core::tensor::TensorSource;
use crate::format::ggufrs::ComponentRole;
use crate::models::whistle::{
    final_rms_norm, load_encoder_layer, load_mhc_layer, EncoderLayerWeights, MhcLayer,
    MHC_LANES,
};
use crate::needle_ffi;

use crate::app::cli::CliOptions;

const WHISTLE_HIDDEN: usize = 512;

/// Where the .cact binary lives. Defaults to `models/whistle-probe/whistle.cact`.
fn cact_path() -> PathBuf {
    if let Ok(p) = std::env::var("WHISTLE_CACT_PATH") {
        return PathBuf::from(p);
    }
    let manifest = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string()),
    );
    manifest.join("models").join("whistle-probe").join("whistle.cact")
}

/// Read a 16-bit / 16kHz mono PCM WAV into mono `f32` in `[-1, 1]`.
fn read_wav_mono_16k(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{path:?}: not a RIFF/WAVE file"));
    }
    // Walk chunks to find `fmt ` and `data`.
    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut audio_format = 0u16;
    let mut data_off = 0usize;
    let mut data_len = 0usize;
    while pos + 8 <= bytes.len() {
        let tag = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;
        if tag == b"fmt " {
            audio_format = u16::from_le_bytes([bytes[pos + 8], bytes[pos + 9]]);
            channels = u16::from_le_bytes([bytes[pos + 10], bytes[pos + 11]]);
            sample_rate = u32::from_le_bytes([
                bytes[pos + 12],
                bytes[pos + 13],
                bytes[pos + 14],
                bytes[pos + 15],
            ]);
            bits_per_sample = u16::from_le_bytes([bytes[pos + 22], bytes[pos + 23]]);
        } else if tag == b"data" {
            data_off = pos + 8;
            data_len = size;
            break;
        }
        pos += 8 + size + (size & 1); // word-align
    }
    if data_off == 0 {
        return Err(format!("{path:?}: no `data` chunk"));
    }
    if sample_rate != 16000 {
        return Err(format!(
            "{path:?}: 16 kHz mono required, got {sample_rate} Hz"
        ));
    }
    if bits_per_sample != 16 {
        return Err(format!(
            "{path:?}: 16-bit PCM required, got {bits_per_sample} bits"
        ));
    }
    if audio_format != 1 {
        return Err(format!("{path:?}: PCM required (format={audio_format})"));
    }
    if channels == 0 || channels > 8 {
        return Err(format!("{path:?}: unsupported channel count {channels}"));
    }
    let samples = data_len / (channels as usize * 2);
    let mut out = Vec::with_capacity(samples);
    for s in 0..samples {
        let mut acc = 0i32;
        for c in 0..channels as usize {
            let p = data_off + (s * channels as usize + c) * 2;
            let raw = i16::from_le_bytes([bytes[p], bytes[p + 1]]);
            acc += raw as i32;
        }
        out.push(acc as f32 / (channels as f32 * 32768.0));
    }
    Ok(out)
}

pub fn run_whistle_cli(options: &CliOptions) -> Result<(), String> {
    // The flag wiring lives in main.rs; the dispatcher in src/app/asr/mod.rs
    // hands us a `CliOptions` already decided. We re-check the
    // architecture just to give a clear error if the user mis-routed.
    let model_path = &options.model;
    let source: Arc<dyn TensorSource> =
        Arc::from(open_or_exit(model_path, ComponentRole::Llm));
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    if arch != "whistle" {
        return Err(format!(
            "--audio whistle path: expected general.architecture == \"whistle\", got {arch:?}"
        ));
    }

    // Optional: route to the closed Cactus runtime instead of the
    // placeholder-stem Rust port. Enabled by setting
    // `WHISTLE_NEEDLE_RUNTIME=1` (the CLI doesn't have a dedicated flag
    // because the runtime path requires a real .cact + libneedle3.so
    // pair to be present in the sandbox; the placeholder path is the
    // sensible default). The runtime produces real transcripts; the
    // Rust port currently emits a placeholder token.
    if std::env::var("WHISTLE_NEEDLE_RUNTIME").as_deref() == Ok("1") {
        return run_whistle_needle_runtime(options);
    }

    // Pure-Rust port (placeholder stem).
    eprintln!(
        "whistle (pure Rust): loading metadata from {} (engine={:?})",
        model_path.display(),
        source.metadata("whistle.engine").and_then(|v| v.to_string_val())
    );

    let enc_layers: Vec<_> = (0..8).map(|i| load_encoder_layer(&*source, i)).collect();
    let mhc = load_mhc_layer(&*source);
    let enc_final_scale = crate::core::tensor::load_f32_tensor(
        &*source,
        "encoder/final_norm/scale",
        &[WHISTLE_HIDDEN as u64],
    )
    .map_err(|e| format!("missing encoder/final_norm/scale: {e}"))?;
    let stack_final_scale = crate::core::tensor::load_f32_tensor(
        &*source,
        "stack/final_norm/scale",
        &[WHISTLE_HIDDEN as u64],
    )
    .map_err(|e| format!("missing stack/final_norm/scale: {e}"))?;
    let embedding = crate::core::tensor::load_f32_tensor(
        &*source,
        "embedding/embedding",
        &[8199, WHISTLE_HIDDEN as u64],
    )
    .or_else(|_| {
        crate::core::tensor::load_f32_tensor(
            &*source,
            "token_embd.weight",
            &[8199, WHISTLE_HIDDEN as u64],
        )
    })
    .map_err(|e| format!("missing embedding/embedding: {e}"))?;

    eprintln!(
        "whistle: loaded 8 encoder layers + 1 stack layer (stem=placeholder); use --needle-runtime for real transcripts"
    );

    let hidden = vec![0.0f32; 50 * WHISTLE_HIDDEN];
    let seq_len = hidden.len() / WHISTLE_HIDDEN;
    eprintln!(
        "whistle: stem placeholder produced seq_len={seq_len} frames; \
         this path is a smoke test for the encoder+decoder wiring"
    );

    // Cumulative multi-lane stream
    let plane_4 = seq_len * MHC_LANES * WHISTLE_HIDDEN;
    let mut stream: Vec<f32> = hidden
        .chunks(WHISTLE_HIDDEN)
        .flat_map(|chunk| (0..MHC_LANES).map(move |_| chunk.to_vec()).flatten())
        .collect();
    if stream.len() != plane_4 {
        return Err(format!(
            "stream init mismatch: got {} expected {plane_4}",
            stream.len()
        ));
    }
    for i in 0..8 {
        let layer = &enc_layers[i];
        let layer_w = EncoderLayerWeights {
            zc_in_scale: &layer.zc_in_scale,
            post_attn_scale: &layer.post_attn_scale,
            q_proj: &layer.q_proj,
            k_proj: &layer.k_proj,
            v_proj: &layer.v_proj,
            out_proj: &layer.out_proj,
            q_norm_scale: &layer.q_norm_scale,
            k_norm_scale: &layer.k_norm_scale,
            attn_gate: &layer.attn_gate,
            hadamard: layer.hadamard.clone(),
        };
        let mut next_stream = vec![0.0f32; plane_4];
        crate::models::whistle::mhc::forward_one_stream(
            &layer_w, &mhc, &stream, i, 1, seq_len, &mut next_stream,
        );
        stream = next_stream;
    }
    let mut enc = vec![0.0f32; seq_len * WHISTLE_HIDDEN];
    for bt in 0..seq_len {
        for h in 0..WHISTLE_HIDDEN {
            let mut sum = 0.0f32;
            for lane in 0..MHC_LANES {
                sum += stream[bt * MHC_LANES * WHISTLE_HIDDEN + lane * WHISTLE_HIDDEN + h];
            }
            enc[bt * WHISTLE_HIDDEN + h] = sum / MHC_LANES as f32;
        }
    }

    let mut enc_norm = vec![0.0f32; enc.len()];
    final_rms_norm(&enc, &enc_final_scale, 1, seq_len, &mut enc_norm);

    // Pick the BOS token (id=1, used as prompt for the decoder).
    let mut text_emb = vec![0.0f32; WHISTLE_HIDDEN];
    text_emb.copy_from_slice(&embedding[1 * WHISTLE_HIDDEN..(1 + 1) * WHISTLE_HIDDEN]);
    let stack = crate::models::whistle::load_stack_layer(&*source, 0);
    let mut stack_out = vec![0.0f32; WHISTLE_HIDDEN];
    stack.forward(&text_emb, &enc_norm, 0, 1, 1, seq_len, &mut stack_out);

    let mut logits_emb = vec![0.0f32; WHISTLE_HIDDEN];
    final_rms_norm(&stack_out, &stack_final_scale, 1, 1, &mut logits_emb);

    let mut logits = vec![0.0f32; 8199];
    for v in 0..8199 {
        let mut acc = 0.0f32;
        for h in 0..WHISTLE_HIDDEN {
            acc += logits_emb[h] * embedding[v * WHISTLE_HIDDEN + h];
        }
        logits[v] = acc;
    }
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    eprintln!(
        "whistle (pure Rust): placeholder ASR produced token id={argmax} (stem is not a real mel pipeline)"
    );

    Ok(())
}

/// Dispatch through `libneedle3.so`. Loads the .cact binary and runs
/// `needle_transcribe` on the decoded 16 kHz mono PCM.
fn run_whistle_needle_runtime(options: &CliOptions) -> Result<(), String> {
    let wav_path = options
        .audio
        .as_ref()
        .ok_or_else(|| "WHISTLE_NEEDLE_RUNTIME=1 needs --audio <wav>".to_string())?;
    let wav = std::path::Path::new(wav_path);
    let samples = read_wav_mono_16k(wav)?;
    eprintln!(
        "whistle (needle runtime): loaded {} samples ({} s) from {}",
        samples.len(),
        samples.len() as f32 / 16000.0,
        wav.display()
    );

    let cact = cact_path();
    needle_ffi::load(&cact)?;

    let language = std::env::var("WHISTLE_LANG").ok().or_else(|| Some("en".into()));
    let result = needle_ffi::transcribe(&samples, language.as_deref(), None, false)?;
    eprintln!(
        "whistle (needle runtime): ttft={:.1} ms decode_tps={:.0} lang={:?}",
        result.ttft_ms,
        result.decode_tps,
        result.language
    );
    println!("{}", result.text);
    Ok(())
}