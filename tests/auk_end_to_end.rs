//! End-to-end smoke test for the AuK pipeline.
//!
//! Loads the AuK-Base DiT GGUF, the Qwen2.5-Omni-3B text encoder GGUF, and
//! the BigVGANFlow VAE GGUF; runs `AukPipeline::generate_audio` end-to-end;
//! and validates that the produced WAV samples are finite and non-silent.
//!
//! Skipped when `RMI_AUK_GGUF`, `RMI_AUK_VAE`, or `RMI_AUK_TEXT` are unset.
//!
//! Run with:
//!   RMI_AUK_GGUF=models/auk-gguf/auk-base-f16.gguf \
//!   RMI_AUK_VAE=models/auk-gguf/auk-vae-f32.gguf \
//!   RMI_AUK_TEXT=models/Qwen2.5-Omni-3B-GGUF/Qwen2.5-Omni-3B-Q8_0.gguf \
//!   cargo test --test auk_end_to_end -- --nocapture --test-threads=1

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};

fn env_path(name: &str) -> Option<PathBuf> {
    let raw = std::env::var(name).ok()?;
    if raw.is_empty() {
        return None;
    }
    let path = PathBuf::from(raw);
    if !path.is_file() {
        return None;
    }
    Some(path)
}

fn open_source(path: &Path) -> Option<Arc<dyn TensorSource>> {
    Some(Arc::from(open_model_source(path, ComponentRole::Llm).ok()?))
}

#[test]
#[ignore = "end-to-end takes ~5min; run with --ignored"]
fn pipeline_runs_end_to_end_producing_a_finite_mono_audio_buffer() {
    let Some(dit) = env_path("RMI_AUK_GGUF") else {
        eprintln!("skipping: set RMI_AUK_GGUF to a real AuK DiT GGUF");
        return;
    };
    let Some(vae) = env_path("RMI_AUK_VAE") else {
        eprintln!("skipping: set RMI_AUK_VAE to a real AuK VAE GGUF");
        return;
    };
    let Some(text) = env_path("RMI_AUK_TEXT") else {
        eprintln!("skipping: set RMI_AUK_TEXT to a real Qwen2.5-Omni GGUF");
        return;
    };
    let Some(dit_src) = open_source(&dit) else {
        eprintln!("skipping: could not open AuK DiT GGUF");
        return;
    };
    let Some(vae_src) = open_source(&vae) else {
        eprintln!("skipping: could not open AuK VAE GGUF");
        return;
    };
    let Some(text_src) = open_source(&text) else {
        eprintln!("skipping: could not open Qwen2.5-Omni GGUF");
        return;
    };
    // Confirm the GGUF has the AuK arch tag so we don't silently run on
    // a wrong file.
    let arch = dit_src
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or("");
    if arch != "audiocpp" {
        eprintln!(
            "skipping: AuK DiT GGUF general.architecture = {} (expected audiocpp)",
            arch
        );
        return;
    }
    let pipeline = rust_model_inference::models::diffusion::auk::AukPipeline::load(
        dit_src,
        vae_src,
        Some(text_src),
        1,
    )
    .expect("AuK pipeline load");
    // Enable the Vulkan F16 GPU matmul path so the dispatch tests the
    // F16 GPU branch (see auk_f16_gpu_runtime in mod.rs). Without this
    // the test would fall through to F16 CPU and take ~5 min.
    rust_model_inference::ops::enable_gpu();
    // 1 second of audio at 24 kHz, 2 denoise steps. The smoke test
    // intentionally keeps steps small -- this exercises every code path
    // end-to-end (text encode -> DiT forward -> CFG -> VAE decode -> WAV)
    // without paying for full 32-step convergence.
    let options = rust_model_inference::models::diffusion::auk::AukOptions {
        steps: 2,
        sample_rate: 24_000,
        duration_sec: 1,
        seed: 42,
        guidance_scale: 2.0,
        instruct: None,
    };
    let audio = pipeline
        .generate_audio("Hi", &options)
        .expect("AuK pipeline generate");
    // Output should be non-empty, mono, ~480 * duration_sec samples.
    assert_eq!(audio.channels, 1);
    assert!(audio.samples.len() >= 1, "empty audio buffer");
    let finite_count = audio.samples.iter().filter(|v| v.is_finite()).count();
    assert_eq!(
        finite_count,
        audio.samples.len(),
        "AuK pipeline produced non-finite samples"
    );
    let max_abs = audio
        .samples
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(max_abs > 0.0, "AuK pipeline produced silent audio");
    eprintln!(
        "AuK end-to-end OK: {} samples @ {} Hz, max_abs={:.3}",
        audio.samples.len(),
        audio.sample_rate,
        max_abs
    );
}
