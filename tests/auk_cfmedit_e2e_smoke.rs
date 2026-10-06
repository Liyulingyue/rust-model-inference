//! CFMEdit end-to-end smoke test.
//!
//! Loads the AuK-Base DiT, the Qwen2.5-Omni text encoder (Q8_0), the
//! BigVGANFlow VAE, AND the Qwen2.5-Omni BF16 audio tower. Generates
//! a synthetic 1-second 440Hz reference WAV in memory, runs the full
//! pipeline (WAV -> audio tower -> embeddings -> DiT -> VAE -> WAV) and
//! validates that the produced audio is finite + non-silent.
//!
//! Run with:
//!   RMI_AUK_GGUF=models/auk-gguf/auk-base-f16.gguf \
//!   RMI_AUK_VAE=models/auk-gguf/auk-vae-f32.gguf \
//!   RMI_AUK_TEXT=models/Qwen2.5-Omni-3B-GGUF/Qwen2.5-Omni-3B-Q8_0.gguf \
//!   RMI_QWEN_OMNI_BF16=models/Qwen2.5-Omni-3B-bf16-GGUF/qwen2.5-omni-3b-bf16.gguf \
//!   cargo test --features vulkan --release --test auk_cfmedit_e2e_smoke -- --ignored --nocapture

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
#[ignore = "loads 7GB BF16 Qwen; end-to-end takes ~5min; run with --ignored"]
fn cfmedit_end_to_end_synthetic_reference_audio() {
    let Some(dit) = env_path("RMI_AUK_GGUF") else {
        eprintln!("RMI_AUK_GGUF unset; skipping");
        return;
    };
    let Some(vae) = env_path("RMI_AUK_VAE") else {
        eprintln!("RMI_AUK_VAE unset; skipping");
        return;
    };
    let Some(text) = env_path("RMI_AUK_TEXT") else {
        eprintln!("RMI_AUK_TEXT unset; skipping");
        return;
    };
    let Some(omni_bf16) = env_path("RMI_QWEN_OMNI_BF16") else {
        eprintln!("RMI_QWEN_OMNI_BF16 unset; skipping");
        return;
    };
    let Some(dit_src) = open_source(&dit) else { return; };
    let Some(vae_src) = open_source(&vae) else { return; };
    let Some(text_src) = open_source(&text) else { return; };
    let Some(omni_bf16_src) = open_source(&omni_bf16) else { return; };

    let pipeline = rust_model_inference::models::diffusion::auk::AukPipeline::load_with_audio_tower(
        dit_src,
        vae_src,
        Some(text_src),
        Some(omni_bf16_src),
        1,
    )
    .expect("AukPipeline load (with audio tower)");
    rust_model_inference::ops::enable_gpu();

    // 1 second of 16 kHz mono PCM = 16000 samples of 440 Hz sine wave.
    let mut samples = Vec::with_capacity(8000);
    for i in 0..8000 {
        let t = i as f32 / 16_000.0;
        samples.push((2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5);
    }

    let options = rust_model_inference::models::diffusion::auk::AukOptions {
        steps: 2,
        sample_rate: 24_000,
        duration_sec: 1,
        seed: 42,
        guidance_scale: 2.0,
        instruct: None,
    };
    let audio = pipeline
        .generate_audio_with_reference_wav("Hi", &samples, &options)
        .expect("AuK CFMEdit end-to-end generate");

    assert_eq!(audio.channels, 1);
    assert!(!audio.samples.is_empty());
    let finite_count = audio.samples.iter().filter(|v| v.is_finite()).count();
    assert_eq!(finite_count, audio.samples.len(), "non-finite samples");
    let max_abs = audio.samples.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(max_abs > 0.0, "all-zero audio buffer");
    eprintln!(
        "AuK CFMEdit end-to-end OK: {} samples @ {} Hz, max_abs={}",
        audio.samples.len(),
        audio.sample_rate,
        max_abs,
    );
}
