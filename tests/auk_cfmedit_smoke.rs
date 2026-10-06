//! CFMEdit reference-audio conditioning smoke test for AuK.
//!
//! Validates that `AukPipeline::generate_audio_with_audio` accepts pre-computed
//! audio embeddings (length = `audio_tokens * TEXT_IN`) and produces a finite
//! audio output. The audio tower encoder is NOT yet ported, so we supply zeros
//! as a stand-in for real audio embeddings — the goal here is to prove the
//! joint-sequence integration is wired correctly, not to validate numerics.
//!
//! Run with:
//!   RMI_AUK_GGUF=models/auk-gguf/auk-base-f16.gguf \
//!   RMI_AUK_VAE=models/auk-gguf/auk-vae-f32.gguf \
//!   RMI_AUK_TEXT=models/Qwen2.5-Omni-3B-GGUF/Qwen2.5-Omni-3B-Q8_0.gguf \
//!   cargo test --features vulkan --release --test auk_cfmedit_smoke -- --ignored --nocapture

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
#[ignore = "CFMEdit smoke takes ~5min; run with --ignored"]
fn cfmedit_zero_audio_conditioning_produces_finite_output() {
    let Some(dit) = env_path("RMI_AUK_GGUF") else {
        eprintln!("skipping: set RMI_AUK_GGUF");
        return;
    };
    let Some(vae) = env_path("RMI_AUK_VAE") else {
        eprintln!("skipping: set RMI_AUK_VAE");
        return;
    };
    let Some(text) = env_path("RMI_AUK_TEXT") else {
        eprintln!("skipping: set RMI_AUK_TEXT");
        return;
    };
    let Some(dit_src) = open_source(&dit) else { return; };
    let Some(vae_src) = open_source(&vae) else { return; };
    let Some(text_src) = open_source(&text) else { return; };

    let pipeline = rust_model_inference::models::diffusion::auk::AukPipeline::load(
        dit_src, vae_src, Some(text_src), 1,
    )
    .expect("AuK pipeline load");
    rust_model_inference::ops::enable_gpu();

    // 5 audio tokens × TEXT_IN=2048 zero embeddings (stand-in for the Qwen
    // audio tower output). The DiT should treat this as 5 extra conditioning
    // tokens prepended to the text tokens.
    let audio_tokens = 5usize;
    let audio_conditioning = vec![0.0f32; audio_tokens * 2048];

    let options = rust_model_inference::models::diffusion::auk::AukOptions {
        steps: 2,
        sample_rate: 24_000,
        duration_sec: 1,
        seed: 42,
        guidance_scale: 2.0,
        instruct: None,
    };
    let audio = pipeline
        .generate_audio_with_audio("Hi", &audio_conditioning, audio_tokens, &options)
        .expect("AuK CFMEdit pipeline generate");

    assert_eq!(audio.channels, 1);
    assert!(!audio.samples.is_empty());
    let finite_count = audio.samples.iter().filter(|v| v.is_finite()).count();
    assert_eq!(finite_count, audio.samples.len(), "non-finite samples");
    let max_abs = audio.samples.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(max_abs > 0.0, "all-zero audio buffer");
    eprintln!(
        "AuK CFMEdit smoke OK: {} samples @ {} Hz, max_abs={}",
        audio.samples.len(),
        audio.sample_rate,
        max_abs,
    );
}