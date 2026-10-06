//! AuK-Flash distilled variant smoke test.
//!
//! Loads the AuK-Flash GGUF (distilled variant with 4 steps + guidance 0) and
//! validates that the same `AukPipeline` produces finite audio. Flash uses
//! the identical architecture (Flux2Edit) and tensor layout as AuK-Base,
//! only the weights differ -- the only changes for Flash inference are:
//!   - steps: 4 (Base uses 32)
//!   - guidance_scale: 0.0 (Base uses 2.0; 0 disables CFG unconditional run)
//!
//! Run with:
//!   RMI_AUK_FLASH=models/auk-gguf/auk-flash-f16.gguf \
//!   RMI_AUK_VAE=models/auk-gguf/auk-vae-f32.gguf \
//!   RMI_AUK_TEXT=models/Qwen2.5-Omni-3B-GGUF/Qwen2.5-Omni-3B-Q8_0.gguf \
//!   cargo test --features vulkan --release --test auk_flash_smoke -- --ignored --nocapture

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
#[ignore = "AuK-Flash smoke takes ~5min; run with --ignored"]
fn flash_distilled_4_step_zero_guidance_produces_finite_output() {
    let Some(dit) = env_path("RMI_AUK_FLASH") else {
        eprintln!("skipping: set RMI_AUK_FLASH to the AuK-Flash DiT GGUF");
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
    let Some(dit_src) = open_source(&dit) else {
        return;
    };
    let Some(vae_src) = open_source(&vae) else {
        return;
    };
    let Some(text_src) = open_source(&text) else {
        return;
    };

    // Validate architecture tag so we don't silently run on the wrong file.
    let arch = dit_src
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or("");
    assert_eq!(
        arch, "audiocpp",
        "AuK-Flash GGUF arch tag mismatch (got {arch:?}, expected audiocpp)"
    );

    let pipeline = rust_model_inference::models::diffusion::auk::AukPipeline::load(
        dit_src,
        vae_src,
        Some(text_src),
        1,
    )
    .expect("AuK-Flash pipeline load");
    rust_model_inference::ops::enable_gpu();

    // AuK-Flash distilled variant: 4 steps, guidance 0 (no CFG unconditional pass).
    let options = rust_model_inference::models::diffusion::auk::AukOptions {
        steps: 4,
        sample_rate: 24_000,
        duration_sec: 1,
        seed: 42,
        guidance_scale: 0.0,
        instruct: None,
    };
    let audio = pipeline
        .generate_audio("Hi", &options)
        .expect("AuK-Flash pipeline generate");

    assert_eq!(audio.channels, 1);
    assert!(!audio.samples.is_empty());
    let finite_count = audio.samples.iter().filter(|v| v.is_finite()).count();
    assert_eq!(finite_count, audio.samples.len(), "non-finite samples");
    let max_abs = audio.samples.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(max_abs > 0.0, "all-zero audio buffer");
    eprintln!(
        "AuK-Flash smoke OK: {} samples @ {} Hz, max_abs={}",
        audio.samples.len(),
        audio.sample_rate,
        max_abs,
    );
}
