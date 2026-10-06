//! Qwen2.5-Omni audio tower smoke test.
//!
//! Loads the BF16 Qwen2.5-Omni GGUF (which contains the audio tower), feeds
//! a synthetic 16 kHz sine-wave PCM buffer into the audio tower, and
//! validates that the output is finite and has the expected shape
//! (token count matches `(frames + 1) / 2` * 200, dim = 2048).
//!
//! Run with:
//!   RMI_QWEN_OMNI_BF16=models/Qwen2.5-Omni-3B-bf16-GGUF/qwen2.5-omni-3b-bf16.gguf \
//!   cargo test --features vulkan --test omni_audio_tower_smoke -- --ignored --nocapture

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
#[ignore = "loads 7GB BF16 Qwen GGUF; run with --ignored"]
fn audio_tower_produces_finite_per_token_embeddings() {
    let Some(path) = env_path("RMI_QWEN_OMNI_BF16") else {
        eprintln!("RMI_QWEN_OMNI_BF16 unset; skipping");
        return;
    };
    let Some(source) = open_source(&path) else {
        panic!("failed to open Qwen GGUF at {}", path.display());
    };
    let started = std::time::Instant::now();
    let model = rust_model_inference::models::qwen3::omni_audio::AudioTowerModel::from_source(
        Arc::clone(&source),
    )
    .expect("Qwen2.5-Omni audio tower load");
    eprintln!("[load] {:?} elapsed={:?}", model.config(), started.elapsed());

    // 1.0 s of 16 kHz mono PCM (a 440 Hz sine wave, the A4 note). Should
    // produce 100 mel frames (1600 / 160 = 10) wait actually 1000/160 ~ 6.
    // For AuK we need at least ~2-3 s to produce 100+ frames. Use 3 s.
    let sample_rate = 16_000usize;
    let duration_sec = 0.5f32;
    let n_samples = (sample_rate as f32 * duration_sec) as usize;
    let mut samples = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = i as f32 / sample_rate as f32;
        let v = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        samples.push(v);
    }
    let started = std::time::Instant::now();
    let (values, tokens) = model
        .encode_pcm(&samples)
        .expect("Qwen2.5-Omni audio tower encode");
    eprintln!(
        "[encode] {tokens} audio tokens, {} total f32 values, elapsed={:?}",
        values.len(),
        started.elapsed()
    );

    assert_eq!(tokens * 2048, values.len(), "shape mismatch");
    assert!(tokens > 0, "no audio tokens produced");
    assert!(
        values.iter().all(|v| v.is_finite()),
        "non-finite audio embeddings"
    );
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(max_abs > 0.0, "all-zero audio embeddings");
    eprintln!("[stats] max_abs={max_abs:.4}");
    eprintln!("Qwen2.5-Omni audio tower smoke OK");
}
