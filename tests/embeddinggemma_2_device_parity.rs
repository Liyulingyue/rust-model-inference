//! CPU and device parity for the EmbeddingGemma-2 vision tower.
//!
//! The vision tower is the one EmbeddingGemma-2 path whose attention runs on
//! the device: the patches attend in both directions over every patch, which
//! the decoder-shaped device attention cannot express, so the tower uses the
//! fused bidirectional kernel instead. This test pins the device result to
//! the CPU one.
//!
//! Run with:
//!   RMI_EMBEDDINGGEMMA_2=/path/to/embeddinggemma-2-F16.gguf \
//!   cargo test --profile release-fast --features vulkan \
//!     --test embeddinggemma_2_device_parity -- --nocapture

use rust_model_inference::compute::ComputePolicy;
use rust_model_inference::models::embeddinggemma2::EmbeddingGemmaVisionModel;
use rust_model_inference::TensorSource;
use std::sync::Arc;

fn relative_l2(actual: &[f32], expected: &[f32]) -> f32 {
    let mut difference = 0.0f64;
    let mut norm = 0.0f64;
    for (a, b) in actual.iter().zip(expected.iter()) {
        difference += f64::from(*a - *b) * f64::from(*a - *b);
        norm += f64::from(*b) * f64::from(*b);
    }
    (difference / norm).sqrt() as f32
}

fn cosine(actual: &[f32], expected: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    for (a, b) in actual.iter().zip(expected.iter()) {
        dot += f64::from(*a) * f64::from(*b);
        actual_norm += f64::from(*a) * f64::from(*a);
        expected_norm += f64::from(*b) * f64::from(*b);
    }
    (dot / (actual_norm.sqrt() * expected_norm.sqrt())) as f32
}

#[test]
fn vision_tower_device_matches_cpu() {
    let Some(path) = std::env::var_os("RMI_EMBEDDINGGEMMA_2") else {
        eprintln!("RMI_EMBEDDINGGEMMA_2 is unset; skipping the device parity check");
        return;
    };
    let loader = rust_model_inference::GGUFLoader::from_file(path).expect("load GGUF");
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let model = EmbeddingGemmaVisionModel::from_source(&source, 8).expect("vision model");

    // 768x768 is the tower's default output length: 2304 patches, which is
    // the shape the media path actually runs.
    let (image_h, image_w) = (768usize, 768usize);
    let pixels: Vec<f32> = (0..image_h * image_w * 3)
        .map(|i| (i as f32 * 0.0013).sin() * 0.5 + 0.5)
        .collect();

    let cpu = {
        let _scope = ComputePolicy::Cpu.enter_legacy_scope();
        model
            .encode_tower(&pixels, image_h, image_w)
            .expect("vision tower on the CPU")
    };

    rust_model_inference::ops::float::enable_gpu();
    let device = {
        let _scope = ComputePolicy::Vulkan.enter_legacy_scope();
        model
            .encode_tower(&pixels, image_h, image_w)
            .expect("vision tower on the device")
    };

    // 768x768 is 48x48 patches, merged into 16x16 tokens of 768 floats.
    assert_eq!(cpu.len(), 16 * 16 * 768);
    assert_eq!(device.len(), cpu.len());
    let l2 = relative_l2(&device, &cpu);
    let similarity = cosine(&device, &cpu);
    eprintln!("vision tower device vs cpu: relative_l2={l2:.8} cosine={similarity:.8}");
    assert!(
        l2 < 0.02,
        "device vision tower drifted from the CPU result: relative_l2={l2}"
    );
    assert!(
        similarity > 0.999,
        "device vision tower drifted from the CPU result: cosine={similarity}"
    );
}
