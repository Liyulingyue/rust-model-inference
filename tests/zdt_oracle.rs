//! Run with RMI_ZDT_MMPROJ and, for numerical alignment, RMI_ZDT_REFERENCE
//! pointing to llama.cpp's final projected contiguous little-endian F32 dump.
use rust_model_inference::{
    core::thread_pool::ComputePool, models::zdt::vision::ZdtVisionEncoder, GGUFLoader,
};
use std::sync::Arc;

#[test]
#[ignore = "requires real mmproj and pinned llama.cpp projection dumps"]
fn blue_image_matches_llama_cpp() {
    let path = std::env::var_os("RMI_ZDT_MMPROJ").expect("RMI_ZDT_MMPROJ is required");
    assert!(std::env::var_os("RMI_ZDT_REFERENCE").is_some());
    assert!(std::env::var_os("RMI_ZDT_REFERENCE_BF16").is_some());
    let source = GGUFLoader::from_file(path).expect("load ZDT mmproj");
    let model = ZdtVisionEncoder::from_source(&source, Arc::new(ComputePool::new(8)))
        .expect("ZDT vision contract");
    let side = model.config.image_size;
    let pixels: Vec<f32> = (0..side * side).flat_map(|_| [0.0, 0.0, 1.0]).collect();
    let actual = model.encode(&pixels, side, side).expect("ZDT encode");
    assert_eq!(actual.len(), 256 * 4096);
    assert!(actual.iter().all(|v| v.is_finite()));
    for (key, bound) in [
        ("RMI_ZDT_REFERENCE", 1e-3),
        ("RMI_ZDT_REFERENCE_BF16", 1e-2),
    ] {
        if let Some(path) = std::env::var_os(key) {
            let bytes = std::fs::read(path).expect("read Oracle output");
            assert_eq!(bytes.len(), actual.len() * 4);
            let expected: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            assert!(expected.iter().all(|v| v.is_finite()));
            let norm: f64 = expected.iter().map(|&v| f64::from(v).powi(2)).sum();
            let diff: f64 = actual
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum();
            let relative_l2 = (diff / norm).sqrt();
            eprintln!("ZDT {key} projected relative L2 = {relative_l2:.8}");
            assert!(
                relative_l2 < bound,
                "ZDT projected relative L2 {relative_l2}"
            );
        }
    }
}
