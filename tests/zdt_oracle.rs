//! Numerical alignment of the ZDTaichu5.0-9B vision tower.
//!
//! Run with RMI_ZDT_MMPROJ pointing at the mmproj and, for the references:
//!
//!   RMI_ZDT_REFERENCE        llama.cpp's projection dump, F32 weights widened
//!                            exactly from the same BF16 values
//!   RMI_ZDT_REFERENCE_BF16   llama.cpp's projection dump with the original
//!                            BF16 weights
//!   RMI_ZDT_PYTORCH_REFERENCE the upstream C-RADIO tower run in PyTorch, with
//!                            the input conditioner disabled so it sees the
//!                            same pixels as the Rust encoder
//!
//! See `tools/oracle/zdt/README.md` for the reference build and the commands
//! that produce each dump.
use rust_model_inference::{
    core::thread_pool::ComputePool, models::zdt::vision::ZdtVisionEncoder, GGUFLoader,
};
use std::sync::Arc;

/// The relative L2 between the Rust projection and a reference dump.
fn relative_l2(actual: &[f32], expected: &[f32]) -> f64 {
    let norm: f64 = expected.iter().map(|&v| f64::from(v).powi(2)).sum();
    let diff: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    (diff / norm).sqrt()
}

/// Ask for the device path when the environment says so.
///
/// The projection's matmuls then run on the GPU, and the comparison against
/// the reference is measured on the device's numbers.
fn request_device_if_asked() {
    if std::env::var_os("RMI_ZDT_GPU").is_some() {
        rust_model_inference::ops::enable_gpu();
    }
}

fn read_f32_dump(path: &std::ffi::OsStr, expected_len: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read Oracle output");
    assert_eq!(bytes.len(), expected_len * 4);
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert!(values.iter().all(|v| v.is_finite()));
    values
}

#[test]
#[ignore = "requires real mmproj and pinned reference projection dumps"]
fn blue_image_matches_llama_cpp() {
    let path = std::env::var_os("RMI_ZDT_MMPROJ").expect("RMI_ZDT_MMPROJ is required");
    assert!(std::env::var_os("RMI_ZDT_REFERENCE").is_some());
    assert!(std::env::var_os("RMI_ZDT_REFERENCE_BF16").is_some());
    request_device_if_asked();
    let source = GGUFLoader::from_file(path).expect("load ZDT mmproj");
    let model = ZdtVisionEncoder::from_source(&source, Arc::new(ComputePool::new(8)))
        .expect("ZDT vision contract");
    let side = model.config.image_size;
    let pixels: Vec<f32> = (0..side * side).flat_map(|_| [0.0, 0.0, 1.0]).collect();
    let started = std::time::Instant::now();
    let actual = model.encode(&pixels, side, side).expect("ZDT encode");
    eprintln!(
        "ZDT projection encoded in {:.3}s on {}",
        started.elapsed().as_secs_f64(),
        if std::env::var_os("RMI_ZDT_GPU").is_some() {
            "the device"
        } else {
            "the CPU"
        }
    );
    assert_eq!(actual.len(), 256 * 4096);
    assert!(actual.iter().all(|v| v.is_finite()));
    for (key, bound) in [
        ("RMI_ZDT_REFERENCE", 1e-3),
        ("RMI_ZDT_REFERENCE_BF16", 1e-2),
    ] {
        if let Some(path) = std::env::var_os(key) {
            let expected = read_f32_dump(&path, actual.len());
            let relative_l2 = relative_l2(&actual, &expected);
            eprintln!("ZDT {key} projected relative L2 = {relative_l2:.8}");
            assert!(
                relative_l2 < bound,
                "ZDT projected relative L2 {relative_l2}"
            );
        }
    }
}

/// The upstream C-RADIO tower, run in PyTorch, must agree with the Rust
/// encoder on the same pixels and the same position embeddings.
///
/// This is the check that the tower's arithmetic is right on its own: the
/// llama.cpp reference is itself a port of the same upstream code, so it
/// cannot catch a shared misreading of the checkpoint.
///
/// The dump must come from `pytorch_reference.py --mode conditioner-identity
/// --positions <mmproj>`. Two upstream details differ from the GGUF, and
/// both are excluded here on purpose:
///
///   - the upstream path normalizes twice (the image processor with the
///     ImageNet statistics, then the tower's conditioner with the CLIP
///     statistics), while llama.cpp and the Rust engine normalize once, so
///     the conditioner is replaced by the identity;
///   - the upstream interpolates the 128x128 position table with
///     `align_corners=False`, while the GGUF was converted with
///     `align_corners=True`, so the GGUF's pre-downsampled table is used.
///
/// The measured gap for each is recorded in `tools/oracle/zdt/README.md`.
#[test]
#[ignore = "requires real mmproj and the upstream PyTorch projection dump"]
fn blue_image_matches_upstream_pytorch() {
    let path = std::env::var_os("RMI_ZDT_MMPROJ").expect("RMI_ZDT_MMPROJ is required");
    let reference = std::env::var_os("RMI_ZDT_PYTORCH_REFERENCE")
        .expect("RMI_ZDT_PYTORCH_REFERENCE is required");
    request_device_if_asked();
    let source = GGUFLoader::from_file(path).expect("load ZDT mmproj");
    let model = ZdtVisionEncoder::from_source(&source, Arc::new(ComputePool::new(8)))
        .expect("ZDT vision contract");
    let side = model.config.image_size;
    let pixels: Vec<f32> = (0..side * side).flat_map(|_| [0.0, 0.0, 1.0]).collect();
    let actual = model.encode(&pixels, side, side).expect("ZDT encode");
    let expected = read_f32_dump(&reference, actual.len());
    let relative_l2 = relative_l2(&actual, &expected);
    eprintln!("ZDT upstream PyTorch projected relative L2 = {relative_l2:.8}");
    assert!(
        relative_l2 < 5e-3,
        "ZDT upstream PyTorch projected relative L2 {relative_l2}"
    );
}
