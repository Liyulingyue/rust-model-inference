//! Actual CLI vision encoder versus pinned llama.cpp, including DeepStack.
//!
//! Two runs, because they answer different questions.
//!
//! * The F32 run keeps the vision tower in F32. Its error is the structure
//!   of the tower alone -- patch embedding, spatial merge, position
//!   embedding, RoPE and the DeepStack order -- so it is checked tightly.
//! * The production run is the path the CLI takes: the tower's weights are
//!   quantized to Q8_0, which adds a per-layer error that compounds through
//!   the ViT and is amplified by the massive activations that start at
//!   layer 10. Its bound is the measured envelope, and it is the run that
//!   catches a wrong rotation: with the interleaved axis assignment the
//!   production run measured 0.5600 against llama.cpp, against 0.1635 with
//!   the vision tower's own contiguous assignment.
//!
//! See `tools/oracle/physbrain/README.md` for the reference dump.
use rust_model_inference::{
    core::thread_pool::ComputePool,
    models::qwen3::vision::{VisionEncoder, VisionScratchpad},
    GGUFLoader,
};
use std::sync::Arc;

/// Which weight representation the vision tower runs with.
#[derive(Clone, Copy)]
enum Weights {
    /// F32 weights and activations: the structural reference.
    F32,
    /// What the CLI runs: Q8_0 weights.
    Q8_0,
}

fn compare(model_env: &str, reference_env: &str, weights: Weights, bound: f64) {
    let loader = GGUFLoader::from_file(std::env::var_os(model_env).expect(model_env)).unwrap();
    let reference = std::fs::read(std::env::var_os(reference_env).expect(reference_env)).unwrap();
    let side = 512;
    // mtmd-debug supplies already normalized floats directly.
    let pixels: Vec<f32> = (0..side * side).flat_map(|_| [0.0, 0.0, 1.0]).collect();
    let pool = Arc::new(ComputePool::new(8));
    let mut model = VisionEncoder::from_source(&loader, pool).unwrap();
    match weights {
        Weights::F32 => model.precompute_lossless(),
        Weights::Q8_0 => model.precompute(),
    }
    let mut scratch = VisionScratchpad::new(&model.config);
    model
        .encode_image(&pixels, side, side, &mut scratch)
        .unwrap();
    let dim = model.config.projection_dim;
    let tokens = scratch.projected.len() / dim;
    let stacks = scratch.deepstack.len() / scratch.projected.len();
    // The reference concatenates per token: the projected features and then
    // the three DeepStack outputs of that token.
    let mut actual = Vec::with_capacity(scratch.projected.len() + scratch.deepstack.len());
    for token in 0..tokens {
        actual.extend_from_slice(&scratch.projected[token * dim..(token + 1) * dim]);
        for stack in 0..stacks {
            let start = (stack * tokens + token) * dim;
            actual.extend_from_slice(&scratch.deepstack[start..start + dim]);
        }
    }
    assert_eq!(reference.len(), actual.len() * 4);
    let expected: Vec<f32> = reference
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert!(actual.iter().chain(&expected).all(|v| v.is_finite()));
    let norm: f64 = expected.iter().map(|&v| f64::from(v).powi(2)).sum();
    let diff: f64 = actual
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let relative_l2 = (diff / norm).sqrt();
    let label = match weights {
        Weights::F32 => "F32 weights",
        Weights::Q8_0 => "Q8_0 weights",
    };
    eprintln!(
        "{model_env}: {label}, {tokens} tokens, {stacks} DeepStack outputs, relative L2={relative_l2:.8}"
    );
    assert!(
        relative_l2 < bound,
        "projected relative L2 {relative_l2} exceeds {bound} for {label}"
    );
}

#[test]
#[ignore = "requires real mmproj and pinned llama.cpp projection dump"]
fn physbrain_vision() {
    compare(
        "RMI_PHYSBRAIN_1_5_2B_MMPROJ",
        "RMI_PHYSBRAIN_VISION_REFERENCE",
        Weights::F32,
        0.01,
    );
    compare(
        "RMI_PHYSBRAIN_1_5_2B_MMPROJ",
        "RMI_PHYSBRAIN_VISION_REFERENCE",
        Weights::Q8_0,
        0.25,
    );
}
