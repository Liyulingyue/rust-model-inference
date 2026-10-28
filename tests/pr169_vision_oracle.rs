//! Actual CLI vision encoders versus pinned llama.cpp, including DeepStack.
use rust_model_inference::{
    core::thread_pool::ComputePool,
    models::qwen3::vision::{VisionEncoder, VisionScratchpad},
    GGUFLoader,
};
use std::sync::Arc;

fn compare(model_env: &str, reference_env: &str, hybrid: bool) {
    let loader = GGUFLoader::from_file(std::env::var_os(model_env).expect(model_env)).unwrap();
    let reference = std::fs::read(std::env::var_os(reference_env).expect(reference_env)).unwrap();
    let side = 512;
    // mtmd-debug supplies already normalized floats directly.
    let pixels: Vec<f32> = (0..side * side).flat_map(|_| [0.0, 0.0, 1.0]).collect();
    let pool = Arc::new(ComputePool::new(8));
    let (actual, tokens, stacks) = if hybrid {
        use rust_model_inference::models::qwen35::vision;
        let model = vision::VisionEncoder::from_source(&loader, pool.clone()).unwrap();
        let mut scratch = vision::VisionScratchpad::new(&model.config);
        let grid = model
            .encode_image(&pixels, side, side, &mut scratch, &pool)
            .unwrap();
        (scratch.projected, grid.token_count(), 0)
    } else {
        let mut model = VisionEncoder::from_source(&loader, pool).unwrap();
        model.precompute();
        let mut scratch = VisionScratchpad::new(&model.config);
        model
            .encode_image(&pixels, side, side, &mut scratch)
            .unwrap();
        let dim = model.config.projection_dim;
        let tokens = scratch.projected.len() / dim;
        let stacks = scratch.deepstack.len() / scratch.projected.len();
        let mut actual = Vec::with_capacity(scratch.projected.len() + scratch.deepstack.len());
        for token in 0..tokens {
            actual.extend_from_slice(&scratch.projected[token * dim..(token + 1) * dim]);
            for stack in 0..stacks {
                let start = (stack * tokens + token) * dim;
                actual.extend_from_slice(&scratch.deepstack[start..start + dim]);
            }
        }
        (actual, tokens, stacks)
    };
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
    eprintln!(
        "{model_env}: {tokens} tokens, {stacks} DeepStack outputs, relative L2={relative_l2:.8}"
    );
    assert!(relative_l2 < 0.01, "projected relative L2 {relative_l2}");
}

#[test]
#[ignore = "requires real mmproj and pinned llama.cpp projection dump"]
fn intern_decision_vision() {
    compare(
        "RMI_INTERN_DECISION_0_8B_MMPROJ",
        "RMI_INTERN_VISION_REFERENCE",
        true,
    );
}

#[test]
#[ignore = "requires real mmproj and pinned llama.cpp projection dump"]
fn physbrain_vision() {
    compare(
        "RMI_PHYSBRAIN_1_5_2B_MMPROJ",
        "RMI_PHYSBRAIN_VISION_REFERENCE",
        false,
    );
}

#[test]
#[ignore = "requires real mmproj and pinned llama.cpp projection dump"]
fn occamy_vision() {
    compare("RMI_OCCAMY_MMPROJ", "RMI_OCCAMY_VISION_REFERENCE", true);
}
