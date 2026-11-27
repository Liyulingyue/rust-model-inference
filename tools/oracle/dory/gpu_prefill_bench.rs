//! Compare identical resident prompts with one-row and 32-row prefill calls.
use rust_model_inference::{
    app::cli::KvFormat, compute::ComputePolicy, core::tensor::TensorSource,
    models::dory::DoryModel, ops, GGUFLoader,
};
use std::{sync::Arc, time::Instant};

fn main() {
    assert!(std::env::var_os("RMI_PARITY_TRACE").is_none());
    assert!(std::env::var_os("RMI_SCALAR").is_none());
    ops::enable_gpu();
    let context = ops::get_vulkan_context().expect("hardware Vulkan");
    assert!(!context.is_software_icd());
    let source: Arc<dyn TensorSource> =
        Arc::new(GGUFLoader::from_file(std::env::args().nth(1).expect("model path")).unwrap());
    let model = DoryModel::from_source(source, 4).unwrap();
    let tokens: Vec<u32> = (0..64).map(|i| [22177, 1897, 1429, 5325][i % 4]).collect();
    for kv in [KvFormat::F32, KvFormat::F16] {
        let mut session = model
            .new_session_with_compute(64, kv, ComputePolicy::Vulkan)
            .unwrap();
        println!("kv={kv:?} cache_bytes={}", session.kv_cache_bytes());
        let mut reference: Option<Vec<f32>> = None;
        // One warmup per shape, followed by balanced alternating measurements.
        for (round, batch) in [1, 32, 1, 32, 32, 1, 1, 32].into_iter().enumerate() {
            session.reset(&model, kv).unwrap();
            let before = context.compute_stats();
            let started = Instant::now();
            let mut logits = Vec::new();
            for chunk in tokens.chunks(batch) {
                logits = model.prefill(chunk, &mut session).unwrap();
            }
            let seconds = started.elapsed().as_secs_f64();
            let stats = context.compute_stats().since(before);
            assert!(stats.submissions > 0);
            assert_eq!(stats.static_uploads, 0);
            assert!(logits.iter().all(|v| v.is_finite()));
            if let Some(expected) = &reference {
                let error = logits
                    .iter()
                    .zip(expected)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                    .sum::<f64>();
                let norm = expected.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
                let max = expected.iter().map(|v| v.abs()).fold(1.0, f32::max);
                assert!((error / norm).sqrt() <= 0.01);
                assert!(logits
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() / max <= 0.01));
            } else {
                reference = Some(logits);
            }
            println!(
                "kv={kv:?} round={round} warmup={} batch={batch} seconds={seconds} stats={stats:?}",
                round < 2
            );
        }
    }
}
