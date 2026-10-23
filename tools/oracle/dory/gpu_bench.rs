//! Warm resident decode benchmark. Link against each build's rlib and alternate runs.
use rust_model_inference::{
    app::cli::KvFormat, compute::ComputePolicy, core::tensor::TensorSource,
    models::dory::DoryModel, ops, GGUFLoader,
};
use std::{sync::Arc, time::Instant};

fn main() {
    for key in ["RMI_PARITY_TRACE", "RMI_SCALAR"] {
        assert!(
            std::env::var_os(key).is_none(),
            "unset {key} for benchmarking"
        );
    }
    let args: Vec<_> = std::env::args().collect();
    let kv = match args.get(2).map(String::as_str) {
        Some("f32") => KvFormat::F32,
        Some("f16") => KvFormat::F16,
        _ => panic!("usage: gpu_bench MODEL f32|f16"),
    };
    ops::enable_gpu();
    let context = ops::get_vulkan_context().expect("real Vulkan context");
    assert!(!context.is_software_icd());
    let source: Arc<dyn TensorSource> = Arc::new(GGUFLoader::from_file(&args[1]).unwrap());
    let model = DoryModel::from_source(source, 4).unwrap();
    println!("device={} kv={kv:?}", context.device_name());
    let mut session = model
        .new_session_with_compute(9, kv, ComputePolicy::Vulkan)
        .unwrap();
    let tokens = [22177, 1897, 1429, 22177, 5325, 1897, 1429, 22177, 5325];
    let expected = [1897, 1429, 22177, 5325, 1897, 1429, 22177, 5325];
    for round in 0..4 {
        session.reset(&model, kv);
        let before = context.compute_stats();
        let mut seconds = Vec::new();
        for (step, &token) in tokens.iter().enumerate() {
            let started = Instant::now();
            let logits = model.prefill(&[token], &mut session).unwrap();
            seconds.push(started.elapsed().as_secs_f64());
            let best = (1..logits.len()).fold(0, |b, i| if logits[i] > logits[b] { i } else { b });
            if let Some(&next) = expected.get(step) {
                assert_eq!(best, next, "step={step}");
            }
            assert!(logits.iter().all(|v| v.is_finite()));
        }
        let stats = context.compute_stats().since(before);
        assert!(stats.submissions > 0);
        assert_eq!(stats.static_uploads, 0);
        println!(
            "round={round} warmup={} seconds={seconds:?} total={} stats={stats:?}",
            round == 0,
            seconds.iter().sum::<f64>()
        );
    }
}
