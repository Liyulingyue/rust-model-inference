//! Real-device operator checks and resident Dory Vulkan/CPU comparison; report submission
//! counts separately so CPU fallback cannot be mistaken for GPU validation.

use rust_model_inference::{
    app::cli::KvFormat, compute::ComputePolicy, core::tensor::TensorSource,
    models::dory::DoryModel, ops, GGUFLoader,
};
use std::sync::Arc;

fn main() {
    assert!(
        std::env::var_os("RMI_PARITY_TRACE").is_none(),
        "tracing would disable offload"
    );
    assert!(
        std::env::var_os("RMI_SCALAR").is_none(),
        "scalar mode would disable offload"
    );
    ops::enable_gpu();
    let context = ops::get_vulkan_context().expect("real Vulkan context");
    assert!(!context.is_software_icd());
    println!("device={}", context.device_name());
    for rows in [1, 3] {
        let before = context.compute_stats();
        rust_model_inference::vulkan::run_batched_matmul_check(
            context,
            &["q5_k", "q6_k", "bf16"],
            rows,
        )
        .expect("GPU batched matmul checks");
        let stats = context.compute_stats().since(before);
        assert!(stats.submissions > 0, "operator check must submit GPU work");
        println!("operator_rows={rows} stats={stats:?}");
    }
    let source: Arc<dyn TensorSource> =
        Arc::new(GGUFLoader::from_file(std::env::args().nth(1).expect("model path")).unwrap());
    let model = DoryModel::from_source(source, 4).unwrap();
    for kv in [KvFormat::F32, KvFormat::F16] {
        let before = context.compute_stats();
        let mut gpu = model
            .new_session_with_compute(9, kv, ComputePolicy::Vulkan)
            .unwrap();
        let initialization = context.compute_stats().since(before);
        assert!(initialization.static_uploads > 0);
        let before_inference = context.compute_stats();
        let (mut gpu_time, mut cpu_time) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        let mut cpu = model.new_session(9, kv);
        let mut token = 22177u32;
        let mut generated = Vec::new();
        let mut max_l2 = 0.0f64;
        let mut max_scaled = 0.0f64;
        let mut bit_mismatches = 0usize;
        for step in 0..9 {
            let started = std::time::Instant::now();
            let actual = model.prefill(&[token], &mut gpu).unwrap();
            gpu_time += started.elapsed();
            let before_cpu = context.compute_stats();
            let started = std::time::Instant::now();
            let expected = {
                let _scope = ComputePolicy::Cpu.enter_legacy_scope();
                model.prefill(&[token], &mut cpu).unwrap()
            };
            cpu_time += started.elapsed();
            assert_eq!(context.compute_stats().since(before_cpu).submissions, 0);
            assert_eq!(actual.len(), expected.len());
            let mut diff2 = 0.0f64;
            let mut ref2 = 0.0f64;
            let mut abs = 0.0f64;
            let mut scale = 1.0f64;
            for (&a, &e) in actual.iter().zip(&expected) {
                assert!(a.is_finite() && e.is_finite());
                let delta = f64::from(a) - f64::from(e);
                diff2 += delta * delta;
                ref2 += f64::from(e) * f64::from(e);
                abs = abs.max(delta.abs());
                scale = scale.max(f64::from(e).abs());
                bit_mismatches += usize::from(a.to_bits() != e.to_bits());
            }
            max_l2 = max_l2.max((diff2 / ref2.max(f64::MIN_POSITIVE)).sqrt());
            max_scaled = max_scaled.max(abs / scale);
            let best = |v: &[f32]| (1..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
            assert_eq!(best(&actual), best(&expected));
            if step < 8 {
                token = best(&actual) as u32;
                generated.push(token);
            }
        }
        assert!(max_l2 <= 0.01 && max_scaled <= 0.01);
        assert_eq!(
            generated,
            [1897, 1429, 22177, 5325, 1897, 1429, 22177, 5325]
        );
        let inference = context.compute_stats().since(before_inference);
        assert!(inference.submissions > 0);
        assert_eq!(inference.static_uploads, 0);
        println!("kv={kv:?} initialization={initialization:?} gpu_seconds={} cpu_seconds={} inference={inference:?}", gpu_time.as_secs_f64(), cpu_time.as_secs_f64());
        println!("kv={kv:?} generated={generated:?} max_relative_l2={max_l2} max_scaled_max_abs={max_scaled} bit_mismatches={bit_mismatches} stats={:?}", context.compute_stats().since(before));
    }
}
