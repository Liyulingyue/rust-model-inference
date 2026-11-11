//! Hardware test for the Nemotron-3 Nano 30B-A3B (`nemotron_h_moe`) on
//! the Vulkan device.
//!
//! The 30B is a hybrid Mamba-2 / attention / mixture-of-experts tower: 52
//! layers made of 23 SSM layers, 23 routed-expert layers and 6 attention
//! layers. All of them run on the device.
//!
//! # Why the gate is 8% and not 1%
//!
//! The CPU path of this model is bit-exact with llama.cpp, because it
//! reproduces ggml's F32 accumulation structure. The device path cannot
//! be: its Q5_0, Q5_1 and Q4_K kernels sum the integer dots in a
//! different order than the host's NEON kernels, so a single projection
//! already differs by ~4e-3 relative (the Q5_0 `ssm_in` projection:
//! `max_rel = 3.791e-3`). The Mamba-2 recurrence then amplifies that
//! difference over the 52 layers, and the router can flip a near-tie on
//! the way.
//!
//! The measured numbers on an RTX-class GB10 (see
//! `docs/develop/NEMOTRON3_NANO_30B_MOE.md` §B7 for the full attribution
//! table) are 5.8% / 4.9% / 1.9% relative L2 for a one-, two- and
//! three-token prefill. The gate is therefore 8% on both the relative L2
//! and the scaled maximum, and the test prints both numbers so a
//! regression shows how far it moved.
//!
//! # The Q8_0 tier is bit-exact
//!
//! That 8% bound is a property of the Q4_K_M tier, not of the device path.
//! Q4_K_M's `ssm_in` and attention q/k/v are Q5_0 and its `ssm_out` and
//! `attn_output` are Q4_K, and those two device kernels sum the integer
//! dots in a different order than the host's NEON kernels. The Q8_0 tier
//! has neither: every quantized tensor in it is Q8_0, whose device kernel
//! is bit-exact (`max_abs = 0`), and the rest is F32 (`exact_bits = true`).
//! Measured `relative_l2 = 0` for all three prefill lengths. Set
//! `RMI_NEMOTRON_MOE_EXACT=1` to hold the test to bit-exactness on that
//! tier.
//!
//! The functional gate is stronger than the numeric one: the device path
//! has to pick the same top-1 token as the CPU path.
//!
//! The prefill lengths matter: the Mamba-2 recurrence only starts to
//! differ from the host's when the scan state is non-zero, so a
//! single-token prefill cannot see a broken recurrence. The test runs one,
//! two and three tokens for that reason.
//!
//! The GGUF is not checked in; set `RMI_NEMOTRON_MOE_MODEL` to the
//! `Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf` path.

use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::models::nemotron_h::trunk::{
    NemotronConfig, NemotronModel, NemotronScratch,
};
use rust_model_inference::GGUFLoader;
use std::path::PathBuf;
use std::sync::Arc;

/// The error bound the device path has to stay inside.
///
/// The device kernels sum the quantized dots in a different order than
/// the host's NEON kernels, and the Mamba-2 recurrence amplifies the
/// difference. The worst measurement on the 30B is 5.8% (one-token
/// prefill); the gate leaves room for that without letting a real
/// regression through.
///
/// This bound applies to the tiers that carry Q5_0 / Q5_1 / Q4_K weights.
/// The Q8_0 tier is bit-exact; set `RMI_NEMOTRON_MOE_EXACT=1` to demand
/// that of any tier the test is pointed at.
const LOGIT_TOLERANCE: f64 = 0.08;

/// The bound this run is held to: zero when `RMI_NEMOTRON_MOE_EXACT` is set.
fn logit_tolerance() -> f64 {
    if std::env::var_os("RMI_NEMOTRON_MOE_EXACT").is_some() {
        0.0
    } else {
        LOGIT_TOLERANCE
    }
}

fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("RMI_NEMOTRON_MOE_MODEL")?);
    path.exists().then_some(path)
}

fn source() -> Option<Arc<dyn TensorSource>> {
    let path = model_path()?;
    Some(Arc::new(
        GGUFLoader::from_file(path).expect("open Nemotron-H MoE GGUF"),
    ))
}

#[test]
fn nemotron3_nano_30b_contract() {
    let Some(source) = source() else { return };
    let config = NemotronConfig::from_source(source.as_ref()).expect("Nemotron-H MoE config");
    assert_eq!(config.architecture, "nemotron_h_moe");
    assert_eq!(config.n_embd, 2688);
    assert_eq!(config.n_layer, 52);
    assert_eq!(config.n_head, 32);
    assert_eq!(config.n_head_kv, 2);
    assert_eq!(config.vocab_size, 131_072);
    assert_eq!(config.ssm_conv_kernel, 4);
    assert_eq!(config.ssm_state_size, 128);
    assert_eq!(config.ssm_group_count, 8);
    assert_eq!(config.ssm_inner_size, 4096);
    assert_eq!(config.ssm_time_step_rank, 64);
    assert_eq!(config.rope_dim, 84);
    assert_eq!(config.attention_layers.len(), 6);
    assert_eq!(config.ffn_layers.len(), 23);
    let moe = config.moe.as_ref().expect("routed experts");
    assert_eq!(moe.n_expert, 128);
    assert_eq!(moe.n_expert_used, 6);
    assert_eq!(moe.n_ff_exp, 1856);
    assert_eq!(moe.n_ff_shexp, 3712);
    assert_eq!(moe.n_expert_shared, 1);
    assert!(moe.expert_weights_norm);
    assert_eq!(moe.expert_weights_scale, 2.5);
}

#[cfg(feature = "vulkan")]
fn assert_numerically_aligned(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let (mut diff, mut reference, mut max_abs, mut scale) = (0.0f64, 0.0f64, 0.0f64, 1.0f64);
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        let delta = f64::from(a) - f64::from(b);
        diff += delta * delta;
        reference += f64::from(b) * f64::from(b);
        max_abs = max_abs.max(delta.abs());
        scale = scale.max(f64::from(b).abs());
    }
    let l2 = (diff / reference.max(f64::MIN_POSITIVE)).sqrt();
    let scaled_max_abs = max_abs / scale;
    let tolerance = logit_tolerance();
    eprintln!("GPU relative_l2={l2} scaled_max_abs={scaled_max_abs} tolerance={tolerance}");
    assert!(l2 <= tolerance, "relative L2 {l2} exceeds {tolerance}");
    assert!(
        scaled_max_abs <= tolerance,
        "scaled max abs {scaled_max_abs} exceeds {tolerance}"
    );
    // The functional gate: the device path has to pick the same token.
    let best = |v: &[f32]| (1..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
    assert_eq!(best(actual), best(expected));
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires the real 30B GGUF and a hardware Vulkan device"]
fn nemotron3_nano_30b_vulkan_matches_cpu() {
    use rust_model_inference::compute::ComputePolicy;
    rust_model_inference::ops::enable_gpu();
    let source = source().expect("set RMI_NEMOTRON_MOE_MODEL to the real GGUF");
    let mut cpu_model = NemotronModel::from_source(source.clone(), 4).unwrap();
    let mut gpu_model = NemotronModel::from_source(source, 4).unwrap();
    gpu_model
        .attach_gpu(8, 8, ComputePolicy::Vulkan)
        .expect("Nemotron-H MoE Vulkan session");
    assert!(gpu_model.gpu_available());
    let context = rust_model_inference::ops::get_vulkan_context().unwrap();
    let before = context.compute_stats();
    for tokens in [vec![22177], vec![22177, 1897], vec![22177, 1897, 1429]] {
        let mut cpu_scratch = NemotronScratch::new(&cpu_model.config, tokens.len());
        let mut gpu_scratch = NemotronScratch::new(&gpu_model.config, tokens.len());
        // The reference has to stay on the host: with the GPU enabled the
        // host's matmuls would be offloaded to the device, and the
        // comparison would then measure the device against itself.
        let expected = {
            let _cpu = ComputePolicy::Cpu.enter_legacy_scope();
            cpu_model.prefill(&tokens, &mut cpu_scratch).unwrap()
        };
        let actual = gpu_model.prefill(&tokens, &mut gpu_scratch).unwrap();
        assert_numerically_aligned(&actual, &expected);
    }
    assert!(context.compute_stats().since(before).submissions > 0);
}
