//! Hardware tests for the Nemotron-3 Nano 4B (`nemotron_h`) trunk.
//!
//! All 42 layers run on the device: the 17 FFN layers, the 4 attention
//! layers and the 21 Mamba-2 SSM layers. The device path is bit-exact
//! against the CPU path, so the gate below is a floor rather than the
//! expected value -- the test prints both numbers so a regression shows
//! how far it moved.
//!
//! The gate is the same double-1% gate the Dory port uses:
//! `relative_l2 <= 0.01` and `scaled_max_abs <= 0.01` against the CPU
//! path.
//!
//! The prefill lengths matter: the Mamba-2 recurrence only starts to
//! differ from the host's when the scan state is non-zero, so a
//! single-token prefill cannot see a broken recurrence. The test runs one,
//! two and three tokens for that reason.
//!
//! The GGUF is not checked in; set `RMI_NEMOTRON_Q8_MODEL` to the
//! `Nemotron-3-Nano-4B-Q8_0.gguf` path.

use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::models::nemotron_h::trunk::{
    NemotronConfig, NemotronModel, NemotronScratch,
};
use rust_model_inference::GGUFLoader;
use std::path::PathBuf;
use std::sync::Arc;

fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("RMI_NEMOTRON_Q8_MODEL")?);
    path.exists().then_some(path)
}

fn source() -> Option<Arc<dyn TensorSource>> {
    let path = model_path()?;
    Some(Arc::new(
        GGUFLoader::from_file(path).expect("open Nemotron-H GGUF"),
    ))
}

#[test]
fn nemotron_h_nano_4b_contract_and_tokenizer() {
    let Some(source) = source() else { return };
    let config = NemotronConfig::from_source(source.as_ref()).expect("Nemotron-H config");
    assert_eq!(config.architecture, "nemotron_h");
    assert_eq!(config.n_embd, 3136);
    assert_eq!(config.n_layer, 42);
    assert_eq!(config.n_head, 40);
    assert_eq!(config.vocab_size, 131_072);
    assert_eq!(config.ssm_conv_kernel, 4);
    assert_eq!(config.ssm_state_size, 128);
    assert_eq!(config.ssm_group_count, 8);
    assert_eq!(config.ssm_inner_size, 7680);
    assert_eq!(config.ssm_time_step_rank, 96);
    assert_eq!(config.rope_dim, 78);
    assert_eq!(config.attention_layers.len(), 4);
    assert_eq!(config.ffn_layers.len(), 17);
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
        .expect("Nemotron-H GPT-2 tokenizer");
    assert_eq!(
        tokenizer.encode(
            "Hello",
            EncodeOptions {
                add_special: false,
                parse_special: true,
            }
        ),
        vec![22177]
    );
}

#[test]
fn nemotron_h_nano_4b_single_token_logits_are_finite() {
    let Some(source) = source() else { return };
    let mut model = NemotronModel::from_source(source, 1).expect("load Nemotron-H model");
    let mut scratch = NemotronScratch::new(&model.config, 2);
    let logits = model
        .prefill(&[1], &mut scratch)
        .expect("Nemotron-H forward");
    assert_eq!(logits.len(), 131_072);
    assert!(logits.iter().all(|value| value.is_finite()));
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
    eprintln!("GPU relative_l2={l2} scaled_max_abs={}", max_abs / scale);
    assert!(l2 <= 0.01);
    assert!(max_abs / scale <= 0.01);
    let best = |v: &[f32]| (1..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
    assert_eq!(best(actual), best(expected));
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires the real Q8_0 GGUF and a hardware Vulkan device"]
fn nemotron_h_nano_4b_vulkan_matches_cpu() {
    use rust_model_inference::compute::ComputePolicy;
    assert!(std::env::var_os("RMI_PARITY_TRACE").is_none());
    assert!(std::env::var_os("RMI_SCALAR").is_none());
    rust_model_inference::ops::enable_gpu();
    let source = source().expect("set RMI_NEMOTRON_Q8_MODEL to the real GGUF");
    let mut cpu_model = NemotronModel::from_source(source.clone(), 4).unwrap();
    let mut gpu_model = NemotronModel::from_source(source, 4).unwrap();
    gpu_model
        .attach_gpu(8, 8, ComputePolicy::Vulkan)
        .expect("Nemotron-H Vulkan session");
    assert!(gpu_model.gpu_available());
    let context = rust_model_inference::ops::get_vulkan_context().unwrap();
    let before = context.compute_stats();
    for tokens in [vec![22177], vec![22177, 1897], vec![22177, 1897, 1429]] {
        let mut cpu_scratch = NemotronScratch::new(&cpu_model.config, tokens.len());
        let mut gpu_scratch = NemotronScratch::new(&gpu_model.config, tokens.len());
        let expected = cpu_model.prefill(&tokens, &mut cpu_scratch).unwrap();
        let actual = gpu_model.prefill(&tokens, &mut gpu_scratch).unwrap();
        assert_numerically_aligned(&actual, &expected);
    }
    assert!(context.compute_stats().since(before).submissions > 0);
}

/// Attribute a GPU/CPU difference to one branch.
///
/// With `RMI_NH_CPU_ATTENTION` or `RMI_NH_CPU_FFN` set, the named branch
/// runs on the host instead of the device, so the remaining difference is
/// the other branch's contribution.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires the real Q8_0 GGUF and a hardware Vulkan device"]
fn nemotron_h_nano_4b_vulkan_branch_attribution() {
    use rust_model_inference::compute::ComputePolicy;
    rust_model_inference::ops::enable_gpu();
    let source = source().expect("set RMI_NEMOTRON_Q8_MODEL to the real GGUF");
    let mut cpu_model = NemotronModel::from_source(source.clone(), 4).unwrap();
    let mut gpu_model = NemotronModel::from_source(source, 4).unwrap();
    gpu_model
        .attach_gpu(8, 8, ComputePolicy::Vulkan)
        .expect("Nemotron-H Vulkan session");
    let tokens = vec![22177];
    let mut cpu_scratch = NemotronScratch::new(&cpu_model.config, tokens.len());
    let mut gpu_scratch = NemotronScratch::new(&gpu_model.config, tokens.len());
    let expected = cpu_model.prefill(&tokens, &mut cpu_scratch).unwrap();
    let actual = gpu_model.prefill(&tokens, &mut gpu_scratch).unwrap();
    eprintln!(
        "attribution cpu_attention={} cpu_ffn={}",
        std::env::var_os("RMI_NH_CPU_ATTENTION").is_some(),
        std::env::var_os("RMI_NH_CPU_FFN").is_some()
    );
    let (mut diff, mut reference) = (0.0f64, 0.0f64);
    for (&a, &b) in actual.iter().zip(expected.iter()) {
        let delta = f64::from(a) - f64::from(b);
        diff += delta * delta;
        reference += f64::from(b) * f64::from(b);
    }
    eprintln!(
        "attribution relative_l2={:.6}",
        (diff / reference.max(f64::MIN_POSITIVE)).sqrt()
    );
}
