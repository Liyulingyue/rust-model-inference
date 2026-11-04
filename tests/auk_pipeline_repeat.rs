//! Does a shared AuK pipeline render the same audio twice?
//!
//! The HTTP server keeps one pipeline alive and renders request after
//! request. This test renders the same request twice in one process and
//! compares the results bit for bit, so a state leak between the renders
//! shows up as a difference.
//!
//! The CPU path passes. The Vulkan path currently fails intermittently (about
//! one render in six): the divergence starts at the third Euler step, so this
//! test is the regression guard for that defect, not a new contract.
//!
//! Run with:
//!   RMI_AUK_DIT=models/AuK-Base-and-Flash-GGUF/auk-base-f16.gguf \
//!   RMI_AUK_VAE=models/AuK-Base-and-Flash-GGUF/auk-vae-f32.gguf \
//!   RMI_AUK_TEXT=models/AuK-Base-and-Flash-GGUF/qwen2.5-omni-3b-q8_0.gguf \
//!   cargo test --features vulkan --profile release-fast --test auk_pipeline_repeat -- --ignored --nocapture

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rust_model_inference::core::tensor::TensorSource;
use rust_model_inference::format::ggufrs::{open_model_source, ComponentRole};
use rust_model_inference::models::diffusion::auk::{AukOptions, AukPipeline};

fn component(name: &str, fallback: &str) -> Option<PathBuf> {
    let path = std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(fallback));
    path.is_file().then_some(path)
}

fn open_source(path: &Path) -> Option<Arc<dyn TensorSource>> {
    Some(Arc::from(open_model_source(path, ComponentRole::Llm).ok()?))
}

#[test]
#[ignore = "needs the AuK weights and renders real audio"]
fn the_same_request_renders_the_same_audio_twice() {
    let Some(dit) = component(
        "RMI_AUK_DIT",
        "models/AuK-Base-and-Flash-GGUF/auk-base-f16.gguf",
    ) else {
        eprintln!("skipping: set RMI_AUK_DIT to the AuK DiT GGUF");
        return;
    };
    let Some(vae) = component(
        "RMI_AUK_VAE",
        "models/AuK-Base-and-Flash-GGUF/auk-vae-f32.gguf",
    ) else {
        eprintln!("skipping: set RMI_AUK_VAE to the AuK VAE GGUF");
        return;
    };
    let Some(text) = component(
        "RMI_AUK_TEXT",
        "models/AuK-Base-and-Flash-GGUF/qwen2.5-omni-3b-q8_0.gguf",
    ) else {
        eprintln!("skipping: set RMI_AUK_TEXT to the Qwen2.5-Omni GGUF");
        return;
    };
    let Some(dit_src) = open_source(&dit) else {
        return;
    };
    let Some(vae_src) = open_source(&vae) else {
        return;
    };
    let Some(text_src) = open_source(&text) else {
        return;
    };
    let pipeline = AukPipeline::load_with_audio_tower(dit_src, vae_src, Some(text_src), None, 8)
        .expect("load the AuK pipeline");
    // The Vulkan path needs the GPU enabled and the compute scope entered on
    // this thread, exactly like the CLI does.
    let scope = rust_model_inference::compute::ComputePolicy::Auto
        .configure_cli()
        .expect("enter the compute scope");
    let options = AukOptions {
        steps: 4,
        sample_rate: 24_000,
        duration_sec: 1,
        seed: 0,
        guidance_scale: 2.0,
        instruct: None,
        sway_sampling_coef: Some(-1.0),
        time_grid: None,
    };
    let prompt = "The next train leaves in ten minutes.";
    let first = pipeline
        .generate_audio(prompt, &options)
        .expect("first render");
    let _ = &scope;
    let second = pipeline
        .generate_audio(prompt, &options)
        .expect("second render");
    assert_eq!(
        first.samples.len(),
        second.samples.len(),
        "the two renders have a different length"
    );
    let mut worst = 0.0_f64;
    let mut differing = 0usize;
    for (left, right) in first.samples.iter().zip(second.samples.iter()) {
        let difference = (left - right).abs();
        if difference > 0.0 {
            differing += 1;
        }
        worst = worst.max(difference);
    }
    eprintln!("differing samples: {differing} of {}", first.samples.len());
    assert_eq!(
        differing, 0,
        "the shared pipeline changed the render: {differing} samples differ, worst {worst}"
    );
}
