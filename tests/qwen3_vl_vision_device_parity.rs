//! CPU and device parity for the Qwen3-VL vision tower's attention.
//!
//! The ViT's attention is the whole cost of a vision layer: the patch
//! projections, the FFN and the norms together are a few percent of it.
//! The tower attends over every patch in both directions, which the
//! decoder-shaped device attention cannot express, so the layers run the
//! fused bidirectional kernel instead.
//!
//! The check is deliberately *not* end to end. The tower's patch
//! projection is a Q8_0 matmul, and the device's Q8_0 kernel does not
//! agree with the CPU's bit for bit (see TODO-046), so the two towers
//! start from different embeddings and drift apart over the 24 layers.
//! That difference is the projection's, not the attention's, so the
//! attention is compared against a scalar reference over the *same* q/k/v.
//!
//! Run with:
//!   RMI_QWEN3_VL_MMPROJ=/path/to/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf \
//!   cargo test --profile release-fast --features vulkan \
//!     --test qwen3_vl_vision_device_parity -- --nocapture

use rust_model_inference::compute::ComputePolicy;
use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::models::qwen3::vision::{VisionEncoder, VisionScratchpad};
use rust_model_inference::TensorSource;
use std::sync::Arc;

fn relative_l2(actual: &[f32], expected: &[f32]) -> f32 {
    let mut difference = 0.0f64;
    let mut norm = 0.0f64;
    for (a, b) in actual.iter().zip(expected.iter()) {
        difference += f64::from(*a - *b) * f64::from(*a - *b);
        norm += f64::from(*b) * f64::from(*b);
    }
    (difference / norm).sqrt() as f32
}

fn cosine(actual: &[f32], expected: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    for (a, b) in actual.iter().zip(expected.iter()) {
        dot += f64::from(*a) * f64::from(*b);
        actual_norm += f64::from(*a) * f64::from(*a);
        expected_norm += f64::from(*b) * f64::from(*b);
    }
    (dot / (actual_norm.sqrt() * expected_norm.sqrt())) as f32
}

/// A plain scalar attention over the ViT's `[part][head][token][dim]` q/k/v.
fn scalar_attention(
    attn_buf: &[f32],
    n_tokens: usize,
    n_head: usize,
    d_head: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_head * n_tokens * d_head];
    for head in 0..n_head {
        let q_base = head * n_tokens * d_head;
        let k_base = n_head * n_tokens * d_head + head * n_tokens * d_head;
        let v_base = 2 * n_head * n_tokens * d_head + head * n_tokens * d_head;
        for token in 0..n_tokens {
            let mut scores = vec![0.0f32; n_tokens];
            for key in 0..n_tokens {
                let mut sum = 0.0f32;
                for i in 0..d_head {
                    sum +=
                        attn_buf[q_base + token * d_head + i] * attn_buf[k_base + key * d_head + i];
                }
                scores[key] = sum * scale;
            }
            rust_model_inference::ops::softmax::softmax_inplace(&mut scores);
            for d in 0..d_head {
                let mut accumulator = 0.0f32;
                for key in 0..n_tokens {
                    accumulator += scores[key] * attn_buf[v_base + key * d_head + d];
                }
                out[head * n_tokens * d_head + token * d_head + d] = accumulator;
            }
        }
    }
    out
}

#[test]
fn vision_attention_device_matches_the_scalar_reference() {
    let Some(path) = std::env::var_os("RMI_QWEN3_VL_MMPROJ") else {
        eprintln!("RMI_QWEN3_VL_MMPROJ is unset; skipping the device parity check");
        return;
    };
    let loader = rust_model_inference::GGUFLoader::from_file(path).expect("load mmproj");
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let pool = Arc::new(ComputePool::new(8));
    let mut encoder =
        VisionEncoder::from_source(source.as_ref(), Arc::clone(&pool)).expect("encoder");
    // The DeepStack planes are precomputed from the vision tower's weights.
    encoder.precompute();

    // 448x448 is the Qwen3-VL default input: 32x32 patches of 14 pixels.
    let (image_w, image_h) = (448usize, 448usize);
    let pixels: Vec<f32> = (0..image_w * image_h * 3)
        .map(|i| (i as f32 * 0.0021).sin() * 0.5 + 0.5)
        .collect();

    rust_model_inference::ops::float::enable_gpu();

    let mut device_scratch = VisionScratchpad::new(&encoder.config);
    {
        let _scope = ComputePolicy::Vulkan.enter_legacy_scope();
        encoder
            .encode_image(&pixels, image_w, image_h, &mut device_scratch)
            .expect("vision tower on the device");
    }

    // The last layer's attention output, and the q/k/v it was computed from.
    let device_attention = device_scratch.attn_out_buf.clone();
    let n_head = encoder.config.n_head;
    let d_head = encoder.config.d_head();
    let n_tokens = device_scratch.attn_buf.len() / (3 * n_head * d_head);
    let reference = scalar_attention(
        &device_scratch.attn_buf,
        n_tokens,
        n_head,
        d_head,
        1.0 / (d_head as f32).sqrt(),
    );

    assert!(!device_attention.is_empty());
    assert_eq!(device_attention.len(), reference.len());
    let l2 = relative_l2(&device_attention, &reference);
    let similarity = cosine(&device_attention, &reference);
    eprintln!(
        "vision attention device vs scalar: n_tokens={n_tokens} n_head={n_head} d_head={d_head} \
         relative_l2={l2:.8} cosine={similarity:.8}"
    );
    assert!(
        l2 < 1e-3,
        "the device attention diverged from the scalar reference: relative_l2={l2}"
    );
    assert!(
        similarity > 0.9999,
        "the device attention is not aligned with the scalar reference: cosine={similarity}"
    );
}

#[test]
fn vision_tower_device_and_cpu_differ_only_by_the_q8_projection() {
    let Some(path) = std::env::var_os("RMI_QWEN3_VL_MMPROJ") else {
        eprintln!("RMI_QWEN3_VL_MMPROJ is unset; skipping the tower comparison");
        return;
    };
    let loader = rust_model_inference::GGUFLoader::from_file(path).expect("load mmproj");
    let source: Arc<dyn TensorSource> = Arc::new(loader);
    let pool = Arc::new(ComputePool::new(8));
    let mut encoder =
        VisionEncoder::from_source(source.as_ref(), Arc::clone(&pool)).expect("encoder");
    // The DeepStack planes are precomputed from the vision tower's weights.
    encoder.precompute();

    let (image_w, image_h) = (448usize, 448usize);
    let pixels: Vec<f32> = (0..image_w * image_h * 3)
        .map(|i| (i as f32 * 0.0021).sin() * 0.5 + 0.5)
        .collect();

    let mut cpu_scratch = VisionScratchpad::new(&encoder.config);
    {
        let _scope = ComputePolicy::Cpu.enter_legacy_scope();
        encoder
            .encode_image(&pixels, image_w, image_h, &mut cpu_scratch)
            .expect("vision tower on the CPU");
    }
    rust_model_inference::ops::float::enable_gpu();
    let mut device_scratch = VisionScratchpad::new(&encoder.config);
    {
        let _scope = ComputePolicy::Vulkan.enter_legacy_scope();
        encoder
            .encode_image(&pixels, image_w, image_h, &mut device_scratch)
            .expect("vision tower on the device");
    }

    // The patch embed is the Q8_0 matmul whose device kernel does not match
    // the CPU's bit for bit (TODO-046), so the two towers start apart. The
    // first frame's patch embed is a plain F16 dot and does match, which is
    // what pins the divergence to the second frame's projection.
    let first_frame = relative_l2(&device_scratch.patch_embd, &cpu_scratch.patch_embd);
    let second_frame = relative_l2(
        &device_scratch.project_concat_buf,
        &cpu_scratch.project_concat_buf,
    );
    eprintln!(
        "vision tower device vs cpu: first patch embed relative_l2={first_frame:.8} \
         second patch embed relative_l2={second_frame:.8}"
    );
    assert!(
        first_frame < 1e-6,
        "the F16 patch embed should agree bit for bit, got relative_l2={first_frame}"
    );
    assert!(
        second_frame > 1e-3,
        "the Q8_0 patch embed is expected to differ (TODO-046), got relative_l2={second_frame}"
    );
}
