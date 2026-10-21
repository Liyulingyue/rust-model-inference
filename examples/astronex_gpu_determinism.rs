#![cfg(feature = "vulkan")]
//! Run-to-run determinism probe for the Astronex GPU operators.
//!
//! The Astronex GPU path produced two different videos from the same seed on
//! the same binary, and the divergence was already present in the sampled
//! latent, so it is not the VAE. This probe runs the same operator twice with
//! identical inputs and compares the results bit for bit.
use rust_model_inference::models::diffusion::dreamx::kernels::{
    attention_scalar_f32, AttentionSpec,
};
use rust_model_inference::ops::float::enable_gpu;
use rust_model_inference::ops::get_vulkan_context;

fn synth(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        })
        .collect()
}

fn compare(label: &str, a: &[f32], b: &[f32]) {
    let same = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    let max_diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!(
        "{label}: 逐位相同={same} 最大绝对差={max_diff:.6e} ({} 个元素)",
        a.len()
    );
}

fn probe_attention(query_tokens: usize, key_tokens: usize, heads: usize, head_dim: usize) {
    let spec = AttentionSpec {
        query_tokens,
        key_tokens,
        query_heads: heads,
        key_value_heads: heads,
        head_dim,
        causal: false,
        scale: 1.0 / (head_dim as f32).sqrt(),
    };
    let query = synth(query_tokens * heads * head_dim, 7);
    let key = synth(key_tokens * heads * head_dim, 11);
    let value = synth(key_tokens * heads * head_dim, 13);
    let first = attention_scalar_f32(&query, &key, &value, spec).expect("attention run 1");
    let second = attention_scalar_f32(&query, &key, &value, spec).expect("attention run 2");
    compare(
        &format!("attention {query_tokens}x{key_tokens} heads={heads} dim={head_dim}"),
        &first,
        &second,
    );
}

fn main() {
    enable_gpu();
    let Some(context) = get_vulkan_context() else {
        println!("没有可用的 Vulkan 上下文");
        return;
    };
    println!("设备: {}", context.device_name());
    // Warm up the shader JIT so the first call is not charged to compilation.
    probe_attention(64, 128, 4, 64);
    for (q, k, heads, dim) in [
        (128usize, 256usize, 8usize, 64usize),
        (512, 1024, 8, 64),
        (1024, 2048, 24, 128),
        (160, 320, 12, 128),
    ] {
        probe_attention(q, k, heads, dim);
    }
}
