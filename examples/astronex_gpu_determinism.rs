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

/// Attention correctness against a scalar CPU reference.
///
/// Mage-Flow's DiT uses self-attention with 24 heads of dim 128, while its
/// VAE uses windowed attention with a single head whose head_dim is the
/// channel count, so both shapes have to be checked.
fn probe_correctness() {
    use rust_model_inference::models::diffusion::dreamx::kernels::{
        attention_scalar_f32, AttentionSpec,
    };
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 1024.0 - 8.0
    };
    for (queries, keys, heads, dim) in [
        (1024usize, 1024usize, 1usize, 384usize),
        (1024, 1024, 1, 128),
        (1024, 1024, 1, 96),
        (1024, 1024, 1, 64),
        (1024, 1024, 1, 32),
        (256, 256, 1, 384),
        (64, 64, 1, 384),
    ] {
        let q: Vec<f32> = (0..queries * heads * dim).map(|_| next() / 8.0).collect();
        let k: Vec<f32> = (0..keys * heads * dim).map(|_| next() / 8.0).collect();
        let v: Vec<f32> = (0..keys * heads * dim).map(|_| next() / 8.0).collect();
        let spec = AttentionSpec {
            query_tokens: queries,
            key_tokens: keys,
            query_heads: heads,
            key_value_heads: heads,
            head_dim: dim,
            causal: false,
            scale: 1.0 / (dim as f32).sqrt(),
        };
        let got = attention_scalar_f32(&q, &k, &v, spec).expect("attention");
        let group = spec.query_heads / spec.key_value_heads;
        let scale = 1.0 / (dim as f32).sqrt();
        let mut worst = 0.0f32;
        let mut first = None;
        for qt in (0..queries).step_by((queries / 8).max(1)) {
            let qh = 0usize;
            let qs = (qt * spec.query_heads + qh) * dim;
            let kh = qh / group;
            let mut m = f32::NEG_INFINITY;
            let mut scores = Vec::with_capacity(keys);
            for kt in 0..keys {
                let ks = (kt * spec.key_value_heads + kh) * dim;
                let mut acc = 0.0f32;
                for d in 0..dim {
                    acc += q[qs + d] * k[ks + d];
                }
                let s = acc * scale;
                scores.push(s);
                if s > m {
                    m = s;
                }
            }
            let mut denom = 0.0f32;
            let mut probs = Vec::with_capacity(scores.len());
            for s in &scores {
                let e = (s - m).exp();
                denom += e;
                probs.push(e);
            }
            for d in 0..dim {
                let mut o = 0.0f32;
                for (kt, p) in probs.iter().enumerate() {
                    o += p / denom * v[(kt * spec.key_value_heads + kh) * dim + d];
                }
                let got_v = got[qs + d];
                let rel = (o - got_v).abs() / o.abs().max(1e-3);
                if rel > worst {
                    worst = rel;
                }
                if first.is_none() && rel > 1e-2 {
                    first = Some((qt, qh, d, o, got_v));
                }
            }
        }
        match first {
            None => println!(
                "attention q={queries} k={keys} heads={heads} dim={dim}: OK 最大相对误差={worst:.3e}"
            ),
            Some((qt, qh, d, cpu, gpu)) => println!(
                "attention q={queries} k={keys} heads={heads} dim={dim}: 首个超阈值 q={qt} h={qh} d={d} cpu={cpu:.5e} gpu={gpu:.5e} rel={:.2e}",
                (cpu - gpu).abs() / cpu.abs().max(1e-3)
            ),
        }
    }
}

/// Runtime rebuild interference: the attention runtime is a global that is
/// destroyed and recreated whenever a larger shape shows up, so a shape can
/// give different answers depending on what ran before it.
fn probe_shape_interference() {
    use rust_model_inference::models::diffusion::dreamx::kernels::{
        attention_scalar_f32, AttentionSpec,
    };
    let mut state: u64 = 0x243F6A8885A308D3;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 1024.0 - 8.0
    };
    let mk = |q: usize, k: usize, dim: usize, next: &mut dyn FnMut() -> f32| {
        let query: Vec<f32> = (0..q * dim).map(|_| next() / 8.0).collect();
        let key: Vec<f32> = (0..k * dim).map(|_| next() / 8.0).collect();
        let value: Vec<f32> = (0..k * dim).map(|_| next() / 8.0).collect();
        (query, key, value)
    };
    let dim = 128usize;
    let (q1, k1, v1) = mk(1024, 1024, dim, &mut next);
    let (q2, k2, v2) = mk(512, 512, dim, &mut next);
    let spec = |n: usize| AttentionSpec {
        query_tokens: n,
        key_tokens: n,
        query_heads: 1,
        key_value_heads: 1,
        head_dim: dim,
        causal: false,
        scale: 1.0 / (dim as f32).sqrt(),
    };
    let a1 = attention_scalar_f32(&q1, &k1, &v1, spec(1024)).expect("a1");
    let a2 = attention_scalar_f32(&q2, &k2, &v2, spec(512)).expect("a2");
    // Interleave: a bigger shape, then a smaller one, then the big one again.
    let _ = attention_scalar_f32(&q2, &k2, &v2, spec(512)).expect("a2b");
    let _ = attention_scalar_f32(&q1, &k1, &v1, spec(1024)).expect("a1b");
    let a1b = attention_scalar_f32(&q1, &k1, &v1, spec(1024)).expect("a1c");
    let a1c = attention_scalar_f32(&q1, &k1, &v1, spec(1024)).expect("a1d");
    let same = |x: &[f32], y: &[f32]| {
        x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
    };
    println!(
        "shape 干扰: 大形状首次 vs 交错后 逐位相同={} 交错后两次之间 逐位相同={}",
        same(&a1, &a1b),
        same(&a1b, &a1c)
    );
    let d: f32 = a1
        .iter()
        .zip(a1b.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f32::max);
    println!("  大形状首次 vs 交错后 最大绝对差={d:.6e}");
}

fn main() {
    enable_gpu();
    let Some(context) = get_vulkan_context() else {
        println!("没有可用的 Vulkan 上下文");
        return;
    };
    println!("设备: {}", context.device_name());
    probe_correctness();
    probe_shape_interference();
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
