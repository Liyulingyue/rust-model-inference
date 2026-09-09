//! Partial Neox RoPE.
//!
//! Applies Neox rotation to only the first `n_rot` dims of each head of
//! size `head_dim`. Used by architectures where `n_rot < head_dim`
//! (e.g. Spark 2.5's per-layer heterogeneous RoPE: full-attn layers use
//! `n_rot=64`, SWA layers use `n_rot=256`, both with `head_dim=256`).
//!
//! `x` must be laid out as `[n_heads * head_dim]`. Elements beyond
//! `n_rot` per head are left untouched, matching
//! `ggml_rope_ext(ctx, x, pos, nullptr, n_rot, ...)` in llama.cpp.

use super::neox::rope_sin_cos;

pub fn rope_neox_partial(x: &mut [f32], pos: usize, head_dim: usize, n_rot: usize, freq_base: f32) {
    assert!(
        n_rot <= head_dim,
        "n_rot ({n_rot}) must be <= head_dim ({head_dim})"
    );
    assert!(n_rot % 2 == 0, "n_rot ({n_rot}) must be even");
    let half = n_rot / 2;
    let n_heads = x.len() / head_dim;
    let theta_scale = freq_base.powf(-2.0f32 / n_rot as f32);
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut theta = pos as f32;
        for i in 0..half {
            let (cos_a, sin_a) = rope_sin_cos(theta);
            let x0 = x[base + i];
            let x1 = x[base + i + half];
            x[base + i] = x0.mul_add(cos_a, x1 * -sin_a);
            x[base + i + half] = x0.mul_add(sin_a, x1 * cos_a);
            theta *= theta_scale;
        }
    }
}
