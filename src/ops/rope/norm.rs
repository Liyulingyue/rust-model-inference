//! GGML "normal" (interleaved-pair) RoPE.
//!
//! Used by the classic `llama` GGUF architecture. HF `rotate_half`-style
//! weights are permuted to this layout by the llama.cpp llama-arch
//! converter, so a `llama`-arch GGUF must use this variant, not
//! [`super::rope_neox_inplace`].
//!
//! Inner loop stays scalar because the rotation touches interleaved
//! `(x[2i], x[2i+1])` pairs, which AVX2 can only handle with shuffles
//! that cost more than they save at typical `half ≤ 128`.

use super::neox::rope_sin_cos;

/// Interleaved-pair RoPE rotating **all** `head_dim` lanes of every head.
///
/// Equivalent to [`rope_norm_nrot`] with `n_rot == head_dim`, which is what
/// every caller without a `rope.dimension_count` key wants.
pub fn rope_norm(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    rope_norm_impl(x, pos, head_dim, head_dim, freq_base);
}

/// Interleaved-pair RoPE that rotates only the first `n_rot` lanes of each
/// head, leaving the tail `[n_rot, head_dim)` untouched.
///
/// Mirrors `ggml_compute_forward_rope_f32` (`ggml-cpu/ops.cpp:6098`): `n_dims`
/// is `n_rot`, `theta_scale` is `powf(freq_base, -2.0f/n_dims)` — computed from
/// `n_rot`, **not** from `head_dim` — and the channels past `n_dims` are copied
/// through unchanged (`ops.cpp:6224-6234`).
///
/// `n_rot` must be even and `<= head_dim`; anything else falls back to
/// rotating the whole head, which is what the callers without a
/// `rope.dimension_count` key did before this function existed.
pub fn rope_norm_nrot(x: &mut [f32], pos: usize, head_dim: usize, n_rot: usize, freq_base: f32) {
    if n_rot == 0 || n_rot > head_dim || n_rot % 2 != 0 {
        rope_norm_impl(x, pos, head_dim, head_dim, freq_base);
        return;
    }
    rope_norm_impl(x, pos, head_dim, n_rot, freq_base);
}

fn rope_norm_impl(x: &mut [f32], pos: usize, head_dim: usize, n_rot: usize, freq_base: f32) {
    let half = n_rot / 2;
    let n_heads = x.len() / head_dim;
    if half == 0 || n_heads == 0 {
        return;
    }
    // Cache sin/cos table once across all heads (same for each head at this pos).
    // `rope_norm` uses the recurrence `theta *= theta_scale` (matches ggml's
    // ROPE_TYPE_NORM); we keep it here for bit-exact parity with the original
    // implementation. Reduces `sin_cos` calls from `n_heads × half` to `half`.
    let mut cos_table = vec![0.0f32; half];
    let mut sin_table = vec![0.0f32; half];
    let theta_scale = freq_base.powf(-2.0f32 / n_rot as f32);
    let mut theta = pos as f32;
    for i in 0..half {
        let (c, s) = rope_sin_cos(theta);
        cos_table[i] = c;
        sin_table[i] = s;
        theta *= theta_scale;
    }
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let x0 = x[base + 2 * i];
            let x1 = x[base + 2 * i + 1];
            let c = cos_table[i];
            let sn = sin_table[i];
            if crate::ops::scalar_mode() {
                x[base + 2 * i] = x0 * c - x1 * sn;
                x[base + 2 * i + 1] = x0 * sn + x1 * c;
            } else {
                x[base + 2 * i] = x0.mul_add(c, x1 * -sn);
                x[base + 2 * i + 1] = x0.mul_add(sn, x1 * c);
            }
        }
    }
}
