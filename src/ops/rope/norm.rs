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

pub fn rope_norm(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half = head_dim / 2;
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
    let theta_scale = freq_base.powf(-2.0f32 / head_dim as f32);
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
            x[base + 2 * i] = x0.mul_add(c, x1 * -sn);
            x[base + 2 * i + 1] = x0.mul_add(sn, x1 * c);
        }
    }
}
