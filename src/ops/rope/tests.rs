//! Bit-exact parity tests for the RoPE variants.

use super::{
    neox::{rope_neox_inplace_scalar, rope_sin_cos},
    rope_mrope, rope_neox_inplace, rope_neox_sleef, rope_norm, rope_sin_cos_sleef,
    rope_sin_cos_sleef_table_with_threads, rope_vision,
};

#[test]
fn sleef_rope_sin_cos_matches_torch_arm_bits() {
    let expected = [
        (1.0f32, 0x3f0a5141u32, 0x3f576aa4u32),
        (126.0f32, 0x3f71a8f2u32, 0x3ea8f48fu32),
        (2048.0f32, 0x3f7321cau32, 0xbea04902u32),
    ];
    for (theta, cos_bits, sin_bits) in expected {
        let (cos, sin) = rope_sin_cos_sleef(theta);
        assert_eq!(cos.to_bits(), cos_bits, "cos({theta})");
        assert_eq!(sin.to_bits(), sin_bits, "sin({theta})");
    }
}

#[test]
fn sleef_rope_sin_cos_matches_torch_arm_low_frequency_bits() {
    let expected = [
        (0x3e12bd91u32, 0x3f7d6040u32, 0x3e123d21u32),
        (0x3dec7fd6, 0x3f7e4b84u32, 0x3debf95du32),
        (0x3c301052, 0x3f7ffc37u32, 0x3c300f74u32),
        (0x39b229fb, 0x3f7fffffu32, 0x39b229fbu32),
    ];
    for (theta_bits, cos_bits, sin_bits) in expected {
        let theta = f32::from_bits(theta_bits);
        let (cos, sin) = rope_sin_cos_sleef(theta);
        assert_eq!(cos.to_bits(), cos_bits, "cos(theta={theta})");
        assert_eq!(sin.to_bits(), sin_bits, "sin(theta={theta})");
    }
}

#[test]
fn sleef_rope_neox_matches_torch_vector_pow_bits() {
    let mut x = [0.0f32; 128];
    x[37] = 1.0;
    rope_neox_sleef(&mut x, 10, 128, 1_000_000.0);
    assert_eq!(x[37].to_bits(), 0x3f7fff9f);
    assert_eq!(x[101].to_bits(), 0x3b5eb45e);
}

#[test]
fn sleef_rope_table_matches_torch_openmp_chunk_bits() {
    let positions = (0..185).collect::<Vec<_>>();
    let (cos, sin) = rope_sin_cos_sleef_table_with_threads(&positions, 64, 10_000.0, 12);
    let target = 165 * 64 + 30;
    assert_eq!(cos[target].to_bits(), 0x3f7fe3ca);
    assert_eq!(sin[target].to_bits(), 0x3cf054fd);

    let (cos, _) = rope_sin_cos_sleef_table_with_threads(&positions, 64, 10_000.0, 4);
    assert_eq!(cos[target].to_bits(), 0x3f7fe3cb);
}

#[test]
fn vision_rope_rotates_both_halves_with_independent_axes() {
    let mut values = [0.0f32; 64];
    values[0] = 1.0;
    values[32] = 2.0;
    values[31] = 3.0;
    values[63] = 4.0;

    rope_vision(&mut values, [1, 2, 1, 2], [16, 16, 16, 16], 64, 1.0, 32);

    let (sin_h, cos_h) = 1.0f32.sin_cos();
    let (sin_w, cos_w) = 2.0f32.sin_cos();
    assert!((values[0] - (cos_h - 2.0 * sin_h)).abs() < 1e-6);
    assert!((values[32] - (sin_h + 2.0 * cos_h)).abs() < 1e-6);
    assert!((values[31] - (3.0 * cos_w - 4.0 * sin_w)).abs() < 1e-6);
    assert!((values[63] - (3.0 * sin_w + 4.0 * cos_w)).abs() < 1e-6);
}

/// SIMD path must produce the same result as the scalar fallback.
/// Compares public `rope_neox_inplace` against the explicit scalar helper used
/// when SIMD is unavailable. Catches tail-handling, cache wiring, and
/// instruction-order bugs across the three paths.
#[test]
fn rope_neox_inplace_simd_matches_scalar_fallback() {
    // Vary n_heads × head_dim to exercise SIMD tail loops and edge cases.
    for &(head_dim, n_heads, pos, freq_base) in &[
        (64usize, 4usize, 0usize, 10_000.0f32),
        (128, 8, 1, 1_000_000.0),
        (128, 16, 7, 500_000.0),
        (256, 4, 1024, 50_000.0),
        // head_dim not a multiple of 16 → SIMD tail must fall through to scalar.
        (96, 2, 3, 100_000.0),
        (80, 6, 5, 200_000.0),
        (128, 1, 0, 10_000.0),
    ] {
        let mut a = vec![0.0f32; n_heads * head_dim];
        let mut b = vec![0.0f32; n_heads * head_dim];
        for (i, slot) in a.iter_mut().enumerate() {
            *slot = ((i as f32) * 0.0731).sin() * 3.5 - ((i * 31 % 97) as f32) * 0.013;
        }
        b.copy_from_slice(&a);

        rope_neox_inplace(&mut a, pos, head_dim, freq_base);

        // Scalar reference uses the same formula as the public function's
        // table build, then a plain scalar per-head rotation — the same
        // shape as the AVX2/NEON tail loop, so SIMD-vs-scalar diffs are
        // caught bit-for-bit.
        let half = head_dim / 2;
        let pos_f = pos as f32;
        let mut cos_table = vec![0.0f32; half];
        let mut sin_table = vec![0.0f32; half];
        for i in 0..half {
            let inv_freq = 1.0f32 / freq_base.powf((2 * i) as f32 / head_dim as f32);
            let theta = pos_f * inv_freq;
            let (c, s) = rope_sin_cos(theta);
            cos_table[i] = c;
            sin_table[i] = s;
        }
        rope_neox_inplace_scalar(&mut b, n_heads, head_dim, &cos_table, &sin_table);

        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "head_dim={head_dim} n_heads={n_heads} idx={i}"
            );
        }
    }
}

/// Same for `rope_norm`: SIMD is not used (interleaved-pair layout doesn't
/// vectorize cleanly), but the cached table must produce the same output
/// as the original per-head recurrence loop.
#[test]
fn rope_norm_cached_table_matches_per_head_recurrence() {
    for &(head_dim, n_heads, pos, freq_base) in &[
        (64usize, 4usize, 0usize, 10_000.0f32),
        (128, 8, 1, 1_000_000.0),
        (128, 16, 7, 500_000.0),
        (256, 4, 1024, 50_000.0),
    ] {
        let mut a = vec![0.0f32; n_heads * head_dim];
        let mut b = vec![0.0f32; n_heads * head_dim];
        for (i, slot) in a.iter_mut().enumerate() {
            *slot = ((i as f32) * 0.0731).sin() * 3.5 - ((i * 31 % 97) as f32) * 0.013;
        }
        b.copy_from_slice(&a);

        rope_norm(&mut a, pos, head_dim, freq_base);

        // Reference: original per-head loop with `theta *= theta_scale`.
        let half = head_dim / 2;
        let theta_scale = freq_base.powf(-2.0f32 / head_dim as f32);
        for h in 0..n_heads {
            let base = h * head_dim;
            let mut theta = pos as f32;
            for i in 0..half {
                let (c, s) = rope_sin_cos(theta);
                let x0 = b[base + 2 * i];
                let x1 = b[base + 2 * i + 1];
                b[base + 2 * i] = x0.mul_add(c, x1 * -s);
                b[base + 2 * i + 1] = x0.mul_add(s, x1 * c);
                theta *= theta_scale;
            }
        }

        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "head_dim={head_dim} n_heads={n_heads} idx={i}"
            );
        }
    }
}

/// Keep the test pointer referenced so `rope_mrope` doesn't get flagged
/// unused when no test in the file uses it directly.
#[allow(dead_code)]
fn _exercise_rope_mrope() {
    let mut values = [0.0f32; 64];
    rope_mrope(&mut values, [1, 2, 1, 2], [16, 16, 16, 16], 64, 1.0);
}
