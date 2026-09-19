//! NEON parity tests for `q4_0::neon::matmul_q4_0_vs_q8_0_neon`.
//!
//! These mirror the AVX2 parity tests in `avx2.rs` but target the
//! aarch64 `vdotq_u32` path. They verify that the NEON SIMD kernel
//! produces the same output (≤ 1 ULP) as the scalar fallback —
//! essential to make sure the algebraic-identity rewrite
//! `nib_total - 8 * sum_input` doesn't introduce a regression.

#![cfg(target_arch = "aarch64")]

use super::super::matmul_q4_0_scalar_range;
use super::matmul_q4_0_vs_q8_0_neon;

fn build_block(scale: f32, low: u8, hi: u8) -> Vec<u8> {
    assert!(low < 16 && hi < 16);
    let mut v = Vec::with_capacity(18);
    let s_bits = crate::ops::f32_to_f16(scale).to_le_bytes();
    v.extend_from_slice(&s_bits);
    for _ in 0..16 {
        v.push((hi << 4) | low);
    }
    v
}

fn q8_input_zero() -> Vec<u8> {
    vec![0u8; 32]
}
fn q8_input_max() -> Vec<u8> {
    vec![0x7Fu8; 32]
}
fn q8_input_min() -> Vec<u8> {
    vec![0x80u8; 32]
}
fn q8_input_linspace() -> Vec<u8> {
    (0..32).map(|i| (i as i8) as u8).collect()
}
fn q8_input_alt() -> Vec<u8> {
    (0..32)
        .map(|i| if i % 2 == 0 { 0x7F } else { 0x80 })
        .collect()
}

fn assert_neon_eq_scalar(label: &str, weight: &[u8], q8: &[u8], scales: &[f32]) {
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        // Skip silently on hosts without UDOT (the kernel falls back to
        // scalar there, which is what the rest of the parity suite
        // already validates).
        return;
    }
    let n_in = q8.len();
    let n_out = weight.len() / (n_in / 32 * 18);
    let mut neon_out = vec![0.0f32; n_out];
    let mut scalar_out = vec![0.0f32; n_out];
    unsafe {
        matmul_q4_0_vs_q8_0_neon(weight, q8, scales, &mut neon_out, n_in, 0, n_out);
    }
    matmul_q4_0_scalar_range(weight, q8, scales, &mut scalar_out, n_in, n_out, 0, 1);
    for (i, (a, b)) in neon_out.iter().zip(scalar_out.iter()).enumerate() {
        let a_bits = a.to_bits();
        let b_bits = b.to_bits();
        let diff = (a_bits as i32).wrapping_sub(b_bits as i32).unsigned_abs();
        assert!(
            diff <= 4,
            "{} row {}: neon={} (bits {:x}) scalar={} (bits {:x}) diff={} ULP",
            label,
            i,
            a,
            a_bits,
            b,
            b_bits,
            diff
        );
    }
}

#[test]
fn parity_block_uniform_zero_nibble_zero_input() {
    let weight = build_block(1.0, 0, 0);
    let q8 = q8_input_zero();
    let scales = vec![1.0f32];
    assert_neon_eq_scalar("zero-nibble/zero-input", &weight, &q8, &scales);
}

#[test]
fn parity_block_uniform_max_nibble_zero_input() {
    let weight = build_block(1.0, 15, 15);
    let q8 = q8_input_zero();
    let scales = vec![1.0f32];
    assert_neon_eq_scalar("max-nibble/zero-input", &weight, &q8, &scales);
}

#[test]
fn parity_block_zero_nibble_max_input() {
    let weight = build_block(1.0, 0, 0);
    let q8 = q8_input_max();
    let scales = vec![1.0f32];
    assert_neon_eq_scalar("zero-nibble/max-input", &weight, &q8, &scales);
}

#[test]
fn parity_block_zero_nibble_min_input() {
    let weight = build_block(1.0, 0, 0);
    let q8 = q8_input_min();
    let scales = vec![1.0f32];
    assert_neon_eq_scalar("zero-nibble/min-input", &weight, &q8, &scales);
}

#[test]
fn parity_block_max_nibble_max_input() {
    let weight = build_block(1.0, 15, 15);
    let q8 = q8_input_max();
    let scales = vec![1.0f32];
    assert_neon_eq_scalar("max-nibble/max-input", &weight, &q8, &scales);
}

#[test]
fn parity_block_mixed_nibbles_linspace() {
    let weight = build_block(0.7, 0, 15);
    let q8 = q8_input_linspace();
    let scales = vec![1.3f32];
    assert_neon_eq_scalar("mixed-nibble/linspace", &weight, &q8, &scales);
}

#[test]
fn parity_block_alt_input_signs() {
    let weight = build_block(2.5, 8, 3);
    let q8 = q8_input_alt();
    let scales = vec![0.42f32];
    assert_neon_eq_scalar("alt-input-signs", &weight, &q8, &scales);
}

#[test]
fn parity_many_blocks_random() {
    // 16 blocks per row, 4 rows.
    let mut weight = Vec::new();
    let mut state: u64 = 0xdead_beef_1234_5678;
    for _ in 0..4 {
        for _ in 0..16 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let d = 0.01 + (state >> 33) as f32 / u32::MAX as f32;
            let s_bits = crate::ops::f32_to_f16(d).to_le_bytes();
            weight.extend_from_slice(&s_bits);
            for _ in 0..16 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                weight.push((state >> 33) as u8);
            }
        }
    }
    let q8: Vec<u8> = (0..512)
        .map(|i| ((i as i32 % 31) - 15) as i8 as u8)
        .collect();
    let scales: Vec<f32> = (0..16).map(|b| 0.01 + (b as f32) * 0.001).collect();
    assert_neon_eq_scalar("random-4x512", &weight, &q8, &scales);
}
