//! NEON parity tests for `q4_1::neon::matmul_q4_1_vs_q8_0_neon`.
//!
//! Mirror the AVX2 parity tests in `avx2.rs`. Skips silently on hosts
//! without ARMv8.4-A `dotprod` (UDOT) since the kernel falls back to
//! the scalar baseline there.

#![cfg(target_arch = "aarch64")]

use super::neon::matmul_q4_1_vs_q8_0_neon;
use super::scalar::matmul_q4_1_scalar_range;

fn build_block(scale: f32, min: f32, nibble: u8) -> Vec<u8> {
    assert!(nibble < 16);
    let mut v = Vec::with_capacity(20);
    v.extend_from_slice(&crate::ops::f32_to_f16(scale).to_le_bytes());
    v.extend_from_slice(&crate::ops::f32_to_f16(min).to_le_bytes());
    let packed = nibble | (nibble << 4);
    for _ in 0..16 {
        v.push(packed);
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

fn assert_neon_eq_scalar(
    label: &str,
    weight: &[u8],
    q8: &[u8],
    scales: &[f32],
    input_sums: Option<&[f32]>,
) {
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        return;
    }
    let n_in = q8.len();
    let n_out = weight.len() / (n_in / 32 * 20);
    let mut neon_out = vec![0.0f32; n_out];
    let mut scalar_out = vec![0.0f32; n_out];
    unsafe {
        matmul_q4_1_vs_q8_0_neon(
            weight,
            q8,
            scales,
            input_sums,
            &mut neon_out,
            n_in,
            0,
            n_out,
        );
    }
    matmul_q4_1_scalar_range(
        weight,
        q8,
        scales,
        input_sums,
        &mut scalar_out,
        n_in,
        n_out,
        0,
        1,
    );
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
fn parity_block_min_contribution_only() {
    // min=1.0 (non-zero), dot=0, all min contribution
    let weight = build_block(0.0, 1.0, 0);
    let q8 = q8_input_zero();
    let scales = vec![1.0f32];
    let sums: Vec<f32> = q8
        .chunks_exact(32)
        .map(|c| c.iter().map(|&b| b as i8 as f32).sum())
        .collect();
    assert_neon_eq_scalar("min-only", &weight, &q8, &scales, Some(&sums));
}

#[test]
fn parity_block_dot_product_only() {
    // min=0, all dot contribution
    let weight = build_block(1.0, 0.0, 1);
    let q8 = q8_input_max();
    let scales = vec![1.0f32];
    let sums: Vec<f32> = q8
        .chunks_exact(32)
        .map(|c| c.iter().map(|&b| b as i8 as f32).sum())
        .collect();
    assert_neon_eq_scalar("dot-only", &weight, &q8, &scales, Some(&sums));
}

#[test]
fn parity_block_zero_nibble_min_input() {
    let weight = build_block(0.0, 1.0, 0);
    let q8 = q8_input_min();
    let scales = vec![1.0f32];
    let sums: Vec<f32> = q8
        .chunks_exact(32)
        .map(|c| c.iter().map(|&b| b as i8 as f32).sum())
        .collect();
    assert_neon_eq_scalar("zero-nibble/min-input", &weight, &q8, &scales, Some(&sums));
}

#[test]
fn parity_block_random() {
    let mut weight = Vec::new();
    let mut state: u64 = 0xfeed_face_dead_beef;
    for _ in 0..4 {
        for _ in 0..16 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let d = 0.01 + (state >> 33) as f32 / u32::MAX as f32;
            let m = 0.5 + (state >> 33) as f32 / u32::MAX as f32;
            let s_bits = crate::ops::f32_to_f16(d).to_le_bytes();
            let m_bits = crate::ops::f32_to_f16(m).to_le_bytes();
            weight.extend_from_slice(&s_bits);
            weight.extend_from_slice(&m_bits);
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
    // Let the NEON kernel compute its own input_sums (the None path).
    assert_neon_eq_scalar("random", &weight, &q8, &scales, None);
}
