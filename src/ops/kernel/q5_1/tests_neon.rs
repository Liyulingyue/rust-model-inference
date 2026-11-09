//! NEON parity tests for `q5_1::neon::matmul_q5_1_vs_q8_0_neon`.
//!
//! The NEON kernel and the scalar baseline must agree: the integer
//! dot product is exact in both, and the F32 accumulation runs in the
//! same per-block order, so the results are expected to be identical.
//!
//! Skips silently on hosts without ARMv8.4-A `dotprod` (UDOT), since
//! the kernel falls back to the scalar baseline there.

#![cfg(target_arch = "aarch64")]

use super::neon::matmul_q5_1_vs_q8_0_neon;
use super::scalar::matmul_q5_1_scalar_range;

/// Build one Q5_1 block from a 5-bit level array.
fn build_block(d: f32, m: f32, levels: &[u8; 32]) -> Vec<u8> {
    let mut qs = [0u8; 16];
    let mut qh = 0u32;
    for j in 0..16 {
        let lo = levels[j] & 0x1F;
        let hi = levels[j + 16] & 0x1F;
        qs[j] = (lo & 0x0F) | ((hi & 0x0F) << 4);
        qh |= ((lo >> 4) as u32 & 1) << j;
        qh |= ((hi >> 4) as u32 & 1) << (j + 16);
    }
    let mut block = Vec::with_capacity(24);
    block.extend_from_slice(&crate::ops::f32_to_f16(d).to_le_bytes());
    block.extend_from_slice(&crate::ops::f32_to_f16(m).to_le_bytes());
    block.extend_from_slice(&qh.to_le_bytes());
    block.extend_from_slice(&qs);
    block
}

fn assert_neon_eq_scalar(label: &str, weight: &[u8], q8: &[u8], scales: &[f32]) {
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        return;
    }
    let n_in = q8.len();
    let n_out = weight.len() / (n_in / 32 * 24);
    let mut neon_out = vec![0.0f32; n_out];
    let mut scalar_out = vec![0.0f32; n_out];
    unsafe {
        matmul_q5_1_vs_q8_0_neon(weight, q8, scales, None, &mut neon_out, n_in, 0, n_out);
    }
    matmul_q5_1_scalar_range(weight, q8, scales, None, &mut scalar_out, n_in, n_out, 0, 1);
    for (i, (a, b)) in neon_out.iter().zip(scalar_out.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "{label} row {i}: neon={a} (bits {:x}) scalar={b} (bits {:x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

#[test]
fn parity_all_levels_zero() {
    // Every level is 0: only the `m` term contributes.
    let weight = build_block(1.0, 0.25, &[0u8; 32]);
    let q8 = vec![1i8 as u8; 32];
    let scales = vec![1.0f32];
    assert_neon_eq_scalar("all-zero-levels", &weight, &q8, &scales);
}

#[test]
fn parity_all_levels_max() {
    // Every level is 31: the maximum 5-bit value, only the dot product
    // contributes.
    let weight = build_block(1.0, 0.0, &[31u8; 32]);
    let q8 = vec![127i8 as u8; 32];
    let scales = vec![1.0f32];
    assert_neon_eq_scalar("all-max-levels", &weight, &q8, &scales);
}

#[test]
fn parity_negative_activation() {
    let weight = build_block(0.5, -0.125, &[16u8; 32]);
    let q8 = vec![0x80u8; 32];
    let scales = vec![1.0f32];
    assert_neon_eq_scalar("negative-activation", &weight, &q8, &scales);
}

#[test]
fn parity_random_blocks() {
    // Deterministic pseudo-random blocks: the level sweep covers all
    // 32 values, and the random `qh` exercises every combination of
    // nibble bits and 5-th bits.
    let mut state: u64 = 0xfeed_face_dead_beef;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut weight = Vec::new();
    for _ in 0..8 {
        for _ in 0..4 {
            let d = 0.01 + next() as f32 / u32::MAX as f32;
            let m = 0.5 + next() as f32 / u32::MAX as f32;
            weight.extend_from_slice(&crate::ops::f32_to_f16(d).to_le_bytes());
            weight.extend_from_slice(&crate::ops::f32_to_f16(m).to_le_bytes());
            let mut levels = [0u8; 32];
            for level in levels.iter_mut() {
                *level = (next() % 32) as u8;
            }
            weight.extend_from_slice(&build_block(0.0, 0.0, &levels)[8..]);
        }
    }
    let q8: Vec<u8> = (0..1024)
        .map(|i| ((i as i32 % 61) - 30) as i8 as u8)
        .collect();
    let scales: Vec<f32> = (0..32).map(|b| 0.01 + (b as f32) * 0.001).collect();
    assert_neon_eq_scalar("random", &weight, &q8, &scales);
}
