//! F32 × Q8_0 NEON (aarch64) matmul kernel.
//!
//! Sister to `bf16::neon_q8` and `f16::neon_q8`.  F32 weight lanes are
//! loaded with `vld1q_f32` (no cast), and the Q8 input lands in f32
//! through the same sign-extend ladder
//! (`vld1q_s8` → `vmovl_s8` → `vmovl_s16` → `vcvtq_f32_s32`).  Per-
//! block Q8 scales are applied via `vdupq_n_f32` per 4-lane half so
//! 8-lane groups that straddle a 32-element Q8 block boundary stay
//! correct.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

/// Q8 block size — every 32 input bytes share one scale.
const Q8_BLOCK: usize = 32;

#[target_feature(enable = "neon")]
pub unsafe fn matmul_f32_vs_q8_neon(
    weight: &[f32],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    row_start: usize,
    row_end: usize,
) {
    let w_ptr = weight.as_ptr();
    let q_ptr = input_q8.as_ptr();
    let s_ptr = input_scales.as_ptr();
    let out_ptr = output.as_mut_ptr();

    assert!(
        input_q8.len() >= n_in,
        "F32×Q8 NEON: input_q8 len {} < n_in {n_in}",
        input_q8.len()
    );
    assert!(
        input_scales.len() >= n_in.div_ceil(32),
        "F32×Q8 NEON: input_scales len {} < required {}",
        input_scales.len(),
        n_in.div_ceil(32)
    );

    let mut out_local = 0usize;
    for row in row_start..row_end {
        let row_off = row * n_in;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);

        let mut col = 0usize;
        while col + 8 <= n_in {
            let block_lo = col / Q8_BLOCK;
            let block_hi = (col + 4) / Q8_BLOCK;
            let scale_lo = vdupq_n_f32(*s_ptr.add(block_lo));
            let scale_hi = vdupq_n_f32(*s_ptr.add(block_hi));

            let w_lo = vld1q_f32(w_ptr.add(row_off + col));
            let w_hi = vld1q_f32(w_ptr.add(row_off + col + 4));

            // Q8 input: load 8 i8, sign-extend to s32 in two halves.
            let q8_bytes = vld1q_s8(q_ptr.add(col));
            let q_lo_i32 = vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(q8_bytes))));
            let q_hi_i32 = vmovl_s16(vget_high_s16(vmovl_s8(q8_bytes)));
            let q_lo_f32 = vcvtq_f32_s32(q_lo_i32);
            let q_hi_f32 = vcvtq_f32_s32(q_hi_i32);

            acc0 = vfmaq_f32(acc0, w_lo, vmulq_f32(q_lo_f32, scale_lo));
            acc1 = vfmaq_f32(acc1, w_hi, vmulq_f32(q_hi_f32, scale_hi));
            col += 8;
        }

        let mut total = vaddvq_f32(vaddq_f32(acc0, acc1));
        while col < n_in {
            let w_val = *w_ptr.add(row_off + col);
            let q_val = *q_ptr.add(col) as i8 as f32;
            let scale = *s_ptr.add(col / Q8_BLOCK);
            total += w_val * q_val * scale;
            col += 1;
        }
        let output_index = if output.len() >= n_out {
            row
        } else {
            out_local
        };
        *out_ptr.add(output_index) = total;
        out_local += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::matmul_f32_vs_q8_neon;
    use crate::ops::kernel::f32::scalar::forward_q8_rows_scalar;

    fn assert_neon_eq_scalar(
        label: &str,
        weight: &[f32],
        input: &[f32],
        n_in: usize,
        n_out: usize,
    ) {
        let mut neon_out = vec![0.0f32; n_out];
        let mut scalar_out = vec![0.0f32; n_out];
        let mut input_q8 = vec![0u8; n_in];
        let mut input_scales = vec![0.0f32; n_in.div_ceil(32)];
        crate::ops::quantize_q8_0_into(input, n_in, &mut input_q8, &mut input_scales);
        unsafe {
            matmul_f32_vs_q8_neon(
                weight,
                &input_q8,
                &input_scales,
                &mut neon_out,
                n_in,
                n_out,
                0,
                n_out,
            );
        }
        forward_q8_rows_scalar(
            weight,
            &input_q8,
            &input_scales,
            &mut scalar_out,
            n_in,
            n_out,
            0,
            n_out,
        );
        for (i, (n, s)) in neon_out.iter().zip(scalar_out.iter()).enumerate() {
            let denom = s.abs().max(1.0);
            assert!(
                (n - s).abs() / denom < 1e-2,
                "{label} row {i}: neon={n} scalar={s} diff={}",
                (n - s).abs()
            );
        }
    }

    #[test]
    fn neon_matches_scalar_small() {
        let n_in = 16;
        let n_out = 4;
        let weight: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i as f32 * 0.07).sin()))
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| (i as f32 - 8.0) * 0.05).collect();
        assert_neon_eq_scalar("small 4x16", &weight, &input, n_in, n_out);
    }

    #[test]
    fn neon_matches_scalar_large() {
        let n_in = 256;
        let n_out = 8;
        let weight: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i * 13 + 7) % 47) as f32 * 0.01 - 0.2)
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| ((i % 23) as f32 - 11.0) * 0.03).collect();
        assert_neon_eq_scalar("large 8x256", &weight, &input, n_in, n_out);
    }

    #[test]
    fn neon_matches_scalar_cross_block() {
        let n_in = 96;
        let n_out = 2;
        let weight: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i as f32 * 0.013).cos()))
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.13).sin()).collect();
        assert_neon_eq_scalar("cross-block 2x96", &weight, &input, n_in, n_out);
    }

    #[test]
    fn neon_matches_scalar_tail() {
        let n_in = 41;
        let n_out = 3;
        let weight: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i as f32 * 0.01).sin()))
            .collect();
        let input: Vec<f32> = (0..n_in).map(|i| i as f32 * 0.03).collect();
        assert_neon_eq_scalar("tail 3x41", &weight, &input, n_in, n_out);
    }
}
