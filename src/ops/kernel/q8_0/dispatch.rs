//! Q8_0 per-row dispatch: AVX2 → NEON → scalar fallback.
//!
//! GPU (vulkan/wgpu) dispatch lives in `parallel.rs`, which routes the
//! whole-matmul call to a single full-range dispatch instead of per-row-range
//! GPU calls (the old per-range path re-uploaded weights per call and
//! serialized 18 threads through the context mutex).
//!
//! Phase 2.7-final: split from `ops::matmul`. Entry point is
//! `matmul_q8_0_quantized_range`; called by `parallel::matmul_q8_0_quantized_parallel_rows`
//! and by external callers (`bin/server.rs`, `bin/micro_bench.rs`).

#[cfg(target_arch = "aarch64")]
use crate::ops::has_neon;
#[cfg(target_arch = "x86_64")]
use crate::ops::{has_avx2_fma, has_f16c};

#[cfg(target_arch = "x86_64")]
use super::avx2::matmul_q8_0_vs_q8_0_avx2;
#[cfg(target_arch = "aarch64")]
use super::neon;
#[cfg(target_arch = "aarch64")]
use super::neon::{
    matmul_q8_0_vs_q8_0_dotprod_nrc4, matmul_q8_0_vs_q8_0_neon, matmul_q8_0_vs_q8_0_neon_nrc1,
};
use super::scalar::matmul_q8_0_quantized_scalar_range;

/// Per-row dispatch: GPU → AVX2 → NEON → scalar.
///
/// The production hot path for Q8_0 matmul. Tries GPU first (if feature
/// enabled and context initialized), then architecture-specific SIMD,
/// finally scalar fallback.
pub fn matmul_q8_0_quantized_range(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    debug_assert_eq!(output.len(), row_end - row_start);
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() && has_f16c() {
        unsafe {
            matmul_q8_0_vs_q8_0_avx2(
                weight,
                input_q8,
                input_scales,
                output,
                n_in,
                row_start,
                row_end,
            );
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if has_neon() {
        unsafe {
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                matmul_q8_0_vs_q8_0_dotprod_nrc4(
                    weight,
                    input_q8,
                    input_scales,
                    output,
                    n_in,
                    row_start,
                    row_end,
                );
            } else {
                matmul_q8_0_vs_q8_0_neon(
                    weight,
                    input_q8,
                    input_scales,
                    output,
                    n_in,
                    row_start,
                    row_end,
                );
            }
        }
        return;
    }
    matmul_q8_0_quantized_scalar_range(
        weight,
        input_q8,
        input_scales,
        output,
        n_in,
        row_start,
        row_end,
    );
}

/// Batched dispatch: one weight read serves several activation rows.
///
/// The per-row entry re-reads the weight for every activation row, which
/// makes a batched forward pass cost as much traffic as that many
/// token-at-a-time passes. The batched entry shares the weight reads
/// between the rows and computes the same values.
pub fn matmul_q8_0_quantized_batched_range(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    n_rows: usize,
    ith: usize,
    nth: usize,
) {
    // The weight rows are the parallel dimension: every worker owns a
    // disjoint column range of every activation row.
    let (row_start, row_end) = if nth <= 1 || n_out == 0 {
        (0, n_out)
    } else {
        let per_thread = (n_out + nth - 1) / nth;
        let start = ith * per_thread;
        (start, (start + per_thread).min(n_out))
    };
    if row_start >= row_end {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if has_neon() && n_rows >= 2 {
        // SAFETY: NEON is available (runtime probe); the kernel reads the
        // weight rows and the activation rows and writes `n_rows * n_out`
        // outputs.
        //
        // The kernel's arithmetic is the plain NEON kernel's: the same
        // per-block dot and the same fused multiply-add chain. The dotprod
        // kernel differs only in how the integer dot is computed, and the
        // integer dot is exact, so both produce the same values.
        unsafe {
            neon::matmul_q8_0_vs_q8_0_neon_batched(
                weight,
                input_q8,
                input_scales,
                output,
                n_in,
                n_out,
                n_rows,
                row_start,
                row_end,
            );
        }
        return;
    }
    // Everything else keeps the per-row path.
    let blocks = n_in.div_ceil(32);
    for row in 0..n_rows {
        let q8 = &input_q8[row * n_in..(row + 1) * n_in];
        let scales = &input_scales[row * blocks..(row + 1) * blocks];
        // The per-row kernel writes the columns `row_start..row_end` at the
        // front of the slice it is given.
        let out = &mut output[row * n_out + row_start..row * n_out + row_end];
        matmul_q8_0_quantized_range(weight, q8, scales, out, n_in, row_start, row_end);
    }
}

/// aarch64-only single-row dispatch: NEON (NRC=1 fused FMA) → scalar.
///
/// Used by `parallel_range` when it selects a fused 1-row-at-a-time NEON
/// matmul.
#[cfg(target_arch = "aarch64")]
pub fn matmul_q8_0_quantized_range_nrc1(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    debug_assert_eq!(output.len(), row_end - row_start);
    if has_neon() {
        unsafe {
            matmul_q8_0_vs_q8_0_neon_nrc1(
                weight,
                input_q8,
                input_scales,
                output,
                n_in,
                row_start,
                row_end,
            );
        }
        return;
    }
    matmul_q8_0_quantized_scalar_range(
        weight,
        input_q8,
        input_scales,
        output,
        n_in,
        row_start,
        row_end,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_range_handles_row_offset_and_tail() {
        let mut weights = Vec::new();
        for row in 0..7 {
            for block in 0..2 {
                weights.extend_from_slice(&crate::ops::f32_to_f16(0.5).to_le_bytes());
                weights.extend_from_slice(&[(row + 1) * (block + 1); 32]);
            }
        }
        let mut output = [f32::NAN; 5];
        matmul_q8_0_quantized_range(&weights, &[2; 64], &[1.0, 2.0], &mut output, 64, 1, 6);
        assert_eq!(output, [320.0, 480.0, 640.0, 800.0, 960.0]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "requires AVX2/FMA with F16C disabled; run under QEMU in CI"]
    fn avx2_without_f16c_uses_scalar() {
        assert!(
            !cfg!(target_feature = "f16c"),
            "build with target-cpu=x86-64"
        );
        assert!(crate::ops::has_avx2_fma());
        assert!(!crate::ops::has_f16c());
        quantized_range_handles_row_offset_and_tail();
    }
}
