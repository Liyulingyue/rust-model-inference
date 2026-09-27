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

#[cfg(target_arch = "x86_64")]
use crate::ops::has_avx2_fma;
#[cfg(target_arch = "aarch64")]
use crate::ops::has_neon;

#[cfg(target_arch = "x86_64")]
use super::avx2::matmul_q8_0_vs_q8_0_avx2;
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
    if has_avx2_fma() {
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
