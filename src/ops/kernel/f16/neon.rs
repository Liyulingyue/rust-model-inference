//! F16×F16 NEON (aarch64) matmul kernel — placeholder.
//!
//! TODO-005: implement the NEON path once a CI target exposes aarch64.
//! `crate::ops::dot::dot_f16_fp16_neon` already implements a 32-element
//! f16×f16 dot for the legacy `F16Kernel::forward_scaled_rows` path; the
//! F16×F32 kernel here would mirror `avx2.rs` but with NEON intrinsics.

#[cfg(target_arch = "aarch64")]
pub unsafe fn matmul_f16_vs_f32_neon(
    _weight: &[u8],
    _input: &[f32],
    _output: &mut [f32],
    _n_in: usize,
    _row_start: usize,
    _row_end: usize,
) {
    unreachable!("f16 neon kernel not yet implemented; falls back to scalar")
}
