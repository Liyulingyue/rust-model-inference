//! F32×F32 NEON (aarch64) matmul kernel — placeholder.
//!
//! TODO-005: implement the NEON path to mirror `avx2.rs` once a CI target
//! exposes aarch64.  See `docs/TODO.md` for the plan to share the f32×f32
//! SIMD core across F32 / BF16 / F16 kernels.

#[cfg(target_arch = "aarch64")]
pub unsafe fn matmul_f32_vs_f32_neon(
    _weight: &[f32],
    _input: &[f32],
    _output: &mut [f32],
    _n_in: usize,
    _row_start: usize,
    _row_end: usize,
) {
    unreachable!("f32 neon kernel not yet implemented; falls back to scalar")
}
