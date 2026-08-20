//! Q8_0 legacy f32-input API surface.
//!
//! Phase 2.7-final: split from `ops::matmul`. Accepts raw f32 input and
//! internally quantizes to Q8 via `quantize_q8_0_into` before dispatching.
//!
//! Used only by `bin/micro_bench.rs` for A/B comparison vs the prequantized
//! production path in `parallel`. The hot path goes through
//! `parallel::matmul_q8_0_quantized_parallel_rows` instead.

use crate::ops::has_avx2_fma;
#[cfg(target_arch = "aarch64")]
use crate::ops::has_neon;
use crate::ops::quant::q8_0::quantize_q8_0_into;

use super::avx2::matmul_q8_0_vs_q8_0_avx2;
#[cfg(target_arch = "aarch64")]
use super::neon::matmul_q8_0_vs_q8_0_neon;
use super::scalar::matmul_q8_0_quantized_scalar_range;

/// Q8_0 × f32 matmul: quantize input on-the-fly, then dispatch SIMD path.
///
/// Single-threaded. Caller-provided `q8_buf`/`scale_buf` are reused for
/// scratch; size must be at least `n_in / 32 * 34` bytes and `n_in / 32`
/// floats respectively.
pub fn matmul_q8_0_via_q8(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    q8_buf: &mut [u8],
    scale_buf: &mut [f32],
) {
    quantize_q8_0_into(input, n_in, q8_buf, scale_buf);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe {
                matmul_q8_0_vs_q8_0_avx2(
                    weight,
                    q8_buf,
                    scale_buf,
                    output,
                    n_in,
                    0,
                    n_out,
                );
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            unsafe {
                matmul_q8_0_vs_q8_0_neon(
                    weight,
                    q8_buf,
                    scale_buf,
                    output,
                    n_in,
                    0,
                    n_out,
                );
            }
            return;
        }
    }
    matmul_q8_0_quantized_scalar_range(
        weight,
        q8_buf,
        scale_buf,
        output,
        n_in,
        0,
        n_out,
    );
}

/// Q8_0 × f32 matmul, parallel-row variant.
///
/// Same as `matmul_q8_0_via_q8` but splits rows across threads via
/// `matmul_q8_0_quantized_parallel`.
pub fn matmul_q8_0_via_q8_parallel(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    q8_buf: &mut [u8],
    scale_buf: &mut [f32],
) {
    quantize_q8_0_into(input, n_in, q8_buf, scale_buf);
    crate::ops::matmul::matmul_q8_0_quantized_parallel(weight, q8_buf, scale_buf, output, n_in, n_out);
}