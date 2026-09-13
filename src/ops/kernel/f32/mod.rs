//! F32 matmul kernel implementation.
//!
//! Phase 2.2 + 2.7-final: F32 weights are rare in production (Q8_0 / Q4_K_M
//! dominate). For tests this kernel still exposes a working f32 matmul on
//! the `forward` path; the production `forward_prequantized` is a placeholder
//! that emits zeros because F32 weights do not appear in `LayerWeights`.
//!
//! Module structure mirrors the BF16 kernel so we get x86_64 AVX2+FMA and
//! aarch64 NEON paths transparently:
//! - `scalar.rs` — scalar fallback (also the reference for SIMD tests).
//! - `avx2.rs`    — AVX2+FMA f32×f32 matmul (Breeze `--quant f32` warm path).
//! - `neon.rs`    — NEON f32×f32 matmul.
//!
//! TODO-005 in `docs/TODO.md` tracks the broader plan to share a single
//! `matmul_f32_vs_f32_simd` core across BF16/F16/F32.

use super::Kernel;
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod scalar;

#[derive(Debug, Clone)]
pub struct F32Kernel {
    weight: Vec<f32>,
}

impl F32Kernel {
    pub fn new(weight: Vec<f32>) -> Self {
        Self { weight }
    }
}

impl Kernel for F32Kernel {
    fn f32_slice(&self) -> Option<&[f32]> {
        Some(&self.weight)
    }

    fn weight_bytes(&self) -> Option<&[u8]> {
        Some(bytemuck::cast_slice(&self.weight))
    }

    fn forward_prequantized(
        &self,
        _input_q8: &[u8],
        _input_scales: &[f32],
        output: &mut [f32],
        n_out: usize,
        n_in: usize,
        ith: usize,
        nth: usize,
    ) {
        matmul_f32_scalar_range(&self.weight, output, n_in, n_out, ith, nth);
    }

    fn forward_prepared(
        &self,
        input_f32: &[f32],
        _input_q8: &[u8],
        _input_scales: &[f32],
        _q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        scalar::forward_f32_rows(&self.weight, input_f32, output, n_in, n_out, ith, nth);
    }

    /// F32 has a native f32-input path. The trait default impl quantizes
    /// the input to Q8 then calls `forward_prequantized` (zero for F32);
    /// we override here to do the real f32 matmul via SIMD when available.
    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        forward_f32_rows_dispatch(&self.weight, input, output, n_in, n_out, 0, 1);
    }

    fn forward_batched(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        let n_tokens = input.len() / n_in;
        debug_assert_eq!(input.len(), n_tokens * n_in);
        debug_assert_eq!(output.len(), n_tokens * n_out);
        for token in 0..n_tokens {
            forward_f32_rows_dispatch(
                &self.weight,
                &input[token * n_in..(token + 1) * n_in],
                &mut output[token * n_out..(token + 1) * n_out],
                n_in,
                n_out,
                0,
                1,
            );
        }
    }

    fn embedding_lookup(&self, token_id: u32, n_embd: usize, output: &mut [f32]) {
        let offset = token_id as usize * n_embd;
        output.copy_from_slice(&self.weight[offset..offset + n_embd]);
    }
}

/// F32×F32 row-dispatch helper that selects AVX2 / NEON / scalar at runtime.
/// Mirrors `bf16::BF16Kernel::forward_f32_rows` and lives here so the BF16 /
/// F32 SIMD paths stay symmetric and can later converge into a single
/// `matmul_f32_vs_f32_simd` core (TODO-005).
pub(crate) fn forward_f32_rows_dispatch(
    weight: &[f32],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if n_in % 8 == 0 && crate::ops::has_avx2_fma() {
            let (start, end) = scalar::row_range(n_out, ith, nth);
            if end > start {
                let my_out = &mut output[start..end];
                let weight_bytes: &[u8] = bytemuck::cast_slice(weight);
                unsafe {
                    avx2::matmul_f32_vs_f32_avx2(weight_bytes, input, my_out, n_in, start, end);
                    return;
                }
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if crate::ops::has_neon() {
            let (start, end) = scalar::row_range(n_out, ith, nth);
            if end > start {
                let my_out = &mut output[start..end];
                unsafe {
                    neon::matmul_f32_vs_f32_neon(weight, input, my_out, n_in, start, end);
                    return;
                }
            }
        }
    }
    scalar::forward_f32_rows(weight, input, output, n_in, n_out, ith, nth);
}

/// F32 scalar matmul kernel. Phase 2.7-final: moved from `ops::matmul`.
///
/// Stub: only sums each row of the weight matrix. Real f32×f32 dot
/// product happens via `Kernel::forward` (which overrides this and uses
/// the f32-input path). Kept for completeness so a `Box<dyn Kernel>`
/// containing an F32 kernel still produces something deterministic rather
/// than zeros in the LayerWeights hot path.
pub fn matmul_f32_scalar_range(
    weight: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    scalar::row_dot_range(weight, output, n_in, n_out, ith, nth);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_kernel_row_sum() {
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = [1.0f32, 1.0, 1.0];
        let mut output = [0.0f32; 2];

        let kernel = F32Kernel::new(w);
        kernel.forward(&input, &mut output, 3, 2);

        assert_eq!(output, [6.0, 15.0]);
    }

    #[test]
    fn f32_kernel_weighted_input() {
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = [10.0f32, 20.0, 30.0];
        let mut output = [0.0f32; 2];

        let kernel = F32Kernel::new(w);
        kernel.forward(&input, &mut output, 3, 2);

        assert_eq!(output, [140.0, 320.0]);
    }

    #[test]
    fn f32_kernel_batched_default_loop() {
        let w = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = [1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0];
        let mut output = [0.0f32; 6];

        let kernel = F32Kernel::new(w);
        kernel.forward_batched(&input, &mut output, 2, 2);

        assert_eq!(output, [3.0, 7.0, 6.0, 14.0, 9.0, 21.0]);
    }
}
