//! Q5_1 block matmul kernel implementation.
//!
//! Q5_1 uses 32-element blocks with a 24-byte layout:
//! - 2-byte F16 scale (`d`)
//! - 2-byte F16 min (`m`)
//! - 4-byte high-bit table (the 5-th bit of each element)
//! - 16-byte low-nibble table (the low four bits of each element)
//!
//! The value of an element is `level * d + m` where `level` is the
//! unsigned 5-bit value. See [`scalar`] for the arithmetic and the two
//! places where Q5_1 differs from Q5_0.
//!
//! Module structure mirrors q5_0:
//! - `scalar.rs` — scalar baseline (`matmul_q5_1_scalar_range`).
//! - `avx2.rs`   — AVX2 fast path (x86_64), not implemented yet.
//! - `neon.rs`   — ARMv8.4-A dot-product fast path (aarch64), not
//!   implemented yet.

use super::Kernel;
// SIMD paths are optional: the scalar baseline covers every GGUF Q5_1
// block correctly. The submodule declarations are gated so they only pull
// in files that exist on the matching target.
#[cfg(target_arch = "x86_64")]
#[path = "avx2.rs"]
pub mod avx2;
#[cfg(target_arch = "aarch64")]
#[path = "neon.rs"]
pub mod neon;
pub mod scalar;

pub use scalar::matmul_q5_1_scalar_range;

#[derive(Debug, Clone, Copy)]
pub struct Q5_1Kernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> Q5_1Kernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 32;
    pub const BLOCK_BYTES: usize = 24;

    pub fn new(data: &'a [u8], _n_in: usize, _n_out: usize) -> Self {
        Self { weight: data }
    }
}

impl<'a> Kernel for Q5_1Kernel<'a> {
    fn weight_bytes(&self) -> Option<&[u8]> {
        Some(self.weight)
    }

    fn forward_prequantized(
        &self,
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            let per_thread = (n_out + nth - 1) / nth;
            let my_start = ith * per_thread;
            let my_end = (my_start + per_thread).min(n_out);
            if my_start < my_end {
                unsafe {
                    neon::matmul_q5_1_vs_q8_0_neon(
                        self.weight,
                        input_q8,
                        input_scales,
                        None,
                        &mut output[my_start..my_end],
                        n_in,
                        my_start,
                        my_end,
                    );
                }
                return;
            }
        }
        matmul_q5_1_scalar_range(
            self.weight,
            input_q8,
            input_scales,
            None,
            output,
            n_in,
            n_out,
            ith,
            nth,
        );
    }

    /// `forward_prepared` is the path the llama-style trunks use. Q5_1's
    /// 32-element blocks do not align with the caller's Q8_K 256-element
    /// blocks, so the Q8_K activation is ignored and the F32 input is
    /// re-quantized to Q8_0, the same way Q4_1 / Q5_0 / Q4_K do it.
    ///
    /// Q5_1 carries a per-block minimum `m` in addition to the scale
    /// `d`, so the block also contributes `m * sum(y)`. ggml's
    /// `block_q8_1` stores that sum pre-multiplied by the activation
    /// scale, so it is derived here exactly the way Q4_1 derives it.
    fn forward_prepared(
        &self,
        input_f32: &[f32],
        input_q8: &[u8],
        input_scales: &[f32],
        _q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        let input_sums: Vec<f32> = input_f32[..n_in]
            .chunks_exact(32)
            .zip(input_q8[..n_in].chunks_exact(32))
            .map(|(values, quantized)| {
                let amax = values
                    .iter()
                    .fold(0.0f32, |current, value| current.max(value.abs()));
                let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
                let sum = quantized
                    .iter()
                    .map(|&value| i32::from(value as i8))
                    .sum::<i32>();
                crate::ops::f16_to_f32(crate::ops::f32_to_f16(sum as f32 * scale))
            })
            .collect();

        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            let per_thread = (n_out + nth - 1) / nth;
            let my_start = ith * per_thread;
            let my_end = (my_start + per_thread).min(n_out);
            if my_start < my_end {
                unsafe {
                    neon::matmul_q5_1_vs_q8_0_neon(
                        self.weight,
                        input_q8,
                        input_scales,
                        Some(&input_sums),
                        &mut output[my_start..my_end],
                        n_in,
                        my_start,
                        my_end,
                    );
                }
                return;
            }
        }

        matmul_q5_1_scalar_range(
            self.weight,
            input_q8,
            input_scales,
            Some(&input_sums),
            output,
            n_in,
            n_out,
            ith,
            nth,
        );
    }

    fn embedding_lookup(&self, token_id: u32, n_embd: usize, out: &mut [f32]) {
        let _ = (token_id, n_embd, out);
        unimplemented!("Q5_1 embedding_lookup is not implemented");
    }
}

#[cfg(target_arch = "aarch64")]
#[path = "tests_neon.rs"]
mod tests_neon;

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one Q5_1 block: F16 `d`, F16 `m`, the 4-byte `qh` table and the
    /// 16-byte `qs` table. `level_lo` / `level_hi` are the 5-bit levels of
    /// the elements in the low and the upper nibble.
    fn q5_1_block(d: f32, m: f32, levels: &[u8; 32]) -> Vec<u8> {
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

    /// Dequantize a block exactly the way ggml's `dequantize_row_q5_1`
    /// does, so the matmul can be checked against a direct transcription of
    /// the reference rather than against itself.
    fn reference_dequant(block: &[u8]) -> Vec<f32> {
        let d = crate::ops::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = crate::ops::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24];
        let mut out = vec![0.0f32; 32];
        for j in 0..16 {
            // ggml writes these as `uint8_t`; the `as u8` casts mirror the
            // truncation the C code gets from the assignment.
            let xh_0 = (((qh >> (j + 0)) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = (qs[j] & 0x0F) | xh_0;
            let x1 = (qs[j] >> 4) | xh_1;
            out[j] = x0 as f32 * d + m;
            out[j + 16] = x1 as f32 * d + m;
        }
        out
    }

    #[test]
    fn q5_1_matches_the_reference_dequantization() {
        // A block whose levels sweep 0..31 so that every combination of
        // nibble bits and 5-th bits appears at least once.
        let mut levels = [0u8; 32];
        for (index, level) in levels.iter_mut().enumerate() {
            *level = (index as u8 * 7) % 32;
        }
        let block = q5_1_block(0.03125, -0.5, &levels);
        let reference = reference_dequant(&block);

        // The kernel's output for an all-ones input is the sum of the
        // dequantized values, which the reference reproduces directly.
        let input_q8 = vec![1i8 as u8; 32];
        let input_scales = vec![1.0f32];
        let mut output = [0.0f32; 1];
        let kernel = Q5_1Kernel::new(&block, 32, 1);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);

        let expected: f32 = reference.iter().sum();
        assert!(
            (output[0] - expected).abs() <= 1e-3 * expected.abs().max(1.0),
            "kernel {} vs reference {expected}",
            output[0]
        );
    }

    #[test]
    fn q5_1_min_is_added_once_per_element() {
        // Level 0 everywhere: every element is exactly `m`.
        let levels = [0u8; 32];
        let block = q5_1_block(1.0, 0.25, &levels);
        let input_q8 = vec![1i8 as u8; 32];
        let input_scales = vec![1.0f32];
        let mut output = [0.0f32; 1];
        let kernel = Q5_1Kernel::new(&block, 32, 1);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);
        // 32 elements, each `0 * 1.0 + 0.25`.
        assert!((output[0] - 8.0).abs() < 1e-4, "{}", output[0]);
    }

    #[test]
    fn q5_1_high_bit_of_the_upper_nibble_uses_qh_bit_sixteen() {
        // Element 16 is the first upper-nibble element. Its 5-th bit must
        // come from qh bit 16, so setting only bit 16 has to change the
        // result while setting only bit 12 must not.
        let mut levels = [0u8; 32];
        levels[16] = 16;
        let block = q5_1_block(1.0, 0.0, &levels);
        let mut qs = block[8..24].to_vec();
        let mut qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        qh &= !(1u32 << 12); // clear the bit the wrong layout would read
        qs[0] = (qs[0] & 0xF0) | 0; // low nibble of element 0 is 0
        let mut patched = Vec::with_capacity(24);
        patched.extend_from_slice(&block[0..4]);
        patched.extend_from_slice(&qh.to_le_bytes());
        patched.extend_from_slice(&qs);

        let input_q8 = vec![1i8 as u8; 32];
        let input_scales = vec![1.0f32];
        let mut output = [0.0f32; 1];
        let kernel = Q5_1Kernel::new(&patched, 32, 1);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);
        // Element 16 keeps its 5-th bit, so it contributes 16; every other
        // element is 0.
        assert!((output[0] - 16.0).abs() < 1e-4, "{}", output[0]);
    }
}
