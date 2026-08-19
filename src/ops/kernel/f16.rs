//! F16 matmul kernel implementation.
//!
//! Phase 2.4: Reserved interface for F16 matmul. The `F16Kernel` exists
//! to lock the contract for `ProcessedWeight::F16` (which currently is
//! not part of `ProcessedWeight`'s variants). This is the future home
//! of the F16 path once `QuantizedTensor` replaces `ProcessedWeight`.

use super::Kernel;

/// F16 matmul kernel: `output = weight × input`, all dequantized to f32.
///
/// `weight` is laid out as `[n_out rows × n_in cols]` of f16 values
/// (2 bytes per element, little-endian), total `n_out * n_in * 2` bytes.
#[derive(Debug, Clone, Copy)]
pub struct F16Kernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> F16Kernel<'a> {
    pub fn new(weight: &'a [u8]) -> Self {
        Self { weight }
    }

    /// Number of input columns (= n_in) for a given weight size.
    /// `bytes.len() / 2 / n_out = n_in`. Caller knows n_in already.
    #[inline]
    pub fn element_count(&self) -> usize {
        self.weight.len() / 2
    }
}

impl<'a> Kernel for F16Kernel<'a> {
    /// Single-token matmul. Reuses the scalar `dot_f16_f32` helper from
    /// the parent module — no SIMD dispatch yet. The hot path will gain
    /// AVX2/NEON variants when F16 lands in the production dispatch.
    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        debug_assert_eq!(self.weight.len(), n_out * n_in * 2);
        debug_assert!(input.len() >= n_in);
        debug_assert!(output.len() >= n_out);

        // Zero-copy u16 view of the f16 weight bytes. Safe because:
        // 1. GGUF guarantees f16 weights are packed little-endian.
        // 2. The slice was allocated with at least 2-byte alignment.
        // 3. We only read, never write.
        let weight_u16: &[u16] = unsafe {
            std::slice::from_raw_parts(
                self.weight.as_ptr() as *const u16,
                self.weight.len() / 2,
            )
        };

        for (out_idx, row) in (0..n_out).enumerate() {
            let row_off = row * n_in;
            output[out_idx] =
                crate::ops::dot_f16_f32(input, &weight_u16[row_off..row_off + n_in], n_in);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn f16_bytes(values: &[f16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for v in values {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        bytes
    }

    #[test]
    fn f16_kernel_one_row_one_input() {
        // 1x3 weight: [1, 2, 3] (f16)
        let weight = f16_bytes(&[f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)]);
        let input = [1.0f32, 1.0, 1.0];
        let mut output = [0.0f32; 1];

        let kernel = F16Kernel::new(&weight);
        kernel.forward(&input, &mut output, 3, 1);

        assert_eq!(output[0], 6.0);
    }

    #[test]
    fn f16_kernel_weighted_input() {
        // 1x3 weight: [1, 2, 3]
        let weight = f16_bytes(&[f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)]);
        // input = [10, 20, 30] → 1*10 + 2*20 + 3*30 = 140
        let input = [10.0f32, 20.0, 30.0];
        let mut output = [0.0f32; 1];

        let kernel = F16Kernel::new(&weight);
        kernel.forward(&input, &mut output, 3, 1);

        assert_eq!(output[0], 140.0);
    }

    #[test]
    fn f16_kernel_multi_row() {
        // 2x3 weight:
        //   row 0: [1, 2, 3]
        //   row 1: [4, 5, 6]
        let weight = f16_bytes(&[
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ]);
        let input = [1.0f32, 1.0, 1.0];
        let mut output = [0.0f32; 2];

        let kernel = F16Kernel::new(&weight);
        kernel.forward(&input, &mut output, 3, 2);

        assert_eq!(output, [6.0, 15.0]);
    }

    #[test]
    fn f16_kernel_batched_default_loop() {
        // 1x2 weight: [2, 3]
        let weight = f16_bytes(&[f16::from_f32(2.0), f16::from_f32(3.0)]);
        // 3 tokens: [1,1], [2,2], [3,3]
        let input = [1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0];
        let mut output = [0.0f32; 3];

        let kernel = F16Kernel::new(&weight);
        kernel.forward_batched(&input, &mut output, 2, 1);

        // [2*1+3*1, 2*2+3*2, 2*3+3*3] = [5, 10, 15]
        assert_eq!(output, [5.0, 10.0, 15.0]);
    }
}