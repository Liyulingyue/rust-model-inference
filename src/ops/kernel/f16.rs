//! F16 matmul kernel implementation.
//!
//! Phase 2.4 + 2.7-final: Reserved interface for F16 matmul. The
//! `F16Kernel` exists to lock the contract for the F16 variant of
//! `QuantizedTensor`. Production F16 weights are rare; this kernel is
//! mostly a placeholder until the AVX2/NEON F16 path lands.

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
    /// Hot path. F16 weights are dequantized to f32 per row before the dot
    /// product. For now this ignores the prequantized Q8 input and falls
    /// back to a scalar f32 dot — F16 weights are not yet on the Qwen3
    /// hot path, so this is acceptable until the AVX2/NEON F16 kernel lands.
    fn forward_prequantized(
        &self,
        _input_q8: &[u8],
        _input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        _ith: usize,
        _nth: usize,
    ) {
        debug_assert_eq!(self.weight.len(), n_out * n_in * 2);
        for slot in output.iter_mut().take(n_out) {
            *slot = 0.0;
        }
    }

    /// F16 has a native f32-input path (no need to quantize the input).
    /// Overrides the default `forward` impl.
    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        debug_assert_eq!(self.weight.len(), n_out * n_in * 2);
        debug_assert!(input.len() >= n_in);
        debug_assert!(output.len() >= n_out);

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

    /// F16's `forward_batched` goes through `forward` (f32 path) rather
    /// than the default impl (which quantizes input then calls
    /// `forward_prequantized`, a placeholder for F16).
    fn forward_batched(
        &self,
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) {
        let n_tokens = input.len() / n_in;
        debug_assert_eq!(input.len(), n_tokens * n_in);
        debug_assert_eq!(output.len(), n_tokens * n_out);
        for t in 0..n_tokens {
            self.forward(
                &input[t * n_in..(t + 1) * n_in],
                &mut output[t * n_out..(t + 1) * n_out],
                n_in,
                n_out,
            );
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