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

    pub fn forward_scaled(
        &self,
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        scale: f32,
    ) {
        debug_assert_eq!(self.weight.len(), n_out * n_in * 2);
        debug_assert!(input.len() >= n_in);
        debug_assert!(output.len() >= n_out);
        debug_assert!(scale.is_finite() && scale != 0.0);

        let mut input_f16 = vec![0u16; n_in];
        if scale == 1.0 {
            crate::ops::f32_slice_to_f16(&input[..n_in], &mut input_f16);
        } else {
            for (converted, &value) in input_f16.iter_mut().zip(&input[..n_in]) {
                *converted = crate::ops::f32_to_f16(value * scale);
            }
        }

        let inverse_scale = scale.recip();
        for (out_idx, row) in (0..n_out).enumerate() {
            let row_off = row * n_in * 2;
            output[out_idx] = crate::ops::dot_f16_f16_bytes(
                &input_f16,
                &self.weight[row_off..row_off + n_in * 2],
                n_in,
            ) * inverse_scale;
        }
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

    /// F16 converts the input to F16 before the dot product, matching ggml's
    /// `vec_dot_type = GGML_TYPE_F16` contract.
    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        self.forward_scaled(input, output, n_in, n_out, 1.0);
    }

    /// F16's `forward_batched` goes through `forward` (f32 path) rather
    /// than the default impl (which quantizes input then calls
    /// `forward_prequantized`, a placeholder for F16).
    fn forward_batched(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
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

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[test]
    fn f16_kernel_matches_ggml_f16_input_and_native_accumulation() {
        let input = [
            0x3e06_184f,
            0xbe8b_2d10,
            0x3ea0_d97c,
            0xbf13_c468,
            0x3f35_04f3,
            0xbf31_7218,
            0x3ede_5bd9,
            0xbf6a_7cb9,
            0x3e10_d0c3,
            0xbe31_5cac,
            0x3e64_f92e,
            0xbe7a_d3e7,
            0x3e87_7664,
            0xbe90_d0c3,
            0x3ea1_e89b,
            0xbea9_cfaa,
            0x3eb1_5cac,
            0xbeb8_9aaf,
            0x3ebf_92a7,
            0xbec6_4bf8,
            0x3ed3_1a5f,
            0xbedf_2cf6,
            0x3eea_a0bc,
            0xbef5_8bec,
            0x3f02_88ef,
            0xbf09_dc37,
            0x3f10_d0c3,
            0xbf17_739f,
            0x3f1d_cf1b,
            0xbf23_eb84,
            0x3f29_cfaa,
            0xbf2f_8138,
        ]
        .map(f32::from_bits);
        let weight = [
            0xbb80u16, 0x3a80, 0xb980, 0x3880, 0xb700, 0x3500, 0xb200, 0x2c00, 0x2e00, 0xb300,
            0x3580, 0xb780, 0x38c0, 0xb9c0, 0x3ac0, 0xbbc0, 0x3be0, 0xbae0, 0x39e0, 0xb8e0, 0x37c0,
            0xb5c0, 0x3380, 0xaf00, 0xaa00, 0x3180, 0xb4c0, 0x36c0, 0xb860, 0x3960, 0xba60, 0x3b60,
        ]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
        let mut output = [0.0f32; 1];

        F16Kernel::new(&weight).forward(&input, &mut output, 32, 1);

        assert_eq!(output[0].to_bits(), 0xbf92_1000);
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[test]
    fn f16_kernel_matches_ggml_scaled_input_contract() {
        let input = [
            0xb991_c399,
            0xc1a2_96d4,
            0x4203_f9cb,
            0xb865_c801,
            0x36f0_d07b,
            0x41a2_157e,
            0xc24d_ee89,
            0xbc6d_779b,
            0xc3f9_3823,
            0xbc1e_b9ca,
            0xc460_7c5e,
            0x407a_395e,
            0xbc91_1904,
            0x431a_dfc6,
            0xc311_a020,
            0xbda0_24f9,
            0xbdde_0f85,
            0x39ca_b471,
            0x40ed_3ebd,
            0xbadf_54ca,
            0x39ba_d4a0,
            0x3ab6_787b,
            0xbd77_7bba,
            0x3e59_215e,
            0xc049_06f9,
            0xbbaa_0af4,
            0x3593_848c,
            0x4061_3974,
            0x3e32_c573,
            0x400f_5d18,
            0x3ddc_ecc3,
            0xc39c_faa4,
        ]
        .map(f32::from_bits);
        let weight = [
            0x9853u16, 0x2780, 0xacd2, 0x201f, 0x8cbd, 0x18e2, 0x2917, 0x2edf, 0x9973, 0xb011,
            0x23ca, 0xadfd, 0x9f35, 0x941a, 0xa964, 0x333a, 0x12fe, 0x09b1, 0x8c1d, 0x37ab, 0xae7b,
            0xb14d, 0x0daa, 0x88ee, 0x8c01, 0xa8e9, 0x25b5, 0x87a6, 0x9b59, 0xb411, 0xa8a4, 0x344e,
        ]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
        let mut output = [0.0f32; 1];

        F16Kernel::new(&weight).forward_scaled(&input, &mut output, 32, 1, 1.0 / 128.0);

        assert_eq!(output[0].to_bits(), 0xc2c1_c587);
    }
}
