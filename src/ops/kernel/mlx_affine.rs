//! MLX affine decoding with shared SIMD/scalar dot-product dispatch.

use super::Kernel;
use half::f16;

pub struct MlxAffineKernel<'a> {
    packed: &'a [u8],
    scales: &'a [u8],
    biases: &'a [u8],
    n_in: usize,
    n_out: usize,
    bits: usize,
    lora: Option<(&'a [u8], &'a [u8], usize, f32)>,
}

impl<'a> MlxAffineKernel<'a> {
    pub fn new(
        packed: &'a [u8],
        scales: &'a [u8],
        biases: &'a [u8],
        n_in: usize,
        n_out: usize,
        bits: usize,
        lora: Option<(&'a [u8], &'a [u8], usize, f32)>,
    ) -> Result<Self, String> {
        if !matches!(bits, 4 | 8) || n_in == 0 || n_in % 64 != 0 || n_out == 0 {
            return Err("invalid MLX affine matrix dimensions or bit width".into());
        }
        let packed_len = n_in
            .checked_mul(n_out)
            .and_then(|n| n.checked_mul(bits))
            .map(|n| n / 8)
            .ok_or("MLX affine packed length overflow")?;
        let group_len = n_in
            .checked_div(64)
            .and_then(|n| n.checked_mul(n_out))
            .and_then(|n| n.checked_mul(2))
            .ok_or("MLX affine scale length overflow")?;
        if packed.len() != packed_len || scales.len() != group_len || biases.len() != group_len {
            return Err(format!(
                "MLX affine payload size mismatch: weight={}, scales={}, biases={}, expected={packed_len}/{group_len}",
                packed.len(), scales.len(), biases.len()
            ));
        }
        if let Some((a, b, rank, _)) = lora {
            if rank == 0 || a.len() != rank * n_in * 2 || b.len() != n_out * rank * 2 {
                return Err("MLX affine LoRA shape mismatch".into());
            }
        }
        Ok(Self {
            packed,
            scales,
            biases,
            n_in,
            n_out,
            bits,
            lora,
        })
    }

    #[inline]
    fn bf16(bytes: &[u8], index: usize) -> f32 {
        let i = index * 2;
        f32::from_bits(u32::from(u16::from_le_bytes([bytes[i], bytes[i + 1]])) << 16)
    }

    #[inline]
    fn f16(bytes: &[u8], index: usize) -> f32 {
        let i = index * 2;
        f16::from_bits(u16::from_le_bytes([bytes[i], bytes[i + 1]])).to_f32()
    }

    fn decode_row(&self, row: usize, output: &mut [f32]) {
        let group_bytes = 64 * self.bits / 8;
        let row_offset = row * self.n_in * self.bits / 8;
        for (group, values) in output.chunks_exact_mut(64).enumerate() {
            let scale_index = row * (self.n_in / 64) + group;
            let scale = Self::bf16(self.scales, scale_index);
            let bias = Self::bf16(self.biases, scale_index);
            let offset = row_offset + group * group_bytes;
            let packed = &self.packed[offset..offset + group_bytes];
            if self.bits == 4 {
                for (&codes, pair) in packed.iter().zip(values.chunks_exact_mut(2)) {
                    pair[0] = scale * f32::from(codes & 15) + bias;
                    pair[1] = scale * f32::from(codes >> 4) + bias;
                }
            } else {
                for (&code, value) in packed.iter().zip(values) {
                    *value = scale * f32::from(code) + bias;
                }
            }
        }
    }

    fn dot_f16(input: &[f32], weights: &[u8], decoded: &mut [f32]) -> f32 {
        for (index, value) in decoded[..input.len()].iter_mut().enumerate() {
            *value = Self::f16(weights, index);
        }
        crate::ops::dot_f32_exact(input, decoded, input.len())
    }
}

impl Kernel for MlxAffineKernel<'_> {
    fn forward_prequantized(
        &self,
        _input_q8: &[u8],
        _input_scales: &[f32],
        _output: &mut [f32],
        _n_in: usize,
        _n_out: usize,
        _ith: usize,
        _nth: usize,
    ) {
        panic!("MLX affine requires original F32 activations");
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
        debug_assert_eq!((n_in, n_out), (self.n_in, self.n_out));
        let rows_per_thread = n_out.div_ceil(nth);
        let start = ith * rows_per_thread;
        let end = ((ith + 1) * rows_per_thread).min(n_out);
        if start >= end {
            return;
        }
        let rank = self.lora.map_or(0, |(_, _, rank, _)| rank);
        let mut decoded = vec![0.0; n_in.max(rank)];
        let lora_low = self.lora.map(|(a, _, rank, _)| {
            (0..rank)
                .map(|row| {
                    Self::dot_f16(
                        input_f32,
                        &a[row * n_in * 2..(row + 1) * n_in * 2],
                        &mut decoded,
                    )
                })
                .collect::<Vec<_>>()
        });
        for row in start..end {
            self.decode_row(row, &mut decoded[..n_in]);
            let mut sum = crate::ops::dot_f32_exact(input_f32, &decoded, n_in);
            if let (Some((_, weights, rank, scale)), Some(low)) = (self.lora, &lora_low) {
                sum += scale
                    * Self::dot_f16(
                        low,
                        &weights[row * rank * 2..(row + 1) * rank * 2],
                        &mut decoded,
                    );
            }
            output[row] = sum;
        }
    }

    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        self.forward_prepared(input, &[], &[], None, output, n_in, n_out, 0, 1);
    }

    fn forward_batched(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        for (row_in, row_out) in input.chunks_exact(n_in).zip(output.chunks_exact_mut(n_out)) {
            self.forward(row_in, row_out, n_in, n_out);
        }
    }

    fn embedding_lookup(&self, token_id: u32, n_embd: usize, out: &mut [f32]) {
        assert_eq!(n_embd, self.n_in);
        assert!((token_id as usize) < self.n_out);
        assert_eq!(out.len(), n_embd);
        self.decode_row(token_id as usize, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_and_lora_use_shared_dot_dispatch() {
        let mut input = [0.0f32; 64];
        input[0] = 16_777_216.0;
        input[4] = 1.0;
        input[8] = -16_777_216.0;
        input[12] = 1.0;
        let expected = crate::ops::dot_f32_exact(&input, &[1.0; 64], 64);
        let scales = 0x3f80u16.to_le_bytes();
        let biases = 0u16.to_le_bytes();
        let lora_a = vec![f16::from_f32(1.0).to_bits().to_le_bytes(); 64].concat();
        let lora_b = f16::from_f32(0.5).to_bits().to_le_bytes();
        for bits in [4, 8] {
            let packed = vec![if bits == 4 { 0x11 } else { 1 }; 64 * bits / 8];
            for lora in [None, Some((lora_a.as_slice(), lora_b.as_slice(), 1, 2.0))] {
                let kernel =
                    MlxAffineKernel::new(&packed, &scales, &biases, 64, 1, bits, lora).unwrap();
                let mut output = [f32::NAN];
                kernel.forward(&input, &mut output, 64, 1);
                let expected = if lora.is_some() {
                    expected * 2.0
                } else {
                    expected
                };
                assert_eq!(output[0].to_bits(), expected.to_bits(), "bits={bits}");
            }
        }
    }

    #[test]
    fn affine_rows_match_independent_reference_with_lora_and_partitions() {
        let n_in = 128;
        let n_out = 5;
        let rank = 3;
        let input: Vec<f32> = (0..n_in)
            .map(|index| ((index * 17 % 43) as f32 - 21.0) / 7.0)
            .collect();
        let scales: Vec<f32> = (0..n_out * 2)
            .map(|index| (index as f32 - 4.0) / 16.0)
            .collect();
        let biases: Vec<f32> = (0..n_out * 2)
            .map(|index| (3.0 - index as f32) / 8.0)
            .collect();
        let scale_bytes: Vec<u8> = scales
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let bias_bytes: Vec<u8> = biases
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let lora_a: Vec<f32> = (0..rank * n_in)
            .map(|index| ((index * 7 % 13) as f32 - 6.0) / 32.0)
            .collect();
        let lora_b: Vec<f32> = (0..n_out * rank)
            .map(|index| (index as f32 - 7.0) / 16.0)
            .collect();
        let lora_a_bytes: Vec<u8> = lora_a
            .iter()
            .flat_map(|&value| f16::from_f32(value).to_bits().to_le_bytes())
            .collect();
        let lora_b_bytes: Vec<u8> = lora_b
            .iter()
            .flat_map(|&value| f16::from_f32(value).to_bits().to_le_bytes())
            .collect();
        let scalar_dot = |left: &[f32], right: &[f32]| {
            left.iter()
                .zip(right)
                .fold(0.0f32, |sum, (&left, &right)| sum + left * right)
        };
        let low: Vec<f32> = lora_a
            .chunks_exact(n_in)
            .map(|weights| scalar_dot(&input, weights))
            .collect();
        for bits in [4, 8] {
            let codes: Vec<u8> = (0..n_in * n_out)
                .map(|index| ((index * 37 + 3) % (1 << bits)) as u8)
                .collect();
            let packed: Vec<u8> = if bits == 4 {
                codes
                    .chunks_exact(2)
                    .map(|pair| pair[0] | pair[1] << 4)
                    .collect()
            } else {
                codes.clone()
            };
            let weights: Vec<f32> = codes
                .iter()
                .enumerate()
                .map(|(index, &code)| scales[index / 64] * f32::from(code) + biases[index / 64])
                .collect();
            let kernel = MlxAffineKernel::new(
                &packed,
                &scale_bytes,
                &bias_bytes,
                n_in,
                n_out,
                bits,
                Some((&lora_a_bytes, &lora_b_bytes, rank, 0.75)),
            )
            .unwrap();
            for row in 0..n_out {
                let mut embedding = vec![f32::NAN; n_in];
                kernel.embedding_lookup(row as u32, n_in, &mut embedding);
                assert_eq!(embedding, weights[row * n_in..(row + 1) * n_in]);
            }
            let expected: Vec<f32> = weights
                .chunks_exact(n_in)
                .enumerate()
                .map(|(row, weights)| {
                    scalar_dot(&input, weights)
                        + 0.75 * scalar_dot(&low, &lora_b[row * rank..(row + 1) * rank])
                })
                .collect();
            for worker in 0..8 {
                let mut output = vec![f32::NAN; n_out];
                kernel.forward_prepared(
                    &input,
                    &[],
                    &[],
                    None,
                    &mut output,
                    n_in,
                    n_out,
                    worker,
                    8,
                );
                for (row, &value) in output.iter().enumerate() {
                    if row != worker {
                        assert!(value.is_nan(), "worker {worker} overwrote row {row}");
                    } else if crate::ops::scalar_mode() {
                        assert_eq!(value.to_bits(), expected[row].to_bits());
                    } else {
                        assert!(
                            (value - expected[row]).abs() <= 2e-5 * expected[row].abs().max(1.0),
                            "bits={bits}, row={row}: {value} != {}",
                            expected[row]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn affine4_and_lora_preserve_scalar_math() {
        let packed = vec![0x11111111u32.to_le_bytes(); 8].concat();
        let scales = 0x3f80u16.to_le_bytes();
        let biases = 0u16.to_le_bytes();
        let a = vec![f16::from_f32(0.5).to_bits().to_le_bytes(); 64].concat();
        let b = f16::from_f32(2.0).to_bits().to_le_bytes();
        let kernel =
            MlxAffineKernel::new(&packed, &scales, &biases, 64, 1, 4, Some((&a, &b, 1, 2.0)))
                .unwrap();
        let mut out = [0.0];
        kernel.forward(&[1.0; 64], &mut out, 64, 1);
        assert_eq!(out[0].to_bits(), 192.0f32.to_bits());
        let mut embedding = [0.0; 64];
        kernel.embedding_lookup(0, 64, &mut embedding);
        assert!(embedding.iter().all(|&v| v.to_bits() == 1.0f32.to_bits()));

        let packed = vec![0x11111111u32.to_le_bytes(); 16].concat();
        let scales = [0x3f80u16.to_le_bytes(), 0x4000u16.to_le_bytes()].concat();
        let biases = [0u16.to_le_bytes(), 0x3f80u16.to_le_bytes()].concat();
        let kernel = MlxAffineKernel::new(&packed, &scales, &biases, 128, 1, 4, None).unwrap();
        kernel.forward(&[1.0; 128], &mut out, 128, 1);
        assert_eq!(out[0].to_bits(), 256.0f32.to_bits());

        let mut packed = vec![0u8; 64];
        packed[..4].copy_from_slice(&[0, 1, 2, 3]);
        let scale = 0x3f80u16.to_le_bytes();
        let bias = 0xbf80u16.to_le_bytes();
        let kernel = MlxAffineKernel::new(&packed, &scale, &bias, 64, 1, 8, None).unwrap();
        kernel.forward(&[1.0; 64], &mut out, 64, 1);
        assert_eq!(out[0].to_bits(), (-58.0f32).to_bits());
    }
}
