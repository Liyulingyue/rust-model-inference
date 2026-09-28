//! Scalar MLX affine group quantization used by Edge0's lossless GGUF.

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

    fn value(&self, row: usize, col: usize) -> f32 {
        let values_per_word = 32 / self.bits;
        let word = row * (self.n_in / values_per_word) + col / values_per_word;
        let offset = word * 4;
        let packed = u32::from_le_bytes(self.packed[offset..offset + 4].try_into().unwrap());
        let q = (packed >> ((col % values_per_word) * self.bits)) & ((1 << self.bits) - 1);
        let group = row * (self.n_in / 64) + col / 64;
        Self::bf16(self.scales, group) * q as f32 + Self::bf16(self.biases, group)
    }

    fn row(&self, input: &[f32], row: usize, lora_low: Option<&[f32]>) -> f32 {
        let mut sum = 0.0f32;
        for (col, &x) in input.iter().enumerate() {
            sum += x * self.value(row, col);
        }
        if let (Some((_, b, rank, scale)), Some(low)) = (self.lora, lora_low) {
            let mut delta = 0.0f32;
            for r in 0..rank {
                delta += low[r] * Self::f16(b, row * rank + r);
            }
            sum += scale * delta;
        }
        sum
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
        let lora_low = self.lora.map(|(a, _, rank, _)| {
            (0..rank)
                .map(|r| {
                    input_f32.iter().enumerate().fold(0.0f32, |sum, (col, &x)| {
                        sum + x * Self::f16(a, r * n_in + col)
                    })
                })
                .collect::<Vec<_>>()
        });
        let rows_per_thread = n_out.div_ceil(nth);
        for row in ith * rows_per_thread..((ith + 1) * rows_per_thread).min(n_out) {
            output[row] = self.row(input_f32, row, lora_low.as_deref());
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
        for (col, value) in out.iter_mut().enumerate() {
            *value = self.value(token_id as usize, col);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
