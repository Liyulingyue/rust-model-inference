//! Borrowed weight enum `QuantizedTensor<'a>`: zero-copy from mmap'd GGUF.
//!
//! Each variant holds `&'a [u8]` (or `&'a [f32]` for F32) instead of owning
//! bytes, so models can mmap the GGUF file and dispatch matmul against
//! the borrowed bytes directly. The companion `QTensorOwned` (in
//! `qtensor_owned.rs`) is used for cases that need to own bytes — chiefly
//! `fuse_vstack` for FFN gate+up fusion.
//!
//! Construction: prefer `QuantizedTensor::from_bytes(bytes, ggml_type, ...)`
//! at the GGUF loader layer. Dispatch: `Kernel::forward_prequantized`
//! routes each variant to the matching SIMD kernel in `q4_0/` / `q4_k/`
//! etc.

use crate::core::tensor::GGMLType;
use crate::ops::kernel::Kernel;

/// F16 weight layout reserved for future use.
#[derive(Debug, Clone, Copy)]
pub struct F16Weight<'a> {
    pub bytes: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct BF16Weight<'a> {
    pub bytes: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

/// Unified enum of supported quantized weight formats. Produces a
/// `Box<dyn Kernel>` via [`QuantizedTensor::into_kernel`].
///
/// Note: Q4_K / Q5_K are listed for completeness (they are valid GGUF
/// formats and used by some Qwen3 checkpoints) but their Kernel impl
/// uses the dequantize-to-f32 path — they do not benefit from the
/// Q8_0-prequantized fast path.
pub enum QuantizedTensor<'a> {
    F32(Vec<f32>),
    F16(F16Weight<'a>),
    BF16(BF16Weight<'a>),
    Q8_0 { data: &'a [u8], n_cols: usize, n_rows: usize },
    Q6_K { data: &'a [u8], n_cols: usize, n_rows: usize },
    Q2_K { data: &'a [u8], n_cols: usize, n_rows: usize },
    Q3_K { data: &'a [u8], n_cols: usize, n_rows: usize },
    IQ4NL { data: &'a [u8], n_cols: usize, n_rows: usize },
    IQ2XXS { data: &'a [u8], n_cols: usize, n_rows: usize },
    IQ2S { data: &'a [u8], n_cols: usize, n_rows: usize },
    IQ2XS { data: &'a [u8], n_cols: usize, n_rows: usize },
    IQ3XXS { data: &'a [u8], n_cols: usize, n_rows: usize },
    IQ3S { data: &'a [u8], n_cols: usize, n_rows: usize },
    IQ4XS { data: &'a [u8], n_cols: usize, n_rows: usize },
    IQ1M { data: &'a [u8], n_cols: usize, n_rows: usize },
    IQ1S { data: &'a [u8], n_cols: usize, n_rows: usize },
    Q4_0 { data: &'a [u8], n_cols: usize, n_rows: usize },
    Q4_1 { data: &'a [u8], n_cols: usize, n_rows: usize },
    Q4_K { data: &'a [u8], n_cols: usize, n_rows: usize },
    Q5_K { data: &'a [u8], n_cols: usize, n_rows: usize },
}

impl<'a> crate::ops::kernel::Kernel for QuantizedTensor<'a> {
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
        // Bridge: delegate to the active Kernel in `into_kernel`. This
        // is only used when callers keep a `QuantizedTensor` directly
        // (rare; the common path is `into_kernel()` for `LayerWeights`).
        let k = self.clone_to_kernel();
        k.forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth);
    }

    fn forward_prepared(
        &self,
        input_f32: &[f32],
        input_q8: &[u8],
        input_scales: &[f32],
        q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        self.clone_to_kernel().forward_prepared(
            input_f32,
            input_q8,
            input_scales,
            q8_k,
            output,
            n_in,
            n_out,
            ith,
            nth,
        );
    }
}

impl<'a> QuantizedTensor<'a> {
    /// Helper for the `Kernel for QuantizedTensor` impl above: produces a
    /// `Box<dyn Kernel>` view of self. This is intentionally a separate
    /// method rather than inline because `into_kernel` consumes `self`.
    pub(crate) fn clone_to_kernel(&self) -> Box<dyn crate::ops::kernel::Kernel + 'a> {
        use crate::ops::kernel::{bf16, f16, f32, iq4_nl, iq4_xs, q2_k, q3_k, q4_0, q4_1, q4_k, q5_k, q6_k, q8_0};
        match self {
            Self::F32(slice) => Box::new(f32::F32Kernel::new(slice.clone())),
            Self::F16(w) => Box::new(f16::F16Kernel::new(w.bytes)),
            Self::BF16(w) => Box::new(bf16::BF16Kernel::new(w.bytes)),
            Self::Q8_0 { data, .. } => Box::new(q8_0::Q8Kernel::new(data)),
            Self::Q6_K { data, n_cols, n_rows } => {
                Box::new(q6_k::Q6_KKernel::new(data, *n_cols, *n_rows))
            }
            Self::Q2_K { data, n_cols, n_rows } => {
                Box::new(q2_k::Q2_KKernel::new(data, *n_cols, *n_rows))
            }
            Self::Q3_K { data, n_cols, n_rows } => {
                Box::new(q3_k::Q3_KKernel::new(data, *n_cols, *n_rows))
            }
            Self::IQ4NL { data, n_cols, n_rows } => {
                Box::new(iq4_nl::IQ4NLKernel::new(data, *n_cols, *n_rows))
            }
            Self::IQ2XXS { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ2XXSKernel::new(data, *n_cols, *n_rows))
            }
            Self::IQ2S { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ2SKernel::new(data, *n_cols, *n_rows))
            }
            Self::IQ2XS { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ2XSKernel::new(data, *n_cols, *n_rows))
            }
            Self::IQ3XXS { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ3XXSKernel::new(data, *n_cols, *n_rows))
            }
            Self::IQ3S { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ3SKernel::new(data, *n_cols, *n_rows))
            }
            Self::IQ4XS { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ4XSKernel::new(data, *n_cols, *n_rows))
            }
            Self::IQ1M { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ1MKernel::new(data, *n_cols, *n_rows))
            }
            Self::IQ1S { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ1SKernel::new(data, *n_cols, *n_rows))
            }
            Self::Q4_0 { data, n_cols, n_rows } => {
                Box::new(q4_0::Q4_0Kernel::new(data, *n_cols, *n_rows))
            }
            Self::Q4_1 { data, n_cols, n_rows } => {
                Box::new(q4_1::Q4_1Kernel::new(data, *n_cols, *n_rows))
            }
            Self::Q4_K { data, n_cols, n_rows } => {
                Box::new(q4_k::Q4_KKernel::new(data, *n_cols, *n_rows))
            }
            Self::Q5_K { data, n_cols, n_rows } => {
                Box::new(q5_k::Q5_KKernel::new(data, *n_cols, *n_rows))
            }
        }
    }
}

impl<'a> QuantizedTensor<'a> {
    /// Build a `QuantizedTensor` from raw GGUF bytes. This is the bridge
    /// from the GGUF loader to the `Kernel` trait and replaces the previous
    /// `ProcessedWeight::from_bytes` API.
    pub fn from_bytes(data: &'a [u8], ggml_type: GGMLType, n_in: usize, n_out: usize) -> Self {
        match ggml_type {
            GGMLType::F32 => {
                let f32_data: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Self::F32(f32_data)
            }
            GGMLType::F16 => Self::F16(F16Weight { bytes: data, n_in, n_out }),
            GGMLType::BF16 => Self::BF16(BF16Weight { bytes: data, n_in, n_out }),
            GGMLType::Q8_0 => Self::Q8_0 { data, n_cols: n_in, n_rows: n_out },
            GGMLType::Q6K => Self::Q6_K { data, n_cols: n_in, n_rows: n_out },
            GGMLType::Q2K => Self::Q2_K { data, n_cols: n_in, n_rows: n_out },
            GGMLType::Q3K => Self::Q3_K { data, n_cols: n_in, n_rows: n_out },
            GGMLType::IQ4_NL => Self::IQ4NL { data, n_cols: n_in, n_rows: n_out },
            GGMLType::IQ2_XXS => Self::IQ2XXS { data, n_cols: n_in, n_rows: n_out },
            GGMLType::IQ2_S => Self::IQ2S { data, n_cols: n_in, n_rows: n_out },
            GGMLType::IQ2_XS => Self::IQ2XS { data, n_cols: n_in, n_rows: n_out },
            GGMLType::IQ3_XXS => Self::IQ3XXS { data, n_cols: n_in, n_rows: n_out },
            GGMLType::IQ3_S => Self::IQ3S { data, n_cols: n_in, n_rows: n_out },
            GGMLType::IQ4_XS => Self::IQ4XS { data, n_cols: n_in, n_rows: n_out },
            GGMLType::IQ1_M => Self::IQ1M { data, n_cols: n_in, n_rows: n_out },
            GGMLType::IQ1_S => Self::IQ1S { data, n_cols: n_in, n_rows: n_out },
            GGMLType::Q4_0 => Self::Q4_0 { data, n_cols: n_in, n_rows: n_out },
            GGMLType::Q4_1 => Self::Q4_1 { data, n_cols: n_in, n_rows: n_out },
            GGMLType::Q4K => Self::Q4_K { data, n_cols: n_in, n_rows: n_out },
            GGMLType::Q5K => Self::Q5_K { data, n_cols: n_in, n_rows: n_out },
            _ => panic!("unsupported weight type {:?} - use Q8_0 model", ggml_type),
        }
    }

    pub fn ggml_type(&self) -> GGMLType {
        match self {
            Self::F32(_) => GGMLType::F32,
            Self::F16(_) => GGMLType::F16,
            Self::BF16(_) => GGMLType::BF16,
            Self::Q8_0 { .. } => GGMLType::Q8_0,
            Self::Q6_K { .. } => GGMLType::Q6K,
            Self::Q2_K { .. } => GGMLType::Q2K,
            Self::Q3_K { .. } => GGMLType::Q3K,
            Self::IQ4NL { .. } => GGMLType::IQ4_NL,
            Self::IQ2XXS { .. } => GGMLType::IQ2_XXS,
            Self::IQ2S { .. } => GGMLType::IQ2_S,
            Self::IQ2XS { .. } => GGMLType::IQ2_XS,
            Self::IQ3XXS { .. } => GGMLType::IQ3_XXS,
            Self::IQ3S { .. } => GGMLType::IQ3_S,
            Self::IQ4XS { .. } => GGMLType::IQ4_XS,
            Self::IQ1M { .. } => GGMLType::IQ1_M,
            Self::IQ1S { .. } => GGMLType::IQ1_S,
            Self::Q4_0 { .. } => GGMLType::Q4_0,
            Self::Q4_1 { .. } => GGMLType::Q4_1,
            Self::Q4_K { .. } => GGMLType::Q4K,
            Self::Q5_K { .. } => GGMLType::Q5K,
        }
    }

    pub fn n_in(&self) -> usize {
        match self {
            Self::F32(slice) => slice.len(),
            Self::F16(w) => w.n_in,
            Self::BF16(w) => w.n_in,
            Self::Q8_0 { n_cols, .. } => *n_cols,
            Self::Q6_K { n_cols, .. } => *n_cols,
            Self::Q2_K { n_cols, .. } => *n_cols,
            Self::Q3_K { n_cols, .. } => *n_cols,
            Self::IQ4NL { n_cols, .. } => *n_cols,
            Self::IQ2XXS { n_cols, .. } => *n_cols,
            Self::IQ2S { n_cols, .. } => *n_cols,
            Self::IQ2XS { n_cols, .. } => *n_cols,
            Self::IQ3XXS { n_cols, .. } => *n_cols,
            Self::IQ3S { n_cols, .. } => *n_cols,
            Self::IQ4XS { n_cols, .. } => *n_cols,
            Self::IQ1M { n_cols, .. } => *n_cols,
            Self::IQ1S { n_cols, .. } => *n_cols,
            Self::Q4_0 { n_cols, .. } => *n_cols,
            Self::Q4_1 { n_cols, .. } => *n_cols,
            Self::Q4_K { n_cols, .. } => *n_cols,
            Self::Q5_K { n_cols, .. } => *n_cols,
        }
    }

    /// Build a `Box<dyn Kernel>` from this weight tensor.
    pub fn into_kernel(self) -> Box<dyn crate::ops::kernel::Kernel + 'a> {
        use crate::ops::kernel::{bf16, f16, f32, iq4_nl, iq4_xs, q2_k, q3_k, q4_0, q4_1, q4_k, q5_k, q6_k, q8_0};
        match self {
            Self::F32(slice) => Box::new(f32::F32Kernel::new(slice)),
            Self::F16(w) => Box::new(f16::F16Kernel::new(w.bytes)),
            Self::BF16(w) => Box::new(bf16::BF16Kernel::new(w.bytes)),
            Self::Q8_0 { data, .. } => Box::new(q8_0::Q8Kernel::new(data)),
            Self::Q6_K { data, n_cols, n_rows } => {
                Box::new(q6_k::Q6_KKernel::new(data, n_cols, n_rows))
            }
            Self::Q2_K { data, n_cols, n_rows } => {
                Box::new(q2_k::Q2_KKernel::new(data, n_cols, n_rows))
            }
            Self::Q3_K { data, n_cols, n_rows } => {
                Box::new(q3_k::Q3_KKernel::new(data, n_cols, n_rows))
            }
            Self::IQ4NL { data, n_cols, n_rows } => {
                Box::new(iq4_nl::IQ4NLKernel::new(data, n_cols, n_rows))
            }
            Self::IQ2XXS { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ2XXSKernel::new(data, n_cols, n_rows))
            }
            Self::IQ2S { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ2SKernel::new(data, n_cols, n_rows))
            }
            Self::IQ2XS { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ2XSKernel::new(data, n_cols, n_rows))
            }
            Self::IQ3XXS { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ3XXSKernel::new(data, n_cols, n_rows))
            }
            Self::IQ3S { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ3SKernel::new(data, n_cols, n_rows))
            }
            Self::IQ4XS { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ4XSKernel::new(data, n_cols, n_rows))
            }
            Self::IQ1M { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ1MKernel::new(data, n_cols, n_rows))
            }
            Self::IQ1S { data, n_cols, n_rows } => {
                Box::new(iq4_xs::IQ1SKernel::new(data, n_cols, n_rows))
            }
            Self::Q4_0 { data, n_cols, n_rows } => {
                Box::new(q4_0::Q4_0Kernel::new(data, n_cols, n_rows))
            }
            Self::Q4_1 { data, n_cols, n_rows } => {
                Box::new(q4_1::Q4_1Kernel::new(data, n_cols, n_rows))
            }
            Self::Q4_K { data, n_cols, n_rows } => {
                Box::new(q4_k::Q4_KKernel::new(data, n_cols, n_rows))
            }
            Self::Q5_K { data, n_cols, n_rows } => {
                Box::new(q5_k::Q5_KKernel::new(data, n_cols, n_rows))
            }
        }
    }

    pub fn n_rows(&self) -> usize {
        match self {
            Self::F32(values) => usize::from(!values.is_empty()),
            Self::F16(weight) => weight.n_out,
            Self::BF16(weight) => weight.n_out,
            Self::Q8_0 { n_rows, .. } => *n_rows,
            Self::Q6_K { n_rows, .. }
            | Self::Q2_K { n_rows, .. }
            | Self::Q3_K { n_rows, .. }
            | Self::IQ4NL { n_rows, .. }
            | Self::IQ2XXS { n_rows, .. }
            | Self::IQ2S { n_rows, .. }
            | Self::IQ2XS { n_rows, .. }
            | Self::IQ3XXS { n_rows, .. }
            | Self::IQ3S { n_rows, .. }
            | Self::IQ4XS { n_rows, .. }
            | Self::IQ1M { n_rows, .. }
            | Self::IQ1S { n_rows, .. }
            | Self::Q4_0 { n_rows, .. }
            | Self::Q4_1 { n_rows, .. }
            | Self::Q4_K { n_rows, .. }
            | Self::Q5_K { n_rows, .. } => *n_rows,
        }
    }

    pub fn matmul(&self, input: &[f32]) -> Vec<f32> {
        let n_rows = self.n_rows();
        let mut output = vec![0.0; n_rows];
        let mut input_q8 = vec![0u8; input.len()];
        let mut input_scales = vec![0.0; input.len().div_ceil(32)];
        crate::ops::quantize_q8_0_into(
            input,
            input.len(),
            &mut input_q8,
            &mut input_scales,
        );
        self.forward_prepared(
            input,
            &input_q8,
            &input_scales,
            None,
            &mut output,
            input.len(),
            n_rows,
            0,
            1,
        );
        output
    }

    pub fn quantize_and_matmul(
        &self,
        input: &[f32],
        q8k_buf: &mut [crate::ops::quant::BlockQ8K],
        output: &mut [f32],
    ) {
        let mut q8_buf = vec![0u8; input.len()];
        let mut scale_buf = vec![0.0f32; input.len().div_ceil(32)];
        let pool = crate::core::thread_pool::ComputePool::new(1);
        self.quantize_and_matmul_with_scratch(
            input,
            q8k_buf,
            &mut q8_buf,
            &mut scale_buf,
            output,
            &pool,
        );
    }

    pub fn quantize_and_matmul_with_scratch(
        &self,
        input: &[f32],
        q8k_buf: &mut [crate::ops::quant::BlockQ8K],
        q8_buf: &mut [u8],
        scale_buf: &mut [f32],
        output: &mut [f32],
        pool: &crate::core::thread_pool::ComputePool,
    ) {
        let n_rows = self.n_rows();
        let output_ptr = output.as_mut_ptr();
        match self {
            Self::Q4_K { n_cols, .. }
            | Self::Q5_K { n_cols, .. }
            | Self::Q6_K { n_cols, .. }
            | Self::Q2_K { n_cols, .. }
            | Self::Q3_K { n_cols, .. }
            | Self::IQ2XXS { n_cols, .. }
            | Self::IQ2S { n_cols, .. }
            | Self::IQ2XS { n_cols, .. }
            | Self::IQ3XXS { n_cols, .. }
            | Self::IQ3S { n_cols, .. }
            | Self::IQ4NL { n_cols, .. }
            | Self::IQ4XS { n_cols, .. }
            | Self::IQ1M { n_cols, .. }
            | Self::IQ1S { n_cols, .. } => {
                let blocks = *n_cols / crate::ops::quant::QK_K;
                crate::ops::quant::quantize_row_q8_k_into(input, &mut q8k_buf[..blocks]);
                pool.compute(|ith, nth| {
                    let output = unsafe { std::slice::from_raw_parts_mut(output_ptr, n_rows) };
                    self.forward_prepared(
                        input,
                        q8_buf,
                        scale_buf,
                        Some(&q8k_buf[..blocks]),
                        output,
                        *n_cols,
                        n_rows,
                        ith,
                        nth,
                    );
                });
            }
            _ => {
                let blocks = input.len().div_ceil(32);
                crate::ops::quantize_q8_0_into(
                    input,
                    input.len(),
                    &mut q8_buf[..input.len()],
                    &mut scale_buf[..blocks],
                );
                pool.compute(|ith, nth| {
                    let output = unsafe { std::slice::from_raw_parts_mut(output_ptr, n_rows) };
                    self.forward_prepared(
                        input,
                        &q8_buf[..input.len()],
                        &scale_buf[..blocks],
                        None,
                        output,
                        input.len(),
                        n_rows,
                        ith,
                        nth,
                    );
                });
            }
        }
    }
}

#[inline]
fn q8_0_block_count(bytes: &[u8]) -> usize {
    bytes.len() / 34 // 2-byte F16 scale + 32-byte data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::GGMLType;

    #[test]
    fn q8_0_block_count_basic() {
        assert_eq!(q8_0_block_count(&vec![0u8; 34]), 1);
        assert_eq!(q8_0_block_count(&vec![0u8; 68]), 2);
        assert_eq!(q8_0_block_count(&vec![0u8; 102]), 3);
    }

    #[test]
    fn f16_weight_layout_compiles() {
        let data = vec![0u8; 64];
        let w = F16Weight { bytes: &data, n_in: 32, n_out: 2 };
        assert_eq!(w.n_in, 32);
        assert_eq!(w.n_out, 2);
        assert_eq!(w.bytes.len(), 64);
    }

    #[test]
    fn quantized_tensor_ggml_type_discriminator() {
        let f32_slice = vec![0.0f32; 32];
        let q = QuantizedTensor::F32(f32_slice);
        assert_eq!(q.ggml_type(), GGMLType::F32);

        let q8_bytes = vec![0u8; 34];
        let q = QuantizedTensor::Q8_0 { data: &q8_bytes, n_cols: 32, n_rows: 1 };
        assert_eq!(q.ggml_type(), GGMLType::Q8_0);
        assert_eq!(q.n_in(), 32);
    }

    /// After `Q*KWeight<'a>` newtype removal, the K-quant variants are
    /// struct variants (matching `Q8_0` style). Verify each variant
    /// round-trips through `from_bytes` → `ggml_type` → `into_kernel`.
    #[test]
    fn k_quant_variants_construct_and_dispatch() {
        use crate::ops::quant::{
            BLOCK_Q2K_SIZE, BLOCK_Q3K_SIZE, BLOCK_Q4K_SIZE, BLOCK_Q5K_SIZE, BLOCK_Q6K_SIZE,
            BLOCK_Q80_SIZE,
        };
        // Q4_1 block = 20 bytes (F16 scale + F16 min + 16 nibbles).
        const BLOCK_Q4_1_SIZE: usize = 20;
        // (ggml_type, n_cols, n_rows, block_bytes)
        let cases: &[(GGMLType, usize, usize, usize)] = &[
            (GGMLType::Q4_0, 32, 4, BLOCK_Q80_SIZE),
            (GGMLType::Q4_1, 32, 4, BLOCK_Q4_1_SIZE),
            (GGMLType::Q2K, 256, 2, BLOCK_Q2K_SIZE),
            (GGMLType::Q3K, 256, 2, BLOCK_Q3K_SIZE),
            (GGMLType::Q4K, 256, 2, BLOCK_Q4K_SIZE),
            (GGMLType::Q5K, 256, 2, BLOCK_Q5K_SIZE),
            (GGMLType::Q6K, 256, 2, BLOCK_Q6K_SIZE),
        ];
        for &(ggml_type, n_cols, n_rows, block_bytes) in cases {
            let blocks_per_row = n_cols / 32;
            let bytes = vec![0u8; n_rows * blocks_per_row * block_bytes];
            let q = QuantizedTensor::from_bytes(&bytes, ggml_type, n_cols, n_rows);
            assert_eq!(q.ggml_type(), ggml_type, "ggml_type round-trip for {:?}", ggml_type);
            assert_eq!(q.n_in(), n_cols, "n_in for {:?}", ggml_type);

            // into_kernel must produce a kernel that we can call without panicking.
            let kernel = q.into_kernel();
            let input_q8 = vec![0u8; n_cols];
            let input_scales = vec![0.0f32; (n_cols + 31) / 32];
            let mut out = vec![0.0f32; n_rows];
            kernel.forward_prequantized(
                &input_q8, &input_scales, &mut out, n_cols, n_rows, 0, 1,
            );
        }
    }
}
