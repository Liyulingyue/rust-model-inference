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

/// F16 weight layout reserved for future use.
#[derive(Debug, Clone, Copy)]
pub struct F16Weight<'a> {
    pub bytes: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

/// Concrete weight types for each Kernel impl.
pub use crate::ops::kernel::q4_0::Q4_0Weight;
pub use crate::ops::kernel::q4_1::Q4_1Weight;
pub use crate::ops::kernel::q6_k::Q6_KWeight;
pub use crate::ops::kernel::q4_k::Q4_KWeight;
pub use crate::ops::kernel::q5_k::Q5_KWeight;

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
    Q8_0(&'a [u8]),
    Q6_K(Q6_KWeight<'a>),
    Q4_0(Q4_0Weight<'a>),
    Q4_1(Q4_1Weight<'a>),
    Q4_K(Q4_KWeight<'a>),
    Q5_K(Q5_KWeight<'a>),
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
        use crate::ops::kernel::{f16, f32, q4_0, q4_1, q4_k, q5_k, q6_k, q8_0};
        match self {
            Self::F32(slice) => Box::new(f32::F32Kernel::new(slice.clone())),
            Self::F16(w) => Box::new(f16::F16Kernel::new(w.bytes)),
            Self::Q8_0(bytes) => Box::new(q8_0::Q8Kernel::new(bytes)),
            Self::Q6_K(w) => Box::new(q6_k::Q6_KKernel::new(w.data)),
            Self::Q4_0(w) => Box::new(q4_0::Q4_0Kernel::new(w.data)),
            Self::Q4_1(w) => Box::new(q4_1::Q4_1Kernel::new(w.data)),
            Self::Q4_K(w) => Box::new(q4_k::Q4_KKernel::new(*w)),
            Self::Q5_K(w) => Box::new(q5_k::Q5_KKernel::new(*w)),
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
            GGMLType::Q8_0 => Self::Q8_0(data),
            GGMLType::Q6K => Self::Q6_K(Q6_KWeight { data, n_in, n_out }),
            GGMLType::Q4_0 => Self::Q4_0(Q4_0Weight { data, n_in, n_out }),
            GGMLType::Q4_1 => Self::Q4_1(Q4_1Weight { data, n_in, n_out }),
            GGMLType::Q4K => Self::Q4_K(Q4_KWeight { data, n_in, n_out }),
            GGMLType::Q5K => Self::Q5_K(Q5_KWeight { data, n_in, n_out }),
            _ => panic!("unsupported weight type {:?} - use Q8_0 model", ggml_type),
        }
    }

    pub fn ggml_type(&self) -> GGMLType {
        match self {
            Self::F32(_) => GGMLType::F32,
            Self::F16(_) => GGMLType::F16,
            Self::Q8_0(_) => GGMLType::Q8_0,
            Self::Q6_K(_) => GGMLType::Q6K,
            Self::Q4_0(_) => GGMLType::Q4_0,
            Self::Q4_1(_) => GGMLType::Q4_1,
            Self::Q4_K(_) => GGMLType::Q4K,
            Self::Q5_K(_) => GGMLType::Q5K,
        }
    }

    pub fn n_in(&self) -> usize {
        match self {
            Self::F32(slice) => slice.len(),
            Self::F16(w) => w.n_in,
            Self::Q8_0(bytes) => q8_0_block_count(*bytes) * 32,
            Self::Q6_K(w) => w.n_in,
            Self::Q4_0(w) => w.n_in,
            Self::Q4_1(w) => w.n_in,
            Self::Q4_K(w) => w.n_in,
            Self::Q5_K(w) => w.n_in,
        }
    }

    /// Build a `Box<dyn Kernel>` from this weight tensor.
    pub fn into_kernel(self) -> Box<dyn crate::ops::kernel::Kernel + 'a> {
        use crate::ops::kernel::{f16, f32, q4_0, q4_1, q4_k, q5_k, q6_k, q8_0};
        match self {
            Self::F32(slice) => Box::new(f32::F32Kernel::new(slice)),
            Self::F16(w) => Box::new(f16::F16Kernel::new(w.bytes)),
            Self::Q8_0(bytes) => Box::new(q8_0::Q8Kernel::new(bytes)),
            Self::Q6_K(w) => Box::new(q6_k::Q6_KKernel::new(w.data)),
            Self::Q4_0(w) => Box::new(q4_0::Q4_0Kernel::new(w.data)),
            Self::Q4_1(w) => Box::new(q4_1::Q4_1Kernel::new(w.data)),
            Self::Q4_K(w) => Box::new(q4_k::Q4_KKernel::new(w)),
            Self::Q5_K(w) => Box::new(q5_k::Q5_KKernel::new(w)),
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
        let q = QuantizedTensor::Q8_0(&q8_bytes);
        assert_eq!(q.ggml_type(), GGMLType::Q8_0);
        assert_eq!(q.n_in(), 32);
    }
}
