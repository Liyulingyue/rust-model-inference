//! Kernel trait + QuantizedTensor enum: unified matmul dispatch.
//!
//! Phase 2.7-final: `ProcessedWeight` (in `ops::matmul`) has been replaced by
//! `Box<dyn Kernel>` everywhere. The Kernel trait is the single point of
//! dispatch for quantized matmul; `QuantizedTensor` is the enum that names
//! each supported weight format and produces a `Box<dyn Kernel>`.
//!
//! Design (after the Phase 2.7 cleanup):
//! - `forward_prequantized(input_q8, scales, output, n_in, n_out, ith, nth)`
//!   is the hot-path method. Input is pre-quantized to Q8_0 blocks. The
//!   kernel produces `output[my_rows]` where `my_rows` is the
//!   `[ith, nth)` partition of `n_out` rows. Pass `ith=0, nth=1` for the
//!   scalar single-token path.
//! - `forward(input, output, n_in, n_out)` is a convenience that quantizes
//!   the f32 input and calls `forward_prequantized(..., 0, 1)` internally.
//!   Use this in tests and small one-off callers; the production hot path
//!   uses `forward_prequantized` directly to avoid the per-call allocation.

pub mod f16;
pub mod f32;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;

use crate::core::tensor::GGMLType;

/// Core matmul kernel interface.
pub trait Kernel: Send + Sync {
    /// Hot-path matmul: pre-quantized Q8_0 input, partitioned by row.
    ///
    /// Each call computes `output[i] = sum_k weight[i, k] * dequant(input_q8[k], input_scales[k/32])`
    /// for `i` in `[ith * per_thread, min((ith + 1) * per_thread, n_out))`.
    /// For a scalar single-token call, pass `ith = 0, nth = 1`.
    fn forward_prequantized(
        &self,
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    );

    /// Matmul with both the original F32 input and its Q8_0 view available.
    /// K-quant kernels override this to prepare Q8_K activations; all other
    /// kernels retain the existing Q8_0 path.
    ///
    /// `q8_k` lets the caller pass a pre-quantized Q8_K activation (shared
    /// across threads) instead of letting each thread re-quantize internally.
    /// Pass `None` for kernels that don't need it (Q8_0 path) or when the
    /// caller has not pre-quantized.
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
        let _ = input_f32;
        let _ = q8_k;
        self.forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth);
    }

    /// Convenience: f32 input, single-thread. Default impl quantizes the
    /// input to Q8_0 and delegates to `forward_prequantized`. Kernels that
    /// have a native f32-input path (e.g. F16) override this.
    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        let mut input_q8 = vec![0u8; n_in];
        let mut input_scales = vec![0.0f32; n_in.div_ceil(32)];
        crate::ops::quantize_q8_0_into(input, n_in, &mut input_q8, &mut input_scales);
        self.forward_prepared(
            input,
            &input_q8,
            &input_scales,
            None,
            output,
            n_in,
            n_out,
            0,
            1,
        );
    }

    /// Batched matmul: `input[n_tokens * n_in] → output[n_tokens * n_out]`.
    /// Default impl quantizes the whole batch up front then loops over tokens.
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
        let mut input_q8 = vec![0u8; n_in];
        let mut input_scales = vec![0.0f32; n_in.div_ceil(32)];
        for t in 0..n_tokens {
            crate::ops::quantize_q8_0_into(
                &input[t * n_in..(t + 1) * n_in],
                n_in,
                &mut input_q8,
                &mut input_scales,
            );
            self.forward_prepared(
                &input[t * n_in..(t + 1) * n_in],
                &input_q8,
                &input_scales,
                None,
                &mut output[t * n_out..(t + 1) * n_out],
                n_in,
                n_out,
                0,
                1,
            );
        }
    }
}

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
    fn clone_to_kernel(&self) -> Box<dyn Kernel + 'a> {
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
    pub fn into_kernel(self) -> Box<dyn Kernel + 'a> {
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
