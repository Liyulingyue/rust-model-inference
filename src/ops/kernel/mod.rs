//! Matmul kernel abstraction.
//!
//! Module layout:
//! - `Kernel` trait — per-dtype matmul dispatch interface (`trait.rs`)
//! - `QuantizedTensor<'a>` — borrowed weight enum, zero-copy from mmap
//!   (`quantized_tensor.rs`)
//! - `QTensorOwned` — reserved owned weight enum for fuse / batch /
//!   weight-side transforms (`qtensor_owned.rs`); do not use for ordinary
//!   model weights because loading it copies the tensor bytes
//! - per-dtype SIMD kernels: `f16`, `f32`, `q4_0`, `q4_1`, `q4_k`, `q5_k`,
//!   `q6_k`, `q8_0`
//!
//! Hot path: `Kernel::forward_prequantized` → per-dtype SIMD kernel.
//! Fuse path (FFN gate+up, attention QKV): `QTensorOwned::fuse_vstack`.
//!
//! Design rule (R1): a single weight enum (`QuantizedTensor<'a>`) is used
//! on the hot path; `QTensorOwned` exists only for cases that need to own
//! weight bytes. Future sharded / distributed inference should add a
//! `Sharded` variant to `QuantizedTensor<'a>` rather than introduce a new
//! top-level type.

pub mod bf16;
pub mod f16;
pub mod f32;
pub mod iq4_nl;
pub mod iq4_xs;
pub mod q2_k;
pub mod q3_k;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;

mod qtensor_owned;
mod quantized_tensor;
mod trait_;

/// A model weight whose concrete kernel is selected once at load time.
///
/// The kernel retains the borrowed GGUF bytes, so wrapping a
/// [`QuantizedTensor`] keeps the mmap-backed zero-copy representation.
///
/// The metadata fields (`ggml_type`, `n_in`, `n_out`) are populated once
/// at construction time and enable introspection (logging, validation,
/// pre-fuse / pre-transform assertions) without re-dispatching through
/// the kernel. They do **not** enable fuse or weight-side transforms —
/// those still require raw byte access via [`QuantizedTensor`] or
/// [`QTensorOwned`].
pub struct Weight<'a> {
    pub kernel: Box<dyn Kernel + 'a>,
    pub ggml_type: crate::core::tensor::GGMLType,
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> Weight<'a> {
    pub fn from_quantized(tensor: QuantizedTensor<'a>) -> Self {
        let ggml_type = tensor.ggml_type();
        let n_in = tensor.n_in();
        let n_out = tensor.n_rows();
        Self {
            kernel: tensor.into_kernel(),
            ggml_type,
            n_in,
            n_out,
        }
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
        use crate::core::tensor::GGMLType;

        let n_in = input.len();
        let n_out = self.n_out;
        let (input_q8, input_scales, q8_k) = match self.ggml_type {
            GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 => (&[][..], &[][..], None),
            GGMLType::Q4_0 | GGMLType::Q4_1 | GGMLType::Q8_0 => {
                let blocks = n_in.div_ceil(32);
                crate::ops::quantize_q8_0_into(
                    input,
                    n_in,
                    &mut q8_buf[..n_in],
                    &mut scale_buf[..blocks],
                );
                (&q8_buf[..n_in], &scale_buf[..blocks], None)
            }
            GGMLType::Q2K
            | GGMLType::Q3K
            | GGMLType::Q4K
            | GGMLType::Q5K
            | GGMLType::Q6K
            | GGMLType::IQ1_S
            | GGMLType::IQ1_M
            | GGMLType::IQ2_XXS
            | GGMLType::IQ2_XS
            | GGMLType::IQ2_S
            | GGMLType::IQ3_XXS
            | GGMLType::IQ3_S
            | GGMLType::IQ4_NL
            | GGMLType::IQ4_XS => {
                let blocks = n_in / crate::ops::quant::QK_K;
                crate::ops::quant::quantize_row_q8_k_into(input, &mut q8k_buf[..blocks]);
                (&[][..], &[][..], Some(&q8k_buf[..blocks]))
            }
            other => panic!("prepared matmul does not support {other:?}"),
        };

        let output_ptr = output.as_mut_ptr();
        pool.compute(|ith, nth| {
            let output = unsafe { std::slice::from_raw_parts_mut(output_ptr, n_out) };
            self.kernel.forward_prepared(
                input,
                input_q8,
                input_scales,
                q8_k,
                output,
                n_in,
                n_out,
                ith,
                nth,
            );
        });
    }

    pub fn matmul(&self, input: &[f32]) -> Vec<f32> {
        let n_out = self.n_out;
        let mut output = vec![0.0; n_out];
        self.kernel.forward(input, &mut output, input.len(), n_out);
        output
    }

    pub fn embedding_lookup(&self, token_id: u32, out: &mut [f32]) {
        self.kernel.embedding_lookup(token_id, self.n_in, out);
    }
}

// Re-exports for convenient access from `ops::kernel::*`.
/// Reserved owned-weight representation. Prefer `QuantizedTensor` unless a
/// transform or an independent lifetime requires materialized weight data.
pub use qtensor_owned::QTensorOwned;
pub use quantized_tensor::{F16Weight, QuantizedTensor};
pub use trait_::Kernel;
