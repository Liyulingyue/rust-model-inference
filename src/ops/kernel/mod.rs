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

pub mod f16;
pub mod f32;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;

mod trait_;
mod quantized_tensor;
mod qtensor_owned;

/// A model weight whose concrete kernel is selected once at load time.
///
/// The kernel retains the borrowed GGUF bytes, so wrapping a
/// [`QuantizedTensor`] keeps the mmap-backed zero-copy representation.
pub struct Weight<'a> {
	pub kernel: Box<dyn Kernel + 'a>,
}

impl<'a> Weight<'a> {
	pub fn from_quantized(tensor: QuantizedTensor<'a>) -> Self {
		Self {
			kernel: tensor.into_kernel(),
		}
	}

	pub fn quantize_and_matmul_with_scratch(
		&self,
		input: &[f32],
		q8k_buf: &mut [crate::ops::quant::BlockQ8K],
		q8_buf: &mut [u8],
		scale_buf: &mut [f32],
		output: &mut [f32],
		n_out: usize,
		pool: &crate::core::thread_pool::ComputePool,
	) {
		let n_in = input.len();
		let q8_blocks = n_in.div_ceil(32);
		let q8k_blocks = n_in.div_ceil(crate::ops::quant::QK_K);
		crate::ops::quantize_q8_0_into(input, n_in, &mut q8_buf[..n_in], &mut scale_buf[..q8_blocks]);
		crate::ops::quant::quantize_row_q8_k_into(input, &mut q8k_buf[..q8k_blocks]);

		let output_ptr = output.as_mut_ptr();
		pool.compute(|ith, nth| {
			let output = unsafe { std::slice::from_raw_parts_mut(output_ptr, n_out) };
			self.kernel.forward_prepared(
				input,
				&q8_buf[..n_in],
				&scale_buf[..q8_blocks],
				Some(&q8k_buf[..q8k_blocks]),
				output,
				n_in,
				n_out,
				ith,
				nth,
			);
		});
	}

	pub fn matmul(&self, input: &[f32], n_out: usize) -> Vec<f32> {
		let mut output = vec![0.0; n_out];
		self.kernel.forward(input, &mut output, input.len(), n_out);
		output
	}
}


// Re-exports for convenient access from `ops::kernel::*`.
pub use trait_::Kernel;
pub use quantized_tensor::{F16Weight, QuantizedTensor};
/// Reserved owned-weight representation. Prefer `QuantizedTensor` unless a
/// transform or an independent lifetime requires materialized weight data.
pub use qtensor_owned::QTensorOwned;
