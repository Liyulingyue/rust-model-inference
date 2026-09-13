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

pub(crate) struct PreparedRows {
    max_rows: usize,
    max_n_in: usize,
    rows: usize,
    n_in: usize,
    need_q8: bool,
    need_q8k: bool,
    q8: Vec<u8>,
    scales: Vec<f32>,
    q8k: Vec<crate::ops::quant::BlockQ8K>,
}

impl PreparedRows {
    pub(crate) fn new(max_rows: usize, max_n_in: usize) -> Self {
        Self {
            max_rows,
            max_n_in,
            rows: 0,
            n_in: 0,
            need_q8: false,
            need_q8k: false,
            q8: Vec::new(),
            scales: Vec::new(),
            q8k: Vec::new(),
        }
    }

    pub(crate) fn prepare(
        &mut self,
        input: &[f32],
        rows: usize,
        n_in: usize,
        need_q8: bool,
        need_q8k: bool,
    ) -> Result<(), String> {
        if rows == 0 || n_in == 0 {
            return Err("prepared rows require non-zero rows and width".into());
        }
        if rows > self.max_rows || n_in > self.max_n_in {
            return Err(format!(
                "prepared rows shape {rows}x{n_in} exceeds maximum {}x{}",
                self.max_rows, self.max_n_in
            ));
        }
        let input_len = rows
            .checked_mul(n_in)
            .ok_or("prepared rows input shape overflow")?;
        if input.len() != input_len {
            return Err(format!(
                "prepared rows input length mismatch: expected {input_len}, got {}",
                input.len()
            ));
        }
        if need_q8k && !n_in.is_multiple_of(crate::ops::quant::QK_K) {
            return Err(format!(
                "Q8_K activation width {n_in} must be divisible by {}",
                crate::ops::quant::QK_K
            ));
        }

        if need_q8 {
            let blocks = n_in.div_ceil(32);
            self.q8.resize(input_len, 0);
            self.scales.resize(rows * blocks, 0.0);
            for row in 0..rows {
                crate::ops::quantize_q8_0_into(
                    &input[row * n_in..(row + 1) * n_in],
                    n_in,
                    &mut self.q8[row * n_in..(row + 1) * n_in],
                    &mut self.scales[row * blocks..(row + 1) * blocks],
                );
            }
        }
        if need_q8k {
            let blocks = n_in / crate::ops::quant::QK_K;
            self.q8k.resize(
                rows * blocks,
                crate::ops::quant::BlockQ8K {
                    d: 0.0,
                    qs: [0; crate::ops::quant::QK_K],
                    bsums: [0; crate::ops::quant::QK_K / 16],
                },
            );
            for row in 0..rows {
                crate::ops::quant::quantize_row_q8_k_into(
                    &input[row * n_in..(row + 1) * n_in],
                    &mut self.q8k[row * blocks..(row + 1) * blocks],
                );
            }
        }

        self.rows = rows;
        self.n_in = n_in;
        self.need_q8 = need_q8;
        self.need_q8k = need_q8k;
        Ok(())
    }

    pub(crate) fn matmul(
        &self,
        weight: &Weight<'_>,
        input: &[f32],
        output: &mut [f32],
        pool: &crate::core::thread_pool::ComputePool,
    ) -> Result<(), String> {
        self.matmul_group(input, [(weight, output)], pool)
    }

    pub(crate) fn matmul_group<const N: usize>(
        &self,
        input: &[f32],
        projections: [(&Weight<'_>, &mut [f32]); N],
        pool: &crate::core::thread_pool::ComputePool,
    ) -> Result<(), String> {
        let input_len = self.rows * self.n_in;
        if self.rows == 0 || input.len() != input_len {
            return Err("prepared matmul input shape mismatch".into());
        }
        for (weight, output) in &projections {
            if weight.n_in != self.n_in {
                return Err("prepared rows do not match weight input width".into());
            }
            if (weight.needs_q8_0_activation() && !self.need_q8)
                || (weight.uses_q8_k() && !self.need_q8k)
            {
                return Err("prepared activation format does not match weight".into());
            }
            if output.len() != self.rows * weight.n_out {
                return Err("prepared matmul output shape mismatch".into());
            }
        }

        let blocks = self.n_in.div_ceil(32);
        let q8k_blocks = self.n_in / crate::ops::quant::QK_K;
        let projections = projections.map(|(weight, output)| (weight, output.as_mut_ptr()));
        pool.compute(|ith, nth| {
            for row in 0..self.rows {
                let input_row = &input[row * self.n_in..(row + 1) * self.n_in];
                let q8 = if self.need_q8 {
                    &self.q8[row * self.n_in..(row + 1) * self.n_in]
                } else {
                    &[]
                };
                let scales = if self.need_q8 {
                    &self.scales[row * blocks..(row + 1) * blocks]
                } else {
                    &[]
                };
                for (weight, output_ptr) in projections {
                    let q8k = weight
                        .uses_q8_k()
                        .then(|| &self.q8k[row * q8k_blocks..(row + 1) * q8k_blocks]);
                    let output_row = unsafe {
                        std::slice::from_raw_parts_mut(
                            output_ptr.add(row * weight.n_out),
                            weight.n_out,
                        )
                    };
                    weight.kernel.forward_prepared(
                        input_row,
                        q8,
                        scales,
                        q8k,
                        output_row,
                        self.n_in,
                        weight.n_out,
                        ith,
                        nth,
                    );
                }
            }
        });
        Ok(())
    }

    pub(crate) fn bytes(&self) -> usize {
        self.q8.capacity()
            + self.scales.capacity() * std::mem::size_of::<f32>()
            + self.q8k.capacity() * std::mem::size_of::<crate::ops::quant::BlockQ8K>()
    }

    #[cfg(test)]
    pub(crate) fn q8_capacity_for_test(&self) -> usize {
        self.q8.capacity()
    }

    #[cfg(test)]
    pub(crate) fn q8_ptr_for_test(&self) -> *const u8 {
        self.q8.as_ptr()
    }
}

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

    pub(crate) fn uses_q8_k(&self) -> bool {
        use crate::core::tensor::GGMLType;

        matches!(
            self.ggml_type,
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
                | GGMLType::IQ4_XS
        )
    }

    pub(crate) fn needs_q8_0_activation(&self) -> bool {
        matches!(
            self.ggml_type,
            crate::core::tensor::GGMLType::Q4_0
                | crate::core::tensor::GGMLType::Q4_1
                | crate::core::tensor::GGMLType::Q8_0
        )
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
