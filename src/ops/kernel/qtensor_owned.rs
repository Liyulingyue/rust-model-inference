//! Reserved owned weight enum `QTensorOwned` for future weight transforms.
//!
//! Prefer `QuantizedTensor<'a>` for normal inference. It keeps GGUF/mmap
//! weights zero-copy and should remain the default representation whenever a
//! weight can be used without changing its layout.
//!
//! Holds `Vec<u8>` (or `Vec<f32>` for F32) instead of borrowing. Required for
//! operations that must materialize a new weight buffer, such as
//! `fuse_vstack` (FFN gate+up fusion or attention QKV fusion), dequantization,
//! or reordering. It is intentionally not the default model weight type:
//! constructing it from GGUF bytes copies the selected tensor.
//!
//! The hot path uses `QuantizedTensor<'a>` (borrowed, zero-copy). `QTensorOwned`
//! appears only at model load (when fuse is wired) and in the Kernel impl
//! branch that dispatches `.as_slice()` to the same SIMD kernels as the
//! borrowed path.
//!
//! Construction:
//! - `QTensorOwned::from_bytes_owned(...)` — copy from GGUF bytes
//! - `QTensorOwned::from_quantized(q)` — convert borrowed → owned (clone)
//! - `QTensorOwned::fuse_vstack(&a, &b)` — concat two owned tensors

use super::quantized_tensor::QuantizedTensor;
use crate::core::tensor::GGMLType;

/// Owned fallback for weights that cannot remain borrowed.
///
/// Do not use this type merely to represent an ordinary GGUF tensor. Use
/// `QuantizedTensor<'a>` for that case. Keep `QTensorOwned` at ownership
/// boundaries where a transform creates new bytes or where the weight must
/// outlive its source mapping.
#[derive(Debug, Clone)]
pub enum QTensorOwned {
    F32 {
        data: Vec<f32>,
        n_cols: usize,
        n_rows: usize,
    },
    F16 {
        data: Vec<u8>,
        n_cols: usize,
        n_rows: usize,
    },
    BF16 {
        data: Vec<u8>,
        n_cols: usize,
        n_rows: usize,
    },
    Q8_0 {
        data: Vec<u8>,
        n_cols: usize,
        n_rows: usize,
    },
    Q4_K {
        data: Vec<u8>,
        n_cols: usize,
        n_rows: usize,
    },
    Q5_K {
        data: Vec<u8>,
        n_cols: usize,
        n_rows: usize,
    },
    Q6_K {
        data: Vec<u8>,
        n_cols: usize,
        n_rows: usize,
    },
}

impl QTensorOwned {
    /// Read shape. Mirrors `QuantizedTensor::n_in()` / `n_out()`.
    pub fn n_in(&self) -> usize {
        match self {
            Self::F32 { n_cols, .. }
            | Self::F16 { n_cols, .. }
            | Self::BF16 { n_cols, .. }
            | Self::Q8_0 { n_cols, .. }
            | Self::Q4_K { n_cols, .. }
            | Self::Q5_K { n_cols, .. }
            | Self::Q6_K { n_cols, .. } => *n_cols,
        }
    }

    /// Alias of `n_in()`. Kept because the existing
    /// `qwen35::QWeight::n_cols()` callsite uses this name.
    pub fn n_cols(&self) -> usize {
        self.n_in()
    }

    pub fn n_rows(&self) -> usize {
        match self {
            Self::F32 { n_rows, .. }
            | Self::F16 { n_rows, .. }
            | Self::BF16 { n_rows, .. }
            | Self::Q8_0 { n_rows, .. }
            | Self::Q4_K { n_rows, .. }
            | Self::Q5_K { n_rows, .. }
            | Self::Q6_K { n_rows, .. } => *n_rows,
        }
    }

    /// `GGMLType` discriminator — useful for parity checks and dispatch.
    pub fn ggml_type(&self) -> GGMLType {
        match self {
            Self::F32 { .. } => GGMLType::F32,
            Self::F16 { .. } => GGMLType::F16,
            Self::BF16 { .. } => GGMLType::BF16,
            Self::Q8_0 { .. } => GGMLType::Q8_0,
            Self::Q4_K { .. } => GGMLType::Q4K,
            Self::Q5_K { .. } => GGMLType::Q5K,
            Self::Q6_K { .. } => GGMLType::Q6K,
        }
    }

    /// Construct an owned tensor by copying raw GGUF bytes into a Vec.
    pub fn from_bytes_owned(
        data: &[u8],
        ggml_type: GGMLType,
        n_cols: usize,
        n_rows: usize,
    ) -> Self {
        match ggml_type {
            GGMLType::F32 => {
                let f32_data: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Self::F32 {
                    data: f32_data,
                    n_cols,
                    n_rows,
                }
            }
            GGMLType::F16 => Self::F16 {
                data: data.to_vec(),
                n_cols,
                n_rows,
            },
            GGMLType::BF16 => Self::BF16 {
                data: data.to_vec(),
                n_cols,
                n_rows,
            },
            GGMLType::Q8_0 => Self::Q8_0 {
                data: data.to_vec(),
                n_cols,
                n_rows,
            },
            GGMLType::Q4K => Self::Q4_K {
                data: data.to_vec(),
                n_cols,
                n_rows,
            },
            GGMLType::Q5K => Self::Q5_K {
                data: data.to_vec(),
                n_cols,
                n_rows,
            },
            GGMLType::Q6K => Self::Q6_K {
                data: data.to_vec(),
                n_cols,
                n_rows,
            },
            _ => panic!("unsupported weight type {:?} for QTensorOwned", ggml_type),
        }
    }

    /// Convert a borrowed `QuantizedTensor` into an owned one. Copies bytes.
    /// Use when a model needs to take ownership of weights for fusion.
    pub fn from_quantized(q: QuantizedTensor<'_>) -> Self {
        match q {
            QuantizedTensor::F32(v) => Self::F32 {
                data: v.clone(),
                n_cols: v.len(),
                n_rows: 1,
            },
            QuantizedTensor::F16(w) => Self::F16 {
                data: w.bytes.to_vec(),
                n_cols: w.n_in,
                n_rows: w.n_out,
            },
            QuantizedTensor::BF16(w) => Self::BF16 {
                data: w.bytes.to_vec(),
                n_cols: w.n_in,
                n_rows: w.n_out,
            },
            QuantizedTensor::Q8_0 {
                data: b,
                n_cols,
                n_rows,
            } => Self::Q8_0 {
                data: b.to_vec(),
                n_cols,
                n_rows,
            },
            QuantizedTensor::Q4_K {
                data,
                n_cols,
                n_rows,
            } => Self::Q4_K {
                data: data.to_vec(),
                n_cols,
                n_rows,
            },
            QuantizedTensor::Q5_K {
                data,
                n_cols,
                n_rows,
            } => Self::Q5_K {
                data: data.to_vec(),
                n_cols,
                n_rows,
            },
            QuantizedTensor::Q6_K {
                data,
                n_cols,
                n_rows,
            } => Self::Q6_K {
                data: data.to_vec(),
                n_cols,
                n_rows,
            },
            QuantizedTensor::Q2_K { .. } | QuantizedTensor::Q3_K { .. } => {
                panic!("Q2_K / Q3_K not yet supported in QTensorOwned (only used in Qwen3.0.6B matmul, no fuse needed)")
            }
            QuantizedTensor::IQ4NL { .. } | QuantizedTensor::IQ4XS { .. } => {
                panic!("IQ4_NL / IQ4_XS not yet supported in QTensorOwned")
            }
            // Q4_0 / Q4_1 are not modeled in QTensorOwned yet (Qwen3.5 doesn't
            // use them and qwen3 uses QuantizedTensor directly).
            QuantizedTensor::Q4_0 { .. }
            | QuantizedTensor::Q4_1 { .. }
            | QuantizedTensor::IQ2XXS { .. }
            | QuantizedTensor::IQ2S { .. }
            | QuantizedTensor::IQ2XS { .. }
            | QuantizedTensor::IQ3XXS { .. }
            | QuantizedTensor::IQ3S { .. }
            | QuantizedTensor::IQ1M { .. }
            | QuantizedTensor::IQ1S { .. } => {
                panic!("Q4_0 / Q4_1 / I-quant not yet supported in QTensorOwned")
            }
        }
    }

    /// Concatenate two owned tensors along the row (output) dimension.
    /// Both must have the same `n_cols` and the same dtype. Returns `None`
    /// when invariants don't hold.
    ///
    /// Used for FFN gate+up fusion and attention QKV fusion: two matmuls
    /// with the same input become one.
    pub fn fuse_vstack(a: &Self, b: &Self) -> Option<Self> {
        if std::mem::discriminant(a) != std::mem::discriminant(b) || a.n_cols() != b.n_cols() {
            return None;
        }
        let n_cols = a.n_in();
        let (ad, bd) = match (a, b) {
            (Self::F32 { .. }, _) | (Self::F16 { .. }, _) => {
                panic!("F32 / F16 fuse_vstack not implemented yet")
            }
            (Self::BF16 { data: ad, .. }, Self::BF16 { data: bd, .. }) => (ad.clone(), bd.clone()),
            (Self::Q8_0 { data: ad, .. }, Self::Q8_0 { data: bd, .. }) => (ad.clone(), bd.clone()),
            (Self::Q4_K { data: ad, .. }, Self::Q4_K { data: bd, .. }) => (ad.clone(), bd.clone()),
            (Self::Q5_K { data: ad, .. }, Self::Q5_K { data: bd, .. }) => (ad.clone(), bd.clone()),
            (Self::Q6_K { data: ad, .. }, Self::Q6_K { data: bd, .. }) => (ad.clone(), bd.clone()),
            _ => return None,
        };
        let mut fused = Vec::with_capacity(ad.len() + bd.len());
        fused.extend_from_slice(&ad);
        fused.extend_from_slice(&bd);
        let n_rows = a.n_rows() + b.n_rows();
        Some(match a {
            Self::Q8_0 { .. } => Self::Q8_0 {
                data: fused,
                n_cols,
                n_rows,
            },
            Self::BF16 { .. } => Self::BF16 {
                data: fused,
                n_cols,
                n_rows,
            },
            Self::Q4_K { .. } => Self::Q4_K {
                data: fused,
                n_cols,
                n_rows,
            },
            Self::Q5_K { .. } => Self::Q5_K {
                data: fused,
                n_cols,
                n_rows,
            },
            Self::Q6_K { .. } => Self::Q6_K {
                data: fused,
                n_cols,
                n_rows,
            },
            _ => unreachable!(),
        })
    }
}

impl crate::ops::kernel::Kernel for QTensorOwned {
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
        // Same SIMD dispatch as `QuantizedTensor<'a>` (which routes through
        // `clone_to_kernel`). The data is `.as_slice()` from a Vec, identical
        // SIMD function as borrowed bytes.
        use crate::ops::kernel::{bf16, f16, f32, q4_k, q5_k, q6_k};
        match self {
            Self::F32 { data, .. } => Box::new(f32::F32Kernel::new(data.clone()))
                .forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth),
            Self::F16 { data, .. } => Box::new(f16::F16Kernel::new(data.as_slice()))
                .forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth),
            Self::BF16 { data, .. } => Box::new(bf16::BF16Kernel::new(data.as_slice()))
                .forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth),
            Self::Q8_0 { data, .. } => {
                crate::ops::matmul_q8_0_quantized_parallel_rows(
                    data.as_slice(),
                    input_q8,
                    input_scales,
                    output,
                    n_in,
                    n_out,
                    ith,
                    nth,
                );
            }
            Self::Q4_K { data, .. } => {
                let kernel = q4_k::Q4_KKernel::new(data.as_slice(), n_in, n_out);
                kernel.forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth);
            }
            Self::Q5_K { data, .. } => {
                let kernel = q5_k::Q5_KKernel::new(data.as_slice(), n_in, n_out);
                kernel.forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth);
            }
            Self::Q6_K { data, .. } => {
                Box::new(q6_k::Q6_KKernel::new(data.as_slice(), n_in, n_out)).forward_prequantized(
                    input_q8,
                    input_scales,
                    output,
                    n_in,
                    n_out,
                    ith,
                    nth,
                );
            }
        }
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
        use crate::ops::kernel::{bf16, f16, f32, q4_k, q5_k, q6_k};
        match self {
            Self::F32 { data, .. } => Box::new(f32::F32Kernel::new(data.clone())).forward_prepared(
                input_f32,
                input_q8,
                input_scales,
                q8_k,
                output,
                n_in,
                n_out,
                ith,
                nth,
            ),
            Self::F16 { data, .. } => Box::new(f16::F16Kernel::new(data.as_slice()))
                .forward_prepared(
                    input_f32,
                    input_q8,
                    input_scales,
                    q8_k,
                    output,
                    n_in,
                    n_out,
                    ith,
                    nth,
                ),
            Self::BF16 { data, .. } => Box::new(bf16::BF16Kernel::new(data.as_slice()))
                .forward_prepared(
                    input_f32,
                    input_q8,
                    input_scales,
                    q8_k,
                    output,
                    n_in,
                    n_out,
                    ith,
                    nth,
                ),
            Self::Q8_0 { data, .. } => {
                // Q8_0 ignores q8_k — delegate via prequantized.
                self.forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth);
            }
            Self::Q4_K { data, .. } => {
                let kernel = q4_k::Q4_KKernel::new(data.as_slice(), n_in, n_out);
                kernel.forward_prepared(
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
            Self::Q5_K { data, .. } => {
                let kernel = q5_k::Q5_KKernel::new(data.as_slice(), n_in, n_out);
                kernel.forward_prepared(
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
            Self::Q6_K { data, .. } => {
                Box::new(q6_k::Q6_KKernel::new(data.as_slice(), n_in, n_out)).forward_prepared(
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
    }

    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        // Default impl works, but for Q8_0/Q4_K/Q5_K/Q6_K the kernel wants
        // pre-quantized input. Use the trait default.
        <Self as crate::ops::kernel::Kernel>::forward(self, input, output, n_in, n_out)
    }
}

// ============================================================================
// Convenience / general-purpose SIMD dispatch methods on `QTensorOwned`.
// These mirror the original `qwen35::QWeight` API surface but live on the
// shared type so any model (qwen35, future vision/TTS/embedding variants) can
// use them. The `quantize_and_matmul_with_scratch` entry point that takes
// scratch buffers stays in `qwen35` because the scratch layout is model-
// specific.
// ============================================================================

impl QTensorOwned {
    /// Convenience: allocate `n_rows`-length output and run matmul.
    pub fn matmul(&self, input: &[f32]) -> Vec<f32> {
        let n_rows = self.n_rows();
        let mut output = vec![0.0f32; n_rows];
        self.matmul_into(input, &mut output, 0, n_rows);
        output
    }

    /// Per-row f32-input matmul into `output[row_start..row_end]`. Used by
    /// `matmul` and by `quantize_and_matmul_with_scratch` for F32/F16 path.
    pub fn matmul_into(&self, input: &[f32], output: &mut [f32], row_start: usize, row_end: usize) {
        use crate::ops::quant::{
            vec_dot_q4k_q8k, vec_dot_q5k_q8k, vec_dot_q6k_q8k, BLOCK_Q4K_SIZE, BLOCK_Q5K_SIZE,
            BLOCK_Q6K_SIZE, QK_K,
        };
        use crate::ops::{bf16_to_f32, dot_f32, f16_to_f32, quantize_row_q8_k};

        match self {
            Self::F32 { data, .. } => {
                let in_dim = data.len() / self.n_rows().max(1);
                for o in row_start..row_end {
                    output[o - row_start] =
                        dot_f32(&data[o * in_dim..o * in_dim + in_dim], input, in_dim);
                }
            }
            Self::F16 { data, .. } => {
                let in_dim = input.len();
                let mut row = vec![0.0f32; in_dim];
                for o in row_start..row_end {
                    for j in 0..in_dim {
                        let bits = u16::from_le_bytes([
                            data[(o * in_dim + j) * 2],
                            data[(o * in_dim + j) * 2 + 1],
                        ]);
                        row[j] = f16_to_f32(bits);
                    }
                    output[o - row_start] = dot_f32(&row, input, in_dim);
                }
            }
            Self::BF16 { data, .. } => {
                let in_dim = input.len();
                let mut row = vec![0.0f32; in_dim];
                for o in row_start..row_end {
                    for j in 0..in_dim {
                        let bits = u16::from_le_bytes([
                            data[(o * in_dim + j) * 2],
                            data[(o * in_dim + j) * 2 + 1],
                        ]);
                        row[j] = bf16_to_f32(bits);
                    }
                    output[o - row_start] = dot_f32(&row, input, in_dim);
                }
            }
            Self::Q8_0 { data, n_cols, .. } => {
                let in_dim = *n_cols;
                let blocks_per_row = in_dim / 32;
                for o in row_start..row_end {
                    let row_off = o * blocks_per_row * 34;
                    let mut sum = 0.0f32;
                    let mut dequant_buf = [0.0f32; 32];
                    for b in 0..blocks_per_row {
                        let w_off = row_off + b * 34;
                        let d = f16_to_f32(u16::from_le_bytes([data[w_off], data[w_off + 1]]));
                        for j in 0..32 {
                            dequant_buf[j] = d * (data[w_off + 2 + j] as i8) as f32;
                        }
                        sum += dot_f32(&dequant_buf, &input[b * 32..b * 32 + 32], 32);
                    }
                    output[o - row_start] = sum;
                }
            }
            Self::Q4_K { data, n_cols, .. } => {
                let q8k = quantize_row_q8_k(input);
                let blocks_per_row = *n_cols / QK_K;
                for o in row_start..row_end {
                    let row_data = &data[o * blocks_per_row * BLOCK_Q4K_SIZE..];
                    output[o - row_start] = vec_dot_q4k_q8k(row_data, &q8k);
                }
            }
            Self::Q5_K { data, n_cols, .. } => {
                let q8k = quantize_row_q8_k(input);
                let blocks_per_row = *n_cols / QK_K;
                for o in row_start..row_end {
                    let row_data = &data[o * blocks_per_row * BLOCK_Q5K_SIZE..];
                    output[o - row_start] = vec_dot_q5k_q8k(row_data, &q8k);
                }
            }
            Self::Q6_K { data, n_cols, .. } => {
                let q8k = quantize_row_q8_k(input);
                let blocks_per_row = *n_cols / QK_K;
                for o in row_start..row_end {
                    let row_data = &data[o * blocks_per_row * BLOCK_Q6K_SIZE..];
                    output[o - row_start] = vec_dot_q6k_q8k(row_data, &q8k);
                }
            }
        }
    }

    /// K-quant path: full matmul returning Vec.
    pub fn matmul_with_q8k(&self, q8k: &[crate::ops::quant::BlockQ8K]) -> Vec<f32> {
        let n_rows = self.n_rows();
        let mut output = vec![0.0f32; n_rows];
        self.matmul_into_with_q8k(q8k, &mut output, 0, n_rows);
        output
    }

    /// K-quant path: full matmul into `buf`.
    pub fn matmul_with_q8k_into_buf(&self, q8k: &[crate::ops::quant::BlockQ8K], buf: &mut [f32]) {
        let n_rows = self.n_rows();
        self.matmul_into_with_q8k(q8k, &mut buf[..n_rows], 0, n_rows);
    }

    /// K-quant path: per-row matmul into `output[row_start..row_end]`. Caller
    /// supplies the pre-quantized Q8K activations.
    pub fn matmul_into_with_q8k(
        &self,
        q8k: &[crate::ops::quant::BlockQ8K],
        output: &mut [f32],
        row_start: usize,
        row_end: usize,
    ) {
        use crate::ops::quant::{
            vec_dot_q4k_q8k, vec_dot_q5k_q8k, vec_dot_q6k_q8k, BLOCK_Q4K_SIZE, BLOCK_Q5K_SIZE,
            BLOCK_Q6K_SIZE, QK_K,
        };

        match self {
            Self::Q4_K { data, n_cols, .. } => {
                let blocks_per_row = *n_cols / QK_K;
                for o in row_start..row_end {
                    let row_data = &data[o * blocks_per_row * BLOCK_Q4K_SIZE..];
                    output[o - row_start] = vec_dot_q4k_q8k(row_data, q8k);
                }
            }
            Self::Q5_K { data, n_cols, .. } => {
                let blocks_per_row = *n_cols / QK_K;
                for o in row_start..row_end {
                    let row_data = &data[o * blocks_per_row * BLOCK_Q5K_SIZE..];
                    output[o - row_start] = vec_dot_q5k_q8k(row_data, q8k);
                }
            }
            Self::Q6_K { data, n_cols, .. } => {
                let blocks_per_row = *n_cols / QK_K;
                for o in row_start..row_end {
                    let row_data = &data[o * blocks_per_row * BLOCK_Q6K_SIZE..];
                    output[o - row_start] = vec_dot_q6k_q8k(row_data, q8k);
                }
            }
            _ => panic!("matmul_with_q8k called on non-K-quant weight type"),
        }
    }

    /// K-quant path: pool.compute-partitioned matmul into `buf`.
    pub fn matmul_with_q8k_into_buf_pooled(
        &self,
        q8k: &[crate::ops::quant::BlockQ8K],
        buf: &mut [f32],
        pool: &crate::core::thread_pool::ComputePool,
    ) {
        let n_rows = self.n_rows();
        if pool.n_threads() <= 1 || n_rows < 256 {
            self.matmul_with_q8k_into_buf(q8k, buf);
            return;
        }
        let nth = pool.n_threads();
        let chunk_size = (n_rows + nth - 1) / nth;
        let weight_ptr = self as *const QTensorOwned;
        let q8k_ptr = q8k.as_ptr();
        let q8k_len = q8k.len();
        let buf_ptr = buf.as_mut_ptr();
        pool.compute(move |ith, _nth| {
            let start = ith * chunk_size;
            let end = (start + chunk_size).min(n_rows);
            if start >= end {
                return;
            }
            unsafe {
                let w = &*weight_ptr;
                let q = std::slice::from_raw_parts(q8k_ptr, q8k_len);
                let b = std::slice::from_raw_parts_mut(buf_ptr.add(start), end - start);
                w.matmul_into_with_q8k(q, b, start, end);
            }
        });
    }

    /// F32/F16 path via pool.compute partition. Used by `quantize_and_matmul`
    /// for the F32/F16 fallback.
    pub fn matmul_into_buf_pooled(
        &self,
        input: &[f32],
        buf: &mut [f32],
        pool: &crate::core::thread_pool::ComputePool,
    ) {
        let n_rows = self.n_rows();
        if pool.n_threads() <= 1 || n_rows < 256 {
            self.matmul_into(input, &mut buf[..n_rows], 0, n_rows);
            return;
        }
        let nth = pool.n_threads();
        let chunk_size = (n_rows + nth - 1) / nth;
        let weight_ptr = self as *const QTensorOwned;
        let input_ptr = input.as_ptr();
        let buf_ptr = buf.as_mut_ptr();
        pool.compute(move |ith, _nth| {
            let start = ith * chunk_size;
            let end = (start + chunk_size).min(n_rows);
            if start >= end {
                return;
            }
            unsafe {
                let w = &*weight_ptr;
                let inp = std::slice::from_raw_parts(input_ptr, input.len());
                let b = std::slice::from_raw_parts_mut(buf_ptr.add(start), end - start);
                w.matmul_into(inp, b, start, end);
            }
        });
    }

    /// One-shot quantize-then-matmul. Internally allocates a temporary
    /// single-thread `ComputePool`; for the hot path use
    /// `quantize_and_matmul_with_scratch` directly (free function in
    /// `models::qwen35`) when a real pool and pre-allocated scratch buffers
    /// are available.
    pub fn quantize_and_matmul(
        &self,
        input: &[f32],
        q8k_buf: &mut [crate::ops::quant::BlockQ8K],
        buf: &mut [f32],
    ) {
        let mut q8_buf = vec![0u8; input.len()];
        let mut scale_buf = vec![0.0f32; (input.len() + 31) / 32];
        let pool = crate::core::thread_pool::ComputePool::new(1);
        self.quantize_and_matmul_with_scratch(
            input,
            q8k_buf,
            &mut q8_buf,
            &mut scale_buf,
            buf,
            &pool,
        );
    }

    /// Pool-driven quantize + matmul using pre-allocated scratch buffers.
    /// The three scratch buffers (`q8k_buf`, `q8_buf`, `scale_buf`) must be
    /// sized for the input dim — see `Qwen35Scratchpad` for the exact
    /// sizing. `buf` is the output, sized to `n_rows`.
    pub fn quantize_and_matmul_with_scratch(
        &self,
        input: &[f32],
        q8k_buf: &mut [crate::ops::quant::BlockQ8K],
        q8_buf: &mut [u8],
        scale_buf: &mut [f32],
        buf: &mut [f32],
        pool: &crate::core::thread_pool::ComputePool,
    ) {
        use crate::ops::quant::quantize_row_q8_k_into;
        use crate::ops::quantize_q8_0_into;

        match self {
            Self::Q4_K { n_cols, .. } | Self::Q5_K { n_cols, .. } | Self::Q6_K { n_cols, .. } => {
                let blocks = *n_cols / crate::ops::quant::QK_K;
                quantize_row_q8_k_into(input, &mut q8k_buf[..blocks]);
                self.matmul_with_q8k_into_buf_pooled(&q8k_buf[..blocks], buf, pool);
            }
            Self::Q8_0 {
                data,
                n_cols,
                n_rows,
            } => {
                let blocks = *n_cols / 32;
                quantize_q8_0_into(
                    input,
                    *n_cols,
                    &mut q8_buf[..*n_cols],
                    &mut scale_buf[..blocks],
                );
                let n_cols_local = *n_cols;
                let n_rows_local = *n_rows;
                let q8_ptr = q8_buf.as_ptr();
                let sc_ptr = scale_buf.as_ptr();
                let out_ptr = buf.as_mut_ptr();
                pool.compute(move |ith, nth| {
                    let q8 = unsafe { std::slice::from_raw_parts(q8_ptr, n_cols_local) };
                    let sc = unsafe { std::slice::from_raw_parts(sc_ptr, blocks) };
                    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n_rows_local) };
                    crate::ops::matmul_q8_0_quantized_parallel_rows(
                        data.as_slice(),
                        q8,
                        sc,
                        out,
                        n_cols_local,
                        n_rows_local,
                        ith,
                        nth,
                    );
                });
            }
            Self::F32 { .. } | Self::F16 { .. } | Self::BF16 { .. } => {
                let n_rows = self.n_rows();
                if pool.n_threads() <= 1 || n_rows < 256 {
                    self.matmul_into(input, &mut buf[..n_rows], 0, n_rows);
                } else {
                    let nth = pool.n_threads();
                    let chunk_size = (n_rows + nth - 1) / nth;
                    let weight_ptr = self as *const QTensorOwned;
                    let input_ptr = input.as_ptr();
                    let buf_ptr = buf.as_mut_ptr();
                    pool.compute(move |ith, _nth| {
                        let start = ith * chunk_size;
                        let end = (start + chunk_size).min(n_rows);
                        if start >= end {
                            return;
                        }
                        unsafe {
                            let w = &*weight_ptr;
                            let inp = std::slice::from_raw_parts(input_ptr, input.len());
                            let b = std::slice::from_raw_parts_mut(buf_ptr.add(start), end - start);
                            w.matmul_into(inp, b, start, end);
                        }
                    });
                }
            }
        }
    }

    /// Dequantize any dtype to F32. Identity for F32; per-dtype dequant
    /// for Q8_0 / Q4_K / Q5_K / Q6_K. F16 has no dequant helper here yet.
    pub fn dequant_to_f32(self) -> Self {
        use crate::ops::quant::{
            dequant_q5k_weight, dequant_q6k_weight, dequant_q80_weight, dequantize_row_q4_k,
            BLOCK_Q4K_SIZE, QK_K,
        };
        match self {
            Self::F32 { .. } => self,
            Self::F16 { .. } => panic!("F16 -> F32 dequant not yet implemented in QTensorOwned"),
            Self::BF16 {
                data,
                n_cols,
                n_rows,
            } => {
                let mut out = vec![0.0f32; n_cols * n_rows];
                for (value, chunk) in out.iter_mut().zip(data.chunks_exact(2)) {
                    *value = crate::ops::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                }
                Self::F32 {
                    data: out,
                    n_cols,
                    n_rows,
                }
            }
            Self::Q8_0 {
                data,
                n_cols,
                n_rows,
            } => {
                let dequant = dequant_q80_weight(&data, n_cols, n_rows);
                Self::F32 {
                    data: dequant,
                    n_cols,
                    n_rows,
                }
            }
            Self::Q4_K {
                data,
                n_cols,
                n_rows,
            } => {
                let mut out = vec![0.0f32; n_cols * n_rows];
                let bpr = n_cols / QK_K;
                for row in 0..n_rows {
                    dequantize_row_q4_k(
                        &data[row * bpr * BLOCK_Q4K_SIZE..],
                        &mut out[row * n_cols..row * n_cols + n_cols],
                    );
                }
                Self::F32 {
                    data: out,
                    n_cols,
                    n_rows,
                }
            }
            Self::Q5_K {
                data,
                n_cols,
                n_rows,
            } => {
                let dequant = dequant_q5k_weight(&data, n_cols, n_rows);
                Self::F32 {
                    data: dequant,
                    n_cols,
                    n_rows,
                }
            }
            Self::Q6_K {
                data,
                n_cols,
                n_rows,
            } => {
                let dequant = dequant_q6k_weight(&data, n_cols, n_rows);
                Self::F32 {
                    data: dequant,
                    n_cols,
                    n_rows,
                }
            }
        }
    }
}

#[cfg(test)]
mod qtensor_owned_tests {
    use super::*;
    use crate::ops::kernel::Kernel;
    use crate::ops::quant::BLOCK_Q80_SIZE;
    use crate::ops::quantize_q8_0_into;

    /// Deterministic Q8_0 weight: row r, block b, lane j produces
    /// `(row_offset + r + b + j/4)` clamped to i8. Different rows produce
    /// different matmul outputs.
    fn make_q8_0_weight(n_rows: usize, n_cols: usize, row_offset: i8) -> Vec<u8> {
        let blocks_per_row = n_cols / 32;
        let row_bytes = blocks_per_row * BLOCK_Q80_SIZE;
        let mut data = Vec::with_capacity(n_rows * row_bytes);
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let scale = half::f16::from_f32(0.5 + 0.01 * (r as f32)).to_bits();
                data.extend_from_slice(&scale.to_le_bytes());
                for j in 0..32i8 {
                    let v = (row_offset as i32 + r as i32 + b as i32 + j as i32 / 4) as i8;
                    data.push(v as u8);
                }
            }
        }
        data
    }

    #[test]
    fn qtensor_owned_from_bytes_layout_matches_borrowed() {
        let bytes = make_q8_0_weight(4, 64, 0);
        let owned = QTensorOwned::from_bytes_owned(&bytes, GGMLType::Q8_0, 64, 4);
        assert!(
            matches!(owned, QTensorOwned::Q8_0 { ref data, n_cols: 64, n_rows: 4 } if data == &bytes)
        );
        assert_eq!(owned.ggml_type(), GGMLType::Q8_0);
        assert_eq!(owned.n_in(), 64);
        assert_eq!(owned.n_rows(), 4);
    }

    #[test]
    fn qtensor_owned_bf16_round_trip() {
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|value| crate::ops::f32_to_bf16(*value).to_le_bytes())
            .collect();
        let borrowed = QuantizedTensor::from_bytes(&bytes, GGMLType::BF16, 3, 2);
        let owned = QTensorOwned::from_quantized(borrowed);
        assert_eq!(owned.ggml_type(), GGMLType::BF16);
        assert_eq!(owned.n_in(), 3);
        assert_eq!(owned.n_rows(), 2);
        let mut output = [0.0f32; 2];
        owned.matmul_into(&[1.0, 2.0, 3.0], &mut output, 0, 2);
        assert_eq!(output, [14.0, 32.0]);
    }

    #[test]
    fn qtensor_owned_kernel_matches_borrowed_kernel_bit_exact() {
        let bytes = make_q8_0_weight(8, 64, 3);
        let owned = QTensorOwned::from_bytes_owned(&bytes, GGMLType::Q8_0, 64, 8);

        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.013 - 4.0).collect();
        let mut input_q8 = vec![0u8; 64];
        let mut input_scales = vec![0.0f32; 64 / 32];
        quantize_q8_0_into(&input, 64, &mut input_q8, &mut input_scales);

        let mut out_owned = vec![0.0f32; 8];
        owned.forward_prequantized(&input_q8, &input_scales, &mut out_owned, 64, 8, 0, 1);

        let borrowed = QuantizedTensor::Q8_0 {
            data: &bytes,
            n_cols: 64,
            n_rows: 4,
        };
        let kernel = borrowed.clone_to_kernel();
        let mut out_borrowed = vec![0.0f32; 8];
        kernel.forward_prequantized(&input_q8, &input_scales, &mut out_borrowed, 64, 8, 0, 1);

        for (i, (a, b)) in out_owned.iter().zip(out_borrowed.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "row {i}: owned={a} borrowed={b}");
        }
    }

    #[test]
    fn qtensor_owned_fuse_vstack_layout() {
        let gate = make_q8_0_weight(4, 64, 0);
        let up = make_q8_0_weight(6, 64, 17);
        let gate_owned = QTensorOwned::from_bytes_owned(&gate, GGMLType::Q8_0, 64, 4);
        let up_owned = QTensorOwned::from_bytes_owned(&up, GGMLType::Q8_0, 64, 6);

        let fused = QTensorOwned::fuse_vstack(&gate_owned, &up_owned).expect("fuse");
        assert_eq!(fused.n_rows(), 10);
        assert_eq!(fused.n_in(), 64);
        if let QTensorOwned::Q8_0 { data, .. } = &fused {
            let row_bytes = 2 * BLOCK_Q80_SIZE;
            assert_eq!(&data[..4 * row_bytes], &gate[..]);
            assert_eq!(&data[4 * row_bytes..], &up[..]);
        } else {
            panic!("expected Q8_0 variant");
        }
    }

    #[test]
    fn qtensor_owned_fuse_vstack_returns_none_on_mismatch() {
        let a = QTensorOwned::from_bytes_owned(&make_q8_0_weight(2, 64, 0), GGMLType::Q8_0, 64, 2);
        let b = QTensorOwned::from_bytes_owned(&make_q8_0_weight(2, 32, 0), GGMLType::Q8_0, 32, 2);
        assert!(QTensorOwned::fuse_vstack(&a, &b).is_none());
    }

    #[test]
    fn qtensor_owned_fused_kernel_dispatches_correctly() {
        let gate = make_q8_0_weight(4, 64, 0);
        let up = make_q8_0_weight(4, 64, 17);
        let fused = QTensorOwned::fuse_vstack(
            &QTensorOwned::from_bytes_owned(&gate, GGMLType::Q8_0, 64, 4),
            &QTensorOwned::from_bytes_owned(&up, GGMLType::Q8_0, 64, 4),
        )
        .expect("fuse");

        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.013 - 4.0).collect();
        let mut input_q8 = vec![0u8; 64];
        let mut input_scales = vec![0.0f32; 64 / 32];
        quantize_q8_0_into(&input, 64, &mut input_q8, &mut input_scales);

        let mut out_fused = vec![0.0f32; 8];
        fused.forward_prequantized(&input_q8, &input_scales, &mut out_fused, 64, 8, 0, 1);

        let g_kernel = QuantizedTensor::Q8_0 {
            data: &gate,
            n_cols: 64,
            n_rows: 4,
        }
        .clone_to_kernel();
        let u_kernel = QuantizedTensor::Q8_0 {
            data: &up,
            n_cols: 64,
            n_rows: 4,
        }
        .clone_to_kernel();
        let mut out_g = vec![0.0f32; 4];
        let mut out_u = vec![0.0f32; 4];
        g_kernel.forward_prequantized(&input_q8, &input_scales, &mut out_g, 64, 4, 0, 1);
        u_kernel.forward_prequantized(&input_q8, &input_scales, &mut out_u, 64, 4, 0, 1);

        for (i, (f, g)) in out_fused[..4].iter().zip(out_g.iter()).enumerate() {
            assert_eq!(f.to_bits(), g.to_bits(), "gate row {i}");
        }
        for (i, (f, u)) in out_fused[4..].iter().zip(out_u.iter()).enumerate() {
            assert_eq!(f.to_bits(), u.to_bits(), "up row {i}");
        }
    }
}
