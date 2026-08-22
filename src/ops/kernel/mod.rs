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

/// Owned counterpart of `QuantizedTensor<'a>`. Holds `Vec<u8>` (or
/// `Vec<f32>` for F32) instead of borrowing, which is required for
/// operations that mutate or concatenate weight bytes — chiefly
/// `fuse_vstack` (FFN gate+up fusion, QKV fusion) and any future work
/// that needs to own a fused output.
///
/// `QTensorOwned` is the canonical weight type for any model path that
/// performs fusion or weight-side transformations. Models that only do
/// zero-copy mmap'd matmul (Qwen3 text path) should keep using the
/// borrowed `QuantizedTensor<'a>` to avoid the clone cost.
#[derive(Debug, Clone)]
pub enum QTensorOwned {
    F32 { data: Vec<f32>, n_cols: usize, n_rows: usize },
    F16 { data: Vec<u8>, n_cols: usize, n_rows: usize },
    Q8_0 { data: Vec<u8>, n_cols: usize, n_rows: usize },
    Q4_K { data: Vec<u8>, n_cols: usize, n_rows: usize },
    Q5_K { data: Vec<u8>, n_cols: usize, n_rows: usize },
    Q6_K { data: Vec<u8>, n_cols: usize, n_rows: usize },
}

impl QTensorOwned {
    /// Read shape. Mirrors `QuantizedTensor::n_in()` / `n_out()`.
    pub fn n_in(&self) -> usize {
        match self {
            Self::F32 { n_cols, .. }
            | Self::F16 { n_cols, .. }
            | Self::Q8_0 { n_cols, .. }
            | Self::Q4_K { n_cols, .. }
            | Self::Q5_K { n_cols, .. }
            | Self::Q6_K { n_cols, .. } => *n_cols,
        }
    }

    /// Alias of `n_in()`. Kept because the existing
    /// `qwen35::QWeight::n_cols()` callsite uses this name.
    pub fn n_cols(&self) -> usize { self.n_in() }

    pub fn n_rows(&self) -> usize {
        match self {
            Self::F32 { n_rows, .. }
            | Self::F16 { n_rows, .. }
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
                Self::F32 { data: f32_data, n_cols, n_rows }
            }
            GGMLType::F16 => Self::F16 { data: data.to_vec(), n_cols, n_rows },
            GGMLType::Q8_0 => Self::Q8_0 { data: data.to_vec(), n_cols, n_rows },
            GGMLType::Q4K => Self::Q4_K { data: data.to_vec(), n_cols, n_rows },
            GGMLType::Q5K => Self::Q5_K { data: data.to_vec(), n_cols, n_rows },
            GGMLType::Q6K => Self::Q6_K { data: data.to_vec(), n_cols, n_rows },
            _ => panic!("unsupported weight type {:?} for QTensorOwned", ggml_type),
        }
    }

    /// Convert a borrowed `QuantizedTensor` into an owned one. Copies bytes.
    /// Use when a model needs to take ownership of weights for fusion.
    pub fn from_quantized(q: QuantizedTensor<'_>) -> Self {
        match q {
            QuantizedTensor::F32(v) => Self::F32 { data: v.clone(), n_cols: v.len(), n_rows: 1 },
            QuantizedTensor::F16(w) => Self::F16 { data: w.bytes.to_vec(), n_cols: w.n_in, n_rows: w.n_out },
            QuantizedTensor::Q8_0(b) => Self::Q8_0 { data: b.to_vec(), n_cols: q.n_in(), n_rows: 1 },
            QuantizedTensor::Q4_K(w) => Self::Q4_K { data: w.data.to_vec(), n_cols: w.n_in, n_rows: w.n_out },
            QuantizedTensor::Q5_K(w) => Self::Q5_K { data: w.data.to_vec(), n_cols: w.n_in, n_rows: w.n_out },
            QuantizedTensor::Q6_K(w) => Self::Q6_K { data: w.data.to_vec(), n_cols: w.n_in, n_rows: w.n_out },
            // Q4_0 / Q4_1 are not modeled in QTensorOwned yet (Qwen3.5 doesn't
            // use them and qwen3 uses QuantizedTensor directly).
            QuantizedTensor::Q4_0(_) | QuantizedTensor::Q4_1(_) => {
                panic!("Q4_0 / Q4_1 not yet supported in QTensorOwned")
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
            Self::Q8_0 { .. } => Self::Q8_0 { data: fused, n_cols, n_rows },
            Self::Q4_K { .. } => Self::Q4_K { data: fused, n_cols, n_rows },
            Self::Q5_K { .. } => Self::Q5_K { data: fused, n_cols, n_rows },
            Self::Q6_K { .. } => Self::Q6_K { data: fused, n_cols, n_rows },
            _ => unreachable!(),
        })
    }
}

impl QuantizedTensor<'_> {
    /// For Q8_0 the bytes don't carry explicit n_rows — it has to be
    /// recovered from total byte length. Used by `from_quantized`.
    fn n_rows_for_q8_0(&self) -> usize {
        match self {
            QuantizedTensor::Q8_0(bytes) => {
                let blocks = bytes.len() / 34;
                let n_cols = blocks * 32;
                // n_rows is not stored on the borrowed variant either; for
                // Q8_0 we keep it 1:1 with the borrowed path which treats it
                // as 1 row when computing n_in(). Callers that need explicit
                // n_rows should track it themselves.
                1
            }
            _ => unreachable!(),
        }
    }
}

impl Kernel for QTensorOwned {
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
        match self {
            Self::F32 { data, .. } => Box::new(f32::F32Kernel::new(data.clone())).forward_prequantized(
                input_q8, input_scales, output, n_in, n_out, ith, nth,
            ),
            Self::F16 { data, .. } => Box::new(f16::F16Kernel::new(data.as_slice())).forward_prequantized(
                input_q8, input_scales, output, n_in, n_out, ith, nth,
            ),
            Self::Q8_0 { data, .. } => {
                crate::ops::matmul_q8_0_quantized_parallel_rows(
                    data.as_slice(), input_q8, input_scales, output, n_in, n_out, ith, nth,
                );
            }
            Self::Q4_K { data, .. } => {
                let kernel = q4_k::Q4_KKernel::new(q4_k::Q4_KWeight { data: data.as_slice(), n_in, n_out });
                kernel.forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth);
            }
            Self::Q5_K { data, .. } => {
                let kernel = q5_k::Q5_KKernel::new(q5_k::Q5_KWeight { data: data.as_slice(), n_in, n_out });
                kernel.forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth);
            }
            Self::Q6_K { data, .. } => {
                Box::new(q6_k::Q6_KKernel::new(data.as_slice())).forward_prequantized(
                    input_q8, input_scales, output, n_in, n_out, ith, nth,
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
        match self {
            Self::F32 { data, .. } => Box::new(f32::F32Kernel::new(data.clone())).forward_prepared(
                input_f32, input_q8, input_scales, q8_k, output, n_in, n_out, ith, nth,
            ),
            Self::F16 { data, .. } => Box::new(f16::F16Kernel::new(data.as_slice())).forward_prepared(
                input_f32, input_q8, input_scales, q8_k, output, n_in, n_out, ith, nth,
            ),
            Self::Q8_0 { data, .. } => {
                // Q8_0 ignores q8_k — delegate via prequantized.
                self.forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth);
            }
            Self::Q4_K { data, .. } => {
                let kernel = q4_k::Q4_KKernel::new(q4_k::Q4_KWeight { data: data.as_slice(), n_in, n_out });
                kernel.forward_prepared(input_f32, input_q8, input_scales, q8_k, output, n_in, n_out, ith, nth);
            }
            Self::Q5_K { data, .. } => {
                let kernel = q5_k::Q5_KKernel::new(q5_k::Q5_KWeight { data: data.as_slice(), n_in, n_out });
                kernel.forward_prepared(input_f32, input_q8, input_scales, q8_k, output, n_in, n_out, ith, nth);
            }
            Self::Q6_K { data, .. } => {
                Box::new(q6_k::Q6_KKernel::new(data.as_slice())).forward_prepared(
                    input_f32, input_q8, input_scales, q8_k, output, n_in, n_out, ith, nth,
                );
            }
        }
    }

    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        // Default impl works, but for Q8_0/Q4_K/Q5_K/Q6_K the kernel wants
        // pre-quantized input. Use the trait default.
        <Self as Kernel>::forward(self, input, output, n_in, n_out)
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
    pub fn matmul_into(
        &self,
        input: &[f32],
        output: &mut [f32],
        row_start: usize,
        row_end: usize,
    ) {
        use crate::ops::quant::{
            vec_dot_q4k_q8k_avx2_direct, vec_dot_q4k_q8k_scalar, vec_dot_q5k_q8k_avx2_direct,
            vec_dot_q5k_q8k_scalar, vec_dot_q6k_q8k_avx2_direct, vec_dot_q6k_q8k_scalar,
            BLOCK_Q4K_SIZE, BLOCK_Q5K_SIZE, BLOCK_Q6K_SIZE, QK_K,
        };
        use crate::ops::{dot_f32, f16_to_f32, quantize_row_q8_k};
        use crate::ops::has_avx2_fma;

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
                        let bits = u16::from_le_bytes([data[(o * in_dim + j) * 2], data[(o * in_dim + j) * 2 + 1]]);
                        row[j] = f16_to_f32(bits);
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
                    output[o - row_start] = if has_avx2_fma() {
                        unsafe { vec_dot_q4k_q8k_avx2_direct(row_data, &q8k) }
                    } else {
                        vec_dot_q4k_q8k_scalar(row_data, &q8k)
                    };
                }
            }
            Self::Q5_K { data, n_cols, .. } => {
                let q8k = quantize_row_q8_k(input);
                let blocks_per_row = *n_cols / QK_K;
                for o in row_start..row_end {
                    let row_data = &data[o * blocks_per_row * BLOCK_Q5K_SIZE..];
                    output[o - row_start] = if has_avx2_fma() {
                        unsafe { vec_dot_q5k_q8k_avx2_direct(row_data, &q8k) }
                    } else {
                        vec_dot_q5k_q8k_scalar(row_data, &q8k)
                    };
                }
            }
            Self::Q6_K { data, n_cols, .. } => {
                let q8k = quantize_row_q8_k(input);
                let blocks_per_row = *n_cols / QK_K;
                for o in row_start..row_end {
                    let row_data = &data[o * blocks_per_row * BLOCK_Q6K_SIZE..];
                    output[o - row_start] = if has_avx2_fma() {
                        unsafe { vec_dot_q6k_q8k_avx2_direct(row_data, &q8k) }
                    } else {
                        vec_dot_q6k_q8k_scalar(row_data, &q8k)
                    };
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
    pub fn matmul_with_q8k_into_buf(
        &self,
        q8k: &[crate::ops::quant::BlockQ8K],
        buf: &mut [f32],
    ) {
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
            vec_dot_q4k_q8k_avx2_direct, vec_dot_q4k_q8k_scalar, vec_dot_q5k_q8k_avx2_direct,
            vec_dot_q5k_q8k_scalar, vec_dot_q6k_q8k_avx2_direct, vec_dot_q6k_q8k_scalar,
            BLOCK_Q4K_SIZE, BLOCK_Q5K_SIZE, BLOCK_Q6K_SIZE, QK_K,
        };
        use crate::ops::has_avx2_fma;

        match self {
            Self::Q4_K { data, n_cols, .. } => {
                let blocks_per_row = *n_cols / QK_K;
                for o in row_start..row_end {
                    let row_data = &data[o * blocks_per_row * BLOCK_Q4K_SIZE..];
                    output[o - row_start] = if has_avx2_fma() {
                        unsafe { vec_dot_q4k_q8k_avx2_direct(row_data, q8k) }
                    } else {
                        vec_dot_q4k_q8k_scalar(row_data, q8k)
                    };
                }
            }
            Self::Q5_K { data, n_cols, .. } => {
                let blocks_per_row = *n_cols / QK_K;
                for o in row_start..row_end {
                    let row_data = &data[o * blocks_per_row * BLOCK_Q5K_SIZE..];
                    output[o - row_start] = if has_avx2_fma() {
                        unsafe { vec_dot_q5k_q8k_avx2_direct(row_data, q8k) }
                    } else {
                        vec_dot_q5k_q8k_scalar(row_data, q8k)
                    };
                }
            }
            Self::Q6_K { data, n_cols, .. } => {
                let blocks_per_row = *n_cols / QK_K;
                for o in row_start..row_end {
                    let row_data = &data[o * blocks_per_row * BLOCK_Q6K_SIZE..];
                    output[o - row_start] = if has_avx2_fma() {
                        unsafe { vec_dot_q6k_q8k_avx2_direct(row_data, q8k) }
                    } else {
                        vec_dot_q6k_q8k_scalar(row_data, q8k)
                    };
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
            if start >= end { return; }
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
            if start >= end { return; }
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
        self.quantize_and_matmul_with_scratch(input, q8k_buf, &mut q8_buf, &mut scale_buf, buf, &pool);
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
            Self::Q4_K { n_cols, .. }
            | Self::Q5_K { n_cols, .. }
            | Self::Q6_K { n_cols, .. } => {
                let blocks = *n_cols / crate::ops::quant::QK_K;
                quantize_row_q8_k_into(input, &mut q8k_buf[..blocks]);
                self.matmul_with_q8k_into_buf_pooled(&q8k_buf[..blocks], buf, pool);
            }
            Self::Q8_0 { data, n_cols, n_rows } => {
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
                        data.as_slice(), q8, sc, out, n_cols_local, n_rows_local, ith, nth,
                    );
                });
            }
            Self::F32 { .. } | Self::F16 { .. } => {
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
                        if start >= end { return; }
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
        use crate::ops::quant::{dequant_q5k_weight, dequant_q6k_weight, dequant_q80_weight,
            dequantize_row_q4_k, BLOCK_Q4K_SIZE, QK_K};
        match self {
            Self::F32 { .. } => self,
            Self::F16 { .. } => panic!("F16 -> F32 dequant not yet implemented in QTensorOwned"),
            Self::Q8_0 { data, n_cols, n_rows } => {
                let dequant = dequant_q80_weight(&data, n_cols, n_rows);
                Self::F32 { data: dequant, n_cols, n_rows }
            }
            Self::Q4_K { data, n_cols, n_rows } => {
                let mut out = vec![0.0f32; n_cols * n_rows];
                let bpr = n_cols / QK_K;
                for row in 0..n_rows {
                    dequantize_row_q4_k(
                        &data[row * bpr * BLOCK_Q4K_SIZE..],
                        &mut out[row * n_cols..row * n_cols + n_cols],
                    );
                }
                Self::F32 { data: out, n_cols, n_rows }
            }
            Self::Q5_K { data, n_cols, n_rows } => {
                let dequant = dequant_q5k_weight(&data, n_cols, n_rows);
                Self::F32 { data: dequant, n_cols, n_rows }
            }
            Self::Q6_K { data, n_cols, n_rows } => {
                let dequant = dequant_q6k_weight(&data, n_cols, n_rows);
                Self::F32 { data: dequant, n_cols, n_rows }
            }
        }
    }
}

#[cfg(test)]
mod qtensor_owned_tests {
    use super::*;
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
        assert!(matches!(owned, QTensorOwned::Q8_0 { ref data, n_cols: 64, n_rows: 4 } if data == &bytes));
        assert_eq!(owned.ggml_type(), GGMLType::Q8_0);
        assert_eq!(owned.n_in(), 64);
        assert_eq!(owned.n_rows(), 4);
    }

    #[test]
    fn qtensor_owned_kernel_matches_borrowed_kernel_bit_exact() {
        // Build owned + borrowed from the same bytes, run both kernels, compare.
        let bytes = make_q8_0_weight(8, 64, 3);
        let owned = QTensorOwned::from_bytes_owned(&bytes, GGMLType::Q8_0, 64, 8);

        // Quantize input to Q8_0.
        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.013 - 4.0).collect();
        let mut input_q8 = vec![0u8; 64];
        let mut input_scales = vec![0.0f32; 64 / 32];
        quantize_q8_0_into(&input, 64, &mut input_q8, &mut input_scales);

        // Owned path: dispatch through QTensorOwned::Kernel.
        let mut out_owned = vec![0.0f32; 8];
        owned.forward_prequantized(&input_q8, &input_scales, &mut out_owned, 64, 8, 0, 1);

        // Borrowed path: build QuantizedTensor -> Kernel -> forward.
        let borrowed = QuantizedTensor::Q8_0(&bytes);
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
            // First 4 rows == gate, next 6 == up.
            let row_bytes = 2 * BLOCK_Q80_SIZE; // 2 blocks per row × 34 bytes
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
        // Different n_cols -> None.
        assert!(QTensorOwned::fuse_vstack(&a, &b).is_none());
    }

    #[test]
    fn qtensor_owned_fused_kernel_dispatches_correctly() {
        // Fused output should produce identical first-n_rows to gate matmul,
        // last-n_rows to up matmul.
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

        // Fused: 8 rows
        let mut out_fused = vec![0.0f32; 8];
        fused.forward_prequantized(&input_q8, &input_scales, &mut out_fused, 64, 8, 0, 1);

        // Separate gate / up kernels for reference
        let g_kernel = QuantizedTensor::Q8_0(&gate).clone_to_kernel();
        let u_kernel = QuantizedTensor::Q8_0(&up).clone_to_kernel();
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
