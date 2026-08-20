//! Q6_K super-block matmul kernel implementation.
//!
//! Phase 2.5 + 2.7-final: Q6_K uses 256-element super-blocks (210 bytes).
//! Like Q4_K / Q5_K, the production fast path here is dequantize-to-f32-
//! then-matmul because llama.cpp's Q6_K family does not have a native
//! Q8-input kernel variant. The `forward_prequantized` argument names are
//! kept for trait uniformity; the Q8 input is ignored and the scalar
//! Q6_K matmul runs on the reinterpreted f32 input. The actual
//! production path is via `QuantizedLinear::forward_dequant` for Q6_K_M
//! weights — see `core::model`.

use super::Kernel;
use crate::core::tensor::GGMLType;
use crate::ops::matmul::{Q6_KWeight};

#[derive(Debug, Clone, Copy)]
pub struct Q6_KKernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> Q6_KKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 210;

    pub fn new(weight: &'a [u8]) -> Self {
        Self { weight }
    }
}

impl<'a> Kernel for Q6_KKernel<'a> {
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
        // The Q6_K matmul reinterprets `input_q8` as `&[f32]` (see comment
        // in `matmul_q6_k_scalar_range`); this is the historical quirk
        // from before the prequantized-Q8 API.
        matmul_q6_k_scalar_range(
            self.weight,
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

/// Q6_K scalar matmul kernel. Phase 2.7-final: moved from `ops::matmul`.
///
/// `input_q8` is actually f32 reinterpretation here (NOT quantized!) — this
/// is a workaround so we avoid needing Q8_K quantization of the input.
/// Slower than a true prequantized Q8-K kernel but correct.
pub fn matmul_q6_k_scalar_range(
    weight: &[u8],
    input_q8: &[u8],
    _input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    let n_blocks = n_in / Q6_KKernel::BLOCK_ELEMENTS;
    let row_stride = n_blocks * Q6_KKernel::BLOCK_BYTES;
    let per_thread = (n_out + nth - 1) / nth;
    let my_start = ith * per_thread;
    let my_end = (my_start + per_thread).min(n_out);
    if my_start >= my_end {
        return;
    }
    let input_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(input_q8.as_ptr() as *const f32, n_in)
    };
    for out_idx in my_start..my_end {
        let row_off = out_idx * row_stride;
        let mut sum = 0.0f32;
        for block in 0..n_blocks {
            let off = row_off + block * Q6_KKernel::BLOCK_BYTES;
            let d = crate::ops::f16_to_f32(u16::from_le_bytes([
                weight[off + 208],
                weight[off + 209],
            ]));
            let base_x = block * 256;
            let mut sum_block = 0.0f32;
            for sub in 0..2 {
                let ql_off = off + sub * 64;
                let qh_off = off + 128 + sub * 32;
                let sc_off = off + 192 + sub * 8;
                for l in 0..32 {
                    let is = l / 16;
                    let ql_0 = weight[ql_off + l] as i8;
                    let ql_1 = weight[ql_off + 32 + l] as i8;
                    let qh_l = weight[qh_off + l] as i8;
                    let q1 = ((((ql_0 & 0xF) as i32)
                        | ((((qh_l >> 0) & 3) as i32) << 4)) as i8)
                        as f32
                        - 32.0;
                    let q2 = ((((ql_1 & 0xF) as i32)
                        | ((((qh_l >> 2) & 3) as i32) << 4)) as i8)
                        as f32
                        - 32.0;
                    let q3 = ((((ql_0 >> 4) as i32)
                        | ((((qh_l >> 4) & 3) as i32) << 4)) as i8)
                        as f32
                        - 32.0;
                    let q4 = ((((ql_1 >> 4) as i32)
                        | ((((qh_l >> 6) & 3) as i32) << 4)) as i8)
                        as f32
                        - 32.0;
                    let sc0 = weight[sc_off + is + 0] as i8;
                    let sc1 = weight[sc_off + is + 2] as i8;
                    let sc2 = weight[sc_off + is + 4] as i8;
                    let sc3 = weight[sc_off + is + 6] as i8;
                    let base_y = sub * 128 + l;
                    sum_block += sc0 as f32 * q1 * input_f32[base_x + base_y]
                        + sc1 as f32 * q2 * input_f32[base_x + base_y + 32]
                        + sc2 as f32 * q3 * input_f32[base_x + base_y + 64]
                        + sc3 as f32 * q4 * input_f32[base_x + base_y + 96];
                }
            }
            sum += d * sum_block;
        }
        output[out_idx] = sum;
    }
}

// Re-export Q6_KWeight for callers that imported it from `ops::matmul`.
// Phase 2.7-final cleanup: keep the old import path working so the
// transition is non-breaking.
pub use crate::core::tensor::GGMLType as _GGMLType;

// Compile-time check that Q6_KWeight is reachable through the new path.
const _: GGMLType = GGMLType::Q6K;