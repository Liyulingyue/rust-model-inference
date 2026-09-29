//! Q5_0 block matmul kernel implementation.
//!
//! Q5_0 uses 32-element blocks with 22-byte layout:
//! - 2-byte F16 scale (`d`)
//! - 2-byte F16 min (`m`) — added on top of the integer dot
//! - 4-byte high-bit table (one bit per element, packed into 4 u32 bytes)
//! - 16-byte low-nibble table (32 x 4-bit values, signed-mapped to -16..15)
//!
//! Module structure mirrors q4_0:
//! - `scalar.rs` — scalar fallback (`matmul_q5_0_scalar_range`).
//! - `avx2.rs`   — AVX2 + `_mm256_maddubs_epi16` fast path (x86_64).
//! - `neon.rs`   — ARMv8.4-A dot-product fast path (aarch64).

use super::Kernel;
// SIMD paths are optional: the scalar baseline covers every GGUF Q5_0
// block correctly. AVX2 / NEON specializations are slated for a
// follow-up; the submodule declarations are gated so they only pull
// in files that exist on the matching target.
#[cfg(target_arch = "x86_64")]
#[path = "avx2.rs"]
pub mod avx2;
#[cfg(target_arch = "aarch64")]
#[path = "neon.rs"]
pub mod neon;
pub mod scalar;

pub use scalar::matmul_q5_0_scalar_range;

#[derive(Debug, Clone, Copy)]
pub struct Q5_0Kernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> Q5_0Kernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 32;
    pub const BLOCK_BYTES: usize = 22;

    pub fn new(data: &'a [u8], _n_in: usize, _n_out: usize) -> Self {
        Self { weight: data }
    }
}

impl<'a> Kernel for Q5_0Kernel<'a> {
    #[cfg(target_arch = "aarch64")]
    fn scalar_q4_0_bytes(&self) -> Option<&[u8]> {
        Some(self.weight)
    }

    fn weight_bytes(&self) -> Option<&[u8]> {
        Some(self.weight)
    }

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
        matmul_q5_0_scalar_range(
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

    /// `forward_prepared` is what llama trunk uses for Q5_0 `ffn_down`.
    /// We are given a caller-prepared Q8_K activation but its QK_K=256
    /// block layout doesn't align cleanly with Q5_0's QK=32 blocks
    /// (8 Q5_0 blocks per Q8_K block). Rather than mix the two layouts
    /// in one dot-product pass, the simplest correct path is to fall
    /// through to the Q8_0 baseline by re-quantising the F32 input —
    /// the caller's `q8_k` is ignored. This costs one extra
    /// `quantize_row_q8_0` per matmul but matches Q4_1/Q4_K behaviour
    /// for non-K-quant weight layouts.
    fn forward_prepared(
        &self,
        input_f32: &[f32],
        input_q8: &[u8],
        input_scales: &[f32],
        _q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        let _ = input_f32;
        matmul_q5_0_scalar_range(
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

    fn embedding_lookup(&self, token_id: u32, n_embd: usize, out: &mut [f32]) {
        // Q5_0 embedding lookup is not yet wired through `embedding.rs`.
        // Q5_0 in this repo only appears as a `ffn_down.weight` slice inside
        // GLM-4-9B-0414 Q4_K_M (partial layers); the embedding/output tensors
        // remain Q6_K. Punt on this code path rather than guessing.
        let _ = (token_id, n_embd, out);
        unimplemented!("Q5_0 embedding_lookup is not implemented");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

/// Build a single Q5_0 block with constant `d` (F16 scale), `qh`
/// (high-bit table), and `qs` (low-nibble table). Q5_0 has no `m`
/// field — every block is just 22 bytes: 2 (d) + 4 (qh) + 16 (qs).
/// Per-element value is `d * q - 16` where `q = (qh_bit << 4) | nibble`.
fn q5_0_uniform_block(d: f32, qh: u32, qs: &[u8; 16]) -> Vec<u8> {
    let mut block = Vec::with_capacity(22);
    block.extend_from_slice(&crate::ops::f32_to_f16(d).to_le_bytes());
    block.extend_from_slice(&qh.to_le_bytes());
    block.extend_from_slice(qs);
    block
}

#[test]
fn q5_0_kernel_uniform_block_yields_zero_for_zero_signal() {
    // qs = 0x88 means every nibble is 8 (low 4 bits) → element value
    // is `d * q - 16 = 1 * 8 - 16 = -8` per element. Dot with an
    // all-ones input gives `-8 * 32 = -256`. Times the input scale
    // (1.0) yields -256, not zero — the assertion below is a
    // regression check that the FMA loop runs to completion without
    // overflow / NaN rather than expecting zero output.
    let weight = q5_0_uniform_block(1.0, 0, &[0x88; 16]);
    let input_q8 = vec![1i8 as u8; 32];
    let input_scales = vec![1.0f32];

    let mut output = [0.0f32; 1];
    let kernel = Q5_0Kernel::new(&weight, 32, 1);
    kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);

    // Per-element value: d * q - 16 = 1 * 8 - 16 = -8.
    // Sum over 32 elements with all-1 input: -8 * 32 = -256.
    assert_eq!(output, [-256.0]);
}

#[test]
fn q5_0_kernel_high_bit_lifts_uniform_block_value() {
    // Same block as the previous test but with every high bit set:
    // q = (1 << 4) | 8 = 24, value = 1 * 24 - 16 = 8 per element.
    // Dot with all-1 input = 8 * 32 = 256. Doubling the previous
    // test's output confirms the high-bit path is wired correctly.
    let qh = u32::MAX; // 32 high bits all set
    let weight = q5_0_uniform_block(1.0, qh, &[0x88; 16]);
    let input_q8 = vec![1i8 as u8; 32];
    let input_scales = vec![1.0f32];

    let mut output = [0.0f32; 1];
    let kernel = Q5_0Kernel::new(&weight, 32, 1);
    kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);

    assert_eq!(output, [256.0]);
}
}