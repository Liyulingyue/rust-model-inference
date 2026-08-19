//! Kernel trait + reserved type abstractions for future matmul dispatch.
//!
//! Phase 2.1: This module defines the **type skeleton only** — no behavior
//! changes. The `QuantizedTensor` enum and weight structs reserve the
//! interface for future migration of `ProcessedWeight` (in `super`). Existing
//! code paths are untouched.
//!
//! Reserved interfaces (no impl yet):
//! - `Kernel` trait — single point of dispatch for matmul kernels
//! - `F16Weight` — placeholder for F16 matmul (not yet supported in
//!   `ProcessedWeight`; defined here to lock the contract)
//! - `QuantizedTensor` — future replacement for `ProcessedWeight`
//!
//! Migration timeline:
//! - Phase 2.2: F32 matmul → `Kernel` impl ✓
//! - Phase 2.3: Q8_0 matmul → `Kernel` impl ✓
//! - Phase 2.4: F16 matmul → `Kernel` impl ✓
//! - Phase 2.5: Q6_K / Q4_0 / Q4_1 → `Kernel` impl ✓
//! - Phase 2.4: F16 matmul → `Kernel` impl (deferred)
//! - Phase 2.5: Q6_K / Q4_0 / Q4_1 → `Kernel` impl
//! - Later: `ProcessedWeight` → `QuantizedTensor` substitution

pub mod f16;
pub mod f32;
pub mod q4_0;
pub mod q4_1;
pub mod q6_k;
pub mod q8_0;

use crate::model::GGMLType;

/// Core matmul kernel interface.
///
/// Dispatched via enum (static, zero-cost), not vtable. Single-token
/// `forward` is the hot path; `forward_batched` has a default loop
/// impl that calls `forward` per token.
pub trait Kernel {
    /// Single-token matmul: `input[n_in] → output[n_out]`
    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize);

    /// Batched matmul: `input[n_tokens * n_in] → output[n_tokens * n_out]`
    ///
    /// Default impl loops over tokens calling `forward`. Override for
    /// kernels that can process multiple tokens in one pass (e.g. mat-vec
    /// variants).
    fn forward_batched(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        let n_tokens = input.len() / n_in;
        debug_assert_eq!(input.len(), n_tokens * n_in);
        debug_assert_eq!(output.len(), n_tokens * n_out);
        for t in 0..n_tokens {
            self.forward(
                &input[t * n_in..(t + 1) * n_in],
                &mut output[t * n_out..(t + 1) * n_out],
                n_in,
                n_out,
            );
        }
    }
}

/// Reserved placeholder for F16 weights.
///
/// F16 matmul is not currently dispatched via `ProcessedWeight`. This
/// struct locks the data layout so that when F16 lands in Phase 2.4,
/// the public contract is already defined.
#[derive(Debug, Clone, Copy)]
pub struct F16Weight<'a> {
    pub bytes: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

/// Future tensor type replacing `ProcessedWeight`.
///
/// Reserved interface only — not yet wired into any dispatch path.
/// Implementation deferred to a later phase. Each variant stores
/// enough metadata to drive a `Kernel::forward` call without re-reading
/// the original `TensorInfo`.
pub enum QuantizedTensor<'a> {
    /// F32 weights — `n_in = slice.len() / n_out`
    F32(&'a [f32]),
    /// F16 weights (placeholder, not yet produced)
    F16(F16Weight<'a>),
    /// Q8_0 block-quantized weights (`n_in = bytes.len() / 34 * 32`)
    Q8_0(&'a [u8]),
    /// Q6_K super-block weights
    Q6_K(super::Q6_KWeight<'a>),
    /// Q4_0 block weights
    Q4_0(super::Q4_0Weight<'a>),
    /// Q4_1 block weights
    Q4_1(super::Q4_1Weight<'a>),
}

impl<'a> QuantizedTensor<'a> {
    /// Type discriminator matching `GGMLType`.
    pub fn ggml_type(&self) -> GGMLType {
        match self {
            Self::F32(_) => GGMLType::F32,
            Self::F16(_) => GGMLType::F16,
            Self::Q8_0(_) => GGMLType::Q8_0,
            Self::Q6_K(_) => GGMLType::Q6K,
            Self::Q4_0(_) => GGMLType::Q4_0,
            Self::Q4_1(_) => GGMLType::Q4_1,
        }
    }

    /// Input dimension (K of the matmul).
    pub fn n_in(&self) -> usize {
        match self {
            Self::F32(slice) => slice.len(), // n_in implicit; full slice = K
            Self::F16(w) => w.n_in,
            Self::Q8_0(bytes) => q8_0_block_count(*bytes) * 32,
            Self::Q6_K(w) => w.n_in,
            Self::Q4_0(w) => w.n_in,
            Self::Q4_1(w) => w.n_in,
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
        // 1 block = 34 bytes
        assert_eq!(q8_0_block_count(&vec![0u8; 34]), 1);
        assert_eq!(q8_0_block_count(&vec![0u8; 68]), 2);
        assert_eq!(q8_0_block_count(&vec![0u8; 102]), 3);
    }

    #[test]
    fn f16_weight_layout_compiles() {
        // Reserved struct layout: data + n_in + n_out
        let data = vec![0u8; 64];
        let w = F16Weight { bytes: &data, n_in: 32, n_out: 2 };
        assert_eq!(w.n_in, 32);
        assert_eq!(w.n_out, 2);
        assert_eq!(w.bytes.len(), 64);
    }

    #[test]
    fn quantized_tensor_ggml_type_discriminator() {
        // F32 variant
        let f32_slice = vec![0.0f32; 32];
        let q = QuantizedTensor::F32(&f32_slice);
        assert_eq!(q.ggml_type(), GGMLType::F32);

        // Q8_0 variant
        let q8_bytes = vec![0u8; 34]; // 1 block
        let q = QuantizedTensor::Q8_0(&q8_bytes);
        assert_eq!(q.ggml_type(), GGMLType::Q8_0);
        assert_eq!(q.n_in(), 32);
    }
}