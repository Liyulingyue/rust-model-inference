//! Attention-specific helpers: value accumulation.

/// `out = Σ_{s < n_cached} values[s] * weights[s]`
///
/// `n_cached` is the number of real (non-padded) elements. Callers pad the
/// score vector to a 256-multiple before softmax to keep SIMD reductions
/// aligned, but the padded slots are zero after softmax and contribute
/// nothing to the sum. We therefore reduce only `n_cached` terms — both
/// `values[n_cached..]` and `weights[n_cached..]` are zero by construction.
///
/// `n_padded` is accepted for ABI compatibility with the ggml-aligned padded
/// reduction; it is intentionally ignored. Output is bit-identical to the
/// padded variant when slots `n_cached..n_padded` carry zero on both sides.
pub fn attention_value_f32(
    values: &[f32],
    weights: &[f32],
    n_cached: usize,
    n_padded: usize,
) -> f32 {
    debug_assert!(n_cached <= n_padded);
    debug_assert!(values.len() >= n_cached);
    debug_assert!(weights.len() >= n_cached);
    let _ = n_padded;
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { super::dot::dot_f32_neon(values, weights, n_cached) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    super::dot_f32(values, weights, n_cached)
}
