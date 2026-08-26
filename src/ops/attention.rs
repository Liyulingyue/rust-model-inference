//! Attention-specific helpers: value accumulation.

pub fn attention_value_f32(
    values: &[f32],
    weights: &[f32],
    n_cached: usize,
    n_padded: usize,
) -> f32 {
    debug_assert!(n_cached <= n_padded);
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { super::dot::dot_f32_neon(values, weights, n_padded) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    super::dot_f32(values, weights, n_padded)
}
