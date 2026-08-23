//! Attention-specific helpers: value accumulation.

pub fn attention_value_f32(
    values: &[f32],
    weights: &[f32],
    n_cached: usize,
    n_padded: usize,
) -> f32 {
    debug_assert!(n_cached <= n_padded);
    super::dot_f32(values, weights, n_padded)
}
