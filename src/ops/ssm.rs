//! State Space Model (SSM) operations. Used by qwen3a.

#[inline(always)]
pub fn ssm_state_decay(state: &mut [f32], decay: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2_fma() {
            unsafe { ssm_state_decay_avx2(state, decay) };
            return;
        }
    }
    for v in state.iter_mut() {
        *v *= decay;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn ssm_state_decay_avx2(state: &mut [f32], decay: f32) {
    use std::arch::x86_64::*;
    let vdecay = _mm256_set1_ps(decay);
    let n = state.len();
    let mut i = 0;
    while i + 8 <= n {
        let s = _mm256_loadu_ps(state.as_ptr().add(i));
        _mm256_storeu_ps(state.as_mut_ptr().add(i), _mm256_mul_ps(s, vdecay));
        i += 8;
    }
    while i < n {
        state[i] *= decay;
        i += 1;
    }
}

#[inline(always)]
pub fn ssm_matvec(state: &[f32], vec: &[f32], dim: usize, n_rows: usize, out: &mut [f32]) {
    debug_assert_eq!(state.len(), n_rows * dim);
    debug_assert_eq!(vec.len(), dim);
    debug_assert!(out.len() >= n_rows);
    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2_fma() {
            unsafe { ssm_matvec_avx2(state, vec, dim, n_rows, out) };
            return;
        }
    }
    for r in 0..n_rows {
        out[r] = super::dot_f32(&state[r * dim..][..dim], vec, dim);
    }
}

#[inline(always)]
pub fn ssm_matvec_scaled(
    state: &[f32],
    vec: &[f32],
    dim: usize,
    n_rows: usize,
    out: &mut [f32],
    scale: f32,
) {
    ssm_matvec(state, vec, dim, n_rows, out);
    super::vec_scale_f32(&mut out[..n_rows], scale);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn ssm_matvec_avx2(state: &[f32], vec: &[f32], dim: usize, n_rows: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n8 = dim / 8 * 8;
    for r in 0..n_rows {
        let row = state.as_ptr().add(r * dim);
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i < n8 {
            let vs = _mm256_loadu_ps(row.add(i));
            let vv = _mm256_loadu_ps(vec.as_ptr().add(i));
            acc = _mm256_fmadd_ps(vs, vv, acc);
            i += 8;
        }
        let mut sum = super::hsum_ps(acc);
        while i < dim {
            sum += *row.add(i) * vec[i];
            i += 1;
        }
        out[r] = sum;
    }
}

#[inline(always)]
pub fn ssm_outer_product_update(state: &mut [f32], k: &[f32], d_vec: &[f32], dim: usize) {
    debug_assert_eq!(state.len(), dim * dim);
    debug_assert_eq!(k.len(), dim);
    debug_assert_eq!(d_vec.len(), dim);
    for d in 0..dim {
        super::vec_mad_f32(&mut state[d * dim..(d + 1) * dim], k, d_vec[d]);
    }
}
