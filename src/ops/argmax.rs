//! SIMD argmax over `&[f32]` returning the index of the maximum
//! element. Used by `greedy_ctc_decode` in the SenseVoice ASR head
//! where the caller does this once per output frame over a 25k-entry
//! vocab. The scalar baseline is `O(n)` f32 comparisons; AVX2 packs
//! 8 lanes per iter with a running (value, index) max and avoids the
//! per-element scalar branch.

#[inline]
pub fn argmax_f32(values: &[f32]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::ops::has_avx2_fma() {
            unsafe { return argmax_f32_avx2(values) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if crate::ops::has_neon() {
            unsafe { return argmax_f32_neon(values) };
        }
    }
    argmax_f32_scalar(values)
}

fn argmax_f32_scalar(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// AVX2 argmax with 8-wide packed comparisons. The running max
/// combines (value, lane_index) so we can hsum-reduce at the end
/// without losing the index. Lane indices are `0..8` per packed
/// chunk; the absolute index is recovered by `chunk_base + lane_idx`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn argmax_f32_avx2(values: &[f32]) -> usize {
    use std::arch::x86_64::*;

    let n = values.len();
    if n == 0 {
        return 0;
    }
    let n8 = n / 8 * 8;

    // Running max: pack (max_value, max_lane_index) so the index of
    // the largest lane is preserved when more than one lane ties.
    let mut v_best_val = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut v_best_idx = _mm256_set1_epi32(-1i32);
    let lane_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    let mut chunk_base = 0i32;
    let mut i = 0;
    while i < n8 {
        let v = _mm256_loadu_ps(values.as_ptr().add(i));
        let idx = _mm256_add_epi32(lane_indices, _mm256_set1_epi32(chunk_base));
        // Compare value-against-value: lanes with strictly greater
        // value replace the running (value, idx). `_mm256_cmp_ps`
        // returns `__m256i` with all-1s per lane where the predicate
        // holds; `v_best_val < v` → use new (v, idx); otherwise keep
        // running. Ties (predicate false) keep the *old* index, so
        // argmax breaks ties by lowest index, matching the scalar
        // reference.
        let mask = _mm256_cmp_ps::<{ _CMP_LT_OQ }>(v_best_val, v);
        let merged_val = _mm256_blendv_ps(v_best_val, v, mask);
        // `_mm256_blendv_epi32` isn't in `std::arch::x86_64`; cast the
        // i32 lanes to f32 (same bit pattern → blendv_ps treats the
        // sign bit as the mask, which is exactly what we want since
        // `_mm256_cmp_ps` returned all-1s / all-0s per lane).
        let idx_as_ps = _mm256_castsi256_ps(idx);
        let best_idx_as_ps = _mm256_castsi256_ps(v_best_idx);
        let merged_idx_as_ps =
            _mm256_blendv_ps(best_idx_as_ps, idx_as_ps, mask);
        v_best_val = merged_val;
        v_best_idx = _mm256_castps_si256(merged_idx_as_ps);
        i += 8;
        chunk_base += 8;
    }

    // Horizontal reduction: extract (value, index) pairs across lanes,
    // pick the max-value pair.
    let mut best_val = f32::NEG_INFINITY;
    let mut best_idx = 0i32;
    let mut buf_val = [0.0f32; 8];
    let mut buf_idx = [0i32; 8];
    _mm256_storeu_ps(buf_val.as_mut_ptr(), v_best_val);
    _mm256_storeu_si256(buf_idx.as_mut_ptr() as *mut __m256i, v_best_idx);
    for lane in 0..8 {
        if buf_val[lane] > best_val {
            best_val = buf_val[lane];
            best_idx = buf_idx[lane];
        }
    }

    // Tail: scalar fallback for any leftover lanes.
    for j in n8..n {
        if values[j] > best_val {
            best_val = values[j];
            best_idx = j as i32;
        }
    }

    best_idx as usize
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn argmax_f32_neon(values: &[f32]) -> usize {
    use std::arch::aarch64::*;

    let n = values.len();
    if n == 0 {
        return 0;
    }
    let n4 = n / 4 * 4;

    let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
    let mut v_best_val = neg_inf;
    let mut v_best_idx = vdupq_n_s32(-1i32);
    let lane_inc = vdupq_n_s32(4);
    let mut lane_offset = vdupq_n_s32(0);

    let mut i = 0;
    while i < n4 {
        let v = vld1q_f32(values.as_ptr().add(i));
        let idx = lane_offset;
        // vminq_f32 + vceqq: where v > v_best_val, use new (v, idx);
        // otherwise keep running (v_best_val, v_best_idx). Bitwise
        // blend via vbslq.
        let mask = vcgtq_f32(v, v_best_val);
        v_best_val = vbslq_f32(mask, v, v_best_val);
        v_best_idx = vbslq_s32(mask, idx, v_best_idx);
        lane_offset = vaddq_s32(lane_offset, lane_inc);
        i += 4;
    }

    let mut best_val = vmaxvq_f32(v_best_val);
    let mut buf_val = [0.0f32; 4];
    let mut buf_idx = [0i32; 4];
    vst1q_f32(buf_val.as_mut_ptr(), v_best_val);
    vst1q_s32(buf_idx.as_mut_ptr(), v_best_idx);
    let mut best_idx = 0i32;
    for lane in 0..4 {
        if buf_val[lane] == best_val {
            best_idx = best_idx.max(buf_idx[lane]);
        }
    }

    for j in n4..n {
        if values[j] > best_val {
            best_val = values[j];
            best_idx = j as i32;
        }
    }

    best_idx as usize
}

#[cfg(test)]
mod tests {
    use super::argmax_f32;

    fn assert_argmax_matches_scalar(values: &[f32], expected: usize) {
        let got = argmax_f32(values);
        assert_eq!(got, expected, "values={values:?}");
    }

    #[test]
    fn empty_returns_zero() {
        assert_eq!(argmax_f32(&[]), 0);
    }

    #[test]
    fn single_element() {
        assert_argmax_matches_scalar(&[42.0], 0);
    }

    #[test]
    fn first_max() {
        assert_argmax_matches_scalar(&[10.0, 1.0, 2.0, 3.0, -5.0], 0);
    }

    #[test]
    fn middle_max() {
        assert_argmax_matches_scalar(&[1.0, 2.0, 100.0, 3.0, 4.0], 2);
    }

    #[test]
    fn last_max() {
        assert_argmax_matches_scalar(&[1.0, 2.0, 3.0, 4.0, 5.0], 4);
    }

    #[test]
    fn all_negative() {
        assert_argmax_matches_scalar(&[-3.0, -1.0, -2.0, -5.0], 1);
    }

    #[test]
    fn ties_pick_lowest_index() {
        // Ties are broken by lowest index; matches scalar reference.
        assert_argmax_matches_scalar(&[5.0, 5.0, 5.0, 5.0], 0);
        assert_argmax_matches_scalar(&[1.0, 2.0, 2.0, 2.0, 1.0], 1);
    }

    #[test]
    fn large_random() {
        let values: Vec<f32> = (0..1024)
            .map(|i| ((i * 17 + 13) % 97) as f32 * 0.07 - 4.5)
            .collect();
        let scalar = values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(argmax_f32(&values), scalar);
    }

    #[test]
    fn tail_handles_non_aligned_lengths() {
        for &n in &[1usize, 3, 7, 8, 9, 16, 17, 33, 65, 100, 129, 256, 1024] {
            let values: Vec<f32> =
                (0..n).map(|i| i as f32 * 0.13 - 7.0).collect();
            // Make last element the max for varying lengths.
            let mut values = values;
            if n > 0 {
                values[n - 1] = 100.0;
            }
            let scalar = values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, &0.0))
                .0;
            assert_eq!(
                argmax_f32(&values),
                scalar,
                "n={n}: simd and scalar picked different indices"
            );
        }
    }
}
