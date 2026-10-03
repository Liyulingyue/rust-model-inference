//! RMS normalization + scale helpers.
//!
//! `sum_sq_f32`（通用 reduce Σ x² → f64）和它的 AVX2/NEON 内核放在
//! `ops/dot.rs`，与 `hsum_ps` 等横向归约工具同处。本文件只调用之。

/// Layer normalization (BERT family, `ggml_norm` + affine weight/bias).
///
/// `y[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]`
/// with `mean = Σx / n` and `var = Σ(x - mean)² / n`, both accumulated in
/// f64 to match the reduction order of the pinned ggml kernels.
///
/// `bias` may be empty when a tensor carries only an affine scale.
pub fn layer_norm(input: &[f32], weight: &[f32], bias: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len().min(weight.len()).min(output.len());
    // ModernBERT's LayerNorm declares `bias: false`, so the affine step is scale
    // only. An empty bias means "no bias" rather than a shape mismatch.
    assert!(
        bias.is_empty() || bias.len() == n,
        "layer_norm bias must be empty or match the row length"
    );
    if n == 0 {
        return;
    }
    let sum: f64 = input[..n].iter().map(|&value| f64::from(value)).sum();
    let mean = sum / n as f64;
    let var: f64 = input[..n]
        .iter()
        .map(|&value| {
            let centered = f64::from(value) - mean;
            centered * centered
        })
        .sum();
    let var = var / n as f64;
    let scale = 1.0f32 / (var as f32 + eps).sqrt();
    if bias.is_empty() {
        for i in 0..n {
            output[i] = (input[i] - mean as f32) * scale * weight[i];
        }
        return;
    }
    for i in 0..n {
        output[i] = (input[i] - mean as f32) * scale * weight[i] + bias[i];
    }
}

pub fn rms_norm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = input.len().min(weight.len()).min(output.len());
    let sum_sq = super::sum_sq_f32(&input[..n]);
    let mean_sq = (sum_sq / n as f64) as f32;
    let scale = 1.0f32 / (mean_sq + eps).sqrt();
    for i in 0..n {
        output[i] = input[i] * scale * weight[i];
    }
}

pub fn rms_norm_grouped(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    groups: usize,
    eps: f32,
) {
    let n = input.len().min(weight.len()).min(output.len());
    assert!(groups > 0, "groups must be greater than zero");
    if n == 0 {
        return;
    }
    assert!(
        n.is_multiple_of(groups),
        "normalized length {n} must be divisible by {groups} groups"
    );
    let group_size = n / groups;
    for group in 0..groups {
        let range = group * group_size..(group + 1) * group_size;
        rms_norm(
            &input[range.clone()],
            &weight[range.clone()],
            &mut output[range],
            eps,
        );
    }
}

pub fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len().min(weight.len());
    let sum_sq = super::sum_sq_f32(&x[..n]);
    let mean_sq = (sum_sq / n as f64) as f32;
    let scale = 1.0f32 / (mean_sq + eps).sqrt();
    scale_mul_inplace(scale, &weight[..n], &mut x[..n]);
}

/// In-place RMS normalization with unit weight: `x[i] *= 1.0 / rms(x)`.
/// Equivalent to `rms_norm_inplace(x, &[1.0; x.len()], eps)` but skips the
/// per-element weight multiply.
pub fn rms_unit_inplace(x: &mut [f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let sum_sq = super::sum_sq_f32(x);
    let mean_sq = (sum_sq / n as f64) as f32;
    let scale = 1.0f32 / (mean_sq + eps).sqrt();
    scale_inplace(scale, x);
}

fn scale_inplace(scale: f32, x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2_fma() {
            unsafe { scale_avx2(scale, x) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if super::has_neon() {
            unsafe { scale_neon(scale, x) };
            return;
        }
    }
    for value in x.iter_mut() {
        *value *= scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn scale_avx2(scale: f32, x: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let n8 = n / 8 * 8;
    let vscale = _mm256_set1_ps(scale);
    let mut i = 0;
    while i < n8 {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(vx, vscale));
        i += 8;
    }
    while i < n {
        x[i] *= scale;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_neon(scale: f32, x: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let vscale = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= n {
        let v = vmulq_f32(vld1q_f32(x.as_ptr().add(i)), vscale);
        vst1q_f32(x.as_mut_ptr().add(i), v);
        i += 4;
    }
    while i < n {
        x[i] *= scale;
        i += 1;
    }
}

fn scale_mul_inplace(scale: f32, weight: &[f32], x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2_fma() {
            unsafe { scale_mul_avx2(scale, weight, x) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if super::has_neon() {
            unsafe {
                scale_mul_neon(scale, weight, x);
            }
            return;
        }
    }
    for i in 0..weight.len() {
        x[i] = x[i] * scale * weight[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_mul_neon(scale: f32, weight: &[f32], x: &mut [f32]) {
    use std::arch::aarch64::*;
    let scale_v = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= x.len() {
        let value = vmulq_f32(
            vmulq_f32(vld1q_f32(x.as_ptr().add(i)), scale_v),
            vld1q_f32(weight.as_ptr().add(i)),
        );
        vst1q_f32(x.as_mut_ptr().add(i), value);
        i += 4;
    }
    while i < x.len() {
        x[i] = x[i] * scale * weight[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn scale_mul_avx2(scale: f32, weight: &[f32], x: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = weight.len();
    let n8 = n / 8 * 8;
    let vscale = _mm256_set1_ps(scale);
    let mut i = 0;
    while i < n8 {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        let vw = _mm256_loadu_ps(weight.as_ptr().add(i));
        _mm256_storeu_ps(
            x.as_mut_ptr().add(i),
            _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw),
        );
        i += 8;
    }
    while i < n {
        x[i] = x[i] * scale * weight[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::rms_norm_grouped;

    #[test]
    fn grouped_rms_norm_normalizes_contiguous_groups() {
        let input = [3.0, 4.0, 0.0, 5.0];
        let weight = [1.0, 1.0, 2.0, 2.0];
        let mut output = [0.0; 4];

        rms_norm_grouped(&input, &weight, &mut output, 2, 0.0);

        let expected: [f32; 4] = [0.848_528_15, 1.131_370_9, 0.0, 2.828_427];
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn grouped_rms_norm_accepts_empty_input() {
        rms_norm_grouped(&[], &[], &mut [], 1, 1e-5);
    }

    #[test]
    #[should_panic(expected = "groups must be greater than zero")]
    fn grouped_rms_norm_rejects_zero_groups() {
        rms_norm_grouped(&[], &[], &mut [], 0, 1e-5);
    }
}
