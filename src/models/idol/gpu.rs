//! Idol's synchronous device operators; CPU callers keep their existing reductions.
use rayon::prelude::*;

#[derive(Default)]
pub(super) struct Projection {
    #[cfg(feature = "vulkan")]
    inner: crate::ops::kernel::vulkan::GpuLinear,
}

pub(super) fn enabled() -> bool {
    #[cfg(feature = "vulkan")]
    return crate::ops::kernel::vulkan::offload_enabled();
    #[cfg(not(feature = "vulkan"))]
    false
}

pub(super) fn check_failure() -> Result<(), String> {
    #[cfg(feature = "vulkan")]
    if crate::compute::ComputePolicy::legacy() == crate::compute::ComputePolicy::Vulkan
        && crate::vulkan::gpu_broken()
    {
        return Err("Idol Vulkan device failed; refusing a silent CPU fallback".into());
    }
    Ok(())
}

impl Projection {
    pub fn linear(
        &self,
        weight: &[f32],
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool {
        #[cfg(feature = "vulkan")]
        return n_in > 0
            && self.inner.try_matmul(
                bytemuck::cast_slice(weight),
                crate::vulkan::ops::GpuWeightFormat::F32,
                input,
                output,
                n_in,
                n_out,
                input.len() / n_in,
            );
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (weight, input, output, n_in, n_out);
            false
        }
    }

    /// NHWC im2col tiles bound host staging to 256 pixels, including stride-two asymmetric padding.
    #[allow(clippy::too_many_arguments)]
    pub fn convolution(
        &self,
        weight: &[f32],
        input: &[f32],
        output: &mut [f32],
        channels: [usize; 2],
        input_hw: [usize; 2],
        output_hw: [usize; 2],
        kernel: usize,
        stride: usize,
        padding: usize,
        channel_first_kernel: bool,
    ) -> bool {
        if !enabled() {
            return false;
        }
        let [n_in, n_out] = channels;
        let [height, width] = input_hw;
        let [out_h, out_w] = output_hw;
        let Some(patch_width) = n_in.checked_mul(kernel).and_then(|n| n.checked_mul(kernel)) else {
            return false;
        };
        if [n_in, n_out, height, width, out_h, out_w, kernel, stride].contains(&0)
            || height.checked_mul(width).and_then(|n| n.checked_mul(n_in)) != Some(input.len())
            || out_h.checked_mul(out_w).and_then(|n| n.checked_mul(n_out)) != Some(output.len())
            || patch_width.checked_mul(n_out) != Some(weight.len())
        {
            return false;
        }
        let mut patches = vec![0.; 256 * patch_width];
        for (tile, result) in output.chunks_mut(256 * n_out).enumerate() {
            let rows = result.len() / n_out;
            let patches = &mut patches[..rows * patch_width];
            patches
                .par_chunks_mut(patch_width)
                .enumerate()
                .for_each(|(row, patch)| {
                    let pixel = tile * 256 + row;
                    for (index, value) in patch.iter_mut().enumerate() {
                        let (channel, ky, kx) = if channel_first_kernel {
                            (
                                index / (kernel * kernel),
                                index / kernel % kernel,
                                index % kernel,
                            )
                        } else {
                            (index % n_in, index / n_in / kernel, index / n_in % kernel)
                        };
                        let y = (pixel / out_w * stride + ky) as isize - padding as isize;
                        let x = (pixel % out_w * stride + kx) as isize - padding as isize;
                        *value = if y >= 0 && x >= 0 && y < height as isize && x < width as isize {
                            input[(y as usize * width + x as usize) * n_in + channel]
                        } else {
                            0.
                        };
                    }
                });
            if !self.linear(weight, patches, result, patch_width, n_out) {
                return false;
            }
        }
        true
    }
}

pub(super) fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    width: usize,
    heads: usize,
) -> Option<Vec<f32>> {
    #[cfg(feature = "vulkan")]
    {
        use crate::vulkan::ops::AttentionRuntime;
        use std::sync::{Mutex, OnceLock};
        if !enabled() || width == 0 || heads == 0 || width % heads != 0 {
            return None;
        }
        let context = crate::ops::get_vulkan_context()?;
        static RUNTIME: OnceLock<Mutex<Option<AttentionRuntime>>> = OnceLock::new();
        let mut state = RUNTIME.get_or_init(Mutex::default).lock().ok()?;
        let result = (|| {
            if state
                .as_ref()
                .is_none_or(|r| q.len() > r.capacity.0 || k.len() > r.capacity.1)
            {
                let old = state.as_ref().map_or((0, 0), |r| r.capacity);
                *state = None;
                *state = Some(AttentionRuntime::new(
                    context,
                    (old.0.max(q.len()), old.1.max(k.len())),
                )?);
            }
            let mut output = vec![0.; q.len()];
            state.as_mut().unwrap().forward(
                q,
                k,
                v,
                &mut output,
                crate::ops::nn::AttentionSpec {
                    query_tokens: q.len() / width,
                    key_tokens: k.len() / width,
                    query_heads: heads,
                    key_value_heads: heads,
                    head_dim: width / heads,
                    causal: false,
                    scale: ((width / heads) as f32).sqrt().recip(),
                },
            )?;
            if output.iter().any(|v| !v.is_finite()) {
                return Err(crate::vulkan::VulkanError::InitFailed(
                    "non-finite Idol attention".into(),
                ));
            }
            Ok(output)
        })();
        return match result {
            Ok(output) => Some(output),
            Err(error) => {
                eprintln!("compute arch=idol scope=attention backend=Cpu reason={error}");
                if !matches!(error, crate::vulkan::VulkanError::UnsupportedShape(_)) {
                    crate::vulkan::mark_gpu_broken(&error.to_string());
                }
                *state = None;
                None
            }
        };
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (q, k, v, width, heads);
        None
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;
    fn device_scope() -> crate::compute::LegacyComputeScope {
        let scope = crate::compute::ComputePolicy::Vulkan
            .configure_cli()
            .unwrap();
        let context = crate::ops::get_vulkan_context().unwrap();
        assert!(!context.is_software_icd());
        scope
    }
    #[test]
    #[ignore = "requires a real Vulkan accelerator"]
    fn device_primitives_match_independent_cpu_reductions() {
        let _scope = device_scope();
        let context = crate::ops::get_vulkan_context().unwrap();
        let before = context.submission_count();
        let projection = Projection::default();
        let weight: Vec<_> = (0..96 * 192)
            .map(|i| (i as f32 * 0.13).sin() * 0.1)
            .collect();
        for (phase, rows) in [(0.1, 128), (0.7, 257), (-0.3, 129)] {
            let input: Vec<_> = (0..rows * 96).map(|i| (i as f32 * phase).cos()).collect();
            let mut output = vec![0.; rows * 192];
            assert!(projection.linear(&weight, &input, &mut output, 96, 192));
            for (row, result) in input.chunks_exact(96).zip(output.chunks_exact(192)) {
                for (w, &actual) in weight.chunks_exact(96).zip(result) {
                    let expected = crate::ops::dot_f32(row, w, 96);
                    assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
                }
            }
            for (queries, keys, heads, dim) in [(67, 71, 3, 64), (69, 73, 1, 512), (17, 23, 4, 64)]
            {
                let width = heads * dim;
                let q: Vec<_> = (0..queries * width)
                    .map(|i| (i as f32 * phase).sin())
                    .collect();
                let k: Vec<_> = (0..keys * width)
                    .map(|i| (i as f32 * phase).cos())
                    .collect();
                let v: Vec<_> = k.iter().map(|v| v * 0.7).collect();
                let actual = attention(&q, &k, &v, width, heads).expect("device attention");
                let expected = crate::ops::nn::attention_on_cpu(
                    &q,
                    &k,
                    &v,
                    crate::ops::nn::AttentionSpec {
                        query_tokens: queries,
                        key_tokens: keys,
                        query_heads: heads,
                        key_value_heads: heads,
                        head_dim: dim,
                        causal: false,
                        scale: (dim as f32).sqrt().recip(),
                    },
                )
                .unwrap();
                assert!(actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() < 2e-5));
            }
        }
        assert!(context.submission_count() > before);
        let before_cpu = context.submission_count();
        let _cpu = crate::compute::ComputePolicy::Cpu.enter_legacy_scope();
        assert!(!projection.linear(
            &weight,
            &vec![0.; 128 * 96],
            &mut vec![0.; 128 * 192],
            96,
            192
        ));
        assert!(attention(&[0.; 64], &[0.; 64], &[0.; 64], 64, 1).is_none());
        assert_eq!(context.submission_count(), before_cpu);
    }
    #[test]
    #[ignore = "requires a real Vulkan accelerator"]
    fn device_attention_full_shapes_match_sampled_cpu_queries() {
        let _scope = device_scope();
        for (queries, keys, heads, dim) in [
            (12294, 12294, 16, 64),
            (4101, 4101, 20, 64),
            (4096, 4096, 1, 512),
        ] {
            let width = heads * dim;
            let q: Vec<_> = (0..queries * width)
                .map(|i| (i as f32 * 0.17).sin())
                .collect();
            let k: Vec<_> = (0..keys * width).map(|i| (i as f32 * 0.23).cos()).collect();
            let v: Vec<_> = k.iter().map(|v| v * 0.7).collect();
            let actual = attention(&q, &k, &v, width, heads).expect("full-shape device attention");
            let indices = [0, 31, 32, queries / 2, queries - 1];
            let sampled: Vec<_> = indices
                .iter()
                .flat_map(|&i| q[i * width..(i + 1) * width].iter().copied())
                .collect();
            let expected = crate::ops::nn::attention_on_cpu(
                &sampled,
                &k,
                &v,
                crate::ops::nn::AttentionSpec {
                    query_tokens: indices.len(),
                    key_tokens: keys,
                    query_heads: heads,
                    key_value_heads: heads,
                    head_dim: dim,
                    causal: false,
                    scale: (dim as f32).sqrt().recip(),
                },
            )
            .unwrap();
            for (&i, reference) in indices.iter().zip(expected.chunks_exact(width)) {
                assert!(
                    actual[i * width..(i + 1) * width]
                        .iter()
                        .zip(reference)
                        .all(|(a, b)| (a - b).abs() < 2e-5),
                    "shape={queries}/{keys}/{heads}/{dim}, query={i}"
                );
            }
        }
    }
    #[test]
    #[ignore = "requires a real Vulkan accelerator"]
    fn device_convolution_matches_cpu_padding_layout_and_stride() {
        let _scope = device_scope();
        for (stride, padding, next) in [(1, 1, 32), (2, 0, 16)] {
            for channel_first in [false, true] {
                let projection = Projection::default();
                let weight: Vec<_> = (0..64 * 3 * 3 * 3)
                    .map(|i| (i as f32 * 0.17).sin() * 0.1)
                    .collect();
                let input: Vec<_> = (0..32 * 32 * 3).map(|i| (i as f32 * 0.23).cos()).collect();
                let mut output = vec![0.; next * next * 64];
                assert!(projection.convolution(
                    &weight,
                    &input,
                    &mut output,
                    [3, 64],
                    [32, 32],
                    [next, next],
                    3,
                    stride,
                    padding,
                    channel_first
                ));
                for y in 0..next {
                    for x in 0..next {
                        for oc in 0..64 {
                            let mut expected = 0.;
                            for ic in 0..3 {
                                for ky in 0..3 {
                                    for kx in 0..3 {
                                        let iy = (y * stride + ky) as isize - padding as isize;
                                        let ix = (x * stride + kx) as isize - padding as isize;
                                        if iy >= 0 && ix >= 0 && iy < 32 && ix < 32 {
                                            let w = if channel_first {
                                                ((oc * 3 + ic) * 3 + ky) * 3 + kx
                                            } else {
                                                ((oc * 3 + ky) * 3 + kx) * 3 + ic
                                            };
                                            expected += input
                                                [(iy as usize * 32 + ix as usize) * 3 + ic]
                                                * weight[w];
                                        }
                                    }
                                }
                            }
                            assert!((output[(y * next + x) * 64 + oc] - expected).abs() < 2e-5);
                        }
                    }
                }
            }
        }
    }
}
