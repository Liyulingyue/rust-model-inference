//! Synchronous projection offload. Each cache dies with its owning kernel.

use super::Kernel;
use crate::core::tensor::GGMLType;
use crate::ops::quant::BlockQ8K;
use crate::vulkan::ops::{
    AttentionRuntime, BatchedLinearRuntime, Conv2dRuntime, Conv3dRuntime, GpuMatmulSpec,
    GpuWeightFormat,
};
use std::sync::Mutex;

#[derive(Default)]
pub(crate) struct GpuLinear {
    state: Mutex<Option<LinearState>>,
}

struct LinearState {
    weight: (usize, usize),
    spec: GpuMatmulSpec,
    shape: (usize, usize, usize),
    runtime: BatchedLinearRuntime,
}

/// Smallest number of output elements worth a GPU submission.
///
/// Every `BatchedLinearRuntime::matmul_rows` call ends in its own
/// `submit_and_wait`, and that fence costs ~2.5 ms on the GB10 + driver
/// combination this project targets. A matmul only wins if it does enough
/// work to pay that back.
///
/// LFM2.5-8B (MoE) is the pathological case: decode runs `rows=1`,
/// `n_out=1792` expert projections, 24 layers x top-4 experts x 3, so one
/// token pays 288 fences. Measured on that model, the GPU path took 20.3 s
/// for a 2-token run against 2.1 s with the GPU matmul disabled -- an ~10x
/// regression purely from fence latency. Those calls are 1792 output
/// elements, so they now stay on the CPU pool, while image/diffusion
/// matmuls (millions of output elements per call) are unaffected.
///
/// Override with `RUST_GPU_MIN_OUTPUTS=0` to restore the old behaviour.
fn min_gpu_matmul_outputs() -> usize {
    std::env::var("RUST_GPU_MIN_OUTPUTS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16 * 1024)
}

pub(crate) fn offload_enabled() -> bool {
    crate::ops::gpu_requested()
        && !crate::ops::scalar_mode()
        && !crate::core::thread_pool::gpu_matmul_disabled()
        && !crate::vulkan::gpu_broken()
        && std::env::var_os("RMI_PARITY_TRACE").is_none()
}

impl GpuLinear {
    pub(crate) fn tile_rows(
        spec: impl Into<GpuMatmulSpec>,
        n_in: usize,
        n_out: usize,
        rows: usize,
    ) -> usize {
        let spec = spec.into();
        // The tiled paths amortize a fence over many rows: F16 in the prepared
        // mode, and BF16/F32 in the prepared mode through the transposed float
        // kernel. The dot and rounded modes reduce serially and stay small.
        let limit = if spec.uses_tiled_f16()
            || (matches!(spec.format, GpuWeightFormat::BF16 | GpuWeightFormat::F32)
                && spec.mode == crate::vulkan::ops::GpuMatmulMode::Prepared)
        {
            // Target 16 MiB of host input/output while amortizing fence waits.
            (4 * 1024 * 1024 / n_in.saturating_add(n_out).max(1)).clamp(1, 4096)
        } else {
            64
        };
        rows.min(limit)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_matmul(
        &self,
        weight: &[u8],
        spec: impl Into<GpuMatmulSpec>,
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        rows: usize,
    ) -> bool {
        let spec = spec.into();
        if !offload_enabled()
            || rows == 0
            || n_in == 0
            || n_out == 0
            || rows.checked_mul(n_in) != Some(input.len())
            || rows.checked_mul(n_out) != Some(output.len())
            || input.iter().any(|value| !value.is_finite())
            || n_out
                > std::env::var("RUST_GPU_MAX_ROWS")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(usize::MAX)
            || rows
                .checked_mul(n_out)
                .is_none_or(|outputs| outputs < min_gpu_matmul_outputs())
        {
            return false;
        }
        let Some(context) = crate::ops::get_vulkan_context() else {
            return false;
        };
        let tile_rows = Self::tile_rows(spec, n_in, n_out, rows);
        let key = (weight.as_ptr() as usize, weight.len());
        let mut state = self.state.lock().unwrap();
        let result = (|| {
            if state.as_ref().is_none_or(|cached| {
                cached.weight != key
                    || cached.spec != spec
                    || cached.shape.0 < tile_rows
                    || cached.shape.1 != n_in
                    || cached.shape.2 != n_out
            }) {
                *state = None;
                let runtime = BatchedLinearRuntime::new(context, tile_rows, n_in, n_out, 2)?;
                *state = Some(LinearState {
                    weight: key,
                    spec,
                    shape: (tile_rows, n_in, n_out),
                    runtime,
                });
            }
            let runtime = &mut state.as_mut().unwrap().runtime;
            for (input, output) in input
                .chunks(tile_rows * n_in)
                .zip(output.chunks_mut(tile_rows * n_out))
            {
                runtime.matmul_rows(
                    weight,
                    spec,
                    input,
                    input.len() / n_in,
                    n_in,
                    n_out,
                    output,
                )?;
            }
            if output.iter().any(|value| !value.is_finite()) {
                return Err(crate::vulkan::VulkanError::InitFailed(
                    "non-finite projection output".into(),
                ));
            }
            Ok(())
        })();
        match result {
            Ok(()) => true,
            Err(crate::vulkan::VulkanError::UnsupportedShape(_)) => false,
            Err(error) => {
                crate::vulkan::mark_gpu_broken(&error.to_string());
                *state = None;
                false
            }
        }
    }
}

pub(crate) fn try_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    spec: crate::models::diffusion::dreamx::kernels::AttentionSpec,
) -> Option<Vec<f32>> {
    if !offload_enabled() || spec.head_dim > 1024 {
        return None;
    }
    static RUNTIME: std::sync::OnceLock<Mutex<Option<AttentionRuntime>>> =
        std::sync::OnceLock::new();
    let context = crate::ops::get_vulkan_context()?;
    let mut state = RUNTIME.get_or_init(Mutex::default).lock().unwrap();
    let mut output = vec![0.0; query.len()];
    let result = (|| {
        if state.as_ref().is_none_or(|runtime| {
            query.len() > runtime.capacity.0 || key.len() > runtime.capacity.1
        }) {
            let old = state.as_ref().map_or((0, 0), |runtime| runtime.capacity);
            *state = None;
            *state = Some(AttentionRuntime::new(
                context,
                (old.0.max(query.len()), old.1.max(key.len())),
            )?);
        }
        state
            .as_mut()
            .unwrap()
            .forward(query, key, value, &mut output, spec)
    })();
    match result {
        Ok(()) if output.iter().all(|v| v.is_finite()) => Some(output),
        Ok(()) => {
            crate::vulkan::mark_gpu_broken("non-finite attention output");
            *state = None;
            None
        }
        Err(crate::vulkan::VulkanError::UnsupportedShape(_)) => None,
        Err(error) => {
            crate::vulkan::mark_gpu_broken(&error.to_string());
            *state = None;
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_conv3d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    input_shape: [usize; 4],
    weight_shape: [usize; 5],
    output_shape: [usize; 4],
    spec: crate::models::diffusion::dreamx::kernels::Conv3dSpec,
) -> bool {
    if !offload_enabled() || spec.groups != 1 {
        return false;
    }
    let Some(context) = crate::ops::get_vulkan_context() else {
        return false;
    };
    static RUNTIME: std::sync::OnceLock<Mutex<Option<Conv3dRuntime>>> = std::sync::OnceLock::new();
    let mut state = RUNTIME.get_or_init(Mutex::default).lock().unwrap();
    let needed = [input.len(), weight.len(), output.len(), output_shape[0]];
    let result = (|| {
        if state
            .as_ref()
            .is_none_or(|runtime| needed.iter().zip(runtime.capacity).any(|(&n, c)| n > c))
        {
            let old = state.as_ref().map_or([0; 4], |runtime| runtime.capacity);
            *state = None;
            *state = Some(Conv3dRuntime::new(
                context,
                std::array::from_fn(|i| old[i].max(needed[i])),
            )?);
        }
        state.as_mut().unwrap().forward(
            input,
            weight,
            bias,
            output,
            input_shape,
            weight_shape,
            output_shape,
            spec,
        )
    })();
    match result {
        Ok(()) if output.iter().all(|v| v.is_finite()) => true,
        Ok(()) => {
            crate::vulkan::mark_gpu_broken("non-finite Conv3D output");
            *state = None;
            false
        }
        Err(crate::vulkan::VulkanError::UnsupportedShape(_)) => false,
        Err(error) => {
            crate::vulkan::mark_gpu_broken(&error.to_string());
            *state = None;
            false
        }
    }
}

#[derive(Default)]
pub(crate) struct GpuConv {
    state: Mutex<Option<Conv2dRuntime>>,
}

impl GpuConv {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_conv_f16(
        &self,
        weight: &[u8],
        input: &[f32],
        output: &mut [f32],
        input_channels: usize,
        output_channels: usize,
        side: usize,
        kernel: usize,
        bias: Option<&[f32]>,
    ) -> bool {
        let Some(spatial) = side.checked_mul(side) else {
            return false;
        };
        if !offload_enabled()
            || side == 0
            || input_channels == 0
            || output_channels == 0
            || !matches!(kernel, 1 | 3)
            || input_channels.checked_mul(spatial) != Some(input.len())
            || output_channels.checked_mul(spatial) != Some(output.len())
            || input_channels
                .checked_mul(kernel)
                .and_then(|n| n.checked_mul(kernel))
                .and_then(|n| n.checked_mul(output_channels))
                .and_then(|n| n.checked_mul(2))
                != Some(weight.len())
            || bias.is_some_and(|bias| {
                bias.len() != output_channels || bias.iter().any(|v| !v.is_finite())
            })
            || input.iter().any(|v| !v.is_finite())
            || output_channels
                > std::env::var("RUST_GPU_MAX_ROWS")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(usize::MAX)
        {
            return false;
        }
        let Some(context) = crate::ops::get_vulkan_context() else {
            return false;
        };
        let mut state = self.state.lock().unwrap();
        let result = (|| {
            let old = state.as_ref().map_or((0, 0, 0), |runtime| runtime.capacity);
            if input.len() > old.0 || output.len() > old.1 || output_channels > old.2 {
                *state = None;
                *state = Some(Conv2dRuntime::new(
                    context,
                    (
                        input.len().max(old.0),
                        output.len().max(old.1),
                        output_channels.max(old.2),
                    ),
                )?);
            }
            state.as_mut().unwrap().conv_f16(
                weight,
                input,
                output,
                input_channels,
                output_channels,
                side,
                kernel,
                bias,
            )?;
            if output.iter().any(|value| !value.is_finite()) {
                return Err(crate::vulkan::VulkanError::InitFailed(
                    "non-finite convolution output".into(),
                ));
            }
            Ok(())
        })();
        match result {
            Ok(()) => true,
            Err(crate::vulkan::VulkanError::UnsupportedShape(_)) => false,
            Err(error) => {
                crate::vulkan::mark_gpu_broken(&error.to_string());
                *state = None;
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn dropping_q8_kernel_releases_legacy_cached_weights() {
        let context = crate::ops::float::shared_vulkan_context().unwrap();
        let mut weight = vec![1u8; 34];
        weight[..2].copy_from_slice(&crate::ops::f32_to_f16(1.0).to_le_bytes());
        for value in [1u8, 2] {
            weight[2..].fill(value);
            let kernel = super::super::q8_0::Q8Kernel::new(&weight);
            let mut output = [0.0];
            unsafe {
                context
                    .matmul_q8_0(
                        kernel.weight_bytes().unwrap(),
                        &[1; 32],
                        &[1.0],
                        &mut output,
                        32,
                        1,
                    )
                    .unwrap();
            }
            assert_eq!(output, [32.0 * value as f32]);
        }
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn dropping_owned_q8_kernel_releases_legacy_cached_weights() {
        let context = crate::ops::float::shared_vulkan_context().unwrap();
        let mut weight = vec![1u8; 34];
        weight[..2].copy_from_slice(&crate::ops::f32_to_f16(1.0).to_le_bytes());
        let kernel = super::super::QTensorOwned::from_bytes_owned(&weight, GGMLType::Q8_0, 32, 1);
        let super::super::QTensorOwned::Q8_0 { data, .. } = &kernel else {
            unreachable!()
        };
        let mut output = [0.0];
        unsafe {
            context
                .matmul_q8_0(data, &[1; 32], &[1.0], &mut output, 32, 1)
                .unwrap();
        }
        assert_eq!(output, [32.0]);
        let live = context.compute_stats().live_allocation_bytes;
        drop(kernel);
        assert!(context.compute_stats().live_allocation_bytes < live);
    }

    #[test]
    fn f16_batches_bound_memory_and_preserve_dot_contracts() {
        assert_eq!(
            GpuLinear::tile_rows(GpuWeightFormat::F16, 512, 512, 4225),
            4096
        );
        assert_eq!(GpuLinear::tile_rows(GpuWeightFormat::F16, 512, 512, 17), 17);
        let rows = GpuLinear::tile_rows(GpuWeightFormat::F16, 4608, 512, 4096);
        assert!(rows > 64 && rows * (4608 + 512) * 4 <= 16 * 1024 * 1024);
        for format in [
            GpuWeightFormat::F16.with_mode(crate::vulkan::ops::GpuMatmulMode::Dot),
            GpuWeightFormat::BF16.with_mode(crate::vulkan::ops::GpuMatmulMode::Dot),
            GpuWeightFormat::Q8_0.into(),
        ] {
            assert_eq!(GpuLinear::tile_rows(format, 512, 512, 1089), 64);
        }
    }
}

pub(super) struct VulkanKernel<'a> {
    inner: Box<dyn Kernel + 'a>,
    format: GpuWeightFormat,
    gpu: GpuLinear,
}

impl<'a> VulkanKernel<'a> {
    pub(super) fn wrap(inner: Box<dyn Kernel + 'a>, kind: GGMLType) -> Box<dyn Kernel + 'a> {
        match GpuWeightFormat::from_ggml_type(kind) {
            Ok(format) => Box::new(Self {
                inner,
                format,
                gpu: GpuLinear::default(),
            }),
            Err(_) => inner,
        }
    }
}

impl Kernel for VulkanKernel<'_> {
    fn supports_f16_strict(&self) -> bool {
        self.inner.supports_f16_strict()
    }
    fn rounds_bf16_input(&self) -> bool {
        self.inner.rounds_bf16_input()
    }

    fn try_forward_vulkan_rows(
        &self,
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        rows: usize,
    ) -> bool {
        self.inner.weight_bytes().is_some_and(|bytes| {
            self.gpu
                .try_matmul(bytes, self.format, input, output, n_in, n_out, rows)
        })
    }

    fn f32_slice(&self) -> Option<&[f32]> {
        self.inner.f32_slice()
    }
    fn bf16_bytes(&self) -> Option<&[u8]> {
        self.inner.bf16_bytes()
    }
    fn weight_bytes(&self) -> Option<&[u8]> {
        self.inner.weight_bytes()
    }
    fn scalar_q4_0_bytes(&self) -> Option<&[u8]> {
        self.inner.scalar_q4_0_bytes()
    }

    fn forward_prequantized(
        &self,
        input: &[u8],
        scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        self.inner
            .forward_prequantized(input, scales, output, n_in, n_out, ith, nth);
    }
    fn forward_prepared(
        &self,
        input: &[f32],
        q8: &[u8],
        scales: &[f32],
        q8k: Option<&[BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        self.inner
            .forward_prepared(input, q8, scales, q8k, output, n_in, n_out, ith, nth);
    }
    fn forward_f16_strict(
        &self,
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool {
        self.inner.forward_f16_strict(input, output, n_in, n_out)
    }
    fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        self.inner.forward(input, output, n_in, n_out);
    }
    fn forward_batched(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize) {
        self.inner.forward_batched(input, output, n_in, n_out);
    }
    fn embedding_lookup(&self, token_id: u32, n_embd: usize, output: &mut [f32]) {
        self.inner.embedding_lookup(token_id, n_embd, output);
    }
}
