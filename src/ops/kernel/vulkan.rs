//! Synchronous projection offload. Each cache dies with its owning kernel.

use super::Kernel;
use crate::core::tensor::GGMLType;
use crate::ops::quant::BlockQ8K;
use crate::vulkan::ops::{BatchedLinearRuntime, Conv2dRuntime, GpuWeightFormat};
use std::sync::Mutex;

#[derive(Default)]
pub(crate) struct GpuLinear {
    state: Mutex<Option<LinearState>>,
}

struct LinearState {
    weight: (usize, usize),
    format: GpuWeightFormat,
    shape: (usize, usize, usize),
    runtime: BatchedLinearRuntime,
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
        format: GpuWeightFormat,
        n_in: usize,
        n_out: usize,
        rows: usize,
    ) -> usize {
        let limit = if format == GpuWeightFormat::F16 {
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
        format: GpuWeightFormat,
        input: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        rows: usize,
    ) -> bool {
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
        {
            return false;
        }
        let Some(context) = crate::ops::get_vulkan_context() else {
            return false;
        };
        let tile_rows = Self::tile_rows(format, n_in, n_out, rows);
        let key = (weight.as_ptr() as usize, weight.len());
        let mut state = self.state.lock().unwrap();
        let result = (|| {
            if state.as_ref().is_none_or(|cached| {
                cached.weight != key
                    || cached.format != format
                    || cached.shape.0 < tile_rows
                    || cached.shape.1 != n_in
                    || cached.shape.2 != n_out
            }) {
                *state = None;
                let runtime = BatchedLinearRuntime::new(context, tile_rows, n_in, n_out, 2)?;
                *state = Some(LinearState {
                    weight: key,
                    format,
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
                    format,
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
    fn f16_batches_bound_memory_and_preserve_dot_contracts() {
        assert_eq!(
            GpuLinear::tile_rows(GpuWeightFormat::F16, 512, 512, 4225),
            4096
        );
        assert_eq!(GpuLinear::tile_rows(GpuWeightFormat::F16, 512, 512, 17), 17);
        let rows = GpuLinear::tile_rows(GpuWeightFormat::F16, 4608, 512, 4096);
        assert!(rows > 64 && rows * (4608 + 512) * 4 <= 16 * 1024 * 1024);
        for format in [
            GpuWeightFormat::F16Dot,
            GpuWeightFormat::BF16Dot,
            GpuWeightFormat::Q8_0,
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
