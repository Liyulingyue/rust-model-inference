//! Synchronous projection offload. Each cache dies with its owning kernel.

use super::Kernel;
use crate::core::tensor::GGMLType;
use crate::ops::quant::BlockQ8K;
use crate::vulkan::ops::{BatchedLinearRuntime, GpuWeightFormat};
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
        // ponytail: one arena per projection, 64-row tiles bound scratch memory;
        // share a model arena if target-machine measurements show memory pressure.
        let tile_rows = rows.min(64);
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
