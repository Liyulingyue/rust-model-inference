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
    // Tracing observes the requested backend; CPU/scalar scopes select exact CPU paths.
    crate::ops::gpu_requested()
        && !crate::ops::scalar_mode()
        && !crate::core::thread_pool::gpu_matmul_disabled()
        && !crate::vulkan::gpu_broken()
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
            Ok(()) => {
                if std::env::var_os("RMI_COMPUTE_TRACE").is_some() {
                    eprintln!("compute scope=linear backend=Vulkan rows={rows} counters=device_cumulative {:?}", context.compute_stats());
                }
                true
            }
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
        Ok(()) if output.iter().all(|v| v.is_finite()) => {
            // The device has been seen to return the numbers of an earlier
            // call (TODO-044), so the result is checked against the CPU
            // kernel before it is used. A retry of the same call usually
            // returns the right numbers, and a retry keeps the bits of a
            // healthy run.
            if crate::models::diffusion::dreamx::kernels::attention_matches_cpu(
                query, key, value, &output, spec,
            ) {
                return Some(output);
            }
            let mut attempts = 0;
            while attempts < 3 {
                attempts += 1;
                let mut retry = vec![0.0; query.len()];
                let outcome = state
                    .as_mut()
                    .unwrap()
                    .forward(query, key, value, &mut retry, spec);
                if outcome.is_err() {
                    break;
                }
                if !retry.iter().all(|v| v.is_finite()) {
                    break;
                }
                if crate::models::diffusion::dreamx::kernels::attention_matches_cpu(
                    query, key, value, &retry, spec,
                ) {
                    return Some(retry);
                }
            }
            // The device keeps returning numbers the CPU kernel does not
            // agree with, so the attention is computed on the CPU.
            let _cpu_scope = crate::core::thread_pool::ComputePool::disable_gpu_matmul_for_scope();
            let mut cpu = vec![0.0; query.len()];
            if crate::models::diffusion::dreamx::kernels::attention_on_cpu(query, key, value, spec)
                .map(|values| cpu.copy_from_slice(&values))
                .is_err()
            {
                return None;
            }
            Some(cpu)
        }
        Ok(()) => {
            crate::vulkan::mark_gpu_broken("non-finite attention output");
            *state = None;
            None
        }
        Err(crate::vulkan::VulkanError::UnsupportedShape(error)) => {
            if std::env::var_os("RMI_GPU_VERIFY_TRACE").is_some() {
                eprintln!("[GPU-VERIFY] attention unsupported: {error}");
            }
            None
        }
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

    /// Mage-Flow drives dozens of projections with different shapes in a row,
    /// each with its own runtime and weight upload. Interleave them the way
    /// the model does and check that every projection still reproduces itself.
    #[test]
    #[ignore = "requires a Vulkan device and the real Mage-Flow GGUFs"]
    fn bf16_prepared_many_shapes_interleaved_is_stable() {
        use crate::vulkan::ops::{GpuMatmulSpec, GpuWeightFormat};
        crate::ops::float::enable_gpu();
        if crate::ops::get_vulkan_context().is_none() {
            eprintln!("no vulkan context; skip");
            return;
        }
        let Ok(loader) =
            crate::GGUFLoader::from_file("models/mage-flow/mage-flow-base-dit-BF16.gguf")
        else {
            eprintln!("no real weights; skip");
            return;
        };
        let names = [
            "transformer_blocks.0.attn.add_q_proj.weight",
            "transformer_blocks.0.attn.add_k_proj.weight",
            "transformer_blocks.0.attn.add_v_proj.weight",
            "transformer_blocks.0.attn.to_add_out.weight",
            "transformer_blocks.0.mlp.fc1.weight",
            "transformer_blocks.0.mlp.fc2.weight",
        ];
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        };
        let rows = 1024usize;
        let mut inputs: Vec<Vec<f32>> = Vec::new();
        let mut weights: Vec<&[u8]> = Vec::new();
        let mut shapes: Vec<(usize, usize)> = Vec::new();
        for name in names {
            let Some(bytes) = loader.tensor_slice(name) else {
                continue;
            };
            let Some(info) = loader.tensor_info(name) else {
                continue;
            };
            let n_in = info.dims[1] as usize;
            let n_out = info.dims[0] as usize;
            weights.push(bytes);
            shapes.push((n_in, n_out));
            inputs.push((0..rows * n_in).map(|_| next() / 8.0).collect());
        }
        if weights.is_empty() {
            eprintln!("no real projections; skip");
            return;
        }
        let run = |round: usize| -> Vec<Vec<f32>> {
            let mut out = Vec::new();
            for (index, bytes) in weights.iter().enumerate() {
                let (n_in, n_out) = shapes[index];
                let spec = GpuMatmulSpec {
                    format: GpuWeightFormat::BF16,
                    mode: crate::vulkan::ops::GpuMatmulMode::Prepared,
                };
                let linear = GpuLinear::default();
                let mut gpu = vec![0.0f32; rows * n_out];
                let ok =
                    linear.try_matmul(bytes, spec, &inputs[index], &mut gpu, n_in, n_out, rows);
                if !ok {
                    eprintln!("PROBE round {round} projection {index}: 退回 CPU");
                    break;
                }
                out.push(gpu);
            }
            out
        };
        let first = run(0);
        for round in 1..8 {
            let again = run(round);
            let mut bad = 0usize;
            for (index, (a, b)) in first.iter().zip(again.iter()).enumerate() {
                let differing = a
                    .iter()
                    .zip(b.iter())
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count();
                if differing > 0 {
                    eprintln!(
                        "PROBE round {round} projection {index}: {differing}/{} 个元素不同",
                        a.len()
                    );
                    bad += 1;
                }
            }
            if bad == 0 {
                eprintln!("PROBE round {round}: 全部投影逐位一致");
            }
        }
    }

    /// Mage-Flow feeds the denoiser a [1024, 128] latent patch grid into a
    /// [128, 3072] projection. That shape is nothing like the wide one the
    /// other checks use, and the first GPU run of a model is the one that has
    /// to land on the reference image, so repeat it and compare.
    #[test]
    #[ignore = "requires a Vulkan device and the real Mage-Flow GGUFs"]
    fn bf16_prepared_mageflow_projection_is_reproducible() {
        use crate::vulkan::ops::{GpuMatmulMode, GpuMatmulSpec, GpuWeightFormat};
        crate::ops::float::enable_gpu();
        if crate::ops::get_vulkan_context().is_none() {
            eprintln!("no vulkan context; skip");
            return;
        }
        let Ok(loader) =
            crate::GGUFLoader::from_file("models/mage-flow/mage-flow-base-dit-BF16.gguf")
        else {
            eprintln!("no real weights; skip");
            return;
        };
        let Some(bytes) = loader.tensor_slice("img_in.weight") else {
            eprintln!("no img_in.weight; skip");
            return;
        };
        let Some(info) = loader.tensor_info("img_in.weight") else {
            eprintln!("no img_in info; skip");
            return;
        };
        let n_in = info.dims[0] as usize;
        let n_out = info.dims[1] as usize;
        let rows = 1024usize;
        eprintln!("PROBE img_in: n_in={n_in} n_out={n_out} rows={rows}");
        let mut state: u64 = 0xB5026F5AA96619E9;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        };
        let input: Vec<f32> = (0..rows * n_in).map(|_| next()).collect();
        let spec = GpuMatmulSpec {
            format: GpuWeightFormat::BF16,
            mode: GpuMatmulMode::Prepared,
        };
        let mut first = vec![0.0f32; rows * n_out];
        let mut worst = 0.0f64;
        let rounds: usize = std::env::var("PROBE_ROUNDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8);
        for round in 0..rounds {
            let mut out = vec![0.0f32; rows * n_out];
            // Mage-Flow rebuilds the projection for every call, so each round
            // gets a fresh runtime and a freshly JIT-compiled pipeline.
            let linear = GpuLinear::default();
            let ok = linear.try_matmul(bytes, spec, &input, &mut out, n_in, n_out, rows);
            if !ok {
                eprintln!("PROBE round {round}: 退回 CPU");
                return;
            }
            if round == 0 {
                first = out;
                continue;
            }
            let differing = first
                .iter()
                .zip(out.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            let max_abs = first
                .iter()
                .zip(out.iter())
                .map(|(x, y)| (x - y).abs() as f64)
                .fold(0.0f64, f64::max);
            worst = worst.max(max_abs);
            if differing > 0 || round < 3 || round % 100 == 0 {
                eprintln!(
                    "PROBE round {round}: 与第 0 轮不同元素={differing}/{} 最大绝对差={max_abs:.6e}",
                    first.len()
                );
            }
        }
        eprintln!("PROBE img_in {rounds} 轮最大绝对差={worst:.6e}");
    }

    /// A denoiser step is not a single projection: the blocks in between run
    /// attention on the same device, and the projection that follows has to
    /// give the same answer no matter what those blocks did before it.
    #[test]
    #[ignore = "requires a Vulkan device and the real Mage-Flow GGUFs"]
    fn bf16_prepared_after_attention_is_reproducible() {
        use crate::models::diffusion::dreamx::kernels::AttentionSpec;
        use crate::vulkan::ops::{GpuMatmulMode, GpuMatmulSpec, GpuWeightFormat};
        crate::ops::float::enable_gpu();
        if crate::ops::get_vulkan_context().is_none() {
            eprintln!("no vulkan context; skip");
            return;
        }
        let Ok(loader) =
            crate::GGUFLoader::from_file("models/mage-flow/mage-flow-base-dit-BF16.gguf")
        else {
            eprintln!("no real weights; skip");
            return;
        };
        let Some(bytes) = loader.tensor_slice("img_in.weight") else {
            eprintln!("no img_in.weight; skip");
            return;
        };
        let Some(info) = loader.tensor_info("img_in.weight") else {
            eprintln!("no img_in info; skip");
            return;
        };
        let n_in = info.dims[0] as usize;
        let n_out = info.dims[1] as usize;
        let rows = 1024usize;
        let mut state: u64 = 0x2545F4914F6CDD1D;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        };
        let input: Vec<f32> = (0..rows * n_in).map(|_| next()).collect();
        // The denoiser attention shape: 1024 tokens, 24 heads, 128 dims.
        let tokens = 1024usize;
        let heads = 24usize;
        let dim = 128usize;
        let q: Vec<f32> = (0..tokens * heads * dim).map(|_| next() / 8.0).collect();
        let k: Vec<f32> = (0..tokens * heads * dim).map(|_| next() / 8.0).collect();
        let v: Vec<f32> = (0..tokens * heads * dim).map(|_| next() / 8.0).collect();
        let spec_attn = AttentionSpec {
            query_tokens: tokens,
            key_tokens: tokens,
            query_heads: heads,
            key_value_heads: heads,
            head_dim: dim,
            causal: false,
            scale: 1.0 / (dim as f32).sqrt(),
        };
        let spec = GpuMatmulSpec {
            format: GpuWeightFormat::BF16,
            mode: GpuMatmulMode::Prepared,
        };
        let mut first = vec![0.0f32; rows * n_out];
        for round in 0..8 {
            let attn = crate::ops::kernel::vulkan::try_attention(&q, &k, &v, spec_attn);
            if attn.is_none() {
                eprintln!("PROBE round {round}: attention 走 CPU");
                return;
            }
            let mut out = vec![0.0f32; rows * n_out];
            let ok =
                GpuLinear::default().try_matmul(bytes, spec, &input, &mut out, n_in, n_out, rows);
            if !ok {
                eprintln!("PROBE round {round}: 投影退回 CPU");
                return;
            }
            if round == 0 {
                first = out;
                continue;
            }
            let differing = first
                .iter()
                .zip(out.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            let max_abs = first
                .iter()
                .zip(out.iter())
                .map(|(x, y)| (x - y).abs() as f64)
                .fold(0.0f64, f64::max);
            eprintln!(
                "PROBE round {round}: 投影不同元素={differing}/{} 最大绝对差={max_abs:.6e}",
                first.len()
            );
        }
        eprintln!("PROBE attention 之后的投影 8 轮全部比较完毕");
    }

    /// A transformer block is an attention followed by a projection over the
    /// attention result. Either half on its own is reproducible, so the pair
    /// is the smallest unit that can show the two disagreeing.
    #[test]
    #[ignore = "requires a Vulkan device and the real Mage-Flow GGUFs"]
    fn bf16_prepared_after_attention_projection_is_reproducible() {
        use crate::models::diffusion::dreamx::kernels::AttentionSpec;
        use crate::vulkan::ops::{GpuMatmulMode, GpuMatmulSpec, GpuWeightFormat};
        crate::ops::float::enable_gpu();
        if crate::ops::get_vulkan_context().is_none() {
            eprintln!("no vulkan context; skip");
            return;
        }
        let Ok(loader) =
            crate::GGUFLoader::from_file("models/mage-flow/mage-flow-base-dit-BF16.gguf")
        else {
            eprintln!("no real weights; skip");
            return;
        };
        let mut state: u64 = 0x14057B7EF767814F;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        };
        let tokens = 1024usize;
        let heads = 24usize;
        let dim = 128usize;
        let q: Vec<f32> = (0..tokens * heads * dim).map(|_| next() / 8.0).collect();
        let k: Vec<f32> = (0..tokens * heads * dim).map(|_| next() / 8.0).collect();
        let v: Vec<f32> = (0..tokens * heads * dim).map(|_| next() / 8.0).collect();
        let spec_attn = AttentionSpec {
            query_tokens: tokens,
            key_tokens: tokens,
            query_heads: heads,
            key_value_heads: heads,
            head_dim: dim,
            causal: false,
            scale: 1.0 / (dim as f32).sqrt(),
        };
        let (Some(bytes), Some(info)) = (
            loader.tensor_slice("transformer_blocks.0.attn.to_add_out.weight"),
            loader.tensor_info("transformer_blocks.0.attn.to_add_out.weight"),
        ) else {
            eprintln!("no attn output projection; skip");
            return;
        };
        let n_in = info.dims[0] as usize;
        let n_out = info.dims[1] as usize;
        eprintln!("PROBE attn 投影: n_in={n_in} n_out={n_out}");
        let spec = GpuMatmulSpec {
            format: GpuWeightFormat::BF16,
            mode: GpuMatmulMode::Prepared,
        };
        let mut first: Option<Vec<f32>> = None;
        for round in 0..8 {
            let Some(context) = crate::ops::kernel::vulkan::try_attention(&q, &k, &v, spec_attn)
            else {
                eprintln!("PROBE round {round}: attention 走 CPU");
                return;
            };
            let rows = tokens;
            let mut out = vec![0.0f32; rows * n_out];
            let ok =
                GpuLinear::default().try_matmul(bytes, spec, &context, &mut out, n_in, n_out, rows);
            if !ok {
                eprintln!("PROBE round {round}: 投影退回 CPU");
                return;
            }
            match &first {
                None => first = Some(out),
                Some(reference) => {
                    let differing = reference
                        .iter()
                        .zip(out.iter())
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count();
                    let max_abs = reference
                        .iter()
                        .zip(out.iter())
                        .map(|(x, y)| (x - y).abs() as f64)
                        .fold(0.0f64, f64::max);
                    eprintln!(
                        "PROBE round {round}: attention+投影 不同元素={differing}/{} 最大绝对差={max_abs:.6e}",
                        reference.len()
                    );
                }
            }
        }
    }

    /// Every projection shape a denoiser block actually uses, each one driven
    /// with its own real weight. The narrow output projection and the wide
    /// feed-forward ones are the shapes that a single shape check misses.
    #[test]
    #[ignore = "requires a Vulkan device and the real Mage-Flow GGUFs"]
    fn bf16_prepared_every_block_shape_is_reproducible() {
        use crate::vulkan::ops::{GpuMatmulMode, GpuMatmulSpec, GpuWeightFormat};
        crate::ops::float::enable_gpu();
        if crate::ops::get_vulkan_context().is_none() {
            eprintln!("no vulkan context; skip");
            return;
        }
        let Ok(loader) =
            crate::GGUFLoader::from_file("models/mage-flow/mage-flow-base-dit-BF16.gguf")
        else {
            eprintln!("no real weights; skip");
            return;
        };
        // (tensor, rows) exactly as the denoiser calls them.
        let cases: [(&str, usize); 10] = [
            ("proj_out.weight", 1024),
            ("img_in.weight", 1024),
            ("transformer_blocks.0.attn.to_q.weight", 1024),
            ("transformer_blocks.0.attn.to_out.0.weight", 1024),
            ("transformer_blocks.0.img_mlp.net.0.proj.weight", 1024),
            ("transformer_blocks.0.img_mlp.net.2.weight", 1024),
            ("transformer_blocks.8.txt_mlp.net.0.proj.weight", 20),
            ("transformer_blocks.8.txt_mlp.net.2.weight", 20),
            ("transformer_blocks.8.attn.add_q_proj.weight", 20),
            ("norm_out.linear.weight", 1),
        ];
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        };
        let spec = GpuMatmulSpec {
            format: GpuWeightFormat::BF16,
            mode: GpuMatmulMode::Prepared,
        };
        for (name, rows) in cases {
            let (Some(bytes), Some(info)) = (loader.tensor_slice(name), loader.tensor_info(name))
            else {
                eprintln!("PROBE {name}: 权重缺失，跳过");
                continue;
            };
            let n_in = info.dims[0] as usize;
            let n_out = info.dims[1] as usize;
            let input: Vec<f32> = (0..rows * n_in).map(|_| next()).collect();
            let mut first: Option<Vec<f32>> = None;
            let mut differing = 0usize;
            let mut max_abs = 0.0f64;
            let mut fallback = false;
            for round in 0..8 {
                let mut out = vec![0.0f32; rows * n_out];
                let ok = GpuLinear::default()
                    .try_matmul(bytes, spec, &input, &mut out, n_in, n_out, rows);
                if !ok {
                    fallback = true;
                    break;
                }
                match &first {
                    None => first = Some(out),
                    Some(reference) => {
                        let d = reference
                            .iter()
                            .zip(out.iter())
                            .filter(|(x, y)| x.to_bits() != y.to_bits())
                            .count();
                        let m = reference
                            .iter()
                            .zip(out.iter())
                            .map(|(x, y)| (x - y).abs() as f64)
                            .fold(0.0f64, f64::max);
                        differing += d;
                        max_abs = max_abs.max(m);
                    }
                }
            }
            eprintln!(
                "PROBE {name:44} n_in={n_in:6} n_out={n_out:6} rows={rows:5}{} 不同元素累计={differing} 最大绝对差={max_abs:.6e}",
                if fallback { " [回退 CPU]" } else { "" }
            );
        }
    }

    /// Sweep the three dimensions of a projection against the CPU kernel with
    /// weights that are small enough to add up exactly, so any difference is a
    /// wrong answer and not a rounding difference.
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn bf16_prepared_dimension_sweep_matches_cpu() {
        use crate::vulkan::ops::{GpuMatmulMode, GpuMatmulSpec, GpuWeightFormat};
        crate::ops::float::enable_gpu();
        if crate::ops::get_vulkan_context().is_none() {
            eprintln!("no vulkan context; skip");
            return;
        }
        let spec = GpuMatmulSpec {
            format: GpuWeightFormat::BF16,
            mode: GpuMatmulMode::Prepared,
        };
        let cases: [(usize, usize, usize); 8] = [
            (128, 3072, 64),
            (128, 3072, 1024),
            (3072, 3072, 64),
            (3072, 128, 64),
            (128, 128, 64),
            (128, 3072, 32),
            (128, 3072, 33),
            (64, 3072, 64),
        ];
        for (n_in, n_out, rows) in cases {
            // Position dependent weights, so a wrong index cannot hide behind
            // a constant. The input is one, so the exact product is the sum of
            // the row, accumulated in f64.
            let mut weight = Vec::with_capacity(n_in * n_out * 2);
            let mut sums = vec![0.0f64; n_out];
            // Re-encode every weight as bf16 so the host and the device see the
            // same number, then accumulate the reference in f64.
            weight.clear();
            let mut exact = vec![0.0f64; n_in * n_out];
            for index in 0..n_in * n_out {
                let mixed = ((index as u64).wrapping_mul(0x9E3779B97F4A7C15) >> 40) as u32;
                let value = ((mixed % 65) as f32 - 32.0) / 64.0;
                let bf16 = crate::ops::f32_to_bf16(value);
                weight.extend_from_slice(&bf16.to_le_bytes());
                exact[index] = f64::from(crate::ops::bf16_to_f32(bf16));
            }
            // Row major [n_out][n_in], the layout the CPU kernel reads.
            for column in 0..n_out {
                for k in 0..n_in {
                    sums[column] += exact[column * n_in + k];
                }
            }
            let input = vec![1.0f32; rows * n_in];
            let mut gpu = vec![0.0f32; rows * n_out];
            let ok =
                GpuLinear::default().try_matmul(&weight, spec, &input, &mut gpu, n_in, n_out, rows);
            if !ok {
                eprintln!("PROBE n_in={n_in} n_out={n_out} rows={rows}: 回退 CPU");
                continue;
            }
            let mut wrong = 0usize;
            let mut worst = 0.0f64;
            let mut first_bad = None;
            for index in 0..rows * n_out {
                let column = index % n_out;
                let expected = sums[column];
                let got = f64::from(gpu[index]);
                let error = (got - expected).abs();
                let tolerance = expected.abs().max(1.0) * 1.0e-3;
                if error > tolerance {
                    wrong += 1;
                    if first_bad.is_none() {
                        first_bad = Some((index, got, expected));
                    }
                }
                worst = worst.max(error);
            }
            eprintln!(
                "PROBE n_in={n_in:5} n_out={n_out:5} rows={rows:5}: 错误元素={wrong}/{} 最大绝对误差={worst:.6e} 首个错误={first_bad:?}",
                gpu.len()
            );
        }
    }

    /// Three way arbitration for the narrow projection the denoiser starts
    /// with: the GPU kernel, the CPU kernel and an f64 reference all see the
    /// same weight and the same input, so the majority that is off shows up.
    #[test]
    #[ignore = "requires a Vulkan device and the real Mage-Flow GGUFs"]
    fn bf16_prepared_narrow_projection_three_way_arbitration() {
        use crate::vulkan::ops::{GpuMatmulMode, GpuMatmulSpec, GpuWeightFormat};
        crate::ops::float::enable_gpu();
        if crate::ops::get_vulkan_context().is_none() {
            eprintln!("no vulkan context; skip");
            return;
        }
        let Ok(loader) =
            crate::GGUFLoader::from_file("models/mage-flow/mage-flow-base-dit-BF16.gguf")
        else {
            eprintln!("no real weights; skip");
            return;
        };
        let (Some(bytes), Some(info)) = (
            loader.tensor_slice("img_in.weight"),
            loader.tensor_info("img_in.weight"),
        ) else {
            eprintln!("no img_in.weight; skip");
            return;
        };
        let n_in = info.dims[0] as usize;
        let n_out = info.dims[1] as usize;
        let rows = 64usize;
        let mut state: u64 = 0x243F6A8885A308D3;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        };
        let input: Vec<f32> = (0..rows * n_in).map(|_| next()).collect();
        let spec = GpuMatmulSpec {
            format: GpuWeightFormat::BF16,
            mode: GpuMatmulMode::Prepared,
        };
        let mut gpu = vec![0.0f32; rows * n_out];
        if !GpuLinear::default().try_matmul(bytes, spec, &input, &mut gpu, n_in, n_out, rows) {
            eprintln!("回退 CPU");
            return;
        }
        let mut cpu = vec![0.0f32; rows * n_out];
        for index in 0..rows * n_out {
            let row = index / n_out;
            let column = index % n_out;
            cpu[index] = crate::models::diffusion::dreamx::kernels::dot_bf16_f32(
                &bytes[column * n_in * 2..(column + 1) * n_in * 2],
                &input[row * n_in..(row + 1) * n_in],
            );
        }
        // f64 reference over the same bytes.
        let mut reference = vec![0.0f64; rows * n_out];
        for index in 0..rows * n_out {
            let row = index / n_out;
            let column = index % n_out;
            let mut sum = 0.0f64;
            for k in 0..n_in {
                let bits = u16::from_le_bytes(
                    bytes[column * n_in * 2 + k * 2..column * n_in * 2 + k * 2 + 2]
                        .try_into()
                        .unwrap(),
                );
                sum += f64::from(crate::ops::bf16_to_f32(bits)) * f64::from(input[row * n_in + k]);
            }
            reference[index] = sum;
        }
        let mut gpu_worst = 0.0f64;
        let mut cpu_worst = 0.0f64;
        let mut gpu_scale = 0.0f64;
        let mut cpu_scale = 0.0f64;
        for index in 0..rows * n_out {
            let want = reference[index];
            gpu_worst = gpu_worst.max((f64::from(gpu[index]) - want).abs());
            cpu_worst = cpu_worst.max((f64::from(cpu[index]) - want).abs());
            gpu_scale = gpu_scale.max(want.abs());
            cpu_scale = cpu_scale.max(want.abs());
        }
        eprintln!(
            "PROBE img_in n_in={n_in} n_out={n_out}: GPU 最大误差={gpu_worst:.6e} (相对 {:.3e})  CPU 最大误差={cpu_worst:.6e} (相对 {:.3e})",
            gpu_worst / gpu_scale,
            cpu_worst / cpu_scale
        );
    }

    /// The narrow projection that the denoiser uses for its text stream.
    ///
    /// It is the shape that fails once in a few thousand calls in the real
    /// model, so this drives it thousands of times and watches the result
    /// against the CPU kernel.
    #[test]
    #[ignore = "requires a Vulkan device and the real Mage-Flow GGUFs"]
    fn bf16_prepared_narrow_projection_survives_many_rounds() {
        use crate::vulkan::ops::{GpuMatmulMode, GpuMatmulSpec, GpuWeightFormat};
        crate::ops::float::enable_gpu();
        if crate::ops::get_vulkan_context().is_none() {
            eprintln!("no vulkan context; skip");
            return;
        }
        let Ok(loader) =
            crate::GGUFLoader::from_file("models/mage-flow/mage-flow-base-dit-BF16.gguf")
        else {
            eprintln!("no real weights; skip");
            return;
        };
        let (Some(bytes), Some(info)) = (
            loader.tensor_slice("transformer_blocks.8.attn.to_k.weight"),
            loader.tensor_info("transformer_blocks.8.attn.to_k.weight"),
        ) else {
            eprintln!("no weight; skip");
            return;
        };
        let n_in = info.dims[0] as usize;
        let n_out = info.dims[1] as usize;
        // The text stream uses twenty rows, the image stream hundreds. The
        // failure only shows up on the narrow one, so both are worth running.
        let rows: usize = std::env::var("PROBE_ROWS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(20);
        let rounds: usize = std::env::var("PROBE_ROUNDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4000);
        let mut state: u64 = 0x8E1D4C7B93A2F506;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        };
        let input: Vec<f32> = (0..rows * n_in).map(|_| next()).collect();
        // The CPU reference, computed once.
        let mut reference = vec![0.0f32; rows * n_out];
        for index in 0..rows * n_out {
            let row = index / n_out;
            let column = index % n_out;
            reference[index] = crate::models::diffusion::dreamx::kernels::dot_bf16_f32(
                &bytes[column * n_in * 2..(column + 1) * n_in * 2],
                &input[row * n_in..(row + 1) * n_in],
            );
        }
        let scale = reference
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()))
            .max(1.0);
        let spec = GpuMatmulSpec {
            format: GpuWeightFormat::BF16,
            mode: GpuMatmulMode::Prepared,
        };
        let mut failures = 0usize;
        // PROBE_REUSE=1 keeps one runtime for every round, the way the
        // denoiser reuses a projection; the default builds a new one each
        // round, the way a caller that rebuilds the projection would.
        let reuse = std::env::var("PROBE_REUSE").is_ok();
        let mut linear = GpuLinear::default();
        for round in 0..rounds {
            let mut out = vec![0.0f32; rows * n_out];
            let ok = linear.try_matmul(bytes, spec, &input, &mut out, n_in, n_out, rows);
            if !ok {
                eprintln!("PROBE round {round}: 回退 CPU");
                return;
            }
            let max_abs = out
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let relative = max_abs / scale;
            if relative > 1.0e-2 {
                failures += 1;
                let nonzero = out.iter().filter(|value| **value != 0.0).count();
                // Which rows are wrong, and are the wrong values zero, stale or
                // something else? A tail of unwritten rows points at the
                // driver dropping the end of a dispatch.
                let mut wrong_rows: Vec<usize> = Vec::new();
                let mut all_zero_rows: Vec<usize> = Vec::new();
                let mut stale_rows: Vec<usize> = Vec::new();
                for row in 0..rows {
                    let start = row * n_out;
                    let slice = &out[start..start + n_out];
                    let reference_slice = &reference[start..start + n_out];
                    if slice.iter().zip(reference_slice).all(|(a, b)| a == b) {
                        continue;
                    }
                    wrong_rows.push(row);
                    if slice.iter().all(|value| *value == 0.0) {
                        all_zero_rows.push(row);
                    } else if slice
                        .iter()
                        .zip(reference_slice)
                        .all(|(a, b)| a.abs() < 1.0e-3 && b.abs() > 1.0e-3)
                    {
                        stale_rows.push(row);
                    }
                }
                // Which part of the reduction went wrong? Compare the broken
                // values against references that only use the first or the
                // second half of the input.
                let half = n_in / 2;
                let mut first_half = vec![0.0f32; rows * n_out];
                let mut second_half = vec![0.0f32; rows * n_out];
                for index in 0..rows * n_out {
                    let row = index / n_out;
                    let column = index % n_out;
                    let weight = &bytes[column * n_in * 2..(column + 1) * n_in * 2];
                    let input = &input[row * n_in..(row + 1) * n_in];
                    first_half[index] = crate::models::diffusion::dreamx::kernels::dot_bf16_f32(
                        &weight[..half * 2],
                        &input[..half],
                    );
                    second_half[index] = crate::models::diffusion::dreamx::kernels::dot_bf16_f32(
                        &weight[half * 2..],
                        &input[half..],
                    );
                }
                let mut matches_first = 0usize;
                let mut matches_second = 0usize;
                let mut matches_neither = 0usize;
                for index in 0..rows * n_out {
                    if out[index] == first_half[index] {
                        matches_first += 1;
                    } else if out[index] == second_half[index] {
                        matches_second += 1;
                    } else {
                        matches_neither += 1;
                    }
                }
                eprintln!(
                    "PROBE round {round}: 相对差={relative:.4e} 错行={:?} 全零行={:?} | 错值等于前一半={matches_first} 后一半={matches_second} 都不是={matches_neither}",
                    wrong_rows,
                    all_zero_rows,
                );
            }
        }
        eprintln!("PROBE {rounds} 轮中失败 {failures} 次");
    }

    /// The device has been seen to return a projection that is wrong in
    /// whole rows (TODO-044). This probe changes the activation from round
    /// to round, so a result that belongs to an earlier round stands out.
    ///
    /// The activation is `base + round * step`, and the projection is
    /// linear, so the reference of every round follows from two CPU
    /// projections. A result that matches an earlier round's reference is a
    /// stale activation, and a result that matches none is a wrong one.
    #[test]
    #[ignore = "requires a Vulkan device and the real Mage-Flow GGUFs"]
    fn bf16_prepared_activation_round_is_visible() {
        use crate::vulkan::ops::{GpuMatmulMode, GpuMatmulSpec, GpuWeightFormat};
        crate::ops::float::enable_gpu();
        if crate::ops::get_vulkan_context().is_none() {
            eprintln!("no vulkan context; skip");
            return;
        }
        let Ok(loader) =
            crate::GGUFLoader::from_file("models/mage-flow/mage-flow-base-dit-BF16.gguf")
        else {
            eprintln!("no real weights; skip");
            return;
        };
        let (Some(bytes), Some(info)) = (
            loader.tensor_slice("transformer_blocks.8.attn.to_k.weight"),
            loader.tensor_info("transformer_blocks.8.attn.to_k.weight"),
        ) else {
            eprintln!("no weight; skip");
            return;
        };
        let n_in = info.dims[0] as usize;
        let n_out = info.dims[1] as usize;
        let rows: usize = std::env::var("PROBE_ROWS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(20);
        let rounds: usize = std::env::var("PROBE_ROUNDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4000);
        let mut state: u64 = 0x8E1D4C7B93A2F506;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        };
        let base: Vec<f32> = (0..rows * n_in).map(|_| next()).collect();
        let step: Vec<f32> = (0..rows * n_in).map(|_| next()).collect();
        // The projection is linear, so the reference of every round follows
        // from the references of the base and of the step.
        let project = |input: &[f32]| -> Vec<f32> {
            let mut reference = vec![0.0f32; rows * n_out];
            for index in 0..rows * n_out {
                let row = index / n_out;
                let column = index % n_out;
                reference[index] = crate::models::diffusion::dreamx::kernels::dot_bf16_f32(
                    &bytes[column * n_in * 2..(column + 1) * n_in * 2],
                    &input[row * n_in..(row + 1) * n_in],
                );
            }
            reference
        };
        let base_reference = project(&base);
        let step_reference = project(&step);
        let scale = base_reference
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()))
            .max(1.0);
        let spec = GpuMatmulSpec {
            format: GpuWeightFormat::BF16,
            mode: GpuMatmulMode::Prepared,
        };
        let mut failures = 0usize;
        let mut linear = GpuLinear::default();
        for round in 0..rounds {
            let input: Vec<f32> = base
                .iter()
                .zip(step.iter())
                .map(|(base, step)| base + round as f32 * step)
                .collect();
            let mut out = vec![0.0f32; rows * n_out];
            if !linear.try_matmul(bytes, spec, &input, &mut out, n_in, n_out, rows) {
                eprintln!("PROBE round {round}: fell back to the CPU");
                return;
            }
            // How far is the result from the reference of this round, and how
            // far is it from the reference of the rounds around it?
            let distance = |candidate: &[f32]| -> f32 {
                out.iter()
                    .zip(candidate.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
                    / scale
            };
            let own_reference: Vec<f32> = base_reference
                .iter()
                .zip(step_reference.iter())
                .map(|(base, step)| base + round as f32 * step)
                .collect();
            // The scale must follow the round: the reference grows with the
            // input, so a scale fixed at round 0 would turn a healthy 1e-6
            // into a 1e-2 at round 3000.
            let round_scale = own_reference
                .iter()
                .fold(0.0f32, |peak, value| peak.max(value.abs()))
                .max(1.0);
            let own = distance(&own_reference) * scale / round_scale;
            // The extrapolated reference drifts over thousands of rounds, so
            // the exact reference decides whether this is a failure. Its
            // scale follows the round too, because the reference grows with
            // the input.
            let exact = project(&input);
            let exact_scale = exact
                .iter()
                .fold(0.0f32, |peak, value| peak.max(value.abs()))
                .max(1.0);
            let exact_distance = out
                .iter()
                .zip(exact.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
                / exact_scale;
            if exact_distance <= 1.0e-4 {
                continue;
            }
            failures += 1;
            // Which earlier round does the result look like, if any?
            let mut closest: Option<(usize, f32)> = None;
            for earlier in 0..=round {
                let candidate: Vec<f32> = base_reference
                    .iter()
                    .zip(step_reference.iter())
                    .map(|(base, step)| base + earlier as f32 * step)
                    .collect();
                let distance = distance(&candidate);
                if closest.is_none_or(|(_, best)| distance < best) {
                    closest = Some((earlier, distance));
                }
            }
            // How much of each row is wrong, and where in the row?
            let mut wrong_per_row: Vec<(usize, usize, usize)> = Vec::new();
            for row in 0..rows {
                let mut wrong = 0usize;
                let mut first = usize::MAX;
                let mut last = 0usize;
                for column in 0..n_out {
                    if out[row * n_out + column] != own_reference[row * n_out + column] {
                        wrong += 1;
                        first = first.min(column);
                        last = last.max(column);
                    }
                }
                if wrong > 0 {
                    wrong_per_row.push((
                        row,
                        wrong,
                        if first == usize::MAX { 0 } else { last - first },
                    ));
                }
            }
            let caught = crate::models::diffusion::dreamx::kernels::projection_matches_cpu(
                &crate::core::thread_pool::ComputePool::new(1),
                bytes,
                &input,
                &out,
                rows,
                n_in,
                n_out,
            );
            // How localized is the damage? Count the rows that carry any wrong
            // value and, for those rows, how much of the row is wrong.
            let mut affected_rows = 0usize;
            let mut worst_row_fraction = 0.0f32;
            for row in 0..rows {
                let wrong = (0..n_out)
                    .filter(|column| {
                        out[row * n_out + column] != own_reference[row * n_out + column]
                    })
                    .count();
                if wrong > 0 {
                    affected_rows += 1;
                    worst_row_fraction = worst_row_fraction.max(wrong as f32 / n_out as f32);
                }
            }
            eprintln!(
                "PROBE round {round}: 外推相对差={own:.4e} 精确相对差={exact_distance:.4e} 检查抓到={caught} 受影响的行={affected_rows}/{rows} 最差行错列比例={worst_row_fraction:.3} 历史轮次={closest:?}"
            );
        }
        eprintln!("PROBE {rounds} 轮中失败 {failures} 次");
    }

    /// The transposed BF16/F32 kernel that #174 added has to stay within a
    /// float rounding of the CPU path, including when one call is split into
    /// several dispatches.
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn bf16_prepared_multichunk_matches_cpu() {
        use crate::vulkan::ops::{GpuMatmulSpec, GpuWeightFormat};
        crate::ops::float::enable_gpu();
        if crate::ops::get_vulkan_context().is_none() {
            eprintln!("no vulkan context; skip");
            return;
        }
        for (n_in, n_out, rows) in [
            (3072usize, 3072usize, 1024usize),
            (3072, 12288, 1024),
            (12288, 3072, 1024),
            (3072, 3072, 682),
            (3072, 3072, 704),
        ] {
            let mut weight = vec![0u8; n_out * n_in * 2];
            // Well-conditioned xorshift values: no periodic structure, so the
            // 3072-term sum does not cancel to near zero and a wrong kernel
            // cannot hide inside a large relative error.
            let mut state: u64 = 0x243F6A8885A308D3;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / 1024.0 - 8.0
            };
            for i in 0..(n_out * n_in) {
                let v = next() / 8.0;
                weight[i * 2..i * 2 + 2]
                    .copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
            let input: Vec<f32> = (0..rows * n_in).map(|_| next() / 8.0).collect();
            let spec = GpuMatmulSpec {
                format: GpuWeightFormat::BF16,
                mode: crate::vulkan::ops::GpuMatmulMode::Prepared,
            };
            let tile = GpuLinear::tile_rows(spec, n_in, n_out, rows);
            let linear = GpuLinear::default();
            let mut gpu = vec![0.0f32; rows * n_out];
            let ok = linear.try_matmul(&weight, spec, &input, &mut gpu, n_in, n_out, rows);
            if !ok {
                eprintln!("PROBE {n_in}x{n_out} rows={rows} tile={tile}: fell back (no GPU)");
                continue;
            }
            let mut first = None;
            let mut worst_rel = 0.0f32;
            for row in 0..rows {
                for col in 0..n_out {
                    let off = col * n_in * 2;
                    let cpu = crate::ops::dot_bf16_f32(
                        &input[row * n_in..(row + 1) * n_in],
                        &weight[off..off + n_in * 2],
                        n_in,
                    );
                    let got = gpu[row * n_out + col];
                    let rel = (cpu - got).abs() / cpu.abs().max(1e-6);
                    if rel > worst_rel {
                        worst_rel = rel;
                    }
                    if first.is_none() && (cpu - got).abs() > 1e-4 * cpu.abs().max(1e-2) {
                        first = Some((row, col, cpu, got));
                    }
                }
            }
            match first {
                None => eprintln!(
                    "PROBE {n_in}x{n_out} rows={rows} tile={tile}: OK 最大相对误差={worst_rel:.2e}"
                ),
                Some((row, col, cpu, got)) => eprintln!(
                    "PROBE {n_in}x{n_out} rows={rows} tile={tile}: 首个显著不同 row={row} col={col} cpu={cpu:.6e} gpu={got:.6e}"
                ),
            }
        }
    }

    /// Real trained BF16 weights span a wide dynamic range, so the F64
    /// reference decides which of the two summation orders is actually right.
    #[test]
    #[ignore = "requires a Vulkan device and the real Mage-Flow GGUFs"]
    fn bf16_prepared_real_weight_matches_f64() {
        use crate::vulkan::ops::{GpuMatmulSpec, GpuWeightFormat};
        crate::ops::float::enable_gpu();
        if crate::ops::get_vulkan_context().is_none() {
            eprintln!("no vulkan context; skip");
            return;
        }
        let Ok(loader) =
            crate::GGUFLoader::from_file("models/mage-flow/mage-flow-base-dit-BF16.gguf")
        else {
            eprintln!("no real weights; skip");
            return;
        };
        // One real 3072x3072 BF16 projection, driven exactly like the model does
        // it: a fresh Linear per call, 1024 rows, which is two dispatches.
        let Some(bytes) = loader.tensor_slice("transformer_blocks.0.attn.add_q_proj.weight") else {
            eprintln!("tensor missing; skip");
            return;
        };
        let (n_in, n_out) = (3072usize, 3072usize);
        assert_eq!(bytes.len(), n_in * n_out * 2);
        let rows = 1024usize;
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        };
        let input: Vec<f32> = (0..rows * n_in).map(|_| next() / 8.0).collect();
        let spec = GpuMatmulSpec {
            format: GpuWeightFormat::BF16,
            mode: crate::vulkan::ops::GpuMatmulMode::Prepared,
        };
        let tile = GpuLinear::tile_rows(spec, n_in, n_out, rows);
        let linear = GpuLinear::default();
        let mut gpu = vec![0.0f32; rows * n_out];
        let ok = linear.try_matmul(bytes, spec, &input, &mut gpu, n_in, n_out, rows);
        eprintln!("PROBE real 3072x3072 rows={rows} tile={tile} dispatched={ok}");
        if !ok {
            return;
        }
        let mut first = None;
        let mut worst = 0.0f32;
        let mut cpu_err = 0.0f64;
        let mut gpu_err = 0.0f64;
        for row in 0..rows {
            for col in 0..n_out {
                let off = col * n_in * 2;
                let cpu = crate::ops::dot_bf16_f32(
                    &input[row * n_in..(row + 1) * n_in],
                    &bytes[off..off + n_in * 2],
                    n_in,
                );
                let got = gpu[row * n_out + col];
                let rel = (cpu - got).abs() / cpu.abs().max(1e-3);
                if rel > worst {
                    worst = rel;
                }
                // F64 arbitration: which of CPU/GPU is actually right?
                let mut truth = 0.0f64;
                for k in 0..n_in {
                    let w = crate::ops::bf16_to_f32(u16::from_le_bytes([
                        bytes[off + k * 2],
                        bytes[off + k * 2 + 1],
                    ]));
                    truth += w as f64 * input[row * n_in + k] as f64;
                }
                let scale = truth.abs().max(1e-6);
                let err_cpu = ((cpu as f64 - truth) / scale).abs();
                let err_gpu = ((got as f64 - truth) / scale).abs();
                cpu_err = cpu_err.max(err_cpu);
                gpu_err = gpu_err.max(err_gpu);
                if first.is_none() && (cpu - got).abs() > 1e-4 * cpu.abs().max(1e-2) {
                    first = Some((row, col, cpu, got, err_cpu, err_gpu));
                }
            }
        }
        eprintln!("PROBE real weights: 相对F64真值最大误差  CPU={cpu_err:.3e}  GPU={gpu_err:.3e}",);
        match first {
            None => eprintln!("PROBE real weights: OK 最大相对误差={worst:.2e}"),
            Some((row, col, cpu, got, ec, eg)) => eprintln!(
                "PROBE real weights: 首个不同 row={row} col={col} cpu={cpu:.6e} gpu={got:.6e} | 该点相对F64误差 CPU={ec:.2e} GPU={eg:.2e}"
            ),
        }
    }

    /// Mage-Flow builds a fresh Linear per projection, so the runtime, its
    /// pipelines and the weight upload are rebuilt constantly. The result must
    /// not drift with that churn.
    #[test]
    #[ignore = "requires a Vulkan device and the real Mage-Flow GGUFs"]
    fn bf16_prepared_recreate_churn_is_stable() {
        use crate::vulkan::ops::{GpuMatmulSpec, GpuWeightFormat};
        crate::ops::float::enable_gpu();
        if crate::ops::get_vulkan_context().is_none() {
            eprintln!("no vulkan context; skip");
            return;
        }
        let Ok(loader) =
            crate::GGUFLoader::from_file("models/mage-flow/mage-flow-base-dit-BF16.gguf")
        else {
            eprintln!("no real weights; skip");
            return;
        };
        let Some(bytes) = loader.tensor_slice("transformer_blocks.0.attn.add_q_proj.weight") else {
            eprintln!("tensor missing; skip");
            return;
        };
        let (n_in, n_out) = (3072usize, 3072usize);
        let rows = 1024usize;
        let mut state: u64 = 0x243F6A8885A308D3;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        };
        let input: Vec<f32> = (0..rows * n_in).map(|_| next() / 8.0).collect();
        let spec = GpuMatmulSpec {
            format: GpuWeightFormat::BF16,
            mode: crate::vulkan::ops::GpuMatmulMode::Prepared,
        };
        let mut truth = vec![0.0f32; rows * n_out];
        for row in 0..rows {
            for col in 0..n_out {
                let off = col * n_in * 2;
                truth[row * n_out + col] = crate::ops::dot_bf16_f32(
                    &input[row * n_in..(row + 1) * n_in],
                    &bytes[off..off + n_in * 2],
                    n_in,
                );
            }
        }
        // Exactly how Mage-Flow drives it: a brand-new Linear per call, so a
        // brand-new runtime, pipelines and weight upload every time.
        for round in 0..64usize {
            let linear = GpuLinear::default();
            let mut gpu = vec![0.0f32; rows * n_out];
            let ok = linear.try_matmul(bytes, spec, &input, &mut gpu, n_in, n_out, rows);
            if !ok {
                eprintln!("PROBE churn round {round}: 退回 CPU");
                return;
            }
            let mut bad = 0usize;
            let mut worst = 0.0f32;
            for i in 0..rows * n_out {
                let d = (gpu[i] - truth[i]).abs() / truth[i].abs().max(1e-2);
                if d > 1e-3 {
                    bad += 1;
                }
                if d > worst {
                    worst = d;
                }
            }
            if bad > 0 || round == 0 {
                eprintln!(
                    "PROBE churn round {round}: 超阈值元素={bad}/{} 最差相对={worst:.2e}",
                    rows * n_out
                );
            }
            if bad > rows * n_out / 100 {
                eprintln!("PROBE churn 在第 {round} 轮崩坏");
                return;
            }
        }
        eprintln!("PROBE churn 64 轮全部通过");
    }

    #[test]
    #[ignore = "requires hardware Vulkan and RMI_PARITY_TRACE; run alone"]
    fn tracing_preserves_requested_gpu_attention() {
        assert!(std::env::var_os("RMI_PARITY_TRACE").is_some());
        crate::ops::enable_gpu();
        let _scope = crate::compute::ComputePolicy::Auto.enter_legacy_scope();
        let context = crate::compute::ComputePolicy::Vulkan
            .context()
            .unwrap()
            .unwrap();
        let before = context.submission_count();
        let result = try_attention(
            &[0.0; 8],
            &[0.0; 8],
            &[1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0],
            crate::models::diffusion::dreamx::kernels::AttentionSpec {
                query_tokens: 2,
                key_tokens: 2,
                query_heads: 1,
                key_value_heads: 1,
                head_dim: 4,
                causal: false,
                scale: 0.5,
            },
        )
        .expect("tracing must not force requested GPU attention onto CPU");
        for (value, expected) in result.iter().zip([2.0, 3.0, 4.0, 5.0].repeat(2)) {
            assert!((value - expected).abs() < 1e-5);
        }
        assert!(context.submission_count() > before);
        let _cpu = crate::core::thread_pool::ComputePool::disable_gpu_matmul_for_scope();
        assert!(!offload_enabled());
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
    scalar_reference: bool,
}

impl<'a> VulkanKernel<'a> {
    pub(super) fn wrap(inner: Box<dyn Kernel + 'a>, kind: GGMLType) -> Box<dyn Kernel + 'a> {
        match GpuWeightFormat::from_ggml_type(kind) {
            Ok(format) => Box::new(Self {
                inner,
                format,
                gpu: GpuLinear::default(),
                scalar_reference: false,
            }),
            Err(_) => inner,
        }
    }
}

impl Kernel for VulkanKernel<'_> {
    fn use_scalar_reference(&mut self) {
        self.inner.use_scalar_reference();
        self.scalar_reference = true;
    }

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
        if self.scalar_reference {
            return false;
        }
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
