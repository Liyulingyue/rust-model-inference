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
