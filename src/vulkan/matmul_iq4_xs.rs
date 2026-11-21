//! GPU-accelerated IQ4_XS matmul, dispatched through `BatchedLinearRuntime`.
//!
//! The K-quant matmul shaders live in `src/vulkan/ops.rs` and were originally
//! only used by diffusion-model trunks (qwen3, qwen35, z-image). This module
//! adapts them to per-layer LLM matmul (one weight tensor per call, varying
//! `n_in` / `n_out`).
//!
//! ## Lifecycle
//!
//! - The runtime is held in a `OnceLock<Mutex<Option<...>>>` so the first GPU
//!   dispatch lazily creates it from the singleton `VulkanContext`. A static
//!   only borrows that singleton. Private contexts use a call-scoped runtime.
//! - `matmul_iq4_xs` is safe to call from any thread, but only one matmul can be
//!   in flight at a time (the inner mutex serialises dispatches). This
//!   matches `matmul_q8_0`'s model: pool thread 0 submits the fenced
//!   dispatch; the rest of the pool spins elsewhere.
//!
//! ## Scope
//!
//! - IQ4_XS only. Q5_K / Q6_K / Q4_0 / Q4_1 follow the same pattern but each
//!   needs its own shader + push-constant layout and a separate runtime; the
//!   payoff for IQ4_XS alone is the dominant K-quant format in production
//!   GGUF files, so we keep this initial implementation narrow.
//! - Software ICDs (`llvmpipe`, `swiftshader`) are skipped at the caller via
//!   `crate::ops::gpu_matmul_active()`, which now checks
//!   `VulkanContext::is_software_icd()`. This module only runs on real
//!   hardware. The dispatch cost on a software ICD would dominate any GPU
//!   win over the AVX2/FMA kernel.

use std::sync::{Mutex, OnceLock};

use super::ops::{BatchedLinearRuntime, GpuWeightFormat};
use super::{VulkanContext, VulkanError};

/// Per-device IQ4_XS GPU matmul runtime.
///
/// `None` until the first successful dispatch; recreated only when the cached
/// shape maxima cannot accommodate a later call.
static IQ4_XS_RUNTIME: OnceLock<Mutex<Option<IQ4XSRuntime>>> = OnceLock::new();

pub(crate) fn release_shared_runtime() -> Result<(), VulkanError> {
    if let Some(slot) = IQ4_XS_RUNTIME.get() {
        slot.lock()
            .map_err(|_| VulkanError::InitFailed("IQ4_XS runtime mutex poisoned".into()))?
            .take();
    }
    Ok(())
}

struct IQ4XSRuntime {
    runtime: BatchedLinearRuntime,
    max_rows: usize,
    max_n_in: usize,
    max_n_out: usize,
}

/// Lower bound on the row-batch size we pre-allocate for. Llama prefill
/// batches up to `--prefill-batch-size` (default 64); we use 4 as the floor
/// because single-token decode has rows=1 and the runtime handles it fine.
const DEFAULT_MAX_ROWS: usize = 4;

fn runtime_slot() -> &'static Mutex<Option<IQ4XSRuntime>> {
    IQ4_XS_RUNTIME.get_or_init(|| Mutex::new(None))
}

fn fits_in(runtime: &IQ4XSRuntime, rows: usize, n_in: usize, n_out: usize) -> bool {
    rows <= runtime.max_rows && n_in <= runtime.max_n_in && n_out <= runtime.max_n_out
}

/// Run one IQ4_XS matmul on the GPU, returning `Ok(())` on success or an
/// `UnsupportedShape` error when the cached runtime cannot accommodate this
/// shape (caller falls back to CPU). Other Vulkan errors mark the device
/// broken via `VulkanContext::mark_gpu_broken` and return `Err`.
///
/// `weight` must be the GGUF IQ4_XS byte stream (rows of `n_in / 256 * 136`
/// bytes). `input` is `rows * n_in` F32 elements. `output` must hold at
/// least `rows * n_out` F32 elements; only the first `rows * n_out` are
/// written.
pub fn matmul_iq4_xs(
    context: &VulkanContext,
    weight: &[u8],
    input: &[f32],
    rows: usize,
    n_in: usize,
    n_out: usize,
    output: &mut [f32],
) -> Result<(), VulkanError> {
    debug_assert_eq!(
        weight.len(),
        n_out * (n_in / 256) * 136,
        "IQ4_XS weight layout is rows × (n_in / 256) blocks × 136 bytes/block"
    );
    debug_assert!(input.len() >= rows * n_in, "input shorter than rows × n_in");
    debug_assert!(
        output.len() >= rows * n_out,
        "output shorter than rows × n_out"
    );

    let local_slot = Mutex::new(None);
    let slot = if crate::ops::float::is_shared_vulkan_context(context) {
        runtime_slot()
    } else {
        &local_slot
    };
    let mut guard = slot
        .lock()
        .map_err(|_| VulkanError::InitFailed("IQ4_XS runtime mutex poisoned".into()))?;

    let need_rebuild = match guard.as_ref() {
        Some(rt) => !fits_in(rt, rows, n_in, n_out),
        None => true,
    };
    if need_rebuild {
        // Take the max of (current_size, new_size) so the runtime is grown
        // monotonically across calls. Per-layer dispatch visits each
        // (n_in, n_out) once on the first token; subsequent tokens reuse the
        // cached runtime instead of paying the pipeline-creation cost on
        // every matmul. Without this gemma-2-2b-it paid 11 BatchedLinearRuntime
        // initialisations per prefill (one per unique shape).
        let (new_max_rows, new_max_n_in, new_max_n_out) = match guard.as_ref() {
            Some(rt) => (
                rt.max_rows.max(rows),
                rt.max_n_in.max(n_in),
                rt.max_n_out.max(n_out),
            ),
            None => (DEFAULT_MAX_ROWS.max(rows), n_in, n_out),
        };
        // The shared slot borrows the singleton; local_slot drops before the
        // caller's context borrow ends, so neither runtime can outlive its context.
        let static_ctx: &'static VulkanContext = unsafe { std::mem::transmute(context) };
        match BatchedLinearRuntime::new(
            static_ctx,
            new_max_rows,
            new_max_n_in,
            new_max_n_out,
            // One descriptor set for the arena plus room for every IQ4_XS
            // weight tensor in the model — gemma-2-2b-it has 26 layers ×
            // 7 weights (Q/K/V/O/gate/up/down) = 182 unique uploads, and
            // each entry holds one GPU-resident descriptor set until the
            // runtime is dropped. 256 leaves headroom for one or two
            // atypical layers without re-allocation churn.
            256,
        ) {
            Ok(runtime) => {
                *guard = Some(IQ4XSRuntime {
                    runtime,
                    max_rows: new_max_rows,
                    max_n_in: new_max_n_in,
                    max_n_out: new_max_n_out,
                });
            }
            Err(error) => {
                super::mark_gpu_broken(&error.to_string());
                return Err(error);
            }
        }
    }

    let runtime = guard.as_mut().expect("runtime was just installed");
    match runtime.runtime.matmul_rows(
        weight,
        GpuWeightFormat::IQ4_XS,
        input,
        rows,
        n_in,
        n_out,
        output,
    ) {
        Ok(()) => Ok(()),
        Err(error) => {
            // UnsupportedShape is a per-call failure: the GPU stays alive for
            // other shapes, but this matmul must fall back to CPU. Anything
            // else marks the device broken.
            if !matches!(error, VulkanError::UnsupportedShape(_)) {
                super::mark_gpu_broken(&error.to_string());
            }
            Err(error)
        }
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;
    use crate::vulkan::VulkanContext;

    /// Smoke test: a real IQ4_XS GPU dispatch returns either Ok with finite
    /// outputs or `UnsupportedShape` (when the runtime cannot accommodate the
    /// shape — common on integrated GPUs with tight shared-memory limits).
    /// Skipped on software ICDs because the dispatch would mark the device
    /// broken and we don't want the test suite to assume a real GPU.
    #[test]
    fn iq4_xs_gpu_matmul_smoke() {
        let context = match VulkanContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        if context.is_software_icd() {
            eprintln!("[skip] software ICD detected");
            return;
        }
        // All-zero IQ4_XS weight: d=0, min=0, scales=0, qs=0 → dot product is 0
        // for any input. Verifies the shader runs and writes sane output
        // without depending on a CPU-side quantization path.
        let n_in = 256;
        let n_out = 16;
        let n_blocks = n_in / 256;
        let weight = vec![0u8; n_out * n_blocks * 136];
        let input = vec![0.5f32; n_in];
        let mut gpu_output = vec![std::f32::NAN; n_out];
        let live_before = context.compute_stats().live_allocation_bytes;

        match matmul_iq4_xs(&context, &weight, &input, 1, n_in, n_out, &mut gpu_output) {
            Ok(()) => {
                for (i, &value) in gpu_output.iter().enumerate() {
                    assert!(value.is_finite(), "lane {i}: output {value}");
                }
            }
            Err(crate::vulkan::VulkanError::UnsupportedShape(_)) => {
                eprintln!("[skip] IQ4_XS shape exceeds GPU runtime capacity");
            }
            Err(error) => panic!("IQ4_XS GPU matmul failed: {error}"),
        }
        assert_eq!(
            context.compute_stats().live_allocation_bytes,
            live_before,
            "IQ4_XS runtime must not outlive a private context"
        );
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod gpu_cpu_parity {
    use super::*;
    use crate::core::loader::GGUFLoader;

    /// GPU IQ4_XS matmul must be bit-identical to the CPU AVX2 kernel
    /// (`vec_dot_iq4_xs_q8k`, itself bitwise-verified against llama.cpp) on
    /// real model weights: same per-sub-block exact integer dot, same f32
    /// accumulation order. Skipped on software ICDs.
    #[test]
    fn iq4_xs_gpu_matches_cpu_avx2_bitwise_on_real_weights() {
        let context = match VulkanContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        if context.is_software_icd() {
            eprintln!("[skip] software ICD detected");
            return;
        }
        let path = "models/Qwen3-0.6B-GGUF/Qwen3-0.6B-IQ4_XS.gguf";
        let loader = match GGUFLoader::from_file(path) {
            Ok(l) => l,
            Err(_) => {
                eprintln!("[skip] {path} not present");
                return;
            }
        };
        let tensor = loader
            .tensors()
            .iter()
            .find(|t| {
                t.name == "blk.0.attn_q.weight"
                    && t.ggml_type == crate::core::tensor::GGMLType::IQ4_XS
            })
            .expect("blk.0.attn_q.weight IQ4_XS not found");
        let weight = loader.tensor_slice(&tensor.name).unwrap();
        let n_in = tensor.dims[0] as usize;
        let n_out = tensor.dims[1] as usize;

        let input: Vec<f32> = (0..n_in)
            .map(|i| (((i * 31 + 7) % 97) as f32 - 48.0) / 32.0)
            .collect();

        // CPU reference: the AVX2 kernel over the whole tensor.
        let q8k = crate::ops::quant::quantize_row_q8_k(&input);
        let row_bytes = n_in / crate::ops::quant::QK_K * crate::ops::quant::BLOCK_IQ4_XS_SIZE;
        let mut cpu_output = vec![0.0f32; n_out];
        for (row, out) in cpu_output.iter_mut().enumerate() {
            *out = crate::ops::quant::vec_dot_iq4_xs_q8k(
                &weight[row * row_bytes..(row + 1) * row_bytes],
                &q8k,
            );
        }

        // GPU: the full tensor in one dispatch.
        let mut gpu_output = vec![0.0f32; n_out];
        match matmul_iq4_xs(&context, weight, &input, 1, n_in, n_out, &mut gpu_output) {
            Ok(()) => {}
            Err(crate::vulkan::VulkanError::UnsupportedShape(e)) => {
                eprintln!("[skip] UnsupportedShape: {e}");
                return;
            }
            Err(error) => panic!("IQ4_XS GPU matmul failed: {error}"),
        }

        let mut mismatches = 0usize;
        let mut worst_bits = 0u32;
        for (i, (g, c)) in gpu_output.iter().zip(cpu_output.iter()).enumerate() {
            if g.to_bits() != c.to_bits() {
                mismatches += 1;
                worst_bits = worst_bits.max(
                    (g.to_bits() as i32)
                        .wrapping_sub(c.to_bits() as i32)
                        .unsigned_abs(),
                );
                if mismatches <= 4 {
                    eprintln!(
                        "row {i}: gpu={g} ({:08x}) cpu={c} ({:08x})",
                        g.to_bits(),
                        c.to_bits()
                    );
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "IQ4_XS GPU vs CPU: {mismatches}/{n_out} rows differ (worst ULP {worst_bits})"
        );
    }
}
