//! GPU-accelerated Q4_K matmul, dispatched through `BatchedLinearRuntime`.
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
//!   is fine because every model shares one Vulkan device.
//! - `matmul_q4_k` is safe to call from any thread, but only one matmul can be
//!   in flight at a time (the inner mutex serialises dispatches). This
//!   matches `matmul_q8_0`'s model: pool thread 0 submits the fenced
//!   dispatch; the rest of the pool spins elsewhere.
//!
//! ## Scope
//!
//! - Q4_K only. Q5_K / Q6_K / Q4_0 / Q4_1 follow the same pattern but each
//!   needs its own shader + push-constant layout and a separate runtime; the
//!   payoff for Q4_K alone is the dominant K-quant format in production
//!   GGUF files, so we keep this initial implementation narrow.
//! - Software ICDs (`llvmpipe`, `swiftshader`) are skipped at the caller via
//!   `crate::ops::gpu_matmul_active()`, which now checks
//!   `VulkanContext::is_software_icd()`. This module only runs on real
//!   hardware. The dispatch cost on a software ICD would dominate any GPU
//!   win over the AVX2/FMA kernel.

use std::sync::{Mutex, OnceLock};

use super::ops::{BatchedLinearRuntime, GpuWeightFormat};
use super::{VulkanContext, VulkanError};

/// Per-device Q4_K GPU matmul runtime.
///
/// `None` until the first successful dispatch; recreated only when the cached
/// shape maxima cannot accommodate a later call.
static Q4_K_RUNTIME: OnceLock<Mutex<Option<Q4KRuntime>>> = OnceLock::new();

struct Q4KRuntime {
    runtime: BatchedLinearRuntime,
    max_rows: usize,
    max_n_in: usize,
    max_n_out: usize,
}

/// Lower bound on the row-batch size we pre-allocate for. Llama prefill
/// batches up to `--prefill-batch-size` (default 64); we use 4 as the floor
/// because single-token decode has rows=1 and the runtime handles it fine.
const DEFAULT_MAX_ROWS: usize = 4;

fn runtime_slot() -> &'static Mutex<Option<Q4KRuntime>> {
    Q4_K_RUNTIME.get_or_init(|| Mutex::new(None))
}

fn fits_in(runtime: &Q4KRuntime, rows: usize, n_in: usize, n_out: usize) -> bool {
    rows <= runtime.max_rows && n_in <= runtime.max_n_in && n_out <= runtime.max_n_out
}

/// Run one Q4_K matmul on the GPU, returning `Ok(())` on success or an
/// `UnsupportedShape` error when the cached runtime cannot accommodate this
/// shape (caller falls back to CPU). Other Vulkan errors mark the device
/// broken via `VulkanContext::mark_gpu_broken` and return `Err`.
///
/// `weight` must be the GGUF Q4_K byte stream (rows of `n_in / 256 * 144`
/// bytes). `input` is `rows * n_in` F32 elements. `output` must hold at
/// least `rows * n_out` F32 elements; only the first `rows * n_out` are
/// written.
pub fn matmul_q4_k(
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
        n_out * (n_in / 256) * 144,
        "Q4_K weight layout is rows × (n_in / 256) blocks × 144 bytes/block"
    );
    debug_assert!(input.len() >= rows * n_in, "input shorter than rows × n_in");
    debug_assert!(
        output.len() >= rows * n_out,
        "output shorter than rows × n_out"
    );

    let slot = runtime_slot();
    let mut guard = slot
        .lock()
        .map_err(|_| VulkanError::InitFailed("Q4_K runtime mutex poisoned".into()))?;

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
        // `BatchedLinearRuntime::new` takes a `'static` reference. The
        // singleton VulkanContext lives in a `OnceLock` for the lifetime of
        // the process, so this transmute is sound — the runtime will drop on
        // first error or normal program shutdown, well within the
        // VulkanContext's lifetime.
        let static_ctx: &'static VulkanContext = unsafe { std::mem::transmute(context) };
        match BatchedLinearRuntime::new(
            static_ctx,
            new_max_rows,
            new_max_n_in,
            new_max_n_out,
            // One descriptor set for the arena plus room for every Q4_K
            // weight tensor in the model — gemma-2-2b-it has 26 layers ×
            // 7 weights (Q/K/V/O/gate/up/down) = 182 unique uploads, and
            // each entry holds one GPU-resident descriptor set until the
            // runtime is dropped. 256 leaves headroom for one or two
            // atypical layers without re-allocation churn.
            256,
        ) {
            Ok(runtime) => {
                *guard = Some(Q4KRuntime {
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
        GpuWeightFormat::Q4_K,
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

    /// Smoke test: a real Q4_K GPU dispatch returns either Ok with finite
    /// outputs or `UnsupportedShape` (when the runtime cannot accommodate the
    /// shape — common on integrated GPUs with tight shared-memory limits).
    /// Skipped on software ICDs because the dispatch would mark the device
    /// broken and we don't want the test suite to assume a real GPU.
    #[test]
    fn q4_k_gpu_matmul_smoke() {
        let context = match VulkanContext::new() {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        if context.is_software_icd() {
            eprintln!("[skip] software ICD detected");
            return;
        }
        // All-zero Q4_K weight: d=0, min=0, scales=0, qs=0 → dot product is 0
        // for any input. Verifies the shader runs and writes sane output
        // without depending on a CPU-side quantization path.
        let n_in = 256;
        let n_out = 16;
        let n_blocks = n_in / 256;
        let weight = vec![0u8; n_out * n_blocks * 144];
        let input = vec![0.5f32; n_in];
        let mut gpu_output = vec![std::f32::NAN; n_out];

        match matmul_q4_k(&context, &weight, &input, 1, n_in, n_out, &mut gpu_output) {
            Ok(()) => {
                for (i, &value) in gpu_output.iter().enumerate() {
                    assert!(value.is_finite(), "lane {i}: output {value}");
                }
            }
            Err(crate::vulkan::VulkanError::UnsupportedShape(_)) => {
                eprintln!("[skip] Q4_K shape exceeds GPU runtime capacity");
            }
            Err(error) => panic!("Q4_K GPU matmul failed: {error}"),
        }
    }
}
