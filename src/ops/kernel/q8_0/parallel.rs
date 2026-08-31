//! Q8_0 top-level parallel entry points.
//!
//! Phase 2.7-final: split from `ops::matmul`. `matmul_q8_0_quantized_parallel_rows`
//! is the hot-path entry called by `Q8Kernel::forward_prequantized`,
//! `bin/server.rs`, and `app/text.rs`. The other variants provide
//! different parallelization strategies (dynamic chunking, recursive split,
//! rayon chunks).

use super::dispatch::matmul_q8_0_quantized_range;

/// Row-partitioned parallel entry: the production hot path.
///
/// `ith`/`nth` selects which row range this thread handles. With
/// `nth <= 1` it degenerates to the single-thread `dispatch` call.
pub fn matmul_q8_0_quantized_parallel_rows(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    // GPU path: one dispatch covers ALL rows, so thread 0 submits and every
    // other pool thread returns immediately. On a GPU error the context is
    // marked broken and this thread falls back to CPU for the whole matmul.
    #[cfg(feature = "vulkan")]
    {
        use crate::ops::get_vulkan_context;

        // Shader capacity: the input row is staged in 4096 shared words.
        const MAX_GPU_N_IN: usize = 512 * 32;
        let max_rows = std::env::var("RUST_GPU_MAX_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        let gpu_takes_this =
            !crate::vulkan::gpu_broken() && n_in <= MAX_GPU_N_IN && n_out <= max_rows;
        if gpu_takes_this {
            if let Some(ctx) = get_vulkan_context() {
                if ith == 0 {
                    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    let trace = *TRACE.get_or_init(|| std::env::var("RUST_GPU_TRACE").is_ok());
                    static COUNTER: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(0);
                    let dispatch_idx = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if trace {
                        eprintln!("[GPU] dispatch #{dispatch_idx} n_in={n_in} n_out={n_out}");
                    }
                    // Synchronous dispatch: one fenced submission covers all
                    // rows; the calling thread (pool thread 0) owns the fenced
                    // completion, so any element-wise epilogue the trunk runs
                    // after this call is safe.
                    match unsafe {
                        ctx.matmul_q8_0(weight, input_q8, input_scales, output, n_in, n_out)
                    } {
                        Ok(()) => return,
                        Err(e) => {
                            // UnsupportedShape: CPU fallback for THIS matmul
                            // only (the GPU stays alive for other shapes).
                            if !matches!(e, crate::vulkan::VulkanError::UnsupportedShape(_)) {
                                crate::vulkan::mark_gpu_broken(&e.to_string());
                            }
                            // Thread 0 re-computes ALL rows on CPU: the other
                            // pool threads already returned, so leaving the
                            // fallback to the per-thread partition would skip
                            // their row ranges.
                            matmul_q8_0_quantized_range(
                                weight,
                                input_q8,
                                input_scales,
                                output,
                                n_in,
                                0,
                                n_out,
                            );
                            return;
                        }
                    }
                } else {
                    return;
                }
            }
        }
    }
    if nth <= 1 || n_out == 0 {
        matmul_q8_0_quantized_range(weight, input_q8, input_scales, output, n_in, 0, n_out);
        return;
    }
    let per_thread = (n_out + nth - 1) / nth;
    let my_start = ith * per_thread;
    let my_end = (my_start + per_thread).min(n_out);
    if my_start >= my_end {
        return;
    }
    matmul_q8_0_quantized_range(
        weight,
        input_q8,
        input_scales,
        &mut output[my_start..my_end],
        n_in,
        my_start,
        my_end,
    );
}

/// ComputePool-dispatched chunked parallel entry.
///
/// Chunks rows into `pool.n_threads() * 4` chunks; each chunk gets a
/// copy of the weight/input slice pointers (Send-by-raw-pointer dance).
///
/// # Bug
///
/// `compute_with_chunks` uses a barrier that expects ALL `pool.n_threads()`
/// participants to call `barrier.wait()`. However, when called from
/// `pool.compute` (persistent worker pool), those workers are spinning in
/// the epoch loop and **cannot** reach this barrier. Only the main thread
/// + newly spawned threads (2-3 of 8 participants) arrive at the barrier.
/// This causes output to be zeroed or unwritten instead of deadlocking.
///
/// # Correct Usage
///
/// This function is BROKEN when used with `pool.compute`. The correct approach
/// is to use `pool.compute(|ith, nth| ...)` and call
/// `matmul_q8_0_quantized_parallel_rows` directly, which divides work by
/// thread index — matching how Qwen3 and all other models achieve parallelism.
/// See `matmul_q8_0_quantized_parallel_rows` for the working implementation.
pub fn matmul_q8_0_quantized_dynamic(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    pool: &crate::core::thread_pool::ComputePool,
) {
    if n_out == 0 {
        return;
    }
    let chunk_size = 16.max(n_out / (pool.n_threads() * 4));
    let n_chunks = (n_out as i32 + chunk_size as i32 - 1) / chunk_size as i32;
    let w_ptr = weight.as_ptr() as usize;
    let w_len = weight.len();
    let iq_ptr = input_q8.as_ptr() as usize;
    let iq_len = input_q8.len();
    let sc_ptr = input_scales.as_ptr() as usize;
    let sc_len = input_scales.len();
    let out_ptr = output.as_mut_ptr() as usize;
    pool.compute_with_chunks(n_chunks, move |_ith, chunk_id| {
        let row_start = (chunk_id as usize) * chunk_size;
        let row_end = (row_start + chunk_size).min(n_out);
        if row_start >= row_end {
            return;
        }
        let w = unsafe { std::slice::from_raw_parts(w_ptr as *const u8, w_len) };
        let iq = unsafe { std::slice::from_raw_parts(iq_ptr as *const u8, iq_len) };
        let sc = unsafe { std::slice::from_raw_parts(sc_ptr as *const f32, sc_len) };
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut(
                (out_ptr as *mut f32).add(row_start),
                row_end - row_start,
            )
        };
        matmul_q8_0_quantized_range(w, iq, sc, out_slice, n_in, row_start, row_end);
    });
}

/// Recursive-split parallel entry.
///
/// Splits output range at midpoint via `rayon::join`, recursing until
/// the slice is below `min_rows` (64), then dispatches via AVX2/NEON/scalar.
pub fn matmul_q8_0_quantized_parallel(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
) {
    let min_rows = 64;
    parallel_range(
        weight,
        input_q8,
        input_scales,
        output,
        n_in,
        0,
        n_out,
        min_rows,
    );
}

fn parallel_range(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
    min_rows: usize,
) {
    let n = row_end - row_start;
    if n <= min_rows {
        matmul_q8_0_quantized_range(
            weight,
            input_q8,
            input_scales,
            output,
            n_in,
            row_start,
            row_end,
        );
        return;
    }
    let mid_row = row_start + n / 2;
    let mid_idx = mid_row - row_start;
    let (lo, hi) = output.split_at_mut(mid_idx);
    rayon::join(
        || {
            parallel_range(
                weight,
                input_q8,
                input_scales,
                lo,
                n_in,
                row_start,
                mid_row,
                min_rows,
            )
        },
        || {
            parallel_range(
                weight,
                input_q8,
                input_scales,
                hi,
                n_in,
                mid_row,
                row_end,
                min_rows,
            )
        },
    );
}
