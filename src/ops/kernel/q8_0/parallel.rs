//! Q8_0 top-level parallel entry points.
//!
//! Phase 2.7-final: split from `ops::matmul`. `matmul_q8_0_quantized_parallel_rows`
//! is the hot-path entry called by `Q8Kernel::forward_prequantized`,
//! `bin/server.rs`, and `app/text.rs`. The other variants provide
//! different parallelization strategies (dynamic chunking, recursive split,
//! rayon chunks).

use crate::ops::has_avx2_fma;
#[cfg(target_arch = "aarch64")]
use crate::ops::has_neon;
#[cfg(feature = "vulkan")]
use crate::ops::get_vulkan_context;
#[cfg(feature = "wgpu")]
use crate::ops::get_wgpu_context;

#[cfg(target_arch = "x86_64")]
use super::avx2::{matmul_q8_0_avx2_range, matmul_q8_0_vs_q8_0_avx2};
#[cfg(target_arch = "aarch64")]
use super::neon::matmul_q8_0_vs_q8_0_neon;
use super::scalar::{matmul_q8_0_fallback_range, matmul_q8_0_quantized_scalar_range};
use super::dispatch::matmul_q8_0_quantized_range;

/// Single-thread entry: GPU → AVX2 → NEON → scalar, on the full output.
pub fn matmul_q8_0_quantized(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
) {
    #[cfg(feature = "vulkan")]
    {
        if let Some(ctx) = get_vulkan_context() {
            unsafe {
                ctx.matmul_q8_0(weight, input_q8, input_scales, output, n_in, n_out)
                    .expect("GPU matmul failed");
            }
            return;
        }
    }
    #[cfg(feature = "wgpu")]
    {
        if let Some(ctx) = get_wgpu_context() {
            unsafe {
                ctx.matmul_q8_0(weight, input_q8, input_scales, output, n_in, n_out)
                    .expect("WGPU matmul failed");
            }
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe {
                matmul_q8_0_vs_q8_0_avx2(
                    weight,
                    input_q8,
                    input_scales,
                    output,
                    n_in,
                    0,
                    n_out,
                );
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            unsafe {
                matmul_q8_0_vs_q8_0_neon(
                    weight,
                    input_q8,
                    input_scales,
                    output,
                    n_in,
                    0,
                    n_out,
                );
            }
            return;
        }
    }
    matmul_q8_0_quantized_scalar_range(
        weight,
        input_q8,
        input_scales,
        output,
        n_in,
        0,
        n_out,
    );
}

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
    if nth <= 1 || n_out == 0 {
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
    let use_avx2 = has_avx2_fma();
    let min_rows = 64;
    parallel_range(
        weight,
        input_q8,
        input_scales,
        output,
        n_in,
        0,
        n_out,
        use_avx2,
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
    use_avx2: bool,
    min_rows: usize,
) {
    let n = row_end - row_start;
    if n <= min_rows {
        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            unsafe {
                matmul_q8_0_vs_q8_0_avx2(
                    weight,
                    input_q8,
                    input_scales,
                    output,
                    n_in,
                    row_start,
                    row_end,
                );
            }
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if has_neon() {
            unsafe {
                matmul_q8_0_vs_q8_0_neon(
                    weight,
                    input_q8,
                    input_scales,
                    output,
                    n_in,
                    row_start,
                    row_end,
                );
            }
            return;
        }
        matmul_q8_0_quantized_scalar_range(
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
                use_avx2,
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
                use_avx2,
                min_rows,
            )
        },
    );
}

/// Legacy f32-input matmul: AVX2 → scalar, single-thread.
pub fn matmul_q8_0(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe {
                matmul_q8_0_avx2_range(
                    weight,
                    input,
                    output,
                    n_in,
                    0,
                    n_out,
                );
            }
            return;
        }
    }
    matmul_q8_0_fallback_range(weight, input, output, n_in, 0, n_out);
}

/// Legacy f32-input matmul, parallel via rayon chunks.
pub fn matmul_q8_0_parallel(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    _n_threads: usize,
) {
    use rayon::prelude::*;
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = has_avx2_fma();
    let chunk = 128;
    output
        .par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(i, out_slice)| {
            let rs = i * chunk;
            let re = (rs + chunk).min(n_out);
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                unsafe {
                    matmul_q8_0_avx2_range(weight, input, out_slice, n_in, rs, re);
                }
                return;
            }
            matmul_q8_0_fallback_range(weight, input, out_slice, n_in, rs, re);
        });
}
