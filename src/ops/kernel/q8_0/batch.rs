//! Q8_0 batched matmul tasks.
//!
//! Phase 2.7-final: split from `ops::matmul`. The `MatmulTask` struct and
//! `matmul_q8_0_batch` function are retained for callers that want to
//! bundle multiple f32-input matmuls into a single rayon dispatch.
//!
//! Note: this API uses raw f32 input (interpreted from bytes), not the
//! prequantized Q8 input used by the production hot path in `parallel`.

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use crate::ops::has_avx2_fma;

#[cfg(target_arch = "x86_64")]
use super::avx2::matmul_q8_0_avx2_range;
use super::scalar::matmul_q8_0_fallback_range;

/// One row of a batched Q8_0 × f32 matmul.
pub struct MatmulTask<'a> {
    pub weight: &'a [u8],
    pub input: &'a [f32],
    pub output: &'a mut [f32],
    pub n_in: usize,
    pub n_out: usize,
}

/// Dispatch a batch of Q8_0 × f32 matmuls via rayon.
///
/// Each task is split into 128-row chunks; chunks are dispatched in
/// parallel. AVX2 path is selected when available, otherwise scalar.
pub fn matmul_q8_0_batch(tasks: &mut [MatmulTask<'_>]) {
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = has_avx2_fma();
    let chunk = 128;
    struct TaskInfo {
        w_ptr: usize,
        w_len: usize,
        i_ptr: usize,
        i_len: usize,
        o_ptr: usize,
        n_in: usize,
    }
    unsafe impl Sync for TaskInfo {}
    let mut infos: Vec<TaskInfo> = Vec::new();
    let mut work_items: Vec<(usize, usize, usize)> = Vec::new();
    for task in tasks.iter_mut() {
        infos.push(TaskInfo {
            w_ptr: task.weight.as_ptr() as usize,
            w_len: task.weight.len(),
            i_ptr: task.input.as_ptr() as usize,
            i_len: task.input.len(),
            o_ptr: task.output.as_mut_ptr() as usize,
            n_in: task.n_in,
        });
        let n_chunks = (task.n_out + chunk - 1) / chunk;
        let ti = infos.len() - 1;
        for ci in 0..n_chunks {
            let rs = ci * chunk;
            let re = (rs + chunk).min(task.n_out);
            work_items.push((ti, rs, re));
        }
    }
    work_items.par_iter().for_each(|&(ti, rs, re)| {
        let info = &infos[ti];
        let weight = unsafe { std::slice::from_raw_parts(info.w_ptr as *const u8, info.w_len) };
        let input = unsafe { std::slice::from_raw_parts(info.i_ptr as *const f32, info.i_len) };
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut((info.o_ptr as *mut f32).add(rs), re - rs) };
        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            unsafe {
                matmul_q8_0_avx2_range(weight, input, out_slice, info.n_in, rs, re);
            }
            return;
        }
        matmul_q8_0_fallback_range(weight, input, out_slice, info.n_in, rs, re);
    });
}
