//! Q4_0 scalar matmul kernel.
//!
//! Each Q4_0 block holds 32 elements (18 bytes: 2-byte F16 scale + 16-byte
//! nibbles). The hot path is currently scalar — AVX2/NEON variants can
//! be added in a sibling file (e.g. `avx2.rs`) alongside this baseline
//! without touching any other code.

/// Q4_0 scalar matmul kernel. Phase 2.7-final: moved from `ops::matmul`.
/// Each Q4_0 block holds 32 elements (18 bytes: 2-byte F16 scale + 16-byte
/// nibbles). The hot path is currently scalar — AVX2/NEON variants can
/// be added in this file alongside this baseline without touching any
/// other code.
pub fn matmul_q4_0_scalar_range(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    const BLOCK_ELEMENTS: usize = 32;
    const BLOCK_BYTES: usize = 18;
    let n_blocks = n_in / BLOCK_ELEMENTS;
    let row_stride = n_blocks * BLOCK_BYTES;
    let per_thread = (n_out + nth - 1) / nth;
    let my_start = ith * per_thread;
    let my_end = (my_start + per_thread).min(n_out);
    if my_start >= my_end {
        return;
    }
    for out_idx in my_start..my_end {
        let row_off = out_idx * row_stride;
        let mut sum = 0.0f32;
        for block in 0..n_blocks {
            let off = row_off + block * BLOCK_BYTES;
            let d = crate::ops::f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
            let qx = &weight[off + 2..off + BLOCK_BYTES];
            let base_y = block * BLOCK_ELEMENTS;
            let scale = input_scales[block];
            let mut dot: i32 = 0;
            for l in 0..16 {
                let x0 = (qx[l] & 0x0F) as i32 - 8;
                let x1 = (qx[l] >> 4) as i32 - 8;
                let y0 = input_q8[base_y + l] as i8 as i32;
                let y1 = input_q8[base_y + 16 + l] as i8 as i32;
                dot += x0 * y0 + x1 * y1;
            }
            sum += dot as f32 * d * scale;
        }
        output[out_idx] = sum;
    }
}

/// Reuse each Q4 block across four activation rows. Each output retains the
/// single-row integer dot and sequential F32 block accumulation contract.
///
/// # Safety
/// `output` points to `rows * n_out` writable floats. Concurrent calls must
/// use distinct `ith < nth` values with the same positive `nth`, rows and shape.
/// No other references may access the worker's output columns during this call.
pub(crate) unsafe fn matmul_q4_0_batched_scalar_range(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: *mut f32,
    n_in: usize,
    n_out: usize,
    rows: usize,
    ith: usize,
    nth: usize,
) {
    let blocks = n_in / 32;
    let scale_stride = n_in.div_ceil(32);
    let row_stride = blocks * 18;
    let per_thread = n_out.div_ceil(nth);
    let start = ith * per_thread;
    let end = (start + per_thread).min(n_out);
    if start >= end {
        return;
    }
    let full_rows = rows / 4 * 4;
    for out in start..end {
        for row_base in (0..full_rows).step_by(4) {
            let mut sums = [0.0f32; 4];
            for block in 0..blocks {
                let offset = out * row_stride + block * 18;
                let d = crate::ops::f16_to_f32(u16::from_le_bytes([
                    weight[offset],
                    weight[offset + 1],
                ]));
                let mut dots = [0i32; 4];
                for lane in 0..16 {
                    let packed = weight[offset + 2 + lane];
                    let x0 = (packed & 15) as i32 - 8;
                    let x1 = (packed >> 4) as i32 - 8;
                    for row in 0..4 {
                        let y = (row_base + row) * n_in + block * 32 + lane;
                        dots[row] +=
                            x0 * input_q8[y] as i8 as i32 + x1 * input_q8[y + 16] as i8 as i32;
                    }
                }
                for row in 0..4 {
                    sums[row] += dots[row] as f32
                        * d
                        * input_scales[(row_base + row) * scale_stride + block];
                }
            }
            for row in 0..4 {
                // SAFETY: this worker exclusively owns columns start..end.
                unsafe {
                    output.add((row_base + row) * n_out + out).write(sums[row]);
                }
            }
        }
    }
    for row in full_rows..rows {
        // Only this worker's columns become a mutable slice, including when
        // another worker is processing a different partition of this row.
        let tail =
            unsafe { std::slice::from_raw_parts_mut(output.add(row * n_out + start), end - start) };
        matmul_q4_0_scalar_range(
            &weight[start * row_stride..end * row_stride],
            &input_q8[row * n_in..(row + 1) * n_in],
            &input_scales[row * scale_stride..(row + 1) * scale_stride],
            tail,
            n_in,
            end - start,
            0,
            1,
        );
    }
}
