use crate::ops::f16_to_f32;

pub fn matmul_q8_0_fallback_range(
    weight: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    let blocks_per_row = n_in / 32;
    let row_stride = blocks_per_row * 34;
    for (out_idx, j) in (row_start..row_end).enumerate() {
        let row_off = j * row_stride;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let off = row_off + b * 34;
            let d = f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
            let qs = &weight[off + 2..off + 34];
            let inp = &input[b * 32..];
            let mut local = 0.0f32;
            for k in 0..32 {
                local += (qs[k] as i8 as f32) * inp[k];
            }
            sum += d * local;
        }
        output[out_idx] = sum;
    }
}

pub fn matmul_q8_0_quantized_scalar_range(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    row_start: usize,
    row_end: usize,
) {
    let blocks_per_row = n_in / 32;
    let row_stride = blocks_per_row * 34;
    for (out_idx, row) in (row_start..row_end).enumerate() {
        let row_off = row * row_stride;
        let mut sum = 0.0f32;
        for block in 0..blocks_per_row {
            let off = row_off + block * 34;
            let wd = f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
            let qx = &weight[off + 2..off + 34];
            let qy = &input_q8[block * 32..(block + 1) * 32];
            let mut dot = 0i32;
            for lane in 0..32 {
                dot += (qx[lane] as i8 as i32) * (qy[lane] as i8 as i32);
            }
            sum += wd * input_scales[block] * dot as f32;
        }
        output[out_idx] = sum;
    }
}

pub fn q8_0_dot_row(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    n_in: usize,
    row: usize,
    _use_avx2: bool,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    let blocks_per_row = n_in / 32;
    #[cfg(target_arch = "x86_64")]
    let row_stride = blocks_per_row * 34;
    #[cfg(target_arch = "x86_64")]
    if _use_avx2 {
        return unsafe {
            super::avx2::q8_0_dot_row_avx2(
                weight,
                input_q8,
                input_scales,
                n_in,
                row,
                blocks_per_row,
                row_stride,
            )
        };
    }
    #[cfg(target_arch = "aarch64")]
    if crate::ops::has_neon() {
        let mut output = [0.0];
        unsafe {
            super::neon::matmul_q8_0_vs_q8_0_neon(
                weight,
                input_q8,
                input_scales,
                &mut output,
                n_in,
                row,
                row + 1,
            );
        }
        return output[0];
    }
    let mut output = [0.0];
    matmul_q8_0_quantized_scalar_range(
        weight,
        input_q8,
        input_scales,
        &mut output,
        n_in,
        row,
        row + 1,
    );
    output[0]
}
