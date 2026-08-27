use crate::ops::f16_to_f32;

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
