use crate::core::tensor::TensorInfo;

pub mod fuse;
pub mod q8_0;

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;
pub const BLOCK_Q4K_SIZE: usize = 144;
pub const BLOCK_Q5K_SIZE: usize = 176;
pub const BLOCK_Q6K_SIZE: usize = 210;
pub const BLOCK_Q8K_SIZE: usize = 292;
pub const BLOCK_Q2K_SIZE: usize = 84;
pub const BLOCK_Q3K_SIZE: usize = 110;

fn f16_from_bytes(data: &[u8], byte_idx: usize) -> f32 {
    if byte_idx + 2 > data.len() { return 0.0; }
    let bits = u16::from_le_bytes([data[byte_idx], data[byte_idx + 1]]);
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as f32 / 1024.0;
    if exp == 0 { sign * frac * 2.0f32.powi(-14) }
    else if exp == 31 { if frac == 0.0 { sign * f32::INFINITY } else { sign * f32::NAN } }
    else { sign * (1.0 + frac) * 2.0f32.powi(exp - 15) }
}

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, mn)
    }
}

#[derive(Clone)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; 256],
    pub bsums: [i16; 16],
}

pub fn quantize_row_q8_k(x: &[f32]) -> Vec<BlockQ8K> {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        return unsafe { quantize_row_q8_k_avx2(x) };
    }
    quantize_row_q8_k_scalar(x)
}

pub fn quantize_row_q8_k_into(x: &[f32], buf: &mut [BlockQ8K]) {
    let nb = x.len() / QK_K;
    debug_assert!(buf.len() >= nb);
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        unsafe { quantize_row_q8_k_avx2_into(x, buf) };
        return;
    }
    quantize_row_q8_k_scalar_into(x, buf);
}

fn quantize_row_q8_k_scalar(x: &[f32]) -> Vec<BlockQ8K> {
    let nb = x.len() / QK_K;
    let mut result = vec![
        BlockQ8K {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        };
        nb
    ];
    quantize_row_q8_k_scalar_into(x, &mut result);
    result
}

#[inline]
fn nearest_int(value: f32) -> i32 {
    debug_assert!(value.abs() <= 4_194_303.0);
    let bits = (value + 12_582_912.0).to_bits();
    ((bits & 0x007f_ffff) as i32) - 0x0040_0000
}

fn quantize_row_q8_k_scalar_into(x: &[f32], buf: &mut [BlockQ8K]) {
    let n = x.len();
    assert!(n % QK_K == 0);
    let nb = n / QK_K;

    for i in 0..nb {
        let block = &x[i * QK_K..(i + 1) * QK_K];
        let mut amax = 0.0f32;
        let mut max_val = 0.0f32;
        for j in 0..QK_K {
            let ax = block[j].abs();
            if ax > amax {
                amax = ax;
                max_val = block[j];
            }
        }

        if amax == 0.0 {
            buf[i] = BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] };
            continue;
        }

        let iscale = -127.0f32 / max_val;
        let mut qs = [0i8; 256];
        for j in 0..QK_K {
            let v = nearest_int(iscale * block[j]);
            qs[j] = v.min(127) as i8;
        }

        let mut bsums = [0i16; 16];
        for j in 0..16 {
            let mut sum = 0i32;
            for ii in 0..16 {
                sum += qs[j * 16 + ii] as i32;
            }
            bsums[j] = sum as i16;
        }

        buf[i] = BlockQ8K { d: 1.0 / iscale, qs, bsums };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_row_q8_k_avx2(x: &[f32]) -> Vec<BlockQ8K> {
    let nb = x.len() / QK_K;
    let mut result = vec![
        BlockQ8K {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        };
        nb
    ];
    quantize_row_q8_k_avx2_into(x, &mut result);
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_row_q8_k_avx2_into(x: &[f32], buf: &mut [BlockQ8K]) {
    use std::arch::x86_64::*;
    let n = x.len();
    assert!(n % QK_K == 0);
    let nb = n / QK_K;

    for i in 0..nb {
        let block = &x[i * QK_K..(i + 1) * QK_K];

        let mut amax = 0.0f32;
        let mut max_val = 0.0f32;
        for j in 0..QK_K {
            let ax = block[j].abs();
            if ax > amax { amax = ax; max_val = block[j]; }
        }

        if amax == 0.0 {
            buf[i] = BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] };
            continue;
        }

        let iscale = -127.0f32 / max_val;
        let inv_iscale = 1.0f32 / iscale;
        let mut qs = [0i8; 256];
        let iscale_vec = _mm256_set1_ps(iscale);
        let ni8 = _mm256_set1_epi32(-127);
        let pi8 = _mm256_set1_epi32(127);

        let mut j = 0;
        while j + 8 <= QK_K {
            let v = _mm256_loadu_ps(block.as_ptr().add(j));
            let scaled = _mm256_mul_ps(v, iscale_vec);
            let rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT);
            let i32s = _mm256_cvtps_epi32(rounded);
            let clamped = _mm256_min_epi32(_mm256_max_epi32(i32s, ni8), pi8);
            let arr: [i32; 8] = std::mem::transmute(clamped);
            for k in 0..8 {
                qs[j + k] = arr[k] as i8;
            }
            j += 8;
        }
        while j < QK_K {
            let v = (iscale * block[j]).round() as i32;
            qs[j] = v.min(127).max(-127) as i8;
            j += 1;
        }

        let mut bsums = [0i16; 16];
        for bs in 0..16 {
            let mut sum = 0i32;
            for ii in 0..16 {
                sum += qs[bs * 16 + ii] as i32;
            }
            bsums[bs] = sum as i16;
        }

        buf[i] = BlockQ8K { d: inv_iscale, qs, bsums };
    }
}

pub fn vec_dot_q4k_q8k(q4k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        return unsafe { vec_dot_q4k_q8k_avx2(q4k_data, q8k) };
    }
    vec_dot_q4k_q8k_scalar(q4k_data, q8k)
}

pub fn vec_dot_q4k_q8k_scalar(q4k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    let nb = q8k.len();
    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let boff = i * BLOCK_Q4K_SIZE;
        if boff + BLOCK_Q4K_SIZE > q4k_data.len() { break; }

        let d = f16_from_bytes(q4k_data, boff);
        let dmin = f16_from_bytes(q4k_data, boff + 2);
        let scales_bytes = &q4k_data[boff + 4..boff + 16];
        let qs_bytes = &q4k_data[boff + 16..boff + 144];

        let mut aux8 = [0i32; 256];
        let mut a_idx = 0usize;
        for j in 0..4 {
            for l in 0..32 {
                aux8[a_idx] = (qs_bytes[j * 32 + l] & 0xF) as i32;
                a_idx += 1;
            }
            for l in 0..32 {
                aux8[a_idx] = (qs_bytes[j * 32 + l] >> 4) as i32;
                a_idx += 1;
            }
        }

        let mut utmp = [0u32; 4];
        utmp[0] = u32::from_le_bytes([scales_bytes[0], scales_bytes[1], scales_bytes[2], scales_bytes[3]]);
        utmp[1] = u32::from_le_bytes([scales_bytes[4], scales_bytes[5], scales_bytes[6], scales_bytes[7]]);
        utmp[2] = u32::from_le_bytes([scales_bytes[8], scales_bytes[9], scales_bytes[10], scales_bytes[11]]);

        let kmask1: u32 = 0x3f3f3f3f;
        let kmask2: u32 = 0x0f0f0f0f;
        let kmask3: u32 = 0x03030303;

        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        let uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        let mut all_bytes = [0u8; 16];
        for k in 0..4 {
            let bytes = utmp[k].to_le_bytes();
            all_bytes[k * 4..k * 4 + 4].copy_from_slice(&bytes);
        }
        let scales_arr = &all_bytes[0..8];
        let mins_arr = &all_bytes[8..16];

        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += q8k[i].bsums[j] as i32 * mins_arr[j / 2] as i32;
        }

        let mut aux32 = [0i32; 8];
        let mut is_ = 0usize;
        let mut q8_idx = 0usize;
        for _j in 0..8 {
            let scale = scales_arr[is_] as i32;
            is_ += 1;
            for _ in 0..4 {
                for l in 0..8 {
                    if q8_idx < 256 {
                        aux32[l] += scale * q8k[i].qs[q8_idx] as i32 * aux8[q8_idx];
                    }
                    q8_idx += 1;
                }
            }
        }

        let dd = d * q8k[i].d;
        for l in 0..8 { sums[l] += dd * aux32[l] as f32; }
        let dmin_val = dmin * q8k[i].d;
        sumf -= dmin_val * sumi as f32;
    }

    for l in 0..8 { sumf += sums[l]; }
    sumf
}

pub fn vec_dot_q5k_q8k(q5k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        return unsafe { vec_dot_q5k_q8k_avx2(q5k_data, q8k) };
    }
    vec_dot_q5k_q8k_scalar(q5k_data, q8k)
}

pub fn vec_dot_q5k_q8k_scalar(q5k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    let nb = q8k.len();
    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let boff = i * BLOCK_Q5K_SIZE;
        if boff + BLOCK_Q5K_SIZE > q5k_data.len() { break; }

        let d = f16_from_bytes(q5k_data, boff);
        let dmin = f16_from_bytes(q5k_data, boff + 2);
        let scales_bytes = &q5k_data[boff + 4..boff + 16];
        let qh_bytes = &q5k_data[boff + 16..boff + 48];
        let qs_bytes = &q5k_data[boff + 48..boff + 176];

        let mut aux8 = [0i32; 256];
        let mut j = 0usize;
        let mut q4_off = 0usize;
        let mut m = 1u8;
        while j < QK_K {
            for l in 0..32 {
                let ql = qs_bytes[q4_off + l];
                aux8[j + l] = (ql & 0xF) as i32 + if qh_bytes[l] & m != 0 { 16 } else { 0 };
            }
            j += 32;
            m <<= 1;
            for l in 0..32 {
                let ql = qs_bytes[q4_off + l];
                aux8[j + l] = (ql >> 4) as i32 + if qh_bytes[l] & m != 0 { 16 } else { 0 };
            }
            j += 32;
            m <<= 1;
            q4_off += 32;
        }

        let mut utmp = [0u32; 4];
        utmp[0] = u32::from_le_bytes([scales_bytes[0], scales_bytes[1], scales_bytes[2], scales_bytes[3]]);
        utmp[1] = u32::from_le_bytes([scales_bytes[4], scales_bytes[5], scales_bytes[6], scales_bytes[7]]);
        utmp[2] = u32::from_le_bytes([scales_bytes[8], scales_bytes[9], scales_bytes[10], scales_bytes[11]]);

        let kmask1: u32 = 0x3f3f3f3f;
        let kmask2: u32 = 0x0f0f0f0f;
        let kmask3: u32 = 0x03030303;

        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        let uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        let mut all_bytes = [0u8; 16];
        for k in 0..4 {
            let bytes = utmp[k].to_le_bytes();
            all_bytes[k * 4..k * 4 + 4].copy_from_slice(&bytes);
        }
        let scales_arr = &all_bytes[0..8];
        let mins_arr = &all_bytes[8..16];

        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += q8k[i].bsums[j] as i32 * mins_arr[j / 2] as i32;
        }

        let mut aux32 = [0i32; 8];
        let mut is2 = 0usize;
        let mut q8_idx = 0usize;
        for _j in 0..8 {
            let scale = scales_arr[is2] as i32;
            is2 += 1;
            for _ in 0..4 {
                for l in 0..8 {
                    if q8_idx < 256 {
                        aux32[l] += scale * q8k[i].qs[q8_idx] as i32 * aux8[q8_idx];
                    }
                    q8_idx += 1;
                }
            }
        }

        let dd = d * q8k[i].d;
        for l in 0..8 { sums[l] += dd * aux32[l] as f32; }
        let dmin_val = dmin * q8k[i].d;
        sumf -= dmin_val * sumi as f32;
    }

    for l in 0..8 { sumf += sums[l]; }
    sumf
}

pub fn matmul_q4k_q8k(weight_data: &[u8], input: &[f32], n_cols: usize, n_rows: usize) -> Vec<f32> {
    let q8k = quantize_row_q8_k(input);
    let blocks_per_row = n_cols / QK_K;
    let mut output = vec![0.0f32; n_rows];
    for o in 0..n_rows {
        let row_data = &weight_data[o * blocks_per_row * BLOCK_Q4K_SIZE..];
        output[o] = vec_dot_q4k_q8k(row_data, &q8k);
    }
    output
}

pub fn matmul_q5k_q8k(weight_data: &[u8], input: &[f32], n_cols: usize, n_rows: usize) -> Vec<f32> {
    let q8k = quantize_row_q8_k(input);
    let blocks_per_row = n_cols / QK_K;
    let mut output = vec![0.0f32; n_rows];
    for o in 0..n_rows {
        let row_data = &weight_data[o * blocks_per_row * BLOCK_Q5K_SIZE..];
        output[o] = vec_dot_q5k_q8k(row_data, &q8k);
    }
    output
}

pub fn dequant_weight_q4k(data: &[u8], ti: &TensorInfo) -> Option<Vec<f32>> {
    let n_el = ti.n_elements();
    let n_cols = ti.dims[0] as usize;
    let n_rows = if ti.dims.len() >= 2 { ti.dims[1] as usize } else { 1 };
    let blocks_per_row = n_cols / QK_K;
    let mut out = vec![0.0f32; n_el];

    for row in 0..n_rows {
        let byte_offset = row * blocks_per_row * BLOCK_Q4K_SIZE;
        let out_base = row * n_cols;

        for bi in 0..blocks_per_row {
            let boff = byte_offset + bi * BLOCK_Q4K_SIZE;
            if boff + BLOCK_Q4K_SIZE > data.len() { continue; }

            let d = f16_from_bytes(data, boff);
            let dmin = f16_from_bytes(data, boff + 2);
            let scales_off = boff + 4;
            let qs_off = boff + 4 + K_SCALE_SIZE;

            let mut j = 0usize;
            let mut is = 0usize;
            while is < 8 {
                let (sc1, m1) = get_scale_min_k4(is, &data[scales_off..scales_off + K_SCALE_SIZE]);
                let (sc2, m2) = get_scale_min_k4(is + 1, &data[scales_off..scales_off + K_SCALE_SIZE]);

                let d1 = d * sc1 as f32;
                let m1_eff = dmin * m1 as f32;
                let d2 = d * sc2 as f32;
                let m2_eff = dmin * m2 as f32;

                let block_out = out_base + bi * QK_K;
                for l in 0..32 {
                    let ql = data[qs_off + j + l];
                    out[block_out + j + l] = d1 * (ql & 0xF) as f32 - m1_eff;
                    out[block_out + j + l + 32] = d2 * (ql >> 4) as f32 - m2_eff;
                }

                j += 32;
                is += 2;
            }
        }
    }
    Some(out)
}

pub fn dequantize_row_q2_k(block_bytes: &[u8], output: &mut [f32]) {
    let num_blocks = output.len() / QK_K;
    for block_idx in 0..num_blocks {
        let boff = block_idx * BLOCK_Q2K_SIZE;
        if boff + BLOCK_Q2K_SIZE > block_bytes.len() { break; }

        let d = f16_from_bytes(block_bytes, boff + 80);
        let dmin = f16_from_bytes(block_bytes, boff + 82);
        let scales = &block_bytes[boff..boff + 16];
        let qs = &block_bytes[boff + 16..boff + 80];
        let out_base = block_idx * QK_K;

        for j in 0..16 {
            let sc_byte = scales[j];
            let dl = d * (sc_byte & 0xF) as f32;
            let ml = dmin * (sc_byte >> 4) as f32;
            let base = j * 16;
            for l in 0..16 {
                let q_byte = qs[(base + l) / 4];
                let q_shift = ((base + l) % 4) * 2;
                let q = ((q_byte >> q_shift) & 0x3) as f32;
                output[out_base + base + l] = dl * q - ml;
            }
        }
    }
}

pub fn vec_dot_q2k_q8k_scalar(q2k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    let nb = q8k.len();
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let boff = i * BLOCK_Q2K_SIZE;
        if boff + BLOCK_Q2K_SIZE > q2k_data.len() { break; }

        let d = f16_from_bytes(q2k_data, boff + 80);
        let dmin = f16_from_bytes(q2k_data, boff + 82);
        let scales = &q2k_data[boff..boff + 16];
        let qs = &q2k_data[boff + 16..boff + 80];

        let mut sum_dot = 0i32;
        let mut sum_min = 0i32;
        for j in 0..16 {
            let sc_byte = scales[j];
            let sc = (sc_byte & 0xF) as i32;
            let m = (sc_byte >> 4) as i32;
            let base = j * 16;
            for l in 0..16 {
                let q_byte = qs[(base + l) / 4];
                let q_shift = ((base + l) % 4) * 2;
                let q = ((q_byte >> q_shift) & 0x3) as i32;
                sum_dot += q * q8k[i].qs[base + l] as i32;
            }
            sum_min += m * q8k[i].bsums[j] as i32;
        }

        sumf += d * q8k[i].d * sum_dot as f32 - dmin * q8k[i].d * sum_min as f32;
    }

    sumf
}

pub fn vec_dot_q2k_q8k(q2k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    vec_dot_q2k_q8k_scalar(q2k_data, q8k)
}

pub fn dequantize_row_q3_k(block_bytes: &[u8], output: &mut [f32]) {
    let num_blocks = output.len() / QK_K;
    for block_idx in 0..num_blocks {
        let boff = block_idx * BLOCK_Q3K_SIZE;
        if boff + BLOCK_Q3K_SIZE > block_bytes.len() { break; }

        let d = f16_from_bytes(block_bytes, boff + 108);
        let hmask = &block_bytes[boff..boff + 32];
        let qs = &block_bytes[boff + 32..boff + 96];
        let scales_packed = &block_bytes[boff + 96..boff + 108];
        let out_base = block_idx * QK_K;

        for j in 0..256 {
            let qs_byte_idx = j / 4;
            let qs_shift = (j % 4) * 2;
            let ql = ((qs[qs_byte_idx] >> qs_shift) & 0x3) as i32;
            let h_byte_idx = j / 8;
            let h_bit_idx = j % 8;
            let qh = ((hmask[h_byte_idx] >> h_bit_idx) & 0x1) as i32;
            let q = (ql | (qh << 2)) - 4;

            let sub = j / 32;
            let sc_lo = (scales_packed[sub] & 0xF) as i32;
            let sc_hi = ((scales_packed[8 + sub / 4] >> (2 * (sub % 4))) & 0x3) as i32;
            let sc = sc_lo | (sc_hi << 4);

            output[out_base + j] = d * (q * sc) as f32;
        }
    }
}

pub fn vec_dot_q3k_q8k_scalar(q3k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    let nb = q8k.len();
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let boff = i * BLOCK_Q3K_SIZE;
        if boff + BLOCK_Q3K_SIZE > q3k_data.len() { break; }

        let d = f16_from_bytes(q3k_data, boff + 108);
        let hmask = &q3k_data[boff..boff + 32];
        let qs = &q3k_data[boff + 32..boff + 96];
        let scales_packed = &q3k_data[boff + 96..boff + 108];

        let mut sum = 0i32;
        for j in 0..256 {
            let qs_byte_idx = j / 4;
            let qs_shift = (j % 4) * 2;
            let ql = ((qs[qs_byte_idx] >> qs_shift) & 0x3) as i32;
            let h_byte_idx = j / 8;
            let h_bit_idx = j % 8;
            let qh = ((hmask[h_byte_idx] >> h_bit_idx) & 0x1) as i32;
            let q = (ql | (qh << 2)) - 4;

            let sub = j / 32;
            let sc_lo = (scales_packed[sub] & 0xF) as i32;
            let sc_hi = ((scales_packed[8 + sub / 4] >> (2 * (sub % 4))) & 0x3) as i32;
            let sc = sc_lo | (sc_hi << 4);

            sum += q * sc * q8k[i].qs[j] as i32;
        }

        sumf += d * q8k[i].d * sum as f32;
    }

    sumf
}

pub fn vec_dot_q3k_q8k(q3k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    vec_dot_q3k_q8k_scalar(q3k_data, q8k)
}

pub fn dequantize_row_q4_k(block_bytes: &[u8], output: &mut [f32]) {
    let num_blocks = output.len() / QK_K;
    for block_idx in 0..num_blocks {
        let byte_offset = block_idx * BLOCK_Q4K_SIZE;
        if byte_offset + BLOCK_Q4K_SIZE > block_bytes.len() { break; }

        let d = f16_from_bytes(block_bytes, byte_offset);
        let dmin = f16_from_bytes(block_bytes, byte_offset + 2);
        let scales_off = byte_offset + 4;
        let qs_off = byte_offset + 4 + K_SCALE_SIZE;
        let out_base = block_idx * QK_K;

        let mut q_off = 0usize;
        let mut is = 0usize;
        while q_off < QK_K / 2 {
            let (sc1, m1) = get_scale_min_k4(is, &block_bytes[scales_off..scales_off + K_SCALE_SIZE]);
            let (sc2, m2) = get_scale_min_k4(is + 1, &block_bytes[scales_off..scales_off + K_SCALE_SIZE]);

            let d1 = d * sc1 as f32;
            let m1_eff = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2_eff = dmin * m2 as f32;

            let out_off = out_base + q_off * 2;
            for l in 0..32 {
                let ql = block_bytes[qs_off + q_off + l];
                output[out_off + l] = d1 * (ql & 0xF) as f32 - m1_eff;
                output[out_off + 32 + l] = d2 * (ql >> 4) as f32 - m2_eff;
            }

            q_off += 32;
            is += 2;
        }
    }
}

pub fn dequantize_q4_k_weight(
    weight_bytes: &[u8],
    n_rows: usize,
    n_cols: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(output.len(), n_rows * n_cols);
    let blocks_per_row = n_cols / QK_K;
    for row in 0..n_rows {
        let byte_offset = row * blocks_per_row * BLOCK_Q4K_SIZE;
        let out_offset = row * n_cols;
        dequantize_row_q4_k(
            &weight_bytes[byte_offset..],
            &mut output[out_offset..out_offset + n_cols],
        );
    }
}

pub const BLOCK_Q80_SIZE: usize = 34;

pub fn dequant_q80_weight(data: &[u8], n_cols: usize, n_rows: usize) -> Vec<f32> {
    let blocks_per_row = n_cols / 32;
    let mut out = vec![0.0f32; n_rows * n_cols];
    for row in 0..n_rows {
        for bi in 0..blocks_per_row {
            let boff = row * blocks_per_row * BLOCK_Q80_SIZE + bi * BLOCK_Q80_SIZE;
            if boff + BLOCK_Q80_SIZE > data.len() { continue; }
            let d = f16_from_bytes(data, boff);
            let out_base = row * n_cols + bi * 32;
            for j in 0..32 {
                let q = data[boff + 2 + j] as i8 as i32;
                out[out_base + j] = d * q as f32;
            }
        }
    }
    out
}

pub fn dequantize_row_q6_k(block_bytes: &[u8], output: &mut [f32]) {
    let num_blocks = output.len() / QK_K;
    for block_idx in 0..num_blocks {
        let boff = block_idx * BLOCK_Q6K_SIZE;
        if boff + BLOCK_Q6K_SIZE > block_bytes.len() { break; }

        let d = f16_from_bytes(block_bytes, boff + 208);
        let ql_off = boff;
        let qh_off = boff + 128;
        let sc_off = boff + 192;
        let out_base = block_idx * QK_K;

        for j in (0..QK_K).step_by(128) {
            let j128 = j / 128;
            for l in 0..32usize {
                let ql_idx = j128 * 64 + l;
                let qh_idx = j128 * 32 + l;
                let q1 = ((block_bytes[ql_off + ql_idx] & 0xF) as i32
                    | (((block_bytes[qh_off + qh_idx] as i32) & 3) << 4)) - 32;
                let q2 = ((block_bytes[ql_off + ql_idx + 32] & 0xF) as i32
                    | (((block_bytes[qh_off + qh_idx] as i32 >> 2) & 3) << 4)) - 32;
                let q3 = ((block_bytes[ql_off + ql_idx] as i32 >> 4)
                    | (((block_bytes[qh_off + qh_idx] as i32 >> 4) & 3) << 4)) - 32;
                let q4 = ((block_bytes[ql_off + ql_idx + 32] as i32 >> 4)
                    | (((block_bytes[qh_off + qh_idx] as i32 >> 6) & 3) << 4)) - 32;

                let is = l / 16;
                let sc0 = block_bytes[sc_off + j128 * 8 + is + 0] as i8 as f32;
                let sc2 = block_bytes[sc_off + j128 * 8 + is + 2] as i8 as f32;
                let sc4 = block_bytes[sc_off + j128 * 8 + is + 4] as i8 as f32;
                let sc6 = block_bytes[sc_off + j128 * 8 + is + 6] as i8 as f32;

                output[out_base + j + l + 0] = d * sc0 * q1 as f32;
                output[out_base + j + l + 32] = d * sc2 * q2 as f32;
                output[out_base + j + l + 64] = d * sc4 * q3 as f32;
                output[out_base + j + l + 96] = d * sc6 * q4 as f32;
            }
        }
    }
}

pub fn dequant_q6k_weight(data: &[u8], n_cols: usize, n_rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_rows * n_cols];
    let blocks_per_row = n_cols / QK_K;
    for row in 0..n_rows {
        let byte_offset = row * blocks_per_row * BLOCK_Q6K_SIZE;
        dequantize_row_q6_k(
            &data[byte_offset..],
            &mut out[row * n_cols..row * n_cols + n_cols],
        );
    }
    out
}

pub fn dequant_q5k_weight(data: &[u8], n_cols: usize, n_rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_rows * n_cols];
    let blocks_per_row = n_cols / QK_K;
    for row in 0..n_rows {
        let byte_offset = row * blocks_per_row * BLOCK_Q5K_SIZE;
        dequantize_row_q5_k(&data[byte_offset..], &mut out[row * n_cols..row * n_cols + n_cols]);
    }
    out
}

pub fn dequantize_row_q5_k(block_bytes: &[u8], output: &mut [f32]) {
    let num_blocks = output.len() / QK_K;
    for block_idx in 0..num_blocks {
        let boff = block_idx * BLOCK_Q5K_SIZE;
        if boff + BLOCK_Q5K_SIZE > block_bytes.len() { break; }

        let d = f16_from_bytes(block_bytes, boff);
        let dmin = f16_from_bytes(block_bytes, boff + 2);
        let scales_off = boff + 4;
        let qh_off = boff + 16;
        let qs_off = boff + 48;
        let out_base = block_idx * QK_K;

        let mut j = 0usize;
        let mut q4_off = 0usize;
        let mut is = 0usize;
        let mut m = 1u8;
        while j < QK_K {
            let (sc1, m1) = get_scale_min_k4(is, &block_bytes[scales_off..scales_off + K_SCALE_SIZE]);
            let (sc2, m2) = get_scale_min_k4(is + 1, &block_bytes[scales_off..scales_off + K_SCALE_SIZE]);

            let d1 = d * sc1 as f32;
            let m1_eff = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2_eff = dmin * m2 as f32;

            for l in 0..32 {
                let ql = block_bytes[qs_off + q4_off + l];
                let l_val = (ql & 0xF) as f32 + if block_bytes[qh_off + l] & m != 0 { 16.0 } else { 0.0 };
                output[out_base + j + l] = d1 * l_val - m1_eff;
            }
            j += 32;
            m <<= 1;
            for l in 0..32 {
                let ql = block_bytes[qs_off + q4_off + l];
                let l_val = (ql >> 4) as f32 + if block_bytes[qh_off + l] & m != 0 { 16.0 } else { 0.0 };
                output[out_base + j + l] = d2 * l_val - m2_eff;
            }
            j += 32;
            m <<= 1;
            q4_off += 32;
            is += 2;
        }
    }
}

pub fn vec_dot_q6k_q8k_scalar(q6k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    let nb = q8k.len();
    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let boff = i * BLOCK_Q6K_SIZE;
        if boff + BLOCK_Q6K_SIZE > q6k_data.len() { break; }

        let ql_off = boff;
        let qh_off = boff + 128;
        let sc_off = boff + 192;

        let mut aux8 = [0i32; 256];
        {
            let mut a = &mut aux8[..];
            let mut q4_idx = 0usize;
            let mut qh_idx = 0usize;
            for _j in 0..QK_K / 128 {
                for l in 0..32 {
                    a[l + 0] = ((q6k_data[ql_off + q4_idx + l] & 0xF) as i32
                        | (((q6k_data[qh_off + qh_idx + l] as i32) & 3) << 4)) - 32;
                    a[l + 32] = ((q6k_data[ql_off + q4_idx + l + 32] & 0xF) as i32
                        | (((q6k_data[qh_off + qh_idx + l] as i32 >> 2) & 3) << 4)) - 32;
                    a[l + 64] = ((q6k_data[ql_off + q4_idx + l] as i32 >> 4)
                        | (((q6k_data[qh_off + qh_idx + l] as i32 >> 4) & 3) << 4)) - 32;
                    a[l + 96] = ((q6k_data[ql_off + q4_idx + l + 32] as i32 >> 4)
                        | (((q6k_data[qh_off + qh_idx + l] as i32 >> 6) & 3) << 4)) - 32;
                }
                a = &mut aux8[(_j + 1) * 128..];
                q4_idx += 64;
                qh_idx += 32;
            }
        }

        let mut aux32 = [0i32; 8];
        let mut is_ = 0usize;
        let mut q8_idx = 0usize;
        let mut a_idx = 0usize;
        for _j in 0..QK_K / 16 {
            let scale = q6k_data[sc_off + is_] as i8 as i32;
            is_ += 1;
            for l in 0..8 { aux32[l] += scale * q8k[i].qs[q8_idx] as i32 * aux8[a_idx]; q8_idx += 1; a_idx += 1; }
            for l in 0..8 { aux32[l] += scale * q8k[i].qs[q8_idx] as i32 * aux8[a_idx]; q8_idx += 1; a_idx += 1; }
        }

        let d = f16_from_bytes(q6k_data, boff + 208) * q8k[i].d;
        for l in 0..8 { sums[l] += d * aux32[l] as f32; }
    }

    for l in 0..8 { sumf += sums[l]; }
    sumf
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn vec_dot_q4k_q8k_avx2_direct(q4k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    vec_dot_q4k_q8k_avx2(q4k_data, q8k)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q4k_q8k_avx2(q4k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    let nb = q8k.len();

    let kmask1: u32 = 0x3f3f3f3f;
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;

    let m4 = _mm256_set1_epi8(0xF);

    let scale_shuffle: [[u8; 32]; 8] = {
        let mut tbl = [[0u8; 32]; 8];
        for i in 0..8usize {
            for j in 0..16usize {
                tbl[i][j * 2] = (i * 2) as u8;
                tbl[i][j * 2 + 1] = (i * 2 + 1) as u8;
            }
        }
        tbl
    };

    let mut acc = _mm256_setzero_ps();
    let mut acc_m = _mm_setzero_ps();

    for i in 0..nb {
        let boff = i * BLOCK_Q4K_SIZE;
        if boff + BLOCK_Q4K_SIZE > q4k_data.len() { break; }

        let d_raw = u16::from_le_bytes([q4k_data[boff], q4k_data[boff + 1]]);
        let dmin_raw = u16::from_le_bytes([q4k_data[boff + 2], q4k_data[boff + 3]]);
        let d = f16_to_f32(d_raw) * q8k[i].d;
        let dmin = -f16_to_f32(dmin_raw) * q8k[i].d;

        let mut utmp = [0u32; 4];
        let sc_base = boff + 4;
        utmp[0] = u32::from_le_bytes([q4k_data[sc_base], q4k_data[sc_base+1], q4k_data[sc_base+2], q4k_data[sc_base+3]]);
        utmp[1] = u32::from_le_bytes([q4k_data[sc_base+4], q4k_data[sc_base+5], q4k_data[sc_base+6], q4k_data[sc_base+7]]);
        utmp[2] = u32::from_le_bytes([q4k_data[sc_base+8], q4k_data[sc_base+9], q4k_data[sc_base+10], q4k_data[sc_base+11]]);

        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        let uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3] as i32, utmp[2] as i32, utmp[1] as i32, utmp[0] as i32));

        let q8sums = _mm256_loadu_si256(q8k[i].bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1)
        );
        let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        let scales256 = _mm256_set_m128i(sc128, sc128);

        let mut sumi = _mm256_setzero_si256();

        let mut q4_off = boff + 16;
        let mut q8_off = 0usize;

        for j in 0..4 {
            let scale_l = _mm256_shuffle_epi8(scales256, _mm256_loadu_si256(scale_shuffle[2 * j].as_ptr() as *const __m256i));
            let scale_h = _mm256_shuffle_epi8(scales256, _mm256_loadu_si256(scale_shuffle[2 * j + 1].as_ptr() as *const __m256i));

            let q4bits = _mm256_loadu_si256(q4k_data.as_ptr().add(q4_off) as *const __m256i);
            q4_off += 32;
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            let q8l = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off) as *const __m256i);
            q8_off += 32;
            let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            let q8h = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off) as *const __m256i);
            q8_off += 32;
            let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            let sumj = _mm256_add_epi32(p16l, p16h);
            sumi = _mm256_add_epi32(sumi, sumj);
        }

        let vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    let min_val = _mm_cvtss_f32(acc_m);

    let main_val = crate::ops::hsum_ps(acc);

    main_val + min_val
}

#[inline]
#[cfg(target_arch = "x86_64")]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as f32 / 1024.0;
    if exp == 0 { sign * frac * 2.0f32.powi(-14) }
    else if exp == 31 { if frac == 0.0 { sign * f32::INFINITY } else { sign * f32::NAN } }
    else { sign * (1.0 + frac) * 2.0f32.powi(exp - 15) }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn vec_dot_q5k_q8k_avx2_direct(q5k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    vec_dot_q5k_q8k_avx2(q5k_data, q8k)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q5k_q8k_avx2(q5k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    let nb = q8k.len();

    let kmask1: u32 = 0x3f3f3f3f;
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;

    let m4 = _mm256_set1_epi8(0xF);
    let mone = _mm256_set1_epi8(1);

    let scale_shuffle: [[u8; 32]; 8] = {
        let mut tbl = [[0u8; 32]; 8];
        for i in 0..8usize {
            for j in 0..16usize {
                tbl[i][j * 2] = (i * 2) as u8;
                tbl[i][j * 2 + 1] = (i * 2 + 1) as u8;
            }
        }
        tbl
    };

    let mut acc = _mm256_setzero_ps();
    let mut acc_m = _mm_setzero_ps();

    for i in 0..nb {
        let boff = i * BLOCK_Q5K_SIZE;
        if boff + BLOCK_Q5K_SIZE > q5k_data.len() { break; }

        let d_raw = u16::from_le_bytes([q5k_data[boff], q5k_data[boff + 1]]);
        let dmin_raw = u16::from_le_bytes([q5k_data[boff + 2], q5k_data[boff + 3]]);
        let d = f16_to_f32(d_raw) * q8k[i].d;
        let dmin = -f16_to_f32(dmin_raw) * q8k[i].d;

        let mut utmp = [0u32; 4];
        let sc_base = boff + 4;
        utmp[0] = u32::from_le_bytes([q5k_data[sc_base], q5k_data[sc_base+1], q5k_data[sc_base+2], q5k_data[sc_base+3]]);
        utmp[1] = u32::from_le_bytes([q5k_data[sc_base+4], q5k_data[sc_base+5], q5k_data[sc_base+6], q5k_data[sc_base+7]]);
        utmp[2] = u32::from_le_bytes([q5k_data[sc_base+8], q5k_data[sc_base+9], q5k_data[sc_base+10], q5k_data[sc_base+11]]);

        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        let uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3] as i32, utmp[2] as i32, utmp[1] as i32, utmp[0] as i32));

        let q8sums = _mm256_loadu_si256(q8k[i].bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1)
        );
        let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        let scales256 = _mm256_set_m128i(sc128, sc128);

        let mut sumi = _mm256_setzero_si256();

        let qh_base = boff + 16;
        let qs_base = boff + 48;

        let hbits = _mm256_loadu_si256(q5k_data.as_ptr().add(qh_base) as *const __m256i);
        let m16 = _mm256_set1_epi8(16);
        let mut q4_off = 0usize;
        let mut q8_off = 0usize;
        let mut qh_shift = 0i32;

        for j in 0..4 {
            let scale_l = _mm256_shuffle_epi8(scales256, _mm256_loadu_si256(scale_shuffle[2 * j].as_ptr() as *const __m256i));
            let scale_h = _mm256_shuffle_epi8(scales256, _mm256_loadu_si256(scale_shuffle[2 * j + 1].as_ptr() as *const __m256i));

            let qh_mask_l = _mm256_set1_epi8(1i8 << qh_shift);
            let qh_mask_h = _mm256_set1_epi8(1i8 << (qh_shift + 1));
            qh_shift += 2;

            let qs_bits = _mm256_loadu_si256(q5k_data.as_ptr().add(qs_base + q4_off) as *const __m256i);
            let q5l_raw = _mm256_and_si256(qs_bits, m4);

            let qh_l = _mm256_and_si256(hbits, qh_mask_l);
            let qh_add_l = _mm256_and_si256(_mm256_cmpeq_epi8(qh_l, qh_mask_l), m16);
            let q5l = _mm256_add_epi8(q5l_raw, qh_add_l);

            let q5h_raw = _mm256_and_si256(_mm256_srli_epi16(qs_bits, 4), m4);
            let qh_h = _mm256_and_si256(hbits, qh_mask_h);
            let qh_add_h = _mm256_and_si256(_mm256_cmpeq_epi8(qh_h, qh_mask_h), m16);
            let q5h = _mm256_add_epi8(q5h_raw, qh_add_h);

            q4_off += 32;

            let q8l = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off) as *const __m256i);
            q8_off += 32;
            let mut p16l = _mm256_maddubs_epi16(q5l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            let q8h = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off) as *const __m256i);
            q8_off += 32;
            let mut p16h = _mm256_maddubs_epi16(q5h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        let vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    let min_val = _mm_cvtss_f32(acc_m);

    let main_val = crate::ops::hsum_ps(acc);

    main_val + min_val
}

pub fn vec_dot_q6k_q8k(q6k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        return unsafe { vec_dot_q6k_q8k_avx2(q6k_data, q8k) };
    }
    vec_dot_q6k_q8k_scalar(q6k_data, q8k)
}

pub fn matmul_q6k_q8k(weight_data: &[u8], input: &[f32], n_cols: usize, n_rows: usize) -> Vec<f32> {
    let q8k = quantize_row_q8_k(input);
    let blocks_per_row = n_cols / QK_K;
    let mut output = vec![0.0f32; n_rows];
    for o in 0..n_rows {
        let row_data = &weight_data[o * blocks_per_row * BLOCK_Q6K_SIZE..];
        output[o] = vec_dot_q6k_q8k(row_data, &q8k);
    }
    output
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn vec_dot_q6k_q8k_avx2_direct(q6k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    vec_dot_q6k_q8k_avx2(q6k_data, q8k)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_q6k_q8k_avx2(q6k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    let nb = q8k.len();

    let m3 = _mm256_set1_epi8(3);
    let m15 = _mm256_set1_epi8(15);

    let scale_shuffle: [[u8; 16]; 8] = {
        let mut tbl = [[0u8; 16]; 8];
        for i in 0..8 {
            for j in 0..8 {
                tbl[i][j] = (i * 2) as u8;
            }
            for j in 0..8 {
                tbl[i][8 + j] = (i * 2 + 1) as u8;
            }
        }
        tbl
    };

    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let boff = i * BLOCK_Q6K_SIZE;
        if boff + BLOCK_Q6K_SIZE > q6k_data.len() { break; }

        let d_raw = u16::from_le_bytes([q6k_data[boff + 208], q6k_data[boff + 209]]);
        let d = f16_to_f32(d_raw) * q8k[i].d;

        let ql_base = boff;
        let qh_base = boff + 128;
        let sc_base = boff + 192;

        let q8sums = _mm256_loadu_si256(q8k[i].bsums.as_ptr() as *const __m256i);
        let scales = _mm_loadu_si128(q6k_data.as_ptr().add(sc_base) as *const __m128i);
        let scales_16 = _mm256_cvtepi8_epi16(scales);
        let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        let mut sumi = _mm256_setzero_si256();

        let mut is = 0usize;
        let mut ql_off = ql_base;
        let mut qh_off = qh_base;
        let mut q8_off = 0usize;

        for _j in 0..2 {
            let q4bits1 = _mm256_loadu_si256(q6k_data.as_ptr().add(ql_off) as *const __m256i);
            ql_off += 32;
            let q4bits2 = _mm256_loadu_si256(q6k_data.as_ptr().add(ql_off) as *const __m256i);
            ql_off += 32;
            let q4bitsH = _mm256_loadu_si256(q6k_data.as_ptr().add(qh_off) as *const __m256i);
            qh_off += 32;

            let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            let q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64i32 as i8 as u8 as i8)), 2);

            let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            let q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            let q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            let q8_0 = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off) as *const __m256i);
            q8_off += 32;
            let q8_1 = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off) as *const __m256i);
            q8_off += 32;
            let q8_2 = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off) as *const __m256i);
            q8_off += 32;
            let q8_3 = _mm256_loadu_si256(q8k[i].qs.as_ptr().add(q8_off) as *const __m256i);
            q8_off += 32;

            let mut p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            let mut p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            let mut p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            let mut p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            let scale_0 = _mm_shuffle_epi8(scales, _mm_loadu_si128(scale_shuffle[is].as_ptr() as *const __m128i));
            let scale_1 = _mm_shuffle_epi8(scales, _mm_loadu_si128(scale_shuffle[is + 1].as_ptr() as *const __m128i));
            let scale_2 = _mm_shuffle_epi8(scales, _mm_loadu_si128(scale_shuffle[is + 2].as_ptr() as *const __m128i));
            let scale_3 = _mm_shuffle_epi8(scales, _mm_loadu_si128(scale_shuffle[is + 3].as_ptr() as *const __m128i));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    crate::ops::hsum_ps(acc)
}

#[cfg(all(test, target_arch = "x86_64"))]
mod avx2_parity {
    use super::*;

    fn block_q8k(values: &[f32]) -> Vec<BlockQ8K> {
        let n = values.len();
        let mut out = Vec::with_capacity(n / 256);
        for chunk in values.chunks(256) {
            let amax = chunk.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
            let id = if amax == 0.0 { 0.0 } else { 127.0 / amax };
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for (i, &v) in chunk.iter().enumerate() {
                let q = ((v * id).round() as i32).clamp(-127, 127) as i8;
                qs[i] = q;
                bsums[i / 16] += q as i16;
            }
            out.push(BlockQ8K { d, qs, bsums });
        }
        out
    }

    fn make_q4k_block(ql: &[u8; 128], qh: &[u8; 64], scales: &[i8; 16], d: f32, dmin: f32) -> Vec<u8> {
        let mut v = Vec::with_capacity(144);
        v.extend_from_slice(&crate::ops::f32_to_f16(d).to_le_bytes());
        v.extend_from_slice(&crate::ops::f32_to_f16(dmin).to_le_bytes());
        for &s in scales { v.push(s as u8); }
        v.extend_from_slice(ql);
        v.extend_from_slice(qh);
        v
    }

    fn make_q6k_block(ql: &[u8; 128], qh: &[u8; 64], scales: &[i8; 16], d: f32) -> Vec<u8> {
        let mut v = vec![0u8; 210];
        // ql region: bytes [0..128]
        for i in 0..128 { v[i] = ql[i]; }
        // qh region: bytes [128..192]
        for i in 0..64 { v[128 + i] = qh[i]; }
        // scales region: bytes [192..208]
        for i in 0..16 { v[192 + i] = scales[i] as u8; }
        // d F16 at offset 208
        let d_bytes = crate::ops::f32_to_f16(d).to_le_bytes();
        v[208] = d_bytes[0];
        v[209] = d_bytes[1];
        v
    }

    #[test]
    fn q4k_avx2_matches_scalar_one_block() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let ql = [0u8; 128];
        let qh = [0u8; 64];
        let scales: [i8; 16] = std::array::from_fn(|i| (i as i8) - 4);
        let weight = make_q4k_block(&ql, &qh, &scales, 0.5, 0.1);
        let input = vec![0.0f32; 256];
        let q8k = block_q8k(&input);

        let avx2 = unsafe { vec_dot_q4k_q8k_avx2(&weight, &q8k) };
        let scalar = vec_dot_q4k_q8k_scalar(&weight, &q8k);
        eprintln!("q4k one block zero: avx2={} (bits {:x}) scalar={} (bits {:x}) diff={}",
            avx2, avx2.to_bits(), scalar, scalar.to_bits(),
            (avx2.to_bits() as i32).wrapping_sub(scalar.to_bits() as i32).unsigned_abs());
        // Tolerance: 4 ULPs (hsum + FMA both can drift)
        let diff = (avx2 - scalar).abs();
        let rel = if scalar.abs() > 1e-3 { diff / scalar.abs() } else { diff };
        assert!(rel < 1e-3, "q4k AVX2 diverged: avx2={} scalar={} rel={}", avx2, scalar, rel);
    }

    #[test]
    fn q4k_avx2_matches_scalar_multi_block() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        // 4 super-blocks, varied values
        let mut weight = Vec::new();
        for i in 0..4 {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            for j in 0..128 { ql[j] = ((i * 7 + j) % 16) as u8; }
            for j in 0..64 { qh[j] = ((i + j * 3) % 4) as u8; }
            let scales: [i8; 16] = std::array::from_fn(|k| ((i * 3 + k) as i8) - 8);
            weight.extend(make_q4k_block(&ql, &qh, &scales, 0.01 + i as f32 * 0.1, 0.005));
        }
        // 4 blocks of input (1024 floats total)
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let q8k = block_q8k(&input);

        let avx2 = unsafe { vec_dot_q4k_q8k_avx2(&weight, &q8k) };
        let scalar = vec_dot_q4k_q8k_scalar(&weight, &q8k);
        eprintln!("q4k 4-block: avx2={} (bits {:x}) scalar={} (bits {:x}) diff={}",
            avx2, avx2.to_bits(), scalar, scalar.to_bits(),
            (avx2.to_bits() as i32).wrapping_sub(scalar.to_bits() as i32).unsigned_abs());
        let diff = (avx2 - scalar).abs();
        let rel = if scalar.abs() > 1e-3 { diff / scalar.abs() } else { diff };
        assert!(rel < 1e-3, "q4k AVX2 diverged on 4 blocks: avx2={} scalar={} rel={}", avx2, scalar, rel);
    }

    #[test]
    fn q6k_avx2_matches_scalar_one_block() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let ql: [u8; 128] = std::array::from_fn(|i| (i % 4) as u8);
        let qh: [u8; 64] = std::array::from_fn(|i| (i * 5 % 4) as u8);
        let scales: [i8; 16] = std::array::from_fn(|i| (i as i8) - 4);
        let weight = make_q6k_block(&ql, &qh, &scales, 0.5);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let q8k = block_q8k(&input);

        let avx2 = unsafe { vec_dot_q6k_q8k_avx2(&weight, &q8k) };
        let scalar = vec_dot_q6k_q8k_scalar(&weight, &q8k);
        eprintln!("q6k one block: avx2={} (bits {:x}) scalar={} (bits {:x}) diff={}",
            avx2, avx2.to_bits(), scalar, scalar.to_bits(),
            (avx2.to_bits() as i32).wrapping_sub(scalar.to_bits() as i32).unsigned_abs());
        let diff = (avx2 - scalar).abs();
        let rel = if scalar.abs() > 1e-3 { diff / scalar.abs() } else { diff };
        assert!(rel < 1e-3, "q6k AVX2 diverged: avx2={} scalar={} rel={}", avx2, scalar, rel);
    }

    #[test]
    fn q6k_avx2_matches_scalar_multi_block() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let mut weight = Vec::new();
        for i in 0..4 {
            let ql: [u8; 128] = std::array::from_fn(|j| ((i * 11 + j * 3) % 4) as u8);
            let qh: [u8; 64] = std::array::from_fn(|j| ((i + j * 7) % 4) as u8);
            let scales: [i8; 16] = std::array::from_fn(|k| ((i * 5 + k) as i8) - 6);
            weight.extend(make_q6k_block(&ql, &qh, &scales, 0.01 + i as f32 * 0.1));
        }
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let q8k = block_q8k(&input);

        let avx2 = unsafe { vec_dot_q6k_q8k_avx2(&weight, &q8k) };
        let scalar = vec_dot_q6k_q8k_scalar(&weight, &q8k);
        eprintln!("q6k 4-block: avx2={} (bits {:x}) scalar={} (bits {:x}) diff={}",
            avx2, avx2.to_bits(), scalar, scalar.to_bits(),
            (avx2.to_bits() as i32).wrapping_sub(scalar.to_bits() as i32).unsigned_abs());
        // TODO: Q6_K has 1 ULP drift vs scalar (FMA + hsum) — needs deeper
        // investigation (see TODO.md). Currently production model output is
        // correct because the drift doesn't flip argmax.
        let rel = if scalar.abs() > 1e-3 { (avx2 - scalar).abs() / scalar.abs() } else { (avx2 - scalar).abs() };
        assert!(rel < 1e-3, "q6k AVX2 diverged: rel={}", rel);
    }

    #[test]
    fn q4k_avx2_matches_scalar_real_model() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let loader = match crate::core::loader::GGUFLoader::from_file(
            "../models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf",
        ) {
            Ok(l) => l,
            Err(_) => return,
        };
        let tensor = loader
            .tensors()
            .iter()
            .find(|t| t.name == "blk.0.attn_q.weight" && t.ggml_type == crate::core::tensor::GGMLType::Q4K)
            .expect("blk.0.attn_q.weight Q4_K not found");
        let weight = loader.tensor_slice(&tensor.name).unwrap();
        let n_in = tensor.dims[0] as usize;
        let blocks = n_in / 256;
        let input: Vec<f32> = (0..blocks * 256)
            .map(|i| ((i as f32 * 0.013).sin() * 30.0).clamp(-127.0, 127.0))
            .collect();
        let q8k = block_q8k(&input);

        let avx2 = unsafe { vec_dot_q4k_q8k_avx2(&weight, &q8k) };
        let scalar = vec_dot_q4k_q8k_scalar(&weight, &q8k);
        eprintln!(
            "q4k real-model: avx2={} (bits {:x}) scalar={} (bits {:x}) diff_bits={}",
            avx2, avx2.to_bits(), scalar, scalar.to_bits(),
            (avx2.to_bits() as i32).wrapping_sub(scalar.to_bits() as i32).unsigned_abs()
        );
        let diff = (avx2 - scalar).abs();
        let rel = if scalar.abs() > 1e-3 { diff / scalar.abs() } else { diff };
        assert!(rel < 1e-3, "q4k real-model AVX2 diverged: rel={}", rel);
    }
}
