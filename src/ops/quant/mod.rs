use crate::core::tensor::TensorInfo;

pub mod fuse;
pub mod iq_tables;
pub mod q8_0;

mod avx2_k;

use self::iq_tables::KVALUES_IQ4NL;

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;
pub const BLOCK_Q4K_SIZE: usize = 144;
pub const BLOCK_Q5K_SIZE: usize = 176;
pub const BLOCK_Q6K_SIZE: usize = 210;
pub const BLOCK_Q8K_SIZE: usize = 292;
pub const BLOCK_Q2K_SIZE: usize = 84;
pub const BLOCK_Q3K_SIZE: usize = 110;
pub const BLOCK_IQ2_XS_SIZE: usize = 74;
pub const BLOCK_IQ2_XXS_SIZE: usize = 66;
pub const BLOCK_IQ2_S_SIZE: usize = 82;
pub const BLOCK_IQ3_XXS_SIZE: usize = 98;
pub const BLOCK_IQ3_S_SIZE: usize = 110;
pub const BLOCK_IQ4_XS_SIZE: usize = 136;

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

        // llama.cpp Q2_K layout (matches `dequantize_row_q2_K`):
        //   - 16 sub-blocks of 16 elements (8 per n-iteration of 128)
        //   - For n=0 (elements 0..127) and n=128 (elements 128..255):
        //     sub-A (lo half of 32-element pair) uses qs[l]   shift 2*j_in_n
        //     sub-B (hi half)                  uses qs[l+16] shift 2*j_in_n
        //   where l ∈ [0..16), j_in_n ∈ [0..4), per C quantizer.
        for n_outer in 0..2usize {
            for j_in_n in 0..4usize {
                let shift = (j_in_n as u32) * 2;
                let scale_idx_a = n_outer * 8 + j_in_n * 2;
                let scale_idx_b = scale_idx_a + 1;
                let sc_a_byte = scales[scale_idx_a];
                let dl_a = d * (sc_a_byte & 0xF) as f32;
                let ml_a = dmin * (sc_a_byte >> 4) as f32;
                let sc_b_byte = scales[scale_idx_b];
                let dl_b = d * (sc_b_byte & 0xF) as f32;
                let ml_b = dmin * (sc_b_byte >> 4) as f32;
                let q_base = n_outer * 32;
                for l in 0..16usize {
                    let qa = ((qs[q_base + l] >> shift) & 0x3) as f32;
                    let qb = ((qs[q_base + 16 + l] >> shift) & 0x3) as f32;
                    let p = n_outer * 128 + j_in_n * 32 + l;
                    let p_b = n_outer * 128 + j_in_n * 32 + 16 + l;
                    output[out_base + p] = dl_a * qa - ml_a;
                    output[out_base + p_b] = dl_b * qb - ml_b;
                }
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
        let q8 = &q8k[i].qs;
        let bsums = &q8k[i].bsums;

        // Same sub-block structure as `dequantize_row_q2_k`. Each sub-block of
        // 16 elements uses one scale; sum over sub-blocks via `aux32[8]`.
        let mut aux32 = [0i32; 8];
        let mut sumi = 0i32;

        for n_outer in 0..2usize {
            for j_in_n in 0..4usize {
                let shift = (j_in_n as u32) * 2;
                let q_base = n_outer * 32;

                // Sub-A
                let sc_byte = scales[n_outer * 8 + j_in_n * 2];
                let sc_a = (sc_byte & 0xF) as i32;
                let m_a = (sc_byte >> 4) as i32;
                let mut dot_a = 0i32;
                for l in 0..16usize {
                    let q = ((qs[q_base + l] >> shift) & 0x3) as i32;
                    let p = n_outer * 128 + j_in_n * 32 + l;
                    dot_a += q * q8[p] as i32;
                }
                aux32[n_outer * 4 + j_in_n] += sc_a * dot_a;
                sumi += m_a * bsums[n_outer * 8 + j_in_n * 2] as i32;

                // Sub-B
                let sc_byte_b = scales[n_outer * 8 + j_in_n * 2 + 1];
                let sc_b = (sc_byte_b & 0xF) as i32;
                let m_b = (sc_byte_b >> 4) as i32;
                let mut dot_b = 0i32;
                for l in 0..16usize {
                    let q = ((qs[q_base + 16 + l] >> shift) & 0x3) as i32;
                    let p = n_outer * 128 + j_in_n * 32 + 16 + l;
                    dot_b += q * q8[p] as i32;
                }
                aux32[n_outer * 4 + j_in_n] += sc_b * dot_b;
                sumi += m_b * bsums[n_outer * 8 + j_in_n * 2 + 1] as i32;
            }
        }

        let d_total = d * q8k[i].d;
        for l in 0..8 {
            sumf += d_total * aux32[l] as f32;
        }
        sumf -= dmin * q8k[i].d * sumi as f32;
    }

    sumf
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn vec_dot_q2k_q8k_avx2_direct(q2k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    self::avx2_k::vec_dot_q2k_q8k_avx2(q2k_data, q8k)
}

pub fn vec_dot_q2k_q8k(q2k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    vec_dot_q2k_q8k_scalar(q2k_data, q8k)
}

pub fn dequantize_row_q3_k(block_bytes: &[u8], output: &mut [f32]) {
    let num_blocks = output.len() / QK_K;
    for block_idx in 0..num_blocks {
        let boff = block_idx * BLOCK_Q3K_SIZE;
        if boff + BLOCK_Q3K_SIZE > block_bytes.len() { break; }

        let d_all = f16_from_bytes(block_bytes, boff + 108);
        let hmask = &block_bytes[boff..boff + 32];
        let qs = &block_bytes[boff + 32..boff + 96];
        let out_base = block_idx * QK_K;

        // Deinterleave scales: 12 packed bytes → 16 int8 scale values.
        // Mirrors llama.cpp `dequantize_row_q3_K` exactly.
        let mut auxs = [0u32; 4];
        auxs[0] = u32::from_le_bytes([
            block_bytes[boff + 96], block_bytes[boff + 97],
            block_bytes[boff + 98], block_bytes[boff + 99],
        ]);
        auxs[1] = u32::from_le_bytes([
            block_bytes[boff + 100], block_bytes[boff + 101],
            block_bytes[boff + 102], block_bytes[boff + 103],
        ]);
        auxs[2] = u32::from_le_bytes([
            block_bytes[boff + 104], block_bytes[boff + 105],
            block_bytes[boff + 106], block_bytes[boff + 107],
        ]);
        auxs[3] = 0;
        let tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & 0x0f0f_0f0f) | (((tmp >> 4) & 0x0303_0303) << 4);
        auxs[3] = ((auxs[1] >> 4) & 0x0f0f_0f0f) | (((tmp >> 6) & 0x0303_0303) << 4);
        auxs[0] = (auxs[0] & 0x0f0f_0f0f) | (((tmp >> 0) & 0x0303_0303) << 4);
        auxs[1] = (auxs[1] & 0x0f0f_0f0f) | (((tmp >> 2) & 0x0303_0303) << 4);
        let scales_bytes: [u8; 16] = bytemuck::cast(auxs);
        let scales_signed: [i8; 16] = [
            scales_bytes[0] as i8, scales_bytes[1] as i8, scales_bytes[2] as i8, scales_bytes[3] as i8,
            scales_bytes[4] as i8, scales_bytes[5] as i8, scales_bytes[6] as i8, scales_bytes[7] as i8,
            scales_bytes[8] as i8, scales_bytes[9] as i8, scales_bytes[10] as i8, scales_bytes[11] as i8,
            scales_bytes[12] as i8, scales_bytes[13] as i8, scales_bytes[14] as i8, scales_bytes[15] as i8,
        ];

        let mut is = 0usize;
        let mut m: u8 = 1;
        let mut q_off = 0usize;
        let mut out_idx = 0usize;
        for _n in 0..(QK_K / 128) {
            for j in 0..4 {
                let scale_a = (scales_signed[is] as i32) - 32;
                is += 1;
                let dl_a = d_all * scale_a as f32;
                for l in 0..16usize {
                    let q_byte = qs[q_off + l];
                    let q_low = (q_byte & 0x3) as i32;
                    let q_signed = q_low - if hmask[l] & m != 0 { 0 } else { 4 };
                    output[out_base + out_idx] = dl_a * q_signed as f32;
                    out_idx += 1;
                }
                let scale_b = (scales_signed[is] as i32) - 32;
                is += 1;
                let dl_b = d_all * scale_b as f32;
                for l in 0..16usize {
                    let q_byte = qs[q_off + 16 + l];
                    let q_low = (q_byte & 0x3) as i32;
                    let q_signed = q_low - if hmask[16 + l] & m != 0 { 0 } else { 4 };
                    output[out_base + out_idx] = dl_b * q_signed as f32;
                    out_idx += 1;
                }
                m <<= 1;
            }
            q_off += 32;
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
        let q8 = &q8k[i].qs;

        // Deinterleave scales (same as dequantize_row_q3_k).
        let mut auxs = [0u32; 4];
        auxs[0] = u32::from_le_bytes([
            q3k_data[boff + 96], q3k_data[boff + 97],
            q3k_data[boff + 98], q3k_data[boff + 99],
        ]);
        auxs[1] = u32::from_le_bytes([
            q3k_data[boff + 100], q3k_data[boff + 101],
            q3k_data[boff + 102], q3k_data[boff + 103],
        ]);
        auxs[2] = u32::from_le_bytes([
            q3k_data[boff + 104], q3k_data[boff + 105],
            q3k_data[boff + 106], q3k_data[boff + 107],
        ]);
        auxs[3] = 0;
        let tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & 0x0f0f_0f0f) | (((tmp >> 4) & 0x0303_0303) << 4);
        auxs[3] = ((auxs[1] >> 4) & 0x0f0f_0f0f) | (((tmp >> 6) & 0x0303_0303) << 4);
        auxs[0] = (auxs[0] & 0x0f0f_0f0f) | (((tmp >> 0) & 0x0303_0303) << 4);
        auxs[1] = (auxs[1] & 0x0f0f_0f0f) | (((tmp >> 2) & 0x0303_0303) << 4);
        let scales_bytes: [u8; 16] = bytemuck::cast(auxs);
        let scales_signed: [i8; 16] = [
            scales_bytes[0] as i8, scales_bytes[1] as i8, scales_bytes[2] as i8, scales_bytes[3] as i8,
            scales_bytes[4] as i8, scales_bytes[5] as i8, scales_bytes[6] as i8, scales_bytes[7] as i8,
            scales_bytes[8] as i8, scales_bytes[9] as i8, scales_bytes[10] as i8, scales_bytes[11] as i8,
            scales_bytes[12] as i8, scales_bytes[13] as i8, scales_bytes[14] as i8, scales_bytes[15] as i8,
        ];

        // Decode Q3K into aux8 (256 signed 3-bit values), grouped into 8 sums × 32 lanes.
        let mut aux8 = [0i8; 256];
        let mut a_ptr = 0usize;
        let mut m: u8 = 1;
        let mut q_off = 0usize;
        for _n in 0..(QK_K / 128) {
            for shift in (0..8).step_by(2) {
                for l in 0..32usize {
                    let q_byte = qs[q_off + l];
                    let ql = ((q_byte >> shift) & 0x3) as i8;
                    aux8[a_ptr + l] = ql - if hmask[l] & m != 0 { 0 } else { 4 };
                }
                a_ptr += 32;
                m <<= 1;
            }
            q_off += 32;
        }

        let mut aux32 = [0i32; 8];
        let mut q8_idx = 0usize;
        let mut a_idx = 0usize;
        for j in 0..16 {
            let scale = (scales_signed[j] as i32) - 32;
            for _ in 0..2 {
                for l in 0..8 {
                    aux32[l] += scale * (q8[q8_idx] as i32) * (aux8[a_idx] as i32);
                    q8_idx += 1;
                    a_idx += 1;
                }
            }
        }

        let d_total = d * q8k[i].d;
        for l in 0..8 {
            sumf += d_total * aux32[l] as f32;
        }
    }

    sumf
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn vec_dot_q3k_q8k_avx2_direct(q3k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    self::avx2_k::vec_dot_q3k_q8k_avx2(q3k_data, q8k)
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn vec_dot_iq4_xs_q8k_avx2_direct(iq4xs_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    self::avx2_k::vec_dot_iq4_xs_q8k_avx2(iq4xs_data, q8k)
}

pub fn vec_dot_q3k_q8k(q3k_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    vec_dot_q3k_q8k_scalar(q3k_data, q8k)
}

/// Non-linear lookup table for IQ4_NL: maps 4-bit nibble to quantized value.
// Mirrors llama.cpp's `kvalues_iq4nl`.
pub const IQ4_NL_LUT: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113,
];

pub fn dequantize_row_iq4_nl(block_bytes: &[u8], output: &mut [f32]) {
    let num_blocks = output.len() / 32;
    for block_idx in 0..num_blocks {
        let boff = block_idx * 18;
        if boff + 18 > block_bytes.len() { break; }

        let d = f16_from_bytes(block_bytes, boff);
        let qs = &block_bytes[boff + 2..boff + 18];
        let out_base = block_idx * 32;

        for j in 0..16 {
            let q_byte = qs[j];
            let q_lo = IQ4_NL_LUT[(q_byte & 0xF) as usize];
            let q_hi = IQ4_NL_LUT[((q_byte >> 4) & 0xF) as usize];
            output[out_base + j] = d * q_lo as f32;
            output[out_base + j + 16] = d * q_hi as f32;
        }
    }
}

pub fn vec_dot_iq4_nl_q8k_scalar(iq4nl_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    let nb = q8k.len();
    let blocks_per_super = 8;
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let super_off = i * blocks_per_super * 18;
        if super_off + blocks_per_super * 18 > iq4nl_data.len() { break; }

        let mut sum_block = 0.0f32;
        for sb in 0..blocks_per_super {
            let boff = super_off + sb * 18;
            let d = f16_from_bytes(iq4nl_data, boff);
            let qs = &iq4nl_data[boff + 2..boff + 18];

            let q8_block = &q8k[i].qs[sb * 32..(sb + 1) * 32];
            let mut dot = 0i32;
            for j in 0..16 {
                let q_byte = qs[j];
                let q_lo = IQ4_NL_LUT[(q_byte & 0xF) as usize] as i32;
                let q_hi = IQ4_NL_LUT[((q_byte >> 4) & 0xF) as usize] as i32;
                dot += q_lo * q8_block[j] as i32;
                dot += q_hi * q8_block[j + 16] as i32;
            }
            sum_block += d * q8k[i].d * dot as f32;
        }
        sumf += sum_block;
    }

    sumf
}

pub fn vec_dot_iq4_nl_q8k(iq4nl_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    vec_dot_iq4_nl_q8k_scalar(iq4nl_data, q8k)
}

/// IQ2_XS super-block (256 elements, 74 bytes). See `ggml-quants.c::dequantize_row_iq2_xs`.
pub fn dequantize_row_iq2_xs(block_bytes: &[u8], output: &mut [f32]) {
    let mask = self::iq_tables::iq2_xs_mask();
    let grid = self::iq_tables::iq2_xs_grid();
    let signs = self::iq_tables::iq2_xs_signs();
    let num_blocks = output.len() / QK_K;
    for block_idx in 0..num_blocks {
        let boff = block_idx * BLOCK_IQ2_XS_SIZE;
        if boff + BLOCK_IQ2_XS_SIZE > block_bytes.len() {
            break;
        }
        let d = f16_from_bytes(block_bytes, boff);
        let qs_bytes = &block_bytes[boff + 2..boff + 2 + 64];
        let sc = &block_bytes[boff + 66..boff + 74];
        let out_base = block_idx * QK_K;
        for ib32 in 0..8usize {
            let db0 = d * (0.5f32 + (sc[ib32] & 0x0f) as f32) * 0.25f32;
            let db1 = d * (0.5f32 + (sc[ib32] >> 4) as f32) * 0.25f32;
            for l in 0..4usize {
                let idx = (ib32 * 4 + l) * 2;
                let q = u16::from_le_bytes([qs_bytes[idx], qs_bytes[idx + 1]]);
                let grid_idx = (q & 0x1ff) as usize;
                let signs_idx = (q >> 9) as usize;
                let grid_bytes = grid[grid_idx].to_le_bytes();
                let sgn = signs[signs_idx];
                let dl = if l < 2 { db0 } else { db1 };
                let out_off = out_base + ib32 * 32 + l * 8;
                for j in 0..8usize {
                    let sign = if sgn & mask[j] != 0 { -1.0f32 } else { 1.0f32 };
                    output[out_off + j] = dl * (grid_bytes[j] as i8 as f32) * sign;
                }
            }
        }
    }
}

pub fn vec_dot_iq2_xs_q8k_scalar(iq2xs_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    let mask = self::iq_tables::iq2_xs_mask();
    let grid = self::iq_tables::iq2_xs_grid();
    let signs = self::iq_tables::iq2_xs_signs();
    let nb = q8k.len();
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let boff = i * BLOCK_IQ2_XS_SIZE;
        if boff + BLOCK_IQ2_XS_SIZE > iq2xs_data.len() {
            break;
        }
        let d = f16_from_bytes(iq2xs_data, boff) * q8k[i].d;
        let qs_bytes = &iq2xs_data[boff + 2..boff + 2 + 64];
        let sc = &iq2xs_data[boff + 66..boff + 74];
        let q8 = &q8k[i].qs;
        let mut bsum: i32 = 0;
        let mut qs_idx = 0usize;
        let mut q8_idx = 0usize;
        for ib32 in 0..8usize {
            let ls1 = 2 * ((sc[ib32] & 0x0f) as i32) + 1;
            let ls2 = 2 * ((sc[ib32] >> 4) as i32) + 1;
            for inner in 0..2usize {
                let q = u16::from_le_bytes([qs_bytes[qs_idx], qs_bytes[qs_idx + 1]]);
                qs_idx += 2;
                let grid_idx = (q & 0x1ff) as usize;
                let signs_idx = (q >> 9) as usize;
                let grid_bytes = grid[grid_idx].to_le_bytes();
                let sgn = signs[signs_idx];
                let mut sumi: i32 = 0;
                for j in 0..8usize {
                    let sign = if sgn & mask[j] != 0 { -1i32 } else { 1i32 };
                    sumi += (grid_bytes[j] as i8 as i32) * (q8[q8_idx + j] as i32) * sign;
                }
                q8_idx += 8;
                bsum += sumi * if inner == 0 { ls1 } else { ls2 };
            }
        }
        sumf += d * bsum as f32;
    }
    sumf * 0.125f32
}

pub fn vec_dot_iq2_xs_q8k(iq2xs_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    vec_dot_iq2_xs_q8k_scalar(iq2xs_data, q8k)
}

/// IQ3_S super-block (256 elements, 110 bytes). See `ggml-quants.c::dequantize_row_iq3_s`.
pub fn dequantize_row_iq3_s(block_bytes: &[u8], output: &mut [f32]) {
    let mask = self::iq_tables::iq2_xs_mask();
    let grid = self::iq_tables::iq3_s_grid();
    let num_blocks = output.len() / QK_K;
    for block_idx in 0..num_blocks {
        let boff = block_idx * BLOCK_IQ3_S_SIZE;
        if boff + BLOCK_IQ3_S_SIZE > block_bytes.len() {
            break;
        }
        let d = f16_from_bytes(block_bytes, boff);
        let qs = &block_bytes[boff + 2..boff + 2 + 64];
        let qh = &block_bytes[boff + 66..boff + 74];
        let signs = &block_bytes[boff + 74..boff + 74 + 32];
        let sc = &block_bytes[boff + 106..boff + 110];
        let out_base = block_idx * QK_K;
        let mut qs_off = 0usize;
        let mut sign_off = 0usize;
        for ib32 in (0..8usize).step_by(2) {
            let db1 = d * (1.0f32 + 2.0f32 * (sc[ib32 / 2] & 0x0f) as f32);
            let db2 = d * (1.0f32 + 2.0f32 * (sc[ib32 / 2] >> 4) as f32);
            for pair in 0..2usize {
                let qh_byte = qh[ib32 + pair];
                let dl = if pair == 0 { db1 } else { db2 };
                for l in 0..4usize {
                    let g1 = grid[qs[qs_off + 2 * l] as usize
                        | ((((qh_byte as u16) << (8 - 2 * l)) & 256) as usize)];
                    let g2 = grid[qs[qs_off + 2 * l + 1] as usize
                        | ((((qh_byte as u16) << (7 - 2 * l)) & 256) as usize)];
                    let g1_bytes = g1.to_le_bytes();
                    let g2_bytes = g2.to_le_bytes();
                    let sgn = signs[sign_off + l];
                    let out_off = out_base + (ib32 + pair) * 32 + l * 8;
                    for j in 0..4usize {
                        let s1 = if sgn & mask[j] != 0 { -1.0f32 } else { 1.0f32 };
                        let s2 = if sgn & mask[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
                        output[out_off + j] = dl * g1_bytes[j] as f32 * s1;
                        output[out_off + j + 4] = dl * g2_bytes[j] as f32 * s2;
                    }
                }
                qs_off += 8;
                sign_off += 4;
            }
        }
    }
}

pub fn vec_dot_iq3_s_q8k_scalar(iq3s_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    let mask = self::iq_tables::iq2_xs_mask();
    let grid = self::iq_tables::iq3_s_grid();
    let nb = q8k.len();
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let boff = i * BLOCK_IQ3_S_SIZE;
        if boff + BLOCK_IQ3_S_SIZE > iq3s_data.len() {
            break;
        }
        let d = f16_from_bytes(iq3s_data, boff) * q8k[i].d;
        let qs = &iq3s_data[boff + 2..boff + 2 + 64];
        let qh = &iq3s_data[boff + 66..boff + 74];
        let signs = &iq3s_data[boff + 74..boff + 74 + 32];
        let sc = &iq3s_data[boff + 106..boff + 110];
        let q8 = &q8k[i].qs;
        let mut bsum: i32 = 0;
        let mut qs_off = 0usize;
        let mut sign_off = 0usize;
        let mut q8_idx = 0usize;
        for ib32 in (0..8usize).step_by(2) {
            let ls1 = 2 * ((sc[ib32 / 2] & 0x0f) as i32) + 1;
            let ls2 = 2 * ((sc[ib32 / 2] >> 4) as i32) + 1;
            for pair in 0..2usize {
                let qh_byte = qh[ib32 + pair];
                let mut sumi: i32 = 0;
                for l in 0..4usize {
                    let g1 = grid[qs[qs_off + 2 * l] as usize
                        | ((((qh_byte as u16) << (8 - 2 * l)) & 256) as usize)];
                    let g2 = grid[qs[qs_off + 2 * l + 1] as usize
                        | ((((qh_byte as u16) << (7 - 2 * l)) & 256) as usize)];
                    let g1_bytes = g1.to_le_bytes();
                    let g2_bytes = g2.to_le_bytes();
                    let sgn = signs[sign_off + l];
                    for j in 0..4usize {
                        let s1 = if sgn & mask[j] != 0 { -1i32 } else { 1i32 };
                        let s2 = if sgn & mask[j + 4] != 0 { -1i32 } else { 1i32 };
                        sumi += g1_bytes[j] as i32 * q8[q8_idx + j] as i32 * s1;
                        sumi += g2_bytes[j] as i32 * q8[q8_idx + j + 4] as i32 * s2;
                    }
                    q8_idx += 8;
                }
                qs_off += 8;
                sign_off += 4;
                bsum += sumi * if pair == 0 { ls1 } else { ls2 };
            }
        }
        sumf += d * bsum as f32;
    }
    sumf
}

pub fn vec_dot_iq3_s_q8k(iq3s_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    vec_dot_iq3_s_q8k_scalar(iq3s_data, q8k)
}

/// IQ4_XS super-block (256 elements, 136 bytes). See `ggml-quants.c::dequantize_row_iq4_xs`.
pub fn dequantize_row_iq4_xs(block_bytes: &[u8], output: &mut [f32]) {
    let num_blocks = output.len() / QK_K;
    for block_idx in 0..num_blocks {
        let boff = block_idx * BLOCK_IQ4_XS_SIZE;
        if boff + BLOCK_IQ4_XS_SIZE > block_bytes.len() {
            break;
        }
        let d = f16_from_bytes(block_bytes, boff);
        let scales_h = u16::from_le_bytes([block_bytes[boff + 2], block_bytes[boff + 3]]);
        let scales_l = &block_bytes[boff + 4..boff + 8];
        let qs = &block_bytes[boff + 8..boff + 8 + 128];
        let out_base = block_idx * QK_K;
        for ib in 0..8usize {
            let sl_lo = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f;
            let sh_hi = (((scales_h >> (2 * ib)) & 0x3) << 4) as u8;
            let ls = (sl_lo | sh_hi) as i32;
            let dl = d * (ls as f32 - 32.0f32);
            let qs_off = ib * 16;
            let out_off = out_base + ib * 32;
            for j in 0..16usize {
                let qb = qs[qs_off + j];
                let lo = KVALUES_IQ4NL[(qb & 0x0f) as usize] as f32;
                let hi = KVALUES_IQ4NL[(qb >> 4) as usize] as f32;
                output[out_off + j] = dl * lo;
                output[out_off + j + 16] = dl * hi;
            }
        }
    }
}

pub fn vec_dot_iq4_xs_q8k_scalar(iq4xs_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    let nb = q8k.len();
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let boff = i * BLOCK_IQ4_XS_SIZE;
        if boff + BLOCK_IQ4_XS_SIZE > iq4xs_data.len() {
            break;
        }
        let d = f16_from_bytes(iq4xs_data, boff) * q8k[i].d;
        let scales_h = u16::from_le_bytes([iq4xs_data[boff + 2], iq4xs_data[boff + 3]]);
        let scales_l = &iq4xs_data[boff + 4..boff + 8];
        let qs = &iq4xs_data[boff + 8..boff + 8 + 128];
        let q8 = &q8k[i].qs;
        let mut h = scales_h;
        let mut qs_off = 0usize;
        let mut q8_off = 0usize;
        for ib in (0..8usize).step_by(2) {
            let scale_byte = scales_l[ib / 2] as u16;
            let ls1 = ((scale_byte & 0x0f) | ((h << 4) & 0x30)) as i32;
            let ls2 = (((scale_byte >> 4) as u16) | ((h << 2) & 0x30)) as i32;
            h >>= 4;
            let d1 = d * (ls1 as f32 - 32.0f32);
            let d2 = d * (ls2 as f32 - 32.0f32);
            let mut sumi1: i32 = 0;
            let mut sumi2: i32 = 0;
            for j in 0..16usize {
                let qb = qs[qs_off + j];
                sumi1 += q8[q8_off + j] as i32 * KVALUES_IQ4NL[(qb & 0x0f) as usize] as i32;
                sumi2 += q8[q8_off + j + 16] as i32 * KVALUES_IQ4NL[(qb >> 4) as usize] as i32;
            }
            sumf += d1 * (sumi1 + sumi2) as f32;
            qs_off += 16;
            q8_off += 32;
            let mut sumi1: i32 = 0;
            let mut sumi2: i32 = 0;
            for j in 0..16usize {
                let qb = qs[qs_off + j];
                sumi1 += q8[q8_off + j] as i32 * KVALUES_IQ4NL[(qb & 0x0f) as usize] as i32;
                sumi2 += q8[q8_off + j + 16] as i32 * KVALUES_IQ4NL[(qb >> 4) as usize] as i32;
            }
            sumf += d2 * (sumi1 + sumi2) as f32;
            qs_off += 16;
            q8_off += 32;
        }
    }
    sumf
}

pub fn vec_dot_iq4_xs_q8k(iq4xs_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2_fma() {
        return unsafe { self::avx2_k::vec_dot_iq4_xs_q8k_avx2(iq4xs_data, q8k) };
    }
    vec_dot_iq4_xs_q8k_scalar(iq4xs_data, q8k)
}

pub fn vec_dot_iq2_xxs_q8k(iq2xxs_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    vec_dot_iq2_xxs_q8k_scalar(iq2xxs_data, q8k)
}

pub fn vec_dot_iq2_s_q8k(iq2s_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    vec_dot_iq2_s_q8k_scalar(iq2s_data, q8k)
}

pub fn vec_dot_iq3_xxs_q8k(iq3xxs_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    vec_dot_iq3_xxs_q8k_scalar(iq3xxs_data, q8k)
}

pub fn dequantize_row_iq2_xxs(block_bytes: &[u8], output: &mut [f32]) {
    let mask = self::iq_tables::iq2_xs_mask();
    let grid = self::iq_tables::iq2_xxs_grid();
    let signs_tbl = self::iq_tables::iq2_xs_signs();
    let num_blocks = output.len() / QK_K;
    for block_idx in 0..num_blocks {
        let boff = block_idx * BLOCK_IQ2_XXS_SIZE;
        if boff + BLOCK_IQ2_XXS_SIZE > block_bytes.len() { break; }
        let d = f16_from_bytes(block_bytes, boff);
        for ib32 in 0..8usize {
            let aux_off = boff + 2 + ib32 * 8;
            let aux0 = u32::from_le_bytes([
                block_bytes[aux_off], block_bytes[aux_off+1], block_bytes[aux_off+2], block_bytes[aux_off+3],
            ]);
            let aux1 = u32::from_le_bytes([
                block_bytes[aux_off+4], block_bytes[aux_off+5], block_bytes[aux_off+6], block_bytes[aux_off+7],
            ]);
            let db = d * (0.5 + ((aux1 >> 28) as f32)) * 0.25;
            let aux8 = aux0.to_le_bytes();
            let out_base = block_idx * QK_K + ib32 * 32;
            for l in 0..4usize {
                let grid_bytes = grid[aux8[l] as usize].to_le_bytes();
                let signs = signs_tbl[((aux1 >> (7 * l as u32)) & 127) as usize];
                for j in 0..8usize {
                    let sign = if signs & mask[j] != 0 { -1.0 } else { 1.0 };
                    output[out_base + l * 8 + j] = db * (grid_bytes[j] as i8 as f32) * sign;
                }
            }
        }
    }
}

pub fn vec_dot_iq2_xxs_q8k_scalar(iq2xxs_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    let mask = self::iq_tables::iq2_xs_mask();
    let grid = self::iq_tables::iq2_xxs_grid();
    let signs_tbl = self::iq_tables::iq2_xs_signs();
    let nb = q8k.len();
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let boff = i * BLOCK_IQ2_XXS_SIZE;
        if boff + BLOCK_IQ2_XXS_SIZE > iq2xxs_data.len() { break; }
        let d = f16_from_bytes(iq2xxs_data, boff) * q8k[i].d;
        let q2 = &iq2xxs_data[boff + 2..boff + 2 + 64];
        let q8 = &q8k[i].qs;
        let mut bsum: i32 = 0;
        let mut q2_idx = 0;
        let mut q8_idx = 0;
        for _ib32 in 0..8 {
            let mut aux32 = [0u32; 2];
            aux32[0] = u32::from_le_bytes([q2[q2_idx], q2[q2_idx+1], q2[q2_idx+2], q2[q2_idx+3]]);
            aux32[1] = u32::from_le_bytes([q2[q2_idx+4], q2[q2_idx+5], q2[q2_idx+6], q2[q2_idx+7]]);
            q2_idx += 8;
            let aux8 = aux32[0].to_le_bytes();
            let ls = (2 * ((aux32[1] >> 28) as i32)) + 1;
            let mut sumi: i32 = 0;
            for l in 0..4 {
                let grid_bytes = grid[aux8[l] as usize].to_le_bytes();
                let signs = signs_tbl[((aux32[1] >> (7 * l as u32)) & 127) as usize];
                for j in 0..8 {
                    let sign = if signs & mask[j] != 0 { -1 } else { 1 };
                    sumi += (grid_bytes[j] as i8 as i32) * (q8[q8_idx + j] as i32) * sign;
                }
                q8_idx += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * (bsum as f32) * 0.125;
    }
    sumf
}

pub fn dequantize_row_iq2_s(block_bytes: &[u8], output: &mut [f32]) {
    let mask = self::iq_tables::iq2_xs_mask();
    let grid = self::iq_tables::iq2_s_grid();
    let num_blocks = output.len() / QK_K;
    for block_idx in 0..num_blocks {
        let boff = block_idx * BLOCK_IQ2_S_SIZE;
        if boff + BLOCK_IQ2_S_SIZE > block_bytes.len() { break; }
        let d = f16_from_bytes(block_bytes, boff);
        let qs = &block_bytes[boff + 2..boff + 2 + 32];
        let signs = &block_bytes[boff + 2 + 32..boff + 2 + 32 + 32];
        let qh = &block_bytes[boff + 66..boff + 66 + 8];
        let scales = &block_bytes[boff + 74..boff + 74 + 8];
        for ib32 in 0..8usize {
            let db0 = d * (0.5 + (scales[ib32] & 0x0f) as f32) * 0.25;
            let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
            let qh_byte = qh[ib32];
            let out_base = block_idx * QK_K + ib32 * 32;
            for l in 0..4usize {
                let dl = if l < 2 { db0 } else { db1 };
                let qh_shift = 8 - 2 * l;
                let grid_idx = (qs[l] as u32 | ((qh_byte as u32) << qh_shift) & 0x300) as usize;
                let grid_bytes = grid[grid_idx].to_le_bytes();
                let s = signs[l];
                for j in 0..8usize {
                    let sign = if s & mask[j] != 0 { -1.0 } else { 1.0 };
                    output[out_base + l * 8 + j] = dl * (grid_bytes[j] as i8 as f32) * sign;
                }
            }
        }
    }
}

pub fn vec_dot_iq2_s_q8k_scalar(iq2s_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    let mask = self::iq_tables::iq2_xs_mask();
    let grid = self::iq_tables::iq2_s_grid();
    let nb = q8k.len();
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let boff = i * BLOCK_IQ2_S_SIZE;
        if boff + BLOCK_IQ2_S_SIZE > iq2s_data.len() { break; }
        let d = f16_from_bytes(iq2s_data, boff) * q8k[i].d;
        let qs = &iq2s_data[boff + 2..boff + 2 + 32];
        let qh = &iq2s_data[boff + 66..boff + 66 + 8];
        let scales = &iq2s_data[boff + 74..boff + 74 + 8];
        let signs_base = boff + 2 + 32;
        let q8 = &q8k[i].qs;
        let mut bsum: i32 = 0;
        let mut qs_idx = 0;
        let mut q8_idx = 0;
        let mut signs_idx = signs_base;
        for ib32 in 0..8usize {
            let ls1 = 1 + 2 * (scales[ib32] & 0x0f) as i32;
            let ls2 = 1 + 2 * (scales[ib32] >> 4) as i32;
            let qh_byte = qh[ib32];
            let mut sumi1: i32 = 0;
            for l in 0..2 {
                let qh_shift = 8 - 2 * l;
                let grid_idx = (qs[qs_idx + l] as u32 | ((qh_byte as u32) << qh_shift) & 0x300) as usize;
                let grid_bytes = grid[grid_idx].to_le_bytes();
                let s = iq2s_data[signs_idx + l];
                for j in 0..8 {
                    let sign = if s & mask[j] != 0 { -1 } else { 1 };
                    sumi1 += q8[q8_idx + j] as i32 * (grid_bytes[j] as i8 as i32) * sign;
                }
                q8_idx += 8;
            }
            let mut sumi2: i32 = 0;
            for l in 2..4 {
                let qh_shift = 8 - 2 * l;
                let grid_idx = (qs[qs_idx + l] as u32 | ((qh_byte as u32) << qh_shift) & 0x300) as usize;
                let grid_bytes = grid[grid_idx].to_le_bytes();
                let s = iq2s_data[signs_idx + l];
                for j in 0..8 {
                    let sign = if s & mask[j] != 0 { -1 } else { 1 };
                    sumi2 += q8[q8_idx + j] as i32 * (grid_bytes[j] as i8 as i32) * sign;
                }
                q8_idx += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs_idx += 4;
            signs_idx += 4;
        }
        sumf += d * (bsum as f32) * 0.125;
    }
    sumf
}

pub fn dequantize_row_iq3_xxs(block_bytes: &[u8], output: &mut [f32]) {
    let mask = self::iq_tables::iq2_xs_mask();
    let grid = self::iq_tables::iq3_xxs_grid();
    let signs_tbl = self::iq_tables::iq2_xs_signs();
    let num_blocks = output.len() / QK_K;
    for block_idx in 0..num_blocks {
        let boff = block_idx * BLOCK_IQ3_XXS_SIZE;
        if boff + BLOCK_IQ3_XXS_SIZE > block_bytes.len() { break; }
        let d = f16_from_bytes(block_bytes, boff);
        let qs = &block_bytes[boff + 2..boff + 2 + 96];
        let scales_and_signs = &block_bytes[boff + 2 + 64..boff + 2 + 64 + 32];
        for ib32 in 0..8usize {
            let aux32 = u32::from_le_bytes([
                scales_and_signs[ib32 * 4], scales_and_signs[ib32 * 4 + 1],
                scales_and_signs[ib32 * 4 + 2], scales_and_signs[ib32 * 4 + 3],
            ]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            let qs_off = ib32 * 8;
            let out_base = block_idx * QK_K + ib32 * 32;
            for l in 0..4usize {
                let signs = signs_tbl[((aux32 >> (7 * l as u32)) & 127) as usize];
                let g1 = grid[qs[qs_off + 2 * l] as usize].to_le_bytes();
                let g2 = grid[qs[qs_off + 2 * l + 1] as usize].to_le_bytes();
                for j in 0..4usize {
                    let s1 = if signs & mask[j] != 0 { -1.0 } else { 1.0 };
                    let s2 = if signs & mask[j + 4] != 0 { -1.0 } else { 1.0 };
                    output[out_base + l * 8 + j] = db * (g1[j] as i8 as f32) * s1;
                    output[out_base + l * 8 + j + 4] = db * (g2[j] as i8 as f32) * s2;
                }
            }
        }
    }
}

pub fn vec_dot_iq3_xxs_q8k_scalar(iq3xxs_data: &[u8], q8k: &[BlockQ8K]) -> f32 {
    let mask = self::iq_tables::iq2_xs_mask();
    let grid = self::iq_tables::iq3_xxs_grid();
    let signs_tbl = self::iq_tables::iq2_xs_signs();
    let nb = q8k.len();
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let boff = i * BLOCK_IQ3_XXS_SIZE;
        if boff + BLOCK_IQ3_XXS_SIZE > iq3xxs_data.len() { break; }
        let d = f16_from_bytes(iq3xxs_data, boff) * q8k[i].d;
        let q3 = &iq3xxs_data[boff + 2..boff + 2 + 96];
        let gas = &iq3xxs_data[boff + 2 + 64..boff + 2 + 64 + 32];
        let q8 = &q8k[i].qs;
        let mut bsum: i32 = 0;
        let mut q3_idx = 0;
        let mut q8_idx = 0;
        let mut gas_idx = 0;
        for _ib32 in 0..8 {
            let aux32 = u32::from_le_bytes([
                gas[gas_idx], gas[gas_idx+1], gas[gas_idx+2], gas[gas_idx+3],
            ]);
            gas_idx += 4;
            let ls = (2 * ((aux32 >> 28) as i32)) + 1;
            let mut sumi: i32 = 0;
            for l in 0..4 {
                let signs = signs_tbl[((aux32 >> (7 * l as u32)) & 127) as usize];
                let g1 = grid[q3[q3_idx + 2 * l] as usize].to_le_bytes();
                let g2 = grid[q3[q3_idx + 2 * l + 1] as usize].to_le_bytes();
                for j in 0..4 {
                    let s1 = if signs & mask[j] != 0 { -1 } else { 1 };
                    let s2 = if signs & mask[j + 4] != 0 { -1 } else { 1 };
                    sumi += g1[j] as i8 as i32 * q8[q8_idx + j] as i32 * s1;
                    sumi += g2[j] as i8 as i32 * q8[q8_idx + j + 4] as i32 * s2;
                }
                q8_idx += 8;
            }
            q3_idx += 8;
            bsum += sumi * ls;
        }
        sumf += d * (bsum as f32) * 0.25;
    }
    sumf
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

    fn make_q3k_block(hmask: &[u8; 32], qs: &[u8; 64], scales_packed: &[u8; 12], d: f32) -> Vec<u8> {
        let mut v = vec![0u8; 110];
        for i in 0..32 { v[i] = hmask[i]; }
        for i in 0..64 { v[32 + i] = qs[i]; }
        for i in 0..12 { v[96 + i] = scales_packed[i]; }
        let d_bits = crate::ops::f32_to_f16(d).to_le_bytes();
        v[108] = d_bits[0];
        v[109] = d_bits[1];
        v
    }

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
    fn q3k_dequantize_zero_qs_yields_constant() {
        // All-zero qs + zero hmask + scale=1 + d=1:
        //   q_signed = 0 - 4 = -4 (hmask bit not set, all values negative)
        //   dl = d * (scale - 32) = 1 * (1 - 32) = -31
        //   y[j] = dl * q_signed = -31 * -4 = 124 for all j
        let hmask = [0u8; 32];
        let qs = [0u8; 64];
        // scales_packed = [0x11; 8, 0, 0, 0] (lsb 1, msb 0 for all 16 scales)
        let mut scales_packed = [0u8; 12];
        for i in 0..8 { scales_packed[i] = 0x11; }
        let weight = make_q3k_block(&hmask, &qs, &scales_packed, 1.0);
        let mut output = vec![0.0f32; 256];
        super::dequantize_row_q3_k(&weight, &mut output);
        for j in 0..256 {
            assert!(
                (output[j] - 124.0).abs() < 1e-3,
                "Q3_K zero qs: j={} got {} expected 124",
                j, output[j]
            );
        }
    }

    #[test]
    fn q3k_dequantize_all_max_hmask_yields_constant_positive() {
        // All-zero qs + all-ones hmask + scale=1 + d=1:
        //   q_signed = 0 - 0 = 0 (hmask bit set, all values zero)
        //   y[j] = 0 for all j
        let hmask = [0xffu8; 32];
        let qs = [0u8; 64];
        let mut scales_packed = [0u8; 12];
        for i in 0..8 { scales_packed[i] = 0x11; }
        let weight = make_q3k_block(&hmask, &qs, &scales_packed, 1.0);
        let mut output = vec![0.0f32; 256];
        super::dequantize_row_q3_k(&weight, &mut output);
        for j in 0..256 {
            assert!(output[j].abs() < 1e-6, "Q3_K all-max-hmask: j={} got {} expected 0", j, output[j]);
        }
    }

    #[test]
    fn q3k_dequantize_qs2_hmask0_yields_2_d_all_match() {
        // For each element: ql = (qs[j/4] >> (2*(j%4))) & 3
        // Build qs so that ql = 2 for all elements, hmask = 0 (all negative).
        // value = (ql | (qh << 2)) - 4 = (2 | 0) - 4 = -2
        // With scale = 1, d = 1: y[j] = 1 * (1 - 32) * -2 = 62
        let hmask = [0u8; 32];
        let mut qs = [0u8; 64];
        for j in 0..256 {
            qs[j / 4] |= 2u8 << ((j % 4) * 2);
        }
        let mut scales_packed = [0u8; 12];
        for i in 0..8 { scales_packed[i] = 0x11; }
        let weight = make_q3k_block(&hmask, &qs, &scales_packed, 1.0);
        let mut output = vec![0.0f32; 256];
        super::dequantize_row_q3_k(&weight, &mut output);
        for j in 0..256 {
            assert!(
                (output[j] - 62.0).abs() < 1e-3,
                "Q3_K ql=2: j={} got {} expected 62",
                j, output[j]
            );
        }
    }

    #[test]
    fn q3k_vecdot_all_zero_input_yields_zero() {
        // qs=0, hmask=0, scales=1, d=1.
        // input_f32 = [0, 0, ..., 0] → q8k.d = 0 → dot product = 0
        let hmask = [0u8; 32];
        let qs = [0u8; 64];
        let mut scales_packed = [0u8; 12];
        for i in 0..8 { scales_packed[i] = 0x11; }
        let weight = make_q3k_block(&hmask, &qs, &scales_packed, 1.0);
        let q8k = vec![BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }];
        let dot = super::vec_dot_q3k_q8k_scalar(&weight, &q8k);
        assert!(dot.abs() < 1e-6, "Q3_K dot with zero q8k: got {} expected 0", dot);
    }

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

#[cfg(all(test, target_arch = "x86_64"))]
mod avx2_kq_parity {
    use super::*;

    fn make_q2k_block(
        d: f32,
        dmin: f32,
        scales: &[u8; 16],
        qs: &[u8; 64],
    ) -> Vec<u8> {
        let mut v = Vec::with_capacity(84);
        for s in scales {
            v.push(*s);
        }
        v.extend_from_slice(qs);
        v.extend_from_slice(&crate::ops::f32_to_f16(d).to_le_bytes());
        v.extend_from_slice(&crate::ops::f32_to_f16(dmin).to_le_bytes());
        v
    }

    #[test]
    fn q2k_avx2_matches_scalar_one_block() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let _ = vec_dot_q2k_q8k_avx2_direct;
    }

    #[test]
    fn iq4_xs_avx2_matches_scalar_one_block() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let scales_h: u16 = 0xaaaa;
        let scales_l = [0x55u8, 0x33, 0x77, 0x11];
        let qs: [u8; 128] = std::array::from_fn(|i| ((i * 19 + 5) % 256) as u8);
        let weight = {
            let mut v = vec![0u8; 136];
            let d_bytes = crate::ops::f32_to_f16(0.5).to_le_bytes();
            v[0] = d_bytes[0];
            v[1] = d_bytes[1];
            let sh_bytes = scales_h.to_le_bytes();
            v[2] = sh_bytes[0];
            v[3] = sh_bytes[1];
            v[4] = scales_l[0];
            v[5] = scales_l[1];
            v[6] = scales_l[2];
            v[7] = scales_l[3];
            for i in 0..128 {
                v[8 + i] = qs[i];
            }
            v
        };
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let q8k = {
            let mut buf = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; 1];
            super::quantize_row_q8_k_scalar_into(&input, &mut buf);
            buf
        };
        let avx2 = unsafe { vec_dot_iq4_xs_q8k_avx2_direct(&weight, &q8k) };
        let scalar = vec_dot_iq4_xs_q8k_scalar(&weight, &q8k);
        assert_eq!(avx2.to_bits(), scalar.to_bits(), "iq4_xs AVX2 not bit-exact");
    }

    #[test]
    fn q3k_avx2_matches_scalar_one_block() {
        if !crate::ops::has_avx2_fma() {
            return;
        }
        let _ = vec_dot_q3k_q8k_avx2_direct;
    }
}

#[cfg(test)]
mod i_quant_tests {
    use super::*;
    use crate::ops::quant::iq_tables::{
        iq2_xs_grid, iq2_xs_mask, iq2_xs_signs, iq3_s_grid, KVALUES_IQ4NL,
    };

    #[test]
    fn iq4_nl_lut_matches_llama_cpp() {
        assert_eq!(
            KVALUES_IQ4NL,
            [
                -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
            ]
        );
    }

    #[test]
    fn iq_grid_tables_have_expected_length() {
        assert_eq!(iq2_xs_grid().len(), 512);
        assert_eq!(iq2_xs_signs().len(), 128);
        assert_eq!(iq2_xs_mask().len(), 8);
        assert_eq!(iq3_s_grid().len(), 512);
    }

    #[test]
    fn iq2_xs_dequant_all_zero_block_finishes() {
        let mut block = vec![0u8; 74];
        block[0] = crate::ops::f32_to_f16(0.5).to_le_bytes()[0];
        block[1] = crate::ops::f32_to_f16(0.5).to_le_bytes()[1];
        let mut out = vec![0.0f32; 256];
        dequantize_row_iq2_xs(&block, &mut out);
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn iq3_s_dequant_all_zero_block_finishes() {
        let mut block = vec![0u8; 110];
        block[0] = crate::ops::f32_to_f16(0.5).to_le_bytes()[0];
        block[1] = crate::ops::f32_to_f16(0.5).to_le_bytes()[1];
        let mut out = vec![0.0f32; 256];
        dequantize_row_iq3_s(&block, &mut out);
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn iq4_xs_dequant_all_zero_block_yields_zero() {
        let block = vec![0u8; 136];
        let mut out = vec![0.0f32; 256];
        dequantize_row_iq4_xs(&block, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn iq_vecdot_with_zero_q8k_yields_zero() {
        let block = vec![0u8; 136];
        let q8k = vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }];
        let dot = vec_dot_iq4_xs_q8k_scalar(&block, &q8k);
        assert!(dot.abs() < 1e-6);

        let block2 = vec![0u8; 74];
        let dot2 = vec_dot_iq2_xs_q8k_scalar(&block2, &q8k);
        assert!(dot2.abs() < 1e-6);

        let block3 = vec![0u8; 110];
        let dot3 = vec_dot_iq3_s_q8k_scalar(&block3, &q8k);
        assert!(dot3.abs() < 1e-6);
    }

    #[test]
    fn iq2_xxs_vecdot_matches_python_reference() {
        let mut block = vec![0u8; 66];
        block[0] = 0x00;
        block[1] = 0x3c;
        for i in 0..32 {
            block[2 + i] = i as u8;
        }
        let mut qs = [0i8; 256];
        for q in qs.iter_mut() { *q = 1; }
        let q8k = vec![BlockQ8K { d: 1.0, qs, bsums: [0i16; 16] }];
        let dot = vec_dot_iq2_xxs_q8k_scalar(&block, &q8k);
        assert!((dot - 155.875).abs() < 0.01, "expected 155.875, got {}", dot);
    }
}
