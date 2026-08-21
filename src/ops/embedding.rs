//! Embedding lookup functions for quantized token embedding tables.

pub fn embedding_lookup(weight: &[u8], token_id: u32, n_embd: usize, embd_type: crate::core::tensor::GGMLType, out: &mut [f32]) {
    match embd_type {
        crate::core::tensor::GGMLType::Q8_0 => embedding_lookup_q8_0(weight, token_id, n_embd, out),
        crate::core::tensor::GGMLType::Q4_0 => embedding_lookup_q4_0(weight, token_id, n_embd, out),
        crate::core::tensor::GGMLType::Q6K => embedding_lookup_q6_k(weight, token_id, n_embd, out),
        _ => panic!("unsupported embedding type {:?}", embd_type),
    }
}

pub fn embedding_lookup_q8_0(weight: &[u8], token_id: u32, n_embd: usize, out: &mut [f32]) {
    let blocks_per_row = n_embd / 32;
    let row_off = token_id as usize * blocks_per_row * 34;
    for b in 0..blocks_per_row {
        let off = row_off + b * 34;
        let d = super::f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
        for j in 0..32usize {
            out[b * 32 + j] = d * (weight[off + 2 + j] as i8 as f32);
        }
    }
}

pub fn embedding_lookup_q4_0(weight: &[u8], token_id: u32, n_embd: usize, out: &mut [f32]) {
    let blocks_per_row = n_embd / 32;
    let row_off = token_id as usize * blocks_per_row * 18;
    for b in 0..blocks_per_row {
        let off = row_off + b * 18;
        let d = super::f16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
        for j in 0..16usize {
            let q = weight[off + 2 + j];
            let q0 = ((q & 0x0F) as i8 as f32 - 8.0) * d;
            let q1 = (((q >> 4) & 0x0F) as i8 as f32 - 8.0) * d;
            out[b * 32 + j] = q0;
            out[b * 32 + j + 16] = q1;
        }
    }
}

pub fn embedding_lookup_q6_k(weight: &[u8], token_id: u32, n_embd: usize, out: &mut [f32]) {
    let row_bytes = n_embd / crate::ops::quant::QK_K * crate::ops::quant::BLOCK_Q6K_SIZE;
    let row_start = token_id as usize * row_bytes;
    crate::ops::quant::dequantize_row_q6_k(
        &weight[row_start..row_start + row_bytes],
        &mut out[..n_embd],
    );
}
