//! Embedding lookup functions for quantized token embedding tables.

pub fn embedding_lookup(weight: &[u8], token_id: u32, n_embd: usize, embd_type: crate::model::GGMLType, out: &mut [f32]) {
    match embd_type {
        crate::model::GGMLType::Q8_0 => embedding_lookup_q8_0(weight, token_id, n_embd, out),
        crate::model::GGMLType::Q4_0 => embedding_lookup_q4_0(weight, token_id, n_embd, out),
        crate::model::GGMLType::Q6K => embedding_lookup_q6_k(weight, token_id, n_embd, out),
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
    let blocks_per_row = n_embd / 256;
    let row_off = token_id as usize * blocks_per_row * 210;
    for b in 0..blocks_per_row {
        let off = row_off + b * 210;
        let d = super::f16_to_f32(u16::from_le_bytes([weight[off + 208], weight[off + 209]]));
        let scales_off = off + 192;
        let base_y = b * 256;
        // 16 scales per block, each scale handles 16 consecutive L values
        for j in 0..16 {
            let sw = j % 8;
            let sub = j / 8;
            let mut scale = weight[scales_off + j] as i8;
            // ql/qh byte index within sub_block
            let ql_byte_start = (sw / 2) * 16;
            let ql_shift = ((sw % 2) * 4) as u32;
            let qh_byte_start = (sw / 4) * 16;
            let qh_shift = ((sw % 4) * 2) as u32;
            let ql_off = off + sub * 64 + ql_byte_start;
            let qh_off = off + 128 + sub * 32 + qh_byte_start;
            let scale_f = scale as f32;
            for k in 0..16 {
                let ql_byte = weight[ql_off + k] as i32;
                let qh_byte = weight[qh_off + k] as i32;
                let low = (ql_byte >> ql_shift) & 0x0F;
                let high = (qh_byte >> qh_shift) & 0x03;
                let l_value = (low | (high << 4)) - 32;
                out[base_y + sub * 128 + sw * 16 + k] = d * scale_f * l_value as f32;
            }
        }
    }
}