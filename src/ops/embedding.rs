//! Embedding lookup functions for quantized token embedding tables.

use crate::core::tensor::GGMLType;

pub const SUPPORTED_EMBEDDING_TYPES: &[GGMLType] = &[
    GGMLType::F16,
    GGMLType::BF16,
    GGMLType::Q8_0,
    GGMLType::Q4_0,
    GGMLType::Q6K,
];

pub fn is_supported_embedding(ggml_type: GGMLType) -> bool {
    SUPPORTED_EMBEDDING_TYPES.contains(&ggml_type)
}

pub fn expect_supported_embedding(name: &str, ggml_type: GGMLType) {
    if !is_supported_embedding(ggml_type) {
        panic!(
            "{name} has unsupported type {ggml_type:?}; supported: {:?}",
            SUPPORTED_EMBEDDING_TYPES
        );
    }
}

pub fn embedding_lookup(weight: &[u8], token_id: u32, n_embd: usize, embd_type: GGMLType, out: &mut [f32]) {
    match embd_type {
        GGMLType::F16 => embedding_lookup_f16(weight, token_id, n_embd, out),
        GGMLType::Q8_0 => embedding_lookup_q8_0(weight, token_id, n_embd, out),
        GGMLType::Q4_0 => embedding_lookup_q4_0(weight, token_id, n_embd, out),
        GGMLType::Q6K => embedding_lookup_q6_k(weight, token_id, n_embd, out),
        GGMLType::BF16 => embedding_lookup_bf16(weight, token_id, n_embd, out),
        _ => panic!(
            "unsupported embedding type {embd_type:?}; supported: {:?}",
            SUPPORTED_EMBEDDING_TYPES
        ),
    }
}

pub fn embedding_lookup_f16(weight: &[u8], token_id: u32, n_embd: usize, out: &mut [f32]) {
    let row_start = token_id as usize * n_embd * 2;
    for index in 0..n_embd {
        let offset = row_start + index * 2;
        let bits = u16::from_le_bytes([weight[offset], weight[offset + 1]]);
        out[index] = super::f16_to_f32(bits);
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

pub fn embedding_lookup_bf16(weight: &[u8], token_id: u32, n_embd: usize, out: &mut [f32]) {
    let row_start = token_id as usize * n_embd * 2;
    for index in 0..n_embd {
        let offset = row_start + index * 2;
        let bits = u16::from_le_bytes([weight[offset], weight[offset + 1]]);
        out[index] = crate::ops::bf16_to_f32(bits);
    }
}
