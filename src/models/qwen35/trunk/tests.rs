//! Unit tests for the Qwen3.5 model + session layer.
//!
//! Position-builder tests live in `positions.rs`. Vision-encoder tests live in
//! `vision.rs`. This module covers:
//! - bit-exact `l2_norm` against the llama.cpp reference
//! - model loading (gated on `RMI_QWEN35_MODEL`)
//! - `Q8_0` quantized matmul dispatch (scalar fallback path)
//! - scratchpad sizing invariants
//! - dense-attention softmax + value reduction (aarch64 NEON pinned)
//! - `Qwen35Session` state management + embed-lookup helpers

use super::session::{required_token_count, Qwen35Session};
use super::*;
use crate::core::scratchpad::KvCache;
use crate::core::tensor::GGMLType;
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::QuantizedTensor;
use crate::ops::kernel::Weight;
use crate::ops::quant::{self, BlockQ8K};
use std::sync::Arc;

fn dense_test_config(n_ctx: usize) -> Qwen35Config {
    Qwen35Config {
        n_nextn: 0,
        n_embd: 32,
        n_layer: 1,
        n_head: 4,
        n_head_kv: 1,
        n_ff: 64,
        n_ctx,
        vocab_size: 32,
        rope_freq_base: 1_000_000.0,
        norm_eps: 1e-6,
        rope_dimension_count: 8,
        rope_dimension_sections: [2; 4],
        ssm_d_conv: 2,
        ssm_d_state: 8,
        ssm_n_group: 1,
        ssm_dt_rank: 1,
        ssm_d_inner: 8,
        full_attention_interval: 1,
        is_recurrent: vec![false],
        key_length: 32,
        value_length: 8,
    }
}

fn tiny_dense_model(k_weight: [f32; 4], v_weight: [f32; 4]) -> Qwen35Model<'static> {
    let config = Qwen35Config {
        n_nextn: 0,
        n_embd: 2,
        n_layer: 1,
        n_head: 1,
        n_head_kv: 1,
        n_ff: 2,
        n_ctx: 256,
        vocab_size: 2,
        rope_freq_base: 1_000_000.0,
        norm_eps: 0.0,
        rope_dimension_count: 2,
        rope_dimension_sections: [0; 4],
        ssm_d_conv: 1,
        ssm_d_state: 2,
        ssm_n_group: 1,
        ssm_dt_rank: 1,
        ssm_d_inner: 2,
        full_attention_interval: 1,
        is_recurrent: vec![false],
        key_length: 2,
        value_length: 2,
    };
    let weight = |data: Vec<f32>, _n_rows| Weight::from_quantized(QuantizedTensor::F32(data));
    let identity = || weight(vec![1.0, 0.0, 0.0, 1.0], 2);
    let layer = Qwen35LayerWeights {
        attn_norm: vec![1.0; 2],
        attn_post_norm: vec![1.0; 2],
        wq: Some(weight(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4)),
        wk: Some(weight(k_weight.to_vec(), 2)),
        wv: Some(weight(v_weight.to_vec(), 2)),
        wo: Some(identity()),
        attn_q_norm: Some(vec![1.0; 2]),
        attn_k_norm: Some(vec![4.0, 1.0]),
        wqkv: None,
        wqkv_gate: None,
        ssm_conv1d: None,
        ssm_dt: None,
        ssm_a: None,
        ssm_beta: None,
        ssm_alpha: None,
        ssm_norm: None,
        ssm_out: None,
        ffn_gate: identity(),
        ffn_up: identity(),
        ffn_down: identity(),
    };
    Qwen35Model {
        config,
        tok_embd: Vec::new(),
        output_norm: vec![1.0; 2],
        output_weight: identity(),
        layers: vec![layer],
    }
}

#[test]
fn qwen35_l2_norm_matches_pinned_llama_cpp_bits() {
    const INPUT_BITS: [u32; 128] = [
        0x3b11b23c, 0x3abb616e, 0x3a74a52a, 0xbe36d3f9, 0x3c5be35f, 0xbb7eed3d, 0x3cf3b675,
        0x3c352a95, 0xbd12fa43, 0x3d0ed381, 0xbd019bdf, 0xbaa1fbf3, 0xbb00502c, 0x3b97baf3,
        0x3c0a9eb1, 0x3b2a60d7, 0x3c867f8c, 0x39269d95, 0xbc86036e, 0x3c9ca5fe, 0xbd2beb72,
        0xba3c9646, 0x3c19ee2b, 0x3c6bebc5, 0xba1078eb, 0x3cddb829, 0xb9cb7b05, 0x3c10df58,
        0x38ad8928, 0x3a1bb101, 0xbc9dea12, 0xb830b929, 0xbce68c07, 0x3c77b70a, 0x3a9a5020,
        0x3aa1e526, 0xbc830d2f, 0xbccdf802, 0xbae28161, 0x3c13147a, 0xbab844a1, 0xbd6a98ad,
        0xb92c2313, 0x3b279123, 0x3cda6539, 0x398fc008, 0xb9883eab, 0x3c7a1914, 0x3a9fd962,
        0x3c024411, 0x3cfce9fa, 0x39fe7429, 0x39a78347, 0xbcd4f897, 0xb972edb6, 0x3cd07782,
        0xba48c562, 0x3d226f5a, 0x3a002199, 0x38995247, 0x3d24b1fe, 0x3c95409c, 0xb8b86102,
        0xb9c4d2a9, 0x3d1a4c1e, 0xb9679706, 0xbc94ecb1, 0xbb87b477, 0xbc306760, 0x3ad1617d,
        0x3c85d8db, 0xb886458a, 0x3baf244f, 0xbd5d49dc, 0xbb8260a4, 0xbc4a82db, 0x3aa20bbd,
        0x3d2b8415, 0xba532f00, 0x39b7c6d9, 0xbad2ca65, 0xbd023279, 0xbc77cee6, 0x393b6a88,
        0x3c95ab9c, 0xba920ce0, 0xbb9881f0, 0xbaafd1fc, 0x3b8edc22, 0x390e7b29, 0x3bbe234b,
        0xbb803967, 0xb926490d, 0xbcf4b75a, 0x3c56bbe5, 0x3b016c3f, 0xbc6cef21, 0x3b30b9c6,
        0xb9ef1f3c, 0xb8aecebe, 0xba7b7fec, 0xb929766d, 0x3c9b5ced, 0xbc8ca6a4, 0xbcd36384,
        0x3d2261e8, 0x3ccda1b0, 0xbd298883, 0x3d40c6d6, 0x3969c035, 0x3c9c3466, 0xb991a558,
        0xb976ce00, 0xb9b01921, 0x398eef4c, 0xb5e9fa58, 0xbd416b03, 0x3be68c77, 0x39f3d603,
        0x3d0243cb, 0x3b3fc530, 0xbc016f46, 0x3bb1d80e, 0xba45c19f, 0xba89fe14, 0x3b26ebc8,
        0xb9d28f76, 0xbbec5142,
    ];
    const EXPECTED_BITS: [u32; 128] = [
        0x3c05bcb8, 0x3bac0000, 0x3b60906f, 0xbf27d235, 0x3d49d6dc, 0xbc6a0077, 0x3ddfb552,
        0x3d264bbc, 0xbe06e9d2, 0x3e031a4c, 0xbdedf0cd, 0xbb94b02d, 0xbbeb8fdb, 0x3c8b46a4,
        0x3cfe7bbe, 0x3c1c64ae, 0x3d76eaac, 0x3a18f07d, 0xbd7606d0, 0x3d8fca57, 0xbe1dcee5,
        0xbb2d1b7f, 0x3d0d4ba1, 0x3d588e5d, 0xbb049d1f, 0x3dcb852c, 0xbabac748, 0x3d04fb24,
        0x399f4aa6, 0x3b0ee976, 0xbd90f3d1, 0xb92237ac, 0xbdd39f8b, 0x3d6361ce, 0x3b8da58c,
        0x3b949b3f, 0xbd7096cc, 0xbdbd0ffc, 0xbbcfe9d2, 0x3d0701e2, 0xbba9249b, 0xbe57571a,
        0xba1e01f6, 0x3c19d00d, 0x3dc87814, 0x3a83f369, 0xba7a1f83, 0x3d6591c5, 0x3b92ba79,
        0x3cef2594, 0x3de8277f, 0x3ae99153, 0x3a99c355, 0xbdc37d6e, 0xba5efd0e, 0x3dbf5aff,
        0xbb384a95, 0x3e151a1b, 0x3aeb3a5a, 0x398cbc89, 0x3e172d40, 0x3d89005e, 0xb9a93ea7,
        0xbab4aad2, 0x3e0da1de, 0xba5494a0, 0xbd88b357, 0xbc7921cb, 0xbd21ec9a, 0x3bc031c5,
        0x3d75b8a7, 0xb976802e, 0x3ca0c40e, 0xbe4b1fec, 0xbc6f5a09, 0xbd39e37d, 0x3b94beab,
        0x3e1d7004, 0xbb41d966, 0x3aa8b126, 0xbbc17d0d, 0xbdef0548, 0xbd6377b4, 0x3a2c085b,
        0x3d896296, 0xbb860fec, 0xbc8bfd4c, 0xbba16379, 0x3c832238, 0x3a02c934, 0x3cae87ed,
        0xbc6b660e, 0xba18a2e6, 0xbde0a121, 0x3d451bb1, 0x3bed995e, 0xbd597c6f, 0x3c22383d,
        0xbadb7e90, 0xb9a07583, 0xbb66db29, 0xba1b8d82, 0x3d8e9c48, 0xbd811b24, 0xbdc2099b,
        0x3e150dc4, 0x3dbcc0c0, 0xbe1b9e1c, 0x3e30f405, 0x3a569067, 0x3d8f6212, 0xba85b0e3,
        0xba628be5, 0xbaa1a4c7, 0x3a8333cf, 0xb6d6c5c4, 0xbe318ab8, 0x3cd39ff2, 0x3adfd249,
        0x3def2514, 0x3c300785, 0xbced9eed, 0x3ca33f05, 0xbb35862b, 0xbb7d54e2, 0x3c193845,
        0xbac146f5, 0xbcd8eb85,
    ];

    let mut actual = INPUT_BITS.map(f32::from_bits);
    super::util::l2_norm(&mut actual, 1e-6);

    assert_eq!(actual.map(f32::to_bits), EXPECTED_BITS);
}

#[test]
#[ignore = "requires RMI_QWEN35_MODEL"]
fn qwen35_q8_0_model_loads() {
    let path = std::env::var("RMI_QWEN35_MODEL").expect("RMI_QWEN35_MODEL must be set");
    let source =
        crate::open_model_source(std::path::Path::new(&path), crate::ComponentRole::Llm).unwrap();

    assert_eq!(
        source.tensor_info("token_embd.weight").unwrap().ggml_type,
        GGMLType::Q8_0,
    );
    Qwen35Model::from_source(source.as_ref()).unwrap();
}

#[test]
#[ignore = "requires an F16-token-embedding RMI_QWEN35_MODEL"]
fn qwen35_f16_token_embedding_model_loads() {
    let path = std::env::var("RMI_QWEN35_MODEL").expect("RMI_QWEN35_MODEL must be set");
    let source =
        crate::open_model_source(std::path::Path::new(&path), crate::ComponentRole::Llm).unwrap();

    assert_eq!(
        source.tensor_info("token_embd.weight").unwrap().ggml_type,
        GGMLType::F16,
    );
    Qwen35Model::from_source(source.as_ref()).unwrap();
}

#[test]
fn qwen35_q8_0_quantize_and_matmul_dispatches() {
    let mut data = Vec::with_capacity(2 * 8 * quant::BLOCK_Q80_SIZE);
    for quantized_value in [1u8, (-2i8) as u8] {
        for _ in 0..8 {
            data.extend_from_slice(&[0x00, 0x3c]);
            data.extend(std::iter::repeat_n(quantized_value, 32));
        }
    }
    let weight = QuantizedTensor::Q8_0 {
        data: &data,
        n_cols: 256,
        n_rows: 2,
    };
    let input = [1.0f32; 256];
    let mut q8k_buf = vec![
        BlockQ8K {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16]
        };
        1
    ];
    let mut output = [0.0f32; 2];

    weight.quantize_and_matmul(&input, &mut q8k_buf, &mut output);

    for (actual, expected) in output.into_iter().zip([256.0f32, -512.0]) {
        assert!(
            (actual - expected).abs() < 0.05,
            "actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn qwen35_scratch_covers_dense_attention_output_projection() {
    let config = dense_test_config(8);
    let dense_attn_out_dim = config.n_embd_head() * config.n_head;
    assert!(dense_attn_out_dim > config.n_embd.max(config.n_ff).max(config.value_dim()));

    let scratch = Qwen35Scratchpad::new(&config, 1);

    assert!(
        scratch.q8_buf.len() >= dense_attn_out_dim
            && scratch.scale_buf.len() >= (dense_attn_out_dim + 31) / 32,
        "dense_attn_out_dim={dense_attn_out_dim}, q8={}, scales={}",
        scratch.q8_buf.len(),
        scratch.scale_buf.len(),
    );
}

#[test]
fn qwen35_scratch_pads_attention_buffers_to_ggml_row_size() {
    let scratch = Qwen35Scratchpad::new(&dense_test_config(257), 1);

    assert_eq!(scratch.score_buf.len(), 512);
    assert_eq!(scratch.attention_value_buf.len(), 512);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn qwen35_dense_attention_softmax_uses_ggml_padded_row() {
    let model = tiny_dense_model([1.0, 1.0, 0.0, 0.5], [1.0, 0.0, 0.0, 1.0]);
    let mut scratch = Qwen35Scratchpad::new(&model.config, 2);
    let mut kv_cache = crate::core::scratchpad::KvCache::new_f32(1, model.config.n_ctx, 2);
    let pool = ComputePool::new(1);

    model.forward_dense_attn_layer(
        0,
        &[1.0, 0.0, 0.0, 1.0],
        2,
        &mut kv_cache,
        &mut scratch,
        &pool,
        &[[0; 4]; 2],
        #[cfg(feature = "parity-trace")]
        false,
    );

    assert_eq!(
        [
            scratch.score_buf[0].to_bits(),
            scratch.score_buf[1].to_bits()
        ],
        [0x3f25_1fe0, 0x3eb5_c03f],
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn qwen35_dense_attention_value_uses_ggml_padded_reduction() {
    let model = tiny_dense_model([1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
    let n_tokens = 18;
    let mut scratch = Qwen35Scratchpad::new(&model.config, n_tokens);
    let mut kv_cache = crate::core::scratchpad::KvCache::new_f32(1, model.config.n_ctx, 2);
    let pool = ComputePool::new(1);
    let input: Vec<f32> = std::iter::repeat_n([1.0, 0.0], n_tokens)
        .flatten()
        .collect();

    let output = model.forward_dense_attn_layer(
        0,
        &input,
        n_tokens,
        &mut kv_cache,
        &mut scratch,
        &pool,
        &vec![[0; 4]; n_tokens],
        #[cfg(feature = "parity-trace")]
        false,
    );

    assert_eq!(output[(n_tokens - 1) * 2].to_bits(), 0x3f00_0000);
}

// ------------------------------------------------------------------
// Qwen35Session tests
// ------------------------------------------------------------------

/// Tiny dense-only Qwen35Model used to construct a Session.
/// `tok_embd` row i = [4*i+0, 4*i+1, 4*i+2, 4*i+3]; n_embd=4, vocab=8.
fn tiny_dense_session_model() -> Qwen35Model<'static> {
    let config = Qwen35Config {
        n_nextn: 0,
        n_embd: 4,
        n_layer: 1,
        n_head: 1,
        n_head_kv: 1,
        n_ff: 4,
        n_ctx: 16,
        vocab_size: 8,
        rope_freq_base: 1_000_000.0,
        norm_eps: 1e-6,
        rope_dimension_count: 4,
        rope_dimension_sections: [0; 4],
        ssm_d_conv: 1,
        ssm_d_state: 2,
        ssm_n_group: 1,
        ssm_dt_rank: 1,
        ssm_d_inner: 2,
        full_attention_interval: 1,
        is_recurrent: vec![false],
        key_length: 4,
        value_length: 4,
    };
    let mk_weight = |data: Vec<f32>| Weight::from_quantized(QuantizedTensor::F32(data));
    let layer = Qwen35LayerWeights {
        attn_norm: vec![1.0; 4],
        attn_post_norm: vec![1.0; 4],
        wq: Some(mk_weight(vec![1.0; 16])),
        wk: Some(mk_weight(vec![1.0; 16])),
        wv: Some(mk_weight(vec![1.0; 16])),
        wo: Some(mk_weight(vec![1.0; 16])),
        attn_q_norm: Some(vec![1.0; 4]),
        attn_k_norm: Some(vec![1.0; 4]),
        wqkv: None,
        wqkv_gate: None,
        ssm_conv1d: None,
        ssm_dt: None,
        ssm_a: None,
        ssm_beta: None,
        ssm_alpha: None,
        ssm_norm: None,
        ssm_out: None,
        ffn_gate: mk_weight(vec![1.0; 16]),
        ffn_up: mk_weight(vec![1.0; 16]),
        ffn_down: mk_weight(vec![1.0; 16]),
    };
    let tok_embd: Vec<f32> = (0..32).map(|i| i as f32).collect();
    Qwen35Model {
        config,
        tok_embd,
        output_norm: vec![1.0; 4],
        output_weight: mk_weight(vec![1.0; 32]),
        layers: vec![layer],
    }
}

fn tiny_q8_session_model() -> Qwen35Model<'static> {
    fn q8_weight(n_rows: usize) -> Weight<'static> {
        const N_COLS: usize = 256;
        let mut data = Vec::with_capacity(n_rows * N_COLS / 32 * quant::BLOCK_Q80_SIZE);
        for _ in 0..n_rows {
            for _ in 0..N_COLS / 32 {
                data.extend_from_slice(&crate::ops::f32_to_f16(0.01).to_le_bytes());
                data.extend(std::iter::repeat_n(1, 32));
            }
        }
        let data = Box::leak(data.into_boxed_slice());
        Weight::from_quantized(QuantizedTensor::Q8_0 {
            data,
            n_cols: N_COLS,
            n_rows,
        })
    }

    let config = Qwen35Config {
        n_nextn: 0,
        n_embd: 256,
        n_layer: 1,
        n_head: 1,
        n_head_kv: 1,
        n_ff: 256,
        n_ctx: 3,
        vocab_size: 32,
        rope_freq_base: 1_000_000.0,
        norm_eps: 1e-6,
        rope_dimension_count: 256,
        rope_dimension_sections: [0; 4],
        ssm_d_conv: 1,
        ssm_d_state: 2,
        ssm_n_group: 1,
        ssm_dt_rank: 1,
        ssm_d_inner: 2,
        full_attention_interval: 1,
        is_recurrent: vec![false],
        key_length: 256,
        value_length: 256,
    };
    let layer = Qwen35LayerWeights {
        attn_norm: vec![1.0; 256],
        attn_post_norm: vec![1.0; 256],
        wq: Some(q8_weight(512)),
        wk: Some(q8_weight(256)),
        wv: Some(q8_weight(256)),
        wo: Some(q8_weight(256)),
        attn_q_norm: Some(vec![1.0; 256]),
        attn_k_norm: Some(vec![1.0; 256]),
        wqkv: None,
        wqkv_gate: None,
        ssm_conv1d: None,
        ssm_dt: None,
        ssm_a: None,
        ssm_beta: None,
        ssm_alpha: None,
        ssm_norm: None,
        ssm_out: None,
        ffn_gate: q8_weight(256),
        ffn_up: q8_weight(256),
        ffn_down: q8_weight(256),
    };
    Qwen35Model {
        config,
        tok_embd: (0..32 * 256)
            .map(|index| (index % 17 + 1) as f32 * 0.01)
            .collect(),
        output_norm: vec![1.0; 256],
        output_weight: q8_weight(32),
        layers: vec![layer],
    }
}

fn session_pool() -> Arc<ComputePool> {
    Arc::new(ComputePool::new(1))
}

#[test]
fn session_new_initializes_state_and_allocates_cache() {
    let model = tiny_dense_session_model();
    let pool = session_pool();
    let session = Qwen35Session::new(&model, pool.clone()).unwrap();

    assert_eq!(session.next_position(), 0);
    assert_eq!(session.config().vocab_size, 8);
    assert_eq!(session.config().n_embd, 4);
    assert_eq!(session.model().config.n_layer, model.config.n_layer);
    assert_eq!(session.pool().n_threads(), 1);
    // Scratchpad sized for n_ctx tokens
    assert!(session.scratch().x.len() >= 4 * 4);
}

#[test]
fn session_new_with_capacity_sizes_state_to_requested_limit() {
    let model = tiny_dense_session_model();
    let session = Qwen35Session::new_with_capacity(&model, session_pool(), 4).unwrap();

    assert_eq!(session.scratch().x.len(), 4 * model.config.n_embd);
    let KvCache::F32(cache) = session.kv_cache() else {
        panic!("Qwen3.5 KV cache should be F32");
    };
    assert_eq!(
        cache.k.len(),
        model.config.n_layer_impl() * 4 * model.config.n_embd_head()
    );
    assert_eq!(cache.v.len(), cache.k.len());
}

#[test]
fn session_new_with_capacity_rejects_out_of_range_limits() {
    let model = tiny_dense_session_model();

    let zero = Qwen35Session::new_with_capacity(&model, session_pool(), 0)
        .err()
        .unwrap();
    assert!(zero.contains("within 1..=16"), "unexpected error: {zero}");

    let oversized = Qwen35Session::new_with_capacity(&model, session_pool(), 17)
        .err()
        .unwrap();
    assert!(
        oversized.contains("within 1..=16"),
        "unexpected error: {oversized}"
    );
}

#[test]
fn session_step_rejects_tokens_beyond_requested_capacity() {
    let model = tiny_dense_session_model();
    let mut session = Qwen35Session::new_with_capacity(&model, session_pool(), 1).unwrap();

    let error = session
        .step(&[0.0; 8], 2, &[[0; 4], [1, 1, 1, 0]])
        .unwrap_err();

    assert!(
        error.contains("requires 2 tokens; session capacity is 1"),
        "unexpected error: {error}"
    );
}

#[test]
fn session_step_rejects_empty_token_batches() {
    let model = tiny_dense_session_model();
    let mut session = Qwen35Session::new_with_capacity(&model, session_pool(), 1).unwrap();

    let error = session.step(&[], 0, &[]).unwrap_err();

    assert!(
        error.contains("requires at least one token"),
        "unexpected error: {error}"
    );
}

#[test]
fn session_step_rejects_embedding_length_overflow() {
    let model = tiny_dense_session_model();
    let mut session = Qwen35Session::new_with_capacity(&model, session_pool(), 1).unwrap();

    let error = session.step(&[], usize::MAX, &[]).unwrap_err();

    assert!(
        error.contains("embedding length overflow"),
        "unexpected error: {error}"
    );
}

#[test]
fn session_step_enforces_capacity_across_calls() {
    let error = required_token_count(1, 1, 1).unwrap_err();

    assert!(
        error.contains("requires 2 tokens; session capacity is 1"),
        "unexpected error: {error}"
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn later_gpu_failure_recomputes_from_committed_cpu_shadow_and_stays_session_local() {
    let model = tiny_q8_session_model();
    let fallback_pool = Arc::new(ComputePool::new(2));
    let mut fallback = Qwen35Session::new_with_capacity(&model, fallback_pool.clone(), 3).unwrap();
    let mut cpu =
        Qwen35Session::new_with_capacity(&model, Arc::new(ComputePool::new(2)), 3).unwrap();

    let first = fallback.embed_tokens(&[0]);
    let second = fallback.embed_tokens(&[1]);
    let third = fallback.embed_tokens(&[2]);
    fallback.step(&first, 1, &[[0; 4]]).unwrap();
    cpu.step(&first, 1, &[[0; 4]]).unwrap();
    let KvCache::F32(committed) = fallback.kv_cache() else {
        panic!("Qwen3.5 KV cache should be F32");
    };
    assert!(committed
        .k
        .iter()
        .chain(&committed.v)
        .any(|value| *value != 0.0));

    fallback.fail_gpu_once_for_test("later token failure");
    assert!(fallback.gpu_enabled_for_test());
    assert!(!crate::vulkan::gpu_broken());
    fallback_pool.clear_gpu_disabled_workers_for_test();

    let actual = fallback.step(&second, 1, &[[1; 4]]).unwrap();
    let expected = cpu.step(&second, 1, &[[1; 4]]).unwrap();

    assert_eq!(
        actual.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
        expected
            .iter()
            .copied()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
    );
    assert!(!fallback.gpu_enabled_for_test());
    assert!(!crate::vulkan::gpu_broken());
    assert_eq!(
        fallback_pool.gpu_disabled_workers_for_test() & 0b11,
        0b11,
        "the failed Q8_0 token must keep every worker out of legacy Vulkan",
    );
    fallback_pool.clear_gpu_disabled_workers_for_test();

    let actual = fallback.step(&third, 1, &[[2; 4]]).unwrap();
    let expected = cpu.step(&third, 1, &[[2; 4]]).unwrap();
    assert_eq!(
        actual.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
        expected
            .iter()
            .copied()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        fallback_pool.gpu_disabled_workers_for_test() & 0b11,
        0b11,
        "subsequent Q8_0 tokens must keep every worker out of legacy Vulkan",
    );
    assert!(!crate::vulkan::gpu_broken());
    let (KvCache::F32(actual_cache), KvCache::F32(expected_cache)) =
        (fallback.kv_cache(), cpu.kv_cache())
    else {
        panic!("Qwen3.5 KV cache should be F32");
    };
    assert_eq!(actual_cache.k, expected_cache.k);
    assert_eq!(actual_cache.v, expected_cache.v);
}

#[test]
fn session_embed_token_returns_expected_row() {
    let model = tiny_dense_session_model();
    let session = Qwen35Session::new(&model, session_pool()).unwrap();

    // token id 3 -> row offset 12 -> [12, 13, 14, 15]
    let row = session.embed_token(3).unwrap();
    assert_eq!(row, vec![12.0, 13.0, 14.0, 15.0]);

    // token id 0 -> [0, 1, 2, 3]
    let row0 = session.embed_token(0).unwrap();
    assert_eq!(row0, vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn session_embed_token_out_of_range_errors() {
    let model = tiny_dense_session_model();
    let session = Qwen35Session::new(&model, session_pool()).unwrap();

    let err = session.embed_token(8).unwrap_err();
    assert!(err.contains("out of range"), "unexpected error: {err}");
    let err2 = session.embed_token(u32::MAX).unwrap_err();
    assert!(err2.contains("out of range"));
}

#[test]
fn session_embed_tokens_concatenates_rows() {
    let model = tiny_dense_session_model();
    let session = Qwen35Session::new(&model, session_pool()).unwrap();

    let all = session.embed_tokens(&[0, 1, 2]);
    assert_eq!(all.len(), 12);
    assert_eq!(&all[0..4], &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(&all[4..8], &[4.0, 5.0, 6.0, 7.0]);
    assert_eq!(&all[8..12], &[8.0, 9.0, 10.0, 11.0]);
}

#[test]
fn session_embed_tokens_zeros_out_of_range_but_keeps_valid() {
    // Matches `app/text.rs::inject_vision_embeddings` behavior: invalid
    // token ids produce a zero row, surrounding tokens are unchanged.
    let model = tiny_dense_session_model();
    let session = Qwen35Session::new(&model, session_pool()).unwrap();

    let all = session.embed_tokens(&[0, 99, 2]);
    assert_eq!(all.len(), 12);
    assert_eq!(&all[0..4], &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(&all[4..8], &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&all[8..12], &[8.0, 9.0, 10.0, 11.0]);
}

#[test]
fn session_set_next_position_and_reset() {
    let model = tiny_dense_session_model();
    let mut session = Qwen35Session::new(&model, session_pool()).unwrap();

    assert_eq!(session.next_position(), 0);
    session.set_next_position(42);
    assert_eq!(session.next_position(), 42);

    session.reset();
    assert_eq!(session.next_position(), 0);
    // Cache should be empty (all zeros) after reset
    if let KvCache::F32(c) = session.kv_cache() {
        assert!(c.k.iter().all(|v| *v == 0.0));
        assert!(c.v.iter().all(|v| *v == 0.0));
    } else {
        panic!("Qwen3.5 KV cache should be F32");
    }
}

#[test]
fn session_step_validates_embedding_and_position_lengths() {
    let model = tiny_dense_session_model();
    let mut session = Qwen35Session::new(&model, session_pool()).unwrap();

    // embeddings.len() != n_tokens * n_embd
    let bad = vec![0.0f32; 7];
    let positions = [[0usize; 4]; 2];
    let err = session.step(&bad, 2, &positions).unwrap_err();
    assert!(err.contains("embeddings length"), "unexpected error: {err}");

    // positions.len() != n_tokens
    let good = vec![0.0f32; 8];
    let one_pos = [[0usize; 4]; 1];
    let err = session.step(&good, 2, &one_pos).unwrap_err();
    assert!(err.contains("positions length"), "unexpected error: {err}");
}
