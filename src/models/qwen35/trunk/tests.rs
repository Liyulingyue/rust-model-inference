//! Unit tests for the Qwen3.5 model + session layer.
//!
//! Position-builder tests live in `positions.rs`. Vision-encoder tests live in
//! `vision.rs`. This module covers:
//! - bit-exact `l2_norm` against the llama.cpp reference
//! - model loading (gated on `RMI_QWEN35_MODEL`)
//! - `Q8_0` quantized matmul dispatch (scalar fallback path)
//! - scratchpad sizing invariants
//! - dense-attention softmax + padded value reduction
//! - `Qwen35Session` state management + embed-lookup helpers

use crate::compute::ComputePolicy;

use super::session::required_token_count;
use super::session::Qwen35Session;
use super::*;
use crate::core::scratchpad::KvCache;
use crate::core::tensor::GGMLType;
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::PreparedRows;
use crate::ops::kernel::{Kernel, QuantizedTensor, Weight};
use crate::ops::quant::{self, BlockQ8K};
use std::sync::Arc;

fn f32_test_weight(data: Vec<f32>, n_in: usize, n_out: usize) -> Weight<'static> {
    let mut weight = Weight::from_quantized(QuantizedTensor::F32 { data, n_in, n_out });
    weight.n_in = n_in;
    weight.n_out = n_out;
    weight
}

#[test]
fn qwen35_model_rejects_edge0_architecture() {
    use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};

    struct Edge0Source(MetaValue);
    impl TensorSource for Edge0Source {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            (key == "general.architecture").then_some(&self.0)
        }
        fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
            None
        }
        fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
            None
        }
    }

    let source = Edge0Source(MetaValue::String("edge0".into()));
    let error = match Qwen35Model::from_source(&source) {
        Ok(_) => panic!("Edge0 must not construct Qwen35Model"),
        Err(error) => error,
    };
    assert!(error.contains("Edge0Model"), "{error}");
}

#[test]
fn qwen35_bf16_matmul_rounds_activations_before_dot() {
    use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};
    struct Source(TensorInfo, Vec<u8>);
    impl TensorSource for Source {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }
        fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
            Some(&self.0)
        }
        fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
            Some(&self.1)
        }
    }
    let source = Source(
        TensorInfo {
            name: "blk.0.attn_qkv.weight".into(),
            dims: vec![3, 1],
            ggml_type: GGMLType::BF16,
            offset: 0,
        },
        [1.0f32, 2.0, -1.0]
            .into_iter()
            .flat_map(|x| crate::ops::f32_to_bf16(x).to_le_bytes())
            .collect(),
    );
    let weight = super::weights::load_weight(&source, &source.0.name).unwrap();
    let input = [1.00390625, 1.01171875, 0.501953125];
    // RNE BF16 inputs are 1.0, 1.015625, 0.5; llama.cpp's scalar dot = 2.53125.
    assert_eq!(weight.matmul(&input)[0].to_bits(), 2.53125f32.to_bits());
}

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
        moe: None,
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
        moe: None,
    };
    let weight = |data: Vec<f32>, n_rows| {
        let n_in = data.len() / n_rows;
        f32_test_weight(data, n_in, n_rows)
    };
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
        ffn_gate: Some(identity()),
        ffn_up: Some(identity()),
        ffn_down: Some(identity()),
        moe: None,
    };
    Qwen35Model {
        config,
        tok_embd: f32_test_weight(Vec::new(), 0, 0),
        output_norm: vec![1.0; 2],
        output_weight: identity(),
        layers: vec![layer],
        #[cfg(feature = "vulkan")]
        gpu: None,
        cls_score: None,
        cls_score_bias: Vec::new(),
    }
}

struct RejectQ8KKernel;

impl Kernel for RejectQ8KKernel {
    fn forward_prequantized(
        &self,
        _input_q8: &[u8],
        _input_scales: &[f32],
        output: &mut [f32],
        _n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        let per_thread = n_out.div_ceil(nth);
        let start = ith * per_thread;
        let end = (start + per_thread).min(n_out);
        output[start..end].fill(0.0);
    }

    fn forward_prepared(
        &self,
        _input_f32: &[f32],
        input_q8: &[u8],
        input_scales: &[f32],
        q8_k: Option<&[BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        assert!(
            q8_k.is_none(),
            "non-K-quant weights must not receive Q8_K input"
        );
        self.forward_prequantized(input_q8, input_scales, output, n_in, n_out, ith, nth);
    }
}

fn reject_q8k_weight(n_in: usize, n_out: usize) -> Weight<'static> {
    Weight {
        kernel: Box::new(RejectQ8KKernel),
        ggml_type: GGMLType::F32,
        n_in,
        n_out,
    }
}

#[test]
fn qwen35_non_k_quant_weights_do_not_receive_q8_k_input() {
    let mut model = tiny_dense_model([1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
    let layer = &mut model.layers[0];
    layer.wq = Some(reject_q8k_weight(2, 4));
    layer.wk = Some(reject_q8k_weight(2, 2));
    layer.wv = Some(reject_q8k_weight(2, 2));
    layer.ffn_gate = Some(reject_q8k_weight(2, 2));
    layer.ffn_up = Some(reject_q8k_weight(2, 2));

    let mut scratch = Qwen35Scratchpad::new(&model.config, 1);
    let mut kv_cache = KvCache::new_f32(1, model.config.n_ctx, 2);
    let pool = ComputePool::new(1);
    let hidden = [1.0, 0.0];

    scratch.x[..2].copy_from_slice(&hidden);
    model
        .forward(1, &mut kv_cache, &mut scratch, &pool, &[[0; 4]])
        .unwrap();
}

#[test]
#[should_panic(expected = "Qwen3.5 attention input width 2 must be a multiple of 256")]
fn qwen35_k_quant_weights_reject_non_block_width() {
    let mut model = tiny_dense_model([1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
    let mut bad_wq = reject_q8k_weight(2, 4);
    bad_wq.ggml_type = GGMLType::Q4K;
    model.layers[0].wq = Some(bad_wq);

    let mut scratch = Qwen35Scratchpad::new(&model.config, 1);
    let mut kv_cache = KvCache::new_f32(1, model.config.n_ctx, 2);
    let pool = ComputePool::new(1);
    scratch.x[..2].copy_from_slice(&[1.0, 0.0]);
    let _ = model.forward(1, &mut kv_cache, &mut scratch, &pool, &[[0; 4]]);
}

#[test]
fn qwen35_gdn_norm_matches_pinned_llama_cpp_numerically() {
    // Holo4 token 248045, layer 0, head 0; llama.cpp 11fe0215 build_gdn_l2_norm.
    const INPUT_BITS: [u32; 128] = [
        0x3b314673, 0xbcd3cc3c, 0x3ac03a72, 0xbc665313, 0xbb8ce72d, 0x394956d0, 0x3cb3b1a4,
        0x3d2d63a7, 0xbd4432a2, 0x3b9e0daf, 0xbd454c6e, 0xbd0c2a00, 0xbc52b8e4, 0xbc26d947,
        0x3c52681d, 0x3c7d27ce, 0xbb04c994, 0x39186d4c, 0x3ab50494, 0xbbad7a37, 0xbc3aa816,
        0x3c709926, 0xbd023fe1, 0xbb1deced, 0xbb96f74d, 0x39702d9e, 0xbb2ec019, 0xbbb1b57e,
        0xba0a624d, 0xbc9ecf17, 0x3a3b896a, 0xbbda97be, 0xbb584933, 0xbbb7f206, 0xbcd43915,
        0xbb248636, 0xb9b8c7d6, 0x3ab2b2fb, 0xbab28506, 0xbc89bae6, 0xb98a9e66, 0x3927b1ba,
        0x3992fa6e, 0x3b1076ac, 0x3c780e4d, 0xbc3845d7, 0xbcecdf07, 0x3cc35da7, 0xbc9c43f4,
        0x3c7cfbb0, 0xba29b845, 0xb9426d83, 0xbce6e917, 0xba8513fd, 0x3cf09e99, 0xbac795da,
        0x3c9627b3, 0xbc158569, 0xba1e227b, 0xbcd4caee, 0xb8ea0558, 0x3a5312fb, 0xbca3eb3e,
        0x3ba28167, 0xbb6e3b70, 0x38c25258, 0xbc3734a9, 0xbc974ab4, 0xbc626a45, 0xbb8f8e97,
        0x3c106b26, 0xbc20e625, 0xbc6b3698, 0x3912b427, 0x3d252b07, 0xbc346737, 0xbc05008f,
        0xbd087b49, 0xbc4bbc50, 0x3d0e6711, 0x3b038c3a, 0x3b885cbc, 0xbcb6be5a, 0x3cb44f19,
        0x3bc259c5, 0xbca1e5f7, 0x3906a7a0, 0x3b41c46a, 0xbd06228b, 0xbce3e0cf, 0xbce3c22c,
        0xbc2d1f87, 0xbc73902f, 0xbc08c8f7, 0xba542758, 0x3bab3ad3, 0xb8291ee9, 0x3c19f953,
        0xbc866fcc, 0xbd1bcd0f, 0x3c19aeb1, 0x3cffe3a0, 0x3881396a, 0xbcc819f6, 0xba2b12d6,
        0x3c987165, 0xbe587097, 0xbc284abb, 0xbcd8690d, 0x3d439a11, 0xbb39e030, 0x3f8c382d,
        0x3bb31f34, 0xbca198c5, 0x3ab43a43, 0xb6a90963, 0xbbfe4bc2, 0x3bc05db5, 0x3b5d6787,
        0xbc61c462, 0x3d052fdc, 0xb9d086cd, 0xbc657353, 0xbc35c2d0, 0xbcc4778b, 0x3ae26c27,
        0xba2aede9, 0x3c0b4aa0,
    ];
    const EXPECTED_BITS: [u32; 128] = [
        0x3b1c8cf5, 0xbcbb098c, 0x3aa9c16f, 0xbc4b65ec, 0xbb78dc75, 0x3931cd22, 0x3c9eafc2,
        0x3d191e75, 0xbd2d42d3, 0x3b8b9375, 0xbd2e3bad, 0xbcf78e57, 0xbc3a1665, 0xbc1357d3,
        0x3c39cf10, 0x3c5f8f5e, 0xbaea8708, 0x39069b76, 0x3a9fdb13, 0xbb993262, 0xbc24d5d3,
        0x3c547885, 0xbce60b8b, 0xbb0b7688, 0xbb855130, 0x39541990, 0xbb1a522a, 0xbb9cef04,
        0xb9f4697c, 0xbc8c3e41, 0x3a259cd0, 0xbbc109b3, 0xbb3f0031, 0xbba270e8, 0xbcbb69ab,
        0xbb114a54, 0xb9a32db8, 0x3a9dcedf, 0xba9da64a, 0xbc7341d2, 0xb974d3a1, 0x391416f9,
        0x3981cbaa, 0x3aff2655, 0x3c5b0e7c, 0xbc22baec, 0xbcd12df1, 0x3cac86be, 0xbc89ff3d,
        0x3c5f6867, 0xba15e0e5, 0xb92bb2ad, 0xbccbea66, 0xba6b0a74, 0x3cd47d56, 0xbab040a9,
        0x3c8499db, 0xbc040a8a, 0xba0ba5d3, 0xbcbbea78, 0xb8cea990, 0x3a3a65f4, 0xbc90c179,
        0x3b8f81f0, 0xbb52619f, 0x38ab9aaf, 0xbc21c9ad, 0xbc859ad7, 0xbc47f21e, 0xbb7d8c6e,
        0x3bff11fb, 0xbc0e16c2, 0xbc4fb721, 0x39018d9a, 0x3d11dbe0, 0xbc1f501b, 0xbbeae823,
        0xbcf10d53, 0xbc33eaea, 0x3cfb827b, 0x3ae85687, 0x3b70d75e, 0xbca16134, 0x3c9f3acf,
        0x3baba13e, 0xbc8ef8ac, 0x38edd35a, 0x3b2b1d59, 0xbcece84e, 0xbcc93cdf, 0xbcc921d0,
        0xbc18e24c, 0xbc5716d3, 0xbbf19686, 0xba3b5a02, 0x3b973642, 0xb8155976, 0x3c07f931,
        0xbc6d70c0, 0xbd09963f, 0x3c07b748, 0x3ce1f95f, 0x38643bfa, 0xbcb0b553, 0xba1712f2,
        0x3c869f15, 0xbe3f22fa, 0xbc149e17, 0xbcbf1c52, 0x3d2cbc18, 0xbb24254c, 0x3f77a760,
        0x3b9e2e71, 0xbc8eb480, 0x3a9f2869, 0xb6954675, 0xbbe09130, 0x3ba9e093, 0x3b438555,
        0xbc475fa0, 0x3ceb3bad, 0xb9b82605, 0xbc4aa055, 0xbc208311, 0xbcad7fad, 0x3ac7f3c8,
        0xba16f256, 0x3bf603d1,
    ];

    let mut actual = INPUT_BITS.map(f32::from_bits);
    super::util::l2_norm(&mut actual, 1e-6);
    for (actual, expected) in actual.into_iter().zip(EXPECTED_BITS.map(f32::from_bits)) {
        assert!((actual - expected).abs() < 2e-7, "{actual} != {expected}");
    }
}

#[test]
fn qwen35_gdn_norm_adds_eps_to_sum_of_squares() {
    let mut tiny = [1e-8f32; 128];
    super::util::l2_norm(&mut tiny, 1e-6);
    for value in tiny {
        assert!((value - 1e-5).abs() < 1e-10, "{value}");
    }
    let mut zero = [0.0; 128];
    super::util::l2_norm(&mut zero, 1e-6);
    assert_eq!(zero, [0.0; 128]);
}

#[test]
#[ignore = "requires RMI_QWEN35_MODEL and RMI_QWEN35_PROMPT_TOKENS"]
fn qwen35moe_single_token_trace() {
    // Runs exactly one token so the MoE checkpoints can be joined against the
    // oracle on (name, layer, step=0).
    //
    // The tokens come from an env var rather than a prompt string on purpose.
    // The oracle prints the ids it tokenised and then processes that many
    // tokens; re-tokenising the same text here could add a BOS or split
    // differently and quietly compare two different tokens. Handing both sides
    // the same ids is the only way "one token" means one token on both.
    let path = std::env::var("RMI_QWEN35_MODEL").expect("RMI_QWEN35_MODEL must be set");
    let ids = std::env::var("RMI_QWEN35_PROMPT_TOKENS")
        .expect("RMI_QWEN35_PROMPT_TOKENS must be a comma-separated list");
    let tokens: Vec<u32> = ids
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.parse().expect("token id must be a u32"))
        .collect();
    assert!(
        !tokens.is_empty(),
        "RMI_QWEN35_PROMPT_TOKENS must name at least one token"
    );

    let source =
        crate::open_model_source(std::path::Path::new(&path), crate::ComponentRole::Llm).unwrap();
    let mut model = Qwen35Model::from_source(source.as_ref()).unwrap();
    let n_ctx = model.config.n_ctx.min(tokens.len() + 1);
    let pool = session_pool();
    let mut session =
        Qwen35Session::new_with_prefill_batch_size(&mut model, n_ctx, tokens.len(), pool).unwrap();
    let positions: Vec<[usize; 4]> = (0..tokens.len()).map(|i| [i; 4]).collect();
    // The trace is emitted from moe.rs during this call; nothing to assert here
    // beyond it succeeding, because the comparison is done offline by
    // tools/oracle/qwen35/check_moe.py against the oracle's manifest.
    session.step_with_tokens(&tokens, &positions).unwrap();
}

#[test]
#[ignore = "requires RMI_QWEN35_MODEL"]
fn qwen38_q4_0_model_metadata_and_load() {
    let path = std::env::var("RMI_QWEN35_MODEL").expect("RMI_QWEN35_MODEL must be set");
    let source =
        crate::open_model_source(std::path::Path::new(&path), crate::ComponentRole::Llm).unwrap();
    let config = Qwen35Config::from_source(source.as_ref()).unwrap();
    assert_eq!(
        (config.n_layer, config.n_layer_impl(), config.n_nextn),
        (65, 64, 1)
    );
    assert_eq!(
        (config.n_embd, config.n_head, config.n_head_kv),
        (5120, 24, 4)
    );
    assert_eq!((config.key_length, config.value_length), (256, 256));
    assert_eq!((config.ssm_d_conv, config.ssm_d_state), (4, 128));
    assert_eq!(
        (config.ssm_n_group, config.ssm_dt_rank, config.ssm_d_inner),
        (16, 48, 6144)
    );
    assert_eq!(
        source.tensor_info("token_embd.weight").unwrap().ggml_type,
        GGMLType::Q4_0
    );
    let model = Qwen35Model::from_source(source.as_ref()).unwrap();
    assert_eq!(model.tok_embd.n_out, 248320);
    assert_eq!(model.layers.len(), 64);
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
        None,
        #[cfg(feature = "parity-trace")]
        false,
    );

    for (got, expected) in [
        scratch.score_buf[0].to_bits(),
        scratch.score_buf[1].to_bits(),
    ]
    .into_iter()
    .zip([0x3f25_1fe0, 0x3eb5_c040])
    {
        assert_eq!(got, expected);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn qwen35_dense_attention_value_uses_ggml_padded_reduction() {
    let model = tiny_dense_model([1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
    let n_tokens = 18;
    for capacity in [n_tokens, model.config.n_ctx] {
        let mut scratch = Qwen35Scratchpad::new(&model.config, n_tokens);
        let mut kv_cache = crate::core::scratchpad::KvCache::new_f32(1, capacity, 2);
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
            None,
            #[cfg(feature = "parity-trace")]
            false,
        );

        assert_eq!(output[(n_tokens - 1) * 2].to_bits(), 0x3f00_0000);
    }
}

// ------------------------------------------------------------------
// Qwen35Session tests
// ------------------------------------------------------------------

/// Tiny dense-only Qwen35Model used to construct a Session.
/// `tok_embd` row i = [4*i+0, 4*i+1, 4*i+2, 4*i+3]; n_embd=4, vocab=8.
fn tiny_dense_session_model_with_embedding(
    tok_embd: Weight<'static>,
    n_embd: usize,
    vocab_size: usize,
) -> Qwen35Model<'static> {
    let config = Qwen35Config {
        n_nextn: 0,
        n_embd,
        n_layer: 1,
        n_head: 1,
        n_head_kv: 1,
        n_ff: n_embd,
        n_ctx: 16,
        vocab_size,
        rope_freq_base: 1_000_000.0,
        norm_eps: 1e-6,
        rope_dimension_count: n_embd,
        rope_dimension_sections: [0; 4],
        ssm_d_conv: 1,
        ssm_d_state: 2,
        ssm_n_group: 1,
        ssm_dt_rank: 1,
        ssm_d_inner: 2,
        full_attention_interval: 1,
        is_recurrent: vec![false],
        key_length: n_embd,
        value_length: n_embd,
        moe: None,
    };
    let mk_weight = |n_out| f32_test_weight(vec![1.0; n_embd * n_out], n_embd, n_out);
    let layer = Qwen35LayerWeights {
        attn_norm: vec![1.0; n_embd],
        attn_post_norm: vec![1.0; n_embd],
        wq: Some(mk_weight(2 * n_embd)),
        wk: Some(mk_weight(n_embd)),
        wv: Some(mk_weight(n_embd)),
        wo: Some(mk_weight(n_embd)),
        attn_q_norm: Some(vec![1.0; n_embd]),
        attn_k_norm: Some(vec![1.0; n_embd]),
        wqkv: None,
        wqkv_gate: None,
        ssm_conv1d: None,
        ssm_dt: None,
        ssm_a: None,
        ssm_beta: None,
        ssm_alpha: None,
        ssm_norm: None,
        ssm_out: None,
        ffn_gate: Some(mk_weight(n_embd)),
        ffn_up: Some(mk_weight(n_embd)),
        ffn_down: Some(mk_weight(n_embd)),
        moe: None,
    };
    Qwen35Model {
        config,
        tok_embd,
        output_norm: vec![1.0; n_embd],
        output_weight: mk_weight(vocab_size),
        layers: vec![layer],
        #[cfg(feature = "vulkan")]
        gpu: None,
        cls_score: None,
        cls_score_bias: Vec::new(),
    }
}

pub(crate) fn tiny_dense_session_model() -> Qwen35Model<'static> {
    let tok_embd = f32_test_weight((0..32).map(|i| i as f32).collect(), 4, 8);
    tiny_dense_session_model_with_embedding(tok_embd, 4, 8)
}

#[test]
fn qwen35_q4_0_embedding_lookup_is_row_local_and_checked() {
    let block = |scale: f32, nibble: u8| {
        let mut bytes = crate::ops::f32_to_f16(scale).to_le_bytes().to_vec();
        bytes.extend(std::iter::repeat_n(nibble | (nibble << 4), 16));
        bytes
    };
    let bytes = Box::leak([block(0.5, 8), block(0.5, 10)].concat().into_boxed_slice());
    let embedding =
        Weight::from_quantized(QuantizedTensor::from_bytes(bytes, GGMLType::Q4_0, 32, 2));
    let model = tiny_dense_session_model_with_embedding(embedding, 32, 2);

    assert_eq!(model.embed_tokens(&[1]).unwrap(), vec![1.0; 32]);
    assert!(model.embed_tokens(&[2]).unwrap_err().contains("vocab=2"));
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
        moe: None,
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
        ffn_gate: Some(q8_weight(256)),
        ffn_up: Some(q8_weight(256)),
        ffn_down: Some(q8_weight(256)),
        moe: None,
    };
    Qwen35Model {
        config,
        tok_embd: f32_test_weight(
            (0..32 * 256)
                .map(|index| (index % 17 + 1) as f32 * 0.01)
                .collect(),
            256,
            32,
        ),
        output_norm: vec![1.0; 256],
        output_weight: q8_weight(32),
        layers: vec![layer],
        #[cfg(feature = "vulkan")]
        gpu: None,
        cls_score: None,
        cls_score_bias: Vec::new(),
    }
}

fn session_pool() -> Arc<ComputePool> {
    Arc::new(ComputePool::new(1))
}

#[test]
fn qwen35_scratch_rows_are_bounded_by_prefill_batch_size() {
    let mut model = tiny_dense_session_model();
    let width = model.config.n_embd;
    let session =
        Qwen35Session::new_with_prefill_batch_size(&mut model, 16, 3, session_pool()).unwrap();
    assert_eq!(session.scratch().x.len(), 3 * width);
}

#[test]
fn qwen35_scratch_is_independent_of_prompt_length_and_mrope_axes_are_batched() {
    let embedding = f32_test_weight((0..64).map(|i| i as f32 / 10.0).collect(), 8, 8);
    let mut model = tiny_dense_session_model_with_embedding(embedding, 8, 8);
    model.config.n_ctx = 128;
    model.config.rope_dimension_sections = [1; 4];
    let positions = [[2, 3, 5, 7], [11, 13, 17, 19], [23, 29, 31, 37]];
    let mut run = |batch| {
        let mut session =
            Qwen35Session::new_with_prefill_batch_size(&mut model, 128, batch, session_pool())
                .unwrap();
        let logits = session.step_with_tokens(&[1, 2, 3], &positions).unwrap();
        let before = session.scratch_bytes();
        let snapshot = snapshot_qwen35_recurrent_state(&session);
        session.step_with_tokens(&[1; 64], &[[40; 4]; 64]).unwrap();
        assert_eq!(session.scratch_bytes(), before);
        (
            logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            snapshot,
        )
    };
    assert_eq!(run(64), run(1));
}

fn greedy_decode_three(session: &mut Qwen35Session<'_, '_>, mut logits: Vec<f32>) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(3);
    for _ in 0..3 {
        let token = logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0 as u32;
        tokens.push(token);
        let position = session.next_position();
        logits = session
            .step_with_tokens(&[token], &[[position; 4]])
            .unwrap();
    }
    tokens
}

fn run_qwen35_dense_fixture(
    prompt_len: usize,
    batch_size: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut model = tiny_dense_session_model();
    model.config.n_ctx = prompt_len + 3;
    let mut session = Qwen35Session::new_with_prefill_batch_size(
        &mut model,
        prompt_len + 3,
        batch_size,
        session_pool(),
    )
    .unwrap();
    let tokens = (0..prompt_len)
        .map(|index| (index % 8) as u32)
        .collect::<Vec<_>>();
    let positions = (0..prompt_len).map(|index| [index; 4]).collect::<Vec<_>>();
    let logits = session.step_with_tokens(&tokens, &positions).unwrap();
    let KvCache::F32(cache) = session.kv_cache() else {
        panic!("Qwen3.5 KV cache must be F32");
    };
    let prompt_logits = logits.iter().map(|value| value.to_bits()).collect();
    let prompt_kv = cache
        .k
        .iter()
        .chain(&cache.v)
        .map(|value| value.to_bits())
        .collect();
    let generated = greedy_decode_three(&mut session, logits);
    (prompt_logits, prompt_kv, generated)
}

#[test]
fn qwen35_dense_prefill_matches_batch_one_bits() {
    for len in [1, 2, 3, 63, 64, 65, 127, 128] {
        let expected = run_qwen35_dense_fixture(len, 1);
        for batch_size in [2, 3, 63, 64, 65, 127, 128] {
            assert_eq!(
                run_qwen35_dense_fixture(len, batch_size),
                expected,
                "len={len} batch={batch_size}"
            );
        }
    }
}

fn tiny_recurrent_session_model() -> Qwen35Model<'static> {
    let mut model = tiny_dense_session_model();
    model.config.is_recurrent = vec![true];
    model.config.ssm_d_conv = 2;
    let layer = &mut model.layers[0];
    let weight = |n_in, n_out| f32_test_weight(vec![0.05; n_in * n_out], n_in, n_out);
    layer.wqkv = Some(weight(4, 6));
    layer.wqkv_gate = Some(weight(4, 2));
    layer.ssm_beta = Some(weight(4, 1));
    layer.ssm_alpha = Some(weight(4, 1));
    layer.ssm_conv1d = Some(vec![0.5; 12]);
    layer.ssm_dt = Some(vec![0.1]);
    layer.ssm_a = Some(vec![-0.25]);
    layer.ssm_norm = Some(vec![1.0; 2]);
    layer.ssm_out = Some(weight(2, 4));
    model
}

fn tiny_mixed_session_model() -> Qwen35Model<'static> {
    let mut model = tiny_dense_session_model();
    let mut recurrent = tiny_recurrent_session_model();
    model.config.n_layer = 2;
    model.config.ssm_d_conv = 2;
    model.config.rope_dimension_sections = [1, 1, 0, 0];
    model.config.is_recurrent = vec![false, true];
    model.layers.push(recurrent.layers.remove(0));
    model
}

#[cfg(all(test, feature = "vulkan"))]
fn run_qwen35_cpu_fixture(prompt_len: usize, batch_size: usize) -> Vec<u32> {
    let _cpu = ComputePool::disable_gpu_matmul_for_scope();
    run_qwen35_chunk_fixture(prompt_len, batch_size, None, ComputePolicy::Cpu)
}

#[cfg(all(test, feature = "vulkan"))]
fn run_qwen35_gpu_failure_fixture(
    prompt_len: usize,
    batch_size: usize,
    fail_after_row: usize,
) -> Vec<u32> {
    crate::ops::enable_gpu();
    assert!(crate::ops::get_vulkan_context().is_some());
    run_qwen35_chunk_fixture(
        prompt_len,
        batch_size,
        Some(fail_after_row),
        ComputePolicy::Auto,
    )
}

#[cfg(test)]
fn run_qwen35_chunk_fixture(
    prompt_len: usize,
    batch_size: usize,
    fail_after_row: Option<usize>,
    policy: ComputePolicy,
) -> Vec<u32> {
    let mut session = mixed_fixture_session(prompt_len + 3, batch_size, policy);
    #[cfg(feature = "vulkan")]
    if let Some(row) = fail_after_row {
        assert!(session.gpu_enabled_for_test());
        session.fail_gpu_after_row_for_test(row);
    }
    #[cfg(not(feature = "vulkan"))]
    assert!(fail_after_row.is_none());
    let tokens = (0..prompt_len).map(|i| (i % 8) as u32).collect::<Vec<_>>();
    let positions = (0..prompt_len)
        .map(|i| [i, i + 1, i + 2, i + 3])
        .collect::<Vec<_>>();
    let logits = session.step_with_tokens(&tokens, &positions).unwrap();
    let mut result = vec![session.processed_tokens() as u32];
    result.extend(logits.iter().map(|v| v.to_bits()));
    let KvCache::F32(cache) = session.kv_cache() else {
        unreachable!()
    };
    result.extend(
        cache
            .k
            .iter()
            .chain(&cache.v)
            .chain(session.scratch().conv_states.iter().flatten())
            .chain(session.scratch().ssm_states.iter().flatten())
            .map(|v| v.to_bits()),
    );
    result.extend(greedy_decode_three(&mut session, logits));
    result
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires a Vulkan device"]
fn qwen35_gpu_chunk_failure_restarts_from_chunk_base() {
    let expected = run_qwen35_cpu_fixture(5, 5);
    let actual = run_qwen35_gpu_failure_fixture(5, 5, 2);
    assert_eq!(actual, expected);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires a Vulkan device"]
fn qwen35_cpu_scope_skips_full_session_gpu() {
    crate::ops::enable_gpu();
    assert!(crate::ops::get_vulkan_context().is_some());
    let _cpu = ComputePool::disable_gpu_matmul_for_scope();
    let session = mixed_fixture_session(8, 5, ComputePolicy::Auto);
    assert!(
        !session.gpu_enabled_for_test(),
        "CPU scope must exclude the full Vulkan executor too"
    );
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires a Vulkan device"]
fn qwen35_gpu_chunk_failure_preserves_a_committed_gpu_prefix() {
    crate::ops::enable_gpu();
    let tokens = [0, 1, 2, 3, 4, 5, 6, 7];
    let positions = (0..8).map(|i| [i, i + 1, i + 2, i + 3]).collect::<Vec<_>>();
    let mut actual = mixed_fixture_session(11, 3, ComputePolicy::Auto);
    let mut expected = mixed_fixture_session(11, 3, ComputePolicy::Auto);
    for session in [&mut actual, &mut expected] {
        session
            .step_with_tokens(&tokens[..3], &positions[..3])
            .unwrap();
    }
    let prefix = snapshot_qwen35_recurrent_state(&actual);
    assert_eq!(prefix, snapshot_qwen35_recurrent_state(&expected));
    actual.fail_gpu_after_row_for_test(2);
    expected.fail_gpu_once_for_test("retry from the same committed GPU prefix");
    let a = actual
        .step_with_tokens(&tokens[3..], &positions[3..])
        .unwrap();
    let b = expected
        .step_with_tokens(&tokens[3..], &positions[3..])
        .unwrap();
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        snapshot_qwen35_recurrent_state(&actual),
        snapshot_qwen35_recurrent_state(&expected)
    );
    assert_eq!(
        greedy_decode_three(&mut actual, a),
        greedy_decode_three(&mut expected, b)
    );
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires a Vulkan device"]
fn qwen35_vulkan_chunks_match_token_bits_and_submission_count() {
    crate::ops::enable_gpu();
    let context = crate::ops::get_vulkan_context().unwrap();
    let baseline = run_qwen35_chunk_fixture(5, 1, None, ComputePolicy::Auto);
    for batch in [2, 3, 5] {
        let before = context.submission_count();
        let actual = run_qwen35_chunk_fixture(5, batch, None, ComputePolicy::Auto);
        assert_eq!(actual, baseline, "batch={batch}");
        assert_eq!(
            context.submission_count() - before,
            2 * (5usize.div_ceil(batch) as u64 + 3)
        );
    }
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires a Vulkan device"]
fn qwen35_vulkan_dense_chunks_preserve_four_mrope_axes() {
    crate::ops::enable_gpu();
    let run = |batch, positions: &[[usize; 4]]| {
        let embedding = f32_test_weight((0..64).map(|i| i as f32 / 10.0).collect(), 8, 8);
        let mut model = tiny_dense_session_model_with_embedding(embedding, 8, 8);
        model.config.rope_dimension_sections = [1; 4];
        let mut session =
            Qwen35Session::new_with_prefill_batch_size(&mut model, 6, batch, session_pool())
                .unwrap();
        assert!(session.gpu_enabled_for_test());
        let logits = session.step_with_tokens(&[1, 2, 3], positions).unwrap();
        let KvCache::F32(cache) = session.kv_cache() else {
            unreachable!()
        };
        logits
            .iter()
            .chain(&cache.k)
            .chain(&cache.v)
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    let positions = [[2, 3, 5, 7], [11, 13, 17, 19], [23, 29, 31, 37]];
    let baseline = run(1, &positions);
    assert_eq!(run(3, &positions), baseline);
    for axis in 0..4 {
        let mut changed = positions;
        changed[1][axis] += 1;
        assert_ne!(run(3, &changed), baseline, "mRoPE axis {axis} was ignored");
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn qwen35_forced_vulkan_failure_preserves_state_without_cpu_retry() {
    let _cpu = ComputePool::disable_gpu_matmul_for_scope();
    let mut session = mixed_fixture_session(8, 5, ComputePolicy::Cpu);
    session.step_with_tokens(&[1], &[[0; 4]]).unwrap();
    let before = snapshot_qwen35_recurrent_state(&session);
    session.force_vulkan_policy_for_test();
    session.fail_gpu_once_for_test("forced Vulkan failure");
    session.fail_cpu_chunk_after_row_for_test(0);
    let input = [[1, 2, 3, 4], [5, 6, 7, 8]];
    let error = session.step_with_tokens(&[2, 3], &input).unwrap_err();
    assert_eq!(error, "forced Vulkan failure");
    assert_eq!(snapshot_qwen35_recurrent_state(&session), before);
    assert!(session
        .step_with_tokens(&[2, 3], &input)
        .unwrap_err()
        .contains("unavailable"));
    assert_eq!(snapshot_qwen35_recurrent_state(&session), before);
}

#[cfg(feature = "vulkan")]
#[test]
fn qwen35_gpu_and_cpu_chunk_failure_preserves_committed_state_and_both_errors() {
    let _cpu = ComputePool::disable_gpu_matmul_for_scope();
    let mut session = mixed_fixture_session(8, 5, ComputePolicy::Cpu);
    session.step_with_tokens(&[1], &[[0; 4]]).unwrap();
    let before = snapshot_qwen35_recurrent_state(&session);
    session.fail_gpu_once_for_test("Vulkan chunk error");
    session.fail_cpu_chunk_after_row_for_test(2);
    let error = session
        .step_with_tokens(&[2, 3, 4, 5, 6], &[[1; 4], [2; 4], [3; 4], [4; 4], [5; 4]])
        .unwrap_err();
    assert!(
        error.contains("CPU chunk failure")
            && error.contains("original Vulkan error: Vulkan chunk error"),
        "{error}"
    );
    assert_eq!(snapshot_qwen35_recurrent_state(&session), before);
}

#[test]
fn qwen35_cpu_failure_stops_inside_recurrent_scan() {
    let mut model = tiny_mixed_session_model();
    let mut scratch = Qwen35Scratchpad::new(&model.config, 5);
    let embeddings = model.embed_tokens(&[0, 1, 2, 3, 4]).unwrap();
    let positions = [[0; 4], [1; 4], [2; 4], [3; 4], [4; 4]];
    let mut run = |rows, failure| {
        let mut cache = KvCache::new_f32(2, 8, 4);
        scratch.x[..embeddings.len()].copy_from_slice(&embeddings);
        let mut conv = scratch.conv_states.clone();
        let mut ssm = scratch.ssm_states.clone();
        super::forward::set_cpu_scan_failure(failure);
        let result = model.forward_chunk(
            rows,
            0,
            &mut cache,
            &mut scratch,
            &mut conv,
            &mut ssm,
            &session_pool(),
            &positions[..rows],
        );
        super::forward::set_cpu_scan_failure(None);
        (
            result,
            ssm.into_iter()
                .flatten()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
        )
    };
    let expected = run(3, None);
    assert!(expected.0.is_ok());
    let actual = run(5, Some(2));
    assert!(actual.0.unwrap_err().contains("row 2"));
    assert_eq!(actual.1, expected.1);
}

fn recurrent_fixture_session(
    capacity: usize,
    batch_size: usize,
) -> Qwen35Session<'static, 'static> {
    let model = Box::leak(Box::new(tiny_recurrent_session_model()));
    Qwen35Session::new_with_compute(
        model,
        capacity,
        batch_size,
        session_pool(),
        crate::compute::ComputePolicy::Cpu,
    )
    .unwrap()
}

fn mixed_fixture_session(
    capacity: usize,
    batch_size: usize,
    policy: ComputePolicy,
) -> Qwen35Session<'static, 'static> {
    let model = Box::leak(Box::new(tiny_mixed_session_model()));
    model.config.n_ctx = capacity;
    Qwen35Session::new_with_compute(model, capacity, batch_size, session_pool(), policy).unwrap()
}

#[test]
fn qwen35_mixed_boundaries_preserve_prompt_and_decode_state() {
    for len in [1, 2, 3, 63, 64, 65, 127, 128] {
        let baseline = run_qwen35_chunk_fixture(len, 1, None, ComputePolicy::Cpu);
        assert_eq!(
            run_qwen35_chunk_fixture(len, 64, None, ComputePolicy::Cpu),
            baseline,
            "len={len}"
        );
    }
}

fn snapshot_qwen35_recurrent_state(
    session: &Qwen35Session<'_, '_>,
) -> (usize, usize, Vec<u32>, Vec<u32>, Vec<u32>) {
    let committed = session.next_position();
    let KvCache::F32(cache) = session.kv_cache() else {
        panic!("expected F32 KV");
    };
    let layers = session.config().n_layer_impl();
    let stride = session.config().n_embd_gqa();
    let capacity = cache.k.len() / layers / stride;
    let layer_len = capacity * stride;
    let mut kv = Vec::new();
    for layer in 0..layers {
        let base = layer * layer_len;
        kv.extend(
            cache.k[base..base + committed * stride]
                .iter()
                .map(|value| value.to_bits()),
        );
        for dimension in 0..stride {
            let column = base + dimension * capacity;
            kv.extend(
                cache.v[column..column + committed]
                    .iter()
                    .map(|value| value.to_bits()),
            );
        }
    }
    let conv = session
        .scratch()
        .conv_states
        .iter()
        .flatten()
        .map(|value| value.to_bits())
        .collect();
    let ssm = session
        .scratch()
        .ssm_states
        .iter()
        .flatten()
        .map(|value| value.to_bits())
        .collect();
    (session.processed_tokens(), committed, kv, conv, ssm)
}

fn run_qwen35_recurrent_fixture(
    prompt_len: usize,
    batch_size: usize,
) -> (
    Vec<u32>,
    (usize, usize, Vec<u32>, Vec<u32>, Vec<u32>),
    Vec<u32>,
) {
    let mut session = recurrent_fixture_session(8, batch_size);
    let tokens = (0..prompt_len)
        .map(|index| (index % 8) as u32)
        .collect::<Vec<_>>();
    let positions = (0..prompt_len).map(|index| [index; 4]).collect::<Vec<_>>();
    let logits = session.step_with_tokens(&tokens, &positions).unwrap();
    let prompt_logits = logits.iter().map(|value| value.to_bits()).collect();
    let prompt_state = snapshot_qwen35_recurrent_state(&session);
    let generated = greedy_decode_three(&mut session, logits);
    (prompt_logits, prompt_state, generated)
}

#[test]
fn qwen35_recurrent_prefill_and_state_match_batch_one_bits() {
    let baseline = run_qwen35_recurrent_fixture(5, 1);
    for batch_size in [2, 3, 5] {
        assert_eq!(run_qwen35_recurrent_fixture(5, batch_size), baseline);
    }
}

#[test]
fn qwen35_failed_chunk_keeps_dense_and_recurrent_state() {
    let mut session = mixed_fixture_session(8, 4, ComputePolicy::Cpu);
    session.step_with_tokens(&[1], &[[0; 4]]).unwrap();
    let before = snapshot_qwen35_recurrent_state(&session);
    session.fail_cpu_chunk_after_row_for_test(1);
    assert!(session
        .step_with_tokens(&[2, 3, 4], &[[1; 4], [2; 4], [3; 4]])
        .is_err());
    assert_eq!(snapshot_qwen35_recurrent_state(&session), before);

    let actual = session.step_with_tokens(&[2], &[[1; 4]]).unwrap();
    let actual_state = snapshot_qwen35_recurrent_state(&session);
    let mut baseline = mixed_fixture_session(8, 4, ComputePolicy::Cpu);
    baseline.step_with_tokens(&[1], &[[0; 4]]).unwrap();
    let expected = baseline.step_with_tokens(&[2], &[[1; 4]]).unwrap();
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(actual_state, snapshot_qwen35_recurrent_state(&baseline));
}

#[test]
fn qwen35_nonfinite_input_commits_no_state() {
    let mut session = mixed_fixture_session(4, 2, ComputePolicy::Cpu);
    let before = snapshot_qwen35_recurrent_state(&session);
    assert!(session.step(&[f32::NAN; 4], 1, &[[0; 4]]).is_err());
    assert_eq!(snapshot_qwen35_recurrent_state(&session), before);
}

#[test]
fn qwen35_dense_chunk_finite_check_covers_key_and_value_layouts() {
    let model = tiny_mixed_session_model();
    let cfg = &model.config;
    let capacity = 4;
    let stride = cfg.n_embd_gqa();
    let mut cache = KvCache::new_f32(cfg.n_layer_impl(), capacity, stride);
    assert!(super::session::dense_kv_chunk_is_finite(
        &cache, cfg, capacity, 1, 2
    ));

    let KvCache::F32(values) = &mut cache else {
        unreachable!();
    };
    values.k[stride] = f32::NAN;
    assert!(!super::session::dense_kv_chunk_is_finite(
        &cache, cfg, capacity, 1, 2
    ));
    let KvCache::F32(values) = &mut cache else {
        unreachable!();
    };
    values.k[stride] = 0.0;
    values.v[1] = f32::INFINITY;
    assert!(!super::session::dense_kv_chunk_is_finite(
        &cache, cfg, capacity, 1, 2
    ));
}

#[test]
fn prepared_rows_bytes_keeps_peak_allocation_after_a_short_tail() {
    let mut prepared = PreparedRows::new(4, 32);
    prepared
        .prepare(&vec![1.0; 4 * 32], 4, 32, true, false)
        .unwrap();
    let peak = prepared.bytes();
    prepared
        .prepare(&vec![1.0; 32], 1, 32, true, false)
        .unwrap();
    assert_eq!(prepared.bytes(), peak);
}

#[cfg(feature = "parity-trace")]
#[test]
#[ignore]
fn qwen35_trace_two_tokens_child() {
    let batch_size = std::env::var("RMI_TEST_TRACE_BATCH_SIZE")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let mut session = mixed_fixture_session(2, batch_size, ComputePolicy::Cpu);
    if std::env::var_os("RMI_TEST_TRACE_FAIL_ROW").is_some() {
        session.fail_cpu_chunk_after_row_for_test(1);
        assert!(
            session
                .step_with_tokens(&[1, 2], &[[0; 4], [1; 4]])
                .is_err(),
            "trace must execute a real multi-row chunk"
        );
        return;
    }
    session
        .step_with_tokens(&[1, 2], &[[0; 4], [1; 4]])
        .unwrap();
}

#[cfg(feature = "parity-trace")]
#[test]
fn qwen35_trace_metadata_is_independent_of_prefill_batch_size() {
    use std::process::Command;

    let mut traces = Vec::new();
    for batch_size in [1, 64] {
        let trace = std::env::temp_dir().join(format!(
            "rmi-qwen35-prefill-trace-{}-{batch_size}.jsonl",
            std::process::id()
        ));
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "models::qwen35::trunk::tests::qwen35_trace_two_tokens_child",
            ])
            .env("RMI_PARITY_TRACE", &trace)
            .env("RMI_TEST_TRACE_BATCH_SIZE", batch_size.to_string())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let values = std::fs::read_to_string(&trace)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        traces.push(
            values
                .iter()
                .map(|value| {
                    (
                        value["name"].clone(),
                        value["layer"].clone(),
                        value["shape"].clone(),
                        std::fs::read(value["binary_path"].as_str().unwrap()).unwrap(),
                    )
                })
                .collect::<Vec<_>>(),
        );
        for record in values {
            std::fs::remove_file(record["binary_path"].as_str().unwrap()).unwrap();
        }
        std::fs::remove_file(trace).unwrap();
    }
    assert_eq!(traces[1], traces[0]);
    let trace = std::env::temp_dir().join(format!(
        "rmi-qwen35-trace-chunk-{}.jsonl",
        std::process::id()
    ));
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "models::qwen35::trunk::tests::qwen35_trace_two_tokens_child",
        ])
        .env("RMI_PARITY_TRACE", trace)
        .env("RMI_TEST_TRACE_BATCH_SIZE", "64")
        .env("RMI_TEST_TRACE_FAIL_ROW", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn session_new_initializes_state_and_allocates_cache() {
    let mut model = tiny_dense_session_model();
    let pool = session_pool();
    let session = Qwen35Session::new(&mut model, 4, pool.clone()).unwrap();

    assert_eq!(session.next_position(), 0);
    assert_eq!(session.config().vocab_size, 8);
    assert_eq!(session.config().n_embd, 4);
    let cfg = session.config().clone();
    assert_eq!(session.model().config.n_layer, cfg.n_layer);
    assert_eq!(session.pool().n_threads(), 1);
    assert_eq!(session.scratch().x.len(), 4 * 4);
}

#[test]
fn session_capacity_must_fit_model_context() {
    let mut model = tiny_dense_session_model();

    assert!(Qwen35Session::new(&mut model, 0, session_pool()).is_err());
    assert!(Qwen35Session::new(&mut model, 17, session_pool()).is_err());
}

#[test]
fn session_step_rejects_tokens_above_capacity() {
    let mut model = tiny_dense_session_model();
    let mut session = Qwen35Session::new(&mut model, 1, session_pool()).unwrap();

    let err = session.step(&[0.0; 8], 2, &[[0; 4]; 2]).unwrap_err();
    assert!(
        err.contains("session capacity is 1"),
        "unexpected error: {err}"
    );
}

#[test]
fn session_step_fills_capacity_exactly_without_committing_an_extra_row() {
    let mut model = tiny_dense_session_model();
    let mut session = Qwen35Session::new(&mut model, 4, session_pool()).unwrap();
    let logits = session
        .step_with_tokens(&[0, 1, 2, 3], &[[0; 4], [1; 4], [2; 4], [3; 4]])
        .unwrap();
    assert_eq!(session.processed_tokens(), 4);
    assert!(logits.iter().all(|value| value.is_finite()));
    let before = snapshot_qwen35_recurrent_state(&session);
    assert!(session.step_with_tokens(&[0], &[[4; 4]]).is_err());
    assert_eq!(session.processed_tokens(), 4);
    assert_eq!(snapshot_qwen35_recurrent_state(&session), before);
}

#[test]
fn session_new_sizes_state_to_requested_limit() {
    let mut model = tiny_dense_session_model();
    let cfg = model.config.clone();
    let session = Qwen35Session::new(&mut model, 4, session_pool()).unwrap();

    assert_eq!(session.scratch().x.len(), 4 * cfg.n_embd);
    let KvCache::F32(cache) = session.kv_cache() else {
        panic!("Qwen3.5 KV cache should be F32");
    };
    assert_eq!(cache.k.len(), cfg.n_layer_impl() * 4 * cfg.n_embd_head());
    assert_eq!(cache.v.len(), cache.k.len());
}

#[test]
fn dense_kv_snapshots_return_token_major_post_rotary_values() {
    let mut model = tiny_dense_session_model();
    let mut session = Qwen35Session::new(&mut model, 4, session_pool()).unwrap();
    session.step(&[0.0; 8], 2, &[[0; 4], [1; 4]]).unwrap();
    let KvCache::F32(cache) = session.kv_cache_mut() else {
        panic!("Qwen3.5 KV cache should be F32");
    };
    cache.k[..8].copy_from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    cache.v.copy_from_slice(&[
        100.0, 101.0, 0.0, 0.0, 110.0, 111.0, 0.0, 0.0, 120.0, 121.0, 0.0, 0.0, 130.0, 131.0, 0.0,
        0.0,
    ]);

    let snapshots = session.dense_kv_snapshots(&[0], 2).unwrap();

    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].shape(), [2, 1, 4]);
    assert_eq!(snapshots[0].key, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    assert_eq!(
        snapshots[0].value,
        [100.0, 110.0, 120.0, 130.0, 101.0, 111.0, 121.0, 131.0]
    );
}

#[test]
fn last_hidden_borrows_all_final_norm_rows_from_the_last_step() {
    let mut model = tiny_dense_session_model();
    model.config.norm_eps = 0.0;
    model.layers[0].wo = Some(f32_test_weight(vec![0.0; 16], 4, 4));
    model.layers[0].ffn_down = Some(f32_test_weight(vec![0.0; 16], 4, 4));
    let mut session = Qwen35Session::new_with_compute(
        &mut model,
        2,
        2,
        session_pool(),
        crate::compute::ComputePolicy::Cpu,
    )
    .unwrap();
    session
        .step(
            &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
            2,
            &[[0; 4], [1; 4]],
        )
        .unwrap();

    assert_eq!(
        session.last_hidden(2).unwrap(),
        &[1.2, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
    );
}

#[test]
fn last_hidden_rejects_rows_not_produced_by_the_last_step() {
    let mut model = tiny_dense_session_model();
    let mut session = Qwen35Session::new(&mut model, 2, session_pool()).unwrap();
    session.step(&[0.0; 4], 1, &[[0; 4]]).unwrap();

    assert!(session.last_hidden(2).unwrap_err().contains("last step"));
}

#[test]
fn session_step_rejects_empty_token_batches() {
    let mut model = tiny_dense_session_model();
    let mut session = Qwen35Session::new(&mut model, 1, session_pool()).unwrap();

    let error = session.step(&[], 0, &[]).unwrap_err();

    assert!(
        error.contains("requires at least one token"),
        "unexpected error: {error}"
    );
}

#[test]
fn session_step_rejects_embedding_length_overflow() {
    let mut model = tiny_dense_session_model();
    let mut session = Qwen35Session::new(&mut model, 1, session_pool()).unwrap();

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
#[ignore = "requires a Vulkan device"]
fn cpu_scope_prevents_model_from_creating_a_second_vulkan_session() {
    crate::ops::float::enable_gpu();
    assert!(crate::ops::get_vulkan_context().is_some());
    let mut model = tiny_q8_session_model();
    let pool = ComputePool::new(2);
    let mut scratch = Qwen35Scratchpad::new(&model.config, 3);
    let mut cache = KvCache::new_f32(1, 3, 256);
    scratch.x[..256].copy_from_slice(&model.embed_tokens(&[0]).unwrap());
    let _scope = ComputePool::disable_gpu_matmul_for_scope();
    model
        .forward(1, &mut cache, &mut scratch, &pool, &[[0; 4]])
        .unwrap();
    assert!(
        model.gpu.is_none(),
        "CPU fallback must not create a model-level GPU session"
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn gpu_failure_rechunks_the_remaining_prompt_with_bounded_scratch() {
    let mut model = tiny_dense_session_model();
    let mut baseline_model = tiny_dense_session_model();
    let mut fallback =
        Qwen35Session::new_with_prefill_batch_size(&mut model, 3, 1, session_pool()).unwrap();
    let mut baseline =
        Qwen35Session::new_with_prefill_batch_size(&mut baseline_model, 3, 1, session_pool())
            .unwrap();
    fallback.fail_gpu_once_for_test("prompt failure");

    let actual = fallback
        .step_with_tokens(&[0, 1, 2], &[[0; 4], [1; 4], [2; 4]])
        .unwrap();
    let expected = baseline
        .step_with_tokens(&[0, 1, 2], &[[0; 4], [1; 4], [2; 4]])
        .unwrap();

    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        snapshot_qwen35_recurrent_state(&fallback),
        snapshot_qwen35_recurrent_state(&baseline)
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn later_gpu_failure_recomputes_from_committed_cpu_shadow_and_stays_session_local() {
    let mut model = tiny_q8_session_model();
    let mut cpu_model = tiny_q8_session_model();
    let fallback_pool = Arc::new(ComputePool::new(2));
    let mut fallback = Qwen35Session::new(&mut model, 3, fallback_pool.clone()).unwrap();
    let mut cpu = Qwen35Session::new(&mut cpu_model, 3, Arc::new(ComputePool::new(2))).unwrap();

    let first = fallback.embed_tokens(&[0]).unwrap();
    let second = fallback.embed_tokens(&[1]).unwrap();
    let third = fallback.embed_tokens(&[2]).unwrap();
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
    let mut model = tiny_dense_session_model();
    let session = Qwen35Session::new(&mut model, 1, session_pool()).unwrap();

    // token id 3 -> row offset 12 -> [12, 13, 14, 15]
    let row = session.embed_token(3).unwrap();
    assert_eq!(row, vec![12.0, 13.0, 14.0, 15.0]);

    // token id 0 -> [0, 1, 2, 3]
    let row0 = session.embed_token(0).unwrap();
    assert_eq!(row0, vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn session_embed_token_out_of_range_errors() {
    let mut model = tiny_dense_session_model();
    let session = Qwen35Session::new(&mut model, 1, session_pool()).unwrap();

    let err = session.embed_token(8).unwrap_err();
    assert!(err.contains("out of range"), "unexpected error: {err}");
    let err2 = session.embed_token(u32::MAX).unwrap_err();
    assert!(err2.contains("out of range"));
}

#[test]
fn session_embed_tokens_concatenates_rows() {
    let mut model = tiny_dense_session_model();
    let session = Qwen35Session::new(&mut model, 1, session_pool()).unwrap();

    let all = session.embed_tokens(&[0, 1, 2]).unwrap();
    assert_eq!(all.len(), 12);
    assert_eq!(&all[0..4], &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(&all[4..8], &[4.0, 5.0, 6.0, 7.0]);
    assert_eq!(&all[8..12], &[8.0, 9.0, 10.0, 11.0]);
}

#[test]
fn session_embed_tokens_rejects_out_of_range_ids() {
    let mut model = tiny_dense_session_model();
    let session = Qwen35Session::new(&mut model, 1, session_pool()).unwrap();

    assert!(session
        .embed_tokens(&[0, 99, 2])
        .unwrap_err()
        .contains("vocab=8"));
}

#[test]
fn session_set_next_position_and_reset() {
    let mut model = tiny_dense_session_model();
    let mut session = Qwen35Session::new(&mut model, 2, session_pool()).unwrap();

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
    let mut model = tiny_dense_session_model();
    let mut session = Qwen35Session::new(&mut model, 2, session_pool()).unwrap();

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
