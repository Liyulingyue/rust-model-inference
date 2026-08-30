use super::{
    assemble_input_rows, attend, head_dim, kv_source_layer, load_weight, matmul, require_f32_kv,
    softcap, Gemma4InputRow, Gemma4Layer, Gemma4Model, KvLayer, BASE_FFN_LAYERS, EMBED,
    FULL_HEAD_DIM, HEADS, LAYERS, MAX_FFN, PER_LAYER, PER_LAYER_ALL, SWA_HEAD_DIM, VOCAB,
};
use crate::core::scratchpad::KvFormat;
use crate::core::tensor::{GGMLType, TensorInfo, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::gemma4::Gemma4Config;
use crate::ops::kernel::{Kernel, QuantizedTensor, Weight};
use std::sync::Arc;

struct EmptySource;

impl TensorSource for EmptySource {
    fn metadata(&self, _key: &str) -> Option<&crate::core::tensor::MetaValue> {
        None
    }

    fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
        None
    }

    fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
        None
    }
}

struct ZeroKernel;

impl Kernel for ZeroKernel {
    fn forward_prequantized(
        &self,
        _input_q8: &[u8],
        _input_scales: &[f32],
        output: &mut [f32],
        _n_in: usize,
        n_out: usize,
        _ith: usize,
        _nth: usize,
    ) {
        output[..n_out].fill(0.0);
    }

    fn embedding_lookup(&self, _token_id: u32, n_embd: usize, output: &mut [f32]) {
        assert_eq!(output.len(), n_embd);
        output.fill(0.0);
    }
}

struct ZeroBf16Kernel {
    bytes: Vec<u8>,
}

impl Kernel for ZeroBf16Kernel {
    fn bf16_bytes(&self) -> Option<&[u8]> {
        Some(&self.bytes)
    }

    fn forward_prequantized(
        &self,
        _input_q8: &[u8],
        _input_scales: &[f32],
        output: &mut [f32],
        _n_in: usize,
        n_out: usize,
        _ith: usize,
        _nth: usize,
    ) {
        output[..n_out].fill(0.0);
    }
}

fn zero_weight(n_in: usize, n_out: usize) -> Weight<'static> {
    Weight {
        kernel: Box::new(ZeroKernel),
        ggml_type: GGMLType::F32,
        n_in,
        n_out,
    }
}

fn zero_q8_weight(n_in: usize, n_out: usize) -> Weight<'static> {
    Weight {
        kernel: Box::new(ZeroKernel),
        ggml_type: GGMLType::Q8_0,
        n_in,
        n_out,
    }
}

fn zero_bf16_weight(n_in: usize, n_out: usize) -> Weight<'static> {
    Weight {
        kernel: Box::new(ZeroBf16Kernel {
            bytes: vec![0; n_in * n_out * 2],
        }),
        ggml_type: GGMLType::BF16,
        n_in,
        n_out,
    }
}

fn zero_layer(layer: usize) -> Gemma4Layer {
    let dim = head_dim(layer);
    let ffn = if layer < BASE_FFN_LAYERS {
        6144
    } else {
        MAX_FFN
    };
    Gemma4Layer {
        head_dim: dim,
        attn_norm: vec![1.0; EMBED],
        attn_q: zero_q8_weight(EMBED, HEADS * dim),
        attn_k: zero_q8_weight(EMBED, dim),
        attn_v: zero_q8_weight(EMBED, dim),
        attn_output: zero_q8_weight(HEADS * dim, EMBED),
        attn_q_norm: vec![1.0; dim],
        attn_k_norm: vec![1.0; dim],
        post_attention_norm: vec![1.0; EMBED],
        ffn_norm: vec![1.0; EMBED],
        ffn_gate: zero_q8_weight(EMBED, ffn),
        ffn_up: zero_q8_weight(EMBED, ffn),
        ffn_down: zero_q8_weight(ffn, EMBED),
        post_ffw_norm: vec![1.0; EMBED],
        inp_gate: zero_weight(EMBED, PER_LAYER),
        proj: zero_weight(PER_LAYER, EMBED),
        post_norm: vec![1.0; EMBED],
        output_scale: 1.0,
    }
}

fn post_kv_failure_model() -> Gemma4Model {
    let mut layers = (0..LAYERS).map(zero_layer).collect::<Vec<_>>();
    layers[0].attn_output.n_in += 1;
    Gemma4Model {
        _source: Arc::new(EmptySource),
        config: Gemma4Config {
            layers: LAYERS,
            embd: EMBED,
            heads: HEADS,
            kv_heads: 1,
            vocab: VOCAB,
            full_head_dim: FULL_HEAD_DIM,
            swa_head_dim: SWA_HEAD_DIM,
            shared_kv_layers: 20,
            per_layer_width: PER_LAYER,
            sliding_window: 512,
            logit_softcap: 30.0,
        },
        pool: Arc::new(ComputePool::new(1)),
        token_embedding: zero_weight(EMBED, VOCAB),
        per_layer_token_embedding: zero_weight(PER_LAYER_ALL, VOCAB),
        per_layer_model_proj: zero_bf16_weight(EMBED, PER_LAYER_ALL),
        per_layer_proj_norm: vec![1.0; PER_LAYER],
        output_norm: vec![1.0; EMBED],
        rope_freqs: vec![1.0; FULL_HEAD_DIM / 2],
        layers,
    }
}

#[test]
fn raw_rows_are_not_embedding_scaled_and_use_padding_layer_id() {
    let rows = assemble_input_rows(&[
        Gemma4InputRow::Token(7),
        Gemma4InputRow::Raw {
            values: vec![2.0; 1536],
            per_layer_token: 0,
        },
    ])
    .unwrap();
    assert!(rows[0].scale_token_embedding);
    assert!(!rows[1].scale_token_embedding);
    assert_eq!(rows[1].per_layer_token, 0);
}

#[test]
fn softcap_matches_pinned_reciprocal_scale_bits() {
    // Pinned llama.cpp 3173a56471c, first text raw logit at index 1.
    let raw = f32::from_bits(0x417c_38d8);
    assert_eq!(softcap(raw, 30.0).to_bits(), 0x4167_507f);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn layer_12_attention_softmax_matches_pinned_neon_bits() {
    // Occurrence 7 head-0 KQ words and output are independently pinned from llama.cpp 3173a56471c.
    let keys = [
        0x40b4_85b2,
        0x3ffc_c0c2,
        0x4079_1edf,
        0x4027_f5cc,
        0x407c_44ba,
        0x4078_0503,
        0xbec0_388c,
        0x405a_25f4,
    ]
    .map(f32::from_bits);
    let cache = KvLayer {
        head_dim: 1,
        keys: keys.to_vec(),
        values: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].to_vec(),
    };
    let mut output = [0.0; HEADS];

    attend(
        12,
        7,
        &[1.0; HEADS],
        &cache,
        true,
        &mut output,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .unwrap();

    assert_eq!(output.map(f32::to_bits), [0x3f15_89fe; HEADS]);
}

#[test]
fn ggml_geglu_rounds_gate_and_gelu_through_f16() {
    let mut gate = [0.0; 8];
    let mut up = [1.0; 8];
    gate[0] = f32::from_bits(0x3f12_598e);
    up[0] = f32::from_bits(0xbed7_8765);
    gate[1] = f32::from_bits(0xbfff_e000);

    super::ggml_geglu_fp16_inplace(&mut gate, &up);

    assert_eq!(gate[0].to_bits(), 0xbe30_7c3e);
    assert_eq!(gate[1].to_bits(), 0xbd3a_6000);
}

#[test]
fn f32_projection_rejects_missing_or_wrong_backing_storage() {
    let cases = [
        (zero_weight(2, 1), "F32 kernel"),
        (
            Weight {
                kernel: Box::new(crate::ops::kernel::f32::F32Kernel::new(vec![0.0])),
                ggml_type: GGMLType::F32,
                n_in: 2,
                n_out: 1,
            },
            "expected 2, got 1",
        ),
    ];

    for (weight, expected_error) in cases {
        let mut output = [7.0];
        let mut q8 = [0; 2];
        let mut scales = [0.0];
        let error = matmul(
            "blk.0.inp_gate.weight",
            &weight,
            &[1.0, 2.0],
            &mut output,
            &ComputePool::new(1),
            &mut q8,
            &mut scales,
        )
        .unwrap_err();

        assert!(error.contains(expected_error), "{error}");
        assert_eq!(output, [7.0]);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn per_layer_f32_projection_matches_pinned_neon_dot_bits() {
    // Pinned llama.cpp 3173a56471c, blk.0.inp_gate.weight row 0 and
    // layer-0 FFN output occurrence 0. The first 16 real operands already
    // distinguish its four FMA accumulators from sequential F32 addition.
    let weights = [
        0x3a89_0000,
        0x39fb_0000,
        0xb7e5_0000,
        0xb7a2_0000,
        0x39f0_0000,
        0x3a16_0000,
        0xba2b_0000,
        0xba3c_0000,
        0xb906_0000,
        0x3748_0000,
        0xba28_0000,
        0xb9c4_0000,
        0x377b_0000,
        0xba2a_0000,
        0xb983_0000,
        0xb987_0000,
    ]
    .map(f32::from_bits);
    let input = [
        0xc116_ef77,
        0x413c_c829,
        0x3e4c_d214,
        0xc180_6c96,
        0x400b_f0a2,
        0xc03c_6f04,
        0xbe76_9592,
        0x3cec_5d40,
        0x3f80_3150,
        0x401d_94ed,
        0x3e9a_5ed4,
        0x4093_33e6,
        0x3f1c_0cfe,
        0xc0af_a2b5,
        0xc026_7f9c,
        0xbf27_ece8,
    ]
    .map(f32::from_bits);
    let weight = Weight {
        kernel: Box::new(crate::ops::kernel::f32::F32Kernel::new(weights.to_vec())),
        ggml_type: GGMLType::F32,
        n_in: input.len(),
        n_out: 1,
    };
    let mut output = [0.0];
    let mut q8 = [0; 16];
    let mut scales = [0.0];

    matmul(
        "blk.0.inp_gate.weight",
        &weight,
        &input,
        &mut output,
        &ComputePool::new(1),
        &mut q8,
        &mut scales,
    )
    .unwrap();

    assert_eq!(output[0].to_bits(), 0xbb08_36fd);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn per_layer_f32_projection_matches_pinned_neon_long_rows() {
    const WIDTH: usize = 1536;
    const ROWS: usize = 3;
    let input = (0_u32..WIDTH as u32)
        .map(|index| {
            let mixed = index.wrapping_mul(0x9e37_79b9).wrapping_add(0x243f_6a88);
            f32::from_bits((mixed & 0x8000_0000) | 0x3e00_0000 | ((mixed >> 1) & 0x007f_ffff))
        })
        .collect::<Vec<_>>();
    let weights = (0_u32..ROWS as u32)
        .flat_map(|row| {
            (0_u32..WIDTH as u32).map(move |index| {
                let mixed = index
                    .wrapping_mul(0x85eb_ca6b)
                    .wrapping_add((row + 1).wrapping_mul(0xc2b2_ae35));
                f32::from_bits((mixed & 0x8000_0000) | 0x3d80_0000 | ((mixed >> 1) & 0x007f_ffff))
            })
        })
        .collect::<Vec<_>>();
    let weight = Weight {
        kernel: Box::new(crate::ops::kernel::f32::F32Kernel::new(weights)),
        ggml_type: GGMLType::F32,
        n_in: WIDTH,
        n_out: ROWS,
    };
    let mut output = [0.0; ROWS];
    let mut q8 = vec![0; WIDTH];
    let mut scales = vec![0.0; WIDTH.div_ceil(32)];

    matmul(
        "blk.0.inp_gate.weight",
        &weight,
        &input,
        &mut output,
        &ComputePool::new(1),
        &mut q8,
        &mut scales,
    )
    .unwrap();

    // Independent literals from pinned llama.cpp ggml_vec_dot_f32.
    assert_eq!(
        output.map(f32::to_bits),
        [0xbe74_d6c6, 0x3df2_a865, 0x3ed8_e80e]
    );
}

#[test]
fn per_layer_bf16_projection_matches_pinned_scalar_dot_bits() {
    // Pinned llama.cpp 3173a56471c, Gemma4 text projection, first 16
    // operands from real rows 0, 1, 2, 3, and 5. Its arm64 BF16 dot rounds
    // the activation to BF16, forms F32 products, accumulates them in
    // ggml_float (F64), then casts once. The pinned row-3 and row-5 F32
    // accumulator words are respectively 0x3daed280 and 0xbdef03c0, so
    // those rows make an F32-accumulation mutation observable.
    let input = [
        0xbfd0_8482,
        0xbfc2_eb2b,
        0x3e47_739e,
        0xbfbe_62b9,
        0xbf7d_d8f7,
        0xbd11_0e44,
        0xbee2_a64a,
        0x3e87_fd60,
        0xbfa9_fcb8,
        0x3f7d_d8f7,
        0xbf1a_1f28,
        0xbfa9_fcb8,
        0xbeeb_b72f,
        0x3ee2_a64a,
        0xbf8a_4199,
        0xbebe_62b9,
    ]
    .map(f32::from_bits);
    let weight_rows = [
        [
            0x3d37_u16, 0x3d04, 0xbc50, 0x3d77, 0x3bc7, 0x3cd1, 0xbcdb, 0xbdae, 0xbbe5, 0x3b39,
            0xbbcd, 0x3c9e, 0x3cde, 0x3d16, 0xbd82, 0x3c63,
        ],
        [
            0x3c47, 0x3b92, 0x3ca0, 0xbd46, 0xbd80, 0x3d89, 0x3ce9, 0xbcef, 0xbc48, 0xbcbf, 0xbd18,
            0x3ce0, 0x3d43, 0xbd9e, 0x3c35, 0xbcae,
        ],
        [
            0x3ca8, 0xbc5d, 0x3d50, 0xbd1e, 0xbc40, 0x3da4, 0x3ba4, 0xbc8f, 0x3d2c, 0x3cac, 0xbd3c,
            0x3b94, 0x3d03, 0x3c49, 0x3d79, 0x3c83,
        ],
        [
            0xbb1e, 0xbd4b, 0xbac3, 0xbd35, 0x3cc6, 0x3c9b, 0x3c2c, 0x3d72, 0xbd09, 0xbcf5, 0xbcb6,
            0xbbdb, 0xbc6d, 0xbcea, 0x3d91, 0x3a98,
        ],
        [
            0x3cd0, 0x3d18, 0x3bf3, 0xbb41, 0xbbea, 0xbb32, 0x3c12, 0xbd5e, 0x3afa, 0xbc7c, 0x39ab,
            0x3d39, 0xbc9d, 0x3d0e, 0xbd33, 0x3c94,
        ],
    ];
    let rows = weight_rows.len();
    let weight = weight_rows
        .into_iter()
        .flatten()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
        &weight,
        GGMLType::BF16,
        input.len(),
        rows,
    ));
    let mut output = [0.0; 5];
    let mut q8 = vec![0; input.len() * 2];
    let mut scales = vec![0.0; input.len().div_ceil(32)];

    matmul(
        "per_layer_model_proj.weight",
        &weight,
        &input,
        &mut output,
        &ComputePool::new(3),
        &mut q8,
        &mut scales,
    )
    .unwrap();

    assert_eq!(
        output.map(f32::to_bits),
        [
            0xbe32_95aa,
            0x3be9_7100,
            0xbd1b_8ce0,
            0x3dae_d27f,
            0xbdef_03c1
        ]
    );
}

#[test]
fn per_layer_projection_rejects_non_bf16_weight() {
    let weight = zero_weight(2, 1);
    let mut output = [7.0];
    let mut input_bf16 = [0; 4];
    let mut scales = [0.0];

    let error = matmul(
        "per_layer_model_proj.weight",
        &weight,
        &[1.0, 2.0],
        &mut output,
        &ComputePool::new(1),
        &mut input_bf16,
        &mut scales,
    )
    .unwrap_err();

    assert!(error.contains("requires BF16"), "{error}");
    assert_eq!(output, [7.0]);
}

#[test]
fn per_layer_projection_rejects_wrong_bf16_storage_length() {
    for byte_len in [6, 2] {
        let bytes = vec![0; byte_len];
        let weight =
            Weight::from_quantized(QuantizedTensor::from_bytes(&bytes, GGMLType::BF16, 2, 1));
        let mut output = [7.0];
        let mut input_bf16 = [0; 4];
        let mut scales = [0.0];

        let error = matmul(
            "per_layer_model_proj.weight",
            &weight,
            &[1.0, 2.0],
            &mut output,
            &ComputePool::new(1),
            &mut input_bf16,
            &mut scales,
        )
        .unwrap_err();

        assert!(error.contains("expected 4 bytes"), "{error}");
        assert!(error.contains(&format!("got {byte_len}")), "{error}");
        assert_eq!(output, [7.0]);
    }
}

#[test]
fn input_rows_reject_empty_invalid_and_nonfinite_values() {
    assert!(assemble_input_rows(&[]).unwrap_err().contains("empty"));
    assert!(assemble_input_rows(&[Gemma4InputRow::Token(262_144)])
        .unwrap_err()
        .contains("token"));
    assert!(assemble_input_rows(&[Gemma4InputRow::Raw {
        values: vec![0.0; 1535],
        per_layer_token: 0,
    }])
    .unwrap_err()
    .contains("1536"));
    assert!(assemble_input_rows(&[Gemma4InputRow::Raw {
        values: {
            let mut values = vec![0.0; 1536];
            values[7] = f32::NAN;
            values
        },
        per_layer_token: 0,
    }])
    .unwrap_err()
    .contains("non-finite"));
}

#[test]
fn shared_kv_layers_map_by_attention_kind() {
    assert_eq!(kv_source_layer(0), 0);
    assert_eq!(kv_source_layer(14), 14);
    assert_eq!(kv_source_layer(15), 13);
    assert_eq!(kv_source_layer(19), 14);
    assert_eq!(kv_source_layer(34), 14);
}

#[test]
fn incremental_session_is_f32_only() {
    assert!(require_f32_kv(KvFormat::F32).is_ok());
    assert!(require_f32_kv(KvFormat::F16).unwrap_err().contains("F32"));
}

#[test]
fn post_kv_failure_leaves_session_state_unchanged() {
    let model = post_kv_failure_model();
    let mut session = super::Gemma4Session::new(&model, KvFormat::F32).unwrap();
    let rows = [Gemma4InputRow::Raw {
        values: vec![0.0; EMBED],
        per_layer_token: 0,
    }];

    for _ in 0..2 {
        let error = session.forward_rows(&rows).unwrap_err();
        assert!(error.contains("blk.0.attn_output.weight"), "{error}");
        assert_eq!(session.len(), 0);
        assert!(session
            .kv
            .iter()
            .all(|layer| layer.keys.is_empty() && layer.values.is_empty()));
    }
}

#[test]
fn f32_matrix_loader_preserves_declared_shape() {
    struct F32Matrix {
        info: TensorInfo,
        bytes: Vec<u8>,
    }
    impl TensorSource for F32Matrix {
        fn metadata(&self, _key: &str) -> Option<&crate::core::tensor::MetaValue> {
            None
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            (name == "matrix.weight").then_some(&self.info)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            (name == "matrix.weight").then_some(self.bytes.as_slice())
        }
    }
    let source = F32Matrix {
        info: TensorInfo {
            name: "matrix.weight".into(),
            dims: vec![2, 3],
            ggml_type: GGMLType::F32,
            offset: 0,
        },
        bytes: vec![0; 2 * 3 * 4],
    };
    let weight = load_weight(&source, "matrix.weight", &[2, 3], GGMLType::F32).unwrap();
    assert_eq!((weight.n_in, weight.n_out), (2, 3));
}

#[test]
#[ignore = "requires RMI_GEMMA4_MODEL"]
fn actual_model_one_token_produces_finite_logits() {
    let path = std::env::var_os("RMI_GEMMA4_MODEL").expect("RMI_GEMMA4_MODEL");
    let source = std::sync::Arc::new(crate::core::loader::GGUFLoader::from_file(path).unwrap());
    for (layer, expected_ffn) in [(14, 6144), (15, 12_288), (34, 12_288)] {
        assert_eq!(
            source
                .tensor_info(&format!("blk.{layer}.ffn_gate.weight"))
                .unwrap()
                .dims,
            [1536, expected_ffn]
        );
    }
    let model = super::Gemma4Model::from_source(source, 4).unwrap();
    let mut session = super::Gemma4Session::new(&model, KvFormat::F32).unwrap();
    let logits = session.forward_rows(&[Gemma4InputRow::Token(2)]).unwrap();
    assert_eq!(logits.len(), 262_144);
    assert!(logits.iter().all(|value| value.is_finite()));
}
