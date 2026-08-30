use super::{gemma4_token_table_digest, Gemma4AudioConfig, Gemma4Config, Gemma4VisionConfig};
use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
use std::collections::HashMap;

#[derive(Clone, Default)]
struct MapTensorSource {
    metadata: HashMap<String, MetaValue>,
    tensors: HashMap<String, TensorInfo>,
}

impl TensorSource for MapTensorSource {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
        None
    }
}

fn add_tensor(source: &mut MapTensorSource, name: impl Into<String>, dims: &[u64], ty: GGMLType) {
    let name = name.into();
    source.tensors.insert(
        name.clone(),
        TensorInfo {
            name,
            dims: dims.to_vec(),
            ggml_type: ty,
            offset: 0,
        },
    );
}

fn array(ty: MetaValueType, values: Vec<MetaValue>) -> MetaValue {
    MetaValue::Array(ty, values)
}

fn gemma4_token_values() -> Vec<MetaValue> {
    let mut tokens = vec![MetaValue::String(String::new()); 262_144];
    for (id, token) in [
        (0, "<pad>"),
        (1, "<eos>"),
        (2, "<bos>"),
        (3, "<unk>"),
        (4, "<mask>"),
        (1_024, "▁over"),
        (4_096, "दा"),
        (65_536, "DIST"),
        (131_072, "▁Leh"),
        (196_608, "▁dipilih"),
        (255_999, "<|image>"),
        (256_000, "<|audio>"),
        (258_882, "<image|>"),
        (258_883, "<audio|>"),
        (262_143, "<unused6226>"),
    ] {
        tokens[id] = MetaValue::String(token.into());
    }
    tokens
}

fn valid_gemma4_source() -> MapTensorSource {
    let mut source = MapTensorSource {
        metadata: HashMap::from([
            (
                "general.architecture".into(),
                MetaValue::String("gemma4".into()),
            ),
            ("general.type".into(), MetaValue::String("model".into())),
            ("gemma4.block_count".into(), MetaValue::Uint32(35)),
            ("gemma4.context_length".into(), MetaValue::Uint32(131_072)),
            ("gemma4.embedding_length".into(), MetaValue::Uint32(1536)),
            (
                "gemma4.feed_forward_length".into(),
                array(
                    MetaValueType::Int32,
                    [
                        vec![MetaValue::Int32(6144); 15],
                        vec![MetaValue::Int32(12_288); 20],
                    ]
                    .concat(),
                ),
            ),
            ("gemma4.attention.head_count".into(), MetaValue::Uint32(8)),
            (
                "gemma4.attention.head_count_kv".into(),
                MetaValue::Uint32(1),
            ),
            (
                "gemma4.rope.freq_base".into(),
                MetaValue::Float32(1_000_000.0),
            ),
            (
                "gemma4.rope.freq_base_swa".into(),
                MetaValue::Float32(10_000.0),
            ),
            (
                "gemma4.attention.layer_norm_rms_epsilon".into(),
                MetaValue::Float32(1e-6),
            ),
            ("gemma4.attention.key_length".into(), MetaValue::Uint32(512)),
            (
                "gemma4.attention.value_length".into(),
                MetaValue::Uint32(512),
            ),
            (
                "gemma4.final_logit_softcapping".into(),
                MetaValue::Float32(30.0),
            ),
            (
                "gemma4.attention.sliding_window".into(),
                MetaValue::Uint32(512),
            ),
            (
                "gemma4.attention.shared_kv_layers".into(),
                MetaValue::Uint32(20),
            ),
            (
                "gemma4.embedding_length_per_layer_input".into(),
                MetaValue::Uint32(256),
            ),
            (
                "gemma4.attention.sliding_window_pattern".into(),
                array(
                    MetaValueType::Bool,
                    (0..35).map(|i| MetaValue::Bool(i % 5 != 4)).collect(),
                ),
            ),
            (
                "gemma4.attention.key_length_swa".into(),
                MetaValue::Uint32(256),
            ),
            (
                "gemma4.attention.value_length_swa".into(),
                MetaValue::Uint32(256),
            ),
            ("gemma4.rope.dimension_count".into(), MetaValue::Uint32(512)),
            (
                "gemma4.rope.dimension_count_swa".into(),
                MetaValue::Uint32(256),
            ),
            (
                "tokenizer.ggml.model".into(),
                MetaValue::String("gemma4".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                array(MetaValueType::String, gemma4_token_values()),
            ),
        ]),
        tensors: HashMap::new(),
    };
    for (name, dims, ty) in [
        ("output_norm.weight", &[1536][..], GGMLType::F32),
        (
            "per_layer_model_proj.weight",
            &[1536, 8960][..],
            GGMLType::BF16,
        ),
        ("per_layer_proj_norm.weight", &[256][..], GGMLType::F32),
        (
            "per_layer_token_embd.weight",
            &[8960, 262_144][..],
            GGMLType::Q8_0,
        ),
        ("rope_freqs.weight", &[256][..], GGMLType::F32),
        ("token_embd.weight", &[1536, 262_144][..], GGMLType::Q8_0),
    ] {
        add_tensor(&mut source, name, dims, ty);
    }
    for layer in 0..35 {
        let head_dim = if layer % 5 == 4 { 512 } else { 256 };
        let ffn = if layer < 15 { 6144 } else { 12_288 };
        let prefix = format!("blk.{layer}");
        for (name, dims, ty) in [
            ("attn_k.weight", vec![1536, head_dim], GGMLType::Q8_0),
            ("attn_k_norm.weight", vec![head_dim], GGMLType::F32),
            ("attn_norm.weight", vec![1536], GGMLType::F32),
            (
                "attn_output.weight",
                vec![head_dim * 8, 1536],
                GGMLType::Q8_0,
            ),
            ("attn_q.weight", vec![1536, head_dim * 8], GGMLType::Q8_0),
            ("attn_q_norm.weight", vec![head_dim], GGMLType::F32),
            ("attn_v.weight", vec![1536, head_dim], GGMLType::Q8_0),
            ("ffn_down.weight", vec![ffn, 1536], GGMLType::Q8_0),
            ("ffn_gate.weight", vec![1536, ffn], GGMLType::Q8_0),
            ("ffn_norm.weight", vec![1536], GGMLType::F32),
            ("ffn_up.weight", vec![1536, ffn], GGMLType::Q8_0),
            ("inp_gate.weight", vec![1536, 256], GGMLType::F32),
            ("layer_output_scale.weight", vec![1], GGMLType::F32),
            ("post_attention_norm.weight", vec![1536], GGMLType::F32),
            ("post_ffw_norm.weight", vec![1536], GGMLType::F32),
            ("post_norm.weight", vec![1536], GGMLType::F32),
            ("proj.weight", vec![256, 1536], GGMLType::F32),
        ] {
            add_tensor(&mut source, format!("{prefix}.{name}"), &dims, ty);
        }
    }
    source
}

fn add_clippable(source: &mut MapTensorSource, prefix: &str, dims: &[u64]) {
    for suffix in ["input_max", "input_min", "output_max", "output_min"] {
        add_tensor(source, format!("{prefix}.{suffix}"), &[1], GGMLType::F32);
    }
    add_tensor(source, format!("{prefix}.weight"), dims, GGMLType::F16);
}

fn valid_mmproj_source() -> MapTensorSource {
    let mut source = MapTensorSource {
        metadata: HashMap::from([
            (
                "general.architecture".into(),
                MetaValue::String("clip".into()),
            ),
            ("general.type".into(), MetaValue::String("mmproj".into())),
            ("clip.has_vision_encoder".into(), MetaValue::Bool(true)),
            (
                "clip.vision.projector_type".into(),
                MetaValue::String("gemma4v".into()),
            ),
            ("clip.vision.projection_dim".into(), MetaValue::Uint32(1536)),
            ("clip.vision.image_size".into(), MetaValue::Uint32(224)),
            ("clip.vision.patch_size".into(), MetaValue::Uint32(16)),
            (
                "clip.vision.embedding_length".into(),
                MetaValue::Uint32(768),
            ),
            (
                "clip.vision.feed_forward_length".into(),
                MetaValue::Uint32(3072),
            ),
            ("clip.vision.block_count".into(), MetaValue::Uint32(16)),
            (
                "clip.vision.attention.head_count".into(),
                MetaValue::Uint32(12),
            ),
            (
                "clip.vision.image_mean".into(),
                array(MetaValueType::Float32, vec![MetaValue::Float32(0.0); 3]),
            ),
            (
                "clip.vision.image_std".into(),
                array(MetaValueType::Float32, vec![MetaValue::Float32(1.0); 3]),
            ),
            (
                "clip.vision.attention.layer_norm_epsilon".into(),
                MetaValue::Float32(1e-6),
            ),
            ("clip.has_audio_encoder".into(), MetaValue::Bool(true)),
            (
                "clip.audio.projector_type".into(),
                MetaValue::String("gemma4a".into()),
            ),
            ("clip.audio.projection_dim".into(), MetaValue::Uint32(1536)),
            (
                "clip.audio.embedding_length".into(),
                MetaValue::Uint32(1024),
            ),
            (
                "clip.audio.feed_forward_length".into(),
                MetaValue::Uint32(4096),
            ),
            ("clip.audio.block_count".into(), MetaValue::Uint32(12)),
            (
                "clip.audio.attention.head_count".into(),
                MetaValue::Uint32(8),
            ),
            ("clip.audio.num_mel_bins".into(), MetaValue::Uint32(128)),
            (
                "clip.audio.attention.layer_norm_epsilon".into(),
                MetaValue::Float32(1e-5),
            ),
        ]),
        tensors: HashMap::new(),
    };
    for layer in 0..16 {
        let prefix = format!("v.blk.{layer}");
        add_tensor(
            &mut source,
            format!("{prefix}.ln1.weight"),
            &[768],
            GGMLType::F32,
        );
        for name in ["ffn_down", "ffn_gate", "ffn_up"] {
            let dims = if name == "ffn_down" {
                &[3072, 768][..]
            } else {
                &[768, 3072][..]
            };
            add_clippable(&mut source, &format!("{prefix}.{name}"), dims);
        }
        for name in [
            "attn_post_norm.weight",
            "ffn_post_norm.weight",
            "ln2.weight",
        ] {
            add_tensor(
                &mut source,
                format!("{prefix}.{name}"),
                &[768],
                GGMLType::F32,
            );
        }
        add_tensor(
            &mut source,
            format!("{prefix}.attn_k_norm.weight"),
            &[64],
            GGMLType::F32,
        );
        for name in ["attn_k", "attn_out", "attn_q", "attn_v"] {
            add_clippable(&mut source, &format!("{prefix}.{name}"), &[768, 768]);
            if name == "attn_k" || name == "attn_q" {
                add_tensor(
                    &mut source,
                    format!("{prefix}.{name}_norm.weight"),
                    &[64],
                    GGMLType::F32,
                );
            }
        }
    }
    for (name, dims, ty) in [
        (
            "mm.input_projection.weight",
            &[768, 1536][..],
            GGMLType::F16,
        ),
        ("v.patch_embd.weight", &[16, 16, 3, 768][..], GGMLType::F16),
        (
            "v.position_embd.weight",
            &[768, 10_240, 2][..],
            GGMLType::F32,
        ),
        ("a.pre_encode.out.bias", &[1536][..], GGMLType::F32),
        ("a.pre_encode.out.weight", &[1024, 1536][..], GGMLType::F16),
        (
            "a.input_projection.weight",
            &[1024, 1024][..],
            GGMLType::F32,
        ),
        ("a.conv1d.0.weight", &[3, 3, 1, 128][..], GGMLType::F32),
        ("a.conv1d.0.norm.weight", &[128][..], GGMLType::F32),
        ("a.conv1d.1.weight", &[3, 3, 128, 32][..], GGMLType::F32),
        ("a.conv1d.1.norm.weight", &[32][..], GGMLType::F32),
        (
            "mm.a.input_projection.weight",
            &[1536, 1536][..],
            GGMLType::F16,
        ),
    ] {
        add_tensor(&mut source, name, dims, ty);
    }
    for layer in 0..12 {
        let prefix = format!("a.blk.{layer}");
        for name in ["ffn_up", "ffn_down", "ffn_up_1", "ffn_down_1"] {
            let dims = if name.starts_with("ffn_up") {
                &[1024, 4096][..]
            } else {
                &[4096, 1024][..]
            };
            add_clippable(&mut source, &format!("{prefix}.{name}"), dims);
        }
        for name in [
            "ffn_post_norm.weight",
            "ffn_norm.weight",
            "ffn_post_norm_1.weight",
            "ffn_norm_1.weight",
            "norm_conv.weight",
            "conv_norm.weight",
            "ln2.weight",
            "attn_post_norm.weight",
            "attn_pre_norm.weight",
        ] {
            add_tensor(
                &mut source,
                format!("{prefix}.{name}"),
                &[1024],
                GGMLType::F32,
            );
        }
        add_tensor(
            &mut source,
            format!("{prefix}.conv_dw.weight"),
            &[5, 1024],
            GGMLType::F32,
        );
        add_clippable(&mut source, &format!("{prefix}.conv_pw2"), &[1024, 1024]);
        add_clippable(&mut source, &format!("{prefix}.conv_pw1"), &[1024, 2048]);
        add_clippable(&mut source, &format!("{prefix}.attn_k"), &[1024, 1024]);
        add_tensor(
            &mut source,
            format!("{prefix}.per_dim_scale.weight"),
            &[128],
            GGMLType::F32,
        );
        add_clippable(&mut source, &format!("{prefix}.attn_out"), &[1024, 1024]);
        add_clippable(&mut source, &format!("{prefix}.attn_q"), &[1024, 1024]);
        add_tensor(
            &mut source,
            format!("{prefix}.attn_k_rel.weight"),
            &[1024, 1024],
            GGMLType::F16,
        );
        add_clippable(&mut source, &format!("{prefix}.attn_v"), &[1024, 1024]);
    }
    source
}

#[test]
fn gemma4_e2b_contract_is_exact() {
    assert_eq!(
        Gemma4Config::from_source(&valid_gemma4_source()).unwrap(),
        Gemma4Config {
            layers: 35,
            embd: 1536,
            heads: 8,
            kv_heads: 1,
            vocab: 262_144,
            full_head_dim: 512,
            swa_head_dim: 256,
            shared_kv_layers: 20,
            per_layer_width: 256,
            sliding_window: 512,
            logit_softcap: 30.0,
        }
    );
}

#[test]
fn gemma4_vision_contract_is_exact() {
    assert_eq!(
        Gemma4VisionConfig::from_source(&valid_mmproj_source()).unwrap(),
        Gemma4VisionConfig {
            layers: 16,
            embd: 768,
            heads: 12,
            image_size: 224,
            patch_size: 16,
            merge: 3,
            projection: 1536,
        }
    );
}

#[test]
fn gemma4_audio_contract_is_exact() {
    assert_eq!(
        Gemma4AudioConfig::from_source(&valid_mmproj_source()).unwrap(),
        Gemma4AudioConfig {
            layers: 12,
            embd: 1024,
            heads: 8,
            mel_bins: 128,
            projection: 1536,
        }
    );
}

#[test]
fn contracts_reject_architecture_shape_type_and_name_drift() {
    let mut wrong_arch = valid_gemma4_source();
    wrong_arch.metadata.insert(
        "general.architecture".into(),
        MetaValue::String("llama".into()),
    );
    assert!(Gemma4Config::from_source(&wrong_arch)
        .unwrap_err()
        .contains("general.architecture"));

    let mut wrong_shape = valid_gemma4_source();
    wrong_shape
        .tensors
        .get_mut("blk.0.attn_q.weight")
        .unwrap()
        .dims = vec![1536, 1535];
    assert!(Gemma4Config::from_source(&wrong_shape)
        .unwrap_err()
        .contains("blk.0.attn_q.weight"));

    let mut wrong_type = valid_gemma4_source();
    wrong_type
        .tensors
        .get_mut("token_embd.weight")
        .unwrap()
        .ggml_type = GGMLType::F16;
    assert!(Gemma4Config::from_source(&wrong_type)
        .unwrap_err()
        .contains("token_embd.weight"));

    let mut wrong_name = valid_gemma4_source();
    let tensor = wrong_name.tensors.remove("blk.0.attn_q.weight").unwrap();
    wrong_name
        .tensors
        .insert("blk.0.attn_q.drifted".into(), tensor);
    assert!(Gemma4Config::from_source(&wrong_name)
        .unwrap_err()
        .contains("blk.0.attn_q.weight"));

    let mut vision = valid_mmproj_source();
    vision
        .tensors
        .get_mut("v.blk.0.attn_q.weight")
        .unwrap()
        .ggml_type = GGMLType::F32;
    assert!(Gemma4VisionConfig::from_source(&vision)
        .unwrap_err()
        .contains("v.blk.0.attn_q.weight"));

    let mut audio = valid_mmproj_source();
    audio
        .tensors
        .get_mut("a.blk.0.conv_dw.weight")
        .unwrap()
        .dims = vec![5, 1023];
    assert!(Gemma4AudioConfig::from_source(&audio)
        .unwrap_err()
        .contains("a.blk.0.conv_dw.weight"));
}

#[test]
fn gemma4_e2b_contract_rejects_non_anchor_token_drift() {
    let mut wrong_token = valid_gemma4_source();
    let Some(MetaValue::Array(_, tokens)) = wrong_token.metadata.get_mut("tokenizer.ggml.tokens")
    else {
        panic!("valid source has a token array");
    };
    tokens[12_345] = MetaValue::String("token-id-drift".into());
    assert!(Gemma4Config::from_source(&wrong_token)
        .unwrap_err()
        .contains("tokenizer.ggml.tokens digest"));
}

#[test]
fn gemma4_token_digest_is_ordered_and_length_delimited() {
    let tokens = |values: &[&str]| {
        values
            .iter()
            .map(|value| MetaValue::String((*value).into()))
            .collect::<Vec<_>>()
    };

    assert_ne!(
        gemma4_token_table_digest(&tokens(&["ab", "c"])).unwrap(),
        gemma4_token_table_digest(&tokens(&["a", "bc"])).unwrap()
    );
    assert_ne!(
        gemma4_token_table_digest(&tokens(&["ab", "c"])).unwrap(),
        gemma4_token_table_digest(&tokens(&["c", "ab"])).unwrap()
    );
}
