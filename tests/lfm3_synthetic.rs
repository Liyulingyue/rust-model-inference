//! Model-free regressions for LFM3 loading, causal state and context boundaries.
use rust_model_inference::core::scratchpad::KvFormat;
use rust_model_inference::core::tensor::{
    GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource,
};
use rust_model_inference::models::lfm3::{Lfm3Config, Lfm3Model, Lfm3Session};
use std::collections::HashMap;

#[derive(Default)]
struct Source {
    meta: HashMap<String, MetaValue>,
    tensors: HashMap<String, (TensorInfo, Vec<u8>)>,
}
impl TensorSource for Source {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.meta.get(key)
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name).map(|v| &v.0)
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|v| v.1.as_slice())
    }
}
impl Source {
    fn tensor(&mut self, name: &str, dims: &[u64], ones: bool) {
        let size = dims.iter().product::<u64>() as usize;
        let bytes = (0..size)
            .flat_map(|i| {
                let value = if ones {
                    1.0
                } else {
                    ((i * 17 + name.len()) as f32).sin() * 0.1
                };
                value.to_le_bytes()
            })
            .collect();
        self.tensors.insert(
            name.into(),
            (
                TensorInfo {
                    name: name.into(),
                    dims: dims.into(),
                    ggml_type: GGMLType::F32,
                    offset: 0,
                },
                bytes,
            ),
        );
    }
}
fn source() -> Source {
    let mut s = Source::default();
    s.meta.insert(
        "general.architecture".into(),
        MetaValue::String("lfm3".into()),
    );
    for (key, value) in [
        ("embedding_length", 32),
        ("block_count", 3),
        ("attention.head_count", 2),
        ("attention.key_length", 16),
        ("attention.value_length", 16),
        ("feed_forward_length", 64),
        ("context_length", 19),
        ("vocab_size", 16),
        ("shortconv.l_cache", 7),
        ("attention.sliding_window", 4),
    ] {
        s.meta
            .insert(format!("lfm3.{key}"), MetaValue::Uint32(value));
    }
    for (key, value) in [
        ("rope.freq_base", 1e6),
        ("rope.freq_base_swa", 1e4),
        ("attention.layer_norm_rms_epsilon", 1e-5),
    ] {
        s.meta
            .insert(format!("lfm3.{key}"), MetaValue::Float32(value));
    }
    s.meta.insert(
        "lfm3.attention.head_count_kv".into(),
        MetaValue::Array(
            MetaValueType::Int32,
            vec![
                MetaValue::Int32(0),
                MetaValue::Int32(1),
                MetaValue::Int32(1),
            ],
        ),
    );
    s.meta.insert(
        "lfm3.attention.sliding_window_pattern".into(),
        MetaValue::Array(
            MetaValueType::Bool,
            vec![
                MetaValue::Bool(false),
                MetaValue::Bool(true),
                MetaValue::Bool(false),
            ],
        ),
    );
    s.tensor("token_embd.weight", &[32, 16], false);
    s.tensor("output_norm.weight", &[32], true);
    s.tensor("rope_freqs.weight", &[8], true);
    for layer in 0..3 {
        for name in [
            "attn_norm",
            "ffn_norm",
            "post_attention_norm",
            "post_ffw_norm",
        ] {
            s.tensor(&format!("blk.{layer}.{name}.weight"), &[32], true);
        }
        for (name, dims) in [
            ("ffn_gate", [32, 64]),
            ("ffn_up", [32, 64]),
            ("ffn_down", [64, 32]),
        ] {
            s.tensor(&format!("blk.{layer}.{name}.weight"), &dims, false);
        }
        if layer == 0 {
            for (name, dims) in [
                ("in_proj", [32, 96]),
                ("out_proj", [32, 32]),
                ("conv", [7, 32]),
            ] {
                s.tensor(&format!("blk.0.shortconv.{name}.weight"), &dims, false);
            }
        } else {
            for (name, dims) in [
                ("q", [32, 32]),
                ("k", [32, 16]),
                ("v", [32, 16]),
                ("output", [32, 32]),
                ("gate", [32, 2]),
            ] {
                s.tensor(&format!("blk.{layer}.attn_{name}.weight"), &dims, false);
            }
            for name in ["q", "k"] {
                s.tensor(&format!("blk.{layer}.attn_{name}_norm.weight"), &[16], true);
            }
        }
    }
    s
}

#[test]
fn incremental_prefill_chunks_and_reset_agree_for_both_kv_formats() {
    let source = source();
    let tokens: Vec<u32> = (0..18).map(|i| (i % 16) as u32).collect();
    for format in [KvFormat::F32, KvFormat::F16] {
        let model = Lfm3Model::from_source(&source, 1).unwrap();
        let mut session = Lfm3Session::new(&model, format, 19).unwrap();
        let whole = session.forward_logits_chunked(&tokens, 7).unwrap();
        assert!(whole.iter().all(|v| v.is_finite()));
        session.reset();
        session.forward_logits_chunked(&tokens[..7], 3).unwrap();
        let incremental = session.forward_logits_chunked(&tokens[7..], 1).unwrap();
        assert_eq!(whole, incremental);
        assert_eq!(session.seq_len, 18);
        assert!(session.forward_logits_chunked(&[1, 2], 1).is_err());
        assert!(session.forward_logits_chunked(&[16], 1).is_err());
        assert!(session.forward_logits_chunked(&[], 1).is_err());
        assert!(session.forward_logits_chunked(&[1], 0).is_err());
        assert_eq!(session.seq_len, 18);
        session.forward_logits_chunked(&[1], 1).unwrap();
        assert!(session.forward_logits_chunked(&[1], 1).is_err());
    }
}

#[test]
fn rejects_invalid_metadata_without_panicking() {
    for (key, value) in [
        ("general.architecture", MetaValue::String("lfm2".into())),
        ("lfm3.attention.head_count", MetaValue::Uint32(0)),
        ("lfm3.attention.key_length", MetaValue::Uint32(15)),
        ("lfm3.attention.value_length", MetaValue::Uint32(8)),
        ("lfm3.shortconv.l_cache", MetaValue::Uint32(1)),
        ("lfm3.rope.freq_base_swa", MetaValue::Float32(f32::NAN)),
        (
            "lfm3.attention.head_count_kv",
            MetaValue::Array(MetaValueType::Int32, vec![MetaValue::Int32(-1); 3]),
        ),
        (
            "lfm3.attention.sliding_window_pattern",
            MetaValue::Array(MetaValueType::Bool, vec![MetaValue::Bool(false); 3]),
        ),
    ] {
        let mut source = source();
        source.meta.insert(key.into(), value);
        assert!(Lfm3Config::from_source(&source).is_err(), "{key}");
    }
}

#[test]
fn rejects_missing_mis_shaped_and_truncated_weights() {
    for case in 0..5 {
        let mut source = source();
        match case {
            0 => {
                source.tensors.remove("blk.0.post_ffw_norm.weight");
            }
            1 => {
                source
                    .tensors
                    .get_mut("blk.1.attn_gate.weight")
                    .unwrap()
                    .0
                    .dims = vec![32, 32];
            }
            2 => {
                source
                    .tensors
                    .get_mut("blk.0.shortconv.in_proj.weight")
                    .unwrap()
                    .1
                    .pop();
            }
            3 => {
                source.tensors.remove("rope_freqs.weight");
            }
            _ => {
                source.tensors.get_mut("rope_freqs.weight").unwrap().1[..4]
                    .copy_from_slice(&0f32.to_le_bytes());
            }
        }
        assert!(Lfm3Model::from_source(&source, 1).is_err(), "case {case}");
    }
}

#[test]
fn cpu_dispatch_covers_float_quantized_and_mixed_weights() {
    use rust_model_inference::compute::ComputePolicy;
    let formats = [
        GGMLType::F32,
        GGMLType::F16,
        GGMLType::BF16,
        GGMLType::Q8_0,
        GGMLType::Q4_0,
        GGMLType::Q4_1,
        GGMLType::Q4K,
        GGMLType::Q5K,
        GGMLType::Q6K,
    ];
    for choice in 0..=formats.len() {
        let mut source = source();
        source.tensor("output.weight", &[32, 16], false);
        for (key, value) in [
            ("embedding_length", 256),
            ("attention.head_count", 16),
            ("feed_forward_length", 512),
        ] {
            source
                .meta
                .insert(format!("lfm3.{key}"), MetaValue::Uint32(value));
        }
        let mut matrices = 0;
        let mut names: Vec<_> = source.tensors.keys().cloned().collect();
        names.sort();
        for name in names {
            let (info, bytes) = source.tensors.get_mut(&name).unwrap();
            for dim in &mut info.dims {
                *dim = match *dim {
                    32 => 256,
                    64 => 512,
                    96 => 768,
                    value => value,
                };
            }
            if name.ends_with("attn_gate.weight") {
                info.dims[1] = 16;
            }
            let count = info.dims.iter().product::<u64>() as usize;
            let matrix = info.dims.len() == 2 && !name.ends_with("shortconv.conv.weight");
            let mut format = if matrix {
                let format = formats[if choice == formats.len() {
                    matrices % formats.len()
                } else {
                    choice
                }];
                matrices += 1;
                format
            } else {
                GGMLType::F32
            };
            // Q4_1 projections are supported; embedding uses the shared lookup formats.
            if name == "token_embd.weight"
                && !rust_model_inference::ops::embedding::is_supported_embedding(format)
            {
                format = GGMLType::F32;
            }
            info.ggml_type = format;
            let (block, size) = format.type_traits();
            *bytes = if matches!(format, GGMLType::F32 | GGMLType::F16 | GGMLType::BF16) {
                (0..count)
                    .flat_map(|i| {
                        let value = if !matrix {
                            1.0
                        } else {
                            (i as f32 * 0.17).sin() * 0.01
                        };
                        match format {
                            GGMLType::F32 => value.to_le_bytes().to_vec(),
                            GGMLType::F16 => half::f16::from_f32(value).to_le_bytes().to_vec(),
                            _ => half::bf16::from_f32(value).to_le_bytes().to_vec(),
                        }
                    })
                    .collect()
            } else {
                let mut bytes = vec![1; count / block * size];
                for packed in bytes.chunks_exact_mut(size) {
                    let scale = half::f16::from_f32(1.0 / 256.0).to_le_bytes();
                    let offset = if format == GGMLType::Q6K { 208 } else { 0 };
                    packed[offset..offset + 2].copy_from_slice(&scale);
                    if matches!(format, GGMLType::Q4_1 | GGMLType::Q4K | GGMLType::Q5K) {
                        packed[2..4].fill(0);
                    }
                }
                bytes
            };
        }
        let model = Lfm3Model::from_source(&source, 2).unwrap();
        let mut session =
            Lfm3Session::new_with_compute(&model, KvFormat::F32, 19, ComputePolicy::Cpu).unwrap();
        let whole = session.forward_logits_chunked(&[1, 2, 3, 4, 5], 3).unwrap();
        assert!(whole.iter().all(|v| v.is_finite()), "format {choice}");
        assert!(whole.iter().any(|v| *v != 0.0), "format {choice}");
        session.reset();
        session.forward_logits_chunked(&[1, 2, 3], 1).unwrap();
        let incremental = session.forward_logits_chunked(&[4, 5], 1).unwrap();
        assert_eq!(whole, incremental, "format {choice}");
    }
}

#[test]
fn auto_can_fall_back_to_cpu_and_full_vulkan_is_explicit() {
    use rust_model_inference::compute::ComputePolicy;
    let source = source();
    let model = Lfm3Model::from_source(&source, 1).unwrap();
    let _cpu_scope = ComputePolicy::Cpu.enter_legacy_scope();
    let mut cpu =
        Lfm3Session::new_with_compute(&model, KvFormat::F32, 19, ComputePolicy::Cpu).unwrap();
    let expected = cpu.forward_logits_chunked(&[1, 2, 3], 1).unwrap();
    let mut auto =
        Lfm3Session::new_with_compute(&model, KvFormat::F32, 19, ComputePolicy::Auto).unwrap();
    assert_eq!(
        expected,
        auto.forward_logits_chunked(&[1, 2, 3], 3).unwrap()
    );
    assert!(ComputePolicy::Vulkan.validate_text_arch("lfm3").is_err());
    assert!(
        Lfm3Session::new_with_compute(&model, KvFormat::F32, 19, ComputePolicy::Vulkan).is_err()
    );
}
