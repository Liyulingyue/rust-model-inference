use super::*;
use crate::core::tensor::{GGMLType, MetaValue, TensorInfo};
use std::collections::HashMap;

#[derive(Default)]
struct Source {
    metadata: HashMap<String, MetaValue>,
    tensors: HashMap<String, (TensorInfo, Vec<u8>)>,
}
impl TensorSource for Source {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name).map(|tensor| &tensor.0)
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|tensor| tensor.1.as_slice())
    }
}

impl Source {
    fn with_tensor(name: &str, dims: &[u64], ggml_type: GGMLType, bytes: Vec<u8>) -> Self {
        let mut source = Self::default();
        source.tensors.insert(
            name.into(),
            (
                TensorInfo {
                    name: name.into(),
                    dims: dims.into(),
                    ggml_type,
                    offset: 0,
                },
                bytes,
            ),
        );
        source
    }
}

// Representative original safetensors shapes, reversed into GGUF dimension order.
const NON_MATRIX_TENSORS: &[(&str, &[u64])] = &[
    ("depth_decoder.codebooks_head.weight", &[2051, 1024, 15]),
    ("text_encoder.embed_tokens.eoi_embedding", &[1152]),
    ("text_encoder.norm.weight", &[1152]),
    ("backbone_model.norm.weight", &[2048]),
    ("depth_decoder.model.norm.weight", &[1024]),
    (
        "text_encoder.layers.0.pre_self_attn_layernorm.weight",
        &[1152],
    ),
    (
        "text_encoder.layers.25.post_self_attn_layernorm.weight",
        &[1152],
    ),
    (
        "text_encoder.layers.12.pre_feedforward_layernorm.weight",
        &[1152],
    ),
    (
        "text_encoder.layers.24.post_feedforward_layernorm.weight",
        &[1152],
    ),
    ("text_encoder.layers.25.self_attn.q_norm.weight", &[256]),
    ("text_encoder.layers.24.self_attn.k_norm.weight", &[256]),
    ("backbone_model.layers.27.input_layernorm.weight", &[2048]),
    (
        "backbone_model.layers.26.post_attention_layernorm.weight",
        &[2048],
    ),
    ("backbone_model.layers.27.self_attn.q_norm.weight", &[128]),
    ("backbone_model.layers.26.self_attn.k_norm.weight", &[128]),
    (
        "depth_decoder.model.layers.11.input_layernorm.weight",
        &[1024],
    ),
    (
        "depth_decoder.model.layers.10.post_attention_layernorm.weight",
        &[1024],
    ),
];

#[test]
fn main_source_rejects_non_bf16_heads_eoi_and_norms() {
    for &(name, dims) in NON_MATRIX_TENSORS {
        for dtype in [GGMLType::F32, GGMLType::F16, GGMLType::Q8_0] {
            let source = Source::with_tensor(name, dims, dtype, Vec::new());
            let error = BreezeModel::from_source(&source, 1).err().unwrap();
            assert!(
                error.contains(name) && error.contains("original BF16"),
                "{name} {dtype:?}: {error}"
            );
        }
    }
}

#[test]
fn main_source_preserves_original_bf16_and_legacy_codec_f32() {
    for &(name, dims) in NON_MATRIX_TENSORS.iter().chain(std::iter::once(&(
        "codec_model.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.initialized",
        &[1][..],
    ))) {
        let dtype = if name.starts_with("codec_model.") {
            GGMLType::F32
        } else {
            GGMLType::BF16
        };
        let source = Source::with_tensor(name, dims, dtype, Vec::new());
        let error = BreezeModel::from_source(&source, 1).err().unwrap();
        // This tiny source passes the dtype preflight, then reaches metadata validation.
        assert!(error.contains("general.architecture"), "{name}: {error}");
    }
}

#[test]
fn main_non_matrix_shapes_and_byte_lengths_fail_closed() {
    for &(name, dims) in NON_MATRIX_TENSORS {
        let mut wrong_dims = dims.to_vec();
        wrong_dims.push(1);
        let source = Source::with_tensor(name, &wrong_dims, GGMLType::BF16, Vec::new());
        let error = load_f32_tensor(&source, name, dims).unwrap_err();
        assert!(error.contains(name) && error.contains("shape"), "{error}");

        let source = Source::with_tensor(name, dims, GGMLType::BF16, vec![0]);
        let error = load_f32_tensor(&source, name, dims).unwrap_err();
        assert!(
            error.contains(name) && error.contains("data length"),
            "{error}"
        );
    }
}

#[test]
fn main_matrices_require_original_dtype_exact_shape_and_length() {
    for (name, input, output) in [
        ("lm_head.weight", 2048, 2052),
        ("text_encoder.embed_tokens.weight", 1152, 262158),
        (
            "depth_decoder.model.inputs_embeds_projector.weight",
            2048,
            1024,
        ),
    ] {
        for dtype in [GGMLType::F32, GGMLType::F16, GGMLType::Q8_0] {
            let source = Source::with_tensor(name, &[input, output], dtype, Vec::new());
            let error = matrix(&source, name, input as usize, output as usize)
                .err()
                .unwrap();
            assert!(error.contains("original BF16"), "{name}: {error}");
        }
        let source = Source::with_tensor(name, &[output, input], GGMLType::BF16, Vec::new());
        let error = matrix(&source, name, input as usize, output as usize)
            .err()
            .unwrap();
        assert!(error.contains("shape"), "{name}: {error}");

        let source = Source::with_tensor(name, &[input, output], GGMLType::BF16, vec![0]);
        let error = matrix(&source, name, input as usize, output as usize)
            .err()
            .unwrap();
        assert!(error.contains("data length"), "{name}: {error}");
    }
}

#[test]
fn architecture_and_required_config_fail_closed() {
    let mut source = Source::default();
    source.metadata.insert(
        "general.architecture".into(),
        MetaValue::String("qwen3".into()),
    );
    assert!(validate_config(&source).unwrap_err().contains("breeze"));
    source.metadata.insert(
        "general.architecture".into(),
        MetaValue::String("breeze".into()),
    );
    assert!(validate_config(&source)
        .unwrap_err()
        .contains("breeze.config"));
    source
        .metadata
        .insert("breeze.config".into(), MetaValue::String("{}".into()));
    assert!(validate_config(&source).is_err());
}

#[test]
fn reserved_codes_are_masked_but_zero_and_backbone_eos_remain_valid() {
    let mut scores = vec![0.0; 2052];
    scores[2048] = 100.0;
    scores[0] = 1.0;
    assert_eq!(greedy(&scores, true).unwrap(), 0);
    scores[2051] = 2.0;
    assert_eq!(greedy(&scores, true).unwrap(), 2051);
    assert_eq!(greedy(&scores[..2051], false).unwrap(), 0);
    scores[1] = f32::NAN;
    assert!(greedy(&scores, true).is_err());
}

#[test]
#[ignore = "requires BREEZE_MODEL_DIR containing original tokenizer.json"]
fn tokenizer_matches_official_unicode_special_tokens_and_whitespace() {
    let dir = std::env::var("BREEZE_MODEL_DIR").expect("BREEZE_MODEL_DIR");
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    for (input, expected) in [
        ("[S0]你好。", vec![2, 262146, 144626, 236924]),
        (
            "[S0]<ins_bos>温柔的女声<ins_eos>你好。",
            vec![
                2, 262146, 262156, 171109, 146440, 238389, 262157, 144626, 236924,
            ],
        ),
        (
            "[S0]Hello, world!",
            vec![2, 262146, 9259, 236764, 1902, 236888],
        ),
        (
            "  A\tB\nC 😀",
            vec![2, 138, 236776, 255968, 236799, 107, 236780, 163543],
        ),
    ] {
        assert_eq!(encode_segment(&tokenizer, input).unwrap(), expected);
    }
}

#[test]
fn sampling_is_repeatable_masks_reserved_ids_and_validates_parameters() {
    let mut scores = vec![0.0; VOCAB + 1];
    scores[2048] = 100.0;
    scores[17] = 10.0;
    let mut a = Sampling::new(0.9, 50, 0.95, 42).unwrap();
    let mut b = Sampling::new(0.9, 50, 0.95, 42).unwrap();
    let mut top_one = Sampling::new(0.9, 1, 1.0, 1).unwrap();
    for _ in 0..32 {
        let id = a.draw(&scores, true).unwrap();
        assert_eq!(id, b.draw(&scores, true).unwrap());
        assert!(id < 2048 || id == 2051);
        assert_eq!(top_one.draw(&scores, true).unwrap(), 17);
    }
    assert!(Sampling::new(f32::NAN, 0, 1.0, 42).is_err());
    assert!(Sampling::new(0.0, 0, 0.0, 42).is_err());
}

#[test]
#[ignore = "requires BREEZE_GGUF; optional RMI_PARITY_TRACE writes every main-graph checkpoint"]
fn main_graph_real_weights_replay() {
    use crate::format::ggufrs::{open_model_source, ComponentRole};
    let path = std::env::var("BREEZE_GGUF").unwrap();
    let source = open_model_source(std::path::Path::new(&path), ComponentRole::Llm).unwrap();
    let model = BreezeModel::from_source(source.as_ref(), 4).unwrap();
    if std::env::var_os("BREEZE_REPLAY_PREFILL").is_some() {
        let prompt = model.prepare_prompt("你好。", None, None).unwrap();
        tokens("breeze.prompt_ids", &prompt.ids).unwrap();
        model.pool.install(|| {
            let embeddings = model.prompt_embeddings(&prompt, None).unwrap();
            let (_, logits) = model
                .backbone_step(embeddings, &mut Cache::default(), 0)
                .unwrap();
            assert_eq!(greedy(&logits, true).unwrap(), 404);
        });
        return;
    }
    let frames = model.generate("你好。", None, None, 2, 1.0).unwrap();
    assert_eq!(
        frames,
        vec![
            [
                404, 1380, 1234, 2018, 681, 179, 1453, 1610, 770, 1245, 1839, 1223, 848, 1771, 602,
                1102
            ],
            [
                1630, 997, 439, 810, 783, 396, 293, 975, 870, 1384, 1619, 344, 1459, 170, 1558,
                1861
            ],
        ]
    );
}

#[test]
#[ignore = "requires BREEZE_GGUF and BREEZE_ORACLE_TRACE with its request manifest"]
fn main_mode_real_weights_replay() {
    use crate::format::ggufrs::{open_model_source, ComponentRole};
    let path = std::env::var("BREEZE_GGUF").unwrap();
    let oracle = std::env::var("BREEZE_ORACLE_TRACE").unwrap();
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{oracle}.manifest.json")).unwrap())
            .unwrap();
    let request = &manifest["request"];
    let records = std::fs::read_to_string(&oracle).unwrap();
    let reference_codes = records
        .lines()
        .map(|s| serde_json::from_str::<Value>(s).unwrap())
        .find(|r| r["name"] == "breeze.reference_codes")
        .map(|r| {
            r["token_ids"]
                .as_array()
                .unwrap()
                .chunks_exact(16)
                .map(|row| std::array::from_fn(|i| row[i].as_u64().unwrap() as u32))
                .collect::<Vec<[u32; 16]>>()
        });
    let reference = request["ref_text"].as_str().zip(reference_codes.as_deref());
    let source = open_model_source(std::path::Path::new(&path), ComponentRole::Llm).unwrap();
    let model = BreezeModel::from_source(source.as_ref(), 4).unwrap();
    if std::env::var_os("BREEZE_MODE_PREFILL").is_some() {
        let text = request["text"].as_str().unwrap();
        let instruction = request["instruction"].as_str();
        let prompt = model.prepare_prompt(text, instruction, reference).unwrap();
        let negative = (manifest["cfg_scale"].as_f64().unwrap() != 1.0)
            .then(|| model.prepare_prompt(text, None, reference).unwrap());
        tokens("breeze.prompt_ids", &prompt.ids).unwrap();
        if let Some(negative) = &negative {
            tokens("breeze.cfg_negative_prompt_ids", &negative.ids).unwrap();
        }
        if let Some(codes) = &reference_codes {
            tokens(
                "breeze.reference_codes",
                &codes.iter().flatten().copied().collect::<Vec<_>>(),
            )
            .unwrap();
        }
        model.pool.install(|| {
            for prompt in std::iter::once(&prompt).chain(negative.iter()) {
                let input = model.prompt_embeddings(prompt, reference).unwrap();
                model
                    .backbone_step(input, &mut Cache::default(), 0)
                    .unwrap();
            }
        });
        return;
    }
    let frames = model
        .generate(
            request["text"].as_str().unwrap(),
            request["instruction"].as_str(),
            reference,
            manifest["frames_requested"].as_u64().unwrap() as usize,
            manifest["cfg_scale"].as_f64().unwrap() as f32,
        )
        .unwrap();
    let expected = manifest["frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|frame| std::array::from_fn(|i| frame[i].as_u64().unwrap() as u32))
        .take_while(|frame: &[u32; 16]| *frame != [2050; 16])
        .collect::<Vec<_>>();
    assert_eq!(frames, expected);
}
