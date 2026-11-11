use super::*;
use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo};
use std::collections::HashMap;

#[test]
#[ignore = "requires remote pinned flow/HiFT GGUFs and independent Oracle input fixture"]
fn real_code2wav_checkpoints() {
    let input: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("RMI_SOPHON_CODE2WAV_INPUT").unwrap()).unwrap(),
    )
    .unwrap();
    let floats = |name: &str| -> Vec<f32> {
        input[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };
    let ids = |name: &str| -> Vec<u32> {
        input[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    };
    let pool = crate::core::thread_pool::ComputePool::new(4);
    let _scope = crate::compute::ComputePolicy::Cpu.enter_legacy_scope();
    let flow_source =
        crate::core::loader::GGUFLoader::from_file(&std::env::var("RMI_SOPHON_FLOW").unwrap())
            .unwrap();
    let flow = flow::Flow::load(&flow_source).unwrap();
    let tokens = ids("tokens");
    let prompt_tokens = ids("prompt_tokens");
    let prompt_mel = floats("prompt_mel");
    let speaker = floats("speaker");
    let noise = floats("initial_noise");
    assert!(flow
        .infer_until(
            &flow::FlowRequest::new(&tokens, &speaker, &noise),
            &pool,
            &|| true
        )
        .err()
        .unwrap()
        .contains("cancelled"));
    let result = flow
        .infer(
            &flow::FlowRequest {
                tokens: &tokens,
                prompt_tokens: &prompt_tokens,
                prompt_mel: &prompt_mel,
                speaker: &speaker,
                initial_noise: &noise,
                streaming: input["streaming"].as_bool().unwrap_or(true),
                finalize: input["finalize"].as_bool().unwrap_or(true),
                steps: input["steps"].as_u64().unwrap_or(4) as usize,
            },
            &pool,
        )
        .unwrap();
    assert!(!result.mel.is_empty());
    let hift_source =
        crate::core::loader::GGUFLoader::from_file(&std::env::var("RMI_SOPHON_HIFT").unwrap())
            .unwrap();
    let hift = hift::HiFT::load(&hift_source).unwrap();
    let hift_finalize = input["hift_finalize"].as_bool().unwrap_or(true);
    let mel_frames = input["mel_frames"].as_u64().unwrap() as usize;
    assert!(hift
        .synthesize_until(&[], mel_frames, hift_finalize, &[], &pool, &|| true)
        .unwrap_err()
        .contains("cancelled"));
    let waveform = hift
        .synthesize(
            &floats("mel"),
            mel_frames,
            hift_finalize,
            &floats("excitation_noise"),
            &pool,
        )
        .unwrap();
    assert!(waveform.iter().all(|v| v.is_finite() && v.abs() <= 0.99));
    assert_eq!(
        waveform.len(),
        (mel_frames - if hift_finalize { 0 } else { 8 }) * 480
    );
}

struct Source {
    metadata: HashMap<String, MetaValue>,
    tensors: HashMap<String, TensorInfo>,
}

impl TensorSource for Source {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }
    fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
        None
    }
}

fn source() -> Source {
    let mut source = Source {
        metadata: HashMap::new(),
        tensors: HashMap::new(),
    };
    for (key, value) in [
        ("general.architecture", ARCH),
        ("sophon_realtime.component", "thinker"),
        ("sophon_realtime.source_revision", SOURCE_REVISION),
    ] {
        source
            .metadata
            .insert(key.into(), MetaValue::String(value.into()));
    }
    for (key, value) in [
        ("embedding_length", 5120),
        ("block_count", 64),
        ("attention.head_count", 64),
        ("attention.head_count_kv", 8),
        ("attention.key_length", 128),
        ("attention.value_length", 128),
        ("feed_forward_length", 25600),
        ("vocab_size", 151936),
        ("n_deepstack_layers", 3),
        ("context_length", 262144),
    ] {
        source
            .metadata
            .insert(format!("qwen3vl.{key}"), MetaValue::Uint64(value));
    }
    source.metadata.insert(
        "qwen3vl.rope.freq_base".into(),
        MetaValue::Float32(5_000_000.0),
    );
    source.metadata.insert(
        "qwen3vl.attention.layer_norm_rms_epsilon".into(),
        MetaValue::Float32(1e-6),
    );
    source.metadata.insert(
        "qwen3vl.rope.dimension_sections".into(),
        MetaValue::Array(
            MetaValueType::Int32,
            [24, 20, 20, 0].map(MetaValue::Int32).to_vec(),
        ),
    );
    let mut tensor = |name: String, dims: Vec<u64>, ggml_type| {
        source.tensors.insert(
            name.clone(),
            TensorInfo {
                name,
                dims,
                ggml_type,
                offset: 0,
            },
        );
    };
    tensor(
        "token_embd.weight".into(),
        vec![5120, 151936],
        GGMLType::BF16,
    );
    tensor("output.weight".into(), vec![5120, 151936], GGMLType::BF16);
    tensor("output_norm.weight".into(), vec![5120], GGMLType::F32);
    for layer in 0..64 {
        for (name, dims, kind) in [
            ("attn_norm", vec![5120], GGMLType::F32),
            ("ffn_norm", vec![5120], GGMLType::F32),
            ("attn_q_norm", vec![128], GGMLType::F32),
            ("attn_k_norm", vec![128], GGMLType::F32),
            ("attn_q", vec![5120, 8192], GGMLType::Q8_0),
            ("attn_k", vec![5120, 1024], GGMLType::Q8_0),
            ("attn_v", vec![5120, 1024], GGMLType::Q8_0),
            ("attn_output", vec![8192, 5120], GGMLType::Q8_0),
            ("ffn_gate", vec![5120, 25600], GGMLType::Q8_0),
            ("ffn_up", vec![5120, 25600], GGMLType::Q8_0),
            ("ffn_down", vec![25600, 5120], GGMLType::Q8_0),
        ] {
            tensor(format!("blk.{layer}.{name}.weight"), dims, kind);
        }
    }
    source
}

#[test]
fn thinker_explicitly_exposes_the_matching_decoder_contract() {
    let source = decoder_source(Arc::new(source())).unwrap();
    assert_eq!(
        source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val),
        Some("qwen3vl")
    );
    assert_eq!(
        source
            .metadata("sophon_realtime.source_revision")
            .and_then(MetaValue::to_string_val),
        Some(SOURCE_REVISION)
    );
    assert_eq!(
        source.tensor_info("blk.63.attn_q.weight").unwrap().dims,
        [5120, 8192]
    );
    let config = crate::models::qwen3::Qwen3Config::from_source(source.as_ref()).unwrap();
    assert_eq!(
        (config.n_embd, config.n_head, config.n_embd_head_k),
        (5120, 64, 128)
    );
}

#[test]
fn unrelated_model_source_is_preserved() {
    let mut fixture = source();
    fixture.metadata.insert(
        "general.architecture".into(),
        MetaValue::String("qwen3".into()),
    );
    let original: Arc<dyn TensorSource> = Arc::new(fixture);
    let resolved = decoder_source(original.clone()).unwrap();
    assert!(Arc::ptr_eq(&original, &resolved));
}

#[test]
fn thinker_template_matches_transformers_final_block_whitespace() {
    let mut fixture = source();
    fixture.metadata.insert(
        "tokenizer.chat_template".into(),
        MetaValue::String(
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}\n\n"
                .into(),
        ),
    );
    let source = decoder_source(Arc::new(fixture)).unwrap();
    let template = crate::prompt::jinja::template_from_source(&|k| source.metadata(k).cloned())
        .unwrap()
        .unwrap();
    let rendered = template
        .render(&[], true, false, &Default::default())
        .unwrap();
    assert_eq!(rendered, "<|im_start|>assistant\n");
}

#[test]
fn wrong_component_and_incomplete_attention_fail_before_loading() {
    let mut fixture = source();
    fixture.metadata.insert(
        "sophon_realtime.component".into(),
        MetaValue::String("talker".into()),
    );
    assert!(decoder_source(Arc::new(fixture))
        .err()
        .unwrap()
        .contains("thinker"));
    let mut fixture = source();
    fixture
        .tensors
        .get_mut("blk.63.attn_q.weight")
        .unwrap()
        .dims = vec![5120, 5120];
    assert!(decoder_source(Arc::new(fixture))
        .err()
        .unwrap()
        .contains("blk.63.attn_q.weight"));
}

#[test]
fn mismatched_revision_and_dimensions_are_rejected() {
    let mut fixture = source();
    fixture.metadata.insert(
        "sophon_realtime.source_revision".into(),
        MetaValue::String("unknown".into()),
    );
    assert!(decoder_source(Arc::new(fixture))
        .err()
        .unwrap()
        .contains("revision"));
    let mut fixture = source();
    fixture
        .metadata
        .insert("qwen3vl.attention.head_count".into(), MetaValue::Uint64(40));
    assert!(decoder_source(Arc::new(fixture))
        .err()
        .unwrap()
        .contains("attention.head_count"));
}

#[test]
fn malformed_rope_and_norm_storage_are_rejected() {
    let mut fixture = source();
    fixture.metadata.remove("qwen3vl.rope.dimension_sections");
    assert!(decoder_source(Arc::new(fixture))
        .err()
        .unwrap()
        .contains("rope"));
    let mut fixture = source();
    fixture
        .tensors
        .get_mut("output_norm.weight")
        .unwrap()
        .ggml_type = GGMLType::Q8_0;
    assert!(decoder_source(Arc::new(fixture))
        .err()
        .unwrap()
        .contains("output_norm.weight"));
}
