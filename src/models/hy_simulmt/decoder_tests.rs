use super::*;
use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use std::collections::HashMap;

const HIDDEN: usize = 8;
const HEAD: usize = 4;
const VOCAB: usize = 8;
const CONTEXT: usize = 160;
const TOLERANCE: f32 = 2e-6;

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
        self.tensors.get(name).map(|(info, _)| info)
    }

    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|(_, bytes)| bytes.as_slice())
    }
}

impl Source {
    fn tensor(&mut self, name: &str, dimensions: &[usize], values: Vec<f32>) {
        assert_eq!(dimensions.iter().product::<usize>(), values.len());
        let info = TensorInfo {
            name: name.into(),
            dims: dimensions.iter().map(|&n| n as u64).collect(),
            ggml_type: GGMLType::F32,
            offset: 0,
        };
        self.tensors.insert(
            name.into(),
            (
                info,
                values.into_iter().flat_map(f32::to_le_bytes).collect(),
            ),
        );
    }

    fn matrix(&mut self, name: &str, dimensions: &[usize]) {
        let salt: usize = name.bytes().map(usize::from).sum();
        let values = (0..dimensions.iter().product())
            .map(|i| ((i * 13 + salt * 7) % 29) as f32 / 64.0 - 14.0 / 64.0)
            .collect();
        self.tensor(name, dimensions, values);
    }
}

fn model() -> HyDecoder {
    let mut source = Source::default();
    for (key, value) in [
        ("embedding_length", HIDDEN),
        ("block_count", 2),
        ("attention.head_count", 2),
        ("attention.head_count_kv", 1),
        ("attention.key_length", HEAD),
        ("attention.value_length", HEAD),
        ("feed_forward_length", 12),
        ("context_length", CONTEXT),
        ("vocab_size", VOCAB),
        ("expert_count", 3),
        ("expert_used_count", 2),
        ("expert_feed_forward_length", 6),
        ("expert_shared_feed_forward_length", 4),
        ("expert_gating_func", 2),
    ] {
        source
            .metadata
            .insert(format!("hy_v3.{key}"), MetaValue::Uint32(value as u32));
    }
    for (key, value) in [
        ("general.architecture", MetaValue::String("hy_v3".into())),
        ("hy_v3.expert_weights_norm", MetaValue::Bool(true)),
        ("hy_v3.expert_weights_scale", MetaValue::Float32(1.75)),
        ("hy_v3.rope.freq_base", MetaValue::Float32(10_000.0)),
        (
            "hy_v3.attention.layer_norm_rms_epsilon",
            MetaValue::Float32(1e-5),
        ),
        ("tokenizer.ggml.model", MetaValue::String("gpt2".into())),
        (
            "tokenizer.ggml.pre",
            MetaValue::String("hunyuan-dense".into()),
        ),
        (
            "tokenizer.ggml.tokens",
            MetaValue::Array(
                MetaValueType::String,
                (b'a'..=b'h')
                    .map(|byte| MetaValue::String(char::from(byte).to_string()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type",
            MetaValue::Array(MetaValueType::Int32, vec![MetaValue::Int32(1); VOCAB]),
        ),
        (
            "tokenizer.ggml.merges",
            MetaValue::Array(MetaValueType::String, vec![]),
        ),
    ] {
        source.metadata.insert(key.into(), value);
    }
    source.matrix("token_embd.weight", &[HIDDEN, VOCAB]);
    source.matrix("output.weight", &[HIDDEN, VOCAB]);
    source.tensor("output_norm.weight", &[HIDDEN], vec![1.0; HIDDEN]);
    for layer in 0..2 {
        let prefix = format!("blk.{layer}");
        for (name, width) in [
            ("attn_norm", HIDDEN),
            ("ffn_norm", HIDDEN),
            ("attn_q_norm", HEAD),
            ("attn_k_norm", HEAD),
        ] {
            source.tensor(
                &format!("{prefix}.{name}.weight"),
                &[width],
                (0..width).map(|i| 1.0 + (i % 3) as f32 / 32.0).collect(),
            );
        }
        for (name, input, output) in [
            ("attn_q", HIDDEN, 2 * HEAD),
            ("attn_k", HIDDEN, HEAD),
            ("attn_v", HIDDEN, HEAD),
            ("attn_output", 2 * HEAD, HIDDEN),
        ] {
            source.matrix(&format!("{prefix}.{name}.weight"), &[input, output]);
        }
        if layer == 0 {
            for (name, dimensions) in [
                ("gate", [HIDDEN, 12]),
                ("up", [HIDDEN, 12]),
                ("down", [12, HIDDEN]),
            ] {
                source.matrix(&format!("{prefix}.ffn_{name}.weight"), &dimensions);
            }
        } else {
            source.matrix(&format!("{prefix}.ffn_gate_inp.weight"), &[HIDDEN, 3]);
            source.tensor(
                &format!("{prefix}.exp_probs_b"),
                &[3],
                vec![0.0, 0.25, -0.125],
            );
            for (name, input, output) in
                [("gate", HIDDEN, 6), ("up", HIDDEN, 6), ("down", 6, HIDDEN)]
            {
                source.matrix(
                    &format!("{prefix}.ffn_{name}_exps.weight"),
                    &[input, output, 3],
                );
            }
            for (name, dimensions) in [
                ("gate", [HIDDEN, 4]),
                ("up", [HIDDEN, 4]),
                ("down", [4, HIDDEN]),
            ] {
                source.matrix(&format!("{prefix}.ffn_{name}_shexp.weight"), &dimensions);
            }
        }
    }
    let tokenizer =
        Arc::new(BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap());
    HyDecoder::from_source(Arc::new(source), tokenizer, Arc::new(ComputePool::new(2))).unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs() <= TOLERANCE * expected.abs().max(1.0),
            "{context}, logit {index}: {actual} != {expected}"
        );
    }
}

fn batch_matches_steps(format: KvFormat) {
    let model = model();
    let prompt: Vec<u32> = (0..129).map(|i| ((i * 3 + i / 4) % VOCAB) as u32).collect();
    let mut reference = HySession::new(&model, CONTEXT, format, ComputePolicy::Cpu).unwrap();
    let mut expected = Vec::new();
    for token in &prompt {
        expected = reference.forward(&[*token], None, 1).unwrap();
    }
    assert!(expected.iter().any(|value| value.abs() > 1e-3));
    let next = reference.forward(&[2], None, 1).unwrap();
    for batch in [1, 2, 128] {
        let mut session = HySession::new(&model, CONTEXT, format, ComputePolicy::Cpu).unwrap();
        let actual = session.forward(&prompt, None, batch).unwrap();
        assert_eq!(session.position(), prompt.len());
        assert_close(&actual, &expected, &format!("{format:?}, batch={batch}"));
        assert_close(
            &session.forward(&[2], None, batch).unwrap(),
            &next,
            "continuation",
        );
        assert_eq!(session.position(), prompt.len() + 1);
    }
}

#[test]
fn hy_session_f32_batch_matches_token_steps() {
    batch_matches_steps(KvFormat::F32);
}

#[test]
fn hy_session_f16_batch_matches_token_steps() {
    batch_matches_steps(KvFormat::F16);
}

#[test]
fn hy_session_external_embeddings_match_lookup_and_chunking() {
    let model = model();
    let prompt = [1, 4, 2, 6, 3];
    let embeddings = model.embed_tokens(&prompt).unwrap();
    for format in [KvFormat::F32, KvFormat::F16] {
        let mut reference = HySession::new(&model, 16, format, ComputePolicy::Cpu).unwrap();
        let expected = reference.forward(&prompt, None, 1).unwrap();
        for batch in [1, 2, 128] {
            let mut session = HySession::new(&model, 16, format, ComputePolicy::Cpu).unwrap();
            let actual = session.forward(&prompt, Some(&embeddings), batch).unwrap();
            assert_close(&actual, &expected, "supplied embeddings");
            assert_eq!(session.position(), prompt.len());
        }
        let mut changed = embeddings.clone();
        *changed.last_mut().unwrap() += 0.75;
        let mut session = HySession::new(&model, 16, format, ComputePolicy::Cpu).unwrap();
        let actual = session.forward(&prompt, Some(&changed), 2).unwrap();
        assert!(actual
            .iter()
            .zip(&expected)
            .any(|(a, b)| (a - b).abs() > 1e-4));
    }
}

#[test]
fn hy_session_rejected_inputs_preserve_position_and_cache() {
    let model = model();
    for format in [KvFormat::F32, KvFormat::F16] {
        let mut session = HySession::new(&model, 4, format, ComputePolicy::Cpu).unwrap();
        let mut reference = HySession::new(&model, 4, format, ComputePolicy::Cpu).unwrap();
        session.forward(&[1], None, 1).unwrap();
        reference.forward(&[1], None, 1).unwrap();
        for (ids, embeddings, batch) in [
            (vec![], None, 1),
            (vec![VOCAB as u32], None, 1),
            (vec![2, VOCAB as u32], None, 2),
            (vec![2], None, 0),
            (vec![2], Some(vec![0.0; HIDDEN - 1]), 1),
            (vec![2], Some(vec![f32::NAN; HIDDEN]), 1),
            (vec![2], Some(vec![f32::INFINITY; HIDDEN]), 1),
            (vec![2, 3, 4, 5], None, 128),
        ] {
            assert!(session.forward(&ids, embeddings.as_deref(), batch).is_err());
            assert_eq!(session.position(), 1);
        }
        let expected = reference.forward(&[2, 3, 4], None, 1).unwrap();
        assert_close(
            &session.forward(&[2, 3, 4], None, 2).unwrap(),
            &expected,
            "after rejected input",
        );
        assert_eq!(session.position(), 4);
        assert!(session.forward(&[5], None, 1).is_err());
        assert_eq!(session.position(), 4);
    }
}

#[cfg(not(feature = "vulkan"))]
#[test]
fn hy_session_auto_falls_back_to_cpu_without_vulkan() {
    let model = model();
    let original = ComputePolicy::legacy();
    for format in [KvFormat::F32, KvFormat::F16] {
        let mut cpu = HySession::new(&model, 16, format, ComputePolicy::Cpu).unwrap();
        let mut auto = HySession::new(&model, 16, format, ComputePolicy::Auto).unwrap();
        let expected = cpu.forward(&[1, 2, 5, 3, 4], None, 2).unwrap();
        assert_close(
            &auto.forward(&[1, 2, 5, 3, 4], None, 2).unwrap(),
            &expected,
            "Auto fallback",
        );
        assert_close(
            &auto.forward(&[6], None, 1).unwrap(),
            &cpu.forward(&[6], None, 1).unwrap(),
            "Auto continuation",
        );
        assert_eq!(auto.position(), cpu.position());
        assert_eq!(ComputePolicy::legacy(), original);
    }
}
