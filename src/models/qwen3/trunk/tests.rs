//! Test fixtures shared by `util::tests`, `base::tests`, and downstream
//! integration tests (ASR / TTS) that need a minimal Qwen3Model or a
//! mock GGUF metadata source.

#![cfg(test)]

use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::models::qwen3::trunk::config::{Qwen3Config, Qwen3Rope};
use crate::models::qwen3::trunk::weights::Qwen3Model;
use crate::ops::kernel::{QuantizedTensor, Weight};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) struct TestTensorSource;

impl TensorSource for TestTensorSource {
    fn metadata(&self, _key: &str) -> Option<&MetaValue> {
        None
    }

    fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
        None
    }

    fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
        None
    }
}

#[derive(Default)]
pub(crate) struct MapTensorSource {
    pub(crate) metadata: HashMap<String, MetaValue>,
    pub(crate) tensors: HashMap<String, TensorInfo>,
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

pub(crate) fn qwen3vl_metadata_source() -> MapTensorSource {
    MapTensorSource {
        metadata: HashMap::from([
            (
                "general.architecture".into(),
                MetaValue::String("qwen3vl".into()),
            ),
            ("qwen3vl.embedding_length".into(), MetaValue::Uint32(1024)),
            ("qwen3vl.block_count".into(), MetaValue::Uint32(28)),
            ("qwen3vl.attention.head_count".into(), MetaValue::Uint32(16)),
            (
                "qwen3vl.attention.head_count_kv".into(),
                MetaValue::Uint32(8),
            ),
            (
                "qwen3vl.attention.key_length".into(),
                MetaValue::Uint32(128),
            ),
            (
                "qwen3vl.attention.value_length".into(),
                MetaValue::Uint32(128),
            ),
            (
                "qwen3vl.feed_forward_length".into(),
                MetaValue::Uint32(3072),
            ),
            ("qwen3vl.context_length".into(), MetaValue::Uint32(65_536)),
            (
                "qwen3vl.rope.freq_base".into(),
                MetaValue::Float32(1_000_000.0),
            ),
            (
                "qwen3vl.rope.dimension_sections".into(),
                MetaValue::Array(
                    MetaValueType::Int32,
                    [24, 20, 20, 0].map(MetaValue::Int32).to_vec(),
                ),
            ),
            (
                "qwen3vl.attention.layer_norm_rms_epsilon".into(),
                MetaValue::Float32(1e-6),
            ),
            ("qwen3vl.n_deepstack_layers".into(), MetaValue::Uint32(3)),
            ("qwen3vl.vocab_size".into(), MetaValue::Uint32(151_936)),
        ]),
        tensors: HashMap::new(),
    }
}

pub(crate) fn test_model(tokenizer: Arc<BPETokenizer>, n_ctx: usize, n_embd: usize) -> Qwen3Model {
    assert!(n_embd > 0 && n_embd % 32 == 0);
    let row_bytes = n_embd / 32 * 34;
    let embd_box = vec![0u8; tokenizer.vocab_size() * row_bytes].into_boxed_slice();
    let embd_bytes: &'static [u8] = Box::leak(embd_box);
    let token_embedding = Weight::from_quantized(QuantizedTensor::from_bytes(
        embd_bytes,
        GGMLType::Q8_0,
        n_embd,
        tokenizer.vocab_size(),
    ));
    let output = Weight::from_quantized(QuantizedTensor::from_bytes(
        embd_bytes,
        GGMLType::Q8_0,
        n_embd,
        tokenizer.vocab_size(),
    ));
    Qwen3Model {
        source: Arc::new(TestTensorSource),
        pool: Arc::new(ComputePool::new(1)),
        config: Qwen3Config {
            architecture: "qwen3".into(),
            n_embd,
            n_layer: 0,
            n_head: 1,
            n_head_kv: 1,
            n_embd_head_k: n_embd,
            n_embd_head_v: n_embd,
            n_ff: n_embd,
            vocab: tokenizer.vocab_size(),
            n_ctx,
            eps: 1e-6,
            freq_base: 1_000_000.0,
            has_qk_norm: false,
            has_qkv_bias: false,
            n_deepstack_layers: 0,
            moe: None,
            rope: Qwen3Rope::Neox,
        },
        tokenizer,
        layers: Vec::new(),
        output_norm: vec![1.0; n_embd],
        token_embedding,
        output,
    }
}
