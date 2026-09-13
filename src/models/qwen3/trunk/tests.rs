//! Test fixtures shared by `util::tests`, `base::tests`, and downstream
//! integration tests (ASR / TTS) that need a minimal Qwen3Model or a
//! mock GGUF metadata source.

#![cfg(test)]

use crate::core::scratchpad::{KvCache, KvState};
use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::models::qwen3::trunk::config::{Qwen3Config, Qwen3Rope};
use crate::models::qwen3::trunk::forward::{Qwen3GenerateOptions, Qwen3Input};
use crate::models::qwen3::trunk::session::Qwen3Session;
use crate::models::qwen3::trunk::weights::{Qwen3LayerWeights, Qwen3Model};
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

#[derive(Debug, PartialEq, Eq)]
struct KvSnapshot {
    seq_len: usize,
    words: Vec<u32>,
}

fn fixture_tokenizer() -> Arc<BPETokenizer> {
    let tokens = ["a", "b", "c", "d", "e", "f", "g", "h"];
    let metadata: HashMap<String, MetaValue> = HashMap::from([
        (
            "tokenizer.ggml.model".into(),
            MetaValue::String("gpt2".into()),
        ),
        (
            "tokenizer.ggml.pre".into(),
            MetaValue::String("qwen2".into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            MetaValue::Array(
                MetaValueType::String,
                tokens
                    .into_iter()
                    .map(|token| MetaValue::String(token.into()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            MetaValue::Array(
                MetaValueType::Uint32,
                vec![MetaValue::Uint32(1); tokens.len()],
            ),
        ),
        (
            "tokenizer.ggml.merges".into(),
            MetaValue::Array(MetaValueType::String, Vec::new()),
        ),
    ]);
    Arc::new(BPETokenizer::from_gguf_metadata(|key| metadata.get(key).cloned()).unwrap())
}

fn f32_weight(n_in: usize, n_out: usize, seed: usize) -> Weight<'static> {
    let mut bytes = Vec::with_capacity(n_in * n_out * std::mem::size_of::<f32>());
    for row in 0..n_out {
        for column in 0..n_in {
            let value = ((row * 7 + column * 13 + seed) % 23) as f32 / 64.0 - 0.171875;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
        Box::leak(bytes.into_boxed_slice()),
        GGMLType::F32,
        n_in,
        n_out,
    ));
    weight.n_in = n_in;
    weight.n_out = n_out;
    weight
}

fn deterministic_session_model(n_ctx: usize) -> Qwen3Model {
    const WIDTH: usize = 32;
    const VOCAB: usize = 8;
    Qwen3Model {
        source: Arc::new(TestTensorSource),
        tokenizer: fixture_tokenizer(),
        pool: Arc::new(ComputePool::new(2)),
        config: Qwen3Config {
            architecture: "qwen3".into(),
            n_embd: WIDTH,
            n_layer: 1,
            n_head: 1,
            n_head_kv: 1,
            n_embd_head_k: WIDTH,
            n_embd_head_v: WIDTH,
            n_ff: WIDTH,
            vocab: VOCAB,
            n_ctx,
            eps: 1e-6,
            freq_base: 10_000.0,
            has_qk_norm: false,
            has_qkv_bias: false,
            n_deepstack_layers: 0,
            moe: None,
            rope: Qwen3Rope::Neox,
        },
        layers: vec![Qwen3LayerWeights {
            attn_norm: vec![1.0; WIDTH],
            ffn_norm: vec![1.0; WIDTH],
            q_norm: None,
            k_norm: None,
            q_bias: None,
            k_bias: None,
            v_bias: None,
            moe_router: None,
            moe_gate: None,
            moe_up: None,
            moe_down: None,
            wq: f32_weight(WIDTH, WIDTH, 1),
            wk: f32_weight(WIDTH, WIDTH, 2),
            wv: f32_weight(WIDTH, WIDTH, 3),
            wo: f32_weight(WIDTH, WIDTH, 4),
            w_gate: f32_weight(WIDTH, WIDTH, 5),
            w_up: f32_weight(WIDTH, WIDTH, 6),
            w_down: f32_weight(WIDTH, WIDTH, 7),
        }],
        output_norm: vec![1.0; WIDTH],
        token_embedding: f32_weight(WIDTH, VOCAB, 8),
        output: f32_weight(WIDTH, VOCAB, 9),
    }
}

fn qwen3_fixture_session(capacity: usize, _batch_size: usize) -> Qwen3Session<'static> {
    let model = Box::leak(Box::new(deterministic_session_model(capacity.max(160))));
    Qwen3Session::new(model, capacity).unwrap()
}

fn snapshot_qwen3_kv(state: &KvState) -> KvSnapshot {
    let stride = state.arch.n_head_kv * state.arch.n_embd_head_k.max(state.arch.n_embd_head_v);
    let mut words = Vec::new();
    match &state.cache {
        KvCache::F16(cache) => {
            for layer in 0..state.arch.n_layer {
                let start = layer * state.capacity * stride;
                let end = start + state.seq_len * stride;
                words.extend(cache.k[start..end].iter().map(|&value| value as u32));
                words.extend(cache.v[start..end].iter().map(|&value| value as u32));
            }
        }
        KvCache::F32(cache) => {
            for layer in 0..state.arch.n_layer {
                let start = layer * state.capacity * stride;
                let end = start + state.seq_len * stride;
                words.extend(cache.k[start..end].iter().map(|value| value.to_bits()));
                words.extend(cache.v[start..end].iter().map(|value| value.to_bits()));
            }
        }
    }
    KvSnapshot {
        seq_len: state.seq_len,
        words,
    }
}

fn prefill_qwen3_tokens(
    session: &mut Qwen3Session<'_>,
    token_ids: &[u32],
    batch_size: usize,
) -> Result<(), String> {
    let base = session.kv_state().seq_len;
    let positions = (base..base + token_ids.len())
        .map(|position| [position, 0, 0, 0])
        .collect::<Vec<_>>();
    session
        .prefill_cpu(
            &Qwen3Input {
                token_ids,
                positions: &positions,
                embeddings: None,
                deepstack_embeddings: None,
            },
            batch_size,
        )
        .map(|_| ())
}

fn run_qwen3_fixture(prompt_len: usize, batch_size: usize) -> (Vec<u32>, KvSnapshot, Vec<u32>) {
    let mut session = qwen3_fixture_session(prompt_len + 3, batch_size);
    let token_ids = (0..prompt_len)
        .map(|index| (index % 7) as u32)
        .collect::<Vec<_>>();
    let positions = (0..prompt_len)
        .map(|position| [position, 0, 0, 0])
        .collect::<Vec<_>>();
    let generation = session
        .generate(
            Qwen3Input {
                token_ids: &token_ids,
                positions: &positions,
                embeddings: None,
                deepstack_embeddings: None,
            },
            Qwen3GenerateOptions {
                max_new_tokens: 3,
                temperature: 0.0,
                prefill_batch_size: batch_size,
            },
        )
        .unwrap();
    (
        session
            .last_logits()
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        snapshot_qwen3_kv(session.kv_state()),
        generation.token_ids,
    )
}

fn run_qwen3_prompt_snapshot(prompt_len: usize, batch_size: usize) -> (Vec<u32>, KvSnapshot) {
    let mut session = qwen3_fixture_session(prompt_len, batch_size);
    let token_ids = (0..prompt_len)
        .map(|index| (index % 7) as u32)
        .collect::<Vec<_>>();
    prefill_qwen3_tokens(&mut session, &token_ids, batch_size).unwrap();
    (
        session
            .last_logits()
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        snapshot_qwen3_kv(session.kv_state()),
    )
}

#[test]
fn qwen3_cpu_prefill_matches_batch_one_at_chunk_boundaries() {
    for len in [1, 2, 3, 63, 64, 65, 127, 128] {
        let prompt = run_qwen3_prompt_snapshot(len, 1);
        let baseline = run_qwen3_fixture(len, 1);
        for batch_size in [16, 32, 64, 128] {
            assert_eq!(
                run_qwen3_prompt_snapshot(len, batch_size),
                prompt,
                "prompt len={len} batch={batch_size}"
            );
            assert_eq!(
                run_qwen3_fixture(len, batch_size),
                baseline,
                "len={len} batch={batch_size}"
            );
        }
    }
}

#[test]
fn qwen3_oversized_prompt_commits_nothing() {
    let mut session = qwen3_fixture_session(4, 64);
    let before = snapshot_qwen3_kv(session.kv_state());
    assert!(prefill_qwen3_tokens(&mut session, &[1, 2, 3, 4, 5], 64).is_err());
    assert_eq!(snapshot_qwen3_kv(session.kv_state()), before);
}

#[test]
fn qwen3_failed_cpu_chunk_keeps_visible_kv_at_base_position() {
    let mut session = qwen3_fixture_session(8, 4);
    prefill_qwen3_tokens(&mut session, &[1, 2], 4).unwrap();
    let before = snapshot_qwen3_kv(session.kv_state());
    session.fail_cpu_prefill_after_layer_for_test(0);
    assert!(prefill_qwen3_tokens(&mut session, &[3, 4, 5], 4).is_err());
    assert_eq!(snapshot_qwen3_kv(session.kv_state()), before);
}

#[test]
fn qwen3_nonfinite_tentative_kv_never_commits() {
    let mut model = deterministic_session_model(16);
    let mut bad_key = Weight::from_quantized(QuantizedTensor::F32(vec![f32::NAN; 32 * 32]));
    bad_key.n_in = 32;
    bad_key.n_out = 32;
    model.layers[0].wk = bad_key;
    let mut session = Qwen3Session::new(Box::leak(Box::new(model)), 8).unwrap();
    let before = snapshot_qwen3_kv(session.kv_state());
    assert!(prefill_qwen3_tokens(&mut session, &[1, 2], 2).is_err());
    assert_eq!(snapshot_qwen3_kv(session.kv_state()), before);
}

#[cfg(feature = "parity-trace")]
#[test]
#[ignore = "run in a child process to isolate trace environment variables"]
fn qwen3_trace_two_tokens_child() {
    let batch_size = std::env::var("RMI_TEST_TRACE_BATCH_SIZE")
        .unwrap()
        .parse()
        .unwrap();
    let mut session = qwen3_fixture_session(2, batch_size);
    prefill_qwen3_tokens(&mut session, &[1, 2], batch_size).unwrap();
}

#[cfg(feature = "parity-trace")]
#[test]
fn qwen3_trace_keeps_token_major_checkpoints_for_every_prompt_row() {
    use std::process::Command;

    let expected_row = [
        "model.input_embed",
        "attn_norm-0",
        "Qcur_normed-0",
        "Kcur_normed-0",
        "Qcur-0",
        "Kcur-0",
        "kqv_out-0",
        "ffn_out-0",
        "result_norm",
        "result_output",
    ];
    for batch_size in [1, 2] {
        let trace = std::env::temp_dir().join(format!(
            "rmi-qwen3-prefill-trace-{}-{batch_size}.jsonl",
            std::process::id()
        ));
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "models::qwen3::trunk::tests::qwen3_trace_two_tokens_child",
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
        let records = std::fs::read_to_string(&trace).unwrap();
        let values = records
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        let names = values
            .iter()
            .map(|value| value["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(names, expected_row.repeat(2), "batch={batch_size}");
        for record in values {
            std::fs::remove_file(record["binary_path"].as_str().unwrap()).unwrap();
        }
        std::fs::remove_file(trace).unwrap();
    }
}
