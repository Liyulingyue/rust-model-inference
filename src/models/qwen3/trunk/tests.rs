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

/// `qwen2vl` metadata for the Qwen2.5-VL-7B text tower that the LongCat text
/// encoder drives: `n_embd=3584`, 28 layers, 28 heads, 4 KV heads,
/// `head_dim=128`, `n_ff=18944`, `freq_base=1_000_000.0`,
/// `mrope_section=[16, 24, 24]`. `qwen2vl` is the only arch with QKV bias and
/// the only one that must not be given QK norm.
pub(crate) fn qwen2vl_metadata_source() -> MapTensorSource {
    MapTensorSource {
        metadata: HashMap::from([
            (
                "general.architecture".into(),
                MetaValue::String("qwen2vl".into()),
            ),
            ("qwen2vl.embedding_length".into(), MetaValue::Uint32(3584)),
            ("qwen2vl.block_count".into(), MetaValue::Uint32(28)),
            ("qwen2vl.attention.head_count".into(), MetaValue::Uint32(28)),
            (
                "qwen2vl.attention.head_count_kv".into(),
                MetaValue::Uint32(4),
            ),
            (
                "qwen2vl.attention.key_length".into(),
                MetaValue::Uint32(128),
            ),
            (
                "qwen2vl.attention.value_length".into(),
                MetaValue::Uint32(128),
            ),
            (
                "qwen2vl.feed_forward_length".into(),
                MetaValue::Uint32(18_944),
            ),
            ("qwen2vl.context_length".into(), MetaValue::Uint32(32_768)),
            (
                "qwen2vl.rope.freq_base".into(),
                MetaValue::Float32(1_000_000.0),
            ),
            // Qwen2.5-VL's mrope_section is [16, 24, 24]; the loader pads the
            // fourth axis with 0, which `rope_mrope` must treat as "no fourth
            // axis" rather than as a fourth rotary segment.
            (
                "qwen2vl.rope.dimension_sections".into(),
                MetaValue::Array(
                    MetaValueType::Int32,
                    [16, 24, 24, 0].map(MetaValue::Int32).to_vec(),
                ),
            ),
            (
                "qwen2vl.attention.layer_norm_rms_epsilon".into(),
                MetaValue::Float32(1e-6),
            ),
            ("qwen2vl.vocab_size".into(), MetaValue::Uint32(152_064)),
        ]),
        tensors: HashMap::new(),
    }
}

/// `qwen3vl` metadata for `Qwen3-VL-4B-Instruct` (Qwen/Qwen3-VL-4B-Instruct-GGUF).
/// The LLM backbone is the Qwen3-4B dense trunk (`n_embd=2560`, 36 layers,
/// 32 heads, 8 KV heads, `head_dim=128`, `n_ff=9728`, 256 K context,
/// `freq_base=5_000_000.0`, `mrope_section=[24, 20, 20]`).
pub(crate) fn qwen3vl_4b_metadata_source() -> MapTensorSource {
    MapTensorSource {
        metadata: HashMap::from([
            (
                "general.architecture".into(),
                MetaValue::String("qwen3vl".into()),
            ),
            ("qwen3vl.embedding_length".into(), MetaValue::Uint32(2560)),
            ("qwen3vl.block_count".into(), MetaValue::Uint32(36)),
            ("qwen3vl.attention.head_count".into(), MetaValue::Uint32(32)),
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
                MetaValue::Uint32(9728),
            ),
            ("qwen3vl.context_length".into(), MetaValue::Uint32(262_144)),
            (
                "qwen3vl.rope.freq_base".into(),
                MetaValue::Float32(5_000_000.0),
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
        cls_score: None,
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

pub(super) fn deterministic_session_model(n_ctx: usize) -> Qwen3Model {
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
        cls_score: None,
    }
}

#[test]
fn text_embeddings_mask_hides_middle_padding_from_later_tokens() {
    let model = deterministic_session_model(16);
    let positions = [[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 0], [3, 3, 3, 0]];
    let mask = [true, true, false, true];
    let baseline = model.embed_tokens(&[0, 1, 2, 3]).unwrap();
    let expected = model
        .text_encode_embeddings(baseline.clone(), &positions, &mask)
        .unwrap();
    let mut changed = baseline;
    changed[64..96].fill(10.0);
    let actual = model
        .text_encode_embeddings(changed, &positions, &mask)
        .unwrap();
    assert_eq!(&actual[96..128], &expected[96..128]);
    assert_ne!(&actual[64..96], &expected[64..96]);
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
        .prefill(
            &Qwen3Input {
                token_ids,
                positions: &positions,
                embeddings: None,
                deepstack_embeddings: None,
            },
            batch_size,
            true,
        )
        .map(|_| ())
}

fn run_qwen3_fixture(prompt_len: usize, batch_size: usize) -> (Vec<u32>, KvSnapshot, Vec<u32>) {
    let model = deterministic_session_model((prompt_len + 3).max(160));
    let mut session = Qwen3Session::new(&model, prompt_len + 3).unwrap();
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
    let model = deterministic_session_model(prompt_len.max(160));
    let mut session = Qwen3Session::new(&model, prompt_len).unwrap();
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
        for batch_size in [2, 3, 63, 64, 65, 127, 128] {
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
fn qwen3_four_axis_positions_and_deepstack_match_across_chunks() {
    let mut model = deterministic_session_model(128);
    model.config.rope = Qwen3Rope::Interleaved {
        sections: [4; 4],
        n_dims: 32,
    };
    model.config.n_deepstack_layers = 1;
    let tokens = (0..65).map(|i| (i % 8) as u32).collect::<Vec<_>>();
    let positions = (0..65)
        .map(|i| [i + 2, i * 2 + 3, i * 3 + 5, i * 5 + 7])
        .collect::<Vec<_>>();
    let deepstack = (0..65 * 32)
        .map(|i| (i % 17) as f32 / 64.0)
        .collect::<Vec<_>>();
    let run = |batch| {
        let mut session = Qwen3Session::new(&model, 128).unwrap();
        #[cfg(feature = "vulkan")]
        assert!(
            session.gpu.is_none(),
            "deepstack must be ineligible for the full Vulkan executor"
        );
        session
            .prefill(
                &Qwen3Input {
                    token_ids: &tokens,
                    positions: &positions,
                    embeddings: None,
                    deepstack_embeddings: Some(&deepstack),
                },
                batch,
                true,
            )
            .unwrap();
        (
            session
                .last_logits()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            snapshot_qwen3_kv(session.kv_state()),
        )
    };
    assert_eq!(run(64), run(1));
}

#[test]
fn qwen3_capacity_and_scratch_boundaries() {
    let model = deterministic_session_model(128);
    let mut session = Qwen3Session::new(&model, 128).unwrap();
    assert!(prefill_qwen3_tokens(&mut session, &[], 64).is_err());
    prefill_qwen3_tokens(&mut session, &[1; 64], 64).unwrap();
    let bytes = session.scratch_bytes();
    prefill_qwen3_tokens(&mut session, &[1; 64], 64).unwrap();
    assert_eq!(session.kv_state().seq_len, 128);
    assert_eq!(session.scratch_bytes(), bytes);
    let before = snapshot_qwen3_kv(session.kv_state());
    assert!(prefill_qwen3_tokens(&mut session, &[1], 64).is_err());
    assert_eq!(snapshot_qwen3_kv(session.kv_state()), before);
    let mut one = Qwen3Session::new(&model, 128).unwrap();
    prefill_qwen3_tokens(&mut one, &[1; 128], 1).unwrap();
    assert!(one.scratch_bytes() < bytes);
}

#[test]
fn qwen3_oversized_prompt_commits_nothing() {
    let model = deterministic_session_model(4);
    let mut session = Qwen3Session::new(&model, 4).unwrap();
    let before = snapshot_qwen3_kv(session.kv_state());
    assert!(prefill_qwen3_tokens(&mut session, &[1, 2, 3, 4, 5], 64).is_err());
    assert_eq!(snapshot_qwen3_kv(session.kv_state()), before);
}

#[test]
fn qwen3_failed_cpu_chunk_keeps_visible_kv_at_base_position() {
    let model = deterministic_session_model(8);
    let mut session = Qwen3Session::new(&model, 8).unwrap();
    prefill_qwen3_tokens(&mut session, &[1, 2], 4).unwrap();
    let before = snapshot_qwen3_kv(session.kv_state());
    session.fail_cpu_prefill_after_layer_for_test(0);
    assert!(prefill_qwen3_tokens(&mut session, &[3, 4, 5], 4).is_err());
    assert_eq!(snapshot_qwen3_kv(session.kv_state()), before);
}

#[test]
fn qwen3_nonfinite_tentative_kv_never_commits() {
    let mut model = deterministic_session_model(16);
    let mut bad_key = Weight::from_quantized(QuantizedTensor::F32 {
        data: vec![f32::NAN; 32 * 32],
        n_in: 32,
        n_out: 32,
    });
    model.layers[0].wk = bad_key;
    let mut session = Qwen3Session::new(&model, 8).unwrap();
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
    let model = deterministic_session_model(2);
    let mut session = Qwen3Session::new(&model, 2).unwrap();
    let singleton_scratch = session.scratch_bytes();
    prefill_qwen3_tokens(&mut session, &[1, 2], batch_size).unwrap();
    if batch_size > 1 {
        assert!(
            session.scratch_bytes() > singleton_scratch,
            "trace must retain the requested multi-row scratch"
        );
    }
}

#[cfg(feature = "parity-trace")]
#[test]
fn qwen3_trace_keeps_token_major_checkpoints_for_every_prompt_row() {
    use std::process::Command;

    let expected_row = [
        "model.input_embed",
        "attn_norm-0",
        "Qcur_raw-0",
        "Kcur_raw-0",
        "Qcur_normed-0",
        "Kcur_normed-0",
        "Qcur-0",
        "Kcur-0",
        "kqv_out-0",
        "ffn_out-0",
        "result_norm",
        "result_output",
    ];
    let mut baseline = None;
    for batch_size in [1, 64] {
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
        let snapshot = values
            .iter()
            .map(|record| {
                (
                    record["name"].clone(),
                    record["layer"].clone(),
                    record["shape"].clone(),
                    std::fs::read(record["binary_path"].as_str().unwrap()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        if let Some(expected) = &baseline {
            assert_eq!(&snapshot, expected, "batch={batch_size}");
        } else {
            baseline = Some(snapshot);
        }
        for record in values {
            std::fs::remove_file(record["binary_path"].as_str().unwrap()).unwrap();
        }
        std::fs::remove_file(trace).unwrap();
    }
}

#[test]
fn raw_hidden_sequence_preserves_normalization_and_compute_policy() {
    use crate::compute::ComputePolicy;
    use crate::core::scratchpad::{KvFormat, KvLifecycle};
    let model = deterministic_session_model(8);
    let tokens = [1, 2, 1];
    let positions = super::positions::qwen_text_positions(tokens.len());
    let input = || Qwen3Input {
        token_ids: &tokens,
        positions: &positions,
        embeddings: None,
        deepstack_embeddings: None,
    };
    let session = || {
        Qwen3Session::new_with_compute(
            &model,
            8,
            KvFormat::F32,
            KvLifecycle::Ephemeral,
            ComputePolicy::Cpu,
        )
        .unwrap()
    };
    let mut raw_session = session();
    let raw = raw_session.forward_hidden_sequence_raw(input()).unwrap();
    let mut normalized_session = session();
    let normalized = normalized_session.forward_hidden_sequence(input()).unwrap();
    assert_eq!(raw_session.kv_state().seq_len, tokens.len());
    assert_eq!(normalized_session.kv_state().seq_len, tokens.len());
    assert!(raw
        .iter()
        .zip(&normalized)
        .any(|(a, b)| a.to_bits() != b.to_bits()));
    let width = model.config.n_embd;
    let mut expected = vec![0.0; normalized.len()];
    for (source, destination) in raw
        .chunks_exact(width)
        .zip(expected.chunks_exact_mut(width))
    {
        crate::ops::rms_norm(source, &model.output_norm, destination, model.config.eps);
    }
    assert_eq!(
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        normalized.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    for raw in [false, true] {
        let mut forced = session();
        // Check the shared API guard without initializing a physical device.
        forced.compute_policy = ComputePolicy::Vulkan;
        let result = if raw {
            forced.forward_hidden_sequence_raw(input())
        } else {
            forced.forward_hidden_sequence(input())
        };
        assert!(result.unwrap_err().contains("Vulkan"));
        assert_eq!(forced.kv_state().seq_len, 0);
    }
}

#[test]
fn hidden_sequence_preserves_every_f32_kv_row() {
    use crate::core::scratchpad::{KvFormat, KvLifecycle};
    let model = deterministic_session_model(8);
    let tokens = [1, 2, 1];
    let positions = super::positions::qwen_text_positions(tokens.len());
    let mut batched =
        Qwen3Session::new_with_kv_state(&model, 8, KvFormat::F32, KvLifecycle::Ephemeral).unwrap();
    let actual = batched
        .forward_hidden_sequence(Qwen3Input {
            token_ids: &tokens,
            positions: &positions,
            embeddings: None,
            deepstack_embeddings: None,
        })
        .unwrap();
    let mut sequential =
        Qwen3Session::new_with_kv_state(&model, 8, KvFormat::F32, KvLifecycle::Ephemeral).unwrap();
    let mut expected = Vec::new();
    for i in 0..tokens.len() {
        expected.extend(
            sequential
                .forward_last_hidden(
                    Qwen3Input {
                        token_ids: &tokens[i..i + 1],
                        positions: &positions[i..i + 1],
                        embeddings: None,
                        deepstack_embeddings: None,
                    },
                    1,
                )
                .unwrap(),
        );
    }
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert!(batched
        .forward_hidden_sequence(Qwen3Input {
            token_ids: &tokens,
            positions: &positions,
            embeddings: None,
            deepstack_embeddings: None
        })
        .is_err());
}
