use super::forward::Lfm25Session;
use crate::core::scratchpad::KvFormat;
use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::dspark::DSparkTarget;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Default)]
struct FixtureSource {
    metadata: HashMap<String, MetaValue>,
    tensors: HashMap<String, TensorInfo>,
    data: HashMap<String, Vec<u8>>,
}

impl TensorSource for FixtureSource {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.data.get(name).map(Vec::as_slice)
    }
}

impl FixtureSource {
    fn f16_tensor(mut self, name: &str, dims: &[usize], values: Vec<f32>) -> Self {
        assert_eq!(values.len(), dims.iter().product::<usize>());
        self.tensors.insert(
            name.into(),
            TensorInfo {
                name: name.into(),
                dims: dims.iter().map(|&value| value as u64).collect(),
                ggml_type: GGMLType::F16,
                offset: 0,
            },
        );
        self.data.insert(
            name.into(),
            values
                .into_iter()
                .flat_map(|value| crate::ops::f32_to_f16(value).to_le_bytes())
                .collect(),
        );
        self
    }

    fn tensor(mut self, name: impl Into<String>, dims: &[usize], values: Vec<f32>) -> Self {
        let name = name.into();
        assert_eq!(values.len(), dims.iter().product::<usize>());
        self.tensors.insert(
            name.clone(),
            TensorInfo {
                name: name.clone(),
                dims: dims.iter().map(|&value| value as u64).collect(),
                ggml_type: GGMLType::F32,
                offset: 0,
            },
        );
        self.data.insert(
            name,
            values.into_iter().flat_map(f32::to_le_bytes).collect(),
        );
        self
    }
}

fn diagonal(n_in: usize, n_out: usize, scale: f32) -> Vec<f32> {
    let mut values = vec![0.0; n_in * n_out];
    for output in 0..n_out {
        values[output * n_in + output % n_in] = scale;
    }
    values
}

fn fixture() -> FixtureSource {
    const HIDDEN: usize = 256;
    const VOCAB: usize = 4;
    let metadata = HashMap::from([
        (
            "general.architecture".into(),
            MetaValue::String("lfm2".into()),
        ),
        ("lfm2.embedding_length".into(), MetaValue::Uint32(256)),
        ("lfm2.block_count".into(), MetaValue::Uint32(2)),
        ("lfm2.attention.head_count".into(), MetaValue::Uint32(2)),
        ("lfm2.attention.key_length".into(), MetaValue::Uint32(128)),
        ("lfm2.attention.value_length".into(), MetaValue::Uint32(128)),
        ("lfm2.feed_forward_length".into(), MetaValue::Uint32(256)),
        ("lfm2.context_length".into(), MetaValue::Uint32(16)),
        ("lfm2.vocab_size".into(), MetaValue::Uint32(VOCAB as u32)),
        (
            "lfm2.attention.layer_norm_rms_epsilon".into(),
            MetaValue::Float32(1e-5),
        ),
        ("lfm2.rope.freq_base".into(), MetaValue::Float32(10_000.0)),
        ("lfm2.shortconv.l_cache".into(), MetaValue::Uint32(2)),
        (
            "lfm2.attention.head_count_kv".into(),
            MetaValue::Array(
                MetaValueType::Int32,
                vec![MetaValue::Int32(1), MetaValue::Int32(0)],
            ),
        ),
    ]);
    let mut source = FixtureSource {
        metadata,
        ..Default::default()
    }
    .f16_tensor(
        "token_embd.weight",
        &[HIDDEN, VOCAB],
        diagonal(HIDDEN, VOCAB, 1.0),
    )
    .tensor(
        "output.weight",
        &[HIDDEN, VOCAB],
        diagonal(HIDDEN, VOCAB, 0.5),
    )
    .tensor("token_embd_norm.weight", &[HIDDEN], vec![1.0; HIDDEN]);

    for layer in 0..2 {
        source = source
            .tensor(
                format!("blk.{layer}.attn_norm.weight"),
                &[HIDDEN],
                vec![1.0; HIDDEN],
            )
            .tensor(
                format!("blk.{layer}.ffn_norm.weight"),
                &[HIDDEN],
                vec![1.0; HIDDEN],
            )
            .tensor(
                format!("blk.{layer}.ffn_gate.weight"),
                &[HIDDEN, HIDDEN],
                diagonal(HIDDEN, HIDDEN, 0.25),
            )
            .tensor(
                format!("blk.{layer}.ffn_up.weight"),
                &[HIDDEN, HIDDEN],
                diagonal(HIDDEN, HIDDEN, 0.5),
            )
            .tensor(
                format!("blk.{layer}.ffn_down.weight"),
                &[HIDDEN, HIDDEN],
                diagonal(HIDDEN, HIDDEN, 0.25),
            );
    }

    source = source
        .tensor("blk.0.attn_q_norm.weight", &[128], vec![1.0; 128])
        .tensor("blk.0.attn_k_norm.weight", &[128], vec![1.0; 128])
        .tensor(
            "blk.0.attn_q.weight",
            &[HIDDEN, HIDDEN],
            diagonal(HIDDEN, HIDDEN, 0.5),
        )
        .tensor(
            "blk.0.attn_k.weight",
            &[HIDDEN, 128],
            diagonal(HIDDEN, 128, 0.5),
        )
        .tensor(
            "blk.0.attn_v.weight",
            &[HIDDEN, 128],
            diagonal(HIDDEN, 128, 0.5),
        )
        .tensor(
            "blk.0.attn_output.weight",
            &[HIDDEN, HIDDEN],
            diagonal(HIDDEN, HIDDEN, 0.5),
        );

    let mut shortconv_in = vec![0.0; HIDDEN * HIDDEN * 3];
    for channel in 0..HIDDEN {
        shortconv_in[channel * HIDDEN + channel] = 0.5;
        shortconv_in[(HIDDEN + channel) * HIDDEN + channel] = 1.0;
        shortconv_in[(2 * HIDDEN + channel) * HIDDEN + channel] = 1.0;
    }
    source
        .tensor(
            "blk.1.shortconv.in_proj.weight",
            &[HIDDEN, HIDDEN * 3],
            shortconv_in,
        )
        .tensor(
            "blk.1.shortconv.out_proj.weight",
            &[HIDDEN, HIDDEN],
            diagonal(HIDDEN, HIDDEN, 0.5),
        )
        .tensor(
            "blk.1.shortconv.conv.weight",
            &[2 * HIDDEN],
            (0..HIDDEN).flat_map(|_| [0.25, 0.75]).collect(),
        )
}

#[test]
fn restore_replays_shortconv_state_and_logits_exactly() {
    let source = fixture();
    let pool = Arc::new(ComputePool::new(1));
    let mut restored = Lfm25Session::new(&source, Arc::clone(&pool), 16, KvFormat::F32).unwrap();
    let first = restored.evaluate(&[0], &[0, 1]).unwrap();
    assert_eq!(first.features.len(), 1);
    assert_eq!(first.features[0].len(), 2 * 256);
    restored.finish_prefill();
    let checkpoint = restored.checkpoint();
    restored.evaluate(&[1, 2], &[0, 1]).unwrap();
    restored.restore(&checkpoint);
    restored.evaluate(&[3], &[0, 1]).unwrap();

    let mut fresh = Lfm25Session::new(&source, pool, 16, KvFormat::F32).unwrap();
    fresh.evaluate(&[0], &[0, 1]).unwrap();
    fresh.finish_prefill();
    fresh.evaluate(&[3], &[0, 1]).unwrap();

    assert_eq!(restored.logit_bits(), fresh.logit_bits());
    assert_eq!(
        restored.shortconv_state_bits(),
        fresh.shortconv_state_bits()
    );
}

#[test]
fn prefill_history_matches_token_by_token_decode() {
    let source = fixture();
    let pool = Arc::new(ComputePool::new(1));
    let mut prefilled = Lfm25Session::new(&source, Arc::clone(&pool), 16, KvFormat::F32).unwrap();
    prefilled.evaluate(&[0, 1, 2], &[0, 1]).unwrap();
    prefilled.finish_prefill();
    prefilled.evaluate(&[3], &[0, 1]).unwrap();

    let mut decoded = Lfm25Session::new(&source, pool, 16, KvFormat::F32).unwrap();
    decoded.finish_prefill();
    decoded.evaluate(&[0, 1, 2, 3], &[0, 1]).unwrap();
    assert_eq!(prefilled.logit_bits(), decoded.logit_bits());
    assert_eq!(
        prefilled.shortconv_state_bits(),
        decoded.shortconv_state_bits()
    );
}

#[test]
fn f32_attention_handles_more_than_512_cached_tokens() {
    let mut source = fixture();
    source
        .metadata
        .insert("lfm2.context_length".into(), MetaValue::Uint32(768));
    let pool = Arc::new(ComputePool::new(1));
    let mut session = Lfm25Session::new(&source, pool, 513, KvFormat::F32).unwrap();

    session.evaluate(&vec![0; 513], &[0, 1]).unwrap();
}
