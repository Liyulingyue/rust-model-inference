use super::config::{CONTEXT, EPS};
use super::{
    assemble_input_rows, attend, kv_source_layer, load_weight, matmul, require_f32_kv, softcap,
    Gemma4InputRow, Gemma4Layer, Gemma4Model, KvLayer, PER_LAYER, VOCAB,
};
use crate::core::scratchpad::KvFormat;
use crate::core::tensor::{GGMLType, TensorInfo, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::gemma4::Gemma4Config;
use crate::ops::kernel::{Kernel, QuantizedTensor, Weight};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const TEST_LAYERS: usize = 35;
const TEST_EMBD: usize = 1536;
const TEST_FFN_PER_LAYER: [usize; 35] = [
    6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144,
    12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288,
    12288, 12288, 12288, 12288, 12288, 12288, 12288,
];
const TEST_SWA_PATTERN: [bool; 35] = [
    true, true, true, true, false, true, true, true, true, false, true, true, true, true, false,
    true, true, true, true, false, true, true, true, true, false, true, true, true, true, false,
    true, true, true, true, false,
];

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

struct DeterministicKernel {
    seed: usize,
    output_projection_calls: Option<Arc<AtomicUsize>>,
    fail_after: Option<usize>,
}

impl Kernel for DeterministicKernel {
    fn forward_prequantized(
        &self,
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        if let Some(calls) = &self.output_projection_calls {
            let call = calls.fetch_add(1, Ordering::Relaxed);
            if self.fail_after.is_some_and(|limit| call >= limit) {
                output[..n_out].fill(f32::NAN);
                return;
            }
        }
        let per_thread = n_out.div_ceil(nth);
        let start = ith * per_thread;
        let end = (start + per_thread).min(n_out);
        for out in start..end {
            let input = (out.wrapping_mul(17) + self.seed) % n_in;
            output[out] = input_q8[input] as i8 as f32
                * input_scales[input / 32]
                * (1.0 + (self.seed % 7) as f32 / 16.0);
        }
    }

    fn embedding_lookup(&self, token_id: u32, n_embd: usize, output: &mut [f32]) {
        for (index, value) in output.iter_mut().take(n_embd).enumerate() {
            *value = ((token_id as usize + index + self.seed) % 17) as f32 / 16.0 - 0.5;
        }
    }
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

fn deterministic_weight(n_in: usize, n_out: usize, seed: usize) -> Weight<'static> {
    Weight {
        kernel: Box::new(DeterministicKernel {
            seed,
            output_projection_calls: None,
            fail_after: None,
        }),
        ggml_type: GGMLType::Q8_0,
        n_in,
        n_out,
    }
}

fn counting_output_weight(
    n_in: usize,
    n_out: usize,
    output_projection_calls: Arc<AtomicUsize>,
) -> Weight<'static> {
    Weight {
        kernel: Box::new(DeterministicKernel {
            seed: 1,
            output_projection_calls: Some(output_projection_calls),
            fail_after: None,
        }),
        ggml_type: GGMLType::Q8_0,
        n_in,
        n_out,
    }
}

fn test_config() -> Gemma4Config {
    Gemma4Config {
        layers: TEST_LAYERS,
        embd: TEST_EMBD,
        n_heads: 8,
        kv_heads_per_layer: vec![1; TEST_LAYERS],
        vocab: VOCAB,
        full_head_dim: 512,
        swa_head_dim: 256,
        shared_kv_layers: 20,
        per_layer_width: PER_LAYER,
        sliding_window: 512,
        logit_softcap: 30.0,
        ffn_per_layer: TEST_FFN_PER_LAYER.to_vec(),
        swa_pattern: TEST_SWA_PATTERN.to_vec(),
        n_ctx: CONTEXT,
        rope_freq_base: 1_000_000.0,
        rope_freq_base_swa: 10_000.0,
    }
}

fn zero_layer(layer: usize, cfg: &Gemma4Config) -> Gemma4Layer {
    let dim = cfg.head_dim(layer);
    let q_heads = cfg.n_heads;
    let kv_heads = cfg.kv_heads(layer);
    let ffn = cfg.ffn_per_layer[layer];
    let embd = cfg.embd;
    Gemma4Layer {
        q_heads,
        kv_heads,
        head_dim: dim,
        kv_shared_with_k: false,
        attn_norm: vec![1.0; embd],
        attn_q: zero_q8_weight(embd, q_heads * dim),
        attn_k: zero_q8_weight(embd, kv_heads * dim),
        attn_v: Some(zero_q8_weight(embd, kv_heads * dim)),
        attn_output: zero_q8_weight(q_heads * dim, embd),
        attn_q_norm: vec![1.0; dim],
        attn_k_norm: Some(vec![1.0; dim]),
        post_attention_norm: vec![1.0; embd],
        ffn_norm: vec![1.0; embd],
        ffn_gate: zero_q8_weight(embd, ffn),
        ffn_up: zero_q8_weight(embd, ffn),
        ffn_down: zero_q8_weight(ffn, embd),
        post_ffw_norm: vec![1.0; embd],
        inp_gate: Some(zero_weight(embd, PER_LAYER)),
        proj: Some(zero_weight(PER_LAYER, embd)),
        post_norm: Some(vec![1.0; embd]),
        output_scale: 1.0,
    }
}

fn post_kv_failure_model() -> Gemma4Model {
    let cfg = test_config();
    let mut layers = (0..cfg.layers)
        .map(|l| zero_layer(l, &cfg))
        .collect::<Vec<_>>();
    layers[0].attn_output.n_in += 1;
    let embd = cfg.embd;
    let per_layer_all = cfg.per_layer_all();
    let head_dim_half = cfg.full_head_dim / 2;
    Gemma4Model {
        _source: Arc::new(EmptySource),
        config: cfg,
        pool: Arc::new(ComputePool::new(1)),
        token_embedding: zero_weight(embd, VOCAB),
        per_layer_token_embedding: Some(zero_weight(per_layer_all, VOCAB)),
        per_layer_model_proj: Some(zero_bf16_weight(embd, per_layer_all)),
        per_layer_proj_norm: Some(vec![1.0; PER_LAYER]),
        output_norm: vec![1.0; embd],
        rope_freqs: vec![1.0; head_dim_half],
        layers,
    }
}

fn deterministic_config() -> Gemma4Config {
    Gemma4Config {
        layers: 3,
        embd: 32,
        n_heads: 8,
        kv_heads_per_layer: vec![1; 3],
        vocab: VOCAB,
        full_head_dim: 512,
        swa_head_dim: 256,
        shared_kv_layers: 1,
        per_layer_width: PER_LAYER,
        sliding_window: 512,
        logit_softcap: 30.0,
        ffn_per_layer: vec![64; 3],
        swa_pattern: vec![true, false, true],
        n_ctx: 128,
        rope_freq_base: 1_000_000.0,
        rope_freq_base_swa: 10_000.0,
    }
}

fn deterministic_model(output_projection_calls: Arc<AtomicUsize>) -> Gemma4Model {
    deterministic_model_with_config(deterministic_config(), output_projection_calls)
}

fn deterministic_model_with_config(
    cfg: Gemma4Config,
    output_projection_calls: Arc<AtomicUsize>,
) -> Gemma4Model {
    let layers: Vec<Gemma4Layer> = (0..cfg.layers)
        .map(|layer| {
            let dim = cfg.head_dim(layer);
            let q_heads = cfg.n_heads;
            let kv_heads = cfg.kv_heads(layer);
            let ffn = cfg.ffn_per_layer[layer];
            Gemma4Layer {
                q_heads,
                kv_heads,
                head_dim: dim,
                kv_shared_with_k: false,
                attn_norm: vec![1.0; cfg.embd],
                attn_q: deterministic_weight(cfg.embd, q_heads * dim, layer * 11 + 1),
                attn_k: deterministic_weight(cfg.embd, kv_heads * dim, layer * 11 + 2),
                attn_v: Some(deterministic_weight(cfg.embd, kv_heads * dim, layer * 11 + 3)),
                attn_output: deterministic_weight(q_heads * dim, cfg.embd, layer * 11 + 4),
                attn_q_norm: vec![1.0; dim],
                attn_k_norm: Some(vec![1.0; dim]),
                post_attention_norm: vec![1.0; cfg.embd],
                ffn_norm: vec![1.0; cfg.embd],
                ffn_gate: deterministic_weight(cfg.embd, ffn, layer * 11 + 5),
                ffn_up: deterministic_weight(cfg.embd, ffn, layer * 11 + 6),
                ffn_down: deterministic_weight(ffn, cfg.embd, layer * 11 + 7),
                post_ffw_norm: vec![1.0; cfg.embd],
                inp_gate: Some(deterministic_weight(cfg.embd, PER_LAYER, layer * 11 + 8)),
                proj: Some(deterministic_weight(PER_LAYER, cfg.embd, layer * 11 + 9)),
                post_norm: Some(vec![1.0; cfg.embd]),
                output_scale: 0.75 + layer as f32 / 16.0,
            }
        })
        .collect();
    let embd = cfg.embd;
    let per_layer_all = cfg.per_layer_all();
    let head_dim_half = layers[1].head_dim / 2;
    Gemma4Model {
        _source: Arc::new(EmptySource),
        config: cfg,
        pool: Arc::new(ComputePool::new(1)),
        token_embedding: counting_output_weight(embd, VOCAB, output_projection_calls),
        per_layer_token_embedding: Some(deterministic_weight(per_layer_all, VOCAB, 41)),
        per_layer_model_proj: Some(zero_bf16_weight(embd, per_layer_all)),
        per_layer_proj_norm: Some(vec![1.0; PER_LAYER]),
        output_norm: vec![1.0; embd],
        rope_freqs: vec![1.0; head_dim_half],
        layers,
    }
}

#[derive(Debug, PartialEq)]
struct Gemma4Snapshot {
    logits: Vec<u32>,
    base_kv: Vec<(Vec<u32>, Vec<u32>)>,
    seq_len: usize,
    decode_ids: [u32; 3],
}

#[derive(Debug, PartialEq)]
struct CountingGemma4Snapshot {
    output_projection_calls: usize,
}

#[derive(Debug, PartialEq)]
struct Gemma4StateSnapshot {
    seq_len: usize,
    base_kv: Vec<(Vec<u32>, Vec<u32>)>,
}

fn snapshot_base_kv(session: &super::Gemma4Session<'_>) -> Vec<(Vec<u32>, Vec<u32>)> {
    session
        .kv
        .iter()
        .map(|layer| {
            (
                layer.keys.iter().map(|value| value.to_bits()).collect(),
                layer.values.iter().map(|value| value.to_bits()).collect(),
            )
        })
        .collect()
}

fn snapshot_gemma4_state(session: &super::Gemma4Session<'_>) -> Gemma4StateSnapshot {
    Gemma4StateSnapshot {
        seq_len: session.len(),
        base_kv: snapshot_base_kv(session),
    }
}

fn fixture_rows(len: usize) -> Vec<Gemma4InputRow> {
    (0..len)
        .map(|index| Gemma4InputRow::Token((index % 13 + 1) as u32))
        .collect()
}

fn greedy_id(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0 as u32
}

fn run_gemma4_fixture(len: usize, batch: usize) -> Gemma4Snapshot {
    let output_projection_calls = Arc::new(AtomicUsize::new(0));
    let model = deterministic_model(output_projection_calls);
    let mut session =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, batch).unwrap();
    let logits = session.forward_rows(&fixture_rows(len)).unwrap();
    let prompt_logits = logits.iter().map(|value| value.to_bits()).collect();
    let prompt_base_kv = snapshot_base_kv(&session);
    let prompt_seq_len = session.len();
    let mut decode_logits = logits;
    let mut decode_ids = [0; 3];
    for (index, id) in decode_ids.iter_mut().enumerate() {
        *id = greedy_id(&decode_logits);
        if index + 1 < 3 {
            decode_logits = session.forward_rows(&[Gemma4InputRow::Token(*id)]).unwrap();
        }
    }
    Gemma4Snapshot {
        logits: prompt_logits,
        base_kv: prompt_base_kv,
        seq_len: prompt_seq_len,
        decode_ids,
    }
}

fn run_counting_gemma4_fixture(prompt_len: usize, batch_size: usize) -> CountingGemma4Snapshot {
    let output_projection_calls = Arc::new(AtomicUsize::new(0));
    let model = deterministic_model(Arc::clone(&output_projection_calls));
    let mut session =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, batch_size)
            .unwrap();
    session.forward_rows(&fixture_rows(prompt_len)).unwrap();
    CountingGemma4Snapshot {
        output_projection_calls: output_projection_calls.load(Ordering::Relaxed),
    }
}

#[cfg(feature = "vulkan")]
#[derive(Debug, Clone)]
pub(super) struct LinearCall {
    name: String,
    rows: usize,
}

#[cfg(feature = "vulkan")]
#[derive(Default)]
pub(super) struct LinearDispatcher {
    pub(super) calls: Vec<LinearCall>,
    pub(super) failures: usize,
    pub(super) fail_at: Option<usize>,
}

#[cfg(feature = "vulkan")]
impl LinearDispatcher {
    pub(super) fn dispatch(
        &mut self,
        name: &str,
        rows: usize,
    ) -> Result<(), crate::vulkan::VulkanError> {
        self.calls.push(LinearCall {
            name: name.to_owned(),
            rows,
        });
        if self.fail_at == Some(self.calls.len()) {
            Err(crate::vulkan::VulkanError::Timeout)
        } else {
            Err(crate::vulkan::VulkanError::UnsupportedShape(
                "injected CPU fallback".into(),
            ))
        }
    }
}

#[cfg(feature = "vulkan")]
fn run_counting_gemma4_vulkan_fixture(prompt_len: usize, batch_size: usize) -> Vec<LinearCall> {
    let model = deterministic_model(Arc::new(AtomicUsize::new(0)));
    let mut session =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, batch_size)
            .unwrap();
    let dispatcher = Arc::new(std::sync::Mutex::new(LinearDispatcher::default()));
    session.prefill_linear.dispatcher = Some(Arc::clone(&dispatcher));
    session.forward_rows(&fixture_rows(prompt_len)).unwrap();
    session.prefill_linear.dispatcher = None;
    Arc::try_unwrap(dispatcher)
        .ok()
        .unwrap()
        .into_inner()
        .unwrap()
        .calls
}

#[cfg(feature = "vulkan")]
#[test]
fn gemma4_vulkan_linear_dispatches_once_per_projection_chunk() {
    let calls = run_counting_gemma4_vulkan_fixture(3, 3);
    assert!(calls.iter().all(|call| call.rows == 3));
    assert_eq!(
        calls
            .iter()
            .filter(|call| call.name == "blk.0.attn_q.weight")
            .count(),
        1
    );
    assert_eq!(calls.len(), 26);
}

#[cfg(feature = "vulkan")]
#[test]
fn gemma4_vulkan_linear_batch_one_and_decode_isolation() {
    let calls = run_counting_gemma4_vulkan_fixture(3, 1);
    assert_eq!(calls.len(), 78);
    assert!(calls.iter().all(|call| call.rows == 1));
    let model = deterministic_model(Arc::new(AtomicUsize::new(0)));
    let mut session =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, 3).unwrap();
    let dispatcher = Arc::new(std::sync::Mutex::new(LinearDispatcher::default()));
    session.prefill_linear.dispatcher = Some(Arc::clone(&dispatcher));
    session.forward_rows(&fixture_rows(1)).unwrap();
    assert_eq!(dispatcher.lock().unwrap().calls.len(), 26);
    model.pool().clear_gpu_disabled_workers_for_test();
    session.forward_rows(&fixture_rows(1)).unwrap();
    assert_eq!(dispatcher.lock().unwrap().calls.len(), 26);
    assert_eq!(model.pool().gpu_disabled_workers_for_test(), 1);
    assert_eq!(session.len(), 2);
}

#[cfg(feature = "vulkan")]
#[test]
fn gemma4_vulkan_linear_fallback_preserves_logits_and_kv() {
    let expected = run_gemma4_fixture(3, 3);
    for fail_at in [None, Some(3)] {
        let model = deterministic_model(Arc::new(AtomicUsize::new(0)));
        let mut session =
            super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, 3).unwrap();
        let dispatcher = Arc::new(std::sync::Mutex::new(LinearDispatcher {
            fail_at,
            ..Default::default()
        }));
        session.prefill_linear.dispatcher = Some(Arc::clone(&dispatcher));
        let logits = session.forward_rows(&fixture_rows(3)).unwrap();
        assert_eq!(
            logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.logits
        );
        assert_eq!(snapshot_base_kv(&session), expected.base_kv);
        assert_eq!(session.len(), expected.seq_len);
        let calls = dispatcher.lock().unwrap();
        assert_eq!(calls.calls.len(), fail_at.unwrap_or(26));
        assert_eq!(calls.failures, usize::from(fail_at.is_some()));
        drop(calls);
        if fail_at.is_some() {
            assert!(session.prefill_linear.runtime.is_none());
            assert!(!session.prefill_linear.active());
            session.forward_rows(&fixture_rows(3)).unwrap();
            assert_eq!(dispatcher.lock().unwrap().calls.len(), 3);
            assert_eq!(dispatcher.lock().unwrap().failures, 1);
        }
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn gemma4_vulkan_linear_cpu_scope_disables_dispatch() {
    let _scope = ComputePool::disable_gpu_matmul_for_scope();
    assert!(run_counting_gemma4_vulkan_fixture(3, 3).is_empty());
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "requires a Vulkan device"]
fn gemma4_vulkan_linear_device_rows_match_and_decode_stays_cpu() {
    use super::session::Gemma4PrefillLinear;
    use std::collections::HashMap;

    struct ProjectionSource(HashMap<String, (TensorInfo, Vec<u8>)>);
    impl TensorSource for ProjectionSource {
        fn metadata(&self, _: &str) -> Option<&crate::core::tensor::MetaValue> {
            None
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.0.get(name).map(|entry| &entry.0)
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.0.get(name).map(|entry| entry.1.as_slice())
        }
    }
    let mut source = ProjectionSource(HashMap::new());
    let mut install = |name: String, weight: &mut Weight<'static>, ggml_type: GGMLType| {
        let elements = weight.n_in * weight.n_out;
        let bytes = match ggml_type {
            GGMLType::BF16 => (0..elements)
                .flat_map(|i| crate::ops::f32_to_bf16((i % 7) as f32 / 1024.0).to_le_bytes())
                .collect(),
            GGMLType::F32 => (0..elements)
                .flat_map(|i| ((i % 11) as f32 / 1024.0).to_le_bytes())
                .collect(),
            GGMLType::Q8_0 => (0..elements / 32)
                .flat_map(|block| {
                    let mut bytes = [0u8; 34];
                    bytes[..2].copy_from_slice(&crate::ops::f32_to_f16(1.0 / 128.0).to_le_bytes());
                    for (i, value) in bytes[2..].iter_mut().enumerate() {
                        *value = (((block + i * 3) % 9) as i8 - 4) as u8;
                    }
                    bytes
                })
                .collect(),
            _ => unreachable!(),
        };
        let dims = vec![weight.n_in as u64, weight.n_out as u64];
        source.0.insert(
            name.clone(),
            (
                TensorInfo {
                    name: name.clone(),
                    dims: dims.clone(),
                    ggml_type,
                    offset: 0,
                },
                bytes,
            ),
        );
        *weight = load_weight(&source, &name, &dims, ggml_type).unwrap();
    };
    let mut model = deterministic_model(Arc::new(AtomicUsize::new(0)));
    install(
        "per_layer_model_proj.weight".into(),
        &mut model.per_layer_model_proj,
        GGMLType::BF16,
    );
    for (index, layer) in model.layers.iter_mut().enumerate() {
        for (name, weight) in [
            ("attn_q", &mut layer.attn_q),
            ("attn_k", &mut layer.attn_k),
            ("attn_v", &mut layer.attn_v),
            ("attn_output", &mut layer.attn_output),
            ("ffn_gate", &mut layer.ffn_gate),
            ("ffn_up", &mut layer.ffn_up),
            ("ffn_down", &mut layer.ffn_down),
        ] {
            install(format!("blk.{index}.{name}.weight"), weight, GGMLType::Q8_0);
        }
        install(
            format!("blk.{index}.inp_gate.weight"),
            &mut layer.inp_gate,
            GGMLType::F32,
        );
        install(
            format!("blk.{index}.proj.weight"),
            &mut layer.proj,
            GGMLType::F32,
        );
    }
    model._source = Arc::new(source);
    crate::ops::enable_gpu();
    let context = crate::ops::get_vulkan_context().expect("Vulkan device and warmup");
    let (n_in, n_out, descriptors) = Gemma4PrefillLinear::limits(&model).unwrap();
    assert_eq!((n_in, n_out, descriptors), (4096, 4096, 27));
    let run = |batch| {
        let mut session =
            super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, batch)
                .unwrap();
        assert!(session.prefill_linear.runtime.is_some());
        let before = context.submission_count();
        let logits = session.forward_rows(&fixture_rows(3)).unwrap();
        assert_eq!(
            context.submission_count() - before,
            if batch == 1 { 78 } else { 26 }
        );
        let state = snapshot_gemma4_state(&session);
        let before_decode = context.submission_count();
        let decode = session
            .forward_rows(&[Gemma4InputRow::Token(greedy_id(&logits))])
            .unwrap();
        assert_eq!(context.submission_count(), before_decode);
        (
            logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            state,
            decode.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        )
    };
    assert_eq!(run(3), run(1));
    println!(
        "Gemma4 Vulkan device={} BF16/F32/Q8_0 prompt_rows=3 exact_bits=true decode_dispatches=0",
        context.device_name()
    );
}

#[test]
fn gemma4_prefill_matches_batch_one_across_chunk_boundaries() {
    for len in [1, 2, 3, 63, 64, 65, 127, 128] {
        let expected = run_gemma4_fixture(len, 1);
        for batch in [2, 3, 63, 64, 65, 127, 128] {
            assert_eq!(run_gemma4_fixture(len, batch), expected);
        }
    }
}

#[test]
fn gemma4_fixture_snapshots_prompt_state_before_decode() {
    assert_eq!(run_gemma4_fixture(3, 1).seq_len, 3);
}

#[test]
fn gemma4_reset_reuses_scratch_and_matches_fresh_session() {
    let model = deterministic_model(Arc::new(AtomicUsize::new(0)));
    let mut session =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, 64).unwrap();
    let expected = session
        .forward_rows(&fixture_rows(3))
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>();
    let bytes = session.scratch_bytes();
    let pointer = session.scratch.x.as_ptr();
    session.reset();
    assert_eq!(session.len(), 0);
    assert!(session
        .kv
        .iter()
        .all(|kv| kv.keys.is_empty() && kv.values.is_empty()));
    let actual = session
        .forward_rows(&fixture_rows(3))
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert_eq!(session.scratch_bytes(), bytes);
    assert_eq!(session.scratch.x.as_ptr(), pointer);
}

#[test]
fn gemma4_scratch_is_bounded_by_batch_for_prompt_boundaries() {
    let model = deterministic_model(Arc::new(AtomicUsize::new(0)));
    let mut session =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, 64).unwrap();
    assert!(session.forward_rows(&[]).is_err());
    session.forward_rows(&fixture_rows(64)).unwrap();
    let bytes = session.scratch_bytes();
    session.forward_rows(&fixture_rows(64)).unwrap();
    assert_eq!(session.scratch_bytes(), bytes);
    let mut one =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, 1).unwrap();
    one.forward_rows(&fixture_rows(2)).unwrap();
    assert!(one.scratch_bytes() < bytes);
}

#[cfg(feature = "parity-trace")]
#[test]
#[ignore = "child process isolates trace environment"]
fn gemma4_trace_child() {
    let batch = std::env::var("RMI_TEST_TRACE_BATCH_SIZE")
        .unwrap()
        .parse()
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let model = deterministic_model(Arc::clone(&calls));
    let mut session =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, batch).unwrap();
    session.forward_rows(&fixture_rows(2)).unwrap();
    if std::env::var_os("RMI_TEST_TRACE_END_ONLY").is_some() {
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "unselected per-row logits must not be evaluated"
        );
    }
}

#[cfg(feature = "parity-trace")]
#[test]
fn gemma4_trace_rows_match_batch_one_raw_bits() {
    let mut baseline = None;
    for batch in [1, 64] {
        let trace = std::env::temp_dir().join(format!(
            "rmi-gemma4-chunk-trace-{}-{batch}.jsonl",
            std::process::id()
        ));
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "models::gemma4::trunk::tests::gemma4_trace_child",
            ])
            .env("RMI_PARITY_TRACE", &trace)
            .env("RMI_TEST_TRACE_BATCH_SIZE", batch.to_string())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let records = std::fs::read_to_string(&trace)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        let actual = records
            .iter()
            .map(|r| {
                (
                    r["name"].clone(),
                    r["layer"].clone(),
                    r["shape"].clone(),
                    std::fs::read(r["binary_path"].as_str().unwrap()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            records
                .iter()
                .filter(|r| r["name"] == "gemma4.logits")
                .count(),
            2
        );
        if let Some(expected) = &baseline {
            assert_eq!(&actual, expected);
        } else {
            baseline = Some(actual);
        }
        for record in records {
            std::fs::remove_file(record["binary_path"].as_str().unwrap()).unwrap();
        }
        std::fs::remove_file(trace).unwrap();
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "models::gemma4::trunk::tests::gemma4_trace_child",
        ])
        .env(
            "RMI_PARITY_TRACE",
            std::env::temp_dir().join("rmi-gemma4-filtered-trace.jsonl"),
        )
        .env("RMI_PARITY_FILTER", "gemma4.prompt_logits")
        .env("RMI_TEST_TRACE_END_ONLY", "1")
        .env("RMI_TEST_TRACE_BATCH_SIZE", "64")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn gemma4_only_projects_prompt_logits_for_last_row() {
    let calls = run_counting_gemma4_fixture(65, 64);
    assert_eq!(calls.output_projection_calls, 1);
}

#[test]
fn raw_rows_are_not_embedding_scaled_and_use_padding_layer_id() {
    let rows = assemble_input_rows(
        &[
            Gemma4InputRow::Token(7),
            Gemma4InputRow::Raw {
                values: vec![2.0; 1536],
                per_layer_token: 0,
            },
        ],
        TEST_EMBD,
    )
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
fn layer_12_attention_uses_stable_scalar_softmax() {
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
        row_width: 1,
        group_size: 8,
        keys: keys.to_vec(),
        values: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].to_vec(),
    };
    let mut output = [0.0; 8];

    attend(
        12,
        7,
        &[1.0; 8],
        &cache,
        true,
        512,
        &mut output,
        &mut Vec::new(),
        &mut Vec::new(),
        &ComputePool::new(1),
    )
    .unwrap();

    assert_eq!(output.map(f32::to_bits), [0x3f15_89fd; 8]);
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
                kernel: Box::new(crate::ops::kernel::f32::F32Kernel::new(vec![0.0], 0, 0)),
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
        kernel: Box::new(crate::ops::kernel::f32::F32Kernel::new(
            weights.to_vec(),
            0,
            0,
        )),
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
        kernel: Box::new(crate::ops::kernel::f32::F32Kernel::new(weights, 0, 0)),
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
    assert!(assemble_input_rows(&[], TEST_EMBD)
        .unwrap_err()
        .contains("empty"));
    assert!(
        assemble_input_rows(&[Gemma4InputRow::Token(262_144)], TEST_EMBD)
            .unwrap_err()
            .contains("token")
    );
    assert!(assemble_input_rows(
        &[Gemma4InputRow::Raw {
            values: vec![0.0; 1535],
            per_layer_token: 0,
        }],
        TEST_EMBD
    )
    .unwrap_err()
    .contains("1536"));
    assert!(assemble_input_rows(
        &[Gemma4InputRow::Raw {
            values: {
                let mut values = vec![0.0; 1536];
                values[7] = f32::NAN;
                values
            },
            per_layer_token: 0,
        }],
        TEST_EMBD
    )
    .unwrap_err()
    .contains("non-finite"));
}

#[test]
fn gemma4_executes_rows_ending_at_capacity_and_rejects_the_next_row() {
    let mut cfg = deterministic_config();
    cfg.full_head_dim = 32;
    cfg.swa_head_dim = 32;
    let model = deterministic_model_with_config(cfg, Arc::new(AtomicUsize::new(0)));
    let mut session =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, 64).unwrap();
    // Seed a valid zero-valued prefix so the real attention/KV append runs at
    // the actual 131072-token boundary without a quadratic full-prefix test.
    session.seq_len = CONTEXT - 2;
    for kv in &mut session.kv {
        kv.keys.resize(session.seq_len * kv.row_width, 0.0);
        kv.values.resize(session.seq_len * kv.row_width, 0.0);
    }
    let logits = session
        .forward_rows(&[Gemma4InputRow::Token(1), Gemma4InputRow::Token(2)])
        .unwrap();
    assert_eq!(session.len(), CONTEXT);
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|value| value.is_finite()));
    for kv in &session.kv {
        assert_eq!(kv.keys.len(), CONTEXT * kv.row_width);
        assert_eq!(kv.values.len(), CONTEXT * kv.row_width);
    }
    let before = snapshot_gemma4_state(&session);
    assert!(session
        .forward_rows(&[Gemma4InputRow::Token(3)])
        .unwrap_err()
        .contains("exceeds context"));
    assert_eq!(snapshot_gemma4_state(&session), before);
}

#[test]
fn gemma4_rejects_over_capacity_before_validating_rows() {
    let output_projection_calls = Arc::new(AtomicUsize::new(0));
    let model = deterministic_model(output_projection_calls);
    let mut session =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, 1).unwrap();
    let mut rows = vec![Gemma4InputRow::Token(1); CONTEXT + 1];
    rows[0] = Gemma4InputRow::Raw {
        values: Vec::new(),
        per_layer_token: 0,
    };

    let error = session.forward_rows(&rows).unwrap_err();

    assert!(error.contains("exceeds context"), "{error}");
}

#[test]
fn gemma4_prevalidates_later_chunks_before_running_any_projection() {
    let projection_calls = Arc::new(AtomicUsize::new(0));
    let mut model = deterministic_model(Arc::new(AtomicUsize::new(0)));
    let dim = model.config.head_dim(0);
    model.layers[0].attn_q = counting_output_weight(
        model.config.embd,
        model.config.n_heads * dim,
        Arc::clone(&projection_calls),
    );
    let mut session =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, 1).unwrap();
    let before = snapshot_gemma4_state(&session);
    let rows = [
        Gemma4InputRow::Token(1),
        Gemma4InputRow::Token(2),
        Gemma4InputRow::Raw {
            values: vec![0.0; model.config.embd - 1],
            per_layer_token: 0,
        },
    ];

    let error = session.forward_rows(&rows).unwrap_err();

    assert!(error.contains("raw row 2"), "{error}");
    assert_eq!(projection_calls.load(Ordering::Relaxed), 0);
    assert_eq!(snapshot_gemma4_state(&session), before);
}

#[test]
fn gemma4_scratch_bytes_counts_retained_attention_capacity() {
    let model = deterministic_model(Arc::new(AtomicUsize::new(0)));
    let mut session =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, 1).unwrap();
    let before_bytes = session.scratch_bytes();
    let before_capacity =
        session.scratch.scores.capacity() + session.scratch.attention_values.capacity();
    session.scratch.scores.reserve(1);
    session.scratch.attention_values.reserve(1);
    let retained_values = session.scratch.scores.capacity()
        + session.scratch.attention_values.capacity()
        - before_capacity;

    assert!(retained_values > 0);
    assert_eq!(
        session.scratch_bytes() - before_bytes,
        retained_values * std::mem::size_of::<f32>()
    );
}

#[test]
fn shared_kv_layers_map_by_attention_kind() {
    let cfg = test_config();
    assert_eq!(kv_source_layer(&cfg, 0), 0);
    assert_eq!(kv_source_layer(&cfg, 14), 14);
    assert_eq!(kv_source_layer(&cfg, 15), 13);
    assert_eq!(kv_source_layer(&cfg, 19), 14);
    assert_eq!(kv_source_layer(&cfg, 34), 14);
}

#[test]
fn incremental_session_is_f32_only() {
    assert!(require_f32_kv(KvFormat::F32).is_ok());
    assert!(require_f32_kv(KvFormat::F16).unwrap_err().contains("F32"));
}

#[test]
fn failed_gemma4_chunk_truncates_every_base_kv_layer() {
    let model = post_kv_failure_model();
    let mut session =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, 4).unwrap();
    let before = snapshot_gemma4_state(&session);
    let error = session.forward_rows(&fixture_rows(3)).unwrap_err();
    assert!(error.contains("blk.0.attn_output.weight"), "{error}");
    assert_eq!(snapshot_gemma4_state(&session), before);
}

#[test]
fn failed_later_gemma4_chunk_preserves_successful_prefix() {
    let expected = run_gemma4_fixture(2, 2);
    let mut model = deterministic_model(Arc::new(AtomicUsize::new(0)));
    model.layers[0].attn_output.kernel = Box::new(DeterministicKernel {
        seed: 4,
        output_projection_calls: Some(Arc::new(AtomicUsize::new(0))),
        fail_after: Some(2),
    });
    let mut session =
        super::Gemma4Session::new_with_prefill_batch_size(&model, KvFormat::F32, 2).unwrap();
    #[cfg(feature = "vulkan")]
    let dispatcher = Arc::new(std::sync::Mutex::new(LinearDispatcher {
        fail_at: Some(27),
        ..Default::default()
    }));
    #[cfg(feature = "vulkan")]
    {
        session.prefill_linear.dispatcher = Some(Arc::clone(&dispatcher));
    }
    let error = session.forward_rows(&fixture_rows(5)).unwrap_err();
    assert!(error.contains("blk.0.attn_output.weight"), "{error}");
    assert_eq!(session.len(), 2);
    assert_eq!(snapshot_base_kv(&session), expected.base_kv);
    #[cfg(feature = "vulkan")]
    {
        assert_eq!(dispatcher.lock().unwrap().calls.len(), 27);
        assert_eq!(dispatcher.lock().unwrap().failures, 1);
        assert!(!session.prefill_linear.active());
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
