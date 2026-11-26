//! `hy_v3` text-only entry (`HyV3TextRuntime`) contracts for Hy-MT3-30B-A3B.
//!
//! `hy_v3` is shared by two products: Hy-SimulMT (bundles the `hyv3_speech`
//! audio tower) and Hy-MT3-30B-A3B (plain text). `general.architecture` alone
//! cannot separate them, so `clip.projector_type` is the discriminator
//! (`src/models/hy_simulmt/text.rs::has_speech_tower`). These tests pin that
//! discriminator, the rejection messages of the text adapter, and the
//! `positions_needed` / KV-reset contracts.
//!
//! Every test here runs without weights. The synthetic model in
//! [`tiny_hy_v3`] is a complete but minimal `hy_v3` decoder (2 layers, one
//! dense and one MoE, 8-wide embeddings) so the runtime can be built for real
//! instead of being stubbed. Only the tensor *values* are synthetic; the
//! tensor names, shapes and types are exactly what `validate_decoder`
//! requires, which is what makes the reset test meaningful.
use rust_model_inference::app::text::RuntimeOptions;
use rust_model_inference::compute::ComputePolicy;
use rust_model_inference::core::scratchpad::KvFormat;
use rust_model_inference::core::tensor::{
    GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource,
};
use rust_model_inference::core::tokenizer::BPETokenizer;
use rust_model_inference::models::hy_simulmt::{has_speech_tower, HyConfig, HyV3TextRuntime};
use rust_model_inference::ops::generation_runtime::{
    Flow, GenerationRequest, TextRuntime, TokenSink,
};
use rust_model_inference::ComputePool;
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Metadata-only source. `tensor_info` returns `None` for everything, which is
/// enough for every rejection path in `HyV3TextRuntime::new` because those run
/// *before* `HyDecoder::from_source` touches a single tensor.
struct MetaOnly(HashMap<String, MetaValue>);

impl TensorSource for MetaOnly {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.0.get(key)
    }
    fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
        None
    }
    fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
        None
    }
}

/// Metadata **plus** real (synthetic) F32 tensor bytes.
struct Synthetic {
    meta: HashMap<String, MetaValue>,
    tensors: HashMap<String, (TensorInfo, Vec<u8>)>,
}

impl Synthetic {
    fn new(meta: HashMap<String, MetaValue>) -> Self {
        Self {
            meta,
            tensors: HashMap::new(),
        }
    }

    /// Register a tensor of `dims` filled with a deterministic, finite ramp.
    /// Norm weights get 1.0 so RMSNorm stays well-conditioned; the router bias
    /// gets 0.0 so expert selection is driven by the router alone.
    fn push(&mut self, name: &str, dims: &[u64]) {
        let count: u64 = dims.iter().product();
        let is_norm = name.ends_with("_norm.weight");
        let is_bias = name.ends_with("exp_probs_b");
        let data: Vec<f32> = (0..count)
            .map(|i| {
                if is_norm {
                    1.0
                } else if is_bias {
                    0.0
                } else {
                    // Small deterministic values; keeps logits finite and
                    // makes the argmax depend on position and content.
                    ((i % 17) as f32 - 8.0) * 0.01
                }
            })
            .collect();
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.tensors.insert(
            name.to_string(),
            (
                TensorInfo {
                    name: name.to_string(),
                    dims: dims.to_vec(),
                    ggml_type: GGMLType::F32,
                    offset: 0,
                },
                bytes,
            ),
        );
    }
}

impl TensorSource for Synthetic {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.meta.get(key)
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name).map(|(info, _)| info)
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|(_, bytes)| bytes.as_slice())
    }
}

fn uint(meta: &mut HashMap<String, MetaValue>, key: &str, value: u32) {
    meta.insert(key.to_string(), MetaValue::Uint32(value));
}

/// The real Hy-MT3-30B-A3B header (arch `hy_v3`, no audio tower).
fn hy_mt3_30b_meta() -> HashMap<String, MetaValue> {
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        MetaValue::String("hy_v3".into()),
    );
    for (key, value) in [
        ("embedding_length", 2048),
        ("block_count", 48),
        ("attention.head_count", 32),
        ("attention.head_count_kv", 4),
        ("attention.key_length", 128),
        ("attention.value_length", 128),
        ("feed_forward_length", 6912),
        ("context_length", 262144),
        ("vocab_size", 120832),
        ("expert_count", 128),
        ("expert_used_count", 8),
        ("expert_feed_forward_length", 768),
        ("expert_shared_feed_forward_length", 768),
        ("expert_gating_func", 2),
    ] {
        uint(&mut meta, &format!("hy_v3.{key}"), value);
    }
    meta.insert("hy_v3.expert_weights_norm".into(), MetaValue::Bool(true));
    meta.insert(
        "hy_v3.expert_weights_scale".into(),
        MetaValue::Float32(2.826),
    );
    meta.insert(
        "hy_v3.rope.freq_base".into(),
        MetaValue::Float32(11_158_840.0),
    );
    meta.insert(
        "hy_v3.attention.layer_norm_rms_epsilon".into(),
        MetaValue::Float32(1e-5),
    );
    meta
}

/// The real Hy-MT3-7B header (arch `hunyuan-dense`, `pre = "hunyuan"`).
fn hy_mt3_7b_meta() -> HashMap<String, MetaValue> {
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        MetaValue::String("hunyuan-dense".into()),
    );
    for (key, value) in [
        ("embedding_length", 4096),
        ("block_count", 32),
        ("attention.head_count", 32),
        ("attention.head_count_kv", 8),
        ("feed_forward_length", 14336),
        ("context_length", 262144),
        ("vocab_size", 128167),
    ] {
        uint(&mut meta, &format!("hunyuan-dense.{key}"), value);
    }
    meta.insert(
        "hunyuan-dense.rope.freq_base".into(),
        MetaValue::Float32(11_158_840.0),
    );
    meta.insert(
        "hunyuan-dense.attention.layer_norm_rms_epsilon".into(),
        MetaValue::Float32(1e-5),
    );
    meta.insert(
        "tokenizer.ggml.pre".into(),
        MetaValue::String("hunyuan".into()),
    );
    meta
}

/// 256 byte-level BPE base tokens plus `vocab - 256` control tokens, so
/// `vocab_size()` matches the synthetic decoder's `vocab` exactly (the loader
/// rejects a mismatch at `weights.rs`).
fn tiny_tokenizer(vocab: usize) -> Arc<BPETokenizer> {
    assert!(vocab >= 256, "byte-level BPE needs 256 base tokens");
    let mut extra = 256u32;
    let mut tokens: Vec<MetaValue> = (0..=255u32)
        .map(|byte| {
            // Same byte -> printable-codepoint mapping the Hy runtime tests use.
            let code = if matches!(byte, 33..=126 | 161..=172 | 174..=255) {
                byte
            } else {
                extra += 1;
                extra - 1
            };
            MetaValue::String(char::from_u32(code).unwrap().to_string())
        })
        .collect();
    let mut types: Vec<MetaValue> = (0..256).map(|_| MetaValue::Int32(1)).collect();
    for index in 0..(vocab - 256) {
        tokens.push(MetaValue::String(format!("<ctl{index}>")));
        types.push(MetaValue::Int32(3));
    }
    let meta = HashMap::from([
        (
            "tokenizer.ggml.model".to_string(),
            MetaValue::String("gpt2".into()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            MetaValue::String("hunyuan-dense".into()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            MetaValue::Array(MetaValueType::String, tokens),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            MetaValue::Array(MetaValueType::Int32, types),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            MetaValue::Array(MetaValueType::String, Vec::new()),
        ),
    ]);
    Arc::new(
        BPETokenizer::from_gguf_metadata(|key| meta.get(key).cloned())
            .expect("synthetic tokenizer"),
    )
}

const TINY_VOCAB: usize = 264;
const TINY_CTX: usize = 64;

/// A complete but minimal `hy_v3` decoder: 2 layers, layer 0 dense and layer 1
/// MoE, so both `HyFfn` arms and the dense-replace probe are exercised.
fn tiny_hy_v3() -> Synthetic {
    let (h, hd, nq, nkv, n_ff) = (8u64, 4u64, 8u64, 4u64, 16u64);
    let (experts, expert_ff, shared_ff) = (2u64, 4u64, 4u64);

    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        MetaValue::String("hy_v3".into()),
    );
    for (key, value) in [
        ("embedding_length", h as u32),
        ("block_count", 2),
        ("attention.head_count", 2),
        ("attention.head_count_kv", 1),
        ("attention.key_length", hd as u32),
        ("attention.value_length", hd as u32),
        ("feed_forward_length", n_ff as u32),
        ("context_length", TINY_CTX as u32),
        ("vocab_size", TINY_VOCAB as u32),
        ("expert_count", experts as u32),
        ("expert_used_count", 1),
        ("expert_feed_forward_length", expert_ff as u32),
        ("expert_shared_feed_forward_length", shared_ff as u32),
        ("expert_gating_func", 2),
    ] {
        uint(&mut meta, &format!("hy_v3.{key}"), value);
    }
    meta.insert("hy_v3.expert_weights_norm".into(), MetaValue::Bool(true));
    meta.insert(
        "hy_v3.expert_weights_scale".into(),
        MetaValue::Float32(2.826),
    );
    meta.insert(
        "hy_v3.rope.freq_base".into(),
        MetaValue::Float32(11_158_840.0),
    );
    meta.insert(
        "hy_v3.attention.layer_norm_rms_epsilon".into(),
        MetaValue::Float32(1e-5),
    );

    let mut source = Synthetic::new(meta);
    source.push("token_embd.weight", &[h, TINY_VOCAB as u64]);
    source.push("output.weight", &[h, TINY_VOCAB as u64]);
    source.push("output_norm.weight", &[h]);
    for layer in 0..2u64 {
        let p = |suffix: &str| format!("blk.{layer}.{suffix}");
        source.push(&p("attn_norm.weight"), &[h]);
        source.push(&p("ffn_norm.weight"), &[h]);
        source.push(&p("attn_q_norm.weight"), &[hd]);
        source.push(&p("attn_k_norm.weight"), &[hd]);
        source.push(&p("attn_q.weight"), &[h, nq]);
        source.push(&p("attn_k.weight"), &[h, nkv]);
        source.push(&p("attn_v.weight"), &[h, nkv]);
        source.push(&p("attn_output.weight"), &[nq, h]);
        if layer == 0 {
            // Dense leading layer: no `ffn_gate_inp`, so the loader's
            // per-layer probe must choose `HyFfn::Dense`.
            source.push(&p("ffn_gate.weight"), &[h, n_ff]);
            source.push(&p("ffn_up.weight"), &[h, n_ff]);
            source.push(&p("ffn_down.weight"), &[n_ff, h]);
        } else {
            source.push(&p("ffn_gate_inp.weight"), &[h, experts]);
            source.push(&p("exp_probs_b"), &[experts]);
            source.push(&p("ffn_gate_exps.weight"), &[h, expert_ff, experts]);
            source.push(&p("ffn_up_exps.weight"), &[h, expert_ff, experts]);
            source.push(&p("ffn_down_exps.weight"), &[expert_ff, h, experts]);
            source.push(&p("ffn_gate_shexp.weight"), &[h, shared_ff]);
            source.push(&p("ffn_up_shexp.weight"), &[h, shared_ff]);
            source.push(&p("ffn_down_shexp.weight"), &[shared_ff, h]);
        }
    }
    source
}

fn options(source: Arc<dyn TensorSource>, mmproj: Option<Arc<dyn TensorSource>>) -> RuntimeOptions {
    RuntimeOptions {
        compute: ComputePolicy::Cpu,
        threads: 1,
        kv_format: KvFormat::F32,
        max_context: TINY_CTX,
        prefill_batch_size: 8,
        source,
        pool: Arc::new(ComputePool::new(1)),
        tokenizer: tiny_tokenizer(TINY_VOCAB),
        mmproj,
        mtp_draft: 0,
        // No MTP/NextN tensors in these GGUFs and no draft sidecar: the plain
        // decode is the only mode this model has.
        dspark_model: None,
        ngram_draft: 0,
        ngram_lookup: rust_model_inference::core::ngram::NGRAM_SIMPLE_SIZE_N,
    }
}

/// The arm both front-ends actually use.
///
/// `build_text_runtime` is the single seam the CLI and the HTTP server share,
/// so a green test on `HyV3TextRuntime` alone would still pass if the registry
/// arm were dropped or if `ComputeTextRuntime` started rejecting `hy_v3`.
/// Driving the trait object is the closest a weightless test can get to a
/// real request.
#[test]
fn build_text_runtime_registers_hy_v3_for_both_front_ends() {
    let mut runtime = rust_model_inference::app::text::build_text_runtime(
        "hy_v3",
        options(Arc::new(tiny_hy_v3()), None),
    )
    .expect("hy_v3 must resolve to a TextRuntime adapter");
    assert_eq!(runtime.arch(), "hy_v3");
    assert_eq!(runtime.context_length(), TINY_CTX);
    let request = GenerationRequest::new(vec![7u32, 8, 9], 3);
    let mut sink = Sink(Vec::new());
    let generated = runtime.generate(&request, &mut sink).expect("generate");
    assert_eq!(generated.token_ids.len(), 3);
}

/// `HyV3TextRuntime` deliberately does not implement `Debug` (it owns a
/// multi-GB weight layout), so `Result::unwrap_err` is unavailable. This helper
/// is the readable equivalent for the rejection tests.
fn reject(options: RuntimeOptions) -> String {
    match HyV3TextRuntime::new(options) {
        Ok(_) => panic!("expected the text runtime to reject this checkpoint"),
        Err(error) => error,
    }
}

/// Collects streamed text; never asks to stop.
struct Sink(Vec<String>);

impl TokenSink for Sink {
    fn push_text(&mut self, chunk: &str) -> Flow {
        self.0.push(chunk.to_string());
        Flow::Continue
    }
}

// ---------------------------------------------------------------------------
// 1. The two-product discriminator
// ---------------------------------------------------------------------------

#[test]
fn speech_tower_probe_separates_the_two_products() {
    let text_only = MetaOnly(hy_mt3_30b_meta());
    assert!(
        !has_speech_tower(&text_only),
        "Hy-MT3-30B-A3B has no audio tower and must not be treated as Hy-SimulMT"
    );

    let mut with_tower = hy_mt3_30b_meta();
    with_tower.insert(
        "clip.projector_type".into(),
        MetaValue::String("hyv3_speech".into()),
    );
    assert!(
        has_speech_tower(&MetaOnly(with_tower)),
        "clip.projector_type=hyv3_speech is the only discriminator between the two hy_v3 products"
    );

    // Any other projector type must not be mistaken for the speech tower.
    let mut other = hy_mt3_30b_meta();
    other.insert(
        "clip.projector_type".into(),
        MetaValue::String("some_other_projector".into()),
    );
    assert!(!has_speech_tower(&MetaOnly(other)));
}

// ---------------------------------------------------------------------------
// 2. The text adapter's three rejections, with distinguishable messages
// ---------------------------------------------------------------------------

#[test]
fn text_runtime_rejects_wrong_arch() {
    let mut meta = hy_mt3_7b_meta();
    meta.remove("general.architecture");
    meta.insert(
        "general.architecture".into(),
        MetaValue::String("hunyuan-dense".into()),
    );
    // Rejection happens before any tensor is read, so a metadata-only source
    // is enough: `new()` checks arch, then the speech tower, then mmproj, and
    // only afterwards calls `HyDecoder::from_source`.
    let error = reject(options(Arc::new(MetaOnly(meta)), None));
    assert!(
        error.contains("requires general.architecture=hy_v3"),
        "{error}"
    );
    assert!(error.contains("hunyuan-dense"), "{error}");
}

#[test]
fn text_runtime_rejects_a_speech_checkpoint() {
    let mut meta = hy_mt3_30b_meta();
    meta.insert(
        "clip.projector_type".into(),
        MetaValue::String("hyv3_speech".into()),
    );
    let error = reject(options(Arc::new(MetaOnly(meta)), None));
    assert!(error.contains("hyv3_speech"), "{error}");
    assert!(error.contains("--audio"), "{error}");
}

#[test]
fn text_runtime_rejects_mmproj() {
    let mmproj: Arc<dyn TensorSource> = Arc::new(MetaOnly(HashMap::new()));
    let error = reject(options(Arc::new(MetaOnly(hy_mt3_30b_meta())), Some(mmproj)));
    assert!(error.contains("no vision path"), "{error}");
    assert!(error.contains("--mmproj"), "{error}");
}

#[test]
fn the_three_rejections_are_distinguishable() {
    let wrong_arch = {
        let mut meta = hy_mt3_30b_meta();
        meta.insert(
            "general.architecture".into(),
            MetaValue::String("llama".into()),
        );
        reject(options(Arc::new(MetaOnly(meta)), None))
    };
    let tower = {
        let mut meta = hy_mt3_30b_meta();
        meta.insert(
            "clip.projector_type".into(),
            MetaValue::String("hyv3_speech".into()),
        );
        reject(options(Arc::new(MetaOnly(meta)), None))
    };
    let vision = {
        let mmproj: Arc<dyn TensorSource> = Arc::new(MetaOnly(HashMap::new()));
        reject(options(Arc::new(MetaOnly(hy_mt3_30b_meta())), Some(mmproj)))
    };
    assert_ne!(wrong_arch, tower);
    assert_ne!(tower, vision);
    assert_ne!(wrong_arch, vision);
}

// ---------------------------------------------------------------------------
// 3. `HyConfig` owns `hy_v3` only; the dense siblings stay on `hunyuan-dense`
// ---------------------------------------------------------------------------

#[test]
fn hy_config_accepts_the_real_30b_header() {
    let config = HyConfig::from_source(&MetaOnly(hy_mt3_30b_meta())).expect("30B header must load");
    assert_eq!(
        (
            config.n_embd,
            config.n_layer,
            config.n_head,
            config.n_head_kv,
            config.n_embd_head_k,
            config.n_embd_head_v,
            config.n_ff,
            config.vocab,
            config.n_ctx,
        ),
        (2048, 48, 32, 4, 128, 128, 6912, 120832, 262144)
    );
    assert_eq!(config.freq_base, 11_158_840.0);
    assert!((config.eps - 1e-5).abs() < 1e-12);
    assert_eq!(config.moe.expert_count, 128);
    assert_eq!(config.moe.expert_used_count, 8);
    assert_eq!(config.moe.expert_ffn, 768);
    assert_eq!(config.moe.shared_expert_ffn, 768);
    assert!(config.moe.normalize_weights);
    assert_eq!(config.moe.scale, 2.826);
}

#[test]
fn hy_config_rejects_the_dense_siblings() {
    // The 7B (and 1.8B) are `hunyuan-dense` and belong to the Qwen3 trunk via
    // `app::text::run_hunyuan_inference`; `HyConfig` must not claim them.
    for meta in [hy_mt3_7b_meta()] {
        let error = HyConfig::from_source(&MetaOnly(meta)).unwrap_err();
        assert!(
            error.contains("requires general.architecture=hy_v3"),
            "{error}"
        );
    }
}

#[test]
fn dense_sibling_metadata_still_loads_through_the_shared_loader() {
    // Guards the other direction: the `hy_v3` work must not have disturbed the
    // generic loader's `hunyuan-dense` contract. Head dim is derived as
    // n_embd / n_head (4096 / 32), and rope.freq_base must be honoured rather
    // than defaulted.
    let config =
        rust_model_inference::core::loader::model_config_from_source(&MetaOnly(hy_mt3_7b_meta()))
            .expect("7B header must load");
    assert_eq!(
        (
            config.n_embd,
            config.n_layer,
            config.n_head,
            config.n_head_kv,
            config.n_embd_head,
            config.n_ff,
            config.vocab_size,
        ),
        (4096, 32, 32, 8, 128, 14336, 128167)
    );
    assert_eq!(config.rope_freq_base, 11_158_840.0);
}

// ---------------------------------------------------------------------------
// 4. `positions_needed = prompt + max_new - 1`
// ---------------------------------------------------------------------------

/// `positions_needed` is private to the adapter, so the contract is driven
/// through the public `TextRuntime::validate_request` on a real (synthetic)
/// runtime. The last generated token needs no forward pass, which is why the
/// bound is `len + max_new - 1` rather than `len + max_new`.
#[test]
fn positions_needed_boundary_is_prompt_plus_max_new_minus_one() {
    let runtime = HyV3TextRuntime::new(options(Arc::new(tiny_hy_v3()), None))
        .expect("synthetic hy_v3 runtime must build");
    assert_eq!(runtime.context_length(), TINY_CTX);

    let prompt = |len: usize| vec![0u32; len];

    // Exactly at capacity: 60 + 5 - 1 == 64. Must be accepted.
    runtime
        .validate_request(&prompt(60), &[], 5)
        .expect("60 + 5 - 1 == 64 must fit");

    // One more generated token: 60 + 6 - 1 == 65. Must be rejected.
    let error = runtime.validate_request(&prompt(60), &[], 6).unwrap_err();
    assert!(error.contains("65"), "{error}");
    assert!(error.contains("64"), "{error}");

    // One more prompt token, same max_new: 61 + 5 - 1 == 65. Also rejected.
    let error = runtime.validate_request(&prompt(61), &[], 5).unwrap_err();
    assert!(error.contains("65"), "{error}");

    // A prompt that exactly fills the cache with a single new token fits:
    // 64 + 1 - 1 == 64.
    runtime
        .validate_request(&prompt(64), &[], 1)
        .expect("64 + 1 - 1 == 64 must fit");

    // Empty prompt and images are separate, explicit failures.
    assert!(runtime.validate_request(&[], &[], 1).is_err());
    assert!(runtime
        .validate_request(&prompt(4), &[vec![0u8; 4]], 1)
        .is_err());
}

// ---------------------------------------------------------------------------
// 5. The cached session is reset between requests
// ---------------------------------------------------------------------------

/// The runtime caches its `HySession` so the prepared weight layout is reused,
/// but the KV position is per-request state. Before `HySession::reset()` the
/// second call started at the first call's `seq_len`, attended over the first
/// call's keys, and eventually failed with "exceeds KV context capacity".
///
/// `60 + 5 - 1 == 64 == capacity`, so the first request consumes the whole
/// cache. Without the reset the second prefill would need 60 more positions
/// and fail, which makes this test sensitive to the bug without depending on
/// the synthetic weights producing any particular token.
#[test]
fn consecutive_requests_reset_the_kv_position() {
    let mut runtime = HyV3TextRuntime::new(options(Arc::new(tiny_hy_v3()), None))
        .expect("synthetic hy_v3 runtime must build");

    let mut request = GenerationRequest::new(vec![7u32; 60], 5);
    // Greedy: `LlamaSampler` short-circuits to argmax at temperature <= 0, so
    // the two runs are comparable. (The adapter takes top_k/top_p from
    // `general.sampling.*`, not from `request.sampling`.)
    request.sampling.temperature = 0.0;

    let mut first_sink = Sink(Vec::new());
    let first = runtime
        .generate(&request, &mut first_sink)
        .expect("first request must fit the KV cache");
    assert_eq!(first.token_ids.len(), 5, "expected max_new_tokens tokens");

    let mut second_sink = Sink(Vec::new());
    let second = runtime
        .generate(&request, &mut second_sink)
        .expect("second request must start from position 0 after the reset");
    assert_eq!(second.token_ids.len(), 5);

    assert_eq!(
        first.token_ids, second.token_ids,
        "the same request on the same runtime must be reproducible; a drifting KV position changes the logits"
    );
    assert_eq!(first.text, second.text);

    // A third call proves the position does not accumulate across requests
    // either (a leak would only show up after several).
    let mut third_sink = Sink(Vec::new());
    let third = runtime
        .generate(&request, &mut third_sink)
        .expect("third request must also start from position 0");
    assert_eq!(third.token_ids, first.token_ids);
}

/// The same reset contract on a request that does **not** fill the cache, so a
/// passing run cannot be explained by the capacity check alone.
#[test]
fn consecutive_requests_reset_with_spare_capacity() {
    let mut runtime = HyV3TextRuntime::new(options(Arc::new(tiny_hy_v3()), None))
        .expect("synthetic hy_v3 runtime must build");

    let mut request = GenerationRequest::new(vec![3u32; 8], 3);
    request.sampling.temperature = 0.0;

    let mut runs = Vec::new();
    for round in 0..4 {
        let mut sink = Sink(Vec::new());
        let generated = runtime
            .generate(&request, &mut sink)
            .unwrap_or_else(|error| panic!("round {round} failed: {error}"));
        runs.push(generated.token_ids);
    }
    assert!(
        runs.windows(2).all(|pair| pair[0] == pair[1]),
        "every round must produce the same tokens, got {runs:?}"
    );
}

/// Why the two tests above are meaningful: `HySession` has no implicit reset,
/// so a session that is *not* rewound genuinely cannot serve a second request.
///
/// This pins the underlying behaviour the adapter's `reset()` call exists to
/// fix. If someone removes `session.reset()` from `generate()` and the cached
/// session keeps accumulating, `consecutive_requests_reset_the_kv_position`
/// fails here first — with a capacity error, not a silent wrong answer.
#[test]
fn a_session_that_is_not_rewound_cannot_serve_a_second_request() {
    use rust_model_inference::models::hy_simulmt::{HyDecoder, HySession};

    let source: Arc<dyn TensorSource> = Arc::new(tiny_hy_v3());
    let decoder = HyDecoder::from_source(
        source,
        tiny_tokenizer(TINY_VOCAB),
        Arc::new(ComputePool::new(1)),
    )
    .expect("synthetic decoder must build");
    let mut session = HySession::new(&decoder, TINY_CTX, KvFormat::F32, ComputePolicy::Cpu)
        .expect("synthetic session must build");

    // Consume the whole cache: 60 prefill positions + 4 decode steps == 64.
    session
        .forward(&vec![0u32; 60], None, 8)
        .expect("prefill must fit");
    for step in 0..4 {
        session
            .forward(&[1u32], None, 1)
            .unwrap_or_else(|error| panic!("decode step {step} failed: {error}"));
    }
    assert_eq!(session.position(), TINY_CTX);

    // Without a rewind the second prefill has nowhere to go.
    let error = session
        .forward(&vec![0u32; 60], None, 8)
        .expect_err("a full session must reject a second prefill");
    assert!(error.contains("capacity"), "{error}");

    // With the rewind it is reusable, which is what the adapter relies on.
    session.reset();
    assert_eq!(session.position(), 0);
    session
        .forward(&vec![0u32; 60], None, 8)
        .expect("reset must rewind the position so the session is reusable");
    assert_eq!(session.position(), 60);
}

/// The HTTP prompt path for `hy_v3`, driven through the same function the
/// server calls.
///
/// Without a compiled template `build_prompt` falls through to the
/// hand-written renderer, whose arch whitelist rejects `hy_v3` — that is why
/// the server now enables the model's own template for this arch (the GGUF
/// ships an 11 KB one whose first token is `<｜hy_begin▁of▁sentence｜>`).
/// This test pins both halves: the template path yields the BOS-prefixed
/// Hy-MT prompt, and the fallback is still an explicit error rather than a
/// silently different ChatML prompt.
///
/// Needs a tokenizer-only GGUF (metadata section, zero tensors, ~5 MB) so it
/// runs without the 33 GB checkpoint:
///   RMI_HY_MT3_TOKENIZER_GGUF=/path/to/30B-tokenizer.gguf \
///     cargo test --test hy_mt3_text -- --ignored hy_v3_http_prompt
#[test]
#[ignore = "needs RMI_HY_MT3_TOKENIZER_GGUF (tokenizer-only GGUF, no weights)"]
fn hy_v3_http_prompt_uses_the_models_own_template() {
    use rust_model_inference::app::server::api::protocol::{Message, ToolChoice};
    use rust_model_inference::app::server::api::tools::build_prompt;

    let path = std::env::var("RMI_HY_MT3_TOKENIZER_GGUF")
        .expect("set RMI_HY_MT3_TOKENIZER_GGUF to a tokenizer-only hy_v3 GGUF");
    let source: Arc<dyn TensorSource> = Arc::from(
        rust_model_inference::format::ggufrs::open_model_source(
            std::path::Path::new(&path),
            rust_model_inference::format::ggufrs::ComponentRole::Llm,
        )
        .expect("open tokenizer-only GGUF"),
    );
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .expect("tokenizer from GGUF metadata");
    let options = rust_model_inference::prompt::jinja::Options {
        jinja: true,
        file: None,
    };
    let compiled = options
        .resolve(&|k| source.metadata(k).cloned())
        .expect("the Hy-MT3 GGUF must ship a compilable chat template")
        .expect("jinja requested, so a template must resolve");
    let messages = vec![Message {
        role: "user".into(),
        text: "机器学习是人工智能的一个分支。".into(),
        calls: Vec::new(),
        call_id: None,
        images: Vec::new(),
    }];

    let (ids, images) = build_prompt(
        source.as_ref(),
        &tokenizer,
        "hy_v3",
        &messages,
        &[],
        &ToolChoice::None,
        None,
        &options,
        Some(&compiled),
    )
    .expect("hy_v3 must render through its own template, not the arch whitelist");
    assert!(images.is_empty());
    // The template opens with two newlines and a `<｜reasoning_mode｜>` preamble
    // before the BOS, so the contract is "BOS in the first few tokens", not
    // "BOS at index 0". What matters is that no control token precedes it: that
    // is the arrangement both llama.cpp and this repository degenerate on.
    let bos_at = ids.iter().take(4).position(|&id| id == 120000);
    assert!(
        bos_at.is_some(),
        "<｜hy_begin▁of▁sentence｜> must come first: without it a control token sits \
         at position 0 and the model degenerates; rendered={:?}",
        &ids[..ids.len().min(8)]
    );
    assert!(
        ids.contains(&120006),
        "rendered prompt must contain <｜hy_User｜>; rendered={:?}",
        &ids[..ids.len().min(12)]
    );

    // Negative control: no template -> the old explicit rejection, never a
    // silently different ChatML prompt.
    let fallback = build_prompt(
        source.as_ref(),
        &tokenizer,
        "hy_v3",
        &messages,
        &[],
        &ToolChoice::None,
        None,
        &rust_model_inference::prompt::jinja::Options::default(),
        None,
    );
    assert!(
        fallback.is_err(),
        "a template-less hy_v3 must fail loudly rather than render ChatML"
    );
}
