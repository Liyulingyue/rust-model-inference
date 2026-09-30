//! Loader for `gliner2.variant = "boundary"` GGUF files (BoundaryExtractor).
//!
//! Layout (see ``tools/converter/gliner/convert_boundary.py``):
//!
//! - encoder: same tensor names as Decide (``token_embd.weight``,
//!   ``blk.{i}.attn_*``, ``rel_embeddings.weight``, ``tok_norm``,
//!   ``rel_norm``). The base-v1 variant uses DeBERTa-v3-base dims
//!   (768 / 12 / 12 / 3072).
//! - classifier: ``classifier.0.weight`` / ``.bias`` (1536×768 for base-v1)
//!   and ``classifier.3.weight`` / ``.bias`` (1×1536). The GeLU between
//!   them is part of the activation choice (``gliner2.classifier.activation
//!   = "gelu"``); the Decide loader's hardcoded relu check doesn't apply.
//! - boundary_head.*: BOS / EOS states, left/right projections,
//!   output projection, layer norm, attention blocks, refinement blocks.
//! - relation_scorer.* / record_decoder.*: bundled but not loaded yet
//!   (Phase 7 in ``glinerTODO.md``).
//!
//! The encoder side of the GGUF is byte-compatible with Decide's
//! ``crate::models::gliner::weights::load_weights`` apart from the
//! classifier layer index (3 instead of 2). We re-implement the encoder
//! loading here so the classifier branch can diverge, and to avoid
//! pulling Decide's ``classifier.2.*`` requirement into the loader API.

use crate::core::tensor::{load_f32_tensor, GGMLType, MetaValue, TensorInfo, TensorSource};
use crate::models::gliner::compute::EncoderConfig;
use crate::models::gliner::weights::{LayerWeights, Norm};
use crate::ops::kernel::{QuantizedTensor, Weight};
use std::sync::Arc;

use super::forward::BoundaryEncoder;
use super::marginals::BoundaryQueryHead;
use super::pair_scorer::PairScorer;
use super::pool::{DocumentCandidatePool, SharedPoolScorer};
use super::proposer::BoundaryProposer;
use super::settings::BoundarySettings;

/// Loaded BoundaryExtractor weights + cached `EncoderConfig`.
pub struct BoundaryModel<'a> {
    pub config: EncoderConfig,
    pub tokenizer: crate::models::gliner::ModelTokenizer,
    pub encoder: crate::models::gliner::weights::ModelWeights<'a>,
    /// GeLU sits between layers 0 and 3 (the boundary classifier lives at
    /// indices 0 and 3 with dropout+GeLU at indices 1, 2). Loaded
    /// separately from ``encoder.classifier_0`` / ``_2`` because the
    /// existing ``ModelWeights`` only knows about index 2.
    pub classifier_0: Weight<'a>,
    pub classifier_0_bias: Vec<f32>,
    pub classifier_3: Weight<'a>,
    pub classifier_3_bias: Vec<f32>,
    /// `boundary_head.null_projection` — one scalar per extractive query; the
    /// reference drops a query's spans when its sigmoid clears
    /// `abstention_threshold` (`engine.py:269`).
    pub null_projection: Weight<'a>,
    pub null_projection_bias: Vec<f32>,
    /// `boundary_head.count_head` — per-query count log-rate, only consumed when
    /// `adaptive_threshold` is on.
    pub count_head: Weight<'a>,
    pub count_head_bias: Vec<f32>,
    pub boundary: BoundaryEncoder<'a>,
    pub query_head: BoundaryQueryHead<'a>,
    pub proposer: BoundaryProposer<'a>,
    pub pair_scorer: PairScorer<'a>,
    /// The transcoded `boundary_head` settings, the single source of truth for
    /// which optional feature sources exist.
    pub settings: BoundarySettings,
    /// The shared document span pool. This is the mainline when
    /// `candidate_pool = "shared"` (base-v1's setting); `proposer` /
    /// `pair_scorer` then only serve `score_explicit_spans`.
    pub pool_builder: DocumentCandidatePool<'a>,
    pub pool_scorer: SharedPoolScorer<'a>,
}

impl<'a> BoundaryModel<'a> {
    /// Load a boundary-variant GGUF. Validates the ``gliner2.variant =
    /// "boundary"`` flag and the DeBERTa-v3 base architecture before
    /// pulling the rest of the tensors.
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        // 1. Architecture + variant gate
        let arch = meta_str(source, "general.architecture")?;
        if arch != "gliner2" {
            return Err(format!("expected gliner2 architecture, got {arch:?}"));
        }
        let variant = meta_str(source, "gliner2.variant").unwrap_or_default();
        if variant != "boundary" {
            return Err(format!(
                "BoundaryModel requires gliner2.variant = \"boundary\", got {variant:?}"
            ));
        }

        // 2. Encoder dims
        let n_embd = meta_usize(source, "gliner2.embedding_length")?;
        let n_layer = meta_usize(source, "gliner2.block_count")?;
        let n_head = meta_usize(source, "gliner2.attention.head_count")?;
        let n_ff = meta_usize(source, "gliner2.feed_forward_length")?;
        let head_dim = meta_usize(source, "gliner2.attention.head_dim")?;
        let scale_divisor = meta_usize(source, "gliner2.attention.scale_divisor")?;
        if scale_divisor != 3 {
            return Err(format!(
                "gliner2.attention.scale_divisor must be 3 (p2c|c2p), got {scale_divisor}"
            ));
        }
        let bucket_size = meta_usize(source, "gliner2.relative_attention.bucket_size")?;
        let max_relative = meta_usize(source, "gliner2.relative_attention.max_relative_positions")?;
        let norm_rel_embeddings = meta_bool(source, "gliner2.norm_rel_embeddings")?;
        let vocab_size = meta_usize(source, "gliner2.vocab_size")?;
        let classifier_intermediate = meta_usize(source, "gliner2.classifier.intermediate_size")?;
        if classifier_intermediate != n_embd * 2 {
            return Err(format!(
                "classifier.intermediate_size {classifier_intermediate} != n_embd * 2 ({n_embd} * 2)"
            ));
        }

        let config = EncoderConfig {
            n_embd,
            n_layer,
            n_head,
            n_ff,
            head_dim,
            eps: meta_f32(source, "gliner2.attention.layer_norm_epsilon")?,
            att_span: bucket_size,
            pos_ebd_size: bucket_size * 2,
            bucket_size,
            max_relative_positions: max_relative,
            norm_rel_embeddings,
            vocab_size,
        };

        // 3. Tokenizer (re-use Decide's loader for the embedded SPM / fast vocab)
        let tokenizer = crate::models::gliner::load_tokenizer(source)?;

        // 4. Encoder weights (token_embd / rel_embeddings / rel_norm / tok_norm / blk.X.*)
        let encoder = load_encoder(source, n_layer, n_embd, n_head, n_ff, bucket_size * 2)?;

        // 5. Boundary classifier (layers 0 and 3)
        let classifier_0 = load_weight(
            source,
            "classifier.0.weight",
            n_embd,
            classifier_intermediate,
        )?;
        let classifier_0_bias = load_vec(source, "classifier.0.bias", classifier_intermediate)?;
        let classifier_3 = load_weight(source, "classifier.3.weight", classifier_intermediate, 1)?;
        let classifier_3_bias = load_vec(source, "classifier.3.bias", 1)?;
        let null_projection =
            load_weight(source, "boundary_head.null_projection.weight", n_embd, 1)?;
        let null_projection_bias = load_vec(source, "boundary_head.null_projection.bias", 1)?;
        let count_head = load_weight(source, "boundary_head.count_head.weight", n_embd, 1)?;
        let count_head_bias = load_vec(source, "boundary_head.count_head.bias", 1)?;

        // 6. Settings. Read before the heads, since they size themselves from
        //    it (`boundary_attention_window` in particular changes the encoder
        //    attention mask).
        let settings = BoundarySettings::from_source(source)?;

        // 7. BoundaryEncoder weights
        let boundary = BoundaryEncoder::load(source, n_embd, settings.boundary_attention_window)?;

        // 8. BoundaryQueryHead weights (per-query marginals over boundary positions)
        let query_head = BoundaryQueryHead::load(source, n_embd)?;

        // 9. BoundaryProposer weights (endpoint projections + rotary).
        let proposer = BoundaryProposer::load(source, n_embd)?;

        // 10. PairScorer weights (endpoint projections + query gate +
        //    compat_mix + length_query_projection + optional rotary,
        //    span content, inside weight, endpoint difference).
        let pair_scorer = PairScorer::load(source, n_embd, &settings)?;

        // 11. Shared document pool + scorer. `candidate_pool = "shared"` makes
        //     this the mainline; the pair scorer above then only serves
        //     `score_explicit_spans`.
        let pool_builder = DocumentCandidatePool::load(
            source,
            settings.boundary_dim,
            settings.pool_boundary_top_k,
            settings.pool_size,
            settings.min_pool_per_query,
        )?;
        let pool_scorer = SharedPoolScorer::load(source, n_embd, &settings)?;

        Ok(Self {
            config,
            tokenizer,
            encoder,
            classifier_0,
            classifier_0_bias,
            classifier_3,
            classifier_3_bias,
            null_projection,
            null_projection_bias,
            count_head,
            count_head_bias,
            boundary,
            query_head,
            proposer,
            pair_scorer,
            settings,
            pool_builder,
            pool_scorer,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn meta_str(source: &dyn TensorSource, name: &str) -> Result<String, String> {
    source
        .metadata(name)
        .and_then(MetaValue::to_string_val)
        .map(str::to_string)
        .ok_or_else(|| format!("missing or invalid metadata: {name}"))
}

fn meta_usize(source: &dyn TensorSource, name: &str) -> Result<usize, String> {
    source
        .metadata(name)
        .and_then(MetaValue::to_u64)
        .map(|value| value as usize)
        .ok_or_else(|| format!("missing or invalid metadata: {name}"))
}

fn meta_f32(source: &dyn TensorSource, name: &str) -> Result<f32, String> {
    source
        .metadata(name)
        .and_then(|value| value.to_f64())
        .map(|value| value as f32)
        .ok_or_else(|| format!("missing or invalid metadata: {name}"))
}

fn meta_bool(source: &dyn TensorSource, name: &str) -> Result<bool, String> {
    match source.metadata(name) {
        Some(MetaValue::Bool(value)) => Ok(*value),
        Some(MetaValue::Uint32(value)) => Ok(*value != 0),
        Some(MetaValue::Uint64(value)) => Ok(*value != 0),
        _ => Err(format!("missing or invalid metadata: {name}")),
    }
}

fn load_vec(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    load_f32_tensor(source, name, &[len as u64]).map_err(|e| format!("{name}: {e}"))
}

fn load_norm(source: &dyn TensorSource, name: &str, width: usize) -> Result<Norm, String> {
    Ok(Norm {
        weight: load_vec(source, &format!("{name}.weight"), width)?,
        bias: load_vec(source, &format!("{name}.bias"), width)?,
    })
}

fn load_weight<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] {
        return Err(format!(
            "tensor {name} has dims {:?}, expected [{n_in}, {n_out}]",
            info.dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("missing tensor data {name}"))?;
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in,
        n_out,
    )))
}

fn load_encoder<'a>(
    source: &'a dyn TensorSource,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    n_ff: usize,
    pos_ebd_size: usize,
) -> Result<crate::models::gliner::weights::ModelWeights<'a>, String> {
    let d = n_embd;
    let embd_info = source
        .tensor_info("token_embd.weight")
        .ok_or("missing token_embd.weight")?;
    if embd_info.dims[0] as usize != d {
        return Err(format!(
            "token_embd.weight ne0 is {}, expected {d}",
            embd_info.dims[0]
        ));
    }
    let token_embd = source
        .tensor_slice("token_embd.weight")
        .ok_or("missing token_embd.weight data")?;

    let rel_info = source
        .tensor_info("rel_embeddings.weight")
        .ok_or("missing rel_embeddings.weight")?;
    if rel_info.dims != [d as u64, pos_ebd_size as u64] {
        return Err(format!(
            "rel_embeddings.weight has dims {:?}, expected [{d}, {}]",
            rel_info.dims, pos_ebd_size
        ));
    }
    let rel_embeddings = source
        .tensor_slice("rel_embeddings.weight")
        .ok_or("missing rel_embeddings.weight data")?;

    let mut layers = Vec::with_capacity(n_layer);
    for index in 0..n_layer {
        let name = |suffix: &str| format!("blk.{index}.{suffix}");
        layers.push(LayerWeights {
            attn_q: load_weight(source, &name("attn_q.weight"), d, d)?,
            attn_q_bias: load_vec(source, &name("attn_q.bias"), d)?,
            attn_k: load_weight(source, &name("attn_k.weight"), d, d)?,
            attn_k_bias: load_vec(source, &name("attn_k.bias"), d)?,
            attn_v: load_weight(source, &name("attn_v.weight"), d, d)?,
            attn_v_bias: load_vec(source, &name("attn_v.bias"), d)?,
            attn_output: load_weight(source, &name("attn_output.weight"), d, d)?,
            attn_output_bias: load_vec(source, &name("attn_output.bias"), d)?,
            attn_out_norm: load_norm(source, &name("attn_out_norm"), d)?,
            ffn_up: load_weight(source, &name("ffn_up.weight"), d, n_ff)?,
            ffn_up_bias: load_vec(source, &name("ffn_up.bias"), n_ff)?,
            ffn_down: load_weight(source, &name("ffn_down.weight"), n_ff, d)?,
            ffn_down_bias: load_vec(source, &name("ffn_down.bias"), d)?,
            output_norm: load_norm(source, &name("output_norm"), d)?,
        });
    }

    // Stub classifier_0 / classifier_2 for the Decide `ModelWeights` shape.
    // We never use them — the real classifier lives at index 3 and is held
    // separately on `BoundaryModel`. The stub weights are dummy F32
    // tensors with shape (1, 1) so we never accidentally route through
    // Decide's classifier path.
    // Stub weights so the Decide-shaped ModelWeights can hold classifier_0 /
    // classifier_2 fields. We never read them — the real classifier lives
    // at index 3 in BoundaryModel. Build two separate stubs because
    // ``Weight`` is not ``Copy``.
    let stub_0 =
        Weight::from_quantized(QuantizedTensor::from_bytes(&[0u8; 4], GGMLType::F32, 1, 1));
    let stub_2 =
        Weight::from_quantized(QuantizedTensor::from_bytes(&[0u8; 4], GGMLType::F32, 1, 1));

    Ok(crate::models::gliner::weights::ModelWeights {
        token_embd,
        token_embd_type: embd_info.ggml_type,
        tok_norm: load_norm(source, "tok_norm", d)?,
        rel_embeddings,
        rel_embeddings_type: rel_info.ggml_type,
        rel_norm: load_norm(source, "rel_norm", d)?,
        layers,
        classifier_0: stub_0,
        classifier_0_bias: vec![0.0],
        classifier_2: stub_2,
        classifier_2_bias: vec![0.0],
    })
}
