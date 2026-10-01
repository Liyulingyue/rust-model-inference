//! `gliner2.boundary.*` GGUF metadata → a single typed settings struct.
//!
//! `tools/converter/gliner/convert_boundary.py` transcribes the checkpoint's
//! `config.json` `boundary_head` block into flat `gliner2.boundary.*` keys and
//! cross-checks the bundled tensor shapes against it. This module is the only
//! place that reads them, so the heads cannot disagree about which optional
//! feature sources exist.
//!
//! Every field here changes a score or a decoded label, so a missing or
//! mistyped key is a wrong-answer bug, not a tuning knob. `from_source` refuses
//! to fall back to defaults and names the re-conversion step instead.

use crate::core::tensor::{MetaValue, TensorSource};

const MISSING_HINT: &str = "missing gliner2.boundary.* settings metadata; re-convert the \
                           checkpoint with tools/converter/gliner/convert_boundary.py";

/// The full boundary-head configuration for one checkpoint.
#[derive(Clone, Debug)]
pub struct BoundarySettings {
    // --- dims ---
    pub boundary_dim: usize,
    pub pair_dim: usize,
    pub content_dim: usize,
    pub record_dim: usize,
    pub multihead_pair_compat_heads: usize,
    pub rotary_base: f32,

    // --- `SparseBoundaryPairScorer` feature sources ---
    pub use_inside_evidence: bool,
    pub enable_span_content: bool,
    pub content_soft_max_pool: bool,
    pub query_conditioned_inside_weight: bool,
    pub endpoint_difference_features: bool,
    pub enable_rotary_endpoints: bool,
    pub reranker_endpoint_compat: bool,

    // --- `DocumentCandidatePool` ---
    /// `"shared"` routes ordinary inference through the pool + `SharedPoolScorer`
    /// instead of the per-query proposer; anything else uses the historical
    /// `SparseBoundaryProposer` path.
    pub candidate_pool: String,
    pub pool_boundary_top_k: usize,
    pub pool_size: usize,
    pub min_pool_per_query: usize,

    // --- `BoundaryEncoder` attention ---
    /// Local attention band `|i - j| <= window` in every boundary attention
    /// block. 128 for base-v1, which only binds once a document has more than
    /// 257 boundary positions.
    pub boundary_attention_window: usize,
    pub boundary_attention_layers: usize,
    pub boundary_attention_heads: usize,

    // --- `SharedPoolScorer` optional attention ---
    pub candidate_attention_layers: usize,
    pub candidate_attention_heads: usize,
    pub query_attention_layers: usize,

    // --- decoding ---
    pub adaptive_threshold: bool,
    pub overlap_policy: String,

    // --- decoding ---
    pub pair_temperature: f32,
    pub classification_temperature: f32,
    pub abstention_threshold: f32,

    // --- `TypedRelationPairGenerator` + `SparseRelationScorer` ---
    /// Width of the relation query state: `2 * hidden_size` when the two role
    /// states are concatenated (`directional_relation_states`), else `hidden_size`.
    pub directional_relation_states: bool,
    /// The three content projections and the content linear are only present
    /// when this is set. It is a *shape* switch, not just a feature switch.
    pub relation_biaffine_content: bool,
    /// Per relation type, how many head-typed and tail-typed mentions survive
    /// the top-k, and how many of their cross product are scored.
    pub relation_heads_per_type: usize,
    pub relation_tails_per_type: usize,
    pub relation_pair_cap: usize,
    /// A mention only qualifies as a relation argument at or above this
    /// probability. base-v1 uses 0.2, not the reference default of 0.0.
    pub relation_argument_proposal_threshold: f32,
    pub relation_temperature: f32,

    // --- task enables ---
    pub enable_relations: bool,
    pub enable_records: bool,
    pub enable_count_head: bool,
    pub enable_abstention: bool,
}

impl BoundarySettings {
    /// Read the settings. Errors if the metadata block is absent so a GGUF
    /// produced by an older converter cannot be loaded as a down-graded model.
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        if source
            .metadata("gliner2.boundary.use_inside_evidence")
            .is_none()
        {
            return Err(MISSING_HINT.to_string());
        }
        Ok(Self {
            boundary_dim: required_usize(source, "boundary_dim")?,
            pair_dim: required_usize(source, "pair_dim")?,
            content_dim: required_usize(source, "content_dim")?,
            record_dim: required_usize(source, "record_dim")?,
            multihead_pair_compat_heads: required_usize(source, "multihead_pair_compat_heads")?,
            rotary_base: required_f32(source, "rotary_base")?,

            use_inside_evidence: required_flag(source, "use_inside_evidence")?,
            enable_span_content: required_flag(source, "enable_span_content")?,
            content_soft_max_pool: required_flag(source, "content_soft_max_pool")?,
            query_conditioned_inside_weight: required_flag(
                source,
                "query_conditioned_inside_weight",
            )?,
            endpoint_difference_features: required_flag(source, "endpoint_difference_features")?,
            enable_rotary_endpoints: required_flag(source, "enable_rotary_endpoints")?,
            reranker_endpoint_compat: required_flag(source, "reranker_endpoint_compat")?,

            candidate_pool: required_str(source, "candidate_pool")?,
            pool_boundary_top_k: required_usize(source, "pool_boundary_top_k")?,
            pool_size: required_usize(source, "pool_size")?,
            min_pool_per_query: required_usize(source, "min_pool_per_query")?,

            boundary_attention_window: required_usize(source, "boundary_attention_window")?,
            boundary_attention_layers: required_usize(source, "boundary_attention_layers")?,
            boundary_attention_heads: required_usize(source, "boundary_attention_heads")?,

            candidate_attention_layers: required_usize(source, "candidate_attention_layers")?,
            candidate_attention_heads: required_usize(source, "candidate_attention_heads")?,
            query_attention_layers: required_usize(source, "query_attention_layers")?,

            adaptive_threshold: required_flag(source, "adaptive_threshold")?,
            overlap_policy: required_str(source, "overlap_policy")?,
            pair_temperature: required_f32(source, "pair_temperature")?,
            classification_temperature: required_f32(source, "classification_temperature")?,
            abstention_threshold: required_f32(source, "abstention_threshold")?,

            directional_relation_states: required_flag(source, "directional_relation_states")?,
            relation_biaffine_content: required_flag(source, "relation_biaffine_content")?,
            relation_heads_per_type: required_usize(source, "relation_heads_per_type")?,
            relation_tails_per_type: required_usize(source, "relation_tails_per_type")?,
            relation_pair_cap: required_usize(source, "relation_pair_cap")?,
            relation_argument_proposal_threshold: required_f32(
                source,
                "relation_argument_proposal_threshold",
            )?,
            relation_temperature: required_f32(source, "relation_temperature")?,

            enable_relations: required_flag(source, "enable_relations")?,
            enable_records: required_flag(source, "enable_records")?,
            enable_count_head: required_flag(source, "enable_count_head")?,
            enable_abstention: required_flag(source, "enable_abstention")?,
        })
    }

    /// The relation query-state width, which the scorer's input layer is
    /// built against. `directional_relation_states` is the only setting that
    /// changes a *shape* rather than a computation.
    pub fn relation_query_dim(&self, hidden_size: usize) -> usize {
        if self.directional_relation_states {
            2 * hidden_size
        } else {
            hidden_size
        }
    }

    /// True when ordinary inference goes through the shared document pool
    /// rather than the per-query proposer.
    pub fn uses_shared_pool(&self) -> bool {
        self.candidate_pool == "shared"
    }
}

fn key(name: &str) -> String {
    format!("gliner2.boundary.{name}")
}

fn required_flag(source: &dyn TensorSource, name: &str) -> Result<bool, String> {
    match source.metadata(&key(name)) {
        Some(MetaValue::Bool(value)) => Ok(*value),
        Some(MetaValue::Uint32(value)) => Ok(*value != 0),
        Some(MetaValue::Int32(value)) => Ok(*value != 0),
        Some(MetaValue::Uint64(value)) => Ok(*value != 0),
        Some(MetaValue::Int64(value)) => Ok(*value != 0),
        Some(other) => Err(format!("{} is not a bool: {other:?}", key(name))),
        None => Err(format!("missing metadata {}", key(name))),
    }
}

fn required_f32(source: &dyn TensorSource, name: &str) -> Result<f32, String> {
    match source.metadata(&key(name)) {
        Some(value) => value
            .to_f64()
            .map(|v| v as f32)
            .ok_or_else(|| format!("{} is not a number: {value:?}", key(name))),
        None => Err(format!("missing metadata {}", key(name))),
    }
}

fn required_usize(source: &dyn TensorSource, name: &str) -> Result<usize, String> {
    required_f32(source, name).map(|v| v as usize)
}

fn required_str(source: &dyn TensorSource, name: &str) -> Result<String, String> {
    match source.metadata(&key(name)) {
        Some(MetaValue::String(value)) => Ok(value.clone()),
        Some(other) => Err(format!("{} is not a string: {other:?}", key(name))),
        None => Err(format!("missing metadata {}", key(name))),
    }
}
