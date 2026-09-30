//! GLiNER2.5 BoundaryExtractor: multi-task structured prediction model.
//!
//! Architecture differs from `SpanExtractor` (Decide, large-v1):
//!  - Encoder: DeBERTa-v3-base for ``gliner2.5-base-v1`` (the variant we
//!    package today), shared forward with Decide via
//!    ``crate::models::gliner::compute::encode``.
//!  - BoundaryEncoder: shift text states left/right with BOS/EOS, project
//!    each side to a per-boundary dim, concat + project + LayerNorm.
//!    Optionally refines with self-attention and SwiGLU blocks.
//!  - BoundaryProposer: scores each boundary position with a query/key
//!    projection and selects top-K starts and ends (rotary endpoint
//!    embeddings if ``enable_rotary_endpoints``).
//!  - PairScorer: combines start/end marginals with endpoint compatibility,
//!    length features, and the classifier head to produce one logit per
//!    (start, end, query) candidate.
//!
//! See ``target/gliner2-oracle/gliner2/models/boundary/`` for the
//! reference (8149 lines of Python). The current commit implements the
//! encoder + boundary state encoding pieces; the proposer / pair scorer /
//! relation scorer / record decoder are tracked in ``glinerTODO.md`` and
//! will follow in subsequent commits.

pub mod forward;
pub mod loader;
pub mod marginals;
pub mod pair_scorer;
pub mod proposer;

pub use forward::{BoundaryEncoder, BoundaryEncoding};
pub use loader::BoundaryModel;
pub use marginals::{BoundaryMarginals, BoundaryQueryHead};
pub use pair_scorer::PairScorer;
pub use proposer::{BoundaryProposer, RotaryBoundaryEmbedding};

use crate::core::tensor::TensorSource;

/// Detect whether a GGUF carries the boundary variant produced by
/// ``tools/converter/gliner/convert_boundary.py``. Kept here so callers
/// (server dispatch, CLI flag check) have a single source of truth.
pub fn is_boundary_gguf(source: &dyn TensorSource) -> bool {
    source
        .metadata("general.architecture")
        .and_then(crate::core::tensor::MetaValue::to_string_val)
        .is_some_and(|arch| arch == "gliner2")
        && source
            .metadata("gliner2.variant")
            .and_then(crate::core::tensor::MetaValue::to_string_val)
            .is_some_and(|variant| variant == "boundary")
}