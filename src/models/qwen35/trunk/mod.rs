//! Qwen3.5 (hybrid Mamba SSM + dense attention) transformer trunk
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2:
//! - `config.rs` — `Qwen35Config`
//! - `weights.rs` — shared `HybridTrunk`, `Qwen35Model` alias, and load helpers
//! - `forward.rs` — `forward` / `_dense_attn_layer` / `_recurrent_layer` / `_ffn_parallel`
//! - `session.rs` — shared `HybridSession` and `Qwen35Session` alias
//! - `scratch.rs` — `Qwen35Scratchpad` + KV cache helpers
//! - `util.rs` — f16 decode + scalar Mamba helpers
//! - `positions.rs` — `build_qwen35_positions` for mRoPE-aware VL inputs
//! - `tests.rs` — unit tests

pub mod config;
pub mod forward;
pub mod positions;
pub mod scratch;
pub mod session;
pub mod tests;
pub mod util;
pub mod weights;

pub use config::Qwen35Config;
pub use forward::{run_classify_qwen35_with_batch, run_forward_logits_qwen35_with_batch};
pub use positions::build_qwen35_positions;
pub use scratch::Qwen35Scratchpad;
pub use session::{HybridSession, HybridTrunkModel, Qwen35DenseKvSnapshot, Qwen35Session};
pub use weights::{HybridTrunk, Qwen35LayerWeights, Qwen35Model};
