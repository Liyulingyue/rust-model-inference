//! Qwen3.5 (hybrid Mamba SSM + dense attention) transformer trunk
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2:
//! - `config.rs` — `Qwen35Config`
//! - `weights.rs` — `Qwen35Model` + `Qwen35LayerWeights` + load helpers
//! - `forward.rs` — `forward` / `_dense_attn_layer` / `_recurrent_layer` / `_ffn_parallel`
//! - `session.rs` — `Qwen35Session`
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
pub use positions::build_qwen35_positions;
pub use scratch::Qwen35Scratchpad;
pub use session::Qwen35Session;
pub use weights::{Qwen35LayerWeights, Qwen35Model};