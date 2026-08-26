//! Qwen3.5 (hybrid Mamba SSM + dense attention) inference.
//!
//! ## Module map
//!
//! - `clip_config` — `Qwen35Config` parsed from GGUF metadata
//! - `vision` — vision encoder (patch embed + transformer + projector)
//! - `positions` — `build_qwen35_positions` for mRoPE-aware VL inputs
//! - `loader` — `Qwen35Model::from_source` + per-tensor helpers
//! - `forward` — `forward` / `_dense_attn_layer` / `_recurrent_layer` / `_ffn_parallel`
//! - `scratchpad` — `Qwen35Scratchpad` + KV cache helpers
//! - `util` — f16 decode + scalar Mamba helpers
//! - `session` — high-level inference state (`Qwen35Session`)
//! - `tests` — unit tests (only compiled under `cfg(test)`)
//!
//! ## Public surface
//!
//! `Qwen35Model` owns weights + config and exposes a low-level `forward`.
//! `Qwen35Session<'a>` wraps a model with the per-request KV cache,
//! scratchpad, and thread pool for incremental decode.

pub mod clip_config;
pub mod vision;

pub(crate) mod forward;
pub(crate) mod loader;
pub(crate) mod positions;
pub(crate) mod scratchpad;
pub(crate) mod session;
pub(crate) mod tests;
pub(crate) mod util;

// Re-exports for callers (e.g. `bin/server.rs`, `app/text.rs`).
pub use clip_config::Qwen35Config;
pub use positions::build_qwen35_positions;
pub use scratchpad::Qwen35Scratchpad;
pub use session::Qwen35Session;
pub use vision::VisionGrid;

use crate::core::tensor::TensorSource;
use crate::ops::kernel::Weight;

/// All weights for a single Qwen3.5 layer.
///
/// The `Option` fields distinguish dense-attention layers (which fill
/// `wq`/`wk`/`wv`/`wo`/`attn_q_norm`/`attn_k_norm`) from recurrent (Mamba
/// SSM) layers (which fill `wqkv`/`wqkv_gate`/`ssm_*`). `config.is_recurrent`
/// selects which group is active.
pub struct Qwen35LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub attn_post_norm: Vec<f32>,
    pub wq: Option<Weight<'a>>,
    pub wk: Option<Weight<'a>>,
    pub wv: Option<Weight<'a>>,
    pub wo: Option<Weight<'a>>,
    pub attn_q_norm: Option<Vec<f32>>,
    pub attn_k_norm: Option<Vec<f32>>,
    pub wqkv: Option<Weight<'a>>,
    pub wqkv_gate: Option<Weight<'a>>,
    pub ssm_conv1d: Option<Vec<f32>>,
    pub ssm_dt: Option<Vec<f32>>,
    pub ssm_a: Option<Vec<f32>>,
    pub ssm_beta: Option<Weight<'a>>,
    pub ssm_alpha: Option<Weight<'a>>,
    pub ssm_norm: Option<Vec<f32>>,
    pub ssm_out: Option<Weight<'a>>,
    pub ffn_gate: Weight<'a>,
    pub ffn_up: Weight<'a>,
    pub ffn_down: Weight<'a>,
}

/// Loaded Qwen3.5 model weights + parsed config.
///
/// `from_source` is defined in `loader.rs`. `forward` and friends are
/// defined in `forward.rs`. This struct is the source of truth shared by
/// `Qwen35Session` and the existing `app/text.rs` / `bin/server.rs`
/// call sites.
pub struct Qwen35Model<'a> {
    pub config: Qwen35Config,
    pub tok_embd: Vec<f32>,
    pub output_norm: Vec<f32>,
    pub output_weight: Weight<'a>,
    pub layers: Vec<Qwen35LayerWeights<'a>>,
}

// Convenience alias so that `impl Qwen35Model { fn from_source(...) }` in
// `loader.rs` and `impl Qwen35Model { fn forward(...) }` in `forward.rs`
// can refer to a common TensorSource without redundant imports.
pub(crate) type Source<'a> = &'a dyn TensorSource;