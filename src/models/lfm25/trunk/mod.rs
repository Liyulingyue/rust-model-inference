//! LFM2.5 transformer trunk
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2.
//! LFM2.5 has no `Model` struct (weights and config are passed around
//! independently), so `trunk/` exposes the config / weights / forward /
//! session helpers directly.

pub mod config;
pub mod forward;
pub mod weights;

pub use config::Lfm25Config;
pub use forward::run_inference;
pub use weights::{get_f32_tensor, load_layers, Lfm25LayerWeights};