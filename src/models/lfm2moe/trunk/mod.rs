//! LFM2-MoE transformer trunk
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2.
//! Mirrors the `lfm2` trunk layout; the FFN stage branches between dense
//! SwiGLU (leading blocks) and MoE (remaining blocks) in `forward.rs`.

pub mod config;
pub mod forward;
pub mod weights;

pub use config::Lfm2MoeConfig;
pub use forward::run_inference;
pub use weights::{get_f32_tensor, load_layers, Lfm2MoeLayerWeights};
