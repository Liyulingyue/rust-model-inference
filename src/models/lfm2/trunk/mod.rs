//! LFM2 transformer trunk
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2.
//! LFM2 has no `Model` struct (weights and config are passed around
//! independently), so `trunk/` exposes the config / weights / forward /
//! session helpers directly.

pub mod config;
pub mod forward;
pub mod weights;

pub use config::Lfm2Config;
pub use forward::{run_forward_logits_lfm2, run_inference, run_inference_stream};
pub use weights::{get_f32_tensor, load_layers, Lfm2LayerWeights};
