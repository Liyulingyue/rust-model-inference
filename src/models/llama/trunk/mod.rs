//! LLaMA transformer trunk
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2.
//! LLaMA has no explicit `Config`/`Session` struct — config is parsed via
//! [`crate::core::loader::model_config_from_source`] and the forward loop
//! runs as a single function call.

pub mod forward;
pub mod weights;

pub use forward::{run_inference, run_inference_tokens};
pub use weights::{get_f32_tensor, load_layers, load_layers_static, LlamaLayerWeights};