//! LLaMA transformer trunk
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2.
//! LLaMA has no explicit `Config` struct — config is parsed via
//! [`crate::core::loader::model_config_from_source`]. The forward loop
//! is wrapped by `LlamaSession` (see `session.rs`), which implements
//! [`crate::core::prefill::ChunkedPrefill`]. The legacy free-function
//! entry points (`run_forward_logits_llama_with_batch`, `run_inference`,
//! `run_inference_tokens`) are kept for callers that haven't migrated.

pub mod forward;
pub mod session;
pub mod weights;

pub use forward::{
    run_forward_logits_llama_with_batch, run_inference, run_inference_tokens,
};
pub use session::{LlamaSession, LlamaSessionConfig, LlamaWeights};
pub use weights::{get_f32_tensor, load_layers, load_layers_static, LlamaLayerWeights};
