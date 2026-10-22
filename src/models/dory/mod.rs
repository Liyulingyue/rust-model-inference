//! NVIDIA Nemotron-Labs Dory 1.0 (looped normalized-GPT).

mod config;
mod forward;
pub(crate) mod weights;
#[cfg(feature = "vulkan")]
pub(crate) use forward::trace;

pub use config::DoryConfig;
pub use forward::{run_forward_logits_dory, run_inference, DoryModel, DorySession};
