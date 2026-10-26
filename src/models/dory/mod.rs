//! NVIDIA Nemotron-Labs Dory 1.0 (looped normalized-GPT).

mod config;
mod forward;
pub mod runtime;
pub(crate) mod weights;
#[cfg(feature = "vulkan")]
pub(crate) use forward::{trace, PREFILL_BATCH};

pub use config::DoryConfig;
pub use forward::{run_forward_logits_dory, run_inference, DoryModel, DorySession};
pub use runtime::DoryTextRuntime;
