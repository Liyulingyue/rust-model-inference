//! DFlash / DSpark draft trunk.
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2:
//! - `config.rs` — `DflashConfig`
//! - `weights.rs` — `DflashWeights`, `DflashLayerWeights`, the Markov table
//!   and the load helpers
//! - `scratch.rs` — `DflashScratchpad`
//! - `forward.rs` — the encoder pass (`inject_target_features`) and the
//!   decoder pass (`decode_block`)
//! - `session.rs` — `DflashSession`, `DflashKvState`, `DsparkDraft` and the
//!   DSpark head application (Markov bias, confidence gate, greedy pick)

pub mod config;
pub mod forward;
pub mod scratch;
pub mod session;
#[cfg(test)]
pub mod tests;
pub mod weights;

pub use config::DflashConfig;
pub use session::{DflashSession, DsparkDraft};
pub use weights::{load_weights, DflashWeights};
