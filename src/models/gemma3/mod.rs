//! Gemma3 model family (decoder-only).
//!
//! Currently houses the BitNet b1.58 trunk for
//! `microsoft/bitnet-embedding-270m` (`general.architecture =
//! "gemma3"`, `general.file_type = 40`). See the module-level docs
//! of [`crate::models::gemma3::trunk`] for the architectural
//! differences vs the qwen3 trunk (4-norm sandwich, 640-dim
//! hidden state, 4:1 GQA at head_dim=256, 18 layers).
//!
//! ## Public surface
//!
//! Re-exported from `trunk/`:
//! - `Gemma3Config`, `Gemma3Rope`
//! - `Gemma3Model`, `Gemma3LayerWeights`, `Weight`
//! - `load_layers`, `load_layers_static`, `static_weight`
//! - `get_f32_tensor`
//! - `text_encode`, `run_shared_inference`
//! - `BitLinearSlot`, `BitLinearWeights` (re-exported from
//!   `crate::ops::bitnet`)
//!
//! Re-exported from `embedding.rs`:
//! - `compute_embedding`, `run_embedding_tokens`, `print_embedding`,
//!   `run_embedding` (the CLI entry point that mirrors
//!   `qwen3::embedding::run_embedding`).

pub mod embedding;
pub mod trunk;

pub use embedding::{compute_embedding, print_embedding, run_embedding, run_embedding_tokens};
pub use trunk::{
    get_f32_tensor, load_layers, load_layers_static, run_shared_inference, static_weight,
    text_encode, BitLinearSlot, BitLinearWeights, Gemma3Config, Gemma3LayerWeights, Gemma3Model,
    Gemma3Rope, Weight,
};
