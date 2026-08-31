//! Thin dispatch layer for embedding inference.
//!
//! Delegates to `models::qwen3::embedding::run_embedding`.

pub use crate::models::qwen3::embedding::{compute_embedding, run_embedding};
