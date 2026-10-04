//! Qwen3 Transformer Trunk
//!
//! Pure re-export file. All actual type definitions live in submodules:
//! - `config.rs` — `Qwen3Config` + `Qwen3Rope`
//! - `weights.rs` — `Qwen3Model` struct (weight tables) + `Qwen3LayerWeights` + load helpers
//! - `forward.rs` — `text_encode` + `run_shared_inference` + `Qwen3Input`/`Qwen3GenerateOptions`/`Qwen3Generation` + `Qwen3Model::generate` / `text_encode` methods
//! - `session.rs` — `Qwen3Session` struct + `impl Qwen3Session` (KV cache management)
//! - `util.rs` — helpers + unit tests
//! - `positions.rs` — `qwen_text_positions` (RoPE position builder)
//! - `tests.rs` — test fixtures
//!
//! # BitNet separation
//!
//! This trunk is BitNet-free. Microsoft BitNet b1.58 (file_type=40)
//! variants ride a separate trunk family under
//! [`crate::models::bitnet`] (`qwen3_arch` for the
//! `bitnet-embedding-0.6b` Qwen3-architecture model,
//! `gemma3_arch` for `bitnet-embedding-270m`). The standard
//! Qwen3 forward here is unaffected by BitNet's per-projection
//! RMSNorm / I2_S matmul pattern — see
//! `docs/usage/bitnet_embedding.md` for the architectural
//! rationale and `TODO.md` for the I2_S provenance note.

pub mod config;
pub mod forward;
pub mod positions;
mod prefill;
pub mod rerank;
pub mod session;
pub mod tests;
pub mod util;
pub mod weights;

pub use config::{Qwen3Config, Qwen3Rope};
pub use forward::{
    run_shared_inference, text_encode, Qwen3GenerateOptions, Qwen3Generation, Qwen3Input,
};
pub use positions::qwen_text_positions;
pub use rerank::score_qwen3_rerank;
pub use session::Qwen3Session;
pub use weights::{
    get_f32_tensor, load_layers, load_layers_static, static_weight, Qwen3LayerWeights,
    Qwen3Model,
};
