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

pub mod config;
pub mod forward;
pub mod positions;
pub mod session;
pub mod tests;
pub mod util;
pub mod weights;

pub use config::{Qwen3Config, Qwen3Rope};
pub use forward::{
    Qwen3Generation, Qwen3GenerateOptions, Qwen3Input, run_shared_inference, text_encode,
};
pub use positions::qwen_text_positions;
pub use session::Qwen3Session;
pub use weights::{
    get_f32_tensor, load_layers, load_layers_static, static_weight, Qwen3LayerWeights,
    Qwen3Model,
};