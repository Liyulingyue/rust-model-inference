//! Qwen3 model family
//!
//! Per [`MODEL_ORGANIZATION.md`](../../../../docs/MODEL_ORGANIZATION.md) §2.1:
//! - `trunk/` contains the pure transformer decoder (`Qwen3Model` + `Qwen3Session`).
//! - `asr/` and `tts/` are siblings to `trunk/` (not nested).
//!
//! ## Public surface (re-exported from `trunk/`)
//!
//! `Qwen3Model`, `Qwen3Session`, `Qwen3Config`, `Qwen3Rope`, `Qwen3Input`,
//! `Qwen3GenerateOptions`, `Qwen3Generation`, `text_encode`,
//! `run_shared_inference`, `qwen_text_positions`, `Qwen3LayerWeights`,
//! `get_f32_tensor`, `load_layers`, `load_layers_static`, `static_weight`.

pub mod asr;
pub mod embedding;
pub mod hunyuan;
pub mod omni;
pub mod text;
pub mod tts;
pub mod trunk;
pub mod vision;

pub use embedding::run_embedding;
pub use text::{run_inference, run_inference_tokens};
pub use trunk::{
    get_f32_tensor, load_layers, load_layers_static, static_weight, qwen_text_positions,
    run_shared_inference, text_encode, Qwen3Config, Qwen3Generation, Qwen3GenerateOptions,
    Qwen3Input, Qwen3LayerWeights, Qwen3Model, Qwen3Rope, Qwen3Session,
};
