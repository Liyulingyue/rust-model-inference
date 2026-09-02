//! dots.tts — fully continuous autoregressive flow-matching TTS (dots-studio).
//!
//! Two GGUF files are consumed:
//!
//! - `dots-tts-<variant>.gguf` (arch `qwen2`) — the Qwen2 text LLM (28×1536,
//!   12 Q / 2 KV heads, θ=1e6). Loaded through the same weight layout as the
//!   qwen3 trunk; this module only adds a step-by-step session that reports
//!   hidden states (see [`llm`]).
//! - `dots-tts-<variant>-mmproj.gguf` (arch `dotstts`) — patch encoder
//!   (audio → LLM embeddings), DiT velocity-field predictor (flow matching
//!   over 128-dim latents), CAM++ speaker (x-vector) encoder, AudioVAE
//!   vocoder and the latent statistics.
//!
//! Generation is an autoregressive loop over 4-frame latent patches:
//! LLM hidden state → `hidden_proj` → FM sequence; flow matching with CFG
//! (default euler/10 steps/guidance 1.2) decodes one 4×128 patch; the patch
//! is re-encoded by the patch encoder, fed back into the LLM, and the loop
//! continues until the EOS head fires. The patch stream is decoded to 48 kHz
//! mono by the AudioVAE vocoder.

pub mod config;
pub mod dit;
pub mod generate;
pub mod llm;
pub mod patch_encoder;
pub mod schedule;
pub mod speaker;
pub mod vocoder;

pub use config::DotsTtsConfig;
pub use generate::DotsTtsModel;