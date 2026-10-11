//! EmbeddingGemma-2 (Google) — multimodal encoder (text + vision + audio).
//!
//! `arch = "gemma4-embedding"` — a single-GGUF multi-section encoder that
//! produces 768-dim embeddings via mean pooling + a learned projection.
//! The text backbone is `gemma4-text` (bidirectional + PLE block + 5-norm
//! sandwich + sliding window attention with per-layer head_dim/kv_heads).
//! The vision tower is `gemma4-vision` (SigLIP-style + axial RoPE, 16
//! layers). The audio tower is `gemma4-audio` (Conformer, 12 layers).
//!
//! Reference for tensor naming: `tools/converter/embeddinggemma2/convert_embeddinggemma2.py`.
//! End-to-end forward path lives in this module; convert metadata + tensor
//! contract tests live in `tests/embeddinggemma_2.rs`.

pub mod audio;
pub mod text;
pub mod vision;

pub use audio::{EmbeddingGemmaAudioConfig, EmbeddingGemmaAudioModel};
pub use text::{EmbeddingGemmaTextConfig, EmbeddingGemmaTextModel};
pub use vision::{EmbeddingGemmaVisionConfig, EmbeddingGemmaVisionModel};
