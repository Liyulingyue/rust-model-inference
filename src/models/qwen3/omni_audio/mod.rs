//! Qwen2.5-Omni audio tower encoder for AuK CFMEdit reference-audio conditioning.
//!
//! Architecture (matches `references/audio.cpp/src/community_models/auk/audio_conditioning.cpp`):
//! - conv1: Conv1d 128 mel channels -> 1280, kernel 3, padding 1, GELU(erf)
//! - conv2: Conv1d 1280 -> 1280, kernel 3, stride 2, padding 1, GELU(erf) — halves time
//! - 100-frame learnable sinusoidal position encoding (sin/cos split, 640+640 channels)
//! - 32 transformer encoder layers (hidden 1280, heads 20, head_dim 64, ffn 5120)
//! - ln_post (LayerNorm 1280, eps 1e-5)
//! - proj: 1280 -> 2048 (matches Qwen2.5-Omni text hidden space, TEXT_IN in AuK)
//!
//! GGUF tensor prefix: `thinker.audio_tower.*` (in audio.cpp's format).
//! k_proj has no bias (zero_bias is used in audio.cpp's load helper; we just skip it).
//!
//! Run with the BF16 Qwen2.5-Omni GGUF (7.0 GB) which contains the audio tower
//! (`models/Qwen2.5-Omni-3B-bf16-GGUF/qwen2.5-omni-3b-bf16.gguf`). The Q8_0 GGUF
//! only has the text trunk and is insufficient for CFMEdit.
//!
//! The module is a fresh implementation specific to the AuK tensor prefix and
//! is not to be confused with `qwen3::omni` (which targets llama.cpp's qwen2.5o
//! format using the `a.conv1d.1` prefix).

pub mod tower;

pub use tower::{AudioTowerConfig, AudioTowerModel, QWEN25_OMNI_AUDIO_TOWER_PREFIX};
