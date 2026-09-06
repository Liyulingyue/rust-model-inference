//! VibeVoice ASR — microsoft/VibeVoice-ASR-Streaming-7B (arch `qwen2` LLM +
//! `clip` mmproj with `vibevoice_asr` projector).
//!
//! Two GGUF files are consumed:
//!
//! - `VibeVoice-ASR-Streaming-7B-Q8_0.gguf` (arch `qwen2`) — the Qwen2.5-7B decoder
//!   (28×3584, 28 Q / 4 KV heads, θ=1e6, untied lm_head, Q/K/V biases, Q8_0).
//! - `mmproj-VibeVoice-ASR-Streaming-7B-BF16.gguf` (arch `clip`) — acoustic and
//!   semantic tokenizer encoders (ConvNeXt-style causal conv nets) plus their
//!   speech connectors (fc1 → RMSNorm → fc2), BF16.
//!
//! Transcription follows the official streaming protocol: audio is cut into
//! 2.933 s chunks with a 0.533 s lookahead (22+4 latent frames at 7.5 fps),
//! each chunk is encoded independently into `acoustic + semantic` projected
//! features, and per chunk the session prefills
//! `[speech_start, features…, speech_end]` before greedily decoding text up
//! to the `<|text_chunk_end|>` control token (or EOS).

pub mod config;
pub mod encoder;
pub mod generate;
pub mod llm;

pub use config::{is_vibevoice_asr_mmproj, VibeVoiceAsrConfig};
pub use generate::{transcribe_streaming, TranscribeOptions, VibeVoiceAsrModel};
pub use llm::VibeVoiceAsrLlm;
