//! Fun-ASR-Nano: SenseVoice SAN-M encoder + Qwen3 LLM for speech recognition.
//!
//! Pipeline: WAV (16 kHz mono) → kaldi 80-mel fbank + LFR(7/6) →
//! SAN-M encoder (50+20 layers) → adaptor (512→1024) →
//! low-frame-rate truncation → [prefix tokens | audio embeds | suffix tokens]
//! → Qwen3 LLM → transcription.
//!
//! Reference: FunASR llama.cpp runtime (runtime-llamacpp-v0.2.6).

pub mod config;
pub mod encoder;
pub mod fbank;
pub mod model;

pub use config::FunAsrConfig;
pub use encoder::FunAsrEncoder;
pub use model::{is_funasr_encoder, run_funasr_cli, FunAsrTranscription};

/// GGUF architecture string for the Fun-ASR-Nano encoder.
pub const ENCODER_ARCH: &str = "funasr-sensevoice-encoder";
