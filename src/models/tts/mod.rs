//! Text-to-Speech inference.
//!
//! Qwen3-TTS-12Hz 1.7B Base is split across two GGUF files:
//!
//! - [`Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf`] — the autoregressive "Talker" LLM
//!   that maps text tokens to a stream of audio-codebook token ids at 12 Hz.
//! - [`mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf`] — the codec decoder
//!   (speaker encoder + RVQ code predictor + DAC upsampler) that turns those
//!   audio codes into a 24 kHz waveform.
//!
//! Stage 1 ships only the Talker; the codec decoder is added in Stage 2.

pub mod codec;
pub mod speaker;
pub mod talker;

pub use talker::{Qwen3TtsGeneration, Qwen3TtsTalker, Qwen3TtsTalkerConfig};

/// Format used by the 12 Hz codec for its audio tokens. Each emitted token is
/// a single codebook index in the range `audio_codebook_offset..audio_codebook_offset+3072`.
pub const AUDIO_CODEBOOK_SIZE: usize = 3072;

/// EOS token id observed in the Base model metadata (`tokenizer.ggml.eos_token_id`).
/// The base Talker terminates generation when this audio-codebook token is sampled.
pub const TTS_EOS_TOKEN_ID: u32 = 154086;

/// Sampling temperature suggested by `general.sampling.temp = 0.9` in the
/// Qwen3-TTS Base GGUF metadata.
pub const TTS_DEFAULT_TEMP: f32 = 0.9;
