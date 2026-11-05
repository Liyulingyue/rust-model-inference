//! Qwen3-Audio model family.
//!
//! Phase 4b split: the audio pre-processing pipeline (WAV decode + log-mel
//! spectrogram) lives in [`audio_processor`], while the transformer-based
//! audio encoder and `validate_qwen3a_source` live in [`model`].
//!
//! Dependency rules:
//! - `audio_processor` depends only on `core::tensor`.
//! - `model` depends on `core::*`, `ops::*`, and `super::audio_processor`
//!   (uses `MelWindow`, `log_mel_windows`, `AsrAudioError`).
//! - `runtime` depends on `core::*`, `format::*`, `models::qwen3::base`,
//!   and `super::audio_processor` and `super::model`.

pub mod audio_processor;
pub mod mel_encoder;
pub mod model;
