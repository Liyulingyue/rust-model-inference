//! Qwen3-TTS-12Hz codec decoder.
//!
//! Given a stream of audio-codebook token ids produced by the Talker LLM, this
//! module reconstructs a 24 kHz waveform through a 4-stage neural codec:
//!
//! 1. [`predictor`] — 5-layer transformer that lifts the Talker's audio tokens
//!    to a continuous 1024-dim representation and predicts the 15 residual
//!    RVQ codes per timestep.
//! 2. [`rvq`] — Residual Vector Quantisation decode (1 first + 15 residual
//!    codebooks of 2048 entries × 256 dims each).
//! 3. [`tfm`] — 8-layer waveform transformer that mixes the RVQ codebook
//!    vectors with the speaker embedding.
//! 4. [`dac`] — 4-stage DAC upsampler (1536 → 768 → 384 → 192 → 96 → 1) with
//!    Snake1d activation that produces the final waveform.

pub mod conv;
pub mod dac;
pub mod predictor;
pub mod rvq;
pub mod snake;
pub mod tfm;
pub mod wav;

pub use dac::DacDecoder;
pub use predictor::CodePredictor;
pub use rvq::RvqDecoder;
pub use tfm::WaveformTransformer;
pub use wav::{write_wav_f32, WavError};

/// Number of RVQ codebook levels (1 first + 15 residual).
pub const RVQ_LEVELS: usize = 16;
/// Per-codebook vector dimensionality.
pub const RVQ_CODE_DIM: usize = 256;
/// Per-codebook vocabulary size.
pub const RVQ_CODEBOOK_SIZE: usize = 2048;
/// Speaker embedding dimensionality (ECAPA-TDNN output before projection).
pub const SPEAKER_EMB_DIM: usize = 1536;
/// Waveform sample rate in Hz. The DAC upsampler chain has cumulative stride
/// `16 * 10 * 8 * 6 = 7680`; with a 12 Hz codebook frame rate, the resulting
/// waveform sample rate is `12 * 7680 = 92160 Hz`, which we resample to
/// `WAVEFORM_SAMPLE_RATE` when writing the WAV file.
pub const WAVEFORM_SAMPLE_RATE: u32 = 24_000;