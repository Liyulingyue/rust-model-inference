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

use crate::core::tensor::TensorSource;
use dac::DacState;
use tfm::WaveformTransformerState;

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
pub use wav::{encode_wav_pcm16, write_wav_f32, WavError};

const CODE2WAV_WINDOW: usize = 72;
const SAMPLES_PER_FRAME: usize = 1920;

pub struct Code2WavDecoder {
    rvq: RvqDecoder,
    transformer: WaveformTransformer,
    dac: DacDecoder,
}

impl Code2WavDecoder {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        Ok(Self {
            rvq: RvqDecoder::from_source(source)?,
            transformer: WaveformTransformer::from_source(source)?,
            dac: DacDecoder::from_source(source)?,
        })
    }

    pub fn decode(&self, frames: &[[u32; RVQ_LEVELS]]) -> Result<Vec<f32>, String> {
        self.decode_with_chunk_size(frames, CODE2WAV_WINDOW)
    }

    pub fn decode_with_chunk_size(
        &self,
        frames: &[[u32; RVQ_LEVELS]],
        chunk_size: usize,
    ) -> Result<Vec<f32>, String> {
        if frames.is_empty() {
            return Err("Code2Wav requires at least one frame".into());
        }
        if !(1..=CODE2WAV_WINDOW).contains(&chunk_size) {
            return Err(format!(
                "Code2Wav chunk size {chunk_size} must be in 1..={CODE2WAV_WINDOW}"
            ));
        }
        let expected_samples = frames
            .len()
            .checked_mul(SAMPLES_PER_FRAME)
            .ok_or_else(|| "Code2Wav sample count overflow".to_string())?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(expected_samples)
            .map_err(|error| format!("Failed to allocate Code2Wav output: {error}"))?;
        let mut transformer_state = WaveformTransformerState::new();
        let mut dac_state = DacState::new();
        let t_start = std::time::Instant::now();
        let mut t_rvq = std::time::Duration::ZERO;
        let mut t_dac_pre = std::time::Duration::ZERO;
        let mut t_transformer = std::time::Duration::ZERO;
        let mut t_dac_decode = std::time::Duration::ZERO;
        for chunk in frames.chunks(chunk_size) {
            let t0 = std::time::Instant::now();
            let hidden = self.rvq.decode_frames(chunk)?;
            t_rvq += t0.elapsed();

            let t1 = std::time::Instant::now();
            let pre_conv = self
                .dac
                .pre_conv_window(&hidden, chunk.len(), &mut dac_state)?;
            t_dac_pre += t1.elapsed();

            let t2 = std::time::Instant::now();
            let transformed =
                self.transformer
                    .forward_window(&pre_conv, chunk.len(), &mut transformer_state)?;
            t_transformer += t2.elapsed();

            let t3 = std::time::Instant::now();
            output.extend(
                self.dac
                    .decode_window(&transformed, chunk.len(), &mut dac_state)?,
            );
            t_dac_decode += t3.elapsed();
        }
        eprintln!("  [codec_decode] total={:.3}s rvq={:.3}s dac_pre={:.3}s transformer={:.3}s dac_decode={:.3}s",
            t_start.elapsed().as_secs_f64(), t_rvq.as_secs_f64(), t_dac_pre.as_secs_f64(), t_transformer.as_secs_f64(), t_dac_decode.as_secs_f64());
        if output.len() != expected_samples {
            return Err(format!(
                "Code2Wav produced {} samples, expected {expected_samples}",
                output.len()
            ));
        }
        if output.iter().any(|sample| !sample.is_finite()) {
            return Err("Code2Wav produced non-finite PCM".into());
        }
        Ok(output)
    }
}

/// Number of RVQ codebook levels (1 first + 15 residual).
pub const RVQ_LEVELS: usize = 16;
/// Per-codebook vector dimensionality.
pub const RVQ_CODE_DIM: usize = 256;
/// Per-codebook vocabulary size.
pub const RVQ_CODEBOOK_SIZE: usize = 2048;
/// Speaker embedding dimensionality (ECAPA-TDNN output before projection).
pub const SPEAKER_EMB_DIM: usize = 1536;
/// Waveform sample rate in Hz. Two 2× stages followed by DAC strides 8×5×4×3
/// produce 1920 samples per 12.5 Hz codec frame directly at 24 kHz.
pub const WAVEFORM_SAMPLE_RATE: u32 = 24_000;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::ggufrs::{open_model_source, ComponentRole};
    use std::path::Path;

    #[test]
    #[ignore = "requires QWEN3_TTS_MMPROJ"]
    fn code2wav_matches_across_decode_window_boundaries() {
        let path = std::env::var("QWEN3_TTS_MMPROJ").unwrap();
        let source = open_model_source(Path::new(&path), ComponentRole::Mmproj).unwrap();
        let decoder = Code2WavDecoder::from_source(source.as_ref()).unwrap();
        let frames = vec![[0u32; 16]; 80];
        let whole = decoder.decode_with_chunk_size(&frames, 72).unwrap();
        let split = decoder.decode_with_chunk_size(&frames, 17).unwrap();
        assert_eq!(whole.len(), 80 * 1920);
        assert_eq!(
            whole
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            split
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
