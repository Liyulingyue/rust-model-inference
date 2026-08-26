//! WAV writer for the Qwen3-TTS codec decoder.
//!
//! Writes a single-channel 16-bit PCM WAV file (the de-facto default for
//! speech samples). Sample rate is taken from the caller.

use std::io;
use std::path::Path;

#[derive(Debug)]
pub enum WavError {
    Io(io::Error),
    Empty,
    Invalid(String),
}

impl std::fmt::Display for WavError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WavError::Io(err) => write!(formatter, "WAV I/O error: {err}"),
            WavError::Empty => write!(formatter, "WAV input has zero samples"),
            WavError::Invalid(message) => write!(formatter, "Invalid WAV input: {message}"),
        }
    }
}

impl From<io::Error> for WavError {
    fn from(err: io::Error) -> Self {
        WavError::Io(err)
    }
}

impl std::error::Error for WavError {}

/// Write a mono PCM 16-bit WAV from `samples`. `sample_rate` is in Hz.
pub fn write_wav_f32<P: AsRef<Path>>(
    path: P,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(), WavError> {
    let bytes = encode_wav_pcm16(samples, sample_rate)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, WavError> {
    if samples.is_empty() {
        return Err(WavError::Empty);
    }
    if sample_rate == 0 {
        return Err(WavError::Invalid("sample rate must be nonzero".into()));
    }
    let data_bytes = samples
        .len()
        .checked_mul(2)
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| WavError::Invalid("PCM data exceeds RIFF limits".into()))?;
    let chunk_size = 36u32
        .checked_add(data_bytes)
        .ok_or_else(|| WavError::Invalid("RIFF chunk size overflow".into()))?;
    let byte_rate = sample_rate
        .checked_mul(2)
        .ok_or_else(|| WavError::Invalid("byte rate overflow".into()))?;
    let capacity = 44usize
        .checked_add(data_bytes as usize)
        .ok_or_else(|| WavError::Invalid("WAV allocation overflow".into()))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| WavError::Invalid(format!("WAV allocation failed: {error}")))?;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&chunk_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_bytes.to_le_bytes());
    for &sample in samples {
        if !sample.is_finite() {
            return Err(WavError::Invalid("PCM contains a non-finite sample".into()));
        }
        let clamped = sample.clamp(-1.0, 1.0);
        let pcm = (clamped * i16::MAX as f32) as i16;
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_serialization_is_mono_24k_pcm16_and_checked() {
        let bytes = encode_wav_pcm16(&[-2.0, 0.0, 2.0], 24_000).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            24_000
        );
        assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 16);
        assert_eq!(
            i16::from_le_bytes(bytes[44..46].try_into().unwrap()),
            -32767
        );
        assert_eq!(i16::from_le_bytes(bytes[48..50].try_into().unwrap()), 32767);
    }
}
