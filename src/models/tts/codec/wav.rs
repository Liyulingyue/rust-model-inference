//! WAV writer for the Qwen3-TTS codec decoder.
//!
//! Writes a single-channel 16-bit PCM WAV file (the de-facto default for
//! speech samples). Sample rate is taken from the caller.

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

#[derive(Debug)]
pub enum WavError {
    Io(io::Error),
    Empty,
}

impl std::fmt::Display for WavError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WavError::Io(err) => write!(formatter, "WAV I/O error: {err}"),
            WavError::Empty => write!(formatter, "WAV input has zero samples"),
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
    if samples.is_empty() {
        return Err(WavError::Empty);
    }
    let mut file = File::create(path)?;
    write_wav_header(&mut file, samples.len() as u32, sample_rate)?;
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let pcm = (clamped * i16::MAX as f32) as i16;
        file.write_all(&pcm.to_le_bytes())?;
    }
    Ok(())
}

fn write_wav_header(file: &mut File, num_samples: u32, sample_rate: u32) -> io::Result<()> {
    let bits_per_sample = 16u16;
    let num_channels = 1u16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_bytes = num_samples * block_align as u32;
    let chunk_size = 36 + data_bytes;

    file.write_all(b"RIFF")?;
    file.write_all(&chunk_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_bytes.to_le_bytes())?;
    Ok(())
}