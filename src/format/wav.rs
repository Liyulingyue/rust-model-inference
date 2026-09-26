//! WAV writer for the Qwen3-TTS codec decoder.
//!
//! Writes interleaved 16-bit PCM WAV files. Sample rate and channel count are
//! taken from the caller.

use std::path::Path;

pub fn write_wav_f32_channels<P: AsRef<Path>>(
    path: P,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<(), String> {
    let bytes = encode_wav_pcm16_channels(samples, sample_rate, channels)?;
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}

pub fn encode_wav_pcm16_channels(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<Vec<u8>, String> {
    if samples.is_empty() {
        return Err("WAV input has zero samples".into());
    }
    if sample_rate == 0 {
        return Err("sample rate must be nonzero".into());
    }
    if channels == 0 {
        return Err("channel count must be nonzero".into());
    }
    if samples.len() % usize::from(channels) != 0 {
        return Err("PCM sample count must contain complete frames".into());
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("PCM contains a non-finite sample".into());
    }
    let data_bytes = samples
        .len()
        .checked_mul(2)
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| "PCM data exceeds RIFF limits".to_string())?;
    let chunk_size = 36u32
        .checked_add(data_bytes)
        .ok_or_else(|| "RIFF chunk size overflow".to_string())?;
    let block_align = channels
        .checked_mul(2)
        .ok_or_else(|| "block alignment overflow".to_string())?;
    let byte_rate = sample_rate
        .checked_mul(u32::from(block_align))
        .ok_or_else(|| "byte rate overflow".to_string())?;
    let capacity = 44usize
        .checked_add(data_bytes as usize)
        .ok_or_else(|| "WAV allocation overflow".to_string())?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| format!("WAV allocation failed: {error}"))?;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&chunk_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_bytes.to_le_bytes());
    for &sample in samples {
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
        let bytes = encode_wav_pcm16_channels(&[-2.0, 0.0, 2.0], 24_000, 1).unwrap();
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

    #[test]
    fn wav_serialization_supports_interleaved_stereo_48k_and_preserves_mono() {
        let stereo = encode_wav_pcm16_channels(&[-1.0, 1.0, 0.5, -0.5], 48_000, 2).unwrap();
        assert_eq!(u16::from_le_bytes(stereo[22..24].try_into().unwrap()), 2);
        assert_eq!(
            u32::from_le_bytes(stereo[24..28].try_into().unwrap()),
            48_000
        );
        assert_eq!(
            u32::from_le_bytes(stereo[28..32].try_into().unwrap()),
            192_000
        );
        assert_eq!(u16::from_le_bytes(stereo[32..34].try_into().unwrap()), 4);
        assert_eq!(
            encode_wav_pcm16_channels(&[0.0], 24_000, 1).unwrap()[22..24],
            1u16.to_le_bytes()
        );
    }

    #[test]
    fn wav_rejects_zero_channels_partial_frames_and_non_finite_samples() {
        assert!(encode_wav_pcm16_channels(&[0.0], 48_000, 0).is_err());
        assert!(encode_wav_pcm16_channels(&[0.0, 1.0, 2.0], 48_000, 2).is_err());
        assert!(encode_wav_pcm16_channels(&[0.0, f32::NAN], 48_000, 2).is_err());
    }
}
