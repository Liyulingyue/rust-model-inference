use crate::models::qwen3::asr::audio_processor::{decode_pcm16_wav_any, RealFft};

const SAMPLE_RATE: u32 = 16_000;
const FFT_SIZE: usize = 512;
const WINDOW_SIZE: usize = 320;
const HOP: usize = 160;
const MEL_BINS: usize = 128;
const CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE as usize;

/// Gemma4A log-mel data in mel-major `[128, frames]` layout.
pub struct Gemma4AudioFeatures {
    pub values: Vec<f32>,
    pub frames: usize,
}

pub fn gemma4_audio_features(bytes: &[u8]) -> Result<Gemma4AudioFeatures, String> {
    let decoded = decode_pcm16_wav_any(bytes).map_err(|error| format!("{error:?}"))?;
    let channels = usize::from(decoded.channels);
    let mut mono = Vec::new();
    mono.try_reserve_exact(decoded.samples.len() / channels)
        .map_err(|_| "Gemma4 audio allocation failed")?;
    for frame in decoded.samples.chunks_exact(channels) {
        let sample = frame.iter().map(|value| f64::from(*value)).sum::<f64>() / channels as f64;
        if !sample.is_finite() {
            return Err("non-finite mixed audio sample".into());
        }
        mono.push(sample as f32);
    }
    let samples = resample_linear(&mono, decoded.sample_rate)?;
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("non-finite resampled audio sample".into());
    }

    let mut chunks = Vec::new();
    let mut frames = 0usize;
    for chunk in samples.chunks(CHUNK_SAMPLES) {
        let (values, chunk_frames) = gemma4_chunk_mel(chunk)?;
        frames = frames
            .checked_add(chunk_frames)
            .ok_or("Gemma4 audio frame count overflow")?;
        chunks.push((values, chunk_frames));
    }
    if frames == 0 {
        return Err("Gemma4 audio produced no frames".into());
    }
    let mut values = zeroed(
        MEL_BINS
            .checked_mul(frames)
            .ok_or("Gemma4 Mel size overflow")?,
    )?;
    let mut frame_offset = 0usize;
    for (chunk, chunk_frames) in chunks {
        for mel in 0..MEL_BINS {
            let source = &chunk[mel * chunk_frames..(mel + 1) * chunk_frames];
            let destination = &mut values
                [mel * frames + frame_offset..mel * frames + frame_offset + chunk_frames];
            destination.copy_from_slice(source);
        }
        frame_offset += chunk_frames;
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("non-finite Gemma4 Mel value".into());
    }

    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(
        "gemma4.audio.mel",
        None,
        &[MEL_BINS, frames],
        &values,
    ));
    Ok(Gemma4AudioFeatures { values, frames })
}

fn resample_linear(samples: &[f32], source_rate: u32) -> Result<Vec<f32>, String> {
    if samples.is_empty() || source_rate == 0 {
        return Err("invalid Gemma4 audio sample rate or data".into());
    }
    if source_rate == SAMPLE_RATE {
        return Ok(samples.to_vec());
    }
    let input_last = u64::try_from(samples.len() - 1).map_err(|_| "Gemma4 audio is too large")?;
    let output_last = input_last
        .checked_mul(u64::from(SAMPLE_RATE))
        .ok_or("Gemma4 resampled length overflow")?
        / u64::from(source_rate);
    let output_len = usize::try_from(
        output_last
            .checked_add(1)
            .ok_or("Gemma4 resampled length overflow")?,
    )
    .map_err(|_| "Gemma4 resampled length overflow")?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| "Gemma4 resample allocation failed")?;
    for output_index in 0..output_len {
        let position = u64::try_from(output_index)
            .map_err(|_| "Gemma4 resample position overflow")?
            .checked_mul(u64::from(source_rate))
            .ok_or("Gemma4 resample position overflow")?;
        let input_index = usize::try_from(position / u64::from(SAMPLE_RATE))
            .map_err(|_| "Gemma4 resample index overflow")?;
        let current = *samples
            .get(input_index)
            .ok_or("Gemma4 resample exceeded input")?;
        let next = samples.get(input_index + 1).copied().unwrap_or(current);
        let fraction = (position % u64::from(SAMPLE_RATE)) as f32 / SAMPLE_RATE as f32;
        let value = current + (next - current) * fraction;
        if !value.is_finite() {
            return Err("non-finite resampled audio sample".into());
        }
        output.push(value);
    }
    Ok(output)
}

fn gemma4_chunk_mel(samples: &[f32]) -> Result<(Vec<f32>, usize), String> {
    let left_pad = WINDOW_SIZE / 2;
    let with_left = i64::try_from(samples.len())
        .map_err(|_| "Gemma4 audio chunk is too large")?
        .checked_add(i64::try_from(left_pad).unwrap())
        .ok_or("Gemma4 audio frame count overflow")?;
    let frames =
        (with_left - i64::try_from(WINDOW_SIZE + 1).unwrap()) / i64::try_from(HOP).unwrap() + 1;
    let frames = usize::try_from(frames).map_err(|_| "Gemma4 audio is too short for a frame")?;
    if frames == 0 {
        return Err("Gemma4 audio is too short for a frame".into());
    }
    let padded_len = frames
        .checked_sub(1)
        .and_then(|value| value.checked_mul(HOP))
        .and_then(|value| value.checked_add(FFT_SIZE))
        .ok_or("Gemma4 padding size overflow")?;
    let total_pad = padded_len
        .checked_sub(samples.len())
        .ok_or("Gemma4 padding size overflow")?
        .max(left_pad);
    let mut padded = zeroed(
        total_pad
            .checked_add(samples.len())
            .ok_or("Gemma4 padding size overflow")?,
    )?;
    padded[left_pad..left_pad + samples.len()].copy_from_slice(samples);
    let hann = periodic_hann_window();
    let filters = htk_mel_filters()?;
    let fft_bins = FFT_SIZE / 2 + 1;
    let mut values = zeroed(
        MEL_BINS
            .checked_mul(frames)
            .ok_or("Gemma4 Mel size overflow")?,
    )?;
    let mut input = zeroed(FFT_SIZE)?;
    let mut magnitude = zeroed(fft_bins)?;
    let mut fft = RealFft::new(FFT_SIZE).map_err(|error| format!("{error:?}"))?;
    for frame in 0..frames {
        let start = frame.checked_mul(HOP).ok_or("Gemma4 FFT offset overflow")?;
        let source = padded
            .get(start..start + FFT_SIZE)
            .ok_or("Gemma4 padded FFT frame is truncated")?;
        for index in 0..FFT_SIZE {
            input[index] = source[index] * hann[index];
        }
        fft.magnitude(&input, &mut magnitude);
        if magnitude.iter().any(|value| !value.is_finite()) {
            return Err("non-finite Gemma4 FFT output".into());
        }
        for mel in 0..MEL_BINS {
            let filter = &filters[mel * fft_bins..(mel + 1) * fft_bins];
            let mut sum = 0.0f64;
            let mut bin = 0usize;
            while bin + 3 < fft_bins {
                sum += f64::from(
                    magnitude[bin] * filter[bin]
                        + magnitude[bin + 1] * filter[bin + 1]
                        + magnitude[bin + 2] * filter[bin + 2]
                        + magnitude[bin + 3] * filter[bin + 3],
                );
                bin += 4;
            }
            while bin < fft_bins {
                sum += f64::from(magnitude[bin] * filter[bin]);
                bin += 1;
            }
            let value = sum.max(0.001).ln() as f32;
            if !value.is_finite() {
                return Err("non-finite Gemma4 Mel value".into());
            }
            values[mel * frames + frame] = value;
        }
    }
    Ok((values, frames))
}

fn periodic_hann_window() -> Vec<f32> {
    (0..FFT_SIZE)
        .map(|index| {
            if index < WINDOW_SIZE {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / WINDOW_SIZE as f32).cos()
            } else {
                0.0
            }
        })
        .collect()
}

fn htk_mel_filters() -> Result<Vec<f32>, String> {
    let fft_bins = FFT_SIZE / 2 + 1;
    let max_mel = htk_hz_to_mel(SAMPLE_RATE as f64 / 2.0);
    let points: Vec<f64> = (0..MEL_BINS + 2)
        .map(|index| htk_mel_to_hz(max_mel * index as f64 / (MEL_BINS + 1) as f64))
        .collect();
    let mut filters = zeroed(
        MEL_BINS
            .checked_mul(fft_bins)
            .ok_or("Gemma4 Mel filter size overflow")?,
    )?;
    for mel in 0..MEL_BINS {
        let left = points[mel];
        let center = points[mel + 1];
        let right = points[mel + 2];
        for bin in 0..fft_bins {
            let hz = bin as f64 * SAMPLE_RATE as f64 / FFT_SIZE as f64;
            let weight = if hz >= left && hz <= center {
                (hz - left) / (center - left).max(1e-30)
            } else if hz > center && hz <= right {
                (right - hz) / (right - center).max(1e-30)
            } else {
                0.0
            };
            if !weight.is_finite() {
                return Err("non-finite Gemma4 Mel filter".into());
            }
            filters[mel * fft_bins + bin] = weight as f32;
        }
    }
    Ok(filters)
}

fn htk_hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn htk_mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

fn zeroed(len: usize) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| "Gemma4 audio allocation failed")?;
    values.resize(len, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pcm16_wav(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data_len = u32::try_from(samples.len() * 2).unwrap();
        let block_align = channels * 2;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn gemma4_frontend_returns_one_finite_128_bin_frame_for_20ms_stereo() {
        let samples: Vec<i16> = (0..960)
            .flat_map(|i| {
                let sample = (i as f32 * 0.03125).sin() * 16_384.0;
                [sample as i16, sample as i16]
            })
            .collect();
        let mel = gemma4_audio_features(&pcm16_wav(48_000, 2, &samples)).unwrap();
        assert_eq!(mel.frames, 1);
        assert_eq!(mel.values.len(), 128);
        assert!(mel.values.iter().all(|value| value.is_finite()));
    }
}
