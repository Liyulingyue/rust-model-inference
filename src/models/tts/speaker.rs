use crate::models::qwen3a::audio_processor::{decode_pcm16_wav_any, RealFft};

const SAMPLE_RATE: u32 = 24_000;
const FFT_SIZE: usize = 1024;
const HOP: usize = 256;
const REFLECT_PAD: usize = 384;
const MEL_BINS: usize = 128;

pub struct SpeakerMel {
    pub values: Vec<f32>,
    pub frames: usize,
}

pub fn reference_wav_to_mel(bytes: &[u8]) -> Result<SpeakerMel, String> {
    let decoded = decode_pcm16_wav_any(bytes).map_err(|error| format!("{error:?}"))?;
    let channels = usize::from(decoded.channels);
    let mono: Vec<f32> = decoded
        .samples
        .chunks_exact(channels)
        .map(|frame| {
            (frame.iter().map(|value| f64::from(*value)).sum::<f64>() / channels as f64) as f32
        })
        .collect();
    let mono = resample_linear(&mono, decoded.sample_rate, SAMPLE_RATE)?;
    if mono.len() <= REFLECT_PAD {
        return Err(format!(
            "reference audio must contain more than {REFLECT_PAD} samples after resampling"
        ));
    }
    let padded = reflect_pad(&mono, REFLECT_PAD)?;
    let frames = padded
        .len()
        .checked_sub(FFT_SIZE)
        .and_then(|length| length.checked_div(HOP))
        .and_then(|value| value.checked_add(1))
        .ok_or("reference audio is too short for the speaker STFT")?;
    let mut values = vec![
        0.0;
        MEL_BINS
            .checked_mul(frames)
            .ok_or("speaker Mel size overflow")?
    ];
    let hann: Vec<f32> = (0..FFT_SIZE)
        .map(|index| {
            let angle = 2.0 * std::f32::consts::PI * index as f32 / FFT_SIZE as f32;
            0.5 * (1.0 - angle.cos())
        })
        .collect();
    let filters = mel_filters();
    let fft_bins = FFT_SIZE / 2 + 1;
    let mut input = vec![0.0; FFT_SIZE];
    let mut magnitude = vec![0.0; fft_bins];
    let mut fft = RealFft::new(FFT_SIZE).map_err(|error| format!("{error:?}"))?;
    for frame in 0..frames {
        let start = frame
            .checked_mul(HOP)
            .ok_or("speaker STFT offset overflow")?;
        for index in 0..FFT_SIZE {
            input[index] = padded[start + index] * hann[index];
        }
        fft.magnitude(&input, &mut magnitude);
        if magnitude.iter().any(|value| !value.is_finite()) {
            return Err("non-finite speaker FFT output".into());
        }
        for mel in 0..MEL_BINS {
            let filter = &filters[mel * fft_bins..(mel + 1) * fft_bins];
            let sum = magnitude
                .iter()
                .zip(filter)
                .map(|(value, weight)| f64::from(*value * *weight))
                .sum::<f64>();
            let value = sum.max(1e-5).ln() as f32;
            if !value.is_finite() {
                return Err("non-finite speaker Mel value".into());
            }
            values[mel * frames + frame] = value;
        }
    }
    Ok(SpeakerMel { values, frames })
}

fn resample_linear(
    samples: &[f32],
    source_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>, String> {
    if samples.is_empty() || source_rate == 0 || target_rate == 0 {
        return Err("invalid reference audio sample rate or data".into());
    }
    if source_rate == target_rate {
        return Ok(samples.to_vec());
    }
    let output_len = u64::try_from(samples.len())
        .ok()
        .and_then(|length| length.checked_mul(u64::from(target_rate)))
        .and_then(|length| length.checked_add(u64::from(source_rate) / 2))
        .map(|length| length / u64::from(source_rate))
        .and_then(|length| usize::try_from(length).ok())
        .filter(|length| *length > 0)
        .ok_or("resampled reference audio size overflow")?;
    let scale = source_rate as f64 / target_rate as f64;
    let last = samples.len() - 1;
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let position = (index as f64 * scale).min(last as f64);
        let left = position.floor() as usize;
        let right = (left + 1).min(last);
        let fraction = position - left as f64;
        output.push(
            (f64::from(samples[left]) * (1.0 - fraction) + f64::from(samples[right]) * fraction)
                as f32,
        );
    }
    Ok(output)
}

fn reflect_pad(samples: &[f32], pad: usize) -> Result<Vec<f32>, String> {
    if samples.is_empty() || pad >= samples.len() {
        return Err("reflect padding must be smaller than the input".into());
    }
    let mut output = Vec::with_capacity(
        samples
            .len()
            .checked_add(pad * 2)
            .ok_or("reflect padding size overflow")?,
    );
    output.extend((1..=pad).rev().map(|offset| samples[offset]));
    output.extend_from_slice(samples);
    output.extend((1..=pad).map(|offset| samples[samples.len() - 1 - offset]));
    Ok(output)
}

fn slaney_hz_to_mel(hz: f64) -> f64 {
    if hz >= 1_000.0 {
        15.0 + (hz / 1_000.0).ln() / (6.4f64.ln() / 27.0)
    } else {
        hz / (200.0 / 3.0)
    }
}

fn slaney_mel_to_hz(mel: f64) -> f64 {
    if mel >= 15.0 {
        1_000.0 * ((mel - 15.0) * (6.4f64.ln() / 27.0)).exp()
    } else {
        mel * (200.0 / 3.0)
    }
}

fn mel_filters() -> Vec<f32> {
    let fft_bins = FFT_SIZE / 2 + 1;
    let max_mel = slaney_hz_to_mel(SAMPLE_RATE as f64 / 2.0);
    let points: Vec<f64> = (0..MEL_BINS + 2)
        .map(|index| slaney_mel_to_hz(max_mel * index as f64 / (MEL_BINS + 1) as f64))
        .collect();
    let mut filters = vec![0.0; MEL_BINS * fft_bins];
    for mel in 0..MEL_BINS {
        let lower_width = points[mel + 1] - points[mel];
        let upper_width = points[mel + 2] - points[mel + 1];
        let norm = 2.0 / (points[mel + 2] - points[mel]);
        for bin in 0..fft_bins {
            let hz = bin as f64 * SAMPLE_RATE as f64 / FFT_SIZE as f64;
            filters[mel * fft_bins + bin] = (((hz - points[mel]) / lower_width)
                .min((points[mel + 2] - hz) / upper_width)
                .max(0.0)
                * norm) as f32;
        }
    }
    filters
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
    fn reference_frontend_mixes_resamples_and_matches_frame_shape() {
        let stereo: Vec<i16> = (0..960)
            .flat_map(|i| {
                let value = ((i as f32 * 0.04).sin() * 16_000.0) as i16;
                [value, value]
            })
            .collect();
        let wav = pcm16_wav(48_000, 2, &stereo);
        let mel = reference_wav_to_mel(&wav).unwrap();
        assert_eq!(mel.values.len(), 128 * mel.frames);
        assert!(mel.frames > 0);
        assert!(mel.values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn speaker_reflect_padding_matches_pytorch_reflect() {
        assert_eq!(
            reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2).unwrap(),
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
        assert!(reflect_pad(&[1.0, 2.0], 2).is_err());
    }
}
