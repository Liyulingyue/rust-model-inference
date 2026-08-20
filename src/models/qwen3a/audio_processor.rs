// Qwen3-Audio audio pre-processing: WAV decoding + log-mel feature extraction.
//
// Phase 4b split from qwen3a.rs. Owns: AsrAudioError, decode_pcm16_wav, MelWindow,
// LogMel, AudioFft, compute_log_mel, split_mel_windows, log_mel_windows.

use crate::core::tensor::{GGMLType, MetaValue, TensorSource};

pub(crate) const SAMPLE_RATE: usize = 16_000;
pub(crate) const FFT_SIZE: usize = 400;
pub(crate) const HOP: usize = 160;
pub(crate) const MEL_BINS: usize = 128;
pub(crate) const WINDOW_FRAMES: usize = 800;
pub(crate) const CHUNK_FRAMES: usize = 100;

unsafe extern "C" {
    fn cosf(value: f32) -> f32;
    fn erff(value: f32) -> f32;
    fn log10(value: f64) -> f64;
    fn sinf(value: f32) -> f32;
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vDSP_sve(input: *const f32, stride: isize, sum: *mut f32, count: usize);
    fn vDSP_vsadd(
        input: *const f32,
        input_stride: isize,
        scalar: *const f32,
        output: *mut f32,
        output_stride: isize,
        count: usize,
    );
    fn vDSP_measqv(input: *const f32, stride: isize, result: *mut f32, count: usize);
    fn vDSP_vsmul(
        input: *const f32,
        input_stride: isize,
        scalar: *const f32,
        output: *mut f32,
        output_stride: isize,
        count: usize,
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsrAudioError {
    Unsupported(String),
    Invalid(String),
}

fn wav_u16(bytes: &[u8], offset: usize) -> Result<u16, AsrAudioError> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| AsrAudioError::Invalid("WAV offset overflow".into()))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| AsrAudioError::Invalid("truncated WAV field".into()))?;
    Ok(u16::from_le_bytes(value.try_into().unwrap()))
}

fn wav_u32(bytes: &[u8], offset: usize) -> Result<u32, AsrAudioError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| AsrAudioError::Invalid("WAV offset overflow".into()))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| AsrAudioError::Invalid("truncated WAV field".into()))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

pub fn decode_pcm16_wav(bytes: &[u8]) -> Result<Vec<f32>, AsrAudioError> {
    if bytes.get(0..4) != Some(b"RIFF") {
        return Err(AsrAudioError::Unsupported("expected RIFF/WAVE".into()));
    }
    let wave = bytes
        .get(8..12)
        .ok_or_else(|| AsrAudioError::Invalid("truncated RIFF/WAVE header".into()))?;
    if wave != b"WAVE" {
        return Err(AsrAudioError::Unsupported("expected RIFF/WAVE".into()));
    }
    let riff_end = 8usize
        .checked_add(wav_u32(bytes, 4)? as usize)
        .ok_or_else(|| AsrAudioError::Invalid("RIFF size overflow".into()))?;
    if riff_end > bytes.len() {
        return Err(AsrAudioError::Invalid("truncated RIFF".into()));
    }

    let mut format = None;
    let mut pcm = None;
    let mut offset = 12usize;
    while offset < riff_end {
        let id_end = offset
            .checked_add(4)
            .ok_or_else(|| AsrAudioError::Invalid("chunk offset overflow".into()))?;
        let header_end = offset
            .checked_add(8)
            .ok_or_else(|| AsrAudioError::Invalid("chunk header overflow".into()))?;
        let id = bytes
            .get(offset..id_end)
            .ok_or_else(|| AsrAudioError::Invalid("truncated chunk header".into()))?;
        let len = wav_u32(bytes, id_end)? as usize;
        let data_end = header_end
            .checked_add(len)
            .ok_or_else(|| AsrAudioError::Invalid("chunk size overflow".into()))?;
        let padded_end = data_end
            .checked_add(len & 1)
            .ok_or_else(|| AsrAudioError::Invalid("chunk padding overflow".into()))?;
        if padded_end > riff_end {
            return Err(AsrAudioError::Invalid(
                "truncated chunk data or padding".into(),
            ));
        }
        let chunk = bytes
            .get(header_end..data_end)
            .ok_or_else(|| AsrAudioError::Invalid("truncated chunk data".into()))?;
        match id {
            b"fmt " => {
                if format.replace(chunk).is_some() {
                    return Err(AsrAudioError::Invalid("duplicate fmt chunk".into()));
                }
            }
            b"data" => {
                if pcm.replace(chunk).is_some() {
                    return Err(AsrAudioError::Invalid("duplicate data chunk".into()));
                }
            }
            _ => {}
        }
        offset = padded_end;
    }

    let format = format.ok_or_else(|| AsrAudioError::Invalid("missing fmt chunk".into()))?;
    let expected = [
        (wav_u16(format, 0)? as u32, 1, "PCM format"),
        (wav_u16(format, 2)? as u32, 1, "mono audio"),
        (wav_u32(format, 4)?, 16_000, "16000 Hz sample rate"),
        (wav_u32(format, 8)?, 32_000, "32000 byte rate"),
        (wav_u16(format, 12)? as u32, 2, "2-byte block align"),
        (wav_u16(format, 14)? as u32, 16, "16-bit samples"),
    ];
    if let Some((_, _, contract)) = expected
        .iter()
        .find(|(actual, expected, _)| actual != expected)
    {
        return Err(AsrAudioError::Unsupported(format!("expected {contract}")));
    }

    let pcm = pcm.ok_or_else(|| AsrAudioError::Invalid("missing data chunk".into()))?;
    if pcm.is_empty() || pcm.len() & 1 != 0 {
        return Err(AsrAudioError::Invalid(
            "PCM data must contain complete samples".into(),
        ));
    }
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(pcm.len() / 2)
        .map_err(|_| AsrAudioError::Invalid("PCM allocation failed".into()))?;
    for bytes in pcm.chunks_exact(2) {
        let sample = i16::from_le_bytes(bytes.try_into().unwrap()) as f32 / 32768.0;
        if !sample.is_finite() {
            return Err(AsrAudioError::Invalid("non-finite PCM sample".into()));
        }
        samples.push(sample);
    }

    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(
        "asr.pcm",
        None,
        &[samples.len()],
        &samples,
    ));
    Ok(samples)
}

pub(crate) struct MelWindow {
    pub values: Vec<f32>,
    pub frames: usize,
    pub valid_frames: usize,
}

pub(crate) struct LogMel {
    pub(crate) raw: Vec<f32>,
    pub(crate) normalized: Vec<f32>,
    pub(crate) frames: usize,
}

fn zeroed_f32(len: usize) -> Result<Vec<f32>, AsrAudioError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| AsrAudioError::Invalid("audio allocation failed".into()))?;
    values.resize(len, 0.0);
    Ok(values)
}

pub(crate) fn reflect_pad(samples: &[f32]) -> Result<Vec<f32>, AsrAudioError> {
    let padded_len = samples
        .len()
        .checked_add(FFT_SIZE)
        .ok_or_else(|| AsrAudioError::Invalid("padded audio size overflow".into()))?;
    let mut padded = zeroed_f32(padded_len)?;
    let center_end = (FFT_SIZE / 2)
        .checked_add(samples.len())
        .ok_or_else(|| AsrAudioError::Invalid("padded audio range overflow".into()))?;
    padded[FFT_SIZE / 2..center_end].copy_from_slice(samples);
    for i in 0..FFT_SIZE / 2 {
        if FFT_SIZE / 2 - i < samples.len() {
            padded[i] = samples[FFT_SIZE / 2 - i];
        }
        if let Some(source) = samples.len().checked_sub(2 + i) {
            padded[center_end + i] = samples[source];
        }
    }
    Ok(padded)
}

fn slaney_mel_hz(mel: f64) -> f64 {
    let min_log_hz = 1_000.0;
    let min_log_mel = min_log_hz / (200.0 / 3.0);
    if mel >= min_log_mel {
        min_log_hz * ((mel - min_log_mel) * (6.4f64.ln() / 27.0)).exp()
    } else {
        mel * (200.0 / 3.0)
    }
}

fn mel_filters() -> Result<Vec<f32>, AsrAudioError> {
    let fft_bins = FFT_SIZE / 2 + 1;
    let filter_len = MEL_BINS
        .checked_mul(fft_bins)
        .ok_or_else(|| AsrAudioError::Invalid("Mel filter size overflow".into()))?;
    let mut filters = zeroed_f32(filter_len)?;
    let max_mel = 15.0 + (8.0f64).ln() / (6.4f64.ln() / 27.0);
    let mel_hz: Vec<f64> = (0..MEL_BINS + 2)
        .map(|i| slaney_mel_hz(max_mel * i as f64 / (MEL_BINS + 1) as f64))
        .collect();
    for mel in 0..MEL_BINS {
        let lower_width = mel_hz[mel + 1] - mel_hz[mel];
        let upper_width = mel_hz[mel + 2] - mel_hz[mel + 1];
        let norm = 2.0 / (mel_hz[mel + 2] - mel_hz[mel]);
        for bin in 0..fft_bins {
            let hz = bin as f64 * SAMPLE_RATE as f64 / FFT_SIZE as f64;
            let weight = ((hz - mel_hz[mel]) / lower_width)
                .min((mel_hz[mel + 2] - hz) / upper_width)
                .max(0.0)
                * norm;
            if !weight.is_finite() {
                return Err(AsrAudioError::Invalid("non-finite Mel filter".into()));
            }
            filters[mel * fft_bins + bin] = weight as f32;
        }
    }
    Ok(filters)
}

pub(crate) fn periodic_hann_window() -> Vec<f32> {
    (0..FFT_SIZE)
        .map(|i| {
            let angle = (2.0 * std::f64::consts::PI * i as f64 / FFT_SIZE as f64) as f32;
            (0.5 * (1.0 - f64::from(unsafe { cosf(angle) }))) as f32
        })
        .collect()
}

struct AudioFft {
    sin: Vec<f32>,
    cos: Vec<f32>,
    input: Vec<f32>,
    output: Vec<f32>,
}

impl AudioFft {
    fn new() -> Self {
        let mut sin = Vec::with_capacity(FFT_SIZE);
        let mut cos = Vec::with_capacity(FFT_SIZE);
        for index in 0..FFT_SIZE {
            let angle = (2.0 * std::f64::consts::PI * index as f64 / FFT_SIZE as f64) as f32;
            sin.push(unsafe { sinf(angle) });
            cos.push(unsafe { cosf(angle) });
        }
        Self {
            sin,
            cos,
            input: vec![0.0; FFT_SIZE * 2],
            output: vec![0.0; FFT_SIZE * 8],
        }
    }

    fn transform(&mut self, input: &[f32]) {
        self.input[..FFT_SIZE].copy_from_slice(input);
        fft_real(
            &self.sin,
            &self.cos,
            &mut self.input,
            0,
            FFT_SIZE,
            &mut self.output,
            0,
        );
    }

    fn power(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), FFT_SIZE);
        debug_assert_eq!(output.len(), FFT_SIZE / 2 + 1);
        self.transform(input);
        for (bin, value) in output.iter_mut().enumerate() {
            let real = self.output[bin * 2];
            let imaginary = self.output[bin * 2 + 1];
            *value = real.mul_add(real, imaginary * imaginary);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fft_real(
    sin: &[f32],
    cos: &[f32],
    input: &mut [f32],
    input_offset: usize,
    n: usize,
    output: &mut [f32],
    output_offset: usize,
) {
    if n == 1 {
        output[output_offset] = input[input_offset];
        output[output_offset + 1] = 0.0;
        return;
    }
    let half = n / 2;
    if n % 2 != 0 {
        let step = FFT_SIZE / n;
        for k in 0..n {
            let mut real = 0.0f32;
            let mut imaginary = 0.0f32;
            for index in 0..n {
                let table = (k * index * step) % FFT_SIZE;
                let value = input[input_offset + index];
                real = value.mul_add(cos[table], real);
                imaginary = (-value).mul_add(sin[table], imaginary);
            }
            output[output_offset + k * 2] = real;
            output[output_offset + k * 2 + 1] = imaginary;
        }
        return;
    }

    let scratch = input_offset + n;
    for index in 0..half {
        input[scratch + index] = input[input_offset + index * 2];
    }
    let even = output_offset + n * 2;
    fft_real(sin, cos, input, scratch, half, output, even);
    for index in 0..half {
        input[scratch + index] = input[input_offset + index * 2 + 1];
    }
    let odd = even + n;
    fft_real(sin, cos, input, scratch, half, output, odd);

    let step = FFT_SIZE / n;
    for k in 0..half {
        let real = cos[k * step];
        let sine = sin[k * step];
        let odd_real = output[odd + k * 2];
        let odd_imaginary = output[odd + k * 2 + 1];
        let even_real = output[even + k * 2];
        let even_imaginary = output[even + k * 2 + 1];
        output[output_offset + k * 2] =
            sine.mul_add(odd_imaginary, real.mul_add(odd_real, even_real));
        output[output_offset + k * 2 + 1] =
            (-sine).mul_add(odd_real, real.mul_add(odd_imaginary, even_imaginary));
        output[output_offset + (k + half) * 2] =
            (-sine).mul_add(odd_imaginary, (-real).mul_add(odd_real, even_real));
        output[output_offset + (k + half) * 2 + 1] =
            sine.mul_add(odd_real, (-real).mul_add(odd_imaginary, even_imaginary));
    }
}

pub(crate) fn compute_log_mel(samples: &[f32]) -> Result<LogMel, AsrAudioError> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err(AsrAudioError::Invalid(
            "audio samples must be non-empty and finite".into(),
        ));
    }
    let padded = reflect_pad(samples)?;
    let stft_frames = padded
        .len()
        .checked_sub(FFT_SIZE)
        .and_then(|length| length.checked_div(HOP))
        .and_then(|frames| frames.checked_add(1))
        .ok_or_else(|| AsrAudioError::Invalid("STFT frame count overflow".into()))?;
    let sample_frames = samples
        .len()
        .checked_div(HOP)
        .and_then(|frames| frames.checked_add(1))
        .ok_or_else(|| AsrAudioError::Invalid("effective frame count overflow".into()))?;
    let frames = stft_frames.min(sample_frames);
    let raw_len = MEL_BINS
        .checked_mul(frames)
        .ok_or_else(|| AsrAudioError::Invalid("log-Mel size overflow".into()))?;
    let mut raw = zeroed_f32(raw_len)?;
    let filters = mel_filters()?;
    let fft_bins = FFT_SIZE / 2 + 1;
    let mut frame = zeroed_f32(FFT_SIZE)?;
    let hann = periodic_hann_window();
    let mut fft = AudioFft::new();
    let mut power = zeroed_f32(fft_bins)?;

    for frame_index in 0..frames {
        let start = frame_index
            .checked_mul(HOP)
            .ok_or_else(|| AsrAudioError::Invalid("STFT offset overflow".into()))?;
        let end = start
            .checked_add(FFT_SIZE)
            .ok_or_else(|| AsrAudioError::Invalid("STFT range overflow".into()))?;
        let input = padded
            .get(start..end)
            .ok_or_else(|| AsrAudioError::Invalid("truncated padded frame".into()))?;
        for i in 0..FFT_SIZE {
            frame[i] = input[i] * hann[i];
        }
        fft.power(&frame, &mut power);
        if power.iter().any(|value| !value.is_finite()) {
            return Err(AsrAudioError::Invalid("non-finite FFT output".into()));
        }
        for mel in 0..MEL_BINS {
            let filter = &filters[mel * fft_bins..(mel + 1) * fft_bins];
            let mut power_sum = 0.0f64;
            let mut bin = 0;
            while bin + 3 < fft_bins {
                let sum = power[bin + 1] * filter[bin + 1];
                let sum = power[bin].mul_add(filter[bin], sum);
                let sum = power[bin + 2].mul_add(filter[bin + 2], sum);
                let sum = power[bin + 3].mul_add(filter[bin + 3], sum);
                power_sum += f64::from(sum);
                bin += 4;
            }
            while bin < fft_bins {
                power_sum += f64::from(power[bin] * filter[bin]);
                bin += 1;
            }
            let value = unsafe { log10(power_sum.max(f64::from(5.960464477539063e-8f32))) } as f32;
            if !value.is_finite() {
                return Err(AsrAudioError::Invalid("non-finite log-Mel value".into()));
            }
            raw[mel * frames + frame_index] = value;
        }
    }

    let global_max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !global_max.is_finite() {
        return Err(AsrAudioError::Invalid("non-finite log-Mel maximum".into()));
    }
    let threshold = f64::from(global_max) - 8.0;
    let mut normalized = zeroed_f32(raw_len)?;
    for (normalized, value) in normalized.iter_mut().zip(&raw) {
        let value = if f64::from(*value) < threshold {
            threshold as f32
        } else {
            *value
        };
        *normalized = ((f64::from(value) + 4.0) / 4.0) as f32;
    }
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err(AsrAudioError::Invalid(
            "non-finite normalized Mel value".into(),
        ));
    }

    #[cfg(feature = "parity-trace")]
    {
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "asr.raw_log_mel",
            None,
            &[MEL_BINS, frames],
            &raw,
        ));
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "asr.normalized_mel",
            None,
            &[MEL_BINS, frames],
            &normalized,
        ));
    }
    Ok(LogMel {
        raw,
        normalized,
        frames,
    })
}

pub(crate) fn split_mel_windows(normalized: &[f32], frames: usize) -> Result<Vec<MelWindow>, AsrAudioError> {
    let expected_len = MEL_BINS
        .checked_mul(frames)
        .ok_or_else(|| AsrAudioError::Invalid("normalized Mel size overflow".into()))?;
    if frames == 0
        || normalized.len() != expected_len
        || normalized.iter().any(|value| !value.is_finite())
    {
        return Err(AsrAudioError::Invalid(
            "invalid normalized Mel layout".into(),
        ));
    }

    let window_count = frames
        .checked_add(WINDOW_FRAMES - 1)
        .and_then(|value| value.checked_div(WINDOW_FRAMES))
        .ok_or_else(|| AsrAudioError::Invalid("Mel window count overflow".into()))?;
    let mut windows = Vec::new();
    windows
        .try_reserve_exact(window_count)
        .map_err(|_| AsrAudioError::Invalid("Mel window allocation failed".into()))?;
    let mut start = 0usize;
    while start < frames {
        let valid_frames = (frames - start).min(WINDOW_FRAMES);
        let padded_frames = valid_frames
            .checked_add(CHUNK_FRAMES - 1)
            .and_then(|value| value.checked_div(CHUNK_FRAMES))
            .and_then(|value| value.checked_mul(CHUNK_FRAMES))
            .ok_or_else(|| AsrAudioError::Invalid("padded Mel frame count overflow".into()))?;
        let values_len = MEL_BINS
            .checked_mul(padded_frames)
            .ok_or_else(|| AsrAudioError::Invalid("padded Mel size overflow".into()))?;
        let mut values = zeroed_f32(values_len)?;
        for mel in 0..MEL_BINS {
            let source_start = mel
                .checked_mul(frames)
                .and_then(|offset| offset.checked_add(start))
                .ok_or_else(|| AsrAudioError::Invalid("Mel source offset overflow".into()))?;
            let source_end = source_start
                .checked_add(valid_frames)
                .ok_or_else(|| AsrAudioError::Invalid("Mel source range overflow".into()))?;
            let destination_start = mel
                .checked_mul(padded_frames)
                .ok_or_else(|| AsrAudioError::Invalid("Mel destination offset overflow".into()))?;
            let destination_end = destination_start
                .checked_add(valid_frames)
                .ok_or_else(|| AsrAudioError::Invalid("Mel destination range overflow".into()))?;
            values[destination_start..destination_end]
                .copy_from_slice(&normalized[source_start..source_end]);
        }

        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "asr.padded_mel",
            None,
            &[MEL_BINS, padded_frames],
            &values,
        ));
        windows.push(MelWindow {
            values,
            frames: padded_frames,
            valid_frames,
        });
        start = start
            .checked_add(valid_frames)
            .ok_or_else(|| AsrAudioError::Invalid("Mel window offset overflow".into()))?;
    }
    Ok(windows)
}

pub(crate) fn log_mel_windows(samples: &[f32]) -> Result<Vec<MelWindow>, AsrAudioError> {
    let log_mel = compute_log_mel(samples)?;
    split_mel_windows(&log_mel.normalized, log_mel.frames)
}