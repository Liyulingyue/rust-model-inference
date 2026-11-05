//! Voxtral Realtime's 16 kHz, 128-bin streaming Mel frontend.

use crate::models::qwen3::asr::audio_processor::{periodic_hann_window, reflect_pad};

const FFT_SIZE: usize = 400;
const HOP: usize = 160;
const MEL_BINS: usize = 128;
const SAMPLE_RATE: usize = 16_000;

pub struct LogMel {
    pub normalized: Vec<f32>,
    pub frames: usize,
}

fn slaney_mel_hz(mel: f64) -> f64 {
    let min_log_mel = 1_000.0 / (200.0 / 3.0);
    if mel >= min_log_mel {
        1_000.0 * ((mel - min_log_mel) * (6.4f64.ln() / 27.0)).exp()
    } else {
        mel * (200.0 / 3.0)
    }
}

fn mel_filters() -> Vec<f32> {
    let fft_bins = FFT_SIZE / 2 + 1;
    let mut filters = vec![0.0; MEL_BINS * fft_bins];
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
            filters[mel * fft_bins + bin] = (((hz - mel_hz[mel]) / lower_width)
                .min((mel_hz[mel + 2] - hz) / upper_width)
                .max(0.0)
                * norm) as f32;
        }
    }
    filters
}

#[inline]
fn fft_mul_add(a: f32, b: f32, c: f32) -> f32 {
    if crate::ops::scalar_mode() {
        std::hint::black_box(a * b) + c
    } else {
        a.mul_add(b, c)
    }
}

struct RealFft {
    sin: Vec<f32>,
    cos: Vec<f32>,
    input: Vec<f32>,
    output: Vec<f32>,
}

impl RealFft {
    fn new() -> Self {
        let (sin, cos): (Vec<f32>, Vec<f32>) = (0..FFT_SIZE)
            .map(|index| {
                let angle = (2.0 * std::f64::consts::PI * index as f64 / FFT_SIZE as f64) as f32;
                angle.sin_cos()
            })
            .unzip();
        Self {
            sin,
            cos,
            input: vec![0.0; FFT_SIZE * 2],
            output: vec![0.0; FFT_SIZE * 8],
        }
    }

    fn power(&mut self, input: &[f32], output: &mut [f32]) {
        self.input[..FFT_SIZE].copy_from_slice(input);
        fft_real(
            &self.sin,
            &self.cos,
            &mut self.input,
            0,
            FFT_SIZE,
            FFT_SIZE,
            &mut self.output,
            0,
        );
        for (bin, value) in output.iter_mut().enumerate() {
            let real = self.output[bin * 2];
            let imaginary = self.output[bin * 2 + 1];
            *value = fft_mul_add(real, real, imaginary * imaginary);
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
    root_size: usize,
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
        let step = root_size / n;
        for k in 0..n {
            let mut real = 0.0f32;
            let mut imaginary = 0.0f32;
            for index in 0..n {
                let table = (k * index * step) % root_size;
                let value = input[input_offset + index];
                real = fft_mul_add(value, cos[table], real);
                imaginary = fft_mul_add(-value, sin[table], imaginary);
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
    fft_real(sin, cos, input, scratch, half, root_size, output, even);
    for index in 0..half {
        input[scratch + index] = input[input_offset + index * 2 + 1];
    }
    let odd = even + n;
    fft_real(sin, cos, input, scratch, half, root_size, output, odd);

    let step = root_size / n;
    for k in 0..half {
        let real = cos[k * step];
        let sine = sin[k * step];
        let odd_real = output[odd + k * 2];
        let odd_imaginary = output[odd + k * 2 + 1];
        let even_real = output[even + k * 2];
        let even_imaginary = output[even + k * 2 + 1];
        output[output_offset + k * 2] =
            fft_mul_add(sine, odd_imaginary, fft_mul_add(real, odd_real, even_real));
        output[output_offset + k * 2 + 1] = fft_mul_add(
            -sine,
            odd_real,
            fft_mul_add(real, odd_imaginary, even_imaginary),
        );
        output[output_offset + (k + half) * 2] = fft_mul_add(
            -sine,
            odd_imaginary,
            fft_mul_add(-real, odd_real, even_real),
        );
        output[output_offset + (k + half) * 2 + 1] = fft_mul_add(
            sine,
            odd_real,
            fft_mul_add(-real, odd_imaginary, even_imaginary),
        );
    }
}

pub fn compute_voxtral_log_mel(samples: &[f32]) -> Result<LogMel, String> {
    if samples.is_empty() || samples.iter().any(|value| !value.is_finite()) {
        return Err("audio samples must be non-empty and finite".into());
    }
    let padded = reflect_pad(samples).map_err(|error| format!("{error:?}"))?;
    let frames = samples.len() / HOP;
    if frames == 0 {
        return Err("audio window is too short for a Mel frame".into());
    }
    let mut raw = vec![0.0; MEL_BINS * frames];
    let filters = mel_filters();
    let fft_bins = FFT_SIZE / 2 + 1;
    let mut frame = vec![0.0; FFT_SIZE];
    let hann = periodic_hann_window();
    let mut fft = RealFft::new();
    let mut power = vec![0.0; fft_bins];
    for frame_index in 0..frames {
        let start = frame_index * HOP;
        let input = padded
            .get(start..start + FFT_SIZE)
            .ok_or("truncated padded frame")?;
        for i in 0..FFT_SIZE {
            frame[i] = input[i] * hann[i];
        }
        fft.power(&frame, &mut power);
        if power.iter().any(|value| !value.is_finite()) {
            return Err("non-finite FFT output".into());
        }
        for mel in 0..MEL_BINS {
            let filter = &filters[mel * fft_bins..(mel + 1) * fft_bins];
            let mut sum = 0.0f32;
            for bin in 0..fft_bins - 1 {
                sum += power[bin] * filter[bin];
            }
            raw[mel * frames + frame_index] = sum.max(1e-10).log10();
        }
    }
    let normalized: Vec<f32> = raw
        .iter()
        .map(|value| (value.max(-6.5) + 4.0) / 4.0)
        .collect();
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
    Ok(LogMel { normalized, frames })
}

#[cfg(test)]
mod tests {
    #[test]
    fn voxtral_mel_omits_last_centered_frame() {
        let mel = super::compute_voxtral_log_mel(&vec![0.0; 2_160]).unwrap();
        assert_eq!(mel.frames, 13);
        assert_eq!(mel.normalized.len(), 128 * 13);
    }
}
