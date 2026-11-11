use crate::ops::audio::frontend::RealFft;

pub(super) fn parameters(rate: u32) -> Result<(usize, usize), String> {
    if !(8_000..=48_000).contains(&rate) {
        return Err("RE-USE supports sample rates from 8000 to 48000 Hz".into());
    }
    let even = |base| {
        let n = base * rate as usize / 8000;
        n + n % 2
    };
    Ok((even(320), even(40)))
}

pub(super) fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let angle = i as f32 * (std::f64::consts::TAU / n as f64) as f32;
            // Phase inputs retain the sign of tiny FFT residuals at the branch cut.
            0.5 - 0.5 * crate::ops::rope::rope_sin_cos_sleef(angle).0
        })
        .collect()
}

pub(super) fn stft(
    samples: &[f32],
    rate: u32,
) -> Result<(Vec<f32>, Vec<f32>, usize, usize), String> {
    let (n, hop) = parameters(rate)?;
    let (t, f) = (samples.len() / hop + 1, n / 2 + 1);
    let window = hann(n);
    let mut fft = RealFft::new(n).map_err(|e| format!("RE-USE FFT: {e:?}"))?;
    let mut frame = vec![0.0; n];
    let mut bins = vec![[0.0; 2]; f];
    let mut mag = vec![0.0; t * f];
    let mut phase = vec![0.0; t * f];
    for ti in 0..t {
        for i in 0..n {
            let offset = (ti * hop + i) as isize - (n / 2) as isize;
            let index = if offset < 0 {
                -offset
            } else if offset >= samples.len() as isize {
                2 * samples.len() as isize - 2 - offset
            } else {
                offset
            } as usize;
            frame[i] = samples[index] * window[i];
        }
        fft.complex(&frame, &mut bins);
        for (fi, &[re, im]) in bins.iter().enumerate() {
            mag[ti * f + fi] = re.hypot(im).ln_1p();
            phase[ti * f + fi] = im.atan2(re);
        }
    }
    Ok((mag, phase, t, f))
}

pub(super) fn istft(
    mag: &[f32],
    phase: &[f32],
    t: usize,
    f: usize,
    length: usize,
    rate: u32,
) -> Result<Vec<f32>, String> {
    let (n, hop) = parameters(rate)?;
    let window = hann(n);
    let mut fft = RealFft::new(n).map_err(|e| format!("RE-USE FFT: {e:?}"))?;
    let mut frame = vec![0.0; n];
    let mut bins = vec![[0.0; 2]; f];
    let mut output = vec![0.0; (t - 1) * hop + n];
    let mut norm = vec![0.0; output.len()];
    for ti in 0..t {
        let row = &mag[ti * f..(ti + 1) * f];
        for fi in 0..f {
            let amplitude = row[fi].max(0.0).exp_m1();
            let (sin, cos) = phase[ti * f + fi].sin_cos();
            bins[fi] = [amplitude * cos, amplitude * sin];
        }
        fft.inverse(&bins, &mut frame);
        for i in 0..n {
            output[ti * hop + i] += frame[i] * window[i];
            norm[ti * hop + i] += window[i] * window[i];
        }
    }
    let available = (t - 1) * hop;
    let mut samples = vec![1e-8; length];
    for i in 0..length.min(available) {
        samples[i] = output[i + n / 2] / norm[i + n / 2];
    }
    if samples.iter().any(|v| !v.is_finite()) {
        return Err("RE-USE produced non-finite waveform".into());
    }
    Ok(samples)
}
