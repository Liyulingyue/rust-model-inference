//! Kaldi-compatible 80-mel filterbank + LFR stacking.
//!
//! Implements the exact fbank pipeline from the FunASR llama.cpp reference:
//!   - 16 kHz, 25 ms Hamming window, 10 ms shift
//!   - Pre-emphasis 0.97, per-frame DC removal
//!   - 512-point radix-2 FFT → power spectrum
//!   - 80 triangular mel filters (mel = 1127*ln(1+f/700), 20–8000 Hz)
//!   - Log floor = FLT_EPSILON (1.1920929e-7)
//!   - LFR: left-pad 3 copies of frame 0, stack 7 frames with stride 6 → 560-dim

use std::f32::consts::PI;

const FS: usize = 16_000;
const WINLEN: usize = 400;
const SHIFT: usize = 160;
const NFFT: usize = 512;
const NMEL: usize = 80;
const LFR_M: usize = 7;
const LFR_N: usize = 6;
const PREEMPH: f32 = 0.97;
const LOWF: f32 = 20.0;
const HIGHF: f32 = 8000.0;
const FLT_EPS: f32 = 1.1920929e-7;

#[inline]
fn mel_hz(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

/// Radix-2 in-place FFT (iterative, bit-reversed input order).
fn fft(re: &mut [f32], im: &mut [f32], n: usize) {
    let mut j = 0;
    for i in 1..n {
        let mut b = n >> 1;
        while j & b != 0 {
            j ^= b;
            b >>= 1;
        }
        j ^= b;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let a = -2.0 * PI / len as f32;
        let (wr, wi) = (a.cos(), a.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let ur = re[i + k];
                let ui = im[i + k];
                let vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                let vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let n2 = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = n2;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Compute 80-mel fbank features with LFR stacking.
///
/// Input: `wav` = 16 kHz mono f32 samples (any length).
/// Output: `[T_lfr * 560]` row-major f32, plus `T_lfr` (number of output frames).
pub fn compute_fbank(wav: &[f32]) -> (Vec<f32>, usize) {
    if wav.len() < WINLEN {
        return (Vec::new(), 0);
    }
    let mut samples: Vec<f32> = wav.to_vec();
    for v in &mut samples {
        *v *= 32768.0;
    }
    let n = samples.len();
    let t_frames = (n - WINLEN) / SHIFT + 1;
    let nbin = NFFT / 2 + 1;
    let bw = FS as f32 / NFFT as f32;
    let ml = mel_hz(LOWF);
    let mh = mel_hz(HIGHF);
    let dm = (mh - ml) / (NMEL + 1) as f32;
    let mut fb = vec![vec![0.0f32; nbin]; NMEL];
    for m in 0..NMEL {
        let l = ml + m as f32 * dm;
        let c = ml + (m + 1) as f32 * dm;
        let r = ml + (m + 2) as f32 * dm;
        for k in 0..nbin {
            let mf = mel_hz(bw * k as f32);
            if mf > l && mf < r {
                fb[m][k] = if mf <= c {
                    (mf - l) / (c - l)
                } else {
                    (r - mf) / (r - c)
                };
            }
        }
    }
    let mut feat = vec![0.0f32; t_frames * NMEL];
    let mut re = vec![0.0f32; NFFT];
    let mut im = vec![0.0f32; NFFT];
    let mut frame = vec![0.0f32; WINLEN];
    let mut win = vec![0.0f32; WINLEN];
    for i in 0..WINLEN {
        // Compute the angle in f64 first so the ratio `i / (WINLEN - 1)`
        // is not truncated to f32 precision before the multiply by 2*PI.
        // The original all-f32 form
        //   `(2.0 * PI * i as f32 / (WINLEN - 1) as f32).cos()`
        // rounds the intermediate division at ~7 decimal digits and can
        // drift in the last few bits of the Hann window; doing the
        // division in f64 and casting back to f32 before `cos` matches
        // llama.cpp's reference computation. The window is built once
        // at startup so the f64 path is not a hot-path concern.
        let angle = (2.0f64 * std::f64::consts::PI * i as f64 / (WINLEN - 1) as f64) as f32;
        win[i] = 0.54 - 0.46 * angle.cos();
    }
    for t in 0..t_frames {
        let s = &samples[t * SHIFT..];
        let mut mn = 0.0f64;
        for i in 0..WINLEN {
            mn += s[i] as f64;
        }
        mn /= WINLEN as f64;
        for i in 0..WINLEN {
            frame[i] = s[i] - mn as f32;
        }
        for i in (1..WINLEN).rev() {
            frame[i] -= PREEMPH * frame[i - 1];
        }
        frame[0] -= PREEMPH * frame[0];
        for i in 0..NFFT {
            re[i] = if i < WINLEN { frame[i] * win[i] } else { 0.0 };
            im[i] = 0.0;
        }
        fft(&mut re, &mut im, NFFT);
        for m in 0..NMEL {
            let mut e = 0.0f32;
            for k in 0..nbin {
                if fb[m][k] > 0.0 {
                    e += fb[m][k] * (re[k] * re[k] + im[k] * im[k]);
                }
            }
            feat[t * NMEL + m] = if e > FLT_EPS { e } else { FLT_EPS }.ln();
        }
    }
    let pad = (LFR_M - 1) / 2;
    let t_lfr = (t_frames + LFR_N - 1) / LFR_N;
    let mut padded: Vec<Vec<f32>> = Vec::with_capacity(t_frames + pad + LFR_M);
    for _ in 0..pad {
        padded.push(feat[0..NMEL].to_vec());
    }
    for t in 0..t_frames {
        padded.push(feat[t * NMEL..(t + 1) * NMEL].to_vec());
    }
    while padded.len() < (t_lfr - 1) * LFR_N + LFR_M {
        padded.push(feat[(t_frames - 1) * NMEL..t_frames * NMEL].to_vec());
    }
    let d = LFR_M * NMEL;
    let mut out = vec![0.0f32; t_lfr * d];
    for i in 0..t_lfr {
        for j in 0..LFR_M {
            let src = &padded[i * LFR_N + j];
            let dst = &mut out[i * d + j * NMEL..i * d + (j + 1) * NMEL];
            dst.copy_from_slice(&src[..NMEL]);
        }
    }
    (out, t_lfr)
}

#[cfg(test)]
mod parity_tests {
    use super::compute_fbank;

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn fbank_matches_scalar_oracle_for_pcm16_audio() {
        let samples = (0..800)
            .map(|i| (((i * 37) % 101) - 50) as f32 * 300.0 / 32768.0)
            .collect::<Vec<_>>();
        let (features, frames) = compute_fbank(&samples);
        assert_eq!(frames, 1);
        assert_eq!(features.len(), 560);
        assert_eq!(
            features[..16]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [
                0x415f_d9a5,
                0x416d_e1e7,
                0x4165_c66c,
                0x416f_a58b,
                0x417a_620a,
                0x417c_9cec,
                0x414d_74c1,
                0x4173_dd61,
                0x415f_e1d1,
                0x4179_da07,
                0x4187_5e73,
                0x417b_105c,
                0x4176_6c9d,
                0x4182_5a76,
                0x4196_2d7a,
                0x419b_df64,
            ]
        );
    }
}

/// Low-frame-rate truncation: compute the number of audio tokens from fbank length.
///
/// Three stages of stride-2 downsampling:
///   ol = 1 + (T - 3 + 2) / 2  = (T + 1) / 2
///   ol = 1 + (ol - 3 + 2) / 2 = (ol + 1) / 2
///   n_aud = (ol - 1) / 2 + 1  = (ol + 1) / 2
pub fn lfr_token_count(t_fbank: usize) -> usize {
    let ol = 1 + (t_fbank as i32 - 3 + 2) / 2;
    let ol = 1 + (ol - 3 + 2) / 2;
    ((ol - 1) / 2 + 1) as usize
}

/// Sinusoidal position encoding added in-place to fbank features.
///
/// - depth = input feature dim (560)
/// - positions start at 1 (not 0)
/// - inc = log(10000) / (depth/2 - 1)
pub fn add_position_encoding(x: &mut [f32], t: usize, depth: usize) {
    let inc = (10000.0f64).ln() / (depth as f64 / 2.0 - 1.0);
    for t_idx in 0..t {
        let pos = (t_idx + 1) as f64;
        for i in 0..depth / 2 {
            let its = (i as f64 * -inc).exp();
            let st = pos * its;
            x[t_idx * depth + i] += st.sin() as f32;
            x[t_idx * depth + depth / 2 + i] += st.cos() as f32;
        }
    }
}
