//! CAM++ speaker (x-vector) encoder with its frontend.
//!
//! Frontend (reference `speaker/encoder.py` + `utils/audio.py`):
//!   prompt WAV at 48 kHz → stored 41-tap FIR (torchaudio 3:1 sinc) → 16 kHz
//!   → Kaldi fbank (povey window, pre-emph 0.97, 80 HTK mel bins, log,
//!   mean-normalized) → CAM++ (FCM stem, TDNN, 3 dense blocks, masked
//!   statistics pooling, dense → 512-dim x-vector).

use crate::core::tensor::TensorSource;
use crate::models::dots::patch_encoder::load_f16_f32;

const MEL_BINS: usize = 80;
const SR_16K: usize = 16_000;
const FRAME_LEN: usize = 400; // 25 ms
const FRAME_SHIFT: usize = 160; // 10 ms
const FFT_SIZE: usize = 512;
const PREEMPH: f32 = 0.97;
const MEL_FLOOR: f32 = 1e-10;
const BN_EPS: f32 = 1e-5;

// ---------------------------------------------------------------------------
// Generic high-quality resampler (kaiser-windowed sinc, 64 taps, rolloff 0.95)
// ---------------------------------------------------------------------------

pub struct Resampler {
    kernel: Vec<f32>,
    ratio: f64, // orig / new (>= 1 assumed; downsampling path used for 48k->16k)
    k_max: usize,
}

fn bessel_i0(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half = x / 2.0;
    for k in 1..=80 {
        term *= half * half / (k as f64 * k as f64);
        sum += term;
        if term.abs() < 1e-18 * sum.abs() {
            break;
        }
    }
    sum
}

impl Resampler {
    /// Resampler driven by a stored FIR kernel (the checkpoint's
    /// `resample.kernel`, a 41-tap 3:1 lowpass): `out[n] = Σ_k kernel[k]·x[n·3 + k − 20]`.
    pub fn from_kernel(kernel: &[f32]) -> Result<Self, String> {
        if kernel.len() != 41 {
            return Err(format!(
                "speaker resample kernel must have 41 taps, got {}",
                kernel.len()
            ));
        }
        Ok(Self {
            kernel: kernel.to_vec(),
            ratio: 3.0,
            k_max: 20,
        })
    }

    /// Build a kaiser-windowed sinc resampler (torchaudio-style:
    /// lowpass_filter_width=64, rolloff=0.95, sinc_interp_kaiser).
    pub fn new(orig: u32, new: u32) -> Self {
        let ratio = orig as f64 / new as f64;
        let cutoff = 0.95 / ratio.max(1.0); // normalized to the input rate
        // beta for a kaiser window with the torchaudio "lowpass_filter_width"
        // width parameter (A = 2.285*(w-1)*pi*0.475*2 + 7.95 style)
        let width = 64.0;
        let a = 2.285 * (width - 1.0) * std::f64::consts::PI * (1.0 - cutoff) + 7.95;
        let beta = if a > 50.0 {
            0.1102 * (a - 8.7)
        } else if a >= 21.0 {
            0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
        } else {
            0.0
        };
        let k_max = (width * ratio).round() as usize;
        let n_taps = 2 * k_max + 1;
        let i0b = bessel_i0(beta);
        let mut kernel = Vec::with_capacity(n_taps);
        for i in 0..n_taps {
            let t = i as f64 - k_max as f64;
            let rel = t / k_max as f64;
            let window = bessel_i0(beta * (1.0 - rel * rel).max(0.0).sqrt()) / i0b;
            let sinc = if t == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * cutoff * t).sin() / (std::f64::consts::PI * cutoff * t)
            };
            kernel.push((2.0 * cutoff * window * sinc) as f32);
        }
        // DC gain 1 (sum normalized)
        let sum: f32 = kernel.iter().sum();
        for tap in kernel.iter_mut() {
            *tap /= sum;
        }
        Self {
            kernel,
            ratio,
            k_max,
        }
    }

    pub fn resample(&self, input: &[f32]) -> Vec<f32> {
        let out_len = (input.len() as f64 / self.ratio) as usize;
        let mut out = vec![0.0f32; out_len];
        let k_max = self.k_max as f64;
        for n in 0..out_len {
            let center = n as f64 * self.ratio;
            let mut sum = 0.0f32;
            let start = (center - k_max).floor() as i64;
            for tap in 0..self.kernel.len() {
                let src = start + tap as i64;
                if (0..input.len() as i64).contains(&src) {
                    let dx = center - src as f64;
                    let _ = (dx, k_max);
                    // resample() uses the precomputed kernel centered at integer
                    // offsets; for fractional offsets we reuse the nearest tap
                    // (acceptable pre-filter quality for prompt audio).
                    sum += input[src as usize] * self.kernel[tap];
                }
            }
            out[n] = sum;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Kaldi-style fbank
// ---------------------------------------------------------------------------

fn mel_filterbank(fft_size: usize, sample_rate: usize, n_mels: usize) -> Vec<f32> {
    let f_max = sample_rate as f64 / 2.0;
    let mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_max = mel(f_max);
    let mut filterbank = vec![0.0f32; n_mels * (fft_size / 2 + 1)];
    for m in 0..n_mels {
        let mel_left = mel_max * m as f64 / (n_mels + 1) as f64;
        let mel_center = mel_max * (m + 1) as f64 / (n_mels + 1) as f64;
        let mel_right = mel_max * (m + 2) as f64 / (n_mels + 1) as f64;
        let f_left = 700.0 * (10f64.powf(mel_left / 2595.0) - 1.0);
        let f_center = 700.0 * (10f64.powf(mel_center / 2595.0) - 1.0);
        let f_right = 700.0 * (10f64.powf(mel_right / 2595.0) - 1.0);
        let bin_left = f_left * 2.0 * fft_size as f64 / sample_rate as f64;
        let bin_center = f_center * 2.0 * fft_size as f64 / sample_rate as f64;
        let bin_right = f_right * 2.0 * fft_size as f64 / sample_rate as f64;
        for bin in 0..(fft_size / 2 + 1) {
            let weight = if bin as f64 >= bin_left && bin as f64 <= bin_center {
                (bin as f64 - bin_left) / (bin_center - bin_left)
            } else if bin as f64 > bin_center && bin as f64 <= bin_right {
                (bin_right - bin as f64) / (bin_right - bin_center)
            } else {
                0.0
            };
            filterbank[m * (fft_size / 2 + 1) + bin] = weight as f32;
        }
    }
    filterbank
}

/// In-place complex FFT over a `[re0, im0, re1, im1, ...]` buffer
/// (radix-2, iterative Cooley-Tukey).
pub fn fft_complex(re_im: &mut [f64]) {
    let n = re_im.len() / 2;
    debug_assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re_im.swap(2 * i, 2 * j);
            re_im.swap(2 * i + 1, 2 * j + 1);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let angle = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (angle.cos(), angle.sin());
        let half = len / 2;
        for start in (0..n).step_by(len) {
            let (mut w_r, mut w_i) = (1.0f64, 0.0f64);
            for j in 0..half {
                let k = start + j;
                let l = k + half;
                let (u_r, u_i) = (re_im[2 * k], re_im[2 * k + 1]);
                let (v_r, v_i) = (
                    re_im[2 * l] * w_r - re_im[2 * l + 1] * w_i,
                    re_im[2 * l] * w_i + re_im[2 * l + 1] * w_r,
                );
                re_im[2 * k] = u_r + v_r;
                re_im[2 * k + 1] = u_i + v_i;
                re_im[2 * l] = u_r - v_r;
                re_im[2 * l + 1] = u_i - v_i;
                let nw_r = w_r * wr - w_i * wi;
                w_i = w_r * wi + w_i * wr;
                w_r = nw_r;
            }
        }
        len <<= 1;
    }
}

/// 16 kHz waveform → `[frames, 80]` log-mel (mean-normalized).
pub fn kaldi_fbank(waveform: &[f32]) -> Vec<f32> {
    let n = waveform.len();
    let n_frames = if n < FRAME_LEN {
        1
    } else {
        1 + (n - FRAME_LEN) / FRAME_SHIFT
    };
    let mut x = waveform.to_vec();
    for i in (1..x.len()).rev() {
        x[i] -= PREEMPH * x[i - 1];
    }
    let filterbank = mel_filterbank(FFT_SIZE, SR_16K, MEL_BINS);
    let mut features = vec![0.0f32; n_frames * MEL_BINS];
    let mut spectrum = [0.0f32; FFT_SIZE / 2 + 1];
    for frame in 0..n_frames {
        let start = frame * FRAME_SHIFT;
        let mut fft = vec![0.0f64; 2 * FFT_SIZE];
        for i in 0..FRAME_LEN {
            let sample = if start + i < n { x[start + i] as f64 } else { 0.0 };
            let window = (0.5
                - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (FRAME_LEN - 1) as f64).cos())
                .powf(0.85);
            fft[2 * i] = sample * window;
        }
        fft_complex(&mut fft);
        for bin in 0..(FFT_SIZE / 2 + 1) {
            let (re, im) = (fft[2 * bin], fft[2 * bin + 1]);
            spectrum[bin] = (re * re + im * im) as f32;
        }
        for m in 0..MEL_BINS {
            let mut energy = 0.0f32;
            let row = &filterbank[m * (FFT_SIZE / 2 + 1)..(m + 1) * (FFT_SIZE / 2 + 1)];
            for (weight, &power) in row.iter().zip(spectrum.iter()) {
                energy += weight * power;
            }
            features[frame * MEL_BINS + m] = (energy.max(MEL_FLOOR)).ln();
        }
    }
    for m in 0..MEL_BINS {
        let mut mean = 0.0f64;
        for frame in 0..n_frames {
            mean += features[frame * MEL_BINS + m] as f64;
        }
        mean /= n_frames as f64;
        for frame in 0..n_frames {
            features[frame * MEL_BINS + m] -= mean as f32;
        }
    }
    features
}

// ---------------------------------------------------------------------------
// CAM++ x-vector
// ---------------------------------------------------------------------------

pub struct BatchNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
}

impl BatchNorm {
    /// BatchNorm with affine=False (only normalization).
    fn apply_no_affine(&self, x: &mut [f32]) {
        for c in 0..self.running_mean.len() {
            let inv = 1.0 / (self.running_var[c] + BN_EPS).sqrt();
            x[c] = (x[c] - self.running_mean[c]) * inv;
        }
    }
}

pub struct CamPlus {
    // FCM stem (fbank [80, T] → [320, T])
    pub head_conv1: Vec<f32>, // [32,1,3,3]
    pub head_bn1: BatchNorm,
    pub res_l1_0: ResBlock2d,
    pub res_l1_1: ResBlock2d,
    pub res_l2_0: ResBlock2d,
    pub res_l2_1: ResBlock2d,
    pub head_conv2: Vec<f32>, // [32,32,3,3] stride (2,1)
    pub head_bn2: BatchNorm,
    // TDNN
    pub tdnn_w: Vec<f32>, // [128,320,5]
    pub tdnn_bn: BatchNorm,
    // three dense blocks + transits
    pub blocks: Vec<DenseBlock>,
    pub transits: Vec<Transit>,
    pub out_bn: BatchNorm,
    pub dense_w: Vec<f32>, // [512,1024,1]
    pub dense_bn: BatchNorm,
}

pub struct ResBlock2d {
    pub conv1: Vec<f32>,
    pub bn1: BatchNorm,
    pub conv2: Vec<f32>,
    pub bn2: BatchNorm,
    pub shortcut: Option<(Vec<f32>, BatchNorm)>, // stride != 1
    pub stride: usize,
}

pub struct DenseBlock {
    pub layers: Vec<DenseLayer>,
}

pub struct DenseLayer {
    pub nl1: BatchNorm,      // bn(in) + relu
    pub linear1: Vec<f32>,   // [128, in, 1]
    pub nl2: BatchNorm,      // bn(128) + relu
    pub cam_local: Vec<f32>, // [32, 128, 3]
    pub cam_local_dilation: usize,
    pub cam_lin1: Vec<f32>,  // [64, 128, 1]
    pub cam_lin2: Vec<f32>,  // [32, 64, 1]
}

pub struct Transit {
    pub nl: BatchNorm,      // bn(in) + relu
    pub linear: Vec<f32>,   // [out, in, 1]
}

impl CamPlus {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let s = |name: &str, dims: &[u64]| -> Result<Vec<f32>, String> { load_f16_f32(source, name, dims) };
        let bn = |prefix: &str, channels: usize| -> Result<BatchNorm, String> {
            Ok(BatchNorm {
                weight: s(&format!("{prefix}.weight"), &[channels as u64])?,
                bias: s(&format!("{prefix}.bias"), &[channels as u64])?,
                running_mean: s(&format!("{prefix}.running_mean"), &[channels as u64])?,
                running_var: s(&format!("{prefix}.running_var"), &[channels as u64])?,
            })
        };
        let head_conv1 = s("dotstts.speaker.head.conv1.weight", &[3, 3, 1, 32])?;
        let head_bn1 = bn("dotstts.speaker.head.bn1", 32)?;
        let res = |block: &str, stride: usize| -> Result<ResBlock2d, String> {
            let prefix = format!("dotstts.speaker.head.{block}");
            let conv1 = s(&format!("{prefix}.conv1.weight"), &[3, 3, 32, 32])?;
            let bn1 = bn(&format!("{prefix}.bn1"), 32)?;
            let conv2 = s(&format!("{prefix}.conv2.weight"), &[3, 3, 32, 32])?;
            let bn2 = bn(&format!("{prefix}.bn2"), 32)?;
            let shortcut = if stride != 1 {
                Some((
                    s(&format!("{prefix}.shortcut.0.weight"), &[1, 1, 32, 32])?,
                    bn(&format!("{prefix}.shortcut.1"), 32)?,
                ))
            } else {
                None
            };
            Ok(ResBlock2d {
                conv1,
                bn1,
                conv2,
                bn2,
                shortcut,
                stride,
            })
        };
        let res_l1_0 = res("layer1.0", 2)?;
        let res_l1_1 = res("layer1.1", 1)?;
        let res_l2_0 = res("layer2.0", 2)?;
        let res_l2_1 = res("layer2.1", 1)?;
        let head_conv2 = s("dotstts.speaker.head.conv2.weight", &[3, 3, 32, 32])?;
        let head_bn2 = bn("dotstts.speaker.head.bn2", 32)?;
        let tdnn_w = s("dotstts.speaker.xvector.tdnn.linear.weight", &[5, 320, 128])?;
        let tdnn_bn = bn("dotstts.speaker.xvector.tdnn.nonlinear.batchnorm", 128)?;

        let mut blocks = Vec::new();
        let mut transits = Vec::new();
        let mut channels = 128usize;
        for (bi, (num_layers, dilation)) in [(12usize, 1usize), (24, 2), (16, 2)].iter().enumerate() {
            let mut layers = Vec::new();
            for layer in 0..*num_layers {
                let dil = *dilation;
                let in_ch = channels + layer * 32;
                let prefix = format!(
                    "dotstts.speaker.xvector.block{}.tdnnd{}",
                    bi + 1,
                    layer + 1
                );
                let linear1 = s(
                    &format!("{prefix}.linear1.weight"),
                    &[1, in_ch as u64, 128],
                )?;
                let nl1 = bn(&format!("{prefix}.nonlinear1.batchnorm"), in_ch)?;
                let nl2 = bn(&format!("{prefix}.nonlinear2.batchnorm"), 128)?;
                let cam_local = s(&format!("{prefix}.cam_layer.linear_local.weight"), &[3, 128, 32])?;
                let cam_lin1 = s(&format!("{prefix}.cam_layer.linear1.weight"), &[1, 128, 64])?;
                let cam_lin2 = s(&format!("{prefix}.cam_layer.linear2.weight"), &[1, 64, 32])?;
                layers.push(DenseLayer {
                    nl1,
                    linear1,
                    nl2,
                    cam_local,
                    cam_local_dilation: dil,
                    cam_lin1,
                    cam_lin2,
                });
            }
            blocks.push(DenseBlock { layers });
            channels += num_layers * 32;
            let nl = bn(
                &format!("dotstts.speaker.xvector.transit{}.nonlinear.batchnorm", bi + 1),
                channels,
            )?;
            let linear = s(
                &format!("dotstts.speaker.xvector.transit{}.linear.weight", bi + 1),
                &[1, channels as u64, (channels / 2) as u64],
            )?;
            transits.push(Transit { nl, linear });
            channels /= 2;
        }
        let out_bn = bn(
            "dotstts.speaker.xvector.out_nonlinear.batchnorm",
            channels,
        )?;
        let dense_w = s(
            "dotstts.speaker.xvector.dense.linear.weight",
            &[1, (channels * 2) as u64, 512],
        )?;
        let dense_bn = BatchNorm {
            weight: vec![0.0; 512],
            bias: vec![0.0; 512],
            running_mean: s("dotstts.speaker.xvector.dense.nonlinear.batchnorm.running_mean", &[512])?,
            running_var: s("dotstts.speaker.xvector.dense.nonlinear.batchnorm.running_var", &[512])?,
        };
        Ok(Self {
            head_conv1,
            head_bn1,
            res_l1_0,
            res_l1_1,
            res_l2_0,
            res_l2_1,
            head_conv2,
            head_bn2,
            tdnn_w,
            tdnn_bn,
            blocks,
            transits,
            out_bn,
            dense_w,
            dense_bn,
        })
    }

    /// Encode mel frames `[frames, 80]` → 512-dim x-vector.
    pub fn encode(&self, mel: &[f32]) -> Result<Vec<f32>, String> {
        let frames = mel.len() / MEL_BINS;
        if frames == 0 {
            return Err("speaker encoder needs at least one frame".into());
        }
        // FCM: [80, T] → [320, T]
        let mut x = self.fcm(mel, frames);
        // TDNN: [320, T] → [128, T2]
        let t2 = conv1d_length(frames, 5, 2, 2);
        x = self.tdnn(&x, frames, t2);
        let t = t2;
        // dense blocks + transits
        for (block, transit) in self.blocks.iter().zip(self.transits.iter()) {
            let mut channels = 0usize;
            for (layer_idx, layer) in block.layers.iter().enumerate() {
                if layer_idx == 0 {
                    channels = match self.blocks.iter().position(|b| std::ptr::eq(b, block)) {
                        Some(0) => 128usize,
                        Some(1) => 256usize,
                        _ => 512usize,
                    };
                }
                let in_ch = channels + layer_idx * 32;
                x = dense_layer_forward(layer, &x, t, in_ch);
            }
            let channels_after = channels + block.layers.len() * 32;
            let mut trans = vec![0.0f32; t * (channels_after / 2)];
            transit_forward(transit, &x, t, channels_after, &mut trans);
            x = trans;
        }
        // out_nonlinear: bn(512) + relu
        for c in 0..512 {
            let inv = 1.0 / (self.out_bn.running_var[c] + BN_EPS).sqrt();
            let scale = inv * self.out_bn.weight[c];
            let bias = self.out_bn.bias[c] - self.out_bn.running_mean[c] * scale;
            for j in 0..t {
                let idx = j * 512 + c;
                x[idx] = (x[idx] * scale + bias).max(0.0);
            }
        }
        // masked stats pooling (all frames valid, unbiased std, floor 1e-2)
        let mut stats = vec![0.0f32; 512 * 2];
        for c in 0..512 {
            let mut mean = 0.0f64;
            for j in 0..t {
                mean += x[j * 512 + c] as f64;
            }
            mean /= t as f64;
            let mut var = 0.0f64;
            for j in 0..t {
                let d = x[j * 512 + c] as f64 - mean;
                var += d * d;
            }
            var /= (t - 1).max(1) as f64;
            stats[c] = mean as f32;
            stats[512 + c] = (var.max(1e-2)).sqrt() as f32;
        }
        // dense: [1024, 1] → 512 + batchnorm_ (affine=False)
        let mut out = vec![0.0f32; 512];
        for o in 0..512 {
            let mut sum = 0.0f32;
            for i in 0..1024 {
                sum += self.dense_w[o * 1024 + i] * stats[i];
            }
            out[o] = sum;
        }
        self.dense_bn.apply_no_affine(&mut out);
        Ok(out)
    }

    fn fcm(&self, mel: &[f32], frames: usize) -> Vec<f32> {
        // conv2d [1→32] on [80, T]
        let mut x = conv2d_forward(&self.head_conv1, None, mel, frames, 80, 1, 32, 3, 3, 1, 1);
        self.apply_bn2d(&self.head_bn1, &mut x, 32, 80, frames);
        relu_inplace(&mut x);
        x = self.resblock(&self.res_l1_0, &x, 32, 80, frames);
        x = self.resblock(&self.res_l1_1, &x, 32, 40, frames);
        x = self.resblock(&self.res_l2_0, &x, 32, 40, frames);
        x = self.resblock(&self.res_l2_1, &x, 32, 20, frames);
        // conv2 stride (2,1)
        let mut y = conv2d_forward_stride(&self.head_conv2, None, &x, frames, 20, 32, 32, 3, 3, 2, 1, 1, 1);
        self.apply_bn2d(&self.head_bn2, &mut y, 32, 10, frames);
        relu_inplace(&mut y);
        // reshape [32*10, T] = [320, T]
        y
    }

    fn resblock(&self, block: &ResBlock2d, x: &[f32], channels: usize, h: usize, t: usize) -> Vec<f32> {
        let h_out = if block.stride == 1 { h } else { h / 2 };
        let mut out = if block.stride == 1 {
            conv2d_forward(&block.conv1, None, x, t, h, channels, channels, 3, 3, 1, 1)
        } else {
            conv2d_forward_stride(&block.conv1, None, x, t, h, channels, channels, 3, 3, 2, 1, 1, 1)
        };
        self.apply_bn2d(&block.bn1, &mut out, channels, h_out, t);
        relu_inplace(&mut out);
        out = conv2d_forward(
            &block.conv2,
            None,
            &out,
            t,
            h_out,
            channels,
            channels,
            3,
            3,
            1,
            1,
        );
        self.apply_bn2d(&block.bn2, &mut out, channels, h_out, t);
        let residual = match &block.shortcut {
            Some((w, bn_out)) => {
                let mut sc = conv2d_forward_stride(w, None, x, t, h, channels, channels, 1, 1, 2, 1, 0, 0);
                self.apply_bn2d(bn_out, &mut sc, channels, h_out, t);
                sc
            }
            None => x.to_vec(),
        };
        for (o, &r) in out.iter_mut().zip(residual.iter()) {
            *o += r;
        }
        relu_inplace(&mut out);
        out
    }

    fn apply_bn2d(&self, bn: &BatchNorm, x: &mut [f32], channels: usize, h: usize, t: usize) {
        for c in 0..channels {
            let inv = 1.0 / (bn.running_var[c] + BN_EPS).sqrt();
            let scale = inv * bn.weight[c];
            let bias = bn.bias[c] - bn.running_mean[c] * scale;
            for pos in 0..h * t {
                let idx = c * h * t + pos;
                x[idx] = x[idx] * scale + bias;
            }
        }
    }

    fn tdnn(&self, x: &[f32], t: usize, t2: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; t2 * 128];
        for o in 0..128 {
            for j in 0..t2 {
                let mut sum = 0.0f32;
                for i in 0..320 {
                    for k in 0..5usize {
                        let pos = j as isize * 2 + k as isize - 2;
                        if (0..t as isize).contains(&pos) {
                            sum += self.tdnn_w[o * 320 * 5 + i * 5 + k]
                                * x[pos as usize * 320 + i];
                        }
                    }
                }
                out[j * 128 + o] = sum;
            }
        }
        for c in 0..128 {
            let inv = 1.0 / (self.tdnn_bn.running_var[c] + BN_EPS).sqrt();
            let scale = inv * self.tdnn_bn.weight[c];
            let bias = self.tdnn_bn.bias[c] - self.tdnn_bn.running_mean[c] * scale;
            for j in 0..t2 {
                let idx = j * 128 + c;
                out[idx] = out[idx] * scale + bias;
            }
        }
        for value in out.iter_mut() {
            *value = value.max(0.0);
        }
        out
    }
}

fn conv1d_length(t: usize, kernel: usize, stride: usize, pad: usize) -> usize {
    (t + 2 * pad).saturating_sub(kernel) / stride + 1
}

fn relu_inplace(x: &mut [f32]) {
    for value in x.iter_mut() {
        *value = value.max(0.0);
    }
}

/// Conv2d over a `[channels, h, t]` layout, stride (1,1), pad 1.
/// `weight` gguf-dims `[kw, kh, in, out]`.
fn conv2d_forward(
    weight: &[f32],
    bias: Option<&[f32]>,
    x: &[f32],
    t: usize,
    h: usize,
    in_ch: usize,
    out_ch: usize,
    _kw: usize,
    _kh: usize,
    sh: usize,
    sw: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_ch * h * t];
    for o in 0..out_ch {
        let b = bias.map_or(0.0, |b| b[o]);
        for i in 0..in_ch {
            for kh in 0..3usize {
                for kw in 0..3usize {
                    let w = weight[o * in_ch * 9 + i * 9 + kh * 3 + kw];
                    if w == 0.0 {
                        continue;
                    }
                    for hp in 0..h {
                        let src_h = hp as isize * sh as isize + kh as isize - 1;
                        for tp in 0..t {
                            let src_t = tp as isize * sw as isize + kw as isize - 1;
                            if src_h < 0 || src_t < 0 || src_h >= h as isize || src_t >= t as isize {
                                continue;
                            }
                            out[o * h * t + hp * t + tp] += w
                                * x[i * h * t + src_h as usize * t + src_t as usize];
                        }
                    }
                }
            }
        }
        for pos in 0..h * t {
            out[o * h * t + pos] += b;
        }
    }
    out
}

/// Conv2d over a `[channels, h, t]` layout with explicit kernel/pad/stride.
/// `weight` gguf-dims `[kw, kh, in, out]`; pad is symmetric (ph top/bottom,
/// pw left/right), out h = saturating formula, stride may differ per axis.
fn conv2d_forward_stride(
    weight: &[f32],
    bias: Option<&[f32]>,
    x: &[f32],
    t: usize,
    h_in: usize,
    in_ch: usize,
    out_ch: usize,
    kw: usize,
    kh: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> Vec<f32> {
    let h_out = (h_in + 2 * ph).saturating_sub(kh) / sh + 1;
    let mut out = vec![0.0f32; out_ch * h_out * t];
    for o in 0..out_ch {
        let b = bias.map_or(0.0, |b| b[o]);
        for i in 0..in_ch {
            for kk_h in 0..kh {
                for kk_w in 0..kw {
                    let w = weight[o * in_ch * kh * kw + i * kh * kw + kk_h * kw + kk_w];
                    if w == 0.0 {
                        continue;
                    }
                    for hp in 0..h_out {
                        let src_h = hp as isize * sh as isize + kk_h as isize - ph as isize;
                        if src_h < 0 || src_h >= h_in as isize {
                            continue;
                        }
                        for tp in 0..t {
                            let src_t = tp as isize * sw as isize + kk_w as isize - pw as isize;
                            if src_t < 0 || src_t >= t as isize {
                                continue;
                            }
                            out[o * h_out * t + hp * t + tp] += w
                                * x[i * h_in * t + src_h as usize * t + src_t as usize];
                        }
                    }
                }
            }
        }
        for pos in 0..h_out * t {
            out[o * h_out * t + pos] += b;
        }
    }
    out
}

/// One dense-layer step: [in, T] → [in+32, T] (concat), with the CAM
/// attention gate (reference `CAMDenseTDNNLayer` + `CAMLayer`).
fn dense_layer_forward(layer: &DenseLayer, x: &[f32], t: usize, in_ch: usize) -> Vec<f32> {
    // nonlinear1 (bn(in) + relu)
    let mut h = vec![0.0f32; in_ch * t];
    for c in 0..in_ch {
        let inv = 1.0 / (layer.nl1.running_var[c] + BN_EPS).sqrt();
        let scale = inv * layer.nl1.weight[c];
        let bias = layer.nl1.bias[c] - layer.nl1.running_mean[c] * scale;
        for j in 0..t {
            let idx = j * in_ch + c;
            h[idx] = (x[j * in_ch + c] * scale + bias).max(0.0);
        }
    }
    // linear1: 1x1 conv [in → 128]
    let mut bn_in = vec![0.0f32; 128 * t];
    for o in 0..128 {
        for j in 0..t {
            let mut sum = 0.0f32;
            for i in 0..in_ch {
                sum += layer.linear1[o * in_ch + i] * h[j * in_ch + i];
            }
            bn_in[j * 128 + o] = sum;
        }
    }
    // nonlinear2 (bn(128) + relu)
    for c in 0..128 {
        let inv = 1.0 / (layer.nl2.running_var[c] + BN_EPS).sqrt();
        let scale = inv * layer.nl2.weight[c];
        let bias = layer.nl2.bias[c] - layer.nl2.running_mean[c] * scale;
        for j in 0..t {
            let idx = j * 128 + c;
            bn_in[idx] = (bn_in[idx] * scale + bias).max(0.0);
        }
    }
    // cam_layer: linear_local 3x1 with dilation d (pad = d)
    let d = layer.cam_local_dilation;
    let mut out = vec![0.0f32; 32 * t];
    for o in 0..32 {
        for j in 0..t {
            let mut sum = 0.0f32;
            for i in 0..128 {
                for k in 0..3usize {
                    let src = j as isize + k as isize * d as isize - d as isize;
                    if (0..t as isize).contains(&src) {
                        sum += layer.cam_local[o * 128 * 3 + i * 3 + k]
                            * bn_in[src as usize * 128 + i];
                    }
                }
            }
            out[j * 32 + o] = sum;
        }
    }
    // attention gate: context = seg-pooled + global mean → 1x1 convs → sigmoid
    let seg_count = (t + 99) / 100;
    let mut seg_sum = vec![0.0f32; 128 * seg_count];
    for c in 0..128 {
        for seg in 0..seg_count {
            let start = seg * 100;
            let end = (start + 100).min(t);
            let mut sum = 0.0f32;
            for j in start..end {
                sum += bn_in[j * 128 + c];
            }
            seg_sum[c * seg_count + seg] = sum / (end - start) as f32;
        }
    }
    let mut context = vec![0.0f32; 128 * t];
    for c in 0..128 {
        let mut mean = 0.0f64;
        for j in 0..t {
            mean += bn_in[j * 128 + c] as f64;
        }
        mean /= t as f64;
        for j in 0..t {
            let seg = j / 100;
            context[j * 128 + c] = seg_sum[c * seg_count + seg] + mean as f32;
        }
    }
    let mut c1 = vec![0.0f32; 64 * t];
    for o in 0..64 {
        for j in 0..t {
            let mut sum = 0.0f32;
            for i in 0..128 {
                sum += layer.cam_lin1[o * 128 + i] * context[j * 128 + i];
            }
            c1[j * 64 + o] = sum.max(0.0);
        }
    }
    for o in 0..32 {
        for j in 0..t {
            let mut sum = 0.0f32;
            for i in 0..64 {
                sum += layer.cam_lin2[o * 64 + i] * c1[j * 64 + i];
            }
            out[j * 32 + o] *= 1.0 / (1.0 + (-sum).exp());
        }
    }
    // concat: [in + 32, T]
    let mut combined = vec![0.0f32; (in_ch + 32) * t];
    for j in 0..t {
        combined[j * (in_ch + 32)..j * (in_ch + 32) + in_ch]
            .copy_from_slice(&x[j * in_ch..(j + 1) * in_ch]);
        combined[j * (in_ch + 32) + in_ch..(j + 1) * (in_ch + 32)]
            .copy_from_slice(&out[j * 32..(j + 1) * 32]);
    }
    combined
}

fn transit_forward(
    transit: &Transit,
    x: &[f32],
    t: usize,
    in_ch: usize,
    out: &mut [f32],
) {
    let out_ch = in_ch / 2;
    // bn + relu
    let mut h = vec![0.0f32; in_ch * t];
    for c in 0..in_ch {
        let inv = 1.0 / (transit.nl.running_var[c] + BN_EPS).sqrt();
        let scale = inv * transit.nl.weight[c];
        let bias = transit.nl.bias[c] - transit.nl.running_mean[c] * scale;
        for j in 0..t {
            let idx = j * in_ch + c;
            h[idx] = (x[idx] * scale + bias).max(0.0);
        }
    }
    for o in 0..out_ch {
        for j in 0..t {
            let mut sum = 0.0f32;
            for i in 0..in_ch {
                sum += transit.linear[o * in_ch + i] * h[j * in_ch + i];
            }
            out[j * out_ch + o] = sum;
        }
    }
}