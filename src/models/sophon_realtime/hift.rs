//! Causal 24 kHz HiFT vocoder with explicit excitation noise for numerical comparisons.

use super::flow::{load_f32, Conv};
use crate::core::tensor::{require_tensor_type, GGMLType, MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::audio::frontend::RealFft;
use crate::ops::kernel::Weight;
use crate::ops::nn::Linear;

const MEL: usize = 80;
const HARMONICS: usize = 9;
pub const SAMPLE_RATE: usize = 24_000;
pub const SAMPLES_PER_FRAME: usize = 480;

pub struct HiFT<'a> {
    f0: Vec<Conv>,
    classifier: Linear<'a>,
    source: Linear<'a>,
    pre: Conv,
    up: Vec<Conv>,
    down: Vec<Conv>,
    source_blocks: Vec<ResBlock>,
    blocks: Vec<ResBlock>,
    post: Conv,
}

impl<'a> HiFT<'a> {
    pub fn load(source: &'a dyn TensorSource) -> Result<Self, String> {
        super::require_string(source, "general.architecture", "sophon_realtime_hift")?;
        super::require_string(source, "sophon_realtime.component", "hift")?;
        super::require_string(
            source,
            "sophon_realtime.source_revision",
            super::SOURCE_REVISION,
        )?;
        let config: serde_json::Value = serde_json::from_str(
            source
                .metadata("sophon_realtime.hift_config_json")
                .and_then(MetaValue::to_string_val)
                .ok_or("missing HiFT config")?,
        )
        .map_err(|e| format!("invalid HiFT config: {e}"))?;
        let expected = serde_json::json!({
            "in_channels":80,"base_channels":512,"nb_harmonics":8,"sampling_rate":24000,
            "nsf_alpha":0.1,"nsf_sigma":0.003,"nsf_voiced_threshold":10,
            "upsample_rates":[8,5,3],"upsample_kernel_sizes":[16,11,7],
            "istft_n_fft":16,"istft_hop_len":4,"resblock_kernel_sizes":[3,7,11],
            "resblock_dilation_sizes":[[1,3,5],[1,3,5],[1,3,5]],
            "source_resblock_kernel_sizes":[7,7,11],
            "source_resblock_dilation_sizes":[[1,3,5],[1,3,5],[1,3,5]],
            "lrelu_slope":0.1,"final_lrelu_slope":0.01,"audio_limit":0.99,
            "conv_pre_look_right":4,"f0_cond_channels":512,"f0_num_class":1,
            "snake_alpha_logscale":false,"snake_epsilon":1e-9,
            "upsample_mode":"nearest_then_causal_conv1d","conv_layout":"OIK",
            "stft_center":true,"stft_pad_mode":"reflect","stft_periodic_hann":true
        });
        if config != expected {
            return Err("unsupported HiFT configuration".into());
        }
        let mut up = Vec::new();
        let mut down = Vec::new();
        let mut source_blocks = Vec::new();
        let mut blocks = Vec::new();
        for stage in 0..3 {
            let channels = 256 >> stage;
            up.push(Conv::load(
                source,
                &format!("ups.{stage}"),
                channels * 2,
                channels,
                [16, 11, 7][stage],
                1,
            )?);
            let stride = [15, 3, 1][stage];
            down.push(Conv::load(
                source,
                &format!("source_downs.{stage}"),
                18,
                channels,
                if stride == 1 { 1 } else { stride * 2 },
                1,
            )?);
            source_blocks.push(ResBlock::load(
                source,
                &format!("source_resblocks.{stage}"),
                channels,
                [7, 7, 11][stage],
            )?);
            for (index, kernel) in [3, 7, 11].into_iter().enumerate() {
                blocks.push(ResBlock::load(
                    source,
                    &format!("resblocks.{}", stage * 3 + index),
                    channels,
                    kernel,
                )?);
            }
        }
        Ok(Self {
            f0: (0..5)
                .map(|i| {
                    Conv::load(
                        source,
                        &format!("f0_predictor.condnet.{}", i * 2),
                        if i == 0 { MEL } else { 512 },
                        512,
                        if i == 0 { 4 } else { 3 },
                        1,
                    )
                })
                .collect::<Result<_, _>>()?,
            classifier: linear(source, "f0_predictor.classifier", 512, 1)?,
            source: linear(source, "m_source.l_linear", HARMONICS, 1)?,
            pre: Conv::load(source, "conv_pre", MEL, 512, 5, 1)?,
            up,
            down,
            source_blocks,
            blocks,
            post: Conv::load(source, "conv_post", 64, 18, 7, 1)?,
        })
    }

    pub fn predict_f0(
        &self,
        mel: &[f32],
        frames: usize,
        finalize: bool,
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        validate_mel(mel, frames, finalize)?;
        let (mut hidden, rows) =
            self.f0[0].forward(mel, frames, 0, if finalize { 3 } else { 0 }, 1, 1, pool)?;
        elu(&mut hidden);
        for layer in &self.f0[1..] {
            hidden = layer.forward(&hidden, rows, 2, 0, 1, 1, pool)?.0;
            elu(&mut hidden);
        }
        let hidden = to_rows(&hidden, 512, rows);
        let mut f0 = self.classifier.forward(pool, &hidden, rows)?;
        f0.iter_mut().for_each(|x| *x = x.abs());
        finite("f0", &f0)?;
        trace("f0", &[rows], &f0);
        Ok(f0)
    }

    /// `noise` is [frames * 480, 9] uniform excitation, corresponding to the
    /// reference's nonpersistent causal sine_waves buffer rather than a model weight.
    pub fn synthesize(
        &self,
        mel: &[f32],
        frames: usize,
        finalize: bool,
        noise: &[f32],
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        self.synthesize_until(mel, frames, finalize, noise, pool, &|| false)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn synthesize_until(
        &self,
        mel: &[f32],
        frames: usize,
        finalize: bool,
        noise: &[f32],
        pool: &ComputePool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>, String> {
        super::check_cancelled(cancelled)?;
        let f0 = self.predict_f0(mel, frames, finalize, pool)?;
        super::check_cancelled(cancelled)?;
        let expected = f0.len() * SAMPLES_PER_FRAME * HARMONICS;
        if noise.len() != expected || noise.iter().any(|n| !n.is_finite()) {
            return Err(format!(
                "HiFT excitation must contain {expected} finite values"
            ));
        }
        let harmonics = excitation(&f0, noise)?;
        let mut source = self
            .source
            .forward(pool, &harmonics, f0.len() * SAMPLES_PER_FRAME)?;
        source.iter_mut().for_each(|x| *x = x.tanh());
        trace("source", &[source.len()], &source);
        let cropped = crop_channels(mel, MEL, frames, f0.len());
        self.decode_until(&cropped, f0.len(), &source, finalize, pool, cancelled)
    }

    pub fn decode(
        &self,
        mel: &[f32],
        frames: usize,
        source: &[f32],
        finalize: bool,
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        self.decode_until(mel, frames, source, finalize, pool, &|| false)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_until(
        &self,
        mel: &[f32],
        frames: usize,
        source: &[f32],
        finalize: bool,
        pool: &ComputePool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>, String> {
        super::check_cancelled(cancelled)?;
        if frames == 0
            || mel.len() != MEL * frames
            || source.len() != frames * SAMPLES_PER_FRAME
            || (!finalize && frames <= 5)
        {
            return Err("invalid HiFT mel/excitation dimensions".into());
        }
        finite("mel", mel)?;
        finite("source", source)?;
        let (mut spectrum, mut source_frames) = source_spectrum(source)?;
        let (mut hidden, mut rows) =
            self.pre
                .forward(mel, frames, 0, if finalize { 4 } else { 0 }, 1, 1, pool)?;
        if !finalize {
            let keep = source_frames
                .checked_sub(480)
                .ok_or("HiFT source too short")?;
            spectrum = crop_channels(&spectrum, 18, source_frames, keep);
            source_frames = keep;
        }
        for stage in 0..3 {
            super::check_cancelled(cancelled)?;
            leaky(&mut hidden, 0.1);
            let rate = [8, 5, 3][stage];
            hidden = nearest(&hidden, 512 >> stage, rows, rate)?;
            rows *= rate;
            (hidden, rows) =
                self.up[stage].forward(&hidden, rows, [15, 10, 6][stage], 0, 1, 1, pool)?;
            if stage == 2 {
                hidden = reflect_left_one(&hidden, 64, rows)?;
                rows += 1;
            }
            let stride = [15, 3, 1][stage];
            let (conditioning, conditioning_rows) = self.down[stage].forward(
                &spectrum,
                source_frames,
                stride - 1,
                0,
                stride,
                1,
                pool,
            )?;
            if rows != conditioning_rows {
                return Err("HiFT source/mel frame mismatch".into());
            }
            let conditioning = self.source_blocks[stage].forward(conditioning, rows, pool)?;
            add(&mut hidden, &conditioning)?;
            let mut sum = vec![0.0; hidden.len()];
            for block in &self.blocks[stage * 3..stage * 3 + 3] {
                super::check_cancelled(cancelled)?;
                add(&mut sum, &block.forward(hidden.clone(), rows, pool)?)?;
            }
            sum.iter_mut().for_each(|x| *x /= 3.0);
            hidden = sum;
            trace(&format!("stage.{stage}"), &[256 >> stage, rows], &hidden);
        }
        leaky(&mut hidden, 0.01);
        let (spectral, spectral_frames) = self.post.forward(&hidden, rows, 6, 0, 1, 1, pool)?;
        trace("spectrum", &[18, spectral_frames], &spectral);
        let mut waveform = waveform(&spectral, spectral_frames)?;
        if !finalize {
            waveform.truncate(
                waveform
                    .len()
                    .checked_sub(480)
                    .ok_or("HiFT nonfinal waveform too short")?,
            );
        }
        waveform.iter_mut().for_each(|x| *x = x.clamp(-0.99, 0.99));
        finite("waveform", &waveform)?;
        trace("waveform", &[waveform.len()], &waveform);
        Ok(waveform)
    }
}

struct ResBlock {
    channels: usize,
    kernel: usize,
    first: Vec<(Conv, Vec<f32>)>,
    second: Vec<(Conv, Vec<f32>)>,
}

impl ResBlock {
    fn load(
        source: &dyn TensorSource,
        name: &str,
        channels: usize,
        kernel: usize,
    ) -> Result<Self, String> {
        let branch = |index| {
            (0..3)
                .map(|i| {
                    Ok((
                        Conv::load(
                            source,
                            &format!("{name}.convs{index}.{i}"),
                            channels,
                            channels,
                            kernel,
                            1,
                        )?,
                        load_f32(
                            source,
                            &format!("{name}.activations{index}.{i}.alpha"),
                            &[channels as u64],
                        )?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()
        };
        Ok(Self {
            channels,
            kernel,
            first: branch(1)?,
            second: branch(2)?,
        })
    }
    fn forward(
        &self,
        mut input: Vec<f32>,
        frames: usize,
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        if input.len() != self.channels * frames {
            return Err("invalid HiFT residual input".into());
        }
        for (i, dilation) in [1, 3, 5].into_iter().enumerate() {
            let mut x = input.clone();
            snake(&mut x, frames, &self.first[i].1)?;
            let (mut x, _) = self.first[i].0.forward(
                &x,
                frames,
                (self.kernel - 1) * dilation,
                0,
                1,
                dilation,
                pool,
            )?;
            snake(&mut x, frames, &self.second[i].1)?;
            let (x, _) = self.second[i]
                .0
                .forward(&x, frames, self.kernel - 1, 0, 1, 1, pool)?;
            add(&mut input, &x)?;
        }
        Ok(input)
    }
}

fn linear<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    input: usize,
    output: usize,
) -> Result<Linear<'a>, String> {
    let key = format!("{name}.weight");
    require_tensor_type(
        source,
        &key,
        &[input as u64, output as u64],
        &[GGMLType::F32],
    )?;
    let weight = Weight::from_f32_bytes(
        source
            .tensor_slice(&key)
            .ok_or_else(|| format!("missing {key}"))?,
        input,
        output,
    )
    .ok_or("invalid HiFT matrix")?;
    Linear::from_weight(
        weight,
        Some(load_f32(source, &format!("{name}.bias"), &[output as u64])?),
    )
}

fn validate_mel(mel: &[f32], frames: usize, finalize: bool) -> Result<(), String> {
    if frames == 0 || frames > 15000 || mel.len() != frames * MEL || (!finalize && frames <= 8) {
        return Err(
            "HiFT requires 1..=15000 mel frames, or at least 9 for a nonfinal chunk".into(),
        );
    }
    finite("mel", mel)
}
fn finite(name: &str, values: &[f32]) -> Result<(), String> {
    if values.iter().any(|x| !x.is_finite()) {
        Err(format!("nonfinite HiFT {name}"))
    } else {
        Ok(())
    }
}
fn elu(values: &mut [f32]) {
    for x in values {
        if *x < 0.0 {
            *x = x.exp_m1();
        }
    }
}
fn leaky(values: &mut [f32], slope: f32) {
    for x in values {
        if *x < 0.0 {
            *x *= slope;
        }
    }
}
fn add(left: &mut [f32], right: &[f32]) -> Result<(), String> {
    if left.len() != right.len() {
        return Err("HiFT residual shape mismatch".into());
    }
    crate::ops::vec_add_into(right, left);
    Ok(())
}
fn snake(values: &mut [f32], frames: usize, alpha: &[f32]) -> Result<(), String> {
    if frames == 0 || values.len() != frames * alpha.len() {
        return Err("invalid Snake dimensions".into());
    }
    for (channel, &a) in values.chunks_exact_mut(frames).zip(alpha) {
        for value in channel {
            *value += (*value * a).sin().powi(2) / (a + 1e-9);
        }
    }
    finite("Snake", values)
}
fn crop_channels(input: &[f32], channels: usize, frames: usize, keep: usize) -> Vec<f32> {
    debug_assert!(keep <= frames && input.len() == channels * frames);
    input
        .chunks_exact(frames)
        .flat_map(|row| row[..keep].iter().copied())
        .collect()
}
fn to_rows(input: &[f32], channels: usize, frames: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for c in 0..channels {
        for t in 0..frames {
            output[t * channels + c] = input[c * frames + t];
        }
    }
    output
}
fn nearest(
    input: &[f32],
    channels: usize,
    frames: usize,
    scale: usize,
) -> Result<Vec<f32>, String> {
    if input.len() != channels * frames || scale == 0 {
        return Err("invalid HiFT upsample".into());
    }
    Ok(input
        .iter()
        .flat_map(|&v| std::iter::repeat_n(v, scale))
        .collect())
}
fn reflect_left_one(input: &[f32], channels: usize, frames: usize) -> Result<Vec<f32>, String> {
    if frames < 2 || input.len() != channels * frames {
        return Err("HiFT reflection requires at least 2 frames".into());
    }
    Ok(input
        .chunks_exact(frames)
        .flat_map(|row| std::iter::once(row[1]).chain(row.iter().copied()))
        .collect())
}

fn excitation(f0: &[f32], noise: &[f32]) -> Result<Vec<f32>, String> {
    if noise.len() != f0.len() * SAMPLES_PER_FRAME * HARMONICS {
        return Err("invalid excitation length".into());
    }
    let mut phase = [0.0f64; HARMONICS];
    let mut output = vec![0.0; noise.len()];
    for (frame, &frequency) in f0.iter().enumerate() {
        let voiced = frequency > 10.0;
        let amplitude = if voiced { 0.003 } else { 0.1 / 3.0 };
        let mut sine = [0.0; HARMONICS];
        // Linear downsampling selects the centre of each constant 480-sample F0 frame.
        for h in 0..HARMONICS {
            let increment = (frequency * (h + 1) as f32 / SAMPLE_RATE as f32).rem_euclid(1.0);
            phase[h] += f64::from(increment);
            sine[h] = if voiced {
                ((phase[h] as f32 * (2.0 * std::f32::consts::PI)) * SAMPLES_PER_FRAME as f32).sin()
                    * 0.1
            } else {
                0.0
            };
        }
        for sample in frame * SAMPLES_PER_FRAME..(frame + 1) * SAMPLES_PER_FRAME {
            for h in 0..HARMONICS {
                let i = sample * HARMONICS + h;
                output[i] = sine[h] + amplitude * noise[i];
            }
        }
    }
    Ok(output)
}

fn hann() -> [f32; 16] {
    std::array::from_fn(|n| {
        (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / 16.0).cos()) as f32
    })
}
fn source_spectrum(source: &[f32]) -> Result<(Vec<f32>, usize), String> {
    if source.len() <= 8 {
        return Err("HiFT excitation too short for reflection STFT".into());
    }
    let frames = source.len() / 4 + 1;
    let mut output = vec![0.0; 18 * frames];
    let mut fft = RealFft::new(16).map_err(|e| format!("HiFT FFT: {e:?}"))?;
    let window = hann();
    for t in 0..frames {
        let mut input = [0.0; 16];
        for n in 0..16 {
            let p = (t * 4 + n) as isize - 8;
            let index = if p < 0 {
                -p as usize
            } else if p as usize >= source.len() {
                2 * source.len() - 2 - p as usize
            } else {
                p as usize
            };
            input[n] = source[index] * window[n];
        }
        let mut bins = [[0.0; 2]; 9];
        fft.complex(&input, &mut bins);
        for k in 0..9 {
            output[k * frames + t] = bins[k][0];
            output[(k + 9) * frames + t] = bins[k][1];
        }
    }
    Ok((output, frames))
}
fn waveform(spectral: &[f32], frames: usize) -> Result<Vec<f32>, String> {
    if frames < 2 || spectral.len() != 18 * frames {
        return Err("invalid HiFT output spectrum".into());
    }
    let mut fft = RealFft::new(16).map_err(|e| format!("HiFT FFT: {e:?}"))?;
    let window = hann();
    let len = (frames - 1) * 4 + 16;
    let mut output = vec![0.0; len];
    let mut envelope = vec![0.0; len];
    for t in 0..frames {
        let mut bins = [[0.0; 2]; 9];
        for k in 0..9 {
            let magnitude = spectral[k * frames + t].exp().min(100.0);
            let phase = spectral[(k + 9) * frames + t].sin();
            bins[k][0] = magnitude * phase.cos();
            bins[k][1] = magnitude * phase.sin();
        }
        let mut values = [0.0; 16];
        fft.inverse(&bins, &mut values);
        for n in 0..16 {
            let i = t * 4 + n;
            output[i] += values[n] * window[n];
            envelope[i] += window[n] * window[n];
        }
    }
    let end = len - 8;
    for i in 8..end {
        if envelope[i] < 1e-11 {
            return Err("HiFT iSTFT has a zero window envelope".into());
        }
        output[i] /= envelope[i];
    }
    Ok(output[8..end].to_vec())
}

fn trace(name: &str, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(
        &format!("sophon.hift.{name}"),
        None,
        shape,
        values,
    ));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, shape, values);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn causal_upsample_and_reflection_preserve_channel_layout() {
        assert_eq!(
            nearest(&[1., 2., 3., 4.], 2, 2, 2).unwrap(),
            [1., 1., 2., 2., 3., 3., 4., 4.]
        );
        assert_eq!(
            reflect_left_one(&[1., 2., 3., 4.], 2, 2).unwrap(),
            [2., 1., 2., 4., 3., 4.]
        );
        assert!(reflect_left_one(&[1.], 1, 1).is_err());
    }
    #[test]
    fn waveform_keeps_centered_istft_length_and_rejects_invalid_shapes() {
        let frames = 121;
        assert_eq!(
            waveform(&vec![0.0; 18 * frames], frames).unwrap().len(),
            480
        );
        assert!(waveform(&[], 0).is_err());
        assert!(validate_mel(&[0.; 80], 1, false).is_err());
        assert!(validate_mel(&[f32::NAN; 80], 1, true).is_err());
    }
    #[test]
    fn unvoiced_excitation_uses_uniform_noise_and_no_sine() {
        let noise = vec![0.6; 480 * 9];
        let result = excitation(&[0.], &noise).unwrap();
        assert!(result.iter().all(|v| (*v - 0.02).abs() < 1e-7));
        assert!(excitation(&[1.], &[]).is_err());
    }
}
