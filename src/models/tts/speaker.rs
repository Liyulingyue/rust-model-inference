use crate::core::tensor::TensorSource;
use crate::models::qwen3::load_f32_tensor;
use crate::models::qwen3a::audio_processor::{decode_pcm16_wav_any, RealFft};

use super::load_f16_or_f32_tensor;

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

fn conv1d_same(
    input: &[f32],
    in_channels: usize,
    frames: usize,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    dilation: usize,
    kernel: usize,
) -> Result<Vec<f32>, String> {
    if kernel == 0 || kernel % 2 == 0 || dilation == 0 {
        return Err("speaker convolution requires an odd kernel and nonzero dilation".into());
    }
    if input.len()
        != in_channels
            .checked_mul(frames)
            .ok_or("speaker input size overflow")?
        || weight.len()
            != kernel
                .checked_mul(in_channels)
                .and_then(|value| value.checked_mul(out_channels))
                .ok_or("speaker weight size overflow")?
        || bias.len() != out_channels
    {
        return Err("speaker convolution shape mismatch".into());
    }
    let pad = (kernel - 1)
        .checked_mul(dilation)
        .and_then(|value| value.checked_div(2))
        .ok_or("speaker convolution padding overflow")?;
    if pad >= frames {
        return Err("speaker convolution padding must be smaller than the frame count".into());
    }
    let padded_frames = frames
        .checked_add(pad.checked_mul(2).ok_or("speaker padding size overflow")?)
        .ok_or("speaker padding size overflow")?;
    let mut padded = vec![0.0; in_channels * padded_frames];
    for channel in 0..in_channels {
        let source = &input[channel * frames..(channel + 1) * frames];
        let reflected = reflect_pad(source, pad)?;
        padded[channel * padded_frames..(channel + 1) * padded_frames].copy_from_slice(&reflected);
    }
    let mut output = vec![0.0; out_channels * frames];
    for out_channel in 0..out_channels {
        for frame in 0..frames {
            let mut sum = bias[out_channel];
            for in_channel in 0..in_channels {
                for tap in 0..kernel {
                    let weight_index =
                        (tap * in_channels + in_channel) * out_channels + out_channel;
                    let input_index = in_channel * padded_frames + frame + tap * dilation;
                    sum = weight[weight_index].mul_add(padded[input_index], sum);
                }
            }
            output[out_channel * frames + frame] = sum;
        }
    }
    Ok(output)
}

fn res2net_chain<F>(chunks: Vec<Vec<f32>>, mut process: F) -> Result<Vec<f32>, String>
where
    F: FnMut(usize, &[f32]) -> Result<Vec<f32>, String>,
{
    let mut chunks = chunks.into_iter();
    let first = chunks
        .next()
        .ok_or("Res2Net requires at least two chunks")?;
    let mut output = first;
    let mut previous: Option<Vec<f32>> = None;
    for (index, mut chunk) in chunks.enumerate() {
        if let Some(previous) = &previous {
            if previous.len() != chunk.len() {
                return Err("Res2Net chunk shape mismatch".into());
            }
            for (value, add) in chunk.iter_mut().zip(previous) {
                *value += *add;
            }
        }
        let processed = process(index, &chunk)?;
        if processed.len() != chunk.len() {
            return Err("Res2Net convolution changed the chunk shape".into());
        }
        output.extend_from_slice(&processed);
        previous = Some(processed);
    }
    Ok(output)
}

#[cfg(test)]
fn res2net_chain_for_test<F>(chunks: Vec<Vec<f32>>, mut process: F) -> Vec<f32>
where
    F: FnMut(&[f32]) -> Vec<f32>,
{
    res2net_chain(chunks, |_, input| Ok(process(input))).unwrap()
}

fn weighted_stats(
    input: &[f32],
    weights: &[f32],
    channels: usize,
    frames: usize,
) -> Result<Vec<f32>, String> {
    let expected = channels
        .checked_mul(frames)
        .ok_or("speaker statistics size overflow")?;
    if frames == 0 || input.len() != expected || weights.len() != expected {
        return Err("speaker statistics shape mismatch".into());
    }
    let mut output = vec![0.0; channels * 2];
    for channel in 0..channels {
        let values = &input[channel * frames..(channel + 1) * frames];
        let weights = &weights[channel * frames..(channel + 1) * frames];
        let mean = values
            .iter()
            .zip(weights)
            .map(|(value, weight)| f64::from(*value) * f64::from(*weight))
            .sum::<f64>();
        let second = values
            .iter()
            .zip(weights)
            .map(|(value, weight)| {
                let value = f64::from(*value);
                value * value * f64::from(*weight)
            })
            .sum::<f64>();
        output[channel] = mean as f32;
        output[channels + channel] = (second - mean * mean).max(1e-12).sqrt() as f32;
    }
    Ok(output)
}

struct SpeakerConv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    dilation: usize,
}

impl SpeakerConv {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        kernel: usize,
        in_channels: usize,
        out_channels: usize,
        dilation: usize,
    ) -> Result<Self, String> {
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");
        Ok(Self {
            weight: load_f16_or_f32_tensor(
                source,
                &weight_name,
                &[kernel as u64, in_channels as u64, out_channels as u64],
            )?,
            bias: load_f32_tensor(source, &bias_name, &[out_channels as u64])?,
            in_channels,
            out_channels,
            kernel,
            dilation,
        })
    }

    fn forward(&self, input: &[f32], frames: usize) -> Result<Vec<f32>, String> {
        conv1d_same(
            input,
            self.in_channels,
            frames,
            &self.weight,
            &self.bias,
            self.out_channels,
            self.dilation,
            self.kernel,
        )
    }
}

struct SeRes2Block {
    pw1: SpeakerConv,
    res2: Vec<SpeakerConv>,
    pw2: SpeakerConv,
    se1: SpeakerConv,
    se2: SpeakerConv,
}

impl SeRes2Block {
    fn load(source: &dyn TensorSource, index: usize, dilation: usize) -> Result<Self, String> {
        let prefix = format!("a.blk.{index}");
        let mut res2 = Vec::with_capacity(7);
        for subindex in 0..7 {
            res2.push(SpeakerConv::load(
                source,
                &format!("{prefix}.res2.{subindex}"),
                3,
                64,
                64,
                dilation,
            )?);
        }
        Ok(Self {
            pw1: SpeakerConv::load(source, &format!("{prefix}.conv_pw1"), 1, 512, 512, 1)?,
            res2,
            pw2: SpeakerConv::load(source, &format!("{prefix}.conv_pw2"), 1, 512, 512, 1)?,
            se1: SpeakerConv::load(source, &format!("{prefix}.se_conv1"), 1, 512, 128, 1)?,
            se2: SpeakerConv::load(source, &format!("{prefix}.se_conv2"), 1, 128, 512, 1)?,
        })
    }

    fn forward(&self, input: &[f32], frames: usize) -> Result<Vec<f32>, String> {
        if input.len() != 512 * frames {
            return Err("SE-Res2Net input shape mismatch".into());
        }
        let mut projected = self.pw1.forward(input, frames)?;
        relu_inplace(&mut projected);
        let chunks = projected
            .chunks_exact(64 * frames)
            .map(<[f32]>::to_vec)
            .collect();
        let mut res2 = res2net_chain(chunks, |index, chunk| {
            let mut output = self.res2[index].forward(chunk, frames)?;
            relu_inplace(&mut output);
            Ok(output)
        })?;
        res2 = self.pw2.forward(&res2, frames)?;
        relu_inplace(&mut res2);

        let mut pooled = vec![0.0; 512];
        for channel in 0..512 {
            pooled[channel] = (res2[channel * frames..(channel + 1) * frames]
                .iter()
                .map(|value| f64::from(*value))
                .sum::<f64>()
                / frames as f64) as f32;
        }
        let mut scale = self.se1.forward(&pooled, 1)?;
        relu_inplace(&mut scale);
        scale = self.se2.forward(&scale, 1)?;
        for value in &mut scale {
            *value = 1.0 / (1.0 + (-*value).exp());
        }
        let mut output = input.to_vec();
        for channel in 0..512 {
            for frame in 0..frames {
                output[channel * frames + frame] += res2[channel * frames + frame] * scale[channel];
            }
        }
        Ok(output)
    }
}

fn relu_inplace(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

pub struct Qwen3TtsSpeakerEncoder {
    stem: SpeakerConv,
    blocks: Vec<SeRes2Block>,
    mfa: SpeakerConv,
    asp_tdnn: SpeakerConv,
    asp_attn: SpeakerConv,
    projection: SpeakerConv,
}

impl Qwen3TtsSpeakerEncoder {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        Ok(Self {
            stem: SpeakerConv::load(source, "a.conv1d.0", 5, 128, 512, 1)?,
            blocks: vec![
                SeRes2Block::load(source, 1, 2)?,
                SeRes2Block::load(source, 2, 3)?,
                SeRes2Block::load(source, 3, 4)?,
            ],
            mfa: SpeakerConv::load(source, "a.conv_out", 1, 1536, 1536, 1)?,
            asp_tdnn: SpeakerConv::load(source, "a.asp_tdnn", 1, 4608, 128, 1)?,
            asp_attn: SpeakerConv::load(source, "a.asp_attn", 1, 128, 1536, 1)?,
            projection: SpeakerConv::load(source, "mm.a.fc", 1, 3072, 2048, 1)?,
        })
    }

    pub fn encode(&self, mel: &SpeakerMel) -> Result<Vec<f32>, String> {
        if mel.frames == 0 || mel.values.len() != 128 * mel.frames {
            return Err("speaker Mel shape mismatch".into());
        }
        let mut current = self.stem.forward(&mel.values, mel.frames)?;
        relu_inplace(&mut current);
        let mut block_outputs = Vec::with_capacity(1536 * mel.frames);
        for block in &self.blocks {
            current = block.forward(&current, mel.frames)?;
            block_outputs.extend_from_slice(&current);
        }
        let mut mfa = self.mfa.forward(&block_outputs, mel.frames)?;
        relu_inplace(&mut mfa);

        let mut context = vec![0.0; 4608 * mel.frames];
        for channel in 0..1536 {
            let row = &mfa[channel * mel.frames..(channel + 1) * mel.frames];
            let mean = row.iter().map(|value| f64::from(*value)).sum::<f64>() / mel.frames as f64;
            let variance = row
                .iter()
                .map(|value| {
                    let difference = f64::from(*value) - mean;
                    difference * difference
                })
                .sum::<f64>()
                / mel.frames as f64;
            let std = variance.max(1e-12).sqrt() as f32;
            context[channel * mel.frames..(channel + 1) * mel.frames].copy_from_slice(row);
            context[(1536 + channel) * mel.frames..(1537 + channel) * mel.frames].fill(mean as f32);
            context[(3072 + channel) * mel.frames..(3073 + channel) * mel.frames].fill(std);
        }
        let mut attention = self.asp_tdnn.forward(&context, mel.frames)?;
        for value in &mut attention {
            *value = value.tanh();
        }
        attention = self.asp_attn.forward(&attention, mel.frames)?;
        softmax_rows(&mut attention, 1536, mel.frames)?;
        let statistics = weighted_stats(&mfa, &attention, 1536, mel.frames)?;
        let embedding = self.projection.forward(&statistics, 1)?;
        if embedding.len() != 2048 || embedding.iter().any(|value| !value.is_finite()) {
            return Err("speaker embedding must contain 2048 finite values".into());
        }
        Ok(embedding)
    }
}

fn softmax_rows(values: &mut [f32], rows: usize, columns: usize) -> Result<(), String> {
    if columns == 0 || values.len() != rows * columns {
        return Err("speaker attention shape mismatch".into());
    }
    for row in values.chunks_exact_mut(columns) {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !maximum.is_finite() {
            return Err("non-finite speaker attention".into());
        }
        let mut sum = 0.0f64;
        for value in row.iter_mut() {
            *value = (*value - maximum).exp();
            sum += f64::from(*value);
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err("invalid speaker attention normalization".into());
        }
        for value in row {
            *value = (f64::from(*value) / sum) as f32;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::ggufrs::{open_model_source, ComponentRole};
    use std::path::Path;

    fn open_mmproj_from_env() -> Box<dyn TensorSource> {
        let path = std::env::var("QWEN3_TTS_MMPROJ").unwrap();
        open_model_source(Path::new(&path), ComponentRole::Mmproj).unwrap()
    }

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

    #[test]
    fn reflect_same_conv_preserves_time_and_uses_dilation() {
        let output =
            conv1d_same(&[1.0, 2.0, 3.0], 1, 3, &[1.0, 1.0, 1.0], &[0.0], 1, 1, 3).unwrap();
        assert_eq!(output, vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn res2net_chains_only_after_the_first_processed_chunk() {
        let chunks = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let output = res2net_chain_for_test(chunks, |input| {
            input.iter().map(|value| value * 10.0).collect()
        });
        assert_eq!(output, vec![1.0, 20.0, 230.0, 2340.0]);
    }

    #[test]
    fn attentive_statistics_returns_weighted_mean_then_std() {
        let x = vec![1.0, 3.0, 2.0, 4.0];
        let weights = vec![0.25, 0.75, 0.5, 0.5];
        let stats = weighted_stats(&x, &weights, 2, 2).unwrap();
        assert_eq!(stats.len(), 4);
        assert!((stats[0] - 2.5).abs() < 1e-6);
        assert!((stats[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires QWEN3_TTS_MMPROJ and QWEN3_TTS_REF_WAV"]
    fn local_speaker_encoder_returns_one_talker_row() {
        let source = open_mmproj_from_env();
        let wav = std::fs::read(std::env::var("QWEN3_TTS_REF_WAV").unwrap()).unwrap();
        let mel = reference_wav_to_mel(&wav).unwrap();
        let embedding = Qwen3TtsSpeakerEncoder::from_source(source.as_ref())
            .unwrap()
            .encode(&mel)
            .unwrap();
        assert_eq!(embedding.len(), 2048);
        assert!(embedding.iter().all(|value| value.is_finite()));
    }
}
