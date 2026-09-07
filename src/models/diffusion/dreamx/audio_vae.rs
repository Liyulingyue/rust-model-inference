use std::sync::Arc;

use super::kernels::{checked_len, dot_bf16_f32, load_float_values};
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;

const PREFIX: &str = "dreamx.audio_vae";
const SAMPLE_RATE: usize = 48_000;
const LATENT_CHANNELS: usize = 128;
const DECODER_CHANNELS: usize = 2048;
const RATES: [usize; 5] = [8, 5, 4, 3, 2];
const HOP_LENGTH: usize = 960;

struct Conv1 {
    weight: String,
    weight_shape: [usize; 3],
    bias: Vec<f32>,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv1 {
    #[allow(clippy::too_many_arguments)]
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self, String> {
        let weight = format!("{PREFIX}.{prefix}.weight");
        let weight_shape = [output_channels, input_channels, kernel];
        validate_bf16(source, &weight, &weight_shape)?;
        Ok(Self {
            weight,
            weight_shape,
            bias: load_values(
                source,
                &format!("{PREFIX}.{prefix}.bias"),
                &[output_channels],
            )?,
            stride,
            padding,
            dilation,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &[f32],
        input_channels: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        conv1d_bf16(
            pool,
            input,
            input_channels,
            tensor_bytes(source, &self.weight)?,
            self.weight_shape,
            Some(&self.bias),
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}

struct ConvTranspose1 {
    weight: String,
    weight_shape: [usize; 3],
    bias: Vec<f32>,
    stride: usize,
    padding: usize,
    output_padding: usize,
}

impl ConvTranspose1 {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        stride: usize,
    ) -> Result<Self, String> {
        let weight = format!("{PREFIX}.{prefix}.weight");
        let weight_shape = [input_channels, output_channels, stride * 2];
        validate_bf16(source, &weight, &weight_shape)?;
        Ok(Self {
            weight,
            weight_shape,
            bias: load_values(
                source,
                &format!("{PREFIX}.{prefix}.bias"),
                &[output_channels],
            )?,
            stride,
            padding: stride.div_ceil(2),
            output_padding: stride % 2,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: &[f32],
        input_channels: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        conv_transpose1d_bf16(
            pool,
            input,
            input_channels,
            tensor_bytes(source, &self.weight)?,
            self.weight_shape,
            Some(&self.bias),
            self.stride,
            self.padding,
            self.output_padding,
        )
    }
}

struct Snake {
    alpha: Vec<f32>,
}

impl Snake {
    fn load(source: &dyn TensorSource, prefix: &str, channels: usize) -> Result<Self, String> {
        Ok(Self {
            alpha: load_values(
                source,
                &format!("{PREFIX}.{prefix}.alpha"),
                &[1, channels, 1],
            )?,
        })
    }

    fn forward(&self, values: &mut [f32], channels: usize) -> Result<(), String> {
        if channels != self.alpha.len() || !values.len().is_multiple_of(channels) {
            return Err("Invalid DreamX Snake tensor".into());
        }
        let length = values.len() / channels;
        for channel in 0..channels {
            let alpha = self.alpha[channel];
            for value in &mut values[channel * length..(channel + 1) * length] {
                let sine = (alpha * *value).sin();
                *value += sine * sine / (alpha + 1e-9);
            }
        }
        Ok(())
    }
}

struct ResidualUnit {
    snake1: Snake,
    conv1: Conv1,
    snake2: Snake,
    conv2: Conv1,
}

impl ResidualUnit {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        channels: usize,
        dilation: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            snake1: Snake::load(source, &format!("{prefix}.block.0"), channels)?,
            conv1: Conv1::load(
                source,
                &format!("{prefix}.block.1"),
                channels,
                channels,
                7,
                1,
                3 * dilation,
                dilation,
            )?,
            snake2: Snake::load(source, &format!("{prefix}.block.2"), channels)?,
            conv2: Conv1::load(
                source,
                &format!("{prefix}.block.3"),
                channels,
                channels,
                1,
                1,
                0,
                1,
            )?,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        input: Vec<f32>,
        channels: usize,
    ) -> Result<Vec<f32>, String> {
        let mut residual = input.clone();
        self.snake1.forward(&mut residual, channels)?;
        let (mut residual, length) = self.conv1.forward(source, pool, &residual, channels)?;
        self.snake2.forward(&mut residual, channels)?;
        let (mut residual, output_length) =
            self.conv2.forward(source, pool, &residual, channels)?;
        if length != output_length || residual.len() != input.len() {
            return Err("DreamX DAC residual unit changed tensor length".into());
        }
        for (output, input) in residual.iter_mut().zip(input) {
            *output += input;
        }
        Ok(residual)
    }
}

struct DecoderBlock {
    input_channels: usize,
    output_channels: usize,
    snake: Snake,
    upsample: ConvTranspose1,
    residuals: Vec<ResidualUnit>,
}

impl DecoderBlock {
    fn load(
        source: &dyn TensorSource,
        index: usize,
        input_channels: usize,
        output_channels: usize,
        stride: usize,
    ) -> Result<Self, String> {
        let prefix = format!("decoder.model.{}", index + 1);
        let mut residuals = Vec::with_capacity(3);
        for (block, dilation) in [1, 3, 9].into_iter().enumerate() {
            residuals.push(ResidualUnit::load(
                source,
                &format!("{prefix}.block.{}", block + 2),
                output_channels,
                dilation,
            )?);
        }
        Ok(Self {
            input_channels,
            output_channels,
            snake: Snake::load(source, &format!("{prefix}.block.0"), input_channels)?,
            upsample: ConvTranspose1::load(
                source,
                &format!("{prefix}.block.1"),
                input_channels,
                output_channels,
                stride,
            )?,
            residuals,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        mut input: Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        self.snake.forward(&mut input, self.input_channels)?;
        let (mut output, _) = self
            .upsample
            .forward(source, pool, &input, self.input_channels)?;
        for residual in &self.residuals {
            output = residual.forward(source, pool, output, self.output_channels)?;
        }
        Ok(output)
    }
}

struct DacDecoder {
    post_quant: Conv1,
    input: Conv1,
    blocks: Vec<DecoderBlock>,
    output_snake: Snake,
    output: Conv1,
}

impl DacDecoder {
    fn load(source: &dyn TensorSource) -> Result<Self, String> {
        let post_quant = Conv1::load(
            source,
            "post_quant_conv",
            LATENT_CHANNELS,
            LATENT_CHANNELS,
            1,
            1,
            0,
            1,
        )?;
        let input = Conv1::load(
            source,
            "decoder.model.0",
            LATENT_CHANNELS,
            DECODER_CHANNELS,
            7,
            1,
            3,
            1,
        )?;
        let mut blocks = Vec::with_capacity(RATES.len());
        for (index, stride) in RATES.into_iter().enumerate() {
            let input_channels = DECODER_CHANNELS >> index;
            blocks.push(DecoderBlock::load(
                source,
                index,
                input_channels,
                input_channels / 2,
                stride,
            )?);
        }
        Ok(Self {
            post_quant,
            input,
            blocks,
            output_snake: Snake::load(source, "decoder.model.6", 64)?,
            output: Conv1::load(source, "decoder.model.7", 64, 1, 7, 1, 3, 1)?,
        })
    }

    fn forward(
        &self,
        source: &dyn TensorSource,
        pool: &ComputePool,
        latent: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, String> {
        let (input, length) = self
            .post_quant
            .forward(source, pool, latent, LATENT_CHANNELS)?;
        if length != frames {
            return Err("DreamX DAC post-quant convolution changed latent length".into());
        }
        let (mut output, length) = self.input.forward(source, pool, &input, LATENT_CHANNELS)?;
        if length != frames {
            return Err("DreamX DAC input convolution changed latent length".into());
        }
        for block in &self.blocks {
            output = block.forward(source, pool, output)?;
        }
        self.output_snake.forward(&mut output, 64)?;
        let (mut waveform, samples) = self.output.forward(source, pool, &output, 64)?;
        let expected = frames
            .checked_mul(HOP_LENGTH)
            .ok_or("DreamX DAC waveform length overflow")?;
        if samples != expected || waveform.len() != expected {
            return Err(format!(
                "DreamX DAC output length mismatch: expected {expected}, got {samples}"
            ));
        }
        for sample in &mut waveform {
            *sample = sample.tanh().clamp(-1.0, 1.0);
        }
        Ok(waveform)
    }
}

pub struct CreatorDacVae {
    source: Option<Arc<dyn TensorSource>>,
    pool: Arc<ComputePool>,
    decoder: Option<DacDecoder>,
}

impl CreatorDacVae {
    pub fn load(source: Arc<dyn TensorSource>, pool: Arc<ComputePool>) -> Result<Self, String> {
        let decoder = DacDecoder::load(source.as_ref())?;
        Ok(Self {
            source: Some(source),
            pool,
            decoder: Some(decoder),
        })
    }

    pub fn latent_frames(duration_seconds: f32) -> Result<usize, String> {
        if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
            return Err("DreamX audio duration must be finite and positive".into());
        }
        let samples = (duration_seconds as f64 * SAMPLE_RATE as f64).ceil();
        if samples > usize::MAX as f64 {
            return Err("DreamX audio duration is too large".into());
        }
        Ok((samples as usize).div_ceil(HOP_LENGTH))
    }

    pub fn decode(&self, latent: &[f32], frames: usize) -> Result<Vec<f32>, String> {
        if frames == 0
            || latent.len()
                != frames
                    .checked_mul(LATENT_CHANNELS)
                    .ok_or("DreamX audio latent length overflow")?
            || latent.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid DreamX audio latent".into());
        }
        #[cfg(test)]
        if self.decoder.is_none() {
            return Ok(vec![0.0; frames * HOP_LENGTH]);
        }
        self.decoder
            .as_ref()
            .ok_or("DreamX DAC decoder is unavailable")?
            .forward(
                self.source
                    .as_deref()
                    .ok_or("DreamX DAC tensor source is unavailable")?,
                &self.pool,
                latent,
                frames,
            )
    }

    #[cfg(test)]
    fn testing() -> Self {
        Self {
            source: None,
            pool: Arc::new(ComputePool::new(1)),
            decoder: None,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn conv1d_bf16(
    pool: &ComputePool,
    input: &[f32],
    input_channels: usize,
    weight: &[u8],
    weight_shape: [usize; 3],
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<(Vec<f32>, usize), String> {
    let [output_channels, weight_channels, kernel] = weight_shape;
    if input_channels == 0
        || output_channels == 0
        || weight_channels != input_channels
        || stride == 0
        || dilation == 0
        || !input.len().is_multiple_of(input_channels)
        || weight.len() != checked_len("DreamX BF16 Conv1D weight", &weight_shape)? * 2
        || bias.is_some_and(|bias| bias.len() != output_channels)
    {
        return Err("Invalid DreamX BF16 Conv1D tensors".into());
    }
    let input_length = input.len() / input_channels;
    let effective_kernel = dilation
        .checked_mul(kernel - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or("DreamX BF16 Conv1D kernel overflow")?;
    let padded = input_length
        .checked_add(padding * 2)
        .ok_or("DreamX BF16 Conv1D padding overflow")?;
    if padded < effective_kernel {
        return Err("DreamX BF16 Conv1D kernel exceeds input".into());
    }
    let output_length = (padded - effective_kernel) / stride + 1;
    let output_len = output_channels
        .checked_mul(output_length)
        .ok_or("DreamX BF16 Conv1D output overflow")?;
    let mut output = vec![0.0; output_len];
    let output_address = output.as_mut_ptr() as usize;
    pool.compute(|thread, threads| {
        for output_index in (thread..output_len).step_by(threads) {
            let position = output_index % output_length;
            let output_channel = output_index / output_length;
            let mut sum = bias.map_or(0.0, |bias| bias[output_channel]);
            for input_channel in 0..input_channels {
                let input_row = input_channel * input_length;
                let weight_row = (output_channel * input_channels + input_channel) * kernel * 2;
                if dilation == 1 {
                    let base = position * stride;
                    let first = padding.saturating_sub(base).min(kernel);
                    let last =
                        kernel.min(input_length.saturating_add(padding).saturating_sub(base));
                    if first < last {
                        let input_start = input_row + base + first - padding;
                        sum += dot_bf16_f32(
                            &weight[weight_row + first * 2..weight_row + last * 2],
                            &input[input_start..input_start + last - first],
                        );
                    }
                } else {
                    for kernel_index in 0..kernel {
                        let padded_index = position * stride + kernel_index * dilation;
                        if padded_index >= padding {
                            let input_index = padded_index - padding;
                            if input_index < input_length {
                                let offset = weight_row + kernel_index * 2;
                                let value = crate::ops::bf16_to_f32(u16::from_le_bytes([
                                    weight[offset],
                                    weight[offset + 1],
                                ]));
                                sum += input[input_row + input_index] * value;
                            }
                        }
                    }
                }
            }
            unsafe { (output_address as *mut f32).add(output_index).write(sum) };
        }
    });
    Ok((output, output_length))
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_bf16(
    pool: &ComputePool,
    input: &[f32],
    input_channels: usize,
    weight: &[u8],
    weight_shape: [usize; 3],
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> Result<(Vec<f32>, usize), String> {
    let [weight_input_channels, output_channels, kernel] = weight_shape;
    if input_channels == 0
        || output_channels == 0
        || weight_input_channels != input_channels
        || stride == 0
        || output_padding >= stride
        || !input.len().is_multiple_of(input_channels)
        || weight.len() != checked_len("DreamX BF16 ConvTranspose1D weight", &weight_shape)? * 2
        || bias.is_some_and(|bias| bias.len() != output_channels)
    {
        return Err("Invalid DreamX BF16 ConvTranspose1D tensors".into());
    }
    let input_length = input.len() / input_channels;
    let output_length = (input_length - 1)
        .checked_mul(stride)
        .and_then(|value| value.checked_add(kernel + output_padding + 1))
        .and_then(|value| value.checked_sub(padding * 2 + 1))
        .ok_or("DreamX BF16 ConvTranspose1D output overflow")?;
    let output_len = output_channels
        .checked_mul(output_length)
        .ok_or("DreamX BF16 ConvTranspose1D output overflow")?;
    let mut output = vec![0.0; output_len];
    let output_address = output.as_mut_ptr() as usize;
    pool.compute(|thread, threads| {
        for output_index in (thread..output_len).step_by(threads) {
            let position = output_index % output_length;
            let output_channel = output_index / output_length;
            let mut sum = bias.map_or(0.0, |bias| bias[output_channel]);
            for input_channel in 0..input_channels {
                let weight_row = (input_channel * output_channels + output_channel) * kernel * 2;
                for kernel_index in 0..kernel {
                    let source = position + padding;
                    if source >= kernel_index && (source - kernel_index).is_multiple_of(stride) {
                        let input_position = (source - kernel_index) / stride;
                        if input_position < input_length {
                            let offset = weight_row + kernel_index * 2;
                            let value = crate::ops::bf16_to_f32(u16::from_le_bytes([
                                weight[offset],
                                weight[offset + 1],
                            ]));
                            sum += input[input_channel * input_length + input_position] * value;
                        }
                    }
                }
            }
            unsafe { (output_address as *mut f32).add(output_index).write(sum) };
        }
    });
    Ok((output, output_length))
}

fn validate_bf16(
    source: &dyn TensorSource,
    name: &str,
    source_shape: &[usize],
) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let expected: Vec<u64> = source_shape
        .iter()
        .rev()
        .map(|&value| value as u64)
        .collect();
    if info.dims != expected || info.ggml_type != GGMLType::BF16 {
        return Err(format!(
            "Invalid tensor {name}: shape {:?}, type {:?}; expected {:?}, BF16",
            info.dims, info.ggml_type, expected
        ));
    }
    let bytes = tensor_bytes(source, name)?;
    if bytes.len() != checked_len("DreamX BF16 tensor", source_shape)? * 2 {
        return Err(format!("Invalid tensor data length for {name}"));
    }
    Ok(())
}

fn tensor_bytes<'a>(source: &'a dyn TensorSource, name: &str) -> Result<&'a [u8], String> {
    source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))
}

fn load_values(
    source: &dyn TensorSource,
    name: &str,
    source_shape: &[usize],
) -> Result<Vec<f32>, String> {
    let expected: Vec<u64> = source_shape
        .iter()
        .rev()
        .map(|&value| value as u64)
        .collect();
    load_float_values(source, name, &expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn five_seconds_maps_to_250_audio_latent_frames() {
        assert_eq!(CreatorDacVae::latent_frames(5.0).unwrap(), 250);
    }

    #[test]
    fn tiny_decoder_produces_hop_length_samples_per_frame() {
        let waveform = CreatorDacVae::testing()
            .decode(&vec![0.0; 128 * 2], 2)
            .unwrap();
        assert_eq!(waveform.len(), 2 * 960);
    }

    #[test]
    fn bf16_convolutions_use_pytorch_weight_layout() {
        let pool = ComputePool::new(2);
        let (convolved, length) = conv1d_bf16(
            &pool,
            &[1.0, 2.0, 3.0],
            1,
            &bf16(&[1.0, 10.0]),
            [1, 1, 2],
            Some(&[0.5]),
            1,
            0,
            1,
        )
        .unwrap();
        assert_eq!(length, 2);
        assert_eq!(convolved, [21.5, 32.5]);

        let (upsampled, length) = conv_transpose1d_bf16(
            &pool,
            &[1.0, 2.0],
            1,
            &bf16(&[1.0, 10.0, 100.0]),
            [1, 1, 3],
            None,
            2,
            1,
            1,
        )
        .unwrap();
        assert_eq!(length, 4);
        assert_eq!(upsampled, [10.0, 102.0, 20.0, 200.0]);
    }
}
