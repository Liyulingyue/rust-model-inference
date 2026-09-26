use std::sync::Arc;

use rayon::prelude::*;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::ops::math::torch28_exp;

use super::config::YuE2VaeConfig;

struct Snake {
    alpha: Vec<f32>,
    beta: Vec<f32>,
}

#[inline(never)]
fn snake_beta_scalar(value: f32, alpha: f32, beta: f32) -> f32 {
    let sine = crate::ops::rope::rope_sin_cos_sleef(value * alpha).1;
    value + (1.0 / (beta + 1e-9)) * (sine * sine)
}

impl Snake {
    fn load(source: &dyn TensorSource, prefix: &str, channels: usize) -> Result<Self, String> {
        let alpha = f32_tensor(source, &format!("{prefix}.alpha"), &[channels])?
            .into_iter()
            .map(torch28_exp)
            .collect();
        let beta = f32_tensor(source, &format!("{prefix}.beta"), &[channels])?
            .into_iter()
            .map(torch28_exp)
            .collect();
        Ok(Self { alpha, beta })
    }

    fn forward(&self, values: &mut [f32], frames: usize) {
        for (channel, row) in values.chunks_exact_mut(frames).enumerate() {
            let a = self.alpha[channel];
            let b = self.beta[channel];
            for x in row {
                *x = snake_beta_scalar(*x, a, b);
            }
        }
    }
}

struct Conv {
    weights: Vec<f32>, // [output, input, kernel] or [input, output, kernel]
    bias: Vec<f32>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    transpose: bool,
}

fn torch_large_f32_dot(input: &[f32], weights: &[f32]) -> f32 {
    debug_assert_eq!(input.len(), weights.len());
    debug_assert_eq!(input.len() % 64, 0);
    let mut accumulators = [0.0f32; 64];
    for (input, weights) in input.chunks_exact(64).zip(weights.chunks_exact(64)) {
        for lane in 0..64 {
            accumulators[lane] = input[lane].mul_add(weights[lane], accumulators[lane]);
        }
    }
    let mut groups = [0.0f32; 16];
    for lane in 0..16 {
        groups[lane] = ((accumulators[lane] + accumulators[lane + 16]) + accumulators[lane + 32])
            + accumulators[lane + 48];
    }
    let mut quarters = [0.0f32; 4];
    for quarter in 0..4 {
        let start = quarter * 4;
        quarters[quarter] =
            ((groups[start] + groups[start + 1]) + groups[start + 2]) + groups[start + 3];
    }
    ((quarters[0] + quarters[1]) + quarters[2]) + quarters[3]
}

impl Conv {
    #[allow(clippy::too_many_arguments)]
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        transpose: bool,
        bias: bool,
    ) -> Result<Self, String> {
        let norm_channels = if transpose { input } else { output };
        let v = f32_tensor(
            source,
            &format!("{prefix}.weight_v"),
            &[
                kernel,
                if transpose { output } else { input },
                norm_channels,
            ],
        )?;
        let g = f32_tensor(
            source,
            &format!("{prefix}.weight_g"),
            &[1, 1, norm_channels],
        )?;
        let weights = materialize_weight_norm(&g, &v, norm_channels, v.len() / norm_channels)
            .map_err(|error| format!("{prefix}.weight_v: {error}"))?;
        let bias = if bias {
            f32_tensor(source, &format!("{prefix}.bias"), &[output])?
        } else {
            vec![0.0; output]
        };
        Ok(Self {
            weights,
            bias,
            input,
            output,
            kernel,
            stride,
            padding,
            dilation,
            transpose,
        })
    }

    fn forward(&self, input: &[f32], frames: usize) -> Result<(Vec<f32>, usize), String> {
        let length = if self.transpose {
            (frames - 1)
                .checked_mul(self.stride)
                .and_then(|v| v.checked_add(self.dilation * (self.kernel - 1) + 1))
                .and_then(|v| v.checked_sub(2 * self.padding))
        } else {
            frames
                .checked_add(2 * self.padding)
                .and_then(|v| v.checked_sub(self.dilation * (self.kernel - 1) + 1))
                .map(|v| v / self.stride + 1)
        }
        .ok_or("YuE2 VAE convolution length overflow")?;
        let mut out = vec![
            0.0;
            self.output
                .checked_mul(length)
                .ok_or("YuE2 VAE output overflow")?
        ];
        if !self.transpose
            && frames == 1
            && length == 1
            && self.input == 64
            && self.output == 2048
            && self.kernel == 7
            && self.stride == 1
            && self.padding == 3
            && self.dilation == 1
        {
            let mut weights = [0.0f32; 64];
            for oc in 0..self.output {
                for ic in 0..self.input {
                    weights[ic] = self.weights[(oc * self.input + ic) * self.kernel + 3];
                }
                out[oc] = torch_large_f32_dot(input, &weights) + self.bias[oc];
            }
            return Ok((out, length));
        }
        if self.transpose {
            out.par_chunks_mut(length)
                .enumerate()
                .for_each(|(oc, row)| {
                    for time in 0..length {
                        let mut sum = 0.0;
                        for tap in 0..self.kernel {
                            let offset = time as isize + self.padding as isize
                                - (tap * self.dilation) as isize;
                            if offset < 0 || offset % self.stride as isize != 0 {
                                continue;
                            }
                            let source_time = (offset / self.stride as isize) as usize;
                            if source_time >= frames {
                                continue;
                            }
                            let mut partial = 0.0;
                            for ic in 0..self.input {
                                let index = (ic * self.output + oc) * self.kernel + tap;
                                partial = input[ic * frames + source_time]
                                    .mul_add(self.weights[index], partial);
                            }
                            sum += partial;
                        }
                        row[time] = sum + self.bias[oc];
                    }
                });
            return Ok((out, length));
        }
        // ponytail: re-probe this pinned BLAS tile cutoff if the Oracle wheel or CPU path changes.
        let pointwise_wide = self
            .input
            .saturating_mul(length.saturating_add(self.output))
            >= 1 << 22;
        let bias_block = if (self.kernel == 1 && pointwise_wide)
            || (self.kernel > 1 && (self.output >= 256 || length >= 4_784))
        {
            32
        } else {
            16
        };
        let bias_first = length / bias_block * bias_block;
        out.par_chunks_mut(length)
            .enumerate()
            .for_each(|(oc, row)| {
                for time in 0..length {
                    if self.input * self.kernel > 2048
                        && (self.input * self.kernel > 4096 || bias_first >= 640)
                        && bias_first != 0
                        && time >= bias_first
                    {
                        let mut sum = self.bias[oc];
                        let mut partial = 0.0f32;
                        let mut k = 0;
                        for ic in 0..self.input {
                            for tap in 0..self.kernel {
                                if k != 0 && k % 2048 == 0 {
                                    sum += partial;
                                    partial = 0.0;
                                }
                                let source_time = (time * self.stride + tap * self.dilation)
                                    as isize
                                    - self.padding as isize;
                                if source_time >= 0 && (source_time as usize) < frames {
                                    let index = (oc * self.input + ic) * self.kernel + tap;
                                    partial = input[ic * frames + source_time as usize]
                                        .mul_add(self.weights[index], partial);
                                }
                                k += 1;
                            }
                        }
                        row[time] = sum + partial;
                        continue;
                    }
                    let mut sum = if time < bias_first {
                        self.bias[oc]
                    } else {
                        0.0
                    };
                    for ic in 0..self.input {
                        for tap in 0..self.kernel {
                            let source_time = (time * self.stride + tap * self.dilation) as isize
                                - self.padding as isize;
                            if source_time >= 0 && (source_time as usize) < frames {
                                let index = (oc * self.input + ic) * self.kernel + tap;
                                sum = input[ic * frames + source_time as usize]
                                    .mul_add(self.weights[index], sum);
                            }
                        }
                    }
                    row[time] = if time < bias_first {
                        sum
                    } else {
                        sum + self.bias[oc]
                    };
                }
            });
        Ok((out, length))
    }
}

struct Residual {
    first_snake: Snake,
    first: Conv,
    second_snake: Snake,
    second: Conv,
}

impl Residual {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        channels: usize,
        dilation: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            first_snake: Snake::load(source, &format!("{prefix}.layers.0"), channels)?,
            first: Conv::load(
                source,
                &format!("{prefix}.layers.1"),
                channels,
                channels,
                7,
                1,
                3 * dilation,
                dilation,
                false,
                true,
            )?,
            second_snake: Snake::load(source, &format!("{prefix}.layers.2"), channels)?,
            second: Conv::load(
                source,
                &format!("{prefix}.layers.3"),
                channels,
                channels,
                1,
                1,
                0,
                1,
                false,
                true,
            )?,
        })
    }

    fn forward(&self, values: &mut Vec<f32>, frames: usize) -> Result<(), String> {
        let mut branch = values.clone();
        self.first_snake.forward(&mut branch, frames);
        let (mut branch, length) = self.first.forward(&branch, frames)?;
        self.second_snake.forward(&mut branch, length);
        let (branch, length) = self.second.forward(&branch, length)?;
        if length != frames {
            return Err("YuE2 VAE residual length changed".into());
        }
        for (value, addend) in values.iter_mut().zip(branch) {
            *value += addend;
        }
        Ok(())
    }
}

struct DecoderBlock {
    snake: Snake,
    up: Conv,
    residuals: [Residual; 3],
}

impl DecoderBlock {
    fn load(
        source: &dyn TensorSource,
        index: usize,
        input: usize,
        output: usize,
        stride: usize,
    ) -> Result<Self, String> {
        let prefix = format!("decoder.layers.{index}.layers");
        Ok(Self {
            snake: Snake::load(source, &format!("{prefix}.0"), input)?,
            up: Conv::load(
                source,
                &format!("{prefix}.1"),
                input,
                output,
                stride * 2,
                stride,
                stride.div_ceil(2),
                1,
                true,
                true,
            )?,
            residuals: [1, 3, 9]
                .map(|dilation| {
                    let index = match dilation {
                        1 => 2,
                        3 => 3,
                        _ => 4,
                    };
                    Residual::load(source, &format!("{prefix}.{index}"), output, dilation)
                })
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_| "YuE2 VAE residual count")?,
        })
    }

    fn forward(
        &self,
        mut values: Vec<f32>,
        frames: usize,
        layer: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        self.snake.forward(&mut values, frames);
        trace(
            "yue2.vae.decoder.block",
            Some(layer),
            &[1, self.up.input, frames],
            &values,
        );
        let (mut values, frames) = self.up.forward(&values, frames)?;
        trace(
            "yue2.vae.decoder.block",
            Some(layer),
            &[1, self.up.output, frames],
            &values,
        );
        for residual in &self.residuals {
            residual.forward(&mut values, frames)?;
            trace(
                "yue2.vae.decoder.block",
                Some(layer),
                &[1, self.up.output, frames],
                &values,
            );
        }
        Ok((values, frames))
    }
}

pub struct YuE2Vae {
    config: YuE2VaeConfig,
    input: Conv,
    blocks: Vec<DecoderBlock>,
    final_snake: Snake,
    output: Conv,
}

impl YuE2Vae {
    pub fn from_source(source: Arc<dyn TensorSource>) -> Result<Self, String> {
        let config = YuE2VaeConfig::from_source(source.as_ref())?;
        let widths = [2048, 1024, 512, 256, 128, 64, 64];
        let strides = [6, 5, 4, 4, 2, 2];
        let input = Conv::load(
            source.as_ref(),
            "decoder.layers.0",
            64,
            widths[0],
            7,
            1,
            3,
            1,
            false,
            true,
        )?;
        let mut blocks = Vec::with_capacity(6);
        for index in 0..6 {
            blocks.push(DecoderBlock::load(
                source.as_ref(),
                index + 1,
                widths[index],
                widths[index + 1],
                strides[index],
            )?);
        }
        let final_snake = Snake::load(source.as_ref(), "decoder.layers.7", 64)?;
        let output = Conv::load(
            source.as_ref(),
            "decoder.layers.8",
            64,
            2,
            7,
            1,
            3,
            1,
            false,
            false,
        )?;
        Ok(Self {
            config,
            input,
            blocks,
            final_snake,
            output,
        })
    }

    pub fn decode(&self, latents: &[f32], frames: usize) -> Result<Vec<f32>, String> {
        if frames == 0
            || frames.checked_mul(self.config.latent_channels) != Some(latents.len())
            || latents.iter().any(|v| !v.is_finite())
        {
            return Err("YuE2 VAE expects finite nonempty frame-major [frames,64] latents".into());
        }
        let mut channel_major = vec![0.0; latents.len()];
        for time in 0..frames {
            for channel in 0..self.config.latent_channels {
                channel_major[channel * frames + time] =
                    latents[time * self.config.latent_channels + channel];
            }
        }
        let (mut values, mut length) = self.input.forward(&channel_major, frames)?;
        trace(
            "yue2.vae.decoder",
            Some(0),
            &[1, self.input.output, length],
            &values,
        );
        for (index, block) in self.blocks.iter().enumerate() {
            (values, length) = block.forward(values, length, index + 1)?;
            trace(
                "yue2.vae.decoder",
                Some(index + 1),
                &[1, block.up.output, length],
                &values,
            );
        }
        self.final_snake.forward(&mut values, length);
        trace(
            "yue2.vae.decoder",
            Some(7),
            &[1, self.output.input, length],
            &values,
        );
        let (values, length) = self.output.forward(&values, length)?;
        trace(
            "yue2.vae.decoder",
            Some(8),
            &[1, self.output.output, length],
            &values,
        );
        let expected = frames
            .checked_mul(self.config.ratio)
            .and_then(|n| n.checked_sub(64))
            .ok_or("YuE2 VAE output length overflow")?;
        if length != expected || values.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "YuE2 VAE output length/values invalid: {length}, expected {expected}"
            ));
        }
        Ok(values)
    }

    pub fn decode_tiled(
        &self,
        latents: &[f32],
        frames: usize,
        core: usize,
        halo: usize,
    ) -> Result<Vec<f32>, String> {
        if core == 0
            || halo < self.config.halo
            || frames == 0
            || frames.checked_mul(self.config.latent_channels) != Some(latents.len())
        {
            return Err("YuE2 VAE requires nonempty latents, positive core, and halo >= 16".into());
        }
        let total = frames
            .checked_mul(self.config.ratio)
            .and_then(|n| n.checked_sub(64))
            .ok_or("YuE2 VAE tiled length overflow")?;
        let mut output = vec![
            0.0;
            total
                .checked_mul(2)
                .ok_or("YuE2 VAE tiled output overflow")?
        ];
        for start in (0..frames).step_by(core) {
            let end = frames.min(start.saturating_add(core));
            let left = start.saturating_sub(halo);
            let right = frames.min(end.saturating_add(halo));
            let tile = self.decode(&latents[left * 64..right * 64], right - left)?;
            let start_sample = start * self.config.ratio;
            let end_sample = (end * self.config.ratio).min(total);
            let crop = (start - left) * self.config.ratio;
            for channel in 0..2 {
                let tile_len = tile.len() / 2;
                let count = end_sample - start_sample;
                if crop + count > tile_len {
                    return Err("YuE2 VAE tile does not cover core".into());
                }
                output[channel * total + start_sample..channel * total + end_sample]
                    .copy_from_slice(
                        &tile[channel * tile_len + crop..channel * tile_len + crop + count],
                    );
            }
        }
        Ok(output)
    }

    #[cfg(test)]
    pub(super) fn tiny_for_test() -> Self {
        fn snake(channels: usize) -> Snake {
            Snake {
                alpha: vec![1.0; channels],
                beta: vec![1.0; channels],
            }
        }
        fn conv(
            input: usize,
            output: usize,
            kernel: usize,
            stride: usize,
            padding: usize,
            dilation: usize,
            transpose: bool,
            active: bool,
        ) -> Conv {
            let mut weights = vec![0.0; input * output * kernel];
            if active {
                for channel in 0..output.min(input) {
                    let tap = if transpose { padding } else { kernel / 2 };
                    weights[(channel * if transpose { output } else { input } + channel)
                        * kernel
                        + tap] = 0.5;
                }
            }
            Conv {
                weights,
                bias: vec![0.0; output],
                input,
                output,
                kernel,
                stride,
                padding,
                dilation,
                transpose,
            }
        }
        let residual = || Residual {
            first_snake: snake(2),
            first: conv(2, 2, 7, 1, 3, 1, false, false),
            second_snake: snake(2),
            second: conv(2, 2, 1, 1, 0, 1, false, false),
        };
        Self {
            config: YuE2VaeConfig {
                strides: [2, 2, 4, 4, 5, 6],
                latent_channels: 64,
                output_channels: 2,
                sample_rate: 48_000,
                ratio: 1920,
                core: 1024,
                halo: 16,
            },
            input: conv(64, 2, 7, 1, 3, 1, false, true),
            blocks: [6, 5, 4, 4, 2, 2]
                .into_iter()
                .map(|stride| DecoderBlock {
                    snake: snake(2),
                    up: conv(2, 2, 2 * stride, stride, stride.div_ceil(2), 1, true, true),
                    residuals: [residual(), residual(), residual()],
                })
                .collect(),
            final_snake: snake(2),
            output: conv(2, 2, 7, 1, 3, 1, false, true),
        }
    }
}

fn trace(name: &str, layer: Option<usize>, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint_at(
        name, layer, None, shape, values,
    ));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, shape, values);
}

pub(super) fn materialize_weight_norm(
    g: &[f32],
    v: &[f32],
    channels: usize,
    per_channel: usize,
) -> Result<Vec<f32>, String> {
    if channels == 0
        || per_channel == 0
        || g.len() != channels
        || v.len() != channels.checked_mul(per_channel).unwrap_or(0)
    {
        return Err("invalid weight norm shape".into());
    }
    let mut weights = Vec::with_capacity(v.len());
    for (channel, row) in v.chunks_exact(per_channel).enumerate() {
        let mut lanes = [0.0f32; 4];
        for chunk in row.chunks(4) {
            for (lane, &value) in chunk.iter().enumerate() {
                lanes[lane] += value * value;
            }
        }
        let norm = ((lanes[0] + lanes[2]) + (lanes[1] + lanes[3])).sqrt();
        if norm == 0.0 || !norm.is_finite() || !g[channel].is_finite() {
            return Err(format!("invalid weight norm channel {channel}"));
        }
        let scale = g[channel] / norm;
        weights.extend(row.iter().map(|value| value * scale));
    }
    if weights.iter().any(|v| !v.is_finite()) {
        return Err("non-finite materialized weight".into());
    }
    Ok(weights)
}

fn f32_tensor(source: &dyn TensorSource, name: &str, dims: &[usize]) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let dims = dims.iter().map(|&n| n as u64).collect::<Vec<_>>();
    if info.ggml_type != GGMLType::F32 || info.dims != dims {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {dims:?} F32",
            info.dims, info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let expected = info
        .checked_nbytes()
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
    if bytes.len() as u64 != expected {
        return Err(format!("Invalid tensor data length: {name}"));
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(format!("Non-finite tensor: {name}"));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{torch28_exp, torch_large_f32_dot, Conv, Residual, Snake};

    fn assert_saved_residual_matches_oracle_bits(
        layer: usize,
        occurrence: usize,
        channels: usize,
        frames: usize,
    ) {
        let oracle_trace = std::path::PathBuf::from(
            std::env::var("YUE2_E2E_ORACLE_TRACE").expect("YUE2_E2E_ORACLE_TRACE"),
        );
        let oracle_dir = oracle_trace.parent().unwrap();
        let read_f32 = |name: &str| {
            std::fs::read(oracle_dir.join(name))
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let mut actual = read_f32(&format!("yue2.vae.decoder.block.{occurrence}.f32"));
        let expected = read_f32(&format!("yue2.vae.decoder.block.{}.f32", occurrence + 1));
        assert_eq!(actual.len(), channels * frames);
        assert_eq!(expected.len(), actual.len());
        let loader =
            crate::GGUFLoader::from_file(std::env::var("YUE2_VAE_GGUF").expect("YUE2_VAE_GGUF"))
                .unwrap();
        Residual::load(
            &loader,
            &format!("decoder.layers.{layer}.layers.2"),
            channels,
            1,
        )
        .unwrap()
        .forward(&mut actual, frames)
        .unwrap();
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "layer {layer} first residual element {index}: Rust={:#010x} Oracle={:#010x}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    #[ignore = "requires fixed YuE2 VAE GGUF and pinned real E2E Oracle"]
    fn saved_200_frame_first_residual_matches_oracle_bits() {
        assert_saved_residual_matches_oracle_bits(1, 1, 1024, 1200);
    }

    #[test]
    #[ignore = "requires fixed YuE2 VAE GGUF and pinned real E2E Oracle"]
    fn saved_200_frame_third_block_residual_matches_oracle_bits() {
        assert_saved_residual_matches_oracle_bits(3, 11, 256, 23_996);
    }

    #[test]
    #[ignore = "requires fixed YuE2 VAE GGUF and pinned real E2E Oracle"]
    fn saved_200_frame_fourth_block_residual_matches_oracle_bits() {
        assert_saved_residual_matches_oracle_bits(4, 16, 128, 95_984);
    }

    #[test]
    fn snake_parameter_exp_matches_torch_bits() {
        assert_eq!(
            torch28_exp(f32::from_bits(0x3e066672)).to_bits(),
            0x3f91f3d0
        );
        assert_eq!(
            torch28_exp(f32::from_bits(0xbec869fd)).to_bits(),
            0x3f2d1408
        );
    }

    #[test]
    fn snake_forward_matches_torch_bits_across_channels() {
        let mut alpha = vec![1.0; 345];
        let mut beta = vec![1.0; 345];
        alpha[344] = torch28_exp(f32::from_bits(0x3dcf8201));
        beta[344] = torch28_exp(f32::from_bits(0xbf31fce7));
        assert_eq!(alpha[344].to_bits(), 0x3f8da627);
        assert_eq!(beta[344].to_bits(), 0x3eff7557);
        assert_eq!(
            super::snake_beta_scalar(f32::from_bits(0xbe0dab31), alpha[344], beta[344]).to_bits(),
            0xbdbbdfa1
        );
        let snake = Snake { alpha, beta };
        let mut values = vec![0.0; 345];
        values[344] = f32::from_bits(0xbe0dab31);
        snake.forward(&mut values, 1);
        assert_eq!(values[344].to_bits(), 0xbdbbdfa1);
    }

    #[test]
    fn large_f32_matrix_vector_matches_accelerate_amx_bits() {
        let reduction = 448;
        let values = (0..reduction)
            .map(|index| (index % 11) as f32 * 0.01)
            .collect::<Vec<_>>();
        let weights = (0..2048 * reduction)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.001)
            .collect::<Vec<_>>();
        let output = weights
            .chunks_exact(reduction)
            .map(|row| torch_large_f32_dot(&values, row))
            .collect::<Vec<_>>();
        let hash = output.iter().fold(0xcbf29ce484222325u64, |hash, value| {
            (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100000001b3)
        });
        assert_eq!(output[6].to_bits(), 0x39a28780);
        assert_eq!(hash, 0x6877889a4e1339ed);
    }

    #[test]
    fn transposed_convolution_groups_each_tap_like_torch() {
        let input = (0..1024 * 2)
            .map(|index| (index % 11) as f32 * 0.01)
            .collect::<Vec<_>>();
        let weights = (0..1024 * 4)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.001)
            .collect::<Vec<_>>();
        let convolution = Conv {
            weights,
            bias: vec![-0.037],
            input: 1024,
            output: 1,
            kernel: 4,
            stride: 2,
            padding: 1,
            dilation: 1,
            transpose: true,
        };
        let output = convolution.forward(&input, 2).unwrap().0;
        assert_eq!(output[1].to_bits(), 0xbd14af4f);
        assert_eq!(output[2].to_bits(), 0xbd1e6eec);
    }

    #[test]
    fn convolution_adds_bias_after_the_reduction() {
        let input = (0..64)
            .map(|index| (index % 11) as f32 * 0.01)
            .collect::<Vec<_>>();
        let weights = (0..64)
            .map(|index| ((index % 13) as f32 - 6.0) * 0.002)
            .collect::<Vec<_>>();
        let convolution = Conv {
            weights,
            bias: vec![-0.1],
            input: 64,
            output: 1,
            kernel: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            transpose: false,
        };
        assert_eq!(
            convolution.forward(&input, 1).unwrap().0[0].to_bits(),
            0xbdd0092d
        );
    }

    #[test]
    fn convolution_fuses_bias_for_torch_matrix_tiles() {
        let input = (0..64 * 32)
            .map(|index| (index % 11) as f32 * 0.01)
            .collect::<Vec<_>>();
        let weights = (0..256 * 64)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.001)
            .collect::<Vec<_>>();
        let convolution = Conv {
            weights,
            bias: vec![-0.037; 256],
            input: 64,
            output: 256,
            kernel: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            transpose: false,
        };
        let output = convolution.forward(&input, 32).unwrap().0;
        assert_eq!(output[0].to_bits(), 0xbd24b341);
        assert_eq!(output[31].to_bits(), 0xbd20663e);
    }

    #[test]
    fn convolution_matches_torch_strided_tail_k_blocks() {
        let frames = 198;
        let mut input = vec![0.0; 1024 * frames];
        let mut weights = vec![0.0; 1024 * 1024 * 7];
        for channel in 0..1024 {
            for tap in 0..7 {
                let index = channel * 7 + tap;
                input[channel * frames + 189 + tap] = ((index * 198 + 192) % 11) as f32 * 0.01;
                weights[index] = ((index % 29) as f32 - 14.0) * 0.001;
            }
        }
        let convolution = Conv {
            weights,
            bias: vec![-0.037; 1024],
            input: 1024,
            output: 1024,
            kernel: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            transpose: false,
        };
        let output = convolution.forward(&input, frames).unwrap().0;
        assert_eq!(output[192].to_bits(), 0xbd23d70b);
    }

    #[test]
    fn convolution_matches_torch_single_k_block_tail() {
        let frames = 509;
        let input = (0..512 * frames)
            .map(|index| (index % 11) as f32 * 0.01)
            .collect::<Vec<_>>();
        let convolution = Conv {
            weights: (0..512 * 512 * 7)
                .map(|index| ((index % 29) as f32 - 14.0) * 0.001)
                .collect(),
            bias: vec![-0.037; 512],
            input: 512,
            output: 512,
            kernel: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            transpose: false,
        };
        assert_eq!(
            convolution.forward(&input, frames).unwrap().0[480].to_bits(),
            0xbd33482c
        );
    }

    #[test]
    fn convolution_selects_torch_bias_tile_width() {
        let frames = 4_784;
        let input = (0..128 * frames)
            .map(|index| (index % 11) as f32 * 0.01)
            .collect::<Vec<_>>();
        let convolution = Conv {
            weights: (0..128 * 128 * 7)
                .map(|index| ((index % 29) as f32 - 14.0) * 0.001)
                .collect(),
            bias: vec![-0.037; 128],
            input: 128,
            output: 128,
            kernel: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            transpose: false,
        };
        assert_eq!(
            convolution.forward(&input, frames).unwrap().0[4769].to_bits(),
            0xbd20125a
        );

        let convolution = Conv {
            weights: (0..128 * 128)
                .map(|index| ((index % 29) as f32 - 14.0) * 0.001)
                .collect(),
            bias: vec![-0.037; 128],
            input: 128,
            output: 128,
            kernel: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            transpose: false,
        };
        assert_eq!(
            convolution.forward(&input, frames).unwrap().0[4768].to_bits(),
            0xbd33721b
        );
    }
}
