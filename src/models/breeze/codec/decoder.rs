use super::*;

struct TransposeConv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
}

impl TransposeConv {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self, String> {
        let original = tensor(
            source,
            &format!("{prefix}.weight"),
            &[input, output, kernel],
        )?;
        let mut weight = vec![0.; original.len()];
        for ic in 0..input {
            for oc in 0..output {
                for tap in 0..kernel {
                    weight[(oc * kernel + tap) * input + ic] =
                        original[(ic * output + oc) * kernel + tap];
                }
            }
        }
        Ok(Self {
            weight,
            bias: tensor(source, &format!("{prefix}.bias"), &[output])?,
            input,
            output,
            kernel,
            stride,
        })
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let frames = input.len() / self.input;
        let length = frames
            .checked_mul(self.stride)
            .ok_or("Breeze transposed convolution length overflow")?;
        let size = length
            .checked_mul(self.output)
            .ok_or("Breeze transposed convolution output overflow")?;
        let mut output = vec![0.; size];
        output
            .par_chunks_mut(self.output)
            .enumerate()
            .for_each(|(time, row)| {
                let first = (time + 1).saturating_sub(self.kernel).div_ceil(self.stride);
                let last = (time / self.stride).min(frames - 1);
                for t in (first..=last).rev() {
                    let tap = time - t * self.stride;
                    let source = &input[t * self.input..(t + 1) * self.input];
                    for (oc, value) in row.iter_mut().enumerate() {
                        let start = (oc * self.kernel + tap) * self.input;
                        *value +=
                            dot_f32(source, &self.weight[start..start + self.input], self.input);
                    }
                }
                for (value, bias) in row.iter_mut().zip(&self.bias) {
                    *value += bias;
                }
            });
        Ok(output)
    }
}

struct Snake {
    alpha: Vec<f32>,
    inverse_beta: Vec<f32>,
}

impl Snake {
    fn load(source: &dyn TensorSource, prefix: &str, channels: usize) -> Result<Self, String> {
        let mut alpha = tensor(source, &format!("{prefix}.alpha"), &[channels])?;
        let mut beta = tensor(source, &format!("{prefix}.beta"), &[channels])?;
        crate::ops::exp_inplace(&mut alpha);
        crate::ops::exp_inplace(&mut beta);
        for b in &mut beta {
            *b = 1. / (*b + 1e-9);
        }
        if alpha.iter().chain(&beta).any(|v| !v.is_finite()) {
            return Err(format!("Breeze Snake parameters overflow: {prefix}"));
        }
        Ok(Self {
            alpha,
            inverse_beta: beta,
        })
    }

    fn forward(&self, x: &mut [f32]) -> Result<(), String> {
        for row in x.chunks_exact_mut(self.alpha.len()) {
            for (channel, value) in row.iter_mut().enumerate() {
                let sine = (self.alpha[channel] * *value).sin();
                *value += self.inverse_beta[channel] * (sine * sine);
            }
        }
        Ok(())
    }
}

struct ConvNeXt {
    up: TransposeConv,
    depthwise: Conv,
    norm: Norm,
    linear1: Linear,
    linear2: Linear,
    gamma: Vec<f32>,
}

impl ConvNeXt {
    fn load(source: &dyn TensorSource, index: usize) -> Result<Self, String> {
        let prefix = format!("decoder.upsample.{index}");
        Ok(Self {
            up: TransposeConv::load(source, &format!("{prefix}.0.conv"), 1024, 1024, 2, 2)?,
            depthwise: Conv::load(
                source,
                &format!("{prefix}.1.dwconv.conv"),
                1024,
                1024,
                7,
                1,
                1,
                1024,
                true,
                false,
            )?,
            norm: Norm::load(source, &format!("{prefix}.1.norm"), 1024, true, 1e-6)?,
            linear1: Linear::load(source, &format!("{prefix}.1.pwconv1"), 1024, 4096, true)?,
            linear2: Linear::load(source, &format!("{prefix}.1.pwconv2"), 4096, 1024, true)?,
            gamma: tensor(source, &format!("{prefix}.1.gamma"), &[1024])?,
        })
    }

    fn forward(&self, x: &[f32], index: usize) -> Result<Vec<f32>, String> {
        let mut residual = self.up.forward(x)?;
        trace(
            &format!("breeze.codec.decoder.upsample.{index}.0"),
            None,
            &residual,
            1024,
        );
        let x = self.depthwise.forward(&residual)?;
        trace(
            &format!("breeze.codec.decoder.upsample.{index}.1.dwconv"),
            None,
            &x,
            1024,
        );
        let x = self.norm.forward(&x);
        trace(
            &format!("breeze.codec.decoder.upsample.{index}.1.norm"),
            None,
            &x,
            1024,
        );
        let mut x = self.linear1.forward(&x);
        trace(
            &format!("breeze.codec.decoder.upsample.{index}.1.pwconv1"),
            None,
            &x,
            4096,
        );
        gelu(&mut x);
        trace(
            &format!("breeze.codec.decoder.upsample.{index}.1.act"),
            None,
            &x,
            4096,
        );
        add_scaled(&mut residual, &self.linear2.forward(&x), &self.gamma);
        Ok(residual)
    }
}

struct Residual {
    act1: Snake,
    conv1: Conv,
    act2: Snake,
    conv2: Conv,
}

impl Residual {
    fn load(
        source: &dyn TensorSource,
        prefix: &str,
        channels: usize,
        dilation: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            act1: Snake::load(source, &format!("{prefix}.act1"), channels)?,
            conv1: Conv::load(
                source,
                &format!("{prefix}.conv1.conv"),
                channels,
                channels,
                7,
                1,
                dilation,
                1,
                true,
                false,
            )?,
            act2: Snake::load(source, &format!("{prefix}.act2"), channels)?,
            conv2: Conv::load(
                source,
                &format!("{prefix}.conv2.conv"),
                channels,
                channels,
                1,
                1,
                1,
                1,
                true,
                false,
            )?,
        })
    }

    fn forward(&self, mut x: Vec<f32>, prefix: &str) -> Result<Vec<f32>, String> {
        let mut branch = x.clone();
        self.act1.forward(&mut branch)?;
        trace(&format!("{prefix}.act1"), None, &branch, self.conv1.input);
        branch = self.conv1.forward(&branch)?;
        trace(&format!("{prefix}.conv1"), None, &branch, self.conv1.output);
        self.act2.forward(&mut branch)?;
        trace(&format!("{prefix}.act2"), None, &branch, self.conv2.input);
        branch = self.conv2.forward(&branch)?;
        trace(&format!("{prefix}.conv2"), None, &branch, self.conv2.output);
        for (v, add) in x.iter_mut().zip(branch) {
            *v += add;
        }
        Ok(x)
    }
}

struct DacBlock {
    snake: Snake,
    up: TransposeConv,
    residuals: Vec<Residual>,
}

impl DacBlock {
    fn load(source: &dyn TensorSource, index: usize, stride: usize) -> Result<Self, String> {
        let prefix = format!("decoder.decoder.{}.block", index + 1);
        let channels = 1536 >> index;
        Ok(Self {
            snake: Snake::load(source, &format!("{prefix}.0"), channels)?,
            up: TransposeConv::load(
                source,
                &format!("{prefix}.1.conv"),
                channels,
                channels / 2,
                stride * 2,
                stride,
            )?,
            residuals: [1, 3, 9]
                .into_iter()
                .enumerate()
                .map(|(i, d)| {
                    Residual::load(source, &format!("{prefix}.{}", i + 2), channels / 2, d)
                })
                .collect::<Result<_, _>>()?,
        })
    }

    fn forward(&self, mut x: Vec<f32>, index: usize) -> Result<Vec<f32>, String> {
        let prefix = format!("breeze.codec.decoder.dac.{}.block", index + 1);
        self.snake.forward(&mut x)?;
        trace(&format!("{prefix}.0"), None, &x, self.up.input);
        x = self.up.forward(&x)?;
        trace(&format!("{prefix}.1"), None, &x, self.up.output);
        for (index, residual) in self.residuals.iter().enumerate() {
            let prefix = format!("{prefix}.{}", index + 2);
            x = residual.forward(x, &prefix)?;
            trace(&prefix, None, &x, self.up.output);
        }
        Ok(x)
    }
}

pub(super) struct Decoder {
    semantic: Codebook,
    acoustic: Vec<Codebook>,
    semantic_projection: Conv,
    acoustic_projection: Conv,
    pre_conv: Conv,
    input_projection: Linear,
    transformer: Vec<TransformerLayer>,
    norm: Norm,
    output_projection: Linear,
    upsample: Vec<ConvNeXt>,
    input: Conv,
    blocks: Vec<DacBlock>,
    final_snake: Snake,
    output: Conv,
}

impl Decoder {
    pub(super) fn load(source: &dyn TensorSource) -> Result<Self, String> {
        let input_projection = Linear::load(
            source,
            "decoder.pre_transformer.input_proj",
            1024,
            DIM,
            true,
        )?;
        Ok(Self {
            semantic: Codebook::load(
                source,
                "decoder.quantizer.rvq_first.vq.layers.0._codebook",
                true,
            )?,
            acoustic: (0..15)
                .map(|i| {
                    Codebook::load(
                        source,
                        &format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook"),
                        true,
                    )
                })
                .collect::<Result<_, _>>()?,
            semantic_projection: Conv::load(
                source,
                "decoder.quantizer.rvq_first.output_proj",
                CODE_DIM,
                DIM,
                1,
                1,
                1,
                1,
                false,
                false,
            )?,
            acoustic_projection: Conv::load(
                source,
                "decoder.quantizer.rvq_rest.output_proj",
                CODE_DIM,
                DIM,
                1,
                1,
                1,
                1,
                false,
                false,
            )?,
            pre_conv: Conv::load(
                source,
                "decoder.pre_conv.conv",
                DIM,
                1024,
                3,
                1,
                1,
                1,
                true,
                false,
            )?,
            input_projection,
            transformer: (0..8)
                .map(|i| {
                    TransformerLayer::load(
                        source,
                        &format!("decoder.pre_transformer.layers.{i}"),
                        true,
                    )
                })
                .collect::<Result<_, _>>()?,
            norm: Norm::load(source, "decoder.pre_transformer.norm", DIM, false, 1e-5)?,
            output_projection: Linear::load(
                source,
                "decoder.pre_transformer.output_proj",
                DIM,
                1024,
                true,
            )?,
            upsample: (0..2)
                .map(|i| ConvNeXt::load(source, i))
                .collect::<Result<_, _>>()?,
            input: Conv::load(
                source,
                "decoder.decoder.0.conv",
                1024,
                1536,
                7,
                1,
                1,
                1,
                true,
                false,
            )?,
            blocks: [8, 5, 4, 3]
                .into_iter()
                .enumerate()
                .map(|(i, s)| DacBlock::load(source, i, s))
                .collect::<Result<_, _>>()?,
            final_snake: Snake::load(source, "decoder.decoder.5", 96)?,
            output: Conv::load(
                source,
                "decoder.decoder.6.conv",
                96,
                1,
                7,
                1,
                1,
                1,
                true,
                false,
            )?,
        })
    }

    pub(super) fn forward(&self, codes: &[[u32; 16]]) -> Result<Vec<f32>, String> {
        let mut first = vec![0.; codes.len() * CODE_DIM];
        self.semantic.add_codes(&mut first, codes, 0);
        trace("breeze.codec.decoder.semantic", None, &first, CODE_DIM);
        let mut rest = vec![0.; first.len()];
        for (index, book) in self.acoustic.iter().enumerate() {
            book.add_codes(&mut rest, codes, index + 1);
        }
        trace("breeze.codec.decoder.acoustic", None, &rest, CODE_DIM);
        let mut x = self.semantic_projection.forward(&first)?;
        let rest = self.acoustic_projection.forward(&rest)?;
        for (v, add) in x.iter_mut().zip(rest) {
            *v += add;
        }
        trace("breeze.codec.decoder.rvq", None, &x, DIM);
        x = self.pre_conv.forward(&x)?;
        trace("breeze.codec.decoder.pre_conv", None, &x, 1024);
        x = self.input_projection.forward(&x);
        trace("breeze.codec.decoder.input_projection", None, &x, DIM);
        for (index, layer) in self.transformer.iter().enumerate() {
            x = layer.forward(x, index);
            trace("breeze.codec.decoder.transformer", Some(index), &x, DIM);
        }
        x = self.norm.forward(&x);
        trace("breeze.codec.decoder.transformer.norm", None, &x, DIM);
        x = self.output_projection.forward(&x);
        trace("breeze.codec.decoder.output_projection", None, &x, 1024);
        trace("breeze.codec.decoder.transformer_output", None, &x, 1024);
        for (index, up) in self.upsample.iter().enumerate() {
            x = up.forward(&x, index)?;
            trace("breeze.codec.decoder.upsample", Some(index), &x, 1024);
        }
        x = self.input.forward(&x)?;
        trace("breeze.codec.decoder.dac.0", None, &x, 1536);
        for (index, block) in self.blocks.iter().enumerate() {
            x = block.forward(x, index)?;
            trace("breeze.codec.decoder.dac", Some(index), &x, 768 >> index);
        }
        self.final_snake.forward(&mut x)?;
        trace("breeze.codec.decoder.dac.5", None, &x, 96);
        x = self.output.forward(&x)?;
        trace("breeze.codec.decoder.dac.6", None, &x, 1);
        for v in &mut x {
            *v = v.clamp(-1., 1.);
        }
        trace("breeze.codec.decoder.pcm", None, &x, 1);
        Ok(x)
    }
}
