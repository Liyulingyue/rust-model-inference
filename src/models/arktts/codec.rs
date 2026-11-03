//! Published causal codec; buffers are time-major [frames, channels].
use super::transformer::{Cache, Shape, Transformer};
use super::{config, trace, Frame, Matrix, CONTEXT, SAMPLES_PER_FRAME};
use crate::core::tensor::{load_f32_tensor, require_tensor_type, GGMLType, TensorSource};
use crate::models::breeze::CausalConv;
use crate::models::qwen3::tts::codec::conv::{conv_transpose1d_causal, ConvTranspose1dState};
use crate::models::qwen3::tts::codec::snake::snake1d_inplace;
use rayon::prelude::*;

fn tensor(source: &dyn TensorSource, name: &str, shape: &[u64]) -> Result<Vec<f32>, String> {
    require_tensor_type(source, name, shape, &[GGMLType::F32])?;
    let values = load_f32_tensor(source, name, shape)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(format!("non-finite ArkTTS codec tensor: {name}"));
    }
    Ok(values)
}

fn conv(
    s: &dyn TensorSource,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<CausalConv, String> {
    CausalConv::load(
        s,
        &format!("{prefix}.conv"),
        input,
        output,
        kernel,
        stride,
        dilation,
        groups,
        true,
        false,
    )
}

struct Snake {
    alpha: Vec<f32>,
    inverse: Vec<f32>,
}
impl Snake {
    fn load(s: &dyn TensorSource, prefix: &str, channels: usize) -> Result<Self, String> {
        let alpha = tensor(s, &format!("{prefix}.alpha"), &[1, channels as u64, 1])?;
        let inverse = alpha.iter().map(|a| 1.0 / (a + 1e-9)).collect();
        Ok(Self { alpha, inverse })
    }
    fn apply(&self, x: &mut [f32]) -> Result<(), String> {
        snake1d_inplace(x, x.len() / self.alpha.len(), &self.alpha, &self.inverse)
    }
}

struct Transpose {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
}
impl Transpose {
    fn load(
        s: &dyn TensorSource,
        p: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: tensor(
                s,
                &format!("{p}.conv.weight"),
                &[kernel as u64, output as u64, input as u64],
            )?,
            bias: tensor(s, &format!("{p}.conv.bias"), &[output as u64])?,
            input,
            output,
            kernel,
            stride,
        })
    }
    fn apply(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        conv_transpose1d_causal(
            &self.weight,
            Some(&self.bias),
            x,
            self.input,
            x.len() / self.input,
            self.output,
            self.kernel,
            self.stride,
            &mut ConvTranspose1dState::default(),
        )
    }
}

struct Residual {
    first_snake: Snake,
    first: CausalConv,
    second_snake: Snake,
    second: CausalConv,
}
impl Residual {
    fn load(
        s: &dyn TensorSource,
        p: &str,
        channels: usize,
        dilation: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            first_snake: Snake::load(s, &format!("{p}.block.0"), channels)?,
            first: conv(
                s,
                &format!("{p}.block.1"),
                channels,
                channels,
                7,
                1,
                dilation,
                1,
            )?,
            second_snake: Snake::load(s, &format!("{p}.block.2"), channels)?,
            second: conv(s, &format!("{p}.block.3"), channels, channels, 1, 1, 1, 1)?,
        })
    }
    fn apply(&self, mut x: Vec<f32>) -> Result<Vec<f32>, String> {
        let mut branch = x.clone();
        self.first_snake.apply(&mut branch)?;
        branch = self.first.forward(&branch)?;
        self.second_snake.apply(&mut branch)?;
        branch = self.second.forward(&branch)?;
        for (out, add) in x.iter_mut().zip(branch) {
            *out += add;
        }
        Ok(x)
    }
}

struct Encoder<'a> {
    input: CausalConv,
    stages: Vec<(Vec<Residual>, Snake, CausalConv)>,
    transformer: Transformer<'a>,
    snake: Snake,
    output: CausalConv,
}

fn codec_transformer<'a>(
    s: &'a dyn TensorSource,
    p: &str,
    count: usize,
    heads: usize,
    kv: usize,
    intermediate: usize,
    window: usize,
) -> Result<Transformer<'a>, String> {
    Transformer::load(
        s,
        &format!("{p}.layers"),
        &format!("{p}.norm"),
        count,
        Shape {
            width: 1024,
            heads,
            kv,
            intermediate,
            epsilon: 1e-5,
            rope_base: 10000.0,
            window: Some(window),
        },
        false,
        true,
    )
}

impl<'a> Encoder<'a> {
    fn load(s: &'a dyn TensorSource) -> Result<Self, String> {
        let mut stages = Vec::new();
        for (index, stride) in [2, 4, 8, 8].into_iter().enumerate() {
            let channels = 64 << index;
            let p = format!("encoder.block.{}.block", index + 1);
            let residual = [1, 3, 9]
                .into_iter()
                .enumerate()
                .map(|(i, d)| Residual::load(s, &format!("{p}.{i}"), channels, d))
                .collect::<Result<_, _>>()?;
            stages.push((
                residual,
                Snake::load(s, &format!("{p}.3"), channels)?,
                conv(
                    s,
                    &format!("{p}.4"),
                    channels,
                    channels * 2,
                    2 * stride,
                    stride,
                    1,
                    1,
                )?,
            ));
        }
        Ok(Self {
            input: conv(s, "encoder.block.0", 1, 64, 7, 1, 1, 1)?,
            stages,
            transformer: codec_transformer(s, "encoder.block.4.block.5", 4, 16, 16, 3072, 512)?,
            snake: Snake::load(s, "encoder.block.5", 1024)?,
            output: conv(s, "encoder.block.6", 1024, 1024, 3, 1, 1, 1)?,
        })
    }
    fn apply(&self, audio: &[f32]) -> Result<Vec<f32>, String> {
        let mut padded = audio.to_vec();
        padded.resize(
            audio.len().div_ceil(SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME,
            0.0,
        );
        let mut x = self.input.forward(&padded)?;
        for (index, (residual, snake, down)) in self.stages.iter().enumerate() {
            for block in residual {
                x = block.apply(x)?;
            }
            snake.apply(&mut x)?;
            x = down.forward(&x)?;
            if index == 3 {
                x = self
                    .transformer
                    .normalize(&self.transformer.forward(x, &mut Cache::default())?);
            }
        }
        self.snake.apply(&mut x)?;
        self.output.forward(&x)
    }
}

struct ConvNeXt<'a> {
    depth: CausalConv,
    norm: Vec<f32>,
    bias: Vec<f32>,
    first: Matrix<'a>,
    second: Matrix<'a>,
    gamma: Vec<f32>,
}
impl<'a> ConvNeXt<'a> {
    fn load(s: &'a dyn TensorSource, p: &str) -> Result<Self, String> {
        Ok(Self {
            depth: conv(s, &format!("{p}.dwconv"), 1024, 1024, 7, 1, 1, 1024)?,
            norm: tensor(s, &format!("{p}.norm.weight"), &[1024])?,
            bias: tensor(s, &format!("{p}.norm.bias"), &[1024])?,
            first: Matrix::load(s, &format!("{p}.pwconv1"), 1024, 4096, true)?,
            second: Matrix::load(s, &format!("{p}.pwconv2"), 4096, 1024, true)?,
            gamma: tensor(s, &format!("{p}.gamma"), &[1024])?,
        })
    }
    fn apply(&self, mut x: Vec<f32>) -> Result<Vec<f32>, String> {
        let depth = self.depth.forward(&x)?;
        let mut normalized = vec![0.0; depth.len()];
        for (out, row) in normalized
            .chunks_exact_mut(1024)
            .zip(depth.chunks_exact(1024))
        {
            crate::ops::layer_norm(row, &self.norm, &self.bias, 1e-6, out);
        }
        let mut branch = self.first.forward(&normalized);
        for value in &mut branch {
            *value = crate::ops::gelu_erf(*value);
        }
        let branch = self.second.forward(&branch);
        for (i, (out, add)) in x.iter_mut().zip(branch).enumerate() {
            *out += add * self.gamma[i % 1024];
        }
        Ok(x)
    }
}

struct Codebook {
    input: CausalConv,
    output: CausalConv,
    embeddings: Vec<f32>,
    normalized: Vec<f32>,
    size: usize,
}
impl Codebook {
    fn load(s: &dyn TensorSource, p: &str, size: usize) -> Result<Self, String> {
        let embeddings = tensor(s, &format!("{p}.codebook.weight"), &[8, size as u64])?;
        let mut normalized = embeddings.clone();
        for row in normalized.chunks_exact_mut(8) {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for value in row {
                *value /= norm;
            }
        }
        Ok(Self {
            input: CausalConv::load(s, &format!("{p}.in_proj"), 1024, 8, 1, 1, 1, 1, true, false)?,
            output: CausalConv::load(
                s,
                &format!("{p}.out_proj"),
                8,
                1024,
                1,
                1,
                1,
                1,
                true,
                false,
            )?,
            embeddings,
            normalized,
            size,
        })
    }
    fn decode(&self, codes: &[u32]) -> Result<Vec<f32>, String> {
        let mut input = Vec::with_capacity(codes.len() * 8);
        for &code in codes {
            let code = (code as usize).min(self.size - 1);
            input.extend_from_slice(&self.embeddings[code * 8..(code + 1) * 8]);
        }
        self.output.forward(&input)
    }
    fn encode(&self, x: &[f32]) -> Result<(Vec<f32>, Vec<u32>), String> {
        let input = self.input.forward(x)?;
        let codes = input
            .par_chunks(8)
            .map(|row| {
                let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                let row: Vec<_> = row.iter().map(|v| v / norm).collect();
                let square = row.iter().map(|v| v * v).sum::<f32>();
                self.normalized
                    .chunks_exact(8)
                    .enumerate()
                    .map(|(id, code)| {
                        let distance = square - 2.0 * crate::ops::dot_f32(&row, code, 8)
                            + code.iter().map(|v| v * v).sum::<f32>();
                        (id as u32, distance)
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                    .unwrap()
                    .0
            })
            .collect::<Vec<_>>();
        let mut quantized = input;
        for (row, &code) in quantized.chunks_exact_mut(8).zip(&codes) {
            for (projected, &embedding) in row
                .iter_mut()
                .zip(&self.embeddings[code as usize * 8..(code as usize + 1) * 8])
            {
                *projected += embedding - *projected;
            }
        }
        Ok((self.output.forward(&quantized)?, codes))
    }
}

struct DecoderStage {
    snake: Snake,
    up: Transpose,
    residual: Vec<Residual>,
}
struct Decoder {
    input: CausalConv,
    stages: Vec<DecoderStage>,
    snake: Snake,
    output: CausalConv,
}
impl Decoder {
    fn load(s: &dyn TensorSource) -> Result<Self, String> {
        let mut stages = Vec::new();
        for (index, stride) in [8, 8, 4, 2].into_iter().enumerate() {
            let input = 1536 >> index;
            let output = input / 2;
            let p = format!("decoder.model.{}.block", index + 1);
            stages.push(DecoderStage {
                snake: Snake::load(s, &format!("{p}.0"), input)?,
                up: Transpose::load(s, &format!("{p}.1"), input, output, stride * 2, stride)?,
                residual: [1, 3, 9]
                    .into_iter()
                    .enumerate()
                    .map(|(i, d)| Residual::load(s, &format!("{p}.{}", i + 2), output, d))
                    .collect::<Result<_, _>>()?,
            });
        }
        Ok(Self {
            input: conv(s, "decoder.model.0", 1024, 1536, 7, 1, 1, 1)?,
            stages,
            snake: Snake::load(s, "decoder.model.5", 96)?,
            output: conv(s, "decoder.model.6", 96, 1, 7, 1, 1, 1)?,
        })
    }
    fn apply(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        let mut x = self.input.forward(x)?;
        for (index, stage) in self.stages.iter().enumerate() {
            stage.snake.apply(&mut x)?;
            x = stage.up.apply(&x)?;
            for block in &stage.residual {
                x = block.apply(x)?;
            }
            trace(
                "arktts.codec.decoder",
                Some(index),
                None,
                &[x.len() / (768 >> index), 768 >> index],
                &x,
            )?;
        }
        self.snake.apply(&mut x)?;
        let mut x = self.output.forward(&x)?;
        for v in &mut x {
            *v = v.tanh();
        }
        Ok(x)
    }
}

pub struct Codec<'a> {
    pub(super) bundle: String,
    encoder: Encoder<'a>,
    down: Vec<(CausalConv, ConvNeXt<'a>)>,
    pre: Transformer<'a>,
    codebooks: Vec<Codebook>,
    post: Transformer<'a>,
    up: Vec<(Transpose, ConvNeXt<'a>)>,
    decoder: Decoder,
}
impl<'a> Codec<'a> {
    pub fn from_source(s: &'a dyn TensorSource) -> Result<Self, String> {
        config(s, "arktts_codec")?;
        let mut down = Vec::new();
        let mut up = Vec::new();
        for index in 0..2 {
            let p = format!("quantizer.downsample.{index}");
            down.push((
                conv(s, &format!("{p}.0"), 1024, 1024, 2, 2, 1, 1)?,
                ConvNeXt::load(s, &format!("{p}.1"))?,
            ));
            let p = format!("quantizer.upsample.{index}");
            up.push((
                Transpose::load(s, &format!("{p}.0"), 1024, 1024, 2, 2)?,
                ConvNeXt::load(s, &format!("{p}.1"))?,
            ));
        }
        let mut codebooks = vec![Codebook::load(
            s,
            "quantizer.semantic_quantizer.quantizers.0",
            4096,
        )?];
        for index in 0..9 {
            codebooks.push(Codebook::load(
                s,
                &format!("quantizer.quantizer.quantizers.{index}"),
                1024,
            )?);
        }
        Ok(Self {
            bundle: super::bundle(s)?,
            encoder: Encoder::load(s)?,
            down,
            up,
            pre: codec_transformer(s, "quantizer.pre_module", 8, 16, 16, 3072, 128)?,
            post: codec_transformer(s, "quantizer.post_module", 8, 16, 8, 1216, 128)?,
            codebooks,
            decoder: Decoder::load(s)?,
        })
    }

    pub fn encode(&self, audio: &[f32]) -> Result<Vec<Frame>, String> {
        let _scope = crate::compute::ComputePolicy::Cpu.cpu_scope();
        if audio.is_empty()
            || audio.len() > CONTEXT * SAMPLES_PER_FRAME
            || audio.iter().any(|v| !v.is_finite())
        {
            return Err("ArkTTS codec requires finite nonempty mono 44100 Hz audio within 2048 codec frames".into());
        }
        let mut x = self.encoder.apply(audio)?;
        trace(
            "arktts.codec.encoder",
            None,
            None,
            &[x.len() / 1024, 1024],
            &x,
        )?;
        for (conv, next) in &self.down {
            x = next.apply(conv.forward(&x)?)?;
        }
        x = self
            .pre
            .normalize(&self.pre.forward(x, &mut Cache::default())?);
        trace("arktts.codec.pre", None, None, &[x.len() / 1024, 1024], &x)?;
        let mut frames = vec![[0; 10]; x.len() / 1024];
        for (index, codebook) in self.codebooks.iter().enumerate() {
            let (quantized, codes) = codebook.encode(&x)?;
            for (frame, code) in frames.iter_mut().zip(codes) {
                frame[index] = code;
            }
            for (residual, value) in x.iter_mut().zip(quantized) {
                *residual -= value;
            }
        }
        super::tokens(
            "arktts.reference_codes",
            &frames.iter().flatten().copied().collect::<Vec<_>>(),
        )?;
        Ok(frames)
    }

    pub fn decode(&self, frames: &[Frame]) -> Result<Vec<f32>, String> {
        let _scope = crate::compute::ComputePolicy::Cpu.cpu_scope();
        if frames.len() > CONTEXT {
            return Err("ArkTTS codec exceeds 2048 frames".into());
        }
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let mut x = vec![0.0; frames.len() * 1024];
        for (index, codebook) in self.codebooks.iter().enumerate() {
            let codes = frames.iter().map(|frame| frame[index]).collect::<Vec<_>>();
            let branch = codebook.decode(&codes)?;
            for (out, add) in x.iter_mut().zip(branch) {
                *out += add;
            }
        }
        trace(
            "arktts.codec.quantized",
            None,
            None,
            &[frames.len(), 1024],
            &x,
        )?;
        x = self
            .post
            .normalize(&self.post.forward(x, &mut Cache::default())?);
        trace("arktts.codec.post", None, None, &[frames.len(), 1024], &x)?;
        for (up, next) in &self.up {
            x = next.apply(up.apply(&x)?)?;
        }
        trace(
            "arktts.codec.upsampled",
            None,
            None,
            &[x.len() / 1024, 1024],
            &x,
        )?;
        let waveform = self.decoder.apply(&x)?;
        if waveform.len() != frames.len() * SAMPLES_PER_FRAME
            || waveform.iter().any(|v| !v.is_finite())
        {
            return Err("ArkTTS codec produced invalid waveform".into());
        }
        trace(
            "arktts.codec.audio",
            None,
            None,
            &[waveform.len()],
            &waveform,
        )?;
        Ok(waveform)
    }
}
