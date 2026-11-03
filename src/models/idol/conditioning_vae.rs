//! FLUX.2's encoder only; raw Diffusers names, BF16 module boundaries, no decoder allocation.
use super::super::common::{finite, round_bf16, round_bf16_checked, tensor, trace, Linear};
use super::{attention, validate_rgb};
use crate::core::tensor::TensorSource;
use rayon::prelude::*;
fn b(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

struct Conv {
    input: usize,
    output: usize,
    kernel: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
    gpu: super::super::gpu::Projection,
}
impl Conv {
    fn load(
        s: &dyn TensorSource,
        p: &str,
        input: usize,
        output: usize,
        kernel: usize,
    ) -> Result<Self, String> {
        let mut weight = tensor(s, &format!("{p}.weight"), &[output, input, kernel, kernel])?;
        let mut bias = tensor(s, &format!("{p}.bias"), &[output])?;
        round_bf16_checked(&mut weight, "BF16 VAE convolution weight")?;
        round_bf16_checked(&mut bias, "BF16 VAE convolution bias")?;
        Ok(Self {
            input,
            output,
            kernel,
            weight,
            bias,
            gpu: Default::default(),
        })
    }
    // NHWC throughout; stride two uses Diffusers' zero padding on right and bottom only.
    fn forward(&self, x: &[f32], side: usize, stride: usize) -> Result<Vec<f32>, String> {
        if side == 0
            || x.len() != side * side * self.input
            || !(stride == 1 || stride == 2)
            || side % stride != 0
        {
            return Err("Invalid Idol VAE convolution shape".into());
        }
        let next = side / stride;
        let patch_len = self.input * self.kernel * self.kernel;
        let mut out = vec![0.; next * next * self.output];
        if self.gpu.convolution(
            &self.weight,
            x,
            &mut out,
            [self.input, self.output],
            [side, side],
            [next, next],
            self.kernel,
            stride,
            if stride == 1 { self.kernel / 2 } else { 0 },
            true,
        ) {
            for row in out.chunks_exact_mut(self.output) {
                for (value, bias) in row.iter_mut().zip(&self.bias) {
                    *value = b(*value + bias);
                }
            }
            finite(&out, "GPU VAE convolution")?;
            return Ok(out);
        }
        super::super::gpu::check_failure()?;
        out.par_chunks_mut(self.output).enumerate().for_each_init(
            || vec![0.; patch_len],
            |patch, (p, row)| {
                patch.fill(0.);
                let padding = if stride == 1 { self.kernel / 2 } else { 0 };
                for c in 0..self.input {
                    for ky in 0..self.kernel {
                        for kx in 0..self.kernel {
                            let iy = (p / next * stride + ky) as isize - padding as isize;
                            let ix = (p % next * stride + kx) as isize - padding as isize;
                            if iy >= 0 && ix >= 0 && iy < side as isize && ix < side as isize {
                                patch[(c * self.kernel + ky) * self.kernel + kx] =
                                    x[(iy as usize * side + ix as usize) * self.input + c];
                            }
                        }
                    }
                }
                for (o, y) in row.iter_mut().enumerate() {
                    *y = b(crate::ops::dot_f32(
                        patch,
                        &self.weight[o * patch_len..(o + 1) * patch_len],
                        patch_len,
                    ) + self.bias[o]);
                }
            },
        );
        finite(&out, "VAE convolution")?;
        Ok(out)
    }
}
struct GroupNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl GroupNorm {
    fn load(s: &dyn TensorSource, p: &str, c: usize) -> Result<Self, String> {
        let mut weight = tensor(s, &format!("{p}.weight"), &[c])?;
        let mut bias = tensor(s, &format!("{p}.bias"), &[c])?;
        round_bf16_checked(&mut weight, "BF16 VAE normalization weight")?;
        round_bf16_checked(&mut bias, "BF16 VAE normalization bias")?;
        Ok(Self { weight, bias })
    }
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let channels = self.weight.len();
        let group = channels / 32;
        let spatial = x.len() / channels;
        let stats: Vec<_> = (0..32)
            .into_par_iter()
            .map(|g| {
                let mut values = Vec::with_capacity(spatial * group);
                for c in g * group..(g + 1) * group {
                    values.extend(x.chunks_exact(channels).map(|r| r[c]));
                }
                let mean = (crate::ops::sum_f32(&values) / values.len() as f64) as f32;
                let variance =
                    (crate::ops::sum_sq_centered_f32(&values, mean) / values.len() as f64) as f32;
                (mean, (variance + 1e-6).sqrt().recip())
            })
            .collect();
        let mut out = vec![0.; x.len()];
        out.par_chunks_mut(channels)
            .zip(x.par_chunks(channels))
            .for_each(|(row, input)| {
                for c in 0..channels {
                    let (mean, inv) = stats[c / group];
                    row[c] = b((input[c] - mean) * inv * self.weight[c] + self.bias[c]);
                }
            });
        out
    }
}
fn silu(mut x: Vec<f32>) -> Vec<f32> {
    crate::ops::silu_inplace(&mut x);
    round_bf16(&mut x);
    x
}
fn add(mut branch: Vec<f32>, residual: &[f32]) -> Vec<f32> {
    branch
        .par_iter_mut()
        .zip(residual)
        .for_each(|(a, &r)| *a = b(*a + r));
    branch
}
struct Residual {
    norm1: GroupNorm,
    norm2: GroupNorm,
    conv1: Conv,
    conv2: Conv,
    shortcut: Option<Conv>,
}
impl Residual {
    fn load(s: &dyn TensorSource, p: &str, input: usize, output: usize) -> Result<Self, String> {
        Ok(Self {
            norm1: GroupNorm::load(s, &format!("{p}.norm1"), input)?,
            norm2: GroupNorm::load(s, &format!("{p}.norm2"), output)?,
            conv1: Conv::load(s, &format!("{p}.conv1"), input, output, 3)?,
            conv2: Conv::load(s, &format!("{p}.conv2"), output, output, 3)?,
            shortcut: if input != output {
                Some(Conv::load(
                    s,
                    &format!("{p}.conv_shortcut"),
                    input,
                    output,
                    1,
                )?)
            } else {
                None
            },
        })
    }
    fn forward(&self, x: &[f32], side: usize) -> Result<Vec<f32>, String> {
        let a = self.conv1.forward(&silu(self.norm1.forward(x)), side, 1)?;
        let a = self.conv2.forward(&silu(self.norm2.forward(&a)), side, 1)?;
        let residual = if let Some(c) = &self.shortcut {
            c.forward(x, side, 1)?
        } else {
            x.to_vec()
        };
        Ok(add(a, &residual))
    }
}
struct MidAttention {
    norm: GroupNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}
impl MidAttention {
    fn load(s: &dyn TensorSource) -> Result<Self, String> {
        let p = "encoder.mid_block.attentions.0";
        let linear = |name: &str| -> Result<Linear, String> {
            let mut l = Linear::load(s, &format!("{p}.{name}"), 512, 512, true)?;
            l.to_bf16()?;
            Ok(l)
        };
        Ok(Self {
            norm: GroupNorm::load(s, &format!("{p}.group_norm"), 512)?,
            q: linear("to_q")?,
            k: linear("to_k")?,
            v: linear("to_v")?,
            o: linear("to_out.0")?,
        })
    }
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        let a = self.norm.forward(x);
        let mut q = self.q.forward(&a)?;
        let mut k = self.k.forward(&a)?;
        let mut v = self.v.forward(&a)?;
        round_bf16(&mut q);
        round_bf16(&mut k);
        round_bf16(&mut v);
        let mut a = attention(&q, &k, &v, 512, 1)?;
        round_bf16(&mut a);
        let mut a = self.o.forward(&a)?;
        round_bf16(&mut a);
        Ok(add(a, x))
    }
}
struct Stage {
    blocks: Vec<Residual>,
    down: Option<Conv>,
}
pub(super) struct Encoder {
    conv_in: Conv,
    stages: Vec<Stage>,
    mid1: Residual,
    attention: MidAttention,
    mid2: Residual,
    norm: GroupNorm,
    conv_out: Conv,
    quant: Conv,
    mean: Vec<f32>,
    std: Vec<f32>,
}
impl Encoder {
    pub fn load(s: &dyn TensorSource) -> Result<Self, String> {
        let mut stages = Vec::with_capacity(4);
        let mut input = 128;
        for (i, output) in [128, 256, 512, 512].into_iter().enumerate() {
            let mut blocks = Vec::with_capacity(2);
            for j in 0..2 {
                blocks.push(Residual::load(
                    s,
                    &format!("encoder.down_blocks.{i}.resnets.{j}"),
                    if j == 0 { input } else { output },
                    output,
                )?);
            }
            let down = if i < 3 {
                Some(Conv::load(
                    s,
                    &format!("encoder.down_blocks.{i}.downsamplers.0.conv"),
                    output,
                    output,
                    3,
                )?)
            } else {
                None
            };
            stages.push(Stage { blocks, down });
            input = output;
        }
        let mut mean = tensor(s, "bn.running_mean", &[128])?;
        round_bf16_checked(&mut mean, "BF16 VAE batch normalization mean")?;
        let mut var = tensor(s, "bn.running_var", &[128])?;
        round_bf16_checked(&mut var, "BF16 VAE batch normalization variance")?;
        if var.iter().any(|&v| v < 0.) {
            return Err("Idol FLUX.2 VAE batch normalization variance is negative".into());
        }
        let std = var.into_iter().map(|v| b(b(v + 1e-4).sqrt())).collect();
        Ok(Self {
            conv_in: Conv::load(s, "encoder.conv_in", 3, 128, 3)?,
            stages,
            mid1: Residual::load(s, "encoder.mid_block.resnets.0", 512, 512)?,
            attention: MidAttention::load(s)?,
            mid2: Residual::load(s, "encoder.mid_block.resnets.1", 512, 512)?,
            norm: GroupNorm::load(s, "encoder.conv_norm_out", 512)?,
            conv_out: Conv::load(s, "encoder.conv_out", 512, 64, 3)?,
            quant: Conv::load(s, "quant_conv", 64, 64, 1)?,
            mean,
            std,
        })
    }
    pub fn tokens(&self, rgb: &[u8], side: usize, noise: &[f32]) -> Result<Vec<f32>, String> {
        validate_rgb(rgb, side)?;
        if noise.len() != 32 * (side / 8) * (side / 8) {
            return Err("Idol VAE noise must have shape [32, side/8, side/8]".into());
        }
        finite(noise, "VAE posterior noise")?;
        let image: Vec<_> = rgb
            .iter()
            .map(|&x| b(b(b(x as f32 / 255.) * 2.) - 1.))
            .collect();
        let mut a = self.conv_in.forward(&image, side, 1)?;
        let mut current = side;
        trace("idol.vae.conv_in", &[1, current, current, 128], &a);
        for (i, stage) in self.stages.iter().enumerate() {
            for (j, block) in stage.blocks.iter().enumerate() {
                a = block.forward(&a, current)?;
                trace(
                    &format!("idol.vae.down.{i}.{j}"),
                    &[1, current, current, block.conv2.output],
                    &a,
                );
            }
            if let Some(down) = &stage.down {
                a = down.forward(&a, current, 2)?;
                current /= 2;
            }
        }
        a = self.mid1.forward(&a, current)?;
        trace("idol.vae.mid.resnet0", &[1, current, current, 512], &a);
        a = self.attention.forward(&a)?;
        trace("idol.vae.mid.attention", &[1, current, current, 512], &a);
        a = self.mid2.forward(&a, current)?;
        a = self
            .conv_out
            .forward(&silu(self.norm.forward(&a)), current, 1)?;
        a = self.quant.forward(&a, current, 1)?;
        trace("idol.vae.moments_nhwc", &[1, current, current, 64], &a);
        let tokens = posterior_tokens(&a, current, noise, &self.mean, &self.std)?;
        trace(
            "idol.condition.feature2",
            &[1, 5 + (current / 2) * (current / 2), 128],
            &tokens,
        );
        Ok(tokens)
    }
}
fn posterior_tokens(
    moments: &[f32],
    side: usize,
    noise: &[f32],
    mean: &[f32],
    std: &[f32],
) -> Result<Vec<f32>, String> {
    let spatial = side * side;
    let packed_side = side / 2;
    let mut tokens = vec![0.; (5 + packed_side * packed_side) * 128];
    tokens[5 * 128..]
        .par_chunks_mut(128)
        .enumerate()
        .for_each(|(p, row)| {
            for c in 0..32 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let pixel = (p / packed_side * 2 + dy) * side + p % packed_side * 2 + dx;
                        let channel = c * 4 + dy * 2 + dx;
                        let mu = moments[pixel * 64 + c];
                        let logvar = moments[pixel * 64 + 32 + c].clamp(-30., 20.);
                        let sigma = b(b(0.5 * logvar).exp());
                        let sampled = b(mu + b(sigma * b(noise[c * spatial + pixel])));
                        row[channel] = b(b(sampled - mean[channel]) / std[channel]);
                    }
                }
            }
        });
    finite(&tokens, "VAE tokens")?;
    Ok(tokens)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn posterior_noise_is_chw_and_pack_is_channel_y_x() {
        let mut moments = vec![0.; 4 * 64];
        for pixel in 0..4 {
            for c in 0..32 {
                moments[pixel * 64 + c] = (c * 4 + pixel) as f32;
            }
        }
        let out = posterior_tokens(&moments, 2, &vec![0.; 128], &[0.; 128], &[1.; 128]).unwrap();
        assert_eq!(&out[..640], &vec![0.; 640]);
        assert_eq!(&out[640..], &(0..128).map(|v| v as f32).collect::<Vec<_>>());
        let noisy =
            posterior_tokens(&vec![0.; 256], 2, &vec![1.; 128], &[0.; 128], &[1.; 128]).unwrap();
        assert!(noisy[640..].iter().all(|&v| v == 1.));
    }
}
