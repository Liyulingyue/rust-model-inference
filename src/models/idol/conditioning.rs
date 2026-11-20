//! Idol's released DINOv3 ViT-H+/16 and FLUX.2 image conditioning.
use super::common::{finite, norm, tensor, trace, Linear};
use crate::core::tensor::TensorSource;
use rayon::prelude::*;
#[path = "conditioning_vae.rs"]
mod vae;

const WIDTH: usize = 1280;
const HEADS: usize = 20;
const HEAD: usize = 64;

pub(super) struct Conditioning {
    dino: Dino,
    vae: vae::Encoder,
}
impl Conditioning {
    pub fn load(dino: &dyn TensorSource, vae: &dyn TensorSource) -> Result<Self, String> {
        Ok(Self {
            dino: Dino::load(dino)?,
            vae: vae::Encoder::load(vae)?,
        })
    }
    pub fn encode(&self, rgb: &[u8], noise: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        Ok((
            self.dino_tokens(rgb, 1024)?,
            self.vae_tokens(rgb, 1024, noise)?,
        ))
    }
    /// Float32 DINO path, corresponding to the official `dinov3_dtype="fp32"` option.
    pub fn dino_tokens(&self, rgb: &[u8], side: usize) -> Result<Vec<f32>, String> {
        self.dino.forward(rgb, side)
    }
    /// `noise` uses PyTorch's [32, side/8, side/8] channel-first order.
    pub fn vae_tokens(&self, rgb: &[u8], side: usize, noise: &[f32]) -> Result<Vec<f32>, String> {
        self.vae.tokens(rgb, side, noise)
    }
}

pub(super) fn validate_rgb(rgb: &[u8], side: usize) -> Result<(), String> {
    if side == 0
        || side % 16 != 0
        || side > 1024
        || side.checked_mul(side).and_then(|n| n.checked_mul(3)) != Some(rgb.len())
    {
        return Err(
            "Idol encoder expects square RGB bytes, side a positive multiple of 16 at most 1024"
                .into(),
        );
    }
    Ok(())
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl LayerNorm {
    fn load(s: &dyn TensorSource, p: &str) -> Result<Self, String> {
        Ok(Self {
            weight: tensor(s, &format!("{p}.weight"), &[WIDTH])?,
            bias: tensor(s, &format!("{p}.bias"), &[WIDTH])?,
        })
    }
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        norm(x, WIDTH, Some(&self.weight), Some(&self.bias), 1e-5)
    }
}
struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    scale1: Vec<f32>,
    scale2: Vec<f32>,
}
impl Block {
    fn load(s: &dyn TensorSource, p: &str) -> Result<Self, String> {
        let linear = |suffix: &str, input, output, bias| {
            Linear::load(s, &format!("{p}.{suffix}"), input, output, bias)
        };
        Ok(Self {
            norm1: LayerNorm::load(s, &format!("{p}.norm1"))?,
            norm2: LayerNorm::load(s, &format!("{p}.norm2"))?,
            q: linear("attention.q_proj", WIDTH, WIDTH, true)?,
            k: linear("attention.k_proj", WIDTH, WIDTH, false)?,
            v: linear("attention.v_proj", WIDTH, WIDTH, true)?,
            o: linear("attention.o_proj", WIDTH, WIDTH, true)?,
            gate: linear("mlp.gate_proj", WIDTH, 5120, true)?,
            up: linear("mlp.up_proj", WIDTH, 5120, true)?,
            down: linear("mlp.down_proj", 5120, WIDTH, true)?,
            scale1: tensor(s, &format!("{p}.layer_scale1.lambda1"), &[WIDTH])?,
            scale2: tensor(s, &format!("{p}.layer_scale2.lambda1"), &[WIDTH])?,
        })
    }
    fn forward(&self, x: &mut Vec<f32>, side: usize) -> Result<(), String> {
        let normalized = self.norm1.forward(x);
        let mut q = self.q.forward(&normalized)?;
        let mut k = self.k.forward(&normalized)?;
        let v = self.v.forward(&normalized)?;
        rope(&mut q, side);
        rope(&mut k, side);
        let a = attention(&q, &k, &v, WIDTH, HEADS)?;
        let a = self.o.forward(&a)?;
        for (i, (out, &a)) in x.iter_mut().zip(&a).enumerate() {
            *out += a * self.scale1[i % WIDTH];
        }
        let normalized = self.norm2.forward(x);
        let gate = self.gate.forward(&normalized)?;
        let mut up = self.up.forward(&normalized)?;
        crate::ops::silu_mul_inplace(&gate, &mut up);
        let a = self.down.forward(&up)?;
        for (i, (out, &a)) in x.iter_mut().zip(&a).enumerate() {
            *out += a * self.scale2[i % WIDTH];
        }
        Ok(())
    }
}
struct Dino {
    patch: Vec<f32>,
    bias: Vec<f32>,
    prefix: Vec<f32>,
    blocks: Vec<Block>,
    norm: LayerNorm,
}
impl Dino {
    fn load(s: &dyn TensorSource) -> Result<Self, String> {
        let mut prefix = tensor(s, "embeddings.cls_token", &[1, 1, WIDTH])?;
        prefix.extend(tensor(s, "embeddings.register_tokens", &[1, 4, WIDTH])?);
        Ok(Self {
            patch: tensor(s, "embeddings.patch_embeddings.weight", &[WIDTH, 3, 16, 16])?,
            bias: tensor(s, "embeddings.patch_embeddings.bias", &[WIDTH])?,
            prefix,
            blocks: (0..32)
                .map(|i| Block::load(s, &format!("layer.{i}")))
                .collect::<Result<_, _>>()?,
            norm: LayerNorm::load(s, "norm")?,
        })
    }
    fn forward(&self, rgb: &[u8], side: usize) -> Result<Vec<f32>, String> {
        validate_rgb(rgb, side)?;
        let patches = side / 16;
        let mut x = vec![0.; (5 + patches * patches) * WIDTH];
        x[..5 * WIDTH].copy_from_slice(&self.prefix);
        x[5 * WIDTH..]
            .par_chunks_mut(WIDTH)
            .enumerate()
            .for_each(|(index, out)| {
                let mut patch = [0.; 768];
                for c in 0..3 {
                    for y in 0..16 {
                        for z in 0..16 {
                            let byte =
                                rgb[((index / patches * 16 + y) * side + index % patches * 16 + z)
                                    * 3
                                    + c];
                            patch[(c * 16 + y) * 16 + z] = (byte as f32 / 255.
                                - [0.485, 0.456, 0.406][c])
                                / [0.229, 0.224, 0.225][c];
                        }
                    }
                }
                for (i, value) in out.iter_mut().enumerate() {
                    *value = crate::ops::dot_f32(&patch, &self.patch[i * 768..(i + 1) * 768], 768)
                        + self.bias[i];
                }
            });
        trace(
            "idol.dino.embeddings",
            &[1, 5 + patches * patches, WIDTH],
            &x,
        );
        for (i, b) in self.blocks.iter().enumerate() {
            b.forward(&mut x, patches)?;
            finite(&x, "DINO block")?;
            trace(
                &format!("idol.dino.block.{i}"),
                &[1, 5 + patches * patches, WIDTH],
                &x,
            );
        }
        x = self.norm.forward(&x);
        trace(
            "idol.dino.last_hidden_state",
            &[1, 5 + patches * patches, WIDTH],
            &x,
        );
        x = norm(&x, WIDTH, None, None, 1e-5);
        finite(&x, "DINO tokens")?;
        trace(
            "idol.condition.feature1",
            &[1, 5 + patches * patches, WIDTH],
            &x,
        );
        Ok(x)
    }
}
fn rope(x: &mut [f32], patches: usize) {
    x[5 * WIDTH..]
        .par_chunks_mut(WIDTH)
        .enumerate()
        .for_each(|(p, row)| {
            for head in row.chunks_mut(HEAD) {
                for i in 0..HEAD / 2 {
                    let coordinate = if i < HEAD / 4 {
                        p / patches
                    } else {
                        p % patches
                    };
                    let coordinate = 2. * ((coordinate as f32 + 0.5) / patches as f32) - 1.;
                    let angle = 2.
                        * std::f32::consts::PI
                        * coordinate
                        * 100.0f32.powf(-((i % (HEAD / 4)) as f32) / (HEAD / 4) as f32);
                    let (sin, cos) = angle.sin_cos();
                    let a = head[i];
                    let b = head[i + HEAD / 2];
                    head[i] = a * cos - b * sin;
                    head[i + HEAD / 2] = b * cos + a * sin;
                }
            }
        });
}
pub(super) fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    width: usize,
    heads: usize,
) -> Result<Vec<f32>, String> {
    if let Some(output) = super::gpu::attention(q, k, v, width, heads) {
        return Ok(output);
    }
    super::gpu::check_failure()?;
    let tokens = q.len() / width;
    let head = width / heads;
    let scale = (head as f32).sqrt().recip();
    let mut output = vec![0.; q.len()];
    output
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, out)| {
            let mut scores = vec![0.; tokens];
            for h in 0..heads {
                let offset = h * head;
                let query = &q[row * width + offset..row * width + offset + head];
                for (t, score) in scores.iter_mut().enumerate() {
                    *score = crate::ops::dot_f32(
                        query,
                        &k[t * width + offset..t * width + offset + head],
                        head,
                    ) * scale;
                }
                crate::ops::softmax_inplace(&mut scores);
                crate::ops::attention_value_reduce(
                    v,
                    &scores,
                    &mut out[offset..offset + head],
                    0,
                    width,
                    offset,
                    tokens,
                    head,
                );
            }
        });
    Ok(output)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attention_identity_and_rope_prefix() {
        assert_eq!(
            attention(&[0., 0.], &[0., 0.], &[2., 4.], 1, 1).unwrap(),
            vec![3., 3.]
        );
        let mut x = vec![1.; 6 * WIDTH];
        rope(&mut x, 1);
        assert!(x.iter().all(|&v| v == 1.));
        assert!(validate_rgb(&[0; 3], 0).is_err());
        assert!(validate_rgb(&[0; 3], 16).is_err());
    }
}
