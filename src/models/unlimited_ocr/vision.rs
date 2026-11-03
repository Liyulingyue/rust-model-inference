use super::{linear, trace, vector, HIDDEN};
use crate::models::diffusion::dreamx::kernels::{layer_norm_rows, load_float_values, Linear};
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::{ComputePool, GGMLType, TensorSource};
use rayon::prelude::*;
use std::path::Path;

pub(super) fn preprocess(path: &Path) -> Result<Vec<f32>, String> {
    let image = image::open(path).map_err(|e| format!("Read OCR page {}: {e}", path.display()))?;
    preprocess_image(image)
}

pub(super) fn preprocess_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let image = image::load_from_memory(bytes).map_err(|e| format!("Read OCR page: {e}"))?;
    preprocess_image(image)
}

fn preprocess_image(image: image::DynamicImage) -> Result<Vec<f32>, String> {
    if !matches!(
        image.color(),
        image::ColorType::L8
            | image::ColorType::La8
            | image::ColorType::Rgb8
            | image::ColorType::Rgba8
    ) {
        return Err(
            "Unlimited-OCR requires 8-bit page images (convert higher bit depths explicitly)"
                .into(),
        );
    }
    let image = image.to_rgb8();
    let (w, h) = (image.width() as usize, image.height() as usize);
    let (nw, nh) = if w > h {
        (
            1024,
            ((h as f64 * 1024.0 / w as f64).round_ties_even() as usize).max(1),
        )
    } else {
        (
            ((w as f64 * 1024.0 / h as f64).round_ties_even() as usize).max(1),
            1024,
        )
    };
    let resized =
        crate::models::gemma4::vision::resize_bicubic_pillow(image.as_raw(), w, h, nw, nh)?;
    let mut padded = vec![127_u8; 1024 * 1024 * 3];
    // Pillow ImageOps.pad centers odd margins with Python's ties-to-even round.
    let left = ((1024 - nw) as f64 / 2.0).round_ties_even() as usize;
    let top = ((1024 - nh) as f64 / 2.0).round_ties_even() as usize;
    for y in 0..nh {
        let dst = ((top + y) * 1024 + left) * 3;
        padded[dst..dst + nw * 3].copy_from_slice(&resized[y * nw * 3..(y + 1) * nw * 3]);
    }
    Ok(padded
        .into_iter()
        .map(|p| half::bf16::from_f32((p as f32 / 255.0 - 0.5) / 0.5).to_f32())
        .collect())
}
struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    eps: f32,
}
impl Norm {
    fn load(s: &dyn TensorSource, p: &str, d: usize, eps: f32) -> Result<Self, String> {
        Ok(Self {
            weight: vector(s, &format!("{p}.weight"), d)?,
            bias: vector(s, &format!("{p}.bias"), d)?,
            eps,
        })
    }
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        layer_norm_rows(
            x,
            x.len() / self.weight.len(),
            Some(&self.weight),
            Some(&self.bias),
            self.eps,
        )
    }
}
struct Conv<'a> {
    linear: Linear<'a>,
    cin: usize,
    k: usize,
    stride: usize,
    pad: usize,
}
impl<'a> Conv<'a> {
    fn load(
        s: &'a dyn TensorSource,
        p: &str,
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        pad: usize,
        bias: bool,
    ) -> Result<Self, String> {
        let name = format!("{p}.weight");
        let info = s
            .tensor_info(&name)
            .ok_or_else(|| format!("Missing {name}"))?;
        if info.dims != [k as u64, k as u64, cin as u64, cout as u64]
            || !matches!(
                info.ggml_type,
                GGMLType::BF16 | GGMLType::F16 | GGMLType::F32
            )
        {
            return Err(format!("Invalid OCR convolution {name}"));
        }
        let bytes = s
            .tensor_slice(&name)
            .ok_or_else(|| format!("Missing {name} bytes"))?;
        if info.checked_nbytes() != Some(bytes.len() as u64) {
            return Err(format!("Invalid OCR convolution storage {name}"));
        }
        let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
            bytes,
            info.ggml_type,
            cin * k * k,
            cout,
        ));
        weight.n_in = cin * k * k;
        weight.n_out = cout;
        Ok(Self {
            linear: Linear::from_weight(
                weight,
                if bias {
                    Some(vector(s, &format!("{p}.bias"), cout)?)
                } else {
                    None
                },
            )?,
            cin,
            k,
            stride,
            pad,
        })
    }
    fn forward(
        &self,
        pool: &ComputePool,
        x: &[f32],
        side: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        if x.len() != side * side * self.cin || side + 2 * self.pad < self.k {
            return Err("Invalid OCR convolution input".into());
        }
        let out = (side + 2 * self.pad - self.k) / self.stride + 1;
        let width = self.cin * self.k * self.k;
        let mut patches = vec![0.0; out * out * width];
        patches
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, patch)| {
                for c in 0..self.cin {
                    for ky in 0..self.k {
                        for kx in 0..self.k {
                            let y = (row / out * self.stride + ky) as isize - self.pad as isize;
                            let xx = (row % out * self.stride + kx) as isize - self.pad as isize;
                            if y >= 0 && xx >= 0 && y < side as isize && xx < side as isize {
                                patch[(c * self.k + ky) * self.k + kx] =
                                    x[(y as usize * side + xx as usize) * self.cin + c];
                            }
                        }
                    }
                }
            });
        Ok((
            self.linear.forward_verified(pool, &patches, out * out)?,
            out,
        ))
    }
}
struct Block<'a> {
    n1: Norm,
    n2: Norm,
    qkv: Linear<'a>,
    proj: Linear<'a>,
    up: Linear<'a>,
    down: Linear<'a>,
    rel_h: Vec<f32>,
    rel_w: Vec<f32>,
    window: usize,
}

pub(super) struct Vision<'a> {
    patch: Conv<'a>,
    sam_pos: Vec<f32>,
    sam: Vec<Block<'a>>,
    neck0: Conv<'a>,
    neck1: Norm,
    neck2: Conv<'a>,
    neck3: Norm,
    net2: Conv<'a>,
    net3: Conv<'a>,
    cls: Vec<f32>,
    clip_pos: Vec<f32>,
    pre: Norm,
    clip: Vec<Block<'a>>,
    projector: Linear<'a>,
    newline: Vec<f32>,
    separator: Vec<f32>,
}
impl<'a> Vision<'a> {
    pub fn load(s: &'a dyn TensorSource) -> Result<Self, String> {
        let ids = "model.vision_model.embeddings.position_ids";
        let info = s.tensor_info(ids).ok_or_else(|| format!("Missing {ids}"))?;
        let bytes = s
            .tensor_slice(ids)
            .ok_or_else(|| format!("Missing {ids} bytes"))?;
        if info.dims != [257, 1]
            || info.ggml_type != GGMLType::I64
            || bytes.len() != 257 * 8
            || bytes
                .chunks_exact(8)
                .enumerate()
                .any(|(i, v)| i64::from_le_bytes(v.try_into().unwrap()) != i as i64)
        {
            return Err("Unlimited-OCR requires the original CLIP position_ids 0..256".into());
        }
        let mut sam = Vec::new();
        for i in 0..12 {
            let p = format!("model.sam_model.blocks.{i}");
            let window = if [2, 5, 8, 11].contains(&i) { 64 } else { 14 };
            sam.push(Block {
                n1: Norm::load(s, &format!("{p}.norm1"), 768, 1e-6)?,
                n2: Norm::load(s, &format!("{p}.norm2"), 768, 1e-6)?,
                qkv: linear(s, &format!("{p}.attn.qkv"), 768, 2304, true)?,
                proj: linear(s, &format!("{p}.attn.proj"), 768, 768, true)?,
                up: linear(s, &format!("{p}.mlp.lin1"), 768, 3072, true)?,
                down: linear(s, &format!("{p}.mlp.lin2"), 3072, 768, true)?,
                rel_h: load_float_values(
                    s,
                    &format!("{p}.attn.rel_pos_h"),
                    &[64, (window * 2 - 1) as u64],
                )?,
                rel_w: load_float_values(
                    s,
                    &format!("{p}.attn.rel_pos_w"),
                    &[64, (window * 2 - 1) as u64],
                )?,
                window,
            });
        }
        let mut clip = Vec::new();
        for i in 0..24 {
            let p = format!("model.vision_model.transformer.layers.{i}");
            clip.push(Block {
                n1: Norm::load(s, &format!("{p}.layer_norm1"), 1024, 1e-5)?,
                n2: Norm::load(s, &format!("{p}.layer_norm2"), 1024, 1e-5)?,
                qkv: linear(s, &format!("{p}.self_attn.qkv_proj"), 1024, 3072, true)?,
                proj: linear(s, &format!("{p}.self_attn.out_proj"), 1024, 1024, true)?,
                up: linear(s, &format!("{p}.mlp.fc1"), 1024, 4096, true)?,
                down: linear(s, &format!("{p}.mlp.fc2"), 4096, 1024, true)?,
                rel_h: Vec::new(),
                rel_w: Vec::new(),
                window: 0,
            });
        }
        Ok(Self {
            patch: Conv::load(
                s,
                "model.sam_model.patch_embed.proj",
                3,
                768,
                16,
                16,
                0,
                true,
            )?,
            sam_pos: load_float_values(s, "model.sam_model.pos_embed", &[768, 64, 64, 1])?,
            sam,
            neck0: Conv::load(s, "model.sam_model.neck.0", 768, 256, 1, 1, 0, false)?,
            neck1: Norm::load(s, "model.sam_model.neck.1", 256, 1e-6)?,
            neck2: Conv::load(s, "model.sam_model.neck.2", 256, 256, 3, 1, 1, false)?,
            neck3: Norm::load(s, "model.sam_model.neck.3", 256, 1e-6)?,
            net2: Conv::load(s, "model.sam_model.net_2", 256, 512, 3, 2, 1, false)?,
            net3: Conv::load(s, "model.sam_model.net_3", 512, 1024, 3, 2, 1, false)?,
            cls: vector(s, "model.vision_model.embeddings.class_embedding", 1024)?,
            clip_pos: load_float_values(
                s,
                "model.vision_model.embeddings.position_embedding.weight",
                &[1024, 257],
            )?,
            pre: Norm::load(s, "model.vision_model.pre_layrnorm", 1024, 1e-5)?,
            clip,
            projector: linear(s, "model.projector.layers", 2048, HIDDEN, true)?,
            newline: vector(s, "model.image_newline", HIDDEN)?,
            separator: vector(s, "model.view_seperator", HIDDEN)?,
        })
    }
    pub fn encode(&self, pool: &ComputePool, pixels: &[f32]) -> Result<Vec<f32>, String> {
        let (mut x, side) = self.patch.forward(pool, pixels, 1024)?;
        assert_eq!(side, 64);
        for (v, p) in x.iter_mut().zip(&self.sam_pos) {
            *v += p;
        }
        trace("uocr.sam_embed", None, &[64, 64, 768], &x);
        for (i, b) in self.sam.iter().enumerate() {
            let norm = b.n1.forward(&x)?;
            let mut attended = vec![0.0; x.len()];
            let w = b.window;
            for wy in 0..64_usize.div_ceil(w) {
                for wx in 0..64_usize.div_ceil(w) {
                    let mut window = vec![0.0; w * w * 768];
                    for y in 0..w {
                        for xx in 0..w {
                            let gy = wy * w + y;
                            let gx = wx * w + xx;
                            if gy < 64 && gx < 64 {
                                window[(y * w + xx) * 768..(y * w + xx + 1) * 768].copy_from_slice(
                                    &norm[(gy * 64 + gx) * 768..(gy * 64 + gx + 1) * 768],
                                );
                            }
                        }
                    }
                    let qkv = b.qkv.forward_verified(pool, &window, w * w)?;
                    let attn = attention(&qkv, w * w, 12, Some((&b.rel_h, &b.rel_w, w)));
                    let values = b.proj.forward_verified(pool, &attn, w * w)?;
                    for y in 0..w {
                        for xx in 0..w {
                            let gy = wy * w + y;
                            let gx = wx * w + xx;
                            if gy < 64 && gx < 64 {
                                attended[(gy * 64 + gx) * 768..(gy * 64 + gx + 1) * 768]
                                    .copy_from_slice(
                                        &values[(y * w + xx) * 768..(y * w + xx + 1) * 768],
                                    );
                            }
                        }
                    }
                }
            }
            for (v, a) in x.iter_mut().zip(attended) {
                *v += a;
            }
            let norm = b.n2.forward(&x)?;
            let mut up = b.up.forward_verified(pool, &norm, 4096)?;
            crate::ops::gelu_erf_inplace(&mut up);
            let down = b.down.forward_verified(pool, &up, 4096)?;
            for (v, d) in x.iter_mut().zip(down) {
                *v += d;
            }
            trace("uocr.sam_layer", Some(i), &[64, 64, 768], &x);
        }
        let (x, _) = self.neck0.forward(pool, &x, 64)?;
        let x = self.neck1.forward(&x)?;
        let (x, _) = self.neck2.forward(pool, &x, 64)?;
        let x = self.neck3.forward(&x)?;
        let (x, side) = self.net2.forward(pool, &x, 64)?;
        let (sam, side) = self.net3.forward(pool, &x, side)?;
        assert_eq!(side, 16);
        trace("uocr.sam", None, &[16, 16, 1024], &sam);
        let mut x = self.cls.clone();
        x.extend_from_slice(&sam);
        for (v, p) in x.iter_mut().zip(&self.clip_pos) {
            *v += p;
        }
        let mut x = self.pre.forward(&x)?;
        trace("uocr.clip_embed", None, &[257, 1024], &x);
        for (i, b) in self.clip.iter().enumerate() {
            let norm = b.n1.forward(&x)?;
            let qkv = b.qkv.forward_verified(pool, &norm, 257)?;
            let attn = attention(&qkv, 257, 16, None);
            let projected = b.proj.forward_verified(pool, &attn, 257)?;
            for (v, a) in x.iter_mut().zip(projected) {
                *v += a;
            }
            let norm = b.n2.forward(&x)?;
            let mut up = b.up.forward_verified(pool, &norm, 257)?;
            for v in &mut up {
                *v *= 1.0 / (1.0 + (-1.702 * *v).exp());
            }
            let down = b.down.forward_verified(pool, &up, 257)?;
            for (v, a) in x.iter_mut().zip(down) {
                *v += a;
            }
            trace("uocr.clip_layer", Some(i), &[257, 1024], &x);
        }
        let mut combined = Vec::with_capacity(256 * 2048);
        for (clip, sam) in x[1024..].chunks_exact(1024).zip(sam.chunks_exact(1024)) {
            combined.extend_from_slice(clip);
            combined.extend_from_slice(sam);
        }
        let projected = self.projector.forward_verified(pool, &combined, 256)?;
        let mut result = Vec::with_capacity(273 * HIDDEN);
        for row in projected.chunks_exact(16 * HIDDEN) {
            result.extend_from_slice(row);
            result.extend_from_slice(&self.newline);
        }
        result.extend_from_slice(&self.separator);
        if result.iter().any(|v| !v.is_finite()) {
            return Err("Non-finite Unlimited-OCR visual features".into());
        }
        trace("uocr.visual", None, &[273, HIDDEN], &result);
        Ok(result)
    }
}
fn attention(
    qkv: &[f32],
    tokens: usize,
    heads: usize,
    rel: Option<(&[f32], &[f32], usize)>,
) -> Vec<f32> {
    let width = heads * 64;
    let mut out = vec![0.0; tokens * width];
    out.par_chunks_mut(64).enumerate().for_each(|(i, output)| {
        let row = i / heads;
        let head = i % heads;
        let qoff = row * width * 3 + head * 64;
        let query = &qkv[qoff..qoff + 64];
        let (mut rh, mut rw) = (Vec::new(), Vec::new());
        if let Some((h, w, side)) = rel {
            for k in 0..side {
                let ih = (row / side + side - 1 - k) * 64;
                let iw = (row % side + side - 1 - k) * 64;
                rh.push(crate::ops::dot_f32(query, &h[ih..ih + 64], 64));
                rw.push(crate::ops::dot_f32(query, &w[iw..iw + 64], 64));
            }
        }
        let mut scores: Vec<_> = (0..tokens)
            .map(|key| {
                let koff = key * width * 3 + width + head * 64;
                let score = crate::ops::dot_f32(query, &qkv[koff..koff + 64], 64) * 0.125;
                score + rel.map_or(0.0, |(_, _, side)| rh[key / side] + rw[key % side])
            })
            .collect();
        crate::ops::softmax_inplace(&mut scores);
        for (key, p) in scores.into_iter().enumerate() {
            let voff = key * width * 3 + width * 2 + head * 64;
            for d in 0..64 {
                output[d] += p * qkv[voff + d];
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    #[test]
    fn rejects_16_bit_images_instead_of_rescaling_pillow_integer_samples() {
        let image = image::ImageBuffer::from_pixel(2, 2, image::Luma([256_u16]));
        let path = std::env::temp_dir().join(format!("rmi-uocr-gray16-{}.png", std::process::id()));
        image.save(&path).unwrap();
        let error = preprocess(&path).unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert!(error.contains("8-bit"));
    }

    #[test]
    fn normalized_pixels_match_official_pillow_and_bf16_input() {
        let image = image::RgbImage::from_fn(23, 17, |x, y| {
            image::Rgb([
                (x * 11 % 256) as u8,
                (y * 15 % 256) as u8,
                ((x * 7 + y * 13) % 256) as u8,
            ])
        });
        let path = std::env::temp_dir().join(format!("rmi-uocr-pixels-{}.png", std::process::id()));
        image.save(&path).unwrap();
        let pixels = preprocess(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let bytes: Vec<_> = pixels.iter().flat_map(|x| x.to_le_bytes()).collect();
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/unlimited_ocr_oracle.json"
        ))
        .unwrap();
        assert_eq!(
            format!("{:x}", sha2::Sha256::digest(bytes)),
            fixture["pixel_sha256"].as_str().unwrap()
        );
    }

    #[test]
    fn convolution_uses_pytorch_channel_kernel_order_and_padding() {
        let values = [1_f32, 2., 3., 4., 10., 20., 30., 40.];
        let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let weight =
            Weight::from_quantized(QuantizedTensor::from_bytes(&bytes, GGMLType::F32, 8, 1));
        let conv = Conv {
            linear: Linear::from_weight(weight, Some(vec![0.5])).unwrap(),
            cin: 2,
            k: 2,
            stride: 1,
            pad: 1,
        };
        // NHWC: channel zero is 1..4; channel one is 5..8.
        let (out, side) = conv
            .forward(&ComputePool::new(1), &[1., 5., 2., 6., 3., 7., 4., 8.], 2)
            .unwrap();
        assert_eq!(side, 3);
        assert_eq!(out[0], 204.5);
        assert_eq!(out[4], 730.5);
        assert_eq!(out[8], 84.5);
        assert!(conv.forward(&ComputePool::new(1), &[], 2).is_err());
    }

    #[test]
    fn visual_attention_respects_qkv_layout_and_relative_position() {
        let mut qkv = vec![0.0; 4 * 192];
        for row in 0..4 {
            qkv[row * 192] = 1.0;
            qkv[row * 192 + 128] = (row + 1) as f32;
        }
        let mut rh = vec![0.0; 3 * 64];
        rh[64] = 2_f32.ln();
        let rw = vec![0.0; 3 * 64];
        let out = attention(&qkv, 4, 1, Some((&rh, &rw, 2)));
        // Same-row keys receive twice the probability of the other row.
        assert!((out[0] - 13.0 / 6.0).abs() < 1e-6);
        assert!((out[128] - 17.0 / 6.0).abs() < 1e-6);
    }
}
