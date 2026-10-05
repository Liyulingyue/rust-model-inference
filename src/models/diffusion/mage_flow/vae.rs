use super::dit::{trace, unweighted_norm};
use crate::core::tensor::{load_f32_tensor, GGMLType, MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::diffusion::dreamx::kernels::{
    attention_scalar_f32, checked_len, conv2d, group_norm_ncthw, layer_norm_rows, rms_norm_rows,
    AttentionSpec, Linear,
};
use crate::ops::{gelu_erf_inplace, silu_inplace, sum_f32};
use std::sync::Arc;

struct Map {
    data: Vec<f32>,
    shape: [usize; 3],
}
impl Map {
    fn zeros(shape: [usize; 3]) -> Result<Self, String> {
        Ok(Self {
            data: vec![0.0; checked_len("MageVAE map", &shape)?],
            shape,
        })
    }
    fn rows(&self) -> Vec<f32> {
        let [c, h, w] = self.shape;
        let n = h * w;
        (0..n * c).map(|i| self.data[(i % c) * n + i / c]).collect()
    }
    fn from_rows(data: &[f32], shape: [usize; 3]) -> Self {
        let [c, h, w] = shape;
        let n = h * w;
        Self {
            data: (0..data.len()).map(|i| data[(i % n) * c + i / n]).collect(),
            shape,
        }
    }
}

/// Deterministic MageVAE: the released config selects posterior mean, without sampling.
pub struct MageVae<'a> {
    source: &'a dyn TensorSource,
    pool: Arc<ComputePool>,
}
impl<'a> MageVae<'a> {
    pub fn load(source: &'a dyn TensorSource, pool: Arc<ComputePool>) -> Result<Self, String> {
        if source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            != Some("mage_vae")
            || !matches!(
                source.metadata("mage_vae.sample_posterior"),
                Some(MetaValue::Bool(false))
            )
        {
            return Err("Expected deterministic mage_vae GGUF".into());
        }
        Ok(Self { source, pool })
    }
    fn vector(&self, name: &str, width: usize) -> Result<Vec<f32>, String> {
        load_f32_tensor(self.source, name, &[width as u64])
    }
    fn linear(
        &self,
        name: &str,
        input: &[f32],
        rows: usize,
        width: usize,
        out: usize,
    ) -> Result<Vec<f32>, String> {
        let weight_name = format!("{name}.weight");
        if self
            .source
            .tensor_info(&weight_name)
            .is_none_or(|i| i.ggml_type != GGMLType::BF16)
        {
            return Err(format!("Expected BF16 {weight_name}"));
        }
        let layer = Linear::from_source(
            self.source,
            &weight_name,
            Some(&format!("{name}.bias")),
            width,
            out,
        )?;
        let y = layer.forward(&self.pool, input, rows)?;
        trace(&format!("vae.{name}"), 1, y.len(), &y)?;
        Ok(y)
    }
    fn conv(
        &self,
        name: &str,
        input: &Map,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Map, String> {
        let weight_name = format!("{name}.weight");
        let info = self
            .source
            .tensor_info(&weight_name)
            .ok_or_else(|| format!("Missing {weight_name}"))?;
        if info.ggml_type != GGMLType::BF16
            || info.dims.len() != 4
            || info.dims[..3]
                != [
                    kernel as u64,
                    kernel as u64,
                    (input.shape[0] / groups) as u64,
                ]
        {
            return Err(format!("Invalid MageVAE convolution {weight_name}"));
        }
        let out = usize::try_from(info.dims[3]).map_err(|e| e.to_string())?;
        let w = load_f32_tensor(self.source, &weight_name, &info.dims)?;
        let b = if bias {
            Some(self.vector(&format!("{name}.bias"), out)?)
        } else {
            None
        };
        let (data, shape) = conv2d(
            &self.pool,
            &input.data,
            input.shape,
            &w,
            [out, input.shape[0] / groups, kernel, kernel],
            b.as_deref(),
            [stride; 2],
            [padding; 2],
            [1; 2],
            groups,
        )?;
        trace(&format!("vae.{name}"), 1, data.len(), &data)?;
        Ok(Map { data, shape })
    }
    fn norm(&self, input: &Map, name: Option<&str>) -> Result<Map, String> {
        let [width, h, w] = input.shape;
        let rows = input.rows();
        let values = if let Some(name) = name {
            let weight = self.vector(&format!("{name}.weight"), width)?;
            let bias = self.vector(&format!("{name}.bias"), width)?;
            layer_norm_rows(&rows, h * w, Some(&weight), Some(&bias), 1e-6)?
        } else {
            unweighted_norm(&rows, h * w, width)?
        };
        Ok(Map::from_rows(&values, input.shape))
    }
    fn group_norm(&self, input: &Map, name: &str) -> Result<Map, String> {
        let [c, h, w] = input.shape;
        let data = group_norm_ncthw(
            &input.data,
            [c, 1, h, w],
            32,
            &self.vector(&format!("{name}.weight"), c)?,
            &self.vector(&format!("{name}.bias"), c)?,
            1e-6,
        )?;
        Ok(Map {
            data,
            shape: input.shape,
        })
    }
    fn time(&self, prefix: &str) -> Result<Vec<f32>, String> {
        let mut time = vec![0.0; 256];
        time[..128].fill(1.0);
        let mut time = self.linear(&format!("{prefix}.t_embedder.mlp.0"), &time, 1, 256, 384)?;
        silu_inplace(&mut time);
        self.linear(&format!("{prefix}.t_embedder.mlp.2"), &time, 1, 384, 384)
    }
    fn dico(&self, input: Map, name: &str, condition: Option<&[f32]>) -> Result<Map, String> {
        let [c, h, w] = input.shape;
        let plane = h * w;
        let modulation = if let Some(condition) = condition {
            let mut condition = condition.to_vec();
            silu_inplace(&mut condition);
            Some(self.linear(
                &format!("{name}.adaLN_modulation.1"),
                &condition,
                1,
                c,
                6 * c,
            )?)
        } else {
            None
        };
        let mut norm = self.norm(
            &input,
            condition
                .is_none()
                .then_some(format!("{name}.norm1"))
                .as_deref(),
        )?;
        if let Some(m) = &modulation {
            modulate_map(&mut norm, &m[..c], &m[c..2 * c]);
        }
        let x = self.conv(&format!("{name}.conv1"), &norm, 1, 1, 0, 1, true)?;
        let mut x = self.conv(&format!("{name}.conv2"), &x, 3, 1, 1, c, true)?;
        gelu_erf_inplace(&mut x.data);
        let avg = Map {
            data: x
                .data
                .chunks_exact(plane)
                .map(|row| (sum_f32(row) / plane as f64) as f32)
                .collect(),
            shape: [c, 1, 1],
        };
        let mut ca = self.conv(&format!("{name}.ca.1"), &avg, 1, 1, 0, 1, true)?;
        for v in &mut ca.data {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
        for (ch, row) in x.data.chunks_exact_mut(plane).enumerate() {
            for v in row {
                *v *= ca.data[ch];
            }
        }
        let x = self.conv(&format!("{name}.conv3"), &x, 1, 1, 0, 1, true)?;
        let mut result = input;
        for (i, (v, b)) in result.data.iter_mut().zip(&x.data).enumerate() {
            *v += modulation.as_ref().map_or(1.0, |m| m[2 * c + i / plane]) * b;
        }
        let mut norm = self.norm(
            &result,
            condition
                .is_none()
                .then_some(format!("{name}.norm2"))
                .as_deref(),
        )?;
        if let Some(m) = &modulation {
            modulate_map(&mut norm, &m[3 * c..4 * c], &m[4 * c..5 * c]);
        }
        let mut x = self.conv(&format!("{name}.conv4"), &norm, 1, 1, 0, 1, true)?;
        gelu_erf_inplace(&mut x.data);
        let x = self.conv(&format!("{name}.conv5"), &x, 1, 1, 0, 1, true)?;
        for (i, (v, b)) in result.data.iter_mut().zip(&x.data).enumerate() {
            *v += modulation.as_ref().map_or(1.0, |m| m[5 * c + i / plane]) * b;
        }
        trace(&format!("vae.{name}"), 1, result.data.len(), &result.data)?;
        Ok(result)
    }
    pub fn encode(&self, image: &[f32], height: usize, width: usize) -> Result<Vec<f32>, String> {
        validate_shape(image, [3, height, width])?;
        let p = "student.dconv_encoder";
        let input = Map {
            data: image.to_vec(),
            shape: [3, height, width],
        };
        let mut cond = self.conv(&format!("{p}.patch_cond_embed"), &input, 16, 16, 0, 1, true)?;
        for i in 0..2 {
            cond = self.dico(cond, &format!("{p}.head_blocks.{i}"), None)?;
        }
        let cond = self.conv(&format!("{p}.proj_down"), &cond, 1, 1, 0, 1, true)?;
        let z = Map::zeros([128, height / 16, width / 16])?;
        let z = self.conv(&format!("{p}.z_proj"), &z, 1, 1, 0, 1, true)?;
        let mut data = cond.data;
        data.extend(z.data);
        let mut x = self.conv(
            &format!("{p}.fuse_proj"),
            &Map {
                data,
                shape: [768, height / 16, width / 16],
            },
            1,
            1,
            0,
            1,
            true,
        )?;
        let time = self.time(p)?;
        for i in 0..21 {
            x = self.dico(x, &format!("{p}.blocks.{i}"), Some(&time))?;
        }
        let x = self.norm(&x, Some(&format!("{p}.norm_out")))?;
        let mut x = self.conv(&format!("{p}.proj_out"), &x, 1, 1, 0, 1, true)?;
        x.data.truncate(128 * (height / 16) * (width / 16));
        trace("vae.mean", 1, x.data.len(), &x.data)?;
        Ok(x.data)
    }
    fn resnet(&self, input: Map, name: &str) -> Result<Map, String> {
        let mut x = self.group_norm(&input, &format!("{name}.norm1"))?;
        nonlinearity(&mut x.data);
        let x = self.conv(&format!("{name}.conv1"), &x, 3, 1, 1, 1, true)?;
        let mut x = self.group_norm(&x, &format!("{name}.norm2"))?;
        nonlinearity(&mut x.data);
        let mut x = self.conv(&format!("{name}.conv2"), &x, 3, 1, 1, 1, true)?;
        for (v, &r) in x.data.iter_mut().zip(&input.data) {
            *v += r;
        }
        Ok(x)
    }
    fn attention(&self, input: Map, name: &str) -> Result<Map, String> {
        let norm = self.group_norm(&input, &format!("{name}.norm"))?;
        let q = self.conv(&format!("{name}.q"), &norm, 1, 1, 0, 1, true)?;
        let k = self.conv(&format!("{name}.k"), &norm, 1, 1, 0, 1, true)?;
        let v = self.conv(&format!("{name}.v"), &norm, 1, 1, 0, 1, true)?;
        let [c, h, w] = input.shape;
        let d = 32;
        let n = d * d;
        let mut result = Map::zeros(input.shape)?;
        for by in (0..h).step_by(d) {
            for bx in (0..w).step_by(d) {
                let patch = |m: &Map| -> Vec<f32> {
                    (0..n * c)
                        .map(|i| {
                            let channel = i % c;
                            let pixel = i / c;
                            m.data[(channel * h + (by + pixel / d).min(h - 1)) * w
                                + (bx + pixel % d).min(w - 1)]
                        })
                        .collect()
                };
                let out = attention_scalar_f32(
                    &patch(&q),
                    &patch(&k),
                    &patch(&v),
                    AttentionSpec {
                        query_tokens: n,
                        key_tokens: n,
                        query_heads: 1,
                        key_value_heads: 1,
                        head_dim: c,
                        causal: false,
                        scale: (1.0 / (c as f64).sqrt()) as f32,
                    },
                )?;
                for y in 0..d.min(h - by) {
                    for x in 0..d.min(w - bx) {
                        for ch in 0..c {
                            result.data[(ch * h + by + y) * w + bx + x] = out[(y * d + x) * c + ch];
                        }
                    }
                }
            }
        }
        let mut result = self.conv(&format!("{name}.proj_out"), &result, 1, 1, 0, 1, true)?;
        for (v, &r) in result.data.iter_mut().zip(&input.data) {
            *v += r;
        }
        Ok(result)
    }
    pub fn decode(
        &self,
        latent: &[f32],
        latent_height: usize,
        latent_width: usize,
    ) -> Result<Vec<f32>, String> {
        validate_shape(latent, [128, latent_height, latent_width])?;
        let p = "pipeline.y_embedder.decoder";
        let z = Map {
            data: latent.to_vec(),
            shape: [128, latent_height, latent_width],
        };
        let mut cond = self.conv(&format!("{p}.conv_in"), &z, 3, 1, 1, 1, true)?;
        for i in 0..5 {
            let name = format!("{p}.block.{i}");
            cond = if i % 2 == 0 {
                self.resnet(cond, &name)?
            } else {
                self.attention(cond, &name)?
            };
        }
        let mut cond = self.group_norm(&cond, &format!("{p}.norm_out"))?;
        nonlinearity(&mut cond.data);
        let cond = self.conv(&format!("{p}.conv_out"), &cond, 3, 1, 1, 1, true)?;
        let p = "pipeline";
        let time = self.time(p)?;
        let [_, h, w] = cond.shape;
        let n = h * w;
        let noise = Map::zeros([3, h * 16, w * 16])?;
        let patch = self.conv(
            &format!("{p}.s_embedder.proj1"),
            &noise,
            16,
            16,
            0,
            1,
            false,
        )?;
        let mut data = patch.data;
        data.extend(&cond.data);
        let mut s = self.conv(
            &format!("{p}.s_embedder.proj2"),
            &Map {
                data,
                shape: [512, h, w],
            },
            1,
            1,
            0,
            1,
            true,
        )?;
        for i in 0..21 {
            s = self.dico(s, &format!("{p}.blocks.{i}"), Some(&time))?;
        }
        let conditioning = self.conv(&format!("{p}.y_embedder_x"), &cond, 1, 1, 0, 1, true)?;
        let dct = patch_dct();
        let mut x = vec![0.0; n * 256 * 99];
        for patch in 0..n {
            for pixel in 0..256 {
                let off = (patch * 256 + pixel) * 99;
                for ch in 0..32 {
                    x[off + 3 + ch] = conditioning.data[(ch * 256 + pixel) * n + patch];
                }
                x[off + 35..off + 99].copy_from_slice(&dct[pixel * 64..(pixel + 1) * 64]);
            }
        }
        let x = self.linear(&format!("{p}.x_embedder.embedder.0"), &x, n * 256, 99, 32)?;
        let mut x = self.linear(&format!("{p}.dec_net.input_proj"), &x, n * 256, 32, 32)?;
        let cond = self.linear(&format!("{p}.dec_net.cond_embed"), &s.rows(), n, 384, 8192)?;
        for i in 0..3 {
            let name = format!("{p}.dec_net.res_blocks.{i}");
            let mut c = cond.clone();
            silu_inplace(&mut c);
            let m = self.linear(&format!("{name}.adaLN_modulation.1"), &c, n * 256, 32, 96)?;
            let norm_weight = self.vector(&format!("{name}.in_ln.weight"), 32)?;
            let norm_bias = self.vector(&format!("{name}.in_ln.bias"), 32)?;
            let mut norm =
                layer_norm_rows(&x, n * 256, Some(&norm_weight), Some(&norm_bias), 1e-6)?;
            for r in 0..n * 256 {
                for j in 0..32 {
                    norm[r * 32 + j] =
                        norm[r * 32 + j] * (1.0 + m[r * 96 + 32 + j]) + m[r * 96 + j];
                }
            }
            let mut branch = self.linear(&format!("{name}.mlp.0"), &norm, n * 256, 32, 32)?;
            silu_inplace(&mut branch);
            let branch = self.linear(&format!("{name}.mlp.2"), &branch, n * 256, 32, 32)?;
            for r in 0..n * 256 {
                for j in 0..32 {
                    x[r * 32 + j] += m[r * 96 + 64 + j] * branch[r * 32 + j];
                }
            }
        }
        let x = rms_norm_rows(
            &x,
            n * 256,
            &self.vector(&format!("{p}.final_layer.norm.weight"), 32)?,
            1e-6,
        )?;
        let x = self.linear(&format!("{p}.final_layer.linear"), &x, n * 256, 32, 3)?;
        let mut image = vec![0.0; 3 * h * w * 256];
        for py in 0..h {
            for px in 0..w {
                for y in 0..16 {
                    for xx in 0..16 {
                        for ch in 0..3 {
                            image[(ch * h * 16 + py * 16 + y) * w * 16 + px * 16 + xx] =
                                x[(((py * w + px) * 256 + y * 16 + xx) * 3) + ch];
                        }
                    }
                }
            }
        }
        trace("vae.decoded", 1, image.len(), &image)?;
        Ok(image)
    }
}

fn modulate_map(input: &mut Map, shift: &[f32], scale: &[f32]) {
    let plane = input.shape[1] * input.shape[2];
    for (ch, row) in input.data.chunks_exact_mut(plane).enumerate() {
        for v in row {
            *v = *v * (1.0 + scale[ch]) + shift[ch];
        }
    }
}
// The official decoder spells this as x * sigmoid(x), with two F32 roundings.
fn nonlinearity(values: &mut [f32]) {
    for value in values {
        *value *= 1.0 / (1.0 + (-*value).exp());
    }
}
fn validate_shape(values: &[f32], shape: [usize; 3]) -> Result<(), String> {
    let limit = if shape[0] == 128 { 128 } else { 2048 };
    if shape[1] > limit
        || shape[2] > limit
        || values.len() != checked_len("MageVAE input", &shape)?
        || values.iter().any(|v| !v.is_finite())
    {
        return Err("Invalid MageVAE input".into());
    }
    if shape[0] == 3 && (shape[1] % 16 != 0 || shape[2] % 16 != 0) {
        return Err("MageVAE pixel sides must be multiples of 16".into());
    }
    Ok(())
}
fn patch_dct() -> Vec<f32> {
    let mut values = Vec::with_capacity(256 * 64);
    for y in 0..16 {
        for x in 0..16 {
            for fx in 0..8 {
                for fy in 0..8 {
                    let px = linspace(1.0, 16, x);
                    let py = linspace(1.0, 16, y);
                    let fx = linspace(8.0, 8, fx);
                    let fy = linspace(8.0, 8, fy);
                    let coeff = 1.0 / (1.0 + fx * fy);
                    values.push(
                        ((px * fx) * std::f32::consts::PI).cos()
                            * ((py * fy) * std::f32::consts::PI).cos()
                            * coeff,
                    );
                }
            }
        }
    }
    values
}

fn linspace(end: f32, n: usize, i: usize) -> f32 {
    let step = end / (n - 1) as f32;
    if i < n / 2 {
        step * i as f32
    } else {
        end - step * (n - 1 - i) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_attention_preserves_sequential_f32_score_reduction() {
        if !crate::ops::scalar_mode() {
            return;
        }
        let result = attention_scalar_f32(
            &[1e10, 1.0, -1e10],
            &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            AttentionSpec {
                query_tokens: 1,
                key_tokens: 2,
                query_heads: 1,
                key_value_heads: 1,
                head_dim: 3,
                causal: false,
                scale: 1.0,
            },
        )
        .unwrap();
        assert_eq!(
            result.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vec![0.5f32.to_bits(); 3]
        );
    }
}
