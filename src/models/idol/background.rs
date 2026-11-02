//! Native F32 BiRefNet/Swin-L foreground alpha, upstream Idol 4702a7f.
use super::common::{finite, norm, tensor, trace, Linear};
use crate::core::tensor::TensorSource;
use rayon::prelude::*;

const INPUT: usize = 1024;
const WINDOW: usize = 12;

/// Private feature maps are NHWC, with one image per inference.
struct Map {
    x: Vec<f32>,
    h: usize,
    w: usize,
    c: usize,
}
impl Map {
    fn checkpoint(&self, label: &str) -> Result<(), String> {
        finite(&self.x, label)?;
        trace(label, &[1, self.h, self.w, self.c], &self.x);
        Ok(())
    }
    fn resize(&self, h: usize, w: usize) -> Self {
        let mut x = vec![0.; h * w * self.c];
        x.par_chunks_mut(w * self.c)
            .enumerate()
            .for_each(|(y, row)| {
                let sy = if h > 1 {
                    (self.h - 1) as f32 / (h - 1) as f32 * y as f32
                } else {
                    0.
                };
                let y0 = sy.floor() as usize;
                let y1 = (y0 + 1).min(self.h - 1);
                let fy = sy - y0 as f32;
                for (px, pixel) in row.chunks_exact_mut(self.c).enumerate() {
                    let sx = if w > 1 {
                        (self.w - 1) as f32 / (w - 1) as f32 * px as f32
                    } else {
                        0.
                    };
                    let x0 = sx.floor() as usize;
                    let x1 = (x0 + 1).min(self.w - 1);
                    let fx = sx - x0 as f32;
                    for (c, value) in pixel.iter_mut().enumerate() {
                        let a = self.x[(y0 * self.w + x0) * self.c + c];
                        let b = self.x[(y0 * self.w + x1) * self.c + c];
                        let d = self.x[(y1 * self.w + x0) * self.c + c];
                        let e = self.x[(y1 * self.w + x1) * self.c + c];
                        *value =
                            (1. - fy) * ((1. - fx) * a + fx * b) + fy * ((1. - fx) * d + fx * e);
                    }
                }
            });
        Self { x, h, w, c: self.c }
    }
    fn add(&mut self, other: &Self) -> Result<(), String> {
        if (self.h, self.w, self.c) != (other.h, other.w, other.c) {
            return Err("BiRefNet residual shape mismatch".into());
        }
        for (a, &b) in self.x.iter_mut().zip(&other.x) {
            *a += b;
        }
        Ok(())
    }
}
fn concat(maps: &[&Map]) -> Result<Map, String> {
    let first = maps.first().ok_or("Empty BiRefNet concatenation")?;
    if maps.iter().any(|m| m.h != first.h || m.w != first.w) {
        return Err("BiRefNet concatenation shape mismatch".into());
    }
    let c = maps.iter().map(|m| m.c).sum();
    let mut x = vec![0.; first.h * first.w * c];
    x.par_chunks_mut(c).enumerate().for_each(|(pixel, row)| {
        let mut offset = 0;
        for map in maps {
            row[offset..offset + map.c].copy_from_slice(&map.x[pixel * map.c..(pixel + 1) * map.c]);
            offset += map.c;
        }
    });
    Ok(Map {
        x,
        h: first.h,
        w: first.w,
        c,
    })
}

struct Conv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
}
impl Conv {
    #[allow(clippy::too_many_arguments)]
    fn load(
        s: &dyn TensorSource,
        p: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Result<Self, String> {
        let raw = tensor(s, &format!("{p}.weight"), &[output, input, kernel, kernel])?;
        let mut weight = vec![0.; raw.len()];
        for oc in 0..output {
            for k in 0..kernel * kernel {
                for ic in 0..input {
                    weight[(oc * kernel * kernel + k) * input + ic] =
                        raw[(oc * input + ic) * kernel * kernel + k];
                }
            }
        }
        Ok(Self {
            weight,
            bias: if bias {
                tensor(s, &format!("{p}.bias"), &[output])?
            } else {
                vec![0.; output]
            },
            input,
            output,
            kernel,
            stride,
            padding,
        })
    }
    fn forward(&self, x: &Map) -> Result<Map, String> {
        if x.c != self.input
            || x.h + 2 * self.padding < self.kernel
            || x.w + 2 * self.padding < self.kernel
        {
            return Err("Invalid BiRefNet convolution shape".into());
        }
        let h = (x.h + 2 * self.padding - self.kernel) / self.stride + 1;
        let w = (x.w + 2 * self.padding - self.kernel) / self.stride + 1;
        let patch_width = self.input * self.kernel * self.kernel;
        let mut out = vec![0.; h * w * self.output];
        out.par_chunks_mut(self.output).enumerate().for_each_init(
            || vec![0.; patch_width],
            |patch, (pixel, row)| {
                let oy = pixel / w;
                let ox = pixel % w;
                for ky in 0..self.kernel {
                    for kx in 0..self.kernel {
                        let y = (oy * self.stride + ky) as isize - self.padding as isize;
                        let px = (ox * self.stride + kx) as isize - self.padding as isize;
                        let dst = &mut patch[(ky * self.kernel + kx) * self.input
                            ..(ky * self.kernel + kx + 1) * self.input];
                        if y >= 0 && px >= 0 && y < x.h as isize && px < x.w as isize {
                            let start = (y as usize * x.w + px as usize) * self.input;
                            dst.copy_from_slice(&x.x[start..start + self.input]);
                        } else {
                            dst.fill(0.);
                        }
                    }
                }
                for (oc, value) in row.iter_mut().enumerate() {
                    *value = crate::ops::dot_f32(
                        patch,
                        &self.weight[oc * patch_width..(oc + 1) * patch_width],
                        patch_width,
                    ) + self.bias[oc];
                }
            },
        );
        finite(&out, "background convolution")?;
        Ok(Map {
            x: out,
            h,
            w,
            c: self.output,
        })
    }
}
struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl LayerNorm {
    fn load(s: &dyn TensorSource, p: &str, width: usize) -> Result<Self, String> {
        Ok(Self {
            weight: tensor(s, &format!("{p}.weight"), &[width])?,
            bias: tensor(s, &format!("{p}.bias"), &[width])?,
        })
    }
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        norm(
            x,
            self.weight.len(),
            Some(&self.weight),
            Some(&self.bias),
            1e-5,
        )
    }
}
struct BatchNorm {
    scale: Vec<f32>,
    shift: Vec<f32>,
}
impl BatchNorm {
    fn load(s: &dyn TensorSource, p: &str, width: usize) -> Result<Self, String> {
        let weight = tensor(s, &format!("{p}.weight"), &[width])?;
        let bias = tensor(s, &format!("{p}.bias"), &[width])?;
        let mean = tensor(s, &format!("{p}.running_mean"), &[width])?;
        let variance = tensor(s, &format!("{p}.running_var"), &[width])?;
        if variance.iter().any(|&v| v < 0.) {
            return Err(format!("Negative BiRefNet BatchNorm variance at {p}"));
        }
        let scale: Vec<_> = weight
            .iter()
            .zip(variance)
            .map(|(&a, v)| a / (v + 1e-5).sqrt())
            .collect();
        let shift: Vec<_> = bias
            .iter()
            .zip(mean)
            .zip(&scale)
            .map(|((&b, m), &a)| b - m * a)
            .collect();
        finite(&scale, &format!("{p} BatchNorm scale"))?;
        finite(&shift, &format!("{p} BatchNorm shift"))?;
        Ok(Self { scale, shift })
    }
    fn apply(&self, x: &mut Map, relu: bool) {
        x.x.par_chunks_mut(x.c).for_each(|row| {
            for (c, value) in row.iter_mut().enumerate() {
                *value = *value * self.scale[c] + self.shift[c];
                if relu && *value < 0. {
                    *value = 0.;
                }
            }
        });
    }
}

struct SwinBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    proj: Linear,
    up: Linear,
    down: Linear,
    bias: Vec<f32>,
    width: usize,
    heads: usize,
    shift: usize,
}
fn window_region(y: usize, x: usize, h: usize, w: usize) -> usize {
    let segment = |p: usize, n: usize| {
        if p < n - WINDOW {
            0
        } else if p < n - WINDOW / 2 {
            1
        } else {
            2
        }
    };
    segment(y, h) * 3 + segment(x, w)
}
impl SwinBlock {
    fn load(
        s: &dyn TensorSource,
        p: &str,
        width: usize,
        heads: usize,
        shift: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            norm1: LayerNorm::load(s, &format!("{p}.norm1"), width)?,
            norm2: LayerNorm::load(s, &format!("{p}.norm2"), width)?,
            qkv: Linear::load(s, &format!("{p}.attn.qkv"), width, 3 * width, true)?,
            proj: Linear::load(s, &format!("{p}.attn.proj"), width, width, true)?,
            up: Linear::load(s, &format!("{p}.mlp.fc1"), width, 4 * width, true)?,
            down: Linear::load(s, &format!("{p}.mlp.fc2"), 4 * width, width, true)?,
            bias: tensor(
                s,
                &format!("{p}.attn.relative_position_bias_table"),
                &[(2 * WINDOW - 1).pow(2), heads],
            )?,
            width,
            heads,
            shift,
        })
    }
    fn forward(&self, mut map: Map, label: &str) -> Result<Map, String> {
        let h = map.h.div_ceil(WINDOW) * WINDOW;
        let w = map.w.div_ceil(WINDOW) * WINDOW;
        let nw = w / WINDOW;
        let n = WINDOW * WINDOW;
        let normalized = self.norm1.apply(&map.x);
        let mut windows = vec![0.; h * w * self.width];
        windows
            .par_chunks_mut(n * self.width)
            .enumerate()
            .for_each(|(window, out)| {
                for q in 0..n {
                    let y = (window / nw * WINDOW + q / WINDOW + self.shift) % h;
                    let x = (window % nw * WINDOW + q % WINDOW + self.shift) % w;
                    if y < map.h && x < map.w {
                        out[q * self.width..(q + 1) * self.width].copy_from_slice(
                            &normalized
                                [(y * map.w + x) * self.width..(y * map.w + x + 1) * self.width],
                        );
                    }
                }
            });
        let mut qkv = self.qkv.forward(&windows)?;
        let d = self.width / self.heads;
        let scale = (d as f32).sqrt().recip();
        qkv.par_chunks_mut(3 * self.width).for_each(|row| {
            for q in &mut row[..self.width] {
                *q *= scale;
            }
        });
        windows
            .par_chunks_mut(self.width)
            .enumerate()
            .for_each_init(
                || vec![0.; n],
                |scores, (query, out)| {
                    let window = query / n;
                    let q = query % n;
                    let y = window / nw * WINDOW;
                    let x = window % nw * WINDOW;
                    for head in 0..self.heads {
                        let start = query * 3 * self.width + head * d;
                        let qv = &qkv[start..start + d];
                        for (k, score) in scores.iter_mut().enumerate() {
                            let start = (window * n + k) * 3 * self.width + self.width + head * d;
                            let ry = (q / WINDOW) as isize - (k / WINDOW) as isize
                                + (WINDOW - 1) as isize;
                            let rx = (q % WINDOW) as isize - (k % WINDOW) as isize
                                + (WINDOW - 1) as isize;
                            let relative = ry as usize * (2 * WINDOW - 1) + rx as usize;
                            *score = crate::ops::dot_f32(qv, &qkv[start..start + d], d)
                                + self.bias[relative * self.heads + head];
                            if self.shift != 0
                                && window_region(y + q / WINDOW, x + q % WINDOW, h, w)
                                    != window_region(y + k / WINDOW, x + k % WINDOW, h, w)
                            {
                                *score -= 100.;
                            }
                        }
                        crate::ops::softmax_inplace(scores);
                        crate::ops::attention_value_reduce(
                            &qkv,
                            scores,
                            &mut out[head * d..(head + 1) * d],
                            window * n * 3 * self.width,
                            3 * self.width,
                            2 * self.width + head * d,
                            n,
                            d,
                        );
                    }
                },
            );
        let projected = self.proj.forward(&windows)?;
        for y in 0..map.h {
            for x in 0..map.w {
                let sy = (y + h - self.shift) % h;
                let sx = (x + w - self.shift) % w;
                let window = sy / WINDOW * nw + sx / WINDOW;
                let token = sy % WINDOW * WINDOW + sx % WINDOW;
                let source = (window * n + token) * self.width;
                let destination = (y * map.w + x) * self.width;
                for c in 0..self.width {
                    map.x[destination + c] += projected[source + c];
                }
            }
        }
        map.checkpoint(&format!("{label}.attention"))?;
        let mut ff = self.up.forward(&self.norm2.apply(&map.x))?;
        crate::ops::gelu_erf_inplace(&mut ff);
        let ff = self.down.forward(&ff)?;
        for (value, &f) in map.x.iter_mut().zip(&ff) {
            *value += f;
        }
        map.checkpoint(label)?;
        Ok(map)
    }
}
struct SwinStage {
    blocks: Vec<SwinBlock>,
    norm: LayerNorm,
    merge: Option<(LayerNorm, Linear)>,
}
struct Swin {
    patch: Conv,
    norm: LayerNorm,
    stages: Vec<SwinStage>,
}
impl Swin {
    fn load(s: &dyn TensorSource) -> Result<Self, String> {
        let mut stages = Vec::new();
        for (i, (&depth, &heads)) in [2, 2, 18, 2].iter().zip(&[6, 12, 24, 48]).enumerate() {
            let c = 192 << i;
            let p = format!("bb.layers.{i}");
            let mut blocks = Vec::new();
            for b in 0..depth {
                blocks.push(SwinBlock::load(
                    s,
                    &format!("{p}.blocks.{b}"),
                    c,
                    heads,
                    if b % 2 == 0 { 0 } else { WINDOW / 2 },
                )?);
            }
            stages.push(SwinStage {
                blocks,
                norm: LayerNorm::load(s, &format!("bb.norm{i}"), c)?,
                merge: if i == 3 {
                    None
                } else {
                    Some((
                        LayerNorm::load(s, &format!("{p}.downsample.norm"), 4 * c)?,
                        Linear::load(s, &format!("{p}.downsample.reduction"), 4 * c, 2 * c, false)?,
                    ))
                },
            });
        }
        Ok(Self {
            patch: Conv::load(s, "bb.patch_embed.proj", 3, 192, 4, 4, 0, true)?,
            norm: LayerNorm::load(s, "bb.patch_embed.norm", 192)?,
            stages,
        })
    }
    fn forward(&self, input: &Map, label: &str) -> Result<Vec<Map>, String> {
        let mut x = self.patch.forward(input)?;
        x.x = self.norm.apply(&x.x);
        x.checkpoint(&format!("{label}.patch"))?;
        let mut outputs = Vec::new();
        for (i, stage) in self.stages.iter().enumerate() {
            for (b, block) in stage.blocks.iter().enumerate() {
                x = block.forward(x, &format!("{label}.stage.{i}.block.{b}"))?;
            }
            let output = Map {
                x: stage.norm.apply(&x.x),
                h: x.h,
                w: x.w,
                c: x.c,
            };
            output.checkpoint(&format!("{label}.stage.{i}.out"))?;
            outputs.push(output);
            if let Some((norm, reduction)) = &stage.merge {
                let merged = merge_patches(&x)?;
                x = Map {
                    x: reduction.forward(&norm.apply(&merged.x))?,
                    h: merged.h,
                    w: merged.w,
                    c: 2 * x.c,
                };
                x.checkpoint(&format!("{label}.stage.{i}.merge"))?;
            }
        }
        Ok(outputs)
    }
}
fn merge_patches(x: &Map) -> Result<Map, String> {
    if x.h % 2 != 0 || x.w % 2 != 0 {
        return Err("BiRefNet patch merging requires even dimensions".into());
    }
    let h = x.h / 2;
    let w = x.w / 2;
    let c = 4 * x.c;
    let mut output = vec![0.; h * w * c];
    for y in 0..h {
        for px in 0..w {
            for (index, (dy, dx)) in [(0, 0), (1, 0), (0, 1), (1, 1)].iter().enumerate() {
                let source = ((2 * y + dy) * x.w + 2 * px + dx) * x.c;
                let target = (y * w + px) * c + index * x.c;
                output[target..target + x.c].copy_from_slice(&x.x[source..source + x.c]);
            }
        }
    }
    Ok(Map { x: output, h, w, c })
}

fn sample_zero(x: &Map, y: f32, px: f32, c: usize) -> f32 {
    if y <= -1. || y >= x.h as f32 || px <= -1. || px >= x.w as f32 {
        return 0.;
    }
    let y0 = y.floor() as isize;
    let x0 = px.floor() as isize;
    let fy = y - y0 as f32;
    let fx = px - x0 as f32;
    let at = |yy: isize, xx: isize| {
        if yy >= 0 && xx >= 0 && yy < x.h as isize && xx < x.w as isize {
            x.x[(yy as usize * x.w + xx as usize) * x.c + c]
        } else {
            0.
        }
    };
    (1. - fy) * (1. - fx) * at(y0, x0)
        + (1. - fy) * fx * at(y0, x0 + 1)
        + fy * (1. - fx) * at(y0 + 1, x0)
        + fy * fx * at(y0 + 1, x0 + 1)
}
struct Deform {
    offset: Conv,
    mask: Conv,
    regular: Conv,
}
impl Deform {
    fn load(
        s: &dyn TensorSource,
        p: &str,
        input: usize,
        output: usize,
        kernel: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            offset: Conv::load(
                s,
                &format!("{p}.offset_conv"),
                input,
                2 * kernel * kernel,
                kernel,
                1,
                kernel / 2,
                true,
            )?,
            mask: Conv::load(
                s,
                &format!("{p}.modulator_conv"),
                input,
                kernel * kernel,
                kernel,
                1,
                kernel / 2,
                true,
            )?,
            regular: Conv::load(
                s,
                &format!("{p}.regular_conv"),
                input,
                output,
                kernel,
                1,
                kernel / 2,
                false,
            )?,
        })
    }
    fn forward(&self, x: &Map) -> Result<Map, String> {
        let offset = self.offset.forward(x)?;
        let mask = self.mask.forward(x)?;
        let k = self.regular.kernel;
        let patch_width = x.c * k * k;
        let mut output = vec![0.; x.h * x.w * self.regular.output];
        output
            .par_chunks_mut(self.regular.output)
            .enumerate()
            .for_each_init(
                || vec![0.; patch_width],
                |patch, (pixel, out)| {
                    for ky in 0..k {
                        for kx in 0..k {
                            let j = ky * k + kx;
                            let y = (pixel / x.w + ky) as f32 - self.regular.padding as f32
                                + offset.x[pixel * offset.c + 2 * j];
                            let px = (pixel % x.w + kx) as f32 - self.regular.padding as f32
                                + offset.x[pixel * offset.c + 2 * j + 1];
                            let multiplier = 2. / (1. + (-mask.x[pixel * mask.c + j]).exp());
                            for c in 0..x.c {
                                patch[j * x.c + c] = sample_zero(x, y, px, c) * multiplier;
                            }
                        }
                    }
                    for (c, value) in out.iter_mut().enumerate() {
                        *value = crate::ops::dot_f32(
                            patch,
                            &self.regular.weight[c * patch_width..(c + 1) * patch_width],
                            patch_width,
                        );
                    }
                },
            );
        finite(&output, "background deformable convolution")?;
        Ok(Map {
            x: output,
            h: x.h,
            w: x.w,
            c: self.regular.output,
        })
    }
}
struct DeformBranch {
    conv: Deform,
    norm: BatchNorm,
}
impl DeformBranch {
    fn load(s: &dyn TensorSource, p: &str, kernel: usize) -> Result<Self, String> {
        Ok(Self {
            conv: Deform::load(s, &format!("{p}.atrous_conv"), 64, 256, kernel)?,
            norm: BatchNorm::load(s, &format!("{p}.bn"), 256)?,
        })
    }
    fn forward(&self, x: &Map) -> Result<Map, String> {
        let mut x = self.conv.forward(x)?;
        self.norm.apply(&mut x, true);
        Ok(x)
    }
}
struct Aspp {
    branches: Vec<DeformBranch>,
    pool: Conv,
    pool_norm: BatchNorm,
    out: Conv,
    out_norm: BatchNorm,
}
impl Aspp {
    fn load(s: &dyn TensorSource, p: &str) -> Result<Self, String> {
        let mut branches = vec![DeformBranch::load(s, &format!("{p}.aspp1"), 1)?];
        for (i, k) in [1, 3, 7].into_iter().enumerate() {
            branches.push(DeformBranch::load(s, &format!("{p}.aspp_deforms.{i}"), k)?);
        }
        Ok(Self {
            branches,
            pool: Conv::load(
                s,
                &format!("{p}.global_avg_pool.1"),
                64,
                256,
                1,
                1,
                0,
                false,
            )?,
            pool_norm: BatchNorm::load(s, &format!("{p}.global_avg_pool.2"), 256)?,
            out: Conv::load(s, &format!("{p}.conv1"), 5 * 256, 64, 1, 1, 0, false)?,
            out_norm: BatchNorm::load(s, &format!("{p}.bn1"), 64)?,
        })
    }
    fn forward(&self, x: &Map, label: &str) -> Result<Map, String> {
        let mut outputs = Vec::new();
        for (i, branch) in self.branches.iter().enumerate() {
            let output = branch.forward(x)?;
            output.checkpoint(&format!("{label}.branch.{i}"))?;
            outputs.push(output);
        }
        let mut pooled = Map {
            x: vec![0.; x.c],
            h: 1,
            w: 1,
            c: x.c,
        };
        for row in x.x.chunks_exact(x.c) {
            for (out, &value) in pooled.x.iter_mut().zip(row) {
                *out += value;
            }
        }
        for value in &mut pooled.x {
            *value /= (x.h * x.w) as f32;
        }
        let mut pooled = self.pool.forward(&pooled)?;
        self.pool_norm.apply(&mut pooled, true);
        outputs.push(pooled.resize(x.h, x.w));
        let mut output = self
            .out
            .forward(&concat(&outputs.iter().collect::<Vec<_>>())?)?;
        self.out_norm.apply(&mut output, true);
        output.checkpoint(label)?;
        Ok(output)
    }
}
struct DecBlock {
    input: Conv,
    norm_in: BatchNorm,
    aspp: Aspp,
    out: Conv,
    norm_out: BatchNorm,
}
impl DecBlock {
    fn load(s: &dyn TensorSource, p: &str, input: usize, output: usize) -> Result<Self, String> {
        Ok(Self {
            input: Conv::load(s, &format!("{p}.conv_in"), input, 64, 3, 1, 1, true)?,
            norm_in: BatchNorm::load(s, &format!("{p}.bn_in"), 64)?,
            aspp: Aspp::load(s, &format!("{p}.dec_att"))?,
            out: Conv::load(s, &format!("{p}.conv_out"), 64, output, 3, 1, 1, true)?,
            norm_out: BatchNorm::load(s, &format!("{p}.bn_out"), output)?,
        })
    }
    fn forward(&self, x: &Map, label: &str) -> Result<Map, String> {
        let mut h = self.input.forward(x)?;
        self.norm_in.apply(&mut h, true);
        h.checkpoint(&format!("{label}.input"))?;
        let h = self.aspp.forward(&h, &format!("{label}.aspp"))?;
        let mut h = self.out.forward(&h)?;
        self.norm_out.apply(&mut h, false);
        h.checkpoint(label)?;
        Ok(h)
    }
}
struct Simple {
    input: Conv,
    out: Conv,
}
impl Simple {
    fn load(s: &dyn TensorSource, p: &str, input: usize, output: usize) -> Result<Self, String> {
        Ok(Self {
            input: Conv::load(s, &format!("{p}.conv1"), input, 64, 3, 1, 1, true)?,
            out: Conv::load(s, &format!("{p}.conv_out"), 64, output, 3, 1, 1, true)?,
        })
    }
    fn forward(&self, x: &Map) -> Result<Map, String> {
        self.out.forward(&self.input.forward(x)?)
    }
}
fn image_patches(image: &Map, h: usize, w: usize) -> Result<Map, String> {
    if h == 0 || w == 0 || image.h % h != 0 || image.w % w != 0 {
        return Err("Invalid BiRefNet image-patch shape".into());
    }
    let hg = image.h / h;
    let wg = image.w / w;
    let c = image.c * hg * wg;
    let mut x = vec![0.; h * w * c];
    x.par_chunks_mut(c).enumerate().for_each(|(pixel, row)| {
        for channel in 0..image.c {
            for gy in 0..hg {
                for gx in 0..wg {
                    row[(channel * hg + gy) * wg + gx] = image.x
                        [((gy * h + pixel / w) * image.w + gx * w + pixel % w) * image.c + channel];
                }
            }
        }
    });
    Ok(Map { x, h, w, c })
}
struct Gate {
    conv: Conv,
    norm: BatchNorm,
    out: Conv,
}
impl Gate {
    fn load(s: &dyn TensorSource, index: usize, input: usize) -> Result<Self, String> {
        Ok(Self {
            conv: Conv::load(
                s,
                &format!("decoder.gdt_convs_{index}.0"),
                input,
                16,
                3,
                1,
                1,
                true,
            )?,
            norm: BatchNorm::load(s, &format!("decoder.gdt_convs_{index}.1"), 16)?,
            out: Conv::load(
                s,
                &format!("decoder.gdt_convs_attn_{index}.0"),
                16,
                1,
                1,
                1,
                0,
                true,
            )?,
        })
    }
    fn apply(&self, x: &mut Map) -> Result<(), String> {
        let mut h = self.conv.forward(x)?;
        self.norm.apply(&mut h, true);
        let h = self.out.forward(&h)?;
        for (row, &gate) in x.x.chunks_exact_mut(x.c).zip(&h.x) {
            let gate = 1. / (1. + (-gate).exp());
            for value in row {
                *value *= gate;
            }
        }
        Ok(())
    }
}
pub(super) struct Background {
    swin: Swin,
    squeeze: DecBlock,
    ipt: Vec<Simple>,
    blocks: Vec<DecBlock>,
    lateral: Vec<Conv>,
    gates: Vec<Gate>,
    out: Conv,
}
impl Background {
    pub(super) fn load(s: &dyn TensorSource) -> Result<Self, String> {
        let mut ipt = Vec::new();
        for ((&index, &input), &output) in [5, 4, 3, 2, 1]
            .iter()
            .zip(&[3072, 768, 192, 48, 3])
            .zip(&[384, 384, 192, 96, 48])
        {
            ipt.push(Simple::load(
                s,
                &format!("decoder.ipt_blk{index}"),
                input,
                output,
            )?);
        }
        let mut blocks = Vec::new();
        let mut lateral = Vec::new();
        let mut gates = Vec::new();
        for (i, (&input, &output)) in [3456, 1920, 960, 480]
            .iter()
            .zip(&[1536, 768, 384, 192])
            .enumerate()
        {
            blocks.push(DecBlock::load(
                s,
                &format!("decoder.decoder_block{}", 4 - i),
                input,
                output,
            )?);
            if i < 3 {
                lateral.push(Conv::load(
                    s,
                    &format!("decoder.lateral_block{}.conv", 4 - i),
                    output,
                    output,
                    1,
                    1,
                    0,
                    true,
                )?);
                gates.push(Gate::load(s, 4 - i, output)?);
            }
        }
        Ok(Self {
            swin: Swin::load(s)?,
            squeeze: DecBlock::load(s, "squeeze_module.0", 5760, 3072)?,
            ipt,
            blocks,
            lateral,
            gates,
            out: Conv::load(s, "decoder.conv_out1.0", 240, 1, 1, 1, 0, true)?,
        })
    }
    pub(super) fn alpha(&self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>, String> {
        let pixels = width
            .checked_mul(height)
            .ok_or("BiRefNet image dimensions overflow")?;
        if width == 0 || height == 0 || pixels.checked_mul(3) != Some(rgb.len()) {
            return Err("BiRefNet expects nonempty RGB pixels matching width and height".into());
        }
        let image = Map {
            x: rgb.iter().map(|&v| v as f32 / 255.).collect(),
            h: height,
            w: width,
            c: 3,
        };
        let mut image = image.resize(INPUT, INPUT);
        for row in image.x.chunks_exact_mut(3) {
            for (c, value) in row.iter_mut().enumerate() {
                *value = (*value - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c];
            }
        }
        let alpha = self.forward(&image)?;
        let alpha = alpha.resize(height, width);
        finite(&alpha.x, "background alpha")?;
        Ok(alpha
            .x
            .iter()
            .map(|&v| (v.clamp(0., 1.) * 255.) as u8)
            .collect())
    }
    /// Diagnostic normalized NHWC input; production alpha always uses 1024².
    pub(super) fn forward_fixture(
        &self,
        normalized: &[f32],
        side: usize,
    ) -> Result<Vec<f32>, String> {
        if !(64..=INPUT).contains(&side) || side % 64 != 0 || normalized.len() != side * side * 3 {
            return Err("BiRefNet fixture requires normalized square NHWC RGB, side 64..1024 divisible by 64".into());
        }
        finite(normalized, "background fixture input")?;
        Ok(self
            .forward(&Map {
                x: normalized.to_vec(),
                h: side,
                w: side,
                c: 3,
            })?
            .x)
    }
    fn forward(&self, image: &Map) -> Result<Map, String> {
        image.checkpoint("idol.background.input")?;
        let full = self.swin.forward(image, "idol.background.full")?;
        let half = self.swin.forward(
            &image.resize(image.h / 2, image.w / 2),
            "idol.background.half",
        )?;
        let mut features = Vec::new();
        for (i, (a, b)) in full.iter().zip(&half).enumerate() {
            let feature = concat(&[a, &b.resize(a.h, a.w)])?;
            feature.checkpoint(&format!("idol.background.multiscale.{i}"))?;
            features.push(feature);
        }
        let last = &features[3];
        let context = concat(&[
            &features[0].resize(last.h, last.w),
            &features[1].resize(last.h, last.w),
            &features[2].resize(last.h, last.w),
            last,
        ])?;
        let mut current = self.squeeze.forward(&context, "idol.background.squeeze")?;
        for i in 0..4 {
            let patches = self.ipt[i].forward(&image_patches(image, current.h, current.w)?)?;
            patches.checkpoint(&format!("idol.background.image_patch.{i}"))?;
            current = self.blocks[i].forward(
                &concat(&[&current, &patches])?,
                &format!("idol.background.decoder.{i}"),
            )?;
            if i < 3 {
                self.gates[i].apply(&mut current)?;
                current.checkpoint(&format!("idol.background.gated.{i}"))?;
                let feature = &features[2 - i];
                current = current.resize(feature.h, feature.w);
                current.add(&self.lateral[i].forward(feature)?)?;
            }
        }
        current = current.resize(image.h, image.w);
        let patches = self.ipt[4].forward(&image_patches(image, image.h, image.w)?)?;
        let mut output = self.out.forward(&concat(&[&current, &patches])?)?;
        output.checkpoint("idol.background.logits")?;
        for value in &mut output.x {
            *value = 1. / (1. + (-*value).exp());
        }
        output.checkpoint("idol.background.alpha")?;
        Ok(output)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn align_corners_resize_preserves_endpoints_and_handles_singletons() {
        let x = Map {
            x: vec![0., 2., 4., 6.],
            h: 2,
            w: 2,
            c: 1,
        };
        assert_eq!(x.resize(3, 3).x, [0., 1., 2., 2., 3., 4., 4., 5., 6.]);
        assert_eq!(x.resize(1, 1).x, [0.]);
        let singleton = Map {
            x: vec![7.],
            h: 1,
            w: 1,
            c: 1,
        };
        assert_eq!(singleton.resize(2, 3).x, [7.; 6]);
    }
    #[test]
    fn patch_merge_and_image_stack_have_distinct_torch_orders() {
        let x = Map {
            x: (0..16).map(|v| v as f32).collect(),
            h: 4,
            w: 4,
            c: 1,
        };
        assert_eq!(&merge_patches(&x).unwrap().x[..4], &[0., 4., 1., 5.]);
        assert_eq!(&image_patches(&x, 2, 2).unwrap().x[..4], &[0., 2., 8., 10.]);
        assert!(image_patches(&x, 3, 3).is_err());
    }
    #[test]
    fn deform_sampling_blends_zero_padding_instead_of_clamping() {
        let x = Map {
            x: vec![2., 4., 6., 8.],
            h: 2,
            w: 2,
            c: 1,
        };
        assert_eq!(sample_zero(&x, 0.5, 0.5, 0), 5.);
        assert_eq!(sample_zero(&x, -0.5, 0., 0), 1.);
        assert_eq!(sample_zero(&x, -1., 0., 0), 0.);
        assert_eq!(sample_zero(&x, 2., 0., 0), 0.);
    }
    #[test]
    fn deform_applies_y_x_offsets_and_twice_sigmoid_mask() {
        let x = Map {
            x: vec![2., 4., 6., 8.],
            h: 2,
            w: 2,
            c: 1,
        };
        let deform = Deform {
            offset: Conv {
                weight: vec![0.; 2],
                bias: vec![0.5, 0.],
                input: 1,
                output: 2,
                kernel: 1,
                stride: 1,
                padding: 0,
            },
            mask: Conv {
                weight: vec![0.],
                bias: vec![0.],
                input: 1,
                output: 1,
                kernel: 1,
                stride: 1,
                padding: 0,
            },
            regular: Conv {
                weight: vec![2.],
                bias: vec![0.],
                input: 1,
                output: 1,
                kernel: 1,
                stride: 1,
                padding: 0,
            },
        };
        assert_eq!(deform.forward(&x).unwrap().x, [8., 12., 6., 8.]);
    }
    #[test]
    fn convolution_preserves_channel_kernel_layout_and_bias() {
        let x = Map {
            x: vec![1., 10., 2., 20., 3., 30., 4., 40.],
            h: 2,
            w: 2,
            c: 2,
        };
        let conv = Conv {
            weight: vec![1., 2., 3., 4., 5., 6., 7., 8.],
            bias: vec![5.],
            input: 2,
            output: 1,
            kernel: 2,
            stride: 2,
            padding: 0,
        };
        assert_eq!(conv.forward(&x).unwrap().x, [655.]);
    }
    #[test]
    fn shifted_window_mask_separates_wraparound_regions() {
        assert_eq!(window_region(0, 0, 24, 24), window_region(11, 11, 24, 24));
        assert_ne!(window_region(12, 12, 24, 24), window_region(18, 12, 24, 24));
        assert_ne!(window_region(12, 12, 24, 24), window_region(12, 18, 24, 24));
    }
    #[test]
    fn relu_preserves_nan_for_finite_validation() {
        let mut x = Map {
            x: vec![f32::NAN],
            h: 1,
            w: 1,
            c: 1,
        };
        BatchNorm {
            scale: vec![1.],
            shift: vec![0.],
        }
        .apply(&mut x, true);
        assert!(x.x[0].is_nan());
        assert!(x.checkpoint("fixture").is_err());
    }
}
