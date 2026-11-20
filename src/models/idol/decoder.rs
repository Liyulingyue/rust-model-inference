//! Idol octree and Gaussian decoders; tensors use token-major, natural PyTorch order.
use super::common::{finite, norm, round_f16_checked, tensor, trace, Linear};
use crate::core::tensor::TensorSource;
use rand::Rng;
use rayon::prelude::*;

const MAX_ANCHORS: usize = 32768;
const MAX_CONTEXT: usize = 8192;
const LN_EPS: f32 = 1e-6;

fn half(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}
fn half_linear(linear: &Linear, x: &[f32]) -> Result<Vec<f32>, String> {
    let mut y = linear.forward(x)?;
    round_f16_checked(&mut y, "FP16 decoder linear output")?;
    Ok(y)
}
fn load_half(s: &dyn TensorSource, p: &str, input: usize, output: usize) -> Result<Linear, String> {
    let mut linear = Linear::load(s, p, input, output, true)?;
    linear.to_f16()?;
    Ok(linear)
}
fn normalized(
    x: &[f32],
    width: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    let mut y = norm(x, width, weight, bias, LN_EPS);
    round_f16_checked(&mut y, "FP16 decoder normalization")?;
    Ok(y)
}
fn add_half(x: &mut [f32], y: &[f32], gate: Option<&[f32]>) {
    for (i, (a, &b)) in x.iter_mut().zip(y).enumerate() {
        *a = half(*a + gate.map_or(b, |g| half(b * g[i % g.len()])));
    }
}
fn modulate(x: &mut [f32], shift: &[f32], scale: &[f32]) {
    for (i, v) in x.iter_mut().enumerate() {
        let j = i % shift.len();
        *v = half(half(*v * half(1.0 + scale[j])) + shift[j]);
    }
}

struct Attention {
    q: Linear,
    kv: Option<Linear>,
    out: Linear,
    q_gamma: Vec<f32>,
    k_gamma: Vec<f32>,
    channels: usize,
    heads: usize,
}
impl Attention {
    fn load(
        s: &dyn TensorSource,
        p: &str,
        channels: usize,
        heads: usize,
        context: Option<usize>,
    ) -> Result<Self, String> {
        let hd = channels / heads;
        Ok(Self {
            q: load_half(
                s,
                &format!("{p}.{}", if context.is_some() { "to_q" } else { "to_qkv" }),
                channels,
                channels * if context.is_some() { 1 } else { 3 },
            )?,
            kv: context
                .map(|c| load_half(s, &format!("{p}.to_kv"), c, channels * 2))
                .transpose()?,
            out: load_half(s, &format!("{p}.to_out"), channels, channels)?,
            q_gamma: tensor(s, &format!("{p}.q_rms_norm.gamma"), &[heads, hd])?,
            k_gamma: tensor(s, &format!("{p}.k_rms_norm.gamma"), &[heads, hd])?,
            channels,
            heads,
        })
    }
    fn rms(&self, x: &mut [f32], gamma: &[f32]) {
        let hd = self.channels / self.heads;
        x.par_chunks_mut(self.channels).for_each(|row| {
            for (h, head) in row.chunks_mut(hd).enumerate() {
                let inv = crate::ops::dot_f32(head, head, hd)
                    .sqrt()
                    .max(1e-12)
                    .recip();
                for (i, v) in head.iter_mut().enumerate() {
                    *v = half((*v * inv) * gamma[h * hd + i] * (hd as f32).sqrt());
                }
            }
        });
    }
    fn forward(&self, x: &[f32], context: Option<&[f32]>) -> Result<Vec<f32>, String> {
        let c = self.channels;
        let hd = c / self.heads;
        let projected = half_linear(&self.q, x)?;
        let (mut q, mut k, v) = if let Some(kv) = &self.kv {
            let ctx = context.ok_or("Idol cross attention requires context")?;
            let projected_kv = half_linear(kv, ctx)?;
            let mut k = Vec::with_capacity(projected_kv.len() / 2);
            let mut v = Vec::with_capacity(projected_kv.len() / 2);
            for row in projected_kv.chunks_exact(2 * c) {
                k.extend_from_slice(&row[..c]);
                v.extend_from_slice(&row[c..]);
            }
            (projected, k, v)
        } else {
            let mut q = Vec::with_capacity(x.len());
            let mut k = Vec::with_capacity(x.len());
            let mut v = Vec::with_capacity(x.len());
            for row in projected.chunks_exact(3 * c) {
                q.extend_from_slice(&row[..c]);
                k.extend_from_slice(&row[c..2 * c]);
                v.extend_from_slice(&row[2 * c..]);
            }
            (q, k, v)
        };
        self.rms(&mut q, &self.q_gamma);
        self.rms(&mut k, &self.k_gamma);
        finite(&q, "decoder attention queries")?;
        finite(&k, "decoder attention keys")?;
        if let Some(mut attended) = super::gpu::attention(&q, &k, &v, c, self.heads) {
            round_f16_checked(&mut attended, "FP16 decoder attention")?;
            return half_linear(&self.out, &attended);
        }
        super::gpu::check_failure()?;
        let source_rows = k.len() / c;
        let scale = (hd as f32).sqrt().recip();
        let mut attended = vec![0.; q.len()];
        // One score row per worker keeps memory O(tokens * channels), never O(tokens²).
        attended
            .par_chunks_mut(c)
            .zip(q.par_chunks(c))
            .for_each_init(
                || vec![0.; source_rows],
                |scores, (out, row)| {
                    for head in 0..self.heads {
                        let offset = head * hd;
                        for (j, score) in scores.iter_mut().enumerate() {
                            *score = crate::ops::dot_f32(
                                &row[offset..offset + hd],
                                &k[j * c + offset..j * c + offset + hd],
                                hd,
                            ) * scale;
                        }
                        crate::ops::softmax_inplace(scores);
                        crate::ops::attention_value_reduce(
                            &v,
                            scores,
                            &mut out[offset..offset + hd],
                            0,
                            c,
                            offset,
                            source_rows,
                            hd,
                        );
                    }
                },
            );
        round_f16_checked(&mut attended, "FP16 decoder attention")?;
        half_linear(&self.out, &attended)
    }
}

struct FeedForward {
    up: Linear,
    down: Linear,
}
impl FeedForward {
    fn load(s: &dyn TensorSource, p: &str, c: usize, hidden: usize) -> Result<Self, String> {
        Ok(Self {
            up: load_half(s, &format!("{p}.mlp.0"), c, hidden)?,
            down: load_half(s, &format!("{p}.mlp.2"), hidden, c)?,
        })
    }
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        let mut h = half_linear(&self.up, x)?;
        crate::ops::gelu_inplace(&mut h);
        round_f16_checked(&mut h, "FP16 decoder feed-forward")?;
        half_linear(&self.down, &h)
    }
}

struct Block {
    self_attention: Option<Attention>,
    cross: Attention,
    norm_weight: Option<Vec<f32>>,
    norm_bias: Option<Vec<f32>>,
    ffn: FeedForward,
}
impl Block {
    fn load(
        s: &dyn TensorSource,
        p: &str,
        c: usize,
        heads: usize,
        ctx: usize,
        hidden: usize,
        gaussian: bool,
    ) -> Result<Self, String> {
        Ok(Self {
            self_attention: gaussian
                .then(|| Attention::load(s, &format!("{p}.self_attn"), c, heads, None))
                .transpose()?,
            cross: Attention::load(s, &format!("{p}.cross_attn"), c, heads, Some(ctx))?,
            norm_weight: gaussian
                .then(|| tensor(s, &format!("{p}.norm2.weight"), &[c]))
                .transpose()?,
            norm_bias: gaussian
                .then(|| tensor(s, &format!("{p}.norm2.bias"), &[c]))
                .transpose()?,
            ffn: FeedForward::load(s, &format!("{p}.mlp"), c, hidden)?,
        })
    }
    fn forward(
        &self,
        x: &mut [f32],
        ctx: &[f32],
        modulation: Option<&[f32]>,
    ) -> Result<(), String> {
        let c = self.cross.channels;
        if let Some(attention) = &self.self_attention {
            let h = attention.forward(&normalized(x, c, None, None)?, None)?;
            add_half(x, &h, None);
        }
        let mut h = normalized(x, c, self.norm_weight.as_deref(), self.norm_bias.as_deref())?;
        if let Some(m) = modulation {
            modulate(&mut h, &m[..c], &m[c..2 * c]);
        }
        h = self.cross.forward(&h, Some(ctx))?;
        add_half(x, &h, modulation.map(|m| &m[2 * c..3 * c]));
        let mut h = normalized(x, c, None, None)?;
        if let Some(m) = modulation {
            modulate(&mut h, &m[3 * c..4 * c], &m[4 * c..5 * c]);
        }
        h = self.ffn.forward(&h)?;
        add_half(x, &h, modulation.map(|m| &m[5 * c..6 * c]));
        finite(x, "decoder block output")
    }
}

struct QueryDecoder {
    input: Linear,
    input_layer: Linear,
    output: Linear,
    blocks: Vec<Block>,
    channels: usize,
    context: usize,
}
impl QueryDecoder {
    fn load(s: &dyn TensorSource, gaussian: bool, output: usize) -> Result<Self, String> {
        let c = 1024;
        let heads = 16;
        let context = 16;
        let hidden = 4096;
        let count = if gaussian { 16 } else { 4 };
        let blocks = (0..count)
            .map(|i| {
                Block::load(
                    s,
                    &format!("blocks.{i}"),
                    c,
                    heads,
                    context,
                    hidden,
                    gaussian,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if s.tensor_info(&format!("blocks.{count}.cross_attn.to_q.weight"))
            .is_some()
        {
            return Err("Unsupported Idol decoder block count".into());
        }
        Ok(Self {
            input: Linear::load(s, "in_proj", 3, c, true)?,
            input_layer: Linear::load(s, "input_layer", c, c, true)?,
            output: Linear::load(s, "out_proj", c, output, true)?,
            blocks,
            channels: c,
            context,
        })
    }
    fn validate(&self, points: &[f32], latent: &[f32]) -> Result<(), String> {
        validate_points(points)?;
        if latent.is_empty()
            || latent.len() % self.context != 0
            || latent.len() / self.context > MAX_CONTEXT
        {
            return Err("Invalid Idol decoder latent shape".into());
        }
        finite(latent, "decoder latent")
    }
    fn embed(&self, points: &[f32]) -> Result<Vec<f32>, String> {
        let mut h = self.input.forward(points)?;
        let fd = self.channels / 6;
        for (row, point) in h.chunks_mut(self.channels).zip(points.chunks_exact(3)) {
            for axis in 0..3 {
                for f in 0..fd {
                    let frequency = 2.0_f32.powf(10.0 * f as f32 / (fd - 1).max(1) as f32);
                    let angle = (point[axis] * frequency) * std::f32::consts::PI;
                    row[axis * fd * 2 + f] += angle.sin();
                    row[axis * fd * 2 + fd + f] += angle.cos();
                }
            }
        }
        let mut h = self.input_layer.forward(&h)?;
        round_f16_checked(&mut h, "FP16 decoder embedding")?;
        Ok(h)
    }
    fn forward(
        &self,
        points: &[f32],
        latent: &[f32],
        modulation: Option<&[f32]>,
        name: &str,
    ) -> Result<Vec<f32>, String> {
        self.validate(points, latent)?;
        let mut h = self.embed(points)?;
        trace(
            &format!("idol.{name}.input"),
            &[points.len() / 3, self.channels],
            &h,
        );
        let mut ctx = latent.to_vec();
        round_f16_checked(&mut ctx, "FP16 decoder context")?;
        for (i, block) in self.blocks.iter().enumerate() {
            block.forward(&mut h, &ctx, modulation)?;
            trace(
                &format!("idol.{name}.block.{i}"),
                &[points.len() / 3, self.channels],
                &h,
            );
        }
        let output = self
            .output
            .forward(&norm(&h, self.channels, None, None, 1e-5))?;
        finite(&output, "decoder output")?;
        trace(
            &format!("idol.{name}.output"),
            &[points.len() / 3, self.output.output],
            &output,
        );
        Ok(output)
    }
}
fn validate_points(points: &[f32]) -> Result<(), String> {
    if points.is_empty() || points.len() % 3 != 0 || points.len() / 3 > MAX_ANCHORS {
        return Err("Idol decoder points must be [1..32768,3]".into());
    }
    if points
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err("Idol decoder points must be finite and in [0,1]".into());
    }
    Ok(())
}

pub(super) struct Decoder {
    octree: QueryDecoder,
    gaussian: QueryDecoder,
    level_in: Linear,
    level_out: Linear,
    modulation: Linear,
    perturbation: Vec<f32>,
    base_offset_scale: f32,
}
impl Decoder {
    pub fn load(
        octree_source: &dyn TensorSource,
        gs_source: &dyn TensorSource,
    ) -> Result<Self, String> {
        let perturbation = tensor(gs_source, "points_offset_perturbation", &[32, 3])?;
        let octree = QueryDecoder::load(octree_source, false, 8)?;
        let c = octree.channels;
        Ok(Self {
            octree,
            gaussian: QueryDecoder::load(gs_source, true, 32 * 15)?,
            level_in: Linear::load(octree_source, "l_embedder.mlp.0", 256, c, true)?,
            level_out: Linear::load(octree_source, "l_embedder.mlp.2", c, c, true)?,
            modulation: Linear::load(octree_source, "adaLN_modulation.1", c, 6 * c, true)?,
            perturbation,
            base_offset_scale: tensor(gs_source, "base_offset_scale", &[])?[0],
        })
    }
    pub fn octree_logits(
        &self,
        points: &[f32],
        level: f32,
        latent: &[f32],
    ) -> Result<Vec<f32>, String> {
        if !level.is_finite() || level <= 0.0 {
            return Err("Idol octree resolution must be finite and positive".into());
        }
        let mut embed = vec![0.; 256];
        for f in 0..128 {
            let freq = (-1024.0_f32.ln() * f as f32 / 128.0).exp();
            let angle = (level * freq) * 2.0 * std::f32::consts::PI;
            embed[f] = angle.cos();
            embed[128 + f] = angle.sin();
        }
        let mut h = self.level_in.forward(&embed)?;
        for v in &mut h {
            *v = crate::ops::silu(*v);
        }
        h = self.level_out.forward(&h)?;
        for v in &mut h {
            *v = crate::ops::silu(*v);
        }
        let mut modulation = self.modulation.forward(&h)?;
        round_f16_checked(&mut modulation, "octree modulation")?;
        trace(
            "idol.octree.modulation",
            &[1, modulation.len()],
            &modulation,
        );
        self.octree
            .forward(points, latent, Some(&modulation), "octree")
    }
    pub fn gaussian_features(&self, points: &[f32], latent: &[f32]) -> Result<Vec<f32>, String> {
        self.gaussian.forward(points, latent, None, "gaussian")
    }
    pub fn sample_anchors(
        &self,
        latent: &[f32],
        num_points: usize,
        level: usize,
        rng: &mut impl Rng,
    ) -> Result<Vec<f32>, String> {
        if num_points == 0 || num_points > MAX_ANCHORS || !(1..=20).contains(&level) {
            return Err("Idol requires 1..32768 anchors and octree level 1..20".into());
        }
        let mut cells = vec![([0u32; 3], num_points)];
        for l in 1..=level {
            let parent_resolution = (1u32 << (l - 1)) as f32;
            let points: Vec<_> = cells
                .iter()
                .flat_map(|(cell, _)| cell.iter().map(|&v| (v as f32 + 0.5) / parent_resolution))
                .collect();
            let mut probabilities = self.octree_logits(&points, (1u32 << l) as f32, latent)?;
            probabilities
                .chunks_mut(8)
                .for_each(crate::ops::softmax_inplace);
            let mut children = Vec::with_capacity(num_points);
            for ((cell, count), probs) in cells.iter().zip(probabilities.chunks_exact(8)) {
                let counts = systematic_sample(probs, *count, rng.gen())?;
                for (child, n) in counts.into_iter().enumerate() {
                    if n > 0 {
                        children.push((
                            [
                                cell[0] * 2 + (child as u32 & 1),
                                cell[1] * 2 + ((child as u32 >> 1) & 1),
                                cell[2] * 2 + ((child as u32 >> 2) & 1),
                            ],
                            n,
                        ));
                    }
                }
            }
            cells = children;
        }
        let resolution = (1u32 << level) as f32;
        let mut points = Vec::with_capacity(num_points * 3);
        for (cell, count) in cells {
            for _ in 0..count {
                for coordinate in cell {
                    points.push((coordinate as f32 + rng.gen::<f32>()) / resolution);
                }
            }
        }
        validate_points(&points)?;
        trace("idol.anchors", &[num_points, 3], &points);
        Ok(points)
    }

    /// Standard 3DGS PLY columns, after activation, inverse activation and z-up -> y-up.
    pub fn representation(
        &self,
        points: &[f32],
        features: &[f32],
    ) -> Result<Vec<[f32; 17]>, String> {
        representation(points, features, &self.perturbation, self.base_offset_scale)
    }
}

/// The supplied offset is uniform in [0,1); exposing it makes resampling fixtures deterministic.
fn systematic_sample(probs: &[f32], count: usize, offset: f32) -> Result<Vec<usize>, String> {
    if probs.is_empty()
        || probs.iter().any(|v| !v.is_finite())
        || !offset.is_finite()
        || !(0.0..1.0).contains(&offset)
    {
        return Err("Invalid Idol resampling probabilities or offset".into());
    }
    let mut p: Vec<_> = probs.iter().map(|v| v.max(0.)).collect();
    let sum: f32 = p.iter().sum();
    if !sum.is_finite() {
        return Err("Idol resampling probability sum overflow".into());
    }
    if sum == 0. {
        let uniform = 1. / p.len() as f32;
        p.fill(uniform);
    } else {
        for v in &mut p {
            *v /= sum.max(1.);
        }
    }
    for i in 1..p.len() {
        p[i] += p[i - 1];
    }
    let mut out = vec![0; probs.len()];
    let mut bin = 0;
    for i in 0..count {
        let u = (offset / count as f32 + i as f32 / count as f32).min(1.);
        while bin + 1 < p.len() && p[bin] < u {
            bin += 1;
        }
        out[bin] += 1;
    }
    Ok(out)
}

fn softplus(x: f32) -> f32 {
    if x > 20. {
        x
    } else {
        x.exp().ln_1p()
    }
}
fn representation(
    points: &[f32],
    features: &[f32],
    perturbation: &[f32],
    base_offset_scale: f32,
) -> Result<Vec<[f32; 17]>, String> {
    validate_points(points)?;
    if features.len() != points.len() / 3 * 480
        || perturbation.len() != 96
        || !base_offset_scale.is_finite()
    {
        return Err("Invalid Idol Gaussian representation shape".into());
    }
    finite(features, "Gaussian features")?;
    finite(perturbation, "Gaussian offset perturbation")?;
    let mut output = Vec::with_capacity(points.len() / 3 * 32);
    // Pinned Idol representation_config values are checked by the model's config loader.
    const XYZ_LR: f32 = 1.;
    const COLOR_LR: f32 = 1.;
    const SCALE_LR: f32 = 1.;
    const ROTATION_LR: f32 = 0.1;
    const OPACITY_LR: f32 = 1.;
    const PERTURB_SIZE: f32 = 1.5;
    const SCALING_BIAS: f32 = 0.004;
    const OPACITY_BIAS: f32 = 0.1;
    const MIN_KERNEL: f32 = 0.0009;
    let scaling_offset = SCALING_BIAS + (-(-SCALING_BIAS).exp_m1()).ln();
    let opacity_offset = (OPACITY_BIAS / (1. - OPACITY_BIAS)).ln();
    for (point, feature) in points.chunks_exact(3).zip(features.chunks_exact(480)) {
        for g in 0..32 {
            let mut row = [0.; 17];
            let offset_scale = softplus(feature[448 + g] + base_offset_scale);
            let mut xyz = [0.; 3];
            for axis in 0..3 {
                let offset = (feature[g * 3 + axis] * XYZ_LR + perturbation[g * 3 + axis]).tanh()
                    * 0.5
                    * PERTURB_SIZE
                    * offset_scale;
                xyz[axis] = point[axis] + offset - 0.5;
                row[6 + axis] = feature[96 + g * 3 + axis] * COLOR_LR;
                let scale = softplus(feature[192 + g * 3 + axis] * SCALE_LR + scaling_offset);
                row[10 + axis] = (scale * scale + MIN_KERNEL * MIN_KERNEL)
                    .sqrt()
                    .max(1e-6)
                    .ln();
            }
            row[..3].copy_from_slice(&[xyz[0], -xyz[2], xyz[1]]);
            let opacity = (1. + (-(feature[416 + g] * OPACITY_LR + opacity_offset)).exp())
                .recip()
                .clamp(1e-6, 1. - 1e-6);
            row[9] = (opacity / (1. - opacity)).ln();
            let mut q = [
                feature[288 + g * 4] * ROTATION_LR + 1.,
                feature[288 + g * 4 + 1] * ROTATION_LR,
                feature[288 + g * 4 + 2] * ROTATION_LR,
                feature[288 + g * 4 + 3] * ROTATION_LR,
            ];
            let inv = crate::ops::dot_f32(&q, &q, 4).sqrt().max(1e-12).recip();
            for v in &mut q {
                *v *= inv;
            }
            let a = std::f64::consts::FRAC_1_SQRT_2;
            row[13] = (a * (q[0] as f64 - q[1] as f64)) as f32;
            row[14] = (a * (q[1] as f64 + q[0] as f64)) as f32;
            row[15] = (a * (q[2] as f64 - q[3] as f64)) as f32;
            row[16] = (a * (q[3] as f64 + q[2] as f64)) as f32;
            finite(&row, "PLY attributes")?;
            output.push(row);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn systematic_resampling_preserves_mass_and_uses_left_boundaries() {
        assert_eq!(
            systematic_sample(&[0.25, 0.25, 0.25, 0.25], 8, 0.5).unwrap(),
            [2, 2, 2, 2]
        );
        assert_eq!(
            systematic_sample(&[0., 0., 0., 0.], 4, 0.5).unwrap(),
            [1, 1, 1, 1]
        );
        assert_eq!(systematic_sample(&[0., 1., 0.], 7, 0.5).unwrap(), [0, 7, 0]);
        assert_eq!(systematic_sample(&[0.25, 0.75], 2, 0.5).unwrap(), [1, 1]);
        assert_eq!(systematic_sample(&[1., 1.], 0, 0.5).unwrap(), [0, 0]);
        assert!(systematic_sample(&[f32::NAN, 1.], 1, 0.5).is_err());
        assert!(systematic_sample(&[1.], 1, 1.).is_err());
        // Pinned oracle divides rows by sum.clamp_min(1), even for sub-unit mass.
        assert_eq!(systematic_sample(&[0.1, 0.1], 10, 0.5).unwrap(), [1, 9]);
    }

    #[test]
    fn gaussian_baseline_rotates_positions_and_identity_quaternions() {
        let output = representation(&[0.2, 0.3, 0.4], &[0.; 480], &[0.; 96], 0.).unwrap();
        assert_eq!(output.len(), 32);
        for row in output {
            for (&actual, expected) in row[..3].iter().zip([-0.3, 0.1, -0.2]) {
                assert!((actual - expected).abs() < 1e-6);
            }
            assert_eq!(&row[3..9], &[0.; 6]);
            assert!((row[9] - (0.1f32 / 0.9).ln()).abs() < 1e-6);
            let expected = (0.004f32.powi(2) + 0.0009f32.powi(2)).sqrt().ln();
            for &scale in &row[10..13] {
                assert!((scale - expected).abs() < 1e-6);
            }
            let a = std::f32::consts::FRAC_1_SQRT_2;
            assert_eq!(&row[13..], &[a, a, 0., 0.]);
        }
    }

    #[test]
    fn gaussian_layout_offsets_rotations_and_saturation_are_finite() {
        let mut features = [0.; 480];
        features[0] = 1.;
        features[96] = 2.;
        features[97] = 3.;
        features[98] = 4.;
        features[288] = -10.;
        features[289] = 10.;
        features[290] = 20.;
        features[291] = 30.;
        features[416] = 1000.;
        features[417] = -1000.;
        let output = representation(&[0.5; 3], &features, &[0.; 96], 0.).unwrap();
        let first = &output[0];
        assert!((first[0] - 1f32.tanh() * 0.75 * 2f32.ln()).abs() < 1e-6);
        assert_eq!(&first[6..9], &[2., 3., 4.]);
        let a = std::f32::consts::FRAC_1_SQRT_2;
        let r = 14f32.sqrt().recip();
        for (&actual, expected) in first[13..].iter().zip([-a * r, a * r, -a * r, 5. * a * r]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for row in output {
            assert!(row.iter().all(|v| v.is_finite()));
        }
        assert!(representation(&[0.5; 3], &features[..479], &[0.; 96], 0.).is_err());
        assert!(representation(&[f32::NAN; 3], &features, &[0.; 96], 0.).is_err());
    }

    #[test]
    fn modulation_rounds_each_half_precision_operation() {
        let mut x = [1.0009766, 2.];
        let shift = [0.0009, 0.1];
        let scale = [0.0006, -0.9998];
        modulate(&mut x, &shift, &scale);
        assert_eq!(
            x,
            [
                half(half(1.0009766 * half(1. + scale[0])) + shift[0]),
                half(half(2. * half(1. + scale[1])) + shift[1])
            ]
        );
    }
}
