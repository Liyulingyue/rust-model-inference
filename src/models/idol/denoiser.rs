//! Idol's joint latent/image/camera flow transformer, matching upstream 4702a7f.
use super::common::{finite, norm, round_f16, round_f16_checked, tensor, trace, Linear};
use crate::core::tensor::TensorSource;
use rayon::prelude::*;

const LATENT_WIDTH: usize = 16;
const CAMERA_WIDTH: usize = 5;
const LATENT_TOKENS: usize = 8192;
const WIDTH: usize = 1024;
const HEADS: usize = 16;
const BLOCKS: usize = 24;
const REFINERS: usize = 2;

fn half(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

fn checkpoint(name: &str, x: &[f32], width: usize) -> Result<(), String> {
    finite(x, name)?;
    trace(name, &[1, x.len() / width, width], x);
    Ok(())
}

fn linear_half(linear: &Linear, x: &[f32]) -> Result<Vec<f32>, String> {
    let mut h = linear.forward(x)?;
    round_f16_checked(&mut h, "FP16 denoiser linear output")?;
    Ok(h)
}

fn norm_half(x: &[f32], width: usize, affine: Option<&AffineNorm>) -> Result<Vec<f32>, String> {
    let mut h = norm(
        x,
        width,
        affine.map(|n| n.weight.as_slice()),
        affine.map(|n| n.bias.as_slice()),
        1e-6,
    );
    round_f16_checked(&mut h, "FP16 denoiser normalization")?;
    Ok(h)
}

struct AffineNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl AffineNorm {
    fn load(s: &dyn TensorSource, p: &str, width: usize) -> Result<Self, String> {
        Ok(Self {
            weight: tensor(s, &format!("{p}.weight"), &[width])?,
            bias: tensor(s, &format!("{p}.bias"), &[width])?,
        })
    }
}

struct RePo {
    norm: AffineNorm,
    gate: Linear,
    content: Linear,
    final_map: Linear,
    frequencies: Vec<f32>,
    axes: Vec<usize>,
    heads: usize,
    head_dim: usize,
}

impl RePo {
    fn load(s: &dyn TensorSource, p: &str, width: usize, heads: usize) -> Result<Self, String> {
        let head_dim = width / heads;
        let first = head_dim / 6;
        let sizes = [first, first, head_dim / 2 - 2 * first];
        let mut frequencies = Vec::with_capacity(head_dim / 2);
        let mut axes = Vec::with_capacity(head_dim / 2);
        for (axis, &size) in sizes.iter().enumerate() {
            frequencies.extend(tensor(s, &format!("{p}.freqs_{axis}"), &[size])?);
            axes.extend(std::iter::repeat_n(axis, size));
        }
        Ok(Self {
            norm: AffineNorm::load(s, &format!("{p}.norm"), width)?,
            gate: Linear::load(s, &format!("{p}.gate_map"), width, width / 8, false)?,
            content: Linear::load(s, &format!("{p}.content_map"), width, width / 8, false)?,
            final_map: Linear::load(s, &format!("{p}.final_map"), width / 8, heads * 3, false)?,
            frequencies,
            axes,
            heads,
            head_dim,
        })
    }

    fn forward(&self, x: &[f32], label: &str) -> Result<Vec<f32>, String> {
        let width = self.heads * self.head_dim;
        let h = norm(
            x,
            width,
            Some(&self.norm.weight),
            Some(&self.norm.bias),
            1e-5,
        );
        checkpoint(&format!("{label}.norm"), &h, width)?;
        let mut gate = self.gate.forward(&h)?;
        let content = self.content.forward(&h)?;
        checkpoint(&format!("{label}.gate"), &gate, width / 8)?;
        checkpoint(&format!("{label}.content"), &content, width / 8)?;
        for (g, &c) in gate.iter_mut().zip(&content) {
            *g = crate::ops::silu(*g) * c;
        }
        checkpoint(&format!("{label}.hidden"), &gate, width / 8)?;
        let delta = self.final_map.forward(&gate)?;
        checkpoint(&format!("{label}.delta"), &delta, self.heads * 3)?;
        Ok(delta)
    }

    fn rotate(&self, x: &mut [f32], delta: &[f32]) {
        rotate_head(x, delta, &self.frequencies, &self.axes);
    }
}

fn rotate_head(x: &mut [f32], delta: &[f32], frequencies: &[f32], axes: &[usize]) {
    for (pair, (&frequency, &axis)) in frequencies.iter().zip(axes).enumerate() {
        // clamp_mul's detached term still participates in inference arithmetic.
        let bounded = frequency.tanh();
        let angle =
            (delta[axis] * bounded + delta[axis] * (frequency - bounded)) * std::f32::consts::PI;
        let (sin, cos) = angle.sin_cos();
        let a = x[2 * pair];
        let b = x[2 * pair + 1];
        x[2 * pair] = half(a * cos - b * sin);
        x[2 * pair + 1] = half(a * sin + b * cos);
    }
}

fn head_norm(x: &mut [f32], gamma: &[f32]) {
    let length = (crate::ops::sum_sq_f32(x) as f32).sqrt().max(1e-12);
    let scale = (x.len() as f32).sqrt();
    for (value, &weight) in x.iter_mut().zip(gamma) {
        *value = half((*value / length) * weight * scale);
    }
}

/// QKV stays interleaved; each worker allocates one score row, never an L×L matrix.
fn attention(qkv: &[f32], width: usize, heads: usize) -> Result<Vec<f32>, String> {
    if super::gpu::enabled() {
        let mut q = Vec::with_capacity(qkv.len() / 3);
        let mut k = Vec::with_capacity(qkv.len() / 3);
        let mut v = Vec::with_capacity(qkv.len() / 3);
        for row in qkv.chunks_exact(3 * width) {
            q.extend_from_slice(&row[..width]);
            k.extend_from_slice(&row[width..2 * width]);
            v.extend_from_slice(&row[2 * width..]);
        }
        if let Some(mut output) = super::gpu::attention(&q, &k, &v, width, heads) {
            round_f16_checked(&mut output, "FP16 denoiser attention")?;
            return Ok(output);
        }
    }
    super::gpu::check_failure()?;
    let tokens = qkv.len() / (3 * width);
    let head_dim = width / heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.; tokens * width];
    // The rounding happens inside the workers, so an overflow is collected
    // instead of returned from there.
    let overflow = std::sync::atomic::AtomicBool::new(false);
    out.par_chunks_mut(width).enumerate().for_each_init(
        || vec![0.; tokens],
        |scores, (row, output)| {
            for head in 0..heads {
                let offset = head * head_dim;
                let q_start = row * 3 * width + offset;
                let q = &qkv[q_start..q_start + head_dim];
                for (key, score) in scores.iter_mut().enumerate() {
                    let k_start = key * 3 * width + width + offset;
                    *score =
                        crate::ops::dot_f32(q, &qkv[k_start..k_start + head_dim], head_dim) * scale;
                }
                crate::ops::softmax_inplace(scores);
                crate::ops::attention_value_reduce(
                    qkv,
                    scores,
                    &mut output[offset..offset + head_dim],
                    0,
                    3 * width,
                    2 * width + offset,
                    tokens,
                    head_dim,
                );
            }
            round_f16(output);
            if output.iter().any(|value| !value.is_finite()) {
                overflow.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        },
    );
    if overflow.load(std::sync::atomic::Ordering::Relaxed) {
        return Err("FP16 denoiser attention overflowed".into());
    }
    Ok(out)
}

struct Attention {
    qkv: Linear,
    out: Linear,
    q_gamma: Vec<f32>,
    k_gamma: Vec<f32>,
    heads: usize,
    width: usize,
}

impl Attention {
    fn load(s: &dyn TensorSource, p: &str, width: usize, heads: usize) -> Result<Self, String> {
        let mut qkv = Linear::load(s, &format!("{p}.qkv"), width, 3 * width, true)?;
        let mut out = Linear::load(s, &format!("{p}.out"), width, width, true)?;
        qkv.to_f16()?;
        out.to_f16()?;
        Ok(Self {
            qkv,
            out,
            q_gamma: tensor(s, &format!("{p}.q_norm.gamma"), &[heads, width / heads])?,
            k_gamma: tensor(s, &format!("{p}.k_norm.gamma"), &[heads, width / heads])?,
            heads,
            width,
        })
    }

    fn forward(
        &self,
        x: &[f32],
        repo: &RePo,
        delta: &[f32],
        label: &str,
    ) -> Result<Vec<f32>, String> {
        let mut qkv = linear_half(&self.qkv, x)?;
        checkpoint(&format!("{label}.qkv"), &qkv, 3 * self.width)?;
        let head_dim = self.width / self.heads;
        qkv.par_chunks_mut(3 * self.width)
            .enumerate()
            .for_each(|(row, values)| {
                for head in 0..self.heads {
                    let start = head * head_dim;
                    let d =
                        &delta[(row * self.heads + head) * 3..(row * self.heads + head + 1) * 3];
                    let q = &mut values[start..start + head_dim];
                    repo.rotate(q, d);
                    let k = &mut values[self.width + start..self.width + start + head_dim];
                    repo.rotate(k, d);
                }
            });
        checkpoint(&format!("{label}.qkv_rotary"), &qkv, 3 * self.width)?;
        qkv.par_chunks_mut(3 * self.width).for_each(|values| {
            for head in 0..self.heads {
                let start = head * head_dim;
                head_norm(
                    &mut values[start..start + head_dim],
                    &self.q_gamma[start..start + head_dim],
                );
                let k = &mut values[self.width + start..self.width + start + head_dim];
                head_norm(k, &self.k_gamma[start..start + head_dim]);
            }
        });
        checkpoint(&format!("{label}.qkv_rotary_norm"), &qkv, 3 * self.width)?;
        let h = attention(&qkv, self.width, self.heads)?;
        checkpoint(&format!("{label}.weighted_values"), &h, self.width)?;
        let h = linear_half(&self.out, &h)?;
        checkpoint(&format!("{label}.out"), &h, self.width)?;
        Ok(h)
    }
}

struct Block {
    attn: Attention,
    up: Linear,
    down: Linear,
    norm1: Option<AffineNorm>,
    norm2: Option<AffineNorm>,
    shift: Option<Vec<f32>>,
    width: usize,
}

fn modulate_half(h: &mut [f32], shift: &[f32], scale: &[f32]) {
    for row in h.chunks_exact_mut(shift.len()) {
        for ((value, &shift), &scale) in row.iter_mut().zip(shift).zip(scale) {
            *value = half(half(*value * half(1. + scale)) + shift);
        }
    }
}

fn residual_half(x: &mut [f32], h: &[f32], gate: Option<&[f32]>) {
    for (index, (value, &h)) in x.iter_mut().zip(h).enumerate() {
        let branch = gate.map_or(h, |g| half(h * g[index % g.len()]));
        *value = half(*value + branch);
    }
}

impl Block {
    fn load(
        s: &dyn TensorSource,
        p: &str,
        width: usize,
        heads: usize,
        modulation: bool,
    ) -> Result<Self, String> {
        let mut up = Linear::load(s, &format!("{p}.mlp.mlp.0"), width, 4 * width, true)?;
        let mut down = Linear::load(s, &format!("{p}.mlp.mlp.2"), 4 * width, width, true)?;
        up.to_f16()?;
        down.to_f16()?;
        Ok(Self {
            attn: Attention::load(s, &format!("{p}.attn"), width, heads)?,
            up,
            down,
            norm1: if modulation {
                None
            } else {
                Some(AffineNorm::load(s, &format!("{p}.norm1"), width)?)
            },
            norm2: if modulation {
                None
            } else {
                Some(AffineNorm::load(s, &format!("{p}.norm2"), width)?)
            },
            shift: if modulation {
                Some(tensor(s, &format!("{p}.shift_table"), &[1, 6 * width])?)
            } else {
                None
            },
            width,
        })
    }

    fn forward(
        &self,
        mut x: Vec<f32>,
        modulation: Option<&[f32]>,
        repo: &RePo,
        label: &str,
    ) -> Result<Vec<f32>, String> {
        let delta = repo.forward(&x, &format!("{label}.repo"))?;
        let modulation = match (&self.shift, modulation) {
            (Some(shift), Some(m)) if m.len() == 6 * self.width => Some(
                m.iter()
                    .zip(shift)
                    .map(|(&a, &b)| half(a + half(b)))
                    .collect::<Vec<_>>(),
            ),
            (None, None) => None,
            _ => return Err(format!("Invalid Idol block modulation at {label}")),
        };
        if let Some(m) = &modulation {
            checkpoint(&format!("{label}.modulation"), m, self.width)?;
        }
        let mut h = norm_half(&x, self.width, self.norm1.as_ref())?;
        checkpoint(&format!("{label}.norm1"), &h, self.width)?;
        if let Some(m) = &modulation {
            modulate_half(&mut h, &m[..self.width], &m[self.width..2 * self.width]);
        }
        checkpoint(&format!("{label}.attn_input"), &h, self.width)?;
        let h = self
            .attn
            .forward(&h, repo, &delta, &format!("{label}.attn"))?;
        residual_half(
            &mut x,
            &h,
            modulation
                .as_ref()
                .map(|m| &m[2 * self.width..3 * self.width]),
        );
        checkpoint(&format!("{label}.attn_residual"), &x, self.width)?;
        let mut h = norm_half(&x, self.width, self.norm2.as_ref())?;
        checkpoint(&format!("{label}.norm2"), &h, self.width)?;
        if let Some(m) = &modulation {
            modulate_half(
                &mut h,
                &m[3 * self.width..4 * self.width],
                &m[4 * self.width..5 * self.width],
            );
        }
        checkpoint(&format!("{label}.mlp_input"), &h, self.width)?;
        let mut h = linear_half(&self.up, &h)?;
        checkpoint(&format!("{label}.mlp_up"), &h, 4 * self.width)?;
        crate::ops::gelu_inplace(&mut h);
        round_f16_checked(&mut h, "FP16 denoiser feed-forward")?;
        checkpoint(&format!("{label}.mlp_gelu"), &h, 4 * self.width)?;
        let h = linear_half(&self.down, &h)?;
        checkpoint(&format!("{label}.mlp_down"), &h, self.width)?;
        residual_half(
            &mut x,
            &h,
            modulation
                .as_ref()
                .map(|m| &m[5 * self.width..6 * self.width]),
        );
        checkpoint(&format!("{label}.out"), &x, self.width)?;
        Ok(x)
    }
}

fn position_embedding(positions: &[f32], width: usize) -> Vec<f32> {
    let freq_dim = width / 3 / 2;
    let residual = freq_dim.saturating_sub(16);
    let frequencies: Vec<_> = (0..freq_dim)
        .map(|i| {
            let exponent = if i < 16 {
                i as f32
            } else {
                (i - 16) as f32 / residual as f32 * 16.
            };
            2.0f32.powf(exponent)
        })
        .collect();
    let mut output = vec![0.; positions.len() / 3 * width];
    for (point, out) in positions
        .chunks_exact(3)
        .zip(output.chunks_exact_mut(width))
    {
        for axis in 0..3 {
            for (i, &frequency) in frequencies.iter().enumerate() {
                let angle = point[axis] * frequency * 2. * std::f32::consts::PI;
                out[axis * 2 * freq_dim + i] = half(angle.sin());
                out[axis * 2 * freq_dim + freq_dim + i] = half(angle.cos());
            }
        }
    }
    output
}

fn validate_inputs(
    latent: &[f32],
    camera: &[f32],
    t: f32,
    f1: &[f32],
    f2: &[f32],
    fixture: bool,
) -> Result<usize, String> {
    let rows = latent.len() / LATENT_WIDTH;
    if latent.is_empty()
        || latent.len() % LATENT_WIDTH != 0
        || rows > LATENT_TOKENS
        || (!fixture && rows != LATENT_TOKENS)
        || camera.len() != CAMERA_WIDTH
        || f1.is_empty()
        || f1.len() % 1280 != 0
        || f2.len() != f1.len() / 1280 * 128
        || !t.is_finite()
        || !(0.0..=1000.).contains(&t)
    {
        return Err("Invalid Idol denoiser input: expected latent [8192,16], camera [1,5], time 0..1000, matched feature1 [L,1280] and feature2 [L,128]".into());
    }
    for (name, values) in [
        ("latent", latent),
        ("camera", camera),
        ("feature1", f1),
        ("feature2", f2),
    ] {
        finite(values, name)?;
    }
    Ok(rows)
}

pub(super) struct Denoiser {
    width: usize,
    time_in: Linear,
    time_out: Linear,
    ada: Linear,
    input: Linear,
    cond1: Linear,
    cond2: Linear,
    positions: Vec<f32>,
    noise: Vec<(RePo, Block)>,
    context: Vec<(RePo, Block)>,
    cam: Vec<Linear>,
    blocks: Vec<(RePo, Block)>,
    shift: Vec<f32>,
    out: Linear,
    cam_out: Linear,
}

impl Denoiser {
    pub(super) fn load(s: &dyn TensorSource) -> Result<Self, String> {
        let width = WIDTH;
        let heads = HEADS;
        let load_blocks = |block_prefix: &str,
                           repo_prefix: &str,
                           modulation: bool,
                           count: usize|
         -> Result<Vec<(RePo, Block)>, String> {
            let mut blocks = Vec::with_capacity(count);
            for i in 0..count {
                blocks.push((
                    RePo::load(s, &format!("{repo_prefix}.{i}"), width, heads)?,
                    Block::load(s, &format!("{block_prefix}.{i}"), width, heads, modulation)?,
                ));
            }
            if s.tensor_info(&format!("{block_prefix}.{count}.attn.qkv.weight"))
                .is_some()
            {
                return Err(format!(
                    "Unexpected extra Idol denoiser {block_prefix} block {count}"
                ));
            }
            Ok(blocks)
        };
        let noise = load_blocks("noise_refiner", "noise_repo_layers", true, REFINERS)?;
        let context = load_blocks("context_refiner", "context_repo_layers", false, REFINERS)?;
        let mut cam = Vec::with_capacity(noise.len());
        for i in 0..noise.len() {
            cam.push(Linear::load(
                s,
                &format!("cam_refiner.mlp.{}", 2 * i),
                if i == 0 { CAMERA_WIDTH } else { width },
                width,
                true,
            )?);
        }
        let positions = tensor(s, "pos_pe", &[1, LATENT_TOKENS, 3])?;
        let positions = position_embedding(&positions, width);
        finite(&positions, "denoiser positional embedding")?;
        Ok(Self {
            width,
            time_in: Linear::load(s, "t_embedder.mlp.0", 256, width, true)?,
            time_out: Linear::load(s, "t_embedder.mlp.2", width, width, true)?,
            ada: Linear::load(s, "adaLN_modulation.1", width, 6 * width, true)?,
            input: Linear::load(s, "input_layer", LATENT_WIDTH, width, true)?,
            cond1: Linear::load(s, "cond_embedder", 1280, width, true)?,
            cond2: Linear::load(s, "cond_embedder2", 128, width, true)?,
            positions,
            noise,
            context,
            cam,
            blocks: load_blocks("blocks", "repo_layers", true, BLOCKS)?,
            shift: tensor(s, "shift_table", &[1, 2, width])?,
            out: Linear::load(s, "out_layer", width, LATENT_WIDTH, true)?,
            cam_out: Linear::load(s, "cam_out_layer", width, CAMERA_WIDTH, true)?,
        })
    }

    pub(super) fn forward(
        &self,
        latent: &[f32],
        camera: &[f32],
        t: f32,
        feature1: &[f32],
        feature2: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        self.forward_impl(latent, camera, t, feature1, feature2, false)
    }

    /// Diagnostic oracle fixtures may use a prefix of the stored Sobol positions.
    pub(super) fn forward_fixture(
        &self,
        latent: &[f32],
        camera: &[f32],
        t: f32,
        feature1: &[f32],
        feature2: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        self.forward_impl(latent, camera, t, feature1, feature2, true)
    }

    fn forward_impl(
        &self,
        latent: &[f32],
        camera: &[f32],
        t: f32,
        feature1: &[f32],
        feature2: &[f32],
        fixture: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let latent_rows = validate_inputs(latent, camera, t, feature1, feature2, fixture)?;
        let mut h_x = self.input.forward(latent)?;
        round_f16_checked(&mut h_x, "FP16 denoiser input")?;
        checkpoint("idol.denoiser.input", &h_x, self.width)?;
        let mut h_cond = self.cond1.forward(feature1)?;
        let h_cond2 = self.cond2.forward(feature2)?;
        checkpoint("idol.denoiser.condition1", &h_cond, self.width)?;
        checkpoint("idol.denoiser.condition2", &h_cond2, self.width)?;
        for (value, &other) in h_cond.iter_mut().zip(&h_cond2) {
            *value = half(*value + other);
        }
        checkpoint("idol.denoiser.condition", &h_cond, self.width)?;
        let mut time = vec![0.; 256];
        for i in 0..128 {
            let arg = t * (-10000.0f32.ln() * i as f32 / 128.).exp();
            time[i] = arg.cos();
            time[128 + i] = arg.sin();
        }
        checkpoint("idol.denoiser.time_frequency", &time, 256)?;
        let mut time = self.time_in.forward(&time)?;
        checkpoint("idol.denoiser.time_linear1", &time, self.width)?;
        crate::ops::silu_inplace(&mut time);
        checkpoint("idol.denoiser.time_silu", &time, self.width)?;
        let time = self.time_out.forward(&time)?;
        checkpoint("idol.denoiser.time", &time, self.width)?;
        let mut ada_input = time.clone();
        crate::ops::silu_inplace(&mut ada_input);
        let mut modulation = self.ada.forward(&ada_input)?;
        round_f16_checked(&mut modulation, "FP16 denoiser modulation")?;
        checkpoint("idol.denoiser.modulation", &modulation, self.width)?;
        for (h, &position) in h_x.iter_mut().zip(&self.positions) {
            *h = half(*h + position);
        }
        checkpoint("idol.denoiser.position_added", &h_x, self.width)?;
        for (i, (repo, block)) in self.noise.iter().enumerate() {
            h_x = block.forward(
                h_x,
                Some(&modulation),
                repo,
                &format!("idol.denoiser.noise.{i}"),
            )?;
        }
        for (i, (repo, block)) in self.context.iter().enumerate() {
            h_cond = block.forward(h_cond, None, repo, &format!("idol.denoiser.context.{i}"))?;
        }
        let mut h_cam = camera.to_vec();
        for (i, linear) in self.cam.iter().enumerate() {
            h_cam = linear.forward(&h_cam)?;
            checkpoint(
                &format!("idol.denoiser.camera_linear.{i}"),
                &h_cam,
                self.width,
            )?;
            if i + 1 < self.cam.len() {
                crate::ops::gelu_inplace(&mut h_cam);
            }
            checkpoint(&format!("idol.denoiser.camera.{i}"), &h_cam, self.width)?;
        }
        round_f16_checked(&mut h_cam, "FP16 denoiser camera")?;
        h_x.extend_from_slice(&h_cond);
        h_x.extend_from_slice(&h_cam);
        drop(h_cond);
        for (i, (repo, block)) in self.blocks.iter().enumerate() {
            h_x = block.forward(
                h_x,
                Some(&modulation),
                repo,
                &format!("idol.denoiser.joint.{i}"),
            )?;
        }
        let mut latent_h = norm(
            &h_x[..latent_rows * self.width],
            self.width,
            None,
            None,
            1e-5,
        );
        let mut camera_h = norm(&h_x[h_x.len() - self.width..], self.width, None, None, 1e-5);
        checkpoint("idol.denoiser.final_norm_latent", &latent_h, self.width)?;
        checkpoint("idol.denoiser.final_norm_camera", &camera_h, self.width)?;
        for h in [&mut latent_h, &mut camera_h] {
            for row in h.chunks_exact_mut(self.width) {
                for (i, value) in row.iter_mut().enumerate() {
                    let shift = self.shift[i] + time[i];
                    let scale = self.shift[self.width + i] + time[i];
                    *value = *value * (1. + scale) + shift;
                }
            }
        }
        checkpoint(
            "idol.denoiser.final_modulated_latent",
            &latent_h,
            self.width,
        )?;
        checkpoint(
            "idol.denoiser.final_modulated_camera",
            &camera_h,
            self.width,
        )?;
        let latent = self.out.forward(&latent_h)?;
        let camera = self.cam_out.forward(&camera_h)?;
        checkpoint("idol.denoiser.velocity_latent", &latent, LATENT_WIDTH)?;
        checkpoint("idol.denoiser.velocity_camera", &camera, CAMERA_WIDTH)?;
        Ok((latent, camera))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_uses_future_tokens_and_preserves_head_layout() {
        // Q and K are zero, hence both heads must average both values.
        let qkv = [
            0., 0., 0., 0., 0., 0., 0., 0., 2., 4., 10., 20., 0., 0., 0., 0., 0., 0., 0., 0., 6.,
            8., 30., 40.,
        ];
        assert_eq!(
            attention(&qkv, 4, 2).unwrap(),
            [4., 6., 20., 30., 4., 6., 20., 30.]
        );
    }

    #[test]
    fn rotary_uses_learned_axis_frequencies_then_per_head_norm() {
        let mut x = [1., 0., 1., 0., 1., 0.];
        let frequencies: [f32; 3] = [1., 2., 3.];
        let axes = [0, 1, 2];
        let delta = [0.5, 0.25, 1. / 6.];
        rotate_head(&mut x, &delta, &frequencies, &axes);
        head_norm(&mut x, &[1., 2., 1., 2., 1., 2.]);
        for pair in x.chunks_exact(2) {
            assert!(pair[0].abs() < 1e-5);
            assert!((pair[1] - half(2. * 2.0f32.sqrt())).abs() < 0.002);
        }
        let mut zero = [0.; 6];
        head_norm(&mut zero, &[1.; 6]);
        assert_eq!(zero, [0.; 6]);
    }

    #[test]
    fn half_modulation_rounds_each_tensor_operation() {
        let mut x = [1.];
        let scale = [half(0.0003)];
        let shift = [half(0.0003)];
        modulate_half(&mut x, &shift, &scale);
        assert_eq!(x, [1.]);
        assert_ne!(x[0], half(1. * (1. + scale[0]) + shift[0]));
    }

    #[test]
    fn position_embedding_orders_sin_cos_by_coordinate_and_pads() {
        let embedding = position_embedding(&[0., 0.25, 0.5], 14);
        assert_eq!(&embedding[..4], &[0., 0., 1., 1.]);
        assert!((embedding[4] - 1.).abs() < 0.001);
        assert!(embedding[6].abs() < 0.001);
        assert_eq!(&embedding[12..], &[0., 0.]);
    }

    #[test]
    fn input_validation_keeps_short_fixtures_out_of_production() {
        let latent = [0.; 16];
        let camera = [0.; 5];
        let f1 = [0.; 1280];
        let f2 = [0.; 128];
        assert!(validate_inputs(&latent, &camera, 500., &f1, &f2, true).is_ok());
        assert!(validate_inputs(&latent, &camera, 500., &f1, &f2, false).is_err());
        assert!(validate_inputs(&latent, &camera, f32::NAN, &f1, &f2, true).is_err());
        assert!(validate_inputs(&latent, &camera, 1001., &f1, &f2, true).is_err());
        assert!(validate_inputs(&latent, &camera, 500., &f1, &f2[..127], true).is_err());
        let mut bad = f1;
        bad[100] = f32::INFINITY;
        assert!(validate_inputs(&latent, &camera, 500., &bad, &f2, true).is_err());
    }
}
