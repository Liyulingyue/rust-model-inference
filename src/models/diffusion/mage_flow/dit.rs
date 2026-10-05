use super::{
    validate_dit, AXES_DIM, CONTEXT_DIM, FFN, HEADS, HEAD_DIM, HIDDEN, IN_CHANNELS, LAYERS,
};
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::diffusion::dreamx::kernels::{
    attention_scalar, checked_len, layer_norm_rows, rms_norm_rows, AttentionSpec, Linear,
};
use crate::ops::{gelu_inplace, rms_norm_inplace, rope_sin_cos, silu_inplace};
use std::sync::Arc;

const EPS: f32 = 1e-6;

/// BF16 weights are widened exactly; activations and arithmetic stay in F32.
/// Use RMI_SCALAR=1 for the independent scalar Oracle contract.
pub struct MageFlowDit<'a> {
    source: &'a dyn TensorSource,
    pool: Arc<ComputePool>,
}

impl<'a> MageFlowDit<'a> {
    pub fn load(source: &'a dyn TensorSource, pool: Arc<ComputePool>) -> Result<Self, String> {
        validate_dit(source)?;
        Ok(Self { source, pool })
    }

    fn linear(
        &self,
        name: &str,
        input: &[f32],
        rows: usize,
        width: usize,
        out: usize,
    ) -> Result<Vec<f32>, String> {
        let layer = Linear::from_source(
            self.source,
            &format!("{name}.weight"),
            Some(&format!("{name}.bias")),
            width,
            out,
        )?;
        let values = layer.forward(&self.pool, input, rows)?;
        trace(name, rows, out, &values)?;
        Ok(values)
    }

    pub fn sample(
        &self,
        packed: &[f32],
        shapes: &[[usize; 2]],
        context: &[f32],
        negative: Option<&[f32]>,
        steps: usize,
        cfg: f32,
    ) -> Result<Vec<f32>, String> {
        if steps == 0
            || steps > 1000
            || !cfg.is_finite()
            || cfg < 1.0
            || (cfg > 1.0 && negative.is_none())
        {
            return Err("Invalid Mage steps/CFG or missing negative conditioning".into());
        }
        image_positions(shapes)?;
        let total: usize = shapes.iter().map(|s| s[0] * s[1] * IN_CHANNELS).sum();
        if packed.len() != total {
            return Err("Invalid packed Mage sample".into());
        }
        let target = shapes[0][0] * shapes[0][1] * IN_CHANNELS;
        let sigmas = static_sigmas(steps);
        trace("sample.sigmas", 1, sigmas.len(), &sigmas)?;
        let mut image = packed.to_vec();
        for step in 0..steps {
            let mut velocity = self.forward(&image, shapes, context, sigmas[step])?;
            if cfg > 1.0 {
                let unconditioned =
                    self.forward(&image, shapes, negative.unwrap(), sigmas[step])?;
                for (v, u) in velocity.iter_mut().zip(unconditioned) {
                    *v = u + cfg * (*v - u);
                }
            }
            trace("sample.velocity", 1, target, &velocity[..target])?;
            let dt = sigmas[step + 1] - sigmas[step];
            for (x, v) in image[..target].iter_mut().zip(&velocity) {
                *x += dt * v;
            }
            trace("sample.latent", 1, target, &image[..target])?;
        }
        Ok(image[..target].to_vec())
    }

    /// One packed sample: shapes are [target, reference_1, ...], each [H/16,W/16].
    /// Reference velocities are returned, but only target tokens are stepped by the sampler.
    pub fn forward(
        &self,
        image: &[f32],
        shapes: &[[usize; 2]],
        context: &[f32],
        sigma: f32,
    ) -> Result<Vec<f32>, String> {
        let positions = image_positions(shapes)?;
        let image_tokens = positions.len();
        if image.len() != checked_len("Mage-Flow image", &[image_tokens, IN_CHANNELS])?
            || context.is_empty()
            || context.len() % CONTEXT_DIM != 0
            || context.len() / CONTEXT_DIM > 2048
            || !sigma.is_finite()
            || !(0.0..=1.0).contains(&sigma)
            || image.iter().chain(context).any(|v| !v.is_finite())
        {
            return Err("Invalid Mage-Flow image/context/sigma".into());
        }
        let text_tokens = context.len() / CONTEXT_DIM;
        let mut img = self.linear("img_in", image, image_tokens, IN_CHANNELS, HIDDEN)?;
        let txt_weight = load_f32_tensor(self.source, "txt_norm.weight", &[CONTEXT_DIM as u64])?;
        let txt_norm = rms_norm_rows(context, text_tokens, &txt_weight, EPS)?;
        trace("txt_norm", text_tokens, CONTEXT_DIM, &txt_norm)?;
        let time = timestep_embedding(sigma);
        trace("time_proj", 1, 256, &time)?;
        let mut time = self.linear(
            "time_text_embed.timestep_embedder.linear_1",
            &time,
            1,
            256,
            HIDDEN,
        )?;
        silu_inplace(&mut time);
        let time = self.linear(
            "time_text_embed.timestep_embedder.linear_2",
            &time,
            1,
            HIDDEN,
            HIDDEN,
        )?;
        let mut txt = self.linear("txt_in", &txt_norm, text_tokens, CONTEXT_DIM, HIDDEN)?;
        let mut time_silu = time.clone();
        silu_inplace(&mut time_silu);
        let rotary = rotary_frequencies(&positions);
        for layer in 0..LAYERS {
            let p = format!("transformer_blocks.{layer}");
            let im = self.linear(&format!("{p}.img_mod.1"), &time_silu, 1, HIDDEN, 6 * HIDDEN)?;
            let tm = self.linear(&format!("{p}.txt_mod.1"), &time_silu, 1, HIDDEN, 6 * HIDDEN)?;
            let ni = modulated_norm(&img, image_tokens, &im[..3 * HIDDEN])?;
            let nt = modulated_norm(&txt, text_tokens, &tm[..3 * HIDDEN])?;
            trace(&format!("{p}.img_modulated"), image_tokens, HIDDEN, &ni)?;
            trace(&format!("{p}.txt_modulated"), text_tokens, HIDDEN, &nt)?;
            let mut iq =
                self.linear(&format!("{p}.attn.to_q"), &ni, image_tokens, HIDDEN, HIDDEN)?;
            let mut ik =
                self.linear(&format!("{p}.attn.to_k"), &ni, image_tokens, HIDDEN, HIDDEN)?;
            let iv = self.linear(&format!("{p}.attn.to_v"), &ni, image_tokens, HIDDEN, HIDDEN)?;
            let mut tq = self.linear(
                &format!("{p}.attn.add_q_proj"),
                &nt,
                text_tokens,
                HIDDEN,
                HIDDEN,
            )?;
            let mut tk = self.linear(
                &format!("{p}.attn.add_k_proj"),
                &nt,
                text_tokens,
                HIDDEN,
                HIDDEN,
            )?;
            let tv = self.linear(
                &format!("{p}.attn.add_v_proj"),
                &nt,
                text_tokens,
                HIDDEN,
                HIDDEN,
            )?;
            for (name, values) in [
                ("norm_q", &mut iq),
                ("norm_k", &mut ik),
                ("norm_added_q", &mut tq),
                ("norm_added_k", &mut tk),
            ] {
                let w = load_f32_tensor(
                    self.source,
                    &format!("{p}.attn.{name}.weight"),
                    &[HEAD_DIM as u64],
                )?;
                for head in values.chunks_exact_mut(HEAD_DIM) {
                    rms_norm_inplace(head, &w, EPS);
                }
                trace(
                    &format!("{p}.attn.{name}"),
                    values.len() / HEAD_DIM,
                    HEAD_DIM,
                    values,
                )?;
            }
            apply_rope(&mut iq, &rotary);
            apply_rope(&mut ik, &rotary);
            trace(&format!("{p}.attn.rope_q"), image_tokens, HIDDEN, &iq)?;
            trace(&format!("{p}.attn.rope_k"), image_tokens, HIDDEN, &ik)?;
            tq.extend(iq);
            tk.extend(ik);
            let mut value = tv;
            value.extend(iv);
            let n = text_tokens + image_tokens;
            let attn = attention_scalar(
                &tq,
                &tk,
                &value,
                AttentionSpec {
                    query_tokens: n,
                    key_tokens: n,
                    query_heads: HEADS,
                    key_value_heads: HEADS,
                    head_dim: HEAD_DIM,
                    causal: false,
                    scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                },
            )?;
            trace(&format!("{p}.attn.context"), n, HIDDEN, &attn)?;
            let ti = text_tokens * HIDDEN;
            let pi = self.linear(
                &format!("{p}.attn.to_out.0"),
                &attn[ti..],
                image_tokens,
                HIDDEN,
                HIDDEN,
            )?;
            let pt = self.linear(
                &format!("{p}.attn.to_add_out"),
                &attn[..ti],
                text_tokens,
                HIDDEN,
                HIDDEN,
            )?;
            residual(&mut img, &pi, &im[2 * HIDDEN..3 * HIDDEN]);
            residual(&mut txt, &pt, &tm[2 * HIDDEN..3 * HIDDEN]);
            trace(&format!("{p}.img_after_attn"), image_tokens, HIDDEN, &img)?;
            trace(&format!("{p}.txt_after_attn"), text_tokens, HIDDEN, &txt)?;
            for (stream, hidden, rows, params) in [
                ("img", &mut img, image_tokens, &im[3 * HIDDEN..]),
                ("txt", &mut txt, text_tokens, &tm[3 * HIDDEN..]),
            ] {
                let norm = modulated_norm(hidden, rows, params)?;
                let mut up = self.linear(
                    &format!("{p}.{stream}_mlp.net.0.proj"),
                    &norm,
                    rows,
                    HIDDEN,
                    FFN,
                )?;
                gelu_inplace(&mut up);
                trace(&format!("{p}.{stream}_mlp.gelu"), rows, FFN, &up)?;
                let down =
                    self.linear(&format!("{p}.{stream}_mlp.net.2"), &up, rows, FFN, HIDDEN)?;
                residual(hidden, &down, &params[2 * HIDDEN..]);
                trace(&format!("{p}.{stream}"), rows, HIDDEN, hidden)?;
            }
        }
        let modulation = self.linear("norm_out.linear", &time_silu, 1, HIDDEN, 2 * HIDDEN)?;
        let mut img = unweighted_norm(&img, image_tokens, HIDDEN)?;
        for row in img.chunks_exact_mut(HIDDEN) {
            for j in 0..HIDDEN {
                row[j] = row[j] * (1.0 + modulation[j]) + modulation[HIDDEN + j];
            }
        }
        trace("norm_out", image_tokens, HIDDEN, &img)?;
        self.linear("proj_out", &img, image_tokens, HIDDEN, IN_CHANNELS)
    }
}

pub(crate) fn trace(name: &str, rows: usize, width: usize, values: &[f32]) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    {
        let name = format!("mage_flow.{name}");
        if crate::parity_trace::enabled(&name) {
            crate::parity_trace::checkpoint(&name, None, &[rows, width], values)
                .map_err(|e| e.to_string())?;
        }
    }
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, rows, width, values);
    Ok(())
}

pub(crate) fn unweighted_norm(
    input: &[f32],
    rows: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    layer_norm_rows(input, rows, Some(&vec![1.0; width]), None, EPS)
}

fn modulated_norm(input: &[f32], rows: usize, params: &[f32]) -> Result<Vec<f32>, String> {
    let mut values = unweighted_norm(input, rows, HIDDEN)?;
    for row in values.chunks_exact_mut(HIDDEN) {
        for j in 0..HIDDEN {
            row[j] = row[j] * (1.0 + params[HIDDEN + j]) + params[j];
        }
    }
    Ok(values)
}

fn residual(hidden: &mut [f32], branch: &[f32], gate: &[f32]) {
    for (i, (x, &y)) in hidden.iter_mut().zip(branch).enumerate() {
        *x += gate[i % HIDDEN] * y;
    }
}

pub(crate) fn timestep_embedding(sigma: f32) -> Vec<f32> {
    let mut values = vec![0.0; 256];
    for i in 0..128 {
        let frequency = ((-(10_000.0f64).ln() as f32) * i as f32 / 128.0).exp();
        let angle = (sigma * frequency) * 1000.0;
        let (cos, sin) = rope_sin_cos(angle);
        values[i] = cos;
        values[128 + i] = sin;
    }
    values
}

pub(crate) fn image_positions(shapes: &[[usize; 2]]) -> Result<Vec<[f32; 3]>, String> {
    if shapes.is_empty() || shapes.len() > 4 {
        return Err("Mage-Flow needs a target and at most 3 references".into());
    }
    let mut total = 0usize;
    for shape in shapes {
        if shape.iter().any(|&v| v == 0 || v > 4096) {
            return Err("Mage-Flow latent sides must be in 1..=4096".into());
        }
        total = total
            .checked_add(checked_len("Mage-Flow latent", shape)?)
            .ok_or("Mage-Flow token count overflow")?;
    }
    if total > 16384 {
        return Err("Mage-Flow packed latent exceeds 16384 tokens".into());
    }
    let mut positions = Vec::with_capacity(total);
    for (index, &[height, width]) in shapes.iter().enumerate() {
        for h in 0..height {
            for w in 0..width {
                positions.push([
                    index as f32,
                    h as f32 - height.div_ceil(2) as f32,
                    w as f32 - width.div_ceil(2) as f32,
                ]);
            }
        }
    }
    Ok(positions)
}

fn rotary_frequencies(positions: &[[f32; 3]]) -> Vec<[f32; 2]> {
    let mut values = Vec::with_capacity(positions.len() * HEAD_DIM / 2);
    for position in positions {
        for (axis, dim) in AXES_DIM.into_iter().enumerate() {
            for pair in 0..dim / 2 {
                let inverse = 1.0 / 10_000.0f32.powf(2.0 * pair as f32 / dim as f32);
                let angle = position[axis] * inverse;
                let (cos, sin) = rope_sin_cos(angle);
                values.push([cos, sin]);
            }
        }
    }
    values
}

fn apply_rope(values: &mut [f32], rotary: &[[f32; 2]]) {
    for (r, row) in values.chunks_exact_mut(HIDDEN).enumerate() {
        for head in row.chunks_exact_mut(HEAD_DIM) {
            for (pair, &[cos, sin]) in head
                .chunks_exact_mut(2)
                .zip(&rotary[r * HEAD_DIM / 2..(r + 1) * HEAD_DIM / 2])
            {
                let [a, b] = [pair[0], pair[1]];
                pair[0] = a * cos - b * sin;
                pair[1] = a * sin + b * cos;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_trace_does_not_require_a_destination() {
        if std::env::var_os("RMI_PARITY_TRACE").is_none() {
            trace("disabled", 1, 1, &[1.0]).unwrap();
        }
    }
    #[test]
    fn reference_rope_uses_its_own_frame_and_centered_grid() {
        assert_eq!(
            image_positions(&[[1, 2], [3, 1]]).unwrap(),
            vec![
                [0., -1., -1.],
                [0., -1., 0.],
                [1., -2., -1.],
                [1., -1., -1.],
                [1., 0., -1.]
            ]
        );
        assert!(image_positions(&[[usize::MAX, 1]]).is_err());
        assert!(image_positions(&[[0, 1]]).is_err());
        assert!(image_positions(&[[4096, 4096]]).is_err());
    }
    #[test]
    fn scheduler_matches_pinned_scalar_oracle_bits() {
        // Mage 76bec2bb build_scheduler, C scalar linspace, static shift=6.
        assert_eq!(
            static_sigmas(4)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            vec![0x3f800000, 0x3f7286bd, 0x3f5b6db7, 0x3f2aaaab, 0x00000000]
        );
        assert_eq!(
            static_sigmas(20)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            vec![
                0x3f800000, 0x3f7dc61f, 0x3f7b586f, 0x3f78af8c, 0x3f75c290, 0x3f7286bd, 0x3f6eeeee,
                0x3f6aeaea, 0x3f666667, 0x3f6147ae, 0x3f5b6db7, 0x3f54ad4b, 0x3f4ccccd, 0x3f437dad,
                0x3f3851ec, 0x3f2aaaab, 0x3f19999a, 0x3f03a83b, 0x3ecccccd, 0x3e75c290, 0x00000000
            ]
        );
        assert_eq!(
            static_sigmas(30)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            vec![
                0x3f800000, 0x3f7e8982, 0x3f7cfcfd, 0x3f7b586f, 0x3f799999, 0x3f77bdee, 0x3f75c290,
                0x3f73a432, 0x3f715f15, 0x3f6eeef0, 0x3f6c4ec4, 0x3f6978d5, 0x3f666667, 0x3f630f96,
                0x3f5f6b0f, 0x3f5b6db7, 0x3f570a3c, 0x3f523082, 0x3f4ccccc, 0x3f46c6c6, 0x3f400000,
                0x3f3851eb, 0x3f2f8af8, 0x3f256a56, 0x3f199999, 0x3f0ba2e9, 0x3ef5c28f, 0x3ecccccc,
                0x3e99999a, 0x3e2f8afa, 0x00000000
            ]
        );
    }
    #[test]
    fn time_zero_is_cos_then_sin() {
        let t = timestep_embedding(0.0);
        assert!(t[..128].iter().all(|&v| v.to_bits() == 1.0f32.to_bits()));
        assert!(t[128..].iter().all(|&v| v.to_bits() == 0.0f32.to_bits()));
    }
}

pub(crate) fn static_sigmas(steps: usize) -> Vec<f32> {
    let end = 1.0 / steps as f32;
    let step = if steps == 1 {
        0.0
    } else {
        (end - 1.0) / (steps - 1) as f32
    };
    let mut values: Vec<f32> = (0..steps)
        .map(|i| {
            let value = if i < steps / 2 {
                1.0 + step * i as f32
            } else {
                end - step * (steps - 1 - i) as f32
            };
            6.0 * value / (1.0 + 5.0 * value)
        })
        .collect();
    values.push(0.0);
    values
}
