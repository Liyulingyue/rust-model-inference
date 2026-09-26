use crate::ops::quant::BlockQ8K;

use super::ar::{
    add_in_place, dot, rms_norm, rms_norm_heads, rope, silu, softmax,
    YuE2AttentionWeights, YuE2MlpWeights, YuE2Weight,
};
use super::protocol::{CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END, VOCAB_SIZE};
use super::YuE2Model;

const LATENT_CHANNELS: usize = 64;

#[derive(Debug, Clone)]
pub struct YuE2Chunk {
    pub ar_tokens: Vec<u32>,
    pub noise: Vec<f32>,
    pub context_start: usize,
    pub context_end: usize,
}

pub fn song_chunks(
    prefix: &[u32],
    codec: &[u32],
    seed: u64,
    context: usize,
) -> Result<Vec<YuE2Chunk>, String> {
    if prefix.is_empty()
        || prefix.iter().any(|&token| token as usize >= VOCAB_SIZE)
        || codec.is_empty()
        || codec.iter().any(|&token| token as usize >= CODEC_SIZE)
    {
        return Err("YuE2 prefix or codec tokens are empty or outside their vocabulary".into());
    }
    if !(1..=CONTEXT).contains(&context) {
        return Err(format!("YuE2 context must be within 1..={CONTEXT}"));
    }
    let room = context
        .checked_sub(prefix.len().saturating_add(3))
        .ok_or("YuE2 prefix leaves no acoustic context")?;
    let chunk_size = (room / 2).min(CONTEXT);
    if chunk_size == 0 {
        return Err("YuE2 prefix leaves no acoustic context".into());
    }
    let noise = torch_randn(seed, codec.len() * LATENT_CHANNELS);
    let chunks = (0..codec.len())
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(codec.len());
            let mut ar_tokens = Vec::with_capacity(prefix.len() + end - start + 1);
            ar_tokens.extend_from_slice(prefix);
            ar_tokens.extend(codec[start..end].iter().map(|&token| token + CODEC_OFFSET));
            ar_tokens.push(MUSIC_END);
            YuE2Chunk {
                ar_tokens,
                noise: noise[start * LATENT_CHANNELS..end * LATENT_CHANNELS].to_vec(),
                context_start: start,
                context_end: end,
            }
        })
        .collect::<Vec<_>>();
    trace(
        "yue2.nar.noise",
        None,
        None,
        &[codec.len(), LATENT_CHANNELS],
        &noise,
    );
    trace_tokens(
        "yue2.nar.chunk_ranges",
        &chunks
            .iter()
            .flat_map(|chunk| [chunk.context_start as u32, chunk.context_end as u32])
            .collect::<Vec<_>>(),
    );
    Ok(chunks)
}

pub(super) struct VisibilityMask {
    ar_len: usize,
    total_len: usize,
}

pub(super) fn visibility_mask(ar_len: usize, nar_len: usize) -> VisibilityMask {
    VisibilityMask {
        ar_len,
        total_len: ar_len.saturating_add(nar_len),
    }
}

impl VisibilityMask {
    pub(super) fn visible(&self, query: usize, key: usize) -> bool {
        if query >= self.total_len || key >= self.total_len {
            return false;
        }
        if query < self.ar_len {
            key <= query && key < self.ar_len
        } else {
            true
        }
    }
}

pub(super) fn midpoint_times(steps: usize) -> Result<Vec<(f64, f64)>, String> {
    if steps == 0 {
        return Err("YuE2 midpoint steps must be positive".into());
    }
    let dt = 1.0 / steps as f64;
    Ok((0..steps)
        .map(|step| {
            let time = 1.0 - step as f64 * dt;
            (time, time - dt / 2.0)
        })
        .collect())
}

pub struct YuE2NarSession<'model> {
    model: &'model YuE2Model,
    chunk: YuE2Chunk,
    prefix_kv: Vec<(Vec<f32>, Vec<f32>)>,
}

impl<'model> YuE2NarSession<'model> {
    pub fn new(model: &'model YuE2Model, chunk: YuE2Chunk) -> Result<Self, String> {
        let channels = model.config().latent_channels;
        if chunk.ar_tokens.is_empty()
            || chunk
                .ar_tokens
                .iter()
                .any(|&token| token as usize >= model.config().vocab)
        {
            return Err("YuE2 NAR prefix is empty or outside the model vocabulary".into());
        }
        if chunk.noise.is_empty()
            || chunk.noise.len() % channels != 0
            || chunk.noise.iter().any(|value| !value.is_finite())
        {
            return Err("YuE2 acoustic noise must be finite frame-major data".into());
        }
        let nar_len = chunk.noise.len() / channels + 2;
        if chunk.ar_tokens.len() + nar_len > model.config().context {
            return Err("YuE2 acoustic chunk exceeds the model context".into());
        }
        let prefix_kv = prefix_kv(model, &chunk.ar_tokens);
        for (layer, (keys, values)) in prefix_kv.iter().enumerate() {
            trace(
                "yue2.nar.prefix_k",
                Some(layer),
                None,
                &[
                    chunk.ar_tokens.len(),
                    model.config().kv_heads,
                    model.config().head_dim,
                ],
                keys,
            );
            trace(
                "yue2.nar.prefix_v",
                Some(layer),
                None,
                &[
                    chunk.ar_tokens.len(),
                    model.config().kv_heads,
                    model.config().head_dim,
                ],
                values,
            );
        }
        Ok(Self {
            model,
            chunk,
            prefix_kv,
        })
    }

    pub fn velocity(&self, state: &[f32], raw_t: f64) -> Result<Vec<f32>, String> {
        if state.len() != self.chunk.noise.len() {
            return Err("YuE2 ODE state shape changed".into());
        }
        if !raw_t.is_finite() || state.iter().any(|value| !value.is_finite()) {
            return Err("YuE2 ODE state or timestep is non-finite".into());
        }
        let config = self.model.config();
        let frames = state.len() / config.latent_channels;
        let nar_len = frames + 2;
        let hidden = config.hidden;
        let q_width = config.q_heads * config.head_dim;
        let kv_width = config.kv_heads * config.head_dim;
        let mut scratch = LinearScratch::new(config.ffn.max(hidden).max(q_width));

        let shifted = shifted_time(raw_t, config.timestep_shift);
        let time = time_embedding(self.model, shifted, &mut scratch);
        let mut position = vec![0.0; nar_len * hidden];
        for (row, output) in position.chunks_exact_mut(hidden).enumerate() {
            self.model
                .aux
                .latent_position
                .embedding_lookup(row as u32, output);
        }
        trace("yue2.nar.time_embedding", None, None, &[hidden], &time);
        trace(
            "yue2.nar.position_embedding",
            None,
            None,
            &[nar_len, hidden],
            &position,
        );

        let mut x = vec![0.0; nar_len * hidden];
        let mut latent_row = vec![0.0; config.latent_channels];
        for row in 0..nar_len {
            latent_row.fill(0.0);
            if (1..=frames).contains(&row) {
                let start = (row - 1) * config.latent_channels;
                for (output, &input) in latent_row
                    .iter_mut()
                    .zip(&state[start..start + config.latent_channels])
                {
                    *output = bf16(input);
                }
            }
            let output = &mut x[row * hidden..(row + 1) * hidden];
            self.model.aux.vae2llm.matmul_bias(
                &latent_row,
                &self.model.aux.vae2llm_bias,
                output,
                &self.model.pool,
                &mut scratch.q8,
                &mut scratch.scales,
                &mut scratch.q8k,
            );
            for ((value, &time), &position) in output
                .iter_mut()
                .zip(&time)
                .zip(&position[row * hidden..(row + 1) * hidden])
            {
                *value = bf16(*value + time);
                *value = bf16(*value + position);
            }
        }
        trace("yue2.nar.input", None, None, &[nar_len, hidden], &x);

        let mut normed = vec![0.0; nar_len * hidden];
        let mut q = vec![0.0; nar_len * q_width];
        let mut k = vec![0.0; nar_len * kv_width];
        let mut v = vec![0.0; nar_len * kv_width];
        let mut attention = vec![0.0; nar_len * q_width];
        let mut projected = vec![0.0; nar_len * hidden];
        let mut gate = vec![0.0; nar_len * config.ffn];
        let mut up = vec![0.0; nar_len * config.ffn];
        let mut down = vec![0.0; nar_len * hidden];

        for (layer_index, layer) in self.model.layers.iter().enumerate() {
            normalize_rows(
                &x,
                &layer.nar_attention.norm,
                &mut normed,
                hidden,
                config.rms_eps,
            );
            trace(
                "yue2.nar.attn_norm",
                Some(layer_index),
                None,
                &[nar_len, hidden],
                &normed,
            );
            project_qkv_rows(
                self.model,
                &layer.nar_attention,
                &normed,
                self.chunk.ar_tokens.len(),
                &mut q,
                &mut k,
                &mut v,
                &mut scratch,
            );
            trace(
                "yue2.nar.q",
                Some(layer_index),
                None,
                &[nar_len, config.q_heads, config.head_dim],
                &q,
            );
            trace(
                "yue2.nar.k",
                Some(layer_index),
                None,
                &[nar_len, config.kv_heads, config.head_dim],
                &k,
            );
            trace(
                "yue2.nar.v",
                Some(layer_index),
                None,
                &[nar_len, config.kv_heads, config.head_dim],
                &v,
            );
            hybrid_attention(
                config,
                &q,
                &self.prefix_kv[layer_index],
                &k,
                &v,
                &mut attention,
            );
            trace(
                "yue2.nar.attn",
                Some(layer_index),
                None,
                &[nar_len, q_width],
                &attention,
            );
            linear_rows(
                &layer.nar_attention.output,
                &attention,
                &mut projected,
                q_width,
                hidden,
                self.model,
                &mut scratch,
            );
            trace(
                "yue2.nar.attn_output",
                Some(layer_index),
                None,
                &[nar_len, hidden],
                &projected,
            );
            for (row, update) in x
                .chunks_exact_mut(hidden)
                .zip(projected.chunks_exact(hidden))
            {
                add_in_place(row, update);
            }
            trace(
                "yue2.nar.attn_residual",
                Some(layer_index),
                None,
                &[nar_len, hidden],
                &x,
            );

            normalize_rows(&x, &layer.nar_mlp.norm, &mut normed, hidden, config.rms_eps);
            trace(
                "yue2.nar.ffn_norm",
                Some(layer_index),
                None,
                &[nar_len, hidden],
                &normed,
            );
            mlp_rows(
                self.model,
                &layer.nar_mlp,
                &normed,
                &mut gate,
                &mut up,
                &mut down,
                &mut scratch,
            );
            trace(
                "yue2.nar.ffn_gate",
                Some(layer_index),
                None,
                &[nar_len, config.ffn],
                &gate,
            );
            trace(
                "yue2.nar.ffn_up",
                Some(layer_index),
                None,
                &[nar_len, config.ffn],
                &up,
            );
            trace(
                "yue2.nar.ffn_down",
                Some(layer_index),
                None,
                &[nar_len, hidden],
                &down,
            );
            for (row, update) in x.chunks_exact_mut(hidden).zip(down.chunks_exact(hidden)) {
                add_in_place(row, update);
            }
            trace(
                "yue2.nar.ffn_residual",
                Some(layer_index),
                None,
                &[nar_len, hidden],
                &x,
            );
        }

        normalize_rows(
            &x,
            &self.model.final_norm,
            &mut normed,
            hidden,
            config.rms_eps,
        );
        trace(
            "yue2.nar.final_norm",
            None,
            None,
            &[nar_len, hidden],
            &normed,
        );
        let mut velocity = vec![0.0; nar_len * config.latent_channels];
        linear_bias_rows(
            &self.model.aux.llm2vae,
            &self.model.aux.llm2vae_bias,
            &normed,
            &mut velocity,
            hidden,
            config.latent_channels,
            self.model,
            &mut scratch,
        );
        let content =
            velocity[config.latent_channels..(frames + 1) * config.latent_channels].to_vec();
        if content.iter().any(|value| !value.is_finite()) {
            return Err("YuE2 NAR produced non-finite velocity".into());
        }
        Ok(content)
    }

    pub fn solve(&self, steps: usize) -> Result<Vec<f32>, String> {
        let times = midpoint_times(steps)?;
        let mut state = self
            .chunk
            .noise
            .iter()
            .copied()
            .map(bf16)
            .collect::<Vec<_>>();
        let dt = 1.0 / steps as f32;
        let channels = self.model.config().latent_channels;
        for (step, (time, midpoint)) in times.into_iter().enumerate() {
            let first = self.velocity(&state, raw_time(time))?;
            trace(
                "yue2.nar.velocity_first",
                None,
                Some(step),
                &[state.len() / channels, channels],
                &first,
            );
            let middle = state
                .iter()
                .zip(&first)
                .map(|(&value, &velocity)| bf16(value - bf16(velocity * (dt / 2.0))))
                .collect::<Vec<_>>();
            let second = self.velocity(&middle, raw_time(midpoint))?;
            trace(
                "yue2.nar.velocity_midpoint",
                None,
                Some(step),
                &[state.len() / channels, channels],
                &second,
            );
            for (value, velocity) in state.iter_mut().zip(second) {
                *value = bf16(*value - bf16(velocity * dt));
            }
        }
        if state.iter().any(|value| !value.is_finite()) {
            return Err("YuE2 acoustic flow matching produced non-finite latents".into());
        }
        trace(
            "yue2.nar.latents",
            None,
            None,
            &[state.len() / channels, channels],
            &state,
        );
        Ok(state)
    }
}

struct LinearScratch {
    q8: Vec<u8>,
    scales: Vec<f32>,
    q8k: Vec<BlockQ8K>,
}

fn prefix_kv(model: &YuE2Model, tokens: &[u32]) -> Vec<(Vec<f32>, Vec<f32>)> {
    let config = model.config();
    let rows = tokens.len();
    let hidden = config.hidden;
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    let mut scratch = LinearScratch::new(config.ffn.max(hidden).max(q_width));
    let mut x = vec![0.0; rows * hidden];
    for (&token, output) in tokens.iter().zip(x.chunks_exact_mut(hidden)) {
        model.token_embedding.embedding_lookup(token, output);
    }
    let mut normed = vec![0.0; rows * hidden];
    let mut q = vec![0.0; rows * q_width];
    let mut k = vec![0.0; rows * kv_width];
    let mut v = vec![0.0; rows * kv_width];
    let mut attention = vec![0.0; rows * q_width];
    let mut projected = vec![0.0; rows * hidden];
    let mut gate = vec![0.0; rows * config.ffn];
    let mut up = vec![0.0; rows * config.ffn];
    let mut down = vec![0.0; rows * hidden];
    let mut cache = Vec::with_capacity(config.layers);

    for layer in &model.layers {
        normalize_rows(
            &x,
            &layer.ar_attention.norm,
            &mut normed,
            hidden,
            config.rms_eps,
        );
        project_qkv_rows(
            model,
            &layer.ar_attention,
            &normed,
            0,
            &mut q,
            &mut k,
            &mut v,
            &mut scratch,
        );
        cache.push((k.clone(), v.clone()));
        causal_prefix_attention(config, &q, &k, &v, &mut attention);
        linear_rows(
            &layer.ar_attention.output,
            &attention,
            &mut projected,
            q_width,
            hidden,
            model,
            &mut scratch,
        );
        for (row, update) in x
            .chunks_exact_mut(hidden)
            .zip(projected.chunks_exact(hidden))
        {
            add_in_place(row, update);
        }
        normalize_rows(&x, &layer.ar_mlp.norm, &mut normed, hidden, config.rms_eps);
        mlp_rows(
            model,
            &layer.ar_mlp,
            &normed,
            &mut gate,
            &mut up,
            &mut down,
            &mut scratch,
        );
        for (row, update) in x.chunks_exact_mut(hidden).zip(down.chunks_exact(hidden)) {
            add_in_place(row, update);
        }
    }
    cache
}

fn causal_prefix_attention(
    config: &super::YuE2Config,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
) {
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    let rows = k.len() / kv_width;
    let group_size = config.q_heads / config.kv_heads;
    let scale = (config.head_dim as f32).sqrt().recip();
    let mut scores = vec![0.0; rows];
    for row in 0..rows {
        for head in 0..config.q_heads {
            let kv_head = head / group_size;
            let q_start = row * q_width + head * config.head_dim;
            let kv_offset = kv_head * config.head_dim;
            for (position, score) in scores.iter_mut().enumerate() {
                let key = &k[position * kv_width + kv_offset
                    ..position * kv_width + kv_offset + config.head_dim];
                *score = if position <= row {
                    dot(&q[q_start..q_start + config.head_dim], key) * scale
                } else {
                    f32::NEG_INFINITY
                };
            }
            if row < 512 {
                let inverse_sum = flash_softmax(&mut scores);
                for dimension in 0..config.head_dim {
                    let mut sum = 0.0f32;
                    for position in 0..rows {
                        sum += scores[position] * v[position * kv_width + kv_offset + dimension];
                    }
                    output[q_start + dimension] = bf16(sum * inverse_sum);
                }
                continue;
            }

            let result = &mut output[q_start..q_start + config.head_dim];
            result.fill(0.0);
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            for start in (0..=row).step_by(512) {
                let end = (start + 512).min(rows);
                let block = &mut scores[start..end];
                let block_max = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let next_max = running_max.max(block_max);
                let block_sum = flash_exp_sum(block, next_max);
                let rescale = (running_max - next_max).exp();
                running_sum = rescale.mul_add(running_sum, block_sum);
                if start > 0 {
                    for value in result.iter_mut() {
                        *value *= rescale;
                    }
                }
                for (dimension, value) in result.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for offset in 0..block.len() {
                        sum += block[offset] * v[(start + offset) * kv_width + kv_offset + dimension];
                    }
                    *value += sum;
                }
                running_max = next_max;
            }
            let inverse_sum = running_sum.recip();
            for value in result.iter_mut() {
                *value = bf16(*value * inverse_sum);
            }
        }
    }
}

fn flash_softmax(values: &mut [f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    flash_exp_sum(values, max).recip()
}

fn flash_exp_sum(values: &mut [f32], max: f32) -> f32 {
    let vector_end = values.len() / 4 * 4;
    let mut partial = [0.0f32; 4];
    for base in (0..vector_end).step_by(4) {
        let shifted = [
            values[base] - max,
            values[base + 1] - max,
            values[base + 2] - max,
            values[base + 3] - max,
        ];
        let special = shifted
            .iter()
            .any(|value| value.abs() > f32::from_bits(0x42ae_af15));
        for lane in 0..4 {
            let value = if special {
                flash_sleef_exp(shifted[lane])
            } else {
                flash_exp_u20(shifted[lane])
            };
            partial[lane] += value;
            values[base + lane] = bf16(value);
        }
    }
    let mut sum = (partial[0] + partial[2]) + (partial[1] + partial[3]);
    for value in &mut values[vector_end..] {
        let exponential = (*value - max).exp();
        sum += exponential;
        *value = bf16(exponential);
    }
    sum
}

// PyTorch's ARM vector fallback uses SLEEF exp when any lane is out of range.
fn flash_sleef_exp(value: f32) -> f32 {
    if value == f32::NEG_INFINITY {
        return 0.0;
    }
    if value < -104.0 {
        return 0.0;
    }
    let q = (value * 1.442_695_f32).round_ties_even();
    let s = q.mul_add(-0.693_145_75_f32, value);
    let s = q.mul_add(-1.428_606_8e-6_f32, s);
    let mut u = 0.000_198_527_62_f32;
    for coefficient in [
        0.001_393_043_6_f32,
        0.008_333_361_f32,
        0.041_666_485_f32,
        0.166_666_67_f32,
        0.5_f32,
    ] {
        u = u.mul_add(s, coefficient);
    }
    let u = (s * s).mul_add(u, s) + 1.0;
    let exponent = q as i32;
    let half = exponent >> 1;
    let pow2 = |power: i32| f32::from_bits(((power + 127) as u32) << 23);
    (u * pow2(half)) * pow2(exponent - half)
}

// PyTorch 2.10's ARM CPU FlashAttention uses the vectorized u20 exp before BF16 rounding.
fn flash_exp_u20(value: f32) -> f32 {
    let n = (value * f32::from_bits(0x3fb8_aa3b)).round();
    let r = (-n).mul_add(f32::from_bits(0x3f31_7200), value);
    let r = (-n).mul_add(f32::from_bits(0x35bf_be8e), r);
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    let r2 = r * r;
    let p = r.mul_add(f32::from_bits(0x3c07_2010), f32::from_bits(0x3d2b_9f17));
    let q = r.mul_add(f32::from_bits(0x3e2a_af33), f32::from_bits(0x3eff_fedb));
    let q = p.mul_add(r2, q);
    let p = f32::from_bits(0x3f7f_fff6) * r;
    let poly = q.mul_add(r2, p);
    poly.mul_add(scale, scale)
}

#[cfg(test)]
mod flash_tests {
    use super::{flash_exp_sum, flash_softmax, hybrid_attention};

    #[test]
    fn bf16_midpoint_matches_cpu_flash_exponential() {
        let mut scores = [0.0, 0.0, 0.0, f32::from_bits(0xc03f_d218)];
        flash_softmax(&mut scores);
        assert_eq!(scores[3].to_bits(), 0x3d4c_0000);
    }

    #[test]
    fn masked_vector_fallback_matches_sleef_bits() {
        let mut scores = [f32::from_bits(0xbff9_8d48), 0.0, f32::NEG_INFINITY, 0.0];
        flash_softmax(&mut scores);
        assert_eq!(scores[0].to_bits(), 0x3e12_0000);
    }

    #[test]
    fn unmasked_large_negative_uses_finite_sleef_exponential() {
        let mut midpoint = [0.0, -88.377_235, 0.0, 0.0];
        flash_exp_sum(&mut midpoint, 0.0);
        assert_eq!(midpoint[1].to_bits(), 0x002d_0000);
        for bits in (0x42ae_af15..=0x42d00000).step_by(257) {
            let value = -f32::from_bits(bits);
            let mut scores = [0.0, value, 0.0, 0.0];
            flash_exp_sum(&mut scores, 0.0);
            assert!(scores[1].is_finite(), "value {value} bits 0x{bits:08x}");
            assert!(scores[1] >= 0.0, "value {value} bits 0x{bits:08x}");
        }
    }

    #[test]
    #[ignore = "requires the pinned real E2E Oracle"]
    fn hybrid_attention_2034_keys_matches_cpu_flash() {
        let root = std::env::var_os("YUE2_E2E_ORACLE_TRACE")
            .map(std::path::PathBuf::from)
            .expect("YUE2_E2E_ORACLE_TRACE")
            .parent()
            .unwrap()
            .to_owned();
        let read = |name: &str| {
            std::fs::read(root.join(format!("{name}.f32")))
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let config = super::super::YuE2Config {
            hidden: 2048,
            layers: 28,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            ffn: 6144,
            vocab: 156_288,
            context: 24_576,
            rms_eps: 1e-6,
            rope_base: 1_000_000.0,
            latent_channels: 64,
            timestep_shift: 1.0,
        };
        for (layer, occurrence) in [(0, 0), (10, 66)] {
            let q = read(&format!("yue2.nar.q.{occurrence}"));
            let prefix = (
                read(&format!("yue2.nar.prefix_k.{layer}")),
                read(&format!("yue2.nar.prefix_v.{layer}")),
            );
            let k = read(&format!("yue2.nar.k.{occurrence}"));
            let v = read(&format!("yue2.nar.v.{occurrence}"));
            let expected = read(&format!("yue2.nar.attn.{occurrence}"));
            let mut output = vec![0.0; q.len()];
            hybrid_attention(&config, &q, &prefix, &k, &v, &mut output);
            for (index, (actual, expected)) in output.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "layer {layer} element {index}"
                );
            }
        }
    }
}

impl LinearScratch {
    fn new(max_input: usize) -> Self {
        Self {
            q8: vec![0; max_input],
            scales: vec![0.0; max_input.div_ceil(32)],
            q8k: vec![
                BlockQ8K {
                    d: 0.0,
                    qs: [0; 256],
                    bsums: [0; 16],
                };
                max_input.div_ceil(256)
            ],
        }
    }
}

fn normalize_rows(input: &[f32], weight: &[f32], output: &mut [f32], width: usize, eps: f32) {
    for (input, output) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        rms_norm(input, weight, output, eps);
    }
}

#[allow(clippy::too_many_arguments)]
fn project_qkv_rows(
    model: &YuE2Model,
    weights: &YuE2AttentionWeights,
    input: &[f32],
    position_start: usize,
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    scratch: &mut LinearScratch,
) {
    let config = model.config();
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    for (row, input) in input.chunks_exact(config.hidden).enumerate() {
        let q_row = &mut q[row * q_width..(row + 1) * q_width];
        let k_row = &mut k[row * kv_width..(row + 1) * kv_width];
        let v_row = &mut v[row * kv_width..(row + 1) * kv_width];
        weights.q.matmul(
            input,
            q_row,
            &model.pool,
            &mut scratch.q8,
            &mut scratch.scales,
            &mut scratch.q8k,
        );
        weights.k.matmul(
            input,
            k_row,
            &model.pool,
            &mut scratch.q8,
            &mut scratch.scales,
            &mut scratch.q8k,
        );
        weights.v.matmul(
            input,
            v_row,
            &model.pool,
            &mut scratch.q8,
            &mut scratch.scales,
            &mut scratch.q8k,
        );
        rms_norm_heads(q_row, &weights.q_norm, config.head_dim, config.rms_eps);
        rms_norm_heads(k_row, &weights.k_norm, config.head_dim, config.rms_eps);
        rope(
            q_row,
            position_start + row,
            config.head_dim,
            config.rope_base,
        );
        rope(
            k_row,
            position_start + row,
            config.head_dim,
            config.rope_base,
        );
    }
}

fn hybrid_attention(
    config: &super::YuE2Config,
    q: &[f32],
    prefix: &(Vec<f32>, Vec<f32>),
    nar_k: &[f32],
    nar_v: &[f32],
    output: &mut [f32],
) {
    let q_width = config.q_heads * config.head_dim;
    let kv_width = config.kv_heads * config.head_dim;
    let prefix_len = prefix.0.len() / kv_width;
    let nar_len = nar_k.len() / kv_width;
    let total_len = prefix_len + nar_len;
    let group_size = config.q_heads / config.kv_heads;
    let scale = (config.head_dim as f32).sqrt().recip();
    let mut scores = vec![0.0; total_len];
    for row in 0..nar_len {
        for head in 0..config.q_heads {
            let kv_head = head / group_size;
            let q_start = row * q_width + head * config.head_dim;
            let kv_offset = kv_head * config.head_dim;
            for (position, score) in scores.iter_mut().enumerate() {
                let key = if position < prefix_len {
                    &prefix.0[position * kv_width + kv_offset
                        ..position * kv_width + kv_offset + config.head_dim]
                } else {
                    let position = position - prefix_len;
                    &nar_k[position * kv_width + kv_offset
                        ..position * kv_width + kv_offset + config.head_dim]
                };
                *score = dot(&q[q_start..q_start + config.head_dim], key) * scale;
            }
            let value_at = |position: usize, dimension: usize| {
                if position < prefix_len {
                    prefix.1[position * kv_width + kv_offset + dimension]
                } else {
                    nar_v[(position - prefix_len) * kv_width + kv_offset + dimension]
                }
            };
            let result = &mut output[q_start..q_start + config.head_dim];
            if total_len <= 512 {
                let inverse_sum = softmax(&mut scores);
                for (dimension, value) in result.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for position in 0..total_len {
                        sum += scores[position] * value_at(position, dimension);
                    }
                    *value = bf16(sum * inverse_sum);
                }
                continue;
            }

            result.fill(0.0);
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            for start in (0..total_len).step_by(512) {
                let end = (start + 512).min(total_len);
                let block = &mut scores[start..end];
                let block_max = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let next_max = running_max.max(block_max);
                let block_sum = flash_exp_sum(block, next_max);
                let rescale = (running_max - next_max).exp();
                running_sum = rescale.mul_add(running_sum, block_sum);
                if start > 0 {
                    for value in result.iter_mut() {
                        *value *= rescale;
                    }
                }
                for (dimension, value) in result.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for offset in 0..block.len() {
                        sum += block[offset] * value_at(start + offset, dimension);
                    }
                    *value += sum;
                }
                running_max = next_max;
            }
            let inverse_sum = running_sum.recip();
            for value in result.iter_mut() {
                *value = bf16(*value * inverse_sum);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn linear_rows(
    weight: &YuE2Weight,
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    model: &YuE2Model,
    scratch: &mut LinearScratch,
) {
    for (input, output) in input.chunks_exact(n_in).zip(output.chunks_exact_mut(n_out)) {
        weight.matmul(
            input,
            output,
            &model.pool,
            &mut scratch.q8,
            &mut scratch.scales,
            &mut scratch.q8k,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn linear_bias_rows(
    weight: &YuE2Weight,
    bias: &[f32],
    input: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    model: &YuE2Model,
    scratch: &mut LinearScratch,
) {
    for (input, output) in input.chunks_exact(n_in).zip(output.chunks_exact_mut(n_out)) {
        weight.matmul_bias(
            input,
            bias,
            output,
            &model.pool,
            &mut scratch.q8,
            &mut scratch.scales,
            &mut scratch.q8k,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn mlp_rows(
    model: &YuE2Model,
    weights: &YuE2MlpWeights,
    input: &[f32],
    gate: &mut [f32],
    up: &mut [f32],
    down: &mut [f32],
    scratch: &mut LinearScratch,
) {
    let config = model.config();
    linear_rows(
        &weights.gate,
        input,
        gate,
        config.hidden,
        config.ffn,
        model,
        scratch,
    );
    linear_rows(
        &weights.up,
        input,
        up,
        config.hidden,
        config.ffn,
        model,
        scratch,
    );
    for (gate, &up) in gate.iter_mut().zip(up.iter()) {
        *gate = bf16(bf16(silu(*gate)) * up);
    }
    linear_rows(
        &weights.down,
        gate,
        down,
        config.ffn,
        config.hidden,
        model,
        scratch,
    );
}

fn shifted_time(raw_t: f64, shift: f32) -> f32 {
    let raw = bf16(raw_t as f32);
    let sigmoid = bf16(1.0 / (1.0 + (-raw).exp()));
    bf16((shift * sigmoid) / (1.0 + (shift - 1.0) * sigmoid))
}

fn time_embedding(model: &YuE2Model, shifted: f32, scratch: &mut LinearScratch) -> Vec<f32> {
    let half = 128;
    let mut frequency = vec![0.0; 256];
    let log_base = 10_000.0f32.ln();
    for index in 0..half {
        let value = (-log_base * index as f32 / half as f32).exp();
        let angle = shifted * value;
        frequency[index] = bf16(angle.cos());
        frequency[index + half] = bf16(angle.sin());
    }
    let mut hidden = vec![0.0; model.config().hidden];
    model.aux.time_in.matmul_bias(
        &frequency,
        &model.aux.time_in_bias,
        &mut hidden,
        &model.pool,
        &mut scratch.q8,
        &mut scratch.scales,
        &mut scratch.q8k,
    );
    for value in &mut hidden {
        *value = bf16(silu(*value));
    }
    let mut output = vec![0.0; model.config().hidden];
    model.aux.time_out.matmul_bias(
        &hidden,
        &model.aux.time_out_bias,
        &mut output,
        &model.pool,
        &mut scratch.q8,
        &mut scratch.scales,
        &mut scratch.q8k,
    );
    output
}

#[inline]
fn bf16(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

fn trace(name: &str, layer: Option<usize>, step: Option<usize>, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint_at(
        name, layer, step, shape, values,
    ));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, step, shape, values);
}

fn trace_tokens(name: &str, values: &[u32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::token_ids(name, values));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, values);
}

fn raw_time(time: f64) -> f64 {
    (time / (1.0 - time)).ln().clamp(-20.0, 20.0)
}

pub(super) fn torch_randn(seed: u64, count: usize) -> Vec<f32> {
    let mut generator = Mt19937::new(seed);
    debug_assert!(count >= 16);
    let mut values = (0..count)
        .map(|_| uniform(&mut generator))
        .collect::<Vec<_>>();
    for start in (0..=count - 16).step_by(16) {
        normal_fill_16(&mut values[start..start + 16]);
    }
    if count % 16 != 0 {
        let start = count - 16;
        for value in &mut values[start..] {
            *value = uniform(&mut generator);
        }
        normal_fill_16(&mut values[start..]);
    }
    values
}

fn uniform(generator: &mut Mt19937) -> f32 {
    (generator.next() & 0x00ff_ffff) as f32 * (1.0 / 16_777_216.0)
}

fn normal_fill_16(values: &mut [f32]) {
    debug_assert_eq!(values.len(), 16);
    for pair in 0..8 {
        let radius = (-2.0 * (1.0 - values[pair]).ln()).sqrt();
        let theta = (2.0 * std::f64::consts::PI * f64::from(values[pair + 8])) as f32;
        let (sine, cosine) = torch_sin_cos(theta);
        values[pair] = radius * cosine;
        values[pair + 8] = radius * sine;
    }
}

#[cfg(target_vendor = "apple")]
fn torch_sin_cos(value: f32) -> (f32, f32) {
    #[repr(C)]
    struct SinCos {
        sine: f32,
        cosine: f32,
    }
    unsafe extern "C" {
        #[link_name = "__sincosf_stret"]
        fn apple_sin_cos(value: f32) -> SinCos;
    }
    let result = unsafe { apple_sin_cos(value) };
    (result.sine, result.cosine)
}

#[cfg(not(target_vendor = "apple"))]
fn torch_sin_cos(value: f32) -> (f32, f32) {
    value.sin_cos()
}

pub(super) struct Mt19937 {
    state: [u32; 624],
    left: usize,
    next: usize,
}

impl Mt19937 {
    pub(super) fn new(seed: u64) -> Self {
        let mut state = [0; 624];
        state[0] = seed as u32;
        for index in 1..state.len() {
            state[index] = 1_812_433_253u32
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        Self {
            state,
            left: 1,
            next: 0,
        }
    }

    pub(super) fn next(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.twist();
        }
        let mut value = self.state[self.next];
        self.next += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    fn twist(&mut self) {
        const M: usize = 397;
        const N: usize = 624;
        const MATRIX_A: u32 = 0x9908_b0df;
        self.left = N;
        self.next = 0;
        for index in 0..N - M {
            let mixed = (self.state[index] & 0x8000_0000) | (self.state[index + 1] & 0x7fff_ffff);
            self.state[index] =
                self.state[index + M] ^ (mixed >> 1) ^ if mixed & 1 != 0 { MATRIX_A } else { 0 };
        }
        for index in N - M..N - 1 {
            let mixed = (self.state[index] & 0x8000_0000) | (self.state[index + 1] & 0x7fff_ffff);
            self.state[index] = self.state[index + M - N]
                ^ (mixed >> 1)
                ^ if mixed & 1 != 0 { MATRIX_A } else { 0 };
        }
        let mixed = (self.state[N - 1] & 0x8000_0000) | (self.state[0] & 0x7fff_ffff);
        self.state[N - 1] =
            self.state[M - 1] ^ (mixed >> 1) ^ if mixed & 1 != 0 { MATRIX_A } else { 0 };
    }
}
