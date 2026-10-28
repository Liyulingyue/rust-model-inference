use super::{check_tensor, trace};
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::diffusion::dreamx::kernels::{conv2d, Linear};
use crate::models::qwen3::asr::audio_processor::{compute_whisper_log_mel, HOP};
use crate::models::qwen3::asr::mel_encoder::{
    apply_gelu_erf, full_attention_into, layer_norm_rows, LayerNormWeights,
};
use std::sync::Arc;

struct Conv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
}
struct Block {
    ln1: LayerNormWeights,
    ln2: LayerNormWeights,
    q: Linear<'static>,
    k: Linear<'static>,
    v: Linear<'static>,
    out: Linear<'static>,
    up: Linear<'static>,
    down: Linear<'static>,
}

pub struct HyAudioEncoder {
    _source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    conv: Vec<Conv>,
    conv_out: Linear<'static>,
    positions: Vec<f32>,
    blocks: Vec<Block>,
    post_ln: LayerNormWeights,
    proj1: Linear<'static>,
    proj2: Linear<'static>,
    mlp0: Linear<'static>,
    mlp3: Linear<'static>,
    output_norm: LayerNormWeights,
    eps: f32,
}

impl HyAudioEncoder {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let s = source.as_ref();
        if s.metadata("clip.projector_type")
            .and_then(|v| v.to_string_val())
            != Some("hyv3_speech")
        {
            return Err("Hy-SimulMT requires the bundled hyv3_speech audio tower".into());
        }
        for (key, expected) in [
            ("num_mel_bins", 128),
            ("embedding_length", 1024),
            ("feed_forward_length", 4096),
            ("block_count", 24),
            ("projection_dim", 2048),
            ("attention.head_count", 16),
            ("projector.stack_factor", 2),
            ("projector.downsample_rate", 2),
        ] {
            if s.metadata(&format!("clip.audio.{key}"))
                .and_then(|v| v.to_u64())
                != Some(expected)
            {
                return Err(format!(
                    "Unsupported Hy-SimulMT clip.audio.{key}; expected {expected}"
                ));
            }
        }
        let eps = s
            .metadata("clip.audio.attention.layer_norm_epsilon")
            .and_then(|v| v.to_f64())
            .ok_or("missing audio epsilon")? as f32;
        if !eps.is_finite() || eps <= 0.0 {
            return Err("invalid audio epsilon".into());
        }
        // The owning Arc stays in the encoder, as in the decoder's mmap weights.
        let s: &'static dyn TensorSource = unsafe { std::mem::transmute(s) };
        let linear = |name: &str, i: usize, o: usize, bias: bool| {
            check_tensor(s, &format!("{name}.weight"), &[i as u64, o as u64], true)?;
            Linear::from_source(
                s,
                &format!("{name}.weight"),
                bias.then(|| format!("{name}.bias")).as_deref(),
                i,
                o,
            )
        };
        let norm = |name: &str, h| LayerNormWeights::load(s, name, &[h]);
        let mut conv = Vec::new();
        for (idx, input) in [(1, 1), (2, 480), (3, 480)] {
            let p = format!("a.conv2d.{idx}");
            conv.push(Conv {
                weight: load_f32_tensor(s, &format!("{p}.weight"), &[3, 3, input as u64, 480])?,
                bias: load_f32_tensor(s, &format!("{p}.bias"), &[480])?,
                input,
            });
        }
        let mut blocks = Vec::new();
        for l in 0..24 {
            let p = format!("a.blk.{l}");
            blocks.push(Block {
                ln1: norm(&format!("{p}.ln1"), 1024)?,
                ln2: norm(&format!("{p}.ln2"), 1024)?,
                q: linear(&format!("{p}.attn_q"), 1024, 1024, true)?,
                k: linear(&format!("{p}.attn_k"), 1024, 1024, true)?,
                v: linear(&format!("{p}.attn_v"), 1024, 1024, true)?,
                out: linear(&format!("{p}.attn_out"), 1024, 1024, true)?,
                up: linear(&format!("{p}.ffn_up"), 1024, 4096, true)?,
                down: linear(&format!("{p}.ffn_down"), 4096, 1024, true)?,
            });
        }
        Ok(Self {
            conv,
            blocks,
            pool,
            conv_out: linear("a.conv_out", 7680, 1024, false)?,
            positions: load_f32_tensor(s, "a.sinusoid_pos_embd", &[1024, 1500])?,
            post_ln: norm("a.post_ln", 1024)?,
            proj1: linear("a.proj1", 1024, 1024, true)?,
            proj2: linear("a.proj2", 1024, 2048, true)?,
            mlp0: linear("mm.a.mlp.0", 4096, 2048, true)?,
            mlp3: linear("mm.a.mlp.3", 2048, 2048, true)?,
            output_norm: norm("mm.a.output_norm", 2048)?,
            eps,
            _source: source,
        })
    }

    /// One protocol turn: at most 1.6 seconds, mono PCM at 16 kHz.
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<f32>, String> {
        let _cpu = crate::compute::ComputePolicy::Cpu.cpu_scope();
        if samples.is_empty() || samples.len() > 25_600 || samples.iter().any(|v| !v.is_finite()) {
            return Err("Hy-SimulMT audio turn must contain 1..=25600 finite samples".into());
        }
        let mut padded = samples.to_vec();
        padded.resize(samples.len().div_ceil(HOP) * HOP, 0.0);
        let mel = compute_whisper_log_mel(&padded).map_err(|e| format!("Hy-SimulMT mel: {e:?}"))?;
        trace("hy.mel", &[128, mel.frames], &mel.normalized);
        self.encode_mel(&mel.normalized, mel.frames)
    }

    fn encode_mel(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>, String> {
        if frames == 0
            || frames > 160
            || mel.len() != 128 * frames
            || mel.iter().any(|v| !v.is_finite())
        {
            return Err("invalid Hy-SimulMT mel dimensions or values".into());
        }
        let rows = encoder_rows(frames);
        let mut hidden = Vec::with_capacity(rows * 1024);
        for start in (0..frames).step_by(100) {
            let valid = (frames - start).min(100);
            let mut x = vec![0.0; 128 * 100];
            for m in 0..128 {
                x[m * 100..m * 100 + valid]
                    .copy_from_slice(&mel[m * frames + start..m * frames + start + valid]);
            }
            let mut shape = [1, 128, 100];
            for conv in &self.conv {
                (x, shape) = conv2d(
                    &self.pool,
                    &x,
                    shape,
                    &conv.weight,
                    [480, conv.input, 3, 3],
                    Some(&conv.bias),
                    [2, 2],
                    [1, 1],
                    [1, 1],
                    1,
                )?;
                apply_gelu_erf(&mut x)?;
            }
            let flattened = flatten_conv(&x, shape);
            let projected = self.conv_out.forward(&self.pool, &flattened, shape[2])?;
            let keep = valid.div_ceil(8);
            hidden.extend(
                projected[..keep * 1024]
                    .iter()
                    .zip(&self.positions)
                    .map(|(v, p)| v + p),
            );
        }
        trace("hy.enc_inp", &[rows, 1024], &hidden);
        let mut normed = Vec::new();
        let mut scores = Vec::new();
        let mut attention = Vec::new();
        for block in &self.blocks {
            layer_norm_rows(&hidden, rows, &block.ln1, self.eps, &mut normed)?;
            let q = block.q.forward(&self.pool, &normed, rows)?;
            let k = block.k.forward(&self.pool, &normed, rows)?;
            let v = block.v.forward(&self.pool, &normed, rows)?;
            full_attention_into(&q, &k, &v, rows, 16, 64, &mut scores, &mut attention)?;
            let a = block.out.forward(&self.pool, &attention, rows)?;
            for (x, a) in hidden.iter_mut().zip(a) {
                *x += a;
            }
            layer_norm_rows(&hidden, rows, &block.ln2, self.eps, &mut normed)?;
            let mut f = block.up.forward(&self.pool, &normed, rows)?;
            apply_gelu_erf(&mut f)?;
            let f = block.down.forward(&self.pool, &f, rows)?;
            for (x, f) in hidden.iter_mut().zip(f) {
                *x += f;
            }
        }
        trace("hy.after_transformer", &[rows, 1024], &hidden);
        layer_norm_rows(&hidden, rows, &self.post_ln, self.eps, &mut normed)?;
        let mut x = self.proj1.forward(&self.pool, &normed, rows)?;
        apply_gelu_erf(&mut x)?;
        let mut x = self.proj2.forward(&self.pool, &x, rows)?;
        trace("hy.after_proj2", &[rows, 2048], &x);
        let tokens = rows.div_ceil(2);
        x.resize(tokens * 4096, 0.0);
        trace("hy.stacked", &[tokens, 4096], &x);
        let mut x = self.mlp0.forward(&self.pool, &x, tokens)?;
        crate::ops::activation::silu_inplace(&mut x);
        let x = self.mlp3.forward(&self.pool, &x, tokens)?;
        layer_norm_rows(&x, tokens, &self.output_norm, self.eps, &mut normed)?;
        if normed.iter().any(|v| !v.is_finite()) {
            return Err("non-finite Hy-SimulMT audio embeddings".into());
        }
        trace("hy.projected", &[tokens, 2048], &normed);
        Ok(normed)
    }
}

fn encoder_rows(frames: usize) -> usize {
    (frames / 100) * 13 + (frames % 100).div_ceil(8)
}
fn flatten_conv(x: &[f32], [channels, mel, frames]: [usize; 3]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for t in 0..frames {
        for c in 0..channels {
            for m in 0..mel {
                out[t * channels * mel + c * mel + m] = x[(c * mel + m) * frames + t];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hy_chunk_grid_and_feature_layout() {
        assert_eq!(
            [1, 8, 9, 99, 100, 101, 160].map(encoder_rows),
            [1, 1, 2, 13, 13, 14, 21]
        );
        assert_eq!(
            flatten_conv(&(0..12).map(|x| x as f32).collect::<Vec<_>>(), [2, 2, 3]),
            [0., 3., 6., 9., 1., 4., 7., 10., 2., 5., 8., 11.]
        );
    }
}
