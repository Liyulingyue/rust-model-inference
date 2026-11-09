//! Noncausal decision attention; the optional thinker uses causal attention.

use crate::core::thread_pool::ComputePool;
use crate::models::qwen35::trunk::{HybridTrunk, Qwen35Config, Qwen35Scratchpad};
use crate::ops::kernel::{PreparedRows, Weight};
use rayon::prelude::*;

pub(super) fn trace(name: &str, layer: Option<usize>, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    if crate::parity_trace::enabled(name) {
        crate::parity_trace::report(crate::parity_trace::checkpoint_rows(
            name, layer, shape, values,
        ));
    }
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, shape, values);
}

pub(super) fn linear(
    weight: &Weight<'_>,
    input: &[f32],
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    let rows = input.len() / weight.n_in;
    let mut prepared = PreparedRows::new(rows, weight.n_in);
    prepared.prepare(
        input,
        rows,
        weight.n_in,
        weight.needs_q8_0_activation(),
        weight.uses_q8_k(),
    )?;
    let mut output = vec![0.; rows * weight.n_out];
    prepared.matmul(weight, input, &mut output, pool)?;
    Ok(output)
}

// K/V are token-major; query heads sharing a key are adjacent in the HF checkpoint.
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    kv_heads: usize,
    dim: usize,
    past: usize,
    causal: bool,
    output: &mut [f32],
) {
    let width = heads * dim;
    let kv_width = kv_heads * dim;
    let keys = k.len() / kv_width;
    #[cfg(feature = "vulkan")]
    if let Some(values) = crate::ops::kernel::vulkan::try_attention(
        q,
        k,
        v,
        crate::ops::nn::AttentionSpec {
            query_tokens: q.len() / width,
            key_tokens: keys,
            query_heads: heads,
            key_value_heads: kv_heads,
            head_dim: dim,
            causal,
            scale: 1. / (dim as f32).sqrt(),
        },
    ) {
        output.copy_from_slice(&values);
        return;
    }
    output
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(index, out)| {
            let row = index / heads;
            let head = index % heads;
            let kv = head / (heads / kv_heads);
            let query = &q[row * width + head * dim..][..dim];
            let end = if causal { past + row + 1 } else { keys };
            let mut scores = (0..end)
                .map(|key| {
                    crate::ops::dot_f32(query, &k[key * kv_width + kv * dim..][..dim], dim)
                        / (dim as f32).sqrt()
                })
                .collect::<Vec<_>>();
            crate::ops::softmax_inplace(&mut scores);
            out.fill(0.);
            for (key, p) in scores.into_iter().enumerate() {
                crate::ops::vec_scaled_add_inplace(p, &v[key * kv_width + kv * dim..][..dim], out);
            }
        });
}

pub(super) struct Session {
    scratch: Qwen35Scratchpad,
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    causal: bool,
    tokens: usize,
}

impl Session {
    pub fn new(config: &Qwen35Config, max_rows: usize, causal: bool) -> Self {
        Self {
            scratch: Qwen35Scratchpad::new(config, max_rows),
            kv: (0..config.n_layer_impl())
                .map(|_| (Vec::new(), Vec::new()))
                .collect(),
            causal,
            tokens: 0,
        }
    }

    pub fn run(
        &mut self,
        model: &HybridTrunk<'_>,
        mut x: Vec<f32>,
        positions: &[[usize; 4]],
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        let c = &model.config;
        let rows = positions.len();
        let width = c.n_embd;
        if rows == 0 || x.len() != rows * width || self.tokens + rows > c.n_ctx {
            return Err("THX-02 invalid forward dimensions or context exceeded".into());
        }
        trace("thx02.embedding", None, &[rows, width], &x);
        for (i, layer) in model.layers.iter().enumerate() {
            let mut norm = x.clone();
            for row in norm.chunks_exact_mut(width) {
                crate::ops::rms_norm_inplace(row, &layer.attn_norm, c.norm_eps);
            }
            let mixed = if c.is_recurrent[i] {
                model.forward_recurrent_layer(
                    i,
                    &norm,
                    rows,
                    &mut self.scratch,
                    pool,
                    false,
                    #[cfg(feature = "parity-trace")]
                    false,
                )?
            } else {
                let dim = c.n_embd_head();
                let qw = dim * c.n_head;
                let kw = dim * c.n_head_kv;
                let qg = linear(layer.wq.as_ref().unwrap(), &norm, pool)?;
                let mut q = vec![0.; rows * qw];
                let mut gate = vec![0.; rows * qw];
                let mut k = linear(layer.wk.as_ref().unwrap(), &norm, pool)?;
                let v = linear(layer.wv.as_ref().unwrap(), &norm, pool)?;
                for t in 0..rows {
                    for h in 0..c.n_head {
                        let off = t * qw + h * dim;
                        let src = t * qw * 2 + h * dim * 2;
                        q[off..off + dim].copy_from_slice(&qg[src..src + dim]);
                        gate[off..off + dim].copy_from_slice(&qg[src + dim..src + dim * 2]);
                        crate::ops::rms_norm_inplace(
                            &mut q[off..off + dim],
                            layer.attn_q_norm.as_ref().unwrap(),
                            c.norm_eps,
                        );
                    }
                    for head in k[t * kw..(t + 1) * kw].chunks_exact_mut(dim) {
                        crate::ops::rms_norm_inplace(
                            head,
                            layer.attn_k_norm.as_ref().unwrap(),
                            c.norm_eps,
                        );
                    }
                    for values in [&mut q[t * qw..(t + 1) * qw], &mut k[t * kw..(t + 1) * kw]] {
                        crate::ops::rope_mrope_interleaved(
                            values,
                            positions[t],
                            c.rope_dimension_sections,
                            dim,
                            c.rope_freq_base,
                            c.rope_dimension_count,
                        );
                    }
                }
                let (keys, values) = &mut self.kv[i];
                keys.extend_from_slice(&k);
                values.extend_from_slice(&v);
                let mut attended = vec![0.; rows * qw];
                attention(
                    &q,
                    keys,
                    values,
                    c.n_head,
                    c.n_head_kv,
                    dim,
                    self.tokens,
                    self.causal,
                    &mut attended,
                );
                crate::ops::sigmoid_inplace(&mut gate);
                crate::ops::vec_mul_inplace(&gate, &mut attended);
                linear(layer.wo.as_ref().unwrap(), &attended, pool)?
            };
            crate::ops::vec_add_into(&mixed, &mut x);
            norm.copy_from_slice(&x);
            for row in norm.chunks_exact_mut(width) {
                crate::ops::rms_norm_inplace(row, &layer.attn_post_norm, c.norm_eps);
            }
            model.forward_ffn_parallel(layer, &norm, rows, &mut self.scratch, pool, false);
            crate::ops::vec_add_into(&self.scratch.buf[..x.len()], &mut x);
            trace("thx02.layer", Some(i), &[rows, width], &x);
        }
        for row in x.chunks_exact_mut(width) {
            crate::ops::rms_norm_inplace(row, &model.output_norm, c.norm_eps);
        }
        trace("thx02.hidden", None, &[rows, width], &x);
        self.tokens += rows;
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_readout_attends_future_but_thinker_does_not() {
        let q = [1., 1.];
        let k = [0., 0.];
        let v = [2., 6.];
        let mut out = [0.; 2];
        attention(&q, &k, &v, 1, 1, 1, 0, false, &mut out);
        assert_eq!(out, [4., 4.]);
        attention(&q, &k, &v, 1, 1, 1, 0, true, &mut out);
        assert_eq!(out, [2., 4.]);
        attention(&[1.], &k, &v, 1, 1, 1, 1, true, &mut out[..1]);
        assert_eq!(out[0], 4.);
    }

    #[test]
    fn grouped_attention_uses_adjacent_query_heads() {
        let mut out = [0.; 4];
        attention(&[0.; 4], &[0.; 2], &[3., 7.], 4, 2, 1, 0, false, &mut out);
        assert_eq!(out, [3., 3., 7., 7.]);
    }
}
