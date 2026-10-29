use super::weights::{HyDecoder, HyFfn};
use super::HyConfig;
use crate::compute::linear::{LinearBinding, LinearExecutor, LinearMode};
use crate::compute::ComputePolicy;
use crate::core::scratchpad::{KvArch, KvCache, KvFormat, KvState};
use crate::ops::kernel::Weight;
use crate::ops::{self, rms_norm, rms_norm_inplace};
use std::sync::Arc;

const ROWS: usize = 128;

pub struct HySession<'model> {
    model: &'model HyDecoder,
    kv: KvState,
    compute: ComputePolicy,
    linear: LinearExecutor<'model, 'static>,
    experts: Vec<Option<LinearExecutor<'model, 'static>>>,
}

impl<'model> HySession<'model> {
    pub fn new(
        model: &'model HyDecoder,
        capacity: usize,
        format: KvFormat,
        compute: ComputePolicy,
    ) -> Result<Self, String> {
        compute.check_build()?;
        if compute == ComputePolicy::Vulkan {
            return Err(
                "Hy-SimulMT uses hybrid linear offload; choose --compute auto or cpu".into(),
            );
        }
        let c = &model.config;
        if capacity == 0 || capacity > c.n_ctx {
            return Err(format!(
                "Hy session capacity must be within 1..={}",
                c.n_ctx
            ));
        }
        let _scope = compute.enter_legacy_scope();
        let values = c
            .n_layer
            .checked_mul(capacity)
            .and_then(|v| v.checked_mul(c.n_head_kv))
            .and_then(|v| v.checked_mul(c.n_embd_head_k))
            .ok_or("Hy KV allocation overflow")?;
        let bytes = values
            .checked_mul(if format == KvFormat::F32 { 8 } else { 4 })
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or("Hy KV byte size overflow")?;
        let _ = bytes;
        let mut bindings = Vec::new();
        let mut experts = Vec::new();
        for layer in &model.layers {
            for weight in [&layer.q, &layer.k, &layer.v, &layer.out] {
                bindings.push(LinearBinding {
                    weight,
                    mode: LinearMode::Prepared,
                });
            }
            experts.push(match &layer.ffn {
                HyFfn::Dense(e) => {
                    for weight in [&e.gate, &e.up, &e.down] {
                        bindings.push(LinearBinding {
                            weight,
                            mode: LinearMode::Prepared,
                        });
                    }
                    None
                }
                HyFfn::Moe(moe) => Some(
                    LinearExecutor::new(compute, moe.bindings(), 1, model.pool())
                        .map_err(|e| e.to_string())?,
                ),
            });
        }
        bindings.push(LinearBinding {
            weight: &model.output,
            mode: LinearMode::Prepared,
        });
        let linear = LinearExecutor::new(compute, bindings, ROWS, model.pool())
            .map_err(|e| e.to_string())?;
        let arch = Arc::new(KvArch::new(
            c.n_layer,
            c.n_head_kv,
            c.n_embd_head_k,
            c.n_embd_head_v,
            c.n_ctx,
        ));
        Ok(Self {
            model,
            kv: KvState::new(arch, format, capacity),
            compute,
            linear,
            experts,
        })
    }

    pub fn position(&self) -> usize {
        self.kv.seq_len
    }

    pub fn forward(
        &mut self,
        ids: &[u32],
        embeddings: Option<&[f32]>,
        batch: usize,
    ) -> Result<Vec<f32>, String> {
        let c = &self.model.config;
        if ids.is_empty() || batch == 0 || ids.iter().any(|&id| id as usize >= c.vocab) {
            return Err("Hy forward requires valid tokens and a positive batch size".into());
        }
        let end = self
            .kv
            .seq_len
            .checked_add(ids.len())
            .ok_or("Hy position overflow")?;
        if end > self.kv.capacity {
            return Err("Hy forward exceeds KV context capacity".into());
        }
        let expected = ids
            .len()
            .checked_mul(c.n_embd)
            .ok_or("Hy embedding size overflow")?;
        if embeddings.is_some_and(|e| e.len() != expected || e.iter().any(|v| !v.is_finite())) {
            return Err("Invalid Hy embedding dimensions or non-finite values".into());
        }
        let _scope = self.compute.enter_legacy_scope();
        let mut last = Vec::new();
        for start in (0..ids.len()).step_by(batch.min(ROWS)) {
            let end = (start + batch.min(ROWS)).min(ids.len());
            let hidden = match embeddings {
                Some(e) => e[start * c.n_embd..end * c.n_embd].to_vec(),
                None => self.model.embed_tokens(&ids[start..end])?,
            };
            last = self.chunk(hidden, end - start)?;
        }
        let mut normalized = vec![0.0; c.n_embd];
        rms_norm(&last, &self.model.output_norm, &mut normalized, c.eps);
        let mut logits = vec![0.0; c.vocab];
        project(
            &mut self.linear,
            &normalized,
            1,
            [(&self.model.output, &mut logits)],
        )?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err("non-finite Hy logits".into());
        }
        Ok(logits)
    }

    fn chunk(&mut self, mut hidden: Vec<f32>, rows: usize) -> Result<Vec<f32>, String> {
        let c = &self.model.config;
        let nq = c.n_head * c.n_embd_head_k;
        let nkv = c.n_head_kv * c.n_embd_head_k;
        for (index, layer) in self.model.layers.iter().enumerate() {
            let mut normed = vec![0.0; hidden.len()];
            for (x, y) in hidden
                .chunks_exact(c.n_embd)
                .zip(normed.chunks_exact_mut(c.n_embd))
            {
                rms_norm(x, &layer.attn_norm, y, c.eps);
            }
            let mut q = vec![0.0; rows * nq];
            let mut k = vec![0.0; rows * nkv];
            let mut v = vec![0.0; rows * nkv];
            project(
                &mut self.linear,
                &normed,
                rows,
                [(&layer.q, &mut q), (&layer.k, &mut k), (&layer.v, &mut v)],
            )?;
            for row in 0..rows {
                for head in q[row * nq..(row + 1) * nq].chunks_exact_mut(c.n_embd_head_k) {
                    rms_norm_inplace(head, &layer.q_norm, c.eps);
                    ops::rope_neox_inplace(
                        head,
                        self.kv.seq_len + row,
                        c.n_embd_head_k,
                        c.freq_base,
                    );
                }
                for head in k[row * nkv..(row + 1) * nkv].chunks_exact_mut(c.n_embd_head_k) {
                    rms_norm_inplace(head, &layer.k_norm, c.eps);
                    ops::rope_neox_inplace(
                        head,
                        self.kv.seq_len + row,
                        c.n_embd_head_k,
                        c.freq_base,
                    );
                }
                let dst = (index * self.kv.capacity + self.kv.seq_len + row) * nkv;
                let src = row * nkv..(row + 1) * nkv;
                match &mut self.kv.cache {
                    KvCache::F32(cache) => {
                        cache.k[dst..dst + nkv].copy_from_slice(&k[src.clone()]);
                        cache.v[dst..dst + nkv].copy_from_slice(&v[src]);
                    }
                    KvCache::F16(cache) => {
                        ops::f32_slice_to_f16(&k[src.clone()], &mut cache.k[dst..dst + nkv]);
                        ops::f32_slice_to_f16(&v[src], &mut cache.v[dst..dst + nkv]);
                    }
                }
            }
            let attention = self.attention(&q, rows, index);
            let mut residual = vec![0.0; hidden.len()];
            project(
                &mut self.linear,
                &attention,
                rows,
                [(&layer.out, &mut residual)],
            )?;
            ops::vec_add_into(&residual, &mut hidden);
            for (x, y) in hidden
                .chunks_exact(c.n_embd)
                .zip(normed.chunks_exact_mut(c.n_embd))
            {
                rms_norm(x, &layer.ffn_norm, y, c.eps);
            }
            match &layer.ffn {
                HyFfn::Dense(e) => {
                    let mut gate = vec![0.0; rows * e.gate.n_out];
                    let mut up = vec![0.0; gate.len()];
                    project(
                        &mut self.linear,
                        &normed,
                        rows,
                        [(&e.gate, &mut gate), (&e.up, &mut up)],
                    )?;
                    ops::silu_mul_approx_inplace(&gate, &mut up);
                    project(&mut self.linear, &up, rows, [(&e.down, &mut residual)])?;
                }
                HyFfn::Moe(moe) => {
                    let executor = self.experts[index]
                        .as_mut()
                        .ok_or("missing Hy MoE executor")?;
                    for (input, output) in normed
                        .chunks_exact(c.n_embd)
                        .zip(residual.chunks_exact_mut(c.n_embd))
                    {
                        moe.forward(input, &c.moe, executor, output)?;
                    }
                }
            }
            ops::vec_add_into(&residual, &mut hidden);
        }
        self.kv.seq_len += rows;
        Ok(hidden[hidden.len() - c.n_embd..].to_vec())
    }

    fn attention(&self, q: &[f32], rows: usize, layer: usize) -> Vec<f32> {
        let c = &self.model.config;
        let width = c.n_embd_head_k;
        let nq = c.n_head * width;
        let stride = c.n_head_kv * width;
        let base = layer * self.kv.capacity * stride;
        let scale = 1.0 / (width as f32).sqrt();
        let mut output = vec![0.0; rows * nq];
        let ptr = output.as_mut_ptr();
        self.model.pool.compute(|ith, nth| {
            let mut scores = vec![f32::NEG_INFINITY; (self.kv.seq_len + rows).div_ceil(256) * 256];
            let mut query16 = vec![0u16; width];
            let mut weights16 = vec![0u16; scores.len()];
            let mut values16 = vec![0u16; scores.len()];
            let mut values32 = vec![0.0; scores.len()];
            for head in ith * c.n_head / nth..(ith + 1) * c.n_head / nth {
                let kv_head = head / (c.n_head / c.n_head_kv);
                for row in 0..rows {
                    let visible = self.kv.seq_len + row + 1;
                    let padded = visible.div_ceil(256) * 256;
                    let query = &q[row * nq + head * width..row * nq + (head + 1) * width];
                    // SAFETY: workers own disjoint heads in every output row.
                    let out = unsafe {
                        std::slice::from_raw_parts_mut(ptr.add(row * nq + head * width), width)
                    };
                    scores[..padded].fill(f32::NEG_INFINITY);
                    match &self.kv.cache {
                        KvCache::F32(cache) => {
                            for token in 0..visible {
                                let pos = base + token * stride + kv_head * width;
                                scores[token] = ops::dot::dot_f32_ggml(
                                    query,
                                    &cache.k[pos..pos + width],
                                    width,
                                ) * scale;
                            }
                            ops::softmax_inplace(&mut scores[..padded]);
                            values32[visible..padded].fill(0.0);
                            for dimension in 0..width {
                                for token in 0..visible {
                                    values32[token] = cache.v
                                        [base + token * stride + kv_head * width + dimension];
                                }
                                out[dimension] = ops::dot::dot_f32_ggml(
                                    &values32[..padded],
                                    &scores[..padded],
                                    padded,
                                );
                            }
                        }
                        KvCache::F16(cache) => {
                            ops::f32_slice_to_f16(query, &mut query16);
                            for token in 0..visible {
                                let pos = base + token * stride + kv_head * width;
                                scores[token] = ops::dot::dot_f16_ggml(
                                    &query16,
                                    &cache.k[pos..pos + width],
                                    width,
                                ) * scale;
                            }
                            ops::softmax_inplace(&mut scores[..padded]);
                            ops::f32_slice_to_f16(&scores[..padded], &mut weights16[..padded]);
                            values16[visible..padded].fill(0);
                            for dimension in 0..width {
                                for token in 0..visible {
                                    values16[token] = cache.v
                                        [base + token * stride + kv_head * width + dimension];
                                }
                                out[dimension] = ops::dot::dot_f16_ggml(
                                    &values16[..padded],
                                    &weights16[..padded],
                                    padded,
                                );
                            }
                        }
                    }
                }
            }
        });
        output
    }
}

fn project<const N: usize>(
    executor: &mut LinearExecutor<'_, 'static>,
    input: &[f32],
    rows: usize,
    projections: [(&Weight<'static>, &mut Vec<f32>); N],
) -> Result<(), String> {
    let mut ids = Vec::with_capacity(N);
    for (weight, _) in &projections {
        ids.push(executor.id_for(weight).map_err(|e| e.to_string())?);
    }
    let ids = std::array::from_fn(|i| ids[i]);
    executor
        .run_group(
            ids,
            input,
            rows,
            projections.map(|(_, out)| out.as_mut_slice()),
        )
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
#[path = "decoder_tests.rs"]
mod tests;
