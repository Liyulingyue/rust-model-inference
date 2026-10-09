//! Standard dense adapter; the shared recipe owns the execution order.
use super::session::LlamaSession;
use crate::compute::dense::{run_dense_layer, DenseStep};
use crate::core::scratchpad::KvCache;
use crate::ops::kernel::{QuantizedTensor, Weight};

pub(super) fn eligible(s: &LlamaSession<'_>) -> Result<(), String> {
    let c = &s.config;
    if s.arch != "llama"
        || s.loop_final_norm
        || c.n_layer != s.weights.layers.len()
        || s.sliding_window != 0
        || s.attn_softcap != 0.0
        || s.final_logit_softcap != 0.0
        || s.embedding_scale != 0.0
        || s.residual_scale != 0.0
        || s.logit_scale != 0.0
        || s.norm_groups != 1
        || c.rope_dim != c.n_embd_head_k
        || c.attn_factor != 1.0
        || c.yarn_thetas.is_some()
        || c.n_embd_head_k != c.n_embd_head_v
        || s.kq_scale != 1.0 / (c.n_embd_head_k as f32).sqrt()
        || s.weights.layers.iter().any(|l| {
            l.bq.is_some()
                || l.bk.is_some()
                || l.bv.is_some()
                || l.attn_post_norm.is_some()
                || l.ffn_post_norm.is_some()
        })
    {
        return Err("full dense execution requires standard Llama: no SWA, softcap, scaling, partial/YaRN RoPE, bias or post-norm".into());
    }
    Ok(())
}

pub(super) fn forward_cpu(
    s: &mut LlamaSession<'_>,
    tokens: &[u32],
    project_logits: bool,
) -> Result<(), String> {
    let c = &s.config;
    let rows = tokens.len();
    let base = s.seq_len;
    let width = c.n_embd;
    let q_width = c.n_embd_q;
    let kv_width = c.n_embd_gqa;
    let ff = c.n_ff;
    let x_len = rows * width;
    let scratch = &mut s.scratch;
    let prepared = &mut s.prepared_rows;
    let pool = &s.pool;
    let cache = &mut s.kv_cache;
    for (token, row) in tokens
        .iter()
        .zip(scratch.x[..x_len].chunks_exact_mut(width))
    {
        crate::ops::embedding_lookup(
            s.weights.embd_weight,
            *token,
            width,
            s.weights.embd_type,
            row,
        );
    }
    for (layer, l) in s.weights.layers.iter().enumerate() {
        run_dense_layer(
            &mut |_: usize, step| -> Result<(), String> {
                match step {
                    DenseStep::AttnNorm | DenseStep::FfnNorm => {
                        let norm = if step == DenseStep::AttnNorm {
                            &l.attn_norm
                        } else {
                            &l.ffn_norm
                        };
                        for (x, n) in scratch.x[..x_len]
                            .chunks_exact(width)
                            .zip(scratch.normed[..x_len].chunks_exact_mut(width))
                        {
                            crate::ops::rms_norm(x, norm, n, c.eps);
                        }
                    }
                    DenseStep::Qkv => {
                        let weights = [&l.wq, &l.wk, &l.wv];
                        prepared.prepare(
                            &scratch.normed[..x_len],
                            rows,
                            width,
                            weights.iter().any(|w| w.needs_q8_0_activation()),
                            weights.iter().any(|w| w.uses_q8_k()),
                        )?;
                        prepared.matmul_group(
                            &scratch.normed[..x_len],
                            [
                                (&l.wq, &mut scratch.q[..rows * q_width]),
                                (&l.wk, &mut scratch.k_new[..rows * kv_width]),
                                (&l.wv, &mut scratch.v_new[..rows * kv_width]),
                            ],
                            pool,
                        )?;
                    }
                    DenseStep::QkNormRope => {
                        for row in 0..rows {
                            for values in [
                                &mut scratch.q[row * q_width..(row + 1) * q_width],
                                &mut scratch.k_new[row * kv_width..(row + 1) * kv_width],
                            ] {
                                super::forward::apply_rope(
                                    "llama",
                                    values,
                                    base + row,
                                    c.n_embd_head_k,
                                    c.freq_base,
                                    c.rope_dim,
                                    1.0,
                                    None,
                                );
                            }
                        }
                    }
                    DenseStep::AppendKv => {
                        let start = (layer * c.max_ctx + base) * kv_width;
                        let end = start + rows * kv_width;
                        let k = &scratch.k_new[..rows * kv_width];
                        let v = &scratch.v_new[..rows * kv_width];
                        match cache {
                            KvCache::F16(cache) => {
                                crate::ops::f32_slice_to_f16(k, &mut cache.k[start..end]);
                                crate::ops::f32_slice_to_f16(v, &mut cache.v[start..end]);
                            }
                            KvCache::F32(cache) => {
                                cache.k[start..end].copy_from_slice(k);
                                cache.v[start..end].copy_from_slice(v);
                            }
                        }
                    }
                    DenseStep::Attention => super::forward::run_attention_chunked(
                        pool,
                        &scratch.q[..rows * q_width],
                        &mut scratch.attn_out[..rows * q_width],
                        cache,
                        c.n_layer * c.max_ctx * kv_width,
                        base + rows,
                        base,
                        rows,
                        q_width,
                        kv_width,
                        c.n_head,
                        c.n_embd_head_k,
                        c.n_embd_head_v,
                        s.group_size,
                        s.kq_scale,
                        layer * c.max_ctx * kv_width,
                        pool.n_threads(),
                        c.max_ctx,
                        0.0,
                        0,
                    ),
                    DenseStep::AttnOut => {
                        prepared.prepare(
                            &scratch.attn_out[..rows * q_width],
                            rows,
                            q_width,
                            l.wo.needs_q8_0_activation(),
                            l.wo.uses_q8_k(),
                        )?;
                        prepared.matmul_group(
                            &scratch.attn_out[..rows * q_width],
                            [(&l.wo, &mut scratch.attn_proj[..x_len])],
                            pool,
                        )?;
                    }
                    DenseStep::AttnResidual => crate::ops::vec_add_into(
                        &scratch.attn_proj[..x_len],
                        &mut scratch.x[..x_len],
                    ),
                    DenseStep::GateUp => {
                        prepared.prepare(
                            &scratch.normed[..x_len],
                            rows,
                            width,
                            l.w_gate.needs_q8_0_activation() || l.w_up.needs_q8_0_activation(),
                            l.w_gate.uses_q8_k() || l.w_up.uses_q8_k(),
                        )?;
                        prepared.matmul_group(
                            &scratch.normed[..x_len],
                            [
                                (&l.w_gate, &mut scratch.gate_buf[..rows * ff]),
                                (&l.w_up, &mut scratch.up_buf[..rows * ff]),
                            ],
                            pool,
                        )?;
                    }
                    DenseStep::SiluMul => {
                        if rows == 1 {
                            crate::ops::silu_mul_approx_inplace(
                                &scratch.gate_buf[..ff],
                                &mut scratch.up_buf[..ff],
                            );
                        } else {
                            crate::ops::silu_mul_inplace(
                                &scratch.gate_buf[..rows * ff],
                                &mut scratch.up_buf[..rows * ff],
                            );
                        }
                    }
                    DenseStep::Down => {
                        prepared.prepare(
                            &scratch.up_buf[..rows * ff],
                            rows,
                            ff,
                            l.w_down.needs_q8_0_activation(),
                            l.w_down.uses_q8_k(),
                        )?;
                        prepared.matmul_group(
                            &scratch.up_buf[..rows * ff],
                            [(&l.w_down, &mut scratch.down_buf[..x_len])],
                            pool,
                        )?;
                    }
                    DenseStep::FfnResidual => crate::ops::vec_add_into(
                        &scratch.down_buf[..x_len],
                        &mut scratch.x[..x_len],
                    ),
                }
                Ok(())
            },
            layer,
        )?;
    }
    if !project_logits {
        return Ok(());
    }
    crate::ops::rms_norm(
        &scratch.x[(rows - 1) * width..x_len],
        &s.weights.output_norm,
        &mut scratch.normed[..width],
        c.eps,
    );
    let output = Weight::from_quantized(QuantizedTensor::from_bytes(
        s.weights.output_weight,
        s.weights.output_type,
        width,
        c.vocab,
    ));
    prepared.prepare(
        &scratch.normed[..width],
        1,
        width,
        output.needs_q8_0_activation(),
        output.uses_q8_k(),
    )?;
    prepared.matmul_group(
        &scratch.normed[..width],
        [(&output, &mut scratch.logits)],
        pool,
    )?;
    Ok(())
}

#[cfg(feature = "vulkan")]
pub(super) fn gpu_shape(s: &LlamaSession<'_>) -> crate::vulkan::dense::DenseShape {
    let c = &s.config;
    crate::vulkan::dense::DenseShape {
        n_embd: c.n_embd,
        n_ff: c.n_ff,
        n_layer: c.n_layer,
        n_head: c.n_head,
        n_head_kv: c.n_head_kv,
        n_embd_head_k: c.n_embd_head_k,
        n_embd_head_v: c.n_embd_head_v,
        eps: c.eps,
        has_qk_norm: false,
        vocab: c.vocab,
        freq_base: c.freq_base,
        rope_layout: crate::vulkan::ops::RopeLayout::Interleaved,
    }
}

#[cfg(feature = "vulkan")]
pub(super) fn gpu_weights<'a>(
    s: &'a LlamaSession<'_>,
) -> Result<crate::vulkan::dense::DenseWeights<'a>, crate::vulkan::VulkanError> {
    use crate::core::tensor::GGMLType;
    use crate::vulkan::{
        dense::{DenseLayer, DenseWeight, DenseWeights},
        VulkanError,
    };
    let view = |w: &'a Weight<'_>| -> Result<DenseWeight<'a>, VulkanError> {
        if !matches!(
            w.ggml_type,
            GGMLType::Q8_0
                | GGMLType::Q4_0
                | GGMLType::Q4_1
                | GGMLType::Q4K
                | GGMLType::Q6K
                | GGMLType::F16
        ) {
            return Err(VulkanError::UnsupportedShape(
                "unsupported Llama dense weight format".into(),
            ));
        }
        Ok(DenseWeight {
            bytes: w.kernel.weight_bytes().ok_or_else(|| {
                VulkanError::UnsupportedShape("Llama weight has no storage view".into())
            })?,
            ggml_type: w.ggml_type,
            n_in: w.n_in,
            n_out: w.n_out,
        })
    };
    let layers = s
        .weights
        .layers
        .iter()
        .map(|l| {
            Ok(DenseLayer {
                attn_norm: &l.attn_norm,
                ffn_norm: &l.ffn_norm,
                q_norm: None,
                k_norm: None,
                wq: view(&l.wq)?,
                wk: view(&l.wk)?,
                wv: view(&l.wv)?,
                wo: view(&l.wo)?,
                w_gate: view(&l.w_gate)?,
                w_up: view(&l.w_up)?,
                w_down: view(&l.w_down)?,
            })
        })
        .collect::<Result<_, VulkanError>>()?;
    let output = DenseWeight {
        bytes: s.weights.output_weight,
        ggml_type: s.weights.output_type,
        n_in: s.config.n_embd,
        n_out: s.config.vocab,
    };
    Ok(DenseWeights {
        layers,
        output_norm: &s.weights.output_norm,
        output,
    })
}

impl LlamaSession<'_> {
    pub fn used_backend(&self) -> crate::compute::UsedBackend {
        #[cfg(feature = "vulkan")]
        if self.gpu.is_some() {
            return crate::compute::UsedBackend::Vulkan;
        }
        crate::compute::UsedBackend::Cpu
    }

    pub(super) fn compute_chunk(
        &mut self,
        tokens: &[u32],
        project_logits: bool,
    ) -> Result<(), String> {
        use crate::compute::ComputePolicy;
        let rows = tokens.len();
        let base = self.seq_len;
        let end = base
            .checked_add(rows)
            .ok_or("Llama chunk length overflow")?;
        if rows == 0
            || rows > self.prepared_rows.max_rows()
            || end > self.config.max_ctx
            || tokens.iter().any(|&t| t as usize >= self.config.vocab)
        {
            return Err("invalid Llama chunk dimensions or token".into());
        }
        let _scope = self.compute_policy.cpu_scope();
        #[cfg(feature = "vulkan")]
        let gpu_error = {
            if self.compute_policy == ComputePolicy::Vulkan && self.gpu.is_none() {
                return Err("Vulkan session unavailable; create a new session to retry".into());
            }
            let width = self.config.n_embd;
            let supported = eligible(self);
            if let Some(gpu) = &mut self.gpu {
                let result = (|| {
                    supported?;
                    #[cfg(feature = "parity-trace")]
                    if std::env::var_os("RMI_PARITY_TRACE").is_some() {
                        return Err("Vulkan does not support token-major tracing".into());
                    }
                    for (token, row) in tokens
                        .iter()
                        .zip(self.scratch.x[..rows * width].chunks_exact_mut(width))
                    {
                        crate::ops::embedding_lookup(
                            self.weights.embd_weight,
                            *token,
                            width,
                            self.weights.embd_type,
                            row,
                        );
                    }
                    let result = gpu
                        .forward_chunk(&self.scratch.x[..rows * width], base, rows, project_logits)
                        .map_err(|e| e.to_string())?;
                    crate::compute::state::commit_kv_cache(
                        &mut self.kv_cache,
                        self.config.n_layer,
                        self.config.max_ctx,
                        self.config.n_embd_gqa,
                        base,
                        base,
                        rows,
                        result.k_delta,
                        result.v_delta,
                    )?;
                    if project_logits {
                        self.scratch.logits.copy_from_slice(result.logits);
                    }
                    gpu.commit_token();
                    Ok::<_, String>(())
                })();
                match result {
                    Ok(()) => {
                        self.seq_len = end;
                        self.compute_policy.trace(
                            "resident_decoder",
                            crate::compute::UsedBackend::Vulkan,
                            rows,
                        );
                        return Ok(());
                    }
                    Err(error) => {
                        gpu.abort_token();
                        self.gpu = None;
                        if self.compute_policy == ComputePolicy::Vulkan {
                            return Err(error);
                        }
                        log::info!("compute Llama: retry chunk {base}..{end} on CPU: {error}");
                        Some(error)
                    }
                }
            } else {
                None
            }
        };
        // A CPU chunk must never re-enter legacy per-matmul GPU dispatch.
        let _cpu = ComputePolicy::Cpu.cpu_scope();
        let result = if eligible(self).is_ok() {
            forward_cpu(self, tokens, project_logits)
        } else if rows == 1 {
            self.forward_one_token(tokens[0], project_logits)
        } else {
            self.forward_chunk_batched_real(tokens, rows, base, project_logits)
        };
        self.seq_len = base;
        let result = result.and_then(|()| {
            let stride = self.config.n_embd_gqa;
            let finite = |values: &[f32]| values.iter().all(|x| x.is_finite());
            let valid = (0..self.config.n_layer).all(|layer| {
                let range = (layer * self.config.max_ctx + base) * stride
                    ..(layer * self.config.max_ctx + end) * stride;
                match &self.kv_cache {
                    KvCache::F16(c) => c.k[range.clone()]
                        .iter()
                        .chain(&c.v[range])
                        .all(|&v| crate::ops::f16_to_f32(v).is_finite()),
                    KvCache::F32(c) => finite(&c.k[range.clone()]) && finite(&c.v[range]),
                }
            });
            if valid
                && finite(&self.scratch.x[..rows * self.config.n_embd])
                && (!project_logits || finite(&self.scratch.logits))
            {
                Ok(())
            } else {
                Err("Llama chunk produced non-finite state".into())
            }
        });
        if let Err(error) = result {
            #[cfg(feature = "vulkan")]
            if let Some(gpu_error) = gpu_error {
                return Err(format!("{error}; original Vulkan error: {gpu_error}"));
            }
            return Err(error);
        }
        self.seq_len = end;
        self.compute_policy
            .trace("decoder", crate::compute::UsedBackend::Cpu, rows);
        Ok(())
    }
}
