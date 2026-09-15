//! Forward-pass implementations for `Qwen35Model`.
//!
//! Layer dispatch:
//!   `forward` → `forward_dense_attn_layer` (attention) | `forward_recurrent_layer` (Mamba SSM)
//!             → `forward_ffn_parallel` (shared SwiGLU FFN)
//!
//! The dense and recurrent paths are mutually exclusive per layer
//! (selected by `config.is_recurrent`). Both produce a `[n_tokens, n_embd]`
//! post-attention tensor that the outer `forward` loop post-norms and feeds
//! to FFN.

use super::config::Qwen35Config;
use super::scratch::{kv_cache_pos, kv_cache_store};
use super::util::{l2_norm, softplus_f32};
use super::weights::Qwen35LayerWeights;
use crate::core::scratchpad::KvCache;
use crate::core::thread_pool::ComputePool;
use crate::ops::{
    dot_f32, rope_mrope, rope_neox_inplace, sigmoid_inplace, silu_approx_inplace,
    silu_mul_approx_inplace, softmax_inplace,
};
#[cfg(feature = "parity-trace")]
use crate::parity_trace;
#[cfg(feature = "vulkan")]
use crate::vulkan::qwen35::Qwen35VulkanSession;

#[cfg(test)]
thread_local! {
    static CPU_SCAN_FAILURE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(super) fn set_cpu_scan_failure(row: Option<usize>) {
    CPU_SCAN_FAILURE.set(row);
}

impl<'a> super::weights::Qwen35Model<'a> {
    pub fn forward(
        &mut self,
        n_tokens: usize,
        kv_cache: &mut KvCache,
        scratch: &mut super::scratch::Qwen35Scratchpad,
        pool: &ComputePool,
        mrope_positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        self.forward_impl(n_tokens, None, kv_cache, scratch, pool, mrope_positions)
    }

    pub(crate) fn forward_at(
        &mut self,
        n_tokens: usize,
        base_position: usize,
        kv_cache: &mut KvCache,
        scratch: &mut super::scratch::Qwen35Scratchpad,
        pool: &ComputePool,
        mrope_positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        self.forward_impl(
            n_tokens,
            Some(base_position),
            kv_cache,
            scratch,
            pool,
            mrope_positions,
        )
    }

    pub(crate) fn forward_chunk(
        &mut self,
        n_tokens: usize,
        base_position: usize,
        kv_cache: &mut KvCache,
        scratch: &mut super::scratch::Qwen35Scratchpad,
        conv_states: &mut [Vec<f32>],
        ssm_states: &mut [Vec<f32>],
        pool: &ComputePool,
        mrope_positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        if conv_states.len() != scratch.conv_states.len()
            || ssm_states.len() != scratch.ssm_states.len()
        {
            return Err("Qwen3.5 recurrent state layer count mismatch".into());
        }
        for (persistent, working) in scratch.conv_states.iter_mut().zip(conv_states.iter_mut()) {
            std::mem::swap(persistent, working);
        }
        for (persistent, working) in scratch.ssm_states.iter_mut().zip(ssm_states.iter_mut()) {
            std::mem::swap(persistent, working);
        }
        let result = self.forward_at(
            n_tokens,
            base_position,
            kv_cache,
            scratch,
            pool,
            mrope_positions,
        );
        for (persistent, working) in scratch.conv_states.iter_mut().zip(conv_states.iter_mut()) {
            std::mem::swap(persistent, working);
        }
        for (persistent, working) in scratch.ssm_states.iter_mut().zip(ssm_states.iter_mut()) {
            std::mem::swap(persistent, working);
        }
        result
    }

    fn forward_impl(
        &mut self,
        n_tokens: usize,
        base_position: Option<usize>,
        kv_cache: &mut KvCache,
        scratch: &mut super::scratch::Qwen35Scratchpad,
        pool: &ComputePool,
        mrope_positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        if mrope_positions.len() != n_tokens {
            return Err(format!(
                "Qwen3.5 position count mismatch: tokens={n_tokens}, positions={}",
                mrope_positions.len()
            ));
        }

        // ---- GPU dispatch (decode-only; prefill falls through to CPU) ----
        // `forward_token` is single-token; the multimodal/text path's first
        // call is the prefill (n_tokens > 1), subsequent calls decode
        // (n_tokens == 1). We lazily build the session on the first decode.
        #[cfg(feature = "vulkan")]
        let gpu_allowed = base_position.is_none()
            && n_tokens == 1
            && !crate::core::thread_pool::gpu_matmul_disabled();
        #[cfg(feature = "vulkan")]
        if gpu_allowed && self.gpu.is_none() {
            if let Some(context) = crate::ops::get_vulkan_context() {
                let cfg_probe = &self.config;
                let n_layer = cfg_probe.n_layer_impl();
                let stride = cfg_probe.n_head_kv * cfg_probe.key_length.max(cfg_probe.value_length);
                let capacity = match kv_cache {
                    KvCache::F32(c) if n_layer > 0 && stride > 0 => c.k.len() / n_layer / stride,
                    KvCache::F16(c) if n_layer > 0 && stride > 0 => c.k.len() / n_layer / stride,
                    _ => 0,
                };
                if capacity > 0 {
                    match Qwen35VulkanSession::try_new(self, capacity, context) {
                        Ok(Some(gpu)) => {
                            eprintln!("[GPU] Qwen3.5 Vulkan session ready (capacity={capacity})");
                            self.gpu = Some(gpu);
                        }
                        Ok(None) => {}
                        Err(error) => {
                            eprintln!(
                                "[GPU] Qwen3.5 Vulkan session init failed: {error}. Falling back to CPU."
                            );
                        }
                    }
                }
            }
        }

        #[cfg(feature = "vulkan")]
        if gpu_allowed {
            if let Some(gpu) = self.gpu.as_mut() {
                let gpu_capacity = gpu.capacity;
                let cache_position = mrope_positions[0][0];
                let mrope_pos = mrope_positions[0];
                let cfg = &self.config;
                let stride = cfg.n_head_kv * cfg.key_length.max(cfg.value_length);
                let input = &scratch.x[..cfg.n_embd];
                enum DispatchOutcome {
                    Logits(Vec<f32>),
                    Failed,
                }
                let outcome = {
                    let result = gpu.forward_token(input, cache_position, mrope_pos);
                    match result {
                        Ok(result) => {
                            let commit_result = crate::vulkan::qwen35::commit_shadow_state(
                                kv_cache,
                                &mut scratch.conv_states,
                                &mut scratch.ssm_states,
                                cache_position,
                                gpu_capacity,
                                stride,
                                result.k_delta,
                                result.v_delta,
                                result.conv_state,
                                result.ssm_state,
                            );
                            if let Err(error) = commit_result {
                                eprintln!(
                                    "[GPU] Qwen3.5 Vulkan commit failed: {error}. Falling back to CPU."
                                );
                                DispatchOutcome::Failed
                            } else {
                                let mut out = vec![0.0f32; cfg.vocab_size];
                                let n = result.logits.len().min(cfg.vocab_size);
                                out[..n].copy_from_slice(&result.logits[..n]);
                                DispatchOutcome::Logits(out)
                            }
                        }
                        Err(error) => {
                            eprintln!(
                                "[GPU] Qwen3.5 Vulkan forward_token failed: {error}. Falling back to CPU."
                            );
                            DispatchOutcome::Failed
                        }
                    }
                };
                match outcome {
                    DispatchOutcome::Logits(out) => {
                        if let Some(gpu) = self.gpu.as_mut() {
                            gpu.commit_token();
                        }
                        return Ok(out);
                    }
                    DispatchOutcome::Failed => {
                        if let Some(gpu) = self.gpu.as_mut() {
                            gpu.abort_token();
                        }
                        self.gpu = None;
                        // fall through to CPU
                    }
                }
            }
        }
        // ---- end GPU dispatch ----

        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let n_layer = cfg.n_layer_impl();
        let eps = cfg.norm_eps;
        let profile = std::env::var("PROFILE_QWEN35").is_ok();
        let mut t_attn: f64 = 0.0;
        let mut t_ffn: f64 = 0.0;
        #[cfg(feature = "parity-trace")]
        let first_dense_layer = self.config.is_recurrent.iter().position(|value| !*value);
        #[cfg(feature = "parity-trace")]
        let trace_layer =
            |layer: usize| layer == 0 || first_dense_layer == Some(layer) || layer + 1 == n_layer;

        for il in 0..n_layer {
            let layer = &self.layers[il];
            let is_recr = cfg.is_recurrent[il];

            for t in 0..n_tokens {
                let off = t * n_embd;
                scratch.normed_buf[off..off + n_embd]
                    .copy_from_slice(&scratch.x[off..off + n_embd]);
                crate::ops::rms_norm_inplace(
                    &mut scratch.normed_buf[off..off + n_embd],
                    &layer.attn_norm,
                    eps,
                );
            }
            #[cfg(feature = "parity-trace")]
            if !is_recr && trace_layer(il) {
                parity_trace::report(parity_trace::checkpoint_rows(
                    &format!("attn_norm-{il}"),
                    Some(il),
                    &[n_tokens, n_embd],
                    &scratch.normed_buf[..n_tokens * n_embd],
                ));
            }

            let t0 = std::time::Instant::now();
            let normed_ptr = scratch.normed_buf.as_ptr();
            let normed_len = n_tokens * n_embd;
            let attn_out = if is_recr {
                let normed_input = unsafe { std::slice::from_raw_parts(normed_ptr, normed_len) };
                #[cfg(feature = "parity-trace")]
                {
                    self.forward_recurrent_layer(
                        il,
                        normed_input,
                        n_tokens,
                        scratch,
                        pool,
                        trace_layer(il),
                    )?
                }
                #[cfg(not(feature = "parity-trace"))]
                {
                    self.forward_recurrent_layer(il, normed_input, n_tokens, scratch, pool)?
                }
            } else {
                let normed_input = unsafe { std::slice::from_raw_parts(normed_ptr, normed_len) };
                #[cfg(feature = "parity-trace")]
                {
                    self.forward_dense_attn_layer(
                        il,
                        normed_input,
                        n_tokens,
                        kv_cache,
                        scratch,
                        pool,
                        mrope_positions,
                        base_position,
                        trace_layer(il),
                    )
                }
                #[cfg(not(feature = "parity-trace"))]
                {
                    self.forward_dense_attn_layer(
                        il,
                        normed_input,
                        n_tokens,
                        kv_cache,
                        scratch,
                        pool,
                        mrope_positions,
                        base_position,
                    )
                }
            };
            t_attn += t0.elapsed().as_secs_f64();

            for t in 0..n_tokens {
                let off = t * n_embd;
                crate::ops::vec_add_into(
                    &attn_out[off..off + n_embd],
                    &mut scratch.x[off..off + n_embd],
                );
            }

            for t in 0..n_tokens {
                let off = t * n_embd;
                scratch.buf[off..off + n_embd].copy_from_slice(&scratch.x[off..off + n_embd]);
                crate::ops::rms_norm_inplace(
                    &mut scratch.buf[off..off + n_embd],
                    &layer.attn_post_norm,
                    eps,
                );
            }

            let t0 = std::time::Instant::now();
            let buf_ptr = scratch.buf.as_ptr();
            let buf_len = n_tokens * n_embd;
            let ffn_input = unsafe { std::slice::from_raw_parts(buf_ptr, buf_len) };
            self.forward_ffn_parallel(layer, ffn_input, n_tokens, scratch, pool);
            t_ffn += t0.elapsed().as_secs_f64();

            for t in 0..n_tokens {
                let off = t * n_embd;
                crate::ops::vec_add_into(
                    &scratch.buf[off..off + n_embd],
                    &mut scratch.x[off..off + n_embd],
                );
            }
            #[cfg(feature = "parity-trace")]
            {
                parity_trace::report(parity_trace::checkpoint_rows(
                    &format!("layer_output-{il}"),
                    Some(il),
                    &[n_tokens, n_embd],
                    &scratch.x[..n_tokens * n_embd],
                ));
            }
        }

        if profile {
            let total = t_attn + t_ffn;
            eprintln!(
                "PROFILE: attn={:.1}% ({:.3}s) ffn={:.1}% ({:.3}s)",
                t_attn / total * 100.0,
                t_attn,
                t_ffn / total * 100.0,
                t_ffn
            );
        }

        for t in 0..n_tokens {
            let off = t * n_embd;
            scratch.normed_buf[off..off + n_embd].copy_from_slice(&scratch.x[off..off + n_embd]);
            crate::ops::rms_norm_inplace(
                &mut scratch.normed_buf[off..off + n_embd],
                &self.output_norm,
                eps,
            );
        }

        #[cfg(feature = "parity-trace")]
        let trace_all = std::env::var_os("RMI_PARITY_TRACE").is_some();
        #[cfg(not(feature = "parity-trace"))]
        let trace_all = false;
        let mut result = vec![0.0f32; cfg.vocab_size];
        for row in if trace_all {
            0..n_tokens
        } else {
            n_tokens - 1..n_tokens
        } {
            let last_normed = &scratch.normed_buf[row * n_embd..(row + 1) * n_embd];
            #[cfg(feature = "parity-trace")]
            parity_trace::report(parity_trace::checkpoint_row(
                row,
                "result_norm",
                None,
                &[n_embd],
                last_normed,
            ));
            self.output_weight.quantize_and_matmul_with_scratch(
                last_normed,
                &mut scratch.q8k_buf,
                &mut scratch.q8_buf,
                &mut scratch.scale_buf,
                &mut scratch.matmul_out,
                pool,
            );
            let n = scratch.matmul_out.len().min(cfg.vocab_size);
            result[..n].copy_from_slice(&scratch.matmul_out[..n]);
            #[cfg(feature = "parity-trace")]
            parity_trace::report(parity_trace::checkpoint_row(
                row,
                "result_output",
                None,
                &[cfg.vocab_size],
                &result[..cfg.vocab_size],
            ));
        }
        Ok(result)
    }

    pub(super) fn forward_dense_attn_layer(
        &self,
        il: usize,
        input: &[f32],
        n_tokens: usize,
        kv_cache: &mut KvCache,
        scratch: &mut super::scratch::Qwen35Scratchpad,
        pool: &ComputePool,
        mrope_positions: &[[usize; 4]],
        base_position: Option<usize>,
        #[cfg(feature = "parity-trace")] trace_layer: bool,
    ) -> Vec<f32> {
        let profile = std::env::var("PROFILE_QWEN35").is_ok();
        let cfg: &Qwen35Config = &self.config;
        let n_embd = cfg.n_embd;
        let n_head = cfg.n_head;
        let n_head_kv = cfg.n_head_kv;
        let n_embd_head = cfg.n_embd_head();
        let eps = cfg.norm_eps;
        let _nth = pool.n_threads();
        let layer = &self.layers[il];
        let wq = layer.wq.as_ref().unwrap();
        let wk = layer.wk.as_ref().unwrap();
        let wv = layer.wv.as_ref().unwrap();
        let wo = layer.wo.as_ref().unwrap();
        let q_norm_w = layer.attn_q_norm.as_ref().unwrap();
        let k_norm_w = layer.attn_k_norm.as_ref().unwrap();
        let q_dim = n_embd_head * n_head * 2;
        let k_dim = n_embd_head * n_head_kv;
        let v_dim = n_embd_head * n_head_kv;
        let n_embd_heads_total = n_embd_head * n_head;

        let mut t_qkv: f64 = 0.0;
        let mut t_score: f64 = 0.0;
        let mut t_wo: f64 = 0.0;
        let q8k_required = wq.uses_q8_k() || wk.uses_q8_k() || wv.uses_q8_k();
        assert!(
            !q8k_required || n_embd % crate::ops::quant::QK_K == 0,
            "Qwen3.5 attention input width {n_embd} must be a multiple of {} for K-quant weights",
            crate::ops::quant::QK_K
        );

        let t0 = std::time::Instant::now();
        let need_q8 = [wq, wk, wv]
            .iter()
            .any(|weight| weight.needs_q8_0_activation());
        scratch
            .prepared
            .prepare(input, n_tokens, n_embd, need_q8, q8k_required)
            .expect("validated Qwen3.5 dense projection shape");
        scratch
            .prepared
            .matmul_group(
                input,
                [
                    (wq, &mut scratch.q_buf[..n_tokens * q_dim]),
                    (wk, &mut scratch.k_buf[..n_tokens * k_dim]),
                    (wv, &mut scratch.v_buf[..n_tokens * v_dim]),
                ],
                pool,
            )
            .expect("validated Qwen3.5 dense projection shape");
        t_qkv += t0.elapsed().as_secs_f64();

        for t in 0..n_tokens {
            for h in 0..n_head {
                let q_off = t * q_dim + h * n_embd_head * 2;
                crate::ops::rms_norm_inplace(
                    &mut scratch.q_buf[q_off..q_off + n_embd_head],
                    q_norm_w,
                    eps,
                );
            }
            for h in 0..n_head_kv {
                crate::ops::rms_norm_inplace(
                    &mut scratch.k_buf[t * k_dim + h * n_embd_head..][..n_embd_head],
                    k_norm_w,
                    eps,
                );
            }
        }
        #[cfg(feature = "parity-trace")]
        let mut q_trace = Vec::with_capacity(n_tokens * n_head * n_embd_head);
        #[cfg(feature = "parity-trace")]
        if trace_layer {
            for token in 0..n_tokens {
                for head in 0..n_head {
                    let offset = token * q_dim + head * n_embd_head * 2;
                    q_trace.extend_from_slice(&scratch.q_buf[offset..offset + n_embd_head]);
                }
            }
            parity_trace::report(parity_trace::checkpoint_rows(
                &format!("Qcur_normed-{il}"),
                Some(il),
                &[n_tokens, n_head, n_embd_head],
                &q_trace,
            ));
            parity_trace::report(parity_trace::checkpoint_rows(
                &format!("Kcur_normed-{il}"),
                Some(il),
                &[n_tokens, n_head_kv, n_embd_head],
                &scratch.k_buf[..n_tokens * k_dim],
            ));
        }

        let kv_pos =
            base_position.unwrap_or_else(|| kv_cache_pos(kv_cache, il, k_dim, cfg.n_layer_impl()));
        let sections = cfg.rope_dimension_sections;
        let use_mrope = sections[0] > 0 && sections[1] > 0;
        for t in 0..n_tokens {
            let positions = mrope_positions[t];
            for h in 0..n_head {
                let q_off = t * q_dim + h * n_embd_head * 2;
                if use_mrope {
                    rope_mrope(
                        &mut scratch.q_buf[q_off..q_off + cfg.rope_dimension_count],
                        positions,
                        sections,
                        cfg.rope_dimension_count,
                        cfg.rope_freq_base,
                    );
                } else {
                    rope_neox_inplace(
                        &mut scratch.q_buf[q_off..q_off + cfg.rope_dimension_count],
                        positions[0],
                        cfg.rope_dimension_count,
                        cfg.rope_freq_base,
                    );
                }
            }
            for h in 0..n_head_kv {
                let k_off = t * k_dim + h * n_embd_head;
                if use_mrope {
                    rope_mrope(
                        &mut scratch.k_buf[k_off..k_off + cfg.rope_dimension_count],
                        positions,
                        sections,
                        cfg.rope_dimension_count,
                        cfg.rope_freq_base,
                    );
                } else {
                    rope_neox_inplace(
                        &mut scratch.k_buf[k_off..k_off + cfg.rope_dimension_count],
                        positions[0],
                        cfg.rope_dimension_count,
                        cfg.rope_freq_base,
                    );
                }
            }
        }
        #[cfg(feature = "parity-trace")]
        let mut q_trace = Vec::with_capacity(n_tokens * n_head * n_embd_head);
        #[cfg(feature = "parity-trace")]
        if trace_layer {
            for token in 0..n_tokens {
                for head in 0..n_head {
                    let offset = token * q_dim + head * n_embd_head * 2;
                    q_trace.extend_from_slice(&scratch.q_buf[offset..offset + n_embd_head]);
                }
            }
            parity_trace::report(parity_trace::checkpoint_rows(
                &format!("Qcur-{il}"),
                Some(il),
                &[n_tokens, n_head, n_embd_head],
                &q_trace,
            ));
            parity_trace::report(parity_trace::checkpoint_rows(
                &format!("Kcur-{il}"),
                Some(il),
                &[n_tokens, n_head_kv, n_embd_head],
                &scratch.k_buf[..n_tokens * k_dim],
            ));
        }

        kv_cache_store(
            kv_cache,
            il,
            cfg.n_layer_impl(),
            n_head_kv,
            n_embd_head,
            &scratch.k_buf[..n_tokens * k_dim],
            &scratch.v_buf[..n_tokens * v_dim],
            k_dim,
            v_dim,
            kv_pos,
        );
        let _n_kv = kv_pos + n_tokens;
        let scale = 1.0 / (n_embd_head as f32).sqrt();

        let (k_cache, v_cache) = match kv_cache {
            KvCache::F32(c) => (&c.k, &c.v),
            _ => return vec![0.0; n_tokens * n_embd],
        };
        let k_len = k_cache.len() / cfg.n_layer_impl();
        let v_len = v_cache.len() / cfg.n_layer_impl();

        let t0 = std::time::Instant::now();
        for t in 0..n_tokens {
            for h in 0..n_head {
                let q_off = t * q_dim + h * n_embd_head * 2;
                let kv_h = h / (n_head / n_head_kv);
                let n_attend = kv_pos + t + 1;
                let n_padded = n_attend.div_ceil(256) * 256;
                for s in 0..n_attend {
                    let k_off = il * k_len + s * k_dim + kv_h * n_embd_head;
                    let dot = dot_f32(
                        &scratch.q_buf[q_off..q_off + n_embd_head],
                        &k_cache[k_off..k_off + n_embd_head],
                        n_embd_head,
                    );
                    scratch.score_buf[s] = dot * scale;
                }
                scratch.score_buf[n_attend..n_padded].fill(f32::NEG_INFINITY);
                softmax_inplace(&mut scratch.score_buf[..n_padded]);
                let out_base = t * n_embd_heads_total + h * n_embd_head;
                // Pad the reduction even when the physical cache is shorter than
                // ggml's row. Otherwise changing the generation limit changes
                // the SIMD reduction order and therefore the prompt logits.
                let v_capacity = v_len / (n_head_kv * n_embd_head);
                let v_layer_base = il * v_len + kv_h * (n_embd_head * v_capacity);
                scratch.attention_value_buf[n_attend..n_padded].fill(0.0);
                for d in 0..n_embd_head {
                    let v_col_start = v_layer_base + d * v_capacity;
                    scratch.attention_value_buf[..n_attend]
                        .copy_from_slice(&v_cache[v_col_start..v_col_start + n_attend]);
                    scratch.attn_out_buf[out_base + d] = dot_f32(
                        &scratch.attention_value_buf[..n_padded],
                        &scratch.score_buf[..n_padded],
                        n_padded,
                    );
                }
            }
        }
        t_score += t0.elapsed().as_secs_f64();

        for t in 0..n_tokens {
            for h in 0..n_head {
                let gate_off = t * q_dim + h * n_embd_head * 2 + n_embd_head;
                let out_off = t * n_embd_heads_total + h * n_embd_head;
                let gate = &mut scratch.q_buf[gate_off..gate_off + n_embd_head];
                let attn_out = &mut scratch.attn_out_buf[out_off..out_off + n_embd_head];
                sigmoid_inplace(gate);
                for d in 0..n_embd_head {
                    attn_out[d] *= gate[d];
                }
            }
        }

        let mut result = vec![0.0f32; n_tokens * n_embd];
        let t0 = std::time::Instant::now();
        let wo_input = &scratch.attn_out_buf[..n_tokens * n_embd_heads_total];
        scratch
            .prepared
            .prepare(
                wo_input,
                n_tokens,
                n_embd_heads_total,
                wo.needs_q8_0_activation(),
                wo.uses_q8_k(),
            )
            .expect("validated Qwen3.5 output projection shape");
        scratch
            .prepared
            .matmul(wo, wo_input, &mut result, pool)
            .expect("validated Qwen3.5 output projection shape");
        t_wo += t0.elapsed().as_secs_f64();
        if profile {
            eprintln!(
                "  dense_attn[{}]: qkv={:.3}s score={:.3}s wo={:.3}s",
                il, t_qkv, t_score, t_wo
            );
        }
        result
    }

    fn forward_recurrent_layer(
        &self,
        il: usize,
        input: &[f32],
        n_tokens: usize,
        scratch: &mut super::scratch::Qwen35Scratchpad,
        pool: &ComputePool,
        #[cfg(feature = "parity-trace")] trace_layer: bool,
    ) -> Result<Vec<f32>, String> {
        let profile = std::env::var("PROFILE_QWEN35").is_ok();
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let d_inner = cfg.ssm_d_inner;
        let head_k_dim = cfg.ssm_d_state;
        let num_k_heads = cfg.ssm_n_group;
        let num_v_heads = cfg.ssm_dt_rank;
        let head_v_dim = d_inner / num_v_heads;
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();
        let d_conv = cfg.ssm_d_conv;
        let eps = cfg.norm_eps;

        let layer = &self.layers[il];
        let wqkv = layer.wqkv.as_ref().unwrap();
        let wqkv_gate = layer.wqkv_gate.as_ref().unwrap();
        let ssm_conv1d = layer.ssm_conv1d.as_ref().unwrap();
        let ssm_dt = layer.ssm_dt.as_ref().unwrap();
        let ssm_a = layer.ssm_a.as_ref().unwrap();
        let ssm_beta = layer.ssm_beta.as_ref().unwrap();
        let ssm_alpha = layer.ssm_alpha.as_ref().unwrap();
        let ssm_norm_w = layer.ssm_norm.as_ref().unwrap();
        let ssm_out = layer.ssm_out.as_ref().unwrap();

        let t0 = std::time::Instant::now();
        let projections = [wqkv, wqkv_gate, ssm_beta, ssm_alpha];
        let need_q8 = projections
            .iter()
            .any(|weight| weight.needs_q8_0_activation());
        let need_q8k = projections.iter().any(|weight| weight.uses_q8_k());
        scratch
            .prepared
            .prepare(input, n_tokens, n_embd, need_q8, need_q8k)
            .expect("validated Qwen3.5 recurrent projection shape");
        scratch
            .prepared
            .matmul_group(
                input,
                [
                    (wqkv, &mut scratch.qkv_buf[..n_tokens * conv_dim]),
                    (wqkv_gate, &mut scratch.z_buf[..n_tokens * value_dim]),
                    (ssm_beta, &mut scratch.beta_buf[..n_tokens * num_v_heads]),
                    (ssm_alpha, &mut scratch.alpha_buf[..n_tokens * num_v_heads]),
                ],
                pool,
            )
            .expect("validated Qwen3.5 recurrent projection shape");
        for t in 0..n_tokens {
            let n_beta = num_v_heads;
            sigmoid_inplace(&mut scratch.beta_buf[t * num_v_heads..t * num_v_heads + n_beta]);
            for v in 0..num_v_heads {
                let a_biased = scratch.alpha_buf[t * num_v_heads + v] + ssm_dt[v % ssm_dt.len()];
                scratch.alpha_buf[t * num_v_heads + v] =
                    softplus_f32(a_biased) * ssm_a[v % ssm_a.len()];
            }
        }
        let t_matmul = t0.elapsed().as_secs_f64();

        let tc0 = std::time::Instant::now();
        let conv_state = &mut scratch.conv_states[il];
        #[cfg(feature = "parity-trace")]
        let mut conv_raw = if trace_layer {
            vec![0.0f32; n_tokens * conv_dim]
        } else {
            Vec::new()
        };
        for t in 0..n_tokens {
            let qkv_off = t * conv_dim;
            for c in 0..conv_dim {
                for k in 0..d_conv - 1 {
                    conv_state[k * conv_dim + c] = conv_state[(k + 1) * conv_dim + c];
                }
                conv_state[(d_conv - 1) * conv_dim + c] = scratch.qkv_buf[qkv_off + c];
            }
            for c in 0..conv_dim {
                let mut conv_val = 0.0f32;
                for k in 0..d_conv {
                    conv_val += ssm_conv1d[c * d_conv + k] * conv_state[k * conv_dim + c];
                }
                #[cfg(feature = "parity-trace")]
                if trace_layer {
                    conv_raw[t * conv_dim + c] = conv_val;
                }
                scratch.qkv_buf[qkv_off + c] = conv_val;
            }
            silu_approx_inplace(&mut scratch.qkv_buf[qkv_off..qkv_off + conv_dim]);
        }
        #[cfg(feature = "parity-trace")]
        if trace_layer {
            parity_trace::report(parity_trace::checkpoint_rows(
                &format!("conv_output_raw-{il}"),
                Some(il),
                &[n_tokens, conv_dim],
                &conv_raw,
            ));
        }

        for t in 0..n_tokens {
            let qkv_off = t * conv_dim;
            for h in 0..num_k_heads {
                for d in 0..head_k_dim {
                    scratch.q_buf[t * key_dim + h * head_k_dim + d] =
                        scratch.qkv_buf[qkv_off + h * head_k_dim + d];
                }
                for d in 0..head_k_dim {
                    scratch.k_buf2[t * key_dim + h * head_k_dim + d] =
                        scratch.qkv_buf[qkv_off + key_dim + h * head_k_dim + d];
                }
            }
            for h in 0..num_v_heads {
                for d in 0..head_v_dim {
                    scratch.v_buf2[t * value_dim + h * head_v_dim + d] =
                        scratch.qkv_buf[qkv_off + 2 * key_dim + h * head_v_dim + d];
                }
            }
            for h in 0..num_k_heads {
                l2_norm(
                    &mut scratch.q_buf[t * key_dim + h * head_k_dim..][..head_k_dim],
                    eps,
                );
                l2_norm(
                    &mut scratch.k_buf2[t * key_dim + h * head_k_dim..][..head_k_dim],
                    eps,
                );
            }
        }
        #[cfg(feature = "parity-trace")]
        if trace_layer {
            parity_trace::report(parity_trace::checkpoint_rows(
                &format!("q_conv_predelta-{il}"),
                Some(il),
                &[n_tokens, num_k_heads, head_k_dim],
                &scratch.q_buf[..n_tokens * key_dim],
            ));
            parity_trace::report(parity_trace::checkpoint_rows(
                &format!("k_conv_predelta-{il}"),
                Some(il),
                &[n_tokens, num_k_heads, head_k_dim],
                &scratch.k_buf2[..n_tokens * key_dim],
            ));
        }

        let tc = tc0.elapsed().as_secs_f64();

        let ts0 = std::time::Instant::now();
        let q_scale = 1.0 / (head_k_dim as f32).sqrt();
        let ssm_state = &mut scratch.ssm_states[il];
        for t in 0..n_tokens {
            #[cfg(feature = "parity-trace")]
            if trace_layer {
                parity_trace::report(parity_trace::checkpoint_row(
                    t,
                    &format!("state_predelta-{il}"),
                    Some(il),
                    &[num_v_heads, head_v_dim, head_v_dim],
                    ssm_state,
                ));
            }
            let q_off = t * key_dim;
            let k2_off = t * key_dim;
            let v2_off = t * value_dim;
            for v_h in 0..num_v_heads {
                let gate_val = scratch.alpha_buf[t * num_v_heads + v_h];
                let beta_val = scratch.beta_buf[t * num_v_heads + v_h];
                let state_off = v_h * head_v_dim * head_v_dim;
                let k_h = v_h % num_k_heads;
                let decay = gate_val.exp();
                crate::ops::ssm_state_decay(
                    &mut ssm_state[state_off..state_off + head_v_dim * head_v_dim],
                    decay,
                );
                let k_slice = &scratch.k_buf2[k2_off + k_h * head_k_dim..][..head_v_dim];
                let mut sk = [0.0f32; 128];
                crate::ops::ssm_matvec(
                    &ssm_state[state_off..][..head_v_dim * head_v_dim],
                    k_slice,
                    head_v_dim,
                    head_v_dim,
                    &mut sk[..head_v_dim],
                );
                let v_slice = &scratch.v_buf2[v2_off + v_h * head_v_dim..][..head_v_dim];
                let mut d_vec = [0.0f32; 128];
                for d in 0..head_v_dim {
                    d_vec[d] = (v_slice[d] - sk[d]) * beta_val;
                }
                crate::ops::ssm_outer_product_update(
                    &mut ssm_state[state_off..][..head_v_dim * head_v_dim],
                    k_slice,
                    &d_vec[..head_v_dim],
                    head_v_dim,
                );
                let q_slice = &scratch.q_buf[q_off + k_h * head_k_dim..][..head_v_dim];
                let out_off = t * value_dim + v_h * head_v_dim;
                crate::ops::ssm_matvec_scaled(
                    &ssm_state[state_off..][..head_v_dim * head_v_dim],
                    q_slice,
                    head_v_dim,
                    head_v_dim,
                    &mut scratch.attn_out_buf[out_off..out_off + head_v_dim],
                    q_scale,
                );
            }
            #[cfg(test)]
            if CPU_SCAN_FAILURE.get() == Some(t) {
                return Err(format!(
                    "injected Qwen3.5 CPU chunk failure after recurrent row {t}"
                ));
            }
            #[cfg(feature = "parity-trace")]
            if trace_layer {
                parity_trace::report(parity_trace::checkpoint_row(
                    t,
                    &format!("new_state-{il}"),
                    Some(il),
                    &[num_v_heads, head_v_dim, head_v_dim],
                    ssm_state,
                ));
            }
        }

        let tssm = ts0.elapsed().as_secs_f64();
        let tn0 = std::time::Instant::now();
        for t in 0..n_tokens {
            for h in 0..num_v_heads {
                let off = t * value_dim + h * head_v_dim;
                crate::ops::rms_norm_inplace(
                    &mut scratch.attn_out_buf[off..off + head_v_dim],
                    ssm_norm_w,
                    eps,
                );
            }
            let z_off = t * value_dim;
            crate::ops::silu_mul_approx_inplace(
                &scratch.z_buf[z_off..z_off + value_dim],
                &mut scratch.attn_out_buf[t * value_dim..t * value_dim + value_dim],
            );
        }
        #[cfg(feature = "parity-trace")]
        if trace_layer {
            parity_trace::report(parity_trace::checkpoint_rows(
                &format!("final_output-{il}"),
                Some(il),
                &[n_tokens, num_v_heads, head_v_dim],
                &scratch.attn_out_buf[..n_tokens * value_dim],
            ));
        }

        let tnorm = tn0.elapsed().as_secs_f64();
        let mut result = vec![0.0f32; n_tokens * n_embd];
        let t0 = std::time::Instant::now();
        let output_input = &scratch.attn_out_buf[..n_tokens * value_dim];
        scratch
            .prepared
            .prepare(
                output_input,
                n_tokens,
                value_dim,
                ssm_out.needs_q8_0_activation(),
                ssm_out.uses_q8_k(),
            )
            .expect("validated Qwen3.5 recurrent output shape");
        scratch
            .prepared
            .matmul(ssm_out, output_input, &mut result, pool)
            .expect("validated Qwen3.5 recurrent output shape");
        let t_out_matmul = t0.elapsed().as_secs_f64();
        if profile {
            eprintln!(
                "  recr[{}]: matmul={:.3}s conv={:.3}s ssm={:.3}s norm={:.3}s out={:.3}s",
                il, t_matmul, tc, tssm, tnorm, t_out_matmul
            );
        }
        Ok(result)
    }

    fn forward_ffn_parallel(
        &self,
        layer: &Qwen35LayerWeights,
        hidden: &[f32],
        n_tokens: usize,
        scratch: &mut super::scratch::Qwen35Scratchpad,
        pool: &ComputePool,
    ) {
        let n_embd = self.config.n_embd;
        let n_ff = self.config.n_ff;
        let q8k_required = layer.ffn_gate.uses_q8_k() || layer.ffn_up.uses_q8_k();
        assert!(
            !q8k_required || n_embd % crate::ops::quant::QK_K == 0,
            "Qwen3.5 FFN input width {n_embd} must be a multiple of {} for K-quant weights",
            crate::ops::quant::QK_K
        );

        let need_q8 =
            layer.ffn_gate.needs_q8_0_activation() || layer.ffn_up.needs_q8_0_activation();
        scratch
            .prepared
            .prepare(hidden, n_tokens, n_embd, need_q8, q8k_required)
            .expect("validated Qwen3.5 FFN input shape");
        scratch
            .prepared
            .matmul_group(
                hidden,
                [
                    (
                        &layer.ffn_gate,
                        &mut scratch.ffn_gate_buf[..n_tokens * n_ff],
                    ),
                    (&layer.ffn_up, &mut scratch.ffn_up_buf[..n_tokens * n_ff]),
                ],
                pool,
            )
            .expect("validated Qwen3.5 FFN input shape");

        silu_mul_approx_inplace(
            &scratch.ffn_gate_buf[..n_tokens * n_ff],
            &mut scratch.ffn_up_buf[..n_tokens * n_ff],
        );

        let down_input = &scratch.ffn_up_buf[..n_tokens * n_ff];
        scratch
            .prepared
            .prepare(
                down_input,
                n_tokens,
                n_ff,
                layer.ffn_down.needs_q8_0_activation(),
                layer.ffn_down.uses_q8_k(),
            )
            .expect("validated Qwen3.5 FFN output shape");
        scratch
            .prepared
            .matmul(
                &layer.ffn_down,
                down_input,
                &mut scratch.buf[..n_tokens * n_embd],
                pool,
            )
            .expect("validated Qwen3.5 FFN output shape");
    }
}
