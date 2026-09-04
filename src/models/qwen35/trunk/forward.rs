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
use super::util::{l2_norm, sigmoid_f32, softplus_f32};
use super::weights::Qwen35LayerWeights;
use crate::core::scratchpad::KvCache;
use crate::core::thread_pool::ComputePool;
use crate::ops::{
    attention_value_f32, dot_f32, rope_mrope, rope_neox, silu_approx_inplace,
    silu_mul_approx_inplace, softmax_inplace,
};
#[cfg(feature = "parity-trace")]
use crate::parity_trace;

impl<'a> super::weights::Qwen35Model<'a> {
    pub fn forward(
        &self,
        n_tokens: usize,
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
                parity_trace::report(parity_trace::checkpoint(
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
                    )
                }
                #[cfg(not(feature = "parity-trace"))]
                {
                    self.forward_recurrent_layer(il, normed_input, n_tokens, scratch, pool)
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
            if trace_layer(il) {
                parity_trace::report(parity_trace::checkpoint(
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

        let mut normed = vec![0.0f32; n_tokens * n_embd];
        for t in 0..n_tokens {
            let off = t * n_embd;
            normed[off..off + n_embd].copy_from_slice(&scratch.x[off..off + n_embd]);
            crate::ops::rms_norm_inplace(&mut normed[off..off + n_embd], &self.output_norm, eps);
        }

        let last_normed = &normed[(n_tokens - 1) * n_embd..n_tokens * n_embd];
        #[cfg(feature = "parity-trace")]
        parity_trace::report(parity_trace::checkpoint(
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
        let mut result = vec![0.0f32; cfg.vocab_size];
        let n = scratch.matmul_out.len().min(cfg.vocab_size);
        result[..n].copy_from_slice(&scratch.matmul_out[..n]);
        #[cfg(feature = "parity-trace")]
        parity_trace::report(parity_trace::checkpoint(
            "result_output",
            None,
            &[cfg.vocab_size],
            &result[..cfg.vocab_size],
        ));
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

        for t in 0..n_tokens {
            let inp_off = t * n_embd;
            let t0 = std::time::Instant::now();
            let inp_slice = &input[inp_off..inp_off + n_embd];
            wq.quantize_and_matmul_with_scratch(
                inp_slice,
                &mut scratch.q8k_buf,
                &mut scratch.q8_buf,
                &mut scratch.scale_buf,
                &mut scratch.matmul_out,
                pool,
            );
            scratch.q_buf[t * q_dim..t * q_dim + q_dim]
                .copy_from_slice(&scratch.matmul_out[..q_dim]);
            wk.quantize_and_matmul_with_scratch(
                inp_slice,
                &mut scratch.q8k_buf,
                &mut scratch.q8_buf,
                &mut scratch.scale_buf,
                &mut scratch.matmul_out,
                pool,
            );
            scratch.k_buf[t * k_dim..t * k_dim + k_dim]
                .copy_from_slice(&scratch.matmul_out[..k_dim]);
            wv.quantize_and_matmul_with_scratch(
                inp_slice,
                &mut scratch.q8k_buf,
                &mut scratch.q8_buf,
                &mut scratch.scale_buf,
                &mut scratch.matmul_out,
                pool,
            );
            scratch.v_buf[t * v_dim..t * v_dim + v_dim]
                .copy_from_slice(&scratch.matmul_out[..v_dim]);
            t_qkv += t0.elapsed().as_secs_f64();
        }

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
            parity_trace::report(parity_trace::checkpoint(
                &format!("Qcur_normed-{il}"),
                Some(il),
                &[n_tokens, n_head, n_embd_head],
                &q_trace,
            ));
            parity_trace::report(parity_trace::checkpoint(
                &format!("Kcur_normed-{il}"),
                Some(il),
                &[n_tokens, n_head_kv, n_embd_head],
                &scratch.k_buf[..n_tokens * k_dim],
            ));
        }

        let kv_pos = kv_cache_pos(kv_cache, il, k_dim, cfg.n_layer_impl());
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
                    rope_neox(
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
                    rope_neox(
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
            parity_trace::report(parity_trace::checkpoint(
                &format!("Qcur-{il}"),
                Some(il),
                &[n_tokens, n_head, n_embd_head],
                &q_trace,
            ));
            parity_trace::report(parity_trace::checkpoint(
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
                // V cache layout is [layer, kv_head, head_dim, seq]; the column
                // `v[il, kv_h, d, 0..capacity]` is contiguous in `seq`. Score is
                // NEG_INFINITY-padded past `n_attend`, so the corresponding
                // (zero-initialized) V tail contributes nothing after softmax.
                let v_capacity = v_len / (n_head_kv * n_embd_head);
                let v_layer_base = il * v_len + kv_h * (n_embd_head * v_capacity);
                let v_col_end = kv_pos + t + 1;
                let v_col_end_padded = v_col_end.div_ceil(256) * 256;
                let v_col_end_padded = v_col_end_padded.min(v_capacity);
                for d in 0..n_embd_head {
                    let v_col = v_col_end_padded;
                    let v_col_start = v_layer_base + d * v_capacity;
                    scratch.attn_out_buf[out_base + d] = attention_value_f32(
                        &v_cache[v_col_start..v_col_start + v_col],
                        &scratch.score_buf[..v_col],
                        v_col_end,
                        v_col,
                    );
                }
            }
        }
        t_score += t0.elapsed().as_secs_f64();

        for t in 0..n_tokens {
            for h in 0..n_head {
                let gate_off = t * q_dim + h * n_embd_head * 2 + n_embd_head;
                let out_off = t * n_embd_heads_total + h * n_embd_head;
                for d in 0..n_embd_head {
                    scratch.attn_out_buf[out_off + d] *= sigmoid_f32(scratch.q_buf[gate_off + d]);
                }
            }
        }

        let mut result = vec![0.0f32; n_tokens * n_embd];
        let t0 = std::time::Instant::now();
        for t in 0..n_tokens {
            let wo_input = &scratch.attn_out_buf
                [t * n_embd_heads_total..t * n_embd_heads_total + n_embd_heads_total];
            wo.quantize_and_matmul_with_scratch(
                wo_input,
                &mut scratch.q8k_buf,
                &mut scratch.q8_buf,
                &mut scratch.scale_buf,
                &mut scratch.matmul_out,
                pool,
            );
            result[t * n_embd..t * n_embd + n_embd].copy_from_slice(&scratch.matmul_out[..n_embd]);
        }
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
    ) -> Vec<f32> {
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
        for t in 0..n_tokens {
            let inp_off = t * n_embd;
            let inp_slice = &input[inp_off..inp_off + n_embd];
            wqkv.quantize_and_matmul_with_scratch(
                inp_slice,
                &mut scratch.q8k_buf,
                &mut scratch.q8_buf,
                &mut scratch.scale_buf,
                &mut scratch.matmul_out,
                pool,
            );
            scratch.qkv_buf[t * conv_dim..t * conv_dim + conv_dim]
                .copy_from_slice(&scratch.matmul_out[..conv_dim]);
            wqkv_gate.quantize_and_matmul_with_scratch(
                inp_slice,
                &mut scratch.q8k_buf,
                &mut scratch.q8_buf,
                &mut scratch.scale_buf,
                &mut scratch.matmul_out,
                pool,
            );
            scratch.z_buf[t * value_dim..t * value_dim + value_dim]
                .copy_from_slice(&scratch.matmul_out[..value_dim]);
            ssm_beta.quantize_and_matmul_with_scratch(
                inp_slice,
                &mut scratch.q8k_buf,
                &mut scratch.q8_buf,
                &mut scratch.scale_buf,
                &mut scratch.matmul_out,
                pool,
            );
            let n_beta = num_v_heads;
            scratch.beta_buf[t * num_v_heads..t * num_v_heads + n_beta]
                .copy_from_slice(&scratch.matmul_out[..n_beta]);
            for v in 0..num_v_heads {
                scratch.beta_buf[t * num_v_heads + v] =
                    sigmoid_f32(scratch.beta_buf[t * num_v_heads + v]);
            }
            ssm_alpha.quantize_and_matmul_with_scratch(
                inp_slice,
                &mut scratch.q8k_buf,
                &mut scratch.q8_buf,
                &mut scratch.scale_buf,
                &mut scratch.matmul_out,
                pool,
            );
            let n_alpha = num_v_heads;
            scratch.alpha_buf[t * num_v_heads..t * num_v_heads + n_alpha]
                .copy_from_slice(&scratch.matmul_out[..n_alpha]);
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
            parity_trace::report(parity_trace::checkpoint(
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
            parity_trace::report(parity_trace::checkpoint(
                &format!("q_conv_predelta-{il}"),
                Some(il),
                &[n_tokens, num_k_heads, head_k_dim],
                &scratch.q_buf[..n_tokens * key_dim],
            ));
            parity_trace::report(parity_trace::checkpoint(
                &format!("k_conv_predelta-{il}"),
                Some(il),
                &[n_tokens, num_k_heads, head_k_dim],
                &scratch.k_buf2[..n_tokens * key_dim],
            ));
        }

        let tc = tc0.elapsed().as_secs_f64();

        let ts0 = std::time::Instant::now();
        let q_scale = 1.0 / (head_k_dim as f32).sqrt();
        #[cfg(feature = "parity-trace")]
        let state_before = if trace_layer {
            Some(scratch.ssm_states[il].clone())
        } else {
            None
        };
        #[cfg(feature = "parity-trace")]
        if let Some(state_before) = state_before.as_deref() {
            parity_trace::report(parity_trace::checkpoint(
                &format!("state_predelta-{il}"),
                Some(il),
                &[num_v_heads, head_v_dim, head_v_dim],
                state_before,
            ));
        }
        let ssm_state = &mut scratch.ssm_states[il];
        for t in 0..n_tokens {
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
        }
        #[cfg(feature = "parity-trace")]
        if trace_layer {
            parity_trace::report(parity_trace::checkpoint(
                &format!("new_state-{il}"),
                Some(il),
                &[num_v_heads, head_v_dim, head_v_dim],
                ssm_state,
            ));
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
            parity_trace::report(parity_trace::checkpoint(
                &format!("final_output-{il}"),
                Some(il),
                &[n_tokens, num_v_heads, head_v_dim],
                &scratch.attn_out_buf[..n_tokens * value_dim],
            ));
        }

        let tnorm = tn0.elapsed().as_secs_f64();
        let mut result = vec![0.0f32; n_tokens * n_embd];
        let t0 = std::time::Instant::now();
        for t in 0..n_tokens {
            let inp = &scratch.attn_out_buf[t * value_dim..][..value_dim];
            ssm_out.quantize_and_matmul_with_scratch(
                inp,
                &mut scratch.q8k_buf,
                &mut scratch.q8_buf,
                &mut scratch.scale_buf,
                &mut scratch.matmul_out,
                pool,
            );
            result[t * n_embd..t * n_embd + n_embd].copy_from_slice(&scratch.matmul_out[..n_embd]);
        }
        let t_out_matmul = t0.elapsed().as_secs_f64();
        if profile {
            eprintln!(
                "  recr[{}]: matmul={:.3}s conv={:.3}s ssm={:.3}s norm={:.3}s out={:.3}s",
                il, t_matmul, tc, tssm, tnorm, t_out_matmul
            );
        }
        result
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

        for t in 0..n_tokens {
            let off = t * n_embd;
            let inp = &hidden[off..off + n_embd];
            layer.ffn_gate.quantize_and_matmul_with_scratch(
                inp,
                &mut scratch.q8k_buf,
                &mut scratch.q8_buf,
                &mut scratch.scale_buf,
                &mut scratch.matmul_out,
                pool,
            );
            scratch.ffn_gate_buf[t * n_ff..t * n_ff + n_ff]
                .copy_from_slice(&scratch.matmul_out[..n_ff]);
            layer.ffn_up.quantize_and_matmul_with_scratch(
                inp,
                &mut scratch.q8k_buf,
                &mut scratch.q8_buf,
                &mut scratch.scale_buf,
                &mut scratch.matmul_out,
                pool,
            );
            scratch.ffn_up_buf[t * n_ff..t * n_ff + n_ff]
                .copy_from_slice(&scratch.matmul_out[..n_ff]);
        }

        silu_mul_approx_inplace(
            &scratch.ffn_gate_buf[..n_tokens * n_ff],
            &mut scratch.ffn_up_buf[..n_tokens * n_ff],
        );

        for t in 0..n_tokens {
            let down_inp = &scratch.ffn_up_buf[t * n_ff..][..n_ff];
            layer.ffn_down.quantize_and_matmul_with_scratch(
                down_inp,
                &mut scratch.q8k_buf,
                &mut scratch.q8_buf,
                &mut scratch.scale_buf,
                &mut scratch.matmul_out,
                pool,
            );
            scratch.buf[t * n_embd..t * n_embd + n_embd]
                .copy_from_slice(&scratch.matmul_out[..n_embd]);
        }
    }
}
