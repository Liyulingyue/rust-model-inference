//! Prompt-only forward pass for `Qwen3Model`.
//!
//! This module hosts the `text_encode` helper that runs the full transformer
//! stack (embeddings -> N decoder layers -> output norm) over a pre-tokenized
//! sequence without using a KV cache. It is used by external encoders that
//! need the final hidden state of every prompt position (e.g. the Pig image
//! text encoder pipeline).
//!
//! The implementation mirrors the layer-level forward found in the KV-cached
//! `Qwen3Session::forward` path, but allocates fresh activation buffers per
//! layer and re-uses [`Qwen3Rope`] semantics for rotary embedding. Causal
//! masking matches the autoregressive prefill step so the produced hidden
//! states match the first `n_tokens` positions of a KV-cached run.

use crate::models::qwen3::qwen3_multimodal::{
    checked_product, Qwen3Model, Qwen3Rope,
};
use crate::ops::{
    dot_f32, matmul_q8_0_quantized_parallel_rows, quantize_q8_0_into, rms_norm, rms_norm_inplace,
    rope_mrope_interleaved, rope_neox, silu,
};

pub fn text_encode(
    model: &Qwen3Model,
    token_ids: &[u32],
    positions: &[[usize; 4]],
) -> Result<Vec<f32>, String> {
    crate::models::qwen3::qwen3_multimodal::validate_token_ids(token_ids, model.config.vocab)?;
    let n_tokens = token_ids.len();
    if positions.len() != n_tokens {
        return Err(format!(
            "positions length {} != token_ids length {}",
            positions.len(),
            n_tokens
        ));
    }
    if n_tokens == 0 {
        return Ok(Vec::new());
    }

    let cfg = &model.config;
    let n_embd_q = checked_product("query width", cfg.n_head, cfg.n_embd_head_k)?;
    let n_embd_k = checked_product("key width", cfg.n_head_kv, cfg.n_embd_head_k)?;
    let n_embd_v = checked_product("value width", cfg.n_head_kv, cfg.n_embd_head_v)?;
    let n_attn = checked_product("attn width", cfg.n_head, cfg.n_embd_head_v)?;
    let group_size = cfg.n_head / cfg.n_head_kv;
    let kq_scale = 1.0 / (cfg.n_embd_head_k as f32).sqrt();

    let embeddings = model.embed_tokens(token_ids)?;
    let mut hidden = embeddings;

    for layer_idx in 0..cfg.n_layer {
        let layer = &model.layers[layer_idx];

        let mut normed = vec![0.0; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            rms_norm(
                &hidden[off..off + cfg.n_embd],
                &layer.attn_norm,
                &mut normed[off..off + cfg.n_embd],
                cfg.eps,
            );
        }

        let mut q_all = vec![0.0; n_tokens * n_embd_q];
        let mut k_all = vec![0.0; n_tokens * n_embd_k];
        let mut v_all = vec![0.0; n_tokens * n_embd_v];
        for tok in 0..n_tokens {
            let norm_row = &normed[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd];
            let q_off = tok * n_embd_q;
            let k_off = tok * n_embd_k;
            let v_off = tok * n_embd_v;

            let blocks = (cfg.n_embd + 31) / 32;
            let mut q8_buf = vec![0u8; cfg.n_embd];
            let mut scale_buf = vec![0.0f32; blocks];
            quantize_q8_0_into(norm_row, cfg.n_embd, &mut q8_buf, &mut scale_buf);

            matmul_q8_0_quantized_parallel_rows(
                layer.wq,
                &q8_buf,
                &scale_buf,
                &mut q_all[q_off..q_off + n_embd_q],
                cfg.n_embd,
                n_embd_q,
                0,
                1,
            );
            matmul_q8_0_quantized_parallel_rows(
                layer.wk,
                &q8_buf,
                &scale_buf,
                &mut k_all[k_off..k_off + n_embd_k],
                cfg.n_embd,
                n_embd_k,
                0,
                1,
            );
            matmul_q8_0_quantized_parallel_rows(
                layer.wv,
                &q8_buf,
                &scale_buf,
                &mut v_all[v_off..v_off + n_embd_v],
                cfg.n_embd,
                n_embd_v,
                0,
                1,
            );
        }

        if let (Some(q_norm), Some(k_norm)) = (layer.q_norm.as_deref(), layer.k_norm.as_deref()) {
            for head in 0..cfg.n_head {
                let off = head * cfg.n_embd_head_k;
                rms_norm_inplace(&mut q_all[off..off + cfg.n_embd_head_k], q_norm, cfg.eps);
            }
            for head in 0..cfg.n_head_kv {
                let off = head * cfg.n_embd_head_k;
                rms_norm_inplace(&mut k_all[off..off + cfg.n_embd_head_k], k_norm, cfg.eps);
            }
        }

        for tok in 0..n_tokens {
            let pos = positions[tok];
            for head in 0..cfg.n_head {
                let off = head * cfg.n_embd_head_k;
                let q_slice = &mut q_all[off..off + cfg.n_embd_head_k];
                match cfg.rope {
                    Qwen3Rope::Neox => {
                        rope_neox(q_slice, pos[0], cfg.n_embd_head_k, cfg.freq_base);
                    }
                    Qwen3Rope::Interleaved { sections, n_dims } => {
                        rope_mrope_interleaved(
                            q_slice,
                            pos,
                            sections,
                            cfg.n_embd_head_k,
                            cfg.freq_base,
                            n_dims,
                        );
                    }
                }
            }
            for head in 0..cfg.n_head_kv {
                let off = head * cfg.n_embd_head_k;
                let k_slice = &mut k_all[off..off + cfg.n_embd_head_k];
                match cfg.rope {
                    Qwen3Rope::Neox => {
                        rope_neox(k_slice, pos[0], cfg.n_embd_head_k, cfg.freq_base);
                    }
                    Qwen3Rope::Interleaved { sections, n_dims } => {
                        rope_mrope_interleaved(
                            k_slice,
                            pos,
                            sections,
                            cfg.n_embd_head_k,
                            cfg.freq_base,
                            n_dims,
                        );
                    }
                }
            }
        }

        let mut attn_out = vec![0.0; n_tokens * n_attn];
        for head in 0..cfg.n_head {
            let kv_head = head / group_size;
            let q_off = head * cfg.n_embd_head_k;
            let k_off = kv_head * cfg.n_embd_head_k;
            let v_off = kv_head * cfg.n_embd_head_v;
            let attn_off = head * cfg.n_embd_head_v;

            for i in 0..n_tokens {
                let mut max_val = f32::NEG_INFINITY;
                let mut scores = vec![0.0; n_tokens];
                for j in 0..=i {
                    let q_row = &q_all[i * n_embd_q + q_off..i * n_embd_q + q_off + cfg.n_embd_head_k];
                    let k_row = &k_all[j * n_embd_k + k_off..j * n_embd_k + k_off + cfg.n_embd_head_k];
                    scores[j] = dot_f32(q_row, k_row, cfg.n_embd_head_k) * kq_scale;
                    if scores[j] > max_val {
                        max_val = scores[j];
                    }
                }
                let mut exp_sum = 0.0f32;
                for j in 0..=i {
                    scores[j] = (scores[j] - max_val).exp();
                    exp_sum += scores[j];
                }
                for j in 0..=i {
                    scores[j] /= exp_sum;
                }
                for dim in 0..cfg.n_embd_head_v {
                    let mut sum = 0.0f32;
                    for j in 0..=i {
                        let v_row = &v_all[j * n_embd_v + v_off..j * n_embd_v + v_off + cfg.n_embd_head_v];
                        sum += scores[j] * v_row[dim];
                    }
                    attn_out[i * n_attn + attn_off + dim] = sum;
                }
            }
        }

        let mut attn_proj_out = vec![0.0; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let attn_row = &attn_out[tok * n_attn..tok * n_attn + n_attn];
            let blocks = (n_attn + 31) / 32;
            let mut q8_buf = vec![0u8; n_attn];
            let mut scale_buf = vec![0.0f32; blocks];
            quantize_q8_0_into(attn_row, n_attn, &mut q8_buf, &mut scale_buf);
            matmul_q8_0_quantized_parallel_rows(
                layer.wo,
                &q8_buf,
                &scale_buf,
                &mut attn_proj_out[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd],
                n_attn,
                cfg.n_embd,
                0,
                1,
            );
        }

        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            for j in 0..cfg.n_embd {
                hidden[off + j] += attn_proj_out[off + j];
            }
        }

        let mut ffn_normed = vec![0.0; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            rms_norm(
                &hidden[off..off + cfg.n_embd],
                &layer.ffn_norm,
                &mut ffn_normed[off..off + cfg.n_embd],
                cfg.eps,
            );
        }

        let mut gate_buf = vec![0.0; n_tokens * cfg.n_ff];
        let mut up_buf = vec![0.0; n_tokens * cfg.n_ff];
        for tok in 0..n_tokens {
            let ffn_row = &ffn_normed[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd];
            let blocks = (cfg.n_embd + 31) / 32;
            let mut q8_buf = vec![0u8; cfg.n_embd];
            let mut scale_buf = vec![0.0f32; blocks];
            quantize_q8_0_into(ffn_row, cfg.n_embd, &mut q8_buf, &mut scale_buf);
            matmul_q8_0_quantized_parallel_rows(
                layer.w_gate,
                &q8_buf,
                &scale_buf,
                &mut gate_buf[tok * cfg.n_ff..tok * cfg.n_ff + cfg.n_ff],
                cfg.n_embd,
                cfg.n_ff,
                0,
                1,
            );
            matmul_q8_0_quantized_parallel_rows(
                layer.w_up,
                &q8_buf,
                &scale_buf,
                &mut up_buf[tok * cfg.n_ff..tok * cfg.n_ff + cfg.n_ff],
                cfg.n_embd,
                cfg.n_ff,
                0,
                1,
            );
        }

        for tok in 0..n_tokens {
            let off = tok * cfg.n_ff;
            for i in 0..cfg.n_ff {
                gate_buf[off + i] = silu(gate_buf[off + i]) * up_buf[off + i];
            }
        }

        let mut down_buf = vec![0.0; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let blocks = (cfg.n_ff + 31) / 32;
            let mut q8_buf = vec![0u8; cfg.n_ff];
            let mut scale_buf = vec![0.0f32; blocks];
            quantize_q8_0_into(
                &gate_buf[tok * cfg.n_ff..tok * cfg.n_ff + cfg.n_ff],
                cfg.n_ff,
                &mut q8_buf,
                &mut scale_buf,
            );
            matmul_q8_0_quantized_parallel_rows(
                layer.w_down,
                &q8_buf,
                &scale_buf,
                &mut down_buf[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd],
                cfg.n_ff,
                cfg.n_embd,
                0,
                1,
            );
        }

        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            for i in 0..cfg.n_embd {
                hidden[off + i] += down_buf[off + i];
            }
        }
    }

    let mut output = vec![0.0; n_tokens * cfg.n_embd];
    for tok in 0..n_tokens {
        let off = tok * cfg.n_embd;
        rms_norm(
            &hidden[off..off + cfg.n_embd],
            &model.output_norm,
            &mut output[off..off + cfg.n_embd],
            cfg.eps,
        );
    }

    Ok(output)
}