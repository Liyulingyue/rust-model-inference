//! Gemma3 decoder-only forward (BitNet b1.58 + standard variants).
//!
//! Implements the architecture described in the module-level docs
//! of [`crate::models::gemma3`]. The forward loop is laid out as:
//!
//! ```text
//! for each layer:
//!     h_norm = rms_norm(h, attn_norm)
//!     q, k, v = projection(h_norm)            # BitLinear OR Q4_K matmul
//!     q = qk_norm(q, q_norm); k = qk_norm(k, k_norm)
//!     q = rope(q); k = rope(k)
//!     attn = causal_self_attention(q, k, v, sliding_window)   # GQA repeat
//!     attn_out = projection(attn, attn_output)
//!     h = rms_norm(attn_out, post_attention_norm) + h
//!     ffn_in = rms_norm(h, ffn_norm)
//!     gate = projection(ffn_in, ffn_gate)
//!     up   = projection(ffn_in, ffn_up)
//!     ffn_out = projection(silu(gate)*up, ffn_down)
//!     h = rms_norm(ffn_out, post_ffw_norm) + h
//! h_final = rms_norm(h, output_norm)
//! ```
//!
//! Two forward paths share the same loop body:
//!
//! - **BitNet b1.58** (`cfg.is_bitnet == true`): every projection
//!   goes through [`crate::ops::bitnet::bitlinear_forward_packed`]
//!   (pre-dequant `{-1, 0, +1}` int8 SIMD path; SIMD hot path
//!   added in afbb172). The `weight` slot in each layer is unused.
//! - **Standard** (`cfg.is_bitnet == false`): every projection goes
//!   through the standard Q8_0-quantized-activation × quantized-weight
//!   matmul (mixed Q4_K / Q5_0 / Q6K / Q8_0 / F32 matmul weights, the
//!   same kernels the qwen3 trunk uses). The `bitlinear` slot is
//!   `BitLinearSlot::default()`.
//!
//! Both paths apply the same 4-norm sandwich (attn_norm /
//! post_attention_norm / ffn_norm / post_ffw_norm) and the same QK-norm
//! before RoPE. Sliding-window attention (declared as
//! `gemma3.attention.sliding_window` in standard gemma3 GGUFs,
//! omitted in BitNet 270M) is applied in [`causal_self_attention`].

use super::config::{Gemma3Config, Gemma3Rope};
use super::weights::{BitLinearWeights, Gemma3LayerWeights, Gemma3Model, Weight};
use crate::ops::bitnet::{bitlinear_forward, quantize_activation_per_token};
use crate::ops::quantize_q8_0_into;
use crate::ops::rope::rope_neox_inplace_with_factor;

/// BitLinear per-projection: rms_norm → absmax int8 quant →
/// ternary matmul → rescale. Same scalar reference impl as
/// `qwen3::trunk::forward::bitlinear_projection` — duplicated
/// here so the gemma3 trunk doesn't have to reach across.
fn bitlinear_projection(
    input: &[f32],
    proj: &BitLinearWeights,
    output: &mut [f32],
    eps: f32,
) {
    debug_assert_eq!(input.len(), proj.n_in);
    debug_assert_eq!(output.len(), proj.n_out);
    let n_in = proj.n_in;
    let mut normed = vec![0.0f32; n_in];
    crate::ops::norm::rms_norm(input, &proj.norm_in, &mut normed, eps);
    let (x_q, absmax) = quantize_activation_per_token(&normed);
    bitlinear_forward(&proj.weight, &x_q, absmax, n_in, proj.n_out, output);
}

/// Standard (non-BitLinear) per-projection: Q8_0-quantize the F32
/// activation × quantized weight → output. Mirrors the matmul
/// pattern in [`crate::models::qwen3::trunk::forward::text_encode`]
/// (per-token `quantize_q8_0_into` then
/// `weight.kernel.forward_prepared(...)`).
///
/// `weight_bytes` holds the GGUF payload for one projection; the
/// kernel is constructed on-demand from the byte buffer + ggml
/// type. Construction is cheap (the kernel just holds the `&[u8]`
/// view of the buffer + dimensions), so this avoids storing one
/// `Weight` per layer (which would force a `'static` byte buffer
/// owned by the model — `Gemma3Model` is rebuilt each forward call
/// from the borrowed `&dyn TensorSource`, so we can't keep
/// `Weight<'static>` alive without duplicating the GGUF payload).
///
/// `input.len() == n_in`, `output.len() == n_out`. Allocates one
/// `q8_buf: Vec<u8>(n_in)` and one `scale_buf: Vec<f32>((n_in +
/// 31) / 32)` per call (cheap — `n_in` is 640/1024/3072 in 270M
/// 4B/12B/27B respectively).
fn standard_projection(
    input: &[f32],
    weight_bytes: &[u8],
    weight_type: crate::core::tensor::GGMLType,
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
) {
    let weight = Weight::from_quantized(crate::ops::kernel::QuantizedTensor::from_bytes(
        weight_bytes,
        weight_type,
        n_in,
        n_out,
    ));
    let blocks = (n_in + 31) / 32;
    let mut q8_buf = vec![0u8; n_in];
    let mut scale_buf = vec![0.0f32; blocks];
    quantize_q8_0_into(input, n_in, &mut q8_buf, &mut scale_buf);
    weight.kernel.forward_prepared(
        input,
        &q8_buf,
        &scale_buf,
        None,
        output,
        n_in,
        n_out,
        0,
        1,
    );
}

/// Per-head RMSNorm applied to Q (and K) before RoPE.
///
/// Layout assumption: Q/K are stored row-major as
/// `[n_tokens, n_heads, head_dim]`. The RMSNorm is over each
/// `head_dim`-wide slice independently. 270M declares
/// `gemma3.rope.dimension_count = 256` (= `head_dim`), so the
/// norm covers the full head.
fn apply_qk_norm(
    x: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    gain: &[f32],
    eps: f32,
) {
    assert_eq!(gain.len(), head_dim);
    for tok in 0..n_tokens {
        for head in 0..n_heads {
            let off = tok * n_heads * head_dim + head * head_dim;
            let row = &mut x[off..off + head_dim];
            let mut mean_sq = 0.0f32;
            for v in row.iter() {
                mean_sq += v * v;
            }
            mean_sq /= head_dim as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for (v, g) in row.iter_mut().zip(gain) {
                *v = *v * inv * *g;
            }
        }
    }
}

/// Standard causal self-attention with GQA (kv-head repeat).
///
/// `n_attn = n_head * n_embd_head_v`. `group_size = n_head / n_head_kv`.
/// Each Q-head `h` reads from KV-head `h / group_size`.
///
/// `sliding_window` (0 = full causal) optionally restricts each
/// query to the most recent `sliding_window` keys: a key at
/// position `j` is masked out for query `i` when
/// `i - j > sliding_window`. Causal ordering is preserved
/// independently (j > i is always masked). Standard gemma3 270M-it
/// declares `sliding_window = 512`; BitNet 270M does not declare
/// this metadata and is treated as 0.
fn causal_self_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_tokens: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim_k: usize,
    head_dim_v: usize,
    kq_scale: f32,
    sliding_window: usize,
    attn_out: &mut [f32],
) {
    let n_embd_q = n_head * head_dim_k;
    let n_embd_k = n_head_kv * head_dim_k;
    let n_embd_v = n_head_kv * head_dim_v;
    let n_attn = n_head * head_dim_v;
    let group_size = n_head / n_head_kv;
    for head in 0..n_head {
        let kv_head = head / group_size;
        let q_off = head * head_dim_k;
        let k_off = kv_head * head_dim_k;
        let v_off = kv_head * head_dim_v;
        let attn_off = head * head_dim_v;

        for i in 0..n_tokens {
            let q_row = &q[i * n_embd_q + q_off..i * n_embd_q + q_off + head_dim_k];
            // The first key position this query is allowed to
            // attend to. The sliding-window lower bound and the
            // causal lower bound (j ≤ i) both apply.
            let j_start = if sliding_window > 0 && i + 1 > sliding_window {
                i + 1 - sliding_window
            } else {
                0
            };
            let mut max_val = f32::NEG_INFINITY;
            let mut scores = vec![0.0f32; n_tokens];
            for j in j_start..=i {
                let k_row = &k[j * n_embd_k + k_off..j * n_embd_k + k_off + head_dim_k];
                let mut dot = 0.0f32;
                for d in 0..head_dim_k {
                    dot += q_row[d] * k_row[d];
                }
                let s = dot * kq_scale;
                scores[j] = s;
                if s > max_val {
                    max_val = s;
                }
            }
            // Causal guarantees j = i is in range, so max_val is
            // always finite here. The defensive fallback is just
            // belt-and-suspenders for any future sliding-window
            // tweaks.
            if !max_val.is_finite() {
                max_val = 0.0;
            }
            let mut exp_sum = 0.0f32;
            for j in j_start..=i {
                scores[j] = (scores[j] - max_val).exp();
                exp_sum += scores[j];
            }
            for j in j_start..=i {
                scores[j] /= exp_sum;
            }
            for dim in 0..head_dim_v {
                let mut sum = 0.0f32;
                for j in j_start..=i {
                    let v_row = &v[j * n_embd_v + v_off..j * n_embd_v + v_off + head_dim_v];
                    sum += scores[j] * v_row[dim];
                }
                attn_out[i * n_attn + attn_off + dim] = sum;
            }
        }
    }
}

/// Embedding extraction entry point. Takes a pre-loaded
/// `Gemma3Model` and a `token_ids` sequence, runs the decoder,
/// and returns the **last-token** row of the hidden state (the
/// BitNet pooling convention; `cfg.pooling_type == 1` +
/// `cfg.is_bitnet` triggers this).
pub fn text_encode(
    model: &Gemma3Model,
    token_ids: &[u32],
) -> Result<Vec<f32>, String> {
    if token_ids.is_empty() {
        return Err("gemma3::text_encode: empty token sequence".into());
    }
    let n_tokens = token_ids.len();
    let cfg = &model.config;

    let mut hidden = vec![0.0f32; n_tokens * cfg.n_embd];
    for (row, &tid) in hidden
        .chunks_exact_mut(cfg.n_embd)
        .zip(token_ids.iter())
    {
        if (tid as usize) >= cfg.vocab {
            return Err(format!(
                "gemma3::text_encode: token id {tid} >= vocab {}",
                cfg.vocab
            ));
        }
        let src = &model.token_embedding_rows[tid as usize * cfg.n_embd
            ..(tid as usize + 1) * cfg.n_embd];
        row.copy_from_slice(src);
    }

    let n_embd_q = cfg.n_embd_q();
    let n_embd_k = cfg.n_embd_kv();
    let n_embd_v = cfg.n_embd_kv();
    let n_attn = cfg.n_head * cfg.n_embd_head_v;
    let kq_scale = 1.0 / (cfg.n_embd_head_k as f32).sqrt();

    for layer_idx in 0..cfg.n_layer {
        let layer = &model.layers[layer_idx];

        let mut normed = vec![0.0f32; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            crate::ops::norm::rms_norm(
                &hidden[off..off + cfg.n_embd],
                &layer.attn_norm,
                &mut normed[off..off + cfg.n_embd],
                cfg.eps,
            );
        }

        let mut q_all = vec![0.0f32; n_tokens * n_embd_q];
        let mut k_all = vec![0.0f32; n_tokens * n_embd_k];
        let mut v_all = vec![0.0f32; n_tokens * n_embd_v];
        if cfg.is_bitnet {
            for tok in 0..n_tokens {
                let norm_row = &normed[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd];
                let q_off = tok * n_embd_q;
                let k_off = tok * n_embd_k;
                let v_off = tok * n_embd_v;
                bitlinear_projection(
                    norm_row,
                    layer.bitlinear.attn_q.as_ref().expect(
                        "gemma3 BitNet layer missing attn_q BitLinear slot",
                    ),
                    &mut q_all[q_off..q_off + n_embd_q],
                    cfg.eps,
                );
                bitlinear_projection(
                    norm_row,
                    layer
                        .bitlinear
                        .attn_k
                        .as_ref()
                        .expect("gemma3 BitNet layer missing attn_k BitLinear slot"),
                    &mut k_all[k_off..k_off + n_embd_k],
                    cfg.eps,
                );
                bitlinear_projection(
                    norm_row,
                    layer
                        .bitlinear
                        .attn_v
                        .as_ref()
                        .expect("gemma3 BitNet layer missing attn_v BitLinear slot"),
                    &mut v_all[v_off..v_off + n_embd_v],
                    cfg.eps,
                );
            }
        } else {
            // Standard Q4_K / Q5_0 / Q6K / Q8_0 matmul path
            // (mixed-quant GGUF weights, pre-quantized Q8_0
            // activations). Each projection goes through
            // `standard_projection` which dequantizes per token
            // and calls the matmul kernel.
            let wq = layer.wq.as_ref().expect("gemma3 standard layer missing wq StdProjection");
            let wk = layer.wk.as_ref().expect("gemma3 standard layer missing wk StdProjection");
            let wv = layer.wv.as_ref().expect("gemma3 standard layer missing wv StdProjection");
            for tok in 0..n_tokens {
                let norm_row = &normed[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd];
                let q_off = tok * n_embd_q;
                let k_off = tok * n_embd_k;
                let v_off = tok * n_embd_v;
                standard_projection(norm_row, &wq.bytes, wq.ggml_type, &mut q_all[q_off..q_off + n_embd_q], wq.n_in, wq.n_out);
                standard_projection(norm_row, &wk.bytes, wk.ggml_type, &mut k_all[k_off..k_off + n_embd_k], wk.n_in, wk.n_out);
                standard_projection(norm_row, &wv.bytes, wv.ggml_type, &mut v_all[v_off..v_off + n_embd_v], wv.n_in, wv.n_out);
            }
        }

        apply_qk_norm(
            &mut q_all,
            n_tokens,
            cfg.n_head,
            cfg.n_embd_head_k,
            &layer.q_norm,
            cfg.eps,
        );
        apply_qk_norm(
            &mut k_all,
            n_tokens,
            cfg.n_head_kv,
            cfg.n_embd_head_k,
            &layer.k_norm,
            cfg.eps,
        );

        for tok in 0..n_tokens {
            for head in 0..cfg.n_head {
                let off = tok * n_embd_q + head * cfg.n_embd_head_k;
                let q_slice = &mut q_all[off..off + cfg.n_embd_head_k];
                match cfg.rope {
                    Gemma3Rope::Neox => {
                        rope_neox_inplace_with_factor(
                            q_slice,
                            tok,
                            cfg.n_embd_head_k,
                            cfg.freq_base,
                            cfg.rope_factor,
                        );
                    }
                }
            }
            for head in 0..cfg.n_head_kv {
                let off = tok * n_embd_k + head * cfg.n_embd_head_k;
                let k_slice = &mut k_all[off..off + cfg.n_embd_head_k];
                match cfg.rope {
                    Gemma3Rope::Neox => {
                        rope_neox_inplace_with_factor(
                            k_slice,
                            tok,
                            cfg.n_embd_head_k,
                            cfg.freq_base,
                            cfg.rope_factor,
                        );
                    }
                }
            }
        }

        let mut attn_out = vec![0.0f32; n_tokens * n_attn];
        causal_self_attention(
            &q_all,
            &k_all,
            &v_all,
            n_tokens,
            cfg.n_head,
            cfg.n_head_kv,
            cfg.n_embd_head_k,
            cfg.n_embd_head_v,
            kq_scale,
            cfg.sliding_window,
            &mut attn_out,
        );

        let mut attn_proj_out = vec![0.0f32; n_tokens * cfg.n_embd];
        if cfg.is_bitnet {
            for tok in 0..n_tokens {
                let attn_row = &attn_out[tok * n_attn..tok * n_attn + n_attn];
                bitlinear_projection(
                    attn_row,
                    layer
                        .bitlinear
                        .attn_output
                        .as_ref()
                        .expect("gemma3 BitNet layer missing attn_output BitLinear slot"),
                    &mut attn_proj_out[tok * cfg.n_embd..(tok + 1) * cfg.n_embd],
                    cfg.eps,
                );
            }
        } else {
            let wo = layer.wo.as_ref().expect("gemma3 standard layer missing wo StdProjection");
            for tok in 0..n_tokens {
                let attn_row = &attn_out[tok * n_attn..tok * n_attn + n_attn];
                standard_projection(
                    attn_row,
                    &wo.bytes,
                    wo.ggml_type,
                    &mut attn_proj_out[tok * cfg.n_embd..(tok + 1) * cfg.n_embd],
                    n_attn,
                    cfg.n_embd,
                );
            }
        }

        let mut post_attn = vec![0.0f32; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            crate::ops::norm::rms_norm(
                &attn_proj_out[off..off + cfg.n_embd],
                &layer.post_attention_norm,
                &mut post_attn[off..off + cfg.n_embd],
                cfg.eps,
            );
            for d in 0..cfg.n_embd {
                hidden[off + d] += post_attn[off + d];
            }
        }

        let mut ffn_normed = vec![0.0f32; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            crate::ops::norm::rms_norm(
                &hidden[off..off + cfg.n_embd],
                &layer.ffn_norm,
                &mut ffn_normed[off..off + cfg.n_embd],
                cfg.eps,
            );
        }

        let mut gate_buf = vec![0.0f32; n_tokens * cfg.n_ff];
        let mut up_buf = vec![0.0f32; n_tokens * cfg.n_ff];
        if cfg.is_bitnet {
            for tok in 0..n_tokens {
                let ffn_row = &ffn_normed[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd];
                bitlinear_projection(
                    ffn_row,
                    layer
                        .bitlinear
                        .ffn_gate
                        .as_ref()
                        .expect("gemma3 BitNet layer missing ffn_gate BitLinear slot"),
                    &mut gate_buf[tok * cfg.n_ff..(tok + 1) * cfg.n_ff],
                    cfg.eps,
                );
                bitlinear_projection(
                    ffn_row,
                    layer
                        .bitlinear
                        .ffn_up
                        .as_ref()
                        .expect("gemma3 BitNet layer missing ffn_up BitLinear slot"),
                    &mut up_buf[tok * cfg.n_ff..(tok + 1) * cfg.n_ff],
                    cfg.eps,
                );
                for d in 0..cfg.n_ff {
                    let g = crate::ops::silu(gate_buf[tok * cfg.n_ff + d]);
                    gate_buf[tok * cfg.n_ff + d] = g * up_buf[tok * cfg.n_ff + d];
                }
            }
        } else {
            let w_gate = layer.w_gate.as_ref().expect("gemma3 standard layer missing w_gate StdProjection");
            let w_up = layer.w_up.as_ref().expect("gemma3 standard layer missing w_up StdProjection");
            for tok in 0..n_tokens {
                let ffn_row = &ffn_normed[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd];
                standard_projection(
                    ffn_row,
                    &w_gate.bytes,
                    w_gate.ggml_type,
                    &mut gate_buf[tok * cfg.n_ff..(tok + 1) * cfg.n_ff],
                    cfg.n_embd,
                    cfg.n_ff,
                );
                standard_projection(
                    ffn_row,
                    &w_up.bytes,
                    w_up.ggml_type,
                    &mut up_buf[tok * cfg.n_ff..(tok + 1) * cfg.n_ff],
                    cfg.n_embd,
                    cfg.n_ff,
                );
                for d in 0..cfg.n_ff {
                    let g = crate::ops::silu(gate_buf[tok * cfg.n_ff + d]);
                    gate_buf[tok * cfg.n_ff + d] = g * up_buf[tok * cfg.n_ff + d];
                }
            }
        }

        let mut ffn_out = vec![0.0f32; n_tokens * cfg.n_embd];
        if cfg.is_bitnet {
            for tok in 0..n_tokens {
                let act_row = &gate_buf[tok * cfg.n_ff..(tok + 1) * cfg.n_ff];
                bitlinear_projection(
                    act_row,
                    layer
                        .bitlinear
                        .ffn_down
                        .as_ref()
                        .expect("gemma3 BitNet layer missing ffn_down BitLinear slot"),
                    &mut ffn_out[tok * cfg.n_embd..(tok + 1) * cfg.n_embd],
                    cfg.eps,
                );
            }
        } else {
            let w_down = layer.w_down.as_ref().expect("gemma3 standard layer missing w_down StdProjection");
            for tok in 0..n_tokens {
                let act_row = &gate_buf[tok * cfg.n_ff..(tok + 1) * cfg.n_ff];
                standard_projection(
                    act_row,
                    &w_down.bytes,
                    w_down.ggml_type,
                    &mut ffn_out[tok * cfg.n_embd..(tok + 1) * cfg.n_embd],
                    cfg.n_ff,
                    cfg.n_embd,
                );
            }
        }

        let mut post_ffw = vec![0.0f32; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            crate::ops::norm::rms_norm(
                &ffn_out[off..off + cfg.n_embd],
                &layer.post_ffw_norm,
                &mut post_ffw[off..off + cfg.n_embd],
                cfg.eps,
            );
            for d in 0..cfg.n_embd {
                hidden[off + d] += post_ffw[off + d];
            }
        }
    }

    let mut final_hidden = vec![0.0f32; n_tokens * cfg.n_embd];
    for tok in 0..n_tokens {
        let off = tok * cfg.n_embd;
        crate::ops::norm::rms_norm(
            &hidden[off..off + cfg.n_embd],
            &model.output_norm,
            &mut final_hidden[off..off + cfg.n_embd],
            cfg.eps,
        );
    }

    // Last-token pooling (BitNet convention for `pooling_type=1`).
    let last_off = (n_tokens - 1) * cfg.n_embd;
    Ok(final_hidden[last_off..last_off + cfg.n_embd].to_vec())
}

/// CLI-style shared inference stub. The gemma3 trunk currently
/// only supports the embedding extraction path; text generation
/// requires a sampling loop that this session did not implement.
/// Returns an error if invoked.
pub fn run_shared_inference(
    _model: &Gemma3Model,
    _token_ids: &[u32],
) -> Result<Vec<f32>, String> {
    Err("gemma3::run_shared_inference: text generation not implemented; use compute_embedding via app::run_embedding".into())
}

// `Gemma3LayerWeights` is held by `Gemma3Model` and never
// re-exported directly; the type is referenced from forward.rs so
// `cargo` keeps the public surface coherent.
#[allow(dead_code)]
fn _layer_weights_marker(_layer: &Gemma3LayerWeights<'_>) {}
