// GQA self-attention with per-head RMSNorm on q/k, scalar gate, no RoPE.
//
// Two layouts supported via flags:
//   - self-attn (encoder/decoder):  GQA 4:1, 8 query heads, 2 kv heads
//   - cross-attn (decoder only):    full MHA, 8 query heads, 8 kv heads
//
// Per-head RMSNorm shape: [n_heads, head_dim_q]. head_dim_v == 64.
//
// Layout convention (matches safetensors / GGUF row-major):
//   q_proj: [HIDDEN, n_heads_q * head_dim_q]
//   k_proj: [HIDDEN, n_kv_heads * head_dim_k]
//   v_proj: [HIDDEN, n_kv_heads * head_dim_v]
//   out_proj: [HIDDEN, HIDDEN]

use super::{HEAD_DIM_K, HEAD_DIM_Q, HEAD_DIM_V, HIDDEN, NUM_HEADS, NUM_KV_HEADS};

/// Apply per-head ZCRMSNorm where the scale is **shared** across heads
/// (single `[head_dim]` vector). The norm is computed per-head slice,
/// then the same scale is applied to every head.
///
/// Real algorithm (per cactus-compute/needle `architecture.py`
/// `ZCRMSNorm`):
///   out = (1 + scale) * x / rms(x)
#[inline]
fn apply_qk_norm(proj: &[f32], scale: &[f32], n_heads: usize, head_dim: usize, out: &mut [f32]) {
    debug_assert_eq!(proj.len(), n_heads * head_dim);
    debug_assert_eq!(scale.len(), head_dim);
    debug_assert_eq!(out.len(), n_heads * head_dim);

    for h in 0..n_heads {
        let base = h * head_dim;
        let chunk = &proj[base..base + head_dim];

        let mut sum_sq = 0.0f32;
        for &v in chunk {
            sum_sq += v * v;
        }
        let rms = 1.0f32 / (sum_sq / head_dim as f32 + 1e-6).sqrt();

        for i in 0..head_dim {
            out[base + i] = (1.0 + scale[i]) * chunk[i] * rms;
        }
    }
}

fn matmul(input: &[f32], weight: &[f32], out_dim: usize, output: &mut [f32]) {
    debug_assert_eq!(input.len(), HIDDEN);
    debug_assert_eq!(weight.len(), HIDDEN * out_dim);
    debug_assert_eq!(output.len(), out_dim);
    // Inner product = `input @ weight.T` (PyTorch convention), so for each
    // output column o, accumulate over i: input[i] * weight[i, o].
    for o in 0..out_dim {
        let mut acc = 0.0f32;
        for i in 0..HIDDEN {
            acc += input[i] * weight[i * out_dim + o];
        }
        output[o] = acc;
    }
}

/// Self-attention GQA 4:1 forward (no RoPE, no taps — used by the encoder
/// where architecture.py does not apply them).
///
/// `input`: `[B, T, HIDDEN]` contiguous.
/// Writes the attended output `[B, T, HIDDEN]` to `output`.
pub fn gqa_self_attention(
    input: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_out: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    attn_gate: f32,
    batch: usize,
    seq_len: usize,
    output: &mut [f32],
) {
    let q_dim = NUM_HEADS * HEAD_DIM_Q;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM_K;
    let v_per_head = HEAD_DIM_V;
    let rep = NUM_HEADS / NUM_KV_HEADS;

    for b in 0..batch {
        for t in 0..seq_len {
            let tok_base = (b * seq_len + t) * HIDDEN;
            let tok = &input[tok_base..tok_base + HIDDEN];

            let mut q_raw = vec![0.0f32; q_dim];
            let mut k_raw = vec![0.0f32; kv_dim];
            let mut v_raw = vec![0.0f32; NUM_KV_HEADS * v_per_head];
            matmul(tok, w_q, q_dim, &mut q_raw);
            matmul(tok, w_k, kv_dim, &mut k_raw);
            matmul(tok, w_v, NUM_KV_HEADS * v_per_head, &mut v_raw);

            let mut q_normed = vec![0.0f32; q_dim];
            apply_qk_norm(&q_raw, q_norm, NUM_HEADS, HEAD_DIM_Q, &mut q_normed);

            // Per-position k/v projection + per-head norm
            let mut k_all = vec![0.0f32; seq_len * kv_dim];
            let mut v_all = vec![0.0f32; seq_len * NUM_KV_HEADS * v_per_head];
            let mut k_normed = vec![0.0f32; seq_len * kv_dim];
            for s in 0..seq_len {
                let src_base = (b * seq_len + s) * HIDDEN;
                let src = &input[src_base..src_base + HIDDEN];
                let mut k_buf = vec![0.0f32; kv_dim];
                let mut v_buf = vec![0.0f32; NUM_KV_HEADS * v_per_head];
                matmul(src, w_k, kv_dim, &mut k_buf);
                matmul(src, w_v, NUM_KV_HEADS * v_per_head, &mut v_buf);
                k_all[s * kv_dim..(s + 1) * kv_dim].copy_from_slice(&k_buf);
                v_all[s * (NUM_KV_HEADS * v_per_head)..(s + 1) * (NUM_KV_HEADS * v_per_head)]
                    .copy_from_slice(&v_buf);

                let mut normed = vec![0.0f32; kv_dim];
                apply_qk_norm(&k_buf, k_norm, NUM_KV_HEADS, HEAD_DIM_K, &mut normed);
                k_normed[s * kv_dim..(s + 1) * kv_dim].copy_from_slice(&normed);
            }

            // Attention over full sequence
            let mut head_out = vec![0.0f32; NUM_HEADS * v_per_head];
            for h in 0..NUM_HEADS {
                let kv_h = h / rep;
                let q = &q_normed[h * HEAD_DIM_Q..(h + 1) * HEAD_DIM_Q];
                let k_start = kv_h * HEAD_DIM_K;
                let v_start = kv_h * v_per_head;

                let mut scores = vec![0.0f32; seq_len];
                for s in 0..seq_len {
                    let k_src = &k_normed[s * kv_dim + k_start..s * kv_dim + k_start + HEAD_DIM_K];
                    let mut dot = 0.0f32;
                    for i in 0..HEAD_DIM_K {
                        dot += q[i] * k_src[i];
                    }
                    scores[s] = dot / (HEAD_DIM_Q as f32).sqrt();
                }
                let max = scores.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
                let mut weights = vec![0.0f32; seq_len];
                let mut denom = 0.0f32;
                for s in 0..seq_len {
                    let e = (scores[s] - max).exp();
                    weights[s] = e;
                    denom += e;
                }
                for s in 0..seq_len {
                    weights[s] /= denom;
                }

                let out_base = h * v_per_head;
                for i in 0..v_per_head {
                    let mut acc = 0.0f32;
                    for s in 0..seq_len {
                        let v_base = s * (NUM_KV_HEADS * v_per_head) + v_start;
                        acc += weights[s] * v_all[v_base + i];
                    }
                    head_out[out_base + i] = acc;
                }
            }

            // Out projection
            let mut proj = vec![0.0f32; HIDDEN];
            matmul(&head_out, w_out, HIDDEN, &mut proj);

            for i in 0..HIDDEN {
                output[tok_base + i] = attn_gate * proj[i];
            }
        }
    }
}

/// Self-attention GQA 4:1 with 3-tap causal FIR on q/k/v and RoPE on
/// q/k. Taps tensor layout is `[n_taps, dim]` and applied per token
/// along the seq axis as `out = Σ_j taps[j] * _shift_right(input, j)`
/// (causal, zero-fill on the left). RoPE cos/sin precomputed as
/// `[seq_len, head_dim/2]`; for each position t and head_dim index i:
///   out[..., :half] = x1 * cos[t, i] - x2 * sin[t, i]
///   out[..., half:] = x2 * cos[t, i] + x1 * sin[t, i]
/// where x1, x2 are the head-dim halves of the per-head vector.
pub fn gqa_self_attention_with_taps_and_rope(
    input: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_out: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    attn_gate: f32,
    batch: usize,
    seq_len: usize,
    q_taps: &[f32],
    k_taps: &[f32],
    v_taps: &[f32],
    cos: &[f32],
    sin: &[f32],
    output: &mut [f32],
) {
    let q_dim = NUM_HEADS * HEAD_DIM_Q;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM_K;
    let v_per_head = HEAD_DIM_V;
    let rep = NUM_HEADS / NUM_KV_HEADS;
    let n_taps = 3;
    debug_assert_eq!(q_taps.len(), n_taps * q_dim);
    debug_assert_eq!(k_taps.len(), n_taps * kv_dim);
    debug_assert_eq!(v_taps.len(), n_taps * (NUM_KV_HEADS * v_per_head));
    debug_assert_eq!(cos.len(), seq_len * (HEAD_DIM_Q / 2));
    debug_assert_eq!(sin.len(), seq_len * (HEAD_DIM_Q / 2));
    debug_assert_eq!(output.len(), batch * seq_len * HIDDEN);

// Pre-compute q/k/v for every (batch, position) token once, then
    // apply the causal FIR to position `t`. This matches the closed
    // Cactus runtime's layout: `q_raw[t]` reads from `[b, t, q_dim]`.
    let mut q_all = vec![0.0f32; batch * seq_len * q_dim];
    let mut k_all = vec![0.0f32; batch * seq_len * kv_dim];
    let mut v_all = vec![0.0f32; batch * seq_len * (NUM_KV_HEADS * v_per_head)];
    let mut q_normed_buf = vec![0.0f32; batch * seq_len * q_dim];
    let mut k_normed_buf = vec![0.0f32; batch * seq_len * kv_dim];
    let mut v_filt_buf = vec![0.0f32; batch * seq_len * (NUM_KV_HEADS * v_per_head)];
    for b in 0..batch {
        for s in 0..seq_len {
            let tok_base = (b * seq_len + s) * HIDDEN;
            let tok = &input[tok_base..tok_base + HIDDEN];
            matmul(
                tok,
                w_q,
                q_dim,
                &mut q_all[(b * seq_len + s) * q_dim..(b * seq_len + s + 1) * q_dim],
            );
            matmul(
                tok,
                w_k,
                kv_dim,
                &mut k_all[(b * seq_len + s) * kv_dim..(b * seq_len + s + 1) * kv_dim],
            );
            matmul(
                tok,
                w_v,
                NUM_KV_HEADS * v_per_head,
                &mut v_all[(b * seq_len + s) * (NUM_KV_HEADS * v_per_head)
                    ..(b * seq_len + s + 1) * (NUM_KV_HEADS * v_per_head)],
            );
        }
    }

    for b in 0..batch {
        for t in 0..seq_len {
            let tok_base = (b * seq_len + t) * HIDDEN;

            // Apply 3-tap causal FIR to q/k/v (no-op for tap j > t).
            let mut q_filt = vec![0.0f32; q_dim];
            let mut k_filt = vec![0.0f32; kv_dim];
            let mut v_filt = vec![0.0f32; NUM_KV_HEADS * v_per_head];
            for j in 0..n_taps {
                if j > t {
                    continue;
                }
                let q_src_off = (b * seq_len + (t - j)) * q_dim;
                let k_src_off = (b * seq_len + (t - j)) * kv_dim;
                let v_src_off = (b * seq_len + (t - j)) * (NUM_KV_HEADS * v_per_head);
                let tap_row_q = &q_taps[j * q_dim..(j + 1) * q_dim];
                let tap_row_k = &k_taps[j * kv_dim..(j + 1) * kv_dim];
                let tap_row_v = &v_taps[j * (NUM_KV_HEADS * v_per_head)
                    ..(j + 1) * (NUM_KV_HEADS * v_per_head)];
                for o in 0..q_dim {
                    q_filt[o] += tap_row_q[o] * q_all[q_src_off + o];
                }
                for o in 0..kv_dim {
                    k_filt[o] += tap_row_k[o] * k_all[k_src_off + o];
                }
                for o in 0..(NUM_KV_HEADS * v_per_head) {
                    v_filt[o] += tap_row_v[o] * v_all[v_src_off + o];
                }
            }
            let mut q_normed = vec![0.0f32; q_dim];
            apply_qk_norm(&q_filt, q_norm, NUM_HEADS, HEAD_DIM_Q, &mut q_normed);
            let mut k_normed = vec![0.0f32; kv_dim];
            apply_qk_norm(&k_filt, k_norm, NUM_KV_HEADS, HEAD_DIM_K, &mut k_normed);
            // Store q/k per-position for reshape+RoPE outside the t loop.
            let pos_off_q = (b * seq_len + t) * q_dim;
            let pos_off_k = (b * seq_len + t) * kv_dim;
            let pos_off_v = (b * seq_len + t) * (NUM_KV_HEADS * v_per_head);
            q_normed_buf[pos_off_q..pos_off_q + q_dim].copy_from_slice(&q_normed);
            k_normed_buf[pos_off_k..pos_off_k + kv_dim].copy_from_slice(&k_normed);
            v_filt_buf[pos_off_v..pos_off_v + NUM_KV_HEADS * v_per_head].copy_from_slice(&v_filt);
        }
    }

    // Reshape [B, T, q_dim] -> [H, T, head_dim] and apply RoPE per position.
    let half = HEAD_DIM_Q / 2;
    let mut head_q = vec![0.0f32; NUM_HEADS * seq_len * HEAD_DIM_Q];
    let mut head_k = vec![0.0f32; NUM_KV_HEADS * seq_len * HEAD_DIM_Q];
    for b in 0..batch {
        for t in 0..seq_len {
            for h in 0..NUM_HEADS {
                let src = &q_normed_buf[(b * seq_len + t) * q_dim + h * HEAD_DIM_Q
                    ..(b * seq_len + t) * q_dim + (h + 1) * HEAD_DIM_Q];
                let dst = (h * seq_len + t) * HEAD_DIM_Q;
                head_q[dst..dst + HEAD_DIM_Q].copy_from_slice(src);
            }
            for h in 0..NUM_KV_HEADS {
                let src = &k_normed_buf[(b * seq_len + t) * kv_dim + h * HEAD_DIM_K
                    ..(b * seq_len + t) * kv_dim + (h + 1) * HEAD_DIM_K];
                let dst = (h * seq_len + t) * HEAD_DIM_Q;
                head_k[dst..dst + HEAD_DIM_Q].copy_from_slice(src);
            }
        }
    }
    for h in 0..NUM_HEADS {
        for s in 0..seq_len {
            let dst = (h * seq_len + s) * HEAD_DIM_Q;
            for i in 0..half {
                let c = cos[s * half + i];
                let sn = sin[s * half + i];
                let x1 = head_q[dst + i];
                let x2 = head_q[dst + half + i];
                head_q[dst + i] = x1 * c - x2 * sn;
                head_q[dst + half + i] = x2 * c + x1 * sn;
            }
        }
    }
    for h in 0..NUM_KV_HEADS {
        for s in 0..seq_len {
            let dst = (h * seq_len + s) * HEAD_DIM_Q;
            for i in 0..half {
                let c = cos[s * half + i];
                let sn = sin[s * half + i];
                let x1 = head_k[dst + i];
                let x2 = head_k[dst + half + i];
                head_k[dst + i] = x1 * c - x2 * sn;
                head_k[dst + half + i] = x2 * c + x1 * sn;
            }
        }
    }

    for b in 0..batch {
        for t in 0..seq_len {
            let tok_base = (b * seq_len + t) * HIDDEN;
            // Attention over full sequence (GQA: each q head sees one kv head
            // repeated `rep` times).
            let mut head_out = vec![0.0f32; NUM_HEADS * v_per_head];
            for h in 0..NUM_HEADS {
                let kv_h = h / rep;
                let q_off = (h * seq_len + t) * HEAD_DIM_Q;
                let mut scores = vec![0.0f32; seq_len];
                for s in 0..seq_len {
                    let k_off = (kv_h * seq_len + s) * HEAD_DIM_Q;
                    let mut dot = 0.0f32;
                    for i in 0..HEAD_DIM_Q {
                        dot += head_q[q_off + i] * head_k[k_off + i];
                    }
                    scores[s] = dot / (HEAD_DIM_Q as f32).sqrt();
                }
                let max = scores.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
                let mut weights = vec![0.0f32; seq_len];
                let mut denom = 0.0f32;
                for s in 0..seq_len {
                    let e = (scores[s] - max).exp();
                    weights[s] = e;
                    denom += e;
                }
                for s in 0..seq_len {
                    weights[s] /= denom;
                }
                let out_base = h * v_per_head;
                for i in 0..v_per_head {
                    let mut acc = 0.0f32;
                    for s in 0..seq_len {
                        // v_filt_buf is stored as [B, T, n_kv, V] (matching the
                        // np reshape layout v.reshape(b, T, n_kv, V).transpose
                        // (0,2,1,3) means the head_k/head_q read order needs the
                        // [B, H, T, V] index, but the storage is still
                        // [B, T, n_kv, V]).
                        let v_off = (s * NUM_KV_HEADS + kv_h) * v_per_head + i;
                        acc += weights[s] * v_filt_buf[v_off];
                    }
                    head_out[out_base + i] = acc;
                }
            }

            let mut proj = vec![0.0f32; HIDDEN];
            matmul(&head_out, w_out, HIDDEN, &mut proj);

            for i in 0..HIDDEN {
                output[tok_base + i] = attn_gate * proj[i];
            }
        }
    }
}

/// Cross-attention (decoder): full MHA, 8 query heads, 8 kv heads.
pub fn gqa_cross_attention(
    query_input: &[f32],
    kv_input: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_out: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    attn_gate: f32,
    batch: usize,
    q_seq: usize,
    kv_seq: usize,
    output: &mut [f32],
) {
    let q_dim = NUM_HEADS * HEAD_DIM_Q;
    let kv_dim = NUM_HEADS * HEAD_DIM_K;
    let v_per_head = HEAD_DIM_V;

    for b in 0..batch {
        for t in 0..q_seq {
            let tok_base = (b * q_seq + t) * HIDDEN;
            let tok = &query_input[tok_base..tok_base + HIDDEN];

            let mut q_raw = vec![0.0f32; q_dim];
            matmul(tok, w_q, q_dim, &mut q_raw);

            let mut k_all = vec![0.0f32; kv_seq * kv_dim];
            let mut v_all = vec![0.0f32; kv_seq * NUM_HEADS * v_per_head];
            for s in 0..kv_seq {
                let kv_base = (b * kv_seq + s) * HIDDEN;
                let kv_tok = &kv_input[kv_base..kv_base + HIDDEN];
                let mut k_buf = vec![0.0f32; kv_dim];
                let mut v_buf = vec![0.0f32; NUM_HEADS * v_per_head];
                matmul(kv_tok, w_k, kv_dim, &mut k_buf);
                matmul(kv_tok, w_v, NUM_HEADS * v_per_head, &mut v_buf);
                k_all[s * kv_dim..(s + 1) * kv_dim].copy_from_slice(&k_buf);
                v_all[s * (NUM_HEADS * v_per_head)..(s + 1) * (NUM_HEADS * v_per_head)]
                    .copy_from_slice(&v_buf);
            }

            let mut q_normed = vec![0.0f32; q_dim];
            apply_qk_norm(&q_raw, q_norm, NUM_HEADS, HEAD_DIM_Q, &mut q_normed);

            // Per-position k/v norm
            let mut k_normed = vec![0.0f32; kv_seq * kv_dim];
            for s in 0..kv_seq {
                let mut normed = vec![0.0f32; kv_dim];
                let k_chunk = &k_all[s * kv_dim..(s + 1) * kv_dim];
                apply_qk_norm(k_chunk, k_norm, NUM_HEADS, HEAD_DIM_K, &mut normed);
                k_normed[s * kv_dim..(s + 1) * kv_dim].copy_from_slice(&normed);
            }

            let mut head_out = vec![0.0f32; NUM_HEADS * v_per_head];
            for h in 0..NUM_HEADS {
                let q = &q_normed[h * HEAD_DIM_Q..(h + 1) * HEAD_DIM_Q];
                let k_start = h * HEAD_DIM_K;
                let v_start = h * v_per_head;

                let mut scores = vec![0.0f32; kv_seq];
                for s in 0..kv_seq {
                    let k_src = &k_normed[s * kv_dim + k_start..s * kv_dim + k_start + HEAD_DIM_K];
                    let mut dot = 0.0f32;
                    for i in 0..HEAD_DIM_K {
                        dot += q[i] * k_src[i];
                    }
                    scores[s] = dot / (HEAD_DIM_Q as f32).sqrt();
                }
                let max = scores.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
                let mut weights = vec![0.0f32; kv_seq];
                let mut denom = 0.0f32;
                for s in 0..kv_seq {
                    let e = (scores[s] - max).exp();
                    weights[s] = e;
                    denom += e;
                }
                for s in 0..kv_seq {
                    weights[s] /= denom;
                }

                let out_base = h * v_per_head;
                for i in 0..v_per_head {
                    let mut acc = 0.0f32;
                    for s in 0..kv_seq {
                        let v_base = s * (NUM_HEADS * v_per_head) + v_start;
                        acc += weights[s] * v_all[v_base + i];
                    }
                    head_out[out_base + i] = acc;
                }
            }

            let mut proj = vec![0.0f32; HIDDEN];
            matmul(&head_out, w_out, HIDDEN, &mut proj);

            for i in 0..HIDDEN {
                output[tok_base + i] = attn_gate * proj[i];
            }
        }
    }
}
