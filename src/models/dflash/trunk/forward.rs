//! The two graphs of a DFlash / DSpark draft model.
//!
//! llama.cpp builds them in `src/models/dflash.cpp` (`graph<false>`):
//!
//! **The encoder pass** (`ubatch.embd` set) turns the target model's hidden
//! states into the draft's own key/value cache:
//!
//! ```text
//! hidden (the target's layers, concatenated) -> fc -> RMSNorm
//!     -> per layer: K = wk(.), V = wv(.), RMSNorm(K) over each head, RoPE
//!     -> written into the draft's cache at the target's positions
//! ```
//!
//! **The decoder pass** denoises a block of tokens:
//!
//! ```text
//! embed(anchor, MASK, ...) -> per layer:
//!     RMSNorm -> Q, K, V -> RMSNorm over each head of Q and K -> RoPE
//!     -> attention over the whole cache (non-causal: the mask positions see
//!        each other) -> Wo -> residual
//!     -> RMSNorm -> SwiGLU FFN -> residual
//! -> RMSNorm -> the head -> d2t scatter into the target's vocabulary
//! ```
//!
//! The DSpark heads then run on the decoder's output (in [`super::session`]):
//! the **Markov head** biases each position's logits by the previously
//! committed token, and the **confidence head** predicts how likely each
//! position is to be accepted.
//!
//! The parallel closure contract is the repo's audited one
//! (`docs/develop/PARALLEL_MATMUL_SAFETY.md`): raw pointers are extracted
//! *before* the closure and disjoint slices are rebuilt inside it, because
//! `ComputePool::compute` re-invokes one `Fn` from several workers.

use super::scratch::DflashScratchpad;
use super::session::DflashKvState;
use super::weights::DflashWeights;
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{PreparedRows, Weight};

/// One projection through the shared quantized matmul path.
fn matmul(
    weight: &Weight<'_>,
    input: &[f32],
    rows: usize,
    n_in: usize,
    prepared: &mut PreparedRows,
    pool: &ComputePool,
    output: &mut [f32],
) -> Result<(), String> {
    prepared.prepare(
        input,
        rows,
        n_in,
        weight.needs_q8_0_activation(),
        weight.uses_q8_k(),
    )?;
    prepared.matmul(weight, input, output, pool)
}

/// Several projections off one input.
fn matmul_group<const N: usize>(
    input: &[f32],
    rows: usize,
    n_in: usize,
    projections: [(&Weight<'_>, &mut [f32]); N],
    prepared: &mut PreparedRows,
    pool: &ComputePool,
) -> Result<(), String> {
    let need_q8 = projections
        .iter()
        .any(|(weight, _)| weight.needs_q8_0_activation());
    let need_q8k = projections.iter().any(|(weight, _)| weight.uses_q8_k());
    prepared.prepare(input, rows, n_in, need_q8, need_q8k)?;
    prepared.matmul_group(input, projections, pool)
}

/// Write one row's keys and values into the draft's cache.
///
/// The layout is `[layer][position][kv head][dimension]`; positions are
/// linear (they never exceed the cache capacity, which the caller sizes to
/// the target's context).
fn store_kv(kv: &mut DflashKvState, layer: usize, position: usize, k: &[f32], v: &[f32]) {
    let stride = kv.kv_stride;
    let at = layer * kv.capacity * stride + position * stride;
    kv.k[at..at + stride].copy_from_slice(&k[..stride]);
    kv.v[at..at + stride].copy_from_slice(&v[..stride]);
    kv.n_filled = kv.n_filled.max(position + 1);
}

/// The encoder pass: turn the target's hidden states into the draft's cache.
///
/// `hidden` holds, for every token, the target's hidden state at each of the
/// draft's `target_layers`, concatenated in layer order — the layout
/// llama.cpp's `llama_get_embeddings_layer_inp` hands over.
pub(crate) fn inject_target_features(
    weights: &DflashWeights<'_>,
    hidden: &[f32],
    positions: &[usize],
    kv: &mut DflashKvState,
    scratch: &mut DflashScratchpad,
    pool: &ComputePool,
) -> Result<(), String> {
    let config = &weights.config;
    let n_embd_enc = weights.n_embd_enc;
    let n_embd = config.n_embd;
    let rows = positions.len();
    if rows == 0 {
        return Ok(());
    }
    if hidden.len() != rows * n_embd_enc {
        return Err(format!(
            "DFlash encoder got {} hidden values for {rows} tokens, expected {}",
            hidden.len(),
            rows * n_embd_enc
        ));
    }
    if rows > scratch.max_rows {
        return Err(format!(
            "DFlash encoder got {rows} rows but the scratchpad holds {}",
            scratch.max_rows
        ));
    }

    // fc: the target's hidden states into the draft's space.
    matmul(
        &weights.fc,
        hidden,
        rows,
        n_embd_enc,
        &mut scratch.prepared,
        pool,
        &mut scratch.normed[..rows * n_embd],
    )?;
    for row in 0..rows {
        let at = row * n_embd;
        crate::ops::rms_norm_inplace(
            &mut scratch.normed[at..at + n_embd],
            &weights.enc_output_norm,
            config.norm_eps,
        );
    }

    // Per layer: the draft's own K and V, written into its cache.
    for (layer_index, layer) in weights.layers.iter().enumerate() {
        matmul_group(
            &scratch.normed[..rows * n_embd],
            rows,
            n_embd,
            [
                (&layer.wk, &mut scratch.k[..rows * scratch.n_embd_gqa]),
                (&layer.wv, &mut scratch.v[..rows * scratch.n_embd_gqa]),
            ],
            &mut scratch.prepared,
            pool,
        )?;
        for row in 0..rows {
            let k_at = row * scratch.n_embd_gqa;
            let position = positions[row];
            for head in 0..config.n_head_kv {
                let at = k_at + head * config.n_embd_head_k;
                crate::ops::rms_norm_inplace(
                    &mut scratch.k[at..at + config.n_embd_head_k],
                    &layer.k_norm,
                    config.norm_eps,
                );
                // The injected keys are rotated exactly like the decoder's
                // own, so the draft reads the target's positions.
                crate::ops::rope::rope_mrope(
                    &mut scratch.k[at..at + config.rope_dimension_count],
                    [position, position, position, 0],
                    config.rope_sections,
                    config.rope_dimension_count,
                    config.rope_freq_base,
                );
            }
            store_kv(
                kv,
                layer_index,
                position,
                &scratch.k[k_at..k_at + scratch.n_embd_gqa],
                &scratch.v[k_at..k_at + scratch.n_embd_gqa],
            );
        }
    }
    Ok(())
}

/// The decoder pass: denoise one block of tokens.
///
/// `tokens` is the block — the anchor followed by the mask tokens — and
/// `positions` their absolute positions. On return,
/// `scratch.logits[..rows * n_vocab]` holds the base logits (already
/// scattered into the target's vocabulary, `-inf` outside the draft's rows)
/// and `scratch.hidden[..rows * n_embd]` the final-normed hidden states the
/// confidence head reads.
pub(crate) fn decode_block(
    weights: &DflashWeights<'_>,
    token_embd: &Weight<'_>,
    tokens: &[u32],
    positions: &[usize],
    kv: &mut DflashKvState,
    scratch: &mut DflashScratchpad,
    pool: &ComputePool,
) -> Result<(), String> {
    let config = &weights.config;
    let n_embd = config.n_embd;
    let rows = tokens.len();
    if rows == 0 {
        return Ok(());
    }
    if positions.len() != rows {
        return Err("DFlash decoder got a different number of positions than tokens".to_string());
    }
    if rows > scratch.max_rows {
        return Err(format!(
            "DFlash decoder got {rows} tokens but the scratchpad holds {}",
            scratch.max_rows
        ));
    }
    if kv.n_filled == 0 {
        return Err("DFlash decoder runs before anything is cached".to_string());
    }

    // The block's embeddings, from the target's table.
    let debug = std::env::var("RMI_DFLASH_DEBUG").is_ok();
    for row in 0..rows {
        token_embd.embedding_lookup(
            tokens[row],
            &mut scratch.x[row * n_embd..(row + 1) * n_embd],
        );
    }
    if debug {
        eprintln!(
            "dflash embed: x[0][0..4] = {:?}",
            &scratch.x[..4.min(n_embd)]
        );
    }

    for (layer_index, layer) in weights.layers.iter().enumerate() {
        // The norm in front of the attention.
        for row in 0..rows {
            let at = row * n_embd;
            scratch.normed[at..at + n_embd].copy_from_slice(&scratch.x[at..at + n_embd]);
            crate::ops::rms_norm_inplace(
                &mut scratch.normed[at..at + n_embd],
                &layer.attn_norm,
                config.norm_eps,
            );
        }

        matmul_group(
            &scratch.normed[..rows * n_embd],
            rows,
            n_embd,
            [
                (&layer.wq, &mut scratch.q[..rows * scratch.n_embd_q]),
                (&layer.wk, &mut scratch.k[..rows * scratch.n_embd_gqa]),
                (&layer.wv, &mut scratch.v[..rows * scratch.n_embd_gqa]),
            ],
            &mut scratch.prepared,
            pool,
        )?;

        // The per-head norms and the rotary embedding.
        for row in 0..rows {
            let q_at = row * scratch.n_embd_q;
            let k_at = row * scratch.n_embd_gqa;
            for head in 0..config.n_head {
                let at = q_at + head * config.n_embd_head_k;
                crate::ops::rms_norm_inplace(
                    &mut scratch.q[at..at + config.n_embd_head_k],
                    &layer.q_norm,
                    config.norm_eps,
                );
            }
            for head in 0..config.n_head_kv {
                let at = k_at + head * config.n_embd_head_k;
                crate::ops::rms_norm_inplace(
                    &mut scratch.k[at..at + config.n_embd_head_k],
                    &layer.k_norm,
                    config.norm_eps,
                );
            }
            let position = positions[row];
            for head in 0..config.n_head {
                let at = q_at + head * config.n_embd_head_k;
                crate::ops::rope::rope_mrope(
                    &mut scratch.q[at..at + config.rope_dimension_count],
                    [position, position, position, 0],
                    config.rope_sections,
                    config.rope_dimension_count,
                    config.rope_freq_base,
                );
            }
            for head in 0..config.n_head_kv {
                let at = k_at + head * config.n_embd_head_k;
                crate::ops::rope::rope_mrope(
                    &mut scratch.k[at..at + config.rope_dimension_count],
                    [position, position, position, 0],
                    config.rope_sections,
                    config.rope_dimension_count,
                    config.rope_freq_base,
                );
            }
            // The block's own keys and values go into the cache before the
            // attention reads it: the mask positions see each other.
            store_kv(
                kv,
                layer_index,
                position,
                &scratch.k[k_at..k_at + scratch.n_embd_gqa],
                &scratch.v[k_at..k_at + scratch.n_embd_gqa],
            );
        }

        attention(weights, layer_index, rows, positions, kv, scratch, pool)?;

        // The output projection and the residual.
        matmul(
            &layer.wo,
            &scratch.attn_out[..rows * scratch.n_embd_q],
            rows,
            scratch.n_embd_q,
            &mut scratch.prepared,
            pool,
            &mut scratch.proj[..rows * n_embd],
        )?;
        for row in 0..rows {
            let at = row * n_embd;
            for (value, addend) in scratch.x[at..at + n_embd]
                .iter_mut()
                .zip(scratch.proj[at..at + n_embd].iter())
            {
                *value += addend;
            }
        }

        // The FFN.
        for row in 0..rows {
            let at = row * n_embd;
            scratch.normed[at..at + n_embd].copy_from_slice(&scratch.x[at..at + n_embd]);
            crate::ops::rms_norm_inplace(
                &mut scratch.normed[at..at + n_embd],
                &layer.ffn_norm,
                config.norm_eps,
            );
        }
        matmul_group(
            &scratch.normed[..rows * n_embd],
            rows,
            n_embd,
            [
                (&layer.w_gate, &mut scratch.gate[..rows * scratch.n_ff]),
                (&layer.w_up, &mut scratch.up[..rows * scratch.n_ff]),
            ],
            &mut scratch.prepared,
            pool,
        )?;
        for row in 0..rows {
            let at = row * scratch.n_ff;
            crate::ops::silu_mul_inplace(
                &scratch.gate[at..at + scratch.n_ff],
                &mut scratch.up[at..at + scratch.n_ff],
            );
        }
        matmul(
            &layer.w_down,
            &scratch.up[..rows * scratch.n_ff],
            rows,
            scratch.n_ff,
            &mut scratch.prepared,
            pool,
            &mut scratch.down[..rows * n_embd],
        )?;
        for row in 0..rows {
            let at = row * n_embd;
            for (value, addend) in scratch.x[at..at + n_embd]
                .iter_mut()
                .zip(scratch.down[at..at + n_embd].iter())
            {
                *value += addend;
            }
        }
    }

    // The final norm: the confidence head reads these rows, the head
    // projects them.
    for row in 0..rows {
        let at = row * n_embd;
        crate::ops::rms_norm(
            &scratch.x[at..at + n_embd],
            &weights.output_norm,
            &mut scratch.hidden[at..at + n_embd],
            config.norm_eps,
        );
    }

    // The head, onto the draft's own (possibly reduced) vocabulary.
    let head = weights
        .output
        .as_ref()
        .ok_or_else(|| "DFlash draft requires its own output head".to_string())?;
    matmul(
        head,
        &scratch.hidden[..rows * n_embd],
        rows,
        n_embd,
        &mut scratch.prepared,
        pool,
        &mut scratch.head[..rows * head.n_out],
    )?;

    // The draft's own vocabulary is scattered into the vocabulary the
    // sampler reads; the rows the draft does not have are impossible.
    if let Some(map) = &weights.d2t {
        for row in 0..rows {
            let source = &scratch.head[row * head.n_out..(row + 1) * head.n_out];
            let at = row * scratch.n_vocab;
            scratch.logits[at..at + scratch.n_vocab].fill(f32::NEG_INFINITY);
            for (draft_row, value) in source.iter().enumerate() {
                let target = map[draft_row] as usize;
                if target < scratch.n_vocab {
                    scratch.logits[at + target] = *value;
                }
            }
        }
    } else {
        if head.n_out != scratch.n_vocab {
            return Err(
                "DFlash draft head width does not match the sampler vocabulary".to_string(),
            );
        }
        for row in 0..rows {
            let at = row * scratch.n_vocab;
            scratch.logits[at..at + scratch.n_vocab]
                .copy_from_slice(&scratch.head[row * head.n_out..(row + 1) * head.n_out]);
        }
    }
    Ok(())
}

/// The attention over the draft's own cache.
///
/// The attention is **not** causal: the mask positions of a block see each
/// other, which is what makes the block denoising work.
fn attention(
    weights: &DflashWeights<'_>,
    layer_index: usize,
    rows: usize,
    positions: &[usize],
    kv: &mut DflashKvState,
    scratch: &mut DflashScratchpad,
    pool: &ComputePool,
) -> Result<(), String> {
    let config = &weights.config;
    let n_head = config.n_head;
    let group_size = config.n_head / config.n_head_kv;
    let n_embd_head = config.n_embd_head_k;
    if config.n_embd_head_k != config.n_embd_head_v {
        return Err("DFlash draft requires equal key and value head widths".to_string());
    }
    let kq_scale = 1.0f32 / (n_embd_head as f32).sqrt();
    let stride = kv.kv_stride;
    let capacity = kv.capacity;
    let layer_base = layer_index * capacity * stride;
    let sliding = config.layer_is_sliding(layer_index);
    let n_swa = config.sliding_window;
    let n_filled = kv.n_filled;
    let n_embd_q = scratch.n_embd_q;
    let score_stride = scratch.score_stride;

    if n_filled == 0 {
        return Err("DFlash draft attends before anything is cached".to_string());
    }
    let hi = n_filled - 1;

    // Raw pointers, extracted before the closure: `ComputePool::compute`
    // re-invokes one `Fn` from several workers, so every slice the closure
    // touches must be disjoint per worker — the per-thread score window and
    // the per-thread head ranges below are.
    let q_ptr = scratch.q.as_ptr();
    let attn_ptr = scratch.attn_out.as_mut_ptr();
    let scores_ptr = scratch.scores.as_mut_ptr();
    let k_ptr = kv.k.as_ptr();
    let v_ptr = kv.v.as_ptr();
    let positions_owned: Vec<usize> = positions.to_vec();

    pool.compute(move |ith: usize, nth: usize| {
        let scores = unsafe {
            std::slice::from_raw_parts_mut(scores_ptr.add(ith * score_stride), score_stride)
        };
        let head_start = n_head * ith / nth;
        let head_end = n_head * (ith + 1) / nth;
        for row in 0..rows {
            let position = positions_owned[row];
            let mut lo = 0usize;
            if sliding {
                lo = (position + 1).saturating_sub(n_swa);
            }
            lo = lo.max(hi.saturating_sub(capacity - 1));
            if hi < lo {
                continue;
            }
            let visible = hi - lo + 1;
            for head in head_start..head_end {
                let kv_head = head / group_size;
                let query = unsafe {
                    std::slice::from_raw_parts(
                        q_ptr.add(row * n_embd_q + head * n_embd_head),
                        n_embd_head,
                    )
                };
                for (index, slot) in scores[..visible].iter_mut().enumerate() {
                    let at = layer_base + (lo + index) * stride + kv_head * n_embd_head;
                    let key = unsafe { std::slice::from_raw_parts(k_ptr.add(at), n_embd_head) };
                    *slot = crate::ops::dot::dot_f32(query, key, n_embd_head) * kq_scale;
                }
                crate::ops::softmax::softmax_inplace(&mut scores[..visible]);
                let out_at = row * n_embd_q + head * n_embd_head;
                for dimension in 0..n_embd_head {
                    let mut value = 0.0f32;
                    for index in 0..visible {
                        let at = layer_base + (lo + index) * stride + kv_head * n_embd_head;
                        value += scores[index] * unsafe { *v_ptr.add(at + dimension) };
                    }
                    unsafe { *attn_ptr.add(out_at + dimension) = value };
                }
            }
        }
    });
    Ok(())
}
