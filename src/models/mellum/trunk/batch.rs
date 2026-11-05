//! Batched prefill for the Mellum trunk.
//!
//! `MellumRunner::forward_token` walks one token at a time. That is the
//! shape ggml's reference graph uses, and the shape the pinned oracle
//! compares against, but it is a bad shape for a *prompt*: every layer
//! holds 22.5 MB of attention weights and 64 experts of 3 x 2.19 MB, so a
//! single token streams 2.1 GB of weights and the prompt streams that once
//! per token.
//!
//! A prompt, however, is a known batch of tokens. [`MellumRunner::forward_tokens`]
//! evaluates the same tokens with the same arithmetic, but hands every
//! projection all the rows at once, so a weight matrix is read once per
//! batch instead of once per token. For a 32 token prompt that is ~6x less
//! weight traffic, and the rows are also the shape the Vulkan device path
//! is fast at: the device reads the weight once for the whole batch, while
//! the CPU re-reads it per row.
//!
//! # Bit parity
//!
//! The batched path is not an approximation of the single token walk, it is
//! the same arithmetic:
//!
//! * every projection is a per-row product, and the Q8_0 kernels accumulate
//!   each row independently, so a row's result does not depend on how many
//!   rows share the call (verified against the single row device path:
//!   0 mismatches for the 2304 -> 98304 output head at 2, 4 and 8 rows);
//! * attention is computed per query row by the same
//!   [`attend_heads`] helper the single token walk uses, so the scores, the
//!   softmax and the value reduction are bit-for-bit the same;
//! * the MoE routing, the top-8 selection and the weight normalisation run
//!   per row, and the expert outputs are folded back per row in slot order,
//!   which is the order ggml's `ggml_add` chain uses.
//!
//! # The parity trace
//!
//! The batched path does not emit the per-slot expert checkpoints
//! (`ffn_moe_up` and friends), because assembling them per row would need a
//! copy of every expert output. When the parity trace is recording, the
//! prefill therefore keeps using the single token walk, so the trace stays
//! complete and comparable with the oracle.

use super::config::MellumConfig;
use super::forward::{
    apply_rope, attend_heads, build_rope_cache, ggml_softmax, project, AttnScratch, MellumRunner,
    MOE_WEIGHT_SUM_FLOOR,
};
use super::weights::{MellumLayerWeights, MellumModel};
use crate::core::scratchpad::{KvCache, KvState};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{PreparedRows, Weight};
use crate::ops::{dot_f32, rms_norm, silu};
use std::sync::Arc;

/// Largest number of tokens one batched projection covers.
///
/// The batch buffers are sized from this, so it bounds the extra memory to
/// roughly `rows * 40 KB` — about 10 MB at 256 rows for the 12B checkpoint.
///
/// The size also decides how often the prompt is split: every chunk reads
/// all 64 experts of every layer once, so fewer chunks means less expert
/// traffic. A 183 token prompt fits in one chunk here.
pub(crate) const MAX_BATCH_ROWS: usize = 256;

/// Smallest weight the device should take in the batched path.
///
/// The device amortises the weight read across the rows of one call, but
/// every call ends in its own submission, and that submission costs ~0.4 ms
/// on the GB10 box. Measured on that box (8 threads, 32 rows, Q8_0):
///
/// | projection | weight | device | CPU |
/// | --- | --- | --- | --- |
/// | `wq` 2304 -> 4096 | 10.0 MB | 0.71 ms | 0.70 ms |
/// | `wk` 2304 -> 512 | 1.25 MB | 0.39 ms | 0.09 ms |
/// | expert 2304 -> 896 | 2.19 MB | 0.61 ms | 0.14 ms |
/// | output head 2304 -> 98304 | 240 MB | 5.3 ms | 75 ms |
///
/// so the device only pays off once the weight is big enough that the CPU
/// has to stream it many times: the 240 MB output head is 14x faster on the
/// device at 32 rows, while the 1.25 MB projections are 4x faster on the
/// CPU.
fn device_worthwhile(weight: &Weight<'_>, rows: usize) -> bool {
    if rows < 2 || !device_prefill_enabled() {
        return false;
    }
    // Q8_0 stores 32 weights plus a scale in 34 bytes, so the element count
    // is a good enough proxy for the byte count.
    weight.n_in.saturating_mul(weight.n_out) >= 8 * 1024 * 1024
}

/// Whether the batched prefill may hand a projection to the device.
///
/// Off by default. The device reads a weight once for the whole batch, but
/// it has to *upload* that weight first, and the host -> device path runs at
/// ~10 GB/s while a DRAM read runs at ~100 GB/s. A prefill touches every
/// weight exactly once, so the upload is pure overhead: measured on the GB10
/// box with a 183 token prompt, the device took `wq` and `wo` of all 28
/// layers and the prefill went from 1073 ms to 1460 ms.
///
/// The knob stays because the trade-off flips in a serving loop, where the
/// weights are uploaded once and reused by every request, and because it is
/// what the bit-parity check of the device path runs with.
fn device_prefill_enabled() -> bool {
    std::env::var("RMI_MELLUM_DEVICE_PREFILL")
        .ok()
        .map(|value| value != "0")
        .unwrap_or(false)
}

/// Whether the prompt goes through the batched prefill.
///
/// Set `RMI_MELLUM_BATCHED_PREFILL=0` to fall back to the single token
/// walk; the two produce the same logits, so this only exists to compare
/// them or to work around a device that misbehaves.
fn batched_prefill_enabled() -> bool {
    std::env::var("RMI_MELLUM_BATCHED_PREFILL")
        .ok()
        .map(|value| value != "0")
        .unwrap_or(true)
}

/// Buffers for the batched prefill.
pub(crate) struct BatchScratch {
    /// Maximum number of rows one batch covers.
    pub(crate) rows: usize,
    hidden: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    attn_proj: Vec<f32>,
    ffn_inp: Vec<f32>,
    ffn_normed: Vec<f32>,
    router: Vec<f32>,
    expert_ids: Vec<u32>,
    expert_weights: Vec<f32>,
    /// The weighted expert output of every `(token, slot)` pair, so the fold
    /// can run in slot order (ggml's `ggml_add` chain is a left fold).
    expert_out: Vec<f32>,
    moe_out: Vec<f32>,
    /// For every expert, the tokens routed to it and the slot they hold it
    /// in.
    group_rows: Vec<u32>,
    group_slots: Vec<u32>,
    group_counts: Vec<usize>,
    /// Scratch for one expert group.
    group_input: Vec<f32>,
    up_buf: Vec<f32>,
    gate_buf: Vec<f32>,
    down_buf: Vec<f32>,
    attn_scratch: AttnScratch,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    /// Quantized activation rows for the batched matmuls.
    prepared: PreparedRows,
}

impl BatchScratch {
    pub(crate) fn new(config: &MellumConfig, capacity: usize) -> Self {
        let rows = capacity.max(1).min(MAX_BATCH_ROWS);
        let n_embd = config.n_embd;
        let n_embd_q = config.n_head * config.n_embd_head_k;
        let n_embd_k = config.n_head_kv * config.n_embd_head_k;
        let n_embd_v = config.n_head_kv * config.n_embd_head_v;
        let n_used = config.expert_used_count;
        Self {
            rows,
            hidden: vec![0.0; rows * n_embd],
            normed: vec![0.0; rows * n_embd],
            q: vec![0.0; rows * n_embd_q],
            k: vec![0.0; rows * n_embd_k],
            v: vec![0.0; rows * n_embd_v],
            attn: vec![0.0; rows * config.n_attn()],
            attn_proj: vec![0.0; rows * n_embd],
            ffn_inp: vec![0.0; rows * n_embd],
            ffn_normed: vec![0.0; rows * n_embd],
            router: vec![0.0; rows * config.expert_count],
            expert_ids: vec![0; rows * n_used],
            expert_weights: vec![0.0; rows * n_used],
            expert_out: vec![0.0; rows * n_used * n_embd],
            moe_out: vec![0.0; rows * n_embd],
            group_rows: vec![0; config.expert_count * rows],
            group_slots: vec![0; config.expert_count * rows],
            group_counts: vec![0; config.expert_count],
            group_input: vec![0.0; rows * n_embd],
            up_buf: vec![0.0; rows * config.expert_ffn],
            gate_buf: vec![0.0; rows * config.expert_ffn],
            down_buf: vec![0.0; rows * n_embd],
            attn_scratch: AttnScratch::new(config, capacity),
            rope_cos: vec![0.0; config.n_rot / 2],
            rope_sin: vec![0.0; config.n_rot / 2],
            prepared: PreparedRows::new(rows, n_embd.max(config.n_attn()).max(config.expert_ffn)),
        }
    }
}

impl MellumRunner<'_> {
    /// Evaluate `tokens` as one batch starting at `start_position`.
    ///
    /// The result is bit-identical to calling [`MellumRunner::forward_token`]
    /// for every token in turn: the batch only changes how many rows a
    /// projection carries, never the arithmetic of a row. The final logits
    /// are the logits of the last token of the batch.
    ///
    /// When the parity trace is recording, this falls back to the single
    /// token walk, because the batched path does not emit the per-slot
    /// expert checkpoints.
    pub fn forward_tokens(&mut self, tokens: &[u32], start_position: usize) -> Result<(), String> {
        if tokens.is_empty() {
            return Ok(());
        }
        let model = self.model;
        let config = model.config();
        let capacity = self.capacity;
        if start_position + tokens.len() > capacity {
            return Err(format!(
                "batch of {} tokens at position {start_position} exceeds session capacity {capacity}",
                tokens.len()
            ));
        }
        for token in tokens {
            if *token as usize >= config.vocab_size {
                return Err(format!(
                    "token id {token} is outside vocabulary {}",
                    config.vocab_size
                ));
            }
        }
        if super::trace::enabled() || !batched_prefill_enabled() {
            for (offset, token) in tokens.iter().enumerate() {
                self.forward_token(*token, start_position + offset)?;
            }
            return Ok(());
        }
        // The batch buffers are bounded, so a long prompt runs in chunks.
        let mut offset = 0;
        while offset < tokens.len() {
            let rows = (tokens.len() - offset).min(self.batch.rows);
            self.forward_batch(&tokens[offset..offset + rows], start_position + offset)?;
            offset += rows;
        }
        Ok(())
    }

    /// Evaluate one batch of tokens and leave the last token's logits in
    /// [`MellumRunner::logits`].
    fn forward_batch(&mut self, tokens: &[u32], start_position: usize) -> Result<(), String> {
        let model = self.model;
        let config = model.config();
        let pool = model.pool();
        let capacity = self.capacity;
        let rows = tokens.len();
        let n_embd = config.n_embd;
        let Self {
            kv, scratch, batch, ..
        } = self;

        for (row, token) in tokens.iter().enumerate() {
            model
                .token_embedding
                .embedding_lookup(*token, &mut batch.hidden[row * n_embd..(row + 1) * n_embd]);
        }

        for layer_index in 0..config.n_layer {
            batch_layer(
                model,
                kv,
                batch,
                &pool,
                config,
                layer_index,
                start_position,
                rows,
                capacity,
            );
        }

        // Only the last row's logits are needed; the head is a 240 MB
        // projection, so the earlier rows are not worth computing.
        let last = rows - 1;
        rms_norm(
            &batch.hidden[last * n_embd..(last + 1) * n_embd],
            &model.output_norm,
            &mut scratch.normed,
            config.eps,
        );
        project(
            &pool,
            &model.output,
            &scratch.normed,
            n_embd,
            config.vocab_size,
            &mut scratch.logits,
            &mut scratch.project,
        );
        kv.seq_len = start_position + rows;
        kv.update_access();
        Ok(())
    }
}

/// One layer of the batched forward pass.
#[allow(clippy::too_many_arguments)]
fn batch_layer(
    model: &MellumModel,
    kv: &mut KvState,
    batch: &mut BatchScratch,
    pool: &Arc<ComputePool>,
    config: &MellumConfig,
    layer_index: usize,
    start_position: usize,
    rows: usize,
    capacity: usize,
) {
    let layer = &model.layers[layer_index];
    let n_embd = config.n_embd;
    let n_embd_q = config.n_head * config.n_embd_head_k;
    let n_embd_k = config.n_head_kv * config.n_embd_head_k;
    let n_embd_v = config.n_head_kv * config.n_embd_head_v;
    let head_dim = config.n_embd_head_k;

    for row in 0..rows {
        rms_norm(
            &batch.hidden[row * n_embd..(row + 1) * n_embd],
            &layer.attn_norm,
            &mut batch.normed[row * n_embd..(row + 1) * n_embd],
            config.eps,
        );
    }

    // wq / wk / wv share one activation, so they share one quantised copy
    // of it.
    batch
        .prepared
        .prepare(&batch.normed[..rows * n_embd], rows, n_embd, true, false)
        .expect("batch activation fits the prepared rows buffer");
    project_rows(
        pool,
        &batch.prepared,
        &layer.wq,
        &batch.normed[..rows * n_embd],
        &mut batch.q[..rows * n_embd_q],
        device_worthwhile(&layer.wq, rows),
    );
    project_rows(
        pool,
        &batch.prepared,
        &layer.wk,
        &batch.normed[..rows * n_embd],
        &mut batch.k[..rows * n_embd_k],
        device_worthwhile(&layer.wk, rows),
    );
    project_rows(
        pool,
        &batch.prepared,
        &layer.wv,
        &batch.normed[..rows * n_embd],
        &mut batch.v[..rows * n_embd_v],
        device_worthwhile(&layer.wv, rows),
    );

    // Per-head Q/K RMSNorm with the same eps as the block norm.
    for row in 0..rows {
        for head in 0..config.n_head {
            let offset = row * n_embd_q + head * head_dim;
            crate::ops::rms_norm_inplace(
                &mut batch.q[offset..offset + head_dim],
                &layer.q_norm,
                config.eps,
            );
        }
        for head in 0..config.n_head_kv {
            let offset = row * n_embd_k + head * head_dim;
            crate::ops::rms_norm_inplace(
                &mut batch.k[offset..offset + head_dim],
                &layer.k_norm,
                config.eps,
            );
        }
    }

    // RoPE depends on the position, so every row gets its own table.
    let flavour = config.rope_for(layer_index);
    for row in 0..rows {
        build_rope_cache(
            flavour,
            config.freq_base,
            &mut batch.rope_cos,
            &mut batch.rope_sin,
            start_position + row,
        );
        for head in 0..config.n_head {
            let offset = row * n_embd_q + head * head_dim;
            apply_rope(
                &mut batch.q[offset..offset + head_dim],
                &batch.rope_cos,
                &batch.rope_sin,
            );
        }
        for head in 0..config.n_head_kv {
            let offset = row * n_embd_k + head * head_dim;
            apply_rope(
                &mut batch.k[offset..offset + head_dim],
                &batch.rope_cos,
                &batch.rope_sin,
            );
        }
    }

    // The keys and values of the whole batch go into the cache before any
    // query attends to them.
    store_kv_batch(
        &mut kv.cache,
        config,
        capacity,
        layer_index,
        start_position,
        rows,
        &batch.k,
        &batch.v,
    );

    // Attention: one query row at a time, exactly like the single token
    // walk, but against the keys the batch already wrote.
    for row in 0..rows {
        let position = start_position + row;
        let q = &batch.q[row * n_embd_q..(row + 1) * n_embd_q];
        let attn = &mut batch.attn[row * config.n_attn()..(row + 1) * config.n_attn()];
        attend_heads(
            &kv.cache,
            capacity,
            config,
            layer_index,
            position,
            q,
            attn,
            &mut batch.attn_scratch,
        );
    }

    batch
        .prepared
        .prepare(
            &batch.attn[..rows * config.n_attn()],
            rows,
            config.n_attn(),
            true,
            false,
        )
        .expect("batch activation fits the prepared rows buffer");
    project_rows(
        pool,
        &batch.prepared,
        &layer.wo,
        &batch.attn[..rows * config.n_attn()],
        &mut batch.attn_proj[..rows * n_embd],
        device_worthwhile(&layer.wo, rows),
    );

    // Mellum's parallel residual: the FFN consumes `attn_out + hidden` and
    // its output is added to that same sum.
    for row in 0..rows {
        let offset = row * n_embd;
        for lane in 0..n_embd {
            batch.ffn_inp[offset + lane] =
                batch.attn_proj[offset + lane] + batch.hidden[offset + lane];
        }
        rms_norm(
            &batch.ffn_inp[offset..offset + n_embd],
            &layer.ffn_norm,
            &mut batch.ffn_normed[offset..offset + n_embd],
            config.eps,
        );
    }

    batch_moe(layer, batch, pool, config, rows);

    for row in 0..rows {
        let offset = row * n_embd;
        for lane in 0..n_embd {
            batch.hidden[offset + lane] =
                batch.moe_out[offset + lane] + batch.ffn_inp[offset + lane];
        }
    }
}

/// The MoE block of one layer, batched over the tokens.
fn batch_moe(
    layer: &MellumLayerWeights<'_>,
    batch: &mut BatchScratch,
    pool: &Arc<ComputePool>,
    config: &MellumConfig,
    rows: usize,
) {
    let n_embd = config.n_embd;
    let n_expert = config.expert_count;
    let n_used = config.expert_used_count;
    let expert_ffn = config.expert_ffn;

    // Router: a plain f32 matvec per token, exactly like ggml's
    // `ggml_mul_mat(gate_inp, cur)`.
    for row in 0..rows {
        let input = &batch.ffn_normed[row * n_embd..(row + 1) * n_embd];
        let router = &mut batch.router[row * n_expert..(row + 1) * n_expert];
        for expert in 0..n_expert {
            let weights = &layer.moe_router[expert * n_embd..(expert + 1) * n_embd];
            router[expert] = dot_f32(input, weights, n_embd);
        }
        ggml_softmax(router);
    }

    // Top-8 selection and the weight normalisation, per token.
    let mut order: Vec<u32> = Vec::with_capacity(n_expert);
    for row in 0..rows {
        let router = &batch.router[row * n_expert..(row + 1) * n_expert];
        order.clear();
        order.extend(0..n_expert as u32);
        order.sort_by(|&a, &b| {
            router[b as usize]
                .partial_cmp(&router[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let ids = &mut batch.expert_ids[row * n_used..(row + 1) * n_used];
        for (index, slot) in order.iter().take(n_used).enumerate() {
            ids[index] = *slot;
        }
        let weights = &mut batch.expert_weights[row * n_used..(row + 1) * n_used];
        let mut sum = 0.0f64;
        for index in 0..n_used {
            let probability = router[ids[index] as usize];
            weights[index] = probability;
            sum += f64::from(probability);
        }
        // ggml_sum_rows narrows the f64 accumulator to f32, then clamps,
        // then divides in f32.
        let sum = (sum as f32).max(MOE_WEIGHT_SUM_FLOOR);
        for weight in weights.iter_mut() {
            *weight /= sum;
        }
    }

    // Group the tokens by expert, so one expert's weight matrix is read
    // once for all the tokens that routed to it.
    for count in batch.group_counts.iter_mut() {
        *count = 0;
    }
    for row in 0..rows {
        for slot in 0..n_used {
            let expert = batch.expert_ids[row * n_used + slot] as usize;
            let count = batch.group_counts[expert];
            batch.group_rows[expert * batch.rows + count] = row as u32;
            batch.group_slots[expert * batch.rows + count] = slot as u32;
            batch.group_counts[expert] = count + 1;
        }
    }

    for expert in 0..n_expert {
        let count = batch.group_counts[expert];
        if count == 0 {
            continue;
        }
        // Gather the rows that routed to this expert.
        for index in 0..count {
            let row = batch.group_rows[expert * batch.rows + index] as usize;
            batch.group_input[index * n_embd..(index + 1) * n_embd]
                .copy_from_slice(&batch.ffn_normed[row * n_embd..(row + 1) * n_embd]);
        }
        batch
            .prepared
            .prepare(
                &batch.group_input[..count * n_embd],
                count,
                n_embd,
                true,
                false,
            )
            .expect("expert group fits the prepared rows buffer");
        let device = device_worthwhile(&layer.moe_up[expert], count);
        project_rows(
            pool,
            &batch.prepared,
            &layer.moe_up[expert],
            &batch.group_input[..count * n_embd],
            &mut batch.up_buf[..count * expert_ffn],
            device,
        );
        project_rows(
            pool,
            &batch.prepared,
            &layer.moe_gate[expert],
            &batch.group_input[..count * n_embd],
            &mut batch.gate_buf[..count * expert_ffn],
            device,
        );
        for index in 0..count {
            let slot = index * expert_ffn;
            for lane in 0..expert_ffn {
                batch.gate_buf[slot + lane] =
                    silu(batch.gate_buf[slot + lane]) * batch.up_buf[slot + lane];
            }
        }
        batch
            .prepared
            .prepare(
                &batch.gate_buf[..count * expert_ffn],
                count,
                expert_ffn,
                true,
                false,
            )
            .expect("expert group fits the prepared rows buffer");
        project_rows(
            pool,
            &batch.prepared,
            &layer.moe_down[expert],
            &batch.gate_buf[..count * expert_ffn],
            &mut batch.down_buf[..count * n_embd],
            device_worthwhile(&layer.moe_down[expert], count),
        );
        // `ggml_mul(down, weight)`: the weighted expert output, not the
        // running sum.
        for index in 0..count {
            let row = batch.group_rows[expert * batch.rows + index] as usize;
            let slot = batch.group_slots[expert * batch.rows + index] as usize;
            let weight = batch.expert_weights[row * n_used + slot];
            let source = index * n_embd;
            let target = (row * n_used + slot) * n_embd;
            for lane in 0..n_embd {
                batch.expert_out[target + lane] = batch.down_buf[source + lane] * weight;
            }
        }
    }

    // ggml folds the expert outputs left to right in slot order.
    for row in 0..rows {
        let output = &mut batch.moe_out[row * n_embd..(row + 1) * n_embd];
        for slot in 0..n_used {
            let source = (row * n_used + slot) * n_embd;
            if slot == 0 {
                output.copy_from_slice(&batch.expert_out[source..source + n_embd]);
            } else {
                for lane in 0..n_embd {
                    output[lane] += batch.expert_out[source + lane];
                }
            }
        }
    }
}

/// Run one batched projection.
///
/// `allow_device` decides whether the Vulkan path may take it; the device
/// is asked first when it is, and the CPU pool runs the matmul otherwise.
fn project_rows(
    pool: &Arc<ComputePool>,
    prepared: &PreparedRows,
    weight: &Weight<'_>,
    input: &[f32],
    output: &mut [f32],
    allow_device: bool,
) {
    // The scope has to cover the call itself: the Vulkan path is asked
    // before the pool dispatch, on this thread.
    #[cfg(feature = "vulkan")]
    let _cpu_scope = if allow_device {
        None
    } else {
        Some(ComputePool::disable_gpu_matmul_for_scope())
    };
    prepared
        .matmul(weight, input, output, pool)
        .expect("batched projection shape matches the prepared rows");
}

/// Write a batch's keys and values into the KV cache.
#[allow(clippy::too_many_arguments)]
fn store_kv_batch(
    kv_cache: &mut KvCache,
    config: &MellumConfig,
    capacity: usize,
    layer_index: usize,
    start_position: usize,
    rows: usize,
    k: &[f32],
    v: &[f32],
) {
    let kv_heads_k = config.n_head_kv * config.n_embd_head_k;
    let kv_heads_v = config.n_head_kv * config.n_embd_head_v;
    let stride = config.n_head_kv * config.n_embd_head_k.max(config.n_embd_head_v);
    let base = layer_index * capacity + start_position;
    for row in 0..rows {
        let offset = (base + row) * stride;
        let k_row = &k[row * kv_heads_k..(row + 1) * kv_heads_k];
        let v_row = &v[row * kv_heads_v..(row + 1) * kv_heads_v];
        match kv_cache {
            KvCache::F32(cache) => {
                cache.k[offset..offset + kv_heads_k].copy_from_slice(k_row);
                cache.v[offset..offset + kv_heads_v].copy_from_slice(v_row);
            }
            KvCache::F16(cache) => {
                for lane in 0..kv_heads_k {
                    cache.k[offset + lane] = crate::ops::f32_to_f16(k_row[lane]);
                }
                for lane in 0..kv_heads_v {
                    cache.v[offset + lane] = crate::ops::f32_to_f16(v_row[lane]);
                }
            }
        }
    }
}
