//! Mellum forward pass: one token at a time against a KV cache.
//!
//! Every arithmetic step mirrors ggml's CPU kernels in the order ggml runs
//! them, because the parity contract is bit-identical output rather than
//! "close enough":
//!
//! * attention scores — `dot_f32` (f32 products, f64 accumulation), then the
//!   `1/sqrt(head_dim)` scale, then `-inf` for masked keys, then ggml's
//!   two-pass softmax;
//! * sliding-window layers mask with `key_pos + n_swa <= query_pos`;
//! * MoE routing — full 64-way softmax, then top-8 with llama.cpp's
//!   post-`partial_sort` swap of the first two slots, then re-normalise;
//! * the weighted expert outputs are folded left to right in that (swapped)
//!   order, exactly like llama.cpp's `ggml_add` chain in `build_moe_ffn`.

use super::config::{MellumConfig, MellumRope};
use super::trace;
use super::weights::{MellumLayerWeights, MellumModel};
use crate::core::scratchpad::{KvCache, KvFormat, KvState};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::Weight;
use crate::ops::{dot_f32, rms_norm, silu};
use std::sync::Arc;

/// ggml's softmax clamp: `ggml_clamp(weights_sum, 6.103515625e-5, INFINITY)`.
/// `6.103515625e-5` is the smallest positive normal f16.
pub(crate) const MOE_WEIGHT_SUM_FLOOR: f32 = 6.103_515_6e-5;

/// Scratch buffers for the quantized matmul path, shared across all projections.
///
/// `quantize_and_matmul_with_scratch` needs three temporary buffers whose
/// sizes depend only on `n_in` (the widest projection input, `n_embd`).
/// Reusing them across calls avoids per-projection allocation.
pub struct ProjectScratch {
    pub q8k_buf: Vec<crate::ops::quant::BlockQ8K>,
    pub q8_buf: Vec<u8>,
    pub scale_buf: Vec<f32>,
}

impl ProjectScratch {
    fn new(max_n_in: usize) -> Self {
        Self {
            q8k_buf: vec![
                crate::ops::quant::BlockQ8K {
                    d: 0.0,
                    qs: [0; crate::ops::quant::QK_K],
                    bsums: [0; crate::ops::quant::QK_K / 16]
                };
                max_n_in / crate::ops::quant::QK_K
            ],
            q8_buf: vec![0u8; max_n_in],
            scale_buf: vec![0.0f32; max_n_in.div_ceil(32)],
        }
    }
}

/// Buffers one attention step needs, sized once so the decode loop never
/// allocates.
///
/// The batched prefill keeps its own copy: the two paths never run at the
/// same time, but they must not share a buffer either, because the batched
/// path walks several rows of the same layer back to back.
pub(crate) struct AttnScratch {
    /// Reused for the per-head softmax weights; sized to the widest visible
    /// window (`min(capacity, n_swa)` for SWA layers, `capacity` otherwise).
    pub probs: Vec<f32>,
    pub k_window: Vec<f32>,
    pub v_window: Vec<f32>,
    /// V window transposed to `[head_dim_v, visible]` so each attention output
    /// lane is a contiguous dot-product input.
    pub attn_transposed: Vec<f32>,
    /// Debug rows for the whole cache (masked keys = -INF), matching ggml's
    /// `kq` / `kq_soft_max` shapes.
    pub kq_row: Vec<f32>,
    pub kq_probs: Vec<f32>,
    /// The `kq` / `kq_soft_max` rows of every head, concatenated.
    pub kq_debug: Vec<f32>,
    pub kq_soft_max_debug: Vec<f32>,
}

impl AttnScratch {
    pub(crate) fn new(config: &MellumConfig, capacity: usize) -> Self {
        let head_dim = config.n_embd_head_k;
        let head_dim_v = config.n_embd_head_v;
        // The visible key window is bounded by the *sliding window* on SWA
        // layers but by the whole sequence on the rest, so size every
        // per-key buffer by `capacity` — the worst case is a non-SWA layer at
        // the last position.
        let window = capacity.max(1);
        Self {
            probs: vec![0.0; window],
            k_window: vec![0.0; window * head_dim],
            v_window: vec![0.0; window * head_dim_v],
            attn_transposed: vec![0.0; window * head_dim_v],
            kq_row: vec![0.0; window],
            kq_probs: vec![0.0; window],
            kq_debug: Vec::new(),
            kq_soft_max_debug: Vec::new(),
        }
    }
}

/// Reusable per-token buffers, sized once so the decode loop never allocates.
pub struct MellumScratch {
    pub hidden: Vec<f32>,
    pub normed: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn: Vec<f32>,
    pub attn_proj: Vec<f32>,
    pub ffn_inp: Vec<f32>,
    pub ffn_normed: Vec<f32>,
    pub router: Vec<f32>,
    pub router_order: Vec<u32>,
    pub expert_ids: Vec<u32>,
    pub expert_weights: Vec<f32>,
    pub gate_buf: Vec<f32>,
    pub up_buf: Vec<f32>,
    pub down_buf: Vec<f32>,
    pub moe_out: Vec<f32>,
    pub logits: Vec<f32>,
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
    /// Buffers for the attention step.
    pub attn_scratch: AttnScratch,
    moe_debug: MellumMoeDebug,
    /// Quantized matmul scratch (Q8_0 and Q8_K activation buffers).
    pub project: ProjectScratch,
}

impl MellumScratch {
    pub fn new(config: &MellumConfig, capacity: usize) -> Self {
        let n_embd_q = config.n_head * config.n_embd_head_k;
        let n_embd_k = config.n_head_kv * config.n_embd_head_k;
        let n_embd_v = config.n_head_kv * config.n_embd_head_v;
        Self {
            hidden: vec![0.0; config.n_embd],
            normed: vec![0.0; config.n_embd],
            q: vec![0.0; n_embd_q],
            k: vec![0.0; n_embd_k],
            v: vec![0.0; n_embd_v],
            attn: vec![0.0; config.n_attn()],
            attn_proj: vec![0.0; config.n_embd],
            ffn_inp: vec![0.0; config.n_embd],
            ffn_normed: vec![0.0; config.n_embd],
            router: vec![0.0; config.expert_count],
            router_order: vec![0; config.expert_count],
            expert_ids: vec![0; config.expert_used_count],
            expert_weights: vec![0.0; config.expert_used_count],
            gate_buf: vec![0.0; config.expert_ffn],
            up_buf: vec![0.0; config.expert_ffn],
            down_buf: vec![0.0; config.n_embd],
            moe_out: vec![0.0; config.n_embd],
            logits: vec![0.0; config.vocab_size],
            rope_cos: vec![0.0; config.n_rot / 2],
            rope_sin: vec![0.0; config.n_rot / 2],
            attn_scratch: AttnScratch::new(config, capacity),
            moe_debug: MellumMoeDebug::new(config),
            project: ProjectScratch::new(config.n_embd.max(config.n_attn())),
        }
    }
}

/// A loaded model plus its KV cache and scratch buffers.
pub struct MellumRunner<'model> {
    pub model: &'model MellumModel,
    pub kv: KvState,
    pub scratch: MellumScratch,
    /// Buffers for the batched prefill.
    pub(crate) batch: super::batch::BatchScratch,
    pub(crate) capacity: usize,
}

impl<'model> MellumRunner<'model> {
    pub fn new(
        model: &'model MellumModel,
        capacity: usize,
        kv_format: KvFormat,
    ) -> Result<Self, String> {
        let config = model.config();
        if capacity == 0 || capacity > config.n_ctx {
            return Err(format!(
                "Session capacity {capacity} must be within 1..={}",
                config.n_ctx
            ));
        }
        let arch = Arc::new(crate::core::scratchpad::KvArch::new(
            config.n_layer,
            config.n_head_kv,
            config.n_embd_head_k,
            config.n_embd_head_v,
            config.n_ctx,
        ));
        Ok(Self {
            model,
            kv: KvState::new(arch, kv_format, capacity),
            scratch: MellumScratch::new(config, capacity),
            batch: super::batch::BatchScratch::new(config, capacity),
            capacity,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn reset(&mut self) {
        self.kv.reset();
    }

    pub fn logits(&self) -> &[f32] {
        &self.scratch.logits
    }

    /// Evaluate `token` at `position`, appending its KV entry, and leave the
    /// next-token logits in [`Self::logits`].
    pub fn forward_token(&mut self, token: u32, position: usize) -> Result<(), String> {
        if position >= self.capacity {
            return Err(format!(
                "position {position} exceeds session capacity {}",
                self.capacity
            ));
        }
        let config = self.model.config();
        if token as usize >= config.vocab_size {
            return Err(format!(
                "token id {token} is outside vocabulary {}",
                config.vocab_size
            ));
        }
        let pool = self.model.pool();
        self.model
            .token_embedding
            .embedding_lookup(token, &mut self.scratch.hidden);
        trace::global("embd", position, &self.scratch.hidden);

        for layer_index in 0..config.n_layer {
            self.forward_layer(layer_index, position, &pool);
        }

        rms_norm(
            &self.scratch.hidden,
            &self.model.output_norm,
            &mut self.scratch.normed,
            config.eps,
        );
        trace::global("result_norm", position, &self.scratch.normed);
        project(
            &pool,
            &self.model.output,
            &self.scratch.normed,
            config.n_embd,
            config.vocab_size,
            &mut self.scratch.logits,
            &mut self.scratch.project,
        );
        trace::global("result_output", position, &self.scratch.logits);
        self.kv.seq_len = position + 1;
        self.kv.update_access();
        Ok(())
    }

    fn forward_layer(&mut self, layer_index: usize, position: usize, pool: &Arc<ComputePool>) {
        let config = self.model.config();
        let layer = &self.model.layers[layer_index];
        let n_embd = config.n_embd;

        rms_norm(
            &self.scratch.hidden,
            &layer.attn_norm,
            &mut self.scratch.normed,
            config.eps,
        );
        trace::layer_1d("attn_norm", layer_index, position, &self.scratch.normed);
        project(
            pool,
            &layer.wq,
            &self.scratch.normed,
            n_embd,
            config.n_head * config.n_embd_head_k,
            &mut self.scratch.q,
            &mut self.scratch.project,
        );
        project(
            pool,
            &layer.wk,
            &self.scratch.normed,
            n_embd,
            config.n_head_kv * config.n_embd_head_k,
            &mut self.scratch.k,
            &mut self.scratch.project,
        );
        project(
            pool,
            &layer.wv,
            &self.scratch.normed,
            n_embd,
            config.n_head_kv * config.n_embd_head_v,
            &mut self.scratch.v,
            &mut self.scratch.project,
        );

        // Per-head Q/K RMSNorm with the same eps as the block norm.
        let head_dim = config.n_embd_head_k;
        for head in 0..config.n_head {
            let offset = head * head_dim;
            crate::ops::rms_norm_inplace(
                &mut self.scratch.q[offset..offset + head_dim],
                &layer.q_norm,
                config.eps,
            );
        }
        for head in 0..config.n_head_kv {
            let offset = head * head_dim;
            crate::ops::rms_norm_inplace(
                &mut self.scratch.k[offset..offset + head_dim],
                &layer.k_norm,
                config.eps,
            );
        }
        trace::layer_1d("Qcur_normed", layer_index, position, &self.scratch.q);
        trace::layer_1d("Kcur_normed", layer_index, position, &self.scratch.k);
        trace::layer_1d("Vcur", layer_index, position, &self.scratch.v);

        build_rope_cache(
            config.rope_for(layer_index),
            config.freq_base,
            &mut self.scratch.rope_cos,
            &mut self.scratch.rope_sin,
            position,
        );
        for head in 0..config.n_head {
            let offset = head * head_dim;
            apply_rope(
                &mut self.scratch.q[offset..offset + head_dim],
                &self.scratch.rope_cos,
                &self.scratch.rope_sin,
            );
        }
        for head in 0..config.n_head_kv {
            let offset = head * head_dim;
            apply_rope(
                &mut self.scratch.k[offset..offset + head_dim],
                &self.scratch.rope_cos,
                &self.scratch.rope_sin,
            );
        }
        trace::layer_1d("Qcur", layer_index, position, &self.scratch.q);
        trace::layer_1d("Kcur", layer_index, position, &self.scratch.k);

        self.store_kv(layer_index, position);
        self.attend(layer_index, position);
        trace::layer_1d("kqv", layer_index, position, &self.scratch.attn);
        // `kqv_out` is ggml's name for the same tensor; the oracle dumps both.
        trace::layer_1d("kqv_out", layer_index, position, &self.scratch.attn);
        project(
            pool,
            &layer.wo,
            &self.scratch.attn,
            config.n_attn(),
            n_embd,
            &mut self.scratch.attn_proj,
            &mut self.scratch.project,
        );

        // Mellum's parallel residual: the FFN consumes `attn_out + hidden` and
        // its output is added to that same sum, not to the pre-attention state.
        let scratch = &mut self.scratch;
        for index in 0..n_embd {
            scratch.ffn_inp[index] = scratch.attn_proj[index] + scratch.hidden[index];
        }
        trace::layer_1d("ffn_inp", layer_index, position, &scratch.ffn_inp);
        rms_norm(
            &scratch.ffn_inp,
            &layer.ffn_norm,
            &mut scratch.ffn_normed,
            config.eps,
        );
        trace::layer_1d("ffn_norm", layer_index, position, &scratch.ffn_normed);
        {
            let scratch = &mut self.scratch;
            moe_forward(
                pool,
                layer,
                config,
                layer_index,
                position,
                &scratch.ffn_normed,
                &mut scratch.moe_out,
                &mut scratch.router,
                &mut scratch.router_order,
                &mut scratch.expert_ids,
                &mut scratch.expert_weights,
                &mut scratch.gate_buf,
                &mut scratch.up_buf,
                &mut scratch.down_buf,
                &mut scratch.moe_debug,
                &mut scratch.project,
            );
        }
        let scratch = &mut self.scratch;
        trace::layer_1d("ffn_moe_out", layer_index, position, &scratch.moe_out);
        for index in 0..n_embd {
            scratch.hidden[index] = scratch.moe_out[index] + scratch.ffn_inp[index];
        }
        trace::layer_1d("ffn_out", layer_index, position, &scratch.hidden);
        trace::layer_1d("l_out", layer_index, position, &scratch.hidden);
    }

    fn store_kv(&mut self, layer_index: usize, position: usize) {
        let config = &self.model.config;
        let kv_heads_k = config.n_head_kv * config.n_embd_head_k;
        let kv_heads_v = config.n_head_kv * config.n_embd_head_v;
        let stride = config.n_head_kv * config.n_embd_head_k.max(config.n_embd_head_v);
        let base = (layer_index * self.capacity + position) * stride;
        let scratch = &mut self.scratch;
        match &mut self.kv.cache {
            KvCache::F32(cache) => {
                cache.k[base..base + kv_heads_k].copy_from_slice(&scratch.k);
                cache.v[base..base + kv_heads_v].copy_from_slice(&scratch.v);
            }
            KvCache::F16(cache) => {
                for lane in 0..kv_heads_k {
                    cache.k[base + lane] = crate::ops::f32_to_f16(scratch.k[lane]);
                }
                for lane in 0..kv_heads_v {
                    cache.v[base + lane] = crate::ops::f32_to_f16(scratch.v[lane]);
                }
            }
        }
    }

    fn attend(&mut self, layer_index: usize, position: usize) {
        let model = self.model;
        let config = model.config();
        let kv_cache = &self.kv.cache;
        let capacity = self.capacity;
        let scratch = &mut self.scratch;
        let q = &scratch.q;
        let attn = &mut scratch.attn;
        let attn_scratch = &mut scratch.attn_scratch;
        attend_heads(
            kv_cache,
            capacity,
            config,
            layer_index,
            position,
            q,
            attn,
            attn_scratch,
        );
        trace::layer_2d(
            "kq",
            layer_index,
            position,
            config.n_head,
            position + 1,
            &attn_scratch.kq_debug,
        );
        trace::layer_2d(
            "kq_soft_max",
            layer_index,
            position,
            config.n_head,
            position + 1,
            &attn_scratch.kq_soft_max_debug,
        );
    }
}

/// One attention step: every head against the keys visible to `position`.
///
/// ggml's `kqv = ggml_mul_mat(v, kq)` is one `ggml_vec_dot_f32` per output
/// lane, i.e. f32 products accumulated in f64 and narrowed once, and the
/// softmax is ggml's two-pass `max` / `expf` / f64 sum. Both the single
/// token walk and the batched prefill call this, so the two paths cannot
/// drift apart.
///
/// `q` holds the query rows of one token (or one row of a batch) and
/// `attn_out` receives the attention output of the same token.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attend_heads(
    kv_cache: &KvCache,
    capacity: usize,
    config: &MellumConfig,
    layer_index: usize,
    position: usize,
    q: &[f32],
    attn_out: &mut [f32],
    scratch: &mut AttnScratch,
) {
    let head_dim = config.n_embd_head_k;
    let head_dim_v = config.n_embd_head_v;
    let stride = config.n_head_kv * head_dim.max(head_dim_v);
    let group = config.n_head / config.n_head_kv;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let swa = config.is_swa(layer_index);
    let first = if swa {
        (position + 1).saturating_sub(config.n_swa)
    } else {
        0
    };
    let visible = position + 1 - first;
    debug_assert!(visible <= scratch.probs.len());
    let base = layer_index * capacity;
    scratch.kq_debug.clear();
    scratch.kq_soft_max_debug.clear();

    for head in 0..config.n_head {
        let kv_head = head / group;
        let q_start = head * head_dim;
        let attn_start = head * head_dim_v;
        // Materialise this head's visible K/V window once; the F16 cache
        // needs the widening anyway.
        let k_start = kv_head * head_dim;
        let v_start = kv_head * head_dim_v;
        {
            let k_window = &mut scratch.k_window[..visible * head_dim];
            let v_window = &mut scratch.v_window[..visible * head_dim_v];
            for index in 0..visible {
                let offset = (base + first + index) * stride;
                let k_slot = index * head_dim;
                let v_slot = index * head_dim_v;
                read_into(
                    kv_cache,
                    true,
                    offset + k_start,
                    head_dim,
                    &mut k_window[k_slot..k_slot + head_dim],
                );
                read_into(
                    kv_cache,
                    false,
                    offset + v_start,
                    head_dim_v,
                    &mut v_window[v_slot..v_slot + head_dim_v],
                );
            }
        }

        // ggml: `wp[i] = sp[i] * scale`, then the mask adds -INF, then
        // `max`, then `expf`, summed in f64 and inverted once in f64.
        let q_row = &q[q_start..q_start + head_dim];
        let k_window = &scratch.k_window[..visible * head_dim];
        let v_window = &scratch.v_window[..visible * head_dim_v];
        let probs = &mut scratch.probs[..visible];
        let attn = &mut attn_out[attn_start..attn_start + head_dim_v];
        let mut max = f32::NEG_INFINITY;
        for index in 0..visible {
            let k_row = &k_window[index * head_dim..(index + 1) * head_dim];
            let score = dot_f32(q_row, k_row, head_dim) * scale;
            probs[index] = score;
            if score > max {
                max = score;
            }
        }

        let mut sum = 0.0f64;
        for index in 0..visible {
            let value = (probs[index] - max).exp();
            probs[index] = value;
            sum += f64::from(value);
        }
        let inv = (1.0f64 / sum) as f32;
        for index in 0..visible {
            probs[index] *= inv;
        }

        // Transpose this head's V window into `attn_transposed` so each
        // output lane is a contiguous row.
        {
            let transposed = &mut scratch.attn_transposed[..visible * head_dim_v];
            for dim in 0..head_dim_v {
                for index in 0..visible {
                    transposed[dim * visible + index] = v_window[index * head_dim_v + dim];
                }
            }
        }
        let transposed = &scratch.attn_transposed[..visible * head_dim_v];
        for dim in 0..head_dim_v {
            attn[dim] = dot_f32(
                probs,
                &transposed[dim * visible..(dim + 1) * visible],
                visible,
            );
        }

        // Debug rows for the whole cache (masked keys = -INF), matching
        // ggml's `kq` / `kq_soft_max` shapes.
        let row = &mut scratch.kq_row[..position + 1];
        let row_probs = &mut scratch.kq_probs[..position + 1];
        for index in 0..first {
            // ggml fills the masked entries with -INF before the softmax,
            // and `kq` is captured before that, so keep the raw dot only
            // for the visible window.
            row[index] = 0.0;
            row_probs[index] = 0.0;
        }
        for index in 0..visible {
            let k_row = &k_window[index * head_dim..(index + 1) * head_dim];
            row[first + index] = dot_f32(q_row, k_row, head_dim);
            row_probs[first + index] = probs[index];
        }
        scratch.kq_debug.extend_from_slice(row);
        scratch.kq_soft_max_debug.extend_from_slice(row_probs);
    }
}

/// Read one contiguous K or V row out of the cache, widening f16 on the fly.
pub(crate) fn read_into(cache: &KvCache, keys: bool, offset: usize, len: usize, out: &mut [f32]) {
    match cache {
        KvCache::F32(cache) => {
            let source = if keys { &cache.k } else { &cache.v };
            out.copy_from_slice(&source[offset..offset + len]);
        }
        KvCache::F16(cache) => {
            let source = if keys { &cache.k } else { &cache.v };
            for (index, value) in out.iter_mut().enumerate() {
                *value = crate::ops::f16_to_f32(source[offset + index]);
            }
        }
    }
}

/// Fill `cos` / `sin` for the layer's flavour at `position`.
pub(crate) fn build_rope_cache(
    flavour: MellumRope,
    freq_base: f32,
    cos: &mut [f32],
    sin: &mut [f32],
    position: usize,
) {
    let half = cos.len();
    match flavour {
        MellumRope::Plain => {
            let scale = freq_base.powf(-2.0 / (2 * half) as f32);
            let mut theta = position as f32;
            for index in 0..half {
                let (value_cos, value_sin) = crate::ops::rope::neox::rope_sin_cos(theta);
                cos[index] = value_cos;
                sin[index] = value_sin;
                theta *= scale;
            }
        }
        MellumRope::Yarn(yarn) => {
            let theta_scale = yarn.theta_scale();
            let mut theta = position as f32;
            for index in 0..half {
                let (value_cos, value_sin) = yarn.sin_cos(yarn.theta_from_extrap(theta, index));
                cos[index] = value_cos;
                sin[index] = value_sin;
                theta *= theta_scale;
            }
        }
    }
}

/// ggml's NEOX `rotate_pairs` with `n_offset = n_rot / 2`.
pub(crate) fn apply_rope(values: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    for index in 0..half {
        let x0 = values[index];
        let x1 = values[index + half];
        values[index] = x0 * cos[index] - x1 * sin[index];
        values[index + half] = x0 * sin[index] + x1 * cos[index];
    }
}

/// Borrow a `&mut [f32]` out of a raw pointer so the pool closure can `Send` it.
/// The slice is the **full** output buffer; the kernel partitions rows itself
/// from `(ith, nth)`, exactly like `qwen3::trunk::forward::project_tokens`.
struct Rows(*mut f32, usize);
unsafe impl Send for Rows {}
unsafe impl Sync for Rows {}
impl Rows {
    /// Hand the worker its slice of the full output buffer.
    ///
    /// # Safety
    /// The caller must own the whole buffer and no other view may be live
    /// while any worker holds this handle. This is the same raw-pointer
    /// split-worker pattern `qwen3::trunk::forward::RowsWrap` uses, and the
    /// pool joins every worker before `project` returns.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self) -> &mut [f32] {
        std::slice::from_raw_parts_mut(self.0, self.1)
    }
}

/// Single-row quantized matmul, on the CPU.
///
/// The Mellum trunk evaluates one token at a time, so every projection is a
/// single row. The device only wins when a projection carries many rows: a
/// one-row matmul reads the whole weight matrix once, and the Q8_0 device
/// kernel walks the reduction axis with a stride, so it reaches about half
/// the bandwidth the NEON path gets. Measured on this GB10 box with the
/// 12B Q8_0 checkpoint (4 steps, 21 token prompt, 8 threads):
///
/// | projection | CPU | Vulkan |
/// | --- | --- | --- |
/// | output head (2304 -> 98304) | 2.6 ms/token | 4.0 ms/token |
///
/// plus a one-off ~320 ms device setup (the 226 MB weight upload and the
/// arena), so `--compute auto` was 40% *slower* than `--compute cpu` for the
/// decode loop while producing bit-identical output. The projections
/// therefore stay on the CPU until the trunk can hand the device a whole
/// token stream at once; see the TODO entry for the batched prefill.
///
/// The scope only covers this call, so the CPU kernel keeps its own
/// dispatch and the pool workers see the same decision.
pub(crate) fn project(
    pool: &Arc<ComputePool>,
    weight: &Weight<'_>,
    input: &[f32],
    n_in: usize,
    n_out: usize,
    output: &mut [f32],
    scratch: &mut ProjectScratch,
) {
    debug_assert_eq!(input.len(), n_in);
    debug_assert_eq!(output.len(), n_out);
    if n_out == 0 {
        return;
    }
    #[cfg(feature = "vulkan")]
    let _cpu_scope = crate::core::thread_pool::ComputePool::disable_gpu_matmul_for_scope();
    weight.quantize_and_matmul_with_scratch(
        input,
        &mut scratch.q8k_buf,
        &mut scratch.q8_buf,
        &mut scratch.scale_buf,
        output,
        pool,
    );
}

/// ggml's `build_moe_ffn` for `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` with
/// `norm_w = true` and `w_scale = 0`.
#[allow(clippy::too_many_arguments)]
fn moe_forward(
    pool: &Arc<ComputePool>,
    layer: &MellumLayerWeights<'_>,
    config: &MellumConfig,
    layer_index: usize,
    step: usize,
    input: &[f32],
    output: &mut [f32],
    router: &mut [f32],
    router_order: &mut [u32],
    expert_ids: &mut [u32],
    expert_weights: &mut [f32],
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    down_buf: &mut [f32],
    expert_debug: &mut MellumMoeDebug,
    project_scratch: &mut ProjectScratch,
) {
    let n_embd = config.n_embd;
    let n_expert = config.expert_count;
    let n_used = config.expert_used_count;

    // `ggml_mul_mat(gate_inp, cur)` — the router stays F32, so this is a plain
    // f32 matvec with ggml's dot accumulation.
    for expert in 0..n_expert {
        let row = &layer.moe_router[expert * n_embd..(expert + 1) * n_embd];
        router[expert] = dot_f32(input, row, n_embd);
    }
    trace::layer_1d("ffn_moe_logits", layer_index, step, router);
    ggml_softmax(router);
    trace::layer_1d("ffn_moe_probs", layer_index, step, router);

    for (index, slot) in router_order.iter_mut().enumerate() {
        *slot = index as u32;
    }
    router_order.sort_by(|&a, &b| {
        router[b as usize]
            .partial_cmp(&router[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    for index in 0..n_used {
        expert_ids[index] = router_order[index];
    }
    // `ggml_argsort_top_k` is `ggml_argsort(DESC)` + a view, so the slot
    // order is plain descending. (`ggml_top_k` — the op that swaps the first
    // two slots — is a different entry point that Mellum does not use.)
    trace::ids("ffn_moe_topk", layer_index, step, expert_ids);

    let mut sum = 0.0f64;
    for index in 0..n_used {
        let probability = router[expert_ids[index] as usize];
        expert_weights[index] = probability;
        sum += f64::from(probability);
    }
    // ggml: `cb(weights)` fires before the norm_w division.
    trace::layer_1d("ffn_moe_weights", layer_index, step, expert_weights);
    // ggml_sum_rows narrows the f64 accumulator to f32, then clamps, then
    // divides in f32.
    let sum = (sum as f32).max(MOE_WEIGHT_SUM_FLOOR);
    for weight in expert_weights.iter_mut() {
        *weight /= sum;
    }
    trace::layer_1d("ffn_moe_weights_norm", layer_index, step, expert_weights);

    for index in 0..n_used {
        let expert = expert_ids[index] as usize;
        let weight = expert_weights[index];
        project(
            pool,
            &layer.moe_up[expert],
            input,
            n_embd,
            config.expert_ffn,
            up_buf,
            project_scratch,
        );
        project(
            pool,
            &layer.moe_gate[expert],
            input,
            n_embd,
            config.expert_ffn,
            gate_buf,
            project_scratch,
        );
        let ffn_slot = index * config.expert_ffn;
        expert_debug.up[ffn_slot..ffn_slot + config.expert_ffn]
            .copy_from_slice(&up_buf[..config.expert_ffn]);
        expert_debug.gate[ffn_slot..ffn_slot + config.expert_ffn]
            .copy_from_slice(&gate_buf[..config.expert_ffn]);
        for (gate, up) in gate_buf.iter_mut().zip(&*up_buf) {
            *gate = silu(*gate) * *up;
        }
        expert_debug.swiglu[ffn_slot..ffn_slot + config.expert_ffn]
            .copy_from_slice(&gate_buf[..config.expert_ffn]);
        project(
            pool,
            &layer.moe_down[expert],
            gate_buf,
            config.expert_ffn,
            n_embd,
            down_buf,
            project_scratch,
        );
        let down_slot = index * n_embd;
        expert_debug.down[down_slot..down_slot + n_embd].copy_from_slice(&down_buf[..n_embd]);
        // `ggml_mul(down, weight)` then the left-folded `ggml_add` chain.
        // ggml weights every expert column *before* folding them, so the
        // trace column is `down * weight`, not the running sum.
        for lane in 0..n_embd {
            expert_debug.weighted[down_slot + lane] = down_buf[lane] * weight;
        }
        if index == 0 {
            output.copy_from_slice(&expert_debug.weighted[down_slot..down_slot + n_embd]);
        } else {
            for lane in 0..n_embd {
                output[lane] += expert_debug.weighted[down_slot + lane];
            }
        }
    }
    let used = config.expert_used_count;
    trace::layer_2d(
        "ffn_moe_up",
        layer_index,
        step,
        used,
        config.expert_ffn,
        &expert_debug.up,
    );
    trace::layer_2d(
        "ffn_moe_gate",
        layer_index,
        step,
        used,
        config.expert_ffn,
        &expert_debug.gate,
    );
    trace::layer_2d(
        "ffn_moe_swiglu",
        layer_index,
        step,
        used,
        config.expert_ffn,
        &expert_debug.swiglu,
    );
    trace::layer_2d(
        "ffn_moe_down",
        layer_index,
        step,
        used,
        n_embd,
        &expert_debug.down,
    );
    trace::layer_2d(
        "ffn_moe_weighted",
        layer_index,
        step,
        used,
        n_embd,
        &expert_debug.weighted,
    );
}

/// Per-selected-expert intermediates, kept only so the parity trace can emit
/// the same checkpoints the oracle's `build_moe_ffn` labels.
#[derive(Debug, Clone)]
pub struct MellumMoeDebug {
    up: Vec<f32>,
    gate: Vec<f32>,
    swiglu: Vec<f32>,
    down: Vec<f32>,
    weighted: Vec<f32>,
}

impl MellumMoeDebug {
    pub fn new(config: &MellumConfig) -> Self {
        let ffn = config.expert_ffn * config.expert_used_count;
        let down = config.n_embd * config.expert_used_count;
        Self {
            up: vec![0.0; ffn],
            gate: vec![0.0; ffn],
            swiglu: vec![0.0; ffn],
            down: vec![0.0; down],
            weighted: vec![0.0; down],
        }
    }
}

/// ggml's `soft_max` with `scale = 1.0` and no mask: subtract the row max,
/// `expf`, accumulate the sum in f64, then one f32 reciprocal multiply.
pub(crate) fn ggml_softmax(values: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &value in values.iter() {
        if value > max {
            max = value;
        }
    }
    let mut sum = 0.0f64;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += f64::from(*value);
    }
    let inv = (1.0f64 / sum) as f32;
    for value in values.iter_mut() {
        *value *= inv;
    }
}
