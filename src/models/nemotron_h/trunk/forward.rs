//! Nemotron-3 Nano forward pass.
//!
//! This is a **partial** implementation intended to get the model
//! loading, routing, and tokenization pipeline working end-to-end. The
//! Mamba2 SSM branch is currently a no-op (the SSM tensors are loaded
//! but the SSM forward pass is not implemented; SSM output is treated
//! as zero). Attention and FFN follow the Qwen3 conventions.
//!
//! Once a parity test against the pinned llama.cpp commit exists
//! (see `docs/REFERENCE_IMPLEMENTATIONS.md`), the SSM and any
//! attention-specific quirks (e.g. partial RoPE dim) should be
//! implemented to match the reference.

use half::f16;
use std::sync::Arc;

use super::config::NemotronConfig;
use super::weights::NemotronLayerWeights;

use crate::core::scratchpad::{KvArch, KvCache, KvLifecycle, KvState};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{Kernel, Weight};
use crate::ops::{
    dot_f32_exact, f16_to_f32, f32_slice_to_f16, quantize_q8_0_into, rms_norm, rope_neox_inplace,
    softmax_inplace,
};
use std::io::{self, Write};

pub struct NemotronModel {
    pub config: NemotronConfig,
    pub layers: Vec<NemotronLayerWeights<'static>>,
    pub tok_embd: Weight<'static>,
    pub output_norm: Vec<f32>,
    pub output: Weight<'static>,
}

impl NemotronModel {
    pub fn from_source(source: Arc<dyn TensorSource>) -> Result<Self, String> {
        let config = NemotronConfig::from_source(source.as_ref())?;
        let layers = super::weights::load_layers(source.as_ref(), &config)?;
        let output_norm = crate::core::tensor::load_f32_tensor(
            source.as_ref(),
            "output_norm.weight",
            &[config.n_embd as u64],
        )?;
        let tok_embd_info = source
            .tensor_info("token_embd.weight")
            .ok_or_else(|| "Missing tensor: token_embd.weight".to_string())?;
        let tok_embd_bytes = source
            .tensor_slice("token_embd.weight")
            .ok_or_else(|| "Missing tensor: token_embd.weight".to_string())?;
        let bytes_static: &'static [u8] =
            unsafe { std::mem::transmute::<&[u8], &'static [u8]>(tok_embd_bytes) };
        let tok_embd = Weight::from_quantized(crate::ops::kernel::QuantizedTensor::from_bytes(
            bytes_static,
            tok_embd_info.ggml_type,
            config.n_embd,
            config.vocab_size,
        ));
        let output_info = source
            .tensor_info("output.weight")
            .ok_or_else(|| "Missing tensor: output.weight".to_string())?;
        let output_bytes = source
            .tensor_slice("output.weight")
            .ok_or_else(|| "Missing tensor: output.weight".to_string())?;
        let output_static: &'static [u8] =
            unsafe { std::mem::transmute::<&[u8], &'static [u8]>(output_bytes) };
        let output = Weight::from_quantized(crate::ops::kernel::QuantizedTensor::from_bytes(
            output_static,
            output_info.ggml_type,
            config.n_embd,
            config.vocab_size,
        ));
        Ok(Self {
            config,
            layers,
            tok_embd,
            output_norm,
            output,
        })
    }
}

pub struct NemotronSession {
    pub model: NemotronModel,
    pub scratch: NemotronScratch,
    pub kv_state: KvState,
    pub next_position: usize,
    pub capacity: usize,
}

pub struct NemotronScratch {
    pub hidden: Vec<f32>,
    pub normed: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub q8_buf: Vec<u8>,
    pub scale_buf: Vec<f32>,
    /// Per-layer SSM conv1d history buffer. Layout
    /// `[layer * kernel * conv_cols + k * conv_cols + c]`
    /// with `kernel = ssm_conv_kernel = 4` and
    /// `conv_cols = d_inner + 2*n_group*d_state = 9728`. Holds the
    /// most recent `kernel-1` input slices so the next token's
    /// causal conv1d can include contributions from past tokens.
    pub ssm_conv_hist: Vec<f32>,
    /// Per-layer SSM scan state. Layout
    /// `[layer * n_group * dt_rank + g * dt_rank + r]`.
    /// Carried across tokens within one prefill, reset per prefill.
    pub ssm_scan_state: Vec<f32>,
    pub scores: Vec<f32>,
    pub logits: Vec<f32>,
    pub ssm_state: Vec<f32>,
}

impl NemotronScratch {
    pub fn new(config: &NemotronConfig, capacity: usize) -> Self {
        let n_attn_q = config.n_head * config.n_embd_head_k;
        let n_attn_kv = config.n_head_kv * config.n_embd_head_k;
        let n_attn_v = config.n_head_kv * config.n_embd_head_v;
        let n_embd = config.n_embd;
        // Per-layer SSM state. Sized so we can index by
        // [layer * stride + offset]. conv_cols =
        // d_inner + 2*n_group*d_state = 9728 for 4B Nano.
        let conv_cols = config.ssm_inner_size
            + 2 * config.ssm_group_count * config.ssm_state_size;
        Self {
            hidden: vec![0.0; capacity * n_embd],
            normed: vec![0.0; n_embd],
            q: vec![0.0; capacity * n_attn_q],
            k: vec![0.0; capacity * n_attn_kv],
            v: vec![0.0; capacity * n_attn_v],
            attn_out: vec![0.0; capacity * n_attn_v],
            ffn_out: vec![0.0; n_embd],
            q8_buf: vec![0u8; n_embd],
            scale_buf: vec![0.0; n_embd.div_ceil(32)],
            scores: vec![0.0; capacity * capacity],
            logits: vec![0.0; config.vocab_size],
            ssm_state: vec![0.0; config.ssm_state_size * config.ssm_inner_size],
            ssm_conv_hist: vec![0.0;
                config.n_layer * config.ssm_conv_kernel * conv_cols],
            // Per-layer scan state shape: (n_head, headdim, d_state).
            // Total per layer = n_head * headdim * d_state
            //                  = dt_rank * (d_inner / dt_rank) * d_state
            //                  = d_inner * d_state.
            // For 4B Nano: 7680 * 128 = 983040 floats/layer = 3.75 MB,
            // 42 layers = 158 MB total. Reasonable.
            ssm_scan_state: vec![0.0;
                config.n_layer * config.ssm_inner_size * config.ssm_state_size],
        }
    }

    /// Reset per-prefill SSM state (conv1d history and scan state).
    /// Called at the top of `prefill` so each sequence starts clean.
    pub fn reset_ssm_state(&mut self) {
        for v in &mut self.ssm_conv_hist {
            *v = 0.0;
        }
        for v in &mut self.ssm_scan_state {
            *v = 0.0;
        }
    }
}

impl NemotronModel {
    /// One forward pass over a sequence of prefill tokens. Returns logits
    /// for the last position. Per-layer attention + residual are wired;
    /// SSM and FFN outputs are zero placeholders until a parity test
    /// pins the SSM math.
    pub fn prefill(&self, token_ids: &[u32], scratch: &mut NemotronScratch) -> Result<Vec<f32>, String> {
        let n = token_ids.len();
        if n == 0 {
            return Err("Nemotron prefill: empty token sequence".into());
        }
        if n > scratch.hidden.len() / self.config.n_embd {
            return Err(format!(
                "Nemotron prefill: need capacity {} but scratch has {}",
                n,
                scratch.hidden.len() / self.config.n_embd
            ));
        }
        // Reset per-prefill SSM state so each sequence starts from a
        // clean conv1d history and zeroed scan state. Without this,
        // leftover state from the previous prefill call would
        // contaminate the next sequence.
        scratch.reset_ssm_state();
        // 1) Embed input tokens.
        for (i, &tid) in token_ids.iter().enumerate() {
            let row_start = i * self.config.n_embd;
            self.tok_embd
                .embedding_lookup(tid, &mut scratch.hidden[row_start..row_start + self.config.n_embd]);
        }
        // 2) Per-layer forward.
        for (layer_idx, lw) in self.layers.iter().enumerate() {
            self.forward_layer(layer_idx, lw, n, scratch)?;
        }
        // 3) Final norm + logits.
        let last_pos = n - 1;
        let off = last_pos * self.config.n_embd;
        rms_norm(
            &scratch.hidden[off..off + self.config.n_embd],
            &self.output_norm,
            &mut scratch.normed,
            self.config.norm_eps,
        );
        let blocks = (self.config.n_embd + 31) / 32;
        quantize_q8_0_into(
            &scratch.normed,
            self.config.n_embd,
            &mut scratch.q8_buf[..self.config.n_embd],
            &mut scratch.scale_buf[..blocks],
        );
        self.output.kernel.forward_prepared(
            &scratch.normed,
            &scratch.q8_buf[..self.config.n_embd],
            &scratch.scale_buf[..blocks],
            None,
            &mut scratch.logits,
            self.config.n_embd,
            self.config.vocab_size,
            0,
            1,
        );
        Ok(scratch.logits.clone())
    }

    fn forward_layer(
        &self,
        layer_idx: usize,
        lw: &NemotronLayerWeights,
        length: usize,
        scratch: &mut NemotronScratch,
    ) -> Result<(), String> {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let n_head = cfg.n_head;
        let n_head_kv = cfg.n_head_kv;
        let head_dim_k = cfg.n_embd_head_k;
        let head_dim_v = cfg.n_embd_head_v;
        let group_size = if n_head_kv > 0 { n_head / n_head_kv } else { 1 };
        let n_attn_q = n_head * head_dim_k;
        let n_attn_kv = n_head_kv * head_dim_k;
        // Attention output dim is per-query-head, not per-KV-head. With
        // GQA each Q head produces its own output; the wo matrix then
        // projects back to n_embd.
        let n_attn_v = n_head * head_dim_v;
        let kq_scale = 1.0 / (head_dim_k as f32).sqrt();

        for t in 0..length {
            let off = t * n_embd;
            let row = &mut scratch.hidden[off..off + n_embd];
            // Pre-attention norm.
            rms_norm(row, &lw.attn_norm, &mut scratch.normed, cfg.norm_eps);
            // QKV projections (Q8 matmul, SIMD via existing Q8 path).
            let mut q = vec![0.0f32; n_attn_q];
            let mut k = vec![0.0f32; n_attn_kv];
            let mut v = vec![0.0f32; n_attn_v];
            let blocks = (n_embd + 31) / 32;
            quantize_q8_0_into(
                &scratch.normed,
                n_embd,
                &mut scratch.q8_buf[..n_embd],
                &mut scratch.scale_buf[..blocks],
            );
            let q8 = &scratch.q8_buf[..n_embd];
            let sc = &scratch.scale_buf[..blocks];
            // Entire attention branch (QKV + QK norm + RoPE + causal attn
            // + out projection) only runs on attention layers. SSM / FFN
            // layers skip it; their pre-norm `attn_norm` is consumed by the
            // SSM/FFN branch below.
                if let (Some(wq), Some(wk), Some(wv)) = (&lw.wq, &lw.wk, &lw.wv) {
                    wq.kernel
                        .forward_prepared(&scratch.normed, q8, sc, None, &mut q, n_embd, n_attn_q, 0, 1);
                    wk.kernel
                        .forward_prepared(&scratch.normed, q8, sc, None, &mut k, n_embd, n_attn_kv, 0, 1);
                    wv.kernel
                        .forward_prepared(&scratch.normed, q8, sc, None, &mut v, n_embd, n_attn_v, 0, 1);
                    // Persist per-token K and V into the layer's
                    // scratch cache so the next tokens can attend to
                    // this token. Without this write, the K/V cache
                    // access below reads only the current token (and
                    // panics on out-of-bounds for j > 0).
                    scratch.k[t * n_attn_kv..(t + 1) * n_attn_kv]
                        .copy_from_slice(&k);
                    scratch.v[t * n_attn_v..(t + 1) * n_attn_v]
                        .copy_from_slice(&v);
                if let (Some(qn), Some(kn)) = (&lw.attn_q_norm, &lw.attn_k_norm) {
                    for h in 0..n_head {
                        let o = h * head_dim_k;
                        crate::ops::rms_norm_inplace(
                            &mut q[o..o + head_dim_k],
                            qn,
                            cfg.norm_eps,
                        );
                    }
                    for h in 0..n_head_kv {
                        let o = h * head_dim_k;
                        crate::ops::rms_norm_inplace(
                            &mut k[o..o + head_dim_k],
                            kn,
                            cfg.norm_eps,
                        );
                    }
                }
                for h in 0..n_head {
                    rope_neox_inplace(
                        &mut q[h * head_dim_k..h * head_dim_k + head_dim_k],
                        t,
                        head_dim_k,
                        cfg.rope_freq_base,
                    );
                }
                for h in 0..n_head_kv {
                    rope_neox_inplace(
                        &mut k[h * head_dim_k..h * head_dim_k + head_dim_k],
                        t,
                        head_dim_k,
                        cfg.rope_freq_base,
                    );
                }
                let mut attn_out = vec![0.0f32; n_attn_v];
                // q/k/v each hold n_attn_q / n_attn_kv / n_attn_v values for
                // a single token. Convert ALL of them to f16, not just the
                // first head — each head reads its own slot.
                let mut q_f16 = vec![0u16; n_attn_q];
                f32_slice_to_f16(&q[..n_attn_q], &mut q_f16);
                // Convert ALL cached K rows to f16 in one shot — they
                // were written into scratch.k above. (Old code accessed
                // a freshly-allocated per-token `k` vector, which was
                // out of bounds for j > 0.)
                let mut k_f16_storage: Vec<Vec<u16>> =
                    (0..=t).map(|_| vec![0u16; n_attn_kv]).collect();
                for j in 0..=t {
                    f32_slice_to_f16(
                        &scratch.k[j * n_attn_kv..(j + 1) * n_attn_kv],
                        &mut k_f16_storage[j],
                    );
                }
                let mut v_storage: Vec<Vec<f32>> =
                    (0..=t).map(|_| vec![0.0f32; n_attn_v]).collect();
                for j in 0..=t {
                    v_storage[j].copy_from_slice(
                        &scratch.v[j * n_attn_v..(j + 1) * n_attn_v],
                    );
                }
                for h in 0..n_head {
                    let kv_h = h / group_size;
                    let mut scores = vec![0.0f32; t + 1];
                    for j in 0..=t {
                        let k_row = &k_f16_storage[j]
                            [kv_h * head_dim_k..(kv_h + 1) * head_dim_k];
                        let q_row = &q_f16[h * head_dim_k..(h + 1) * head_dim_k];
                        scores[j] =
                            crate::ops::dot_f16(q_row, k_row, head_dim_k) * kq_scale;
                    }
                    softmax_inplace(&mut scores);
                    let head_out =
                        &mut attn_out[h * head_dim_v..(h + 1) * head_dim_v];
                    for j in 0..=t {
                        let s = scores[j];
                        let v_row = &v_storage[j]
                            [kv_h * head_dim_v..(kv_h + 1) * head_dim_v];
                        for d in 0..head_dim_v {
                            head_out[d] += s * v_row[d];
                        }
                    }
                }
                if let Some(wo) = &lw.wo {
                    let blocks2 = (n_attn_v + 31) / 32;
                    quantize_q8_0_into(
                        &attn_out,
                        n_attn_v,
                        &mut scratch.q8_buf[..n_attn_v],
                        &mut scratch.scale_buf[..blocks2],
                    );
                    wo.kernel.forward_prepared(
                        &attn_out,
                        &scratch.q8_buf[..n_attn_v],
                        &scratch.scale_buf[..blocks2],
                        None,
                        &mut scratch.ffn_out,
                        n_attn_v,
                        n_embd,
                        0,
                        1,
                    );
                }
            }
            // Residual: hidden += ffn_out (which is attn_proj for attention
            // layers, zero otherwise). On SSM / FFN layers this still
            // adds the SSM/FFN output below.
            for d in 0..n_embd {
                row[d] += scratch.ffn_out[d];
            }
            // SSM branch: Mamba2 selective-state-space forward.
            //
            // Tensor layout (per Nemotron-3 Nano 4B reference):
            //   ssm_in.weight         (inner, n_embd)        Q4_0   in_proj
            //   ssm_conv1d.weight     (4, 9728)              F32   fused conv + b + c
            //   ssm_conv1d.bias       (9728,)                F32   fused conv + b + c bias
            //   ssm_dt.bias           (dt_rank,)             F32   dt bias
            //   ssm_a                 (1, dt_rank)            F32   A_log
            //   ssm_d                 (1, dt_rank)            F32   D skip
            //   ssm_norm.weight       (per_group, n_groups) F32   group RMSNorm
            //   ssm_out.weight        (n_embd, inner)        Q5_K  out_proj
            //
            // 9728 = 7680 (x after conv) + 1024 + 1024 (B, C groupings).
            //
            // The 4B Nano checkpoint uses a slimmed Mamba2 layout: dt/a/d
            // are per-time_step_rank (96) rather than per-channel, and
            // the in_proj (`ssm_in`) produces only x — no z/B/C/dt
            // concatenated. B/C are picked up from `ssm_conv1d` output's
            // trailing 2048 channels. This implementation approximates the
            // selective scan by routing each (group, rank) to a contiguous
            // 10-channel slot in d_inner and per-rank dt/a/d broadcast:
            //
            //   state[t][g, r] = exp(A[r] * dt[r]) * state[t-1][g, r]
            //               + dt[r] * B[t][g * d_state + r] * x_act[inner(g, r)]
            //   y[inner(g, r)] = C[t][g * d_state + r] * state[t][g, r]
            //                 + D[r] * x_act[inner(g, r)]
            //
            // The exact inner(g, r) ↔ channel mapping is unknown for
            // this 4B Nano variant (no reference commit exposes it), so
            // we use a simple per-group chunking: inner(g, r) =
            // g * (d_inner / n_group) + r * (d_inner / n_group / dt_rank).
            // This produces structurally-valid output but is not
            // guaranteed to match llama.cpp byte-for-byte without a
            // parity test against the exact reference.
            if let (
                Some(ssm_in),
                Some(ssm_conv1d_w),
                Some(ssm_conv1d_b),
                Some(ssm_dt_bias),
                Some(ssm_a_log),
                Some(ssm_d),
                Some(ssm_norm),
                Some(ssm_out),
            ) = (
                &lw.ssm_in,
                &lw.ssm_conv1d_w,
                &lw.ssm_conv1d_b,
                &lw.ssm_dt_bias,
                &lw.ssm_a_log,
                &lw.ssm_d,
                &lw.ssm_norm,
                &lw.ssm_out,
            ) {
                // Input projection: x = ssm_in @ hidden.
                let blocks_in = (n_embd + 31) / 32;
                quantize_q8_0_into(
                    row,
                    n_embd,
                    &mut scratch.q8_buf[..n_embd],
                    &mut scratch.scale_buf[..blocks_in],
                );
                let q8 = &scratch.q8_buf[..n_embd];
                let sc = &scratch.scale_buf[..blocks_in];
                let inner_size = cfg.ssm_inner_size;
                let n_group = cfg.ssm_group_count;
                let d_state = cfg.ssm_state_size;
                let dt_rank = cfg.ssm_time_step_rank;
                let conv_kernel = cfg.ssm_conv_kernel;
                let per_group = inner_size / n_group;
                // ssm_in output layout (matches llama.cpp nemotron-h.cpp):
                //   d_in_proj = 2 * d_inner + 2 * n_group * d_state + dt_rank
                // Split into [x_proj, z_proj, B, C, dt]:
                //   [0  .. d_inner)            = x (will go through conv1d)
                //   [d_inner .. 2*d_inner)      = z (the gate)
                //   [2*d_inner .. 2*d_inner + 2*n_group*d_state) = [B, C]
                //   [2*d_inner + 2*n_group*d_state .. end) = dt (per dt_rank)
                let d_in_proj = 2 * inner_size
                    + 2 * n_group * d_state
                    + dt_rank;
                let z_offset = inner_size;
                let b_offset = 2 * inner_size;
                let dt_offset = 2 * inner_size + 2 * n_group * d_state;
                // 9728 = d_inner + 2 * n_group * d_state (the conv1d fused
                // input is just [x, B, C]; z and dt are split out from
                // ssm_in's larger output).
                let conv_out_cols = inner_size + 2 * n_group * d_state;
                let mut ssm_in_out = vec![0.0f32; d_in_proj];
                ssm_in.kernel.forward_prepared(
                    row,
                    q8,
                    sc,
                    None,
                    &mut ssm_in_out,
                    n_embd,
                    d_in_proj,
                    0,
                    1,
                );
                // Causal depthwise conv1d producing [x_conv, B, C] in the
                // fused output. The conv1d weight has shape
                // (kernel=4, channels=conv_out_cols); for each output
                // channel c:
                //   conv_out[t, c] = sum_{k=0..K-1} weight[k, c] *
                //                     history[k, c] + bias[c]
                // where history[0] = current input, history[k] = input
                // t-k. After the conv1d we shift the buffer so the
                // current input becomes history[1] for the next token
                // (and history[0] gets a fresh write).
                let hist_base = layer_idx * conv_kernel * conv_out_cols;
                let cur_input = &ssm_in_out[..conv_out_cols];
                let mut conv_out = ssm_conv1d_b.to_vec();
                for ki in 0..conv_kernel {
                    let w_row = ki * conv_out_cols;
                    let hist_row = hist_base + ki * conv_out_cols;
                    for c in 0..conv_out_cols {
                        conv_out[c] += ssm_conv1d_w[w_row + c]
                            * scratch.ssm_conv_hist[hist_row + c];
                    }
                }
                // Shift history[1..K] <- history[0..K-1] so the
                // current input becomes history[1] for the next
                // token. We do the shift in-place: start from the
                // oldest tap and move downward.
                for k in (1..conv_kernel).rev() {
                    let dst_off = hist_base + k * conv_out_cols;
                    let src_off = hist_base + (k - 1) * conv_out_cols;
                    scratch.ssm_conv_hist
                        .copy_within(src_off..src_off + conv_out_cols, dst_off);
                }
                // Write current input into history[0].
                let h0 = &mut scratch.ssm_conv_hist[hist_base..hist_base + conv_out_cols];
                h0.copy_from_slice(cur_input);
                // The conv_out contains: x_conv (first d_inner channels),
                // B, C (next 2 * n_group * d_state channels). Per
                // llama.cpp mamba-base.cpp:
                //   xBC = ggml_silu(xBC)   // SiLU applied to ALL channels
                // then B = silu(conv_B), C = silu(conv_C) feed the
                // scan, x = silu(conv_x) feeds both the scan and the
                // D-skip. The z gate is APPLIED AFTER the scan and
                // D-skip via ggml_swiglu_split(z, y) = silu(z) * y.
                // So x is NOT gated by z here; gating happens later.
                let x_pre: Vec<f32> = conv_out[..inner_size]
                    .iter()
                    .map(|&v| crate::ops::silu(v))
                    .collect();
                // B and C are the trailing 2 * n_group * d_state
                // channels of the conv1d output, stored as (n_group *
                // d_state) per token. dt_rank = 96 controls 10
                // d_inner channels each (inner / (n_group * dt_rank) =
                // 7680 / (8 * 96) = 10).
                let b: Vec<f32> = conv_out[inner_size..inner_size + n_group * d_state].to_vec();
                let c: Vec<f32> = conv_out[inner_size + n_group * d_state..].to_vec();
                // Canonical Mamba2 scan (matches llama.cpp's
                // ggml_compute_forward_ssm_scan_f32 in ggml-cpu/ops.cpp):
                //
                //   n_head = ssm_dt_rank        (= 96)
                //   headdim = d_inner / n_head  (= 80)
                //   d_state = nc                (= 128)
                //   n_group = ng                (= 8)
                //   heads_per_group = n_head / n_group (= 12)
                //
                // State shape per layer: (n_head, headdim, d_state).
                // For each (head, channel-in-head, state-element):
                //   g = head / heads_per_group         # B[g], C[g] are shared
                //   x_dt = x[head, k] * dt[head]
                //   state[h, k, n] = state[h, k, n] * exp(dt * A_log[h])
                //                   + B[g, n] * x_dt
                // Output for each (head, channel-in-head):
                //   y[head, k] = sum_n state[h, k, n] * C[g, n] + D * x_orig[head, k]
                //
                // Key differences from the previous (incorrect)
                // implementation:
                //   * state is per (head, channel-in-head, state-element),
                //     not a scalar per (group, rank).
                //   * y[head, k] is a single scalar (the dot product of
                //     state and C), not a scalar multiplied across 10
                //     inner channels.
                //   * B, C are vector-valued per group (size d_state),
                //     shared across heads within the group.
                //   * dt is per-head (size n_head, from
                //     ssm_in_out[dt_offset..dt_offset+n_head]), not
                //     per-time_step_rank.
                let n_head = cfg.ssm_n_head();
                let headdim = cfg.ssm_headdim();
                let heads_per_group = n_head / n_group;
                let state_stride_layer = n_head * headdim * d_state;
                let state_base = layer_idx * state_stride_layer;
                let state_end = state_base + state_stride_layer;
                let mut state: Vec<f32> = scratch.ssm_scan_state[state_base..state_end].to_vec();
                let mut y_buf = vec![0.0f32; inner_size];
                // Snapshot dt per head, then apply softplus. dt_base is
                // a per-head projection of x (size n_head).
                let mut dt_per_head = [0.0f32; 96];
                for h in 0..n_head {
                    // The GGUF stores ssm_dt.bias as (dt_rank,) but
                    // n_head = dt_rank for this model, so it pairs 1:1
                    // with heads in head order.
                    let dt_bias_h = if h < dt_rank {
                        ssm_dt_bias[h]
                    } else {
                        0.0
                    };
                    let dt_base = ssm_in_out[dt_offset + h];
                    let z = dt_base + dt_bias_h;
                    dt_per_head[h] = if z > 20.0 {
                        z
                    } else if z < -20.0 {
                        z.exp()
                    } else {
                        z.exp().ln_1p()
                    };
                }
                // Per-head dA factor (decay): exp(dt_soft_plus * A_log[h]).
                // A_log stored raw (negative). llama.cpp matches this.
                let mut dA_per_head = [0.0f32; 96];
                for h in 0..n_head {
                    dA_per_head[h] = (dt_per_head[h] * ssm_a_log[h]).exp();
                }
                // Scan loop. The canonical layout (see llama.cpp
                // ggml_compute_forward_ssm_scan_f32):
                //   * state[(h * headdim + k) * d_state + n] for the
                //     d_state-dim state vector at (h, k).
                //   * B[g, n] = b[g * d_state + n]
                //   * C[g, n] = c[g * d_state + n]
                //   * x_dt = silu(conv_x)[h * headdim + k] * dt_per_head[h]
                // The D-skip in llama.cpp is `D * silu(conv_x)` (NOT
                // gated by z), and the z-gate is applied AFTER the
                // scan and D-skip via ggml_swiglu_split(z, y) =
                // silu(z) * y. See mamba-base.cpp lines 283-285:
                //   y = ggml_add(y, ggml_mul(x, ssm_d));   // +D*silu(conv_x)
                //   y = ggml_swiglu_split(z, y);            // silu(z) * y
                for h in 0..n_head {
                    let g = h / heads_per_group;
                    let g_b_off = g * d_state;
                    let g_c_off = g * d_state;
                    let head_x_off = h * headdim;
                    let head_state_off = h * headdim * d_state;
                    let dA = dA_per_head[h];
                    let dt_h = dt_per_head[h];
                    let d_h = ssm_d[h];
                    for k in 0..headdim {
                        // x_dt uses the post-SiLU conv1d x (no z gate).
                        let x_dt = x_pre[head_x_off + k] * dt_h;
                        let state_row_off = head_state_off + k * d_state;
                        let mut sumf = 0.0f32;
                        for n in 0..d_state {
                            let state_idx = state_row_off + n;
                            let b_gn = b[g_b_off + n];
                            let c_gn = c[g_c_off + n];
                            let new_state = state[state_idx] * dA + b_gn * x_dt;
                            state[state_idx] = new_state;
                            sumf += new_state * c_gn;
                        }
                        // D-skip on post-SiLU conv1d x, NOT gated by z.
                        y_buf[head_x_off + k] = sumf + d_h * x_pre[head_x_off + k];
                    }
                }
                // z-gate applied AFTER the scan+D-skip:
                //   y_final[h, k] = silu(z[h, k]) * y_buf[h, k]
                // (equivalent to ggml_swiglu_split(z, y_buf)).
                let z_slice = &ssm_in_out[z_offset..z_offset + inner_size];
                for j in 0..inner_size {
                    y_buf[j] = crate::ops::silu(z_slice[j]) * y_buf[j];
                }
                // Persist scan state for the next token in the same
                // prefill.
                scratch.ssm_scan_state[state_base..state_end].copy_from_slice(&state);
                // Group RMSNorm on the scan output.
                for g in 0..n_group {
                    let group_start = g * per_group;
                    let group_end = group_start + per_group;
                    let group = &mut y_buf[group_start..group_end];
                    let mean_sq = group.iter().map(|x| x * x).sum::<f32>() / per_group as f32;
                    let rstd = 1.0 / (mean_sq + cfg.norm_eps).sqrt();
                    for (j, v) in group.iter_mut().enumerate() {
                        *v = *v * rstd * ssm_norm[g * per_group + j];
                    }
                }
                // Output projection: y = ssm_out @ y_buf.
                let blocks_out = (inner_size + 31) / 32;
                quantize_q8_0_into(
                    &y_buf,
                    inner_size,
                    &mut scratch.q8_buf[..inner_size],
                    &mut scratch.scale_buf[..blocks_out],
                );
                ssm_out.kernel.forward_prepared(
                    &y_buf,
                    &scratch.q8_buf[..inner_size],
                    &scratch.scale_buf[..blocks_out],
                    None,
                    &mut scratch.ffn_out,
                    inner_size,
                    n_embd,
                    0,
                    1,
                );
                // DEBUG: disable SSM residual to isolate attention/FFN
                // let _ = scratch.ffn_out; // skip residual
                for d in 0..n_embd {
                    row[d] += scratch.ffn_out[d];
                }
            }
            // FFN branch (plain 2-layer): only if FFN weights are present.
            // Some Nemotron-H layers appear to skip FFN. When ffn_norm is
            // absent, fall back to attn_norm (the model's pre-norm is
            // reused as the FFN's pre-norm).
            if let (Some(w_up), Some(w_down)) = (&lw.w_up, &lw.w_down) {
                let ffn_norm_ref: &[f32] = lw
                    .ffn_norm
                    .as_deref()
                    .unwrap_or(&lw.attn_norm);
                rms_norm(row, ffn_norm_ref, &mut scratch.normed, cfg.norm_eps);
                let mut up_buf = vec![0.0f32; cfg.n_ff];
                let blocks3 = (n_embd + 31) / 32;
                quantize_q8_0_into(
                    &scratch.normed,
                    n_embd,
                    &mut scratch.q8_buf[..n_embd],
                    &mut scratch.scale_buf[..blocks3],
                );
                let q8f = &scratch.q8_buf[..n_embd];
                let scf = &scratch.scale_buf[..blocks3];
                w_up.kernel.forward_prepared(
                    &scratch.normed,
                    q8f,
                    scf,
                    None,
                    &mut up_buf,
                    n_embd,
                    cfg.n_ff,
                    0,
                    1,
                );
                // Activation: ReLU(x)^2 (LLM_FFN_RELU_SQR).
                // llama.cpp's nemotron-h.cpp builds FFN as
                //   build_ffn(cur, ..., LLM_FFN_RELU_SQR, LLM_FFN_PAR, ...)
                // which is `cur = relu(cur); cur = sqr(cur)`. Earlier
                // we used SiLU which is the LLM_FFN_SILU activation —
                // completely wrong for this model.
                for j in 0..cfg.n_ff {
                    let x = up_buf[j];
                    up_buf[j] = if x > 0.0 { x * x } else { 0.0 };
                }
                quantize_q8_0_into(
                    &up_buf,
                    cfg.n_ff,
                    &mut scratch.q8_buf[..cfg.n_ff],
                    &mut scratch.scale_buf[..cfg.n_ff.div_ceil(32)],
                );
                w_down.kernel.forward_prepared(
                    &up_buf,
                    &scratch.q8_buf[..cfg.n_ff],
                    &scratch.scale_buf[..cfg.n_ff.div_ceil(32)],
                    None,
                    &mut scratch.ffn_out,
                    cfg.n_ff,
                    n_embd,
                    0,
            1,
                );
                for d in 0..n_embd {
                    row[d] += scratch.ffn_out[d];
                }
            }
            let _ = layer_idx;
        }
        Ok(())
    }
}

pub fn run_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    _kv_format: crate::app::cli::KvFormat,
) -> Result<(), String> {
    use crate::core::tokenizer::{BPETokenizer, EncodeOptions};

    let _ = (n_threads_arg);
    eprintln!("Loading Nemotron-3 Nano from model");
    let model = NemotronModel::from_source(source.clone())?;
    println!(
        "Model: {} | n_embd={} n_layer={} n_head={} n_head_kv={} vocab={}",
        model.config.architecture,
        model.config.n_embd,
        model.config.n_layer,
        model.config.n_head,
        model.config.n_head_kv,
        model.config.vocab_size,
    );

    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|e| format!("Failed to initialize tokenizer: {e}"))?;
    let prompt_ids = tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    if prompt_ids.is_empty() {
        return Err("Nemotron prompt produced no tokens".into());
    }

    // Scratch sized for the worst-case attention layer (n_head=40, key=128,
    // value=128). KV cache intentionally skipped for the first cut.
    let n_attn_q = model.config.n_head * model.config.n_embd_head_k;
    let n_attn_v = model.config.n_head_kv.max(1) * model.config.n_embd_head_v;
    let scratch_capacity = (prompt_ids.len() + max_tokens).max(8);
    let conv_cols = model.config.ssm_inner_size
        + 2 * model.config.ssm_group_count * model.config.ssm_state_size;
    let mut scratch = NemotronScratch {
        hidden: vec![0.0; scratch_capacity * model.config.n_embd],
        normed: vec![0.0; model.config.n_embd],
        q: vec![0.0; scratch_capacity * n_attn_q],
        k: vec![0.0; scratch_capacity * n_attn_v],
        v: vec![0.0; scratch_capacity * n_attn_v],
        attn_out: vec![0.0; n_attn_v],
        ffn_out: vec![0.0; model.config.n_embd],
        q8_buf: vec![0u8; model.config.n_embd
            .max(model.config.n_ff)
            .max(n_attn_v)
            .max(model.config.n_embd_head_v * model.config.n_head)
            .max(model.config.ssm_inner_size)],
        scale_buf: vec![0.0; model.config.n_embd
            .max(model.config.n_ff)
            .max(n_attn_v)
            .max(model.config.n_embd_head_v * model.config.n_head)
            .max(model.config.ssm_inner_size)
            .div_ceil(32)],
        scores: vec![0.0; scratch_capacity * scratch_capacity],
        logits: vec![0.0; model.config.vocab_size],
        ssm_state: vec![0.0; model.config.ssm_state_size * model.config.ssm_inner_size],
        ssm_conv_hist: vec![0.0;
            model.config.n_layer * model.config.ssm_conv_kernel * conv_cols],
        ssm_scan_state: vec![0.0;
            model.config.n_layer * model.config.ssm_inner_size * model.config.ssm_state_size],
    };

    // Prefill each prompt token as a separate step (no KV cache yet).
    let started = std::time::Instant::now();
    let mut logits = Vec::new();
    for &tid in &prompt_ids {
        logits = model.prefill(&[tid], &mut scratch)?;
    }
    let mut next_token = sample_argmax(&logits, temperature);
    let mut generated: Vec<u32> = vec![next_token];
    let t_prefill = started.elapsed();
    eprintln!(
        "Nemotron: prefill {} tokens in {:.2?}",
        prompt_ids.len(),
        t_prefill
    );

    // Decode one token at a time. Each step reuses the full prefill
    // scratch (we don't have KV cache yet, so this is O(n²)).
    for step in 0..max_tokens {
        let next_logits = model.prefill(&[next_token], &mut scratch)?;
        // DEBUG: dump logits stats + argmax to stderr for parity diff
        // against llama.cpp oracle (commit 96013c511).
        {
            let mut l2 = 0.0f64;
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in next_logits.iter().enumerate() {
                let vv = v as f64;
                l2 += vv * vv;
                if v < mn { mn = v; }
                if v > mx { mx = v; }
                if v > best_v { best_v = v; best = i; }
            }
            eprintln!(
                "[OURS-LOGITS] step {step}: L2={:.3} min={:.3} max={:.3} argmax={best} (logit={:.3})",
                l2.sqrt(), mn, mx, best_v
            );
        }
        next_token = sample_argmax(&next_logits, temperature);
        if let Some(eos) = tokenizer.eos_id() {
            if next_token == eos {
                break;
            }
        }
        generated.push(next_token);
    }
    let piece = tokenizer.decode(&generated, true);
    println!("Output: {}", piece);
    Ok(())
}

fn sample_argmax(logits: &[f32], temperature: f32) -> u32 {
    if temperature == 0.0 {
        let mut best = f32::NEG_INFINITY;
        let mut idx = 0u32;
        for (i, &v) in logits.iter().enumerate() {
            if v > best {
                best = v;
                idx = i as u32;
            }
        }
        idx
    } else {
        sample_argmax(logits, 0.0)
    }
}
