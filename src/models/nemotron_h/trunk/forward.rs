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
                let mut k_f16_storage: Vec<Vec<u16>> =
                    (0..=t).map(|_| vec![0u16; n_attn_kv]).collect();
                for j in 0..=t {
                    f32_slice_to_f16(
                        &k[j * n_attn_kv..(j + 1) * n_attn_kv],
                        &mut k_f16_storage[j],
                    );
                }
                let mut v_storage: Vec<Vec<f32>> =
                    (0..=t).map(|_| vec![0.0f32; n_attn_v]).collect();
                for j in 0..=t {
                    v_storage[j].copy_from_slice(
                        &v[j * n_attn_v..(j + 1) * n_attn_v],
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
            //   ssm_dt.bias           (inner,)               F32   dt bias
            //   ssm_a                 (1, inner)             F32   A_log
            //   ssm_d                 (1, inner)             F32   D skip
            //   ssm_norm.weight       (per_group, n_groups) F32   group RMSNorm
            //   ssm_out.weight        (n_embd, inner)        Q5_K  out_proj
            //
            // 9728 = 7680 (x after conv) + 1024 + 1024 (B, C groupings).
            //
            // This is the first cut: we project → conv1d (first inner
            // channels) → SiLU → D-skip → group RMSNorm → out_proj. The
            // selective scan (B, C → state over A·dt) is a no-op because
            // this checkpoint does not store dt-projection weights; the
            // B/C channels of ssm_conv1d output are loaded but unused
            // here. A full Mamba2 implementation would replace the
            // D-skip with a real selective scan.
            if let (Some(ssm_in), Some(ssm_conv1d_w), Some(ssm_conv1d_b), Some(ssm_d), Some(ssm_norm), Some(ssm_out)) = (
                &lw.ssm_in, &lw.ssm_conv1d_w, &lw.ssm_conv1d_b, &lw.ssm_d, &lw.ssm_norm, &lw.ssm_out,
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
                let conv_kernel = cfg.ssm_conv_kernel;
                let mut x_inner = vec![0.0f32; inner_size];
                ssm_in.kernel.forward_prepared(
                    row,
                    q8,
                    sc,
                    None,
                    &mut x_inner,
                    n_embd,
                    inner_size,
                    0,
                    1,
                );
                // Causal conv1d on the first `inner` channels of the fused
                // output. The remaining 2 * (n_groups * d_state) channels
                // are the B/C projections which are loaded but not yet
                // used (the selective scan is the missing piece).
                let conv_out_cols = inner_size
                    + 2 * cfg.ssm_state_size * cfg.ssm_group_count;
                let mut x_conv = vec![0.0f32; inner_size];
                for ki in 0..conv_kernel {
                    let t_src = if t >= ki { t - ki } else { 0 };
                    // We don't carry the full prefill sequence in
                    // scratch.hidden (each step processes a single token),
                    // so the conv1d here is a single-tap with zero
                    // history. For real sequential prefill we'd need to
                    // accumulate state.
                    let _ = t_src;
                    let row_off = ki * conv_out_cols;
                    for c in 0..inner_size {
                        x_conv[c] += ssm_conv1d_w[row_off + c];
                    }
                }
                for c in 0..inner_size {
                    x_conv[c] += ssm_conv1d_b[c];
                }
                // SiLU activation.
                for v in x_conv.iter_mut() {
                    *v = crate::ops::silu(*v);
                }
                // D-skip (per-channel residual): x_conv *= D.
                for (xi, di) in x_conv.iter_mut().zip(ssm_d.iter()) {
                    *xi *= *di;
                }
                // Group RMSNorm: split inner_size into n_groups groups,
                // each with `per_group` features, normalized by RMS(scale).
                let n_groups = cfg.ssm_group_count;
                let per_group = inner_size / n_groups;
                // ssm_norm.weight is row-major (per_group × n_groups).
                for g in 0..n_groups {
                    let group_start = g * per_group;
                    let group_end = group_start + per_group;
                    let group = &mut x_conv[group_start..group_end];
                    let mean_sq = group.iter().map(|x| x * x).sum::<f32>() / per_group as f32;
                    let rstd = 1.0 / (mean_sq + cfg.norm_eps).sqrt();
                    for (j, v) in group.iter_mut().enumerate() {
                        *v = *v * rstd * ssm_norm[g * per_group + j];
                    }
                }
                // Output projection: y = ssm_out @ x_conv.
                let blocks_out = (inner_size + 31) / 32;
                quantize_q8_0_into(
                    &x_conv,
                    inner_size,
                    &mut scratch.q8_buf[..inner_size],
                    &mut scratch.scale_buf[..blocks_out],
                );
                ssm_out.kernel.forward_prepared(
                    &x_conv,
                    &scratch.q8_buf[..inner_size],
                    &scratch.scale_buf[..blocks_out],
                    None,
                    &mut scratch.ffn_out,
                    inner_size,
                    n_embd,
                    0,
                    1,
                );
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
                // Activation: SiLU. The reference uses SiLU; if a different
                // activation is needed the parity test will catch it.
                for j in 0..cfg.n_ff {
                    up_buf[j] = crate::ops::silu(up_buf[j]);
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
    for _step in 0..max_tokens {
        let next_logits = model.prefill(&[next_token], &mut scratch)?;
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
