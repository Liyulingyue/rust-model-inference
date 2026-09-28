//! Xing4.0-29B-A4B forward pass.
//!
//! Three cooperating mechanisms, in the order a token sees them:
//!
//! 1. **mHC** (hyper-connection) — the residual is `hc_count`
//!    interleaved streams. Before every sub-block the streams are mixed
//!    into one lane by a learned gate (`pre`); after the sub-block the
//!    lane is spread back across the streams by `post` plus a
//!    Sinkhorn-normalised mixing matrix (`comb`). This is DeepSeek-V4's
//!    hyper-connection without the head-side projector — Xing4.0 has no
//!    `hc_head_*` tensors, so the output head averages the streams.
//! 2. **MLA** — queries pass through a rank-`q_lora_rank` bottleneck; the
//!    KV cache holds one shared `kv_lora_rank + rope` latent per token
//!    instead of per-head K/V. Queries are absorbed into the same latent
//!    space by `wk_b`, so the dot products are single-head over the
//!    latent (32x cheaper than materialising every cached key), and
//!    `wv_b` maps the attended latent back to per-head values.
//! 3. **MoE** — the first `n_layer_dense_lead` layers run a dense SwiGLU
//!    FFN; the rest route each token to `n_expert_used` of `n_expert`
//!    experts (sigmoid routing, renormalised weights) plus one shared
//!    expert that always fires.

use super::config::Xing4Config;
use super::weights::{MoeFfn, Xing4LayerWeights, Xing4Weights};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::Weight;
use crate::ops::{quantize_q8_0_into, quantize_row_q8_k_into, rms_norm_grouped};

pub struct Xing4Scratch {
    /// `hc_count` interleaved residual streams, `[hc * n_embd]`.
    pub hc_state: Vec<f32>,
    /// Snapshot of `hc_state` before a sub-block (input to `hc_post`).
    pub residual: Vec<f32>,
    /// RMS-normalised `hc_state`, input to the mHC gate matmuls.
    pub hc_normed: Vec<f32>,
    /// Mixed lane fed to one sub-block, `[n_embd]`.
    pub lane: Vec<f32>,
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    pub comb: Vec<f32>,
    pub mixes: Vec<f32>,
    /// RMS-normed lane, `[n_embd]`; also reused as the normed `q_a`
    /// buffer, so it is sized `max(n_embd, q_lora_rank)`.
    pub normed: Vec<f32>,
    /// `[q_lora_rank]` — raw `attn_q_a` output.
    pub q_a: Vec<f32>,
    /// `[n_head * (nope + rope)]`.
    pub q: Vec<f32>,
    /// `[n_head * (kv_lora_rank + rope)]` — absorbed queries.
    pub q_latent: Vec<f32>,
    /// `[kv_lora_rank]` — RMS-normed latent ready for the cache.
    pub latent: Vec<f32>,
    /// `[kv_lora_rank + rope]` — raw `attn_kv_a_mqa` output.
    pub kv_a: Vec<f32>,
    pub scores: Vec<f32>,
    /// `[n_head * v_dim]` — per-head values before `wo`.
    pub o_heads: Vec<f32>,
    /// `[kv_lora_rank]` scratch for one head's attended latent.
    pub o_latent: Vec<f32>,
    /// Dense/shared-expert FFN hidden.
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub router_probs: Vec<f32>,
    pub moe_out: Vec<f32>,
    /// Weighted sum of the MoE experts, `[n_embd]`.
    pub routed_acc: Vec<f32>,
    pub expert_sel: Vec<usize>,
    pub expert_w: Vec<f32>,
    pub q8_buf: Vec<u8>,
    pub q8_scales: Vec<f32>,
    pub q8k_buf: Vec<crate::ops::quant::BlockQ8K>,
    pub logits: Vec<f32>,
}

impl Xing4Scratch {
    pub fn new(cfg: &Xing4Config, max_ctx: usize) -> Self {
        let hc_dim = cfg.hc_count * cfg.n_embd;
        let ffn_w = cfg.n_ff.max(cfg.n_ff_exp).max(cfg.n_expert_used * cfg.n_ff_exp);
        let q_lat_w = cfg.n_head * (cfg.kv_lora_rank + cfg.n_embd_head_qk_rope);
        let attn_out_w = cfg.n_head * cfg.n_embd_head_v;
        let max_n_in = cfg
            .n_embd
            .max(cfg.n_head * cfg.n_embd_head_qk())
            .max(attn_out_w)
            .max(ffn_w)
            .max(q_lat_w);
        Self {
            hc_state: vec![0.0; hc_dim],
            residual: vec![0.0; hc_dim],
            hc_normed: vec![0.0; hc_dim],
            lane: vec![0.0; cfg.n_embd],
            pre: vec![0.0; cfg.hc_count],
            post: vec![0.0; cfg.hc_count],
            comb: vec![0.0; cfg.hc_count * cfg.hc_count],
            mixes: vec![0.0; cfg.hc_mix_dim()],
            normed: vec![0.0; cfg.n_embd.max(cfg.q_lora_rank)],
            q_a: vec![0.0; cfg.q_lora_rank],
            q: vec![0.0; cfg.n_head * cfg.n_embd_head_qk()],
            q_latent: vec![0.0; q_lat_w],
            latent: vec![0.0; cfg.kv_lora_rank],
            kv_a: vec![0.0; cfg.kv_cache_width()],
            scores: vec![0.0; max_ctx],
            o_heads: vec![0.0; attn_out_w],
            o_latent: vec![0.0; cfg.kv_lora_rank],
            ffn_gate: vec![0.0; ffn_w],
            ffn_up: vec![0.0; ffn_w],
            router_logits: vec![0.0; cfg.n_expert],
            router_probs: vec![0.0; cfg.n_expert],
            moe_out: vec![0.0; cfg.n_embd],
            routed_acc: vec![0.0; cfg.n_embd],
            expert_sel: vec![0; cfg.n_expert_used],
            expert_w: vec![0.0; cfg.n_expert_used],
            q8_buf: vec![0; max_n_in],
            q8_scales: vec![0.0; max_n_in / 32],
            q8k_buf: vec![
                crate::ops::quant::BlockQ8K { d: 0.0, qs: [0; 256], bsums: [0; 16] };
                max_n_in / 256
            ],
            logits: vec![0.0; cfg.n_vocab],
        }
    }
}

pub struct Xing4Runtime<'a> {
    source: &'a dyn TensorSource,
    cfg: Xing4Config,
    layers: Vec<Xing4LayerWeights<'a>>,
    globals: Xing4Weights<'a>,
    pool: ComputePool,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    max_ctx: usize,
    /// `[n_layer * max_ctx * cache_w]` — the shared MLA latent per token.
    kv_cache: Vec<f32>,
}

impl<'a> Xing4Runtime<'a> {
    pub fn new(
        source: &'a dyn TensorSource,
        cfg: Xing4Config,
        n_threads: usize,
        max_ctx: usize,
    ) -> Result<Self, String> {
        let layers = super::weights::load_layers(source, &cfg)?;
        let globals = super::weights::load_global(source, &cfg)?;
        let pool = ComputePool::new(n_threads);
        let max_ctx = max_ctx.min(cfg.n_ctx).max(1);
        let cache_w = cfg.kv_cache_width();
        let kv_cache = vec![0.0f32; cfg.n_layer * max_ctx * cache_w];
        let mut rt = Self {
            source,
            cfg,
            layers,
            globals,
            pool,
            rope_cos: Vec::new(),
            rope_sin: Vec::new(),
            max_ctx,
            kv_cache,
        };
        rt.build_rope();
        Ok(rt)
    }

    pub fn cfg(&self) -> &Xing4Config {
        &self.cfg
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn scratch(&self) -> Xing4Scratch {
        Xing4Scratch::new(&self.cfg, self.max_ctx)
    }

    pub fn clear_cache(&mut self) {
        self.kv_cache.fill(0.0);
    }

    fn build_rope(&mut self) {
        let c = &self.cfg;
        let dim = c.n_embd_head_qk_rope;
        let half = dim / 2;
        let freq_scale = 1.0f64 / c.rope_yarn_factor as f64;
        let base = c.rope_freq_base as f64;
        let n_ctx_orig = c.rope_yarn_orig_ctx as f64;
        let corr = |n_rot: f64| {
            (dim as f64) * (n_ctx_orig / (n_rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let start = corr(c.rope_yarn_beta_fast as f64).floor().max(0.0);
        let end = corr(c.rope_yarn_beta_slow as f64)
            .ceil()
            .min((dim - 1) as f64);
        let span = (end - start).max(0.001);
        for pos in 0..self.max_ctx {
            for i in 0..half {
                let theta_extrap = (pos as f64) * base.powf(-(2.0 * i as f64) / dim as f64);
                let theta_interp = freq_scale * theta_extrap;
                let ramp = 1.0 - ((i as f64 - start) / span).clamp(0.0, 1.0);
                let theta = theta_interp * (1.0 - ramp) + theta_extrap * ramp;
                self.rope_cos.push(theta.cos() as f32);
                self.rope_sin.push(theta.sin() as f32);
            }
        }
    }

    /// Rotate adjacent pairs in place (`GGML_ROPE_TYPE_NORMAL` style).
    fn rope_in_place(&self, x: &mut [f32], pos: usize) {
        let half = self.cfg.n_embd_head_qk_rope / 2;
        let base = pos * half;
        for i in 0..half {
            let a = x[2 * i];
            let b = x[2 * i + 1];
            let cos = self.rope_cos[base + i];
            let sin = self.rope_sin[base + i];
            x[2 * i] = a * cos - b * sin;
            x[2 * i + 1] = a * sin + b * cos;
        }
    }

    pub fn embed(&self, token: u32, out: &mut [f32]) {
        self.globals
            .tok_embd
            .kernel
            .embedding_lookup(token, self.cfg.n_embd, out);
    }

    /// Seed the residual streams with a token's embedding, repeated
    /// `hc_count` times (DeepSeek-V4's `hc_init`).
    pub fn init_hc(&self, token: u32, hc_state: &mut [f32]) {
        let n = self.cfg.n_embd;
        let mut first = vec![0.0f32; n];
        self.globals.tok_embd.kernel.embedding_lookup(token, n, &mut first);
        for s in 0..self.cfg.hc_count {
            hc_state[s * n..(s + 1) * n].copy_from_slice(&first);
        }
    }

    /// Average the hc streams and project to logits.
    pub fn logits_from_hc(&self, hc_state: &[f32], s: &mut Xing4Scratch) -> Vec<f32> {
        let c = &self.cfg;
        let n = c.n_embd;
        let hc = c.hc_count;
        for d in 0..n {
            let mut acc = 0.0f32;
            for i in 0..hc {
                acc += hc_state[i * n + d];
            }
            s.lane[d] = acc / hc as f32;
        }
        rms_norm_grouped(&s.lane, &self.globals.output_norm, &mut s.normed[..n], 1, c.norm_eps);
        let vocab = c.n_vocab;
        self.globals.output.kernel.forward(&s.normed[..n], &mut s.logits[..vocab], n, vocab);
        s.logits[..vocab].to_vec()
    }

    /// Fill `pre` / `post` / `comb` from the mHC gate matmul.
    /// Callers pass the scratch fields separately so the borrow
    /// checker can see they do not alias.
    #[allow(clippy::too_many_arguments)]
    fn hc_gate(
        &self,
        hc_state: &[f32],
        fn_w: &Weight,
        base: &[f32],
        scale: &[f32],
        hc_normed: &mut [f32],
        mixes: &mut [f32],
        pre: &mut [f32],
        post: &mut [f32],
        comb: &mut [f32],
    ) {
        let c = &self.cfg;
        let (hc, n) = (c.hc_count, c.n_embd);
        let flat = n * hc;
        let inv = 1.0 / rms_scale(&hc_state[..flat], c.norm_eps);
        for (dst, src) in hc_normed[..flat].iter_mut().zip(hc_state[..flat].iter()) {
            *dst = *src * inv;
        }
        let mix_dim = c.hc_mix_dim();
        fn_w.kernel
            .forward(&hc_normed[..flat], &mut mixes[..mix_dim], flat, mix_dim);

        let (sp, sq, sr) = (scale[0], scale[1], scale[2]);
        let eps = c.hc_eps;
        for i in 0..hc {
            pre[i] = sigmoid(mixes[i] * sp + base[i]) + eps;
        }
        for i in 0..hc {
            post[i] = sigmoid(mixes[hc + i] * sq + base[hc + i]) * 2.0;
        }
        let off = 2 * hc;
        for i in 0..hc * hc {
            comb[i] = mixes[off + i] * sr + base[off + i];
        }
        sinkhorn(comb, hc, eps, c.hc_sinkhorn_iters);
    }

    /// `lane = Σ_i pre_i * hc_state[i]`
    fn mix_lane(&self, hc_state: &[f32], pre: &[f32], lane: &mut [f32]) {
        let (n, hc) = (self.cfg.n_embd, self.cfg.hc_count);
        for d in 0..n {
            let mut acc = 0.0f32;
            for i in 0..hc {
                acc += pre[i] * hc_state[i * n + d];
            }
            lane[d] = acc;
        }
    }

    /// `streams[dst] = lane * post[dst] + Σ_src comb[dst][src] * residual[src]`
    fn spread_lane(&self, lane: &[f32], post: &[f32], comb: &[f32], residual: &[f32], hc_state: &mut [f32]) {
        let (n, hc) = (self.cfg.n_embd, self.cfg.hc_count);
        for dst in 0..hc {
            for d in 0..n {
                let mut acc = lane[d] * post[dst];
                for src in 0..hc {
                    acc += comb[dst * hc + src] * residual[src * n + d];
                }
                hc_state[dst * n + d] = acc;
            }
        }
    }

    /// Quantize `src` into the shared Q8_0/Q8_K buffers.
    fn prep<'b>(
        src: &[f32],
        len: usize,
        q8: &'b mut [u8],
        scales: &'b mut [f32],
        q8k: &'b mut [crate::ops::quant::BlockQ8K],
    ) {
        quantize_q8_0_into(src, len, &mut q8[..len], &mut scales[..len / 32]);
        quantize_row_q8_k_into(src, &mut q8k[..len / 256]);
    }

    pub fn forward_token(&mut self, pos: usize, hc_state: &mut [f32], s: &mut Xing4Scratch) {
        let c = &self.cfg;
        let n_embd = c.n_embd;
        let hc = c.hc_count;
        let flat = n_embd * hc;
        let nope = c.n_embd_head_k_nope();
        let rope_dim = c.n_embd_head_qk_rope;
        let head_k = c.n_embd_head_qk();
        let kv_lora = c.kv_lora_rank;
        let v_dim = c.n_embd_head_v;
        let cache_w = c.kv_cache_width();
        let q_lat_w = kv_lora + rope_dim;
        let attn_out_w = c.n_head * v_dim;
        let kq_scale = 1.0f32 / (head_k as f32).sqrt();
        let n_cached = pos + 1;

        for il in 0..c.n_layer {
            let lw = &self.layers[il];

            // ===================== attention =====================
            s.residual.copy_from_slice(&hc_state[..flat]);
            {
                let (fn_w, base, scale) = (&lw.hc_attn_fn, &lw.hc_attn_base, &lw.hc_attn_scale);
                self.hc_gate(&s.residual, fn_w, base, scale,
                    &mut s.hc_normed, &mut s.mixes, &mut s.pre, &mut s.post, &mut s.comb);
            }
            self.mix_lane(&s.residual, &s.pre, &mut s.lane);

            rms_norm_grouped(&s.lane, &lw.attn_norm, &mut s.normed[..n_embd], 1, c.norm_eps);
            Self::prep(&s.normed[..n_embd], n_embd, &mut s.q8_buf, &mut s.q8_scales, &mut s.q8k_buf);

            // q lane: n_embd -> q_lora -> n_head * head_k
            lw.wq_a.kernel.forward_prepared(
                &s.normed[..n_embd], &s.q8_buf[..n_embd], &s.q8_scales[..n_embd / 32],
                Some(&s.q8k_buf[..n_embd / 256]), &mut s.q_a[..c.q_lora_rank],
                n_embd, c.q_lora_rank, 0, 1,
            );
            rms_norm_grouped(
                &s.q_a[..c.q_lora_rank],
                &lw.attn_q_a_norm,
                &mut s.normed[..c.q_lora_rank],
                1,
                c.norm_eps,
            );
            lw.wq_b.kernel.forward(&s.normed[..c.q_lora_rank], &mut s.q[..c.n_head * head_k], c.q_lora_rank, c.n_head * head_k);

            // kv latent for this position: n_embd -> cache_w
            lw.wkv_a_mqa.kernel.forward_prepared(
                &s.normed[..n_embd], &s.q8_buf[..n_embd], &s.q8_scales[..n_embd / 32],
                Some(&s.q8k_buf[..n_embd / 256]), &mut s.kv_a[..cache_w],
                n_embd, cache_w, 0, 1,
            );

            // RoPE: per-head query rope part + shared key rope part.
            for h in 0..c.n_head {
                let off = h * head_k + nope;
                self.rope_in_place(&mut s.q[off..off + rope_dim], pos);
            }
            let kpe = kv_lora + rope_dim;
            self.rope_in_place(&mut s.kv_a[kv_lora..kpe], pos);

            // Cache concat(rms_norm(latent), roped key part).
            rms_norm_grouped(
                &s.kv_a[..kv_lora],
                &lw.attn_kv_a_norm,
                &mut s.latent[..kv_lora],
                1,
                c.norm_eps,
            );
            let base = il * self.max_ctx * cache_w + pos * cache_w;
            self.kv_cache[base..base + kv_lora].copy_from_slice(&s.latent[..kv_lora]);
            self.kv_cache[base + kv_lora..base + cache_w].copy_from_slice(&s.kv_a[kv_lora..cache_w]);

            // Absorb queries: q_latent[h] = concat(wk_b[h] @ q_nope[h], q_pe[h]).
            for h in 0..c.n_head {
                let q_off = h * head_k;
                let dst = h * q_lat_w;
                lw.wk_b[h].kernel.forward(&s.q[q_off..q_off + nope], &mut s.q_latent[dst..dst + kv_lora], nope, kv_lora);
                let after = dst + kv_lora;
                s.q_latent[after..after + rope_dim].copy_from_slice(&s.q[q_off + nope..q_off + head_k]);
            }

            // Attention over the latent.
            let layer_base = il * self.max_ctx * cache_w;
            for h in 0..c.n_head {
                let q_off = h * q_lat_w;
                let qv = &s.q_latent[q_off..q_off + q_lat_w];
                for t in 0..n_cached {
                    let off = layer_base + t * cache_w;
                    let mut acc = 0.0f32;
                    for d in 0..cache_w {
                        acc += qv[d] * self.kv_cache[off + d];
                    }
                    s.scores[t] = acc * kq_scale;
                }
                softmax_inplace(&mut s.scores[..n_cached]);
                let o = &mut s.o_latent[..kv_lora];
                o.fill(0.0);
                for t in 0..n_cached {
                    let off = layer_base + t * cache_w;
                    let w = s.scores[t];
                    for (d, slot) in o.iter_mut().enumerate() {
                        *slot += w * self.kv_cache[off + d];
                    }
                }
                lw.wv_b[h].kernel.forward(&s.o_latent[..kv_lora], &mut s.o_heads[h * v_dim..(h + 1) * v_dim], kv_lora, v_dim);
            }

            if std::env::var_os("RUST_XING4_DEBUG").is_some() && il < 2 {
                let st: f64 = s.lane.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let mx = s.lane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                eprintln!("[xing4] il={il} pos={pos} attn_lane sq={st:.3} max={mx:.4}");
            }
            // wo: [n_head * v_dim] -> n_embd
            Self::prep(&s.o_heads, attn_out_w, &mut s.q8_buf, &mut s.q8_scales, &mut s.q8k_buf);
            lw.wo.kernel.forward_prepared(
                &s.o_heads, &s.q8_buf[..attn_out_w], &s.q8_scales[..attn_out_w / 32],
                Some(&s.q8k_buf[..attn_out_w / 256]), &mut s.lane,
                attn_out_w, n_embd, 0, 1,
            );
            let lane = s.lane.clone();
            self.spread_lane(&lane, &s.post, &s.comb, &s.residual, hc_state);
            if std::env::var_os("RUST_XING4_DEBUG").is_some() && il < 2 {
                let st: f64 = hc_state[..flat].iter().map(|&v| (v as f64) * (v as f64)).sum();
                let mx = hc_state[..flat].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                eprintln!("[xing4] il={il} pos={pos} after_attn_post sq={st:.3} max={mx:.4}");
            }

            // ===================== FFN =====================
            s.residual.copy_from_slice(&hc_state[..flat]);
            {
                let (fn_w, base, scale) = (&lw.hc_ffn_fn, &lw.hc_ffn_base, &lw.hc_ffn_scale);
                self.hc_gate(&s.residual, fn_w, base, scale,
                    &mut s.hc_normed, &mut s.mixes, &mut s.pre, &mut s.post, &mut s.comb);
            }
            self.mix_lane(&s.residual, &s.pre, &mut s.lane);
            rms_norm_grouped(&s.lane, &lw.ffn_norm, &mut s.normed[..n_embd], 1, c.norm_eps);
            if std::env::var_os("RUST_XING4_DEBUG").is_some() && il < 2 {
                let st: f64 = s.lane.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let mx = s.lane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                eprintln!("[xing4] il={il} pos={pos} ffn_lane sq={st:.3} max={mx:.4}");
            }
            Self::prep(&s.normed[..n_embd], n_embd, &mut s.q8_buf, &mut s.q8_scales, &mut s.q8k_buf);

            let lane: Vec<f32> = if c.is_dense_ffn(il) {
                let ff = c.n_ff;
                lw.dense.w_gate.kernel.forward_prepared(
                    &s.normed[..n_embd], &s.q8_buf[..n_embd], &s.q8_scales[..n_embd / 32],
                    Some(&s.q8k_buf[..n_embd / 256]), &mut s.ffn_gate[..ff],
                    n_embd, ff, 0, 1,
                );
                lw.dense.w_up.kernel.forward_prepared(
                    &s.normed[..n_embd], &s.q8_buf[..n_embd], &s.q8_scales[..n_embd / 32],
                    Some(&s.q8k_buf[..n_embd / 256]), &mut s.ffn_up[..ff],
                    n_embd, ff, 0, 1,
                );
                crate::ops::silu_mul_approx_inplace(&s.ffn_gate[..ff], &mut s.ffn_up[..ff]);
                if std::env::var_os("RUST_XING4_DEBUG").is_some() && il == 0 {
                    let gsq: f64 = s.ffn_up[..ff].iter().map(|&v| (v as f64) * (v as f64)).sum();
                    let gmx = s.ffn_up[..ff].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let nsq: f64 = s.normed[..n_embd].iter().map(|&v| (v as f64) * (v as f64)).sum();
                    let nmx = s.normed[..n_embd].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    eprintln!("[xing4] il=0 normed sq={nsq:.4} max={nmx:.4} | h sq={gsq:.1} max={gmx:.4} | norm_w[0]={:.4} ffnnorm_w[0]={:.4}", lw.attn_norm[0], lw.ffn_norm[0]);
                }
                Self::prep(&s.ffn_up[..ff], ff, &mut s.q8_buf, &mut s.q8_scales, &mut s.q8k_buf);
                lw.dense.w_down.kernel.forward_prepared(
                    &s.ffn_up[..ff], &s.q8_buf[..ff], &s.q8_scales[..ff / 32],
                    Some(&s.q8k_buf[..ff / 256]), &mut s.moe_out[..n_embd],
                    ff, n_embd, 0, 1,
                );
                if std::env::var_os("RUST_XING4_DEBUG").is_some() && il == 0 {
                    let dsq: f64 = s.moe_out[..n_embd].iter().map(|&v| (v as f64) * (v as f64)).sum();
                    let dmx = s.moe_out[..n_embd].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    eprintln!("[xing4] il=0 down_out sq={dsq:.2} max={dmx:.4}");
                }
                s.moe_out[..n_embd].to_vec()
            } else {
                let moe = lw.moe.as_ref().expect("MoE layer missing weights");
                s.moe_out[..n_embd].fill(0.0);
                self.run_moe(moe, s);
                s.moe_out[..n_embd].to_vec()
            };
            self.spread_lane(&lane, &s.post, &s.comb, &s.residual, hc_state);
            if std::env::var_os("RUST_XING4_DEBUG").is_some() && il < 2 {
                let st: f64 = hc_state[..flat].iter().map(|&v| (v as f64) * (v as f64)).sum();
                let mx = hc_state[..flat].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                eprintln!("[xing4] il={il} pos={pos} after_ffn_post sq={st:.3} max={mx:.4}");
            }
        }
    }

    /// Routed experts plus the always-on shared expert, accumulated into
    /// `s.moe_out`.
    fn run_moe(&self, moe: &MoeFfn, s: &mut Xing4Scratch) {
        let c = &self.cfg;
        let n = c.n_embd;
        let n_exp = c.n_expert;
        let n_used = c.n_expert_used;
        let ff = c.n_ff_exp;

        // Router logits: F32 dot per expert (like LFM2-MoE).
        for e in 0..n_exp {
            let row = &moe.router[e * n..(e + 1) * n];
            let mut acc = 0.0f32;
            for d in 0..n {
                acc += row[d] * s.normed[d];
            }
            s.router_logits[e] = acc;
        }
        // Sigmoid gating; selection scores add the bias term.
        let probs = &mut s.router_probs[..n_exp];
        for (p, &l) in probs.iter_mut().zip(s.router_logits.iter()) {
            *p = sigmoid(l);
        }
        let mut sel: Vec<(usize, f32)> = probs
            .iter()
            .zip(moe.exp_probs_b.iter())
            .enumerate()
            .map(|(e, (&p, &b))| (e, p + b))
            .collect();
        sel.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sel.truncate(n_used);

        let mut w: Vec<f32> = sel.iter().map(|&(e, _)| probs[e]).collect();
        if c.expert_weights_norm {
            let sum: f32 = w.iter().sum();
            let sum = sum.max(6.103515625e-5);
            for v in w.iter_mut() {
                *v /= sum;
            }
        }
        // llama.cpp scales the (already normalised) weights by
        // `expert_weights_scale`, so the selected set can sum to more than
        // 1.0. Skipping this makes routed experts contribute half as much
        // as they should on models that set it to 2.0.
        if c.expert_weights_scale != 0.0 && c.expert_weights_scale != 1.0 {
            let scale = c.expert_weights_scale;
            for v in w.iter_mut() {
                *v *= scale;
            }
        }
        s.expert_sel.copy_from_slice(&sel.iter().map(|&(e, _)| e).collect::<Vec<_>>());
        s.expert_w.copy_from_slice(&w);
        s.routed_acc[..n].fill(0.0);

        // Routed experts, weighted-accumulated into `routed_acc`.
        for k in 0..n_used {
            let e = s.expert_sel[k];
            moe.gate_exps[e].kernel.forward_prepared(
                &s.normed[..n], &s.q8_buf[..n], &s.q8_scales[..n / 32],
                Some(&s.q8k_buf[..n / 256]), &mut s.ffn_gate[..ff], n, ff, 0, 1,
            );
            moe.up_exps[e].kernel.forward_prepared(
                &s.normed[..n], &s.q8_buf[..n], &s.q8_scales[..n / 32],
                Some(&s.q8k_buf[..n / 256]), &mut s.ffn_up[..ff], n, ff, 0, 1,
            );
            crate::ops::silu_mul_approx_inplace(&s.ffn_gate[..ff], &mut s.ffn_up[..ff]);
            Self::prep(&s.ffn_up[..ff], ff, &mut s.q8_buf, &mut s.q8_scales, &mut s.q8k_buf);
            moe.down_exps[e].kernel.forward_prepared(
                &s.ffn_up[..ff], &s.q8_buf[..ff], &s.q8_scales[..ff / 32],
                Some(&s.q8k_buf[..ff / 256]), &mut s.moe_out[..n],
                ff, n, 0, 1,
            );
            let wk = s.expert_w[k];
            for d in 0..n {
                s.routed_acc[d] += s.moe_out[d] * wk;
            }
        }

        // Shared expert, always on, added unweighted.
        if c.n_expert_shared > 0 {
            moe.shared_gate.kernel.forward_prepared(
                &s.normed[..n], &s.q8_buf[..n], &s.q8_scales[..n / 32],
                Some(&s.q8k_buf[..n / 256]), &mut s.ffn_gate[..ff], n, ff, 0, 1,
            );
            moe.shared_up.kernel.forward_prepared(
                &s.normed[..n], &s.q8_buf[..n], &s.q8_scales[..n / 32],
                Some(&s.q8k_buf[..n / 256]), &mut s.ffn_up[..ff], n, ff, 0, 1,
            );
            crate::ops::silu_mul_approx_inplace(&s.ffn_gate[..ff], &mut s.ffn_up[..ff]);
            Self::prep(&s.ffn_up[..ff], ff, &mut s.q8_buf, &mut s.q8_scales, &mut s.q8k_buf);
            moe.shared_down.kernel.forward_prepared(
                &s.ffn_up[..ff], &s.q8_buf[..ff], &s.q8_scales[..ff / 32],
                Some(&s.q8k_buf[..ff / 256]), &mut s.moe_out[..n],
                ff, n, 0, 1,
            );
            for d in 0..n {
                s.routed_acc[d] += s.moe_out[d];
            }
        }

        s.moe_out[..n].copy_from_slice(&s.routed_acc[..n]);
    }
}

fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let n = x.len();
    let mut sum = 0.0f64;
    for &v in &x[..n] {
        sum += (v as f64) * (v as f64);
    }
    let mean = (sum / n as f64) as f32;
    1.0 / (mean + eps).sqrt()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Sinkhorn normalisation of the `hc × hc` mixing matrix, matching
/// llama.cpp's DeepSeek-V4 sequence: a softmax over the destination
/// rows first, then alternating row/column normalisations.
fn sinkhorn(m: &mut [f32], hc: usize, eps: f32, iters: usize) {
    // Softmax over dim 0 (destination), i.e. each column normalised
    // across destinations.
    for col in 0..hc {
        let mut max = f32::NEG_INFINITY;
        for r in 0..hc {
            max = max.max(m[r * hc + col]);
        }
        let mut sum = 0.0f32;
        for r in 0..hc {
            m[r * hc + col] = (m[r * hc + col] - max).exp();
            sum += m[r * hc + col];
        }
        for r in 0..hc {
            m[r * hc + col] /= sum;
        }
    }
    for v in m.iter_mut() {
        *v += eps;
    }
    for it in 0..iters.max(1) {
        // column normalisation
        for col in 0..hc {
            let mut s = 0.0f32;
            for r in 0..hc {
                s += m[r * hc + col];
            }
            let denom = s + eps;
            for r in 0..hc {
                m[r * hc + col] /= denom;
            }
        }
        // row normalisation (llama.cpp skips the first round, matching its
        // `for (i = 1; i < iters; ++i)` loop after the initial column pass)
        if it == 0 {
            continue;
        }
        for r in 0..hc {
            let row: f32 = m[r * hc..(r + 1) * hc].iter().sum();
            let denom = row + eps;
            for v in m[r * hc..(r + 1) * hc].iter_mut() {
                *v /= denom;
            }
        }
    }
}
