//! Falcon-H1 forward pass.
//!
//! Parallel hybrid per layer (llama.cpp `src/models/falcon-h1.cpp`):
//!   cur  = attn(rms(inpL, attn_norm)) + mamba2(rms(inpL, attn_norm))
//!   inpL = cur + inpL
//!   cur  = swiglu_ffn(rms(inpL, ffn_norm))
//!   inpL = cur + inpL
//! The Mamba2 branch mirrors `build_mamba2_layer` (mamba-base.cpp) and
//! reuses the bit-verified `ssm_scan_row` kernel from the Nemotron trunk.

use half::f16;
use std::sync::Arc;

use super::config::FalconH1Config;
use super::weights::FalconH1LayerWeights;

use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{Kernel, Weight};
use crate::ops::{
    f32_slice_to_f16, quantize_q8_0_into, rope_neox_inplace, rope_norm, sample_llama_cpp,
    softmax_inplace,
};

/// Split the fused `ssm_in` projection into [z, xBC, dt]. Layout matches
/// llama.cpp mamba-base.cpp: z first (`head_dim x n_head`), then xBC
/// (`d_inner + 2*n_group*d_state`), then dt (`n_head`).
fn split_mamba2_projection(
    values: &[f32],
    inner: usize,
    groups: usize,
    state: usize,
    heads: usize,
) -> (&[f32], &[f32], &[f32]) {
    let conv_width = inner + 2 * groups * state;
    assert_eq!(values.len(), inner + conv_width + heads);
    let (z, rest) = values.split_at(inner);
    let (xbc, dt) = rest.split_at(conv_width);
    (z, xbc, dt)
}

/// Causal depthwise conv1d step over the per-layer history ring. Writes
/// into `output` (pre-sized to `current.len()`) so callers can reuse a
/// scratch buffer across layers/positions.
fn mamba2_conv_step(
    weights: &[f32],
    bias: &[f32],
    history: &mut [f32],
    current: &[f32],
    kernel: usize,
    output: &mut [f32],
) {
    let cols = current.len();
    assert_eq!(weights.len(), cols * kernel);
    assert_eq!(bias.len(), cols);
    assert_eq!(history.len(), cols * (kernel - 1));
    assert_eq!(output.len(), cols);
    for c in 0..cols {
        output[c] = 0.0;
        for k in 0..kernel - 1 {
            output[c] += weights[c * kernel + k] * history[k * cols + c];
        }
        output[c] += weights[c * kernel + kernel - 1] * current[c];
        output[c] += bias[c];
    }
    if kernel > 2 {
        history.copy_within(cols.., 0);
    }
    history[(kernel - 2) * cols..].copy_from_slice(current);
}

pub struct FalconH1Model {
    _source: Arc<dyn TensorSource>,
    pub config: FalconH1Config,
    pub layers: Vec<FalconH1LayerWeights<'static>>,
    pub tok_embd: Weight<'static>,
    pub output_norm: Vec<f32>,
    pub output: Weight<'static>,
    pub output_bytes: &'static [u8],
    pub pool: Arc<ComputePool>,
    /// Toggle for the falcon-local `q8_matmul_ggml` path that uses
    /// ggml's exact `vec_dot_q8_0_q8_0` reduction order. Off by default;
    /// enable via `FalconH1Model::with_ggml_kernel` or the `RMI_FALCON_GGML`
    /// env var during construction.
    pub ggml_kernel: bool,
}

impl FalconH1Model {
    pub fn from_source(source: Arc<dyn TensorSource>, n_threads: usize) -> Result<Self, String> {
        let config = FalconH1Config::from_source(source.as_ref())?;
        let layers = super::weights::load_layers(source.as_ref(), &config)?;
        let output_norm = crate::core::tensor::load_f32_tensor(
            source.as_ref(),
            "output_norm.weight",
            &[config.n_embd as u64],
        )?;
        let tok_embd = super::weights::load_weight_and_bytes(
            source.as_ref(),
            "token_embd.weight",
            config.n_embd,
            config.vocab_size,
        )?;
        let (tok_embd, tok_embd_bytes) = tok_embd;
        // llama.cpp falls back to the tied embedding when `output.weight`
        // is absent; every released falcon-h1 GGUF ships both, so require
        // the explicit tensor and fail loudly otherwise.
        let output = super::weights::load_weight_and_bytes(
            source.as_ref(),
            "output.weight",
            config.n_embd,
            config.vocab_size,
        )?;
        let (output, output_bytes) = output;
        // Falcon-local Q8_0×Q8_0 matmul (ggml reduction order). Off by
        // default because the shared kernel happens to suppress an EOS
        // bias this 1.5B checkpoint exhibits under greedy decoding; opt
        // in with `RMI_FALCON_GGML=1` (or any non-"0" value).
        let ggml_kernel = std::env::var_os("RMI_FALCON_GGML")
            .map(|v| v != *"0")
            .unwrap_or(false);
        let pool = Arc::new(ComputePool::new(n_threads.max(1)));
        Ok(Self {
            _source: source,
            config,
            layers,
            tok_embd,
            output_norm,
            output,
            output_bytes,
            pool,
            ggml_kernel,
        })
    }

    /// One Q8_0 (or F32/F16/BF16/QK) matmul with output rows partitioned
    /// across `self.pool`. Mirrors `NemotronModel::run_matmul`.
    #[inline]
    fn run_matmul(
        &self,
        kernel: &dyn Kernel,
        input: &[f32],
        q8: &[u8],
        sc: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) {
        debug_assert_eq!(input.len(), n_in);
        debug_assert_eq!(q8.len(), n_in);
        debug_assert_eq!(sc.len(), n_in.div_ceil(32));
        debug_assert_eq!(output.len(), n_out);

        let input_ptr = input.as_ptr();
        let q8_ptr = q8.as_ptr();
        let sc_ptr = sc.as_ptr();
        let out_ptr = output.as_mut_ptr();
        self.pool.compute(move |ith, nth| {
            // SAFETY: each thread writes to the disjoint
            // `[my_start, my_end)` row range computed inside the kernel.
            let my_in = unsafe { std::slice::from_raw_parts(input_ptr, n_in) };
            let my_q8 = unsafe { std::slice::from_raw_parts(q8_ptr, n_in) };
            let my_sc = unsafe { std::slice::from_raw_parts(sc_ptr, n_in.div_ceil(32)) };
            let my_out = unsafe { std::slice::from_raw_parts_mut(out_ptr, n_out) };
            kernel.forward_prepared(my_in, my_q8, my_sc, None, my_out, n_in, n_out, ith, nth);
        });
    }

    /// Falcon-local Q8_0 × Q8_0 matmul using ggml's exact reduction
    /// order. Caller is responsible for sizing `q8_buf`/`scale_buf` to
    /// `n_in` / `n_in.div_ceil(32)` respectively.
    fn run_matmul_ggml(
        &self,
        weight_bytes: &[u8],
        activation: &[f32],
        q8_buf: &mut [u8],
        scale_buf: &mut [f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) {
        q8_matmul_ggml(
            weight_bytes,
            activation,
            q8_buf,
            scale_buf,
            output,
            n_in,
            n_out,
            &self.pool,
        );
    }

    /// Evaluate the next tokens and return logits for the last position.
    pub fn prefill(
        &self,
        token_ids: &[u32],
        scratch: &mut FalconH1Scratch,
    ) -> Result<Vec<f32>, String> {
        let n = token_ids.len();
        if n == 0 {
            return Err("Falcon-H1 prefill: empty token sequence".into());
        }
        let base_position = scratch.next_position;
        if n > scratch.capacity.saturating_sub(base_position)
            || n > self.config.n_ctx.saturating_sub(base_position)
        {
            return Err(format!(
                "Falcon-H1 prefill: need {} positions but scratch has capacity {}",
                base_position + n,
                scratch.capacity
            ));
        }
        // 1) Embed input tokens.
        for (i, &tid) in token_ids.iter().enumerate() {
            let row_start = i * self.config.n_embd;
            self.tok_embd.embedding_lookup(
                tid,
                &mut scratch.hidden[row_start..row_start + self.config.n_embd],
            );
        }
        #[cfg(feature = "parity-trace")]
        parity_log(
            "embd",
            &scratch.hidden[..n * self.config.n_embd],
        );
        // 2) Per-layer forward (layers outer, positions inner — the SSM
        // state and conv history must advance sequentially per layer, and
        // attention reads the progressively-filled KV cache).
        for (layer_idx, lw) in self.layers.iter().enumerate() {
            self.forward_layer(layer_idx, lw, n, base_position, scratch)?;
        }
        // 3) Final norm + logits.
        let last_pos = n - 1;
        let off = last_pos * self.config.n_embd;
        rms_norm_ggml(
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
        if self.ggml_kernel {
            self.run_matmul_ggml(
                self.output_bytes,
                &scratch.normed,
                &mut scratch.q8_buf[..self.config.n_embd],
                &mut scratch.scale_buf[..blocks],
                &mut scratch.logits,
                self.config.n_embd,
                self.config.vocab_size,
            );
        } else {
            self.run_matmul(
                &*self.output.kernel,
                &scratch.normed,
                &scratch.q8_buf[..self.config.n_embd],
                &scratch.scale_buf[..blocks],
                &mut scratch.logits,
                self.config.n_embd,
                self.config.vocab_size,
            );
        }
        scratch.next_position += n;
        #[cfg(feature = "parity-trace")]
        {
            parity_log("result_norm", &scratch.normed);
            parity_log("result_output", &scratch.logits);
        }
        Ok(scratch.logits.clone())
    }

    fn forward_layer(
        &self,
        layer_idx: usize,
        lw: &FalconH1LayerWeights,
        length: usize,
        base_position: usize,
        scratch: &mut FalconH1Scratch,
    ) -> Result<(), String> {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let n_head = cfg.n_head;
        let n_head_kv = cfg.n_head_kv;
        let head_dim_k = cfg.n_embd_head_k;
        let head_dim_v = cfg.n_embd_head_v;
        let group_size = n_head / n_head_kv;
        let n_attn_q = n_head * head_dim_k;
        let n_attn_kv = n_head_kv * head_dim_k;
        let n_attn_v = n_head * head_dim_v;
        let n_attn_v_kv = n_head_kv * head_dim_v;
        // Optional per-stage parity probe (build with `--features parity-trace`,
        // see `.agents/skills/adapting-new-models`). The macro and buffers
        // stay compiled in but expand to nothing without the feature so
        // there is no hot-path overhead.
        let parity = cfg!(feature = "parity-trace")
            && std::env::var_os("RMI_FALCON_PARITY").is_some();
        let mut acc: Vec<(&'static str, Vec<f32>)> = Vec::new();
        macro_rules! par_push {
            ($tag:expr, $buf:expr) => {
                if parity {
                    if let Some(slot) = acc.iter_mut().find(|(t, _)| *t == $tag) {
                        slot.1.extend_from_slice($buf);
                    } else {
                        acc.push(($tag, $buf.to_vec()));
                    }
                }
            };
        }
        let kq_scale = 1.0 / (head_dim_k as f32).sqrt();

        // mm_q8! dispatches Q8_0 matmul to either the shared kernel or
        // the falcon-local `q8_matmul_ggml` path (ggml reduction order).
        macro_rules! mm_q8 {
            ($weight:expr, $kernel:expr, $act:expr, $out:expr, $n_in:expr, $n_out:expr) => {{
                if self.ggml_kernel {
                    let blocks = ($n_in + 31) / 32;
                    quantize_q8_0_into(
                        $act,
                        $n_in,
                        &mut scratch.q8_buf[..$n_in],
                        &mut scratch.scale_buf[..blocks],
                    );
                    self.run_matmul_ggml(
                        $weight,
                        $act,
                        &mut scratch.q8_buf[..$n_in],
                        &mut scratch.scale_buf[..blocks],
                        $out,
                        $n_in,
                        $n_out,
                    );
                } else {
                    let blocks = ($n_in + 31) / 32;
                    quantize_q8_0_into(
                        $act,
                        $n_in,
                        &mut scratch.q8_buf[..$n_in],
                        &mut scratch.scale_buf[..blocks],
                    );
                    self.run_matmul(
                        &*$kernel,
                        $act,
                        &scratch.q8_buf[..$n_in],
                        &scratch.scale_buf[..blocks],
                        $out,
                        $n_in,
                        $n_out,
                    );
                }
            }};
        }

        for t in 0..length {
            let position = base_position + t;
            let off = t * n_embd;
            let row = &mut scratch.hidden[off..off + n_embd];
            // The same normed input feeds BOTH branches (llama.cpp builds
            // `build_norm(inpL, attn_norm)` twice — bit-identical result,
            // so one rms_norm reuse is exact). `quantize_into` re-quantizes
            // the shared scratch buffer at each use site so the raw f32
            // input always drives the next branch.
            fn quantize_into(normed: &[f32], q8_buf: &mut [u8], scale_buf: &mut [f32], n_in: usize) {
                let blocks = (n_in + 31) / 32;
                quantize_q8_0_into(normed, n_in, &mut q8_buf[..n_in], &mut scale_buf[..blocks]);
            }
            rms_norm_ggml(row, &lw.attn_norm, &mut scratch.normed, cfg.norm_eps);
            par_push!("attn_norm", &scratch.normed);

            // ---- Attention branch (GQA + NORM-type RoPE over the full
            // head, freq_base from metadata; no QK norm, no biases on the
            // 1.5B checkpoint) ----
            let (q, k, v) = (
                &mut scratch.q_buf[..n_attn_q],
                &mut scratch.k_buf[..n_attn_kv],
                &mut scratch.v_buf[..n_attn_v],
            );
            quantize_into(&scratch.normed, &mut scratch.q8_buf, &mut scratch.scale_buf, n_embd);
            let q8 = &scratch.q8_buf[..n_embd];
            let sc = &scratch.scale_buf[..(n_embd + 31) / 32];
            self.run_matmul(&*lw.wq.kernel, &scratch.normed, q8, sc, q, n_embd, n_attn_q);
            self.run_matmul(&*lw.wk.kernel, &scratch.normed, q8, sc, k, n_embd, n_attn_kv);
            self.run_matmul(&*lw.wv.kernel, &scratch.normed, q8, sc, v, n_embd, n_attn_v);
            par_push!("Qcur", q);
            par_push!("Vcur", v);
            par_push!("Kcur", k);
            if parity {
                #[cfg(feature = "parity-trace")]
                eprintln!(
                    "V t{} v[0..3]={:?} v[-3..]={:?}",
                    t,
                    &v[..3],
                    &v[v.len() - 3..]
                );
            }
            for h in 0..n_head {
                let o = h * head_dim_k;
                rope_norm(&mut q[o..o + head_dim_k], position, head_dim_k, cfg.rope_freq_base);
            }
            for h in 0..n_head_kv {
                let o = h * head_dim_k;
                rope_norm(&mut k[o..o + head_dim_k], position, head_dim_k, cfg.rope_freq_base);
            }
            par_push!("Qcur-post-rope", q);
            par_push!("Kcur-post-rope", k);
            #[cfg(feature = "parity-trace")]
            if parity {
                eprintln!(
                    "V q[0..6]={:?} q[-6..]={:?}",
                    &q[..6],
                    &q[head_dim_k - 6..head_dim_k]
                );
            }
            let k_base = layer_idx * scratch.capacity * n_attn_kv;
            let v_base = layer_idx * scratch.capacity * n_attn_v;
            scratch.k[k_base + position * n_attn_kv..k_base + (position + 1) * n_attn_kv]
                .copy_from_slice(k);
            // Cache V as f16-rounded (matches ggml cache_v storage).
            for (cached, &value) in scratch.v
                [v_base + position * n_attn_v..v_base + (position + 1) * n_attn_v]
                .iter_mut()
                .zip(v.iter())
            {
                *cached = f16::from_f32(value).to_f32();
            }
            let attn_out = &mut scratch.attn_out[..];
            f32_slice_to_f16(q, &mut scratch.q_f16[..n_attn_q]);
            // Flat K / V f16 caches, sliced per position.
            let k_flat = &mut scratch.k_f16_storage[..(position + 1) * n_attn_kv];
            let v_flat = &mut scratch.v_f16_storage[..(position + 1) * n_attn_v];
            for j in 0..=position {
                f32_slice_to_f16(
                    &scratch.k[k_base + j * n_attn_kv..k_base + (j + 1) * n_attn_kv],
                    &mut k_flat[j * n_attn_kv..(j + 1) * n_attn_kv],
                );
                f32_slice_to_f16(
                    &scratch.v[v_base + j * n_attn_v..v_base + (j + 1) * n_attn_v],
                    &mut v_flat[j * n_attn_v..(j + 1) * n_attn_v],
                );
            }
            let q_f16 = &scratch.q_f16[..n_attn_q];
            let scores = &mut scratch.scores[..position + 1];
            let probs_f16 = &mut scratch.probs_f16[..position + 1];
            let v_col = &mut scratch.v_col[..position + 1];
            for h in 0..n_head {
                let kv_h = h / group_size;
                for j in 0..=position {
                    let k_row = &k_flat[j * n_attn_kv + kv_h * head_dim_k
                        ..j * n_attn_kv + (kv_h + 1) * head_dim_k];
                    let q_row = &q_f16[h * head_dim_k..(h + 1) * head_dim_k];
                    scores[j] = dot_f16_ggml(q_row, k_row, head_dim_k) * kq_scale;
                }
                if parity {
                    acc.extend(vec![("kq", scores.to_vec())]);
                }
                softmax_inplace(scores);
                f32_slice_to_f16(scores, probs_f16);
                // KQV as per-output-dim dots over the position axis, in
                // ggml's mul_mat(vec_dot_f16) order. Build the per-d column
                // once into the scratch `v_col` buffer.
                let head_out = &mut attn_out[h * head_dim_v..(h + 1) * head_dim_v];
                for d in 0..head_dim_v {
                    let col_off = kv_h * head_dim_v + d;
                    for j in 0..=position {
                        v_col[j] = v_flat[j * n_attn_v + col_off];
                    }
                    head_out[d] = dot_f16_ggml(probs_f16, v_col, position + 1);
                }
            }
            let attn_proj = &mut scratch.attn_proj[..];
            par_push!("kqv_out-0", attn_out);
            let blocks2 = (n_attn_v + 31) / 32;
            quantize_q8_0_into(
                attn_out,
                n_attn_v,
                &mut scratch.q8_buf[..n_attn_v],
                &mut scratch.scale_buf[..blocks2],
            );
            let wo_q8 = &scratch.q8_buf[..n_attn_v];
            let wo_sc = &scratch.scale_buf[..blocks2];
            self.run_matmul(
                &*lw.wo.kernel,
                attn_out,
                wo_q8,
                wo_sc,
                attn_proj,
                n_attn_v,
                n_embd,
            );
            par_push!("attn_out-0", attn_proj);
            for d in 0..n_embd {
                row[d] += attn_proj[d];
            }

            // ---- Mamba2 branch on the SAME normed input ----
            let inner_size = cfg.ssm_inner_size;
            let n_group = cfg.ssm_group_count;
            let d_state = cfg.ssm_state_size;
            let n_ssm_head = cfg.ssm_n_head();
            let headdim = cfg.ssm_headdim();
            let heads_per_group = n_ssm_head / n_group;
            let conv_cols = cfg.ssm_conv_cols();
            let d_in_proj = cfg.ssm_in_proj_dim();
            let ssm_in_out = &mut scratch.ssm_in_out[..d_in_proj];
            quantize_into(&scratch.normed, &mut scratch.q8_buf, &mut scratch.scale_buf, n_embd);
            let ssm_q8 = &scratch.q8_buf[..n_embd];
            let ssm_sc = &scratch.scale_buf[..(n_embd + 31) / 32];
            self.run_matmul(
                &*lw.ssm_in.kernel,
                &scratch.normed,
                ssm_q8,
                ssm_sc,
                ssm_in_out,
                n_embd,
                d_in_proj,
            );
            par_push!("node_40", ssm_in_out);
            let (z_slice, cur_input, dt_input) =
                split_mamba2_projection(ssm_in_out, inner_size, n_group, d_state, n_ssm_head);
            let hist_len = (cfg.ssm_conv_kernel - 1) * conv_cols;
            let hist_base = layer_idx * hist_len;
            let conv_out = &mut scratch.conv_out[..conv_cols];
            mamba2_conv_step(
                &lw.ssm_conv1d_w,
                &lw.ssm_conv1d_b,
                &mut scratch.ssm_conv_hist[hist_base..hist_base + hist_len],
                cur_input,
                cfg.ssm_conv_kernel,
                conv_out,
            );
            par_push!("node_52", conv_out);
            for value in conv_out.iter_mut() {
                *value = crate::ops::silu(*value);
            }
            let x_pre = &conv_out[..inner_size];
            // b/c are slices into the live `conv_out`; the scan only reads
            // them while conv_out is immutable there.
            let b = &conv_out[inner_size..inner_size + n_group * d_state];
            let c = &conv_out[inner_size + n_group * d_state..];
            let dt_raw = &mut scratch.dt_raw[..n_ssm_head];
            let dt_per_head = &mut scratch.dt_per_head[..n_ssm_head];
            for h in 0..n_ssm_head {
                let z = dt_input[h] + lw.ssm_dt_bias[h];
                dt_raw[h] = z;
                dt_per_head[h] = if z > 20.0 { z } else { (1.0 + z.exp()).ln() };
            }
            par_push!("node_53", dt_raw);
            let da_per_head = &mut scratch.da_per_head[..n_ssm_head];
            for h in 0..n_ssm_head {
                da_per_head[h] = (dt_per_head[h] * lw.ssm_a[h]).exp();
            }
            let state_stride_layer = n_ssm_head * headdim * d_state;
            let state_base = layer_idx * state_stride_layer;
            let state_end = state_base + state_stride_layer;
            let state = &mut scratch.ssm_scan_state[state_base..state_end];
            let y_buf = &mut scratch.y_buf[..inner_size];
            for h in 0..n_ssm_head {
                let g = h / heads_per_group;
                let g_b_off = g * d_state;
                let g_c_off = g * d_state;
                let head_x_off = h * headdim;
                let head_state_off = h * headdim * d_state;
                let da = da_per_head[h];
                let dt_h = dt_per_head[h];
                let d_h = lw.ssm_d[h];
                for k in 0..headdim {
                    let x_dt = x_pre[head_x_off + k] * dt_h;
                    let state_row = &mut state
                        [head_state_off + k * d_state..head_state_off + (k + 1) * d_state];
                    let sumf = ssm_scan_row_ggml(
                        state_row,
                        &b[g_b_off..g_b_off + d_state],
                        &c[g_c_off..g_c_off + d_state],
                        da,
                        x_dt,
                    );
                    y_buf[head_x_off + k] = sumf + d_h * x_pre[head_x_off + k];
                }
            }
            // z-gate AFTER the scan + D-skip: y = silu(z) * y.
            par_push!("mamba2_y_add_d", y_buf);
            for j in 0..inner_size {
                y_buf[j] = crate::ops::silu(z_slice[j]) * y_buf[j];
            }
            par_push!("node_74", y_buf);
            // Group RMSNorm (n_group == 1 for the 1.5B: one group over the
            // full d_inner), ggml-exact rms semantics. The scratch `normed_ssm`
            // buffer aliases `y_buf` element-wise so we still need a copy.
            let per_group = inner_size / n_group;
            let normed_ssm = &mut scratch.normed_ssm[..inner_size];
            for g in 0..n_group {
                let group_start = g * per_group;
                rms_norm_ggml(
                    &y_buf[group_start..group_start + per_group],
                    &lw.ssm_norm[group_start..group_start + per_group],
                    &mut normed_ssm[group_start..group_start + per_group],
                    cfg.norm_eps,
                );
            }
            y_buf.copy_from_slice(normed_ssm);
            par_push!("ssm_norm_w", y_buf);
            let blocks_out = (inner_size + 31) / 32;
            quantize_q8_0_into(
                y_buf,
                inner_size,
                &mut scratch.q8_buf[..inner_size],
                &mut scratch.scale_buf[..blocks_out],
            );
            let ssm_proj = &mut scratch.ssm_proj[..];
            mm_q8!(lw.ssm_out_bytes, lw.ssm_out.kernel, y_buf, ssm_proj, inner_size, n_embd);
            par_push!("ssm_out-0", ssm_proj);
            for d in 0..n_embd {
                row[d] += ssm_proj[d];
            }
            if parity {
                let mut cur = vec![0.0f32; n_embd];
                for d in 0..n_embd {
                    cur[d] = attn_proj[d] + ssm_proj[d];
                }
                par_push!("layer_out-0", &cur);
                par_push!("ffn_inp-0", row);
            }

            // ---- FFN branch (SwiGLU): inpL is now inpSA. ----
            rms_norm_ggml(row, &lw.ffn_norm, &mut scratch.normed, cfg.norm_eps);
            par_push!("ffn_norm-0", &scratch.normed);
            quantize_into(&scratch.normed, &mut scratch.q8_buf, &mut scratch.scale_buf, n_embd);
            let q8f = &scratch.q8_buf[..n_embd];
            let scf = &scratch.scale_buf[..(n_embd + 31) / 32];
            let (gate_buf, up_buf) = (
                &mut scratch.gate_buf[..cfg.n_ff],
                &mut scratch.up_buf[..cfg.n_ff],
            );
            self.run_matmul(
                &*lw.w_gate.kernel,
                &scratch.normed,
                q8f,
                scf,
                gate_buf,
                n_embd,
                cfg.n_ff,
            );
            self.run_matmul(
                &*lw.w_up.kernel,
                &scratch.normed,
                q8f,
                scf,
                up_buf,
                n_embd,
                cfg.n_ff,
            );
            par_push!("ffn_gate-0", gate_buf);
            par_push!("ffn_up-0", up_buf);
            // cur = silu(gate) * up
            for j in 0..cfg.n_ff {
                up_buf[j] = crate::ops::silu(gate_buf[j]) * up_buf[j];
            }
            par_push!("ffn_swiglu-0", up_buf);
            let ffn_out = &mut scratch.ffn_out[..];
            let blocks4 = (cfg.n_ff + 31) / 32;
            quantize_q8_0_into(
                up_buf,
                cfg.n_ff,
                &mut scratch.q8_buf[..cfg.n_ff],
                &mut scratch.scale_buf[..blocks4],
            );
            self.run_matmul(
                &*lw.w_down.kernel,
                up_buf,
                &scratch.q8_buf[..cfg.n_ff],
                &scratch.scale_buf[..blocks4],
                ffn_out,
                cfg.n_ff,
                n_embd,
            );
            par_push!("ffn_out-0", ffn_out);
            for d in 0..n_embd {
                row[d] += ffn_out[d];
            }
            par_push!("l_out-0", row);
        }
        if parity {
            for (tag, values) in &acc {
                parity_log(&format!("L{layer_idx} {tag}"), values);
            }
        }
        Ok(())
    }
}

/// ggml-exact `hsum_float_8` (the AVX2 horizontal sum used by
/// `vec_dot_q8_0_q8_0`). Differs from the shared `hsum_ps` in dot.rs
/// only by the reduction order — local to Falcon so we don't perturb
/// other trunks' pinned parity hashes.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum_float_8(x: std::arch::x86_64::__m256) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let mut res = _mm256_extractf128_ps(x, 1);
        res = _mm_add_ps(res, _mm256_castps256_ps128(x));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        _mm_cvtss_f32(res)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = x;
        unreachable!()
    }
}

/// Single-column Q8_0 × Q8_0 dot product: ggml's exact
/// `vec_dot_q8_0_q8_0` AVX2 path. Returns the f32 dot of a single
/// activation column (already quantised into `q8` + `scales`) against
/// one weight row. `weight` holds the raw GGUF Q8_0 blocks for that row:
/// `weight = [f16 scale | 32 int8 | f16 scale | 32 int8 | ...]`.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn vec_dot_q8_0_q8_0_row_ggml(
    weight: &[u8],
    q8: &[u8],
    scales: &[f32],
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let blocks_per_row = q8.len() / 32;
        let row_stride = blocks_per_row * 34;
        debug_assert_eq!(weight.len(), row_stride);
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_ps();
        for ib in 0..blocks_per_row {
            let qx = _mm256_loadu_si256(weight.as_ptr().add(ib * 34 + 2) as *const __m256i);
            let qy = _mm256_loadu_si256(q8.as_ptr().add(ib * 32) as *const __m256i);
            let w_scale_bits =
                std::ptr::read_unaligned(weight.as_ptr().add(ib * 34) as *const u16);
            let w_scale = f16::to_f32(f16::from_bits(w_scale_bits));
            let d = w_scale * scales[ib];
            let d_v = _mm256_set1_ps(d);
            let ax = _mm256_sign_epi8(qx, qx);
            let sy = _mm256_sign_epi8(qy, qx);
            let pair_i16 = _mm256_maddubs_epi16(ax, sy);
            let summed_i32 = _mm256_madd_epi16(ones, pair_i16);
            acc = _mm256_fmadd_ps(d_v, _mm256_cvtepi32_ps(summed_i32), acc);
        }
        hsum_float_8(acc)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (weight, q8, scales);
        unreachable!()
    }
}

/// Scalar fallback for non-AVX2 targets.
#[cfg(not(target_arch = "x86_64"))]
unsafe fn vec_dot_q8_0_q8_0_row_ggml(
    weight: &[u8],
    q8: &[u8],
    scales: &[f32],
) -> f32 {
    let blocks_per_row = q8.len() / 32;
    let mut acc = 0.0f32;
    for ib in 0..blocks_per_row {
        let w_scale_bits =
            std::ptr::read_unaligned(weight.as_ptr().add(ib * 34) as *const u16);
        let w_scale = f16::to_f32(f16::from_bits(w_scale_bits));
        let d = w_scale * scales[ib];
        let mut sum_i32 = 0i32;
        for j in 0..32 {
            let qi = q8[ib * 32 + j] as i8 as i32;
            let wi = weight[ib * 34 + 2 + j] as i8 as i32;
            sum_i32 += qi * wi;
        }
        acc += d * (sum_i32 as f32);
    }
    acc
}

/// Falcon-local Q8_0 × Q8_0 matmul (single activation column). Mirrors
/// `FalconH1Model::run_matmul` but uses ggml's exact `vec_dot_q8_0_q8_0`
/// reduction order. `weight_bytes` is the raw GGUF Q8_0 tensor (blocks
/// of 34 bytes per output row); `activation` is a single f32 column
/// (length `n_in`); the activation is quantised to Q8_0 once into the
/// provided scratch buffers, then every output row is computed against
/// the same shared quantised activation. Rows are partitioned across
/// the pool's worker threads in the same way as `run_matmul`.
pub(crate) fn q8_matmul_ggml(
    weight_bytes: &[u8],
    activation: &[f32],
    q8_buf: &mut [u8],
    scale_buf: &mut [f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    pool: &crate::core::thread_pool::ComputePool,
) {
    debug_assert_eq!(activation.len(), n_in);
    let blocks_per_row = n_in / 32;
    let row_stride = blocks_per_row * 34;
    debug_assert_eq!(weight_bytes.len(), n_out * row_stride);
    debug_assert!(q8_buf.len() >= n_in);
    let blocks = n_in.div_ceil(32);
    debug_assert!(scale_buf.len() >= blocks);
    crate::ops::quantize_q8_0_into(activation, n_in, &mut q8_buf[..n_in], &mut scale_buf[..blocks]);
    let q8 = &q8_buf[..n_in];
    let scales = &scale_buf[..blocks];
    let wb_ptr = weight_bytes.as_ptr();
    let q8_ptr = q8.as_ptr();
    let sc_ptr = scales.as_ptr();
    let out_ptr = output.as_mut_ptr();
    pool.compute(move |ith, nth| {
        let my_q8 = unsafe { std::slice::from_raw_parts(q8_ptr, n_in) };
        let my_sc = unsafe { std::slice::from_raw_parts(sc_ptr, blocks) };
        let row_start = ith * n_out / nth;
        let row_end = (ith + 1) * n_out / nth;
        for row_idx in row_start..row_end {
            let weight = unsafe {
                std::slice::from_raw_parts(
                    wb_ptr.add(row_idx * row_stride),
                    row_stride,
                )
            };
            let out = unsafe { std::slice::from_raw_parts_mut(out_ptr.add(row_idx), 1) };
            let dot = unsafe { vec_dot_q8_0_q8_0_row_ggml(weight, my_q8, my_sc) };
            out[0] = dot;
        }
    });
}

/// F16 x F16 dot with ggml's `vec_dot_f16` AVX2 reduction order:
/// four FMA accumulators over 32-wide blocks, `(a0+a2)+(a1+a3)` reduce,
/// then a sequential scalar tail.
pub(crate) fn dot_f16_ggml(a: &[u16], b: &[u16], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::ops::has_avx2_fma() && crate::ops::has_f16c() && n >= 32 {
            unsafe { return dot_f16_ggml_avx2(a, b, n) };
        }
    }
    let mut sum = 0.0f64;
    for i in 0..n {
        sum += f64::from(f16::to_f32(f16::from_bits(a[i])) * f16::to_f32(f16::from_bits(b[i])));
    }
    sum as f32
}

#[cfg(target_arch = "x86_64")]
#[cfg_attr(
    target_arch = "x86_64",
    target_feature(enable = "avx2", enable = "fma", enable = "f16c")
)]
#[inline]
unsafe fn dot_f16_ggml_avx2(a: &[u16], b: &[u16], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 32 <= n {
            for (offset, acc) in [
                (0usize, &mut acc0),
                (8, &mut acc1),
                (16, &mut acc2),
                (24, &mut acc3),
            ] {
                let va = _mm256_cvtph_ps(_mm_loadu_si128(a.as_ptr().add(i + offset) as *const __m128i));
                let vb = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i + offset) as *const __m128i));
                *acc = _mm256_fmadd_ps(va, vb, *acc);
            }
            i += 32;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc2), _mm256_add_ps(acc1, acc3));
        let mut sum = f64::from(crate::ops::dot::hsum_ps(acc));
        while i < n {
            sum += f64::from(f16::to_f32(f16::from_bits(a[i])) * f16::to_f32(f16::from_bits(b[i])));
            i += 1;
        }
        sum as f32
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (a, b, n);
        unreachable!()
    }
}

/// One row of the Mamba2 selective-state-space scan, matching
/// `ggml_compute_forward_ssm_scan_f32` (Mamba-2 branch) op-for-op:
///   t0 = s * dA (rounded); t1 = B * x_dt (rounded); s' = t0 + t1;
///   sum = fma(s', C, sum) (fused); scalar tail identical.
/// The shared Nemotron kernel fuses `B * x_dt + s * dA` into one FMA,
/// which differs from ggml's per-product rounding.
#[inline]
pub(crate) fn ssm_scan_row_ggml(
    state_row: &mut [f32],
    b_row: &[f32],
    c_row: &[f32],
    d_a: f32,
    x_dt: f32,
) -> f32 {
    debug_assert_eq!(state_row.len(), b_row.len());
    debug_assert_eq!(state_row.len(), c_row.len());
    #[cfg(target_arch = "x86_64")]
    {
        if crate::ops::has_avx2_fma() {
            unsafe {
                return ssm_scan_row_ggml_avx2(state_row, b_row, c_row, d_a, x_dt);
            }
        }
    }
    let mut sumf = 0.0f32;
    for i in 0..state_row.len() {
        let state = (state_row[i] * d_a) + (b_row[i] * x_dt);
        sumf += state * c_row[i];
        state_row[i] = state;
    }
    sumf
}

#[cfg(target_arch = "x86_64")]
#[cfg_attr(
    target_arch = "x86_64",
    target_feature(enable = "avx2", enable = "fma")
)]
#[inline]
unsafe fn ssm_scan_row_ggml_avx2(
    state_row: &mut [f32],
    b_row: &[f32],
    c_row: &[f32],
    d_a: f32,
    x_dt: f32,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let n = state_row.len();
        // ggml GGML_F32_STEP = 32 (4 x 8-lane accumulators).
        const STEP: usize = 32;
        const EPR: usize = 8;
        const ARR: usize = 4;
        let np = n / STEP * STEP;
        let v_da = _mm256_set1_ps(d_a);
        let v_x_dt = _mm256_set1_ps(x_dt);
        let mut acc = [_mm256_setzero_ps(); ARR];

        let mut i = 0;
        while i < np {
            for j in 0..ARR {
                let off = i + j * EPR;
                let v_t0 = _mm256_mul_ps(_mm256_loadu_ps(state_row.as_ptr().add(off)), v_da);
                let v_t1 = _mm256_mul_ps(_mm256_loadu_ps(b_row.as_ptr().add(off)), v_x_dt);
                let v_new = _mm256_add_ps(v_t0, v_t1);
                _mm256_storeu_ps(state_row.as_mut_ptr().add(off), v_new);
                let v_c = _mm256_loadu_ps(c_row.as_ptr().add(off));
                acc[j] = _mm256_fmadd_ps(v_new, v_c, acc[j]);
            }
            i += STEP;
        }
        // GGML_F32x8_REDUCE: pairwise tree over the 4 accumulators, then
        // low+high 128-bit add and two hadds.
        acc[0] = _mm256_add_ps(acc[0], acc[2]);
        acc[1] = _mm256_add_ps(acc[1], acc[3]);
        acc[0] = _mm256_add_ps(acc[0], acc[1]);
        let t0 = _mm_add_ps(_mm256_castps256_ps128(acc[0]), _mm256_extractf128_ps(acc[0], 1));
        let t1 = _mm_hadd_ps(t0, t0);
        let mut sumf = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
        while i < n {
            let state = (state_row[i] * d_a) + (b_row[i] * x_dt);
            sumf += state * c_row[i];
            state_row[i] = state;
            i += 1;
        }
        sumf
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (state_row, b_row, c_row, d_a, x_dt);
        unreachable!()
    }
}

pub struct FalconH1Session {
    pub model: FalconH1Model,
    pub scratch: FalconH1Scratch,
    pub next_position: usize,
    pub capacity: usize,
}

pub struct FalconH1Scratch {
    pub next_position: usize,
    pub capacity: usize,
    pub hidden: Vec<f32>,
    pub normed: Vec<f32>,
    /// Per-layer attention KV cache, laid out as
    /// `[layer * capacity * n_attn_kv + position * n_attn_kv + i]`.
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub q8_buf: Vec<u8>,
    pub scale_buf: Vec<f32>,
    /// Per-layer Mamba2 conv1d history, laid out as
    /// `[layer * (kernel-1) * conv_cols + k * conv_cols + c]`.
    pub ssm_conv_hist: Vec<f32>,
    /// Per-layer Mamba2 scan state `[layer, head, channel, state]`.
    pub ssm_scan_state: Vec<f32>,
    pub logits: Vec<f32>,
    // --- Reusable per-(layer, position) working buffers. Sized for the
    // largest variant encountered (n_attn_q, n_attn_kv, n_ff, ssm_*).
    pub q_buf: Vec<f32>,
    pub k_buf: Vec<f32>,
    pub v_buf: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_proj: Vec<f32>,
    pub ssm_proj: Vec<f32>,
    pub ssm_in_out: Vec<f32>,
    pub conv_out: Vec<f32>,
    pub dt_raw: Vec<f32>,
    pub dt_per_head: Vec<f32>,
    pub da_per_head: Vec<f32>,
    pub y_buf: Vec<f32>,
    pub normed_ssm: Vec<f32>,
    pub gate_buf: Vec<f32>,
    pub up_buf: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub scores: Vec<f32>,
    /// `q_f16`, `probs_f16`, `v_col`, `k_f16_storage`, `v_f16_storage` all
    /// sized to the maximum `position + 1` (i.e. `capacity`) the scratch
    /// will ever see.
    pub q_f16: Vec<u16>,
    pub probs_f16: Vec<u16>,
    pub v_col: Vec<u16>,
    pub k_f16_storage: Vec<u16>,
    pub v_f16_storage: Vec<u16>,
}

impl FalconH1Scratch {
    pub fn new(config: &FalconH1Config, capacity: usize) -> Self {
        let n_embd = config.n_embd;
        let n_attn_q = config.n_head * config.n_embd_head_k;
        let n_attn_kv = config.n_head_kv * config.n_embd_head_k;
        let n_attn_v = config.n_head * config.n_embd_head_v;
        let conv_cols = config.ssm_conv_cols();
        let n_ff = config.n_ff;
        let n_ssm_head = config.ssm_n_head();
        Self {
            next_position: 0,
            capacity,
            hidden: vec![0.0; capacity * n_embd],
            normed: vec![0.0; n_embd],
            k: vec![0.0; config.n_layer * capacity * n_attn_kv],
            v: vec![0.0; config.n_layer * capacity * n_attn_v],
            q8_buf: vec![0u8; n_embd.max(config.ssm_inner_size).max(n_ff)],
            scale_buf: vec![0.0; n_embd.max(config.ssm_inner_size).max(n_ff).div_ceil(32)],
            ssm_conv_hist: vec![
                0.0;
                config.n_layer
                    * (config.ssm_conv_kernel - 1)
                    * conv_cols
            ],
            // Per layer: n_head * headdim * d_state = d_inner * d_state.
            ssm_scan_state: vec![
                0.0;
                config.n_layer * config.ssm_inner_size * config.ssm_state_size
            ],
            logits: vec![0.0; config.vocab_size],
            q_buf: vec![0.0; n_attn_q],
            k_buf: vec![0.0; n_attn_kv],
            v_buf: vec![0.0; n_attn_v],
            attn_out: vec![0.0; n_attn_v],
            attn_proj: vec![0.0; n_embd],
            ssm_proj: vec![0.0; n_embd],
            ssm_in_out: vec![0.0; config.ssm_in_proj_dim()],
            conv_out: vec![0.0; conv_cols],
            dt_raw: vec![0.0; n_ssm_head],
            dt_per_head: vec![0.0; n_ssm_head],
            da_per_head: vec![0.0; n_ssm_head],
            y_buf: vec![0.0; config.ssm_inner_size],
            normed_ssm: vec![0.0; config.ssm_inner_size],
            gate_buf: vec![0.0; n_ff],
            up_buf: vec![0.0; n_ff],
            ffn_out: vec![0.0; n_embd],
            scores: vec![0.0; capacity],
            q_f16: vec![0u16; n_attn_q],
            probs_f16: vec![0u16; capacity],
            v_col: vec![0u16; capacity],
            // Each (kv slot) holds n_attn_kv (K) or n_attn_v (V) f16 rows.
            k_f16_storage: vec![0u16; capacity * n_attn_kv],
            v_f16_storage: vec![0u16; capacity * n_attn_v],
        }
    }

    pub fn reset_ssm_state(&mut self) {
        for v in &mut self.ssm_conv_hist {
            *v = 0.0;
        }
        for v in &mut self.ssm_scan_state {
            *v = 0.0;
        }
    }

    pub fn reset(&mut self) {
        self.next_position = 0;
        self.k.fill(0.0);
        self.v.fill(0.0);
        self.reset_ssm_state();
    }
}

pub fn run_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    _kv_format: crate::app::cli::KvFormat,
    repetition_penalty: f32,
    chat_template: Option<&str>,
) -> Result<(), String> {
    use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
    use std::collections::HashMap;

    eprintln!("Loading Falcon-H1 from model");
    let model = FalconH1Model::from_source(source.clone(), n_threads_arg)?;
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
    // Wrap the prompt with the model's canonical chat template.
    // `--chat-template <preset>` overrides the per-arch default; an
    // unknown architecture stays in raw base-model mode so parity
    // tests against llama.cpp still match byte-for-byte.
    let formatted_prompt = crate::models::chat_template::format_chat(
        &model.config.architecture,
        chat_template,
        prompt,
    )
    .unwrap_or_else(|| prompt.to_string());
    if formatted_prompt != prompt {
        let preset_name = chat_template.unwrap_or("auto");
        eprintln!(
            "Falcon-H1: chat-template preset = {preset_name} (resolved via {})",
            model.config.architecture
        );
    }
    let prompt_ids = tokenizer.encode(
        &formatted_prompt,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    if prompt_ids.is_empty() {
        return Err("Falcon-H1 prompt produced no tokens".into());
    }

    let mut scratch = FalconH1Scratch::new(&model.config, prompt_ids.len() + max_tokens);
    let infer_started = std::time::Instant::now();
    let started = std::time::Instant::now();
    let mut logits = model.prefill(&prompt_ids, &mut scratch)?;
    let t_prefill = started.elapsed();
    eprintln!(
        "Falcon-H1: prefill {} tokens in {:.2?}",
        prompt_ids.len(),
        t_prefill
    );

    let mut generated = Vec::with_capacity(max_tokens);
    // Track per-token counts so `apply_repetition_penalty` can divide each
    // repeated token's logit by penalty^count (llama.cpp / HF semantics).
    let mut token_counts: HashMap<u32, u32> = HashMap::new();
    let decode_started = std::time::Instant::now();
    for step in 0..max_tokens {
        crate::ops::apply_repetition_penalty(&mut logits, &token_counts, repetition_penalty);
        let next_token = sample_argmax(&logits, temperature);
        if Some(next_token) == tokenizer.eos_id() {
            break;
        }
        generated.push(next_token);
        *token_counts.entry(next_token).or_insert(0) += 1;
        if step + 1 < max_tokens {
            logits = model.prefill(&[next_token], &mut scratch)?;
        }
    }
    let t_decode = decode_started.elapsed();
    let piece = tokenizer.decode(&generated, true);
    println!("Output: {}", piece);

    let infer_ms = infer_started.elapsed().as_millis();
    let tok_s = if infer_ms > 0 {
        generated.len() as f64 / infer_ms as f64 * 1000.0
    } else {
        0.0
    };
    eprintln!(
        "Prompt: {:.1} t/s | Generation: {:.1} t/s | end-to-end: {:.1} tok/s",
        crate::app::cli::per_second(prompt_ids.len(), t_prefill),
        crate::app::cli::per_second(generated.len(), t_decode),
        tok_s
    );
    Ok(())
}

/// Greedy decode if `temperature <= 0.0`; otherwise sample with
/// llama.cpp's chain (temperature + top-k 0 + top-p 1.0).
fn sample_argmax(logits: &[f32], temperature: f32) -> u32 {
    if temperature <= 0.0 {
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
        let mut owned = logits.to_vec();
        let rng_u64 = rand::random::<u64>();
        sample_llama_cpp(&mut owned, 0, 1.0, temperature, rng_u64) as u32
    }
}

/// Parity instrumentation: when `RMI_FALCON_PARITY` is set, print a
/// sequential-f32 sum per checkpoint so it can be diffed against the
/// llama.cpp eval-callback dump (`oracle_sums.txt`).
pub(crate) fn parity_log(tag: &str, values: &[f32]) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if *ENABLED.get_or_init(|| std::env::var_os("RMI_FALCON_PARITY").is_some()) {
        let mut sum = 0.0f32;
        for &v in values {
            sum += v;
        }
        eprintln!("P {tag} sum={sum:.6}");
    }
}

// ---- ggml-exact scalar helpers (bit-identical ports of the AVX2 paths
// ggml-cpu uses for rms_norm / soft_max / silu on x86_64) ----

/// Scalar port of `ggml_v_expf` (AVX2, vec.h). Each SIMD lane computes
/// this exact sequence, so a per-lane scalar version is bit-identical.
#[inline]
fn v_expf_scalar(x: f32) -> f32 {
    const LOG2E: f32 = f32::from_bits(0x3FB8_AA3B); // 0x1.715476p+0
    const C4: f32 = f32::from_bits(0x3F7F_FFF6); // 0x1.ffffecp-1
    const C5: f32 = f32::from_bits(0x3EFF_FEDB); // 0x1.fffdb6p-2
    const C6: f32 = f32::from_bits(0x3E2A_AF33); // 0x1.555e66p-3
    const C7: f32 = f32::from_bits(0x3D2B_9F17); // 0x1.573e2ep-5
    const C8: f32 = f32::from_bits(0x3C07_2010); // 0x1.0e4020p-7
    let r = f32::from_bits(0x4B40_0000); // 0x1.8p23
    let z = x.mul_add(LOG2E, r);
    let n = z - r;
    // b = x - n*c2 - n*c1 (two fnmadd)
    let b = n.mul_add(f32::from_bits(0xBF31_7200), x); // -0x1.62e4p-1
    let b = n.mul_add(f32::from_bits(0xB5BF_BE8E), b); // -0x1.7f7d1cp-20
    let e: u32 = z.to_bits() << 23;
    let k = f32::from_bits(e.wrapping_add(1.0f32.to_bits()));
    let n_abs = f32::from_bits(n.to_bits() & 0x7fff_ffff);
    let big = n_abs > 126.0;
    let u = b * b;
    let inner = C6.mul_add(b, C5);
    let outer = C8.mul_add(b, C7);
    let j = outer.mul_add(u, inner).mul_add(u, C4 * b);
    if !big {
        return j.mul_add(k, k);
    }
    let g: u32 = if n <= 0.0 { 0x8200_0000 } else { 0 };
    let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
    let s2 = f32::from_bits(e.wrapping_sub(g));
    let huge = n_abs > 192.0;
    if huge {
        s1 * s1
    } else {
        s2.mul_add(j, s2) * s1
    }
}

/// `ggml_vec_silu_f32` semantics: AVX2 path uses `v_silu` for n >= 8,
/// scalar tail (libm expf) otherwise.
#[inline]
pub(crate) fn silu_ggml(x: f32) -> f32 {
    x / (1.0f32 + v_expf_scalar(-x))
}

/// `ggml_compute_forward_soft_max_f32` + `ggml_vec_soft_max_f32`:
/// scale-then-max-then-exp with the AVX2 `v_expf` for full 8-lane
/// groups (lane-summed in ggml's movehl/movehdup order into an f64
/// accumulator) and the libm scalar tail for the remainder.
pub(crate) fn softmax_ggml_inplace(x: &mut [f32], scale: f32) {
    if x.is_empty() {
        return;
    }
    for v in x.iter_mut() {
        *v *= scale;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let n = x.len();
    let n8 = n / 8 * 8;
    let mut sum = 0.0f64;
    let mut i = 0;
    while i < n8 {
        let mut lane = [0.0f32; 8];
        for j in 0..8 {
            lane[j] = v_expf_scalar(x[i + j] - max);
        }
        // ggml lane order: (v4+v0), + (v6+v2), + (v5+v1)
        let a = lane[4] + lane[0];
        let a = a + (lane[6] + lane[2]);
        let a = a + (lane[5] + lane[1]);
        sum += f64::from(a);
        for j in 0..8 {
            x[i + j] = lane[j];
        }
        i += 8;
    }
    while i < n {
        let val = (x[i] - max).exp();
        sum += f64::from(val);
        x[i] = val;
        i += 1;
    }
    let inv = (1.0 / sum) as f32;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `ggml_compute_forward_rms_norm_f32`: sequential scalar f64 sum of
/// squares, f32 mean/scale, per-element `(x * scale) * weight`.
pub(crate) fn rms_norm_ggml(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = input.len().min(weight.len()).min(output.len());
    let mut sum = 0.0f64;
    for &value in &input[..n] {
        sum += f64::from(value * value);
    }
    let mean = (sum / n as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();
    for i in 0..n {
        output[i] = input[i] * scale * weight[i];
    }
}
