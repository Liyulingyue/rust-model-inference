//! SAN-M encoder + adaptor for Fun-ASR-Nano.
//!
//! The encoder is a 50+20 layer SAN-M (Self-Attention with Memory Network)
//! stack. Each layer: LN → SAN-M self-attn → +residual → LN → FFN(relu) → +residual.
//!
//! SAN-M attention = standard multi-head attention + FSMN memory branch:
//!   q,k,v = split(fused_linear_q_k_v(x))
//!   fsmn = shift_accumulate(fsmn_kernel, pad(v))
//!   attn  = softmax(qk^T / sqrt(dk)) * v → linear_out
//!   out   = attn + fsmn
//!
//! The adaptor projects encoder output [T, 512] → [T, 1024] for the Qwen3 LLM:
//!   linear1(512→2048) → relu → linear2(2048→1024) → 2 transformer layers

use crate::core::loader::load_static_weight;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::funasr::config::FunAsrConfig;
use crate::ops::kernel::Weight;
use crate::ops::softmax_inplace;
use crate::ops::sum_sq_centered_f32;
use crate::ops::{dot_f32, sum_f32, vec_add_into, vec_mad_per_channel_f32};
use std::sync::Arc;

const LN_EPS: f32 = 1e-5;

// ======================= Pre-loaded weight structs =======================

struct Linear {
    weight: Weight<'static>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    dim: usize,
}

struct SanmLayer {
    norm1: LayerNorm,
    qkv: Linear,
    fsmn: Vec<f32>,
    linear_out: Linear,
    norm2: LayerNorm,
    ff_w1: Linear,
    ff_w2: Linear,
}

struct AdpLayer {
    norm1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    linear_out: Linear,
    norm2: LayerNorm,
    ff_w1: Linear,
    ff_w2: Linear,
}

struct Adaptor {
    linear1: Linear,
    linear2: Linear,
    blocks: Vec<AdpLayer>,
}

pub struct FunAsrEncoder {
    pub config: FunAsrConfig,
    pool: Arc<ComputePool>,
    enc0: SanmLayer,
    encoders: Vec<SanmLayer>,
    after_norm: LayerNorm,
    tp_encoders: Vec<SanmLayer>,
    tp_norm: LayerNorm,
    adaptor: Adaptor,
}

impl FunAsrEncoder {
    pub fn new(source: Arc<dyn TensorSource>, pool: Arc<ComputePool>) -> Result<Self, String> {
        let config = FunAsrConfig::from_source(source.as_ref())?;

        let enc0 = load_sanm_layer(
            source.as_ref(),
            "audio_encoder.encoders0.0.",
            config.input_size,
            config.output_size,
        )?;

        let mut encoders = Vec::with_capacity(config.num_blocks - 1);
        for i in 0..config.num_blocks - 1 {
            encoders.push(load_sanm_layer(
                source.as_ref(),
                &format!("audio_encoder.encoders.{i}."),
                config.output_size,
                config.output_size,
            )?);
        }

        let after_norm = load_layernorm(source.as_ref(), "audio_encoder.after_norm.")?;
        let tp_norm = load_layernorm(source.as_ref(), "audio_encoder.tp_norm.")?;

        let mut tp_encoders = Vec::with_capacity(config.tp_blocks);
        for i in 0..config.tp_blocks {
            tp_encoders.push(load_sanm_layer(
                source.as_ref(),
                &format!("audio_encoder.tp_encoders.{i}."),
                config.output_size,
                config.output_size,
            )?);
        }

        let linear1 = load_linear(source.as_ref(), "audio_adaptor.linear1.")?;
        let linear2 = load_linear(source.as_ref(), "audio_adaptor.linear2.")?;
        let mut blocks = Vec::with_capacity(config.adp_n_layer);
        for i in 0..config.adp_n_layer {
            blocks.push(load_adp_layer(
                source.as_ref(),
                &format!("audio_adaptor.blocks.{i}."),
                config.adp_llm_dim,
            )?);
        }

        Ok(Self {
            config,
            pool,
            enc0,
            encoders,
            after_norm,
            tp_encoders,
            tp_norm,
            adaptor: Adaptor {
                linear1,
                linear2,
                blocks,
            },
        })
    }

    pub fn encode(&self, fbank: &[f32], t: usize) -> Result<Vec<f32>, String> {
        let d = self.config.input_size;
        let d_model = self.config.output_size;
        if fbank.len() != t * d {
            return Err(format!("fbank len {} != {} * {}", fbank.len(), t, d));
        }

        let n_head = self.config.attention_heads;
        let dk = d_model / n_head;
        let kernel = self.config.kernel_size;

        let mut x =
            self.sanm_layer_fwd(&self.enc0, fbank, t, d, d_model, n_head, dk, kernel, false);

        for layer in &self.encoders {
            x = self.sanm_layer_fwd(layer, &x, t, d_model, d_model, n_head, dk, kernel, true);
        }

        x = layernorm_fwd(&self.after_norm, &x, t, &self.pool);

        for layer in &self.tp_encoders {
            x = self.sanm_layer_fwd(layer, &x, t, d_model, d_model, n_head, dk, kernel, true);
        }

        x = layernorm_fwd(&self.tp_norm, &x, t, &self.pool);

        x = linear_fwd(&self.adaptor.linear1, &x, t, &self.pool);
        relu_inplace(&mut x);
        x = linear_fwd(&self.adaptor.linear2, &x, t, &self.pool);

        let adp_n_head = self.config.adp_attention_heads;
        let adp_dk = self.config.adp_llm_dim / adp_n_head;
        for layer in &self.adaptor.blocks {
            x = self.adp_layer_fwd(layer, &x, t, self.config.adp_llm_dim, adp_n_head, adp_dk);
        }

        Ok(x)
    }

    #[inline]
    fn sanm_layer_fwd(
        &self,
        layer: &SanmLayer,
        x: &[f32],
        t: usize,
        _in_dim: usize,
        out_dim: usize,
        n_head: usize,
        dk: usize,
        kernel: usize,
        residual: bool,
    ) -> Vec<f32> {
        let normed = layernorm_fwd(&layer.norm1, x, t, &self.pool);

        let qkv = linear_fwd(&layer.qkv, &normed, t, &self.pool);
        let (q, k, v) = split_qkv(&qkv, t, out_dim);

        let fsmn = fsmn_shift_accumulate(&v, t, out_dim, kernel, &layer.fsmn, &self.pool);

        let attn = multi_head_attention(&q, &k, &v, t, out_dim, n_head, dk, &self.pool);
        let o = linear_fwd(&layer.linear_out, &attn, t, &self.pool);

        let mut h = o;
        vec_add_into(&fsmn, &mut h);
        if residual {
            vec_add_into(&x[..t * out_dim], &mut h);
        }

        let normed2 = layernorm_fwd(&layer.norm2, &h, t, &self.pool);
        let ff1 = linear_fwd(&layer.ff_w1, &normed2, t, &self.pool);
        let mut ff1_relu = ff1;
        relu_inplace(&mut ff1_relu);
        let ff2 = linear_fwd(&layer.ff_w2, &ff1_relu, t, &self.pool);

        add_residual(&h, &ff2, t * out_dim)
    }

    #[inline]
    fn adp_layer_fwd(
        &self,
        layer: &AdpLayer,
        x: &[f32],
        t: usize,
        dim: usize,
        n_head: usize,
        dk: usize,
    ) -> Vec<f32> {
        let normed = layernorm_fwd(&layer.norm1, x, t, &self.pool);

        let q = linear_fwd(&layer.q, &normed, t, &self.pool);
        let k = linear_fwd(&layer.k, &normed, t, &self.pool);
        let v = linear_fwd(&layer.v, &normed, t, &self.pool);
        let attn = multi_head_attention(&q, &k, &v, t, dim, n_head, dk, &self.pool);
        let o = linear_fwd(&layer.linear_out, &attn, t, &self.pool);

        let mut h = add_residual(x, &o, t * dim);

        let normed2 = layernorm_fwd(&layer.norm2, &h, t, &self.pool);
        let ff1 = linear_fwd(&layer.ff_w1, &normed2, t, &self.pool);
        let mut ff1_relu = ff1;
        relu_inplace(&mut ff1_relu);
        let ff2 = linear_fwd(&layer.ff_w2, &ff1_relu, t, &self.pool);

        h = add_residual(&h, &ff2, t * dim);
        h
    }
}

// ======================= Weight loading =======================

fn load_linear(source: &dyn TensorSource, prefix: &str) -> Result<Linear, String> {
    let weight_name = format!("{prefix}weight");
    let bias_name = format!("{prefix}bias");
    let info = source
        .tensor_info(&weight_name)
        .ok_or_else(|| format!("tensor {weight_name} not found"))?;
    let (in_dim, out_dim) = if info.dims.len() == 2 {
        (info.dims[0] as usize, info.dims[1] as usize)
    } else {
        return Err(format!(
            "unexpected dims for {weight_name}: {:?}",
            info.dims
        ));
    };
    let weight = load_static_weight(source, &weight_name, in_dim, out_dim);
    let bias = load_f32_vec(source, &bias_name)?;
    Ok(Linear {
        weight,
        bias,
        in_dim,
        out_dim,
    })
}

fn load_layernorm(source: &dyn TensorSource, prefix: &str) -> Result<LayerNorm, String> {
    let weight = load_f32_vec(source, &format!("{prefix}weight"))?;
    let bias = load_f32_vec(source, &format!("{prefix}bias"))?;
    let dim = weight.len();
    Ok(LayerNorm { weight, bias, dim })
}

fn load_sanm_layer(
    source: &dyn TensorSource,
    prefix: &str,
    _in_dim: usize,
    _out_dim: usize,
) -> Result<SanmLayer, String> {
    let p = format!("{prefix}self_attn.");
    Ok(SanmLayer {
        norm1: load_layernorm(source, &format!("{prefix}norm1."))?,
        qkv: load_linear(source, &format!("{p}linear_q_k_v."))?,
        fsmn: load_f32_vec(source, &format!("{p}fsmn_block.weight"))?,
        linear_out: load_linear(source, &format!("{p}linear_out."))?,
        norm2: load_layernorm(source, &format!("{prefix}norm2."))?,
        ff_w1: load_linear(source, &format!("{prefix}feed_forward.w_1."))?,
        ff_w2: load_linear(source, &format!("{prefix}feed_forward.w_2."))?,
    })
}

fn load_adp_layer(
    source: &dyn TensorSource,
    prefix: &str,
    _dim: usize,
) -> Result<AdpLayer, String> {
    let p = format!("{prefix}self_attn.");
    Ok(AdpLayer {
        norm1: load_layernorm(source, &format!("{prefix}norm1."))?,
        q: load_linear(source, &format!("{p}linear_q."))?,
        k: load_linear(source, &format!("{p}linear_k."))?,
        v: load_linear(source, &format!("{p}linear_v."))?,
        linear_out: load_linear(source, &format!("{p}linear_out."))?,
        norm2: load_layernorm(source, &format!("{prefix}norm2."))?,
        ff_w1: load_linear(source, &format!("{prefix}feed_forward.w_1."))?,
        ff_w2: load_linear(source, &format!("{prefix}feed_forward.w_2."))?,
    })
}

fn load_f32_vec(source: &dyn TensorSource, name: &str) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("tensor {name} not found"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("tensor slice {name} not found"))?;
    let count = info.dims.iter().product::<u64>() as usize;
    let mut out = vec![0.0f32; count];
    match info.ggml_type {
        GGMLType::F32 => {
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        GGMLType::F16 => {
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                out[i] = crate::ops::f16_to_f32(u16::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        _ => return Err(format!("unsupported type {:?} for {name}", info.ggml_type)),
    }
    Ok(out)
}

// ======================= Forward primitives (ComputePool-based) =======================

struct SharedMut<T>(*mut T);
unsafe impl<T> Send for SharedMut<T> {}
unsafe impl<T> Sync for SharedMut<T> {}
impl<T> SharedMut<T> {
    #[inline]
    unsafe fn write(&self, index: usize, value: T) {
        self.0.add(index).write(value);
    }
    #[inline]
    unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(start), len)
    }
}

#[inline]
fn linear_fwd(lin: &Linear, input: &[f32], t: usize, pool: &ComputePool) -> Vec<f32> {
    let in_dim = lin.in_dim;
    let out_dim = lin.out_dim;
    let mut out = vec![0.0f32; t * out_dim];
    let out_ptr = SharedMut(out.as_mut_ptr());
    let bias = &lin.bias;
    let weight = &lin.weight;

    pool.compute(move |ith, nth| {
        let per = t.div_ceil(nth);
        let start = ith * per;
        let end = (start + per).min(t);
        for row in start..end {
            let input_row = &input[row * in_dim..(row + 1) * in_dim];
            let output_row = unsafe { out_ptr.slice(row * out_dim, out_dim) };
            if weight
                .kernel
                .forward_f16_strict(input_row, output_row, in_dim, out_dim)
            {
                // Strict F16xF16 path used by FunASR's mtmd-audio embedding
                // parity oracle. The F16 kernel opts in via the trait
                // method and returns true; other weight types return false
                // by default and fall through to the generic f32-input
                // path below.
            } else {
                weight
                    .kernel
                    .forward(input_row, output_row, in_dim, out_dim);
            }
            if !bias.is_empty() {
                for i in 0..out_dim {
                    output_row[i] += bias[i];
                }
            }
        }
    });
    out
}

#[inline]
fn layernorm_fwd(ln: &LayerNorm, x: &[f32], t: usize, pool: &ComputePool) -> Vec<f32> {
    let dim = ln.dim;
    let weight = &ln.weight;
    let bias = &ln.bias;
    let mut out = vec![0.0f32; t * dim];
    let out_ptr = SharedMut(out.as_mut_ptr());

    pool.compute(move |ith, nth| {
        let per = t.div_ceil(nth);
        let start = ith * per;
        let end = (start + per).min(t);
        for row in start..end {
            let input_row = &x[row * dim..(row + 1) * dim];
            let output_row = unsafe { out_ptr.slice(row * dim, dim) };
            let mean = (sum_f32(input_row) as f32) / dim as f32;
            let variance_sum = sum_sq_centered_f32(input_row, mean);
            let var = (variance_sum / dim as f64) as f32;
            let rstd = 1.0 / (var + LN_EPS).sqrt();
            // output[i] = (input[i] - mean) * rstd * weight[i] + bias[i]
            // Fused as: output = bias + normalized * weight, where normalized = (input - mean) * rstd
            // Step 1: output = (input - mean) * rstd  (scalar, dim ≤ 560)
            for i in 0..dim {
                output_row[i] = (input_row[i] - mean) * rstd;
            }
            // Step 2: output = bias + output * weight  (SIMD via vec_mad_per_channel)
            // vec_mad_per_channel: y[i] += x[i] * scale[i]
            // Need y=bias first, then y += output * weight → but y IS output.
            // Workaround: copy bias into output, then mad with old output values.
            // Simpler: just do fused scalar — dim is small (512/560), overhead negligible.
            for i in 0..dim {
                output_row[i] = output_row[i] * weight[i] + bias[i];
            }
        }
    });
    out
}

fn split_qkv(qkv: &[f32], t: usize, dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; t * dim];
    let mut k = vec![0.0f32; t * dim];
    let mut v = vec![0.0f32; t * dim];
    for i in 0..t {
        let row = &qkv[i * 3 * dim..(i + 1) * 3 * dim];
        q[i * dim..(i + 1) * dim].copy_from_slice(&row[..dim]);
        k[i * dim..(i + 1) * dim].copy_from_slice(&row[dim..2 * dim]);
        v[i * dim..(i + 1) * dim].copy_from_slice(&row[2 * dim..3 * dim]);
    }
    (q, k, v)
}

/// FSMN shift-accumulate: pad v by (K-1)/2 on each time side, then for each
/// output frame t: fsmn[t] = v[t] + sum_j kernel[:,j] * pad_v[t + j].
///
/// `fsmn_w` is [dim, kernel_size] row-major (transposed from PyTorch at export).
fn fsmn_shift_accumulate(
    v: &[f32],
    t: usize,
    dim: usize,
    kernel: usize,
    fsmn_w: &[f32],
    pool: &ComputePool,
) -> Vec<f32> {
    let pad = (kernel - 1) / 2;
    let mut padded = vec![0.0f32; (t + 2 * pad) * dim];
    for i in 0..t {
        padded[(i + pad) * dim..(i + pad + 1) * dim].copy_from_slice(&v[i * dim..(i + 1) * dim]);
    }
    let mut fsmn = vec![0.0f32; t * dim];
    let fsmn_ptr = SharedMut(fsmn.as_mut_ptr());

    pool.compute(move |ith, nth| {
        let per = t.div_ceil(nth);
        let start = ith * per;
        let end = (start + per).min(t);
        for t_idx in start..end {
            let out = unsafe { fsmn_ptr.slice(t_idx * dim, dim) };
            out.copy_from_slice(&v[t_idx * dim..(t_idx + 1) * dim]);
            for j in 0..kernel {
                let pad_idx = t_idx + j;
                let k_row = &fsmn_w[j * dim..(j + 1) * dim];
                let pad_row = &padded[pad_idx * dim..(pad_idx + 1) * dim];
                vec_mad_per_channel_f32(out, pad_row, k_row);
            }
        }
    });
    fsmn
}

/// Standard multi-head attention: softmax(Q @ K^T / sqrt(dk)) @ V.
///
/// QK^T uses `dot_f32` (AVX2/NEON). On ARM, KQV follows ggml's F32 dot
/// reduction order so the attention output matches the pinned CPU oracle.
fn multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    dim: usize,
    n_head: usize,
    dk: usize,
    pool: &ComputePool,
) -> Vec<f32> {
    let scale = 1.0 / (dk as f32).sqrt();
    let mut out = vec![0.0f32; t * dim];
    let out_ptr = SharedMut(out.as_mut_ptr());

    pool.compute(move |ith, nth| {
        let per = n_head.div_ceil(nth);
        let h_start = ith * per;
        let h_end = (h_start + per).min(n_head);
        if h_start >= h_end {
            return;
        }
        let mut scores = vec![0.0f32; t * t];
        for h in h_start..h_end {
            let off = h * dk;
            // QK^T: scores[i, j] = dot(Q[i, off..off+dk], K[j, off..off+dk]) * scale
            for i in 0..t {
                let q_row = &q[i * dim + off..i * dim + off + dk];
                for j in 0..t {
                    let k_row = &k[j * dim + off..j * dim + off + dk];
                    scores[i * t + j] = dot_f32(q_row, k_row, dk) * scale;
                }
            }
            // Softmax per query row
            for i in 0..t {
                let row = &mut scores[i * t..(i + 1) * t];
                softmax_inplace(row);
            }
            // KQV: out[i, off..off+dk] = sum_j scores[i,j] * V[j, off..off+dk]
            //
            // `kqv_dot_f32` is self-contained (zeros the slice and picks
            // SIMD/scalar internally). Do not re-accumulate below — that
            // would double the result.
            for i in 0..t {
                let out_row = unsafe { out_ptr.slice(i * dim + off, dk) };
                kqv_dot_f32(out_row, v, &scores[i * t..(i + 1) * t], t, dim, off);
            }
        }
    });
    out
}

/// `out[d] = sum_j scores[j] * V[j * dim + off + d]` for d in [0, dk).
/// Self-contained: zero-fills `out`, then accumulates via the SIMD
/// path on aarch64 or scalar fallback elsewhere. Mirrors the reduction
/// order in llama.cpp's mtmd-audio attention path so the bit pattern
/// matches the pinned CPU oracle.
pub(crate) fn kqv_dot_f32(
    out: &mut [f32],
    v: &[f32],
    scores: &[f32],
    t: usize,
    dim: usize,
    off: usize,
) {
    out.fill(0.0);
    #[cfg(target_arch = "x86_64")]
    {
        if out.len() % 8 == 0 && t >= 16 && std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                kqv_dot_f32_avx2(out, v, scores, t, dim, off);
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if out.len() % 4 == 0 && t >= 16 && std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                kqv_dot_f32_neon(out, v, scores, t, dim, off);
                return;
            }
        }
    }
    for j in 0..t {
        let s = scores[j];
        if s != 0.0 {
            let v_row = &v[j * dim + off..j * dim + off + out.len()];
            for (o, &vi) in out.iter_mut().zip(v_row.iter()) {
                *o += s * vi;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn kqv_dot_f32_avx2(
    out: &mut [f32],
    v: &[f32],
    scores: &[f32],
    t: usize,
    dim: usize,
    off: usize,
) {
    use std::arch::x86_64::*;

    // Mirror of the NEON path: 16 ymm accumulators per d-block (8 d's
    // wide), each holds the partial sum for one of the 16 j-slices. After
    // the main j-loop, reduce lane-wise across the 16 accumulators to get
    // 8 distinct output values (one per d-lane) and store them as a
    // single ymm.
    for d in (0..out.len()).step_by(8) {
        let mut acc = [_mm256_setzero_ps(); 16];
        let mut j = 0;
        while j + 16 <= t {
            for slot in 0..16 {
                acc[slot] = _mm256_fmadd_ps(
                    _mm256_loadu_ps(v.as_ptr().add((j + slot) * dim + off + d)),
                    _mm256_set1_ps(scores[j + slot]),
                    acc[slot],
                );
            }
            j += 16;
        }
        // Lane-wise reduce: sum across the 16 ymm accumulators keeps the
        // 8 d-lanes separate. acc[k][l] holds sum_j score[j] * V[j][d+l]
        // restricted to the j-slice owned by acc[k]; summing the 16 slices
        // gives the full per-lane result.
        let mut sum = acc[0];
        for k in 1..16 {
            sum = _mm256_add_ps(sum, acc[k]);
        }
        // `sum` is 8-wide with lane l = out[d+l] from the first 16 j's.

        while j + 8 <= t {
            let mut s = _mm256_setzero_ps();
            for slot in 0..8 {
                s = _mm256_fmadd_ps(
                    _mm256_loadu_ps(v.as_ptr().add((j + slot) * dim + off + d)),
                    _mm256_set1_ps(scores[j + slot]),
                    s,
                );
            }
            sum = _mm256_add_ps(sum, s);
            j += 8;
        }
        while j < t {
            let mut lane_add = _mm256_setzero_ps();
            let v_row = v.as_ptr().add(j * dim + off + d);
            let s = scores[j];
            // Broadcast score[j] and FMA into a fresh lane-add vector.
            lane_add = _mm256_fmadd_ps(_mm256_loadu_ps(v_row), _mm256_set1_ps(s), lane_add);
            sum = _mm256_add_ps(sum, lane_add);
            j += 1;
        }
        _mm256_storeu_ps(out.as_mut_ptr().add(d), sum);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn kqv_dot_f32_neon(
    out: &mut [f32],
    v: &[f32],
    scores: &[f32],
    t: usize,
    dim: usize,
    off: usize,
) {
    use std::arch::aarch64::*;

    for d in (0..out.len()).step_by(4) {
        let mut acc = [vdupq_n_f32(0.0); 16];
        let mut j = 0;
        while j + 16 <= t {
            for slot in 0..16 {
                acc[slot] = vfmaq_f32(
                    acc[slot],
                    vld1q_f32(v.as_ptr().add((j + slot) * dim + off + d)),
                    vdupq_n_f32(scores[j + slot]),
                );
            }
            j += 16;
        }

        let a = vaddq_f32(vaddq_f32(acc[0], acc[8]), vaddq_f32(acc[4], acc[12]));
        let b = vaddq_f32(vaddq_f32(acc[1], acc[9]), vaddq_f32(acc[5], acc[13]));
        let c = vaddq_f32(vaddq_f32(acc[2], acc[10]), vaddq_f32(acc[6], acc[14]));
        let e = vaddq_f32(vaddq_f32(acc[3], acc[11]), vaddq_f32(acc[7], acc[15]));
        let mut sum = vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, e));

        while j + 4 <= t {
            for slot in 0..4 {
                let product = vmulq_f32(
                    vld1q_f32(v.as_ptr().add((j + slot) * dim + off + d)),
                    vdupq_n_f32(scores[j + slot]),
                );
                sum = vaddq_f32(sum, product);
            }
            j += 4;
        }
        while j < t {
            sum = vfmaq_f32(
                sum,
                vld1q_f32(v.as_ptr().add(j * dim + off + d)),
                vdupq_n_f32(scores[j]),
            );
            j += 1;
        }
        vst1q_f32(out.as_mut_ptr().add(d), sum);
    }
}

#[inline]
fn relu_inplace(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[inline]
fn add_residual(x: &[f32], delta: &[f32], len: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    for i in 0..len {
        out[i] += delta[i];
    }
    out
}

#[cfg(test)]
mod parity_tests {
    use super::{fsmn_shift_accumulate, layernorm_fwd, linear_fwd, LayerNorm, Linear};
    #[cfg(target_arch = "aarch64")]
    use crate::core::tensor::GGMLType;
    use crate::core::thread_pool::ComputePool;
    #[cfg(target_arch = "aarch64")]
    use crate::ops::kernel::{QuantizedTensor, Weight};

    #[test]
    fn layernorm_rounds_sum_before_dividing_like_ggml() {
        let x = [207.840_07, 251.440_61, -868.942_26];
        let ln = LayerNorm {
            weight: vec![1.0; 3],
            bias: vec![0.0; 3],
            dim: 3,
        };
        let output = layernorm_fwd(&ln, &x, 1, &ComputePool::new(1));
        assert_eq!(
            output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [0x3f2a_2475, 0x3f3f_aebd, 0xbfb4_e99a]
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fsmn_rounds_product_before_adding_like_ggml() {
        let value = f32::from_bits(0xbf0b_ed36);
        let weight = f32::from_bits(0x3f6c_b1ef);
        let output =
            fsmn_shift_accumulate(&[value; 4], 1, 4, 1, &[weight; 4], &ComputePool::new(1));
        assert_eq!(
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [0xbf86_a692; 4]
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn f16_linear_rounds_input_to_half_like_scalar_ggml() {
        let bytes = Box::leak(Box::new([0xb1, 0x30, 0x8d, 0x3a, 0x0c, 0xb6, 0x1a, 0x2d]));
        let linear = Linear {
            weight: Weight::from_quantized(QuantizedTensor::from_bytes(bytes, GGMLType::F16, 4, 1)),
            bias: vec![0.0],
            in_dim: 4,
            out_dim: 1,
        };
        let output = linear_fwd(
            &linear,
            &[0.33331, -0.17299, 0.92345, -0.78231],
            1,
            &ComputePool::new(1),
        );
        assert_eq!(output[0].to_bits(), 0xbf01_0c34);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn f16_neon_dot_matches_scalar_ggml_with_tail() {
        let weight = [
            0xb9a1, 0x336c, 0xb949, 0x36cd, 0x32e7, 0xb367, 0x3385, 0x303d, 0xbb4f, 0x2bb6, 0x33a8,
            0x2905, 0x3a12, 0x3788, 0xb3dd, 0xb85f, 0xb9f4,
        ];
        let input = [
            0x38f7, 0x3613, 0x32ff, 0xaf75, 0x1844, 0xb873, 0x2bc9, 0xb9ad, 0x33c0, 0x342a, 0x3573,
            0xb80f, 0xba87, 0xb558, 0x3601, 0x2513, 0x37ec,
        ];
        let bytes = weight
            .iter()
            .flat_map(|bits: &u16| bits.to_le_bytes())
            .collect::<Vec<_>>();
        let actual = unsafe { super::funasr_f16_dot_f16_neon(&bytes, &input) };
        assert_eq!(actual.to_bits(), 0xbff1_455e);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn layernorm_accumulates_variance_in_ggml_neon_groups() {
        let x = [2.240_905_8, -826.513_1, -855.897_7, -161.255_62];
        let ln = LayerNorm {
            weight: vec![1.0; 4],
            bias: vec![0.0; 4],
            dim: 4,
        };
        let output = layernorm_fwd(&ln, &x, 1, &ComputePool::new(1));
        assert_eq!(
            output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [0x3f99_a896, 0xbf73_3fae, 0xbf83_6289, 0x3f46_b395]
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn kqv_matches_ggml_f32_dot_bits_for_vector_and_scalar_tails() {
        for (t, expected) in [
            (17, [0x3c20_3a26, 0x3c60_7cf1, 0x3c20_7854, 0x3c20_ae2f]),
            (20, [0x3c40_50e8, 0x3c80_4efc, 0x3c40_9c08, 0x3c40_dc28]),
            (70, [0x3d30_f017, 0x3d41_2d1c, 0x3d41_6a24, 0x3d31_6fdc]),
        ] {
            let scores = (0..t)
                .map(|j| f32::from_bits(0x3c00_0000 + (j * 317 % 0x10_0000) as u32))
                .collect::<Vec<_>>();
            let mut v = vec![0.0f32; t * 4];
            for j in 0..t {
                for d in 0..4 {
                    let mut bits = 0x3e80_0000 + ((j * 977 + d * 7919) % 0x10_0000) as u32;
                    if (j + d) % 3 == 0 {
                        bits |= 0x8000_0000;
                    }
                    v[j * 4 + d] = f32::from_bits(bits);
                }
            }
            let mut out = [0.0f32; 4];
            super::kqv_dot_f32(&mut out, &v, &scores, t, 4, 0);
            assert_eq!(out.map(f32::to_bits), expected, "t={t}");
        }
    }

    /// Scalar reference: `out[d] = sum_j scores[j] * V[j * dim + d]`.
    /// Used by the regression tests below to validate that every SIMD/scalar
    /// path inside `kqv_dot_f32` produces the same result — and crucially,
    /// that the caller doesn't double-count.
    fn kqv_dot_f32_scalar_reference(
        out: &mut [f32],
        v: &[f32],
        scores: &[f32],
        t: usize,
        dim: usize,
        off: usize,
    ) {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        for j in 0..t {
            let s = scores[j];
            if s != 0.0 {
                for (o, vi) in out.iter_mut().zip(v[j * dim + off..].iter()) {
                    *o += s * vi;
                }
            }
        }
    }

    /// Regression for the double-counting bug introduced when commit
    /// `ccd39b6` refactored the aarch64 NEON branch into a unified
    /// `kqv_dot_f32` call without removing the pre-existing scalar loop
    /// below it. The output must match the scalar reference exactly, not
    /// 2x it.
    #[test]
    fn kqv_dot_f32_scalar_fallback_matches_reference() {
        // t=4, dk=4 — forces scalar fallback (t < 16 threshold).
        let v: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let scores = [0.1, 0.2, 0.3, 0.4];
        let mut out = [0.0f32; 4];
        super::kqv_dot_f32(&mut out, &v, &scores, 4, 4, 0);
        let mut expected = [0.0f32; 4];
        kqv_dot_f32_scalar_reference(&mut expected, &v, &scores, 4, 4, 0);
        for (a, e) in out.iter().zip(expected.iter()) {
            assert_eq!(a.to_bits(), e.to_bits());
        }
    }

    /// Regression for double-counting: verifies that even with zero
    /// "explicit zero-fill" outside (the caller passes an uninitialised
    /// buffer), the function still produces the correct sum (not 2x).
    /// This catches any future caller that adds a redundant `out.fill(0)`
    /// followed by a fresh `kqv_dot_f32` call.
    #[test]
    fn kqv_dot_f32_does_not_double_count_with_uninitialised_input() {
        // dk=4, t=10 — scalar fallback path. If the function accumulated
        // twice (e.g. if a caller added a manual `out.fill(0)` followed by
        // a `for j in 0..t` loop), output would be 2x.
        let v: Vec<f32> = (0..40).map(|x| (x as f32) * 0.1).collect();
        let scores: Vec<f32> = (0..10).map(|x| (x as f32 + 1.0) * 0.05).collect();
        let mut out = [f32::from_bits(0xdeadbeef); 4]; // uninitialised sentinel
        super::kqv_dot_f32(&mut out, &v, &scores, 10, 4, 0);
        let mut expected = [0.0f32; 4];
        kqv_dot_f32_scalar_reference(&mut expected, &v, &scores, 10, 4, 0);
        for (a, e) in out.iter().zip(expected.iter()) {
            assert_eq!(a.to_bits(), e.to_bits(), "out != 2x reference");
        }
    }

    /// SIMD path coverage (t >= 16, dk divisible by 8 on x86_64 / 4 on aarch64).
    /// The function must select the SIMD kernel when available and still
    /// match the scalar reference within tight ULP tolerance.
    #[test]
    fn kqv_dot_f32_simd_path_matches_scalar_reference() {
        for (t, dk) in [(16, 4), (17, 8), (20, 8), (32, 8), (70, 8)] {
            let dim = dk;
            let v: Vec<f32> = (0..t * dim)
                .map(|x| f32::from_bits(0x3e80_0000 + (x * 977 % 0x10_0000) as u32))
                .collect();
            let scores: Vec<f32> = (0..t)
                .map(|j| f32::from_bits(0x3c00_0000 + (j * 317 % 0x10_0000) as u32))
                .collect();
            let mut out = vec![0.0f32; dk];
            super::kqv_dot_f32(&mut out, &v, &scores, t, dim, 0);
            let mut expected = vec![0.0f32; dk];
            kqv_dot_f32_scalar_reference(&mut expected, &v, &scores, t, dim, 0);
            for (a, e) in out.iter().zip(expected.iter()) {
                // SIMD and scalar reduction orders differ; allow a tiny
                // relative ULP gap (≤2 ULP) for paths that aren't bit-exact.
                let abs_e = e.abs();
                let ulp = (a.to_bits() as i32).abs_diff(e.to_bits() as i32);
                let rel = (a - e).abs() / abs_e.max(1e-30);
                assert!(
                    ulp <= 2 || rel < 1e-5,
                    "t={t} dk={dk}: simd={} ref={} (ulp={}, rel={})",
                    a,
                    e,
                    ulp,
                    rel,
                );
            }
        }
    }
}
