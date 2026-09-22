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
use crate::models::funasr::config::FunAsrConfig;
use crate::ops::kernel::Weight;
use rayon::prelude::*;
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
    fsmn: Vec<f32>,      // [dim, kernel_size] row-major, F32
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
    enc0: SanmLayer,
    encoders: Vec<SanmLayer>,
    after_norm: LayerNorm,
    tp_encoders: Vec<SanmLayer>,
    tp_norm: LayerNorm,
    adaptor: Adaptor,
}

impl FunAsrEncoder {
    pub fn new(source: Arc<dyn TensorSource>) -> Result<Self, String> {
        let config = FunAsrConfig::from_source(source.as_ref())?;

        // enc0: first layer (input dim may differ: input_size → output_size)
        let enc0 = load_sanm_layer(
            source.as_ref(),
            "audio_encoder.encoders0.0.",
            config.input_size,
            config.output_size,
        )?;

        // encoders.0..48
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

        // tp_encoders.0..19
        let mut tp_encoders = Vec::with_capacity(config.tp_blocks);
        for i in 0..config.tp_blocks {
            tp_encoders.push(load_sanm_layer(
                source.as_ref(),
                &format!("audio_encoder.tp_encoders.{i}."),
                config.output_size,
                config.output_size,
            )?);
        }

        // Adaptor
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

    /// Run the full encoder + adaptor pipeline.
    ///
    /// Input: `fbank` = [T * input_size] (already pre-scaled by sqrt(d_model) + position-encoded).
    /// Output: `[T * adp_llm_dim]` row-major f32.
    pub fn encode(&self, fbank: &[f32], t: usize) -> Result<Vec<f32>, String> {
        let d = self.config.input_size;
        let d_model = self.config.output_size;
        if fbank.len() != t * d {
            return Err(format!("fbank len {} != {} * {}", fbank.len(), t, d));
        }

        let n_head = self.config.attention_heads;
        let dk = d_model / n_head;
        let kernel = self.config.kernel_size;

        // enc0: first layer, no residual
        let mut x = self.sanm_layer_fwd(&self.enc0, fbank, t, d, d_model, n_head, dk, kernel, false);

        // encoders.0..48
        for layer in &self.encoders {
            x = self.sanm_layer_fwd(layer, &x, t, d_model, d_model, n_head, dk, kernel, true);
        }

        // after_norm
        x = layernorm_fwd(&self.after_norm, &x, t);

        // tp_encoders.0..19
        for layer in &self.tp_encoders {
            x = self.sanm_layer_fwd(layer, &x, t, d_model, d_model, n_head, dk, kernel, true);
        }

        // tp_norm
        x = layernorm_fwd(&self.tp_norm, &x, t);

        // Adaptor: linear1 → relu → linear2 → adp_layers
        x = linear_fwd(&self.adaptor.linear1, &x, t);
        relu_inplace(&mut x);
        x = linear_fwd(&self.adaptor.linear2, &x, t);

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
        in_dim: usize,
        out_dim: usize,
        n_head: usize,
        dk: usize,
        kernel: usize,
        residual: bool,
    ) -> Vec<f32> {
        let normed = layernorm_fwd(&layer.norm1, x, t);

        // Fused QKV
        let qkv = linear_fwd(&layer.qkv, &normed, t);
        let (q, k, v) = split_qkv(&qkv, t, out_dim);

        // FSMN
        let fsmn = fsmn_shift_accumulate(&v, t, out_dim, kernel, &layer.fsmn);

        // Attention
        let attn = multi_head_attention(&q, &k, &v, t, out_dim, n_head, dk);
        let o = linear_fwd(&layer.linear_out, &attn, t);

        // out = linear_out + fsmn
        let mut h = if residual {
            add_residual(x, &o, t * out_dim)
        } else {
            o
        };
        for i in 0..h.len() {
            h[i] += fsmn[i];
        }

        // FFN
        let normed2 = layernorm_fwd(&layer.norm2, &h, t);
        let ff1 = linear_fwd(&layer.ff_w1, &normed2, t);
        let mut ff1_relu = ff1;
        relu_inplace(&mut ff1_relu);
        let ff2 = linear_fwd(&layer.ff_w2, &ff1_relu, t);

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
        let normed = layernorm_fwd(&layer.norm1, x, t);

        let q = linear_fwd(&layer.q, &normed, t);
        let k = linear_fwd(&layer.k, &normed, t);
        let v = linear_fwd(&layer.v, &normed, t);
        let attn = multi_head_attention(&q, &k, &v, t, dim, n_head, dk);
        let o = linear_fwd(&layer.linear_out, &attn, t);

        let mut h = add_residual(x, &o, t * dim);

        let normed2 = layernorm_fwd(&layer.norm2, &h, t);
        let ff1 = linear_fwd(&layer.ff_w1, &normed2, t);
        let mut ff1_relu = ff1;
        relu_inplace(&mut ff1_relu);
        let ff2 = linear_fwd(&layer.ff_w2, &ff1_relu, t);

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
        return Err(format!("unexpected dims for {weight_name}: {:?}", info.dims));
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
    in_dim: usize,
    out_dim: usize,
) -> Result<SanmLayer, String> {
    let p = format!("{prefix}self_attn.");
    let qkv = load_linear(source, &format!("{p}linear_q_k_v."))?;
    let linear_out = load_linear(source, &format!("{p}linear_out."))?;
    let fsmn = load_f32_vec(source, &format!("{p}fsmn_block.weight"))?;
    let norm1 = load_layernorm(source, &format!("{prefix}norm1."))?;
    let norm2 = load_layernorm(source, &format!("{prefix}norm2."))?;
    let ff_w1 = load_linear(source, &format!("{prefix}feed_forward.w_1."))?;
    let ff_w2 = load_linear(source, &format!("{prefix}feed_forward.w_2."))?;
    Ok(SanmLayer {
        norm1,
        qkv,
        fsmn,
        linear_out,
        norm2,
        ff_w1,
        ff_w2,
    })
}

fn load_adp_layer(
    source: &dyn TensorSource,
    prefix: &str,
    dim: usize,
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

// ======================= Forward primitives =======================

#[inline]
fn linear_fwd(lin: &Linear, input: &[f32], t: usize) -> Vec<f32> {
    let in_dim = lin.in_dim;
    let out_dim = lin.out_dim;
    let mut out = vec![0.0f32; t * out_dim];
    out.par_chunks_mut(out_dim)
        .zip(input.par_chunks(in_dim))
        .for_each(|(o, row)| {
            lin.weight.kernel.forward(row, o, in_dim, out_dim);
            if !lin.bias.is_empty() {
                for i in 0..out_dim {
                    o[i] += lin.bias[i];
                }
            }
        });
    out
}

#[inline]
fn layernorm_fwd(ln: &LayerNorm, x: &[f32], t: usize) -> Vec<f32> {
    let dim = ln.dim;
    let mut out = vec![0.0f32; t * dim];
    out.par_chunks_mut(dim)
        .zip(x.par_chunks(dim))
        .for_each(|(o, row)| {
            let mean = row.iter().sum::<f32>() / dim as f32;
            let mut var = 0.0f32;
            for v in row {
                let d = v - mean;
                var += d * d;
            }
            var /= dim as f32;
            let rstd = 1.0 / (var + LN_EPS).sqrt();
            for i in 0..dim {
                o[i] = (row[i] - mean) * rstd * ln.weight[i] + ln.bias[i];
            }
        });
    out
}

#[inline]
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
fn fsmn_shift_accumulate(v: &[f32], t: usize, dim: usize, kernel: usize, fsmn_w: &[f32]) -> Vec<f32> {
    let pad = (kernel - 1) / 2;
    let mut padded = vec![0.0f32; (t + 2 * pad) * dim];
    for i in 0..t {
        padded[(i + pad) * dim..(i + pad + 1) * dim]
            .copy_from_slice(&v[i * dim..(i + 1) * dim]);
    }
    let mut fsmn = vec![0.0f32; t * dim];
    fsmn.par_chunks_mut(dim)
        .enumerate()
        .for_each(|(t_idx, out)| {
            out.copy_from_slice(&v[t_idx * dim..(t_idx + 1) * dim]);
            for j in 0..kernel {
                let pad_idx = t_idx + j;
                let k_row = &fsmn_w[j * dim..(j + 1) * dim];
                let pad_row = &padded[pad_idx * dim..(pad_idx + 1) * dim];
                for i in 0..dim {
                    out[i] += k_row[i] * pad_row[i];
                }
            }
        });
    fsmn
}

/// Standard multi-head attention: softmax(Q @ K^T / sqrt(dk)) @ V.
fn multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    dim: usize,
    n_head: usize,
    dk: usize,
) -> Vec<f32> {
    let scale = 1.0 / (dk as f32).sqrt();
    let mut out = vec![0.0f32; t * dim];
    let mut scores = vec![0.0f32; t * t];
    for h in 0..n_head {
        let off = h * dk;
        for i in 0..t {
            for j in 0..t {
                let mut s = 0.0f32;
                for d in 0..dk {
                    s += q[i * dim + off + d] * k[j * dim + off + d];
                }
                scores[i * t + j] = s * scale;
            }
        }
        for i in 0..t {
            let row = &mut scores[i * t..(i + 1) * t];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in row.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in row.iter_mut() {
                *s /= sum;
            }
        }
        for i in 0..t {
            for d in 0..dk {
                let mut acc = 0.0f32;
                for j in 0..t {
                    acc += scores[i * t + j] * v[j * dim + off + d];
                }
                out[i * dim + off + d] = acc;
            }
        }
    }
    out
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
