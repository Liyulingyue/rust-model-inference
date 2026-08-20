//! Waveform Transformer for the Qwen3-TTS codec.
//!
//! 8-layer DiT-style transformer with per-channel layer scale (`ls1`, `ls2`).
//! Operates on `[in_dim=1024, length]` sequences — typically the continuous
//! RVQ embedding lifted to 1024 dims, optionally concatenated with the
//! speaker embedding. Outputs a 1024-dim sequence that is then consumed by
//! the DAC upsampler.

use crate::core::tensor::TensorSource;
use crate::models::qwen3::{
    check_allocation, checked_product, load_f32_tensor, static_q8_matrix, static_q8_tensor,
    usize_to_u64,
};
use crate::ops::{
    dot_f32, matmul_q8_0_quantized_parallel_rows, quantize_q8_0_into, rms_norm, silu,
};

const TFM_N_LAYER: usize = 8;
const TFM_IN_DIM: usize = 1024;
const TFM_N_EMBD: usize = 512;
const TFM_N_HEAD: usize = 16;
const TFM_HEAD_DIM: usize = 64;
const TFM_N_FF: usize = 1024;

pub(crate) struct TfmLayer {
    ln1: Vec<f32>,
    ls1: Vec<f32>,
    ln2: Vec<f32>,
    ls2: Vec<f32>,
    wq: &'static [u8],
    wk: &'static [u8],
    wv: &'static [u8],
    wo: &'static [u8],
    w_gate: &'static [u8],
    w_up: &'static [u8],
    w_down: &'static [u8],
}

pub struct WaveformTransformer {
    in_proj_w: &'static [u8],
    in_proj_b: Vec<f32>,
    layers: Vec<TfmLayer>,
    output_norm: Vec<f32>,
    out_proj_w: &'static [u8],
    out_proj_b: Vec<f32>,
}

impl WaveformTransformer {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let in_proj_dims = [
            usize_to_u64(TFM_IN_DIM, "tfm in_proj in")?,
            usize_to_u64(TFM_N_EMBD, "tfm in_proj out")?,
        ];
        let in_proj_w = static_q8_tensor(source, "a.gen.wav.tfm.in_proj.weight", &in_proj_dims)?;
        let in_proj_b = load_f32_tensor(
            source,
            "a.gen.wav.tfm.in_proj.bias",
            &[usize_to_u64(TFM_N_EMBD, "tfm in_proj bias")?],
        )?;

        let output_norm = load_f32_tensor(
            source,
            "a.gen.wav.tfm.output_norm.weight",
            &[usize_to_u64(TFM_N_EMBD, "tfm output_norm")?],
        )?;

        let out_proj_dims = [
            usize_to_u64(TFM_N_EMBD, "tfm out_proj in")?,
            usize_to_u64(TFM_IN_DIM, "tfm out_proj out")?,
        ];
        let out_proj_w = static_q8_tensor(source, "a.gen.wav.tfm.out_proj.weight", &out_proj_dims)?;
        let out_proj_b = load_f32_tensor(
            source,
            "a.gen.wav.tfm.out_proj.bias",
            &[usize_to_u64(TFM_IN_DIM, "tfm out_proj bias")?],
        )?;

        check_allocation(
            "tfm layers",
            TFM_N_LAYER,
            std::mem::size_of::<TfmLayer>(),
        )?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(TFM_N_LAYER)
            .map_err(|error| format!("Failed to allocate tfm layers: {error}"))?;
        for layer_idx in 0..TFM_N_LAYER {
            let prefix = format!("a.gen.wav.tfm.blk.{layer_idx}");
            let n_embd_dim = [usize_to_u64(TFM_N_EMBD, "tfm layer n_embd")?];
            let n_attn = checked_product("tfm attn", TFM_N_HEAD, TFM_HEAD_DIM)?;
            layers.push(TfmLayer {
                ln1: load_f32_tensor(source, &format!("{prefix}.ln1.weight"), &n_embd_dim)?,
                ls1: load_f32_tensor(source, &format!("{prefix}.ls1.weight"), &n_embd_dim)?,
                ln2: load_f32_tensor(source, &format!("{prefix}.ln2.weight"), &n_embd_dim)?,
                ls2: load_f32_tensor(source, &format!("{prefix}.ls2.weight"), &n_embd_dim)?,
                wq: static_q8_matrix(
                    source,
                    &format!("{prefix}.attn_q.weight"),
                    TFM_N_EMBD,
                    n_attn,
                )?,
                wk: static_q8_matrix(
                    source,
                    &format!("{prefix}.attn_k.weight"),
                    TFM_N_EMBD,
                    n_attn,
                )?,
                wv: static_q8_matrix(
                    source,
                    &format!("{prefix}.attn_v.weight"),
                    TFM_N_EMBD,
                    n_attn,
                )?,
                wo: static_q8_matrix(
                    source,
                    &format!("{prefix}.attn_out.weight"),
                    n_attn,
                    TFM_N_EMBD,
                )?,
                w_gate: static_q8_matrix(
                    source,
                    &format!("{prefix}.ffn_gate.weight"),
                    TFM_N_EMBD,
                    TFM_N_FF,
                )?,
                w_up: static_q8_matrix(
                    source,
                    &format!("{prefix}.ffn_up.weight"),
                    TFM_N_EMBD,
                    TFM_N_FF,
                )?,
                w_down: static_q8_matrix(
                    source,
                    &format!("{prefix}.ffn_down.weight"),
                    TFM_N_FF,
                    TFM_N_EMBD,
                )?,
            });
        }

        Ok(Self {
            in_proj_w,
            in_proj_b,
            layers,
            output_norm,
            out_proj_w,
            out_proj_b,
        })
    }

    /// Transform `[in_dim, length]` input to `[in_dim, length]` output via
    /// 8 DiT-style transformer layers. Returns the output buffer.
    pub fn forward(&self, input: &[f32], length: usize) -> Result<Vec<f32>, String> {
        if input.len() != TFM_IN_DIM * length {
            return Err(format!(
                "WaveformTransformer: input length {} != expected {}",
                input.len(),
                TFM_IN_DIM * length,
            ));
        }
        // in_proj: 1024 -> 512 with bias.
        let mut hidden = matmul_q8_bias(
            self.in_proj_w,
            Some(&self.in_proj_b),
            input,
            TFM_IN_DIM,
            TFM_N_EMBD,
            length,
        )?;

        let n_attn = checked_product("tfm attn", TFM_N_HEAD, TFM_HEAD_DIM)?;
        for layer in &self.layers {
            forward_tfm_layer(layer, &mut hidden, length, n_attn)?;
        }

        // output_norm -> out_proj 512 -> 1024 with bias.
        for t in 0..length {
            let off = t * TFM_N_EMBD;
            let mut normed = vec![0.0f32; TFM_N_EMBD];
            rms_norm(
                &hidden[off..off + TFM_N_EMBD],
                &self.output_norm,
                &mut normed,
                1e-6,
            );
            hidden[off..off + TFM_N_EMBD].copy_from_slice(&normed);
        }
        matmul_q8_bias(
            self.out_proj_w,
            Some(&self.out_proj_b),
            &hidden,
            TFM_N_EMBD,
            TFM_IN_DIM,
            length,
        )
    }
}

fn forward_tfm_layer(
    layer: &TfmLayer,
    hidden: &mut [f32],
    length: usize,
    n_attn: usize,
) -> Result<(), String> {
    let n_q = TFM_N_HEAD * TFM_HEAD_DIM;
    let mut normed = vec![0.0f32; TFM_N_EMBD];
    let mut q_all = vec![0.0f32; length * n_q];
    let mut k_all = vec![0.0f32; length * n_q];
    let mut v_all = vec![0.0f32; length * n_q];
    // ln1 -> qkv
    for t in 0..length {
        let off = t * TFM_N_EMBD;
        rms_norm(
            &hidden[off..off + TFM_N_EMBD],
            &layer.ln1,
            &mut normed,
            1e-6,
        );
        let blocks = (TFM_N_EMBD + 31) / 32;
        let mut q8_buf = vec![0u8; TFM_N_EMBD];
        let mut scale_buf = vec![0.0f32; blocks];
        quantize_q8_0_into(&normed, TFM_N_EMBD, &mut q8_buf, &mut scale_buf);
        let q_off = t * n_q;
        matmul_q8_0_quantized_parallel_rows(
            layer.wq,
            &q8_buf,
            &scale_buf,
            &mut q_all[q_off..q_off + n_q],
            TFM_N_EMBD,
            n_q,
            0,
            1,
        );
        matmul_q8_0_quantized_parallel_rows(
            layer.wk,
            &q8_buf,
            &scale_buf,
            &mut k_all[q_off..q_off + n_q],
            TFM_N_EMBD,
            n_q,
            0,
            1,
        );
        matmul_q8_0_quantized_parallel_rows(
            layer.wv,
            &q8_buf,
            &scale_buf,
            &mut v_all[q_off..q_off + n_q],
            TFM_N_EMBD,
            n_q,
            0,
            1,
        );
    }
    // Causal attention per head.
    let kq_scale = 1.0 / (TFM_HEAD_DIM as f32).sqrt();
    let mut attn_out = vec![0.0f32; length * n_attn];
    for head in 0..TFM_N_HEAD {
        let q_off = head * TFM_HEAD_DIM;
        let attn_off = head * TFM_HEAD_DIM;
        for i in 0..length {
            let mut max_val = f32::NEG_INFINITY;
            let mut scores = vec![0.0f32; length];
            for j in 0..=i {
                let q_row = &q_all[i * n_q + q_off..i * n_q + q_off + TFM_HEAD_DIM];
                let k_row = &k_all[j * n_q + q_off..j * n_q + q_off + TFM_HEAD_DIM];
                scores[j] = dot_f32(q_row, k_row, TFM_HEAD_DIM) * kq_scale;
                if scores[j] > max_val {
                    max_val = scores[j];
                }
            }
            let mut exp_sum = 0.0f32;
            for j in 0..=i {
                scores[j] = (scores[j] - max_val).exp();
                exp_sum += scores[j];
            }
            for j in 0..=i {
                scores[j] /= exp_sum;
            }
            for dim in 0..TFM_HEAD_DIM {
                let mut sum = 0.0f32;
                for j in 0..=i {
                    let v_row = &v_all[j * n_q + q_off..j * n_q + q_off + TFM_HEAD_DIM];
                    sum += scores[j] * v_row[dim];
                }
                attn_out[i * n_attn + attn_off + dim] = sum;
            }
        }
    }
    // attn_out projection + ls1 + residual
    let mut attn_proj = vec![0.0f32; length * TFM_N_EMBD];
    for t in 0..length {
        let attn_row = &attn_out[t * n_attn..t * n_attn + n_attn];
        let blocks = (n_attn + 31) / 32;
        let mut q8_buf = vec![0u8; n_attn];
        let mut scale_buf = vec![0.0f32; blocks];
        quantize_q8_0_into(attn_row, n_attn, &mut q8_buf, &mut scale_buf);
        matmul_q8_0_quantized_parallel_rows(
            layer.wo,
            &q8_buf,
            &scale_buf,
            &mut attn_proj[t * TFM_N_EMBD..t * TFM_N_EMBD + TFM_N_EMBD],
            n_attn,
            TFM_N_EMBD,
            0,
            1,
        );
        // apply ls1 (per-channel scale) and residual
        let off = t * TFM_N_EMBD;
        for i in 0..TFM_N_EMBD {
            hidden[off + i] += layer.ls1[i] * attn_proj[off + i];
        }
    }
    // ln2 -> ffn -> ls2 + residual
    for t in 0..length {
        let off = t * TFM_N_EMBD;
        rms_norm(
            &hidden[off..off + TFM_N_EMBD],
            &layer.ln2,
            &mut normed,
            1e-6,
        );
        let blocks = (TFM_N_EMBD + 31) / 32;
        let mut q8_buf = vec![0u8; TFM_N_EMBD];
        let mut scale_buf = vec![0.0f32; blocks];
        quantize_q8_0_into(&normed, TFM_N_EMBD, &mut q8_buf, &mut scale_buf);
        let mut gate = vec![0.0f32; TFM_N_FF];
        let mut up = vec![0.0f32; TFM_N_FF];
        matmul_q8_0_quantized_parallel_rows(
            layer.w_gate,
            &q8_buf,
            &scale_buf,
            &mut gate,
            TFM_N_EMBD,
            TFM_N_FF,
            0,
            1,
        );
        matmul_q8_0_quantized_parallel_rows(
            layer.w_up,
            &q8_buf,
            &scale_buf,
            &mut up,
            TFM_N_EMBD,
            TFM_N_FF,
            0,
            1,
        );
        // silu(gate) * up
        let mut silu_mul = vec![0.0f32; TFM_N_FF];
        for i in 0..TFM_N_FF {
            silu_mul[i] = silu(gate[i]) * up[i];
        }
        let blocks2 = (TFM_N_FF + 31) / 32;
        let mut q8_buf2 = vec![0u8; TFM_N_FF];
        let mut scale_buf2 = vec![0.0f32; blocks2];
        quantize_q8_0_into(&silu_mul, TFM_N_FF, &mut q8_buf2, &mut scale_buf2);
        let mut down = vec![0.0f32; TFM_N_EMBD];
        matmul_q8_0_quantized_parallel_rows(
            layer.w_down,
            &q8_buf2,
            &scale_buf2,
            &mut down,
            TFM_N_FF,
            TFM_N_EMBD,
            0,
            1,
        );
        for i in 0..TFM_N_EMBD {
            hidden[off + i] += layer.ls2[i] * down[i];
        }
    }
    Ok(())
}

fn matmul_q8_bias(
    weight: &[u8],
    bias: Option<&[f32]>,
    input: &[f32],
    in_dim: usize,
    out_dim: usize,
    n_tokens: usize,
) -> Result<Vec<f32>, String> {
    let blocks = (in_dim + 31) / 32;
    let expected_weight = blocks * out_dim * 34;
    if weight.len() != expected_weight {
        return Err(format!(
            "matmul_q8_bias: weight {} != expected {}",
            weight.len(),
            expected_weight
        ));
    }
    if input.len() != n_tokens * in_dim {
        return Err("matmul_q8_bias: input length mismatch".into());
    }
    if let Some(bias) = bias {
        if bias.len() != out_dim {
            return Err("matmul_q8_bias: bias length mismatch".into());
        }
    }
    let mut out = vec![0.0f32; n_tokens * out_dim];
    for t in 0..n_tokens {
        let mut q8_buf = vec![0u8; in_dim];
        let mut scale_buf = vec![0.0f32; blocks];
        quantize_q8_0_into(
            &input[t * in_dim..(t + 1) * in_dim],
            in_dim,
            &mut q8_buf,
            &mut scale_buf,
        );
        let o_off = t * out_dim;
        matmul_q8_0_quantized_parallel_rows(
            weight,
            &q8_buf,
            &scale_buf,
            &mut out[o_off..o_off + out_dim],
            in_dim,
            out_dim,
            0,
            1,
        );
        if let Some(bias) = bias {
            for v in 0..out_dim {
                out[o_off + v] += bias[v];
            }
        }
    }
    Ok(out)
}