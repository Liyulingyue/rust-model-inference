//! Code predictor for the Qwen3-TTS codec.
//!
//! Given a sequence of Talker audio tokens, predicts the 15 residual RVQ
//! indices per timestep. The Talker encodes only the first-level codebook
//! index; the codec decoder needs the full 16-level RVQ tuple to reconstruct
//! the audio embedding.
//!
//! Architecture (mirrors `a.gen.code.*`):
//!
//! 1. Embed Talker audio tokens via `out_embd[2048, 3072]` (transpose lookup).
//! 2. `proj_in [2048, 1024]` projects each 2048-dim vector to 1024-dim.
//! 3. 5 transformer blocks with Q/K per-head RMSNorm (16/8 GQA, head_dim=128).
//! 4. `output_norm` then `head [15, 2048, 1024]` produces 15 × 2048 logits per
//!    position; `argmax` over each head gives 15 residual RVQ codes.

use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::qwen3::{
    check_allocation, checked_product, load_f32_tensor, static_q8_matrix, static_q8_tensor,
    usize_to_u64,
};
use crate::ops::{
    dot_f32, matmul_q8_0_quantized_parallel_rows, quantize_q8_0_into, rms_norm, rms_norm_inplace,
    rope_neox,
};

const CODE_N_LAYER: usize = 5;
const CODE_N_EMBD: usize = 1024;
const CODE_N_HEAD: usize = 16;
const CODE_N_HEAD_KV: usize = 8;
const CODE_HEAD_DIM: usize = 128;
const CODE_N_FF: usize = 3072;
const CODE_RESIDUAL_LEVELS: usize = 15;
const CODE_VOCAB_PER_LEVEL: usize = 2048;

pub(crate) struct CodeLayer {
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    wq: &'static [u8],
    wk: &'static [u8],
    wv: &'static [u8],
    wo: &'static [u8],
    w_gate: &'static [u8],
    w_up: &'static [u8],
    w_down: &'static [u8],
}

pub struct CodePredictor {
    out_embd: Vec<f32>,
    embd: Vec<f32>,
    proj_in_w: &'static [u8],
    proj_in_b: Vec<f32>,
    layers: Vec<CodeLayer>,
    output_norm: Vec<f32>,
    head_w: &'static [u8],
}

impl CodePredictor {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let out_embd_dims = [
            usize_to_u64(CODE_N_EMBD * 2, "code out_embd dim")?,
            3072,
        ];
        let out_embd = load_q8_0_lookup(
            source,
            "a.gen.code.out_embd.weight",
            &out_embd_dims,
            3072,
            CODE_N_EMBD * 2,
        )?;

        let embd_dims = [
            (CODE_N_EMBD * 2) as u64,
            CODE_VOCAB_PER_LEVEL as u64,
            CODE_RESIDUAL_LEVELS as u64,
        ];
        let embd = load_q8_0_lookup(
            source,
            "a.gen.code.embd.weight",
            &embd_dims,
            CODE_RESIDUAL_LEVELS * CODE_VOCAB_PER_LEVEL,
            CODE_N_EMBD * 2,
        )?;

        let proj_in_dims = [
            usize_to_u64(CODE_N_EMBD * 2, "code proj_in in")?,
            usize_to_u64(CODE_N_EMBD, "code proj_in out")?,
        ];
        let proj_in_w = static_q8_tensor(source, "a.gen.code.proj_in.weight", &proj_in_dims)?;
        let proj_in_b = load_f32_tensor(
            source,
            "a.gen.code.proj_in.bias",
            &[usize_to_u64(CODE_N_EMBD, "code proj_in bias")?],
        )?;

        let output_norm = load_f32_tensor(
            source,
            "a.gen.code.output_norm.weight",
            &[usize_to_u64(CODE_N_EMBD, "code output_norm")?],
        )?;

        let head_dims = [
            CODE_N_EMBD as u64,
            CODE_VOCAB_PER_LEVEL as u64,
            CODE_RESIDUAL_LEVELS as u64,
        ];
        let head_w = static_q8_tensor(source, "a.gen.code.head.weight", &head_dims)?;

        check_allocation(
            "code predictor layers",
            CODE_N_LAYER,
            std::mem::size_of::<CodeLayer>(),
        )?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(CODE_N_LAYER)
            .map_err(|error| format!("Failed to allocate code predictor layers: {error}"))?;
        for layer_idx in 0..CODE_N_LAYER {
            let prefix = format!("a.gen.code.blk.{layer_idx}");
            let n_embd_dim = [usize_to_u64(CODE_N_EMBD, "code layer n_embd")?];
            let head_dim = [usize_to_u64(CODE_HEAD_DIM, "code layer head_dim")?];
            let n_attn = checked_product("code attn width", CODE_N_HEAD, CODE_HEAD_DIM)?;
            let n_embd_k = checked_product("code k width", CODE_N_HEAD_KV, CODE_HEAD_DIM)?;
            let n_embd_v = checked_product("code v width", CODE_N_HEAD_KV, CODE_HEAD_DIM)?;
            layers.push(CodeLayer {
                ln1: load_f32_tensor(
                    source,
                    &format!("{prefix}.ln1.weight"),
                    &n_embd_dim,
                )?,
                ln2: load_f32_tensor(
                    source,
                    &format!("{prefix}.ln2.weight"),
                    &n_embd_dim,
                )?,
                q_norm: load_f32_tensor(
                    source,
                    &format!("{prefix}.attn_q_norm.weight"),
                    &head_dim,
                )?,
                k_norm: load_f32_tensor(
                    source,
                    &format!("{prefix}.attn_k_norm.weight"),
                    &head_dim,
                )?,
                wq: static_q8_matrix(
                    source,
                    &format!("{prefix}.attn_q.weight"),
                    CODE_N_EMBD,
                    n_attn,
                )?,
                wk: static_q8_matrix(
                    source,
                    &format!("{prefix}.attn_k.weight"),
                    CODE_N_EMBD,
                    n_embd_k,
                )?,
                wv: static_q8_matrix(
                    source,
                    &format!("{prefix}.attn_v.weight"),
                    CODE_N_EMBD,
                    n_embd_v,
                )?,
                wo: static_q8_matrix(
                    source,
                    &format!("{prefix}.attn_out.weight"),
                    n_attn,
                    CODE_N_EMBD,
                )?,
                w_gate: static_q8_matrix(
                    source,
                    &format!("{prefix}.ffn_gate.weight"),
                    CODE_N_EMBD,
                    CODE_N_FF,
                )?,
                w_up: static_q8_matrix(
                    source,
                    &format!("{prefix}.ffn_up.weight"),
                    CODE_N_EMBD,
                    CODE_N_FF,
                )?,
                w_down: static_q8_matrix(
                    source,
                    &format!("{prefix}.ffn_down.weight"),
                    CODE_N_FF,
                    CODE_N_EMBD,
                )?,
            });
        }

        Ok(Self {
            out_embd,
            embd,
            proj_in_w,
            proj_in_b,
            layers,
            output_norm,
            head_w,
        })
    }

    /// Predict the 15 residual RVQ codes per timestep given a sequence of
    /// Talker audio tokens (each in `[0, 3072)`).
    pub fn predict(&self, talker_tokens: &[u32]) -> Result<Vec<u32>, String> {
        if talker_tokens.is_empty() {
            return Ok(Vec::new());
        }
        let n_tokens = talker_tokens.len();
        // Embed Talker tokens via out_embd (2048-dim per token).
        let mut hidden = embed_tokens(&self.out_embd, talker_tokens, CODE_N_EMBD * 2, n_tokens)?;
        // Project: 2048 -> 1024.
        hidden = matmul_q8_0_into(&self.proj_in_w, &hidden, &self.proj_in_b, 2048, CODE_N_EMBD, n_tokens)?;

        let n_embd_q = checked_product("code q width", CODE_N_HEAD, CODE_HEAD_DIM)?;
        let n_embd_k = checked_product("code k width", CODE_N_HEAD_KV, CODE_HEAD_DIM)?;
        let n_embd_v = checked_product("code v width", CODE_N_HEAD_KV, CODE_HEAD_DIM)?;
        let n_attn = checked_product("code attn", CODE_N_HEAD, CODE_HEAD_DIM)?;
        let group_size = CODE_N_HEAD / CODE_N_HEAD_KV;
        let kq_scale = 1.0 / (CODE_HEAD_DIM as f32).sqrt();

        for layer in &self.layers {
            forward_code_layer(layer, &mut hidden, n_tokens, n_embd_q, n_embd_k, n_embd_v, n_attn, group_size, kq_scale)?;
        }

        // Apply output_norm on the final hidden state.
        for t in 0..n_tokens {
            let off = t * CODE_N_EMBD;
            let mut normed = vec![0.0f32; CODE_N_EMBD];
            rms_norm(
                &hidden[off..off + CODE_N_EMBD],
                &self.output_norm,
                &mut normed,
                1e-6,
            );
            hidden[off..off + CODE_N_EMBD].copy_from_slice(&normed);
        }

        // Predict 15 residual codes via head [15, 2048, 1024].
        let mut residual_codes = Vec::with_capacity(n_tokens * CODE_RESIDUAL_LEVELS);
        for t in 0..n_tokens {
            let block = &hidden[t * CODE_N_EMBD..(t + 1) * CODE_N_EMBD];
            for level in 0..CODE_RESIDUAL_LEVELS {
                let level_offset = level * CODE_VOCAB_PER_LEVEL * CODE_N_EMBD;
                let mut logits = vec![0.0f32; CODE_VOCAB_PER_LEVEL];
                // head is Q8_0 tensor, treat as a sequence of 2048 entries each
                // applied to the same 1024-dim block.
                for v in 0..CODE_VOCAB_PER_LEVEL {
                    let off = level_offset + v * CODE_N_EMBD;
                    let mut acc = 0.0f32;
                    for (i, &x) in block.iter().enumerate() {
                        // Dequantize on the fly for simplicity.
                        let qb = (off + i) / 32 * 34;
                        let idx = (off + i) % 32;
                        let qbyte = self.head_w[qb + 2 + idx];
                        let scale = half::f16::from_le_bytes([self.head_w[qb], self.head_w[qb + 1]]).to_f32();
                        let w = scale * (qbyte as i8 as f32);
                        acc += x * w;
                    }
                    logits[v] = acc;
                }
                let best = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);
                residual_codes.push(best);
            }
        }
        Ok(residual_codes)
    }
}

fn forward_code_layer(
    layer: &CodeLayer,
    hidden: &mut [f32],
    n_tokens: usize,
    n_embd_q: usize,
    n_embd_k: usize,
    n_embd_v: usize,
    n_attn: usize,
    group_size: usize,
    kq_scale: f32,
) -> Result<(), String> {
    // ln1 -> qkv
    let mut q_all = vec![0.0f32; n_tokens * n_embd_q];
    let mut k_all = vec![0.0f32; n_tokens * n_embd_k];
    let mut v_all = vec![0.0f32; n_tokens * n_embd_v];
    let mut normed = vec![0.0f32; CODE_N_EMBD];
    for t in 0..n_tokens {
        let off = t * CODE_N_EMBD;
        rms_norm(
            &hidden[off..off + CODE_N_EMBD],
            &layer.ln1,
            &mut normed,
            1e-6,
        );
        let blocks = (CODE_N_EMBD + 31) / 32;
        let mut q8_buf = vec![0u8; CODE_N_EMBD];
        let mut scale_buf = vec![0.0f32; blocks];
        quantize_q8_0_into(&normed, CODE_N_EMBD, &mut q8_buf, &mut scale_buf);
        let q_off = t * n_embd_q;
        let k_off = t * n_embd_k;
        let v_off = t * n_embd_v;
        matmul_q8_0_quantized_parallel_rows(
            layer.wq,
            &q8_buf,
            &scale_buf,
            &mut q_all[q_off..q_off + n_embd_q],
            CODE_N_EMBD,
            n_embd_q,
            0,
            1,
        );
        matmul_q8_0_quantized_parallel_rows(
            layer.wk,
            &q8_buf,
            &scale_buf,
            &mut k_all[k_off..k_off + n_embd_k],
            CODE_N_EMBD,
            n_embd_k,
            0,
            1,
        );
        matmul_q8_0_quantized_parallel_rows(
            layer.wv,
            &q8_buf,
            &scale_buf,
            &mut v_all[v_off..v_off + n_embd_v],
            CODE_N_EMBD,
            n_embd_v,
            0,
            1,
        );
    }
    // Q/K per-head RMSNorm + Neox RoPE (code predictor uses plain Neox RoPE).
    for t in 0..n_tokens {
        for head in 0..CODE_N_HEAD {
            let off = head * CODE_HEAD_DIM;
            rms_norm_inplace(
                &mut q_all[t * n_embd_q + off..t * n_embd_q + off + CODE_HEAD_DIM],
                &layer.q_norm,
                1e-6,
            );
            rope_neox(
                &mut q_all[t * n_embd_q + off..t * n_embd_q + off + CODE_HEAD_DIM],
                t,
                CODE_HEAD_DIM,
                1_000_000.0,
            );
        }
        for head in 0..CODE_N_HEAD_KV {
            let off = head * CODE_HEAD_DIM;
            rms_norm_inplace(
                &mut k_all[t * n_embd_k + off..t * n_embd_k + off + CODE_HEAD_DIM],
                &layer.k_norm,
                1e-6,
            );
            rope_neox(
                &mut k_all[t * n_embd_k + off..t * n_embd_k + off + CODE_HEAD_DIM],
                t,
                CODE_HEAD_DIM,
                1_000_000.0,
            );
        }
    }
    // Causal attention per head.
    let mut attn_out = vec![0.0f32; n_tokens * n_attn];
    for head in 0..CODE_N_HEAD {
        let kv_head = head / group_size;
        let q_off = head * CODE_HEAD_DIM;
        let k_off = kv_head * CODE_HEAD_DIM;
        let v_off = kv_head * CODE_HEAD_DIM;
        let attn_off = head * CODE_HEAD_DIM;
        for i in 0..n_tokens {
            let mut max_val = f32::NEG_INFINITY;
            let mut scores = vec![0.0f32; n_tokens];
            for j in 0..=i {
                let q_row = &q_all[i * n_embd_q + q_off..i * n_embd_q + q_off + CODE_HEAD_DIM];
                let k_row = &k_all[j * n_embd_k + k_off..j * n_embd_k + k_off + CODE_HEAD_DIM];
                scores[j] = dot_f32(q_row, k_row, CODE_HEAD_DIM) * kq_scale;
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
            for dim in 0..CODE_HEAD_DIM {
                let mut sum = 0.0f32;
                for j in 0..=i {
                    let v_row = &v_all[j * n_embd_v + v_off..j * n_embd_v + v_off + CODE_HEAD_DIM];
                    sum += scores[j] * v_row[dim];
                }
                attn_out[i * n_attn + attn_off + dim] = sum;
            }
        }
    }
    // attn_out projection + residual
    let mut attn_proj = vec![0.0f32; n_tokens * CODE_N_EMBD];
    for t in 0..n_tokens {
        let attn_row = &attn_out[t * n_attn..t * n_attn + n_attn];
        let blocks = (n_attn + 31) / 32;
        let mut q8_buf = vec![0u8; n_attn];
        let mut scale_buf = vec![0.0f32; blocks];
        quantize_q8_0_into(attn_row, n_attn, &mut q8_buf, &mut scale_buf);
        matmul_q8_0_quantized_parallel_rows(
            layer.wo,
            &q8_buf,
            &scale_buf,
            &mut attn_proj[t * CODE_N_EMBD..t * CODE_N_EMBD + CODE_N_EMBD],
            n_attn,
            CODE_N_EMBD,
            0,
            1,
        );
    }
    for t in 0..n_tokens {
        let off = t * CODE_N_EMBD;
        for i in 0..CODE_N_EMBD {
            hidden[off + i] += attn_proj[off + i];
        }
    }
    // ln2 -> ffn
    for t in 0..n_tokens {
        let off = t * CODE_N_EMBD;
        rms_norm(
            &hidden[off..off + CODE_N_EMBD],
            &layer.ln2,
            &mut normed,
            1e-6,
        );
        let blocks = (CODE_N_EMBD + 31) / 32;
        let mut q8_buf = vec![0u8; CODE_N_EMBD];
        let mut scale_buf = vec![0.0f32; blocks];
        quantize_q8_0_into(&normed, CODE_N_EMBD, &mut q8_buf, &mut scale_buf);
        let mut gate = vec![0.0f32; CODE_N_FF];
        let mut up = vec![0.0f32; CODE_N_FF];
        matmul_q8_0_quantized_parallel_rows(
            layer.w_gate,
            &q8_buf,
            &scale_buf,
            &mut gate,
            CODE_N_EMBD,
            CODE_N_FF,
            0,
            1,
        );
        matmul_q8_0_quantized_parallel_rows(
            layer.w_up,
            &q8_buf,
            &scale_buf,
            &mut up,
            CODE_N_EMBD,
            CODE_N_FF,
            0,
            1,
        );
        // down
        let mut down = vec![0.0f32; CODE_N_EMBD];
        let silu_mul: Vec<f32> = (0..CODE_N_FF)
            .map(|i| {
                let s = 1.0 / (1.0 + (-gate[i]).exp());
                s * gate[i] * up[i]
            })
            .collect();
        let q8_blocks = (CODE_N_FF + 31) / 32;
        let mut q8_buf2 = vec![0u8; CODE_N_FF];
        let mut scale_buf2 = vec![0.0f32; q8_blocks];
        quantize_q8_0_into(&silu_mul, CODE_N_FF, &mut q8_buf2, &mut scale_buf2);
        matmul_q8_0_quantized_parallel_rows(
            layer.w_down,
            &q8_buf2,
            &scale_buf2,
            &mut down,
            CODE_N_FF,
            CODE_N_EMBD,
            0,
            1,
        );
        for i in 0..CODE_N_EMBD {
            hidden[off + i] += down[i];
        }
    }
    Ok(())
}

fn embed_tokens(
    table: &[f32],
    tokens: &[u32],
    dim: usize,
    n_tokens: usize,
) -> Result<Vec<f32>, String> {
    let vocab = table.len() / dim;
    let mut out = vec![0.0f32; n_tokens * dim];
    for (t, &tok) in tokens.iter().enumerate() {
        let idx = tok as usize;
        if idx >= vocab {
            return Err(format!("token {idx} >= vocab {vocab}"));
        }
        out[t * dim..(t + 1) * dim]
            .copy_from_slice(&table[idx * dim..(idx + 1) * dim]);
    }
    Ok(out)
}

fn matmul_q8_0_into(
    weight: &[u8],
    input: &[f32],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
    n_tokens: usize,
) -> Result<Vec<f32>, String> {
    let blocks = (in_dim + 31) / 32;
    let expected_weight = blocks * out_dim * 34;
    if weight.len() != expected_weight {
        return Err(format!(
            "matmul_q8_0_into: weight {} != expected {}",
            weight.len(),
            expected_weight,
        ));
    }
    if input.len() != n_tokens * in_dim {
        return Err("matmul_q8_0_into: input length mismatch".into());
    }
    if bias.len() != out_dim {
        return Err("matmul_q8_0_into: bias length mismatch".into());
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
        for v in 0..out_dim {
            out[o_off + v] += bias[v];
        }
    }
    Ok(out)
}

fn load_q8_0_lookup(
    source: &dyn TensorSource,
    name: &str,
    expected_dims: &[u64],
    entries: usize,
    dim: usize,
) -> Result<Vec<f32>, String> {
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor info: {name}"))?;
    if info.dims != expected_dims {
        return Err(format!(
            "{name}: dims {:?} != expected {expected_dims:?}",
            info.dims,
        ));
    }
    if info.ggml_type != GGMLType::Q8_0 {
        return Err(format!("{name}: type {:?} not Q8_0", info.ggml_type));
    }
    let blocks_per_row = dim / 32;
    let bytes_per_row = blocks_per_row * 34;
    let expected_bytes = checked_product("code q8 bytes", entries, bytes_per_row)?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "{name}: bytes {} != expected {}",
            bytes.len(),
            expected_bytes
        ));
    }
    let mut out = vec![0.0f32; entries * dim];
    for row in 0..entries {
        for b in 0..blocks_per_row {
            let off = row * bytes_per_row + b * 34;
            let scale = half::f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
            for j in 0..32usize {
                let q = bytes[off + 2 + j] as i8 as f32;
                out[row * dim + b * 32 + j] = scale * q;
            }
        }
    }
    Ok(out)
}