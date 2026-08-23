//! Per-frame autoregressive code predictor for the Qwen3-TTS codec.
//!
//! For each Talker-generated audio token, this module:
//!
//! 1. Embeds the token via `out_embd` to produce `code0_embd` (2048-dim).
//! 2. Prefills the predictor with `h_state` (2048-dim talker hidden state) at
//!    position 0 (output discarded — only seeds the KV cache), then with
//!    `code0_embd` at position 1 to produce the first acoustic code.
//! 3. Runs 14 autoregressive steps: at step g, embeds the previous step's
//!    code via `embd[g-1]`, applies the predictor, samples from `head[g]`,
//!    writes the new code into the cache.
//! 4. Sums all 16 codebook embeddings to produce `out_embd`, which is fed
//!    back to the talker for the next frame's prediction.
//!
//! The KV cache has 16 slots (one per code level, indexed 0..=15). Slot 0 holds
//! the talker h_state seed; slots 1..=15 hold the 15 acoustic codes.

use rand::Rng;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::qwen3::{
    checked_product, load_f32_tensor, static_q8_matrix, static_q8_tensor, usize_to_u64,
};
use crate::models::tts::talker::{mad_f16_inplace, scale_f16_inplace};
#[cfg(target_arch = "aarch64")]
use crate::ops::kernel::q8_0::dispatch::matmul_q8_0_quantized_range_nrc1;
#[cfg(not(target_arch = "aarch64"))]
use crate::ops::matmul_q8_0_quantized_parallel_rows;
use crate::ops::{
    dot_f16, f16_to_f32, f32_slice_to_f16, quantize_q8_0_into, rms_norm, rms_norm_inplace,
    rope_neox, silu_mul_approx_inplace, vec_scale_f32,
};

use super::{RVQ_CODEBOOK_SIZE, RVQ_LEVELS};

const PRED_N_LAYER: usize = 5;
const PRED_N_EMBD_IN: usize = 2048; // after out_embd / embd lookup
const PRED_N_EMBD: usize = 1024; // internal hidden dim (after proj_in)
const PRED_N_HEAD: usize = 16;
const PRED_N_HEAD_KV: usize = 8;
const PRED_HEAD_DIM: usize = 128;
const PRED_N_FF: usize = 3072;
const PRED_VOCAB: usize = RVQ_CODEBOOK_SIZE; // 2048 per level
const PRED_ACOUSTIC_LEVELS: usize = RVQ_LEVELS - 1; // 15
const PRED_N_SLOTS: usize = RVQ_LEVELS; // 16

fn predictor_matmul(
    weight: &[u8],
    input_q8: &[u8],
    input_scales: &[f32],
    output: &mut [f32],
    n_in: usize,
    n_out: usize,
    ith: usize,
    nth: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        debug_assert_eq!((ith, nth), (0, 1));
        matmul_q8_0_quantized_range_nrc1(weight, input_q8, input_scales, output, n_in, 0, n_out);
    }
    #[cfg(not(target_arch = "aarch64"))]
    matmul_q8_0_quantized_parallel_rows(
        weight,
        input_q8,
        input_scales,
        output,
        n_in,
        n_out,
        ith,
        nth,
    );
}

pub(crate) struct PredLayer {
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
    out_embd: Vec<f32>, // [vocab=3072, embd=2048] lookup for code0
    embd: Vec<f32>,     // [levels=15, vocab=2048, embd=2048] lookup for code[1..15]
    proj_in_w: &'static [u8],
    proj_in_b: Vec<f32>,
    layers: Vec<PredLayer>,
    output_norm: Vec<f32>,
    head_w: &'static [u8],
    eps: f32,
}

impl CodePredictor {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let out_embd_dims = [usize_to_u64(PRED_N_EMBD_IN, "code out_embd dim")?, 3072];
        let out_embd = load_q8_lookup_f32(
            source,
            "a.gen.code.out_embd.weight",
            &out_embd_dims,
            3072,
            PRED_N_EMBD_IN,
        )?;
        let embd_dims = [
            (PRED_N_EMBD_IN) as u64,
            PRED_VOCAB as u64,
            PRED_ACOUSTIC_LEVELS as u64,
        ];
        let embd = load_q8_lookup_f32(
            source,
            "a.gen.code.embd.weight",
            &embd_dims,
            PRED_ACOUSTIC_LEVELS * PRED_VOCAB,
            PRED_N_EMBD_IN,
        )?;
        let proj_in_dims = [
            usize_to_u64(PRED_N_EMBD_IN, "pred proj_in in")?,
            usize_to_u64(PRED_N_EMBD, "pred proj_in out")?,
        ];
        let proj_in_w = static_q8_tensor(source, "a.gen.code.proj_in.weight", &proj_in_dims)?;
        let proj_in_b = load_f32_tensor(
            source,
            "a.gen.code.proj_in.bias",
            &[usize_to_u64(PRED_N_EMBD, "pred proj_in bias")?],
        )?;
        let output_norm = load_f32_tensor(
            source,
            "a.gen.code.output_norm.weight",
            &[usize_to_u64(PRED_N_EMBD, "pred output_norm")?],
        )?;
        let head_dims = [
            PRED_N_EMBD as u64,
            PRED_VOCAB as u64,
            PRED_ACOUSTIC_LEVELS as u64,
        ];
        let head_w = static_q8_tensor(source, "a.gen.code.head.weight", &head_dims)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(PRED_N_LAYER)
            .map_err(|e| format!("alloc pred layers: {e}"))?;
        for layer_idx in 0..PRED_N_LAYER {
            let prefix = format!("a.gen.code.blk.{layer_idx}");
            let n_embd_dim = [usize_to_u64(PRED_N_EMBD, "pred n_embd")?];
            let head_dim = [usize_to_u64(PRED_HEAD_DIM, "pred head_dim")?];
            let n_attn = checked_product("pred attn", PRED_N_HEAD, PRED_HEAD_DIM)?;
            let n_embd_k = checked_product("pred k", PRED_N_HEAD_KV, PRED_HEAD_DIM)?;
            let n_embd_v = checked_product("pred v", PRED_N_HEAD_KV, PRED_HEAD_DIM)?;
            layers.push(PredLayer {
                ln1: load_f32_tensor(source, &format!("{prefix}.ln1.weight"), &n_embd_dim)?,
                ln2: load_f32_tensor(source, &format!("{prefix}.ln2.weight"), &n_embd_dim)?,
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
                    PRED_N_EMBD,
                    n_attn,
                )?,
                wk: static_q8_matrix(
                    source,
                    &format!("{prefix}.attn_k.weight"),
                    PRED_N_EMBD,
                    n_embd_k,
                )?,
                wv: static_q8_matrix(
                    source,
                    &format!("{prefix}.attn_v.weight"),
                    PRED_N_EMBD,
                    n_embd_v,
                )?,
                wo: static_q8_matrix(
                    source,
                    &format!("{prefix}.attn_out.weight"),
                    n_attn,
                    PRED_N_EMBD,
                )?,
                w_gate: static_q8_matrix(
                    source,
                    &format!("{prefix}.ffn_gate.weight"),
                    PRED_N_EMBD,
                    PRED_N_FF,
                )?,
                w_up: static_q8_matrix(
                    source,
                    &format!("{prefix}.ffn_up.weight"),
                    PRED_N_EMBD,
                    PRED_N_FF,
                )?,
                w_down: static_q8_matrix(
                    source,
                    &format!("{prefix}.ffn_down.weight"),
                    PRED_N_FF,
                    PRED_N_EMBD,
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
            eps: 1e-6,
        })
    }

    /// Predict one frame's 16 RVQ codes given the talker hidden state and the
    /// sampled talker audio token id.
    ///
    /// Returns `(codes[0..16], out_embd)` where `out_embd` is the sum of all
    /// 16 codebook embeddings — to be fed back to the talker as the next
    /// frame's embedding.
    pub fn predict_frame<R: Rng + ?Sized>(
        &self,
        h_state: &[f32],
        code0: u32,
        top_k: usize,
        rng: &mut R,
    ) -> Result<([u32; RVQ_LEVELS], Vec<f32>), String> {
        if h_state.len() != PRED_N_EMBD_IN {
            return Err(format!(
                "predict_frame: h_state length {} != {PRED_N_EMBD_IN}",
                h_state.len()
            ));
        }
        if code0 as usize >= PRED_VOCAB {
            return Err(format!(
                "predict_frame: semantic code {code0} >= {PRED_VOCAB}"
            ));
        }
        // KV cache: [layer][n_embd_head * n_head_kv * n_slots] in row-major.
        let n_embd_k = checked_product("pred k", PRED_N_HEAD_KV, PRED_HEAD_DIM)?;
        let cache_stride = n_embd_k;
        let cache_size = cache_stride * PRED_N_SLOTS;
        let mut k_cache: Vec<Vec<u16>> = vec![vec![0; cache_size]; PRED_N_LAYER];
        let mut v_cache: Vec<Vec<u16>> = vec![vec![0; cache_size]; PRED_N_LAYER];

        // Pre-allocated activations.
        let mut hidden = vec![0.0f32; PRED_N_EMBD];

        // ----- Step 0: seed KV cache with h_state at pos 0. Output discarded. -----
        let h_hidden = matmul_with_bias(
            self.proj_in_w,
            &self.proj_in_b,
            h_state,
            PRED_N_EMBD_IN,
            PRED_N_EMBD,
        )?;
        hidden.copy_from_slice(&h_hidden);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            forward_layer_inplace(
                layer,
                layer_idx,
                &mut hidden,
                &mut k_cache,
                &mut v_cache,
                0,
                cache_stride,
                self.eps,
            )?;
        }

        // ----- Step 1: code0_embd at pos 1 -> head[0] -> sample -> codes[1] -----
        let code0_embd = lookup_f32(&self.out_embd, code0 as usize, PRED_N_EMBD_IN);
        let c0_hidden = matmul_with_bias(
            self.proj_in_w,
            &self.proj_in_b,
            &code0_embd,
            PRED_N_EMBD_IN,
            PRED_N_EMBD,
        )?;
        hidden.copy_from_slice(&c0_hidden);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            forward_layer_inplace(
                layer,
                layer_idx,
                &mut hidden,
                &mut k_cache,
                &mut v_cache,
                1,
                cache_stride,
                self.eps,
            )?;
        }
        // Sample from head[0].
        let code1 = sample_at_head(
            &self.head_w,
            &hidden,
            &self.output_norm,
            self.eps,
            0,
            top_k,
            rng,
        )?;

        // ----- Steps 2..16: each step g (g=1..14) reads cache[g], uses embd[g-1]. -----
        let mut codes = [0u32; PRED_N_SLOTS];
        codes[0] = code0;
        codes[1] = code1;
        let mut prev_code = code1;
        for g in 1..PRED_ACOUSTIC_LEVELS as u32 {
            let emb = lookup_embd_f32(
                &self.embd,
                (g - 1) as usize,
                prev_code as usize,
                PRED_N_EMBD_IN,
            );
            let h = matmul_with_bias(
                self.proj_in_w,
                &self.proj_in_b,
                &emb,
                PRED_N_EMBD_IN,
                PRED_N_EMBD,
            )?;
            hidden.copy_from_slice(&h);
            let pos = (g + 1) as usize;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                forward_layer_inplace(
                    layer,
                    layer_idx,
                    &mut hidden,
                    &mut k_cache,
                    &mut v_cache,
                    pos,
                    cache_stride,
                    self.eps,
                )?;
            }
            let sampled = sample_at_head(
                &self.head_w,
                &hidden,
                &self.output_norm,
                self.eps,
                g as usize,
                top_k,
                rng,
            )?;
            codes[(g + 1) as usize] = sampled;
            prev_code = sampled;
        }

        // ----- Sum all 16 codebook embeddings to produce out_embd. -----
        let semantic = lookup_f32(&self.out_embd, codes[0] as usize, PRED_N_EMBD_IN);
        let acoustic: Vec<Vec<f32>> = codes[1..]
            .iter()
            .enumerate()
            .map(|(level, &code)| lookup_embd_f32(&self.embd, level, code as usize, PRED_N_EMBD_IN))
            .collect();
        let sum = sum_frame_embeddings(
            &semantic,
            acoustic.iter().map(Vec::as_slice),
            PRED_N_EMBD_IN,
        )?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::token_ids("tts.frame_codes", &codes));
        Ok((codes, sum))
    }
}

fn lookup_f32(table: &[f32], index: usize, dim: usize) -> Vec<f32> {
    table[index * dim..(index + 1) * dim].to_vec()
}

fn lookup_embd_f32(table: &[f32], level: usize, code: usize, dim: usize) -> Vec<f32> {
    let stride = RVQ_CODEBOOK_SIZE * dim;
    let off = level * stride + code * dim;
    table[off..off + dim].to_vec()
}

fn matmul_with_bias(
    weight: &[u8],
    bias: &[f32],
    input: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>, String> {
    let blocks = (in_dim + 31) / 32;
    let expected = blocks * out_dim * 34;
    if weight.len() != expected {
        return Err(format!(
            "matmul_with_bias: weight {} != expected {expected}",
            weight.len()
        ));
    }
    let mut q8_buf = vec![0u8; in_dim];
    let mut scale_buf = vec![0.0f32; blocks];
    quantize_q8_0_into(input, in_dim, &mut q8_buf, &mut scale_buf);
    let mut out = vec![0.0f32; out_dim];
    predictor_matmul(weight, &q8_buf, &scale_buf, &mut out, in_dim, out_dim, 0, 1);
    for (o, b) in out.iter_mut().zip(bias.iter()) {
        *o += *b;
    }
    Ok(out)
}

/// Forward a single predictor layer at one position with KV cache.
fn forward_layer_inplace(
    layer: &PredLayer,
    layer_idx: usize,
    hidden: &mut [f32],
    k_cache: &mut [Vec<u16>],
    v_cache: &mut [Vec<u16>],
    pos: usize,
    cache_stride: usize,
    eps: f32,
) -> Result<(), String> {
    let n_embd_q = checked_product("pred q", PRED_N_HEAD, PRED_HEAD_DIM)?;
    let n_embd_k = checked_product("pred k", PRED_N_HEAD_KV, PRED_HEAD_DIM)?;
    let n_embd_v = checked_product("pred v", PRED_N_HEAD_KV, PRED_HEAD_DIM)?;
    let n_attn = checked_product("pred attn", PRED_N_HEAD, PRED_HEAD_DIM)?;
    let group_size = PRED_N_HEAD / PRED_N_HEAD_KV;
    let kq_scale = 1.0 / (PRED_HEAD_DIM as f32).sqrt();

    // ln1 -> qkv
    let mut normed = vec![0.0f32; PRED_N_EMBD];
    rms_norm(hidden, &layer.ln1, &mut normed, eps);
    let blocks = (PRED_N_EMBD + 31) / 32;
    let mut q8_buf = vec![0u8; PRED_N_EMBD];
    let mut scale_buf = vec![0.0f32; blocks];
    quantize_q8_0_into(&normed, PRED_N_EMBD, &mut q8_buf, &mut scale_buf);
    let mut q = vec![0.0f32; n_embd_q];
    let mut k = vec![0.0f32; n_embd_k];
    let mut v = vec![0.0f32; n_embd_v];
    predictor_matmul(
        layer.wq,
        &q8_buf,
        &scale_buf,
        &mut q,
        PRED_N_EMBD,
        n_embd_q,
        0,
        1,
    );
    predictor_matmul(
        layer.wk,
        &q8_buf,
        &scale_buf,
        &mut k,
        PRED_N_EMBD,
        n_embd_k,
        0,
        1,
    );
    predictor_matmul(
        layer.wv,
        &q8_buf,
        &scale_buf,
        &mut v,
        PRED_N_EMBD,
        n_embd_v,
        0,
        1,
    );
    // Q/K per-head RMSNorm + Neox RoPE.
    for head in 0..PRED_N_HEAD {
        let off = head * PRED_HEAD_DIM;
        rms_norm_inplace(&mut q[off..off + PRED_HEAD_DIM], &layer.q_norm, eps);
        rope_neox(
            &mut q[off..off + PRED_HEAD_DIM],
            pos,
            PRED_HEAD_DIM,
            1_000_000.0,
        );
    }
    for head in 0..PRED_N_HEAD_KV {
        let off = head * PRED_HEAD_DIM;
        rms_norm_inplace(&mut k[off..off + PRED_HEAD_DIM], &layer.k_norm, eps);
        rope_neox(
            &mut k[off..off + PRED_HEAD_DIM],
            pos,
            PRED_HEAD_DIM,
            1_000_000.0,
        );
    }
    // Write K/V into THIS layer's cache row at `pos`.
    let cache_row = pos * cache_stride;
    for head in 0..PRED_N_HEAD_KV {
        let off = head * PRED_HEAD_DIM;
        let k_dst = &mut k_cache[layer_idx][cache_row + off..cache_row + off + PRED_HEAD_DIM];
        f32_slice_to_f16(&k[off..off + PRED_HEAD_DIM], k_dst);
        let v_dst = &mut v_cache[layer_idx][cache_row + off..cache_row + off + PRED_HEAD_DIM];
        f32_slice_to_f16(&v[off..off + PRED_HEAD_DIM], v_dst);
    }
    // Causal attention: read K/V from THIS layer's cache rows 0..=pos.
    let mut attn_out = vec![0.0f32; n_attn];
    for head in 0..PRED_N_HEAD {
        let kv_head = head / group_size;
        let q_off = head * PRED_HEAD_DIM;
        let attn_off = head * PRED_HEAD_DIM;
        let mut query = vec![0; PRED_HEAD_DIM];
        f32_slice_to_f16(&q[q_off..q_off + PRED_HEAD_DIM], &mut query);
        let mut accumulator = vec![0; PRED_HEAD_DIM];
        let mut sum = 0.0f32;
        let mut max = f32::NEG_INFINITY;
        for j in 0..=pos {
            let k_off = j * cache_stride + kv_head * PRED_HEAD_DIM;
            let k_row = &k_cache[layer_idx][k_off..k_off + PRED_HEAD_DIM];
            let score = dot_f16(&query, k_row, PRED_HEAD_DIM) * kq_scale;
            let mut rescale = 1.0f32;
            let mut weight = 1.0f32;
            if score > max {
                rescale = (max - score).exp();
                max = score;
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    scale_f16_inplace(&mut accumulator, rescale);
                }
                #[cfg(not(target_arch = "aarch64"))]
                scale_f16_inplace(&mut accumulator, rescale);
            } else {
                weight = (score - max).exp();
            }
            let v_off = j * cache_stride + kv_head * PRED_HEAD_DIM;
            #[cfg(target_arch = "aarch64")]
            unsafe {
                mad_f16_inplace(
                    &mut accumulator,
                    &v_cache[layer_idx][v_off..v_off + PRED_HEAD_DIM],
                    weight,
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            mad_f16_inplace(
                &mut accumulator,
                &v_cache[layer_idx][v_off..v_off + PRED_HEAD_DIM],
                weight,
            );
            sum = sum.mul_add(rescale, weight);
        }
        let output = &mut attn_out[attn_off..attn_off + PRED_HEAD_DIM];
        for (output, &value) in output.iter_mut().zip(&accumulator) {
            *output = f16_to_f32(value);
        }
        vec_scale_f32(output, if sum == 0.0 { 0.0 } else { sum.recip() });
    }
    // attn_out projection + residual.
    let mut attn_proj = vec![0.0f32; PRED_N_EMBD];
    let q8_blocks = (n_attn + 31) / 32;
    let mut q8b = vec![0u8; n_attn];
    let mut sb = vec![0.0f32; q8_blocks];
    quantize_q8_0_into(&attn_out, n_attn, &mut q8b, &mut sb);
    predictor_matmul(
        layer.wo,
        &q8b,
        &sb,
        &mut attn_proj,
        n_attn,
        PRED_N_EMBD,
        0,
        1,
    );
    for (h, p) in hidden.iter_mut().zip(attn_proj.iter()) {
        *h += *p;
    }
    // ln2 -> ffn -> residual.
    rms_norm(hidden, &layer.ln2, &mut normed, eps);
    let mut q8b2 = vec![0u8; PRED_N_EMBD];
    let mut sb2 = vec![0.0f32; blocks];
    quantize_q8_0_into(&normed, PRED_N_EMBD, &mut q8b2, &mut sb2);
    let mut gate = vec![0.0f32; PRED_N_FF];
    let mut up = vec![0.0f32; PRED_N_FF];
    predictor_matmul(
        layer.w_gate,
        &q8b2,
        &sb2,
        &mut gate,
        PRED_N_EMBD,
        PRED_N_FF,
        0,
        1,
    );
    predictor_matmul(
        layer.w_up,
        &q8b2,
        &sb2,
        &mut up,
        PRED_N_EMBD,
        PRED_N_FF,
        0,
        1,
    );
    let q8b3 = (PRED_N_FF + 31) / 32;
    silu_mul_approx_inplace(&gate, &mut up);
    let mut q8b3_buf = vec![0u8; PRED_N_FF];
    let mut sb3 = vec![0.0f32; q8b3];
    quantize_q8_0_into(&up, PRED_N_FF, &mut q8b3_buf, &mut sb3);
    let mut down = vec![0.0f32; PRED_N_EMBD];
    predictor_matmul(
        layer.w_down,
        &q8b3_buf,
        &sb3,
        &mut down,
        PRED_N_FF,
        PRED_N_EMBD,
        0,
        1,
    );
    for (h, d) in hidden.iter_mut().zip(down.iter()) {
        *h += *d;
    }
    Ok(())
}

fn sample_at_head<R: Rng + ?Sized>(
    head_w: &[u8],
    hidden: &[f32],
    output_norm: &[f32],
    eps: f32,
    level: usize,
    top_k: usize,
    rng: &mut R,
) -> Result<u32, String> {
    let n_embd = PRED_N_EMBD;
    let vocab = PRED_VOCAB;
    let blocks_per_row = n_embd / 32;
    let bytes_per_row = blocks_per_row * 34;
    let bytes_per_level = bytes_per_row * vocab;
    let level_off = level * bytes_per_level;
    let expected = bytes_per_level;
    if head_w.len() < level_off + expected {
        return Err(format!(
            "sample_at_head: head_w length {} < required {}",
            head_w.len(),
            level_off + expected
        ));
    }
    let mut normed = vec![0.0; n_embd];
    rms_norm(hidden, output_norm, &mut normed, eps);
    let blocks = n_embd.div_ceil(32);
    let mut input_q8 = vec![0; n_embd];
    let mut input_scales = vec![0.0; blocks];
    quantize_q8_0_into(&normed, n_embd, &mut input_q8, &mut input_scales);
    let mut logits = vec![0.0f32; vocab];
    predictor_matmul(
        &head_w[level_off..level_off + bytes_per_level],
        &input_q8,
        &input_scales,
        &mut logits,
        n_embd,
        vocab,
        0,
        1,
    );
    sample_top_k_with_draw(&logits, top_k, rng.gen())
}

/// Top-K sampling with an injected draw for deterministic tests and one RNG
/// stream shared by the complete request.
fn sample_top_k_with_draw(logits: &[f32], k: usize, draw: f32) -> Result<u32, String> {
    if logits.is_empty() || k == 0 {
        return Err("top-K sampling requires non-empty logits and k > 0".into());
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("top-K sampling requires finite logits".into());
    }
    if !(0.0..1.0).contains(&draw) || !draw.is_finite() {
        return Err(format!("top-K sampling draw {draw} is outside [0, 1)"));
    }
    let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    top.truncate(k.min(top.len()));
    let max_val = top
        .iter()
        .map(|&(_, v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, v) in top.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err("top-K probability sum is not finite and positive".into());
    }
    for (_, probability) in &mut top {
        *probability /= sum;
    }
    let mut cumulative = 0.0f32;
    for &(index, probability) in &top {
        cumulative += probability;
        if cumulative > draw {
            return u32::try_from(index).map_err(|_| "sampled code does not fit u32".into());
        }
    }
    u32::try_from(top.last().expect("non-empty top-K").0)
        .map_err(|_| "sampled code does not fit u32".into())
}

fn sum_frame_embeddings<'a, I>(
    semantic: &[f32],
    acoustic: I,
    dim: usize,
) -> Result<Vec<f32>, String>
where
    I: IntoIterator<Item = &'a [f32]>,
{
    if semantic.len() != dim || semantic.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "semantic feedback embedding must contain {dim} finite values"
        ));
    }
    let mut sum = semantic.to_vec();
    let mut levels = 0;
    for (level, embedding) in acoustic.into_iter().enumerate() {
        if embedding.len() != dim || embedding.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "acoustic feedback embedding {level} must contain {dim} finite values"
            ));
        }
        for (sum, value) in sum.iter_mut().zip(embedding) {
            *sum += *value;
        }
        levels += 1;
    }
    if levels != PRED_ACOUSTIC_LEVELS {
        return Err(format!(
            "feedback contains {levels} acoustic levels; expected {PRED_ACOUSTIC_LEVELS}"
        ));
    }
    if sum.iter().any(|value| !value.is_finite()) {
        return Err("feedback embedding sum contains non-finite values".into());
    }
    Ok(sum)
}

fn load_q8_lookup_f32(
    source: &dyn TensorSource,
    name: &str,
    expected_dims: &[u64],
    rows: usize,
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
            info.dims
        ));
    }
    if info.ggml_type != GGMLType::Q8_0 {
        return Err(format!("{name}: type {:?} not Q8_0", info.ggml_type));
    }
    let blocks_per_row = dim / 32;
    let bytes_per_row = blocks_per_row * 34;
    let expected_bytes = rows * bytes_per_row;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "{name}: bytes {} != expected {expected_bytes}",
            bytes.len()
        ));
    }
    let mut out = vec![0.0f32; rows * dim];
    for row in 0..rows {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_draw_selects_a_reproducible_top_k_code() {
        let logits = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(sample_top_k_with_draw(&logits, 2, 0.0).unwrap(), 3);
        assert_eq!(sample_top_k_with_draw(&logits, 2, 0.999_999).unwrap(), 2);
    }

    #[test]
    fn feedback_sum_keeps_semantic_and_all_fifteen_acoustic_levels() {
        let code0 = vec![1.0, 2.0];
        let acoustic: Vec<Vec<f32>> = (1..=15).map(|level| vec![level as f32, 1.0]).collect();
        let sum = sum_frame_embeddings(&code0, acoustic.iter().map(Vec::as_slice), 2).unwrap();
        assert_eq!(sum, vec![121.0, 17.0]);
    }
}
