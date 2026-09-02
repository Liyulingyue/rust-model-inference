//! Patch encoder (`VAESemanticEncoder`): maps 4×128 latent patches to one
//! 1536-dim LLM embedding row.
//!
//! Pipeline per reference `encoder_inference.py`:
//!   raw [4,128] → transpose → causal Conv1d(k2, s2, left pad 1) with carried
//!   tail → [2,128] → in_proj Linear(128→1024) → [2,1024] → 24-layer
//!   transformer with KV cache (rotary θ=1e4; pairs (d, d+32); q/k norms
//!   are weight-free RMSNorm so they only divide by the RMS) → concat the two
//!   tokens → out_proj Linear(2048→1536) → [1,1536].

use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::dots::config::DotsTtsConfig;
use crate::ops::{dot_f32, rms_norm, silu};

pub(crate) fn load_f16_f32(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims || !matches!(info.ggml_type, GGMLType::F16 | GGMLType::F32) {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {dims:?} F16/F32",
            info.dims, info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    Ok(match info.ggml_type {
        GGMLType::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| crate::ops::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect(),
        GGMLType::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        _ => unreachable!(),
    })
}

pub(crate) fn linear_forward(
    weight: &[f32],
    bias: Option<&[f32]>,
    input: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) {
    for row in 0..out_dim {
        let mut sum = bias.map_or(0.0, |b| b[row]);
        let w = &weight[row * in_dim..(row + 1) * in_dim];
        for (wi, &xi) in w.iter().zip(input.iter()) {
            sum = wi.mul_add(xi, sum);
        }
        output[row] = sum;
    }
}

/// Reference rotary: freqs repeat across the two halves, so element `d` is
/// paired with `d+half` and rotated by `pos / theta^(2*(d%half)/head_dim)`.
pub(crate) fn dots_rotary(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half = head_dim / 2;
    for i in 0..half {
        let angle = pos as f32 * freq_base.powf(-2.0 * i as f32 / head_dim as f32);
        let (cos_a, sin_a) = (angle.cos(), angle.sin());
        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0.mul_add(cos_a, x1 * -sin_a);
        x[i + half] = x0.mul_add(sin_a, x1 * cos_a);
    }
}

const ENC_HEADS: usize = 16;
const ENC_HEAD_DIM: usize = 64;
const ENC_HIDDEN: usize = 1024;
const ENC_FFN: usize = 4096;
const ENC_ROPE_THETA: f32 = 10_000.0;
const ENC_NORM_EPS: f32 = 1e-5;
const ENC_TOKENS_PER_PATCH: usize = 2; // patch_size 4 / in_ds_rate 2

pub(crate) struct PatchLayerWeights {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) o: Vec<f32>,
    pub(crate) o_bias: Vec<f32>,
    pub(crate) fc1: Vec<f32>,
    pub(crate) fc1_bias: Vec<f32>,
    pub(crate) fc2: Vec<f32>,
    pub(crate) fc2_bias: Vec<f32>,
}

pub struct PatchEncoder {
    pub ds_proj: Vec<f32>,
    pub ds_bias: Vec<f32>,
    pub in_proj: Vec<f32>,
    pub in_bias: Vec<f32>,
    pub out_proj: Vec<f32>,
    pub out_bias: Vec<f32>,
    pub(crate) layers: Vec<PatchLayerWeights>,
    pub config: DotsTtsConfig,
}

pub struct PatchEncoderState {
    pub conv_tail: Vec<f32>, // [128] last input frame (left-pad slot)
    pub k_cache: Vec<Vec<f32>>,
    pub v_cache: Vec<Vec<f32>>,
    pub seq_len: usize,
}

impl PatchEncoderState {
    fn new(layers: usize, capacity_tokens: usize) -> Self {
        let cap = capacity_tokens * ENC_HIDDEN;
        Self {
            conv_tail: vec![0.0; 128],
            k_cache: vec![vec![0.0; cap]; layers],
            v_cache: vec![vec![0.0; cap]; layers],
            seq_len: 0,
        }
    }
}

impl PatchEncoder {
    pub fn from_source(source: &dyn TensorSource, config: DotsTtsConfig) -> Result<Self, String> {
        let d = config.latent_dim as u64;
        let enc_hid = ENC_HIDDEN as u64;
        let ds = load_f16_f32(
            source,
            "dotstts.patch_encoder.ds_proj.weight",
            &[2, d, d],
        )?;
        let ds_bias = load_f16_f32(source, "dotstts.patch_encoder.ds_proj.bias", &[d])?;
        let in_proj = load_f16_f32(
            source,
            "dotstts.patch_encoder.in_proj.weight",
            &[d, enc_hid],
        )?;
        let in_bias = load_f16_f32(source, "dotstts.patch_encoder.in_proj.bias", &[enc_hid])?;
        let out_proj = load_f16_f32(
            source,
            "dotstts.patch_encoder.out_proj.weight",
            &[(ENC_HIDDEN * 2) as u64, config.llm_hidden_size as u64],
        )?;
        let out_bias = load_f16_f32(
            source,
            "dotstts.patch_encoder.out_proj.bias",
            &[config.llm_hidden_size as u64],
        )?;
        let mut layers = Vec::with_capacity(config.patch_encoder_layers);
        for layer in 0..config.patch_encoder_layers {
            let name = |suffix: &str| {
                format!("dotstts.patch_encoder.encoder.layers.{layer}.{suffix}")
            };
            let hid = [ENC_HIDDEN as u64];
            let hid2 = [ENC_HIDDEN as u64; 2];
            layers.push(PatchLayerWeights {
                attn_norm: load_f16_f32(source, &name("attn_norm.weight"), &hid)?,
                ffn_norm: load_f16_f32(source, &name("ffn_norm.weight"), &hid)?,
                q: load_f16_f32(source, &name("attn_q.weight"), &hid2)?,
                k: load_f16_f32(source, &name("attn_k.weight"), &hid2)?,
                v: load_f16_f32(source, &name("attn_v.weight"), &hid2)?,
                o: load_f16_f32(source, &name("attn_output.weight"), &hid2)?,
                o_bias: load_f16_f32(source, &name("attn_output.bias"), &hid)?,
                fc1: load_f16_f32(
                    source,
                    &name("ffn_fc1.weight"),
                    &[ENC_HIDDEN as u64, ENC_FFN as u64],
                )?,
                fc1_bias: load_f16_f32(source, &name("ffn_fc1.bias"), &[ENC_FFN as u64])?,
                fc2: load_f16_f32(
                    source,
                    &name("ffn_fc2.weight"),
                    &[ENC_FFN as u64, ENC_HIDDEN as u64],
                )?,
                fc2_bias: load_f16_f32(source, &name("ffn_fc2.bias"), &[ENC_HIDDEN as u64])?,
            });
        }
        Ok(Self {
            ds_proj: ds,
            ds_bias,
            in_proj,
            in_bias,
            out_proj,
            out_bias,
            layers,
            config,
        })
    }

    /// Fresh streaming state sized for `capacity_tokens` encoder tokens
    /// (2 tokens per patch).
    pub fn new_state(&self, capacity_tokens: usize) -> PatchEncoderState {
        PatchEncoderState::new(self.layers.len(), capacity_tokens)
    }

    /// Encode all patches of a prompt at once (reference `prefill`).
    /// `latents` is `[patches*4, 128]` in raw latent space.
    pub fn prefill(
        &self,
        latents: &[f32],
        state: &mut PatchEncoderState,
    ) -> Result<Vec<f32>, String> {
        if latents.len() % (4 * 128) != 0 {
            return Err("patch encoder prefill input is not patch-sized".into());
        }
        let patches = latents.len() / (4 * 128);
        if patches == 0 {
            return Ok(Vec::new());
        }
        let tokens = self.downsample(latents, state)?;
        let hidden = self.transformer(&tokens, state.seq_len, state)?;
        let embeddings = self.project(&hidden, patches)?;
        state.seq_len += tokens.len() / ENC_HIDDEN;
        Ok(embeddings)
    }

    /// Encode one generated patch `[4, 128]` (raw latent space) against state.
    pub fn encode_patch(
        &self,
        patch: &[f32],
        state: &mut PatchEncoderState,
    ) -> Result<Vec<f32>, String> {
        if patch.len() != 4 * 128 {
            return Err("patch encoder expects a 4×128 latent patch".into());
        }
        let tokens = self.downsample(patch, state)?;
        let hidden = self.transformer(&tokens, state.seq_len, state)?;
        let embeddings = self.project(&hidden, 1)?;
        state.seq_len += ENC_TOKENS_PER_PATCH;
        Ok(embeddings)
    }

    /// Causal Conv1d(k2, s2, left pad 1) downsample + in_proj.
    /// Input frames `[frames, 128]` (raw latent space); output `[tokens, 1024]`.
    fn downsample(
        &self,
        frames: &[f32],
        state: &mut PatchEncoderState,
    ) -> Result<Vec<f32>, String> {
        let n_frames = frames.len() / 128;
        let n_tokens = n_frames / 2;
        let mut tokens = vec![0.0f32; n_tokens * ENC_HIDDEN];
        for token in 0..n_tokens {
            let mut projected = vec![0.0f32; 128];
            for out in 0..128 {
                let mut sum = self.ds_bias[out];
                // two taps: input positions 2*token + {0, 1} over the padded stream
                for tap in 0..2 {
                    let in_pos = 2 * token + tap;
                    let frame = if in_pos == 0 {
                        &state.conv_tail
                    } else {
                        &frames[(in_pos - 1) * 128..in_pos * 128]
                    };
                    for inp in 0..128 {
                        let w = self.ds_proj[tap * 128 * 128 + inp * 128 + out];
                        sum = w.mul_add(frame[inp], sum);
                    }
                }
                projected[out] = sum;
            }
            linear_forward(
                &self.in_proj,
                Some(&self.in_bias),
                &projected,
                128,
                ENC_HIDDEN,
                &mut tokens[token * ENC_HIDDEN..(token + 1) * ENC_HIDDEN],
            );
        }
        // carry the last 1 frame as the next left-pad slot
        state.conv_tail.copy_from_slice(&frames[(n_frames - 1) * 128..]);
        Ok(tokens)
    }

    /// Run the 24-layer transformer with KV caching. `start` is the absolute
    /// position of the first new token; keys before `start` come from state.
    fn transformer(
        &self,
        tokens: &[f32],
        start: usize,
        state: &mut PatchEncoderState,
    ) -> Result<Vec<f32>, String> {
        let t = tokens.len() / ENC_HIDDEN;
        let mut x = tokens.to_vec();
        let mut h = vec![0.0f32; ENC_HIDDEN];
        let mut q = vec![0.0f32; t * ENC_HIDDEN];
        let mut k = vec![0.0f32; t * ENC_HIDDEN];
        let mut v = vec![0.0f32; t * ENC_HIDDEN];
        let mut attn = vec![0.0f32; t * ENC_HIDDEN];
        let mut out = vec![0.0f32; ENC_HIDDEN];
        let mut ffn = vec![0.0f32; ENC_FFN];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // projections
            for i in 0..t {
                rms_norm(
                    &x[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN],
                    &layer.attn_norm,
                    &mut h,
                    ENC_NORM_EPS,
                );
                linear_forward(
                    &layer.q,
                    None,
                    &h,
                    ENC_HIDDEN,
                    ENC_HIDDEN,
                    &mut q[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN],
                );
                linear_forward(
                    &layer.k,
                    None,
                    &h,
                    ENC_HIDDEN,
                    ENC_HIDDEN,
                    &mut k[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN],
                );
                linear_forward(
                    &layer.v,
                    None,
                    &h,
                    ENC_HIDDEN,
                    ENC_HIDDEN,
                    &mut v[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN],
                );
            }
            // rotary (per token position) then weight-free RMS q/k norms
            for i in 0..t {
                let pos = start + i;
                for head in 0..ENC_HEADS {
                    dots_rotary(
                        &mut q[i * ENC_HIDDEN + head * ENC_HEAD_DIM
                            ..i * ENC_HIDDEN + (head + 1) * ENC_HEAD_DIM],
                        pos,
                        ENC_HEAD_DIM,
                        ENC_ROPE_THETA,
                    );
                    dots_rotary(
                        &mut k[i * ENC_HIDDEN + head * ENC_HEAD_DIM
                            ..i * ENC_HIDDEN + (head + 1) * ENC_HEAD_DIM],
                        pos,
                        ENC_HEAD_DIM,
                        ENC_ROPE_THETA,
                    );
                }
            }
            for chunk in q.chunks_exact_mut(ENC_HEAD_DIM) {
                rms_norm_ones(chunk, ENC_NORM_EPS);
            }
            for chunk in k.chunks_exact_mut(ENC_HEAD_DIM) {
                rms_norm_ones(chunk, ENC_NORM_EPS);
            }
            // write new K/V into the cache
            {
                let end = start + t;
                for i in 0..t {
                    state.k_cache[layer_idx]
                        [(start + i) * ENC_HIDDEN..(start + i + 1) * ENC_HIDDEN]
                        .copy_from_slice(&k[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN]);
                    state.v_cache[layer_idx]
                        [(start + i) * ENC_HIDDEN..(start + i + 1) * ENC_HIDDEN]
                        .copy_from_slice(&v[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN]);
                }
                let _ = end;
            }
            // attention: query i sees keys 0 .. start+i+1 (causal)
            let scale = 1.0 / (ENC_HEAD_DIM as f32).sqrt();
            attn.fill(0.0);
            for i in 0..t {
                let keys = start + i + 1;
                for head in 0..ENC_HEADS {
                    let qh = &q[i * ENC_HIDDEN + head * ENC_HEAD_DIM
                        ..i * ENC_HIDDEN + (head + 1) * ENC_HEAD_DIM];
                    // online softmax over keys
                    let mut acc = [0.0f32; ENC_HEAD_DIM];
                    let mut sum = 0.0f32;
                    let mut max = f32::NEG_INFINITY;
                    for key in 0..keys {
                        let koff = key * ENC_HIDDEN + head * ENC_HEAD_DIM;
                        let krow = &state.k_cache[layer_idx][koff..koff + ENC_HEAD_DIM];
                        let score = dot_f32(qh, krow, ENC_HEAD_DIM) * scale;
                        if score > max {
                            let rescale = (max - score).exp();
                            max = score;
                            for value in acc.iter_mut() {
                                *value *= rescale;
                            }
                            sum = sum.mul_add(rescale, 1.0);
                        } else {
                            let weight = (score - max).exp();
                            sum += weight;
                            let voff = key * ENC_HIDDEN + head * ENC_HEAD_DIM;
                            let vrow = &state.v_cache[layer_idx][voff..voff + ENC_HEAD_DIM];
                            for (value, &vv) in acc.iter_mut().zip(vrow.iter()) {
                                *value += vv * weight;
                            }
                        }
                    }
                    let recip = if sum == 0.0 { 0.0 } else { sum.recip() };
                    let dst = i * ENC_HIDDEN + head * ENC_HEAD_DIM;
                    for (slot, &value) in attn[dst..dst + ENC_HEAD_DIM]
                        .iter_mut()
                        .zip(acc.iter())
                    {
                        *slot = value * recip;
                    }
                }
            }
            // o_proj + residual + FFN
            for i in 0..t {
                linear_forward(
                    &layer.o,
                    Some(&layer.o_bias),
                    &attn[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN],
                    ENC_HIDDEN,
                    ENC_HIDDEN,
                    &mut out,
                );
                let xrow = &mut x[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN];
                for (xs, &o) in xrow.iter_mut().zip(out.iter()) {
                    *xs += o;
                }
                rms_norm(xrow, &layer.ffn_norm, &mut h, ENC_NORM_EPS);
                linear_forward(
                    &layer.fc1,
                    Some(&layer.fc1_bias),
                    &h,
                    ENC_HIDDEN,
                    ENC_FFN,
                    &mut ffn,
                );
                for value in ffn.iter_mut() {
                    *value = silu(*value);
                }
                linear_forward(
                    &layer.fc2,
                    Some(&layer.fc2_bias),
                    &ffn,
                    ENC_FFN,
                    ENC_HIDDEN,
                    &mut out,
                );
                for (xs, &o) in xrow.iter_mut().zip(out.iter()) {
                    *xs += o;
                }
            }
        }
        Ok(x)
    }

    /// Concat every two encoder tokens and project to the LLM width.
    fn project(&self, hidden: &[f32], patches: usize) -> Result<Vec<f32>, String> {
        let mut embeddings = vec![0.0f32; patches * self.config.llm_hidden_size];
        for p in 0..patches {
            let mut concat = vec![0.0f32; ENC_HIDDEN * 2];
            concat[..ENC_HIDDEN].copy_from_slice(
                &hidden[p * 2 * ENC_HIDDEN..p * 2 * ENC_HIDDEN + ENC_HIDDEN],
            );
            concat[ENC_HIDDEN..].copy_from_slice(
                &hidden[p * 2 * ENC_HIDDEN + ENC_HIDDEN..(p + 1) * 2 * ENC_HIDDEN],
            );
            linear_forward(
                &self.out_proj,
                Some(&self.out_bias),
                &concat,
                ENC_HIDDEN * 2,
                self.config.llm_hidden_size,
                &mut embeddings[p * self.config.llm_hidden_size
                    ..(p + 1) * self.config.llm_hidden_size],
            );
        }
        Ok(embeddings)
    }
}

/// RMSNorm with a weight of all ones (the checkpoint has no q/k norm weights).
fn rms_norm_ones(x: &mut [f32], eps: f32) {
    let mut mean_sq = 0.0f64;
    for &value in x.iter() {
        mean_sq += (value as f64) * (value as f64);
    }
    mean_sq /= x.len() as f64;
    let inv = 1.0 / (mean_sq + eps as f64).sqrt();
    for value in x.iter_mut() {
        *value = (*value as f64 * inv) as f32;
    }
}