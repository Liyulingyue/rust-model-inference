//! Patch encoder (`VAESemanticEncoder`): maps 4×128 latent patches to one
//! 1536-dim LLM embedding row.
//!
//! Pipeline per reference `encoder_inference.py`:
//!   raw [4,128] → transpose → causal Conv1d(k2, s2, left pad 1) with carried
//!   tail → [2,128] → in_proj Linear(128→1024) → [2,1024] → 24-layer
//!   transformer with KV cache (causal RMSNorm self-attention) → concat the two
//!   tokens → out_proj Linear(2048→1536) → [1,1536].

pub(crate) use super::weights::linear_forward;
use super::weights::load_weight;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::dots::config::DotsTtsConfig;
use crate::ops::dot_f32;
use crate::ops::kernel::Weight;
use crate::ops::math::torch28_exp;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use crate::ops::silu;

pub(crate) fn load_f16_f32(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims
        || !matches!(
            info.ggml_type,
            GGMLType::F16 | GGMLType::BF16 | GGMLType::F32
        )
    {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {dims:?} F16/BF16/F32",
            info.dims, info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let expected = info
        .checked_nbytes()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    Ok(match info.ggml_type {
        GGMLType::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| crate::ops::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect(),
        GGMLType::BF16 => bytes
            .chunks_exact(2)
            .map(|chunk| crate::core::tensor::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect(),
        GGMLType::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        _ => unreachable!(),
    })
}

pub(crate) fn linear_forward_transposed_input_then_bias(
    weight: &Weight<'_>,
    bias: &[f32],
    input: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), rows * in_dim);
    debug_assert_eq!((weight.n_in, weight.n_out), (in_dim, out_dim));
    debug_assert_eq!(bias.len(), out_dim);
    debug_assert_eq!(output.len(), rows * out_dim);
    let mut contiguous = vec![0.0; input.len()];
    for row in 0..rows {
        for feature in 0..in_dim {
            contiguous[row * in_dim + feature] = input[feature * rows + row];
        }
    }
    linear_forward(weight, Some(bias), &contiguous, in_dim, out_dim, output);
}

/// Rotary helper shared by DiT; PatchEncoder's pinned Oracle does not apply it.
pub(crate) fn dots_rotary(
    x: &mut [f32],
    positions: &[usize],
    n_heads: usize,
    head_dim: usize,
    freq_base: f32,
) {
    crate::ops::rope_neox_sleef_rows(x, positions, n_heads, head_dim, freq_base);
}

const ENC_HEADS: usize = 16;
const ENC_HEAD_DIM: usize = 64;
const ENC_HIDDEN: usize = 1024;
const ENC_FFN: usize = 4096;
// PyTorch nn.RMSNorm(eps=None) defaults to finfo(float32).eps.
const ENC_NORM_EPS: f32 = f32::EPSILON;
const ENC_TOKENS_PER_PATCH: usize = 2; // patch_size 4 / in_ds_rate 2

/// Match the float32 reduction used by Torch's CPU RMSNorm kernel.
///
/// The generic `ops::rms_norm` intentionally uses a higher precision f64
/// reduction for the other model families.  The pinned dots encoder is a
/// float32 Torch graph, whose ARM kernel uses four interleaved accumulators
/// and folds them in a fixed cascade.  Keep this contract local to the
/// encoder instead of changing the shared normalization primitive.
fn torch_rms_norm(input: &[f32], weight: &[f32], output: &mut [f32]) {
    torch_rms_norm_with_eps(input, weight, output, ENC_NORM_EPS);
}

pub(crate) fn torch_rms_norm_with_eps(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = input.len().min(weight.len()).min(output.len());
    debug_assert!(n > 0);
    let sum_sq = torch_sum_squares(&input[..n]);
    let mean_sq = sum_sq / n as f32;
    let scale = (mean_sq + eps).sqrt().recip();
    for i in 0..n {
        output[i] = input[i] * scale * weight[i];
    }
}

fn torch_sum_squares(values: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // For the encoder's 1024-wide rows this is the exact SumKernel
        // NEON-W4 cascade: 4 interleaved rows × 4 lanes, reduced in 16-row
        // groups before the final lane fold.
        if values.len() >= 64 && values.len() % 4 == 0 {
            let vec_size = values.len() / 4;
            let size_ilp = vec_size / 4;
            let level_power = 4usize;
            let level_step = 1usize << level_power;
            let level_mask = level_step - 1;
            let mut acc = [[[0.0f32; 4]; 4]; 4];
            let mut i = 0usize;
            while i + level_step <= size_ilp {
                for _ in 0..level_step {
                    for row in 0..4 {
                        for lane in 0..4 {
                            let value = values[i * 16 + row * 4 + lane];
                            acc[0][row][lane] += value * value;
                        }
                    }
                    i += 1;
                }
                for level in 1..4 {
                    for row in 0..4 {
                        for lane in 0..4 {
                            acc[level][row][lane] += acc[level - 1][row][lane];
                            acc[level - 1][row][lane] = 0.0;
                        }
                    }
                    let mask = level_mask << (level * level_power);
                    if (i & mask) != 0 {
                        break;
                    }
                }
            }
            while i < size_ilp {
                for row in 0..4 {
                    for lane in 0..4 {
                        let value = values[i * 16 + row * 4 + lane];
                        acc[0][row][lane] += value * value;
                    }
                }
                i += 1;
            }
            for level in 1..4 {
                for row in 0..4 {
                    for lane in 0..4 {
                        acc[0][row][lane] += acc[level][row][lane];
                    }
                }
            }
            for vec_i in size_ilp * 4..vec_size {
                for lane in 0..4 {
                    let value = values[vec_i * 4 + lane];
                    acc[0][0][lane] += value * value;
                }
            }
            for row in 1..4 {
                for lane in 0..4 {
                    acc[0][0][lane] += acc[0][row][lane];
                }
            }
            let mut total = 0.0f32;
            for value in &values[vec_size * 4..] {
                total += *value * *value;
            }
            for lane in 0..4 {
                total += acc[0][0][lane];
            }
            return total;
        }
    }
    let mut total = 0.0f32;
    for &value in values {
        total += value * value;
    }
    total
}

pub(crate) struct PatchLayerWeights<'a> {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) q: Weight<'a>,
    pub(crate) k: Weight<'a>,
    pub(crate) v: Weight<'a>,
    pub(crate) o: Weight<'a>,
    pub(crate) o_bias: Vec<f32>,
    pub(crate) fc1: Weight<'a>,
    pub(crate) fc1_bias: Vec<f32>,
    pub(crate) fc2: Weight<'a>,
    pub(crate) fc2_bias: Vec<f32>,
}

pub struct PatchEncoder<'a> {
    pub ds_proj: Weight<'a>,
    pub ds_bias: Vec<f32>,
    pub in_proj: Weight<'a>,
    pub in_bias: Vec<f32>,
    pub out_proj: Weight<'a>,
    pub out_bias: Vec<f32>,
    pub(crate) layers: Vec<PatchLayerWeights<'a>>,
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

fn online_attention_head(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    keys: usize,
    head_offset: usize,
) -> [f32; ENC_HEAD_DIM] {
    debug_assert_eq!(q.len(), ENC_HEAD_DIM);
    let scale = 1.0 / (ENC_HEAD_DIM as f32).sqrt();
    let mut acc = [0.0f32; ENC_HEAD_DIM];
    let mut sum = 0.0f32;
    let mut max = f32::NEG_INFINITY;
    for key in 0..keys {
        let offset = key * ENC_HIDDEN + head_offset;
        let score = dot_f32(q, &k_cache[offset..offset + ENC_HEAD_DIM], ENC_HEAD_DIM) * scale;
        let is_new_max = score > max;
        let weight = if is_new_max {
            let rescale = (max - score).exp();
            max = score;
            for value in &mut acc {
                *value *= rescale;
            }
            sum = sum.mul_add(rescale, 1.0);
            1.0
        } else {
            (score - max).exp()
        };
        let vrow = &v_cache[offset..offset + ENC_HEAD_DIM];
        for (value, &vv) in acc.iter_mut().zip(vrow) {
            *value = vv.mul_add(weight, *value);
        }
        if !is_new_max {
            sum += weight;
        }
    }
    let recip = if sum == 0.0 { 0.0 } else { sum.recip() };
    acc.map(|value| value * recip)
}

/// Causal attention through native dot products and an online softmax.
/// Query/output slices may start at a head offset within a strided row.
fn native_attention_head(
    q: &[f32],
    queries: usize,
    q_stride: usize,
    k_cache: &[f32],
    v_cache: &[f32],
    keys: usize,
    start: usize,
    head_offset: usize,
    output: &mut [f32],
    output_stride: usize,
) -> Result<(), String> {
    let row_len = |rows: usize, stride: usize, width: usize| {
        if rows == 0 {
            Some(0)
        } else {
            (rows - 1).checked_mul(stride)?.checked_add(width)
        }
    };
    if q_stride < ENC_HEAD_DIM
        || output_stride < ENC_HEAD_DIM
        || head_offset > ENC_HIDDEN - ENC_HEAD_DIM
        || head_offset % ENC_HEAD_DIM != 0
    {
        return Err("patch encoder attention has invalid strides or head offset".into());
    }
    let end = start
        .checked_add(queries)
        .ok_or("patch encoder attention causal range overflow")?;
    if end > keys {
        return Err("patch encoder attention queries exceed the cached keys".into());
    }
    let q_len = row_len(queries, q_stride, ENC_HEAD_DIM)
        .ok_or("patch encoder attention query size overflow")?;
    let output_len = row_len(queries, output_stride, ENC_HEAD_DIM)
        .ok_or("patch encoder attention output size overflow")?;
    let cache_len = row_len(keys, ENC_HIDDEN, head_offset + ENC_HEAD_DIM)
        .ok_or("patch encoder attention cache size overflow")?;
    if q.len() < q_len
        || output.len() < output_len
        || k_cache.len() < cache_len
        || v_cache.len() < cache_len
    {
        return Err("patch encoder attention has a truncated query, output, or cache".into());
    }
    for query in 0..queries {
        let acc = online_attention_head(
            &q[query * q_stride..query * q_stride + ENC_HEAD_DIM],
            k_cache,
            v_cache,
            start + query + 1,
            head_offset,
        );
        output[query * output_stride..query * output_stride + ENC_HEAD_DIM].copy_from_slice(&acc);
    }
    Ok(())
}
impl<'a> PatchEncoder<'a> {
    pub fn from_source(
        source: &'a dyn TensorSource,
        config: DotsTtsConfig,
    ) -> Result<Self, String> {
        let d = config.latent_dim as u64;
        let enc_hid = ENC_HIDDEN as u64;
        let ds = load_weight(source, "dotstts.patch_encoder.ds_proj.weight", &[2, d, d])?;
        let ds_bias = load_f16_f32(source, "dotstts.patch_encoder.ds_proj.bias", &[d])?;
        let in_proj = load_weight(
            source,
            "dotstts.patch_encoder.in_proj.weight",
            &[d, enc_hid],
        )?;
        let in_bias = load_f16_f32(source, "dotstts.patch_encoder.in_proj.bias", &[enc_hid])?;
        let out_proj = load_weight(
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
            let name =
                |suffix: &str| format!("dotstts.patch_encoder.encoder.layers.{layer}.{suffix}");
            let hid = [ENC_HIDDEN as u64];
            let hid2 = [ENC_HIDDEN as u64; 2];
            let q = load_weight(source, &name("attn_q.weight"), &hid2)?;
            let k = load_weight(source, &name("attn_k.weight"), &hid2)?;
            let v = load_weight(source, &name("attn_v.weight"), &hid2)?;
            layers.push(PatchLayerWeights {
                attn_norm: load_f16_f32(source, &name("attn_norm.weight"), &hid)?,
                ffn_norm: load_f16_f32(source, &name("ffn_norm.weight"), &hid)?,
                q,
                k,
                v,
                o: load_weight(source, &name("attn_output.weight"), &hid2)?,
                o_bias: load_f16_f32(source, &name("attn_output.bias"), &hid)?,
                fc1: load_weight(
                    source,
                    &name("ffn_fc1.weight"),
                    &[ENC_HIDDEN as u64, ENC_FFN as u64],
                )?,
                fc1_bias: load_f16_f32(source, &name("ffn_fc1.bias"), &[ENC_FFN as u64])?,
                fc2: load_weight(
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
        #[cfg(feature = "parity-trace")]
        for embedding in embeddings.chunks_exact(self.config.llm_hidden_size) {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.patch.embedding",
                None,
                &[1, embedding.len()],
                embedding,
            ));
        }
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
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.patch.embedding",
            None,
            &[1, embeddings.len()],
            &embeddings,
        ));
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

        {
            let projected = self.downsample_projection(frames, &state.conv_tail);
            linear_forward(
                &self.in_proj,
                Some(&self.in_bias),
                &projected,
                128,
                ENC_HIDDEN,
                &mut tokens,
            );
        }
        // carry the last 1 frame as the next left-pad slot
        state
            .conv_tail
            .copy_from_slice(&frames[(n_frames - 1) * 128..]);
        Ok(tokens)
    }

    fn downsample_projection(&self, frames: &[f32], conv_tail: &[f32]) -> Vec<f32> {
        let n_tokens = frames.len() / (2 * 128);
        let mut projected = vec![0.0f32; n_tokens * 128];
        let mut columns = vec![0.0; n_tokens * 256];
        for token in 0..n_tokens {
            for tap in 0..2 {
                let in_pos = 2 * token + tap;
                let frame = if in_pos == 0 {
                    conv_tail
                } else {
                    &frames[(in_pos - 1) * 128..in_pos * 128]
                };
                for inp in 0..128 {
                    columns[token * 256 + inp * 2 + tap] = frame[inp];
                }
            }
        }
        linear_forward(
            &self.ds_proj,
            Some(&self.ds_bias),
            &columns,
            256,
            128,
            &mut projected,
        );
        projected
    }

    /// Run the 24-layer transformer with KV caching. `start` is the absolute
    /// position of the first new token; keys before `start` come from state.
    fn transformer(
        &self,
        tokens: &[f32],
        start: usize,
        state: &mut PatchEncoderState,
    ) -> Result<Vec<f32>, String> {
        if tokens.len() % ENC_HIDDEN != 0 {
            return Err("patch encoder transformer input is not hidden-width aligned".into());
        }
        let t = tokens.len() / ENC_HIDDEN;
        if t == 0 {
            return Ok(Vec::new());
        }
        let end = start
            .checked_add(t)
            .ok_or_else(|| "patch encoder cache length overflow".to_string())?;
        let cache_len = end
            .checked_mul(ENC_HIDDEN)
            .ok_or_else(|| "patch encoder cache element count overflow".to_string())?;
        if state.k_cache.len() != self.layers.len() || state.v_cache.len() != self.layers.len() {
            return Err("patch encoder cache layer count does not match weights".into());
        }
        if state
            .k_cache
            .iter()
            .chain(&state.v_cache)
            .any(|cache| cache.len() < cache_len)
        {
            return Err("patch encoder cache is shorter than requested sequence".into());
        }
        let mut x = tokens.to_vec();
        let mut normed = vec![0.0f32; t * ENC_HIDDEN];
        let mut q = vec![0.0f32; t * ENC_HIDDEN];
        let mut k = vec![0.0f32; t * ENC_HIDDEN];
        let mut v = vec![0.0f32; t * ENC_HIDDEN];
        let mut attn = vec![0.0f32; t * ENC_HIDDEN];
        let mut out = vec![0.0f32; t * ENC_HIDDEN];
        let mut ffn = vec![0.0f32; t * ENC_FFN];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // projections
            for i in 0..t {
                torch_rms_norm(
                    &x[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN],
                    &layer.attn_norm,
                    &mut normed[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN],
                );
            }
            for (weight, output) in [(&layer.q, &mut q), (&layer.k, &mut k), (&layer.v, &mut v)] {
                linear_forward(weight, None, &normed, ENC_HIDDEN, ENC_HIDDEN, output);
            }
            #[cfg(test)]
            if std::env::var_os("DOTS_PATCH_DEBUG").is_some() && layer_idx == 0 {
                let bits = |values: &[f32]| {
                    values
                        .iter()
                        .take(8)
                        .map(|v| format!("{:#010x}", v.to_bits()))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                eprintln!("layer0 norm={}", bits(&normed[..ENC_HIDDEN]));
                eprintln!("layer0 q={}", bits(&q[..ENC_HIDDEN]));
                eprintln!("layer0 k={}", bits(&k[..ENC_HIDDEN]));
                eprintln!("layer0 v={}", bits(&v[..ENC_HIDDEN]));
            }
            // write new K/V into the cache
            {
                for i in 0..t {
                    state.k_cache[layer_idx]
                        [(start + i) * ENC_HIDDEN..(start + i + 1) * ENC_HIDDEN]
                        .copy_from_slice(&k[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN]);
                    state.v_cache[layer_idx]
                        [(start + i) * ENC_HIDDEN..(start + i + 1) * ENC_HIDDEN]
                        .copy_from_slice(&v[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN]);
                }
            }
            // attention: query i sees keys 0 .. start+i+1 (causal)
            attn.fill(0.0);

            for head in 0..ENC_HEADS {
                let offset = head * ENC_HEAD_DIM;
                native_attention_head(
                    &q[offset..],
                    t,
                    ENC_HIDDEN,
                    &state.k_cache[layer_idx],
                    &state.v_cache[layer_idx],
                    end,
                    start,
                    offset,
                    &mut attn[offset..],
                    ENC_HIDDEN,
                )?;
            }
            // Keep weights in their GGUF representation for every projection.
            linear_forward(
                &layer.o,
                Some(&layer.o_bias),
                &attn,
                ENC_HIDDEN,
                ENC_HIDDEN,
                &mut out,
            );
            for (xs, &o) in x.iter_mut().zip(&out) {
                *xs += o;
            }
            #[cfg(test)]
            if std::env::var_os("DOTS_PATCH_DEBUG").is_some() && layer_idx == 0 {
                let bits = |values: &[f32]| {
                    values
                        .iter()
                        .take(8)
                        .map(|v| format!("{:#010x}", v.to_bits()))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                eprintln!("layer0 attn={}", bits(&attn[..ENC_HIDDEN]));
                eprintln!("layer0 after_attn={}", bits(&x[..ENC_HIDDEN]));
            }
            for i in 0..t {
                torch_rms_norm(
                    &x[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN],
                    &layer.ffn_norm,
                    &mut normed[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN],
                );
            }
            linear_forward(
                &layer.fc1,
                Some(&layer.fc1_bias),
                &normed,
                ENC_HIDDEN,
                ENC_FFN,
                &mut ffn,
            );
            for value in &mut ffn {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                {
                    *value /= 1.0 + torch28_exp(-*value);
                }
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                {
                    *value = silu(*value);
                }
            }
            linear_forward(
                &layer.fc2,
                Some(&layer.fc2_bias),
                &ffn,
                ENC_FFN,
                ENC_HIDDEN,
                &mut out,
            );
            for (xs, &o) in x.iter_mut().zip(&out) {
                *xs += o;
            }
            #[cfg(test)]
            if std::env::var_os("DOTS_PATCH_DEBUG").is_some() && layer_idx == 0 {
                let bits = |values: &[f32]| {
                    values
                        .iter()
                        .take(8)
                        .map(|v| format!("{:#010x}", v.to_bits()))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                eprintln!("layer0 ffn_norm={}", bits(&normed[..ENC_HIDDEN]));
                eprintln!("layer0 ffn={}", bits(&ffn[..ENC_FFN]));
                eprintln!("layer0 after_ffn={}", bits(&x[..ENC_HIDDEN]));
            }
            #[cfg(test)]
            if std::env::var_os("DOTS_PATCH_DEBUG").is_some() {
                let bits = |values: &[f32]| {
                    values
                        .iter()
                        .take(4)
                        .map(|v| format!("{:#010x}", v.to_bits()))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                eprintln!("layer{} final={}", layer_idx, bits(&x[..ENC_HIDDEN]));
            }
        }
        Ok(x)
    }

    /// Concat every two encoder tokens and project to the LLM width.
    fn project(&self, hidden: &[f32], patches: usize) -> Result<Vec<f32>, String> {
        let mut embeddings = vec![0.0f32; patches * self.config.llm_hidden_size];
        debug_assert_eq!(hidden.len(), patches * ENC_HIDDEN * 2);
        linear_forward(
            &self.out_proj,
            Some(&self.out_bias),
            hidden,
            ENC_HIDDEN * 2,
            self.config.llm_hidden_size,
            &mut embeddings,
        );
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::TensorInfo;

    fn weight(values: Vec<f32>, n_in: usize, n_out: usize) -> Weight<'static> {
        assert_eq!(values.len(), n_in * n_out);
        let mut weight = Weight::from_quantized(crate::ops::kernel::QuantizedTensor::F32(values));
        weight.n_in = n_in;
        weight.n_out = n_out;
        weight
    }

    fn config(layers: usize) -> DotsTtsConfig {
        DotsTtsConfig {
            patch_size: 4,
            latent_dim: 128,
            hop_size: 1920,
            sample_rate: 48_000,
            fm_hidden_size: ENC_HIDDEN,
            llm_hidden_size: 1536,
            xvec_dim: 512,
            patch_encoder_layers: layers,
            dit_layers: 18,
            dit_heads: 16,
            default_nfe: 10,
            default_guidance: 1.2,
            default_speaker_scale: 1.5,
            default_eos_threshold: 0.8,
        }
    }

    #[derive(Default)]
    struct Source {
        metadata: std::collections::HashMap<String, crate::core::tensor::MetaValue>,
        infos: std::collections::HashMap<String, TensorInfo>,
        bytes: std::collections::HashMap<String, Vec<u8>>,
    }

    impl TensorSource for Source {
        fn metadata(&self, key: &str) -> Option<&crate::core::tensor::MetaValue> {
            self.metadata.get(key)
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.infos.get(name)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.bytes.get(name).map(Vec::as_slice)
        }
    }

    fn one_tensor_source(
        name: &str,
        ggml_type: GGMLType,
        dims: Vec<u64>,
        bytes: Vec<u8>,
    ) -> Source {
        Source {
            infos: std::collections::HashMap::from([(
                name.into(),
                TensorInfo {
                    name: name.into(),
                    dims,
                    ggml_type,
                    offset: 0,
                },
            )]),
            bytes: std::collections::HashMap::from([(name.into(), bytes)]),
            ..Source::default()
        }
    }

    #[test]
    fn component_loader_decodes_bf16_without_f16_rounding() {
        let source = one_tensor_source(
            "x",
            GGMLType::BF16,
            vec![2],
            [0x3f80u16.to_le_bytes(), 0xc020u16.to_le_bytes()].concat(),
        );
        assert_eq!(
            load_f16_f32(&source, "x", &[2])
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            vec![0x3f80_0000, 0xc020_0000],
        );
    }

    #[test]
    fn online_attention_includes_value_when_score_sets_new_maximum() {
        let q = [0.0f32; ENC_HEAD_DIM];
        let keys = vec![0.0f32; ENC_HIDDEN];
        let mut values = vec![0.0f32; ENC_HIDDEN];
        for (index, value) in values.iter_mut().enumerate() {
            *value = index as f32 + 1.0;
        }
        let actual = online_attention_head(&q, &keys, &values, 1, 0);
        assert_eq!(actual, values[..ENC_HEAD_DIM]);
    }

    #[test]
    fn online_attention_fuses_value_accumulation() {
        let mut q = [0.0f32; ENC_HEAD_DIM];
        q[0] = 8.0;
        let mut keys = vec![0.0f32; ENC_HIDDEN * 2];
        keys[ENC_HIDDEN] = -0.0011;
        let mut values = vec![0.0f32; ENC_HIDDEN * 2];
        values[0] = f32::from_bits(0x3e86_10e7);
        values[ENC_HIDDEN] = f32::from_bits(0x3e80_d91a);
        let actual = online_attention_head(&q, &keys, &values, 2, 0);
        assert_eq!(actual[0].to_bits(), 0x3e83_755f);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn transformer_silu_matches_torch28_neon_kernel() {
        let mut fc2 = vec![0.0f32; ENC_HIDDEN * ENC_FFN];
        fc2[28] = 1.0;
        let encoder = PatchEncoder {
            ds_proj: weight(Vec::new(), 0, 0),
            ds_bias: Vec::new(),
            in_proj: weight(Vec::new(), 0, 0),
            in_bias: Vec::new(),
            out_proj: weight(Vec::new(), 0, 0),
            out_bias: Vec::new(),
            layers: vec![PatchLayerWeights {
                attn_norm: vec![1.0; ENC_HIDDEN],
                ffn_norm: vec![1.0; ENC_HIDDEN],
                q: weight(vec![0.0; ENC_HIDDEN * ENC_HIDDEN], ENC_HIDDEN, ENC_HIDDEN),
                k: weight(vec![0.0; ENC_HIDDEN * ENC_HIDDEN], ENC_HIDDEN, ENC_HIDDEN),
                v: weight(vec![0.0; ENC_HIDDEN * ENC_HIDDEN], ENC_HIDDEN, ENC_HIDDEN),
                o: weight(vec![0.0; ENC_HIDDEN * ENC_HIDDEN], ENC_HIDDEN, ENC_HIDDEN),
                o_bias: vec![0.0; ENC_HIDDEN],
                fc1: weight(vec![0.0; ENC_FFN * ENC_HIDDEN], ENC_HIDDEN, ENC_FFN),
                fc1_bias: vec![f32::from_bits(0xbdbd_9888); ENC_FFN],
                fc2: weight(fc2, ENC_FFN, ENC_HIDDEN),
                fc2_bias: vec![0.0; ENC_HIDDEN],
            }],
            config: config(1),
        };
        let mut state = encoder.new_state(1);
        let actual = encoder
            .transformer(&vec![0.0; ENC_HIDDEN], 0, &mut state)
            .unwrap();
        assert_eq!(actual[0].to_bits(), 0xbd34_d37a);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires fixed PatchEncoder mmproj, input, and hidden sidecars"]
    fn production_patch_hidden_matches_pinned_oracle_bitwise() {
        use crate::{open_model_source, ComponentRole};
        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).expect("missing sidecar path"))
                .expect("read sidecar")
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let source = open_model_source(
            std::path::Path::new(&std::env::var_os("DOTS_PATCH_MMPROJ").unwrap()),
            ComponentRole::Mmproj,
        )
        .unwrap();
        let encoder = PatchEncoder::from_source(
            source.as_ref(),
            DotsTtsConfig::from_source(source.as_ref()).unwrap(),
        )
        .unwrap();
        let input = read("DOTS_PATCH_INPUT");
        let expected = read("DOTS_PATCH_HIDDEN");
        let mut state = encoder.new_state(input.len() / (2 * 128));
        let tokens = encoder.downsample(&input, &mut state).unwrap();
        let actual = encoder.transformer(&tokens, 0, &mut state).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "hidden[{index}] rust={:08x} oracle={:08x}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }

    fn flash_fixture_value(kind: u32, row: u32, column: u32) -> f32 {
        let mut mixed = kind.wrapping_mul(0x9e37_79b9)
            ^ row.wrapping_mul(0x85eb_ca6b)
            ^ column.wrapping_mul(0xc2b2_ae35);
        mixed ^= mixed >> 16;
        mixed = mixed.wrapping_mul(0x7feb_352d);
        mixed ^= mixed >> 15;
        ((mixed & 0x3fff) as i32 - 8192) as f32 / 2048.0
    }

    fn flash_fixture_qkv(tokens: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut q = vec![0.0f32; tokens * ENC_HEAD_DIM];
        let mut k = vec![0.0f32; tokens * ENC_HIDDEN];
        let mut v = vec![0.0f32; tokens * ENC_HIDDEN];
        for row in 0..tokens {
            for column in 0..ENC_HEAD_DIM {
                q[row * ENC_HEAD_DIM + column] = flash_fixture_value(1, row as u32, column as u32);
                k[row * ENC_HIDDEN + column] = flash_fixture_value(2, row as u32, column as u32);
                v[row * ENC_HIDDEN + column] = flash_fixture_value(3, row as u32, column as u32);
            }
        }
        (q, k, v)
    }

    // Independent f64 dot/softmax reference checks native operator accuracy;
    // this is not a Torch or llama.cpp bitwise parity assertion.
    fn attention_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        keys: usize,
        offset: usize,
    ) -> Vec<f64> {
        let scores: Vec<f64> = (0..keys)
            .map(|key| {
                q.iter()
                    .enumerate()
                    .map(|(column, &query)| {
                        query as f64 * k[key * ENC_HIDDEN + offset + column] as f64
                    })
                    .sum::<f64>()
                    / (ENC_HEAD_DIM as f64).sqrt()
            })
            .collect();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let probabilities: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let sum: f64 = probabilities.iter().sum();
        (0..ENC_HEAD_DIM)
            .map(|column| {
                probabilities
                    .iter()
                    .enumerate()
                    .map(|(key, probability)| {
                        probability * v[key * ENC_HIDDEN + offset + column] as f64
                    })
                    .sum::<f64>()
                    / sum
            })
            .collect()
    }

    #[test]
    fn native_attention_matches_f64_reference_for_72_tokens_and_scalar_tail() {
        for tokens in [5, 72] {
            let (q, k, v) = flash_fixture_qkv(tokens);
            let mut actual = vec![0.0; tokens * ENC_HEAD_DIM];
            native_attention_head(
                &q,
                tokens,
                ENC_HEAD_DIM,
                &k,
                &v,
                tokens,
                0,
                0,
                &mut actual,
                ENC_HEAD_DIM,
            )
            .unwrap();
            for row in 0..tokens {
                let expected = attention_reference(
                    &q[row * ENC_HEAD_DIM..(row + 1) * ENC_HEAD_DIM],
                    &k,
                    &v,
                    row + 1,
                    0,
                );
                for (column, &expected) in expected.iter().enumerate() {
                    let value = actual[row * ENC_HEAD_DIM + column] as f64;
                    assert!(
                        (value - expected).abs() < 4e-5 * expected.abs().max(1.0),
                        "tokens={tokens}, attention[{row}, {column}]: {value} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn native_attention_streaming_matches_full_with_strided_heads() {
        const TOKENS: usize = 72;
        let mut q = vec![0.0; TOKENS * ENC_HIDDEN];
        let mut k = vec![0.0; q.len()];
        let mut v = vec![0.0; q.len()];
        for row in 0..TOKENS {
            for column in 0..ENC_HIDDEN {
                q[row * ENC_HIDDEN + column] = flash_fixture_value(1, row as u32, column as u32);
                k[row * ENC_HIDDEN + column] = flash_fixture_value(2, row as u32, column as u32);
                v[row * ENC_HIDDEN + column] = flash_fixture_value(3, row as u32, column as u32);
            }
        }
        for head in [0, 7, ENC_HEADS - 1] {
            let offset = head * ENC_HEAD_DIM;
            let mut full = vec![f32::NAN; q.len()];
            let mut streamed = vec![f32::NAN; q.len()];
            native_attention_head(
                &q[offset..],
                TOKENS,
                ENC_HIDDEN,
                &k,
                &v,
                TOKENS,
                0,
                offset,
                &mut full[offset..],
                ENC_HIDDEN,
            )
            .unwrap();
            for (start, queries) in [(0, 5), (5, 27), (32, 40)] {
                native_attention_head(
                    &q[start * ENC_HIDDEN + offset..],
                    queries,
                    ENC_HIDDEN,
                    &k[..(start + queries) * ENC_HIDDEN],
                    &v[..(start + queries) * ENC_HIDDEN],
                    start + queries,
                    start,
                    offset,
                    &mut streamed[start * ENC_HIDDEN + offset..],
                    ENC_HIDDEN,
                )
                .unwrap();
            }
            assert_eq!(
                full.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                streamed.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            for row in 0..TOKENS {
                let expected = attention_reference(
                    &q[row * ENC_HIDDEN + offset..row * ENC_HIDDEN + offset + ENC_HEAD_DIM],
                    &k,
                    &v,
                    row + 1,
                    offset,
                );
                for column in 0..ENC_HIDDEN {
                    let value = full[row * ENC_HIDDEN + column];
                    if (offset..offset + ENC_HEAD_DIM).contains(&column) {
                        assert!(
                            (value as f64 - expected[column - offset]).abs()
                                < 4e-5 * expected[column - offset].abs().max(1.0)
                        );
                    } else {
                        assert!(value.is_nan(), "attention overwrote another head");
                    }
                }
            }
        }
    }

    #[test]
    fn native_attention_rejects_invalid_shapes_before_writing_output() {
        let (q, k, v) = flash_fixture_qkv(2);
        for (queries, stride, keys, start, head, output_stride) in [
            (2, 63, 2, 0, 0, 64),
            (2, 64, 2, 0, 0, 63),
            (2, 64, 2, 0, 1, 64),
            (2, 64, 2, 0, ENC_HIDDEN, 64),
            (2, 64, 2, 1, 0, 64),
            (2, 64, 3, 0, 0, 64),
            (2, 64, 2, usize::MAX, 0, 64),
            (2, usize::MAX, 2, 0, 0, 64),
            (2, 64, 2, 0, 0, usize::MAX),
            (3, 64, 3, 0, 0, 64),
        ] {
            let mut output = [123.0; 128];
            assert!(native_attention_head(
                &q,
                queries,
                stride,
                &k,
                &v,
                keys,
                start,
                head,
                &mut output,
                output_stride
            )
            .is_err());
            assert_eq!(output, [123.0; 128]);
        }
        native_attention_head(&[], 0, 64, &[], &[], 0, 0, 0, &mut [], 64).unwrap();
    }

    #[test]
    fn q8_downsample_preserves_channel_tap_order_and_streaming_carry() {
        let mut bytes = Vec::new();
        for output in 0..128 {
            for block in 0..8 {
                bytes.extend_from_slice(&0x3c00u16.to_le_bytes()); // Q8 scale 1
                for lane in 0..32 {
                    let column = block * 32 + lane;
                    let value = if column == output * 2 {
                        1
                    } else if column == ((output + 17) % 128) * 2 + 1 {
                        2
                    } else {
                        0
                    };
                    bytes.push(value);
                }
            }
        }
        let source = one_tensor_source("ds", GGMLType::Q8_0, vec![256, 128], bytes);
        let mut projection = vec![0.0; ENC_HIDDEN * 128];
        for output in 0..ENC_HIDDEN {
            projection[output * 128 + output % 128] = 1.0;
        }
        let encoder = PatchEncoder {
            ds_proj: load_weight(&source, "ds", &[2, 128, 128]).unwrap(),
            ds_bias: (0..128).map(|channel| channel as f32 / 4.0).collect(),
            in_proj: weight(projection, 128, ENC_HIDDEN),
            in_bias: vec![0.0; ENC_HIDDEN],
            out_proj: weight(Vec::new(), 0, 0),
            out_bias: Vec::new(),
            layers: Vec::new(),
            config: config(0),
        };
        assert_eq!(encoder.ds_proj.ggml_type, GGMLType::Q8_0);
        // Each activation block has max=127, so activation quantization is
        // exact here and the check isolates channel/tap addressing and carry.
        let frames: Vec<f32> = (0..8)
            .flat_map(|frame| {
                (0..128).map(move |channel| {
                    if channel % 16 == 15 {
                        127.0
                    } else {
                        (frame * 10 + channel % 13) as f32
                    }
                })
            })
            .collect();
        let mut state = encoder.new_state(4);
        let actual = encoder.downsample(&frames, &mut state).unwrap();
        let mut streamed_state = encoder.new_state(4);
        let mut streamed = encoder
            .downsample(&frames[..4 * 128], &mut streamed_state)
            .unwrap();
        assert_eq!(streamed_state.conv_tail, frames[3 * 128..4 * 128]);
        streamed.extend(
            encoder
                .downsample(&frames[4 * 128..], &mut streamed_state)
                .unwrap(),
        );
        assert_eq!(actual, streamed);
        assert_eq!(state.conv_tail, frames[7 * 128..]);
        assert_eq!(streamed_state.conv_tail, state.conv_tail);
        for token in 0..4 {
            for output in 0..ENC_HIDDEN {
                let channel = output % 128;
                let previous = if token == 0 {
                    0.0
                } else {
                    frames[(token * 2 - 1) * 128 + channel]
                };
                let current = frames[token * 2 * 128 + (channel + 17) % 128];
                let expected = previous + 2.0 * current + channel as f32 / 4.0;
                assert_eq!(
                    actual[token * ENC_HIDDEN + output],
                    expected,
                    "token={token}, output={output}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires fixed Torch/C layer-0 Q/K/V and attention sidecars"]
    fn production_flash_attention_matches_pinned_layer0_sidecar_bitwise() {
        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).expect("missing sidecar path"))
                .expect("read sidecar")
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let q = read("DOTS_PATCH_Q");
        let k = read("DOTS_PATCH_K");
        let v = read("DOTS_PATCH_V");
        let expected = read("DOTS_PATCH_ATTENTION");
        assert_eq!(q.len(), 72 * ENC_HIDDEN);
        assert_eq!(k.len(), q.len());
        assert_eq!(v.len(), q.len());
        assert_eq!(expected.len(), q.len());
        let mut actual = vec![0.0f32; q.len()];
        for head in 0..ENC_HEADS {
            let offset = head * ENC_HEAD_DIM;
            native_attention_head(
                &q[offset..],
                72,
                ENC_HIDDEN,
                &k,
                &v,
                72,
                0,
                offset,
                &mut actual[offset..],
                ENC_HIDDEN,
            )
            .unwrap();
        }
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "attention[{index}] rust={:08x} oracle={:08x}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    fn patch_encoder_rms_norm_uses_torch_default_epsilon() {
        let input = [1.0f32, 2.0];
        let weight = [1.0f32; 2];
        let mut output = [0.0f32; 2];
        torch_rms_norm(&input, &weight, &mut output);
        assert_eq!(output.map(f32::to_bits), [0x3f21_e89b, 0x3fa1_e89b]);
    }

    #[test]
    fn patch_encoder_rms_norm_matches_torch_f32_reduction() {
        let input: Vec<f32> = (0..ENC_HIDDEN)
            .map(|i| ((i * 37 % 1009) as f32 - 500.0) / 97.0)
            .collect();
        let weight: Vec<f32> = (0..ENC_HIDDEN)
            .map(|i| 1.0 + ((i % 7) as f32 - 3.0) * 0.01)
            .collect();
        let mut output = vec![0.0f32; ENC_HIDDEN];
        torch_rms_norm(&input, &weight, &mut output);
        let expected = [
            0xbfd5_22be,
            0xbfc7_65f9,
            0xbfb9_55f6,
            0xbfaa_f2b1,
            0xbf9c_3c2c,
            0xbf8d_3269,
            0xbf7b_aac9,
            0xbf4d_767c,
        ];
        assert_eq!(
            output[..expected.len()]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    #[ignore]
    fn print_patch_encoder_first_layer_probe() {
        use crate::{open_model_source, ComponentRole};
        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let source = open_model_source(
            std::path::Path::new(&std::env::var_os("DOTS_PATCH_MMPROJ").unwrap()),
            ComponentRole::Mmproj,
        )
        .unwrap();
        let config = DotsTtsConfig::from_source(source.as_ref()).unwrap();
        let encoder = PatchEncoder::from_source(source.as_ref(), config).unwrap();
        let input = read("DOTS_PATCH_INPUT");
        let mut state = encoder.new_state(input.len() / (4 * 128) * 2 + 8);
        let tokens = encoder.downsample(&input, &mut state).unwrap();
        let layer = &encoder.layers[0];
        let mut normed = vec![0.0f32; ENC_HIDDEN];
        torch_rms_norm(&tokens[..ENC_HIDDEN], &layer.attn_norm, &mut normed);
        let mut q = vec![0.0f32; ENC_HIDDEN];
        let mut k = vec![0.0f32; ENC_HIDDEN];
        let mut v = vec![0.0f32; ENC_HIDDEN];
        linear_forward(&layer.q, None, &normed, ENC_HIDDEN, ENC_HIDDEN, &mut q);
        linear_forward(&layer.k, None, &normed, ENC_HIDDEN, ENC_HIDDEN, &mut k);
        linear_forward(&layer.v, None, &normed, ENC_HIDDEN, ENC_HIDDEN, &mut v);
        let zero = vec![0.0f32; ENC_HIDDEN];
        let mut q_blas = vec![0.0f32; ENC_HIDDEN];
        linear_forward(
            &layer.q,
            Some(&zero),
            &normed,
            ENC_HIDDEN,
            ENC_HIDDEN,
            &mut q_blas,
        );
        let bits = |values: &[f32]| {
            values
                .iter()
                .take(8)
                .map(|v| format!("{:#010x}", v.to_bits()))
                .collect::<Vec<_>>()
                .join(",")
        };
        println!("token={}", bits(&tokens));
        println!("norm={}", bits(&normed));
        println!("q={}", bits(&q));
        println!("q_blas={}", bits(&q_blas));
        println!("k={}", bits(&k));
        println!("v={}", bits(&v));
    }

    #[test]
    #[ignore = "requires DOTS_PATCH_MMPROJ, DOTS_PATCH_RAW_INPUT, and DOTS_PATCH_DS_PROJ"]
    fn causal_downsample_projection_matches_pinned_torch_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let source = open_model_source(
            std::path::Path::new(&std::env::var_os("DOTS_PATCH_MMPROJ").unwrap()),
            ComponentRole::Mmproj,
        )
        .unwrap();
        let config = DotsTtsConfig::from_source(source.as_ref()).unwrap();
        let encoder = PatchEncoder::from_source(source.as_ref(), config).unwrap();
        let input = read("DOTS_PATCH_RAW_INPUT");
        let expected = read("DOTS_PATCH_DS_PROJ");

        let actual = encoder.downsample_projection(&input[..144 * 128], &[0.0; 128]);

        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "patch_encoder.ds_proj[{index}]"
            );
        }
    }

    #[test]
    #[ignore = "requires DOTS_PATCH_MMPROJ, DOTS_PATCH_RAW_INPUT, and DOTS_PATCH_IN_PROJ"]
    fn input_projection_matches_pinned_torch_bmm_then_bias_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let read = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let source = open_model_source(
            std::path::Path::new(&std::env::var_os("DOTS_PATCH_MMPROJ").unwrap()),
            ComponentRole::Mmproj,
        )
        .unwrap();
        let config = DotsTtsConfig::from_source(source.as_ref()).unwrap();
        let encoder = PatchEncoder::from_source(source.as_ref(), config).unwrap();
        let input = read("DOTS_PATCH_RAW_INPUT");
        let expected = read("DOTS_PATCH_IN_PROJ");
        let mut state = encoder.new_state(input.len() / (2 * 128));

        let actual = encoder.downsample(&input[..144 * 128], &mut state).unwrap();

        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "patch_encoder.in_proj[{index}]"
            );
        }
    }
}
