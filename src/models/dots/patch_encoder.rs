//! Patch encoder (`VAESemanticEncoder`): maps 4×128 latent patches to one
//! 1536-dim LLM embedding row.
//!
//! Pipeline per reference `encoder_inference.py`:
//!   raw [4,128] → transpose → causal Conv1d(k2, s2, left pad 1) with carried
//!   tail → [2,128] → in_proj Linear(128→1024) → [2,1024] → 24-layer
//!   transformer with KV cache (causal RMSNorm self-attention) → concat the two
//!   tokens → out_proj Linear(2048→1536) → [1,1536].

#[cfg(any(
    all(feature = "accelerate", target_os = "macos"),
    all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
))]
use super::blas::sys;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::dots::config::DotsTtsConfig;
use crate::models::dots::speaker::exp::torch28_exp;
use crate::ops::dot_f32;
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

pub(crate) fn linear_forward(
    weight: &[f32],
    bias: Option<&[f32]>,
    input: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) {
    #[cfg(any(
        all(feature = "accelerate", target_os = "macos"),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
    {
        let rows = output.len() / out_dim;
        debug_assert_eq!(input.len(), rows * in_dim);
        if let Some(bias) = bias {
            for row in output.chunks_exact_mut(out_dim) {
                row.copy_from_slice(bias);
            }
        } else {
            output.fill(0.0);
        }
        unsafe {
            sys::cblas_sgemm(
                101,
                111,
                112,
                rows as i32,
                out_dim as i32,
                in_dim as i32,
                1.0,
                input.as_ptr(),
                in_dim as i32,
                weight.as_ptr(),
                in_dim as i32,
                1.0,
                output.as_mut_ptr(),
                out_dim as i32,
            );
        }
        return;
    }
    #[cfg(not(any(
        all(feature = "accelerate", target_os = "macos"),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    )))]
    {
        let rows = output.len() / out_dim;
        debug_assert_eq!(input.len(), rows * in_dim);
        for input_row in 0..rows {
            for output_feature in 0..out_dim {
                let mut sum = bias.map_or(0.0, |b| b[output_feature]);
                let w = &weight[output_feature * in_dim..(output_feature + 1) * in_dim];
                for (wi, &xi) in w
                    .iter()
                    .zip(&input[input_row * in_dim..(input_row + 1) * in_dim])
                {
                    sum = wi.mul_add(xi, sum);
                }
                output[input_row * out_dim + output_feature] = sum;
            }
        }
    }
}

pub(crate) fn linear_forward_transposed_input_then_bias(
    weight: &[f32],
    bias: &[f32],
    input: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), rows * in_dim);
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(bias.len(), out_dim);
    debug_assert_eq!(output.len(), rows * out_dim);
    #[cfg(any(
        all(feature = "accelerate", target_os = "macos"),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
    unsafe {
        sys::cblas_sgemm(
            101,
            112,
            112,
            rows as i32,
            out_dim as i32,
            in_dim as i32,
            1.0,
            input.as_ptr(),
            rows as i32,
            weight.as_ptr(),
            in_dim as i32,
            0.0,
            output.as_mut_ptr(),
            out_dim as i32,
        );
    }
    #[cfg(not(any(
        all(feature = "accelerate", target_os = "macos"),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    )))]
    for row in 0..rows {
        for output_feature in 0..out_dim {
            let mut sum = 0.0f32;
            for input_feature in 0..in_dim {
                sum = weight[output_feature * in_dim + input_feature]
                    .mul_add(input[input_feature * rows + row], sum);
            }
            output[row * out_dim + output_feature] = sum;
        }
    }
    for row in output.chunks_exact_mut(out_dim) {
        for (value, &bias) in row.iter_mut().zip(bias) {
            *value += bias;
        }
    }
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

pub(crate) struct PatchLayerWeights {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) qkv: Vec<f32>,
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

/// Match Torch 2.8 CPU FlashAttention's macOS path: row-major Accelerate
/// SGEMMs around four-lane SLEEF softmax. The query/output slices may start
/// at a head offset; their row strides keep the full hidden width.
#[cfg(any(
    all(feature = "accelerate", target_os = "macos"),
    all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
))]
#[allow(clippy::too_many_arguments)]
fn torch28_flash_attention_head(
    q: &[f32],
    rows: usize,
    q_stride: usize,
    k_cache: &[f32],
    v_cache: &[f32],
    keys: usize,
    first_query: usize,
    head_offset: usize,
    output: &mut [f32],
    output_stride: usize,
) -> Result<(), String> {
    const QUERY_BLOCK: usize = 32;
    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANSPOSE: i32 = 111;
    const CBLAS_TRANSPOSE: i32 = 112;

    if rows == 0 {
        return Ok(());
    }
    let query_end = first_query
        .checked_add(rows)
        .ok_or_else(|| "patch attention query length overflow".to_string())?;
    if keys < query_end {
        return Err("patch attention cache does not cover all causal queries".into());
    }
    if q_stride < ENC_HEAD_DIM || output_stride < ENC_HEAD_DIM {
        return Err("patch attention head stride is narrower than its head".into());
    }
    if head_offset
        .checked_add(ENC_HEAD_DIM)
        .is_none_or(|end| end > ENC_HIDDEN)
    {
        return Err("patch attention head exceeds hidden width".into());
    }
    let row_span = rows
        .checked_sub(1)
        .and_then(|last| last.checked_mul(q_stride))
        .and_then(|start| start.checked_add(ENC_HEAD_DIM))
        .ok_or_else(|| "patch attention query span overflow".to_string())?;
    let output_span = rows
        .checked_sub(1)
        .and_then(|last| last.checked_mul(output_stride))
        .and_then(|start| start.checked_add(ENC_HEAD_DIM))
        .ok_or_else(|| "patch attention output span overflow".to_string())?;
    let cache_span = keys
        .checked_sub(1)
        .and_then(|last| last.checked_mul(ENC_HIDDEN))
        .and_then(|start| start.checked_add(head_offset + ENC_HEAD_DIM))
        .ok_or_else(|| "patch attention cache span overflow".to_string())?;
    if q.len() < row_span
        || output.len() < output_span
        || k_cache.len() < cache_span
        || v_cache.len() < cache_span
    {
        return Err("patch attention buffer is shorter than its declared shape".into());
    }

    let q_stride =
        i32::try_from(q_stride).map_err(|_| "patch attention query stride exceeds BLAS limits")?;
    let output_stride = i32::try_from(output_stride)
        .map_err(|_| "patch attention output stride exceeds BLAS limits")?;
    let keys_i32 =
        i32::try_from(keys).map_err(|_| "patch attention key count exceeds BLAS limits")?;
    let head_dim_i32 = i32::try_from(ENC_HEAD_DIM).expect("head dimension fits i32");
    let mut scores = vec![
        0.0f32;
        QUERY_BLOCK.checked_mul(keys).ok_or_else(|| {
            "patch attention score allocation overflows".to_string()
        })?
    ];
    let mut reciprocals = [0.0f32; QUERY_BLOCK];
    let scale = 1.0 / (ENC_HEAD_DIM as f32).sqrt();

    for block_start in (0..rows).step_by(QUERY_BLOCK) {
        let block_rows = (rows - block_start).min(QUERY_BLOCK);
        let block_rows_i32 = i32::try_from(block_rows).expect("query block fits i32");
        unsafe {
            sys::cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANSPOSE,
                CBLAS_TRANSPOSE,
                block_rows_i32,
                keys_i32,
                head_dim_i32,
                1.0,
                q.as_ptr().add(block_start * q_stride as usize),
                q_stride,
                k_cache.as_ptr().add(head_offset),
                ENC_HIDDEN as i32,
                0.0,
                scores.as_mut_ptr(),
                keys_i32,
            );
        }
        for row in 0..block_rows {
            let valid = first_query + block_start + row + 1;
            let row_scores = &mut scores[row * keys..(row + 1) * keys];
            let mut max4 = [f32::NEG_INFINITY; 4];
            let vector_end = keys / 4 * 4;
            for column in (0..vector_end).step_by(4) {
                for lane in 0..4 {
                    let index = column + lane;
                    let score = if index < valid {
                        row_scores[index] * scale
                    } else {
                        f32::NEG_INFINITY
                    };
                    row_scores[index] = score;
                    max4[lane] = max4[lane].max(score);
                }
            }
            let mut max = max4[0].max(max4[2]).max(max4[1].max(max4[3]));
            for index in vector_end..keys {
                let score = if index < valid {
                    row_scores[index] * scale
                } else {
                    f32::NEG_INFINITY
                };
                row_scores[index] = score;
                max = max.max(score);
            }
            let mut sum4 = [0.0f32; 4];
            for column in (0..vector_end).step_by(4) {
                for lane in 0..4 {
                    let index = column + lane;
                    let weight = torch28_exp(row_scores[index] - max);
                    row_scores[index] = weight;
                    sum4[lane] += weight;
                }
            }
            let mut sum = (sum4[0] + sum4[2]) + (sum4[1] + sum4[3]);
            for index in vector_end..keys {
                let weight = (row_scores[index] - max).exp();
                row_scores[index] = weight;
                sum += weight;
            }
            reciprocals[row] = sum.recip();
        }
        unsafe {
            sys::cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANSPOSE,
                CBLAS_NO_TRANSPOSE,
                block_rows_i32,
                head_dim_i32,
                keys_i32,
                1.0,
                scores.as_ptr(),
                keys_i32,
                v_cache.as_ptr().add(head_offset),
                ENC_HIDDEN as i32,
                0.0,
                output
                    .as_mut_ptr()
                    .add(block_start * output_stride as usize),
                output_stride,
            );
        }
        for row in 0..block_rows {
            let reciprocal = reciprocals[row];
            let output_start = (block_start + row) * output_stride as usize;
            let output_row = &mut output[output_start..output_start + ENC_HEAD_DIM];
            for value in output_row {
                *value *= reciprocal;
            }
        }
    }
    Ok(())
}

impl PatchEncoder {
    pub fn from_source(source: &dyn TensorSource, config: DotsTtsConfig) -> Result<Self, String> {
        let d = config.latent_dim as u64;
        let enc_hid = ENC_HIDDEN as u64;
        let ds = load_f16_f32(source, "dotstts.patch_encoder.ds_proj.weight", &[2, d, d])?;
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
            let name =
                |suffix: &str| format!("dotstts.patch_encoder.encoder.layers.{layer}.{suffix}");
            let hid = [ENC_HIDDEN as u64];
            let hid2 = [ENC_HIDDEN as u64; 2];
            let q = load_f16_f32(source, &name("attn_q.weight"), &hid2)?;
            let k = load_f16_f32(source, &name("attn_k.weight"), &hid2)?;
            let v = load_f16_f32(source, &name("attn_v.weight"), &hid2)?;
            let mut qkv = Vec::with_capacity(q.len() + k.len() + v.len());
            qkv.extend_from_slice(&q);
            qkv.extend_from_slice(&k);
            qkv.extend_from_slice(&v);
            layers.push(PatchLayerWeights {
                attn_norm: load_f16_f32(source, &name("attn_norm.weight"), &hid)?,
                ffn_norm: load_f16_f32(source, &name("ffn_norm.weight"), &hid)?,
                q,
                k,
                v,
                qkv,
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
        #[cfg(any(
            all(feature = "accelerate", target_os = "macos"),
            all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        ))]
        {
            let projected = self.downsample_projection_channel_major(frames, &state.conv_tail);
            linear_forward_transposed_input_then_bias(
                &self.in_proj,
                &self.in_bias,
                &projected,
                n_tokens,
                128,
                ENC_HIDDEN,
                &mut tokens,
            );
        }
        #[cfg(not(any(
            all(feature = "accelerate", target_os = "macos"),
            all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        )))]
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
        #[cfg(any(
            all(feature = "accelerate", target_os = "macos"),
            all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        ))]
        {
            let channel_major = self.downsample_projection_channel_major(frames, conv_tail);
            for out in 0..128 {
                for token in 0..n_tokens {
                    projected[token * 128 + out] = channel_major[out * n_tokens + token];
                }
            }
        }
        #[cfg(not(any(
            all(feature = "accelerate", target_os = "macos"),
            all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        )))]
        for token in 0..n_tokens {
            for out in 0..128 {
                let mut sum = self.ds_bias[out];
                for tap in 0..2 {
                    let in_pos = 2 * token + tap;
                    let frame = if in_pos == 0 {
                        conv_tail
                    } else {
                        &frames[(in_pos - 1) * 128..in_pos * 128]
                    };
                    for inp in 0..128 {
                        let weight = self.ds_proj[(out * 128 + inp) * 2 + tap];
                        sum = weight.mul_add(frame[inp], sum);
                    }
                }
                projected[token * 128 + out] = sum;
            }
        }
        projected
    }

    #[cfg(any(
        all(feature = "accelerate", target_os = "macos"),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
    fn downsample_projection_channel_major(&self, frames: &[f32], conv_tail: &[f32]) -> Vec<f32> {
        let n_tokens = frames.len() / (2 * 128);
        let mut columns = vec![0.0f32; 2 * 128 * n_tokens];
        let mut channel_major = vec![0.0f32; 128 * n_tokens];
        for out in 0..128 {
            channel_major[out * n_tokens..(out + 1) * n_tokens].fill(self.ds_bias[out]);
        }
        for inp in 0..128 {
            for tap in 0..2 {
                let column = (inp * 2 + tap) * n_tokens;
                for token in 0..n_tokens {
                    let in_pos = 2 * token + tap;
                    let frame = if in_pos == 0 {
                        conv_tail
                    } else {
                        &frames[(in_pos - 1) * 128..in_pos * 128]
                    };
                    columns[column + token] = frame[inp];
                }
            }
        }
        unsafe {
            sys::cblas_sgemm(
                102,
                111,
                111,
                n_tokens as i32,
                128,
                256,
                1.0,
                columns.as_ptr(),
                n_tokens as i32,
                self.ds_proj.as_ptr(),
                256,
                1.0,
                channel_major.as_mut_ptr(),
                n_tokens as i32,
            );
        }
        channel_major
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
        let mut qkv = vec![0.0f32; t * ENC_HIDDEN * 3];
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
            linear_forward(
                &layer.qkv,
                None,
                &normed,
                ENC_HIDDEN,
                ENC_HIDDEN * 3,
                &mut qkv,
            );
            for i in 0..t {
                let qkv_row = &qkv[i * ENC_HIDDEN * 3..(i + 1) * ENC_HIDDEN * 3];
                q[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN].copy_from_slice(&qkv_row[..ENC_HIDDEN]);
                k[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN]
                    .copy_from_slice(&qkv_row[ENC_HIDDEN..ENC_HIDDEN * 2]);
                v[i * ENC_HIDDEN..(i + 1) * ENC_HIDDEN].copy_from_slice(&qkv_row[ENC_HIDDEN * 2..]);
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
            #[cfg(any(
                all(feature = "accelerate", target_os = "macos"),
                all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
            ))]
            for head in 0..ENC_HEADS {
                let offset = head * ENC_HEAD_DIM;
                torch28_flash_attention_head(
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
            #[cfg(not(any(
                all(feature = "accelerate", target_os = "macos"),
                all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
            )))]
            for i in 0..t {
                let keys = start + i + 1;
                for head in 0..ENC_HEADS {
                    let qh = &q[i * ENC_HIDDEN + head * ENC_HEAD_DIM
                        ..i * ENC_HIDDEN + (head + 1) * ENC_HEAD_DIM];
                    let acc = online_attention_head(
                        qh,
                        &state.k_cache[layer_idx],
                        &state.v_cache[layer_idx],
                        keys,
                        head * ENC_HEAD_DIM,
                    );
                    let dst = i * ENC_HIDDEN + head * ENC_HEAD_DIM;
                    for (slot, &value) in attn[dst..dst + ENC_HEAD_DIM].iter_mut().zip(acc.iter()) {
                        *slot = value;
                    }
                }
            }
            // Torch runs the prefill projections as full matrices. Keep the
            // same SGEMM shape because Accelerate's reduction order is part of
            // the pinned bitwise contract.
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
            ds_proj: Vec::new(),
            ds_bias: Vec::new(),
            in_proj: Vec::new(),
            in_bias: Vec::new(),
            out_proj: Vec::new(),
            out_bias: Vec::new(),
            layers: vec![PatchLayerWeights {
                attn_norm: vec![1.0; ENC_HIDDEN],
                ffn_norm: vec![1.0; ENC_HIDDEN],
                q: Vec::new(),
                k: Vec::new(),
                v: Vec::new(),
                qkv: vec![0.0; ENC_HIDDEN * ENC_HIDDEN * 3],
                o: vec![0.0; ENC_HIDDEN * ENC_HIDDEN],
                o_bias: vec![0.0; ENC_HIDDEN],
                fc1: vec![0.0; ENC_FFN * ENC_HIDDEN],
                fc1_bias: vec![f32::from_bits(0xbdbd_9888); ENC_FFN],
                fc2,
                fc2_bias: vec![0.0; ENC_HIDDEN],
            }],
            config: DotsTtsConfig {
                patch_size: 4,
                latent_dim: 128,
                hop_size: 1920,
                sample_rate: 48_000,
                fm_hidden_size: ENC_HIDDEN,
                llm_hidden_size: 1536,
                xvec_dim: 512,
                patch_encoder_layers: 1,
                dit_layers: 18,
                dit_heads: 16,
                default_nfe: 10,
                default_guidance: 1.2,
                default_speaker_scale: 1.5,
                default_eos_threshold: 0.8,
            },
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

    #[cfg(any(
        all(feature = "accelerate", target_os = "macos"),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
    fn flash_fixture_value(kind: u32, row: u32, column: u32) -> f32 {
        let mut mixed = kind.wrapping_mul(0x9e37_79b9)
            ^ row.wrapping_mul(0x85eb_ca6b)
            ^ column.wrapping_mul(0xc2b2_ae35);
        mixed ^= mixed >> 16;
        mixed = mixed.wrapping_mul(0x7feb_352d);
        mixed ^= mixed >> 15;
        ((mixed & 0x3fff) as i32 - 8192) as f32 / 2048.0
    }

    #[cfg(any(
        all(feature = "accelerate", target_os = "macos"),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
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

    /// Torch 2.8 CPU FlashAttention / Accelerate C oracle, generated from the
    /// identical integer fixture. Values are raw F32 words, not tolerances.
    // The pinned bit patterns in `EXPECTED` were captured against Apple's
    // Accelerate sgemm; OpenBLAS's reduction order produces different bits
    // in the same ULP range, so this test is restricted to macOS-Accelerate.
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    #[test]
    fn attention_72_token_fixture_matches_torch28_flash_oracle() {
        const TOKENS: usize = 72;
        const EXPECTED: &[(usize, usize, u32)] = &[
            (0, 0, 0x406d_9800),
            (0, 63, 0x3fb0_d000),
            (1, 7, 0xc072_f800),
            (15, 19, 0x3e72_645c),
            (31, 37, 0xbe41_b800),
            (32, 0, 0xbf33_026b),
            (32, 63, 0x4011_fdb7),
            (47, 11, 0x4033_b535),
            (63, 29, 0x3fea_2e54),
            (64, 3, 0x3ec0_e823),
            (70, 41, 0x403e_57a3),
            (71, 0, 0xbe8a_248a),
            (71, 13, 0xbf18_7df7),
            (71, 37, 0x3dbb_381b),
            (71, 63, 0x3fac_452c),
        ];

        let (q, k, v) = flash_fixture_qkv(TOKENS);
        let mut actual = vec![0.0f32; TOKENS * ENC_HEAD_DIM];
        let legacy = online_attention_head(&q[15 * ENC_HEAD_DIM..16 * ENC_HEAD_DIM], &k, &v, 16, 0);
        assert_ne!(
            legacy[19].to_bits(),
            0x3e72_645c,
            "fixture must reject the legacy online recurrence"
        );
        torch28_flash_attention_head(
            &q,
            TOKENS,
            ENC_HEAD_DIM,
            &k,
            &v,
            TOKENS,
            0,
            0,
            &mut actual,
            ENC_HEAD_DIM,
        )
        .unwrap();
        for &(row, column, expected) in EXPECTED {
            assert_eq!(
                actual[row * ENC_HEAD_DIM + column].to_bits(),
                expected,
                "attention[{row}, {column}]"
            );
        }
    }

    /// The final key is a scalar Torch tail after the four-lane reduction.
    // Pinned against Apple's Accelerate sgemm; see the comment on
    // `attention_72_token_fixture_matches_torch28_flash_oracle` for why this
    // is macOS-only.
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    #[test]
    fn attention_scalar_tail_matches_torch28_flash_oracle() {
        const TOKENS: usize = 5;
        let (q, k, v) = flash_fixture_qkv(TOKENS);
        let mut actual = vec![0.0f32; TOKENS * ENC_HEAD_DIM];
        torch28_flash_attention_head(
            &q,
            TOKENS,
            ENC_HEAD_DIM,
            &k,
            &v,
            TOKENS,
            0,
            0,
            &mut actual,
            ENC_HEAD_DIM,
        )
        .unwrap();
        for &(row, column, expected) in &[
            (0, 0, 0x406d_9800),
            (1, 7, 0xc072_f800),
            (4, 0, 0xbec9_d053),
            (4, 17, 0x3f13_a718),
            (4, 63, 0xbe63_159a),
        ] {
            assert_eq!(
                actual[row * ENC_HEAD_DIM + column].to_bits(),
                expected,
                "attention tail[{row}, {column}]"
            );
        }
    }

    #[cfg(any(
        all(feature = "accelerate", target_os = "macos"),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
    ))]
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
            torch28_flash_attention_head(
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
