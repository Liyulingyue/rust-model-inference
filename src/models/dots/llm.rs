//! dotstts LLM half: a 28×1536 Qwen2 decoder (arch `qwen2`) driven step by
//! step so the flow-matching pipeline can interleave LLM forwards with FM
//! decodes. Mirrors `models::qwen3::tts::talker::TtsSession` but for the
//! plain Qwen2 layout (no per-head Q/K RMSNorm, single scalar Neox rope) and
//! exposes the raw hidden state of every step for `hidden_proj`/`eos_proj`.

use std::sync::Arc;

use crate::core::scratchpad::KvCache;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::models::dots::patch_encoder::torch_rms_norm_with_eps;
use crate::models::dots::speaker::exp::torch28_exp;
use crate::ops::kernel::Weight;
use crate::models::dots::blas::sys;
use crate::ops::{dot_f32, vec_mad_f32};

#[derive(Debug, Clone)]
pub struct DotsLlmConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head: usize,
    pub n_ff: usize,
    pub vocab_size: usize,
    pub n_ctx: usize,
    pub eps: f32,
    pub freq_base: f32,
}

impl DotsLlmConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let cfg = crate::models::qwen3::Qwen3Config::from_source(source)?;
        let vocab_size = source
            .metadata("tokenizer.ggml.tokens")
            .and_then(|v| v.to_arr())
            .map(Vec::len)
            .unwrap_or(0);
        Ok(Self {
            n_embd: cfg.n_embd,
            n_layer: cfg.n_layer,
            n_head: cfg.n_head,
            n_head_kv: cfg.n_head_kv,
            n_embd_head: cfg.n_embd_head_k,
            n_ff: cfg.n_ff,
            vocab_size,
            n_ctx: cfg.n_ctx,
            eps: cfg.eps,
            freq_base: cfg.freq_base,
        })
    }
}

pub(crate) struct DotsLinear {
    pub(crate) data: Vec<f32>,
    pub(crate) n_in: usize,
    pub(crate) n_out: usize,
}

impl DotsLinear {
    fn from_source(
        source: &dyn TensorSource,
        name: &str,
        n_in: usize,
        n_out: usize,
    ) -> Result<Self, String> {
        let info = source
            .tensor_info(name)
            .ok_or_else(|| format!("Missing tensor: {name}"))?;
        let expected_dims = [n_in as u64, n_out as u64];
        if info.dims != expected_dims {
            return Err(format!(
                "Invalid tensor {name}: shape {:?}; expected {:?}",
                info.dims, expected_dims
            ));
        }
        let bytes = source
            .tensor_slice(name)
            .ok_or_else(|| format!("Missing tensor: {name}"))?;
        let bytes_per_element = match info.ggml_type {
            crate::core::tensor::GGMLType::BF16 => 2,
            crate::core::tensor::GGMLType::F32 => 4,
            other => {
                return Err(format!(
                    "Invalid tensor {name}: type {other:?}; expected BF16 or F32"
                ))
            }
        };
        let expected_bytes = n_in
            .checked_mul(n_out)
            .and_then(|count| count.checked_mul(bytes_per_element))
            .ok_or_else(|| format!("Tensor byte size overflow: {name}"))?;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "Invalid tensor data length for {name}: {}; expected {expected_bytes}",
                bytes.len()
            ));
        }
        let data = match info.ggml_type {
            crate::core::tensor::GGMLType::BF16 => bytes
                .chunks_exact(2)
                .map(|chunk| {
                    crate::core::tensor::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))
                })
                .collect(),
            crate::core::tensor::GGMLType::F32 => bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
            _ => unreachable!("tensor type checked above"),
        };
        Ok(Self { data, n_in, n_out })
    }

    fn matmul(&self, input: &[f32], bias: Option<&[f32]>, output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.n_in);
        debug_assert_eq!(output.len(), self.n_out);
        if let Some(bias) = bias {
            debug_assert_eq!(bias.len(), self.n_out);
            output.copy_from_slice(bias);
        } else {
            output.fill(0.0);
        }
        #[cfg(any(
    target_os = "macos",
    all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
))]
        unsafe {
            sys::cblas_sgemm(
                101,
                111,
                112,
                1,
                self.n_out as i32,
                self.n_in as i32,
                1.0,
                input.as_ptr(),
                self.n_in as i32,
                self.data.as_ptr(),
                self.n_in as i32,
                if bias.is_some() { 1.0 } else { 0.0 },
                output.as_mut_ptr(),
                self.n_out as i32,
            );
        }
        #[cfg(not(any(
    target_os = "macos",
    all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
)))]
        for row in 0..self.n_out {
            let row_start = row * self.n_in;
            let mut sum = bias.map_or(0.0, |values| values[row]);
            for index in 0..self.n_in {
                sum = self.data[row_start + index].mul_add(input[index], sum);
            }
            output[row] = sum;
        }
    }

    fn matmul_batch(&self, input: &[f32], bias: Option<&[f32]>, rows: usize, output: &mut [f32]) {
        debug_assert_eq!(input.len(), rows * self.n_in);
        debug_assert_eq!(output.len(), rows * self.n_out);
        #[cfg(any(
    target_os = "macos",
    all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
))]
        {
            if let Some(bias) = bias {
                for row in output.chunks_exact_mut(self.n_out) {
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
                    self.n_out as i32,
                    self.n_in as i32,
                    1.0,
                    input.as_ptr(),
                    self.n_in as i32,
                    self.data.as_ptr(),
                    self.n_in as i32,
                    if bias.is_some() { 1.0 } else { 0.0 },
                    output.as_mut_ptr(),
                    self.n_out as i32,
                );
            }
            return;
        }
        #[cfg(not(any(
    target_os = "macos",
    all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
)))]
        for row in 0..rows {
            self.matmul(
                &input[row * self.n_in..(row + 1) * self.n_in],
                bias,
                &mut output[row * self.n_out..(row + 1) * self.n_out],
            );
        }
    }
}

pub(crate) struct DotsLayerWeights {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) q_bias: Vec<f32>,
    pub(crate) k_bias: Vec<f32>,
    pub(crate) v_bias: Vec<f32>,
    pub(crate) wq: DotsLinear,
    pub(crate) wk: DotsLinear,
    pub(crate) wv: DotsLinear,
    pub(crate) wo: DotsLinear,
    pub(crate) w_gate: DotsLinear,
    pub(crate) w_up: DotsLinear,
    pub(crate) w_down: DotsLinear,
}

/// Loaded Qwen2 LLM for dots.tts.
pub struct DotsLlm {
    /// Keep the source alive: all weights are 'static views into its mmap.
    pub source: Arc<dyn TensorSource>,
    pub pool: Arc<ComputePool>,
    pub config: DotsLlmConfig,
    pub output_norm: Vec<f32>,
    pub(crate) layers: Vec<DotsLayerWeights>,
    pub token_embedding: Weight<'static>,
}

impl DotsLlm {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let config = DotsLlmConfig::from_source(source.as_ref())?;
        let n_embd_q = config.n_head * config.n_embd_head;
        let n_embd_k = config.n_head_kv * config.n_embd_head;
        let output_norm = crate::core::tensor::load_f32_tensor(
            source.as_ref(),
            "output_norm.weight",
            &[config.n_embd as u64],
        )?;
        let token_embedding = crate::models::qwen3::static_weight(
            source.as_ref(),
            "token_embd.weight",
            config.n_embd,
            config.vocab_size,
        );
        let mut layers = Vec::with_capacity(config.n_layer);
        for layer in 0..config.n_layer {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}");
            let n_embd = [config.n_embd as u64];
            layers.push(DotsLayerWeights {
                attn_norm: crate::core::tensor::load_f32_tensor(
                    source.as_ref(),
                    &name("attn_norm.weight"),
                    &n_embd,
                )?,
                ffn_norm: crate::core::tensor::load_f32_tensor(
                    source.as_ref(),
                    &name("ffn_norm.weight"),
                    &n_embd,
                )?,
                q_bias: crate::core::tensor::load_f32_tensor(
                    source.as_ref(),
                    &name("attn_q.bias"),
                    &[n_embd_q as u64],
                )?,
                k_bias: crate::core::tensor::load_f32_tensor(
                    source.as_ref(),
                    &name("attn_k.bias"),
                    &[n_embd_k as u64],
                )?,
                v_bias: crate::core::tensor::load_f32_tensor(
                    source.as_ref(),
                    &name("attn_v.bias"),
                    &[n_embd_k as u64],
                )?,
                wq: DotsLinear::from_source(
                    source.as_ref(),
                    &name("attn_q.weight"),
                    config.n_embd,
                    n_embd_q,
                )?,
                wk: DotsLinear::from_source(
                    source.as_ref(),
                    &name("attn_k.weight"),
                    config.n_embd,
                    n_embd_k,
                )?,
                wv: DotsLinear::from_source(
                    source.as_ref(),
                    &name("attn_v.weight"),
                    config.n_embd,
                    n_embd_k,
                )?,
                wo: DotsLinear::from_source(
                    source.as_ref(),
                    &name("attn_output.weight"),
                    n_embd_q,
                    config.n_embd,
                )?,
                w_gate: DotsLinear::from_source(
                    source.as_ref(),
                    &name("ffn_gate.weight"),
                    config.n_embd,
                    config.n_ff,
                )?,
                w_up: DotsLinear::from_source(
                    source.as_ref(),
                    &name("ffn_up.weight"),
                    config.n_embd,
                    config.n_ff,
                )?,
                w_down: DotsLinear::from_source(
                    source.as_ref(),
                    &name("ffn_down.weight"),
                    config.n_ff,
                    config.n_embd,
                )?,
            });
        }
        Ok(Self {
            source,
            pool,
            config,
            output_norm,
            layers,
            token_embedding,
        })
    }

    pub fn new_session(&self) -> Result<DotsLlmSession<'_>, String> {
        DotsLlmSession::new(self)
    }
}

/// Input row for prefill / decode: a token id (embedded by the table) or an
/// already-computed projection (patch-encoder embeddings or codec feedback).
pub enum LlmInputRow<'a> {
    Token(u32),
    Embedding(&'a [f32]),
}

/// One LLM step with hidden-state capture. Safe single-threaded matmuls
/// (correctness-first; parallelizing is a later optimization).
pub struct DotsLlmSession<'model> {
    model: &'model DotsLlm,
    kv: KvCache,
    /// Reusable scratch buffers (allocated once per session).
    x: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    acc: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    step: usize,
    capacity: usize,
}

fn matmul(weight: &DotsLinear, input: &[f32], output: &mut [f32], bias: Option<&[f32]>) {
    weight.matmul(input, bias, output);
}

/// Match Torch 2.8 CPU SDPA on macOS: Accelerate SGEMMs around the same
/// four-lane softmax used by the dots patch encoder. `q`/`output` may point at
/// one head inside a wider row; their strides retain the full row width.
#[allow(clippy::too_many_arguments)]
fn attention_head(
    q: &[f32],
    rows: usize,
    q_stride: usize,
    k_cache: &[f32],
    v_cache: &[f32],
    keys: usize,
    first_query: usize,
    query_head: usize,
    head_offset: usize,
    head_dim: usize,
    cache_stride: usize,
    output: &mut [f32],
    output_stride: usize,
) -> Result<(), String> {
    if rows == 0 {
        return Ok(());
    }
    let query_end = first_query
        .checked_add(rows)
        .ok_or_else(|| "dots LLM attention query length overflow".to_string())?;
    if keys < query_end {
        return Err("dots LLM attention cache does not cover causal queries".into());
    }
    if head_dim == 0 || q_stride < head_dim || output_stride < head_dim {
        return Err("dots LLM attention head stride is narrower than its head".into());
    }
    let q_span = rows
        .checked_sub(1)
        .and_then(|last| last.checked_mul(q_stride))
        .and_then(|start| start.checked_add(head_dim))
        .ok_or_else(|| "dots LLM attention query span overflow".to_string())?;
    let output_span = rows
        .checked_sub(1)
        .and_then(|last| last.checked_mul(output_stride))
        .and_then(|start| start.checked_add(head_dim))
        .ok_or_else(|| "dots LLM attention output span overflow".to_string())?;
    let cache_span = keys
        .checked_sub(1)
        .and_then(|last| last.checked_mul(cache_stride))
        .and_then(|start| start.checked_add(head_offset + head_dim))
        .ok_or_else(|| "dots LLM attention cache span overflow".to_string())?;
    if q.len() < q_span
        || output.len() < output_span
        || k_cache.len() < cache_span
        || v_cache.len() < cache_span
    {
        return Err("dots LLM attention buffer is shorter than its declared shape".into());
    }

    #[cfg(any(
    target_os = "macos",
    all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
))]
    {
        const CBLAS_ROW_MAJOR: i32 = 101;
        const CBLAS_COL_MAJOR: i32 = 102;
        const CBLAS_NO_TRANSPOSE: i32 = 111;
        const CBLAS_TRANSPOSE: i32 = 112;
        let output_stride_i32 = i32::try_from(output_stride)
            .map_err(|_| "dots LLM attention output stride exceeds BLAS limits")?;
        let q_stride_i32 = i32::try_from(q_stride)
            .map_err(|_| "dots LLM attention query stride exceeds BLAS limits")?;
        let head_dim_i32 = i32::try_from(head_dim)
            .map_err(|_| "dots LLM attention head dimension exceeds BLAS limits")?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scale_sqrt = scale.sqrt();
        // Torch's math SDPA scales both operands by sqrt(scale) before its
        // addmm. Keep the query's [token, head, dim] stride in that addmm.
        let mut q_layout = vec![0.0f32; q_span];
        for row in 0..rows {
            let start = row * q_stride;
            q_layout[start..start + head_dim].copy_from_slice(&q[start..start + head_dim]);
            for value in &mut q_layout[start..start + head_dim] {
                *value *= scale_sqrt;
            }
        }
        let mut k_contiguous = vec![0.0f32; keys * head_dim];
        let mut v_contiguous = vec![0.0f32; keys * head_dim];
        for key in 0..keys {
            let offset = key * cache_stride + head_offset;
            k_contiguous[key * head_dim..(key + 1) * head_dim]
                .copy_from_slice(&k_cache[offset..offset + head_dim]);
            for value in &mut k_contiguous[key * head_dim..(key + 1) * head_dim] {
                *value *= scale_sqrt;
            }
            v_contiguous[key * head_dim..(key + 1) * head_dim]
                .copy_from_slice(&v_cache[offset..offset + head_dim]);
        }

        for block_start in [0usize] {
            let block_rows = rows;
            let max_keys = keys;
            let mut scores = vec![0.0f32; block_rows * max_keys];
            if block_rows == 1 {
                unsafe {
                    sys::cblas_sgemm(
                        CBLAS_ROW_MAJOR,
                        CBLAS_NO_TRANSPOSE,
                        CBLAS_TRANSPOSE,
                        1,
                        max_keys as i32,
                        head_dim_i32,
                        1.0,
                        q_layout.as_ptr().add(block_start * q_stride),
                        q_stride_i32,
                        k_contiguous.as_ptr(),
                        head_dim_i32,
                        0.0,
                        scores.as_mut_ptr(),
                        max_keys as i32,
                    );
                }
            } else {
                let mut k_transposed = vec![0.0f32; max_keys * head_dim];
                for key in 0..max_keys {
                    for index in 0..head_dim {
                        k_transposed[index * max_keys + key] = k_contiguous[key * head_dim + index];
                    }
                }
                unsafe {
                    sys::cblas_sgemm(
                        CBLAS_COL_MAJOR,
                        CBLAS_NO_TRANSPOSE,
                        CBLAS_NO_TRANSPOSE,
                        max_keys as i32,
                        block_rows as i32,
                        head_dim_i32,
                        1.0,
                        k_transposed.as_ptr(),
                        max_keys as i32,
                        q_layout.as_ptr().add(block_start * q_stride),
                        q_stride_i32,
                        0.0,
                        scores.as_mut_ptr(),
                        max_keys as i32,
                    );
                }
            }
            let mut reciprocals = vec![0.0f32; block_rows];
            for row in 0..block_rows {
                let valid = (first_query + block_start + row + 1).min(max_keys);
                let row_scores = &mut scores[row * max_keys..(row + 1) * max_keys];
                let mut max4 = [f32::NEG_INFINITY; 4];
                let vector_end = max_keys / 4 * 4;
                for column in (0..vector_end).step_by(4) {
                    for lane in 0..4 {
                        let index = column + lane;
                        let score = if index < valid {
                            row_scores[index]
                        } else {
                            f32::NEG_INFINITY
                        };
                        row_scores[index] = score;
                        max4[lane] = max4[lane].max(score);
                    }
                }
                let mut max = max4[0].max(max4[2]).max(max4[1].max(max4[3]));
                for index in vector_end..max_keys {
                    let score = if index < valid {
                        row_scores[index]
                    } else {
                        f32::NEG_INFINITY
                    };
                    row_scores[index] = score;
                    max = max.max(score);
                }
                let mut sum4 = [0.0f32; 4];
                for column in (0..max_keys).step_by(4) {
                    for lane in 0..4 {
                        let index = column + lane;
                        let weight = if index < max_keys {
                            let weight = torch28_exp(row_scores[index] - max);
                            row_scores[index] = weight;
                            weight
                        } else {
                            0.0
                        };
                        sum4[lane] += weight;
                    }
                }
                // Torch includes the zero-padded tail in its four-lane reduction.
                let sum = (sum4[0] + sum4[2]) + (sum4[1] + sum4[3]);
                reciprocals[row] = sum.recip();
                for weight in row_scores.iter_mut() {
                    *weight *= reciprocals[row];
                }
            }

            unsafe {
                sys::cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANSPOSE,
                    CBLAS_NO_TRANSPOSE,
                    block_rows as i32,
                    head_dim_i32,
                    max_keys as i32,
                    1.0,
                    scores.as_ptr(),
                    max_keys as i32,
                    v_contiguous.as_ptr(),
                    head_dim_i32,
                    0.0,
                    output.as_mut_ptr().add(block_start * output_stride),
                    output_stride_i32,
                );
            }
        }
        return Ok(());
    }

    #[cfg(not(any(
    target_os = "macos",
    all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
)))]
    {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; keys];
        let mut weights = vec![0.0f32; keys];
        for row in 0..rows {
            let valid = first_query + row + 1;
            let query = &q[row * q_stride..row * q_stride + head_dim];
            let mut max = f32::NEG_INFINITY;
            for key in 0..valid {
                let offset = key * cache_stride + head_offset;
                let score = dot_f32(query, &k_cache[offset..offset + head_dim], head_dim) * scale;
                scores[key] = score;
                max = max.max(score);
            }
            let mut sum = 0.0f32;
            for key in 0..valid {
                let weight = torch28_exp(scores[key] - max);
                weights[key] = weight;
                sum += weight;
            }
            let out = &mut output[row * output_stride..row * output_stride + head_dim];
            out.fill(0.0);
            let reciprocal = sum.recip();
            for key in 0..valid {
                let offset = key * cache_stride + head_offset;
                vec_mad_f32(
                    out,
                    &v_cache[offset..offset + head_dim],
                    weights[key] * reciprocal,
                );
            }
        }
        Ok(())
    }
}

#[cfg(feature = "parity-trace")]
fn debug_bits(label: &str, values: &[f32]) {
    if std::env::var_os("DOTS_LLM_DEBUG").is_some() {
        let bits = values
            .iter()
            .take(if std::env::var_os("DOTS_LLM_DEBUG_ALL").is_some() {
                values.len()
            } else {
                8
            })
            .map(|v| format!("{:08x}", v.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        eprintln!("dots.llm.debug {label} {bits}");
    }
}

// Optional parity-trace sidecar dumps; the helper is a no-op in normal builds.
#[allow(unused_variables)]
fn dump_stage(name: &str, position: usize, layer: usize, values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    if let Some(dir) = std::env::var_os("DOTS_LLM_STAGE_OUT") {
        let dir = std::path::Path::new(&dir);
        if let Err(error) = std::fs::create_dir_all(dir) {
            panic!("create dots LLM stage directory {}: {error}", dir.display());
        }
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let path = dir.join(format!("{name}.p{position}.l{layer}.f32"));
        if let Err(error) = std::fs::write(&path, bytes) {
            panic!("write dots LLM stage {}: {error}", path.display());
        }
    }
}

impl<'model> DotsLlmSession<'model> {
    pub fn new(model: &'model DotsLlm) -> Result<Self, String> {
        let cfg = &model.config;
        let n_embd_q = cfg.n_head * cfg.n_embd_head;
        let n_embd_kv = cfg.n_head_kv * cfg.n_embd_head;
        // the reference runtime caps the static LLM cache at
        // DEFAULT_MAX_SEQUENCE_LENGTH = 2048; keep the same bound so a
        // 131k-context gguf does not force multi-GB caches
        let capacity = cfg.n_ctx.min(2048);
        Ok(Self {
            model,
            kv: KvCache::new_f32(cfg.n_layer, capacity, n_embd_kv),
            x: vec![0.0; cfg.n_embd],
            normed: vec![0.0; cfg.n_embd],
            q: vec![0.0; n_embd_q],
            k: vec![0.0; n_embd_kv],
            v: vec![0.0; n_embd_kv],
            attn_out: vec![0.0; n_embd_q],
            acc: vec![0.0; cfg.n_embd_head],
            gate: vec![0.0; cfg.n_ff],
            up: vec![0.0; cfg.n_ff],
            down: vec![0.0; cfg.n_embd],
            step: 0,
            capacity,
        })
    }

    /// Length of the currently cached prefix.
    pub fn position(&self) -> usize {
        self.step
    }

    /// Embed + run one forward; returns the final normalized hidden row.
    pub fn step_row(&mut self, row: LlmInputRow<'_>) -> Result<Vec<f32>, String> {
        match row {
            LlmInputRow::Token(id) => {
                self.model.token_embedding.embedding_lookup(id, &mut self.x);
            }
            LlmInputRow::Embedding(embedding) => {
                if embedding.len() != self.model.config.n_embd {
                    return Err(format!(
                        "dotstts LLM embedding length {} != {}",
                        embedding.len(),
                        self.model.config.n_embd
                    ));
                }
                self.x.copy_from_slice(embedding);
            }
        }
        self.forward()?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.llm.hidden",
            None,
            &[1, self.x.len()],
            &self.x,
        ));
        self.step += 1;
        Ok(self.x.clone())
    }

    /// Run the initial prefill as one batched Torch-compatible forward.
    pub fn prefill_rows(&mut self, rows: &[LlmInputRow<'_>]) -> Result<Vec<f32>, String> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        if self.step != 0 {
            return Err("dotstts LLM batched prefill requires a fresh session".into());
        }
        let hidden = self.forward_batch(rows)?;
        #[cfg(feature = "parity-trace")]
        for row in hidden.chunks_exact(self.model.config.n_embd) {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.llm.hidden",
                None,
                &[1, self.model.config.n_embd],
                row,
            ));
        }
        self.step = rows.len();
        self.x.copy_from_slice(
            &hidden[(rows.len() - 1) * self.model.config.n_embd
                ..rows.len() * self.model.config.n_embd],
        );
        Ok(hidden)
    }

    /// Run the Qwen2 decoder for the current `self.x` at position `self.step`.
    fn forward(&mut self) -> Result<(), String> {
        let cfg = &self.model.config;
        let n_embd_q = cfg.n_head * cfg.n_embd_head;
        let n_embd_kv = cfg.n_head_kv * cfg.n_embd_head;
        let group_size = cfg.n_head / cfg.n_head_kv;
        let kq_scale = 1.0 / (cfg.n_embd_head as f32).sqrt();
        if self.step >= self.capacity {
            return Err(format!(
                "dotstts LLM session exceeds context {}",
                self.capacity
            ));
        }
        let kv_stride = n_embd_kv;
        let (k_cache, v_cache) = match &mut self.kv {
            KvCache::F32(cache) => (&mut cache.k, &mut cache.v),
            KvCache::F16(_) => return Err("dotstts LLM requires an F32 KV cache".into()),
        };

        for layer in 0..cfg.n_layer {
            let weights = &self.model.layers[layer];
            // 1. attention norm + QKV
            if layer == 0 {
                dump_stage("input", self.step, layer, &self.x);
            }
            torch_rms_norm_with_eps(&self.x, &weights.attn_norm, &mut self.normed, cfg.eps);
            if layer == 0 {
                dump_stage("rms_attn", self.step, layer, &self.normed);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("norm", &self.normed);
            }
            matmul(
                &weights.wq,
                &self.normed,
                &mut self.q,
                Some(&weights.q_bias),
            );
            if layer == 0 {
                dump_stage("q", self.step, layer, &self.q);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("q", &self.q);
            }
            matmul(
                &weights.wk,
                &self.normed,
                &mut self.k,
                Some(&weights.k_bias),
            );
            if layer == 0 {
                dump_stage("k", self.step, layer, &self.k);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("k", &self.k);
            }
            matmul(
                &weights.wv,
                &self.normed,
                &mut self.v,
                Some(&weights.v_bias),
            );
            if layer == 0 {
                dump_stage("v", self.step, layer, &self.v);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("v", &self.v);
            }
            // 2. rope + KV store (F32, matching the float32 Torch oracle)
            for head in self.q.chunks_exact_mut(cfg.n_embd_head) {
                crate::ops::rope::rope_neox_sleef(head, self.step, cfg.n_embd_head, cfg.freq_base);
            }
            for head in self.k.chunks_exact_mut(cfg.n_embd_head) {
                crate::ops::rope::rope_neox_sleef(head, self.step, cfg.n_embd_head, cfg.freq_base);
            }
            if layer == 0 {
                dump_stage("rope_q", self.step, layer, &self.q);
                dump_stage("rope_k", self.step, layer, &self.k);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("q_rope", &self.q);
                debug_bits("k_rope", &self.k);
            }
            let layer_base = layer * self.capacity * kv_stride;
            let row_base = layer_base + self.step * kv_stride;
            for kv_head in 0..cfg.n_head_kv {
                let offset = kv_head * cfg.n_embd_head;
                let dst = row_base + offset;
                k_cache[dst..dst + cfg.n_embd_head]
                    .copy_from_slice(&self.k[offset..offset + cfg.n_embd_head]);
                v_cache[dst..dst + cfg.n_embd_head]
                    .copy_from_slice(&self.v[offset..offset + cfg.n_embd_head]);
            }
            if layer == 0 {
                dump_stage(
                    "cache_k",
                    self.step,
                    layer,
                    &k_cache[layer_base..layer_base + (self.step + 1) * kv_stride],
                );
                dump_stage(
                    "cache_v",
                    self.step,
                    layer,
                    &v_cache[layer_base..layer_base + (self.step + 1) * kv_stride],
                );
            }
            // 3. attention over the F32 cache (safe slices)
            self.attn_out.fill(0.0);
            for head in 0..cfg.n_head {
                let kv_head = head / group_size;
                let q_offset = head * cfg.n_embd_head;
                let out_offset = head * cfg.n_embd_head;
                attention_head(
                    &self.q[q_offset..],
                    1,
                    n_embd_q,
                    &k_cache[layer_base..layer_base + self.capacity * kv_stride],
                    &v_cache[layer_base..layer_base + self.capacity * kv_stride],
                    self.step + 1,
                    self.step,
                    head,
                    kv_head * cfg.n_embd_head,
                    cfg.n_embd_head,
                    kv_stride,
                    &mut self.attn_out[out_offset..],
                    n_embd_q,
                )?;
            }
            if layer == 0 {
                dump_stage("attention", self.step, layer, &self.attn_out);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("attn_out", &self.attn_out);
            }
            // 4. output projection + residual
            matmul(&weights.wo, &self.attn_out, &mut self.down, None);
            if layer == 0 {
                dump_stage("o", self.step, layer, &self.down);
            }
            vec_mad_f32(&mut self.x, &self.down, 1.0);
            if layer == 0 {
                dump_stage("o_residual", self.step, layer, &self.x);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("attn_resid", &self.x);
            }
            // 5. FFN: gate·up with SiLU, then down
            torch_rms_norm_with_eps(&self.x, &weights.ffn_norm, &mut self.normed, cfg.eps);
            if layer == 0 {
                dump_stage("rms_ffn", self.step, layer, &self.normed);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("ffn_norm", &self.normed);
            }
            matmul(&weights.w_gate, &self.normed, &mut self.gate, None);
            matmul(&weights.w_up, &self.normed, &mut self.up, None);
            if layer == 0 {
                dump_stage("gate", self.step, layer, &self.gate);
                dump_stage("up", self.step, layer, &self.up);
            }
            for i in 0..cfg.n_ff {
                let gate = self.gate[i];
                self.gate[i] = gate / (1.0 + torch28_exp(-gate)) * self.up[i];
            }
            if layer == 0 {
                dump_stage("ffn_act", self.step, layer, &self.gate);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("ffn_gate_up", &self.gate);
            }
            matmul(&weights.w_down, &self.gate, &mut self.down, None);
            if layer == 0 {
                dump_stage("ffn_down", self.step, layer, &self.down);
            }
            vec_mad_f32(&mut self.x, &self.down, 1.0);
            if layer == 0 {
                dump_stage("layer0_hidden", self.step, layer, &self.x);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 && layer == 0 {
                debug_bits("ffn_resid", &self.x);
            }
            #[cfg(feature = "parity-trace")]
            if self.step == 0 {
                debug_bits(&format!("layer{layer}.ffn_resid"), &self.x);
            }
        }
        // Qwen2Model applies its final RMSNorm before exposing hidden states.
        torch_rms_norm_with_eps(&self.x, &self.model.output_norm, &mut self.normed, cfg.eps);
        self.x.copy_from_slice(&self.normed);
        dump_stage("hidden", self.step, 0, &self.x);
        #[cfg(feature = "parity-trace")]
        if self.step == 0 {
            debug_bits("final", &self.x);
        }
        Ok(())
    }

    fn forward_batch(&mut self, rows: &[LlmInputRow<'_>]) -> Result<Vec<f32>, String> {
        let cfg = &self.model.config;
        let rows_len = rows.len();
        let n_embd_q = cfg.n_head * cfg.n_embd_head;
        let n_embd_kv = cfg.n_head_kv * cfg.n_embd_head;
        let group_size = cfg.n_head / cfg.n_head_kv;
        let kq_scale = 1.0 / (cfg.n_embd_head as f32).sqrt();
        if rows_len > self.capacity {
            return Err(format!(
                "dotstts LLM prefill exceeds context {}",
                self.capacity
            ));
        }

        let mut x = vec![0.0f32; rows_len * cfg.n_embd];
        for (row_index, row) in rows.iter().enumerate() {
            let dst = &mut x[row_index * cfg.n_embd..(row_index + 1) * cfg.n_embd];
            match row {
                LlmInputRow::Token(id) => self.model.token_embedding.embedding_lookup(*id, dst),
                LlmInputRow::Embedding(embedding) => {
                    if embedding.len() != cfg.n_embd {
                        return Err(format!(
                            "dotstts LLM embedding length {} != {}",
                            embedding.len(),
                            cfg.n_embd
                        ));
                    }
                    dst.copy_from_slice(embedding);
                }
            }
            #[cfg(feature = "parity-trace")]
            dump_stage("batch_input", row_index, 0, dst);
        }
        let mut normed = vec![0.0f32; rows_len * cfg.n_embd];
        let mut q = vec![0.0f32; rows_len * n_embd_q];
        let mut k = vec![0.0f32; rows_len * n_embd_kv];
        let mut v = vec![0.0f32; rows_len * n_embd_kv];
        let mut attn_out = vec![0.0f32; rows_len * n_embd_q];
        let mut acc = vec![0.0f32; cfg.n_embd_head];
        let mut down = vec![0.0f32; rows_len * cfg.n_embd];
        let mut gate = vec![0.0f32; rows_len * cfg.n_ff];
        let mut up = vec![0.0f32; rows_len * cfg.n_ff];

        for layer in 0..cfg.n_layer {
            let weights = &self.model.layers[layer];
            for row in 0..rows_len {
                torch_rms_norm_with_eps(
                    &x[row * cfg.n_embd..(row + 1) * cfg.n_embd],
                    &weights.attn_norm,
                    &mut normed[row * cfg.n_embd..(row + 1) * cfg.n_embd],
                    cfg.eps,
                );
            }
            weights
                .wq
                .matmul_batch(&normed, Some(&weights.q_bias), rows_len, &mut q);
            weights
                .wk
                .matmul_batch(&normed, Some(&weights.k_bias), rows_len, &mut k);
            weights
                .wv
                .matmul_batch(&normed, Some(&weights.v_bias), rows_len, &mut v);
            dump_stage("batch_q", 0, layer, &q);
            dump_stage("batch_k", 0, layer, &k);
            dump_stage("batch_v", 0, layer, &v);
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.norm.row1"),
                    &normed[cfg.n_embd..2 * cfg.n_embd],
                );
                debug_bits(
                    &format!("batch.l{layer}.q.row1"),
                    &q[n_embd_q..2 * n_embd_q],
                );
                debug_bits(
                    &format!("batch.l{layer}.k.row1"),
                    &k[n_embd_kv..2 * n_embd_kv],
                );
                debug_bits(
                    &format!("batch.l{layer}.v.row1"),
                    &v[n_embd_kv..2 * n_embd_kv],
                );
                if std::env::var_os("DOTS_LLM_DEBUG_BATCH_ROW2").is_some() && layer == 0 {
                    debug_bits(
                        "batch.l0.q.row2.head0",
                        &q[2 * n_embd_q..2 * n_embd_q + cfg.n_embd_head],
                    );
                    for row in 0..3 {
                        let start = row * n_embd_kv;
                        debug_bits(
                            &format!("batch.l0.k.row{row}.head0"),
                            &k[start..start + cfg.n_embd_head],
                        );
                        debug_bits(
                            &format!("batch.l0.v.row{row}.head0"),
                            &v[start..start + cfg.n_embd_head],
                        );
                    }
                }
            }
            for row in 0..rows_len {
                for head in
                    q[row * n_embd_q..(row + 1) * n_embd_q].chunks_exact_mut(cfg.n_embd_head)
                {
                    crate::ops::rope::rope_neox_sleef(head, row, cfg.n_embd_head, cfg.freq_base);
                }
                for head in
                    k[row * n_embd_kv..(row + 1) * n_embd_kv].chunks_exact_mut(cfg.n_embd_head)
                {
                    crate::ops::rope::rope_neox_sleef(head, row, cfg.n_embd_head, cfg.freq_base);
                }
            }
            dump_stage("batch_rope_q", 0, layer, &q);
            dump_stage("batch_rope_k", 0, layer, &k);
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.rope_q.row1"),
                    &q[n_embd_q..2 * n_embd_q],
                );
                if std::env::var_os("DOTS_LLM_DEBUG_BATCH_ROW2").is_some() && layer == 0 {
                    for head in 0..cfg.n_head {
                        let start = 2 * n_embd_q + head * cfg.n_embd_head;
                        debug_bits(
                            &format!("batch.l0.rope_q.row2.head{head}"),
                            &q[start..start + cfg.n_embd_head],
                        );
                    }
                    for head in 0..cfg.n_head_kv {
                        let start = 2 * n_embd_kv + head * cfg.n_embd_head;
                        debug_bits(
                            &format!("batch.l0.rope_k.row2.head{head}"),
                            &k[start..start + cfg.n_embd_head],
                        );
                    }
                }
                debug_bits(&format!("batch.l{layer}.rope_k.row0"), &k[..n_embd_kv]);
                debug_bits(
                    &format!("batch.l{layer}.rope_k.row1"),
                    &k[n_embd_kv..2 * n_embd_kv],
                );
            }
            let layer_base = layer * self.capacity * n_embd_kv;
            let (k_cache, v_cache) = match &mut self.kv {
                KvCache::F32(cache) => (&mut cache.k, &mut cache.v),
                KvCache::F16(_) => return Err("dotstts LLM requires an F32 KV cache".into()),
            };
            for row in 0..rows_len {
                let row_base = layer_base + row * n_embd_kv;
                k_cache[row_base..row_base + n_embd_kv]
                    .copy_from_slice(&k[row * n_embd_kv..(row + 1) * n_embd_kv]);
                v_cache[row_base..row_base + n_embd_kv]
                    .copy_from_slice(&v[row * n_embd_kv..(row + 1) * n_embd_kv]);
            }
            for head in 0..cfg.n_head {
                let kv_head = head / group_size;
                let offset = head * cfg.n_embd_head;
                attention_head(
                    &q[offset..],
                    rows_len,
                    n_embd_q,
                    &k_cache[layer_base..layer_base + self.capacity * n_embd_kv],
                    &v_cache[layer_base..layer_base + self.capacity * n_embd_kv],
                    rows_len,
                    0,
                    head,
                    kv_head * cfg.n_embd_head,
                    cfg.n_embd_head,
                    n_embd_kv,
                    &mut attn_out[offset..],
                    n_embd_q,
                )?;
            }
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.attn_out.row1"),
                    &attn_out[n_embd_q..2 * n_embd_q],
                );
                if std::env::var_os("DOTS_LLM_DEBUG_BATCH_ROW2").is_some() && layer == 0 {
                    for head in 0..cfg.n_head {
                        let start = 2 * n_embd_q + head * cfg.n_embd_head;
                        debug_bits(
                            &format!("batch.l0.attn_out.row2.head{head}"),
                            &attn_out[start..start + cfg.n_embd_head],
                        );
                    }
                    debug_bits(
                        "batch.l0.attn_out.row2.head0",
                        &attn_out[2 * n_embd_q..2 * n_embd_q + cfg.n_embd_head],
                    );
                }
            }
            weights
                .wo
                .matmul_batch(&attn_out, None, rows_len, &mut down);
            dump_stage("batch_attn", 0, layer, &attn_out);
            dump_stage("batch_o", 0, layer, &down);
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.o.row1"),
                    &down[n_embd_q..n_embd_q + cfg.n_embd],
                );
                if std::env::var_os("DOTS_LLM_DEBUG_BATCH_ROW2").is_some() && layer == 0 {
                    debug_bits("batch.l0.o.row2", &down[2 * cfg.n_embd..3 * cfg.n_embd]);
                }
            }
            for (value, residual) in down.iter_mut().zip(x.iter()) {
                *value += *residual;
            }
            x.copy_from_slice(&down);
            dump_stage("batch_layer_hidden", 0, layer, &x);
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.o_residual.row1"),
                    &x[cfg.n_embd..2 * cfg.n_embd],
                );
                if std::env::var_os("DOTS_LLM_DEBUG_BATCH_ROW2").is_some() && layer == 0 {
                    debug_bits(
                        "batch.l0.o_residual.row2",
                        &x[2 * cfg.n_embd..3 * cfg.n_embd],
                    );
                }
            }
            for row in 0..rows_len {
                torch_rms_norm_with_eps(
                    &x[row * cfg.n_embd..(row + 1) * cfg.n_embd],
                    &weights.ffn_norm,
                    &mut normed[row * cfg.n_embd..(row + 1) * cfg.n_embd],
                    cfg.eps,
                );
            }
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.rms_ffn.row1"),
                    &normed[cfg.n_embd..2 * cfg.n_embd],
                );
            }
            weights
                .w_gate
                .matmul_batch(&normed, None, rows_len, &mut gate);
            weights.w_up.matmul_batch(&normed, None, rows_len, &mut up);
            dump_stage("batch_gate", 0, layer, &gate);
            dump_stage("batch_up", 0, layer, &up);
            for (gate_value, up_value) in gate.iter_mut().zip(up.iter()) {
                let value = *gate_value;
                *gate_value = value / (1.0 + torch28_exp(-value)) * *up_value;
            }
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.gate.row1"),
                    &gate[cfg.n_ff..2 * cfg.n_ff],
                );
                debug_bits(
                    &format!("batch.l{layer}.up.row1"),
                    &up[cfg.n_ff..2 * cfg.n_ff],
                );
                debug_bits(
                    &format!("batch.l{layer}.ffn_act.row1"),
                    &gate[cfg.n_ff..2 * cfg.n_ff],
                );
            }
            weights
                .w_down
                .matmul_batch(&gate, None, rows_len, &mut down);
            dump_stage("batch_ffn_down", 0, layer, &down);
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.ffn_down.row1"),
                    &down[cfg.n_embd..2 * cfg.n_embd],
                );
            }
            for (value, residual) in down.iter_mut().zip(x.iter()) {
                *value += *residual;
            }
            x.copy_from_slice(&down);
            dump_stage("batch_full_hidden", 0, layer, &x);
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() {
                debug_bits(
                    &format!("batch.layer{layer}.hidden.row1"),
                    &x[cfg.n_embd..2 * cfg.n_embd],
                );
            }
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() && layer <= 1 {
                debug_bits(
                    &format!("batch.l{layer}.hidden.row1"),
                    &x[cfg.n_embd..2 * cfg.n_embd],
                );
            }
        }
        let mut hidden = vec![0.0f32; rows_len * cfg.n_embd];
        for row in 0..rows_len {
            torch_rms_norm_with_eps(
                &x[row * cfg.n_embd..(row + 1) * cfg.n_embd],
                &self.model.output_norm,
                &mut hidden[row * cfg.n_embd..(row + 1) * cfg.n_embd],
                cfg.eps,
            );
        }
        dump_stage("batch_hidden", 0, 0, &hidden);
        #[cfg(feature = "parity-trace")]
        if std::env::var_os("DOTS_LLM_DEBUG_BATCH").is_some() {
            debug_bits("batch.final.row1", &hidden[cfg.n_embd..2 * cfg.n_embd]);
        }
        Ok(hidden)
    }

    /// Final normalized hidden state of the most recent step.
    pub fn last_hidden(&self) -> &[f32] {
        &self.x
    }

    /// Normalized hidden state (after `output_norm.weight`).
    pub fn normalized_hidden(&self) -> Result<Vec<f32>, String> {
        if self.model.output_norm.len() != self.model.config.n_embd {
            return Err("dotstts output norm shape mismatch".into());
        }
        let mut out = vec![0.0; self.model.config.n_embd];
        torch_rms_norm_with_eps(
            &self.x,
            &self.model.output_norm,
            &mut out,
            self.model.config.eps,
        );
        Ok(out)
    }
}
