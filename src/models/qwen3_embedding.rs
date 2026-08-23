//! # Qwen3 Embedding
//!
//! 此模块包含 Qwen3 的文本 embedding 提取实现。

use crate::app::cli::EmbeddingOutput;
use crate::core::loader::model_config_from_source;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::qwen3::get_f32_tensor;
use crate::ops::{attention_value_f32, dot_f32, f16_to_f32, matmul_q8_0_quantized, quantize_q8_0_into, rms_norm, rms_norm_inplace, rope_neox, silu_mul_approx_inplace, softmax_inplace};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug)]
pub struct EmbeddingWeight<'a> {
    bytes: &'a [u8],
    ggml_type: GGMLType,
    n_cols: usize,
    n_rows: usize,
}

impl<'a> EmbeddingWeight<'a> {
    fn load(
        source: &'a dyn TensorSource,
        name: &str,
        n_cols: usize,
        n_rows: usize,
    ) -> Result<Self, String> {
        let info = source
            .tensor_info(name)
            .ok_or_else(|| format!("Embedding tensor {name} not found"))?;
        let expected_dims = [n_cols as u64, n_rows as u64];
        if info.dims != expected_dims {
            return Err(format!(
                "Embedding tensor {name} has shape {:?}; expected {:?}",
                info.dims, expected_dims
            ));
        }
        if !matches!(info.ggml_type, GGMLType::F16 | GGMLType::Q8_0) {
            return Err(format!(
                "Embedding tensor {name} has unsupported type {:?}; expected F16 or Q8_0",
                info.ggml_type
            ));
        }
        if info.ggml_type == GGMLType::Q8_0 && n_cols % 32 != 0 {
            return Err(format!(
                "Embedding tensor {name} has Q8_0 columns {n_cols}; expected a multiple of 32"
            ));
        }
        let n_elements = n_cols
            .checked_mul(n_rows)
            .ok_or_else(|| format!("Embedding tensor {name} shape overflows"))?;
        let expected_bytes = info.ggml_type.nbytes(n_elements);
        let bytes = source
            .tensor_slice(name)
            .ok_or_else(|| format!("Embedding tensor {name} data not found"))?;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "Embedding tensor {name} has {} bytes; expected {expected_bytes}",
                bytes.len()
            ));
        }
        Ok(Self {
            bytes,
            ggml_type: info.ggml_type,
            n_cols,
            n_rows,
        })
    }

    fn get_row(&self, row: usize, output: &mut [f32]) -> Result<(), String> {
        if row >= self.n_rows {
            return Err(format!(
                "Embedding row {row} is out of range for {} rows",
                self.n_rows
            ));
        }
        if output.len() != self.n_cols {
            return Err(format!(
                "Embedding row output has {} values; expected {}",
                output.len(), self.n_cols
            ));
        }
        match self.ggml_type {
            GGMLType::F16 => {
                let offset = row * self.n_cols * 2;
                for (value, bytes) in output
                    .iter_mut()
                    .zip(self.bytes[offset..offset + self.n_cols * 2].chunks_exact(2))
                {
                    *value = f16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap()));
                }
            }
            GGMLType::Q8_0 => crate::ops::embedding_lookup_q8_0(self.bytes, row as u32, self.n_cols, output),
            _ => unreachable!("EmbeddingWeight validates its type"),
        }
        Ok(())
    }

    fn matmul_prepared(
        &self,
        activation: &EmbeddingActivation<'_>,
        output: &mut [f32],
    ) -> Result<(), String> {
        if activation.ggml_type != self.ggml_type
            || activation.n_cols != self.n_cols
            || output.len() != self.n_rows
        {
            return Err(format!(
                "Embedding matmul has activation/output type {:?} shape {}/{}; expected {:?} {}/{}",
                activation.ggml_type,
                activation.n_cols,
                output.len(),
                self.ggml_type,
                self.n_cols,
                self.n_rows
            ));
        }
        match self.ggml_type {
            GGMLType::F16 => {
                for (row, value) in output.iter_mut().enumerate() {
                    let offset = row * self.n_cols * 2;
                    *value = crate::ops::dot_f16_f16_bytes(
                        activation.f16,
                        &self.bytes[offset..offset + self.n_cols * 2],
                        self.n_cols,
                    );
                }
            }
            GGMLType::Q8_0 => matmul_q8_0_quantized(
                self.bytes,
                activation.q8,
                activation.scales,
                output,
                self.n_cols,
                self.n_rows,
            ),
            _ => unreachable!("EmbeddingWeight validates its type"),
        }
        Ok(())
    }
}

struct EmbeddingActivationScratch {
    f16: Vec<u16>,
    q8: Vec<u8>,
    scales: Vec<f32>,
}

struct EmbeddingActivation<'a> {
    ggml_type: GGMLType,
    n_cols: usize,
    f16: &'a [u16],
    q8: &'a [u8],
    scales: &'a [f32],
}

impl EmbeddingActivationScratch {
    fn new(max_cols: usize) -> Self {
        Self {
            f16: vec![0; max_cols],
            q8: vec![0; max_cols],
            scales: vec![0.0; max_cols.div_ceil(32)],
        }
    }

    fn prepare<'a>(
        &'a mut self,
        weight: &EmbeddingWeight<'_>,
        input: &[f32],
    ) -> Result<EmbeddingActivation<'a>, String> {
        if input.len() != weight.n_cols || input.len() > self.f16.len() {
            return Err(format!(
                "Embedding activation has {} values; expected {} (scratch capacity {})",
                input.len(),
                weight.n_cols,
                self.f16.len()
            ));
        }
        match weight.ggml_type {
            GGMLType::F16 => crate::ops::f32_slice_to_f16(input, &mut self.f16[..input.len()]),
            GGMLType::Q8_0 => quantize_q8_0_into(
                input,
                input.len(),
                &mut self.q8[..input.len()],
                &mut self.scales[..input.len() / 32],
            ),
            _ => unreachable!("EmbeddingWeight validates its type"),
        }
        Ok(EmbeddingActivation {
            ggml_type: weight.ggml_type,
            n_cols: input.len(),
            f16: &self.f16[..input.len()],
            q8: &self.q8[..input.len()],
            scales: &self.scales[..input.len() / 32],
        })
    }
}

fn embedding_matmul_group(
    input: &[f32],
    projections: &mut [(&EmbeddingWeight<'_>, &mut [f32])],
    scratch: &mut EmbeddingActivationScratch,
) -> Result<(), String> {
    for ggml_type in [GGMLType::F16, GGMLType::Q8_0] {
        if let Some(index) = projections
            .iter()
            .position(|(weight, _)| weight.ggml_type == ggml_type)
        {
            let activation = scratch.prepare(projections[index].0, input)?;
            for (weight, output) in projections.iter_mut() {
                if weight.ggml_type == ggml_type {
                    weight.matmul_prepared(&activation, output)?;
                }
            }
        }
    }
    Ok(())
}

struct Qwen3EmbeddingLayerWeights<'a> {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    wq: EmbeddingWeight<'a>,
    wk: EmbeddingWeight<'a>,
    wv: EmbeddingWeight<'a>,
    wo: EmbeddingWeight<'a>,
    w_gate: EmbeddingWeight<'a>,
    w_up: EmbeddingWeight<'a>,
    w_down: EmbeddingWeight<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmbeddingPooling {
    Mean,
    Last,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EmbeddingConfig {
    causal_attn: bool,
    pooling: EmbeddingPooling,
}

fn embedding_config(
    arch: &str,
    get_meta: impl Fn(&str) -> Option<crate::core::tensor::MetaValue>,
) -> Result<EmbeddingConfig, String> {
    let pooling_key = format!("{arch}.pooling_type");
    let pooling = match get_meta(&pooling_key).and_then(|value| value.to_u64()) {
        Some(1) => EmbeddingPooling::Mean,
        Some(3) => EmbeddingPooling::Last,
        Some(value) => {
            return Err(format!(
                "Unsupported {pooling_key}: {value}; expected 1=MEAN or 3=LAST"
            ));
        }
        None => return Err(format!("Missing or invalid metadata: {pooling_key}")),
    };

    let causal_key = format!("{arch}.attention.causal");
    let causal_attn = match get_meta(&causal_key) {
        None => true,
        Some(crate::core::tensor::MetaValue::Bool(value)) => value,
        Some(value) => {
            return Err(format!(
                "Invalid metadata {causal_key}: expected bool, got {value:?}"
            ));
        }
    };

    Ok(EmbeddingConfig {
        causal_attn,
        pooling,
    })
}

fn encode_embedding_input(tokenizer: &BPETokenizer, prompt: &str) -> Vec<u32> {
    tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    )
}

fn embedding_positions(n_tokens: usize) -> std::ops::Range<usize> {
    0..n_tokens
}

fn attention_key_end(query: usize, n_tokens: usize, causal: bool) -> usize {
    if causal {
        (query + 1).min(n_tokens)
    } else {
        n_tokens
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_embedding_ffn_typed(
    hidden: &mut [f32],
    normed: &[f32],
    n_embd: usize,
    n_ff: usize,
    w_gate: &EmbeddingWeight<'_>,
    w_up: &EmbeddingWeight<'_>,
    w_down: &EmbeddingWeight<'_>,
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    down_buf: &mut [f32],
    activation_scratch: &mut EmbeddingActivationScratch,
) -> Result<(), String> {
    assert_eq!(hidden.len(), normed.len());
    assert!(n_embd > 0 && n_ff > 0);
    assert_eq!(hidden.len() % n_embd, 0);
    assert_eq!(gate_buf.len(), n_ff);
    assert_eq!(up_buf.len(), n_ff);
    assert_eq!(down_buf.len(), n_embd);

    for (input, residual) in normed
        .chunks_exact(n_embd)
        .zip(hidden.chunks_exact_mut(n_embd))
    {
        embedding_matmul_group(
            input,
            &mut [(w_gate, &mut *gate_buf), (w_up, &mut *up_buf)],
            activation_scratch,
        )?;

        silu_mul_approx_inplace(gate_buf, up_buf);

        embedding_matmul_group(
            up_buf,
            &mut [(w_down, &mut *down_buf)],
            activation_scratch,
        )?;

        for index in 0..n_embd {
            residual[index] += down_buf[index];
        }
    }
    Ok(())
}

fn pool_embedding_rows(
    hidden: &[f32],
    n_tokens: usize,
    n_embd: usize,
    pooling: EmbeddingPooling,
) -> Result<Vec<f32>, String> {
    let expected = n_tokens
        .checked_mul(n_embd)
        .ok_or_else(|| "Embedding shape overflow".to_string())?;
    if n_tokens == 0 || n_embd == 0 || hidden.len() != expected {
        return Err(format!(
            "Invalid embedding shape: rows={n_tokens}, cols={n_embd}, values={}",
            hidden.len()
        ));
    }

    match pooling {
        EmbeddingPooling::Last => Ok(hidden[(n_tokens - 1) * n_embd..n_tokens * n_embd].to_vec()),
        EmbeddingPooling::Mean => {
            let mut pooled = vec![0.0f32; n_embd];
            for row in hidden.chunks_exact(n_embd) {
                for (output, value) in pooled.iter_mut().zip(row) {
                    *output += *value;
                }
            }
            let scale = 1.0 / n_tokens as f32;
            for value in &mut pooled {
                *value *= scale;
            }
            Ok(pooled)
        }
    }
}

fn l2_normalize_embedding(values: &mut [f32]) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("Embedding contains a non-finite value".into());
    }

    let sum = values
        .iter()
        .map(|&value| f64::from(value * value))
        .sum::<f64>();

    let scale = if sum > 0.0 {
        (1.0 / sum.sqrt()) as f32
    } else {
        0.0
    };

    for value in values.iter_mut() {
        *value *= scale;
    }

    if values.iter().any(|value| !value.is_finite()) {
        return Err("Normalized embedding contains a non-finite value".into());
    }
    Ok(())
}

pub fn run_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
    _kv_format: crate::app::cli::KvFormat,
    output: EmbeddingOutput,
) {
    let t0 = Instant::now();
    let config = model_config_from_source(source).expect("Failed to parse model config");

    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    let is_qwen3 = arch == "qwen3";

    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .expect("Failed to init tokenizer");
    let embedding_cfg = embedding_config(&arch, |key| source.metadata(key).cloned())
        .unwrap_or_else(|error| {
            eprintln!("Embedding metadata error: {error}");
            std::process::exit(1);
        });

    let n_embd = config.n_embd;
    let n_layer = config.n_layer;
    let n_head = config.n_head;
    let n_head_kv = config.n_head_kv;
    let n_embd_head = config.n_embd_head;
    let n_embd_head_k = if let Some(v) = source.metadata(&format!("{}.attention.key_length", arch)) {
        v.to_u64().unwrap_or(n_embd_head as u64) as usize
    } else {
        n_embd_head
    };
    let n_embd_head_v =
        if let Some(v) = source.metadata(&format!("{}.attention.value_length", arch)) {
            v.to_u64().unwrap_or(n_embd_head as u64) as usize
        } else {
            n_embd_head
        };
    let n_embd_q = n_head * n_embd_head_k;
    let n_embd_gqa = n_head_kv * n_embd_head_v;
    let n_ff = config.n_ff;
    let eps = config.norm_eps;
    let freq_base = config.rope_freq_base;

    let output_norm = get_f32_tensor(source, "output_norm.weight", n_embd);
    let embd_weight = EmbeddingWeight::load(source, "token_embd.weight", n_embd, tokenizer.vocab_size())
        .unwrap_or_else(|error| panic!("Failed to load embedding token weights: {error}"));

    let layers: Vec<Qwen3EmbeddingLayerWeights> = (0..n_layer)
        .map(|l| Qwen3EmbeddingLayerWeights {
            attn_norm: get_f32_tensor(source, &format!("blk.{}.attn_norm.weight", l), n_embd),
            ffn_norm: get_f32_tensor(source, &format!("blk.{}.ffn_norm.weight", l), n_embd),
            q_norm: if is_qwen3 {
                Some(get_f32_tensor(
                    source,
                    &format!("blk.{}.attn_q_norm.weight", l),
                    n_embd_head_k,
                ))
            } else {
                None
            },
            k_norm: if is_qwen3 {
                Some(get_f32_tensor(
                    source,
                    &format!("blk.{}.attn_k_norm.weight", l),
                    n_embd_head_k,
                ))
            } else {
                None
            },
            wq: EmbeddingWeight::load(source, &format!("blk.{l}.attn_q.weight"), n_embd, n_embd_q)
                .unwrap_or_else(|error| panic!("Failed to load embedding Q weights: {error}")),
            wk: EmbeddingWeight::load(source, &format!("blk.{l}.attn_k.weight"), n_embd, n_embd_gqa)
                .unwrap_or_else(|error| panic!("Failed to load embedding K weights: {error}")),
            wv: EmbeddingWeight::load(source, &format!("blk.{l}.attn_v.weight"), n_embd, n_embd_gqa)
                .unwrap_or_else(|error| panic!("Failed to load embedding V weights: {error}")),
            wo: EmbeddingWeight::load(source, &format!("blk.{l}.attn_output.weight"), n_embd_q, n_embd)
                .unwrap_or_else(|error| panic!("Failed to load embedding output weights: {error}")),
            w_gate: EmbeddingWeight::load(source, &format!("blk.{l}.ffn_gate.weight"), n_embd, n_ff)
                .unwrap_or_else(|error| panic!("Failed to load embedding gate weights: {error}")),
            w_up: EmbeddingWeight::load(source, &format!("blk.{l}.ffn_up.weight"), n_embd, n_ff)
                .unwrap_or_else(|error| panic!("Failed to load embedding up weights: {error}")),
            w_down: EmbeddingWeight::load(source, &format!("blk.{l}.ffn_down.weight"), n_ff, n_embd)
                .unwrap_or_else(|error| panic!("Failed to load embedding down weights: {error}")),
        })
        .collect();

    let load_ms = t0.elapsed().as_millis();
    if output == EmbeddingOutput::Summary {
        println!(
            "Model: {} | n_embd={} n_layer={} n_head={} n_head_kv={} n_ff={} | loaded in {}ms",
            arch, n_embd, n_layer, n_head, n_head_kv, n_ff, load_ms
        );
    }

    let prompt_tokens = encode_embedding_input(&tokenizer, prompt);
    if prompt_tokens.is_empty() {
        eprintln!("Embedding input produced no tokens");
        std::process::exit(1);
    }
    let n_tokens = prompt_tokens.len();
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = crate::app::resolve_thread_count(n_threads_arg, available_threads);

    let pool = Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());
    if output == EmbeddingOutput::Summary {
        println!("Prompt: {} ({} tokens)", prompt, n_tokens);
    }

    let kq_scale = 1.0f32 / (n_embd_head_k as f32).sqrt();
    let group_size = n_head / n_head_kv;

    let mut hidden = vec![0.0f32; n_tokens * n_embd];
    let mut q_buf = vec![0.0f32; n_tokens * n_embd_q];
    let mut k_buf = vec![0.0f32; n_tokens * n_embd_gqa];
    let mut v_buf = vec![0.0f32; n_tokens * n_embd_gqa];
    let mut attn_out = vec![0.0f32; n_tokens * n_embd_q];
    let mut attn_proj = vec![0.0f32; n_tokens * n_embd];
    let mut normed = vec![0.0f32; n_tokens * n_embd];
    let mut gate_buf = vec![0.0f32; n_ff];
    let mut up_buf = vec![0.0f32; n_ff];
    let mut down_buf = vec![0.0f32; n_embd];
    let mut activation_scratch =
        EmbeddingActivationScratch::new(n_embd.max(n_embd_q).max(n_ff));
    let max_n_padded = (n_tokens + 255) / 256 * 256;
    let mut scores = vec![0.0f32; max_n_padded];
    let mut values = vec![0.0f32; max_n_padded];

    for t in 0..n_tokens {
        let token_id = prompt_tokens[t];
        let x_slice = &mut hidden[t * n_embd..(t + 1) * n_embd];
        embd_weight
            .get_row(token_id as usize, x_slice)
            .unwrap_or_else(|error| panic!("Failed to read embedding token row: {error}"));
    }

    let t_embed = Instant::now();
    for layer in 0..n_layer {
        let lw = &layers[layer];

        for t in 0..n_tokens {
            rms_norm(
                &hidden[t * n_embd..(t + 1) * n_embd],
                &lw.attn_norm,
                &mut normed[t * n_embd..(t + 1) * n_embd],
                eps,
            );
        }

        for t in 0..n_tokens {
            let x = &normed[t * n_embd..(t + 1) * n_embd];
            let q = &mut q_buf[t * n_embd_q..(t + 1) * n_embd_q];
            let k = &mut k_buf[t * n_embd_gqa..(t + 1) * n_embd_gqa];
            let v = &mut v_buf[t * n_embd_gqa..(t + 1) * n_embd_gqa];

            embedding_matmul_group(
                x,
                &mut [(&lw.wq, q), (&lw.wk, k), (&lw.wv, v)],
                &mut activation_scratch,
            )
            .unwrap_or_else(|error| panic!("Embedding Q/K/V matmul failed: {error}"));
        }

        if let (Some(qn), Some(kn)) = (&lw.q_norm, &lw.k_norm) {
            for t in 0..n_tokens {
                let q = &mut q_buf[t * n_embd_q..(t + 1) * n_embd_q];
                for h in 0..n_head {
                    rms_norm_inplace(&mut q[h * n_embd_head_k..(h + 1) * n_embd_head_k], qn, eps);
                }
            }
            for t in 0..n_tokens {
                let k = &mut k_buf[t * n_embd_gqa..(t + 1) * n_embd_gqa];
                for h in 0..n_head_kv {
                    rms_norm_inplace(&mut k[h * n_embd_head_k..(h + 1) * n_embd_head_k], kn, eps);
                }
            }
        }

        for t in embedding_positions(n_tokens) {
            let q = &mut q_buf[t * n_embd_q..(t + 1) * n_embd_q];
            for h in 0..n_head {
                rope_neox(
                    &mut q[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                    t,
                    n_embd_head_k,
                    freq_base,
                );
            }
        }
        for t in embedding_positions(n_tokens) {
            let k = &mut k_buf[t * n_embd_gqa..(t + 1) * n_embd_gqa];
            for h in 0..n_head_kv {
                rope_neox(
                    &mut k[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                    t,
                    n_embd_head_k,
                    freq_base,
                );
            }
        }

        for t in 0..n_tokens {
            let q_row = &q_buf[t * n_embd_q..(t + 1) * n_embd_q];
            let attn_row = &mut attn_out[t * n_embd_q..(t + 1) * n_embd_q];

            for h in 0..n_head {
                let kv_h = h / group_size;
                let q_off = h * n_embd_head_k;
                let out_base = h * n_embd_head_v;
                let n_cached = attention_key_end(t, n_tokens, embedding_cfg.causal_attn);
                let n_padded = (n_cached + 255) / 256 * 256;
                for s in 0..n_cached {
                    let k_row = &k_buf[s * n_embd_gqa..(s + 1) * n_embd_gqa];
                    scores[s] = dot_f32(
                        &q_row[q_off..q_off + n_embd_head_k],
                        &k_row[kv_h * n_embd_head_v..kv_h * n_embd_head_v + n_embd_head_k],
                        n_embd_head_k,
                    ) * kq_scale;
                }
                scores[n_cached..n_padded].fill(f32::NEG_INFINITY);
                softmax_inplace(&mut scores[..n_padded]);
                for d in 0..n_embd_head_v {
                    for s in 0..n_cached {
                        values[s] = v_buf[s * n_embd_gqa + kv_h * n_embd_head_v + d];
                    }
                    values[n_cached..n_padded].fill(0.0);
                    attn_row[out_base + d] = attention_value_f32(
                        &values[..n_padded],
                        &scores[..n_padded],
                        n_cached,
                        n_padded,
                    );
                }
            }
        }

        for t in 0..n_tokens {
            let attn = &attn_out[t * n_embd_q..(t + 1) * n_embd_q];
            let proj = &mut attn_proj[t * n_embd..(t + 1) * n_embd];

            embedding_matmul_group(
                attn,
                &mut [(&lw.wo, proj)],
                &mut activation_scratch,
            )
            .unwrap_or_else(|error| panic!("Embedding output matmul failed: {error}"));
        }

        for t in 0..n_tokens {
            let x = &mut hidden[t * n_embd..(t + 1) * n_embd];
            let proj = &attn_proj[t * n_embd..(t + 1) * n_embd];
            for i in 0..n_embd {
                x[i] += proj[i];
            }
        }

        for t in 0..n_tokens {
            rms_norm(
                &hidden[t * n_embd..(t + 1) * n_embd],
                &lw.ffn_norm,
                &mut normed[t * n_embd..(t + 1) * n_embd],
                eps,
            );
        }

        apply_embedding_ffn_typed(
            &mut hidden,
            &normed,
            n_embd,
            n_ff,
            &lw.w_gate,
            &lw.w_up,
            &lw.w_down,
            &mut gate_buf,
            &mut up_buf,
            &mut down_buf,
            &mut activation_scratch,
        )
        .unwrap_or_else(|error| panic!("Embedding FFN failed: {error}"));
    }

    for t in 0..n_tokens {
        let x = &mut hidden[t * n_embd..(t + 1) * n_embd];
        rms_norm(
            x,
            &output_norm,
            &mut normed[t * n_embd..(t + 1) * n_embd],
            eps,
        );
        x.copy_from_slice(&normed[t * n_embd..(t + 1) * n_embd]);
    }

    let mut pooled = pool_embedding_rows(&hidden, n_tokens, n_embd, embedding_cfg.pooling)
        .unwrap_or_else(|error| {
            eprintln!("Embedding pooling error: {error}");
            std::process::exit(1);
        });

    l2_normalize_embedding(&mut pooled).unwrap_or_else(|error| {
        eprintln!("Embedding normalization error: {error}");
        std::process::exit(1);
    });

    let embed_ms = t_embed.elapsed().as_millis();
    match output {
        EmbeddingOutput::Summary => {
            println!(
                "Embedding ({} dims, {} layers, {}ms):",
                n_embd, n_layer, embed_ms
            );
            for value in pooled.iter().take(8) {
                print!("{value:.9} ");
            }
            if n_embd > 8 {
                print!("... ");
                for value in &pooled[n_embd - 4..] {
                    print!("{value:.9} ");
                }
            }
            println!();
        }
        EmbeddingOutput::Raw => {
            print!("embedding_raw:");
            for value in &pooled {
                print!(" {value:.9}");
            }
            println!();
        }
    }
}
