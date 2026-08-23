use crate::core::tensor::{GGMLType, TensorSource};
use crate::ops::{dot_f32, f16_to_f32, matmul_q8_0_quantized, quantize_q8_0_into, rms_norm, rms_norm_inplace, silu_mul_approx_inplace};
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::app::cli::EmbeddingOutput;
use crate::core::thread_pool::ComputePool;
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

struct EmbeddingLayerWeights<'a> {
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
    let config = crate::core::loader::model_config_from_source(source).expect("Failed to parse model config");

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
    let n_embd_head_k = if let Some(v) = source.metadata(&format!("{}.attention.key_length", arch))
    {
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

    let output_norm = crate::app::get_f32_tensor(source, "output_norm.weight", n_embd);
    let embd_weight = EmbeddingWeight::load(source, "token_embd.weight", n_embd, tokenizer.vocab_size())
        .unwrap_or_else(|error| panic!("Failed to load embedding token weights: {error}"));

    let layers: Vec<EmbeddingLayerWeights> = (0..n_layer)
        .map(|l| EmbeddingLayerWeights {
            attn_norm: crate::app::get_f32_tensor(source, &format!("blk.{}.attn_norm.weight", l), n_embd),
            ffn_norm: crate::app::get_f32_tensor(source, &format!("blk.{}.ffn_norm.weight", l), n_embd),
            q_norm: if is_qwen3 {
                Some(crate::app::get_f32_tensor(
                    source,
                    &format!("blk.{}.attn_q_norm.weight", l),
                    n_embd_head_k,
                ))
            } else {
                None
            },
            k_norm: if is_qwen3 {
                Some(crate::app::get_f32_tensor(
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
                crate::ops::rope_neox(
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
                crate::ops::rope_neox(
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
                crate::ops::softmax_inplace(&mut scores[..n_padded]);
                for d in 0..n_embd_head_v {
                    for s in 0..n_cached {
                        values[s] = v_buf[s * n_embd_gqa + kv_h * n_embd_head_v + d];
                    }
                    values[n_cached..n_padded].fill(0.0);
                    attn_row[out_base + d] = crate::ops::attention_value_f32(
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
                print!("{value:+.6} ");
            }
            if n_embd > 8 {
                print!("... ");
                for value in &pooled[n_embd - 4..] {
                    print!("{value:+.6} ");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::ggufrs::{open_model_source, ComponentRole};
    use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
    use std::collections::HashMap;
    use std::path::Path;

    struct TestTensorSource {
        info: TensorInfo,
        bytes: Vec<u8>,
    }

    impl TensorSource for TestTensorSource {
        fn metadata(&self, _key: &str) -> Option<&MetaValue> {
            None
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            (name == self.info.name).then_some(&self.info)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            (name == self.info.name).then_some(&self.bytes)
        }
    }

    #[test]
    fn f16_embedding_rows_decode_little_endian_half_values() {
        let source = TestTensorSource {
            info: TensorInfo {
                name: "token_embd.weight".into(),
                dims: vec![4, 1],
                ggml_type: GGMLType::F16,
                offset: 0,
            },
            bytes: [0x00, 0x3c, 0x00, 0xc0, 0x55, 0x35, 0x00, 0x00].to_vec(),
        };

        let weight = EmbeddingWeight::load(&source, "token_embd.weight", 4, 1).unwrap();
        let mut row = [0.0; 4];
        weight.get_row(0, &mut row).unwrap();
        assert_eq!(row, [1.0, -2.0, 0.333_251_95, 0.0]);
    }

    #[test]
    #[ignore = "environment-specific floating point precision (compiler optimization difference)"]
    fn f16_embedding_matmul_uses_ggml_fp16_vector_accumulation() {
        let bytes = half::f16::from_f32(0.1).to_bits().to_le_bytes();
        let source = TestTensorSource {
            info: TensorInfo {
                name: "weight".into(),
                dims: vec![32, 1],
                ggml_type: GGMLType::F16,
                offset: 0,
            },
            bytes: bytes.repeat(32),
        };
        let weight = EmbeddingWeight::load(&source, "weight", 32, 1).unwrap();
        let mut scratch = EmbeddingActivationScratch::new(32);
        let activation = scratch.prepare(&weight, &[0.1; 32]).unwrap();
        let mut output = [0.0];

        weight.matmul_prepared(&activation, &mut output).unwrap();

        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        let expected = if std::arch::is_aarch64_feature_detected!("fp16") {
            0x3ea3_c000
        } else {
            0x3ea3_c28e
        };
        #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
        let expected = 0x3ea3_c28e;
        assert_eq!(output[0].to_bits(), expected);
    }

    #[test]
    fn q8_embedding_matmul_uses_the_existing_quantized_kernel() {
        let source = TestTensorSource {
            info: TensorInfo {
                name: "weight".into(),
                dims: vec![32, 1],
                ggml_type: GGMLType::Q8_0,
                offset: 0,
            },
            bytes: [half::f16::from_f32(1.0).to_bits().to_le_bytes().as_slice(), &[1; 32]].concat(),
        };
        let weight = EmbeddingWeight::load(&source, "weight", 32, 1).unwrap();
        let mut scratch = EmbeddingActivationScratch::new(32);
        let activation = scratch.prepare(&weight, &[1.0; 32]).unwrap();
        let mut output = [0.0];

        weight.matmul_prepared(&activation, &mut output).unwrap();

        assert_eq!(output, [31.998_047]);
    }

    #[test]
    fn prepared_embedding_activation_is_reused_across_projections() {
        let f16_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes().repeat(32);
        let f16 = EmbeddingWeight {
            bytes: &f16_bytes,
            ggml_type: GGMLType::F16,
            n_cols: 32,
            n_rows: 1,
        };
        let q8_bytes = [
            half::f16::from_f32(1.0).to_bits().to_le_bytes().as_slice(),
            &[1; 32],
        ]
        .concat();
        let q8 = EmbeddingWeight {
            bytes: &q8_bytes,
            ggml_type: GGMLType::Q8_0,
            n_cols: 32,
            n_rows: 1,
        };
        let mut scratch = EmbeddingActivationScratch::new(32);
        let input = [1.0; 32];

        let f16_activation = scratch.prepare(&f16, &input).unwrap();
        let f16_ptr = f16_activation.f16.as_ptr();
        f16.matmul_prepared(&f16_activation, &mut [0.0]).unwrap();
        let f16_activation = scratch.prepare(&f16, &input).unwrap();
        assert_eq!(f16_activation.f16.as_ptr(), f16_ptr);

        let q8_activation = scratch.prepare(&q8, &input).unwrap();
        let q8_ptr = q8_activation.q8.as_ptr();
        q8.matmul_prepared(&q8_activation, &mut [0.0]).unwrap();
        let q8_activation = scratch.prepare(&q8, &input).unwrap();
        assert_eq!(q8_activation.q8.as_ptr(), q8_ptr);
    }

    #[test]
    fn embedding_weight_rejects_invalid_type_shape_length_and_row() {
        let invalid_type = TestTensorSource {
            info: TensorInfo {
                name: "weight".into(),
                dims: vec![4, 1],
                ggml_type: GGMLType::F32,
                offset: 0,
            },
            bytes: vec![0; 16],
        };
        assert!(EmbeddingWeight::load(&invalid_type, "weight", 4, 1)
            .unwrap_err()
            .contains("unsupported type"));

        let wrong_shape = TestTensorSource {
            info: TensorInfo {
                name: "weight".into(),
                dims: vec![2, 2],
                ggml_type: GGMLType::F16,
                offset: 0,
            },
            bytes: vec![0; 8],
        };
        assert!(EmbeddingWeight::load(&wrong_shape, "weight", 4, 1)
            .unwrap_err()
            .contains("shape"));

        let wrong_length = TestTensorSource {
            info: TensorInfo {
                name: "weight".into(),
                dims: vec![4, 1],
                ggml_type: GGMLType::F16,
                offset: 0,
            },
            bytes: vec![0; 7],
        };
        assert!(EmbeddingWeight::load(&wrong_length, "weight", 4, 1)
            .unwrap_err()
            .contains("expected 8"));

        let valid = TestTensorSource {
            info: TensorInfo {
                name: "weight".into(),
                dims: vec![4, 1],
                ggml_type: GGMLType::F16,
                offset: 0,
            },
            bytes: vec![0; 8],
        };
        let weight = EmbeddingWeight::load(&valid, "weight", 4, 1).unwrap();
        assert!(weight.get_row(1, &mut [0.0; 4]).unwrap_err().contains("out of range"));
    }

    fn tiny_embedding_tokenizer() -> BPETokenizer {
        let metadata: HashMap<String, MetaValue> = HashMap::from([
            (
                "tokenizer.ggml.model".into(),
                MetaValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.pre".into(),
                MetaValue::String("qwen2".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                MetaValue::Array(
                    MetaValueType::String,
                    ["h", "e", "l", "o", "<|endoftext|>"]
                        .into_iter()
                        .map(|value| MetaValue::String(value.into()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                MetaValue::Array(
                    MetaValueType::Uint32,
                    [1, 1, 1, 1, 3].into_iter().map(MetaValue::Uint32).collect(),
                ),
            ),
            (
                "tokenizer.ggml.merges".into(),
                MetaValue::Array(MetaValueType::String, vec![]),
            ),
            ("tokenizer.ggml.bos_token_id".into(), MetaValue::Uint32(0)),
            ("tokenizer.ggml.eos_token_id".into(), MetaValue::Uint32(4)),
            (
                "tokenizer.ggml.add_bos_token".into(),
                MetaValue::Bool(false),
            ),
            ("tokenizer.ggml.add_eos_token".into(), MetaValue::Bool(true)),
        ]);

        BPETokenizer::from_gguf_metadata(|key| metadata.get(key).cloned()).unwrap()
    }

    fn q8_identity(size: usize) -> Vec<u8> {
        assert_eq!(size % 32, 0);
        let blocks_per_row = size / 32;
        let row_stride = blocks_per_row * 34;
        let mut weight = vec![0u8; size * row_stride];

        for row in 0..size {
            let block = row / 32;
            let lane = row % 32;
            let offset = row * row_stride + block * 34;
            weight[offset..offset + 2]
                .copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
            weight[offset + 2 + lane] = 1;
        }
        weight
    }

    #[test]
    fn embedding_ffn_keeps_each_tokens_projection_independent() {
        let identity = q8_identity(32);
        let weight = EmbeddingWeight {
            bytes: &identity,
            ggml_type: GGMLType::Q8_0,
            n_cols: 32,
            n_rows: 32,
        };
        let mut normed = vec![0.0f32; 64];
        normed[0] = 1.0;
        normed[33] = 2.0;

        let mut hidden = vec![0.0f32; 64];
        hidden[0] = 10.0;
        hidden[33] = 20.0;

        apply_embedding_ffn_typed(
            &mut hidden,
            &normed,
            32,
            32,
            &weight,
            &weight,
            &weight,
            &mut [0.0; 32],
            &mut [0.0; 32],
            &mut [0.0; 32],
            &mut EmbeddingActivationScratch::new(32),
        )
        .unwrap();

        assert!((hidden[0] - 10.731059).abs() < 1e-4, "{}", hidden[0]);
        assert!((hidden[33] - 23.523041).abs() < 1e-4, "{}", hidden[33]);
        assert_eq!(hidden[1], 0.0);
        assert_eq!(hidden[32], 0.0);
    }

    const EMBEDDING_TOKEN_CASES: &[(&str, &[u32])] = &[
        ("hello", &[14990, 151643]),
        (
            "Hello, 世界! 123",
            &[9707, 11, 220, 99489, 0, 220, 16, 17, 18, 151643],
        ),
        (
            "What is the capital of China?",
            &[3838, 374, 279, 6722, 315, 5616, 30, 151643],
        ),
        (
            "The capital of China is Beijing.",
            &[785, 6722, 315, 5616, 374, 26549, 13, 151643],
        ),
        (
            "Photosynthesis converts light into chemical energy.",
            &[31772, 73667, 32722, 3100, 1119, 11483, 4802, 13, 151643],
        ),
        (
            "中国的首都是北京。",
            &[105538, 59975, 100132, 68990, 1773, 151643],
        ),
    ];

    #[test]
    fn embedding_input_honors_tokenizer_eos_metadata() {
        assert_eq!(
            encode_embedding_input(&tiny_embedding_tokenizer(), "hello"),
            vec![0, 1, 2, 2, 3, 4],
        );
    }

    #[test]
    fn embedding_config_defaults_to_causal_and_reads_last_pooling() {
        let metadata = HashMap::from([("qwen3.pooling_type".to_string(), MetaValue::Uint32(3))]);

        assert_eq!(
            embedding_config("qwen3", |key| metadata.get(key).cloned()).unwrap(),
            EmbeddingConfig {
                causal_attn: true,
                pooling: EmbeddingPooling::Last,
            },
        );
    }

    #[test]
    fn embedding_config_reads_mean_and_non_causal_metadata() {
        let metadata = HashMap::from([
            ("qwen3.pooling_type".to_string(), MetaValue::Uint32(1)),
            ("qwen3.attention.causal".to_string(), MetaValue::Bool(false)),
        ]);

        assert_eq!(
            embedding_config("qwen3", |key| metadata.get(key).cloned()).unwrap(),
            EmbeddingConfig {
                causal_attn: false,
                pooling: EmbeddingPooling::Mean,
            },
        );
    }

    #[test]
    fn causal_embedding_attention_never_reads_future_keys() {
        assert_eq!(
            (0..3)
                .map(|query| attention_key_end(query, 3, true))
                .collect::<Vec<_>>(),
            vec![1, 2, 3],
        );
        assert_eq!(attention_key_end(0, 3, false), 3);
    }

    #[test]
    fn embedding_positions_are_contiguous_from_zero() {
        assert_eq!(embedding_positions(4).collect::<Vec<_>>(), vec![0, 1, 2, 3],);
    }

    #[test]
    fn embedding_config_rejects_missing_malformed_or_unsupported_pooling() {
        assert!(embedding_config("qwen3", |_| None)
            .unwrap_err()
            .contains("qwen3.pooling_type"));

        let error = embedding_config("qwen3", |key| match key {
            "qwen3.pooling_type" => Some(MetaValue::Bool(true)),
            _ => None,
        })
        .unwrap_err();
        assert!(error.contains("qwen3.pooling_type"), "{error}");

        let error = embedding_config("qwen3", |key| match key {
            "qwen3.pooling_type" => Some(MetaValue::Uint32(2)),
            _ => None,
        })
        .unwrap_err();
        assert!(error.contains("expected 1=MEAN or 3=LAST"), "{error}");
    }

    #[test]
    fn embedding_config_rejects_non_boolean_causal_metadata() {
        let metadata = HashMap::from([
            ("qwen3.pooling_type".to_string(), MetaValue::Uint32(3)),
            ("qwen3.attention.causal".to_string(), MetaValue::Uint32(1)),
        ]);

        let error = embedding_config("qwen3", |key| metadata.get(key).cloned()).unwrap_err();
        assert!(error.contains("expected bool"), "{error}");
    }

    #[test]
    fn embedding_pooling_supports_mean_and_last() {
        let hidden = [1.0, 2.0, 5.0, 6.0];
        assert_eq!(
            pool_embedding_rows(&hidden, 2, 2, EmbeddingPooling::Mean).unwrap(),
            vec![3.0, 4.0],
        );
        assert_eq!(
            pool_embedding_rows(&hidden, 2, 2, EmbeddingPooling::Last).unwrap(),
            vec![5.0, 6.0],
        );
    }

    #[test]
    fn embedding_pooling_rejects_invalid_shapes() {
        assert!(pool_embedding_rows(&[], 0, 2, EmbeddingPooling::Last).is_err());
        assert!(pool_embedding_rows(&[1.0], 1, 2, EmbeddingPooling::Last).is_err());
    }

    #[test]
    fn embedding_l2_uses_f64_accumulation_and_preserves_zero() {
        let mut values = vec![1.0f32];
        values.extend(std::iter::repeat(1e-4f32).take(4096));
        l2_normalize_embedding(&mut values).unwrap();
        assert!((values[0] - 0.9999795).abs() < 1e-6, "{}", values[0]);

        let mut zero = [0.0f32, 0.0];
        l2_normalize_embedding(&mut zero).unwrap();
        assert_eq!(zero, [0.0, 0.0]);
    }

    #[test]
    fn embedding_l2_matches_llama_f32_product_and_scale_bits() {
        let mut values = [1.0f32, 3.0];
        l2_normalize_embedding(&mut values).unwrap();
        assert_eq!(
            values.map(f32::to_bits),
            [0x3ea1e89b, 0x3f72dce8],
        );
    }

    #[test]
    fn embedding_l2_matches_llama_subnormal_underflow_to_zero() {
        let mut values = [f32::from_bits(1)];
        l2_normalize_embedding(&mut values).unwrap();
        assert_eq!(values, [0.0]);
    }

    #[test]
    fn embedding_l2_rejects_non_finite_values() {
        for value in [f32::INFINITY, f32::NAN] {
            let mut values = [value];
            assert!(l2_normalize_embedding(&mut values).is_err());
        }
    }

    #[test]
    #[ignore = "requires QWEN3_EMBEDDING_MODEL"]
    fn qwen3_embedding_tokens_match_pinned_llama_cpp() {
        let model = std::env::var("QWEN3_EMBEDDING_MODEL").unwrap();
        let source = open_model_source(Path::new(&model), ComponentRole::Llm).unwrap();
        let tokenizer =
            BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();

        for &(text, expected) in EMBEDDING_TOKEN_CASES {
            assert_eq!(
                tokenizer.encode(
                    text,
                    EncodeOptions {
                        add_special: true,
                        parse_special: true,
                    },
                ),
                expected,
                "{text:?}",
            );
        }
    }
}
