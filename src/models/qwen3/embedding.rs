//! # Qwen3 Embedding
//!
//! 此模块包含 Qwen3 的文本 embedding 提取实现。

use crate::app::cli::EmbeddingOutput;
use crate::core::loader::model_config_from_source;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::qwen3::{get_f32_tensor, load_layers, Qwen3LayerWeights};
use crate::ops::kernel::Weight;
use crate::ops::quant::BlockQ8K;
use crate::ops::{
    attention_value_f32, dot_f32, embedding_lookup, f32_slice_to_f16, rms_norm, rms_norm_inplace,
    rope_neox, silu_mul_approx_inplace, softmax_inplace,
};
use std::sync::Arc;
use std::time::Instant;

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

#[derive(Clone, Copy)]
pub struct MediaEmbeddings<'a> {
    pub placeholder_id: u32,
    pub values: &'a [f32],
}

fn inject_media_embeddings(
    hidden: &mut [f32],
    token_ids: &[u32],
    n_embd: usize,
    media: Option<MediaEmbeddings<'_>>,
) -> Result<(), String> {
    if hidden.len() != token_ids.len().saturating_mul(n_embd) {
        return Err("Token embedding shape mismatch".into());
    }
    let Some(media) = media else {
        return Ok(());
    };
    if n_embd == 0 || media.values.len() % n_embd != 0 {
        return Err("Media embeddings are not row aligned".into());
    }
    let placeholders = token_ids
        .iter()
        .filter(|&&token| token == media.placeholder_id)
        .count();
    let rows = media.values.len() / n_embd;
    if placeholders != rows {
        return Err(format!(
            "Media row count mismatch: placeholders={placeholders}, rows={rows}"
        ));
    }
    let mut row = 0usize;
    for (token, output) in token_ids.iter().zip(hidden.chunks_exact_mut(n_embd)) {
        if *token == media.placeholder_id {
            output.copy_from_slice(&media.values[row * n_embd..(row + 1) * n_embd]);
            row += 1;
        }
    }
    Ok(())
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

/// FFN matmul for embedding extraction.
///
/// Uses `Weight::quantize_and_matmul_with_scratch` instead of the
/// `ExecutionScratchpad` + manual quantization path in base.rs.  This is
/// because embedding has no pre-allocated scratch buffers and each token
/// position processes independently — the "with scratch" helper is a natural
/// fit for the per-row quantization → matmul → silu → down pattern.
///
/// Architecture (per token row):
///   1. quantize_and_matmul(gate) → gate_buf   (quantizes input → matmul)
///   2. quantize_and_matmul(up)   → up_buf     (same quantized input, different weight)
///   3. silu_mul(gate_buf, up_buf)               (in-place SiLU gating)
///   4. quantize_and_matmul(down)  → down_buf   (quantizes gate*up result → matmul)
///   5. residual += down_buf
///
/// Contrast with base.rs: there the FFN gate+up share one quantized input and
/// run fused inside a single `pool.compute` call, then SILU, then down-projection
/// in a second `pool.compute` call — enabled by `ExecutionScratchpad`'s
/// pre-allocated reusable buffers.
#[allow(clippy::too_many_arguments)]
fn apply_embedding_ffn_typed(
    hidden: &mut [f32],
    normed: &[f32],
    n_embd: usize,
    n_ff: usize,
    w_gate: &Weight<'_>,
    w_up: &Weight<'_>,
    w_down: &Weight<'_>,
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    down_buf: &mut [f32],
    q8k_buf: &mut [BlockQ8K],
    q8_buf: &mut [u8],
    scale_buf: &mut [f32],
    pool: &ComputePool,
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
        w_gate.quantize_and_matmul_with_scratch(input, q8k_buf, q8_buf, scale_buf, gate_buf, pool);
        w_up.quantize_and_matmul_with_scratch(input, q8k_buf, q8_buf, scale_buf, up_buf, pool);

        silu_mul_approx_inplace(gate_buf, up_buf);

        w_down.quantize_and_matmul_with_scratch(up_buf, q8k_buf, q8_buf, scale_buf, down_buf, pool);

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

pub fn compute_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    let prompt_tokens = encode_embedding_input(&tokenizer, prompt);
    if prompt_tokens.is_empty() {
        return Err("Embedding input produced no tokens".into());
    }
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::token_ids(
        "embedding.tokens",
        &prompt_tokens,
    ));
    run_embedding_tokens(source, &prompt_tokens, None, n_threads_arg)
}

pub fn run_embedding_tokens(
    source: &dyn TensorSource,
    token_ids: &[u32],
    media: Option<MediaEmbeddings<'_>>,
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    if token_ids.is_empty() {
        return Err("Embedding input produced no tokens".into());
    }
    let config = model_config_from_source(source)?;

    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    let is_qwen3 = arch == "qwen3";
    let embedding_cfg = embedding_config(&arch, |key| source.metadata(key).cloned())?;

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

    let output_norm = get_f32_tensor(source, "output_norm.weight", n_embd);
    let embd_info = source
        .tensor_info("token_embd.weight")
        .ok_or("missing token_embd.weight")?;
    let embd_weight = source
        .tensor_slice("token_embd.weight")
        .ok_or("missing token_embd.weight")?;
    let embd_type = embd_info.ggml_type;

    let layers: Vec<Qwen3LayerWeights> = load_layers(
        source,
        n_layer,
        n_embd,
        n_embd_q,
        n_embd_gqa,
        n_ff,
        n_embd_head_k,
        is_qwen3,
    );

    let n_tokens = token_ids.len();
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let n_threads = crate::app::resolve_thread_count(n_threads_arg, available_threads);

    let pool = Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());
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
    let max_buf_size = n_embd.max(n_ff);
    let q8k_buf_size = max_buf_size.div_ceil(crate::ops::quant::QK_K);
    let q8_buf_size = max_buf_size;
    let scale_buf_size = max_buf_size.div_ceil(32);
    let mut q8k_buf = vec![
        BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16]
        };
        q8k_buf_size
    ];
    let mut q8_buf = vec![0u8; q8_buf_size];
    let mut scale_buf = vec![0.0f32; scale_buf_size];
    let max_n_padded = (n_tokens + 255) / 256 * 256;
    let mut scores = vec![0.0f32; max_n_padded];
    let mut values = vec![0.0f32; max_n_padded];

    for t in 0..n_tokens {
        let token_id = token_ids[t];
        let x_slice = &mut hidden[t * n_embd..(t + 1) * n_embd];
        embedding_lookup(embd_weight, token_id as u32, n_embd, embd_type, x_slice);
    }
    inject_media_embeddings(&mut hidden, token_ids, n_embd, media)?;

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

            lw.wq.quantize_and_matmul_with_scratch(
                x,
                &mut q8k_buf,
                &mut q8_buf,
                &mut scale_buf,
                q,
                &pool,
            );
            lw.wk.quantize_and_matmul_with_scratch(
                x,
                &mut q8k_buf,
                &mut q8_buf,
                &mut scale_buf,
                k,
                &pool,
            );
            lw.wv.quantize_and_matmul_with_scratch(
                x,
                &mut q8k_buf,
                &mut q8_buf,
                &mut scale_buf,
                v,
                &pool,
            );
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

            lw.wo.quantize_and_matmul_with_scratch(
                attn,
                &mut q8k_buf,
                &mut q8_buf,
                &mut scale_buf,
                proj,
                &pool,
            );
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
            &mut q8k_buf,
            &mut q8_buf,
            &mut scale_buf,
            &pool,
        )?;
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

    let mut pooled = pool_embedding_rows(&hidden, n_tokens, n_embd, embedding_cfg.pooling)?;
    l2_normalize_embedding(&mut pooled)?;
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(
        "embedding.final",
        None,
        &[n_embd],
        &pooled,
    ));
    Ok(pooled)
}

pub fn print_embedding(pooled: &[f32], output: EmbeddingOutput, elapsed_ms: u128) {
    match output {
        EmbeddingOutput::Summary => {
            println!("Embedding ({} dims, {}ms):", pooled.len(), elapsed_ms);
            for value in pooled.iter().take(8) {
                print!("{value:.9} ");
            }
            if pooled.len() > 8 {
                print!("... ");
                for value in &pooled[pooled.len() - 4..] {
                    print!("{value:.9} ");
                }
            }
            println!();
        }
        EmbeddingOutput::Raw => {
            print!("embedding_raw:");
            for value in pooled {
                print!(" {value:.9}");
            }
            println!();
        }
    }
}

pub fn run_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
    _kv_format: crate::app::cli::KvFormat,
    output: EmbeddingOutput,
) {
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    let config = model_config_from_source(source).expect("Failed to parse model config");
    let t0 = Instant::now();
    let pooled = match compute_embedding(source, prompt, n_threads_arg) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_millis();
    let n_embd = config.n_embd;
    let n_layer = config.n_layer;
    match output {
        EmbeddingOutput::Summary => {
            println!(
                "Embedding ({} dims, {} layers, arch={} {}ms):",
                n_embd, n_layer, arch, load_ms
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

#[cfg(test)]
mod tests {
    use super::*;

    struct EmptySource;

    impl TensorSource for EmptySource {
        fn metadata(&self, _key: &str) -> Option<&crate::core::tensor::MetaValue> {
            None
        }

        fn tensor_info(&self, _name: &str) -> Option<&crate::core::tensor::TensorInfo> {
            None
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    #[test]
    fn explicit_embedding_tokens_reject_empty_input_before_model_loading() {
        let error = run_embedding_tokens(&EmptySource, &[], None, 1).unwrap_err();
        assert_eq!(error, "Embedding input produced no tokens");
    }

    #[test]
    fn media_rows_replace_placeholder_tokens_in_order() {
        let tokens = [7, 99, 8, 99];
        let mut hidden = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        let media = MediaEmbeddings {
            placeholder_id: 99,
            values: &[10.0, 11.0, 20.0, 21.0],
        };

        inject_media_embeddings(&mut hidden, &tokens, 2, Some(media)).unwrap();

        assert_eq!(hidden, [1.0, 1.0, 10.0, 11.0, 3.0, 3.0, 20.0, 21.0]);
    }

    #[test]
    fn media_row_count_must_match_placeholders() {
        let mut hidden = vec![0.0; 6];
        let error = inject_media_embeddings(
            &mut hidden,
            &[99, 1, 99],
            2,
            Some(MediaEmbeddings {
                placeholder_id: 99,
                values: &[1.0, 2.0],
            }),
        )
        .unwrap_err();
        assert!(error.contains("placeholders=2"), "{error}");
    }

    #[test]
    fn last_pooling_and_l2_contract_are_shared_by_media_inputs() {
        let hidden = [9.0, 9.0, 3.0, 4.0];
        let mut pooled = pool_embedding_rows(&hidden, 2, 2, EmbeddingPooling::Last).unwrap();
        l2_normalize_embedding(&mut pooled).unwrap();
        assert_eq!(pooled[0].to_bits(), 0.6f32.to_bits());
        assert_eq!(pooled[1].to_bits(), 0.8f32.to_bits());
    }
}
