//! Forward pass and CLI entry point for `arch = "gemma-embedding"`.
//!
//! Every step below cites the oracle line that pins its semantics. Deviations
//! from the llama trunk path are deliberate and named in [`super`].

use crate::app::cli::EmbeddingOutput;
use crate::core::loader::model_config_from_source;
use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{EncodeOptions, SPMTokenizer};
use crate::ops::kernel::Weight;
use crate::ops::quant::BlockQ8K;
use crate::ops::{
    rope_neox_inplace,
    embedding_lookup, gelu_ggml_f16_inplace, softmax_inplace,

};
use std::sync::Arc;

use super::weights::{load_weights, GemmaEmbeddingWeights};

/// Fallback RMSNorm epsilon.
///
/// This GGUF reports `gemma-embedding.attention.layer_norm_rms_epsilon = 1e-6`
/// (note `_epsilon`, not `_eps`). We still clamp: a packer that omits the key
/// or writes a literal 0 would otherwise make RMSNorm divide by zero. Clamping
/// only affects the degenerate case — the non-zero path is a plain RMSNorm.
const EPS_FALLBACK: f32 = 1e-6;

fn safe_eps(eps: f32) -> f32 {
    if eps.is_finite() && eps > 0.0 {
        eps
    } else {
        EPS_FALLBACK
    }
}

#[derive(Clone, Copy, Debug)]
struct GemmaEmbeddingConfig {
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    n_head_kv: usize,
    n_embd_head: usize,
    n_embd_head_k: usize,
    n_embd_head_v: usize,
    n_ff: usize,
    eps: f32,
    freq_base: f32,
    freq_base_swa: f32,
    n_swa: usize,
    dense_2_feat_out: usize,
    dense_3_feat_in: usize,
}

fn read_meta(source: &dyn TensorSource) -> Result<GemmaEmbeddingConfig, String> {
    let arch = source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .unwrap_or_default()
        .to_string();
    if arch != "gemma-embedding" {
        return Err(format!("expected arch \"gemma-embedding\", got {arch:?}"));
    }
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let uint = |suffix: &str| {
        source
            .metadata(&key(suffix))
            .and_then(MetaValue::to_u64)
            .map(|value| value as usize)
    };
    let float = |suffix: &str| {
        source
            .metadata(&key(suffix))
            .and_then(|value| value.to_f64())
            .map(|value| value as f32)
    };
    let config = model_config_from_source(source)?;

    let n_embd = uint("embedding_length").ok_or("missing embedding_length")?;
    let n_head = uint("attention.head_count").ok_or("missing attention.head_count")?;
    let n_head_kv = uint("attention.head_count_kv").ok_or("missing attention.head_count_kv")?;
    let n_embd_head = if n_head > 0 { n_embd / n_head } else { 0 };
    let n_embd_head_k = uint("attention.key_length").unwrap_or(n_embd_head);
    let n_embd_head_v = uint("attention.value_length").unwrap_or(n_embd_head_k);
    if n_head == 0 || n_head_kv == 0 || n_head % n_head_kv != 0 {
        return Err(format!("invalid head layout: {n_head}/{n_head_kv}"));
    }
    if n_embd_head_k == 0 {
        return Err("attention.key_length is 0".into());
    }

    let base_freq = float("rope.freq_base").unwrap_or(config.rope_freq_base as f32);
    Ok(GemmaEmbeddingConfig {
        n_embd,
        n_layer: uint("block_count").ok_or("missing block_count")?,
        n_head,
        n_head_kv,
        n_embd_head,
        n_embd_head_k,
        n_embd_head_v,
        n_ff: uint("feed_forward_length").ok_or("missing feed_forward_length")?,
        // GGUF key is `layer_norm_rms_epsilon`; fall back to the parsed ModelConfig
        // norm_eps, then clamp, so a missing/0 key stays finite.
        eps: safe_eps(float("attention.layer_norm_rms_epsilon").unwrap_or(config.norm_eps)),
        freq_base: base_freq,
        freq_base_swa: float("rope.freq_base_swa").unwrap_or(base_freq),
        n_swa: uint("attention.sliding_window").unwrap_or(512),
        dense_2_feat_out: uint("dense_2_feat_out").ok_or("missing dense_2_feat_out")?,
        dense_3_feat_in: uint("dense_3_feat_in").ok_or("missing dense_3_feat_in")?,
    })
}

/// `is_swa(il) = (il % n_pattern < n_pattern - 1)` from `load_swa_pattern(ml, 6)`
/// with `dense_first = false` (`llama-model.cpp:3386-3391`). Layers with
/// `il % 6 == 5` are dense/global and use `rope.freq_base`; the rest use
/// `rope.freq_base_swa`.
const SWA_PATTERN: usize = 6;

fn is_swa(layer: usize) -> bool {
    SWA_PATTERN == 0 || layer % SWA_PATTERN < SWA_PATTERN - 1
}

/// Symmetric-window mask: key `p1` is masked for query `p0` when
/// `|p1 - p0| > n_swa / 2` (`llama-hparams.h:472-500`).
fn is_masked_symmetric(n_swa: usize, p0: usize, p1: usize) -> bool {
    let half = (n_swa / 2) as i64;
    let diff = p1 as i64 - p0 as i64;
    diff < -half || diff > half
}

fn rms_norm_row(src: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(src.len(), weight.len());
    let sum: f64 = src.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    let inv = 1.0f32 / (sum as f32 / src.len() as f32 + eps).sqrt();
    src.iter()
        .zip(weight)
        .map(|(value, scale)| *value * inv * *scale)
        .collect()
}

/// Mean-pool → `dense_2` → GELU → `dense_3`.
///
/// The oracle stops at `res->t_embd = build_norm(cur, output_norm, ...)`
/// (`gemma-embedding.cpp:170-175`) — pooling and the dense modules are added by
/// the sentence-transformers packing, which is what this GGUF's
/// `dense_2_feat_out` / `dense_3_feat_in` metadata advertises.
fn project_pooled(
    pooled: &[f32],
    dense_2: &Weight<'_>,
    dense_3: &Weight<'_>,
    q8k_buf: &mut [BlockQ8K],
    q8_buf: &mut [u8],
    scale_buf: &mut [f32],
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    if pooled.len() != dense_2.n_in || dense_2.n_out != dense_3.n_in {
        return Err(format!(
            "dense shape mismatch: pooled={} dense_2[{}→{}] dense_3[{}→{}]",
            pooled.len(),
            dense_2.n_in,
            dense_2.n_out,
            dense_3.n_in,
            dense_3.n_out,
        ));
    }
    let mut mid = vec![0.0f32; dense_2.n_out];
    dense_2.quantize_and_matmul_with_scratch(pooled, q8k_buf, q8_buf, scale_buf, &mut mid, pool);
    // Same op as the FFN gelu path (`ggml_vec_geglu_f32`), single-tensor form.
    gelu_ggml_f16_inplace(&mut mid);
    let mut out = vec![0.0f32; dense_3.n_out];
    dense_3.quantize_and_matmul_with_scratch(&mid, q8k_buf, q8_buf, scale_buf, &mut out, pool);
    Ok(out)
}

fn l2_normalize(values: &mut [f32]) -> Result<(), String> {
    let sum: f64 = values.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("embedding has zero or non-finite norm".into());
    }
    let scale = (1.0f64 / sum.sqrt()) as f32;
    for value in values.iter_mut() {
        *value *= scale;
    }
    Ok(())
}

pub fn compute_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    // `tokenizer.ggml.model = "llama"` → SentencePiece vocab, which is the
    // only tokenizer family llama.cpp maps `LLM_ARCH_GEMMA_EMBEDDING` onto.
    // EmbeddingGemma ships `add_bos_token = false`, so no BOS is injected.
    let tokenizer = SPMTokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    let prompt_tokens = tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    if prompt_tokens.is_empty() {
        return Err("Embedding input produced no tokens".into());
    }
    run_embedding_tokens(source, &prompt_tokens, n_threads_arg)
}

pub fn run_embedding_tokens(
    source: &dyn TensorSource,
    token_ids: &[u32],
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    if token_ids.is_empty() {
        return Err("Embedding input produced no tokens".into());
    }
    let cfg = read_meta(source)?;
    let (n_embd, n_layer, n_head, n_head_kv) = (cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.n_head_kv);
    let (n_embd_head_k, n_embd_head_v, n_ff) = (cfg.n_embd_head_k, cfg.n_embd_head_v, cfg.n_ff);
    let n_embd_q = n_head * n_embd_head_k;
    let n_embd_gqa = n_head_kv * n_embd_head_v;
    let group_size = n_head / n_head_kv;
    let kq_scale = 1.0f32 / (n_embd_head_k as f32).sqrt();
    // Token embeddings are scaled by sqrt(n_embd) before layer 0
    // (`gemma-embedding.cpp:89`).
    let embd_scale = (n_embd as f32).sqrt();

    let weights: GemmaEmbeddingWeights<'_> = load_weights(
        source,
        n_layer,
        n_embd,
        n_embd_q,
        n_embd_gqa,
        n_embd_head_k,
        n_ff,
    );
    if weights.n_layer != n_layer {
        return Err("layer count mismatch between metadata and tensors".into());
    }

    let available = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(4);
    let n_threads = crate::app::resolve_thread_count(n_threads_arg, available);
    let pool = Arc::new(ComputePool::new(n_threads));

    let n_tokens = token_ids.len();
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

    let max_width = n_embd
        .max(n_ff)
        .max(cfg.dense_2_feat_out)
        .max(cfg.dense_3_feat_in);
    let mut q8k_buf = vec![
        BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        };
        max_width.div_ceil(crate::ops::quant::QK_K)
    ];
    let mut q8_buf = vec![0u8; max_width];
    let mut scale_buf = vec![0.0f32; max_width.div_ceil(32)];

    for (token, row) in token_ids.iter().zip(hidden.chunks_exact_mut(n_embd)) {
        embedding_lookup(
            weights.token_embd,
            *token,
            n_embd,
            weights.token_embd_ggml_type,
            row,
        );
        for value in row.iter_mut() {
            *value *= embd_scale;
        }
    }

    for layer in 0..n_layer {
        let lw = &weights.layers[layer];
        let layer_is_swa = is_swa(layer);
        let freq_base = if layer_is_swa {
            cfg.freq_base_swa
        } else {
            cfg.freq_base
        };

        // 1. pre-attention norm
        for t in 0..n_tokens {
            let row = &hidden[t * n_embd..(t + 1) * n_embd];
            normed[t * n_embd..(t + 1) * n_embd].copy_from_slice(&rms_norm_row(
                row,
                &lw.attn_norm,
                cfg.eps,
            ));
        }

        // 2. Q / K / V projections — each re-quantises the same normed row, so
        //    the three calls share one Q8_0/Q8_K encoding of the input.
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

        // 3. QK norm — per head, over head_dim, **before** RoPE
        //    (`gemma-embedding.cpp:105-118`).
        for t in 0..n_tokens {
            let q = &mut q_buf[t * n_embd_q..(t + 1) * n_embd_q];
            for h in 0..n_head {
                let lane = &mut q[h * n_embd_head_k..(h + 1) * n_embd_head_k];
                lane.copy_from_slice(&rms_norm_row(lane, &lw.q_norm, cfg.eps));
            }
        }
        for t in 0..n_tokens {
            let k = &mut k_buf[t * n_embd_gqa..(t + 1) * n_embd_gqa];
            for h in 0..n_head_kv {
                let lane = &mut k[h * n_embd_head_k..(h + 1) * n_embd_head_k];
                lane.copy_from_slice(&rms_norm_row(lane, &lw.k_norm, cfg.eps));
            }
        }

        // 4. RoPE — NEOX layout (matches `LLM_ARCH_GEMMA_EMBEDDING` in
        //    llama.cpp's rope_type table), per-layer freq_base.
        for t in 0..n_tokens {
            let q = &mut q_buf[t * n_embd_q..(t + 1) * n_embd_q];
            for h in 0..n_head {
                rope_neox_inplace(
                    &mut q[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                    t,
                    n_embd_head_k,
                    freq_base);
            }
        }
        for t in 0..n_tokens {
            let k = &mut k_buf[t * n_embd_gqa..(t + 1) * n_embd_gqa];
            for h in 0..n_head_kv {
                rope_neox_inplace(
                    &mut k[h * n_embd_head_k..(h + 1) * n_embd_head_k],
                    t,
                    n_embd_head_k,
                    freq_base);
            }
        }

        // 5. bidirectional attention. `kq_scale` is the oracle's
        //    `f_attention_scale = 1/sqrt(n_embd_head_k)` applied to Q after
        //    RoPE (`gemma-embedding.cpp:27,122`); folding it into the score is
        //    algebraically identical and saves a buffer.
        for t in 0..n_tokens {
            let attn_row = &mut attn_out[t * n_embd_q..(t + 1) * n_embd_q];
            for h in 0..n_head {
                let kv_h = h / group_size;
                let q_off = h * n_embd_head_k;
                let out_base = h * n_embd_head_v;
                let mut scores = vec![0.0f32; n_tokens];
                for s in 0..n_tokens {
                    let q_row = &q_buf[t * n_embd_q..(t + 1) * n_embd_q];
                    let k_row = &k_buf[s * n_embd_gqa..(s + 1) * n_embd_gqa];
                    let dot: f32 = q_row[q_off..q_off + n_embd_head_k]
                        .iter()
                        .zip(&k_row[kv_h * n_embd_head_v..kv_h * n_embd_head_v + n_embd_head_k])
                        .map(|(a, b)| a * b)
                        .sum();
                    scores[s] = if layer_is_swa && is_masked_symmetric(cfg.n_swa, t, s) {
                        f32::NEG_INFINITY
                    } else {
                        dot * kq_scale
                    };
                }
                softmax_inplace(&mut scores);
                for d in 0..n_embd_head_v {
                    let mut acc = 0.0f32;
                    for s in 0..n_tokens {
                        let v_row = &v_buf[s * n_embd_gqa..(s + 1) * n_embd_gqa];
                        acc += v_row[kv_h * n_embd_head_v + d] * scores[s];
                    }
                    attn_row[out_base + d] = acc;
                }
            }
        }

        // 6. attention output projection
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

        // 7. post-attention norm + residual (`cur + inpL`)
        for t in 0..n_tokens {
            let proj = &attn_proj[t * n_embd..(t + 1) * n_embd];
            let normed_attn = rms_norm_row(proj, &lw.attn_post_norm, cfg.eps);
            let x = &mut hidden[t * n_embd..(t + 1) * n_embd];
            for i in 0..n_embd {
                x[i] += normed_attn[i];
            }
        }

        // 8. pre-FFN norm
        for t in 0..n_tokens {
            let row = &hidden[t * n_embd..(t + 1) * n_embd];
            normed[t * n_embd..(t + 1) * n_embd].copy_from_slice(&rms_norm_row(
                row,
                &lw.ffn_norm,
                cfg.eps,
            ));
        }

        // 9. geglu FFN: gelu(gate) * up, then 10. post-FFN norm + residual.
        for t in 0..n_tokens {
            let x = &normed[t * n_embd..(t + 1) * n_embd];
            lw.w_gate.quantize_and_matmul_with_scratch(
                x,
                &mut q8k_buf,
                &mut q8_buf,
                &mut scale_buf,
                &mut gate_buf,
                &pool,
            );
            lw.w_up.quantize_and_matmul_with_scratch(
                x,
                &mut q8k_buf,
                &mut q8_buf,
                &mut scale_buf,
                &mut up_buf,
                &pool,
            );
            gelu_ggml_f16_inplace(&mut gate_buf);
            for (gate, up) in gate_buf.iter_mut().zip(up_buf.iter()) {
                *gate *= *up;
            }
            lw.w_down.quantize_and_matmul_with_scratch(
                &gate_buf,
                &mut q8k_buf,
                &mut q8_buf,
                &mut scale_buf,
                &mut down_buf,
                &pool,
            );
            let normed_ffn = rms_norm_row(&down_buf, &lw.ffn_post_norm, cfg.eps);
            let x = &mut hidden[t * n_embd..(t + 1) * n_embd];
            for i in 0..n_embd {
                x[i] += normed_ffn[i];
            }
        }
    }

    // 11. final norm → mean pool → dense_2/dense_3 → L2 normalise
    let mut pooled = vec![0.0f32; n_embd];
    for t in 0..n_tokens {
        let row = &hidden[t * n_embd..(t + 1) * n_embd];
        let normed_final = rms_norm_row(row, &weights.output_norm, cfg.eps);
        for i in 0..n_embd {
            pooled[i] += normed_final[i];
        }
    }
    let inv_tokens = 1.0f32 / n_tokens as f32;
    for value in pooled.iter_mut() {
        *value *= inv_tokens;
    }

    let mut embedding = project_pooled(
        &pooled,
        &weights.dense_2,
        &weights.dense_3,
        &mut q8k_buf,
        &mut q8_buf,
        &mut scale_buf,
        &pool,
    )?;
    l2_normalize(&mut embedding)?;
    Ok(embedding)
}

pub fn print_embedding(pooled: &[f32], output: EmbeddingOutput) {
    match output {
        EmbeddingOutput::Summary => {
            print!("Embedding ({} dims):", pooled.len());
            for value in pooled.iter().take(8) {
                print!(" {value:.9}");
            }
            if pooled.len() > 8 {
                print!(" ...");
                for value in &pooled[pooled.len() - 4..] {
                    print!(" {value:.9}");
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
    let pooled = match compute_embedding(source, prompt, n_threads_arg) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    print_embedding(&pooled, output);
}
