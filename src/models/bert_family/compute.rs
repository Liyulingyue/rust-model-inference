//! Forward pass and CLI entry point for the BERT encoder family.
//!
//! Graph order follows `references/llama.cpp/src/models/bert.cpp:70-204`.
//! Every step that differs between the `bert` and `jina-bert-v2` variants is
//! gated on [`BertVariant`] with the oracle line cited inline.

use crate::app::cli::{EmbeddingOutput, KvFormat};
use crate::core::loader::model_config_from_source;
use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{EncodeOptions, WPMTokenizer};
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::{embedding_lookup, gelu_ggml_f16_inplace, layer_norm, rope_norm, softmax_inplace};
use std::sync::Arc;

use super::weights::{load_weights, BertVariant, BertWeights, MAX_ALIBI_BIAS_JINA_V2};

#[derive(Clone, Copy, Debug)]
struct BertConfig {
    variant: BertVariant,
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    n_head_kv: usize,
    n_embd_head: usize,
    n_embd_head_k: usize,
    n_embd_head_v: usize,
    n_ff: usize,
    eps: f32,
    /// `rope.freq_base`. Only meaningful for `nomic-bert` (the one variant that
    /// ropes): it trains at 1000 Hz, not the 10 000 Hz llama default.
    rope_freq_base: f32,
    /// `pooling_type` (`llama.h:177-182`): 0 none, 1 mean, 2 CLS, 3 last.
    /// `bert` ships 2, while jina-bert-v2 and nomic-bert ship 1.
    pooling_type: u64,
}

fn read_meta(source: &dyn TensorSource) -> Result<BertConfig, String> {
    let arch = source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .unwrap_or_default()
        .to_string();
    let variant = BertVariant::from_arch(&arch)
        .ok_or_else(|| format!("unsupported encoder architecture {arch:?}"))?;
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let uint = |suffix: &str| {
        source
            .metadata(&key(suffix))
            .and_then(MetaValue::to_u64)
            .map(|value| value as usize)
    };
    let uint64 = |suffix: &str| source.metadata(&key(suffix)).and_then(MetaValue::to_u64);
    let float = |suffix: &str| {
        source
            .metadata(&key(suffix))
            .and_then(|value| value.to_f64())
            .map(|value| value as f32)
    };
    let config = model_config_from_source(source)?;

    let n_embd = uint("embedding_length").ok_or("missing embedding_length")?;
    let n_head = uint("attention.head_count").ok_or("missing attention.head_count")?;
    if n_head == 0 {
        return Err("attention.head_count is 0".into());
    }
    // Standard BERT is plain MHA and the converted GGUF omits
    // `attention.head_count_kv`, so the KV width equals the Q width.
    let n_head_kv = uint("attention.head_count_kv").unwrap_or(n_head);
    let head_from_count = n_embd / n_head;
    let n_embd_head = uint("attention.head_count")
        .map(|_| head_from_count)
        .unwrap_or(0);
    let n_embd_head_k = uint("attention.key_length").unwrap_or(head_from_count);
    let n_embd_head_v = uint("attention.value_length").unwrap_or(head_from_count);
    if n_embd_head_k == 0 || n_embd_head_v == 0 {
        return Err("attention key/value length is 0".into());
    }
    // `bert.cpp:68` asserts head_k == head_v.
    if n_embd_head_k != n_embd_head_v {
        return Err(format!(
            "encoder requires head_k == head_v, got {n_embd_head_k}/{n_embd_head_v}"
        ));
    }

    Ok(BertConfig {
        variant,
        n_embd,
        n_layer: uint("block_count").ok_or("missing block_count")?,
        n_head,
        n_head_kv,
        n_embd_head,
        n_embd_head_k,
        n_embd_head_v,
        n_ff: uint("feed_forward_length").ok_or("missing feed_forward_length")?,
        // BERT uses `attention.layer_norm_epsilon`; jina-bert-v2 ships 1e-12.
        eps: float("attention.layer_norm_epsilon").unwrap_or(1e-12),
        // `llama-model.cpp:1410` defaults to 10 000 Hz; nomic-bert overrides it
        // with 1000 via `nomic-bert.rope.freq_base`. Only the roped variant
        // reads this, but loading it unconditionally keeps the config honest.
        rope_freq_base: float("rope.freq_base").unwrap_or(10_000.0),
        // 1 (mean) is the safe default: every variant verified before bge-small
        // used it, and `bert` is the one that passes 2 explicitly.
        pooling_type: uint64("pooling_type").unwrap_or(1),
    })
    .and_then(|cfg| {
        if cfg.pooling_type > 3 {
            return Err(format!("unsupported pooling_type {}", cfg.pooling_type));
        }
        Ok(cfg)
    })
}

/// Per-head ALiBi slope, exactly as `soft_max_ext` computes it
/// (`ggml-cpu/ops.cpp:8944`):
///
/// ```text
/// n_head_log2 = 1 << floor(log2(n_head))
/// m0 = 2^(-max_bias / n_head_log2)
/// m1 = 2^(-(max_bias / 2) / n_head_log2)
/// slope(h) = h < n_head_log2 ? powf(m0, h+1) : powf(m1, 2*(h-n_head_log2)+1)
/// ```
fn alibi_slopes(n_head: usize, max_bias: f32) -> Vec<f32> {
    if max_bias <= 0.0 {
        return vec![1.0; n_head];
    }
    let n_head_log2 = 1usize << (n_head as f32).log2().floor() as usize;
    let n_head_log2 = n_head_log2.max(1);
    let m0 = 2.0f32.powf(-max_bias / n_head_log2 as f32);
    let m1 = 2.0f32.powf(-(max_bias / 2.0) / n_head_log2 as f32);
    (0..n_head)
        .map(|h| {
            if h < n_head_log2 {
                m0.powf((h + 1) as f32)
            } else {
                m1.powf((2 * (h - n_head_log2) + 1) as f32)
            }
        })
        .collect()
}

/// Additive ALiBi bias for head `h`: `slope(h) * -|query - key|`
/// (`llama-graph.cpp:441` fills the mask with `-|p0 - p1|`, and
/// `soft_max_ext` multiplies it by the per-head slope).
fn alibi_bias(slope: f32, query: usize, key: usize) -> f32 {
    slope * -((query as i64 - key as i64).unsigned_abs() as f32)
}

/// SiLU in place, matching `ggml_silu_f32` exactly (`ggml-cpu/vec.h:1046`):
/// `x / (1.0f + expf(-x))`. `crate::ops::silu` is the identical scalar helper,
/// so no new operator is introduced here.
fn silu_inplace(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value = crate::ops::silu(*value);
    }
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

/// MoE router gate for one token, matching `LLM_FFN_EXPERT_GATING` +
/// `build_moe_ffn` as `bert.cpp:165-176` calls it.
///
/// `bert.cpp:173` selects `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`, so:
///   1. `probs = softmax(logits)` over **all** experts
///      (`llama-graph.cpp:2052-2055`);
///   2. `selected = argsort_top_k(probs, k)` (`llama-graph.cpp:2118`);
///   3. `weights = probs[selected]`, read straight back out by
///      `ggml_get_rows` (`llama-graph.cpp:2133`) and **not** renormalized,
///      because `bert.cpp:171` passes `norm_w = false` so the
///      `ggml_div`-by-`weights_sum` block at `llama-graph.cpp:2143-2155`
///      is skipped;
///   4. `w_scale` is applied only when it is neither `0.0` (hparams
///      default, "unset") nor `1.0` (`llama-graph.cpp:2156`).
///
/// Consequence worth stating: `selected` weights sum to at most 1, and the
/// shortfall is the mass the unselected experts keep. Renormalizing them to
/// sum to 1 is the `SOFTMAX_WEIGHT` variant, which `bert.cpp` does not use;
/// it scales the FFN output by `1/(sum of selected probs) >= 1`.
///
/// Returns `(selected expert ids sorted by descending probability, weights)`.
pub fn moe_gate(logits: &[f32], k: usize, w_scale: f32) -> (Vec<usize>, Vec<f32>) {
    let n_expert = logits.len();
    let k = k.min(n_expert);

    let mut max_logit = f32::NEG_INFINITY;
    for &v in logits.iter() {
        if v > max_logit {
            max_logit = v;
        }
    }
    // f64 accumulation: the logits are small but the softmax denominator is a
    // sum of exponentials, and the graph is otherwise f32.
    let mut probs = vec![0.0f64; n_expert];
    let mut probs_sum = 0.0f64;
    for (p, &logit) in probs.iter_mut().zip(logits.iter()) {
        let value = ((logit - max_logit).exp()) as f64;
        *p = value;
        probs_sum += value;
    }
    if probs_sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= probs_sum;
        }
    }

    // Top-k by probability, ties broken by lower id. Selecting on `probs`
    // rather than on `logits` picks the same experts (softmax is monotonic)
    // and keeps the comparison in the same domain the oracle uses.
    let mut order: Vec<usize> = (0..n_expert).collect();
    order.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let selected = order[..k].to_vec();

    let scale = f64::from(w_scale);
    let scale = if scale != 0.0 && scale != 1.0 { scale } else { 1.0 };
    let weights = selected
        .iter()
        .map(|&e| (probs[e] * scale) as f32)
        .collect();
    (selected, weights)
}

pub fn compute_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    // The four bert-family variants use three different tokenizers:
    //   * `bert`/`jina-bert-v2`/`nomic-bert` ship `tokenizer.ggml.model =
    //     "bert"` (WordPiece).
    //   * `nomic-bert-moe` ships `tokenizer.ggml.model = "t5"` (UGM, an
    //     XLM unigram with a precompiled XCDA charsmap).
    // We dispatch on the architecture rather than on `tokenizer.ggml.model`
    // because the WPM tokenizer is not a `dyn Tokenizer` and the BPE
    // fallback in `load_tokenizer` would reject "bert".
    let arch = source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .unwrap_or_default()
        .to_string();
    let prompt_tokens = if arch == "nomic-bert-moe" {
        let tok = crate::core::tokenizer::load_tokenizer(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        tok.encode(
            prompt,
            EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        )
    } else {
        let tok = WPMTokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
        tok.encode(
            prompt,
            EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        )
    };
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
    let (head_k, head_v, n_ff) = (cfg.n_embd_head_k, cfg.n_embd_head_v, cfg.n_ff);
    let n_embd_q = n_head * head_k;
    let n_embd_gqa = n_head_kv * head_v;
    let group_size = n_head / n_head_kv;
    // `bert.cpp:147` — score scale is 1/sqrt(head_dim), NOT the llama
    // per-head-q scale used by the causal trunks.
    let score_scale = 1.0f32 / (head_k as f32).sqrt();

    let weights: BertWeights<'_> = load_weights(
        source,
        cfg.variant,
        n_layer,
        n_embd,
        n_embd_q,
        n_embd_gqa,
        head_k,
        n_ff,
    );
    let slopes = if cfg.variant.uses_alibi() {
        alibi_slopes(n_head, MAX_ALIBI_BIAS_JINA_V2)
    } else {
        vec![1.0; n_head]
    };

    let available = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(4);
    let n_threads = crate::app::resolve_thread_count(n_threads_arg, available);
    let pool = Arc::new(ComputePool::new(n_threads));

    let n_tokens = token_ids.len();
    let mut hidden = vec![0.0f32; n_tokens * n_embd];

    // ---- embeddings: token (+ segment row 0 for jina) (+ abs pos for bert)
    for (token, row) in token_ids.iter().zip(hidden.chunks_exact_mut(n_embd)) {
        embedding_lookup(
            weights.token_embd,
            *token,
            n_embd,
            weights.token_embd_ggml_type,
            row,
        );
    }
    if let Some((bytes, _)) = weights.token_types {
        // `bert.cpp:83-86` — token types are hardcoded to 0 ("sentence A"),
        // so only row 0 of `token_types` is ever added. `token_types` is a
        // plain F32 [n_embd, n_token_types] table, not a quantized
        // embedding matrix, so decode the row directly.
        let segment = super::weights::decode_f32_row_public(bytes, n_embd)
            .ok_or("token_types.weight is not decodable as f32")?;
        for row in hidden.chunks_exact_mut(n_embd) {
            for (slot, value) in row.iter_mut().zip(&segment) {
                *slot += *value;
            }
        }
    }
    if let Some((bytes, _)) = weights.pos_embd {
        // `bert.cpp:87-89` — absolute learned position embeddings, `bert` only.
        // `ggml_get_rows(pos_embd, inp_pos)` indexes the table by **position**,
        // so token `t` reads row `t`, not row 0. The row offset must account
        // for the element width, so it is done in the weights helper rather
        // than here. Not covered by a test: no `bert`-arch GGUF is available
        // locally to verify against.
        for (t, row) in hidden.chunks_exact_mut(n_embd).enumerate() {
            let position = super::weights::decode_f32_row_at_public(bytes, t, n_embd)
                .ok_or("pos_embd.weight is not decodable as f32")?;
            for (slot, value) in row.iter_mut().zip(&position) {
                *slot += *value;
            }
        }
    }

    // ---- embedding LayerNorm (`bert.cpp:92`, `LLM_NORM` with bias)
    let mut normed = vec![0.0f32; n_embd];
    for row in hidden.chunks_exact_mut(n_embd) {
        layer_norm(
            row,
            &weights.tok_norm.weight,
            &weights.tok_norm.bias,
            cfg.eps,
            &mut normed,
        );
        row.copy_from_slice(&normed);
    }

    let mut qkv_buf = vec![0.0f32; n_tokens * (n_embd_q + 2 * n_embd_gqa)];
    // Fused QKV width: Q + K + V concatenated along the output dim.
    let qkv_width = n_embd_q + 2 * n_embd_gqa;
    let mut attn_out = vec![0.0f32; n_tokens * n_embd_q];
    let mut attn_proj = vec![0.0f32; n_tokens * n_embd];
    let mut gate_buf = vec![0.0f32; n_ff];
    let mut up_buf = vec![0.0f32; n_ff];
    let mut down_buf = vec![0.0f32; n_embd];
    let max_width = n_embd.max(n_ff);
    let mut q8k_buf = vec![
        crate::ops::quant::BlockQ8K {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        };
        max_width.div_ceil(crate::ops::quant::QK_K)
    ];
    let mut q8_buf = vec![0u8; max_width];
    let mut scale_buf = vec![0.0f32; max_width.div_ceil(32)];

    for layer in 0..n_layer {
        let lw = &weights.layers[layer];
        // `bert.cpp:141` — the layer input is re-added after attention and
        // again after the FFN (residual re-add, not a pre-norm sandwich).
        let residual: Vec<f32> = hidden.clone();

        // 1. Q / K / V with biases. Fused `attn_qkv` for nomic-bert, three
        //    separate projections for the others. `bert.cpp:120-133` ropes Q
        //    and K only for NOMIC_BERT (and its MoE sibling / jina-bert-v3);
        //    BERT and jina-bert-v2 fall through unroped.
        if let Some(wqkv) = lw.wqkv.as_ref() {
            for t in 0..n_tokens {
                let x = &hidden[t * n_embd..(t + 1) * n_embd];
                let out = &mut qkv_buf[t * qkv_width..(t + 1) * qkv_width];
                wqkv.quantize_and_matmul_with_scratch(
                    x,
                    &mut q8k_buf,
                    &mut q8_buf,
                    &mut scale_buf,
                    out,
                    &pool,
                );
            }
        } else {
            for t in 0..n_tokens {
                let x = &hidden[t * n_embd..(t + 1) * n_embd];
                let out = &mut qkv_buf[t * qkv_width..(t + 1) * qkv_width];
                let (q, rest) = out.split_at_mut(n_embd_q);
                let (k, v) = rest.split_at_mut(n_embd_gqa);
                let wq = lw.wq.as_ref().ok_or("bert-family layer is missing wq")?;
                let wk = lw.wk.as_ref().ok_or("bert-family layer is missing wk")?;
                let wv = lw.wv.as_ref().ok_or("bert-family layer is missing wv")?;
                wq.quantize_and_matmul_with_scratch(
                    x,
                    &mut q8k_buf,
                    &mut q8_buf,
                    &mut scale_buf,
                    q,
                    &pool,
                );
                lw.wq_bias.add_to(q);
                wk.quantize_and_matmul_with_scratch(
                    x,
                    &mut q8k_buf,
                    &mut q8_buf,
                    &mut scale_buf,
                    k,
                    &pool,
                );
                lw.wk_bias.add_to(k);
                wv.quantize_and_matmul_with_scratch(
                    x,
                    &mut q8k_buf,
                    &mut q8_buf,
                    &mut scale_buf,
                    v,
                    &pool,
                );
                lw.wv_bias.add_to(v);
            }
        }

        // RoPE for nomic-bert only, applied per head on the Q and K halves.
        // `n_rot` is `n_embd/n_head` (the GGUF carries no
        // `rope.dimension_count`), which equals `head_dim` here, so every lane
        // rotates. The rope variant is `LLAMA_ROPE_TYPE_NORM`
        // (`llama-model.cpp:3055`), i.e. interleaved pairs, not NEOX.
        if cfg.variant.uses_rope() {
            for t in 0..n_tokens {
                let row = &mut qkv_buf[t * qkv_width..(t + 1) * qkv_width];
                let (q, rest) = row.split_at_mut(n_embd_q);
                let (k, _) = rest.split_at_mut(n_embd_gqa);
                rope_norm(q, t, head_k, cfg.rope_freq_base);
                rope_norm(k, t, head_k, cfg.rope_freq_base);
            }
        }

        // 2. bidirectional attention with the ALiBi / zero bias. The ggml
        //    mask is `-|p0-p1|` (or 0 without ALiBi) and never masks a key
        //    inside the same sequence, so no key is hidden.
        for t in 0..n_tokens {
            let attn_row = &mut attn_out[t * n_embd_q..(t + 1) * n_embd_q];
            for h in 0..n_head {
                let kv_h = h / group_size;
                let q_off = h * head_k;
                let out_base = h * head_v;
                let mut scores = vec![0.0f32; n_tokens];
                for s in 0..n_tokens {
                    let q_row = &qkv_buf[t * qkv_width..t * qkv_width + n_embd_q];
                    let k_row = &qkv_buf[s * qkv_width + n_embd_q..s * qkv_width + qkv_width];
                    let dot: f32 = q_row[q_off..q_off + head_k]
                        .iter()
                        .zip(&k_row[kv_h * head_v..kv_h * head_v + head_k])
                        .map(|(a, b)| a * b)
                        .sum();
                    let bias = if cfg.variant.uses_alibi() {
                        alibi_bias(slopes[h], t, s)
                    } else {
                        0.0
                    };
                    scores[s] = dot * score_scale + bias;
                }
                softmax_inplace(&mut scores);
                for d in 0..head_v {
                    let mut acc = 0.0f32;
                    for s in 0..n_tokens {
                        let v_off = n_embd_q + n_embd_gqa + kv_h * head_v + d;
                        acc += qkv_buf[s * qkv_width + v_off] * scores[s];
                    }
                    attn_row[out_base + d] = acc;
                }
            }
        }

        // 3. attention output projection + residual re-add + LayerNorm
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
            lw.wo_bias.add_to(proj);
        }
        for t in 0..n_tokens {
            let x = &mut hidden[t * n_embd..(t + 1) * n_embd];
            // `bert.cpp:151` — `cur = ggml_add(cur, inpL)` where `cur` is the
            // attention *output projection* and `inpL` the layer input. The
            // projected value has to be part of the sum; adding only the
            // residual makes every layer an identity map through `inpL`, which
            // silently drops the attention from the stack.
            for i in 0..n_embd {
                x[i] += attn_proj[t * n_embd + i] + residual[t * n_embd + i];
            }
            // `bert.cpp:154` — attention output LayerNorm. `x` is reborrowed
            // as `&[f32]` for the read; the write goes to `normed`.
            layer_norm(
                x,
                &lw.attn_out_norm.weight,
                &lw.attn_out_norm.bias,
                cfg.eps,
                &mut normed,
            );
            x.copy_from_slice(&normed);
        }

        // 4. FFN: GELU/GEGLU/SwiGLU dense, or 8x top-2 MoE for `nomic-bert-moe` MoE
        //    layers. The MoE branch mirrors `build_moe_ffn`
        //    (`llama-graph.cpp:2002`) reduced to scalar batches — router
        //    logits, top-k, softmax, weighted sum of per-expert (up →
        //    GELU → down) outputs. MoE experts use plain GELU (no gate),
        //    same as the dense `BERT || NOMIC_BERT_MOE` arm at `bert.cpp:179`.
        if lw.is_moe_layer {
            let router = lw
                .ffn_gate_inp
                .as_ref()
                .expect("moe layer missing ffn_gate_inp");
            let up_exps = lw
                .ffn_up_exps
                .as_ref()
                .expect("moe layer missing ffn_up_exps");
            let down_exps = lw
                .ffn_down_exps
                .as_ref()
                .expect("moe layer missing ffn_down_exps");
            assert!(weights.expert_count > 0);
            assert!(weights.expert_used_count > 0);
            let n_expert = weights.expert_count;
            let k = weights.expert_used_count.min(n_expert);

            // Build every expert's kernel once per layer instead of once per
            // selected expert per token. `Weight::from_quantized` boxes a
            // kernel over a `&[u8]` slice, so the previous `for token { for k {`
            // nesting re-boxed them `n_tokens * k` times for no benefit.
            let up_weights: Vec<Weight<'_>> = (0..n_expert)
                .map(|e| {
                    Weight::from_quantized(QuantizedTensor::from_bytes(
                        up_exps.per_expert_bytes(e),
                        up_exps.ggml_type,
                        up_exps.cols, // n_in = inner dim
                        up_exps.rows, // n_out = outer dim
                    ))
                })
                .collect();
            let down_weights: Vec<Weight<'_>> = (0..n_expert)
                .map(|e| {
                    Weight::from_quantized(QuantizedTensor::from_bytes(
                        down_exps.per_expert_bytes(e),
                        down_exps.ggml_type,
                        down_exps.cols, // n_in = inner dim
                        down_exps.rows, // n_out = outer dim
                    ))
                })
                .collect();

            let mut logits = vec![0.0f32; n_expert];
            let mut hidden_e = vec![0.0f32; n_embd];
            for t in 0..n_tokens {
                let x = &hidden[t * n_embd..(t + 1) * n_embd];
                // Router logits [n_expert] per token.
                router.quantize_and_matmul_with_scratch(
                    x,
                    &mut q8k_buf,
                    &mut q8_buf,
                    &mut scale_buf,
                    &mut logits,
                    &pool,
                );

                // `bert.cpp:173` passes LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                // so `llama-graph.cpp:2052-2055` softmaxes over ALL n_expert
                // logits first; `ggml_argsort_top_k` at line 2118 then picks
                // n_expert_used, and `ggml_get_rows` at line 2133 reads the
                // *selected probabilities* straight back out as the weights.
                // The 9th `build_moe_ffn` argument from `bert.cpp:171` is
                // `norm_w = false`, so the block at lines 2143-2155 does NOT
                // renormalize them to sum to 1: the k weights sum to
                // p_a + ... <= 1 and the leftover mass stays on the experts
                // that were not selected.
                //
                // Renormalizing (the SOFTMAX_WEIGHT path, which is what this
                // loop used to implement) multiplies the expert output by
                // 1/(p_a + p_b) >= 1 - up to 2x when the router is unsure - so
                // it is not an equivalent shortcut.
                let (selected, gate) = moe_gate(&logits, k, weights.expert_weights_scale);
                let weight_of = |e: usize| -> f32 {
                    gate[selected
                        .iter()
                        .position(|&s| s == e)
                        .expect("expert id missing from the top-k selection")]
                };

                // Weighted sum of expert (up → gelu → down) outputs.
                for slot in hidden_e.iter_mut() {
                    *slot = 0.0;
                }
                for &e in selected.iter() {
                    let weight_e = weight_of(e);
                    // `expert @ x` (per-expert up is `[n_ff rows × n_embd cols]`
                    // in storage, the matmul reads it row-major as
                    // `[ne0=n_embd, ne1=n_ff]`, so `output[m] = Σ_k
                    // expert[m, k] * x[k]` gives the `n_ff` vector we want).
                    up_weights[e].quantize_and_matmul_with_scratch(
                        x,
                        &mut q8k_buf,
                        &mut q8_buf,
                        &mut scale_buf,
                        &mut up_buf,
                        &pool,
                    );
                    gelu_ggml_f16_inplace(&mut up_buf);
                    down_weights[e].quantize_and_matmul_with_scratch(
                        &up_buf,
                        &mut q8k_buf,
                        &mut q8_buf,
                        &mut scale_buf,
                        &mut down_buf,
                        &pool,
                    );
                    // hidden_e += weight_e * (up_buf @ down_exps[e])
                    for (acc, &v) in hidden_e.iter_mut().zip(down_buf.iter()) {
                        *acc += weight_e * v;
                    }
                }

                // Add the MoE FFN contribution to the residual (the
                // attention-projected residual is added separately above)
                // and run the layer's output LayerNorm.
                let x = &mut hidden[t * n_embd..(t + 1) * n_embd];
                for i in 0..n_embd {
                    x[i] += hidden_e[i];
                }
                layer_norm(
                    x,
                    &lw.layer_out_norm.weight,
                    &lw.layer_out_norm.bias,
                    cfg.eps,
                    &mut normed,
                );
                x.copy_from_slice(&normed);
            }
        } else {
            // ---- Dense FFN (BERT / jina / nomic / nomic-moe dense) ----
            let ffn_up = lw
                .ffn_up
                .as_ref()
                .expect("dense layer missing ffn_up.weight");
            let ffn_down = lw
                .ffn_down
                .as_ref()
                .expect("dense layer missing ffn_down.weight");
            for t in 0..n_tokens {
                let x = &hidden[t * n_embd..(t + 1) * n_embd];
                ffn_up.quantize_and_matmul_with_scratch(
                    x,
                    &mut q8k_buf,
                    &mut q8_buf,
                    &mut scale_buf,
                    &mut up_buf,
                    &pool,
                );
                lw.ffn_up_bias.add_to(&mut up_buf);

                // `ggml_geglu_split(cur, tmp)` with `cur = gate` and `tmp = up`
                // (`llama-graph.cpp:1825-1831,1866-1869`), and
                // `ggml_vec_geglu_f32(y, x, g)` computes `gelu(x) * g` with
                // `x = src0 = gate`. So the FFN is `gelu(gate) * up` — the
                // gate is the activated side and `ffn_up` is the plain
                // multiplier. The down projection always reads `ffn_in`; for
                // the gelu-only variant that is `up_buf` itself.
                let ffn_in: &[f32] = if cfg.variant.uses_gelu_gate() {
                    let gate = lw
                        .ffn_gate
                        .as_ref()
                        .expect("geglu variant requires ffn_gate");
                    gate.quantize_and_matmul_with_scratch(
                        x,
                        &mut q8k_buf,
                        &mut q8_buf,
                        &mut scale_buf,
                        &mut gate_buf,
                        &pool,
                    );
                    lw.ffn_gate_bias.add_to(&mut gate_buf);
                    gelu_ggml_f16_inplace(&mut gate_buf);
                    for (slot, up) in gate_buf.iter_mut().zip(&up_buf) {
                        *slot *= *up;
                    }
                    &gate_buf[..]
                } else if cfg.variant.uses_silu_gate() {
                    // `bert.cpp:196-203` — the `bert.cpp` fall-through arm,
                    // which nomic-bert reaches because it is in neither the
                    // GELU arm (`bert.cpp:179`) nor the GEGLU arm
                    // (`bert.cpp:187`). `build_ffn(up, gate, ...)` with
                    // SILU + FFN_PAR gives `ggml_swiglu_split(cur = gate,
                    // tmp = up)` = `silu(gate) * up`, so again the gate is
                    // the activated side.
                    let gate = lw
                        .ffn_gate
                        .as_ref()
                        .expect("swiglu variant requires ffn_gate");
                    gate.quantize_and_matmul_with_scratch(
                        x,
                        &mut q8k_buf,
                        &mut q8_buf,
                        &mut scale_buf,
                        &mut gate_buf,
                        &pool,
                    );
                    lw.ffn_gate_bias.add_to(&mut gate_buf);
                    silu_inplace(&mut gate_buf);
                    for (slot, up) in gate_buf.iter_mut().zip(&up_buf) {
                        *slot *= *up;
                    }
                    &gate_buf[..]
                } else {
                    // `bert.cpp:155-160` — plain GELU over the single `ffn_up`.
                    gelu_ggml_f16_inplace(&mut up_buf);
                    &up_buf[..]
                };

                ffn_down.quantize_and_matmul_with_scratch(
                    ffn_in,
                    &mut q8k_buf,
                    &mut q8_buf,
                    &mut scale_buf,
                    &mut down_buf,
                    &pool,
                );
                lw.ffn_down_bias.add_to(&mut down_buf);

                let x = &mut hidden[t * n_embd..(t + 1) * n_embd];
                for i in 0..n_embd {
                    x[i] += down_buf[i];
                }
                // `bert.cpp:195` — output LayerNorm closes the layer.
                layer_norm(
                    x,
                    &lw.layer_out_norm.weight,
                    &lw.layer_out_norm.bias,
                    cfg.eps,
                    &mut normed,
                );
                x.copy_from_slice(&normed);
            }
        }
    }

    // ---- pooling, then L2 (`common.cpp:1893`, `embd_normalize = 2`)
    let mut pooled = vec![0.0f32; n_embd];
    match cfg.pooling_type {
        1 => {
            // Mean over every token, specials included.
            // `llm_graph_input_mean::set_input` (`llama-graph.cpp:250-278`)
            // gives each token weight 1/n_tokens.
            for row in hidden.chunks_exact(n_embd) {
                for (slot, value) in pooled.iter_mut().zip(row) {
                    *slot += *value;
                }
            }
            let inv_tokens = 1.0f32 / n_tokens as f32;
            for value in pooled.iter_mut() {
                *value *= inv_tokens;
            }
        }
        2 => {
            // CLS: the row of the lowest-position token. `set_input`
            // (`llama-graph.cpp:303-319`) takes `pos < target_pos`, so for a
            // single fresh sequence that is row 0, which the WPM tokenizer has
            // already filled with [CLS].
            let cls = hidden
                .chunks_exact(n_embd)
                .next()
                .ok_or("CLS pooling found no tokens")?;
            pooled.copy_from_slice(cls);
        }
        3 => {
            // Last token row, equivalent to row `n_tokens - 1` here.
            let last = hidden
                .chunks_exact(n_embd)
                .nth(n_tokens - 1)
                .ok_or("LAST pooling found no tokens")?;
            pooled.copy_from_slice(last);
        }
        other => return Err(format!("unsupported pooling_type {other}")),
    }
    l2_normalize(&mut pooled)?;
    Ok(pooled)
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
    _kv_format: KvFormat,
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

#[cfg(test)]
mod moe_gate_tests {
    use super::moe_gate;

    /// Sums a slice in f64 so the assertion is not itself a float puzzle.
    fn sum(values: &[f32]) -> f64 {
        values.iter().map(|&v| f64::from(v)).sum()
    }

    #[test]
    fn softmax_gate_picks_the_two_largest_logits() {
        // 0.0 0.9 0.2 -0.3 1.5 0.4 -0.8 0.1 -> experts 4 then 1.
        let logits = [0.0f32, 0.9, 0.2, -0.3, 1.5, 0.4, -0.8, 0.1];
        let (selected, weights) = moe_gate(&logits, 2, 1.0);
        assert_eq!(selected, vec![4, 1], "top-2 by logit must be experts 4 and 1");
        // Softmax is monotonic, so the weights must be ordered the same way.
        assert!(weights[0] > weights[1]);
        // ...and they are plain softmax probabilities, not renormalized over
        // the selection: this pair holds only part of the mass.
        let total = sum(&weights);
        assert!(
            total < 1.0 - 1e-6,
            "selected weights must NOT sum to 1 (got {total}); \
             renormalizing is the SOFTMAX_WEIGHT variant bert.cpp does not use"
        );
        assert!(total > 0.5, "the top-2 should still hold most of the mass: {total}");
        // Cross-check against a hand-computed softmax over all 8 logits. The
        // gate returns f32, so the tolerance is f32 ulp, not f64 exactness.
        let expected: Vec<f64> = logits
            .iter()
            .map(|&l| {
                let e = ((l - 1.5f32).exp()) as f64;
                let denom: f64 = logits.iter().map(|&o| ((o - 1.5f32).exp()) as f64).sum();
                e / denom
            })
            .collect();
        assert!((f64::from(weights[0]) - expected[4]).abs() < 1e-7);
        assert!((f64::from(weights[1]) - expected[1]).abs() < 1e-7);
    }

    #[test]
    fn unsure_router_is_left_mostly_unweighted() {
        // Flat logits: every expert gets 1/8, so the top-2 hold only 0.25.
        // This is where renormalizing would have inflated the output 4x.
        let logits = [0.0f32; 8];
        let (selected, weights) = moe_gate(&logits, 2, 1.0);
        assert_eq!(selected, vec![0, 1], "ties break to the lower ids");
        let total = sum(&weights);
        assert!((total - 0.25).abs() < 1e-6, "flat router keeps only 0.25 mass");
    }

    #[test]
    fn ties_break_to_the_lower_expert_id() {
        let logits = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let (selected, _) = moe_gate(&logits, 2, 1.0);
        assert_eq!(selected, vec![0, 1]);
    }

    #[test]
    fn w_scale_only_applies_when_it_is_neither_zero_nor_one() {
        let logits = [0.0f32, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0, 0.0];
        let (_, unscaled) = moe_gate(&logits, 2, 1.0);
        let (_, scaled) = moe_gate(&logits, 2, 0.5);
        for (u, s) in unscaled.iter().zip(scaled.iter()) {
            assert!((f64::from(*u) * 0.5 - f64::from(*s)).abs() < 1e-9);
        }
        // 0.0 is the hparams "unset" default, so it must behave like 1.0
        // (`llama-graph.cpp:2156` guards on both 0.0 and 1.0).
        let (_, via_zero) = moe_gate(&logits, 2, 0.0);
        for (u, z) in unscaled.iter().zip(via_zero.iter()) {
            assert_eq!(u, z, "w_scale = 0.0 must be a no-op, not a zero-out");
        }
    }

    #[test]
    fn k_is_clamped_to_the_expert_count() {
        let logits = [1.0f32, 0.5];
        let (selected, weights) = moe_gate(&logits, 8, 1.0);
        assert_eq!(selected.len(), 2);
        assert_eq!(weights.len(), 2);
    }
}
