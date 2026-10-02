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
use crate::ops::{
    dot_f32, embedding_lookup, gelu_ggml_f16_inplace, layer_norm, rope_norm_nrot, softmax_inplace,
};
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
    /// `rope.dimension_count` — how many lanes of each head RoPE rotates.
    /// Defaults to `n_embd_head_k` (`llama-model.cpp:1438-1445`), which is why
    /// the models that omit the key still rotate every lane.
    n_rot: usize,
    /// `pooling_type` (`llama.h:177-182`): 0 none, 1 mean, 2 CLS, 3 last.
    /// `bert` ships 2, while jina-bert-v2 and nomic-bert ship 1.
    pooling_type: u64,
    /// `context_length` — the trained context. Equals the row count of
    /// `position_embd.weight` for the absolute-position variants and the RoPE
    /// `n_ctx_orig` for the roped ones. `0` means the GGUF did not ship it, in
    /// which case no length check is applied.
    n_ctx_train: usize,
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
        // `llama-model.cpp:1438-1445`: `n_rot` defaults to `n_embd_head_k` and
        // is only overridden when the GGUF ships `rope.dimension_count`. The
        // three roped bert-family GGUFs all omit it, so they rotate every lane
        // — but a partial-rotary encoder would be silently over-rotated if we
        // hardcoded `head_dim` here.
        n_rot: uint("rope.dimension_count").unwrap_or(n_embd_head_k),
        // 1 (mean) is the safe default: every variant verified before bge-small
        // used it, and `bert` is the one that passes 2 explicitly.
        pooling_type: uint64("pooling_type").unwrap_or(1),
        // `context_length` is the size of `position_embd.weight`'s row count
        // for the absolute-position variants (`bert.cpp:31`),
        // and the RoPE `n_ctx_orig` for the roped ones. Either way an input
        // longer than this has no defined positions, so it has to be rejected
        // rather than silently indexed out of range.
        n_ctx_train: uint("context_length").unwrap_or(0),
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
    // Production: each value promoted to f64 first, then squared in f64.
    // Parity mode: square in f32 first (matches llama.cpp scalar), then
    // promote. The two paths differ in the rounding of the squared term
    // (1 ULP at most for typical embedding magnitudes); downstream
    // normalization is unchanged. `scalar_mode()` returns `false`
    // (compile-time const) in non-parity-trace builds, so the runtime
    // branch is DCE'd away in production.
    let sum: f64 = if crate::ops::scalar_mode() {
        values.iter().map(|v| f64::from(*v * *v)).sum()
    } else {
        values.iter().map(|v| f64::from(*v) * f64::from(*v)).sum()
    };
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
    let scale = if scale != 0.0 && scale != 1.0 {
        scale
    } else {
        1.0
    };
    let weights = selected
        .iter()
        .map(|&e| (probs[e] * scale) as f32)
        .collect();
    (selected, weights)
}

/// Add the separate Q/K/V biases on top of a fused-QKV projection output.
///
/// `out` is one token's `[q | k | v]` slice of width
/// `n_embd_q + 2 * n_embd_gqa`; `n_embd_q` / `n_embd_gqa` locate the sub-slices.
///
/// The oracle's guard is `else if (layer.wq_b && layer.wk_b && layer.wv_b)`
/// (`llama-graph.cpp:1663`): all three must be present before **any** of them is
/// added, because it concatenates them into one `[q | k | v]` vector first.
/// A partial set is therefore left untouched here too.
///
/// Returns whether the biases were applied.
fn add_split_qkv_biases(
    q_bias: &super::weights::Bias,
    k_bias: &super::weights::Bias,
    v_bias: &super::weights::Bias,
    out: &mut [f32],
    n_embd_q: usize,
    n_embd_gqa: usize,
) -> bool {
    if q_bias.values.is_none() || k_bias.values.is_none() || v_bias.values.is_none() {
        return false;
    }
    let (q, rest) = out.split_at_mut(n_embd_q);
    let (k, v) = rest.split_at_mut(n_embd_gqa);
    q_bias.add_to(q);
    k_bias.add_to(k);
    v_bias.add_to(v);
    true
}

pub fn compute_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    // The tokenizer is chosen by `tokenizer.ggml.model`, exactly as
    // `llama-vocab.cpp:1804+` does — NOT by the model architecture. The two
    // are independent: `bert`-arch weights ship either WordPiece (`"bert"`,
    // the English BERT family) or a SentencePiece unigram (`"t5"` →
    // LLAMA_VOCAB_TYPE_UGM, the XLM-Roberta family such as bge-m3), while
    // `nomic-bert` uses `"bert"` and `nomic-bert-moe` uses `"t5"`.
    //
    // Keying this off the arch silently breaks every model whose pair does not
    // match the two that happened to be verified first: bge-m3 is `arch=bert`
    // + `tokenizer.ggml.model=t5`, and routing by arch sent it to WordPiece,
    // which rejects the model with "expected bert".
    //
    // WPM lives outside the `Tokenizer` trait, so it gets its own arm and
    // everything else goes through `load_tokenizer` (t5 → UGM, llama → SPM,
    // otherwise BPE).
    let options = EncodeOptions {
        add_special: true,
        parse_special: true,
    };
    let prompt_tokens = {
        let model = source
            .metadata("tokenizer.ggml.model")
            .and_then(MetaValue::to_string_val)
            .unwrap_or_default()
            .to_string();
        if model == "bert" {
            WPMTokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())
                .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?
                .encode(prompt, options)
        } else {
            crate::core::tokenizer::load_tokenizer(|k| source.metadata(k).cloned())
                .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?
                .encode(prompt, options)
        }
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
    // `llama-context.cpp:133` sizes the KV cache (and hence the accepted batch)
    // from `hparams.n_ctx_train`, so an over-long input is a hard error in
    // llama.cpp rather than a silent truncation.
    //
    // Without this the over-length prompt dies much later and much less
    // usefully: the absolute-position variants index `position_embd.weight`
    // row by row, so a 700-token prompt on a 512-row table fails inside
    // `decode_f32_row_at_public` and surfaces as "position_embd.weight is not
    // decodable as f32" — which describes a decode failure that never happened
    // and hides the actual out-of-range index. There is no truncation path on
    // purpose: silently dropping 200 tokens changes what the embedding means.
    if cfg.n_ctx_train != 0 && token_ids.len() > cfg.n_ctx_train {
        return Err(format!(
            "prompt is too long: {} tokens, but this model was trained for {} \
             (context_length); split the input or use a longer-context model",
            token_ids.len(),
            cfg.n_ctx_train
        ));
    }
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
    #[cfg(feature = "parity-trace")]
    if crate::parity_trace::enabled("embedding.tokens") {
        crate::parity_trace::token_ids("embedding.tokens", token_ids).map_err(|e| e.to_string())?;
    }
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

    trace("bert.embedding", None, &[n_tokens, n_embd], &hidden)?;
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
    trace("bert.embedding_norm", None, &[n_tokens, n_embd], &hidden)?;

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
    // Attention scratch, allocated once for the whole forward. `scores` used to
    // be allocated inside the `for token { for head {` nest, i.e.
    // `n_tokens * n_head * n_layer` times — 4608 allocations for a 32-token
    // bge-m3 prompt — for a buffer that depends on none of the three.
    let mut scores = vec![0.0f32; n_tokens];
    let mut value_column = vec![0.0f32; n_tokens];

    for layer in 0..n_layer {
        let lw = &weights.layers[layer];
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
                // A fused `attn_qkv` may still ship *separate* Q/K/V biases:
                // `create_tensor_qkv` (`llama-model.cpp:3358-3363`) loads
                // `wq_b`/`wk_b`/`wv_b` whenever there is no fused
                // `attn_qkv.bias`, and `build_qkv`
                // (`llama-graph.cpp:1663-1668`) concatenates them into one
                // `[q | k | v]` vector and adds it to the fused projection
                // output. Skipping this silently drops the bias, which is the
                // same failure shape as the attention-projection residual bug.
                add_split_qkv_biases(
                    &lw.wq_bias,
                    &lw.wk_bias,
                    &lw.wv_bias,
                    out,
                    n_embd_q,
                    n_embd_gqa,
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

        // RoPE for nomic-bert / nomic-bert-moe / jina-bert-v3, applied per head
        // on the Q and K halves. `n_rot` comes from
        // `rope.dimension_count` and defaults to `n_embd/n_head`
        // (`llama-model.cpp:1438-1445`), which equals `head_dim` for every
        // model verified so far, so all lanes rotate; a partial-rotary encoder
        // would only turn its first `n_rot` lanes. The rope variant is
        // `LLAMA_ROPE_TYPE_NORM` (`llama-model.cpp:3055`), i.e. interleaved
        // pairs, not NEOX.
        if cfg.variant.uses_rope() {
            debug_assert!(cfg.n_rot <= head_k);
            for t in 0..n_tokens {
                let row = &mut qkv_buf[t * qkv_width..(t + 1) * qkv_width];
                let (q, rest) = row.split_at_mut(n_embd_q);
                let (k, _) = rest.split_at_mut(n_embd_gqa);
                rope_norm_nrot(q, t, head_k, cfg.n_rot, cfg.rope_freq_base);
                rope_norm_nrot(k, t, head_k, cfg.n_rot, cfg.rope_freq_base);
            }
        }

        #[cfg(feature = "parity-trace")]
        for (name, offset, heads, width) in [
            ("bert.q", 0, n_head, head_k),
            ("bert.k", n_embd_q, n_head_kv, head_k),
            ("bert.v", n_embd_q + n_embd_gqa, n_head_kv, head_v),
        ] {
            if crate::parity_trace::enabled(name) {
                let values: Vec<f32> = qkv_buf
                    .chunks_exact(qkv_width)
                    .flat_map(|row| row[offset..offset + heads * width].iter().copied())
                    .collect();
                trace(name, Some(layer), &[n_tokens, heads, width], &values)?;
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
                for s in 0..n_tokens {
                    let q_row = &qkv_buf[t * qkv_width..t * qkv_width + n_embd_q];
                    let k_row = &qkv_buf[s * qkv_width + n_embd_q..s * qkv_width + qkv_width];
                    let dot = dot_f32(
                        &q_row[q_off..q_off + head_k],
                        &k_row[kv_h * head_v..kv_h * head_v + head_k],
                        head_k,
                    );
                    let bias = if cfg.variant.uses_alibi() {
                        alibi_bias(slopes[h], t, s)
                    } else {
                        0.0
                    };
                    scores[s] = dot * score_scale + bias;
                }
                softmax_inplace(&mut scores);
                for d in 0..head_v {
                    for (s, value) in value_column.iter_mut().enumerate() {
                        let v_off = n_embd_q + n_embd_gqa + kv_h * head_v + d;
                        *value = qkv_buf[s * qkv_width + v_off];
                    }
                    attn_row[out_base + d] = dot_f32(&value_column, &scores, n_tokens);
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
        trace(
            "bert.attention",
            Some(layer),
            &[n_tokens, n_embd],
            &attn_proj,
        )?;
        for t in 0..n_tokens {
            let x = &mut hidden[t * n_embd..(t + 1) * n_embd];
            // The layer input is still in `x`; add the attention projection once.
            for i in 0..n_embd {
                x[i] = attn_proj[t * n_embd + i] + x[i];
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

        trace("bert.ffn_input", Some(layer), &[n_tokens, n_embd], &hidden)?;
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
        trace(
            "bert.layer_output",
            Some(layer),
            &[n_tokens, n_embd],
            &hidden,
        )?;
    }

    // ---- pooling, then L2 (`common.cpp:1893`, `embd_normalize = 2`)
    let mut pooled = vec![0.0f32; n_embd];
    match cfg.pooling_type {
        1 => {
            // Mean over every token, specials included. Two paths:
            // - Production (default): sum in f32 first, then scale by 1/n
            //   tokens. Simple, fast, and uses our native precision rather
            //   than borrowing llama.cpp's pattern.
            // - Parity mode (`scalar_mode()`): each token multiplied by
            //   1/n_tokens in f32 first, then accumulated in f64. Matches
            //   `llm_graph_input_mean::set_input` (`llama-graph.cpp:250-278`)
            //   so the oracle at `tools/oracle/jina_bert_v2/` stays
            //   bit-equal. The 1–2 ULP gap between the two paths does not
            //   affect similarity ordering.
            let inv_tokens = 1.0f32 / n_tokens as f32;
            if crate::ops::scalar_mode() {
                for (dim, value) in pooled.iter_mut().enumerate() {
                    *value = hidden
                        .chunks_exact(n_embd)
                        .map(|row| f64::from(row[dim] * inv_tokens))
                        .sum::<f64>() as f32;
                }
            } else {
                for row in hidden.chunks_exact(n_embd) {
                    for (slot, value) in pooled.iter_mut().zip(row) {
                        *slot += *value;
                    }
                }
                for value in pooled.iter_mut() {
                    *value *= inv_tokens;
                }
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
    trace("embedding.pooled", None, &[n_embd], &pooled)?;
    l2_normalize(&mut pooled)?;
    trace("embedding.final", None, &[n_embd], &pooled)?;
    Ok(pooled)
}

fn trace(name: &str, layer: Option<usize>, shape: &[usize], values: &[f32]) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    if crate::parity_trace::enabled(name) {
        crate::parity_trace::checkpoint(name, layer, shape, values).map_err(|e| e.to_string())?;
    }
    let _ = (name, layer, shape, values);
    Ok(())
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
        assert_eq!(
            selected,
            vec![4, 1],
            "top-2 by logit must be experts 4 and 1"
        );
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
        assert!(
            total > 0.5,
            "the top-2 should still hold most of the mass: {total}"
        );
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
        assert!(
            (total - 0.25).abs() < 1e-6,
            "flat router keeps only 0.25 mass"
        );
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

#[cfg(test)]
mod fused_qkv_bias_tests {
    use super::add_split_qkv_biases;
    use crate::models::bert_family::weights::Bias;

    fn bias(seed: f32, len: usize) -> Bias {
        Bias {
            values: Some((0..len).map(|i| seed + i as f32).collect()),
        }
    }

    #[test]
    fn all_three_biases_land_in_their_own_slices() {
        // Widths: q = 4, k = 4, v = 4, so `out` is [q | k | v].
        let (n_q, n_kv) = (4usize, 4usize);
        let qb = bias(10.0, n_q);
        let kb = bias(20.0, n_kv);
        let vb = bias(30.0, n_kv);
        let mut out = vec![0.0f32; n_q + 2 * n_kv];

        let applied = add_split_qkv_biases(&qb, &kb, &vb, &mut out, n_q, n_kv);
        assert!(applied, "all three biases present must mean applied");
        assert_eq!(&out[..n_q], &[10.0, 11.0, 12.0, 13.0], "Q slice");
        assert_eq!(&out[n_q..n_q + n_kv], &[20.0, 21.0, 22.0, 23.0], "K slice");
        assert_eq!(&out[n_q + n_kv..], &[30.0, 31.0, 32.0, 33.0], "V slice");
    }

    #[test]
    fn biases_accumulate_onto_existing_projection_output() {
        let (n_q, n_kv) = (2usize, 2usize);
        let mut out = vec![1.0f32; n_q + 2 * n_kv];
        let applied = add_split_qkv_biases(
            &bias(0.5, n_q),
            &bias(0.25, n_kv),
            &bias(0.75, n_kv),
            &mut out,
            n_q,
            n_kv,
        );
        assert!(applied);
        assert_eq!(&out, &[1.5, 2.5, 1.25, 2.25, 1.75, 2.75]);
    }

    #[test]
    fn a_partial_bias_set_adds_nothing() {
        // `llama-graph.cpp:1663` requires wq_b && wk_b && wv_b, so a model that
        // ships only some of them gets none of them. Dropping a single bias
        // while adding the others would be a silent divergence.
        let (n_q, n_kv) = (2usize, 2usize);
        let empty = Bias { values: None };
        let full = bias(1.0, n_q);
        let full_kv = bias(2.0, n_kv);

        for (q, k, v) in [
            (&empty, &full_kv, &full_kv),
            (&full, &empty, &full_kv),
            (&full, &full_kv, &empty),
            (&empty, &empty, &empty),
        ] {
            let mut out = vec![7.0f32; n_q + 2 * n_kv];
            let applied = add_split_qkv_biases(q, k, v, &mut out, n_q, n_kv);
            assert!(!applied, "a partial set must not be applied");
            assert!(
                out.iter().all(|&v| v == 7.0),
                "a partial set must leave the projection untouched, got {out:?}"
            );
        }
    }
}

#[cfg(test)]
mod alibi_tests {
    use super::alibi_slopes;
    use crate::models::bert_family::weights::{BertVariant, MAX_ALIBI_BIAS_JINA_V2};

    /// Oracle formula, transcribed from `ggml-cpu/ops.cpp:5620-5645` inside
    /// `ggml_compute_forward_soft_max_f32`:
    ///
    /// ```text
    /// n_head_log2 = 1 << floor(log2(n_head))
    /// m0 = 2^(-max_bias / n_head_log2)
    /// m1 = 2^(-(max_bias / 2) / n_head_log2)
    /// slope(h) = h < n_head_log2 ? m0^(h+1) : m1^(2*(h-n_head_log2)+1)
    /// ```
    fn oracle_slope(h: usize, n_head: usize, max_bias: f32) -> f32 {
        let log2 = (n_head as f32).log2().floor() as u32;
        let n_head_log2 = 1usize << log2;
        let m0 = 2.0f32.powf(-max_bias / n_head_log2 as f32);
        let m1 = 2.0f32.powf(-(max_bias / 2.0) / n_head_log2 as f32);
        if h < n_head_log2 {
            m0.powf((h + 1) as f32)
        } else {
            m1.powf((2 * (h - n_head_log2) + 1) as f32)
        }
    }

    #[test]
    fn alibi_slopes_match_the_oracle_formula_bit_for_bit() {
        for n_head in [4usize, 8, 12, 16] {
            for max_bias in [1.0f32, 2.0, 8.0] {
                let slopes = alibi_slopes(n_head, max_bias);
                assert_eq!(slopes.len(), n_head);
                for (h, &got) in slopes.iter().enumerate() {
                    let want = oracle_slope(h, n_head, max_bias);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "n_head={n_head} max_bias={max_bias} head={h}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn alibi_slopes_follow_the_two_branch_shape() {
        // The slope sequence is NOT globally monotonic, and that is correct:
        // `ops.cpp:5641` switches from the m0 branch to the m1 branch at
        // n_head_log2, and the second branch restarts from a *larger* value.
        // For n_head = 12, n_head_log2 = 8, so heads 0..7 decay by half each
        // time and heads 8..11 restart at m1^1 = 0.707. Asserting global
        // monotonicity here would be asserting a bug.
        for n_head in [4usize, 8, 12, 16] {
            let slopes = alibi_slopes(n_head, MAX_ALIBI_BIAS_JINA_V2);
            assert_eq!(slopes.len(), n_head);
            let log2 = (n_head as f32).log2().floor() as u32;
            let n_head_log2 = 1usize << log2;

            for branch in [(0usize, n_head_log2), (n_head_log2, n_head)] {
                for w in slopes[branch.0..branch.1].windows(2) {
                    assert!(
                        w[0] > w[1],
                        "branch {branch:?} must strictly decrease: {:?}",
                        slopes
                    );
                }
            }
            // Every slope is a positive, finite, sub-unity discount.
            for (h, &s) in slopes.iter().enumerate() {
                assert!(s > 0.0 && s <= 1.0, "head {h}: {s}");
            }
            // Head 0 is m0^1 = 2^(-max_bias / n_head_log2): 0.25 at n_head 4,
            // 0.5 at 8 and 12, 2^-0.5 at 16. Derived rather than hardcoded so
            // the assertion holds at every width the loop covers.
            let log2f = (n_head as f32).log2().floor();
            let expected_head0 =
                2.0f32.powf(-MAX_ALIBI_BIAS_JINA_V2 / (1usize << log2f as u32) as f32);
            assert!(
                (slopes[0] - expected_head0).abs() < 1e-6,
                "head 0 was {}, expected {expected_head0}",
                slopes[0]
            );
            if n_head_log2 < n_head {
                assert!(
                    (slopes[n_head_log2] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6,
                    "branch switch head was {}",
                    slopes[n_head_log2]
                );
            }
        }
    }

    #[test]
    fn jina_bert_v2_is_the_only_alibi_variant() {
        // This is the decision the ALiBi branch in the forward keys off. If it
        // ever stops being jina-only, the hardcoded
        // MAX_ALIBI_BIAS_JINA_V2 has to be re-derived per variant.
        assert!(BertVariant::JinaBertV2.uses_alibi());
        for other in [
            BertVariant::Bert,
            BertVariant::NomicBert,
            BertVariant::NomicBertMoe,
        ] {
            assert!(
                !other.uses_alibi(),
                "{other:?} must not apply ALiBi under the 8.0 bias"
            );
        }
    }

    #[test]
    fn a_nonzero_bias_means_alibi_is_on() {
        // `llama-model.cpp:1483` — `use_alibi = (f_max_alibi_bias > 0.0f)`.
        // The two jina GGUFs shipped on ModelScope omit
        // `{arch}.attention.max_alibi_bias` entirely, and `llama-model.cpp`
        // never reads that key for this arch anyway; jina-bert-v2.cpp:5 sets
        // 8.0f unconditionally. So "the GGUF lacks the key" must NOT be read as
        // "ALiBi disabled" - this pins the correct constant.
        assert!(
            MAX_ALIBI_BIAS_JINA_V2 > 0.0,
            "jina-bert-v2 must keep its 8.0 bias; defaulting to 0.0 would \
             silently disable ALiBi for every jina-bert-v2 model"
        );
        // ...and the slopes derived from it must not be all-ones, which is what
        // the fully-disabled path produces.
        let alibi = alibi_slopes(12, MAX_ALIBI_BIAS_JINA_V2);
        let disabled = alibi_slopes(12, 0.0);
        assert_ne!(alibi, disabled, "bias 8.0 must not collapse to no ALiBi");
    }
}
