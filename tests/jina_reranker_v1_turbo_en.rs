//! Integration tests for `Jina-Bert-Implementation-38M-F16.gguf`
//! (`jina-reranker-v1-turbo-en`).
//!
//! Run with `RMI_JINA_RERANKER_V1_TURBO_MODEL=/path/to/Jina-Bert-Implementation-38M-F16.gguf`.
//!
//! Oracle: local read-only `references/llama.cpp/src/models/jina-bert-v2.cpp` +
//! `llama-vocab.cpp:2239-2246` (where `jina-v1-en` / `jina-v2-code` /
//! `roberta-bpe` are routed to `LLAMA_VOCAB_PRE_TYPE_GPT2` with `add_sep = true`).
//!
//! ## What's covered here
//!
//! 1. The contract pin: `arch = jina-bert-v2`, `tokenizer.ggml.model = gpt2`,
//!    `pre = jina-v1-en`, eps = 1e-12, 6 layers, 384 dims, 12 heads.
//! 2. The third tokenizer/arch mismatch in this family:
//!    `arch = jina-bert-v2` but `tokenizer.ggml.model = gpt2` (BPE), unlike
//!    `jina-embeddings-v2-base-en` (which is `arch = jina-bert-v2` +
//!    `tokenizer.ggml.model = bert` → WordPiece). `0897baa`'s
//!    tokenizer-by-model dispatch handles it, but jina-v1-en is a new
//!    `tokenizer.ggml.pre` that requires explicit handling
//!    (`tokenizer/mod.rs:413`); without that, `BPETokenizer::from_gguf_metadata`
//!    rejects the file.
//! 3. The 102-tensor inventory including the unique-to-this-GGUF `cls.weight`
//!    + `cls.bias` classification head (used for reranker scoring, but not
//!    reached through the embedding path).
//! 4. Embedding invariants: 384 dims, L2-normalized, no NaN/Inf.
//!
//! ## What is **not** verified here (deferred)
//!
//! - **Reranker scoring**. The reranker takes `query [SEP] document` and
//!   projects the pooled embedding through `cls.weight` + `cls.bias` to a
//!   single logit. The embedding path (`compute_embedding`) does not exercise
//!   `cls.weight` / `cls.bias` — they're loaded but never used. To actually
//!   verify cross-encoder ranking we'd need either (a) a CLI flag like
//!   `--rerank` that pipes `(query, document)` through the head, or (b) a
//!   `compute_rerank_score` function and a new test that asserts score
//!   ordering. Both are larger than this commit.
//! - **Bit-level oracle alignment**. The "TODO: BERT 家族位级 oracle 对齐"
//!   block in `docs/develop/MODEL_ADAPT_PLAN.md` is parked (per its own
//!   warning: "暂缓, 勿与他方核对工作并行").
//!
//! Because the reranker model is not designed to produce semantically
//! meaningful *standalone* embeddings (cross-encoders are optimized for the
//! CLS head, not raw vector geometry), we deliberately do NOT assert
//! "relevant doc above unrelated doc" on the embedding. Per
//! `docs/develop/MODEL_ADAPT_PLAN.md` §"jina-reranker-v1-turbo-en" this PR
//! only verifies "加载 + 输出合理 embedding", not reranker ranking.

use rust_model_inference::GGUFLoader;
use rust_model_inference::models::bert_family::compute_embedding;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_JINA_RERANKER_V1_TURBO_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

fn pick(loader: &GGUFLoader, key: &str) -> usize {
    loader
        .metadata(key)
        .and_then(|v| v.to_u64())
        .map(|v| v as usize)
        .unwrap_or(0)
}

fn pick_f32(loader: &GGUFLoader, key: &str) -> f32 {
    loader
        .metadata(key)
        .and_then(|v| v.to_f64())
        .map(|v| v as f32)
        .unwrap_or(f32::NAN)
}

#[test]
fn contract_loads_and_pins_jina_reranker_metadata() {
    let Some(loader) = loader() else {
        return;
    };
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(arch, "jina-bert-v2", "reranker is `jina-bert-v2` arch");

    let pre = loader
        .metadata("tokenizer.ggml.pre")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(pre, "jina-v1-en", "jina-reranker uses jina-v1-en pre");

    let tok_model = loader
        .metadata("tokenizer.ggml.model")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(
        tok_model, "gpt2",
        "jina-reranker uses GPT-2 byte-level BPE (not WordPiece)"
    );

    // 6 layers / 384 dim / 12 heads(head_dim=32) / 1536 FFN — the only
    // `arch = jina-bert-v2` GGUF in this shape. (`jina-embeddings-v2-base-en`
    // is 12 layers / 768 dim / 12 heads; the size tag on this file is 38M,
    // consistent.)
    assert_eq!(pick(&loader, "jina-bert-v2.block_count"), 6);
    assert_eq!(pick(&loader, "jina-bert-v2.embedding_length"), 384);
    assert_eq!(
        pick(&loader, "jina-bert-v2.feed_forward_length"),
        1536,
        "FFN dim for a 384-embed jina-bert-v2"
    );
    assert_eq!(pick(&loader, "jina-bert-v2.attention.head_count"), 12);
    assert_eq!(pick(&loader, "jina-bert-v2.context_length"), 8192);

    let eps = pick_f32(&loader, "jina-bert-v2.attention.layer_norm_epsilon");
    assert!((eps - 1e-12).abs() < 1e-18, "eps = {eps}");

    // Causal-attn = false ⇒ bidirectional; ALiBi is the jina-bert-v2
    // unconditional default (`jina-bert-v2.cpp:5` hardcodes
    // `f_max_alibi_bias = 8.0f`, which the GGUF omits).
    let causal = match loader.metadata("jina-bert-v2.attention.causal") {
        Some(rust_model_inference::core::tensor::MetaValue::Bool(b)) => *b,
        _ => true,
    };
    assert!(!causal, "bi-directional");

    // No `head_count_kv` ⇒ plain MHA (KV width == Q width), like v2-base-en.
    assert!(loader
        .metadata("jina-bert-v2.attention.head_count_kv")
        .is_none());

    // BOS/EOS/SEP/padding token IDs: this is the `add_special` contract.
    // Notably `seperator_token_id == eos_token_id == 2` (typo on the GGUF
    // key), which is why we can use `add_eos = true` to also act as
    // `add_sep` without a separate flag in our tokenizer.
    assert_eq!(
        pick(&loader, "tokenizer.ggml.bos_token_id"),
        0,
        "<s> at id 0"
    );
    assert_eq!(
        pick(&loader, "tokenizer.ggml.eos_token_id"),
        2,
        "</s> at id 2"
    );
    assert_eq!(
        pick(&loader, "tokenizer.ggml.seperator_token_id"),
        2,
        "SEP id == EOS id == 2 (typo key 'seperator' is what GGUF ships)"
    );
    assert_eq!(
        pick(&loader, "tokenizer.ggml.padding_token_id"),
        1
    );
}

#[test]
fn tensor_inventory_has_102_tensors_and_the_cls_head() {
    let Some(loader) = loader() else {
        return;
    };

    let n_tensors = loader.tensors().len();
    assert_eq!(
        n_tensors, 102,
        "expected 102 tensors: 4 top-level + 6 × 16 per-layer + cls(2)"
    );

    let shape = |name: &str| loader.tensor_info(name).map(|t| t.dims.to_vec());

    // Top-level pieces.
    assert_eq!(shape("token_embd.weight"), Some(vec![384, 61056]));
    assert_eq!(shape("token_embd_norm.weight"), Some(vec![384]));
    assert_eq!(shape("token_embd_norm.bias"), Some(vec![384]));
    assert_eq!(shape("token_types.weight"), Some(vec![384, 2]));

    // CLS scoring head — present (this is what makes it a *reranker*).
    // `cls.weight [384]` is the linear projection from pooled embedding to a
    // scalar logit; `cls.bias [1]` is the scalar bias. Both unused by the
    // embedding path.
    let cls_w = shape("cls.weight");
    let cls_b = shape("cls.bias");
    assert_eq!(cls_w, Some(vec![384]), "cls.weight shape");
    assert_eq!(cls_b, Some(vec![1]), "cls.bias shape");

    // Spot-check a representative per-layer set. The block tensor layout is
    // identical to `jina-embeddings-v2-base-en` (same arch), just smaller:
    //   attn_k/q/v.{weight, bias}  [384, 384] / [384]
    //   attn_output.{weight, bias} [384, 384] / [384]
    //   attn_output_norm.{weight, bias} [384] / [384]
    //   ffn_gate.weight            [384, 1536]   ← GEGLU: two parallel
    //   ffn_up.weight              [384, 1536]   ← projections
    //   ffn_down.{weight, bias}    [1536, 384] / [384]
    //   layer_output_norm.{weight, bias} [384] / [384]
    for l in 0..6 {
        assert_eq!(shape(&format!("blk.{l}.attn_q.weight")), Some(vec![384, 384]));
        assert_eq!(shape(&format!("blk.{l}.attn_v.bias")), Some(vec![384]));
        assert_eq!(
            shape(&format!("blk.{l}.ffn_gate.weight")),
            Some(vec![384, 1536]),
            "blk.{l}.ffn_gate must exist (GEGLU)"
        );
        assert_eq!(
            shape(&format!("blk.{l}.layer_output_norm.weight")),
            Some(vec![384])
        );
    }

    // No RoPE / no `attn_q_norm` / no `position_embd.weight` — same as v2-base-en.
    assert!(loader.tensor_info("position_embd.weight").is_none());
    assert!(loader.tensor_info("blk.0.attn_q_norm.weight").is_none());
}

#[test]
fn bpe_tokenizer_with_jina_v1_en_pre_wraps_with_bos_and_eos() {
    use rust_model_inference::core::tokenizer::load_tokenizer;

    let Some(loader) = loader() else {
        return;
    };

    let get_meta = |k: &str| loader.metadata(k).cloned();

    // `load_tokenizer` dispatches on `tokenizer.ggml.model`: `gpt2` → BPE.
    // The `jina-v1-en` pre is the third token-specific string the BPE
    // recognizes (added in src/core/tokenizer/mod.rs:413 for this GGUF);
    // without that line, `from_gguf_metadata` errors out with
    // "Unsupported tokenizer.ggml.pre \"jina-v1-en\"".
    let tokenizer = load_tokenizer(get_meta).expect("BPE tokenizer must build");
    let encode = |text: &str| {
        tokenizer.encode(
            text,
            rust_model_inference::core::tokenizer::EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        )
    };

    // `add_bos_token = true` and `add_eos_token = true` are both set in this
    // GGUF; the encoded sequence must wrap `[BOS] text [EOS]` (where EOS
    // also serves as SEP since `seperator_token_id == eos_token_id == 2`).
    let ids = encode("hello world");
    assert_eq!(ids.first().copied(), Some(0), "first token must be <s>=0");
    assert_eq!(ids.last().copied(), Some(2), "last token must be </s>=2");
    assert!(ids.len() >= 3, "encoded seq must contain at least [BOS, ..., EOS]");

    // The byte-level GPT-2 BPE folds the leading space into a `Ġ` byte;
    // `hello world` should tokenize to non-empty byte tokens (we don't pin
    // specific IDs because the vocab is large, just that the segment
    // produces something).
    let segment = &ids[1..ids.len() - 1];
    assert!(
        segment.iter().all(|&id| id < 61056),
        "all segment IDs must be in vocab (got len={}, vocab=61056)",
        segment.len()
    );
}

#[test]
fn embedding_returns_a_384_dim_l2_normalized_real_vector() {
    let Some(loader) = loader() else {
        return;
    };

    let prompt = "What is the capital of France?";
    let embed = compute_embedding(&loader, prompt, 4).expect("embedding must succeed");

    assert_eq!(embed.len(), 384, "embedding dim must equal n_embd");

    let norm_sq: f64 = embed.iter().map(|v| f64::from(*v).powi(2)).sum();
    let norm = norm_sq.sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "L2-normalized embedding expected (norm={norm})"
    );

    // No NaN/Inf — this is the "loads + outputs reasonable embedding"
    // check the docs promise. The actual semantic ordering is a separate
    // (deferred) concern that needs the CLS head.
    for (i, v) in embed.iter().enumerate() {
        assert!(v.is_finite(), "non-finite value at dim {i}: {v}");
    }
}

#[test]
fn overlong_prompt_is_rejected_with_context_length_diagnostic() {
    let Some(loader) = loader() else {
        return;
    };

    // n_ctx_train = 8192 (this GGUF). We need a prompt that BPE-encodes to
    // more than 8192 tokens. `tokenizer.ggml.model = "gpt2"` packs
    // `Ġa` (id 268) as a single token, so each `"a "` produces one token
    // rather than two. 10000 `"a "` + 10000 `"b "` ⇒ ~10002 tokens ⇒
    // safely over the 8192 limit, but small enough that the encode + check
    // itself is cheap (the over-length error must fire *before* the
    // forward, not after we've paid for a 10k-token forward).
    let prompt = "a ".repeat(10_000) + &"b ".repeat(10_000);
    let result = compute_embedding(&loader, &prompt, 4);
    let err = result.expect_err("expected an over-length error");
    assert!(
        err.contains("too long") && err.contains("context_length"),
        "error should mention length and context_length, got: {err}"
    );
    assert!(
        err.contains("8192"),
        "error should pin the trained context, got: {err}"
    );
}