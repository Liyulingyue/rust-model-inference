//! Integration tests for `bge-m3-q8_0.gguf`.
//!
//! Run with `RMI_BGE_M3_MODEL=/path/to/bge-m3-q8_0.gguf`.
//!
//! bge-m3 is `arch = "bert"` but it is an **XLM-Roberta** encoder, which makes
//! it the first model in this family where the architecture and the tokenizer
//! disagree with the pair every other check has used:
//!
//! | model | arch | `tokenizer.ggml.model` |
//! |---|---|---|
//! | bert-base-uncased / bge-small | `bert` | `bert` (WordPiece) |
//! | jina-bert-v2 / nomic-bert     | `jina-bert-v2` / `nomic-bert` | `bert` |
//! | nomic-bert-moe                 | `nomic-bert-moe` | `t5` (UGM) |
//! | **bge-m3**                    | **`bert`** | **`t5` (UGM)** |
//!
//! `compute_embedding` used to pick the tokenizer by *architecture*, which
//! sent bge-m3 into WordPiece and died with `Unsupported WPM
//! tokenizer.ggml.model "t5"; expected bert`. llama.cpp keys it off
//! `tokenizer.ggml.model` (`llama-vocab.cpp:1804+`), and so does the code now.
//!
//! Other things this model uniquely covers:
//! * `pooling_type = 2` (CLS) at 1024 dims — bge-small was CLS at 384, and the
//!   two mean-pooled English encoders were 384 / 768.
//! * `attention.layer_norm_epsilon = 1e-5`, the only `-eps` key in the family
//!   that is not 1e-12.
//! * `n_ctx_train = 8192`, so a `position_embd.weight` of [1024, 8192].
//! * `token_types.weight` with a **1-D** [1024] shape (token_type_count = 1),
//!   rather than the [n_embd, n] table the others carry.
//!
//! Oracle: local read-only `references/llama.cpp/src/models/bert.cpp`.

use rust_model_inference::GGUFLoader;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_BGE_M3_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

#[test]
fn contract_loads_and_pins_bge_m3_metadata() {
    let Some(loader) = loader() else { return };
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(arch, "bert");

    let pick = |k: &str| {
        loader
            .metadata(k)
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(0)
    };
    let f = |k: &str| {
        loader
            .metadata(k)
            .and_then(|v| v.to_f64())
            .unwrap_or(f64::NAN)
    };

    // XLM-Roberta-large trunk: 24 layers, 1024 wide, 4096 FFN, 16 heads of 64.
    assert_eq!(pick("bert.block_count"), 24);
    assert_eq!(pick("bert.embedding_length"), 1024);
    assert_eq!(pick("bert.feed_forward_length"), 4096);
    assert_eq!(pick("bert.attention.head_count"), 16);
    assert_eq!(
        pick("bert.embedding_length") / pick("bert.attention.head_count"),
        64
    );
    assert_eq!(pick("bert.context_length"), 8192);
    assert_eq!(pick("bert.attention.causal"), 0);

    // The only bert-family model that does not use 1e-12.
    assert!((f("bert.attention.layer_norm_epsilon") - 1e-5).abs() < 1e-10);

    // CLS pooling, at a width no other CLS-pooled model here has.
    assert_eq!(pick("bert.pooling_type"), 2, "bge-m3 pools CLS");

    // The whole point: arch=bert but an XLM-Roberta SentencePiece unigram
    // tokenizer. This is the combination that used to break the dispatch.
    let tok_model = loader
        .metadata("tokenizer.ggml.model")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(
        tok_model, "t5",
        "bge-m3 must ship the UGM tokenizer; the arch alone does not decide it"
    );
    assert_eq!(pick("tokenizer.ggml.token_type_count"), 1);
    assert_eq!(pick("tokenizer.ggml.bos_token_id"), 0);
    assert_eq!(pick("tokenizer.ggml.eos_token_id"), 2);

    // vocab 250002 is the XLM-Roberta multilingual vocab, not bert-base's 30522.
    let tok_embd = loader
        .tensor_info("token_embd.weight")
        .expect("token_embd.weight");
    assert_eq!(tok_embd.dims, vec![1024, 250002]);

    for absent in [
        "bert.attention.head_count_kv",
        "bert.max_alibi_bias",
        "bert.rope.freq_base",
        "bert.rope.dimension_count",
        "bert.expert_count",
        "bert.moe_every_n_layers",
    ] {
        assert!(loader.metadata(absent).is_none(), "{absent} must not exist");
    }

    let cfg = rust_model_inference::core::loader::model_config_from_source(&loader)
        .expect("model_config_from_source must resolve arch=bert");
    assert_eq!(cfg.n_embd, 1024);
    assert_eq!(cfg.n_layer, 24);
    assert_eq!(cfg.n_ff, 4096);
    assert_eq!(cfg.n_head, 16);
    assert_eq!(cfg.n_head_kv, 16);
}

#[test]
fn tokenizer_is_ugm_and_not_wordpiece() {
    let Some(loader) = loader() else { return };
    use rust_model_inference::core::tokenizer::Tokenizer;
    let tok =
        rust_model_inference::core::tokenizer::load_tokenizer(|k| loader.metadata(k).cloned())
            .expect("t5 must resolve to the UGM tokenizer");

    // 250002 tokens of XLM-Roberta vocab, and UGM's control ids, not WPM's
    // 101 / 102 [CLS] / [SEP].
    assert_eq!(tok.vocab_size(), 250002);
    assert_eq!(tok.bos_id(), Some(0));
    assert_eq!(tok.eos_id(), Some(2));

    // The dispatch in compute_embedding must reach here too. WPM would reject
    // this model outright, so a successful embedding means the arch-keyed
    // dispatch did not come back.
    use rust_model_inference::models::bert_family::compute_embedding;
    let embedding = compute_embedding(&loader, "What is the capital of France?", 4)
        .expect("compute_embedding must build the UGM tokenizer for arch=bert+t5");
    assert_eq!(embedding.len(), 1024);
}

#[test]
fn tensor_inventory_matches_the_xlm_roberta_bert_graph() {
    let Some(loader) = loader() else { return };
    let shape = |name: &str| loader.tensor_info(name).expect(name).dims.clone();

    assert_eq!(shape("token_embd.weight"), vec![1024, 250002]);
    assert_eq!(shape("token_embd_norm.weight"), vec![1024]);
    assert_eq!(shape("token_embd_norm.bias"), vec![1024]);
    assert_eq!(shape("position_embd.weight"), vec![1024, 8192]);
    assert!(loader.tensor_info("pos_embd.weight").is_none());

    // XLM-Roberta converts with token_type_count = 1, and the converter emits a
    // 1-D segment vector rather than the [n_embd, n] table the 2-type English
    // models carry. `decode_f32_row_public` reads the first n_embd words, so it
    // handles either shape; pin the shape so a converter change is visible.
    assert_eq!(shape("token_types.weight"), vec![1024]);

    // Split Q/K/V with all four bias groups, as for the other bert-arch models.
    for proj in ["attn_q", "attn_k", "attn_v", "attn_output"] {
        assert_eq!(shape(&format!("blk.0.{proj}.weight")), vec![1024, 1024]);
        assert_eq!(shape(&format!("blk.0.{proj}.bias")), vec![1024]);
    }
    assert!(loader.tensor_info("blk.0.attn_qkv.weight").is_none());
    assert_eq!(shape("blk.0.ffn_up.weight"), vec![1024, 4096]);
    assert_eq!(shape("blk.0.ffn_up.bias"), vec![4096]);
    assert_eq!(shape("blk.0.ffn_down.weight"), vec![4096, 1024]);
    assert_eq!(shape("blk.0.ffn_down.bias"), vec![1024]);
    assert!(loader.tensor_info("blk.0.ffn_gate.weight").is_none());
    assert_eq!(shape("blk.0.attn_output_norm.weight"), vec![1024]);
    assert_eq!(shape("blk.0.layer_output_norm.weight"), vec![1024]);

    // 16 tensors per layer (q/k/v/out weights + biases = 8, two LayerNorms
    // with biases = 4, ffn_up/ffn_down weights + biases = 4) across 24 layers,
    // plus the 5 top-level: token_embd, its norm+bias, token_types,
    // position_embd.
    assert_eq!(loader.tensors().len(), 24 * 16 + 5);
}

#[test]
fn cls_pooling_reads_the_first_row() {
    let Some(loader) = loader() else { return };
    use rust_model_inference::models::bert_family::compute_embedding;
    let embed = |prompt: &str| {
        compute_embedding(&loader, prompt, 4).unwrap_or_else(|e| panic!("{prompt}: {e}"))
    };

    let query = embed("What is the capital of France?");
    assert_eq!(query.len(), 1024);
    let norm: f64 = query.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    assert!((norm.sqrt() - 1.0).abs() < 1e-5, "norm={norm}");

    // CLS pooling reads the row of the lowest position, which for a single
    // fresh sequence is row 0 - the <s> row the UGM tokenizer prepends. If the
    // row selection were off by one the embedding would still look plausible
    // (it is a different token row) but stop matching the oracle, so the
    // observable here is only that the output stays finite, normalized, and
    // order-sensitive; the row index itself is pinned by the oracle-alignment
    // TODO in MODEL_ADAPT_PLAN.md.
    let shuffled = embed("France the capital is What?");
    assert!(
        !query
            .iter()
            .zip(shuffled.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "reordering must change the CLS embedding"
    );
}

#[test]
fn embedding_orders_relevant_document_above_unrelated() {
    let Some(loader) = loader() else { return };
    use rust_model_inference::models::bert_family::compute_embedding;
    let embed = |prompt: &str| {
        compute_embedding(&loader, prompt, 4).unwrap_or_else(|e| panic!("{prompt}: {e}"))
    };

    let query = embed("What is the capital of France?");
    let cos = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum::<f64>()
    };
    let positive = embed("The capital of France is Paris.");
    let related = embed("The capital of Germany is Berlin.");
    let unrelated = embed("Photosynthesis converts light into chemical energy.");

    let (s_pos, s_rel, s_unrel) = (
        cos(&query, &positive),
        cos(&query, &related),
        cos(&query, &unrelated),
    );
    assert!(
        s_pos > s_rel && s_rel > s_unrel,
        "semantic ordering broken: pos={s_pos} rel={s_rel} unrel={s_unrel}"
    );
    // bge-m3 separates harder than any other model here (observed 0.88 / 0.73 /
    // 0.46), so the gap assertion can be tighter than for the base encoders.
    assert!(s_pos > 0.8, "relevant similarity too low: {s_pos}");
    assert!(
        s_pos - s_unrel > 0.3,
        "separation collapsed: pos={s_pos} unrel={s_unrel}"
    );
}
