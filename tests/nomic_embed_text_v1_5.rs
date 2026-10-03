//! Integration tests for `nomic-embed-text-v1.5.Q8_0.gguf`.
//!
//! Run with `RMI_NOMIC_EMBED_TEXT_V1_5_MODEL=/path/to/nomic-embed-text-v1.5.Q8_0.gguf`.
//!
//! Oracle: local read-only `references/llama.cpp/src/models/bert.cpp` +
//! `nomic-bert.cpp`. This is the third variant of the shared `bert.cpp` graph
//! (`bert` / `jina-bert-v2` / `nomic-bert`) and it is the only one that ropes
//! Q and K, so the contract below pins the parts that make it different
//! rather than re-pinning what jina already covers.
//!
//! Bit-level parity with llama.cpp still needs the oracle binary and is
//! tracked separately, as for the other encoders.

use rust_model_inference::core::tokenizer::{EncodeOptions, WPMTokenizer};
use rust_model_inference::GGUFLoader;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_NOMIC_EMBED_TEXT_V1_5_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

#[test]
fn contract_loads_and_pins_nomic_bert_metadata() {
    let Some(loader) = loader() else { return };
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(arch, "nomic-bert");

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

    assert_eq!(pick("nomic-bert.block_count"), 12);
    assert_eq!(pick("nomic-bert.embedding_length"), 768);
    assert_eq!(pick("nomic-bert.feed_forward_length"), 3072);
    assert_eq!(pick("nomic-bert.attention.head_count"), 12);
    assert_eq!(pick("nomic-bert.context_length"), 2048);
    assert_eq!(pick("nomic-bert.pooling_type"), 1, "mean pooling");
    assert_eq!(pick("nomic-bert.attention.causal"), 0, "bidirectional");
    assert_eq!(pick("tokenizer.ggml.token_type_count"), 2);

    // BERT epsilon, not the RMSNorm spelling.
    assert!((f("nomic-bert.attention.layer_norm_epsilon") - 1e-12).abs() < 1e-18);

    // The one rope knob nomic-bert actually ships and the reason it cannot
    // reuse the 10 000 Hz default of every llama-family trunk: HF trains
    // nomic-bert with freq_base = 1000.
    let freq_base = f("nomic-bert.rope.freq_base");
    assert!(
        (freq_base - 1000.0).abs() < 1.0,
        "rope.freq_base must be 1000, got {freq_base}"
    );

    // No `rope.dimension_count`, so `n_rot` falls back to `n_embd/n_head` = 64.
    assert!(loader.metadata("nomic-bert.rope.dimension_count").is_none());

    // Plain MHA: the converted GGUF omits `attention.head_count_kv` and
    // `attention.{key,value}_length`, so all three default to 64/12.
    assert!(loader
        .metadata("nomic-bert.attention.head_count_kv")
        .is_none());
    assert!(loader.metadata("nomic-bert.attention.key_length").is_none());
    // The loader fallback must resolve instead of erroring.
    let cfg = rust_model_inference::core::loader::model_config_from_source(&loader)
        .expect("model_config_from_source must resolve nomic-bert");
    assert_eq!(cfg.n_head, 12);
    assert_eq!(cfg.n_head_kv, 12, "KV width falls back to Q width");
    assert_eq!(cfg.n_embd, 768);
    assert_eq!(cfg.n_layer, 12);
    assert_eq!(cfg.n_ff, 3072);
    assert_eq!(
        cfg.rope_freq_base, 1000.0,
        "loader must read the arch-namespaced rope.freq_base"
    );
}

#[test]
fn variant_is_distinct_from_the_other_bert_family_members() {
    use rust_model_inference::models::bert_family::weights::BertVariant;
    assert_eq!(
        BertVariant::from_arch("nomic-bert"),
        Some(BertVariant::NomicBert)
    );
    // nomic-bert is neither the ALiBi variant nor the GEGLU one.
    assert!(!BertVariant::NomicBert.uses_alibi());
    assert!(!BertVariant::NomicBert.uses_gelu_gate());
    assert!(BertVariant::NomicBert.uses_silu_gate());
    assert!(BertVariant::NomicBert.uses_rope());
    // It must not be reachable as one of the other two.
    assert_ne!(BertVariant::NomicBert, BertVariant::Bert);
    assert_ne!(BertVariant::NomicBert, BertVariant::JinaBertV2);
    // An unrelated arch must not silently become a BERT variant.
    assert_eq!(BertVariant::from_arch("qwen3"), None);
}

#[test]
fn wordpiece_uses_the_bert_tokenizer_with_cls_and_sep() {
    let Some(loader) = loader() else { return };
    let tok =
        WPMTokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned()).expect("WPM must build");
    assert_eq!(tok.bos_id(), Some(101), "[CLS]");
    assert_eq!(tok.sep_id(), Some(102), "[SEP]");
    assert_eq!(tok.unk_id(), Some(100), "[UNK]");

    let ids = tok.encode(
        "What is the capital of France?",
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert_eq!(ids.first().copied(), Some(101), "must start with [CLS]");
    assert_eq!(ids.last().copied(), Some(102), "must end with [SEP]");
    assert_eq!(tok.decode(&ids, false), "what is the capital of france ?");
}

#[test]
fn tensor_inventory_fuses_qkv_and_omits_every_projection_bias() {
    let Some(loader) = loader() else { return };
    let shape = |name: &str| loader.tensor_info(name).expect(name);

    for name in [
        "token_embd.weight",
        "token_embd_norm.weight",
        "token_embd_norm.bias",
        "token_types.weight",
    ] {
        assert!(shape(name).dims.len() == 1 || shape(name).dims.len() == 2);
    }
    // bert-base-uncased vocab (30522), not jina's extended 30528.
    assert_eq!(shape("token_embd.weight").dims, vec![768, 30522]);
    assert_eq!(shape("token_types.weight").dims, vec![768, 2]);

    // The defining difference of this variant: Q, K and V are one fused
    // projection of width 3*n_embd (768 -> 2304) with no separate q/k/v.
    assert_eq!(shape("blk.0.attn_qkv.weight").dims, vec![768, 2304]);
    for absent in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
    ] {
        assert!(
            loader.tensor_info(absent).is_none(),
            "{absent} must not exist"
        );
    }

    // nomic-bert ships NO projection bias at all, unlike jina-bert-v2.
    for absent in [
        "blk.0.attn_q.bias",
        "blk.0.attn_k.bias",
        "blk.0.attn_v.bias",
        "blk.0.attn_output.bias",
        "blk.0.ffn_up.bias",
        "blk.0.ffn_gate.bias",
        "blk.0.ffn_down.bias",
    ] {
        assert!(
            loader.tensor_info(absent).is_none(),
            "{absent} must not exist"
        );
    }
    assert_eq!(shape("blk.0.attn_output.weight").dims, vec![768, 768]);

    // Two LayerNorms per layer, both with bias (the only biases in the model).
    assert_eq!(shape("blk.0.attn_output_norm.weight").dims, vec![768]);
    assert_eq!(shape("blk.0.attn_output_norm.bias").dims, vec![768]);
    assert_eq!(shape("blk.0.layer_output_norm.weight").dims, vec![768]);
    assert_eq!(shape("blk.0.layer_output_norm.bias").dims, vec![768]);

    // SwiGLU: separate gate and up, both [n_embd, n_ff].
    assert_eq!(shape("blk.0.ffn_gate.weight").dims, vec![768, 3072]);
    assert_eq!(shape("blk.0.ffn_up.weight").dims, vec![768, 3072]);
    assert_eq!(shape("blk.0.ffn_down.weight").dims, vec![3072, 768]);

    // No learned absolute positions and no QK norm: nomic-bert positions by
    // RoPE (`bert.cpp:120-133` ropes only NOMIC_BERT / JINA_BERT_V3).
    assert!(loader.tensor_info("pos_embd.weight").is_none());
    assert!(loader.tensor_info("blk.0.attn_q_norm.weight").is_none());
    assert!(loader.tensor_info("blk.0.attn_k_norm.weight").is_none());
    // No LM head / classifier head.
    assert!(loader.tensor_info("output.weight").is_none());
    assert!(loader.tensor_info("cls.weight").is_none());
}

#[test]
fn embedding_orders_relevant_document_above_unrelated() {
    let Some(loader) = loader() else { return };
    use rust_model_inference::models::bert_family::compute_embedding;
    let embed = |prompt: &str| {
        compute_embedding(&loader, prompt, 4).unwrap_or_else(|e| panic!("{prompt}: {e}"))
    };

    let query = embed("What is the capital of France?");
    assert_eq!(query.len(), 768, "embedding dim must equal n_embd");
    let norm: f64 = query.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    assert!((norm.sqrt() - 1.0).abs() < 1e-5, "norm={norm}");

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
    assert!(s_pos > 0.7, "relevant similarity too low: {s_pos}");
}
