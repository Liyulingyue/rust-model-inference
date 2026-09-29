//! Integration tests for `gte-small-q8_0.gguf`.
//!
//! Run with `RMI_GTE_SMALL_MODEL=/path/to/gte-small-q8_0.gguf`.
//!
//! gte-small is the first `arch = "bert"` model here that combines all three of
//! the traits the other bert-arch checkers split between them:
//!
//! * a **384-dim / head_dim 32** body (like bge-small) rather than
//!   bert-base-uncased's 768 / 64, so the head_dim 32 path is covered by
//!   something other than a CLS-pooling model;
//! * **mean pooling** (`pooling_type = 1`) at that width — bge-small is CLS
//!   (2) and bert-base-uncased ships no key at all, so this is the first model
//!   that *explicitly* ships mean while also carrying
//!   `position_embd.weight`;
//! * the learned absolute position table plus `token_types`, i.e. the plain
//!   BERT encoder with no rope / ALiBi / MoE anywhere.
//!
//! Oracle: local read-only `references/llama.cpp/src/models/bert.cpp`.

use rust_model_inference::core::tokenizer::{EncodeOptions, WPMTokenizer};
use rust_model_inference::GGUFLoader;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_GTE_SMALL_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

#[test]
fn contract_loads_and_pins_gte_small_metadata() {
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

    // gte-small-en-v1.5: 12 layers, 384 wide, 1536 FFN, 12 heads of 32.
    assert_eq!(pick("bert.block_count"), 12);
    assert_eq!(pick("bert.embedding_length"), 384);
    assert_eq!(pick("bert.feed_forward_length"), 1536);
    assert_eq!(pick("bert.attention.head_count"), 12);
    assert_eq!(pick("bert.context_length"), 512);
    // head_dim = 384 / 12 = 32, the narrowest head in the family.
    assert_eq!(
        pick("bert.embedding_length") / pick("bert.attention.head_count"),
        32
    );

    assert!((f("bert.attention.layer_norm_epsilon") - 1e-12).abs() < 1e-18);
    assert_eq!(pick("bert.attention.causal"), 0, "bidirectional");

    // The point of this model: mean pooling shipped *explicitly*, at a width
    // that also carries a position table. bge-small is CLS at the same width,
    // and bert-base-uncased never ships the key at all.
    assert_eq!(pick("bert.pooling_type"), 1, "gte-small is mean-pooled");
    assert_eq!(pick("tokenizer.ggml.token_type_count"), 2);

    let model = loader
        .metadata("tokenizer.ggml.model")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(model, "bert");
    // Same bert-base-uncased vocab (30522) as bge-small and bert-base-uncased.
    let tok_embd = loader
        .tensor_info("token_embd.weight")
        .expect("token_embd.weight");
    assert_eq!(tok_embd.dims, vec![384, 30522]);

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
    assert_eq!(cfg.n_embd, 384);
    assert_eq!(cfg.n_layer, 12);
    assert_eq!(cfg.n_ff, 1536);
    assert_eq!(cfg.n_head, 12);
}

#[test]
fn variant_is_the_plain_bert_base_case() {
    use rust_model_inference::models::bert_family::weights::BertVariant;
    assert_eq!(BertVariant::from_arch("bert"), Some(BertVariant::Bert));
    assert!(!BertVariant::Bert.uses_alibi());
    assert!(!BertVariant::Bert.uses_gelu_gate());
    assert!(!BertVariant::Bert.uses_silu_gate());
    assert!(!BertVariant::Bert.uses_rope());
    assert!(!BertVariant::Bert.uses_moe());
}

#[test]
fn wordpiece_shares_the_bert_base_uncased_vocab() {
    let Some(loader) = loader() else { return };
    let tok = WPMTokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned())
        .expect("WPM must build for tokenizer.ggml.model=bert");
    assert_eq!(tok.bos_id(), Some(101), "[CLS]");
    assert_eq!(tok.sep_id(), Some(102), "[SEP]");
    let ids = tok.encode(
        "What is the capital of France?",
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    // Same interior ids as bert-base-uncased and bge-small: one shared
    // bert-base-uncased WordPiece vocab, only the encoder differs.
    assert_eq!(
        ids,
        vec![101, 2054, 2003, 1996, 3007, 1997, 2605, 29632, 102]
    );
    assert_eq!(tok.decode(&ids, false), "what is the capital of france?");
}

#[test]
fn tensor_inventory_has_the_position_table_and_all_four_biases() {
    let Some(loader) = loader() else { return };
    let shape = |name: &str| loader.tensor_info(name).expect(name).dims.clone();

    assert_eq!(shape("token_embd.weight"), vec![384, 30522]);
    assert_eq!(shape("token_types.weight"), vec![384, 2]);
    assert_eq!(shape("token_embd_norm.weight"), vec![384]);
    assert_eq!(shape("token_embd_norm.bias"), vec![384]);
    // Required by `bert.cpp:32`, named `position_embd.weight`
    // (`llama-arch.cpp:480`), 512 rows for n_ctx_train.
    assert_eq!(shape("position_embd.weight"), vec![384, 512]);
    assert!(loader.tensor_info("pos_embd.weight").is_none());

    // Split Q/K/V with four bias groups.
    for proj in ["attn_q", "attn_k", "attn_v"] {
        assert_eq!(shape(&format!("blk.0.{proj}.weight")), vec![384, 384]);
        assert_eq!(shape(&format!("blk.0.{proj}.bias")), vec![384]);
    }
    assert_eq!(shape("blk.0.attn_output.weight"), vec![384, 384]);
    assert_eq!(shape("blk.0.attn_output.bias"), vec![384]);
    assert!(loader.tensor_info("blk.0.attn_qkv.weight").is_none());

    // A single GELU FFN with both biases: not geglu (jina) nor swiglu (nomic).
    assert_eq!(shape("blk.0.ffn_up.weight"), vec![384, 1536]);
    assert_eq!(shape("blk.0.ffn_up.bias"), vec![1536]);
    assert_eq!(shape("blk.0.ffn_down.weight"), vec![1536, 384]);
    assert_eq!(shape("blk.0.ffn_down.bias"), vec![384]);
    assert!(loader.tensor_info("blk.0.ffn_gate.weight").is_none());
    assert_eq!(shape("blk.0.attn_output_norm.weight"), vec![384]);
    assert_eq!(shape("blk.0.layer_output_norm.weight"), vec![384]);
    // 197 tensors: a loader change that dropped a bias group would move this.
    assert_eq!(loader.tensors().len(), 197);
}

#[test]
fn mean_pooling_over_a_position_table_is_order_sensitive() {
    let Some(loader) = loader() else { return };
    use rust_model_inference::models::bert_family::compute_embedding;
    let embed = |prompt: &str| {
        compute_embedding(&loader, prompt, 4).unwrap_or_else(|e| panic!("{prompt}: {e}"))
    };

    // Same invariant as the bert-base-uncased test, re-checked at 384 dims with
    // mean pooling: permutation-invariant pooling composed with
    // permutation-equivariant attention is exactly invariant, so a dead
    // `position_embd.weight` would make these two bit-identical.
    let ordered = embed("What is the capital of France?");
    let shuffled = embed("France the capital is What?");
    assert_eq!(ordered.len(), 384);
    let identical = ordered
        .iter()
        .zip(shuffled.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(
        !identical,
        "permuting the tokens must change the embedding; identical output means \
         position_embd.weight is being ignored (bag-of-words behaviour)"
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
    assert!(s_pos > 0.9, "relevant similarity too low: {s_pos}");
}
