//! Integration tests for `bge-small-en-v1.5-q8_0.gguf`.
//!
//! Run with `RMI_BGE_SMALL_EN_V1_5_MODEL=/path/to/bge-small-en-v1.5-q8_0.gguf`.
//!
//! This is the first model that exercises the `arch = "bert"` variant of
//! `bert_family`, so it is where the differences between `bert` and the two
//! already-verified variants get pinned. Oracle: local read-only
//! `references/llama.cpp/src/models/bert.cpp`.
//!
//! Two things the `jina-bert-v2` / `nomic-bert` tests never touched:
//!
//! 1. The position table is named `position_embd.weight`, not
//!    `pos_embd.weight` (`llama-arch.cpp:480`), and `bert` *requires* it
//!    (`bert.cpp:32`), unlike nomic-bert which ropes instead.
//! 2. `pooling_type = 2` is CLS pooling, not mean. `bert` is the only
//!    currently-reachable variant that uses it, so the pooling branch has
//!    never run before this test.

use rust_model_inference::core::tokenizer::{EncodeOptions, WPMTokenizer};
use rust_model_inference::GGUFLoader;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_BGE_SMALL_EN_V1_5_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

#[test]
fn contract_loads_and_pins_bert_metadata() {
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

    assert_eq!(pick("bert.block_count"), 12);
    assert_eq!(pick("bert.embedding_length"), 384);
    assert_eq!(pick("bert.feed_forward_length"), 1536);
    assert_eq!(pick("bert.attention.head_count"), 12);
    assert_eq!(
        pick("bert.context_length"),
        512,
        "n_ctx_train caps pos rows"
    );
    assert!((f("bert.attention.layer_norm_epsilon") - 1e-12).abs() < 1e-18);
    assert_eq!(pick("bert.attention.causal"), 0, "bidirectional");

    // The one number that flips the pooling branch: CLS, not mean.
    assert_eq!(pick("bert.pooling_type"), 2, "CLS pooling, not mean");

    // Plain MHA, so the converted GGUF omits the KV key.
    assert!(loader.metadata("bert.attention.head_count_kv").is_none());

    let cfg = rust_model_inference::core::loader::model_config_from_source(&loader)
        .expect("model_config_from_source must resolve bert");
    assert_eq!(cfg.n_embd, 384);
    assert_eq!(cfg.n_layer, 12);
    assert_eq!(cfg.n_head, 12);
    assert_eq!(cfg.n_head_kv, 12);
    assert_eq!(cfg.n_ff, 1536);
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
    // Same bert-base-uncased vocab as nomic-bert, so the interior ids must
    // match that test exactly — a shared-vocab regression would show here.
    assert_eq!(
        ids,
        vec![101, 2054, 2003, 1996, 3007, 1997, 2605, 29632, 102],
        "lowercased, ▁-prefixed greedy longest-match"
    );
    assert_eq!(tok.decode(&ids, false), "what is the capital of france?");
}

#[test]
fn tensor_inventory_is_the_split_qkv_bert_graph() {
    let Some(loader) = loader() else { return };
    let shape = |name: &str| loader.tensor_info(name).expect(name);

    // bert-base-uncased vocab (30522).
    assert_eq!(shape("token_embd.weight").dims, vec![384, 30522]);
    assert_eq!(shape("token_types.weight").dims, vec![384, 2]);

    // The name that bit us: `llama-arch.cpp:480` spells it `position_embd`,
    // and `bert.cpp:32` makes it required, so a wrong spelling is a hard
    // failure rather than a silently skipped term.
    let pos = shape("position_embd.weight");
    assert_eq!(pos.dims, vec![384, 512], "[n_embd, n_ctx_train]");
    assert!(
        loader.tensor_info("pos_embd.weight").is_none(),
        "the shorter spelling does not exist in GGUF"
    );

    // Split q/k/v with full bias sets — unlike nomic-bert.
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
    ] {
        assert_eq!(shape(name).dims, vec![384, 384]);
    }
    for name in [
        "blk.0.attn_q.bias",
        "blk.0.attn_k.bias",
        "blk.0.attn_v.bias",
        "blk.0.attn_output.bias",
        "blk.0.ffn_up.bias",
        "blk.0.ffn_down.bias",
    ] {
        assert_eq!(shape(name).dims.len(), 1);
    }
    assert_eq!(shape("blk.0.attn_output.weight").dims, vec![384, 384]);

    // GELU FFN over a single ffn_up: no gate, unlike jina/nomic.
    assert!(loader.tensor_info("blk.0.ffn_gate.weight").is_none());
    assert_eq!(shape("blk.0.ffn_up.weight").dims, vec![384, 1536]);
    assert_eq!(shape("blk.0.ffn_down.weight").dims, vec![1536, 384]);

    // Two LayerNorms per layer, both with bias.
    for name in [
        "blk.0.attn_output_norm.weight",
        "blk.0.layer_output_norm.weight",
    ] {
        assert_eq!(shape(name).dims, vec![384]);
    }
    for name in [
        "blk.0.attn_output_norm.bias",
        "blk.0.layer_output_norm.bias",
    ] {
        assert_eq!(shape(name).dims, vec![384]);
    }

    // No rope bookkeeping, no QK norm, no MoE, no LM head.
    assert!(loader.tensor_info("blk.0.attn_q_norm.weight").is_none());
    assert!(loader.tensor_info("blk.0.attn_qkv.weight").is_none());
    assert!(loader.tensor_info("blk.0.ffn_gate_inp.weight").is_none());
    assert!(loader.tensor_info("output.weight").is_none());
}

#[test]
fn embedding_orders_relevant_document_above_unrelated() {
    let Some(loader) = loader() else { return };
    use rust_model_inference::models::bert_family::compute_embedding;
    let embed = |prompt: &str| {
        compute_embedding(&loader, prompt, 4).unwrap_or_else(|e| panic!("{prompt}: {e}"))
    };

    let query = embed("What is the capital of France?");
    assert_eq!(query.len(), 384, "embedding dim must equal n_embd");
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
}
