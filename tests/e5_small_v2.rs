//! Integration tests for `e5-small-v2-q8_0.gguf`.
//!
//! Run with `RMI_E5_SMALL_V2_MODEL=/path/to/e5-small-v2-q8_0.gguf`.
//!
//! This is the same plain-`bert` shape as gte-small (12 layers, 384 wide, 1536
//! FFN, 12 heads of 32, explicit `pooling_type = 1`, learned position table,
//! four projection bias groups), so it is not a new code path on its own. What
//! it adds is a second independent checkpoint of that combination, plus a
//! tokenizer-level quirk worth pinning: e5 models are trained with
//! `"query: "` / `"passage: "` prefixes (intfloat/e5-small-v2's README), so the
//! unsuffixed score is meaningful but not what the model is tuned for. The
//! tests here only assert the ordering, never an absolute threshold, so they
//! stay true with or without the prefixes.
//!
//! Oracle: local read-only `references/llama.cpp/src/models/bert.cpp`.

use rust_model_inference::core::tokenizer::{EncodeOptions, WPMTokenizer};
use rust_model_inference::GGUFLoader;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_E5_SMALL_V2_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

#[test]
fn contract_loads_and_pins_e5_small_v2_metadata() {
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

    // e5-small-v2 is the same size as gte-small: 12 x 384 / 1536 / 12 heads.
    assert_eq!(pick("bert.block_count"), 12);
    assert_eq!(pick("bert.embedding_length"), 384);
    assert_eq!(pick("bert.feed_forward_length"), 1536);
    assert_eq!(pick("bert.attention.head_count"), 12);
    assert_eq!(pick("bert.context_length"), 512);
    assert_eq!(
        pick("bert.embedding_length") / pick("bert.attention.head_count"),
        32,
        "head_dim"
    );
    assert!((f("bert.attention.layer_norm_epsilon") - 1e-12).abs() < 1e-18);
    assert_eq!(pick("bert.attention.causal"), 0);
    assert_eq!(pick("bert.pooling_type"), 1, "e5-small-v2 is mean-pooled");

    let model = loader
        .metadata("tokenizer.ggml.model")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(model, "bert");
    assert_eq!(pick("tokenizer.ggml.token_type_count"), 2);
    let tok_embd = loader
        .tensor_info("token_embd.weight")
        .expect("token_embd.weight");
    assert_eq!(tok_embd.dims, vec![384, 30522]);

    for absent in [
        "bert.attention.head_count_kv",
        "bert.max_alibi_bias",
        "bert.rope.freq_base",
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
    assert_eq!(cfg.n_head_kv, 12);
}

#[test]
fn wordpiece_and_the_e5_prefix_convention() {
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
    assert_eq!(
        ids,
        vec![101, 2054, 2003, 1996, 3007, 1997, 2605, 29632, 102]
    );

    // e5 does not encode its "query: " / "passage: " prefixes in the tokenizer;
    // they are plain text the caller prepends. WordPiece must therefore keep
    // them as ordinary subwords instead of a special-token fast path.
    let prefixed = tok.encode(
        "query: What is the capital of France?",
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert_eq!(prefixed.first().copied(), Some(101));
    assert_eq!(prefixed.last().copied(), Some(102));
    assert!(
        prefixed.len() > ids.len(),
        "the prefix must add real subwords, not be swallowed"
    );
    // Round-trip through decode: lowercase, `[CLS]`/`[SEP]` stripped.
    assert_eq!(
        tok.decode(&prefixed, false),
        "query: what is the capital of france?"
    );
}

#[test]
fn tensor_inventory_matches_the_plain_bert_graph() {
    let Some(loader) = loader() else { return };
    let shape = |name: &str| loader.tensor_info(name).expect(name).dims.clone();

    assert_eq!(shape("token_embd.weight"), vec![384, 30522]);
    assert_eq!(shape("token_types.weight"), vec![384, 2]);
    assert_eq!(shape("position_embd.weight"), vec![384, 512]);
    assert!(loader.tensor_info("pos_embd.weight").is_none());
    for proj in ["attn_q", "attn_k", "attn_v", "attn_output"] {
        assert_eq!(shape(&format!("blk.0.{proj}.weight")), vec![384, 384]);
        assert_eq!(shape(&format!("blk.0.{proj}.bias")), vec![384]);
    }
    assert!(loader.tensor_info("blk.0.attn_qkv.weight").is_none());
    assert_eq!(shape("blk.0.ffn_up.weight"), vec![384, 1536]);
    assert_eq!(shape("blk.0.ffn_up.bias"), vec![1536]);
    assert_eq!(shape("blk.0.ffn_down.weight"), vec![1536, 384]);
    assert_eq!(shape("blk.0.ffn_down.bias"), vec![384]);
    assert!(loader.tensor_info("blk.0.ffn_gate.weight").is_none());
    assert_eq!(shape("blk.0.attn_output_norm.weight"), vec![384]);
    assert_eq!(shape("blk.0.layer_output_norm.weight"), vec![384]);
    assert_eq!(loader.tensors().len(), 197);
}

#[test]
fn shuffled_word_order_changes_the_embedding() {
    let Some(loader) = loader() else { return };
    use rust_model_inference::models::bert_family::compute_embedding;
    let embed = |prompt: &str| {
        compute_embedding(&loader, prompt, 4).unwrap_or_else(|e| panic!("{prompt}: {e}"))
    };
    let ordered = embed("What is the capital of France?");
    let shuffled = embed("France the capital is What?");
    assert_eq!(ordered.len(), 384);
    assert!(
        !ordered
            .iter()
            .zip(shuffled.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "identical output for a reordered prompt means position_embd is ignored"
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
