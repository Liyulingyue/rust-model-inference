//! Integration tests for `jina-embeddings-v2-base-en-q8_0.gguf`.
//!
//! Run with `RMI_JINA_V2_BASE_EN_MODEL=/path/to/jina-embeddings-v2-base-en-q8_0.gguf`.
//!
//! Oracle: llama.cpp b96806d96061049a5b574269b049bf6241d63d46.
//! These tests pin the loader, WordPiece wrapping and semantic ordering.
//! Full scalar checkpoint parity: `tools/oracle/jina_bert_v2/verify.py`.

use rust_model_inference::core::tokenizer::{EncodeOptions, WPMTokenizer};
use rust_model_inference::GGUFLoader;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_JINA_V2_BASE_EN_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

#[test]
fn contract_loads_and_pins_jina_bert_v2_metadata() {
    let Some(loader) = loader() else { return };
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(arch, "jina-bert-v2");

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

    assert_eq!(pick("jina-bert-v2.block_count"), 12);
    assert_eq!(pick("jina-bert-v2.embedding_length"), 768);
    assert_eq!(pick("jina-bert-v2.feed_forward_length"), 3072);
    assert_eq!(pick("jina-bert-v2.attention.head_count"), 12);
    assert_eq!(pick("jina-bert-v2.pooling_type"), 1, "mean pooling");
    // BERT epsilon, not the RMSNorm spelling — and much smaller than 1e-5.
    assert!((f("jina-bert-v2.attention.layer_norm_epsilon") - 1e-12).abs() < 1e-18);
    // No `attention.head_count_kv`: plain MHA, so KV width == Q width.
    assert!(loader
        .metadata("jina-bert-v2.attention.head_count_kv")
        .is_none());
}

#[test]
fn wordpiece_wraps_with_cls_and_sep_and_uses_phantom_space() {
    let Some(loader) = loader() else { return };
    let tok = WPMTokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned())
        .expect("WPM tokenizer must build for jina-bert-v2");
    // `llama-vocab.cpp:1982-1996` defaults for the bert model family.
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
    // `llama-vocab.cpp:3521-3543` wraps unconditionally on add_special.
    assert_eq!(ids.first().copied(), Some(101), "must start with [CLS]");
    assert_eq!(ids.last().copied(), Some(102), "must end with [SEP]");
    assert_eq!(
        ids,
        vec![101, 2054, 2003, 1996, 3007, 1997, 2605, 1029, 102],
        "lowercased, ▁-prefixed greedy longest-match"
    );
    // Round-trip drops the phantom prefix and the specials.
    assert_eq!(tok.decode(&ids, false), "what is the capital of france ?");
}

#[test]
fn tensor_inventory_matches_the_bert_graph_contract() {
    let Some(loader) = loader() else { return };
    let shape = |name: &str| loader.tensor_info(name).expect(name);

    // Everything the encoder needs is present, and the LM head is not.
    for name in [
        "token_embd.weight",
        "token_embd_norm.weight",
        "token_embd_norm.bias",
        "token_types.weight",
    ] {
        assert!(shape(name).dims.len() == 2 || shape(name).dims.len() == 1);
    }
    // `token_types` is [n_embd, n_token_types] and `bert.cpp:28` reads row 0.
    assert_eq!(shape("token_types.weight").dims, vec![768, 2]);
    // No `pos_embd` — jina-bert-v2 uses ALiBi (`bert.cpp:77-89`).
    assert!(loader.tensor_info("pos_embd.weight").is_none());

    let q = shape("blk.0.attn_q.weight");
    assert_eq!(q.dims, vec![768, 768], "attn_q: [n_embd, n_embd_q]");
    // Biases are required here, unlike the llama trunks.
    assert_eq!(shape("blk.0.attn_q.bias").dims, vec![768]);
    assert_eq!(shape("blk.0.attn_v.bias").dims, vec![768]);
    assert_eq!(shape("blk.0.attn_output.bias").dims, vec![768]);
    // Two LayerNorms per layer (attention output + layer output), both with bias.
    assert_eq!(shape("blk.0.attn_output_norm.weight").dims, vec![768]);
    assert_eq!(shape("blk.0.attn_output_norm.bias").dims, vec![768]);
    assert_eq!(shape("blk.0.layer_output_norm.weight").dims, vec![768]);
    assert_eq!(shape("blk.0.layer_output_norm.bias").dims, vec![768]);
    // GEGLU: separate gate and up, both [n_embd, n_ff].
    assert_eq!(shape("blk.0.ffn_gate.weight").dims, vec![768, 3072]);
    assert_eq!(shape("blk.0.ffn_up.weight").dims, vec![768, 3072]);
    assert_eq!(shape("blk.0.ffn_down.weight").dims, vec![3072, 768]);
    // No RoPE-position bookkeeping anywhere in the graph.
    assert!(loader.tensor_info("blk.0.attn_q_norm.weight").is_none());
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
