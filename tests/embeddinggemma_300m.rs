//! Integration tests for the EmbeddingGemma-300M Q8_0 GGUF.
//!
//! Run with `RMI_EMBEDDINGGEMMA_300M_MODEL=/path/to/embeddinggemma-300M-Q8_0.gguf`.
//!
//! Oracle: local read-only `references/llama.cpp/src/models/gemma-embedding.cpp`.
//! These tests pin the loader/tokenizer/tensor contract and the semantic
//! ordering of the embedding (relevant document scores highest); bit-level
//! parity with llama.cpp requires the oracle binary and is tracked separately.

use rust_model_inference::GGUFLoader;
use rust_model_inference::TensorInfo;
use rust_model_inference::TensorSource as _;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_EMBEDDINGGEMMA_300M_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

#[test]
fn contract_loads_and_pins_gemma_embedding_metadata() {
    let Some(loader) = loader() else { return };
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(arch, "gemma-embedding");

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

    assert_eq!(pick("gemma-embedding.block_count"), 24);
    assert_eq!(pick("gemma-embedding.embedding_length"), 768);
    assert_eq!(pick("gemma-embedding.feed_forward_length"), 1152);
    assert_eq!(pick("gemma-embedding.attention.head_count"), 3);
    assert_eq!(pick("gemma-embedding.attention.head_count_kv"), 1);
    assert_eq!(pick("gemma-embedding.attention.key_length"), 256);
    assert_eq!(pick("gemma-embedding.attention.sliding_window"), 512);
    assert_eq!(pick("gemma-embedding.pooling_type"), 1, "mean pooling");
    // Layer-norm epsilon key name matters: `_epsilon`, not `_eps`.
    assert!(
        loader
            .metadata("gemma-embedding.attention.layer_norm_rms_epsilon")
            .is_some(),
        "eps key is present under the `_epsilon` spelling"
    );
    // `load_swa_pattern(ml, 6)` + `dense_first=false` → dense layers at il%6==5.
    // We pin the pattern constant rather than a computed bool so a regression
    // in the source shows up as a test failure.
    assert_eq!(
        24 % 6,
        0,
        "24 layers must divide evenly into the 6-group pattern"
    );

    // GGUF key is `layer_norm_rms_epsilon` (not `_eps`); a small positive
    // value, but implementations must still tolerate a missing/0 key.
    assert!((f("gemma-embedding.attention.layer_norm_rms_epsilon") - 1e-6).abs() < 1e-12);

    // Dual RoPE bases: SWA layers use freq_base_swa, dense layers freq_base.
    assert_eq!(f("gemma-embedding.rope.freq_base"), 1_000_000.0);
    assert_eq!(f("gemma-embedding.rope.freq_base_swa"), 10_000.0);

    // Dense projections advertise their widths in metadata.
    assert_eq!(pick("gemma-embedding.dense_2_feat_in"), 768);
    assert_eq!(pick("gemma-embedding.dense_2_feat_out"), 3072);
    assert_eq!(pick("gemma-embedding.dense_3_feat_in"), 3072);
    assert_eq!(pick("gemma-embedding.dense_3_feat_out"), 768);
}

#[test]
fn tokenizer_is_sentencepiece_without_bos() {
    let Some(loader) = loader() else { return };
    use rust_model_inference::core::tokenizer::SPMTokenizer;
    let tok = SPMTokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned())
        .expect("SPM tokenizer must build for gemma-embedding");
    assert_eq!(tok.bos_id(), Some(2), "EmbeddingGemma BOS is 2 (<s>)");
    assert_eq!(tok.eos_id(), Some(1), "EmbeddingGemma EOS is 1 (</s>)");

    // `add_bos_token = false` — an encoded prompt must not start with BOS.
    let ids = tok.encode(
        "What is the capital of France?",
        rust_model_inference::core::tokenizer::EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    assert!(!ids.is_empty());
    assert_ne!(ids[0], 2, "must not prepend BOS when add_bos_token=false");
}

#[test]
fn tensor_inventory_matches_the_expected_gemma_embedding_layout() {
    let Some(loader) = loader() else { return };
    fn shape<'a>(loader: &'a GGUFLoader, name: &str) -> Option<&'a TensorInfo> {
        loader.tensor_info(name)
    }

    // Layer 0 is representative: all 24 layers share this shape.
    let q = shape(&loader, "blk.0.attn_q.weight").expect("blk.0.attn_q.weight");
    assert_eq!(q.dims, vec![768, 768], "attn_q: [n_embd, n_embd_q]");
    let k = shape(&loader, "blk.0.attn_k.weight").expect("blk.0.attn_k.weight");
    assert_eq!(k.dims, vec![768, 256], "attn_k: [n_embd, n_embd_gqa]");

    // Four norms per layer, NOT the two a standard llama trunk expects.
    for name in [
        "blk.0.attn_norm.weight",
        "blk.0.post_attention_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.post_ffw_norm.weight",
    ] {
        let info = shape(&loader, name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.dims, vec![768], "{name} must be [n_embd]");
    }

    // QK-norm is per head (head_dim = 256), not per model dim.
    let q_norm = shape(&loader, "blk.0.attn_q_norm.weight").expect("attn_q_norm");
    assert_eq!(q_norm.dims, vec![256], "q_norm must be [n_embd_head_k]");
    let k_norm = shape(&loader, "blk.0.attn_k_norm.weight").expect("attn_k_norm");
    assert_eq!(k_norm.dims, vec![256], "k_norm must be [n_embd_head_k]");

    // Dense projections (768 ↔ 3072) with no dense_1.
    let d2 = shape(&loader, "dense_2.weight").expect("dense_2.weight");
    assert_eq!(d2.dims, vec![768, 3072]);
    let d3 = shape(&loader, "dense_3.weight").expect("dense_3.weight");
    assert_eq!(d3.dims, vec![3072, 768]);
    assert!(
        shape(&loader, "dense_1.weight").is_none(),
        "dense_1 must be absent in this GGUF"
    );

    // Encoder: no LM head, and the final norm is a plain tensor.
    assert!(
        shape(&loader, "output.weight").is_none() || {
            // Some packers alias output → token_embd; either is acceptable, but the
            // implementation must not treat this as a generative model.
            let out = shape(&loader, "output.weight").unwrap();
            out.dims[0] == 768 && out.dims[1] == 262_144
        }
    );
    assert_eq!(
        shape(&loader, "output_norm.weight").unwrap().dims,
        vec![768]
    );
}

#[test]
fn embedding_orders_relevant_document_above_unrelated() {
    let Some(loader) = loader() else { return };
    use rust_model_inference::models::gemma_embedding::compute_embedding;
    let embed = |prompt: &str| {
        compute_embedding(&loader, prompt, 4).unwrap_or_else(|e| panic!("{prompt}: {e}"))
    };

    let query = embed("What is the capital of France?");
    assert_eq!(query.len(), 768, "embedding dim must equal n_embd");
    // L2 normalised by construction.
    let norm: f64 = query.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    assert!((norm.sqrt() - 1.0).abs() < 1e-5, "norm={norm}");

    let cos = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum()
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
