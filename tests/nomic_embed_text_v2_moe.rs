//! Integration tests for `nomic-embed-text-v2-moe-q8_0.gguf`.
//!
//! Run with
//! `RMI_NOMIC_EMBED_TEXT_V2_MOE_MODEL=/path/to/nomic-embed-text-v2-moe-q8_0.gguf`.
//!
//! Oracle: `references/llama.cpp/src/models/nomic-bert-moe.cpp` + the shared
//! `bert.cpp` graph. Bit-level parity still needs the oracle binary and is
//! tracked separately.
//!
//! This model is the XLM build of nomic-bert with `expert_count=8`,
//! `expert_used_count=2`, and the UGM tokenizer. Dense layers
//! (`l % 2 == 0`) keep the plain `ffn_up`/`ffn_down` GELU graph; MoE layers
//! (`l % 2 == 1`) replace them with `ffn_gate_inp` + `ffn_up_exps` +
//! `ffn_down_exps`.

use rust_model_inference::core::tokenizer::{EncodeOptions, Tokenizer};
use rust_model_inference::GGUFLoader;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_NOMIC_EMBED_TEXT_V2_MOE_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

#[test]
fn contract_loads_and_pins_nomic_bert_moe_metadata() {
    let Some(loader) = loader() else { return };
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(arch, "nomic-bert-moe");

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

    assert_eq!(pick("nomic-bert-moe.block_count"), 12);
    assert_eq!(pick("nomic-bert-moe.embedding_length"), 768);
    assert_eq!(pick("nomic-bert-moe.feed_forward_length"), 3072);
    assert_eq!(pick("nomic-bert-moe.attention.head_count"), 12);
    assert_eq!(pick("nomic-bert-moe.context_length"), 2048);
    assert_eq!(pick("nomic-bert-moe.pooling_type"), 1, "mean pooling");
    assert_eq!(pick("nomic-bert-moe.attention.causal"), 0);
    assert_eq!(pick("nomic-bert-moe.expert_count"), 8);
    assert_eq!(pick("nomic-bert-moe.expert_used_count"), 2);
    assert_eq!(pick("nomic-bert-moe.moe_every_n_layers"), 2);

    assert!((f("nomic-bert-moe.attention.layer_norm_epsilon") - 1e-5).abs() < 1e-10);
    assert!((f("nomic-bert-moe.rope.freq_base") - 10000.0).abs() < 1.0);
    assert!(loader
        .metadata("nomic-bert-moe.rope.dimension_count")
        .is_none());

    let cfg = rust_model_inference::core::loader::model_config_from_source(&loader)
        .expect("model_config_from_source must resolve nomic-bert-moe");
    assert_eq!(cfg.n_embd, 768);
    assert_eq!(cfg.n_layer, 12);
    assert_eq!(cfg.n_ff, 3072);
    assert!((cfg.rope_freq_base - 10000.0).abs() < 1.0);
}

#[test]
fn variant_is_distinct_and_uses_moe() {
    use rust_model_inference::models::bert_family::weights::BertVariant;
    assert_eq!(
        BertVariant::from_arch("nomic-bert-moe"),
        Some(BertVariant::NomicBertMoe)
    );
    assert!(BertVariant::NomicBertMoe.uses_moe());
    assert!(!BertVariant::NomicBertMoe.uses_alibi());
    assert!(!BertVariant::NomicBertMoe.uses_gelu_gate());
    // MoE dense layers use plain GELU; the MoE layers don't have a gate
    // projection (gate_inp is the router, not the activation).
    assert!(!BertVariant::NomicBertMoe.uses_silu_gate());
    assert!(BertVariant::NomicBertMoe.uses_rope());
    assert_ne!(BertVariant::NomicBertMoe, BertVariant::NomicBert);
    assert_eq!(BertVariant::from_arch("not-a-real-arch"), None);
}

#[test]
fn ugm_tokenizer_pins_ids_and_round_trip() {
    let Some(loader) = loader() else { return };
    let model = loader
        .metadata("tokenizer.ggml.model")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(model, "t5", "nomic-bert-moe ships the UGM tokenizer");
    let tok =
        rust_model_inference::core::tokenizer::load_tokenizer(|k| loader.metadata(k).cloned())
            .expect("tokenizer must load");
    assert_eq!(tok.vocab_size(), 250048);
    assert_eq!(tok.bos_id(), Some(0));
    assert_eq!(tok.eos_id(), Some(2));

    // Round-trip a sentence. The UGM tokenizer normalizes spaces to `▁`.
    let ids = tok.encode(
        "What is the capital of France?",
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert_eq!(ids.first().copied(), Some(0), "starts with [CLS]/<s>");
    assert_eq!(ids.last().copied(), Some(2), "ends with [EOS]");
    assert!(
        ids.len() >= 5 && ids.len() <= 32,
        "token count out of range"
    );
    let detok = tok.decode(&ids, false);
    assert!(
        detok.contains("capital"),
        "decode round-trip must keep word meaning"
    );
}

#[test]
fn tensor_inventory_mixes_dense_and_moe_per_two_layers() {
    let Some(loader) = loader() else { return };
    let shape = |name: &str| loader.tensor_info(name).expect(name).dims.clone();

    // Same fused-QKV attention as nomic-bert.
    assert_eq!(shape("blk.0.attn_qkv.weight"), vec![768, 2304]);
    assert_eq!(shape("blk.0.attn_output.weight"), vec![768, 768]);
    // Same two LayerNorms per layer (the only biases in the model).
    assert_eq!(shape("blk.0.attn_output_norm.weight"), vec![768]);
    assert_eq!(shape("blk.0.attn_output_norm.bias"), vec![768]);
    assert_eq!(shape("blk.0.layer_output_norm.weight"), vec![768]);
    assert_eq!(shape("blk.0.layer_output_norm.bias"), vec![768]);

    // Layer 0 is dense: it carries the plain ffn_up/ffn_down (GELU) and no
    // MoE expert tensors.
    assert_eq!(shape("blk.0.ffn_up.weight"), vec![768, 3072]);
    assert_eq!(shape("blk.0.ffn_down.weight"), vec![3072, 768]);
    assert!(loader.tensor_info("blk.0.ffn_gate_inp.weight").is_none());
    assert!(loader.tensor_info("blk.0.ffn_up_exps.weight").is_none());
    assert!(loader.tensor_info("blk.0.ffn_down_exps.weight").is_none());

    // Layer 1 is the first MoE layer: it carries only the 3-D expert
    // tensors and the router, and NOT the dense projections.
    assert!(loader.tensor_info("blk.1.ffn_up.weight").is_none());
    assert!(loader.tensor_info("blk.1.ffn_down.weight").is_none());
    assert_eq!(shape("blk.1.ffn_gate_inp.weight"), vec![768, 8]);
    assert_eq!(shape("blk.1.ffn_up_exps.weight"), vec![768, 3072, 8]);
    assert_eq!(shape("blk.1.ffn_down_exps.weight"), vec![3072, 768, 8]);

    // Even-numbered layers are dense, odd-numbered are MoE.
    for l in [0, 2, 4, 6, 8, 10] {
        assert!(
            loader
                .tensor_info(&format!("blk.{l}.ffn_up_exps.weight"))
                .is_none(),
            "even layer {l} must not have MoE experts"
        );
    }
    for l in [1, 3, 5, 7, 9, 11] {
        assert!(
            loader
                .tensor_info(&format!("blk.{l}.ffn_up.weight"))
                .is_none(),
            "odd layer {l} must not have a dense ffn_up"
        );
        assert!(
            loader
                .tensor_info(&format!("blk.{l}.ffn_down_exps.weight"))
                .is_some(),
            "odd layer {l} must have ffn_down_exps"
        );
    }
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
    assert!(s_pos > 0.8, "relevant similarity too low: {s_pos}");

    // The MoE FFN contribution used to be inflated: `moe_gate` renormalized the
    // top-2 softmax weights to sum to 1, which multiplies every MoE layer's
    // output by 1/(p_a + p_b) >= 1. `bert.cpp:173` selects the SOFTMAX gating
    // variant (not SOFTMAX_WEIGHT) and `bert.cpp:171` passes `norm_w = false`,
    // so the selected probabilities are used as-is. With the fix the
    // positive-vs-unrelated gap is wide enough to pin; before it, the
    // unrelated document still scored ~0.66 against the query.
    assert!(
        s_pos - s_unrel > 0.35,
        "MoE gating regression: pos={s_pos} unrel={s_unrel}; the top-2 softmax \
         weights are probably being renormalized to sum to 1 again"
    );
}

#[test]
fn moe_router_gate_keeps_the_unselected_expert_mass() {
    use rust_model_inference::models::bert_family::compute::moe_gate;

    // The real router is not reachable without the model, so this pins the
    // semantics directly against the values the 489MB GGUF produces in
    // practice: confident logits for the top-2, near-flat for the rest.
    let logits = [0.0f32, 0.9, 0.2, -0.3, 1.5, 0.4, -0.8, 0.1];
    let (selected, weights) = moe_gate(&logits, 2, 1.0);
    assert_eq!(selected, vec![4, 1], "top-2 by logit");
    let total: f64 = weights.iter().map(|&w| f64::from(w)).sum();
    assert!(
        total < 1.0,
        "selected weights must not be renormalized to 1 (got {total}); \
         bert.cpp:173 uses SOFTMAX, not SOFTMAX_WEIGHT"
    );
    assert!(
        total > 0.5,
        "a confident router should still hold most mass: {total}"
    );

    // Flat logits are the degenerate case the old code got most wrong: each of
    // 8 experts gets 1/8, so the top-2 hold exactly 0.25 and the old
    // renormalizing path quadrupled the FFN output.
    let flat = [0.0f32; 8];
    let (_, flat_weights) = moe_gate(&flat, 2, 1.0);
    let flat_total: f64 = flat_weights.iter().map(|&w| f64::from(w)).sum();
    assert!(
        (flat_total - 0.25).abs() < 1e-6,
        "flat router must keep only 0.25 of the mass, got {flat_total}"
    );
}
