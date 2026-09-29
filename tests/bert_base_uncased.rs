//! Integration tests for `bert-base-uncased-Q8_0.gguf`.
//!
//! Run with `RMI_BERT_BASE_UNCASED_MODEL=/path/to/bert-base-uncased-Q8_0.gguf`.
//!
//! This is the **canonical** `arch = "bert"` GGUF, and the first one the
//! `bert` variant has ever been run against. bge-small-en-v1.5 already
//! routed through the same variant, but it is a 384-dim / `pooling_type = 2`
//! (CLS) descendant, while bert-base-uncased is the 768-dim / mean-pooling
//! original that `bert.cpp` is written against.
//!
//! What this test adds over `bge_small_en_v1_5.rs`:
//!
//! 1. It actually executes the `position_embd.weight` path. bge barely
//!    depends on it (CLS pooling reads one row), but mean pooling over a
//!    permutation-invariant bag of words is *exactly* invariant, so a broken
//!    position table is invisible unless the embeddings are compared across
//!    token orders. `position_embd_makes_the_embedding_order_sensitive`
//!    pins that.
//! 2. It pins that `bert` carries all four projection bias groups *and* the
//!    learned position table *and* `token_types`, which together are what
//!    distinguish it from every other variant in the family.
//!
//! Oracle: local read-only `references/llama.cpp/src/models/bert.cpp`.

use rust_model_inference::core::tokenizer::{EncodeOptions, WPMTokenizer};
use rust_model_inference::GGUFLoader;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_BERT_BASE_UNCASED_MODEL")?;
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

    // bert-base-uncased dimensions: 12 layers, 768 wide, 3072 FFN, 12 heads of
    // 64, trained on 512-token contexts.
    assert_eq!(pick("bert.block_count"), 12);
    assert_eq!(pick("bert.embedding_length"), 768);
    assert_eq!(pick("bert.feed_forward_length"), 3072);
    assert_eq!(pick("bert.attention.head_count"), 12);
    assert_eq!(pick("bert.context_length"), 512);

    // `bert.cpp:5` reads exactly this key, and bert-base-uncased ships the
    // canonical 1e-12 (HF's default for this checkpoint).
    assert!((f("bert.attention.layer_norm_epsilon") - 1e-12).abs() < 1e-18);
    assert_eq!(pick("bert.attention.causal"), 0, "bidirectional attention");

    // WordPiece over bert-base-uncased's 30522-token vocab, two token types.
    let model = loader
        .metadata("tokenizer.ggml.model")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(model, "bert");
    assert_eq!(pick("tokenizer.ggml.token_type_count"), 2);

    // The GGUF ships no `pooling_type`, so the loader falls back to 1 (mean).
    // That is what this tests assumes, so pin the absence rather than a value.
    assert!(
        loader.metadata("bert.pooling_type").is_none(),
        "bert-base-uncased ships no pooling_type; mean must be the default"
    );

    // Plain MHA, and none of the rope / ALiBi / MoE knobs: this is the base
    // variant every other bert-family member is derived from.
    for absent in [
        "bert.attention.head_count_kv",
        "bert.attention.key_length",
        "bert.attention.value_length",
        "bert.attention.layer_norm_rms_epsilon",
        "bert.rope.freq_base",
        "bert.rope.dimension_count",
        "bert.max_alibi_bias",
        "bert.expert_count",
        "bert.moe_every_n_layers",
        "bert.cls_out",
    ] {
        assert!(
            loader.metadata(absent).is_none(),
            "{absent} must not exist for plain bert"
        );
    }

    // The loader's fallbacks must resolve without erroring.
    let cfg = rust_model_inference::core::loader::model_config_from_source(&loader)
        .expect("model_config_from_source must resolve arch=bert");
    assert_eq!(cfg.n_embd, 768);
    assert_eq!(cfg.n_layer, 12);
    assert_eq!(cfg.n_ff, 3072);
    assert_eq!(cfg.n_head, 12);
    assert_eq!(cfg.n_head_kv, 12, "KV width falls back to Q width");
}

#[test]
fn variant_is_the_plain_bert_base_case() {
    use rust_model_inference::models::bert_family::weights::BertVariant;
    assert_eq!(BertVariant::from_arch("bert"), Some(BertVariant::Bert));
    // None of the four switches the family grew for the other members.
    assert!(!BertVariant::Bert.uses_alibi());
    assert!(!BertVariant::Bert.uses_gelu_gate());
    assert!(!BertVariant::Bert.uses_silu_gate());
    assert!(!BertVariant::Bert.uses_rope());
    assert!(!BertVariant::Bert.uses_moe());
    assert_ne!(BertVariant::Bert, BertVariant::JinaBertV2);
    assert_ne!(BertVariant::Bert, BertVariant::NomicBert);
    assert_ne!(BertVariant::Bert, BertVariant::NomicBertMoe);
}

#[test]
fn wordpiece_matches_bert_base_uncased() {
    let Some(loader) = loader() else { return };
    let tok = WPMTokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned())
        .expect("WPM must build for tokenizer.ggml.model=bert");
    assert_eq!(tok.bos_id(), Some(101), "[CLS]");
    assert_eq!(tok.sep_id(), Some(102), "[SEP]");
    assert_eq!(tok.unk_id(), Some(100), "[UNK]");

    // The interior ids are the shared bert-base-uncased WordPiece split, which
    // is exactly what bge-small and nomic-embed-text-v1.5 produce too.
    let ids = tok.encode(
        "What is the capital of France?",
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert_eq!(
        ids,
        vec![101, 2054, 2003, 1996, 3007, 1997, 2605, 29632, 102],
        "WordPiece ids must match the bert-base-uncased vocab"
    );
    assert_eq!(tok.decode(&ids, false), "what is the capital of france?");
}

#[test]
fn tensor_inventory_is_the_split_qkv_bert_graph() {
    let Some(loader) = loader() else { return };
    let shape = |name: &str| loader.tensor_info(name).expect(name).dims.clone();

    // bert-base-uncased vocab (30522) and the 2-way segment table.
    assert_eq!(shape("token_embd.weight"), vec![768, 30522]);
    assert_eq!(shape("token_types.weight"), vec![768, 2]);
    // The embedding LayerNorm, with bias.
    assert_eq!(shape("token_embd_norm.weight"), vec![768]);
    assert_eq!(shape("token_embd_norm.bias"), vec![768]);

    // Defining trait #1: a learned absolute position table, required by
    // `bert.cpp:32` and named `position_embd.weight`
    // (`llama-arch.cpp:480`), NOT `pos_embd.weight`.
    assert_eq!(shape("position_embd.weight"), vec![768, 512]);
    assert!(loader.tensor_info("pos_embd.weight").is_none());

    // Defining trait #2: split Q/K/V, all three carrying a bias, plus the
    // attention-output bias. This is the four-bias shape that distinguishes
    // `bert` (and jina-bert-v2) from the bias-free nomic pair.
    assert_eq!(shape("blk.0.attn_q.weight"), vec![768, 768]);
    assert_eq!(shape("blk.0.attn_q.bias"), vec![768]);
    assert_eq!(shape("blk.0.attn_k.weight"), vec![768, 768]);
    assert_eq!(shape("blk.0.attn_k.bias"), vec![768]);
    assert_eq!(shape("blk.0.attn_v.weight"), vec![768, 768]);
    assert_eq!(shape("blk.0.attn_v.bias"), vec![768]);
    assert_eq!(shape("blk.0.attn_output.weight"), vec![768, 768]);
    assert_eq!(shape("blk.0.attn_output.bias"), vec![768]);
    assert!(loader.tensor_info("blk.0.attn_qkv.weight").is_none());

    // Defining trait #3: a single GELU FFN with a bias on both projections,
    // i.e. neither geglu (jina) nor swiglu (nomic).
    assert_eq!(shape("blk.0.ffn_up.weight"), vec![768, 3072]);
    assert_eq!(shape("blk.0.ffn_up.bias"), vec![3072]);
    assert_eq!(shape("blk.0.ffn_down.weight"), vec![3072, 768]);
    assert_eq!(shape("blk.0.ffn_down.bias"), vec![768]);
    assert!(loader.tensor_info("blk.0.ffn_gate.weight").is_none());

    // Two LayerNorms per layer, both with bias.
    assert_eq!(shape("blk.0.attn_output_norm.weight"), vec![768]);
    assert_eq!(shape("blk.0.attn_output_norm.bias"), vec![768]);
    assert_eq!(shape("blk.0.layer_output_norm.weight"), vec![768]);
    assert_eq!(shape("blk.0.layer_output_norm.bias"), vec![768]);

    // No MoE tensors anywhere.
    for l in 0..12 {
        assert!(loader
            .tensor_info(&format!("blk.{l}.ffn_gate_inp.weight"))
            .is_none());
        assert!(loader
            .tensor_info(&format!("blk.{l}.ffn_up_exps.weight"))
            .is_none());
    }
    // 197 tensors is the full inventory; a silent loader change that dropped
    // one of the four bias groups would move this count.
    assert_eq!(loader.tensors().len(), 197);
}

#[test]
fn position_embd_makes_the_embedding_order_sensitive() {
    let Some(loader) = loader() else { return };
    use rust_model_inference::models::bert_family::compute_embedding;
    let embed = |prompt: &str| {
        compute_embedding(&loader, prompt, 4).unwrap_or_else(|e| panic!("{prompt}: {e}"))
    };

    // Mean pooling is permutation-invariant, and attention with no positional
    // signal is permutation-equivariant. So if `position_embd.weight` were read
    // from the wrong row (or not at all), the two prompts below - the same
    // tokens in a different order - would produce bit-identical embeddings.
    // A working position table must make them differ.
    let ordered = embed("What is the capital of France?");
    let shuffled = embed("France the capital is What?");
    assert_eq!(ordered.len(), 768);
    assert_eq!(shuffled.len(), 768);

    let identical = ordered
        .iter()
        .zip(shuffled.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(
        !identical,
        "permuting the tokens must change the embedding; identical output means \
         position_embd.weight is being ignored (bag-of-words behaviour)"
    );

    let cos = ordered
        .iter()
        .zip(shuffled.iter())
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum::<f64>();
    assert!(
        (cos - 1.0).abs() > 1e-3,
        "reordered embedding is suspiciously close to the original (cos={cos}); \
         the position table may not be indexed by position"
    );
    // Same words, so it must still be a *related* sentence, not noise.
    assert!(cos > 0.7, "reordered sentence drifted too far: cos={cos}");
}

#[test]
fn embedding_orders_relevant_document_above_unrelated() {
    let Some(loader) = loader() else { return };
    use rust_model_inference::models::bert_family::compute_embedding;
    let embed = |prompt: &str| {
        compute_embedding(&loader, prompt, 4).unwrap_or_else(|e| panic!("{prompt}: {e}"))
    };

    let query = embed("What is the capital of France?");
    // L2-normalized by construction (`common.cpp:1893`, embd_normalize = 2).
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
    // bert-base-uncased is not a retrieval-trained encoder like bge, so the
    // absolute scores sit lower (observed ~0.83 / 0.79 / 0.63); the ordering is
    // what matters here.
    assert!(s_pos > 0.7, "relevant similarity too low: {s_pos}");
}

#[test]
fn over_long_prompt_is_rejected_with_a_useful_error() {
    // `context_length` is 512 here and it is also the row count of
    // `position_embd.weight`. A longer prompt has no defined positions, and
    // before the length check it died deep inside the position-table decode
    // with "position_embd.weight is not decodable as f32" - a decode failure
    // that never happened, hiding the real out-of-range index.
    let Some(loader) = loader() else { return };
    use rust_model_inference::models::bert_family::compute_embedding;
    let long = "word ".repeat(600);
    let error = compute_embedding(&loader, &long, 4)
        .expect_err("a 600-token prompt must not succeed on a 512-position model");
    assert!(
        error.contains("too long") && error.contains("512"),
        "the error must name the length problem, got: {error}"
    );
    // The old, misleading message must not be what the user sees.
    assert!(
        !error.contains("not decodable"),
        "an over-long prompt must not be reported as a decode failure: {error}"
    );

    // At the boundary it still works: exactly 512 positions is in range.
    let exact = "word ".repeat(502); // 502 words + [CLS]/[SEP] lands under 512
    let embedding = compute_embedding(&loader, &exact, 4)
        .unwrap_or_else(|e| panic!("an in-range prompt must embed: {e}"));
    assert_eq!(embedding.len(), 768);
}

#[test]
fn an_out_of_vocabulary_word_becomes_unk_not_silence() {
    // Regression test for the WordPiece rollback path. `llama-vocab.cpp:829-833`
    // rolls a word back on the first missed position and then *falls through*
    // to the "we didn't find any matches" check, emitting exactly one `[UNK]`.
    // Our implementation used to `return` from that branch instead, so the word
    // disappeared: `"🎉🎊"` produced `[CLS, SEP]` - the same token sequence as
    // `"   "` - and `"hello 🎉 world"` produced `hello world`. The embedding for
    // a prompt full of emoji was therefore bit-identical to a whitespace-only
    // prompt, i.e. the input was silently discarded.
    let Some(loader) = loader() else { return };
    let tok =
        WPMTokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned()).expect("WPM must build");
    assert_eq!(tok.unk_id(), Some(100), "[UNK]");

    let emoji = tok.encode(
        "\u{1F389}\u{1F38A}",
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert_eq!(
        emoji,
        vec![101, 100, 102],
        "an unmatchable word must become a single [UNK], not vanish"
    );

    let mixed = tok.encode(
        "hello \u{1F389} world",
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert_eq!(
        mixed,
        vec![101, 7592, 100, 2088, 102],
        "an OOV word between matching words keeps its [UNK]"
    );

    // Whitespace-only input still produces no word at all, so no [UNK] either.
    let spaces = tok.encode(
        "   ",
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    assert_eq!(spaces, vec![101, 102]);
}
