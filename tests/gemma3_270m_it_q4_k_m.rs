//! Integration test for the **standard** gemma-3-270m-it GGUF
//! metadata + tensor-shape contract (not BitNet).
//!
//! Run with
//! `RMI_GEMMA_3_270M_IT_MODEL=/path/to/gemma-3-270m-it-Q4_K_M.gguf`.
//!
//! # Scope
//!
//! Pins the GGUF layout for `unsloth/gemma-3-270m-it-GGUF`
//! (Q4_K_M, mixed quantization) so we know exactly what a future
//! standard-gemma3 forward path would need to consume:
//!
//! - `general.architecture = "gemma3"` (same as BitNet-270M)
//! - `general.file_type = 15` (Q4_K_M **base**, **not 40** as in BitNet)
//! - 18 layers × dims (640 / 2048 / 4 heads / 1 KV / head_dim 256)
//! - vocab=262144 (from `tokenizer.ggml.tokens` array length), ctx=32768
//! - `gemma3.attention.layer_norm_rms_epsilon = 1e-6`
//! - `gemma3.rope.freq_base = 1e6`
//! - **`gemma3.attention.sliding_window = 512`** — standard
//!   Gemma3 hybrid local/global attention (NOT in BitNet-270M,
//!   which omits `sliding_window`)
//! - SentencePiece tokenizer (`model="llama"`, `pre="default"`,
//!   **`tokens` + `scores` + `token_type` arrays present, NO
//!   `merges`** — pure SPM, identical layout to BitNet-270M)
//! - BOS=2, EOS=106, padding=0, unknown=3
//! - `add_bos_token = true`, `add_eos_token = false`
//! - Chat template at `tokenizer.chat_template`
//! - **Mixed quantization** per-tensor: 109 F32 + 81 Q5_0 +
//!   27 Q4K + 10 Q8_0 + 9 Q6K = 236 total
//!   (most weights Q5_0, sensitive layers higher-precision)
//! - No `*_norm_in` tensors, no I2_S tensors — pure quantized
//!   matmul weights, no BitLinear
//!
//! This test does NOT exercise a forward pass — the gemma3
//! trunk at `src/models/gemma3/trunk/forward.rs` is currently
//! BitNet-only (the standard gemma3 path returns
//! `Err("non-BitNet gemma3 forward not yet implemented")`).
//! See `docs/develop/TODO.md` for the planned standard path.

use rust_model_inference::{GGMLType, GGUFLoader, MetaValue};

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_GEMMA_3_270M_IT_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

fn pick(loader: &GGUFLoader, key: &str) -> usize {
    loader
        .metadata(key)
        .and_then(|v| v.to_u64())
        .map(|v| v as usize)
        .unwrap_or(0)
}

fn pick_f32(loader: &GGUFLoader, key: &str) -> f64 {
    loader
        .metadata(key)
        .and_then(|v| v.to_f64())
        .unwrap_or(f64::NAN)
}

fn pick_str(loader: &GGUFLoader, key: &str) -> String {
    loader
        .metadata(key)
        .and_then(|v| v.to_string_val())
        .unwrap_or_default()
        .to_string()
}

fn vocab_size(loader: &GGUFLoader) -> usize {
    // vocab is encoded as `tokenizer.ggml.tokens` array length
    loader
        .metadata("tokenizer.ggml.tokens")
        .and_then(|v| match v {
            MetaValue::Array(_, items) => Some(items.len()),
            _ => None,
        })
        .unwrap_or(0)
}

#[test]
fn gemma3_270m_it_contract_loads() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_270M_IT_MODEL not set");
        return;
    };
    // Architecture + size (matches BitNet-270M)
    assert_eq!(
        pick_str(&loader, "general.architecture"),
        "gemma3",
        "gemma-3-270m-it rides the gemma3 trunk"
    );
    // Quantization file type
    let file_type = pick(&loader, "general.file_type");
    assert_eq!(
        file_type, 15,
        "general.file_type must be 15 (Q4_K_M base); BitNet I2_S is 40"
    );
    assert_eq!(pick(&loader, "gemma3.block_count"), 18);
    assert_eq!(pick(&loader, "gemma3.embedding_length"), 640);
    assert_eq!(pick(&loader, "gemma3.attention.head_count"), 4);
    assert_eq!(pick(&loader, "gemma3.attention.head_count_kv"), 1);
    assert_eq!(pick(&loader, "gemma3.feed_forward_length"), 2048);
    assert_eq!(pick(&loader, "gemma3.attention.key_length"), 256);
    assert_eq!(pick(&loader, "gemma3.attention.value_length"), 256);
    assert_eq!(
        vocab_size(&loader),
        262144,
        "tokenizer.ggml.tokens array length"
    );
    assert_eq!(pick(&loader, "gemma3.context_length"), 32768);
    let rope_base: f64 = pick_f32(&loader, "gemma3.rope.freq_base");
    assert!(
        (rope_base - 1_000_000.0).abs() < 1.0,
        "rope.freq_base={rope_base}"
    );
    let eps: f64 = pick_f32(&loader, "gemma3.attention.layer_norm_rms_epsilon");
    assert!((eps - 1e-6).abs() < 1e-9, "norm_eps={eps}");
}

#[test]
fn gemma3_270m_it_uses_sliding_window_attention() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_270M_IT_MODEL not set");
        return;
    };
    let sw = pick(&loader, "gemma3.attention.sliding_window");
    assert_eq!(
        sw, 512,
        "standard Gemma3 declares sliding_window=512 (hybrid local/global attn); \
         the gemma3 trunk at src/models/gemma3/trunk/forward.rs does NOT yet apply it"
    );
}

#[test]
fn gemma3_270m_it_tokenizer_is_spm_no_merges() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_270M_IT_MODEL not set");
        return;
    };
    // SPM layout — same as BitNet-270M, even though it's the
    // "llama" model string and not "gemma". Both vocabularies
    // ship pure-SPM (scores + tokens + token_type, no merges).
    assert_eq!(
        pick_str(&loader, "tokenizer.ggml.model"),
        "llama",
        "gemma-3-270m-it declares the llama SPM model type"
    );
    assert_eq!(
        pick_str(&loader, "tokenizer.ggml.pre"),
        "default",
        "default pre-tokenizer"
    );
    assert!(
        loader.metadata("tokenizer.ggml.merges").is_none(),
        "gemma-3-270m-it ships NO merge rules — pure SPM, not BPE"
    );
    assert!(
        loader.metadata("tokenizer.ggml.scores").is_some(),
        "SPM requires token scores"
    );
    assert!(
        loader.metadata("tokenizer.ggml.tokens").is_some(),
        "SPM requires tokens list"
    );
    assert!(
        loader.metadata("tokenizer.ggml.token_type").is_some(),
        "gemma3 needs token_type for special-token classification"
    );
}

#[test]
fn gemma3_270m_it_tokenizer_special_ids() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_270M_IT_MODEL not set");
        return;
    };
    // Different from BitNet-270M: EOS is 106 (start_of_turn /
    // end_of_turn marker), not 1. padding=0 is shared (the raw
    // <pad> token).
    assert_eq!(pick(&loader, "tokenizer.ggml.bos_token_id"), 2);
    assert_eq!(pick(&loader, "tokenizer.ggml.eos_token_id"), 106);
    assert_eq!(pick(&loader, "tokenizer.ggml.padding_token_id"), 0);
    assert_eq!(pick(&loader, "tokenizer.ggml.unknown_token_id"), 3);
    let add_bos = loader
        .metadata("tokenizer.ggml.add_bos_token")
        .and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
        .unwrap_or(false);
    let add_eos = loader
        .metadata("tokenizer.ggml.add_eos_token")
        .and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
        .unwrap_or(true); // default true
    let add_space_prefix = loader
        .metadata("tokenizer.ggml.add_space_prefix")
        .and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
        .unwrap_or(false);
    assert!(add_bos, "gemma-3-270m-it requires BOS prepended");
    assert!(
        !add_eos,
        "gemma-3-270m-it does NOT auto-add EOS (it's a chat model \
         that handles EOS via <end_of_turn>)"
    );
    assert!(
        !add_space_prefix,
        "gemma-3-270m-it does NOT add a space prefix"
    );
}

#[test]
fn gemma3_270m_it_chat_template_present() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_270M_IT_MODEL not set");
        return;
    };
    let tmpl = loader
        .metadata("tokenizer.chat_template")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert!(
        tmpl.contains("<start_of_turn>"),
        "chat template must declare <start_of_turn> marker"
    );
    assert!(
        tmpl.contains("<end_of_turn>"),
        "chat template must declare <end_of_turn> marker"
    );
}

#[test]
fn gemma3_270m_it_tensor_inventory_mixed_quant_no_i2_s() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_270M_IT_MODEL not set");
        return;
    };
    let mut n_total = 0usize;
    let mut n_f32 = 0usize;
    let mut n_q5_0 = 0usize;
    let mut n_q4k = 0usize;
    let mut n_q6k = 0usize;
    let mut n_q8_0 = 0usize;
    let mut n_i2_s = 0usize;
    let mut n_other = 0usize;
    let mut n_norm_in = 0usize;
    for t in loader.tensors() {
        n_total += 1;
        match t.ggml_type {
            GGMLType::F32 => n_f32 += 1,
            GGMLType::Q5_0 => n_q5_0 += 1,
            GGMLType::Q4K => n_q4k += 1,
            GGMLType::Q6K => n_q6k += 1,
            GGMLType::Q8_0 => n_q8_0 += 1,
            GGMLType::I2_S => n_i2_s += 1,
            _ => n_other += 1,
        }
        if t.name.contains("norm_in") {
            n_norm_in += 1;
        }
    }
    assert_eq!(n_i2_s, 0, "standard gemma3 has NO I2_S tensors");
    assert_eq!(
        n_norm_in, 0,
        "standard gemma3 has NO per-projection *_norm_in tensors"
    );
    assert_eq!(n_other, 0, "no exotic types");
    assert_eq!(n_total, 236, "expected tensor count for 270m-it Q4_K_M");
    // The unsloth Q4_K_M conversion uses mixed types:
    //   F32  = 109 (norms + output head + token embedding)
    //   Q5_0 =  81 (the bulk of attention/FFN matmuls)
    //   Q4K  =  27 (less sensitive layers)
    //   Q6K  =   9 (extra precision on a handful)
    //   Q8_0 =  10 (probably the output projection / embed)
    assert_eq!(n_f32, 109);
    assert_eq!(n_q5_0, 81);
    assert_eq!(n_q4k, 27);
    assert_eq!(n_q6k, 9);
    assert_eq!(n_q8_0, 10);
}

#[test]
fn gemma3_270m_it_per_block_tensor_count_is_thirteen() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_270M_IT_MODEL not set");
        return;
    };
    // Each of the 18 layers carries exactly 13 tensors:
    //   4 layer norms (attn_norm, post_attention_norm, ffn_norm, post_ffw_norm)
    // + 2 QK norms (attn_q_norm, attn_k_norm)
    // + 7 quantized matmul tensors (attn_q/k/v/output + ffn_gate/up/down)
    // = 13 tensors per block.
    //
    // Plus per-model: token_embd.weight + output_norm.weight = 2 tensors.
    // Total: 18 × 13 + 2 = 236 tensors.
    let mut per_layer: std::collections::BTreeMap<usize, usize> = Default::default();
    for t in loader.tensors() {
        if let Some(rest) = t.name.strip_prefix("blk.") {
            if let Some((idx_str, _)) = rest.split_once('.') {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    *per_layer.entry(idx).or_insert(0) += 1;
                }
            }
        }
    }
    assert_eq!(per_layer.len(), 18, "expected 18 layer blocks");
    for (idx, count) in per_layer.iter() {
        assert_eq!(*count, 13, "layer {idx} has {count} tensors, expected 13");
    }
}

#[test]
fn gemma3_270m_it_metadata_differs_from_bitnet_270m() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_270M_IT_MODEL not set");
        return;
    };
    // Side-by-side diff between standard 270m-it and BitNet-270M
    // on the metadata keys that distinguish them.
    let standard_keys = ["gemma3.attention.sliding_window"];
    for key in standard_keys {
        assert!(
            loader.metadata(key).is_some(),
            "standard gemma3-270m-it must declare `{key}` (BitNet-270M does not)"
        );
    }
    let bitnet_only_keys = ["gemma3.pooling_type"];
    for key in bitnet_only_keys {
        // `pooling_type` only exists for the BitNet embedding
        // contract; the standard IT model is a text-generation model
        // and doesn't set it.
        assert!(
            loader.metadata(key).is_none(),
            "standard gemma3-270m-it must NOT declare `{key}` (BitNet-270M only)"
        );
    }
    // Different EOS token id (BitNet-270M uses 1, gemma3-270m-it uses 106)
    let eos = pick(&loader, "tokenizer.ggml.eos_token_id");
    assert_ne!(eos, 1, "gemma3 standard uses EOS=106, not 1");
    assert_eq!(eos, 106, "<end_of_turn> marker is the EOS");
}
