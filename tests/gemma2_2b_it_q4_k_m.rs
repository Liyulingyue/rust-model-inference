//! Integration test for the **standard** gemma-2-2b-it GGUF
//! metadata + tensor-shape contract.
//!
//! Run with
//! `RMI_GEMMA_2_2B_IT_MODEL=/path/to/gemma-2-2b-it-Q4_K_M.gguf`.
//!
//! # Scope
//!
//! Pins the GGUF layout for
//! `lmstudio-community/gemma-2-2b-it-GGUF` (Q4_K_M, 1.6 GiB) so the
//! llama trunk's gemma2 dispatch can consume it:
//!
//! - `general.architecture = "gemma2"`
//! - `general.file_type = 15` (Q4_K_M)
//! - 26 layers × dims (2304 / 9216 / 8 heads / 4 KV / head_dim 256)
//! - vocab = 256000 (from `tokenizer.ggml.tokens` array length)
//! - ctx = 8192 (no RoPE scaling — pre-trained base ctx)
//! - `gemma2.attention.layer_norm_rms_epsilon = 1e-6`
//! - `gemma2.rope.freq_base = 1e4` (vs 1e6 in gemma-3)
//! - `gemma2.attention.sliding_window = 4096`
//! - **Logit softcapping** (gemma2-specific, both present):
//!   `gemma2.attention.attn_logit_softcapping = 50.0`
//!   `gemma2.final_logit_softcapping = 30.0`
//! - Tensor layout: llama-style (`blk.{i}.attn_q/k/v/output`,
//!   `blk.{i}.ffn_gate/up/down`) + 4-norm sandwich per block
//!   (`attn_norm` + `post_attention_norm` + `ffn_norm` +
//!   `post_ffw_norm`).
//! - Standard SPM BPE tokenizer (`model = "llama"`, `pre = "default"`,
//!   no merges), BOS=2, EOS=1, add_bos_token=true (so BOS is emitted
//!   via `add_special=true`).

use rust_model_inference::{GGMLType, GGUFLoader, MetaValue};

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_GEMMA_2_2B_IT_MODEL")?;
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
    loader
        .metadata("tokenizer.ggml.tokens")
        .and_then(|v| match v {
            MetaValue::Array(_, items) => Some(items.len()),
            _ => None,
        })
        .unwrap_or(0)
}

#[test]
fn gemma2_2b_it_contract_loads() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_2_2B_IT_MODEL not set");
        return;
    };
    assert_eq!(pick_str(&loader, "general.architecture"), "gemma2");
    assert_eq!(pick(&loader, "general.file_type"), 15, "Q4_K_M file_type");
    assert_eq!(pick(&loader, "gemma2.block_count"), 26);
    assert_eq!(pick(&loader, "gemma2.embedding_length"), 2304);
    assert_eq!(pick(&loader, "gemma2.feed_forward_length"), 9216);
    assert_eq!(pick(&loader, "gemma2.attention.head_count"), 8);
    assert_eq!(pick(&loader, "gemma2.attention.head_count_kv"), 4);
    assert_eq!(pick(&loader, "gemma2.attention.key_length"), 256);
    assert_eq!(pick(&loader, "gemma2.attention.value_length"), 256);
    assert_eq!(vocab_size(&loader), 256000);
    assert_eq!(pick(&loader, "gemma2.context_length"), 8192);
    let eps: f64 = pick_f32(&loader, "gemma2.attention.layer_norm_rms_epsilon");
    assert!(
        (eps - 1e-6).abs() < 1e-9,
        "gemma2 RMS eps ~ 1e-6 (got {eps})"
    );
    // gemma2 rope freq base is 1e4 (vs gemma-3's 1e6) — different
    // attention geometry. The GGUF omits `gemma2.rope.freq_base`; the
    // core `model_config_from_source` hardcodes the 1e4 default for
    // `prefix == "gemma2"`. Pin the resolved config value here so a
    // future refactor of the loader default doesn't silently break
    // Gemma-2's RoPE.
    let config = rust_model_inference::model_config_from_source(&loader).unwrap();
    assert!(
        (config.rope_freq_base - 10_000.0).abs() < 1.0,
        "gemma2 rope_freq_base = 1e4 (got {})",
        config.rope_freq_base
    );
}

#[test]
fn gemma2_2b_it_has_logit_softcapping_and_sliding_window() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_2_2B_IT_MODEL not set");
        return;
    };
    // Logit softcapping is gemma2-specific (gemma-1 has final-only;
    // gemma-3 has neither). The llama trunk reads these via the
    // `arch_prefix.attn_logit_softcapping` (NO `attention.` infix,
    // matching llama.cpp's key spelling) and
    // `arch_prefix.final_logit_softcapping` keys.
    let attn_softcap: f64 = pick_f32(&loader, "gemma2.attn_logit_softcapping");
    assert!(
        (attn_softcap - 50.0).abs() < 1e-3,
        "attn_logit_softcapping = 50.0 (got {attn_softcap})"
    );
    let final_softcap: f64 = pick_f32(&loader, "gemma2.final_logit_softcapping");
    assert!(
        (final_softcap - 30.0).abs() < 1e-3,
        "final_logit_softcapping = 30.0 (got {final_softcap})"
    );
    let sw = pick(&loader, "gemma2.attention.sliding_window");
    assert_eq!(
        sw, 4096,
        "gemma-2 2B uses sliding_window=4096 (vs 1024 in 4B)"
    );
}

#[test]
fn gemma2_2b_it_tokenizer_is_spm_with_bos() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_2_2B_IT_MODEL not set");
        return;
    };
    assert_eq!(pick_str(&loader, "tokenizer.ggml.model"), "llama");
    assert_eq!(pick_str(&loader, "tokenizer.ggml.pre"), "default");
    assert!(
        loader.metadata("tokenizer.ggml.merges").is_none(),
        "gemma-2 is pure SPM, no BPE merges"
    );
    assert_eq!(pick(&loader, "tokenizer.ggml.bos_token_id"), 2);
    assert_eq!(pick(&loader, "tokenizer.ggml.eos_token_id"), 1);
    assert_eq!(pick(&loader, "tokenizer.ggml.padding_token_id"), 0);
    assert_eq!(pick(&loader, "tokenizer.ggml.unknown_token_id"), 3);
    // `add_bos_token = true` means the tokenizer prepends BOS via
    // `add_special = true` (mirrors Mistral / Zephyr behaviour).
    let add_bos = loader
        .metadata("tokenizer.ggml.add_bos_token")
        .and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
        .unwrap_or(false);
    assert!(
        add_bos,
        "gemma-2-it expects add_bos_token=true so BOS is emitted automatically"
    );
    // Chat template is the gemma-2-it jinja with <start_of_turn> /
    // <end_of_turn> markers. We don't parse jinja here; just pin the
    // presence of those literal substrings.
    let tmpl = pick_str(&loader, "tokenizer.chat_template");
    assert!(
        tmpl.contains("<start_of_turn>") && tmpl.contains("<end_of_turn>"),
        "gemma-2 chat_template uses <start_of_turn>/<end_of_turn>"
    );
}

#[test]
fn gemma2_2b_it_tensor_inventory_matches_4norm_sandwich() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_2_2B_IT_MODEL not set");
        return;
    };
    let mut n_total = 0usize;
    let mut n_f32 = 0usize;
    let mut n_q4k = 0usize;
    let mut n_q6k = 0usize;
    let mut n_q8_0 = 0usize;
    let mut n_f16 = 0usize;
    let mut per_layer_counts: std::collections::BTreeMap<usize, usize> = Default::default();
    for t in loader.tensors() {
        n_total += 1;
        match t.ggml_type {
            GGMLType::F32 => n_f32 += 1,
            GGMLType::F16 => n_f16 += 1,
            GGMLType::Q4K => n_q4k += 1,
            GGMLType::Q6K => n_q6k += 1,
            GGMLType::Q8_0 => n_q8_0 += 1,
            _ => {}
        }
        if let Some(rest) = t.name.strip_prefix("blk.") {
            if let Some((idx_str, _)) = rest.split_once('.') {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    *per_layer_counts.entry(idx).or_insert(0) += 1;
                }
            }
        }
    }
    // 26 layers × 11 tensors = 286 (4 F32 norms + 7 Q4K matmuls).
    // Plus `token_embd.weight` + `output_norm.weight` = 288 total.
    assert_eq!(
        n_total, 288,
        "expected 288 tensors (26 layers × 11 + token_embd + output_norm)"
    );
    assert_eq!(per_layer_counts.len(), 26, "expected 26 layer blocks");
    for (idx, count) in per_layer_counts.iter() {
        assert_eq!(
            *count, 11,
            "layer {idx} has {count} tensors, expected 11 (4 norms + 7 matmuls)"
        );
    }
    // Layer-norm tensors are F32, matmul weights are quantized. The
    // exact distribution depends on the GGUF quantization pass; just
    // check that the F32 count >= 26*4+1 = 105 (norms + output_norm)
    // and at least one quantized weight tensor exists.
    assert!(
        n_f32 >= 105,
        "expected ≥105 F32 tensors (104 layer norms + output_norm), got {n_f32}"
    );
    assert!(
        n_q4k + n_q6k + n_q8_0 >= 26 * 7,
        "expected ≥182 quantized matmul weights (26 layers × 7)"
    );
}

#[test]
fn gemma2_2b_it_layer_block_carries_four_norms() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_2_2B_IT_MODEL not set");
        return;
    };
    // Pin that every `blk.{l}` exposes the gemma2 4-norm sandwich
    // (vs gemma-3's 2-norm, llama's 2-norm). The llama trunk's
    // `try_load_post_norms` reads `post_attention_norm` /
    // `post_ffw_norm` and refuses to load if only one is present.
    for l in 0..26 {
        let attn_norm = format!("blk.{l}.attn_norm.weight");
        let post_attn_norm = format!("blk.{l}.post_attention_norm.weight");
        let ffn_norm = format!("blk.{l}.ffn_norm.weight");
        let post_ffw_norm = format!("blk.{l}.post_ffw_norm.weight");
        let names = [
            attn_norm.as_str(),
            post_attn_norm.as_str(),
            ffn_norm.as_str(),
            post_ffw_norm.as_str(),
        ];
        for n in names {
            assert!(
                loader.tensor_info(n).is_some(),
                "blk.{l} missing norm tensor {n}"
            );
        }
    }
}
