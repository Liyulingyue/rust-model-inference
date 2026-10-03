//! Integration test for the BitNet-Embeddings-270M GGUF metadata
//! contract.
//!
//! Run with `RMI_BITNET_EMBEDDING_270M_MODEL=/path/to/bitnet-embeddings-270m-bf16-i2_s.gguf`.
//!
//! # Scope
//!
//! This test pins the GGUF metadata + tensor-shape contract for
//! `microsoft/bitnet-embedding-270m` against the loader
//! (`src/core/loader.rs::GGUFLoader`). It does NOT exercise a
//! forward pass — see `tests/bitnet_embedding_270m_e2e_embed.rs`
//! for the engine's gemma3 trunk forward.
//!
//! # What this pins
//!
//! - `general.architecture = "gemma3"` (decoded by this engine's
//!   new `src/models/gemma3/` trunk — 4-norm sandwich, 640-dim
//!   hidden state, 4:1 GQA at head_dim=256, 18 layers)
//! - `general.file_type = 40` (Microsoft's BitNet I2_S
//!   quantization file type, distinct from Q4_K_M's file_type=12)
//! - 18 layers × dims (640 / 2048 / 4 heads / 1 KV / head_dim 256)
//! - vocab_size=262144 (multilingual SPM vocab)
//! - `gemma3.attention.layer_norm_rms_epsilon = 1e-6`
//! - `gemma3.rope.freq_base = 1e6` (Neox RoPE)
//! - `gemma3.pooling_type = 1` (BitNet last-token pooling
//!   convention, distinct from `gemma-embedding` which is mean)
//! - Tokenizer is SPM (`tokenizer.ggml.model = "llama"`,
//!   `tokenizer.ggml.pre = "default"`, `tokenizer.ggml.scores`
//!   populated, **no** `tokenizer.ggml.merges`)
//! - Tokenizer special IDs: BOS=2, EOS=1, padding=0, unknown=3
//! - `add_bos_token = true`, `add_eos_token = true`,
//!   `add_space_prefix = false`
//! - **362 tensors**: 236 F16 norms + 126 I2_S BitLinear weights
//!   (18 layers × 7 BitLinear layers = attn_q/k/v/o +
//!   ffn_gate/up/down)
//! - Per-layer tensor inventory: 20 tensors per block
//!   (`attn_norm`, `post_attention_norm`, `ffn_norm`,
//!   `post_ffw_norm`, `attn_q_norm`, `attn_k_norm`, plus
//!   7 `*_norm_in` RMSNorm + 7 I2_S BitLinear)
//! - Plus per-model `token_embd.weight` (F16) and `output_norm.weight`
//!   (F16). Total: 18 × 20 + 2 = 362.

use rust_model_inference::{GGUFLoader, GGMLType};

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_BITNET_EMBEDDING_270M_MODEL")?;
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

#[test]
fn bitnet_embedding_270m_contract_loads() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_BITNET_EMBEDDING_270M_MODEL not set");
        return;
    };
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(arch, "gemma3", "BitNet Embeddings 270M rides the gemma3 trunk");
    let file_type = loader
        .metadata("general.file_type")
        .and_then(|v| v.to_u64())
        .unwrap_or(0);
    assert_eq!(file_type, 40, "general.file_type must be 40 (BitNet I2_S)");
    assert_eq!(pick(&loader, "gemma3.block_count"), 18);
    assert_eq!(pick(&loader, "gemma3.embedding_length"), 640);
    assert_eq!(pick(&loader, "gemma3.attention.head_count"), 4);
    assert_eq!(pick(&loader, "gemma3.attention.head_count_kv"), 1);
    assert_eq!(pick(&loader, "gemma3.feed_forward_length"), 2048);
    assert_eq!(pick(&loader, "gemma3.attention.key_length"), 256);
    assert_eq!(pick(&loader, "gemma3.attention.value_length"), 256);
    assert_eq!(pick(&loader, "gemma3.vocab_size"), 262144);
    assert_eq!(pick(&loader, "gemma3.rope.dimension_count"), 256);
    assert_eq!(pick(&loader, "gemma3.pooling_type"), 1, "last-token pooling");
    let rope_base: f64 = pick_f32(&loader, "gemma3.rope.freq_base");
    assert!((rope_base - 1_000_000.0).abs() < 1.0, "rope.freq_base={rope_base}");
    let eps: f64 = pick_f32(&loader, "gemma3.attention.layer_norm_rms_epsilon");
    assert!((eps - 1e-6).abs() < 1e-9, "norm_eps={eps}");
}

#[test]
fn bitnet_embedding_270m_uses_spm_tokenizer_no_bpe_merges() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_BITNET_EMBEDDING_270M_MODEL not set");
        return;
    };
    let model = loader
        .metadata("tokenizer.ggml.model")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(model, "llama", "BitNet 270M declares the llama SPM model type");
    let pre = loader
        .metadata("tokenizer.ggml.pre")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(pre, "default", "BitNet 270M uses the default pre-tokenizer");
    assert!(
        loader.metadata("tokenizer.ggml.merges").is_none(),
        "BitNet 270M ships NO merge rules — it's pure SPM, not BPE"
    );
    assert!(
        loader.metadata("tokenizer.ggml.scores").is_some(),
        "SPM requires token scores"
    );
}

#[test]
fn bitnet_embedding_270m_tokenizer_special_ids() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_BITNET_EMBEDDING_270M_MODEL not set");
        return;
    };
    assert_eq!(pick(&loader, "tokenizer.ggml.bos_token_id"), 2);
    assert_eq!(pick(&loader, "tokenizer.ggml.eos_token_id"), 1);
    assert_eq!(pick(&loader, "tokenizer.ggml.padding_token_id"), 0);
    assert_eq!(pick(&loader, "tokenizer.ggml.unknown_token_id"), 3);
    let add_bos = loader
        .metadata("tokenizer.ggml.add_bos_token")
        .and_then(|v| match v {
            rust_model_inference::MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
        .unwrap_or(false);
    let add_eos = loader
        .metadata("tokenizer.ggml.add_eos_token")
        .and_then(|v| match v {
            rust_model_inference::MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
        .unwrap_or(false);
    let add_space_prefix = loader
        .metadata("tokenizer.ggml.add_space_prefix")
        .and_then(|v| match v {
            rust_model_inference::MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
        .unwrap_or(false);
    assert!(add_bos, "BitNet 270M requires BOS to be prepended");
    assert!(add_eos, "BitNet 270M declares add_eos_token=true");
    assert!(!add_space_prefix, "BitNet 270M does NOT add a space prefix");
}

#[test]
fn bitnet_embedding_270m_tensor_inventory_matches_gemma3_graph() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_BITNET_EMBEDDING_270M_MODEL not set");
        return;
    };
    let mut n_total = 0usize;
    let mut n_f16 = 0usize;
    let mut n_i2s = 0usize;
    let mut n_other = 0usize;
    let mut per_layer_has_norm_in = 0usize;
    for t in loader.tensors() {
        n_total += 1;
        match t.ggml_type {
            GGMLType::F16 => n_f16 += 1,
            GGMLType::I2_S => n_i2s += 1,
            _ => n_other += 1,
        }
        if t.name.contains("norm_in") && t.name.starts_with("blk.") {
            per_layer_has_norm_in += 1;
        }
    }
    assert_eq!(n_total, 362, "BitNet 270M total tensor count");
    assert_eq!(n_f16, 236, "F16 count: 310 norms + 2 per-model");
    assert_eq!(n_i2s, 126, "I2_S count: 18 layers * 7 BitLinear projections");
    assert_eq!(n_other, 0, "no other ggml_types expected");
    assert_eq!(
        per_layer_has_norm_in, 126,
        "per-projection *_norm_in weights (18 layers * 7 projections)"
    );
}

#[test]
fn bitnet_embedding_270m_per_block_tensor_count_is_twenty() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_BITNET_EMBEDDING_270M_MODEL not set");
        return;
    };
    // Each of the 18 layers carries exactly 20 tensors:
    //   4 layer norms (attn_norm, post_attention_norm, ffn_norm, post_ffw_norm)
    // + 2 QK norms (attn_q_norm, attn_k_norm)
    // + 7 BitLinear pre-RMSNorm (`*_norm_in`)
    // + 7 BitLinear weights (`*.weight` I2_S)
    // = 20.
    let mut per_layer_count: std::collections::HashMap<usize, usize> = Default::default();
    for t in loader.tensors() {
        let prefix = "blk.";
        if let Some(rest) = t.name.strip_prefix(prefix) {
            if let Some(dot) = rest.find('.') {
                if let Ok(layer) = rest[..dot].parse::<usize>() {
                    *per_layer_count.entry(layer).or_insert(0) += 1;
                }
            }
        }
    }
    for layer in 0..18 {
        let count = per_layer_count.get(&layer).copied().unwrap_or(0);
        assert_eq!(
            count, 20,
            "layer {layer} has {count} tensors; expected 20 \
             (4 layer norms + 2 QK norms + 7 *_norm_in + 7 I2_S weights)"
        );
    }
    // Per-model tensors (token_embd + output_norm = 2) must NOT
    // match the blk.* prefix and thus are not counted above.
    let n_model = loader
        .tensors()
        .iter()
        .filter(|t| !t.name.starts_with("blk."))
        .count();
    assert_eq!(n_model, 2, "per-model tensors (token_embd + output_norm)");
}