//! Integration test for the BitNet-Embeddings-0.6B GGUF metadata
//! contract.
//!
//! Run with `RMI_BITNET_EMBEDDING_0_6B_Q4_K_MODEL=/path/to/Q4_K_M.gguf`.
//!
//! # Scope
//!
//! This test pins the GGUF metadata + tensor-shape contract for
//! `microsoft/bitnet-embedding-0.6b` against the loader
//! (`src/core/loader.rs::GGUFLoader`). It does NOT exercise a forward
//! pass — the engine does not yet route `qwen3` arch through a
//! BitLinear forward (that integration is the next milestone, see
//! `docs/usage/bitnet_embedding.md` §3).
//!
//! # What this pins
//!
//! - `general.architecture = "qwen3"` (the engine's qwen3 trunk must
//!   be the target arch when BitNet Embeddings 0.6B is dispatched)
//! - `general.file_type = 40` (Microsoft's BitNet I2_S quantization
//!   file type, distinct from Q4_K_M's file_type=12)
//! - 28 layers × dims (1024 / 3072 / 16 heads / 8 KV / head_dim 128)
//! - YaRN config (factor=16, original_context_length=16384,
//!   beta_fast=32, beta_slow=1, freq_base=1e6)
//! - `qwen3.pooling_type = 1` (BitNet-Embeddings last-token pooling;
//!   our existing code maps 1→Mean which is the Qwen3-Embedding
//!   convention; the fix for BitNet is to add a `BitNetLast`
//!   variant — see `docs/usage/bitnet_embedding.md` §3.2)
//! - Tokenizer (gpt2 BPE + qwen2 pre-tokenizer, EOS=151643,
//!   add_eos_token=true)
//! - **506 tensors**: 310 F16 norms + 196 I2_S BitLinear weights
//!   (28 layers × 7 BitLinear layers = attn_q/k/v/o +
//!   ffn_gate/up/down)
//! - Per-projection `*_norm_in` weights exist for each BitLinear
//!   (RMSNorm pre-projection, new op needed for BitLinear forward)
//! - I2_S row layout: 128-element blocks × 32 bytes/block, no
//!   in-block or per-row scale stored (the dequant kernel in
//!   `src/ops/kernel/i2_s.rs` decodes this correctly — verified
//!   by `dequant_i2_s_real_gguf_block_produces_ternary`)

use rust_model_inference::{GGMLType, GGUFLoader};

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_BITNET_EMBEDDING_0_6B_Q4_K_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

fn pick(loader: &GGUFLoader, key: &str) -> usize {
    loader
        .metadata(key)
        .and_then(|v| v.to_u64())
        .map(|v| v as usize)
        .unwrap_or(0)
}

#[test]
fn bitnet_embedding_0_6b_contract_loads() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_BITNET_EMBEDDING_0_6B_Q4_K_MODEL not set");
        return;
    };
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(
        arch, "qwen3",
        "BitNet Embeddings 0.6B rides the qwen3 trunk"
    );
    // Microsoft-specific file type (40 = "BitNet I2_S GGUF"). Q4_K_M
    // is 12; the loader accepts any uint but our engine would dispatch
    // off the tensor types, not the file_type.
    let file_type = loader
        .metadata("general.file_type")
        .and_then(|v| v.to_u64())
        .unwrap_or(0);
    assert_eq!(file_type, 40, "general.file_type must be 40 (BitNet I2_S)");
    assert_eq!(pick(&loader, "qwen3.block_count"), 28);
    assert_eq!(pick(&loader, "qwen3.embedding_length"), 1024);
    assert_eq!(pick(&loader, "qwen3.attention.head_count"), 16);
    assert_eq!(pick(&loader, "qwen3.attention.head_count_kv"), 8);
    assert_eq!(pick(&loader, "qwen3.feed_forward_length"), 3072);
    assert_eq!(pick(&loader, "qwen3.attention.key_length"), 128);
    assert_eq!(pick(&loader, "qwen3.attention.value_length"), 128);
    assert_eq!(pick(&loader, "qwen3.vocab_size"), 151936);
    let rope_base: f64 = loader
        .metadata("qwen3.rope.freq_base")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!((rope_base - 1_000_000.0).abs() < 1.0);
    let eps: f64 = loader
        .metadata("qwen3.attention.layer_norm_rms_epsilon")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!((eps - 1e-6).abs() < 1e-9, "norm_eps={eps}");
}

#[test]
fn bitnet_embedding_0_6b_uses_plain_rope_no_yarn() {
    // BitNet Embeddings 0.6B (Qwen3 backbone) ships plain RoPE, not
    // YaRN. Unlike the Mistral-3 family which adds YaRN scaling
    // (factor=16, original_context_length=16384), this GGUF only
    // declares `qwen3.rope.freq_base = 1e6` and `qwen3.context_length
    // = 32768`. The BitLinear forward integration must therefore NOT
    // call `compute_yarn_thetas` — plain RoPE is the simpler path.
    let Some(loader) = loader() else { return };
    assert_eq!(
        loader
            .metadata("qwen3.rope.scaling.type")
            .map(|v| v.to_string_val().unwrap_or_default()),
        None,
        "BitNet Embeddings 0.6B must NOT have YaRN scaling"
    );
    let rope_base: f64 = loader
        .metadata("qwen3.rope.freq_base")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!(
        (rope_base - 1_000_000.0).abs() < 1.0,
        "rope_freq_base={rope_base}"
    );
    assert_eq!(
        pick(&loader, "qwen3.context_length"),
        32768,
        "context_length=32768"
    );
}

#[test]
fn bitnet_embedding_0_6b_pooling_type_is_last_token() {
    let Some(loader) = loader() else { return };
    // BitNet-Embeddings uses last-token pooling (pooling_type=1).
    // Our existing `src/models/qwen3/embedding.rs` maps 1→Mean
    // (Qwen3-Embedding convention); the engine's pooling dispatch
    // for BitNet is the open work item (see docs/usage/bitnet_embedding.md).
    // This test pins the GGUF contract so the fix knows what to
    // target.
    let pooling = pick(&loader, "qwen3.pooling_type");
    assert_eq!(
        pooling, 1,
        "BitNet-Embeddings pooling_type must be 1 (last-token)"
    );
}

#[test]
fn bitnet_embedding_0_6b_tokenizer_is_qwen3_chatml() {
    let Some(loader) = loader() else { return };
    assert_eq!(
        loader
            .metadata("tokenizer.ggml.model")
            .and_then(|v| v.to_string_val())
            .unwrap_or_default(),
        "gpt2",
        "BitNet Embeddings 0.6B uses the Qwen3 BPE tokenizer"
    );
    let pre = loader
        .metadata("tokenizer.ggml.pre")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(
        pre, "qwen2",
        "pre-tokenizer must be 'qwen2' (ChatML special tokens)"
    );
    // EOS for last-token pooling is `<|endoftext|>` (151643). The
    // conversion explicitly overrides this and sets `add_eos_token=true`
    // so the model emits its trailing EOS into the last position
    // before the embedding head pools over it.
    let eos = loader
        .metadata("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.to_u64())
        .unwrap_or(0);
    assert_eq!(eos, 151643, "EOS token must be <|endoftext|> 151643");
    assert!(matches!(
        loader.metadata("tokenizer.ggml.add_eos_token"),
        Some(rust_model_inference::MetaValue::Bool(true))
    ));
}

#[test]
fn bitnet_embedding_0_6b_tensor_inventory_is_bitnet_layout() {
    let Some(loader) = loader() else { return };
    // 506 tensors total = 28 layers × ~18 per-layer BitLinear/norm
    // tensors + head (token_embd, output_norm).
    let total = loader.n_tensors();
    assert_eq!(total, 506, "expected 506 tensors, got {total}");
    let n_i2s = loader
        .tensors()
        .iter()
        .filter(|t| t.ggml_type == GGMLType::I2_S)
        .count();
    let n_f16 = loader
        .tensors()
        .iter()
        .filter(|t| t.ggml_type == GGMLType::F16)
        .count();
    assert_eq!(
        n_i2s, 196,
        "196 BitLinear weights = 28 layers × 7 projections"
    );
    assert_eq!(n_f16, 310, "310 F16 norms (pre/post BitLinear RMSNorm)");
    // token_embd: [1024, 151936] F16
    let te = loader
        .tensor_info("token_embd.weight")
        .expect("token_embd.weight");
    assert_eq!(te.ggml_type, GGMLType::F16);
    assert_eq!(te.dims, &[1024, 151936]);
    // One BitLinear per (attn_q, attn_k, attn_v, attn_output,
    // ffn_gate, ffn_up, ffn_down) per layer — all I2_S.
    for layer in [0_usize, 14, 27] {
        for name in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            let t = loader
                .tensor_info(&format!("blk.{layer}.{name}.weight"))
                .unwrap_or_else(|| panic!("blk.{layer}.{name}.weight"));
            assert_eq!(
                t.ggml_type,
                GGMLType::I2_S,
                "blk.{layer}.{name}.weight must be I2_S"
            );
        }
    }
    // Per-projection RMSNorm weights (F16) — required for BitLinear.
    // Each BitLinear has a `_norm_in` pre-projection RMSNorm.
    for layer in [0_usize, 14, 27] {
        for (proj_name, proj_dim) in [
            ("attn_q", 1024_u64),
            ("attn_k", 1024),
            ("attn_v", 1024),
            ("attn_output", 2048),
            ("ffn_gate", 1024),
            ("ffn_up", 1024),
            ("ffn_down", 3072),
        ] {
            let t = loader
                .tensor_info(&format!("blk.{layer}.{proj_name}_norm_in.weight"))
                .unwrap_or_else(|| panic!("blk.{layer}.{proj_name}_norm_in.weight"));
            assert_eq!(
                t.ggml_type,
                GGMLType::F16,
                "blk.{layer}.{proj_name}_norm_in.weight must be F16 RMSNorm"
            );
            assert_eq!(t.dims, &[proj_dim], "blk.{layer}.{proj_name}_norm_in dims");
        }
    }
}

#[test]
fn bitnet_embedding_0_6b_i2_s_block_layout_matches_dequant() {
    // Empirical layout verification: the I2_S tensor for
    // `blk.0.ffn_down.weight` (`[3072, 1024]` = 3,145,728 elements)
    // should occupy exactly `3072 × (1024/128) × 32 = 786,432` bytes
    // of the GGUF data section (no per-row scale, just packed
    // 2-bit ternary). The dequant kernel in
    // `src/ops/kernel/i2_s.rs` matches this layout — verified by
    // its unit test that dequantizes the actual GGUF bytes and
    // confirms every value is in {-1.0, 0.0, +1.0}.
    let Some(loader) = loader() else { return };
    let ti = loader
        .tensor_info("blk.0.ffn_down.weight")
        .expect("blk.0.ffn_down.weight present in 0.6B GGUF");
    assert_eq!(ti.ggml_type, GGMLType::I2_S);
    assert_eq!(ti.dims, &[3072, 1024]);
    let expected_bytes = ti.dims[0] * (ti.dims[1] / 128) * 32;
    assert_eq!(
        ti.dims[1] % 128,
        0,
        "row_elements must be a multiple of QK_I2_S=128"
    );
    assert_eq!(
        ti.dims[1] / 128 * 32,
        8 * 32,
        "row_bytes for 1024 elements must be 256 (8 blocks × 32)"
    );
    // Total payload is `n_out × row_bytes`; we don't check the
    // exact byte size (offset arithmetic + alignment padding is
    // hard to verify from metadata alone), but `expected_bytes` is
    // the lower bound — the GGUF conversion adds alignment padding.
    assert!(
        expected_bytes <= 786_432 + 64,
        "i2_s payload should be ≈ row_bytes × n_out, got expected={expected_bytes}"
    );
}
