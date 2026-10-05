//! Integration test for the **standard** gemma-3-1b-it GGUF
//! metadata + tensor-shape contract (non-BitNet, Q4_K_M mixed).
//!
//! Run with
//! `RMI_GEMMA_3_1B_IT_MODEL=/path/to/gemma-3-1b-it-Q4_K_M.gguf`.
//!
//! # Scope
//!
//! Pins the GGUF layout for `unsloth/gemma-3-1b-it-GGUF` (Q4_K_M,
//! 769 MiB) so the standard gemma3 trunk can consume it:
//!
//! - `general.architecture = "gemma3"`
//! - `general.file_type = 15` (Q4_K_M)
//! - 26 layers × dims (1152 / 6912 / 4 heads / 1 KV / head_dim 256)
//! - vocab = 262144 (from `tokenizer.ggml.tokens` array length)
//! - ctx = 32768 (no RoPE scaling — same as 270M-it, vs 131072 + linear
//!   scaling factor=8.0 in 4B-it)
//! - `gemma3.attention.layer_norm_rms_epsilon = 1e-6`
//! - `gemma3.rope.freq_base = 1e6`
//! - `gemma3.attention.sliding_window = 512` (same as 270M-it; vs
//!   1024 in 4B-it; BitNet 270M has no sliding window)
//! - Standard SentencePiece BPE tokenizer (same as 270M-it / 4B-it:
//!   `model = "llama"`, `pre = "default"`, no `merges`, EOS=106)
//! - **Mixed quantization per-tensor** (different mix than 270M-it):
//!   157 F32 + 39 Q4K + 117 Q5_0 + 13 Q6K + 14 Q8_0 = 340 total
//! - Per-block tensor count: 13 (4 layer norms + 2 QK norms +
//!   7 quantized matmuls)
//! - No `*_norm_in` tensors, no I2_S tensors

use rust_model_inference::{GGMLType, GGUFLoader, MetaValue};

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_GEMMA_3_1B_IT_MODEL")?;
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
fn gemma3_1b_it_contract_loads() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_1B_IT_MODEL not set");
        return;
    };
    assert_eq!(pick_str(&loader, "general.architecture"), "gemma3");
    assert_eq!(pick(&loader, "general.file_type"), 15, "Q4_K_M file_type");
    assert_eq!(pick(&loader, "gemma3.block_count"), 26);
    assert_eq!(pick(&loader, "gemma3.embedding_length"), 1152);
    assert_eq!(pick(&loader, "gemma3.attention.head_count"), 4);
    assert_eq!(pick(&loader, "gemma3.attention.head_count_kv"), 1);
    assert_eq!(pick(&loader, "gemma3.feed_forward_length"), 6912);
    assert_eq!(pick(&loader, "gemma3.attention.key_length"), 256);
    assert_eq!(pick(&loader, "gemma3.attention.value_length"), 256);
    assert_eq!(vocab_size(&loader), 262144);
    assert_eq!(pick(&loader, "gemma3.context_length"), 32768);
    let rope_base: f64 = pick_f32(&loader, "gemma3.rope.freq_base");
    assert!((rope_base - 1_000_000.0).abs() < 1.0);
    let eps: f64 = pick_f32(&loader, "gemma3.attention.layer_norm_rms_epsilon");
    assert!((eps - 1e-6).abs() < 1e-9);
}

#[test]
fn gemma3_1b_it_no_rope_scaling() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_1B_IT_MODEL not set");
        return;
    };
    // 1B and 270M-it both omit gemma3.rope.scaling.* entirely; 4B
    // declares linear/factor=8.0. This distinguishes "base
    // context" variants from "extended context" ones.
    assert!(
        loader.metadata("gemma3.rope.scaling.type").is_none(),
        "1B has NO RoPE scaling (vs 4B's linear+8.0)"
    );
    assert!(
        loader.metadata("gemma3.rope.scaling.factor").is_none(),
        "1B has NO RoPE scaling factor"
    );
    let sw = pick(&loader, "gemma3.attention.sliding_window");
    assert_eq!(
        sw, 512,
        "1B uses sliding_window=512 (same as 270M-it, vs 1024 in 4B)"
    );
}

#[test]
fn gemma3_1b_it_tokenizer_is_spm() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_1B_IT_MODEL not set");
        return;
    };
    assert_eq!(pick_str(&loader, "tokenizer.ggml.model"), "llama");
    assert_eq!(pick_str(&loader, "tokenizer.ggml.pre"), "default");
    assert!(
        loader.metadata("tokenizer.ggml.merges").is_none(),
        "standard gemma3 is pure SPM, no merges"
    );
    assert_eq!(pick(&loader, "tokenizer.ggml.bos_token_id"), 2);
    assert_eq!(pick(&loader, "tokenizer.ggml.eos_token_id"), 106);
    assert_eq!(pick(&loader, "tokenizer.ggml.padding_token_id"), 0);
    assert_eq!(pick(&loader, "tokenizer.ggml.unknown_token_id"), 3);
}

#[test]
fn gemma3_1b_it_tensor_inventory_mixed_q4_q5_q6_q8() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_1B_IT_MODEL not set");
        return;
    };
    let mut n_total = 0usize;
    let mut n_f32 = 0usize;
    let mut n_q4k = 0usize;
    let mut n_q5_0 = 0usize;
    let mut n_q6k = 0usize;
    let mut n_q8_0 = 0usize;
    let mut n_i2_s = 0usize;
    let mut n_norm_in = 0usize;
    for t in loader.tensors() {
        n_total += 1;
        match t.ggml_type {
            GGMLType::F32 => n_f32 += 1,
            GGMLType::Q4K => n_q4k += 1,
            GGMLType::Q5_0 => n_q5_0 += 1,
            GGMLType::Q6K => n_q6k += 1,
            GGMLType::Q8_0 => n_q8_0 += 1,
            GGMLType::I2_S => n_i2_s += 1,
            _ => {}
        }
        if t.name.contains("norm_in") {
            n_norm_in += 1;
        }
    }
    assert_eq!(n_i2_s, 0, "no I2_S (BitNet-only marker)");
    assert_eq!(n_norm_in, 0, "no per-projection norm_in");
    // 26 layers × 13 tensors = 338
    // + token_embd + output_norm = 340
    assert_eq!(
        n_total, 340,
        "expected 340 tensors (157 F32 + 39 Q4K + 117 Q5_0 + 13 Q6K + 14 Q8_0)"
    );
    // 26 × 7 = 182 matmul weights, distributed across Q4K/Q5_0/Q6K/Q8_0
    assert_eq!(
        n_q4k + n_q5_0 + n_q6k + n_q8_0,
        183,
        "matmul weights (26*7 + token_embd)"
    );
}

#[test]
fn gemma3_1b_it_per_block_tensor_count_is_thirteen() {
    let Some(loader) = loader() else {
        eprintln!("skipping: RMI_GEMMA_3_1B_IT_MODEL not set");
        return;
    };
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
    assert_eq!(per_layer.len(), 26, "expected 26 layer blocks");
    for (idx, count) in per_layer.iter() {
        assert_eq!(*count, 13, "layer {idx} has {count} tensors, expected 13");
    }
}
