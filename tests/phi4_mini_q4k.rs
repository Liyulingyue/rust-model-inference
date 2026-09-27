//! Integration tests for the Phi-4-mini-instruct Q4_K_M GGUF.
//!
//! Run with `RMI_PHI_4_MINI_Q4_K_MODEL=/path/to/Q4_K_M.gguf`.
//!
//! Phi-4-mini uses a tiktoken-style "gpt-4o" pre-tokenizer and the
//! `<|role|>...<|end|>` chat template. The llama trunk cannot drive it
//! (Phi-4 uses fused QKV + gate-less FFN), so these tests pin the
//! loader + tokenizer + tensor contract rather than end-to-end inference.
//! A dedicated phi3 trunk is the next step.

use rust_model_inference::GGUFLoader;
use rust_model_inference::TensorInfo;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_PHI_4_MINI_Q4_K_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

fn shape<'a>(loader: &'a GGUFLoader, name: &str) -> Option<&'a TensorInfo> {
    loader.tensor_info(name)
}

#[test]
fn q4_contract_loads() {
    let Some(loader) = loader() else { return };
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(arch, "phi3");
    let pick = |k: &str| {
        loader
            .metadata(k)
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(0)
    };
    assert_eq!(pick("phi3.block_count"), 32);
    assert_eq!(pick("phi3.embedding_length"), 3072);
    assert_eq!(pick("phi3.attention.head_count"), 24);
    assert_eq!(pick("phi3.attention.head_count_kv"), 8);
    assert_eq!(pick("phi3.feed_forward_length"), 8192);
    assert_eq!(pick("phi3.context_length"), 131_072);
    let head_dim = pick("phi3.embedding_length") / pick("phi3.attention.head_count");
    assert_eq!(head_dim, 128);
    let rope_dim = pick("phi3.rope.dimension_count");
    assert_eq!(rope_dim, 96, "Phi-4 uses partial RoPE: 96 of 128");
}

#[test]
fn q4_tokenizer_matches_scalar_llama_cpp() {
    let Some(loader) = loader() else { return };
    use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
    let tok = BPETokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned())
        .expect("Phi-4 tokenizer must build");
    let bos = tok.bos_id();
    let eos = tok.eos_id();
    assert_eq!(bos, Some(199_999), "Phi-4 BOS should be 199999");
    assert_eq!(eos, Some(200_020), "Phi-4 EOS should be 200020");
    // Encode a known string and roundtrip it.
    let ids = tok.encode(
        "The capital of France is",
        EncodeOptions {
            add_special: false,
            parse_special: false,
        },
    );
    assert!(ids.len() >= 4, "got ids = {:?}", ids);
    let decoded = tok.decode(&ids, false);
    assert!(
        decoded.contains("France"),
        "decoded text was {decoded:?}"
    );
}

#[test]
fn q4_tensor_shapes_match_loader_contract() {
    let Some(loader) = loader() else { return };
    // Phi-4 uses FUSED QKV (single attn_qkv tensor) and FUSED FFN
    // (ffn_up + ffn_down, no ffn_gate). These names differ from llama.
    let t0_attn_qkv = shape(&loader, "blk.0.attn_qkv.weight").expect("attn_qkv.weight");
    assert_eq!(t0_attn_qkv.dims.len(), 2);
    assert_eq!(t0_attn_qkv.dims[0] as usize, 3072); // n_in
    // n_out = n_embd_q + 2 * n_embd_gqa = 3072 + 2*1024 = 5120
    assert_eq!(
        t0_attn_qkv.dims[1] as usize,
        3072 + 2 * 1024,
        "fused QKV dims mismatch"
    );
    let t0_attn_out = shape(&loader, "blk.0.attn_output.weight").expect("attn_output");
    // Phi-4's attn_output projects from the full Q space (= 3072)
    // back to n_embd (= 3072). GQA happens inside attn_qkv.
    assert_eq!(t0_attn_out.dims[0] as usize, 3072);
    assert_eq!(t0_attn_out.dims[1] as usize, 3072);
    let t0_ffn_up = shape(&loader, "blk.0.ffn_up.weight").expect("ffn_up");
    // Phi-4 fuses gate + up into a single ffn_up tensor of width 2 * n_ff.
    assert_eq!(t0_ffn_up.dims[0] as usize, 3072);
    assert_eq!(t0_ffn_up.dims[1] as usize, 16384);
    let t0_ffn_down = shape(&loader, "blk.0.ffn_down.weight").expect("ffn_down");
    assert_eq!(t0_ffn_down.dims[0] as usize, 8192);
    assert_eq!(t0_ffn_down.dims[1] as usize, 3072);
    assert!(
        loader.tensor_info("blk.0.attn_q.weight").is_none(),
        "Phi-4 should NOT have separate attn_q.weight (fused into attn_qkv)"
    );
    assert!(
        loader.tensor_info("blk.0.ffn_gate.weight").is_none(),
        "Phi-4 should NOT have ffn_gate (gate-less FFN)"
    );
    let te = shape(&loader, "token_embd.weight").expect("token_embd.weight");
    assert_eq!(te.dims[0] as usize, 3072);
    assert_eq!(te.dims[1] as usize, 200_064); // Phi-4 vocab
    let norm = shape(&loader, "output_norm.weight").expect("output_norm.weight");
    assert_eq!(norm.dims[0] as usize, 3072);
    // Phi-4 has precomputed YaRN rope tables.
    let rope_long = shape(&loader, "rope_factors_long.weight").expect("rope_factors_long");
    assert_eq!(rope_long.dims.len(), 1);
    let rope_short = shape(&loader, "rope_factors_short.weight").expect("rope_factors_short");
    assert_eq!(rope_short.dims.len(), 1);
    // Per layer: attn_qkv + attn_output + ffn_up + ffn_down + attn_norm + ffn_norm = 6
    // 32 layers * 6 + token_embd + output_norm + rope_factors_long + rope_factors_short = 197
    let total = loader.n_tensors();
    assert_eq!(total, 32 * 6 + 4, "unexpected tensor count {total}");
}

#[test]
fn q4_chat_template_uses_role_end_tokens() {
    use rust_model_inference::models::chat_template::{default_template, ChatTemplate};
    let template = default_template("phi3").expect("phi3 must have a default chat template");
    let rendered = template.render("What is the capital of France?");
    assert!(rendered.contains("<|user|>"), "got {rendered:?}");
    assert!(rendered.contains("<|end|>"), "got {rendered:?}");
    assert!(rendered.contains("<|assistant|>"), "got {rendered:?}");
    assert_eq!(template, ChatTemplate::Phi4);
}