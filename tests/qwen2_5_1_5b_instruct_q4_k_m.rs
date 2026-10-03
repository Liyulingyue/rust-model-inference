//! Integration tests for the Qwen2.5-1.5B-Instruct Q4_K_M GGUF.
//!
//! Run with `RMI_QWEN2_5_1_5B_INSTRUCT_Q4_K_MODEL=/path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf`.
//!
//! Qwen2.5 rides the existing `qwen2` architecture path through
//! `models::qwen3::trunk`, so this test pins the loader + tokenizer +
//! tensor contract rather than claiming any new architecture support. The
//! model is the same base that the `harshatheg/Qwen-2.5-1B-RLCD` inference
//! engine (`Parallel Constrained Decoding`) drives on Apple Silicon via
//! MLX; the 1.5B parameters, 28 layers, 12 Q heads / 2 KV heads (GQA-6),
//! SwiGLU FFN, RoPE θ=1e6, RMSNorm ε=1e-6, and ChatML chat template
//! match the metadata that we lock in below.
//!
//! SHA-256 not pinned — the test depends only on metadata + tensor shapes,
//! which are stable across the official re-quantizations.

use rust_model_inference::GGUFLoader;
use rust_model_inference::TensorInfo;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_QWEN2_5_1_5B_INSTRUCT_Q4_K_MODEL")?;
    Some(GGUFLoader::from_file(path).unwrap())
}

fn shape<'a>(loader: &'a GGUFLoader, name: &str) -> Option<&'a TensorInfo> {
    loader.tensor_info(name)
}

fn pick(loader: &GGUFLoader, key: &str) -> usize {
    loader
        .metadata(key)
        .and_then(|v| v.to_u64())
        .map(|v| v as usize)
        .unwrap_or(0)
}

#[test]
fn q4_k_m_contract_loads() {
    let Some(loader) = loader() else { return };
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(arch, "qwen2");
    assert_eq!(pick(&loader, "qwen2.block_count"), 28);
    assert_eq!(pick(&loader, "qwen2.embedding_length"), 1536);
    assert_eq!(pick(&loader, "qwen2.attention.head_count"), 12);
    assert_eq!(pick(&loader, "qwen2.attention.head_count_kv"), 2);
    assert_eq!(pick(&loader, "qwen2.feed_forward_length"), 8960);
    assert_eq!(pick(&loader, "qwen2.context_length"), 32768);
    // head_dim = n_embd / n_head = 128 (matches Qwen2/Qwen3 convention)
    let head_dim =
        pick(&loader, "qwen2.embedding_length") / pick(&loader, "qwen2.attention.head_count");
    assert_eq!(head_dim, 128);
    // RoPE θ = 1e6 for Qwen2.5 (vs 1e5 for Qwen2, 1e6 for some Qwen3)
    let rope_base: f64 = loader
        .metadata("qwen2.rope.freq_base")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!(
        (rope_base - 1_000_000.0).abs() < 1.0,
        "rope_freq_base={rope_base}"
    );
    // RMSNorm ε = 1e-6 (Qwen2.5-specific; older Qwen2 was 1e-5)
    let eps: f64 = loader
        .metadata("qwen2.attention.layer_norm_rms_epsilon")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!((eps - 1e-6).abs() < 1e-9, "norm_eps={eps}");
}

#[test]
fn q4_k_m_tokenizer_matches_qwen2_5_chatml() {
    let Some(loader) = loader() else { return };
    use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
    let tok = BPETokenizer::from_gguf_metadata(|k| loader.metadata(k).cloned())
        .expect("Qwen2.5 tokenizer must build");
    // Qwen2.5-Instruct uses ChatML: <|im_start|>{role}\n…<|im_end|>\n.
    // The BPE pretokenizer is `qwen2`; metadata must match.
    assert_eq!(
        loader
            .metadata("tokenizer.ggml.pre")
            .and_then(|v| v.to_string_val())
            .unwrap_or_default(),
        "qwen2"
    );
    // Special-token IDs. Qwen2.5's `<|endoftext|>` is id 151643, `<|im_end|>` is id
    // 151645. The BOS is `<|endoftext|>` (151643), the EOS used by the model is
    // `<|im_end|>` (151645). The padding token is also `<|endoftext|>`.
    assert_eq!(
        loader
            .metadata("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u64()),
        Some(151643)
    );
    assert_eq!(
        loader
            .metadata("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u64()),
        Some(151645)
    );
    assert_eq!(
        loader
            .metadata("tokenizer.ggml.padding_token_id")
            .and_then(|v| v.to_u64()),
        Some(151643)
    );
    // `add_bos_token = false` because the chat template owns BOS insertion.
    // Pattern-match the `Bool` variant directly; `to_u64()` does not handle
    // it (intentional — GGUF Bools are not silently coerced to integers).
    let add_bos = matches!(
        loader
            .metadata("tokenizer.ggml.add_bos_token"),
        Some(rust_model_inference::MetaValue::Bool(false))
    );
    assert!(add_bos, "add_bos_token must be Bool(false)");
    // The vocabulary itself: Qwen2.5 uses 151,643 base tokens + a handful of
    // added special tokens. We don't pin the exact count (it varies between
    // Qwen2.5 re-tokenizations) but it must be ≥ 151,643.
    let vocab = loader
        .metadata("tokenizer.ggml.tokens")
        .and_then(|v| v.to_arr())
        .map(|a| a.len())
        .unwrap_or(0);
    assert!(
        vocab >= 151_643,
        "Qwen2.5 vocab size must be ≥ 151,643, got {vocab}"
    );
    // Roundtrip encode/decode a simple English sentence. "The capital of
    // France is" must produce a non-empty token sequence; decoded text must
    // round-trip the content characters (modulo the leading-space marker).
    let ids = tok.encode(
        "The capital of France is",
        EncodeOptions {
            add_special: false,
            parse_special: false,
        },
    );
    assert!(ids.len() >= 4, "got ids = {:?}", ids);
    let decoded = tok.decode(&ids, false);
    assert!(decoded.contains("France"), "decoded text was {decoded:?}");
    // ChatML marker must tokenize to its pinned special-token IDs:
    // `<|im_start|>` = 151644, `<|im_end|>` = 151645.
    let im_start_ids = tok.encode(
        "<|im_start|>",
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    assert_eq!(
        im_start_ids,
        vec![151_644],
        "<|im_start|> must tokenize to a single id 151644"
    );
    let im_end_ids = tok.encode(
        "<|im_end|>",
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    assert_eq!(
        im_end_ids,
        vec![151_645],
        "<|im_end|> must tokenize to a single id 151645"
    );
}

#[test]
fn q4_k_m_tensor_inventory_matches_qwen2_5_graph() {
    let Some(loader) = loader() else { return };
    // Qwen2.5-1.5B-Instruct graph (Q4_K_M re-quantization):
    // - 28 layers × (4 attn weights + 3 attn biases + 3 ffn weights + 2 norms) = 336
    // - 3 head tensors: token_embd, output (NOT tied), output_norm
    // - Total = 339
    //
    // Note: Qwen2.5-1.5B uses untied embeddings (separate `output.weight`),
    // unlike Qwen2-1.5B-Instruct which historically tied output to
    // token_embd. The README pin in this test catches that distinction.
    let n_layer = 28_usize;
    let n_embd = 1536_usize;
    let n_ff = 8960_usize;
    let n_head_kv = 2_usize;
    let head_k = 128_usize;
    let head_v = 128_usize;

    // Layer 0 spot-checks: 4 attn weights + 3 attn biases + 3 ffn weights + 2 norms.
    let t0_q = shape(&loader, "blk.0.attn_q.weight").expect("blk.0.attn_q.weight");
    assert_eq!(t0_q.dims, &[n_embd as u64, n_embd as u64]);
    let t0_k = shape(&loader, "blk.0.attn_k.weight").expect("blk.0.attn_k.weight");
    assert_eq!(
        t0_k.dims,
        &[n_embd as u64, (n_head_kv * head_k) as u64]
    );
    let t0_v = shape(&loader, "blk.0.attn_v.weight").expect("blk.0.attn_v.weight");
    assert_eq!(
        t0_v.dims,
        &[n_embd as u64, (n_head_kv * head_v) as u64]
    );
    let t0_ao = shape(&loader, "blk.0.attn_output.weight").expect("blk.0.attn_output.weight");
    assert_eq!(t0_ao.dims, &[n_embd as u64, n_embd as u64]);
    // Attn biases — present because Qwen2 uses QKV-bias attention.
    let t0_qb = shape(&loader, "blk.0.attn_q.bias").expect("blk.0.attn_q.bias");
    assert_eq!(t0_qb.dims, &[n_embd as u64]);
    let t0_kb = shape(&loader, "blk.0.attn_k.bias").expect("blk.0.attn_k.bias");
    assert_eq!(t0_kb.dims, &[(n_head_kv * head_k) as u64]);
    let t0_vb = shape(&loader, "blk.0.attn_v.bias").expect("blk.0.attn_v.bias");
    assert_eq!(t0_vb.dims, &[(n_head_kv * head_v) as u64]);
    let t0_fg = shape(&loader, "blk.0.ffn_gate.weight").expect("blk.0.ffn_gate.weight");
    assert_eq!(t0_fg.dims, &[n_embd as u64, n_ff as u64]);
    let t0_fu = shape(&loader, "blk.0.ffn_up.weight").expect("blk.0.ffn_up.weight");
    assert_eq!(t0_fu.dims, &[n_embd as u64, n_ff as u64]);
    let t0_fd = shape(&loader, "blk.0.ffn_down.weight").expect("blk.0.ffn_down.weight");
    assert_eq!(t0_fd.dims, &[n_ff as u64, n_embd as u64]);
    let t0_an = shape(&loader, "blk.0.attn_norm.weight").expect("blk.0.attn_norm.weight");
    assert_eq!(t0_an.dims, &[n_embd as u64]);
    let t0_fn = shape(&loader, "blk.0.ffn_norm.weight").expect("blk.0.ffn_norm.weight");
    assert_eq!(t0_fn.dims, &[n_embd as u64]);

    // Head tensors. Qwen2.5-1.5B-Instruct has UNTIED output projection.
    let te = shape(&loader, "token_embd.weight").expect("token_embd.weight");
    assert_eq!(te.dims, &[n_embd as u64, 151_936]);
    let ow = shape(&loader, "output.weight").expect("output.weight");
    assert_eq!(
        ow.dims,
        &[n_embd as u64, 151_936],
        "Qwen2.5-1.5B-Instruct output is not tied to token_embd"
    );
    let on = shape(&loader, "output_norm.weight").expect("output_norm.weight");
    assert_eq!(on.dims, &[n_embd as u64]);

    // 12 tensors per layer × 28 + 3 head = 336 + 3 = 339.
    let total = loader.n_tensors();
    assert_eq!(
        total,
        n_layer * 12 + 3,
        "unexpected tensor count {total} (expected {})",
        n_layer * 12 + 3
    );
}
