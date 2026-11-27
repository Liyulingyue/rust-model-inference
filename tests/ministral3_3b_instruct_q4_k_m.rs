//! Integration tests for the Ministral-3-3B-Instruct-2512 Q4_K_M GGUF.
//!
//! Run with `RMI_MINISTRAL_3_3B_INSTRUCT_Q4_K_MODEL=/path/to/Q4_K_M.gguf`.
//!
//! Ministral-3 rides the existing `llama` trunk (GQA + SwiGLU + RMSNorm +
//! RoPE are identical shapes) with three deltas:
//!
//! - `general.architecture = "mistral3"` (separate from classic Mistral 1/2
//!   which declare `llama`). Registered in `uses_llama_trunk` and the
//!   loader's arch validation list.
//! - YaRN RoPE: `mistral3.rope.scaling.type = "yarn"` with
//!   `factor=16, original_context_length=16384, yarn_beta_fast=32,
//!   yarn_beta_slow=1, yarn_log_multiplier=1, rope_theta=1e6`.
//! - Tekken tokenizer: `tokenizer.ggml.pre = "tekken"` dispatches to the
//!   same `LlamaBpe` pretokenizer that handles llama / pixtral / dbrx.
//! - General name spelling: Mistral 3 ships as "Ministral-3B-Instruct-2512"
//!   (deliberate misspelling with an extra `n`), so the original
//!   `contains("mistral")` check missed it. The detection now matches both
//!   `"mistral"` and `"ministral"`.

use rust_model_inference::GGUFLoader;
use rust_model_inference::TensorInfo;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_MINISTRAL_3_3B_INSTRUCT_Q4_K_MODEL")?;
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
    assert_eq!(arch, "mistral3");
    assert_eq!(pick(&loader, "mistral3.block_count"), 26);
    assert_eq!(pick(&loader, "mistral3.embedding_length"), 3072);
    assert_eq!(pick(&loader, "mistral3.attention.head_count"), 32);
    assert_eq!(pick(&loader, "mistral3.attention.head_count_kv"), 8);
    assert_eq!(pick(&loader, "mistral3.feed_forward_length"), 9216);
    assert_eq!(pick(&loader, "mistral3.context_length"), 262144);
    // head_dim is pinned by `mistral3.attention.key_length` (= 128),
    // NOT derived from `n_embd / n_head = 3072 / 32 = 96`. The two values
    // intentionally disagree in Mistral 3 (head_dim is a property of the
    // attention op, not the embedding): the GGUF ships 3072 hidden with
    // 32 heads of width 128, so Q projects to 32*128 = 4096 and K/V
    // project to 8*128 = 1024. Reading head_dim from `key_length`
    // mirrors how the llama trunk reads `head_dim = key_length ?? value_length
    // ?? n_embd/n_head`.
    let key_length = pick(&loader, "mistral3.attention.key_length");
    assert_eq!(key_length, 128, "head_dim must be 128 for Mistral 3 3B");
    let value_length = pick(&loader, "mistral3.attention.value_length");
    assert_eq!(value_length, 128);
    let rope_dim = pick(&loader, "mistral3.rope.dimension_count");
    assert_eq!(rope_dim, 128);
    let rope_base: f64 = loader
        .metadata("mistral3.rope.freq_base")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!(
        (rope_base - 1_000_000.0).abs() < 1.0,
        "rope_freq_base={rope_base}"
    );
    let eps: f64 = loader
        .metadata("mistral3.attention.layer_norm_rms_epsilon")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!((eps - 1e-5).abs() < 1e-9, "norm_eps={eps}");
}

#[test]
fn q4_k_m_yarn_scaling_matches_pinned_metadata() {
    let Some(loader) = loader() else { return };
    let yarn_type = loader
        .metadata("mistral3.rope.scaling.type")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(
        yarn_type, "yarn",
        "YaRN scaling must be active for Mistral 3"
    );
    let factor: f64 = loader
        .metadata("mistral3.rope.scaling.factor")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!((factor - 16.0).abs() < 1e-6, "yarn_factor={factor}");
    let orig_ctx = pick(&loader, "mistral3.rope.scaling.original_context_length");
    assert_eq!(orig_ctx, 16384, "yarn_original_context_length={orig_ctx}");
    let beta_fast: f64 = loader
        .metadata("mistral3.rope.scaling.yarn_beta_fast")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!(
        (beta_fast - 32.0).abs() < 1e-6,
        "yarn_beta_fast={beta_fast}"
    );
    let beta_slow: f64 = loader
        .metadata("mistral3.rope.scaling.yarn_beta_slow")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!((beta_slow - 1.0).abs() < 1e-6, "yarn_beta_slow={beta_slow}");
    let log_mult: f64 = loader
        .metadata("mistral3.rope.scaling.yarn_log_multiplier")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    // Mistral 3 ships `yarn_log_multiplier=1.0` (not the 0.1 llama-3
    // default). The trunk must read whatever the GGUF pins.
    assert!(
        (log_mult - 1.0).abs() < 1e-6,
        "yarn_log_multiplier={log_mult}"
    );
}

#[test]
fn q4_k_m_tokenizer_is_tekken_with_special_ids() {
    let Some(loader) = loader() else { return };
    assert_eq!(
        loader
            .metadata("tokenizer.ggml.model")
            .and_then(|v| v.to_string_val())
            .unwrap_or_default(),
        "gpt2"
    );
    let pre = loader
        .metadata("tokenizer.ggml.pre")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(pre, "tekken");
    // Special tokens: BOS=1 (`<s>`), EOS=2 (`</s>`), pad=11 (`<unk>`).
    assert_eq!(
        loader
            .metadata("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u64()),
        Some(1)
    );
    assert_eq!(
        loader
            .metadata("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u64()),
        Some(2)
    );
    assert_eq!(
        loader
            .metadata("tokenizer.ggml.padding_token_id")
            .and_then(|v| v.to_u64()),
        Some(11)
    );
    // `add_bos_token = Bool(true)` so the `[INST] … [/INST]` template
    // gets a leading `<s>` via `add_special=true`. Pattern-match the
    // `Bool` variant directly; `to_u64()` does not handle Bool.
    assert!(matches!(
        loader.metadata("tokenizer.ggml.add_bos_token"),
        Some(rust_model_inference::MetaValue::Bool(true))
    ));
    // vocab_size must be exactly 131072 (Tekken vocab).
    let vocab = loader
        .metadata("mistral3.vocab_size")
        .and_then(|v| v.to_u64())
        .unwrap_or(0);
    assert_eq!(vocab, 131072);
    let gguf_vocab = loader
        .metadata("tokenizer.ggml.tokens")
        .and_then(|v| v.to_arr())
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(gguf_vocab, 131072);
}

#[test]
fn q4_k_m_tensor_inventory_matches_mistral3_graph() {
    let Some(loader) = loader() else { return };
    // Mistral-3 graph (Q4_K_M): 26 layers × 9 tensors + 2 head = 236.
    // NO QKV biases (unlike Qwen2.5 / Qwen3), NO `output.weight` (output
    // is tied to token_embd per `tie_word_embeddings=true` in config.json),
    // NO `rope_freqs.weight` (YaRN is computed analytically).
    let n_layer = 26_usize;
    let n_embd = 3072_usize;
    let n_ff = 9216_usize;
    let n_head = 32_usize;
    let n_head_kv = 8_usize;
    let head_dim = 128_usize;
    let vocab_size = 131072_usize;

    let t0_q = shape(&loader, "blk.0.attn_q.weight").expect("blk.0.attn_q.weight");
    assert_eq!(t0_q.dims, &[n_embd as u64, (n_head * head_dim) as u64]);
    let t0_k = shape(&loader, "blk.0.attn_k.weight").expect("blk.0.attn_k.weight");
    assert_eq!(t0_k.dims, &[n_embd as u64, (n_head_kv * head_dim) as u64]);
    let t0_v = shape(&loader, "blk.0.attn_v.weight").expect("blk.0.attn_v.weight");
    assert_eq!(t0_v.dims, &[n_embd as u64, (n_head_kv * head_dim) as u64]);
    let t0_ao = shape(&loader, "blk.0.attn_output.weight").expect("blk.0.attn_output.weight");
    assert_eq!(t0_ao.dims, &[(n_head * head_dim) as u64, n_embd as u64]);
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

    // Mistral 3 has no QKV biases — unlike Qwen2.5 / Qwen3.
    assert!(loader.tensor_info("blk.0.attn_q.bias").is_none());
    assert!(loader.tensor_info("blk.0.attn_k.bias").is_none());
    assert!(loader.tensor_info("blk.0.attn_v.bias").is_none());

    // Tied output: only `output_norm.weight` is separate; output is tied
    // to `token_embd.weight`.
    let te = shape(&loader, "token_embd.weight").expect("token_embd.weight");
    assert_eq!(te.dims, &[n_embd as u64, vocab_size as u64]);
    assert!(
        loader.tensor_info("output.weight").is_none(),
        "Mistral-3-3B-Instruct ties output to token_embd"
    );
    let on = shape(&loader, "output_norm.weight").expect("output_norm.weight");
    assert_eq!(on.dims, &[n_embd as u64]);

    // YaRN is analytical; no precomputed rope table.
    assert!(
        loader.tensor_info("rope_freqs.weight").is_none(),
        "Mistral 3 uses analytical YaRN; rope_freqs.weight must not exist"
    );

    // 9 per layer × 26 + 2 head = 236.
    let total = loader.n_tensors();
    assert_eq!(
        total,
        n_layer * 9 + 2,
        "unexpected tensor count {total} (expected {})",
        n_layer * 9 + 2
    );
}
