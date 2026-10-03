//! Integration tests for the Ministral-3-8B-Instruct-2512 Q4_K_M GGUF.
//!
//! Run with `RMI_MINISTRAL_3_8B_INSTRUCT_Q4_K_MODEL=/path/to/Q4_K_M.gguf`.
//!
//! Same `mistral3` arch + Tekken + YaRN as the 3B variant; only the dims
//! grow: 34 layers (vs 26), n_embd=4096 (vs 3072), n_ff=14336 (vs 9216),
//! tensor count 308 (vs 236). head_count/head_count_kv/head_dim/rope
//! dimensions stay identical, so the QKV project to the same
//! `n_head * head_dim = 4096` (Q) and `n_head_kv * head_dim = 1024` (KV).
//!
//! Memory profile: 4.96 GB GGUF on disk; with the engine's KV cache and
//! activation scratch the resident set runs ~5.6 GB on a 4-core / 7.5 GiB
//! box (swap will engage if `--max-context` is pushed past 4096). This is
//! why we default to Q4_K_M for the 8B variant — Q5_K_M (5.78 GB) and
//! Q8_0 (8.5+ GB) exceed the box's headroom.

use rust_model_inference::GGUFLoader;
use rust_model_inference::TensorInfo;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_MINISTRAL_3_8B_INSTRUCT_Q4_K_MODEL")?;
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
    // 8B dims (vs 3B's 26/3072/9216).
    assert_eq!(pick(&loader, "mistral3.block_count"), 34);
    assert_eq!(pick(&loader, "mistral3.embedding_length"), 4096);
    assert_eq!(pick(&loader, "mistral3.attention.head_count"), 32);
    assert_eq!(pick(&loader, "mistral3.attention.head_count_kv"), 8);
    assert_eq!(pick(&loader, "mistral3.feed_forward_length"), 14336);
    assert_eq!(pick(&loader, "mistral3.context_length"), 262144);
    assert_eq!(pick(&loader, "mistral3.attention.key_length"), 128);
    assert_eq!(pick(&loader, "mistral3.attention.value_length"), 128);
    assert_eq!(pick(&loader, "mistral3.rope.dimension_count"), 128);
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
    assert_eq!(yarn_type, "yarn", "YaRN scaling must be active for Mistral 3");
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
    assert!((beta_fast - 32.0).abs() < 1e-6, "yarn_beta_fast={beta_fast}");
    let beta_slow: f64 = loader
        .metadata("mistral3.rope.scaling.yarn_beta_slow")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!((beta_slow - 1.0).abs() < 1e-6, "yarn_beta_slow={beta_slow}");
    let log_mult: f64 = loader
        .metadata("mistral3.rope.scaling.yarn_log_multiplier")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.0);
    assert!((log_mult - 1.0).abs() < 1e-6, "yarn_log_multiplier={log_mult}");
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
    assert!(matches!(
        loader.metadata("tokenizer.ggml.add_bos_token"),
        Some(rust_model_inference::MetaValue::Bool(true))
    ));
    let vocab = loader
        .metadata("mistral3.vocab_size")
        .and_then(|v| v.to_u64())
        .unwrap_or(0);
    assert_eq!(vocab, 131072);
}

#[test]
fn q4_k_m_tensor_inventory_matches_mistral3_graph() {
    let Some(loader) = loader() else { return };
    let n_layer = 34_usize;
    let n_embd = 4096_usize;
    let n_ff = 14336_usize;
    let n_head = 32_usize;
    let n_head_kv = 8_usize;
    let head_dim = 128_usize;
    let vocab_size = 131072_usize;

    let t0_q = shape(&loader, "blk.0.attn_q.weight").expect("blk.0.attn_q.weight");
    assert_eq!(t0_q.dims, &[n_embd as u64, (n_head * head_dim) as u64]);
    let t0_k = shape(&loader, "blk.0.attn_k.weight").expect("blk.0.attn_k.weight");
    assert_eq!(
        t0_k.dims,
        &[n_embd as u64, (n_head_kv * head_dim) as u64]
    );
    let t0_v = shape(&loader, "blk.0.attn_v.weight").expect("blk.0.attn_v.weight");
    assert_eq!(
        t0_v.dims,
        &[n_embd as u64, (n_head_kv * head_dim) as u64]
    );
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

    assert!(loader.tensor_info("blk.0.attn_q.bias").is_none());
    assert!(loader.tensor_info("blk.0.attn_k.bias").is_none());
    assert!(loader.tensor_info("blk.0.attn_v.bias").is_none());

    let te = shape(&loader, "token_embd.weight").expect("token_embd.weight");
    assert_eq!(te.dims, &[n_embd as u64, vocab_size as u64]);
    // Ministral-3-8B-Instruct is UNtied (unlike 3B which shares output with
    // token_embd). Q6K quantization on the output head rather than Q4_K_M
    // is typical for Mistral's 8B family because the head dominates the
    // quality of the output distribution.
    let ow = shape(&loader, "output.weight").expect("output.weight (8B is untied)");
    assert_eq!(ow.dims, &[n_embd as u64, vocab_size as u64]);
    let on = shape(&loader, "output_norm.weight").expect("output_norm.weight");
    assert_eq!(on.dims, &[n_embd as u64]);

    assert!(
        loader.tensor_info("rope_freqs.weight").is_none(),
        "Mistral 3 uses analytical YaRN; rope_freqs.weight must not exist"
    );

    // 9 per layer × 34 + 3 head (output + output_norm + token_embd) = 309.
    let total = loader.n_tensors();
    assert_eq!(
        total,
        n_layer * 9 + 3,
        "unexpected tensor count {total} (expected {})",
        n_layer * 9 + 3
    );
}
