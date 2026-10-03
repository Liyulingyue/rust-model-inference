//! Integration tests for the Ministral-3-3B-Reasoning-2512 Q4_K_M GGUF.
//!
//! Run with `RMI_MINISTRAL_3_3B_REASONING_Q4_K_MODEL=/path/to/Q4_K_M.gguf`.
//!
//! Ministral-3-Reasoning-2512 is structurally identical to
//! Ministral-3-3B-Instruct-2512 at the engine layer (same 26-layer graph,
//! same YaRN config, same Tekken tokenizer, same head_dim/RMSNorm/GQA). The
//! only difference is post-training (reasoning vs chat) and the
//! `tokenizer.chat_template`: the Reasoning template wraps the default
//! system prompt in `[SYSTEM_PROMPT] … [/SYSTEM_PROMPT]` and supports
//! assistant `[THINK] … [/THINK]` blocks. The engine doesn't read the
//! chat_template (the CLI emits its own `[INST] … [/INST]`), so all four
//! tests here are contract + chat-template-distinguishing assertions.
//!
//! ## Known limitation (see `src/app/text/generation.rs` TODO)
//!
//! The CLI emits `[INST] {prompt} [/INST]` and never injects the
//! `[SYSTEM_PROMPT] … [/SYSTEM_PROMPT]` block from the chat template,
//! which is what would prompt the model to emit `[THINK] … [/THINK]`
//! in its output. Smoke tests produce free-form reasoning prose without
//! the explicit `[THINK]` markers; reasoning flow is preserved but the
//! canonical Mistral-3-Reasoning output shape is not. Unblocking this
//! requires a `--system-prompt` CLI flag (or chat-template-aware mode),
//! tracked in `src/app/text/generation.rs`.

use rust_model_inference::GGUFLoader;
use rust_model_inference::TensorInfo;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_MINISTRAL_3_3B_REASONING_Q4_K_MODEL")?;
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
fn q4_k_m_contract_matches_instruct_dims() {
    let Some(loader) = loader() else { return };
    let arch = loader
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(arch, "mistral3");
    // Same dims as Instruct (the two share a base model and differ only in
    // post-training). If Mistral ships a different-shape Reasoning 3B in
    // the future, this test should be the one to update.
    assert_eq!(pick(&loader, "mistral3.block_count"), 26);
    assert_eq!(pick(&loader, "mistral3.embedding_length"), 3072);
    assert_eq!(pick(&loader, "mistral3.attention.head_count"), 32);
    assert_eq!(pick(&loader, "mistral3.attention.head_count_kv"), 8);
    assert_eq!(pick(&loader, "mistral3.feed_forward_length"), 9216);
    assert_eq!(pick(&loader, "mistral3.context_length"), 262144);
    assert_eq!(pick(&loader, "mistral3.attention.key_length"), 128);
    assert_eq!(pick(&loader, "mistral3.attention.value_length"), 128);
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

    assert!(loader.tensor_info("blk.0.attn_q.bias").is_none());
    assert!(loader.tensor_info("blk.0.attn_k.bias").is_none());
    assert!(loader.tensor_info("blk.0.attn_v.bias").is_none());

    let te = shape(&loader, "token_embd.weight").expect("token_embd.weight");
    assert_eq!(te.dims, &[n_embd as u64, vocab_size as u64]);
    assert!(
        loader.tensor_info("output.weight").is_none(),
        "Ministral-3-3B-Reasoning ties output to token_embd"
    );
    let on = shape(&loader, "output_norm.weight").expect("output_norm.weight");
    assert_eq!(on.dims, &[n_embd as u64]);

    assert!(
        loader.tensor_info("rope_freqs.weight").is_none(),
        "Mistral 3 uses analytical YaRN; rope_freqs.weight must not exist"
    );

    let total = loader.n_tensors();
    assert_eq!(
        total,
        n_layer * 9 + 2,
        "unexpected tensor count {total} (expected {})",
        n_layer * 9 + 2
    );
}

#[test]
fn q4_k_m_chat_template_advertises_think_blocks() {
    // Distinguishes Reasoning from Instruct: the Reasoning chat template
    // emits `[THINK] … [/THINK]` markers and a default system prompt that
    // asks the model to draft its thinking first. The engine doesn't parse
    // the template (CLI emits its own `[INST] … [/INST]`), but pinning the
    // shape here guards against a future Mistral release silently dropping
    // thinking support.
    let Some(loader) = loader() else { return };
    let tmpl = loader
        .metadata("tokenizer.chat_template")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert!(
        !tmpl.is_empty(),
        "Reasoning GGUF must ship a tokenizer.chat_template"
    );
    assert!(
        tmpl.contains("[THINK]") && tmpl.contains("[/THINK]"),
        "Reasoning chat template must declare [THINK]/[/THINK] markers"
    );
    assert!(
        tmpl.contains("[SYSTEM_PROMPT]") && tmpl.contains("[/SYSTEM_PROMPT]"),
        "Reasoning chat template must declare [SYSTEM_PROMPT]/[/SYSTEM_PROMPT] markers"
    );
    // The default system prompt asks the model to draft its thinking first.
    assert!(
        tmpl.contains("HOW YOU SHOULD THINK AND ANSWER")
            || tmpl.contains("draft your thinking process"),
        "Reasoning chat template must inject a default system prompt that asks the model to draft its thinking"
    );
    // Reasoning also supports the user-image `[IMG]` marker; this is the
    // multimodal-flag carrier. Asserting this guards against a future Mistral
    // release silently dropping image support from the Reasoning build.
    assert!(
        tmpl.contains("[IMG]"),
        "Reasoning chat template must declare an [IMG] marker"
    );
}
