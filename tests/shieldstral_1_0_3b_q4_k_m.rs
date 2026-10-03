//! Integration tests for the Mistral-Shieldstral-1.0-3B Q4_K_M GGUF.
//!
//! Run with `RMI_SHIELDSTRAL_1_0_3B_Q4_K_MODEL=/path/to/Q4_K_M.gguf`.
//!
//! Shieldstral is Mistral's content-moderation / safety classifier,
//! post-trained from `Ministral-3-3B-Base-2512`. At the engine layer it
//! is structurally identical to the 3B Instruct / Reasoning siblings:
//! same 26-layer graph, same YaRN config, same Tekken (BPE-via-`LlamaBpe`)
//! tokenizer with the same vocab and special-token IDs, same GQA-4
//! (32 Q / 8 KV) head_dim=128, same RMSNorm ε=1e-5, tied output
//! (`output.weight` not shipped; `token_embd.weight` is reused as the
//! output projection per `tie_word_embeddings=true`).
//!
//! Three deltas from Instruct / Reasoning matter for the engine:
//!
//! 1. `general.name = "Shieldstral 1.0 3B"` — the product name
//!    "Shieldstral" only shares a suffix with "Mistral" / "Ministral",
//!    so the original `lower.contains("mistral") ||
//!    lower.contains("ministral")` heuristic in
//!    `src/models/llama/trunk/forward.rs::is_mistral` would miss it.
//!    Detection now also matches `contains("shieldstral")` so the
//!    `[INST] … [/INST]` prompt template fires.
//!
//! 2. `tokenizer.ggml.pre = "pixtral"` (not `"tekken"`) — the GGUF
//!    shipper (Metabaron6) wrote `"pixtral"` because the upstream HF
//!    config declares `model_type = "mistral3"` whose `text_config`
//!    is `model_type = "ministral3"` and the BPE pre-tokenizer family
//!    is the Mistral-3 / Pixtral shared one. Both `"pixtral"` and
//!    `"tekken"` already route through `PreTokenizer::LlamaBpe` in
//!    `src/core/tokenizer/mod.rs`, so no tokenizer code change was
//!    needed for this model.
//!
//! 3. NO `tokenizer.ggml.add_bos_token` key in the GGUF — llama.cpp
//!    defaults this to `false` when the key is absent. The tokenized
//!    prompt therefore starts directly with the `[INST]` special
//!    token (id 3) rather than `<s>` (id 1) as a separate prefix,
//!    which is a 1-token difference from Instruct / Reasoning. The
//!    engine handles both shapes uniformly because `[INST]` and
//!    `[/INST]` are recognized as single Tekken special tokens via
//!    `parse_special=true` regardless of whether `<s>` precedes them.
//!
//! ## Use case
//!
//! Shieldstral is intended for content moderation (CSAM / WAF / DLP /
//! PII / SEC / policy-adjacent classifications per Metabaron6's tags),
//! not general chat. The natural CLI surface is `--prompt '[INST]
//! Classify this as safe (yes) or unsafe (no): … [/INST]'` — smoke-
//! confirmed: threat content yields "yes", benign content yields "no".
//! The JEV path runs the model through `run_jev_decision_llama` but
//! exhibits the same Mistral-3-3B JSON-payload positional bias
//! documented in `docs/develop/SUPPORTED_MODELS.md` (A wins ~56%
//! regardless of content); for binary safety classification prefer
//! `--prompt` with explicit yes/no instructions, or write a custom
//! classifier that pre-rolls the model's free-text output.

use rust_model_inference::GGUFLoader;
use rust_model_inference::TensorInfo;

fn loader() -> Option<GGUFLoader> {
    let path = std::env::var_os("RMI_SHIELDSTRAL_1_0_3B_Q4_K_MODEL")?;
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

    // Distinguishing metadata: `general.name = "Shieldstral 1.0 3B"`
    // (with a space, not "Mistral-Shieldstral-*" / "Ministral-3-*-2512").
    // The engine's `is_mistral` heuristic in
    // `src/models/llama/trunk/forward.rs` matches the substring
    // "shieldstral" so the `[INST] … [/INST]` template fires.
    let name = loader
        .metadata("general.name")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert!(
        name.to_ascii_lowercase().contains("shieldstral"),
        "Shieldstral GGUF must ship general.name containing 'Shieldstral'; got {name:?}"
    );
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
    // Distinct from Instruct / Reasoning: Metabaron6 wrote `pre =
    // "pixtral"` rather than `"tekken"`. Both map to
    // `PreTokenizer::LlamaBpe` in the tokenizer dispatch — see
    // `src/core/tokenizer/mod.rs`. Pinning this here guards against a
    // future re-quantization that flips back to `"tekken"` and breaks
    // the special-token routing for some downstream tooling.
    let pre = loader
        .metadata("tokenizer.ggml.pre")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert_eq!(pre, "pixtral");

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

    // Distinct from Instruct / Reasoning: NO `tokenizer.ggml.add_bos_token`
    // key in the GGUF. llama.cpp defaults absent → `false`, so the
    // tokenized prompt starts with `[INST]` (id 3) rather than `<s>`
    // (id 1) + `[INST]`. The contract is encoded as the key being
    // absent (vs `Some(MetaValue::Bool(true))` for Instruct/Reasoning);
    // the matcher below accepts both to keep the test stable against
    // a future re-quantization that adds the key.
    let add_bos = loader.metadata("tokenizer.ggml.add_bos_token");
    assert!(
        add_bos.is_none() || matches!(add_bos, Some(rust_model_inference::MetaValue::Bool(false))),
        "Shieldstral GGUF should not auto-prepend <s>; got add_bos_token={add_bos:?}"
    );

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
    // Tied output (same as 3B Instruct / Reasoning, unlike 3B 8B which
    // is untied). `tie_word_embeddings=true` in the upstream
    // `text_config` confirms the embedding is reused as the output.
    assert!(
        loader.tensor_info("output.weight").is_none(),
        "Mistral-Shieldstral-1.0-3B ties output to token_embd"
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
fn q4_k_m_chat_template_uses_inst_markers() {
    // Distinguishing feature vs Ministral-3-Reasoning: Shieldstral's
    // chat template uses `[INST] … [/INST]` for user turns but does NOT
    // inject a `[THINK]` block or a default system prompt that asks the
    // model to draft thinking. It's a binary classifier prompt, not a
    // reasoning prompt. Pinning the shape guards against a Mistral
    // release that adds `[THINK]` markers (Reasoning-style) to the
    // safety classifier by mistake.
    let Some(loader) = loader() else { return };
    let tmpl = loader
        .metadata("tokenizer.chat_template")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    assert!(
        !tmpl.is_empty(),
        "Shieldstral GGUF must ship a tokenizer.chat_template"
    );
    assert!(
        tmpl.contains("[INST]") && tmpl.contains("[/INST]"),
        "Shieldstral chat template must declare [INST]/[/INST] markers"
    );
    assert!(
        !tmpl.contains("[THINK]") && !tmpl.contains("[/THINK]"),
        "Shieldstral is a classifier, not a reasoning model; chat template must not declare [THINK] blocks"
    );
    // Supports multimodal image input (Mistral's safety classifier
    // accepts image content). The CLI's text-only path won't exercise
    // this but pinning it here guards against a future text-only
    // re-quantization that drops the `[IMG]` marker.
    assert!(
        tmpl.contains("[IMG]"),
        "Shieldstral chat template must declare an [IMG] multimodal marker"
    );
}
