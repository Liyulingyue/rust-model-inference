//! Pin the AuK-Base 1.5B DiT architecture contract against an actual GGUF
//! file loaded from disk. Mirrors the pattern in
//! `tests/ernie_image_di_t_q4_k_m.rs`:
//!
//! Set `RMI_AUK_GGUF` to a published audio-cpp Base/Flash GGUF. Tests skip
//! only when it is unset; a supplied missing or malformed file fails.

use std::path::PathBuf;
use std::sync::Arc;

use rust_model_inference::core::tensor::TensorSource;

fn env_path() -> Option<PathBuf> {
    let raw = std::env::var("RMI_AUK_GGUF").ok()?;
    if raw.is_empty() {
        return None;
    }
    let path = PathBuf::from(raw);
    assert!(
        path.is_file(),
        "RMI_AUK_GGUF is not a file: {}",
        path.display()
    );
    Some(path)
}

fn loader() -> Option<Arc<dyn TensorSource>> {
    let path = env_path()?;
    Some(Arc::from(
        rust_model_inference::format::ggufrs::open_model_source(
            &path,
            rust_model_inference::format::ggufrs::ComponentRole::Llm,
        )
        .expect("RMI_AUK_GGUF must be a valid GGUF"),
    ))
}

fn pick(source: &dyn TensorSource, key: &str) -> Option<String> {
    source
        .metadata(key)
        .and_then(|value| value.to_string_val())
        .map(|s| s.to_string())
}

#[test]
fn contract_pins_auk_dit_architecture_metadata() {
    let Some(source) = loader() else {
        eprintln!("skipping: set RMI_AUK_GGUF to a real AuK GGUF");
        return;
    };
    assert_eq!(
        pick(source.as_ref(), "general.architecture").as_deref(),
        Some("audiocpp")
    );
    assert_eq!(
        pick(source.as_ref(), "general.name").as_deref(),
        Some("AuK-GGUF")
    );
}

#[test]
fn contract_pins_auk_dit_top_level_tensors() {
    let Some(source) = loader() else {
        eprintln!("skipping: set RMI_AUK_GGUF to a real AuK GGUF");
        return;
    };
    // Top-level tensor inventory
    for name in [
        "transformer.audio_embed.linear.weight",
        "transformer.audio_embed.linear.bias",
        "transformer.time_embed.time_mlp.0.weight",
        "transformer.time_embed.time_mlp.0.bias",
        "transformer.time_embed.time_mlp.2.weight",
        "transformer.time_embed.time_mlp.2.bias",
        "transformer.txt_proj.weight",
        "transformer.txt_proj.bias",
        "transformer.txt_norm.weight",
        "transformer.norm_out.linear.weight",
        "transformer.norm_out.linear.bias",
        "transformer.proj_out.weight",
        "transformer.proj_out.bias",
        "transformer.rotary_embed.inv_freq",
    ] {
        assert!(
            source.tensor_info(name).is_some(),
            "Missing top-level tensor: {name}"
        );
    }
}

#[test]
fn contract_pins_auk_dit_double_block_dimensions() {
    let Some(source) = loader() else {
        eprintln!("skipping: set RMI_AUK_GGUF to a real AuK GGUF");
        return;
    };
    // Per-double-block (10 layers) tensor inventory with expected dims.
    for layer in 0..10 {
        let p = format!("transformer.transformer_blocks.{layer}");
        let expected = [
            // AdaLN: hidden -> 6*hidden
            (format!("{p}.attn_norm_x.linear.weight"), vec![1536, 9216]),
            (format!("{p}.attn_norm_x.linear.bias"), vec![9216]),
            (format!("{p}.attn_norm_c.linear.weight"), vec![1536, 9216]),
            (format!("{p}.attn_norm_c.linear.bias"), vec![9216]),
            // Fused QKV: hidden -> 3*hidden
            (format!("{p}.attn.to_qkv.weight"), vec![1536, 4608]),
            (format!("{p}.attn.to_qkv.bias"), vec![4608]),
            (format!("{p}.attn.to_qkv_c.weight"), vec![1536, 4608]),
            (format!("{p}.attn.to_qkv_c.bias"), vec![4608]),
            // Output projection (c-stream only)
            (format!("{p}.attn.to_out_c.weight"), vec![1536, 1536]),
            // Q/K RMS norms (head_dim=64)
            (format!("{p}.attn.q_norm.weight"), vec![64]),
            (format!("{p}.attn.k_norm.weight"), vec![64]),
            // FF gate+up packed (6144=2*3072), down (3072->1536)
            (format!("{p}.ff_x.linear_in.weight"), vec![1536, 6144]),
            (format!("{p}.ff_x.linear_out.weight"), vec![3072, 1536]),
            (format!("{p}.ff_c.linear_in.weight"), vec![1536, 6144]),
            (format!("{p}.ff_c.linear_out.weight"), vec![3072, 1536]),
        ];
        for (name, dims) in expected {
            let info = source
                .tensor_info(&name)
                .unwrap_or_else(|| panic!("Missing tensor: {name}"));
            assert_eq!(&info.dims, &dims, "Invalid dims for {name}");
        }
    }
}

#[test]
fn contract_pins_auk_dit_single_block_dimensions() {
    let Some(source) = loader() else {
        eprintln!("skipping: set RMI_AUK_GGUF to a real AuK GGUF");
        return;
    };
    for layer in 0..20 {
        let p = format!("transformer.single_transformer_blocks.{layer}");
        let expected = [
            (format!("{p}.attn_norm.linear.weight"), vec![1536, 9216]),
            (format!("{p}.attn_norm.linear.bias"), vec![9216]),
            (format!("{p}.attn.to_qkv.weight"), vec![1536, 4608]),
            (format!("{p}.attn.to_qkv.bias"), vec![4608]),
            // Note: special `.0` suffix on to_out for single block
            (format!("{p}.attn.to_out.0.weight"), vec![1536, 1536]),
            (format!("{p}.attn.to_out.0.bias"), vec![1536]),
            (format!("{p}.attn.q_norm.weight"), vec![64]),
            (format!("{p}.attn.k_norm.weight"), vec![64]),
            (format!("{p}.ff.linear_in.weight"), vec![1536, 6144]),
            (format!("{p}.ff.linear_out.weight"), vec![3072, 1536]),
        ];
        for (name, dims) in expected {
            let info = source
                .tensor_info(&name)
                .unwrap_or_else(|| panic!("Missing tensor: {name}"));
            assert_eq!(&info.dims, &dims, "Invalid dims for {name}");
        }
    }
}

#[test]
fn contract_pins_auk_dit_block_count() {
    let Some(source) = loader() else {
        eprintln!("skipping: set RMI_AUK_GGUF to a real AuK GGUF");
        return;
    };
    // Published audio-cpp Base/Flash GGUFs contain 10 double + 20 single blocks.
    for layer in 0..=9 {
        let name = format!("transformer.transformer_blocks.{layer}.attn_norm_x.linear.weight");
        assert!(
            source.tensor_info(&name).is_some(),
            "Missing double-block layer {layer}"
        );
    }
    for layer in 0..20 {
        let name = format!("transformer.single_transformer_blocks.{layer}.attn_norm.linear.weight");
        assert!(
            source.tensor_info(&name).is_some(),
            "Missing single-block layer {layer}"
        );
    }
    // The next block must be absent for both streams.
    assert!(source
        .tensor_info("transformer.transformer_blocks.10.attn_norm_x.linear.weight")
        .is_none());
    assert!(source
        .tensor_info("transformer.single_transformer_blocks.20.attn_norm.linear.weight")
        .is_none());
}

#[test]
fn contract_pins_auk_dit_text_encoder_paired_via_txt_proj() {
    let Some(source) = loader() else {
        eprintln!("skipping: set RMI_AUK_GGUF to a real AuK GGUF");
        return;
    };
    // txt_proj must take Qwen2.5-Omni-3B's hidden (2048) -> AuK hidden (1536).
    let info = source
        .tensor_info("transformer.txt_proj.weight")
        .expect("txt_proj.weight present");
    assert_eq!(
        info.dims[0], 2048,
        "txt_proj source rows must match Qwen2.5-Omni n_embd"
    );
    assert_eq!(
        info.dims[1], 1536,
        "txt_proj target cols must match AuK hidden"
    );
}
