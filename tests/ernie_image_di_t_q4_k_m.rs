//! Pin the ERNIE-Image DiT architecture contract against an actual GGUF file
//! loaded from disk. Mirrors the pattern in
//! `tests/ministral3_3b_instruct_q4_k_m.rs`:
//!
//! - The `loader` helper returns `None` when `RMI_ERNIE_IMAGE_DIT_GGUF` is
//!   unset or the file is missing, so the test silently no-ops on CI.
//! - The `loaded` helper returns `None` if the GGUF couldn't be parsed
//!   (e.g. wrong file). Subsequent tests then bail out without panicking.

use std::path::PathBuf;
use std::sync::Arc;

use rust_model_inference::core::tensor::TensorSource;

fn env_path() -> Option<PathBuf> {
    let raw = std::env::var("RMI_ERNIE_IMAGE_DIT_GGUF").ok()?;
    if raw.is_empty() {
        return None;
    }
    let path = PathBuf::from(raw);
    if !path.is_file() {
        return None;
    }
    Some(path)
}

fn loader() -> Option<Arc<dyn TensorSource>> {
    let path = env_path()?;
    Some(Arc::from(
        rust_model_inference::format::ggufrs::open_model_source(
            &path,
            rust_model_inference::format::ggufrs::ComponentRole::Llm,
        )
        .ok()?,
    ))
}

fn pick(source: &dyn TensorSource, key: &str) -> Option<String> {
    source
        .metadata(key)
        .and_then(|value| value.to_string_val())
        .map(|s| s.to_string())
}

#[test]
fn contract_pins_ernie_image_di_t_via_tensor_signature() {
    let Some(source) = loader() else {
        eprintln!("skipping: set RMI_ERNIE_IMAGE_DIT_GGUF to a real ERNIE-Image GGUF");
        return;
    };
    // Detection: stable-diffusion.cpp probes via
    //   `model.diffusion_model.layers.0.adaLN_sa_ln.weight`
    // but the unsloth export drops the `model.diffusion_model.` prefix and
    // sets `general.architecture = "wan"` (a metadata mis-tag), so the
    // pragmatic check is `layers.0.self_attention.to_q.weight` presence plus
    // `text_proj.weight` -- this combination is unique to ERNIE-Image among
    // diffusion GGUFs in the repo.
    assert!(
        source
            .tensor_info("layers.0.self_attention.to_q.weight")
            .is_some(),
        "ERNIE-Image signature: layers.0.self_attention.to_q.weight missing"
    );
    assert!(
        source.tensor_info("text_proj.weight").is_some(),
        "ERNIE-Image signature: text_proj.weight missing"
    );
}

#[test]
fn contract_pins_ernie_image_di_t_tensor_inventory() {
    let Some(source) = loader() else {
        eprintln!("skipping: set RMI_ERNIE_IMAGE_DIT_GGUF to a real ERNIE-Image GGUF");
        return;
    };
    // 36 main blocks, each contributing 7 matrices (to_q, to_k, to_v, to_out.0,
    // mlp.gate_proj, mlp.up_proj, mlp.linear_fc2) = 252 per-block matrices.
    for layer in 0..36 {
        let prefix = format!("layers.{layer}");
        for suffix in [
            "self_attention.to_q.weight",
            "self_attention.to_k.weight",
            "self_attention.to_v.weight",
            "self_attention.to_out.0.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.linear_fc2.weight",
        ] {
            assert!(
                source.tensor_info(&format!("{prefix}.{suffix}")).is_some(),
                "Missing tensor: {prefix}.{suffix}"
            );
        }
        for suffix in [
            "adaLN_sa_ln.weight",
            "adaLN_mlp_ln.weight",
            "self_attention.norm_q.weight",
            "self_attention.norm_k.weight",
        ] {
            assert!(
                source.tensor_info(&format!("{prefix}.{suffix}")).is_some(),
                "Missing norm: {prefix}.{suffix}"
            );
        }
    }
    // Top-level tensors. The unsloth export drops `final_norm.norm.weight`
    // (the AdaLNContinuous inner norm), so we skip it.
    for name in [
        "x_embedder.proj.weight",
        "x_embedder.proj.bias",
        "adaLN_modulation.1.weight",
        "adaLN_modulation.1.bias",
        "time_embedding.linear_1.weight",
        "time_embedding.linear_1.bias",
        "time_embedding.linear_2.weight",
        "time_embedding.linear_2.bias",
        "final_norm.linear.weight",
        "final_norm.linear.bias",
        "final_linear.weight",
        "final_linear.bias",
        "text_proj.weight",
    ] {
        assert!(
            source.tensor_info(name).is_some(),
            "Missing top-level tensor: {name}"
        );
    }
}

#[test]
fn contract_pins_ernie_image_di_t_block_dimensions() {
    let Some(source) = loader() else {
        eprintln!("skipping: set RMI_ERNIE_IMAGE_DIT_GGUF to a real ERNIE-Image GGUF");
        return;
    };
    // Per-block attention QKV projection: hidden (4096) -> inner (4096).
    // Each tensor is [n_in, n_out] = [4096, 4096].
    let expected_q = source
        .tensor_info("layers.0.self_attention.to_q.weight")
        .expect("layers.0.self_attention.to_q.weight present");
    assert_eq!(expected_q.dims, &[4096, 4096]);
    let expected_k = source
        .tensor_info("layers.0.self_attention.to_k.weight")
        .expect("layers.0.self_attention.to_k.weight present");
    assert_eq!(expected_k.dims, &[4096, 4096]);
    let expected_v = source
        .tensor_info("layers.0.self_attention.to_v.weight")
        .expect("layers.0.self_attention.to_v.weight present");
    assert_eq!(expected_v.dims, &[4096, 4096]);
    let expected_o = source
        .tensor_info("layers.0.self_attention.to_out.0.weight")
        .expect("layers.0.self_attention.to_out.0.weight present");
    assert_eq!(expected_o.dims, &[4096, 4096]);
    // FFN: hidden -> 12288 (SwiGLU gate, up; 12288 -> hidden for fc2).
    let expected_gate = source
        .tensor_info("layers.0.mlp.gate_proj.weight")
        .expect("layers.0.mlp.gate_proj.weight present");
    assert_eq!(expected_gate.dims, &[4096, 12288]);
    let expected_up = source
        .tensor_info("layers.0.mlp.up_proj.weight")
        .expect("layers.0.mlp.up_proj.weight present");
    assert_eq!(expected_up.dims, &[4096, 12288]);
    let expected_fc2 = source
        .tensor_info("layers.0.mlp.linear_fc2.weight")
        .expect("layers.0.mlp.linear_fc2.weight present");
    assert_eq!(expected_fc2.dims, &[12288, 4096]);
    // text_proj: 3072 (Ministral-3 n_embd) -> 4096 (DiT hidden).
    let expected_text_proj = source
        .tensor_info("text_proj.weight")
        .expect("text_proj.weight present");
    assert_eq!(expected_text_proj.dims, &[3072, 4096]);
}

#[test]
fn contract_pins_ernie_image_di_t_modulation_and_time_embedding_shapes() {
    let Some(source) = loader() else {
        eprintln!("skipping: set RMI_ERNIE_IMAGE_DIT_GGUF to a real ERNIE-Image GGUF");
        return;
    };
    // adaLN_modulation.1: hidden -> 6*hidden (shared AdaLN, chunks into
    // shift_msa / scale_msa / gate_msa / shift_mlp / scale_mlp / gate_mlp).
    let adaln = source
        .tensor_info("adaLN_modulation.1.weight")
        .expect("adaLN_modulation.1.weight present");
    assert_eq!(adaln.dims, &[4096, 6 * 4096]);
    // time_embedding: two-linears, both hidden -> hidden.
    let te1 = source
        .tensor_info("time_embedding.linear_1.weight")
        .expect("time_embedding.linear_1.weight present");
    assert_eq!(te1.dims, &[4096, 4096]);
    let te2 = source
        .tensor_info("time_embedding.linear_2.weight")
        .expect("time_embedding.linear_2.weight present");
    assert_eq!(te2.dims, &[4096, 4096]);
    // final_norm: linear hidden -> 2*hidden (AdaLNContinuous, scale+shift).
    let fnl = source
        .tensor_info("final_norm.linear.weight")
        .expect("final_norm.linear.weight present");
    assert_eq!(fnl.dims, &[4096, 2 * 4096]);
    // final_linear: hidden -> 128 (out_channels).
    let flin = source
        .tensor_info("final_linear.weight")
        .expect("final_linear.weight present");
    assert_eq!(flin.dims, &[4096, 128]);
}

#[test]
fn contract_pins_ernie_image_di_t_text_encoder_paired_via_text_proj() {
    let Some(source) = loader() else {
        eprintln!("skipping: set RMI_ERNIE_IMAGE_DIT_GGUF to a real ERNIE-Image GGUF");
        return;
    };
    // The DiT expects text encoder hidden_size == 3072 (Ministral-3 n_embd).
    // We can't assert that from the DiT GGUF alone, but we can pin that
    // text_proj.weight is [3072, 4096] (= 3072 source rows -> 4096 target
    // cols), which transitively pins the contract.
    let tp = source
        .tensor_info("text_proj.weight")
        .expect("text_proj.weight present");
    assert_eq!(
        tp.dims[0], 3072,
        "text_proj source rows must match Ministral-3 n_embd"
    );
    assert_eq!(
        tp.dims[1], 4096,
        "text_proj target cols must match DiT hidden"
    );
}
