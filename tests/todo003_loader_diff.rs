//! Differential test for the TODO-003 loader unification.
//!
//! The old per-model loaders and the new shared
//! `core::tensor::load_weight_any` must produce the **same** `Weight` for
//! the same tensor. This test loads a real checkpoint and compares the two
//! implementations tensor by tensor: the declared format, the matrix shape,
//! and the bytes the kernel actually sees.
//!
//! It is a test, not a script: it runs in CI on any machine that has the
//! checkpoint, and skips with a clear message when it does not.

use rust_model_inference::core::tensor::{GGMLType, TensorSource};
use rust_model_inference::ops::kernel::{QuantizedTensor, Weight};

/// The OLD qwen35 loader, reproduced here so the two can be compared.
///
/// Before TODO-003 the qwen35 trunk read a matrix as
/// `n_cols = dims[0]`, `n_rows = dims[1]`, checked nothing about the byte
/// length, and built the weight straight from `QuantizedTensor`.
fn old_qwen35_load_weight<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
) -> Option<Weight<'a>> {
    let ti = source.tensor_info(name)?;
    let data = source.tensor_slice(name)?;
    let n_cols = ti.dims[0] as usize;
    let n_rows = if ti.dims.len() >= 2 {
        ti.dims[1] as usize
    } else {
        1
    };
    match ti.ggml_type {
        GGMLType::F32 => Weight::from_f32_bytes(data, n_cols, n_rows),
        GGMLType::F16
        | GGMLType::BF16
        | GGMLType::Q8_0
        | GGMLType::Q4_0
        | GGMLType::Q4_1
        | GGMLType::Q4K
        | GGMLType::Q5K
        | GGMLType::Q6K => {
            let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
                data,
                ti.ggml_type,
                n_cols,
                n_rows,
            ));
            weight.n_in = n_cols;
            weight.n_out = n_rows;
            Some(weight)
        }
        _ => None,
    }
}

/// The formats the qwen35 trunk loads as matrices.
const MATRIX_FORMATS: &[GGMLType] = &[
    GGMLType::F32,
    GGMLType::F16,
    GGMLType::BF16,
    GGMLType::Q8_0,
    GGMLType::Q4_0,
    GGMLType::Q4_1,
    GGMLType::Q4K,
    GGMLType::Q5K,
    GGMLType::Q6K,
];

/// The matrix tensors the qwen35 trunk reads.
fn qwen35_matrix_names() -> Vec<String> {
    let mut names = Vec::new();
    for prefix in ["token_embd.weight", "output.weight", "cls.output.weight"] {
        names.push(prefix.to_string());
    }
    names.push("shared_head_head.weight".into());
    for layer in 0..64 {
        for suffix in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_qkv.weight",
            "attn_gate.weight",
            "ssm_beta.weight",
            "ssm_alpha.weight",
            "ssm_out.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "ffn_gate_inp.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
            "ffn_gate_inp_shexp.weight",
        ] {
            names.push(format!("blk.{layer}.{suffix}"));
        }
    }
    names
}

fn checkpoint() -> Option<std::path::PathBuf> {
    for candidate in [
        "models/OvisOCR3-GGUF/OvisOCR3.Q8_0.gguf",
        "models/OvisOCR3-GGUF-converted/bf16/OvisOCR3-BF16.gguf",
        "models/OvisOCR3-GGUF-converted/q8_0/OvisOCR3-Q8_0.gguf",
    ] {
        let path = std::path::PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

/// The old and the new loader must agree on every matrix of a real
/// checkpoint: same format, same shape, same bytes.
#[test]
fn the_shared_loader_matches_the_old_qwen35_loader_on_real_weights() {
    let Some(path) = checkpoint() else {
        eprintln!("skipping: no OvisOCR3 checkpoint on this machine");
        return;
    };
    let source: std::sync::Arc<dyn TensorSource> = std::sync::Arc::from(
        rust_model_inference::format::ggufrs::open_model_source(
            &path,
            rust_model_inference::format::ggufrs::ComponentRole::Llm,
        )
        .unwrap(),
    );

    let mut compared = 0usize;
    let mut matmuls = 0usize;
    for name in qwen35_matrix_names() {
        let Some(info) = source.tensor_info(&name) else {
            continue;
        };
        if !MATRIX_FORMATS.contains(&info.ggml_type) {
            continue;
        }
        let old = old_qwen35_load_weight(source.as_ref(), &name)
            .unwrap_or_else(|| panic!("the old loader rejected {name}"));
        let n_in = info.dims.first().copied().unwrap_or(0) as usize;
        let n_out = info.dims[1..].iter().product::<u64>() as usize;
        let new = rust_model_inference::core::tensor::load_weight_any(
            source.as_ref(),
            &name,
            n_in,
            n_out,
            Some(MATRIX_FORMATS),
        )
        .unwrap_or_else(|error| panic!("the shared loader rejected {name}: {error}"));

        assert_eq!(old.ggml_type, new.ggml_type, "{name}: format");
        assert_eq!(old.n_in, new.n_in, "{name}: input width");
        assert_eq!(old.n_out, new.n_out, "{name}: output width");

        // The bytes the kernel holds must be identical. This is the check
        // that matters: the shape can agree while the payload does not.
        use rust_model_inference::ops::kernel::Kernel;
        let old_bytes = old.kernel.weight_bytes().unwrap_or_default();
        let new_bytes = new.kernel.weight_bytes().unwrap_or_default();
        assert_eq!(old_bytes, new_bytes, "{name}: weight bytes");

        // And the numbers the kernel produces must be identical too, which
        // is the property the model actually depends on.
        if old.n_in > 0 && old.n_out > 0 && old.n_in <= 4096 {
            let input: Vec<f32> = (0..old.n_in)
                .map(|index| ((index % 17) as f32 - 8.0) * 0.125)
                .collect();
            let old_output = old.matmul(&input);
            let new_output = new.matmul(&input);
            assert_eq!(old_output, new_output, "{name}: matmul output");
            matmuls += 1;
        }
        compared += 1;
    }
    assert!(compared > 0, "no matrix tensor was compared; the checkpoint changed");
    assert!(matmuls > 0, "no matmul was compared");
    eprintln!("compared {compared} matrix tensors, {matmuls} matmuls");
}

/// The new loader checks the byte length and the block alignment; the old
/// qwen35 loader checked neither. A tensor the old loader accepted must
/// still be accepted — otherwise the unification would reject checkpoints
/// that used to run.
#[test]
fn the_shared_loader_accepts_every_matrix_the_old_loader_accepted() {
    let Some(path) = checkpoint() else {
        eprintln!("skipping: no OvisOCR3 checkpoint on this machine");
        return;
    };
    let source: std::sync::Arc<dyn TensorSource> = std::sync::Arc::from(
        rust_model_inference::format::ggufrs::open_model_source(
            &path,
            rust_model_inference::format::ggufrs::ComponentRole::Llm,
        )
        .unwrap(),
    );

    let mut checked = 0usize;
    for name in qwen35_matrix_names() {
        let Some(info) = source.tensor_info(&name) else {
            continue;
        };
        if !MATRIX_FORMATS.contains(&info.ggml_type) {
            continue;
        }
        // The old loader accepted it.
        assert!(
            old_qwen35_load_weight(source.as_ref(), &name).is_some(),
            "{name}: the old loader rejected a tensor it should accept"
        );
        // The new loader must accept it too.
        let n_in = info.dims.first().copied().unwrap_or(0) as usize;
        let n_out = info.dims[1..].iter().product::<u64>() as usize;
        assert!(
            rust_model_inference::core::tensor::load_weight_any(
                source.as_ref(),
                &name,
                n_in,
                n_out,
                Some(MATRIX_FORMATS),
            )
            .is_ok(),
            "{name}: the shared loader rejected a tensor the old one accepted"
        );
        checked += 1;
    }
    assert!(checked > 0, "no tensor was checked");
    eprintln!("accepted {checked} tensors under both loaders");
}
