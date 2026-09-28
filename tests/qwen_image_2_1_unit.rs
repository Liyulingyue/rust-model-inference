//! Public self-tests for the Qwen-Image-2.1 numeric kernels and validation
//! contract. These mirror the invariants the pinned stable-diffusion.cpp
//! oracle relies on (tools/oracle/qwen_image_2_1).

use rust_model_inference::core::tensor::{GGMLType, TensorInfo, TensorSource};
use rust_model_inference::models::diffusion::qwen_image_2_1::dit::kernels::{
    apply_rope_row, expf_v, fp32_to_bf16, gelu, rope_frequencies, silu_inplace, softmax_row,
    timestep_embedding_row, vec_dot_f32_ggml,
};
use rust_model_inference::models::diffusion::qwen_image_2_1::{
    config_from_source, matches_signature, validate_dit,
};
use std::collections::HashMap;

const PREFIX: &str = "model.diffusion_model";

#[test]
fn bf16_rounding_matches_ggml_rne() {
    assert_eq!(fp32_to_bf16(1.0), 0x3f80);
    assert_eq!(fp32_to_bf16(-1.5), 0xbfc0);
    assert_eq!(fp32_to_bf16(1.0 + f32::EPSILON), 0x3f80);
}

#[test]
fn expf_v_tracks_libm_expf() {
    for value in [-3.5f32, -1.0, -0.25, 0.0, 0.5, 2.0] {
        let expected = value.exp();
        let actual = expf_v(value);
        let relative = (expected - actual).abs() / expected.abs().max(1.0);
        assert!(
            relative < 2.0e-7,
            "expf_v({value}) = {actual} vs {expected}"
        );
    }
    assert_eq!(expf_v(0.0), 1.0);
}

#[test]
fn silu_matches_reference_values() {
    let mut values = vec![0.0f32, 1.0, -1.0, 2.0];
    silu_inplace(&mut values);
    assert_eq!(values[0], 0.0);
    let expected_one = 1.0f32 / (1.0f32 + (-1.0f32).exp());
    assert!((values[1] - expected_one).abs() < 2.0e-7);
    assert!((values[2] + 1.0f32 / (1.0f32 + 1.0f32.exp())).abs() < 2.0e-7);
}

#[test]
fn gelu_is_zero_at_origin() {
    assert_eq!(gelu(0.0), 0.0);
}

#[test]
fn softmax_row_is_normalized_and_shift_invariant() {
    let mut scores = vec![1.0f32, 2.0, 3.0, 4.0];
    softmax_row(&mut scores);
    let sum: f32 = scores.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    let mut shifted = vec![11.0f32, 12.0, 13.0, 14.0];
    softmax_row(&mut shifted);
    assert_eq!(scores, shifted);
}

#[test]
fn vec_dot_f32_ggml_sums_exactly_for_small_values() {
    let x = vec![1.0f32; 128];
    let mut y = vec![1.0f32; 128];
    y[7] = 2.0;
    assert_eq!(vec_dot_f32_ggml(&x, 1, &y, 1, 128), 129.0);
}

#[test]
fn vec_dot_f32_ggml_supports_strided_rows() {
    let x: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..128).map(|i| (i % 3) as f32).collect();
    let strided = vec_dot_f32_ggml(&x, 2, &y, 1, 128);
    let dense_x: Vec<f32> = x.iter().step_by(2).copied().collect();
    let expected = vec_dot_f32_ggml(&dense_x, 1, &y, 1, 128);
    assert_eq!(strided, expected);
}

#[test]
fn timestep_embedding_is_cos_sin_concat() {
    let mut output = vec![0.0f32; 256];
    timestep_embedding_row(0.0, &mut output);
    assert_eq!(output[0], 1.0);
    assert_eq!(output[128], 0.0);
}

#[test]
fn rope_frequencies_first_and_last_entries() {
    let omega = rope_frequencies(16, 10_000.0);
    assert_eq!(omega.len(), 8);
    assert_eq!(omega[0], 1.0);
    assert!(
        (omega[7] - 1.0 / 10_000.0f32.powf(0.875)).abs() < 1e-9,
        "{}",
        omega[7]
    );
}

#[test]
fn rope_rotation_preserves_norm() {
    let mut values = vec![0.31f32, -0.77, 0.9, 0.12];
    let pe = [0.8f32, -0.6, 0.6, 0.8, 1.0, 0.0, -0.0, 1.0];
    let before: f32 = values.iter().map(|v| v * v).sum();
    apply_rope_row(&mut values, &pe);
    let after: f32 = values.iter().map(|v| v * v).sum();
    assert!((before - after).abs() < 1e-5);
}

#[derive(Default)]
struct TestSource {
    tensors: HashMap<String, TensorInfo>,
    data: HashMap<String, Vec<u8>>,
}

impl TestSource {
    fn add(&mut self, name: String, dims: Vec<u64>, ggml_type: GGMLType) {
        let info = TensorInfo {
            name: name.clone(),
            dims,
            ggml_type,
            offset: 0,
        };
        self.data.insert(name.clone(), vec![0; info.nbytes()]);
        self.tensors.insert(name, info);
    }

    fn without(mut self, name: &str) -> Self {
        self.tensors.remove(name);
        self.data.remove(name);
        self
    }
}

impl TensorSource for TestSource {
    fn metadata(&self, _key: &str) -> Option<&rust_model_inference::core::tensor::MetaValue> {
        None
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.data.get(name).map(Vec::as_slice)
    }
}

fn minimal_source() -> TestSource {
    let mut source = TestSource::default();
    let hidden = 4096u64;
    source.add(
        format!("{PREFIX}.img_in.weight"),
        vec![64, hidden],
        GGMLType::BF16,
    );
    source.add(
        format!("{PREFIX}.proj_out.weight"),
        vec![hidden, 64],
        GGMLType::Q8_0,
    );
    source.add(
        format!("{PREFIX}.txt_in.in_layer.weight"),
        vec![hidden, hidden],
        GGMLType::BF16,
    );
    source.add(
        format!("{PREFIX}.txt_in.out_layer.weight"),
        vec![hidden, hidden],
        GGMLType::BF16,
    );
    source.add(
        format!("{PREFIX}.txt_in.text_norm.weight"),
        vec![hidden],
        GGMLType::BF16,
    );
    source.add(
        format!("{PREFIX}.time_text_embed.timestep_embedder.linear_1.weight"),
        vec![256, hidden],
        GGMLType::Q8_0,
    );
    source.add(
        format!("{PREFIX}.time_text_embed.timestep_embedder.linear_2.weight"),
        vec![hidden, hidden],
        GGMLType::Q8_0,
    );
    source.add(
        format!("{PREFIX}.modulation.1.weight"),
        vec![hidden, 4 * hidden],
        GGMLType::Q8_0,
    );
    source.add(
        format!("{PREFIX}.norm_out.linear.weight"),
        vec![hidden, hidden],
        GGMLType::F32,
    );
    for layer in 0..32u64 {
        let prefix = format!("{PREFIX}.transformer_blocks.{layer}");
        for suffix in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"] {
            source.add(
                format!("{prefix}.{suffix}.weight"),
                vec![hidden, hidden],
                GGMLType::Q8_0,
            );
        }
        source.add(
            format!("{prefix}.attn.norm_q.weight"),
            vec![128],
            GGMLType::F32,
        );
        source.add(
            format!("{prefix}.attn.norm_k.weight"),
            vec![128],
            GGMLType::F32,
        );
        source.add(
            format!("{prefix}.img_mlp.gate_up.weight"),
            vec![hidden, 24576],
            GGMLType::Q8_0,
        );
        source.add(
            format!("{prefix}.img_mlp.out.weight"),
            vec![12288, hidden],
            GGMLType::Q8_0,
        );
    }
    source
}

#[test]
fn full_contract_validates_and_detects_config() {
    let source = minimal_source();
    validate_dit(&source).unwrap();
    let config = config_from_source(&source).unwrap();
    assert_eq!(
        config,
        rust_model_inference::models::diffusion::qwen_image_2_1::QwenImage21Config {
            in_channels: 64,
            out_channels: 64,
            hidden_size: 4096,
            context_dim: 4096,
            head_dim: 128,
            intermediate_size: 12288,
            num_layers: 32,
        }
    );
}

#[test]
fn missing_block_tensor_is_rejected() {
    let source = minimal_source();
    let name = format!("{PREFIX}.transformer_blocks.7.attn.to_v.weight");
    let error = validate_dit(&source.without(&name)).unwrap_err();
    assert!(error.contains(&name), "{error}");
}

#[test]
fn malformed_dimensions_are_rejected_without_panicking() {
    let mut source = minimal_source();
    source
        .tensors
        .get_mut(&format!("{PREFIX}.img_in.weight"))
        .unwrap()
        .dims
        .clear();
    assert!(validate_dit(&source).unwrap_err().contains("img_in.weight"));

    let mut source = minimal_source();
    source
        .tensors
        .get_mut(&format!("{PREFIX}.transformer_blocks.0.attn.norm_q.weight"))
        .unwrap()
        .dims = vec![0];
    assert!(validate_dit(&source).unwrap_err().contains("norm_q.weight"));

    let mut source = minimal_source();
    source
        .tensors
        .get_mut(&format!("{PREFIX}.txt_in.text_norm.weight"))
        .unwrap()
        .dims = vec![4096, 2];
    assert!(validate_dit(&source)
        .unwrap_err()
        .contains("text_norm.weight"));

    let mut source = minimal_source();
    source
        .tensors
        .get_mut(&format!("{PREFIX}.txt_in.text_norm.weight"))
        .unwrap()
        .dims = vec![4096, 1];
    assert!(validate_dit(&source)
        .unwrap_err()
        .contains("text_norm.weight"));
}

#[test]
fn trailing_block_trim_is_rejected_by_the_model_contract() {
    let source = minimal_source();
    let name = format!("{PREFIX}.transformer_blocks.31.attn.to_q.weight");
    let trimmed = source.without(&name);
    let error = validate_dit(&trimmed).unwrap_err();
    assert!(error.contains(&name), "{error}");
}

#[test]
fn signature_requires_the_sniffing_tensor() {
    assert!(matches_signature(&minimal_source()));
    let source = minimal_source();
    let name = format!("{PREFIX}.txt_in.text_norm.weight");
    assert!(!matches_signature(&source.without(&name)));
}

#[test]
fn empty_source_rejects_everything() {
    let source = TestSource::default();
    assert!(validate_dit(&source).is_err());
    assert!(!matches_signature(&source));
}
