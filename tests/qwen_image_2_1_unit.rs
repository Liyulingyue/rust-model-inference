//! Qwen-Image-2.1 tensor contract tests.

use rust_model_inference::core::tensor::{GGMLType, TensorInfo, TensorSource};
use rust_model_inference::models::diffusion::qwen_image_2_1::{
    config_from_source, matches_signature, validate_dit,
};
use std::collections::HashMap;

const PREFIX: &str = "model.diffusion_model";

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
