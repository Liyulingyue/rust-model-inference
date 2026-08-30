use super::super::contract::{
    require_array, require_f32, require_gemma4_token_table, require_string, require_tensor,
    require_u32,
};
use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorSource};

pub(super) const LAYERS: usize = 35;
pub(super) const BASE_KV_LAYERS: usize = 15;
pub(super) const EMBED: usize = 1536;
pub(super) const HEADS: usize = 8;
pub(super) const FULL_HEAD_DIM: usize = 512;
pub(super) const SWA_HEAD_DIM: usize = 256;
pub(super) const BASE_FFN_LAYERS: usize = 15;
pub(super) const MAX_FFN: usize = 12_288;
pub(super) const VOCAB: usize = 262_144;
pub(super) const PER_LAYER: usize = 256;
pub(super) const PER_LAYER_ALL: usize = LAYERS * PER_LAYER;
pub(super) const CONTEXT: usize = 131_072;
pub(super) const EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gemma4Config {
    pub layers: usize,
    pub embd: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub vocab: usize,
    pub full_head_dim: usize,
    pub swa_head_dim: usize,
    pub shared_kv_layers: usize,
    pub per_layer_width: usize,
    pub sliding_window: usize,
    pub logit_softcap: f32,
}

impl Gemma4Config {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        require_string(source, "general.architecture", "gemma4")?;
        require_string(source, "general.type", "model")?;
        require_u32(source, "gemma4.block_count", 35)?;
        require_u32(source, "gemma4.context_length", 131_072)?;
        require_u32(source, "gemma4.embedding_length", 1536)?;
        require_array(
            source,
            "gemma4.feed_forward_length",
            MetaValueType::Int32,
            &[
                vec![MetaValue::Int32(6144); 15],
                vec![MetaValue::Int32(12_288); 20],
            ]
            .concat(),
        )?;
        require_u32(source, "gemma4.attention.head_count", 8)?;
        require_u32(source, "gemma4.attention.head_count_kv", 1)?;
        require_f32(source, "gemma4.rope.freq_base", 1_000_000.0)?;
        require_f32(source, "gemma4.rope.freq_base_swa", 10_000.0)?;
        require_f32(source, "gemma4.attention.layer_norm_rms_epsilon", 1e-6)?;
        require_u32(source, "gemma4.attention.key_length", 512)?;
        require_u32(source, "gemma4.attention.value_length", 512)?;
        require_f32(source, "gemma4.final_logit_softcapping", 30.0)?;
        require_u32(source, "gemma4.attention.sliding_window", 512)?;
        require_u32(source, "gemma4.attention.shared_kv_layers", 20)?;
        require_u32(source, "gemma4.embedding_length_per_layer_input", 256)?;
        require_array(
            source,
            "gemma4.attention.sliding_window_pattern",
            MetaValueType::Bool,
            &(0..35)
                .map(|i| MetaValue::Bool(i % 5 != 4))
                .collect::<Vec<_>>(),
        )?;
        require_u32(source, "gemma4.attention.key_length_swa", 256)?;
        require_u32(source, "gemma4.attention.value_length_swa", 256)?;
        require_u32(source, "gemma4.rope.dimension_count", 512)?;
        require_u32(source, "gemma4.rope.dimension_count_swa", 256)?;
        require_string(source, "tokenizer.ggml.model", "gemma4")?;
        require_gemma4_token_table(source)?;

        for (name, dims, ty) in [
            ("output_norm.weight", &[1536][..], GGMLType::F32),
            (
                "per_layer_model_proj.weight",
                &[1536, 8960][..],
                GGMLType::BF16,
            ),
            ("per_layer_proj_norm.weight", &[256][..], GGMLType::F32),
            (
                "per_layer_token_embd.weight",
                &[8960, 262_144][..],
                GGMLType::Q8_0,
            ),
            ("rope_freqs.weight", &[256][..], GGMLType::F32),
            ("token_embd.weight", &[1536, 262_144][..], GGMLType::Q8_0),
        ] {
            require_tensor(source, name, dims, ty)?;
        }
        for layer in 0..35 {
            let head_dim = if layer % 5 == 4 { 512 } else { 256 };
            let ffn = if layer < 15 { 6144 } else { 12_288 };
            let prefix = format!("blk.{layer}");
            for (name, dims, ty) in [
                ("attn_k.weight", vec![1536, head_dim], GGMLType::Q8_0),
                ("attn_k_norm.weight", vec![head_dim], GGMLType::F32),
                ("attn_norm.weight", vec![1536], GGMLType::F32),
                (
                    "attn_output.weight",
                    vec![head_dim * 8, 1536],
                    GGMLType::Q8_0,
                ),
                ("attn_q.weight", vec![1536, head_dim * 8], GGMLType::Q8_0),
                ("attn_q_norm.weight", vec![head_dim], GGMLType::F32),
                ("attn_v.weight", vec![1536, head_dim], GGMLType::Q8_0),
                ("ffn_down.weight", vec![ffn, 1536], GGMLType::Q8_0),
                ("ffn_gate.weight", vec![1536, ffn], GGMLType::Q8_0),
                ("ffn_norm.weight", vec![1536], GGMLType::F32),
                ("ffn_up.weight", vec![1536, ffn], GGMLType::Q8_0),
                ("inp_gate.weight", vec![1536, 256], GGMLType::F32),
                ("layer_output_scale.weight", vec![1], GGMLType::F32),
                ("post_attention_norm.weight", vec![1536], GGMLType::F32),
                ("post_ffw_norm.weight", vec![1536], GGMLType::F32),
                ("post_norm.weight", vec![1536], GGMLType::F32),
                ("proj.weight", vec![256, 1536], GGMLType::F32),
            ] {
                require_tensor(source, &format!("{prefix}.{name}"), &dims, ty)?;
            }
        }

        Ok(Self {
            layers: 35,
            embd: 1536,
            heads: 8,
            kv_heads: 1,
            vocab: 262_144,
            full_head_dim: 512,
            swa_head_dim: 256,
            shared_kv_layers: 20,
            per_layer_width: 256,
            sliding_window: 512,
            logit_softcap: 30.0,
        })
    }
}
