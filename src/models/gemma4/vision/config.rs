use super::super::contract::{
    require_array, require_bool, require_clip, require_clippable, require_f32, require_string,
    require_tensor, require_u32,
};
use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorSource};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gemma4VisionConfig {
    pub layers: usize,
    pub embd: usize,
    pub heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub merge: usize,
    pub projection: usize,
}

impl Gemma4VisionConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        require_clip(source)?;
        require_bool(source, "clip.has_vision_encoder", true)?;
        require_string(source, "clip.vision.projector_type", "gemma4v")?;
        require_u32(source, "clip.vision.projection_dim", 1536)?;
        require_u32(source, "clip.vision.image_size", 224)?;
        require_u32(source, "clip.vision.patch_size", 16)?;
        require_u32(source, "clip.vision.embedding_length", 768)?;
        require_u32(source, "clip.vision.feed_forward_length", 3072)?;
        require_u32(source, "clip.vision.block_count", 16)?;
        require_u32(source, "clip.vision.attention.head_count", 12)?;
        require_array(
            source,
            "clip.vision.image_mean",
            MetaValueType::Float32,
            &[
                MetaValue::Float32(0.0),
                MetaValue::Float32(0.0),
                MetaValue::Float32(0.0),
            ],
        )?;
        require_array(
            source,
            "clip.vision.image_std",
            MetaValueType::Float32,
            &[
                MetaValue::Float32(1.0),
                MetaValue::Float32(1.0),
                MetaValue::Float32(1.0),
            ],
        )?;
        require_f32(source, "clip.vision.attention.layer_norm_epsilon", 1e-6)?;

        for layer in 0..16 {
            let prefix = format!("v.blk.{layer}");
            require_tensor(
                source,
                &format!("{prefix}.ln1.weight"),
                &[768],
                GGMLType::F32,
            )?;
            for name in ["ffn_down", "ffn_gate", "ffn_up"] {
                let dims = if name == "ffn_down" {
                    &[3072, 768][..]
                } else {
                    &[768, 3072][..]
                };
                require_clippable(source, &format!("{prefix}.{name}"), dims)?;
            }
            for name in [
                "attn_post_norm.weight",
                "ffn_post_norm.weight",
                "ln2.weight",
            ] {
                require_tensor(source, &format!("{prefix}.{name}"), &[768], GGMLType::F32)?;
            }
            require_tensor(
                source,
                &format!("{prefix}.attn_k_norm.weight"),
                &[64],
                GGMLType::F32,
            )?;
            for name in ["attn_k", "attn_out", "attn_q", "attn_v"] {
                require_clippable(source, &format!("{prefix}.{name}"), &[768, 768])?;
                if name == "attn_k" || name == "attn_q" {
                    require_tensor(
                        source,
                        &format!("{prefix}.{name}_norm.weight"),
                        &[64],
                        GGMLType::F32,
                    )?;
                }
            }
        }
        for (name, dims, ty) in [
            (
                "mm.input_projection.weight",
                &[768, 1536][..],
                GGMLType::F16,
            ),
            ("v.patch_embd.weight", &[16, 16, 3, 768][..], GGMLType::F16),
            (
                "v.position_embd.weight",
                &[768, 10_240, 2][..],
                GGMLType::F32,
            ),
        ] {
            require_tensor(source, name, dims, ty)?;
        }

        Ok(Self {
            layers: 16,
            embd: 768,
            heads: 12,
            image_size: 224,
            patch_size: 16,
            merge: 3,
            projection: 1536,
        })
    }
}
