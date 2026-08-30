use super::super::contract::{
    require_bool, require_clip, require_clippable, require_f32, require_string, require_tensor,
    require_u32,
};
use crate::core::tensor::{GGMLType, TensorSource};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gemma4AudioConfig {
    pub layers: usize,
    pub embd: usize,
    pub heads: usize,
    pub mel_bins: usize,
    pub projection: usize,
}

impl Gemma4AudioConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        require_clip(source)?;
        require_bool(source, "clip.has_audio_encoder", true)?;
        require_string(source, "clip.audio.projector_type", "gemma4a")?;
        require_u32(source, "clip.audio.projection_dim", 1536)?;
        require_u32(source, "clip.audio.embedding_length", 1024)?;
        require_u32(source, "clip.audio.feed_forward_length", 4096)?;
        require_u32(source, "clip.audio.block_count", 12)?;
        require_u32(source, "clip.audio.attention.head_count", 8)?;
        require_u32(source, "clip.audio.num_mel_bins", 128)?;
        require_f32(source, "clip.audio.attention.layer_norm_epsilon", 1e-5)?;

        for (name, dims, ty) in [
            ("a.pre_encode.out.bias", &[1536][..], GGMLType::F32),
            ("a.pre_encode.out.weight", &[1024, 1536][..], GGMLType::F16),
            (
                "a.input_projection.weight",
                &[1024, 1024][..],
                GGMLType::F32,
            ),
            ("a.conv1d.0.weight", &[3, 3, 1, 128][..], GGMLType::F32),
            ("a.conv1d.0.norm.weight", &[128][..], GGMLType::F32),
            ("a.conv1d.1.weight", &[3, 3, 128, 32][..], GGMLType::F32),
            ("a.conv1d.1.norm.weight", &[32][..], GGMLType::F32),
            (
                "mm.a.input_projection.weight",
                &[1536, 1536][..],
                GGMLType::F16,
            ),
        ] {
            require_tensor(source, name, dims, ty)?;
        }
        for layer in 0..12 {
            let prefix = format!("a.blk.{layer}");
            for name in ["ffn_up", "ffn_down", "ffn_up_1", "ffn_down_1"] {
                let dims = if name.starts_with("ffn_up") {
                    &[1024, 4096][..]
                } else {
                    &[4096, 1024][..]
                };
                require_clippable(source, &format!("{prefix}.{name}"), dims)?;
            }
            for name in [
                "ffn_post_norm.weight",
                "ffn_norm.weight",
                "ffn_post_norm_1.weight",
                "ffn_norm_1.weight",
                "norm_conv.weight",
                "conv_norm.weight",
                "ln2.weight",
                "attn_post_norm.weight",
                "attn_pre_norm.weight",
            ] {
                require_tensor(source, &format!("{prefix}.{name}"), &[1024], GGMLType::F32)?;
            }
            require_tensor(
                source,
                &format!("{prefix}.conv_dw.weight"),
                &[5, 1024],
                GGMLType::F32,
            )?;
            require_clippable(source, &format!("{prefix}.conv_pw2"), &[1024, 1024])?;
            require_clippable(source, &format!("{prefix}.conv_pw1"), &[1024, 2048])?;
            require_clippable(source, &format!("{prefix}.attn_k"), &[1024, 1024])?;
            require_tensor(
                source,
                &format!("{prefix}.per_dim_scale.weight"),
                &[128],
                GGMLType::F32,
            )?;
            require_clippable(source, &format!("{prefix}.attn_out"), &[1024, 1024])?;
            require_clippable(source, &format!("{prefix}.attn_q"), &[1024, 1024])?;
            require_tensor(
                source,
                &format!("{prefix}.attn_k_rel.weight"),
                &[1024, 1024],
                GGMLType::F16,
            )?;
            require_clippable(source, &format!("{prefix}.attn_v"), &[1024, 1024])?;
        }

        Ok(Self {
            layers: 12,
            embd: 1024,
            heads: 8,
            mel_bins: 128,
            projection: 1536,
        })
    }
}
