use super::super::contract::{
    require_array, require_bool, require_clip, require_f32, require_tensor, require_u32,
};
use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorSource};

/// Gemma4 unified vision configuration (gemma4uv — Unsloth/12B variant).
///
/// This variant has no transformer blocks in the vision encoder; instead it
/// patches images via `im2col`, applies a learnable LayerNorm, a Linear
/// projection to the model dimension, then learns factorized (x, y)
/// positional embeddings and a second projection to the trunk width.
///
/// The reference graph is taken from llama.cpp's `tools/mtmd/models/gemma4uv.cpp`.
/// llama.cpp computes the effective patch size for the unified variant as
/// `patch_size * n_merge` and then sets `n_merge = 1` (see `clip.cpp` line 1636).
/// The metadata `clip.vision.patch_size` therefore reflects the *base* patch
/// size, not the kernel size used by im2col. For Gemma-4 12B the base is 16
/// and `n_merge` defaults to 3, so the effective patch size is 48.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gemma4UvConfig {
    pub embd: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub n_merge: usize,
    pub position_size: usize,
    pub projection: usize,
    pub norm_eps: f32,
    pub rms_eps: f32,
    pub image_min_pixels: usize,
    pub image_max_pixels: usize,
}

impl Gemma4UvConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        require_clip(source)?;
        require_bool(source, "clip.has_vision_encoder", true)?;
        match source.metadata("clip.vision.projector_type") {
            Some(MetaValue::String(value)) if value == "gemma4uv" => {}
            Some(value) => {
                return Err(format!(
                    "Invalid metadata clip.vision.projector_type: expected \"gemma4uv\", got {value:?}"
                ))
            }
            None => return Err("Missing metadata: clip.vision.projector_type".into()),
        }
        require_u32(source, "clip.vision.projection_dim", 3840)?;
        require_u32(source, "clip.vision.image_size", 224)?;
        // The metadata declares 16 here, but llama.cpp derives the effective
        // kernel patch size as base * n_merge and then sets n_merge = 1 (see
        // `clip.cpp` line 1636). For 12B we hardcode base = 16, n_merge = 3 →
        // effective patch_size = 48.
        let base_patch_size = 16usize;
        let n_merge = 3usize;
        let patch_size = base_patch_size * n_merge;
        require_u32(source, "clip.vision.embedding_length", 3840)?;
        require_u32(source, "clip.vision.feed_forward_length", 0)?;
        require_u32(source, "clip.vision.block_count", 0)?;
        require_u32(source, "clip.vision.attention.head_count", 1)?;
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
        // Defaults from llama.cpp gemma4uv load_hparams block; the metadata
        // is not always populated for this projector.
        let image_min_pixels = 161280usize;
        let image_max_pixels = 2580480usize;

        // 6912 = patch_size * patch_size * in_channels = 48 * 48 * 3 for 12B.
        let in_channels = 3usize;
        let patch_dim = patch_size * patch_size * in_channels;
        require_tensor(
            source,
            "v.patch_embd.weight",
            &[patch_dim as u64, 3840],
            GGMLType::F32,
        )?;
        require_tensor(source, "v.patch_embd.bias", &[3840], GGMLType::F32)?;
        require_tensor(
            source,
            "v.patch_norm.1.weight",
            &[patch_dim as u64],
            GGMLType::F32,
        )?;
        require_tensor(
            source,
            "v.patch_norm.1.bias",
            &[patch_dim as u64],
            GGMLType::F32,
        )?;
        require_tensor(source, "v.patch_norm.2.weight", &[3840], GGMLType::F32)?;
        require_tensor(source, "v.patch_norm.2.bias", &[3840], GGMLType::F32)?;
        require_tensor(source, "v.patch_norm.3.weight", &[3840], GGMLType::F32)?;
        require_tensor(source, "v.patch_norm.3.bias", &[3840], GGMLType::F32)?;
        require_tensor(
            source,
            "v.position_embd.weight",
            &[3840, 1120, 2],
            GGMLType::F32,
        )?;
        require_tensor(
            source,
            "mm.input_projection.weight",
            &[3840, 3840],
            GGMLType::F16,
        )?;

        Ok(Self {
            embd: 3840,
            image_size: 224,
            patch_size,
            in_channels,
            n_merge,
            position_size: 1120,
            projection: 3840,
            norm_eps: 1e-5,
            rms_eps: 1e-6,
            image_min_pixels,
            image_max_pixels,
        })
    }
}