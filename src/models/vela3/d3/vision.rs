//! Safetensors view over the existing Qwen3.5 image encoder.

use super::{require_shape, tensor, trace, D3Encoder};
use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::qwen35::vision::clip_config::ClipVisionConfig;
use crate::models::qwen35::vision::{
    qwen_smart_resize, VisionEncoder, VisionGrid, VisionScratchpad,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) struct D3Images {
    pub(crate) grids: Vec<VisionGrid>,
    projected: Vec<Vec<f32>>,
}

pub(crate) struct D3ImageInput {
    pub(crate) ids: Vec<u32>,
    pub(crate) positions: Vec<[usize; 3]>,
    /// Includes the end boundary, so an original token count maps to an expanded count.
    pub(crate) original_to_expanded: Vec<usize>,
    pub(super) embeddings: Vec<f32>,
}

enum Data {
    Alias(String),
    Owned(Vec<u8>),
}

struct View<'a> {
    base: &'a dyn TensorSource,
    metadata: HashMap<String, MetaValue>,
    entries: HashMap<String, (TensorInfo, Data)>,
}

impl TensorSource for View<'_> {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.entries.get(name).map(|v| &v.0)
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        match &self.entries.get(name)?.1 {
            Data::Alias(name) => self.base.tensor_slice(name),
            Data::Owned(bytes) => Some(bytes),
        }
    }
}

impl<'a> View<'a> {
    fn alias(&mut self, name: String, original: String) -> Result<(), String> {
        let mut info = self
            .base
            .tensor_info(&original)
            .ok_or_else(|| format!("missing d3 vision tensor {original}"))?
            .clone();
        info.name = name.clone();
        self.entries.insert(name, (info, Data::Alias(original)));
        Ok(())
    }

    fn owned(&mut self, name: &str, dims: Vec<u64>, dtype: GGMLType, bytes: Vec<u8>) {
        self.entries.insert(
            name.to_owned(),
            (
                TensorInfo {
                    name: name.into(),
                    dims,
                    ggml_type: dtype,
                    offset: 0,
                },
                Data::Owned(bytes),
            ),
        );
    }

    fn new(base: &'a dyn TensorSource, config: &Value) -> Result<Self, String> {
        let n = |name: &str| {
            config[name]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| format!("invalid d3 vision {name}"))
        };
        let (hidden, patch, depth, heads, merge) = (
            n("hidden_size")?,
            n("patch_size")?,
            n("depth")?,
            n("num_heads")?,
            n("spatial_merge_size")?,
        );
        let positions = n("num_position_embeddings")?;
        let side = (positions as f64).sqrt() as usize;
        if n("in_channels")? != 3
            || n("temporal_patch_size")? != 2
            || hidden % heads != 0
            || (hidden / heads) % 4 != 0
            || side * side != positions
            || config["deepstack_visual_indexes"]
                .as_array()
                .is_some_and(|v| !v.is_empty())
            || config["rope_parameters"]["rope_type"] != "axial"
            || config["rope_parameters"]["rope_theta"].as_f64() != Some(10000.)
            || config["hidden_act"] != "gelu_pytorch_tanh"
        {
            return Err(
                "d3 vision configuration differs from the Qwen3.5 image encoder contract".into(),
            );
        }
        let mut view = Self {
            base,
            metadata: HashMap::new(),
            entries: HashMap::new(),
        };
        for (key, value) in [
            ("projection_dim", n("out_hidden_size")?),
            (
                "image_size",
                side.checked_mul(patch)
                    .ok_or("vision image_size overflow")?,
            ),
            ("patch_size", patch),
            ("embedding_length", hidden),
            ("feed_forward_length", n("intermediate_size")?),
            ("block_count", depth),
            ("attention.head_count", heads),
            ("spatial_merge_size", merge),
            ("image_min_pixels", 65536),
            ("image_max_pixels", 1638400),
        ] {
            view.metadata.insert(
                format!("clip.vision.{key}"),
                MetaValue::Uint64(value as u64),
            );
        }
        view.metadata.insert(
            "clip.vision.attention.layer_norm_epsilon".into(),
            MetaValue::Float32(1e-6),
        );
        view.metadata
            .insert("clip.use_gelu".into(), MetaValue::Bool(true));
        for key in ["image_mean", "image_std"] {
            view.metadata.insert(
                format!("clip.vision.{key}"),
                MetaValue::Array(MetaValueType::Float32, vec![MetaValue::Float32(0.5); 3]),
            );
        }
        for (target, original) in [
            ("v.patch_embd.bias", "visual.patch_embed.proj.bias"),
            ("v.post_ln.weight", "visual.merger.norm.weight"),
            ("v.post_ln.bias", "visual.merger.norm.bias"),
            ("mm.0.weight", "visual.merger.linear_fc1.weight"),
            ("mm.0.bias", "visual.merger.linear_fc1.bias"),
            ("mm.2.weight", "visual.merger.linear_fc2.weight"),
            ("mm.2.bias", "visual.merger.linear_fc2.bias"),
        ] {
            view.alias(target.into(), original.into())?;
        }
        for layer in 0..depth {
            for (target, original) in [
                ("ln1", "norm1"),
                ("ln2", "norm2"),
                ("attn_qkv", "attn.qkv"),
                ("attn_out", "attn.proj"),
                ("ffn_up", "mlp.linear_fc1"),
                ("ffn_down", "mlp.linear_fc2"),
            ] {
                for suffix in ["weight", "bias"] {
                    view.alias(
                        format!("v.blk.{layer}.{target}.{suffix}"),
                        format!("visual.blocks.{layer}.{original}.{suffix}"),
                    )?;
                }
            }
        }
        // The shared encoder reads position grids as F32 bytes; all large linear weights stay mmap-backed.
        require_shape(base, "visual.pos_embed.weight", &[positions, hidden])?;
        let position = tensor::values(base, "visual.pos_embed.weight")?;
        view.owned(
            "v.position_embd.weight",
            vec![hidden as u64, positions as u64],
            GGMLType::F32,
            position.into_iter().flat_map(f32::to_le_bytes).collect(),
        );
        require_shape(
            base,
            "visual.patch_embed.proj.weight",
            &[hidden, 3, 2, patch, patch],
        )?;
        let info = base.tensor_info("visual.patch_embed.proj.weight").unwrap();
        let original = base
            .tensor_slice("visual.patch_embed.proj.weight")
            .ok_or("missing vision patch bytes")?;
        let item = match info.ggml_type {
            GGMLType::BF16 | GGMLType::F16 => 2,
            GGMLType::F32 => 4,
            _ => return Err("unsupported d3 vision patch dtype".into()),
        };
        let plane = patch
            .checked_mul(patch)
            .and_then(|v| v.checked_mul(item))
            .ok_or("vision patch dimensions overflow")?;
        for temporal in 0..2 {
            let mut bytes = Vec::with_capacity(hidden * 3 * plane);
            for h in 0..hidden {
                for channel in 0..3 {
                    let start = ((h * 3 + channel) * 2 + temporal) * plane;
                    bytes.extend_from_slice(
                        original
                            .get(start..start + plane)
                            .ok_or("invalid d3 patch data")?,
                    );
                }
            }
            view.owned(
                if temporal == 0 {
                    "v.patch_embd.weight"
                } else {
                    "v.patch_embd.weight.1"
                },
                vec![patch as u64, patch as u64, 3, hidden as u64],
                info.ggml_type,
                bytes,
            );
        }
        Ok(view)
    }
}

fn resize_grid(
    width: usize,
    height: usize,
    config: &ClipVisionConfig,
) -> Result<VisionGrid, String> {
    let shared = qwen_smart_resize(width, height, config)?;
    let factor = config.patch_size * config.spatial_merge_size;
    let x = width as f64 / factor as f64;
    let y = height as f64 / factor as f64;
    // The pinned processor uses Python round (ties to even); shared Qwen resizing rounds away.
    if x.round().max(1.) == x.round_ties_even() && y.round().max(1.) == y.round_ties_even() {
        return Ok(shared);
    }
    let mut aligned_width = x.round_ties_even() as usize * factor;
    let mut aligned_height = y.round_ties_even() as usize * factor;
    let pixels = aligned_width
        .checked_mul(aligned_height)
        .ok_or("d3 aligned image pixel count overflow")?;
    let original = width
        .checked_mul(height)
        .ok_or("d3 image pixel count overflow")? as f64;
    if pixels > config.image_max_pixels {
        let scale = (original / config.image_max_pixels as f64).sqrt();
        aligned_width = (width as f64 / scale / factor as f64).floor().max(1.) as usize * factor;
        aligned_height = (height as f64 / scale / factor as f64).floor().max(1.) as usize * factor;
    } else if pixels < config.image_min_pixels {
        let scale = (config.image_min_pixels as f64 / original).sqrt();
        aligned_width = (width as f64 * scale / factor as f64).ceil() as usize * factor;
        aligned_height = (height as f64 * scale / factor as f64).ceil() as usize * factor;
    }
    VisionGrid::from_image_size(
        1,
        aligned_width,
        aligned_height,
        config.patch_size,
        config.spatial_merge_size,
    )
}

pub(super) fn prepare(
    source: &dyn TensorSource,
    config: &Value,
    pool: Arc<ComputePool>,
    images: &[Value],
) -> Result<D3Images, String> {
    if images.is_empty() {
        return Ok(D3Images {
            grids: Vec::new(),
            projected: Vec::new(),
        });
    }
    let view = View::new(source, config)?;
    // shortcut: F32/F16-GELU tower is approximate; add BF16 activations before claiming image parity.
    let encoder = VisionEncoder::from_source(&view, Arc::clone(&pool))?;
    let mut scratch = VisionScratchpad::new(&encoder.config);
    let mut prepared = D3Images {
        grids: Vec::with_capacity(images.len()),
        projected: Vec::with_capacity(images.len()),
    };
    for image in images {
        let image = super::super::vision::load_image(image)?;
        let grid = resize_grid(
            image.width() as usize,
            image.height() as usize,
            &encoder.config,
        )?;
        let image = super::super::vision::resize(&image, grid.image_width(), grid.image_height())?;
        let pixels = image
            .pixels()
            .flat_map(|pixel| (0..3).map(move |c| (pixel[c] as f32 * (1. / 255.) - 0.5) / 0.5))
            .collect::<Vec<_>>();
        let actual = encoder.encode_image(
            &pixels,
            grid.image_width(),
            grid.image_height(),
            &mut scratch,
            &pool,
        )?;
        if actual != grid {
            return Err("d3 vision preprocessing and encoder grids differ".into());
        }
        let count = grid.checked_token_count()?;
        if scratch.projected.len() != count * encoder.config.projection_dim
            || !scratch.projected.iter().all(|x| x.is_finite())
        {
            return Err("invalid d3 projected image features".into());
        }
        trace(
            "vela3.d3.vision.projected",
            None,
            &[count, encoder.config.projection_dim],
            &scratch.projected,
        );
        prepared.grids.push(grid);
        prepared.projected.push(scratch.projected.clone());
    }
    Ok(prepared)
}

pub(super) fn expand(
    encoder: &D3Encoder,
    ids: &[u32],
    images: &D3Images,
    image_token: u32,
) -> Result<D3ImageInput, String> {
    let mut expanded = Vec::new();
    let mut map = Vec::with_capacity(ids.len() + 1);
    let mut image = 0;
    for &id in ids {
        map.push(expanded.len());
        if id == image_token {
            let grid = images
                .grids
                .get(image)
                .ok_or("image placeholder has no image")?;
            expanded.extend(std::iter::repeat_n(id, grid.checked_token_count()?));
            image += 1;
        } else {
            expanded.push(id);
        }
    }
    map.push(expanded.len());
    if image != images.grids.len() {
        return Err("image count does not match prompt placeholders".into());
    }
    encoder.validate_length(expanded.len())?;
    let mut embeddings = tensor::embed(
        &encoder.source,
        "language_model.embed_tokens.weight",
        &expanded,
    )?;
    let mut row = 0;
    let mut image = 0;
    while row < expanded.len() {
        if expanded[row] == image_token {
            let count = images.grids[image].checked_token_count()?;
            let projected = &images.projected[image];
            if projected.len() != count * encoder.config.hidden {
                return Err("image projection does not fit d3 hidden size".into());
            }
            embeddings[row * encoder.config.hidden..(row + count) * encoder.config.hidden]
                .copy_from_slice(projected);
            row += count;
            image += 1;
        } else {
            row += 1;
        }
    }
    encoder.cast(&mut embeddings);
    let (positions, _) =
        crate::models::qwen35::build_qwen35_positions(&expanded, Some(image_token), &images.grids)?;
    Ok(D3ImageInput {
        ids: expanded,
        positions: positions.into_iter().map(|p| [p[0], p[1], p[2]]).collect(),
        original_to_expanded: map,
        embeddings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vision_fixture() -> (crate::models::vela3::tensor::tests::Source, Value) {
        let config = serde_json::json!({
            "hidden_size": 8, "intermediate_size": 16, "depth": 1, "num_heads": 2,
            "out_hidden_size": 4, "num_position_embeddings": 4, "patch_size": 2,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "in_channels": 3,
            "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": [],
            "rope_parameters": {"rope_type": "axial", "rope_theta": 10000.}
        });
        let mut source = crate::models::vela3::tensor::tests::Source::default();
        for (suffix, shape) in [
            ("patch_embed.proj.weight", vec![8, 3, 2, 2, 2]),
            ("patch_embed.proj.bias", vec![8]),
            ("pos_embed.weight", vec![4, 8]),
            ("blocks.0.norm1.weight", vec![8]),
            ("blocks.0.norm1.bias", vec![8]),
            ("blocks.0.norm2.weight", vec![8]),
            ("blocks.0.norm2.bias", vec![8]),
            ("blocks.0.attn.qkv.weight", vec![24, 8]),
            ("blocks.0.attn.qkv.bias", vec![24]),
            ("blocks.0.attn.proj.weight", vec![8, 8]),
            ("blocks.0.attn.proj.bias", vec![8]),
            ("blocks.0.mlp.linear_fc1.weight", vec![16, 8]),
            ("blocks.0.mlp.linear_fc1.bias", vec![16]),
            ("blocks.0.mlp.linear_fc2.weight", vec![8, 16]),
            ("blocks.0.mlp.linear_fc2.bias", vec![8]),
            ("merger.norm.weight", vec![8]),
            ("merger.norm.bias", vec![8]),
            ("merger.linear_fc1.weight", vec![32, 32]),
            ("merger.linear_fc1.bias", vec![32]),
            ("merger.linear_fc2.weight", vec![4, 32]),
            ("merger.linear_fc2.bias", vec![4]),
        ] {
            let name = format!("visual.{suffix}");
            let salt = name.bytes().map(usize::from).sum::<usize>();
            let values = (0..shape.iter().product())
                .map(|i| (((i + salt) % 17) as f32 - 8.) * 0.025)
                .collect::<Vec<_>>();
            source.add(&name, &shape, &values);
        }
        (source, config)
    }

    #[test]
    fn vision_view_matches_independent_transformers_projection_and_borrows_linear_weights() {
        let (source, config) = vision_fixture();
        let view = View::new(&source, &config).unwrap();
        assert_eq!(
            view.tensor_slice("v.blk.0.attn_qkv.weight")
                .unwrap()
                .as_ptr(),
            source
                .tensor_slice("visual.blocks.0.attn.qkv.weight")
                .unwrap()
                .as_ptr()
        );
        let pool = Arc::new(ComputePool::new(1));
        let encoder = VisionEncoder::from_source(&view, Arc::clone(&pool)).unwrap();
        let mut scratch = VisionScratchpad::new(&encoder.config);
        let pixels = (0..8 * 4 * 3)
            .map(|i| (i % 17) as f32 / 8. - 1.)
            .collect::<Vec<_>>();
        let grid = encoder
            .encode_image(&pixels, 8, 4, &mut scratch, &pool)
            .unwrap();
        assert_eq!(grid.token_count(), 2);
        // Transformers 5.19.0 Qwen3_5VisionModel, independent FP32 forward.
        let expected = [
            0.15833405,
            0.07968416,
            0.047869712,
            -0.22240381,
            0.17183244,
            0.13942735,
            0.054942608,
            -0.3313856,
        ];
        let max_abs = scratch
            .projected
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f32, f32::max);
        assert!(
            max_abs < 5e-4,
            "vision max_abs={max_abs}: {:?}",
            scratch.projected
        );
    }

    #[test]
    fn resize_matches_python_ties_to_even_and_small_dimension_budget_branch() {
        let (source, config) = vision_fixture();
        let view = View::new(&source, &config).unwrap();
        let mut config = ClipVisionConfig::from_source(&view).unwrap();
        config.patch_size = 16;
        config.spatial_merge_size = 2;
        let grid = resize_grid(592, 448, &config).unwrap();
        assert_eq!((grid.image_width(), grid.image_height()), (576, 448));
        let grid = resize_grid(15, 3000, &config).unwrap();
        assert_eq!((grid.image_width(), grid.image_height()), (32, 3648));
    }

    #[test]
    fn image_expansion_keeps_index_mapping_and_uses_existing_qwen35_positions() {
        let fixture = super::super::tests::tiny_fixture();
        let encoder =
            D3Encoder::load(&fixture.0, &fixture.0, Arc::new(ComputePool::new(1))).unwrap();
        let images = D3Images {
            grids: vec![VisionGrid {
                grid_t: 1,
                grid_h: 2,
                grid_w: 3,
                patch_size: 16,
                merge_size: 2,
            }],
            projected: vec![vec![0.25; 6 * 4]],
        };
        let input = expand(&encoder, &[1, 4, 2, 3], &images, 4).unwrap();
        assert_eq!(input.ids, [1, 4, 4, 4, 4, 4, 4, 2, 3]);
        assert_eq!(input.original_to_expanded, [0, 1, 7, 8, 9]);
        assert_eq!(input.positions[6], [1, 2, 3]);
        assert_eq!(input.positions[7], [4, 4, 4]);
        assert_eq!(&input.embeddings[4..28], &[0.25; 24]);
        assert!(expand(&encoder, &[1, 2], &images, 4).is_err());
        assert!(expand(&encoder, &[1, 4, 4], &images, 4).is_err());
    }
}
