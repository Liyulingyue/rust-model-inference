use crate::core::tensor::{MetaValue, TensorSource};

#[derive(Debug, Clone)]
pub struct ClipVisionConfig {
    pub projection_dim: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub spatial_merge_size: usize,
    pub image_min_pixels: usize,
    pub image_max_pixels: usize,
    pub video_min_pixels: usize,
    pub video_max_pixels: usize,
    pub eps: f32,
    pub use_gelu: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub has_deepstack_layers: Vec<bool>,
}

impl ClipVisionConfig {
    pub fn from_source<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let get_u32 = |key: &str| -> Result<u32, String> {
            source.metadata(key)
                .and_then(|v| v.to_u64())
                .map(|v| v as u32)
                .ok_or_else(|| format!("Missing clip metadata: {}", key))
        };

        let get_f32 = |key: &str| -> Result<f32, String> {
            source.metadata(key)
                .and_then(|v| v.to_f64())
                .map(|v| v as f32)
                .ok_or_else(|| format!("Missing clip metadata: {}", key))
        };

        let get_bool = |key: &str| -> bool {
            source.metadata(key)
                .and_then(|v| match v { MetaValue::Bool(b) => Some(*b), _ => None })
                .unwrap_or(false)
        };

        let projection_dim = get_u32("clip.vision.projection_dim")? as usize;
        let image_size = get_u32("clip.vision.image_size")? as usize;
        let patch_size = get_u32("clip.vision.patch_size")? as usize;
        let n_embd = get_u32("clip.vision.embedding_length")? as usize;
        let n_ff = get_u32("clip.vision.feed_forward_length")? as usize;
        let n_layer = get_u32("clip.vision.block_count")? as usize;
        let n_head = get_u32("clip.vision.attention.head_count")? as usize;
        let spatial_merge_size = source.metadata("clip.vision.spatial_merge_size")
            .and_then(|v| v.to_u64())
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "clip.vision.spatial_merge_size does not fit usize")?
            .unwrap_or(2);
        let factor = patch_size
            .checked_mul(spatial_merge_size)
            .ok_or("clip patch/merge factor overflow")?;
        let factor_pixels = factor
            .checked_mul(factor)
            .ok_or("clip patch/merge pixel factor overflow")?;
        let default_min = factor_pixels
            .checked_mul(8)
            .ok_or("clip minimum pixel count overflow")?;
        let default_max = factor_pixels
            .checked_mul(4096)
            .ok_or("clip maximum pixel count overflow")?;
        let image_min_pixels = source
            .metadata("clip.vision.image_min_pixels")
            .and_then(MetaValue::to_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "clip.vision.image_min_pixels does not fit usize")?
            .unwrap_or(default_min);
        let image_max_pixels = source
            .metadata("clip.vision.image_max_pixels")
            .and_then(MetaValue::to_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "clip.vision.image_max_pixels does not fit usize")?
            .unwrap_or(default_max);
        if image_min_pixels == 0 || image_min_pixels > image_max_pixels {
            return Err("Invalid clip vision pixel limits".into());
        }
        let video_min_pixels = source
            .metadata("clip.vision.video_min_pixels")
            .and_then(MetaValue::to_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "clip.vision.video_min_pixels does not fit usize")?
            .unwrap_or(image_min_pixels);
        let video_max_pixels = source
            .metadata("clip.vision.video_max_pixels")
            .and_then(MetaValue::to_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "clip.vision.video_max_pixels does not fit usize")?
            .unwrap_or(image_max_pixels);
        if video_min_pixels == 0 || video_min_pixels > video_max_pixels {
            return Err("Invalid clip vision video pixel limits".into());
        }
        let eps = get_f32("clip.vision.attention.layer_norm_epsilon")?;
        let use_gelu = get_bool("clip.use_gelu");

        let image_mean = match source.metadata("clip.vision.image_mean") {
            Some(MetaValue::Array(_, vals)) => {
                let m: Vec<f32> = vals.iter()
                    .filter_map(|v| v.to_f64().map(|x| x as f32))
                    .collect();
                if m.len() == 3 {
                    [m[0], m[1], m[2]]
                } else {
                    [0.48145466, 0.4578275, 0.40821073]
                }
            }
            _ => {
                [0.48145466, 0.4578275, 0.40821073]
            }
        };

        let image_std = match source.metadata("clip.vision.image_std") {
            Some(MetaValue::Array(_, vals)) => {
                let s: Vec<f32> = vals.iter()
                    .filter_map(|v| v.to_f64().map(|x| x as f32))
                    .collect();
                if s.len() == 3 {
                    [s[0], s[1], s[2]]
                } else {
                    [0.26862954, 0.26130258, 0.27577711]
                }
            }
            _ => {
                [0.26862954, 0.26130258, 0.27577711]
            }
        };

        let has_deepstack_layers = match source.metadata("clip.vision.is_deepstack_layers") {
            Some(MetaValue::Array(_, vals)) => {
                vals.iter()
                    .filter_map(|v| match v { MetaValue::Bool(b) => Some(*b), _ => None })
                    .collect()
            }
            _ => vec![false; n_layer],
        };

        Ok(Self {
            projection_dim,
            image_size,
            patch_size,
            n_embd,
            n_ff,
            n_layer,
            n_head,
            spatial_merge_size,
            image_min_pixels,
            image_max_pixels,
            video_min_pixels,
            video_max_pixels,
            eps,
            use_gelu,
            image_mean,
            image_std,
            has_deepstack_layers,
        })
    }

    pub fn d_head(&self) -> usize {
        self.n_embd / self.n_head
    }

    pub fn n_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    pub fn n_patches(&self) -> usize {
        let ps = self.n_patches_per_side();
        ps * ps
    }

    pub fn n_output_tokens(&self) -> usize {
        let merge = self.spatial_merge_size;
        let ps = self.n_patches_per_side();
        (ps / merge) * (ps / merge)
    }
}
