//! `MiniCpmV47Config` — hyperparameters for the `minicpmv4_7` projector
//! (`clip.projector_type = "minicpmv4_7"`).
//!
//! Cross-checked against the pinned llama.cpp oracle, the merge commit of
//! upstream PR #29416 `[Model] Support MiniCPM-V 4.7`
//! (`69f201a2051ea9b9e9b50c3cc56afd9e2ae64414`): `tools/mtmd/models/minicpmv.cpp`
//! (`clip_graph_minicpmv4_6::build`, shared by 4.6/4.7),
//! `tools/mtmd/clip.cpp` (hparams) and `tools/mtmd/mtmd.cpp` (slicing).
//!
//! Shape of the graph, for one tile:
//!
//! ```text
//! patch_embd (14x14x3 -> 1152) + position_embd
//!   ViT blocks 0..=insert_layer_id            (insert_layer_id = wa_layer_indexes[0])
//!   vit_merger: window attention (4x4 blocks) + residual
//!   vit_merger: 2x2 mean/concat downsample, LN, GELU MLP, + mean residual
//!   ViT blocks insert_layer_id+1 .. block_count
//!   post_ln
//!   final merger: 2x2 concat downsample, mm.input_norm, GELU_erf MLP
//!   v.tok_embd_sep suffix rows
//! ```
//!
//! Two 2x2 merges give `scale_factor = 4` tokens per side. Note the tower uses
//! **LayerNorm**, not RMSNorm (`NORM_TYPE_NORMAL`), with `eps = 1e-6`, plain
//! GELU in the tower and the vit_merger MLP, and erf-GELU only in the final
//! projector MLP.

use crate::core::tensor::{MetaValue, TensorSource};

pub const PROJECTOR_TYPE: &str = "minicpmv4_7";

#[derive(Debug, Clone, PartialEq)]
pub struct MiniCpmV47Config {
    /// `clip.vision.image_size` — the nominal square tile edge (448).
    pub image_size: usize,
    /// `clip.vision.patch_size` (14).
    pub patch_size: usize,
    /// `clip.vision.embedding_length` (1152) — the ViT width.
    pub n_embd: usize,
    /// `clip.vision.feed_forward_length` (4304).
    pub n_ff: usize,
    /// `clip.vision.block_count` (27).
    pub n_layer: usize,
    /// `clip.vision.attention.head_count` (16).
    pub n_head: usize,
    /// `clip.vision.projection_dim` (1024) — the text tower's `n_embd`.
    pub n_embd_text: usize,
    /// `clip.vision.attention.layer_norm_epsilon` (1e-6).
    pub eps: f32,
    /// `clip.vision.projector.scale_factor` (4) — two 2x2 merges.
    pub n_merge: usize,
    /// `clip.vision.max_slice_nums` (9).
    pub max_slice_nums: usize,
    /// `clip.vision.wa_layer_indexes[0]` (6) — where `vit_merger` replaces the
    /// plain ViT block. Borrowed by the converter from MiniCPM's
    /// weight-aware-attention layer index.
    pub insert_layer_id: usize,
    /// `clip.vision.image_mean` / `clip.vision.image_std`.
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// `d_head = n_embd / n_head`.
    pub d_head: usize,
    /// `1 / sqrt(d_head)`, matching ggml's `kq_scale`.
    pub kq_scale: f32,
}

impl MiniCpmV47Config {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let u32_at = |key: &str| -> Result<u32, String> {
            match source.metadata(key) {
                Some(value) => value
                    .to_u64()
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or_else(|| format!("{key}: expected unsigned int, got {value:?}")),
                None => Err(format!("Missing metadata: {key}")),
            }
        };
        let f32_at = |key: &str| -> Result<f32, String> {
            match source.metadata(key) {
                Some(value) => value
                    .to_f64()
                    .map(|v| v as f32)
                    .ok_or_else(|| format!("{key}: expected float, got {value:?}")),
                None => Err(format!("Missing metadata: {key}")),
            }
        };

        let declared = match source.metadata("clip.projector_type") {
            Some(MetaValue::String(value)) => value.as_str(),
            Some(other) => {
                return Err(format!(
                    "clip.projector_type: expected string, got {other:?}"
                ))
            }
            None => return Err("Missing metadata: clip.projector_type".into()),
        };
        if declared != PROJECTOR_TYPE {
            return Err(format!(
                "clip.projector_type is {declared:?}, expected {PROJECTOR_TYPE:?}"
            ));
        }

        let n_embd = u32_at("clip.vision.embedding_length")? as usize;
        let n_head = u32_at("clip.vision.attention.head_count")? as usize;
        if n_head == 0 || n_embd % n_head != 0 {
            return Err(format!(
                "Invalid vision head shape: embedding_length={n_embd}, head_count={n_head}"
            ));
        }
        let n_layer = u32_at("clip.vision.block_count")? as usize;
        let n_merge = u32_at("clip.vision.projector.scale_factor")? as usize;
        if n_merge != 2 && n_merge != 4 {
            return Err(format!(
                "clip.vision.projector.scale_factor must be 2 or 4, got {n_merge}"
            ));
        }
        let insert_layer_id = match source.metadata("clip.vision.wa_layer_indexes") {
            Some(MetaValue::Array(_, values)) if !values.is_empty() => match values[0].to_u64() {
                Some(value) if value < n_layer as u64 => value as usize,
                _ => {
                    return Err(format!(
                        "clip.vision.wa_layer_indexes[0] out of range: {:?}",
                        values[0]
                    ))
                }
            },
            // The converter always writes it; absence would otherwise silently
            // fall back to a different insertion point.
            _ => return Err("Missing or empty clip.vision.wa_layer_indexes".into()),
        };

        let vec3 = |key: &str| -> Result<[f32; 3], String> {
            let Some(MetaValue::Array(_, values)) = source.metadata(key) else {
                return Err(format!("{key}: expected array of 3 floats"));
            };
            if values.len() != 3 {
                return Err(format!("{key}: expected 3 entries, got {}", values.len()));
            }
            let mut out = [0.0f32; 3];
            for (slot, value) in out.iter_mut().zip(values.iter()) {
                *slot = value
                    .to_f64()
                    .map(|v| v as f32)
                    .ok_or_else(|| format!("{key}: expected float, got {value:?}"))?;
            }
            Ok(out)
        };

        let d_head = n_embd / n_head;
        Ok(Self {
            image_size: u32_at("clip.vision.image_size")? as usize,
            patch_size: u32_at("clip.vision.patch_size")? as usize,
            n_embd,
            n_ff: u32_at("clip.vision.feed_forward_length")? as usize,
            n_layer,
            n_head,
            n_embd_text: u32_at("clip.vision.projection_dim")? as usize,
            eps: f32_at("clip.vision.attention.layer_norm_epsilon")?,
            n_merge,
            max_slice_nums: u32_at("clip.vision.max_slice_nums")? as usize,
            insert_layer_id,
            image_mean: vec3("clip.vision.image_mean")?,
            image_std: vec3("clip.vision.image_std")?,
            d_head,
            kq_scale: 1.0 / (d_head as f32).sqrt(),
        })
    }

    /// Tiles are aligned to `patch_size * 4`, independent of `n_merge`: the
    /// 2x2 `vit_merger` slot stays in the grid even in 4x mode.
    pub fn slice_align(&self) -> usize {
        self.patch_size * 4
    }

    /// Tokens one tile contributes to the text sequence.
    ///
    /// `n_out = (n_patches_y/2/2) * (n_patches_x/2/2)` for `n_merge == 4`,
    /// `(n_patches_y/2) * (n_patches_x/2)` for `n_merge == 2`, plus the
    /// separator rows the tile appends.
    pub fn output_tokens(&self, nx: usize, ny: usize, suffix_rows: usize) -> usize {
        let (hx, hy) = (nx / 2, ny / 2);
        let out = if self.n_merge == 2 {
            hx * hy
        } else {
            (hx / 2) * (hy / 2)
        };
        out + suffix_rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_counts_follow_the_two_merges() {
        let cfg = MiniCpmV47Config {
            image_size: 448,
            patch_size: 14,
            n_embd: 1152,
            n_ff: 4304,
            n_layer: 27,
            n_head: 16,
            n_embd_text: 1024,
            eps: 1e-6,
            n_merge: 4,
            max_slice_nums: 9,
            insert_layer_id: 6,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            d_head: 72,
            kq_scale: 1.0 / 72.0f32.sqrt(),
        };
        // 448/14 = 32 patches a side -> 32*32 -> 16*16 -> 8*8 = 64 tokens.
        assert_eq!(cfg.output_tokens(32, 32, 0), 64);
        assert_eq!(cfg.output_tokens(32, 32, 2), 66);
        assert_eq!(cfg.slice_align(), 56);
    }
}
