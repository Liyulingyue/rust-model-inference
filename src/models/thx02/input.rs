use super::{protocol, weights};
use crate::core::{
    tensor::{MetaValue, TensorSource},
    thread_pool::ComputePool,
};
use crate::models::qwen35::{
    build_qwen35_positions,
    vision::{VisionEncoder, VisionGrid, VisionScratchpad},
};
use base64::Engine as _;
use serde_json::Value;
use std::{io::Cursor, sync::Arc};

pub(super) struct Encoder {
    pub tokenizer: tokenizers::Tokenizer,
    pub codes: Vec<String>,
    pub escapes: Vec<String>,
    pub marker: u32,
    pub image: u32,
}

impl Encoder {
    pub fn new(source: &dyn TensorSource) -> Result<Self, String> {
        let raw = weights::string(source, "thx02.tokenizer_json")?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(raw.as_bytes())
            .map_err(|e| format!("THX-02 tokenizer: {e}"))?;
        let MetaValue::Array(_, codes) = source
            .metadata("thx02.codes")
            .ok_or("Missing THX-02 codebook")?
        else {
            return Err("Invalid THX-02 codebook".into());
        };
        let codes = codes
            .iter()
            .map(|v| {
                v.to_string_val()
                    .map(str::to_owned)
                    .ok_or("Invalid codebook entry".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if codes.len() != 255 {
            return Err("THX-02 codebook must have 255 entries".into());
        }
        let MetaValue::Array(_, ids) = source
            .metadata("thx02.code_token_ids")
            .ok_or("Missing THX-02 code token IDs")?
        else {
            return Err("Invalid THX-02 code token IDs".into());
        };
        if ids.len() != 255
            || codes.iter().zip(ids).any(|(code, id)| {
                tokenizer.encode(code.as_str(), false).map_or(true, |e| {
                    e.get_ids().len() != 1 || Some(e.get_ids()[0] as u64) != id.to_u64()
                })
            })
        {
            return Err("THX-02 codebook does not match tokenizer".into());
        }
        // HF's configured special-token map differs from added_tokens[].special.
        let MetaValue::Array(_, values) = source
            .metadata("thx02.escape_tokens")
            .ok_or("Missing THX-02 configured escape tokens")?
        else {
            return Err("Invalid THX-02 escape tokens".into());
        };
        let mut escapes = values
            .iter()
            .map(|v| {
                v.to_string_val()
                    .map(str::to_owned)
                    .ok_or("Invalid THX-02 escape token".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !escapes.iter().any(|s| s == protocol::MARKER) {
            return Err("THX-02 configured escapes must include the readout marker".into());
        }
        escapes.sort_by_key(|s| std::cmp::Reverse(s.len()));
        escapes.dedup();
        let marker = tokenizer
            .token_to_id(protocol::MARKER)
            .ok_or("THX-02 missing readout marker")?;
        let image = tokenizer
            .token_to_id("<|image_pad|>")
            .ok_or("THX-02 missing image marker")?;
        Ok(Self {
            tokenizer,
            codes,
            escapes,
            marker,
            image,
        })
    }

    pub fn encode(
        &self,
        request: &Value,
        grids: &[VisionGrid],
        thinking: bool,
    ) -> Result<Encoded, String> {
        let text = protocol::prompt(request, &self.codes, &self.escapes, thinking)?;
        let tokens = self
            .tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| format!("THX-02 encode: {e}"))?;
        let mut ids = Vec::new();
        let mut index = 0;
        for &id in tokens.get_ids() {
            if id == self.image {
                let grid = grids.get(index).ok_or("THX-02 image marker has no image")?;
                ids.extend(std::iter::repeat_n(id, grid.token_count()));
                index += 1;
            } else {
                ids.push(id);
            }
        }
        if index != grids.len() {
            return Err("THX-02 image count mismatch".into());
        }
        let (positions, next) = build_qwen35_positions(&ids, Some(self.image), grids)?;
        let markers = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == self.marker).then_some(i))
            .collect::<Vec<_>>();
        if markers.len()
            != if thinking {
                0
            } else {
                request["questions"].as_object().unwrap().len()
            }
        {
            return Err("THX-02 readout marker count mismatch".into());
        }
        Ok(Encoded {
            ids,
            positions,
            markers,
            next,
        })
    }
}

pub(super) struct Encoded {
    pub ids: Vec<u32>,
    pub positions: Vec<[usize; 4]>,
    pub markers: Vec<usize>,
    pub next: usize,
}

pub(super) fn open_images(request: &Value) -> Result<Vec<image::DynamicImage>, String> {
    request["images"].as_array().into_iter().flatten().map(|value| {
        let value = value.as_str().ok_or("THX-02 model image must be a data URL")?;
        let (head, data) = value.split_once(',').ok_or("Invalid THX-02 image data URL")?;
        if !head.starts_with("data:image/") || !head.ends_with(";base64") || data.len() > 48 * 1024 * 1024 {
            return Err("THX-02 model images must be base64 data URLs of at most 32 MiB; resolve files at the input boundary".into());
        }
        let bytes = base64::engine::general_purpose::STANDARD.decode(data)
            .map_err(|e| format!("Invalid THX-02 image base64: {e}"))?;
        image::ImageReader::new(Cursor::new(bytes)).with_guessed_format()
            .map_err(|e| e.to_string())?.decode().map_err(|e| format!("THX-02 image decode: {e}"))
    }).collect()
}

#[derive(Default)]
pub(super) struct Visual {
    pub grids: Vec<VisionGrid>,
    pub embeddings: Vec<f32>,
}

fn resize_grid(width: usize, height: usize) -> Result<VisionGrid, String> {
    if width == 0 || height == 0 || width.max(height) as f64 / width.min(height) as f64 > 200. {
        return Err("THX-02 image dimensions must be nonzero with aspect ratio at most 200".into());
    }
    let pixels = width
        .checked_mul(height)
        .ok_or("THX-02 image size overflow")?;
    // HF uses ties-to-even rounding; the shared llama.cpp resize rounds ties away from zero.
    let mut w = (width as f64 / 32.).round_ties_even() as usize * 32;
    let mut h = (height as f64 / 32.).round_ties_even() as usize * 32;
    let aligned = w
        .checked_mul(h)
        .ok_or("THX-02 resized image size overflow")?;
    if aligned > 1_638_400 {
        let beta = (pixels as f64 / 1_638_400.).sqrt();
        w = ((width as f64 / beta / 32.).floor() as usize).max(1) * 32;
        h = ((height as f64 / beta / 32.).floor() as usize).max(1) * 32;
    } else if aligned < 65_536 {
        let beta = (65_536. / pixels as f64).sqrt();
        w = (width as f64 * beta / 32.).ceil() as usize * 32;
        h = (height as f64 * beta / 32.).ceil() as usize * 32;
    }
    VisionGrid::from_image_size(1, w, h, 16, 2)
}

impl Visual {
    pub fn encode(
        source: &dyn TensorSource,
        images: &[image::DynamicImage],
        pool: &Arc<ComputePool>,
    ) -> Result<Self, String> {
        if images.is_empty() {
            return Ok(Self::default());
        }
        let encoder = VisionEncoder::from_source(source, Arc::clone(pool))?;
        if encoder.config.n_embd != 1024
            || encoder.config.n_layer != 24
            || encoder.config.patch_size != 16
            || encoder.config.spatial_merge_size != 2
        {
            return Err("THX-02 requires its published 24-layer vision component".into());
        }
        let mut scratch = VisionScratchpad::new(&encoder.config);
        let mut visual = Self::default();
        for image in images {
            let rgb = image.to_rgb8();
            let grid = resize_grid(rgb.width() as usize, rgb.height() as usize)?;
            let resized = crate::models::gemma4::vision::resize_bicubic_pillow(
                rgb.as_raw(),
                rgb.width() as usize,
                rgb.height() as usize,
                grid.image_width(),
                grid.image_height(),
            )?;
            let pixels = resized
                .into_iter()
                .map(|x| (x as f32 / 255. - 0.5) / 0.5)
                .collect::<Vec<_>>();
            let actual = encoder.encode_image_f32_gelu(
                &pixels,
                grid.image_width(),
                grid.image_height(),
                &mut scratch,
                pool,
            )?;
            if scratch.projected.len() != actual.token_count() * 2560
                || scratch.projected.iter().any(|v| !v.is_finite())
            {
                return Err("THX-02 invalid projector output".into());
            }
            visual.grids.push(actual);
            visual.embeddings.extend_from_slice(&scratch.projected);
        }
        super::forward::trace(
            "thx02.visual",
            None,
            &[visual.embeddings.len() / 2560, 2560],
            &visual.embeddings,
        );
        Ok(visual)
    }

    pub fn inject(&self, ids: &[u32], pad: u32, embeddings: &mut [f32]) -> Result<(), String> {
        crate::models::qwen3::vision::media::inject_qwen_media_embeddings(
            ids,
            pad,
            embeddings,
            &self.embeddings,
            &[],
            2560,
        )
        .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_minimum_pixels_and_python_tie_rounding_match_processor() {
        for (width, height, expected) in [
            (230, 129, (352, 192)),
            (208, 400, (192, 384)),
            (1920, 1080, (1696, 960)),
        ] {
            let grid = resize_grid(width, height).unwrap();
            assert_eq!((grid.image_width(), grid.image_height()), expected);
        }
        assert_eq!(resize_grid(230, 129).unwrap().token_count(), 66);
        assert!(resize_grid(1, 201).is_err());
    }
}
