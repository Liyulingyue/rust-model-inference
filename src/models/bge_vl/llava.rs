use super::{clip, json_metadata, Linear};
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{load_tokenizer, EncodeOptions};
use crate::models::llama::trunk::LlamaSession;
use crate::models::qwen3::embedding::l2_normalize_embedding;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Config {
    vision_config: clip::Config,
    image_token_index: u32,
    vision_feature_layer: i32,
    image_grid_pinpoints: Vec<[usize; 2]>,
}

pub(super) fn compute(
    source: &dyn TensorSource,
    prompt: &str,
    image: Option<&Path>,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    let cfg: Config = json_metadata(source, "bge.config")?;
    let vision_rows = if let Some(path) = image {
        image_features(source, path, &cfg, pool)?
    } else {
        Vec::new()
    };
    let prompt = if prompt.starts_with("[INST]") {
        prompt.to_string()
    } else {
        let mut p = "[INST] ".to_string();
        if !prompt.is_empty() {
            p.push_str(&format!(" {prompt} \n"));
        }
        if image.is_some() {
            p.push_str(" <image>");
        }
        p.push_str(" [/INST]");
        p
    };
    let tokenizer = load_tokenizer(|k| source.metadata(k).cloned())?;
    let ids = tokenizer.encode(
        &prompt,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    let image_count = ids
        .iter()
        .filter(|id| **id == cfg.image_token_index)
        .count();
    if image_count != usize::from(image.is_some()) {
        return Err(
            "BGE LLaVA-NeXT requires exactly one <image> token for the supplied image".into(),
        );
    }
    let width = source
        .metadata("llama.embedding_length")
        .and_then(|m| m.to_u64())
        .ok_or("missing decoder width")? as usize;
    let rows = ids.len() - image_count + vision_rows.len() / width;
    if rows == 0 {
        return Err("BGE LLaVA-NeXT input is empty".into());
    }
    let mut session =
        LlamaSession::from_source(source, pool.n_threads(), crate::app::KvFormat::F32, rows)?;
    let mut row = vec![0.0; width];
    // shortcut: image tokens use single-row CPU prefill; batch this path when retrieval latency is required.
    for id in ids {
        if id == cfg.image_token_index {
            for feature in vision_rows.chunks_exact(width) {
                session.forward_embedding(feature)?;
            }
        } else {
            if id as usize >= session.config.vocab {
                return Err("LLaVA token exceeds vocabulary".into());
            }
            crate::ops::embedding_lookup(
                session.weights.embd_weight,
                id,
                width,
                session.weights.embd_type,
                &mut row,
            );
            session.forward_embedding(&row)?;
        }
    }
    let mut embedding = session.scratch.normed[..width].to_vec();
    super::trace("embedding.pooled", &embedding)?;
    l2_normalize_embedding(&mut embedding)?;
    super::trace("embedding.final", &embedding)?;
    Ok(embedding)
}

fn image_features(
    source: &dyn TensorSource,
    path: &Path,
    cfg: &Config,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    let vision = clip::Vision::load(
        source,
        "vision_tower.vision_model",
        cfg.vision_config.clone(),
    )?;
    let side = vision.cfg.image_size;
    let patches = side / vision.cfg.patch_size;
    let v = vision.cfg.hidden_size;
    let layer = if cfg.vision_feature_layer < 0 {
        vision.cfg.num_hidden_layers as i32 + 1 + cfg.vision_feature_layer
    } else {
        cfg.vision_feature_layer
    };
    if layer < 0 || layer as usize > vision.cfg.num_hidden_layers {
        return Err("invalid LLaVA vision feature layer".into());
    }
    if !cfg.image_grid_pinpoints.contains(&[side * 2, side * 2]) {
        return Err("BGE LLaVA-NeXT requires the official square anyres grid".into());
    }
    let original = crate::app::media::decode_image(path)?.to_rgb8();
    // Official BGE data_process first stretches every image to 512 by 512.
    let resized = crate::models::gemma4::vision::resize_bicubic_pillow(
        original.as_raw(),
        original.width() as usize,
        original.height() as usize,
        512,
        512,
    )?;
    let resized = image::DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(512, 512, resized).ok_or("invalid resized image")?,
    );
    let preprocess: serde_json::Value = json_metadata(source, "bge.preprocessor")?;
    let array = |key: &str| -> Result<[f32; 3], String> {
        let a = preprocess[key]
            .as_array()
            .ok_or("missing LLaVA image normalization")?;
        if a.len() != 3 {
            return Err("invalid image normalization shape".into());
        }
        let mut out = [0.; 3];
        for (x, v) in out.iter_mut().zip(a) {
            *x = v.as_f64().ok_or("invalid image normalization")? as f32;
        }
        Ok(out)
    };
    let mean = array("image_mean")?;
    let std = array("image_std")?;
    let global = crate::app::media::normalize_resized_image(&resized, side, side, &mean, &std)?;
    let canvas =
        crate::app::media::normalize_resized_image(&resized, side * 2, side * 2, &mean, &std)?;
    let width = source
        .metadata("llama.embedding_length")
        .and_then(|m| m.to_u64())
        .ok_or("missing decoder width")? as usize;
    let first = Linear::load(source, "multi_modal_projector.linear_1", v, width, true)?;
    let second = Linear::load(source, "multi_modal_projector.linear_2", width, width, true)?;
    let encode = |pixels: &[f32]| -> Result<Vec<f32>, String> {
        let hidden = vision.encode(pixels, layer as usize, pool)?;
        let mut hidden = first.apply(&hidden[v..], pool)?;
        crate::ops::gelu_erf_inplace(&mut hidden);
        second.apply(&hidden, pool)
    };
    let mut output = encode(&global)?;
    let mut tiles = Vec::with_capacity(4);
    for ty in 0..2 {
        for tx in 0..2 {
            let mut tile = Vec::with_capacity(side * side * 3);
            for y in ty * side..(ty + 1) * side {
                tile.extend_from_slice(
                    &canvas[(y * side * 2 + tx * side) * 3..(y * side * 2 + (tx + 1) * side) * 3],
                );
            }
            tiles.push(encode(&tile)?);
        }
    }
    let newline = load_f32_tensor(source, "image_newline", &[width as u64])?;
    for y in 0..patches * 2 {
        for x in 0..patches * 2 {
            let tile = (y / patches) * 2 + x / patches;
            let token = (y % patches) * patches + x % patches;
            output.extend_from_slice(&tiles[tile][token * width..(token + 1) * width]);
        }
        output.extend_from_slice(&newline);
    }
    Ok(output)
}
