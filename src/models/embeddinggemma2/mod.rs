//! EmbeddingGemma-2 (Google) — multimodal encoder (text + vision + audio).
//!
//! `arch = "gemma4-embedding"` — a single-GGUF multi-section encoder that
//! produces 768-dim embeddings via mean pooling + a learned projection.
//! The text backbone is `gemma4-text` (bidirectional + PLE block + 5-norm
//! sandwich + sliding window attention with per-layer head_dim/kv_heads).
//! The vision tower is `gemma4-vision` (SigLIP-style + axial RoPE, 16
//! layers). The audio tower is `gemma4-audio` (Conformer, 12 layers).
//!
//! Reference for tensor naming: `tools/converter/embeddinggemma2/convert_embeddinggemma2.py`.
//! End-to-end forward path lives in this module; convert metadata + tensor
//! contract tests live in `tests/embeddinggemma_2.rs`.

pub mod audio;
pub mod text;
pub mod vision;

use crate::app::cli::{EmbeddingOutput, KvFormat};
use crate::core::tensor::TensorSource;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use std::time::Instant;
use std::{path::Path, sync::Arc};

pub use audio::{EmbeddingGemmaAudioConfig, EmbeddingGemmaAudioModel};
pub use text::{EmbeddingGemmaTextConfig, EmbeddingGemmaTextModel};
pub use vision::{EmbeddingGemmaVisionConfig, EmbeddingGemmaVisionModel};

pub fn compute_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    _n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    let token_ids = tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    if token_ids.is_empty() {
        return Err("Embedding input produced no tokens".into());
    }
    EmbeddingGemmaTextModel::from_source(source)?.forward(&token_ids)
}

/// Encode a single image or audio file, optionally followed by text. All
/// components live in the same GGUF; an external mmproj is unnecessary.
pub fn compute_media_embedding(
    source: Arc<dyn TensorSource>,
    image: Option<&Path>,
    audio: Option<&Path>,
    prompt: &str,
) -> Result<Vec<f32>, String> {
    if usize::from(image.is_some()) + usize::from(audio.is_some()) != 1 {
        return Err("EmbeddingGemma requires exactly one image or audio input".into());
    }
    let (kind, soft_tokens) = if let Some(path) = image {
        let model = EmbeddingGemmaVisionModel::from_source(&source)?;
        let rgb = crate::app::media::decode_image(path)?.to_rgb8();
        let (width, height) = image_size(
            rgb.width() as usize,
            rgb.height() as usize,
            model.config.patch_size,
            model.config.spatial_merge_size,
            model.config.default_output_length,
        )?;
        let resized = crate::models::gemma4::vision::resize_bicubic_pillow(
            rgb.as_raw(),
            rgb.width() as usize,
            rgb.height() as usize,
            width,
            height,
        )?;
        let pixels: Vec<f32> = resized.iter().map(|&v| v as f32 / 255.0).collect();
        ("image", model.encode(&pixels, height, width)?)
    } else {
        let samples = crate::app::media::decode_audio(audio.unwrap())?;
        let pcm: Vec<i16> = samples
            .iter()
            .map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
            .collect();
        (
            "audio",
            EmbeddingGemmaAudioModel::from_source(&source)?.encode(&pcm)?,
        )
    };
    let get_id = |key: &str| -> Result<u32, String> {
        source
            .metadata(key)
            .and_then(|v| v.to_u64())
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("Missing token ID: {key}"))
    };
    let media_id = get_id(&format!("gemma4-embedding.{kind}_token_id"))?;
    let (start, end) = if kind == "image" {
        ("boi", "eoi")
    } else {
        ("boa", "eoa")
    };
    let start_id = get_id(&format!("gemma4-embedding.{start}_token_id"))?;
    let end_id = get_id(&format!("gemma4-embedding.{end}_token_id"))?;
    let model = EmbeddingGemmaTextModel::from_source(source.as_ref())?;
    let rows = soft_tokens.len() / model.config.n_embd;
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())?;
    let mut tokens = tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: true,
            parse_special: true,
        },
    );
    if tokens.contains(&media_id) {
        return Err(
            "media placeholders are inserted automatically; omit them from --prompt".into(),
        );
    }
    let bos = get_id("tokenizer.ggml.bos_token_id")?;
    let offset = usize::from(tokens.first() == Some(&bos));
    tokens.splice(
        offset..offset,
        std::iter::once(start_id)
            .chain(std::iter::repeat_n(media_id, rows))
            .chain(std::iter::once(end_id)),
    );
    model.forward_with_soft_tokens(&tokens, media_id, &soft_tokens)
}

// HF Gemma4ImageProcessor's aspect-ratio preserving patch budget.
fn image_size(
    width: usize,
    height: usize,
    patch: usize,
    merge: usize,
    budget: usize,
) -> Result<(usize, usize), String> {
    if [width, height, patch, merge, budget].contains(&0) {
        return Err("invalid EmbeddingGemma image dimensions or patch budget".into());
    }
    let align = patch * merge;
    let factor = ((budget as f64 * (align as f64).powi(2)) / (width as f64 * height as f64)).sqrt();
    let mut w = (width as f64 * factor / align as f64).floor() as usize * align;
    let mut h = (height as f64 * factor / align as f64).floor() as usize * align;
    if h == 0 {
        h = align;
        w = (width / height).min(budget) * align;
    } else if w == 0 {
        w = align;
        h = (height / width).min(budget) * align;
    }
    Ok((w, h))
}

#[cfg(test)]
mod tests {
    use super::image_size;
    #[test]
    fn image_resize_matches_hf_patch_budget() {
        assert_eq!(image_size(100, 100, 16, 3, 280).unwrap(), (768, 768));
        assert_eq!(image_size(640, 480, 16, 3, 280).unwrap(), (912, 672));
        assert_eq!(image_size(100000, 1, 16, 3, 280).unwrap(), (13440, 48));
        assert!(image_size(0, 1, 16, 3, 280).is_err());
    }
}

pub fn run_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
    _kv_format: KvFormat,
    output: EmbeddingOutput,
) {
    let started = Instant::now();
    let embedding = match compute_embedding(source, prompt, n_threads_arg) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    match output {
        EmbeddingOutput::Summary => {
            println!(
                "Embedding ({} dims, {}ms):",
                embedding.len(),
                started.elapsed().as_millis()
            );
            for value in embedding.iter().take(8) {
                print!("{value:.9} ");
            }
            if embedding.len() > 8 {
                print!("... ");
                for value in &embedding[embedding.len() - 4..] {
                    print!("{value:.9} ");
                }
            }
            println!();
        }
        EmbeddingOutput::Raw => {
            print!("embedding_raw:");
            for value in embedding {
                print!(" {value:.9}");
            }
            println!();
        }
    }
}
