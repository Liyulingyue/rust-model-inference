use super::json_metadata;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions, Tokenizer};
use crate::models::llama::trunk::LlamaSession;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Config {
    image_token_id: u32,
}

fn retrieval_prompt(text: &str, image: bool) -> String {
    if text.starts_with("<|im_start|>system\n") {
        return text.into();
    }
    let (system, user) = if image {
        let media = "<|vision_start|><|image_pad|><|vision_end|>";
        if text.is_empty() {
            ("Represent the given text-rich image, focusing on extracting and interpreting both its rich text content and visual features.", media.to_string())
        } else {
            (
                "Represent the given image with the given query.",
                format!("Query:{text}{media}"),
            )
        }
    } else {
        ("Represent the given text.", text.to_string())
    };
    format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>")
}

pub(super) fn compute(
    source: &dyn TensorSource,
    prompt: &str,
    image: Option<&Path>,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    if crate::app::arch_of(source) != "qwen2vl" {
        return Err("BGE Screenshot requires qwen2vl architecture".into());
    }
    let cfg: Config = json_metadata(source, "bge.config")?;
    let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())?;
    let mut ids = tokenizer.encode(
        &retrieval_prompt(prompt, image.is_some()),
        EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    );
    let media = image
        .map(|path| crate::app::text::encode_qwen35_image(source, path, pool.n_threads()))
        .transpose()?;
    if let Some((grid, _)) = &media {
        if ids.iter().filter(|id| **id == cfg.image_token_id).count() != 1 {
            return Err("BGE Screenshot requires one image placeholder".into());
        }
        let mut expanded = Vec::new();
        for id in ids {
            if id == cfg.image_token_id {
                expanded.extend(std::iter::repeat_n(id, grid.token_count()));
            } else {
                expanded.push(id);
            }
        }
        ids = expanded;
    } else if ids.contains(&cfg.image_token_id) {
        return Err("BGE Screenshot image placeholder requires an image".into());
    }
    let eos = source
        .metadata("tokenizer.ggml.tokens")
        .and_then(|value| {
            if let crate::core::tensor::MetaValue::Array(_, tokens) = value {
                tokens
                    .iter()
                    .position(|v| v.to_string_val() == Some("<|endoftext|>"))
            } else {
                None
            }
        })
        .ok_or("BGE Screenshot tokenizer has no <|endoftext|>")? as u32;
    if ids.len() > 3072 {
        ids.truncate(3072);
        *ids.last_mut().ok_or("empty Screenshot input")? = eos;
    }
    if ids.last() != Some(&eos) {
        return Err("BGE Screenshot prompt must end with <|endoftext|>".into());
    }
    let grids = media
        .as_ref()
        .map(|(grid, _)| vec![(grid.grid_h, grid.grid_w)])
        .unwrap_or_default();
    let mut positions =
        crate::app::text::build_qwen3_media_positions(&ids, cfg.image_token_id, &grids)?;
    // HF Qwen2.5-VL uses the same position on all three axes for text.
    for (id, position) in ids.iter().zip(&mut positions) {
        if *id != cfg.image_token_id {
            *position = [position[0], position[0], position[0], 0];
        }
    }
    let sections = crate::core::loader::qwen3_arch_knobs(source)?
        .rope_sections
        .ok_or("missing Screenshot M-RoPE sections")?;
    let mut session = LlamaSession::from_source_with_max_rows(
        source,
        pool.n_threads(),
        crate::app::KvFormat::F32,
        ids.len(),
        1,
    )?;
    let width = session.config.n_embd;
    let mut row = vec![0.0; width];
    let mut index = 0;
    for (id, position) in ids.iter().zip(positions) {
        if *id == cfg.image_token_id {
            let values = &media.as_ref().ok_or("missing Screenshot image")?.1;
            let start = index * width;
            row.copy_from_slice(
                values
                    .get(start..start + width)
                    .ok_or("Screenshot image rows were truncated or have the wrong width")?,
            );
            index += 1;
        } else {
            if *id as usize >= session.config.vocab {
                return Err("Screenshot token exceeds vocabulary".into());
            }
            crate::ops::embedding_lookup(
                session.weights.embd_weight,
                *id,
                width,
                session.weights.embd_type,
                &mut row,
            );
        }
        session.forward_embedding_mrope(&row, position, sections)?;
    }
    #[cfg(feature = "parity-trace")]
    if crate::parity_trace::enabled("embedding.tokens") {
        crate::parity_trace::token_ids("embedding.tokens", &ids).map_err(|e| e.to_string())?;
    }
    let mut output = session.scratch.normed[..width].to_vec();
    super::trace("embedding.pooled", &output)?;
    crate::models::qwen3::embedding::l2_normalize_embedding(&mut output)?;
    super::trace("embedding.final", &output)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::retrieval_prompt;
    #[test]
    fn screenshot_uses_official_candidate_and_query_templates() {
        let text = retrieval_prompt("hello", false);
        assert!(text.starts_with("<|im_start|>system\nRepresent the given text."));
        assert!(text.ends_with("<|im_start|>assistant\n<|endoftext|>"));
        assert!(retrieval_prompt("", true).contains("Represent the given text-rich image,"));
        assert!(retrieval_prompt("find", true)
            .contains("Query:find<|vision_start|><|image_pad|><|vision_end|>"));
        assert_eq!(retrieval_prompt(&text, false), text);
    }
}
