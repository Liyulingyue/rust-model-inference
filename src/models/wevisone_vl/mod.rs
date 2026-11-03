//! Tencent WeVisOne-VL: Qwen3-VL backbone and the released FP32 pixel decoder.
//!
//! The backbone is the ordinary `qwen3vl` text trunk plus the `clip`
//! vision tower, so this module holds only what WeVisOne adds: the
//! released decoder, the task prompts, and the prediction pipeline that
//! ties the vision tower, the text trunk and the decoder together.
mod decoder;
pub use decoder::{Decoder, Feature};

use crate::app::CliOptions;
use crate::compute::ComputePolicy;
use crate::core::scratchpad::{KvFormat, KvLifecycle};
use crate::core::{tensor::TensorSource, thread_pool::ComputePool, tokenizer::BPETokenizer};
use crate::models::qwen3::vision::media::{
    build_qwen3_media_positions, inject_qwen_media_embeddings,
};
use crate::models::qwen3::vision::{qwen_smart_resize, VisionEncoder, VisionScratchpad};
use crate::models::qwen3::{Qwen3GenerateOptions, Qwen3Input, Qwen3Model, Qwen3Session};
use crate::prompt::{append_qwen_assistant_prefix, append_qwen_message_tokens};
use base64::Engine as _;
use image::ImageEncoder;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub fn build_prompt(task: &str, query: &str) -> Result<String, String> {
    let query = query.trim();
    let phrase = if query.is_empty() || query.ends_with(['.', '?', '!']) {
        query.to_string()
    } else {
        format!("{query}.")
    };
    let labels = query
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(". ");
    Ok(match task.trim().to_lowercase().as_str() {
        "detection" => format!("Locate all the instances that match the following description: {labels}."),
        "referring" => format!("Locate the instances that match the following description: {phrase}"),
        "point" => format!("point to: {labels}"),
        "count_box" => format!("How many {} in the image? Please count them with boxes.", if query.is_empty() { "objects" } else { query }),
        "count_point" => format!("How many {} in the image?", if query.is_empty() { "objects" } else { query }),
        "segmentation" | "matting" => format!("Locate the instances that match the following description and provide the alpha mattes: {phrase}"),
        "ocr" => "detect all the word in the box format".into(),
        "ocr_grounding" => format!("Locate the following text in the image: {query}"),
        "freeform" => query.into(),
        _ => return Err(format!("unsupported WeVisOne task: {task}")),
    })
}

/// The released decoder consumes only boxes immediately followed by `<OBJ>`.
pub fn parse_seg_boxes(text: &str) -> Vec<[f32; 4]> {
    let boxes = regex::Regex::new(r"(?is)<box>(.*?)(?:</box>|$)").unwrap();
    let coords = regex::Regex::new(r"<(\d{1,4})>").unwrap();
    let obj = regex::Regex::new(r"(?i)<OBJ>").unwrap();
    let mut result = Vec::new();
    for body in boxes.captures_iter(text.split("<|im_end|>").next().unwrap_or_default()) {
        let parts = obj.split(&body[1]).collect::<Vec<_>>();
        for prefix in &parts[..parts.len().saturating_sub(1)] {
            let values = coords
                .captures_iter(prefix)
                .map(|c| c[1].parse::<u32>().unwrap())
                .collect::<Vec<_>>();
            if values.len() >= 4 {
                let v = &values[values.len() - 4..];
                if v.iter().all(|&x| x <= 1000) && v[2] > v[0] && v[3] > v[1] {
                    result.push(std::array::from_fn(|i| v[i] as f32 / 1000.0));
                }
            }
        }
    }
    result
}

#[derive(Clone, Copy)]
pub struct PredictionOptions {
    pub compute: ComputePolicy,
    pub kv_format: KvFormat,
    pub max_context: usize,
    pub prefill_batch_size: usize,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub repetition_penalty: f32,
}
impl PredictionOptions {
    pub fn from_cli(options: &crate::app::CliOptions) -> Result<Self, String> {
        let result = Self {
            compute: options.compute_policy(),
            kv_format: options.kv_format,
            max_context: options.effective_max_context(),
            prefill_batch_size: options.effective_prefill_batch_size()?,
            max_new_tokens: options.max_tokens.unwrap_or(4096),
            temperature: options.temperature.unwrap_or(0.0),
            repetition_penalty: options.repetition_penalty.unwrap_or(1.05),
        };
        if result.max_new_tokens == 0
            || result.max_new_tokens > 4096
            || !result.temperature.is_finite()
            || result.temperature < 0.0
            || !result.repetition_penalty.is_finite()
            || result.repetition_penalty <= 0.0
        {
            return Err("invalid WeVisOne generation options".into());
        }
        Ok(result)
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictRequest {
    pub image: String,
    pub task: String,
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub reference_boxes: Vec<[f32; 4]>,
    #[serde(default = "default_tokens")]
    pub max_new_tokens: usize,
}
fn default_tokens() -> usize {
    4096
}
impl PredictRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.image.is_empty()
            || self.query.len() > 65536
            || self.max_new_tokens == 0
            || self.max_new_tokens > 4096
            || self.reference_boxes.len() > 64
        {
            return Err(
                "WeVisOne requires an image, <=64KiB query and 1..=4096 generated tokens".into(),
            );
        }
        if self.task.trim().eq_ignore_ascii_case("visual_prompt") {
            if self.reference_boxes.is_empty()
                || self.reference_boxes.iter().any(|b| {
                    b.iter().any(|v| !v.is_finite() || *v < 0.0) || b[2] <= b[0] || b[3] <= b[1]
                })
            {
                return Err(
                    "visual_prompt requires finite pixel-space xyxy reference_boxes".into(),
                );
            }
        } else {
            build_prompt(&self.task, &self.query)?;
            if !self.reference_boxes.is_empty() {
                return Err("reference_boxes require visual_prompt".into());
            }
        }
        Ok(())
    }
    fn instruction(&self, width: u32, height: u32) -> Result<String, String> {
        if !self.task.trim().eq_ignore_ascii_case("visual_prompt") {
            return build_prompt(&self.task, &self.query);
        }
        let coords = self
            .reference_boxes
            .iter()
            .map(|b| {
                b.iter()
                    .enumerate()
                    .map(|(i, v)| {
                        format!(
                            "<{}>",
                            (v / if i % 2 == 0 {
                                width as f32
                            } else {
                                height as f32
                            } * 1000.0)
                                .clamp(0.0, 1000.0) as u32
                        )
                    })
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join(",");
        Ok(format!("Locate all similar objects in the image based on the reference boxes: <ref>object 1</ref><box>{coords}</box>"))
    }
}
#[derive(Serialize)]
pub struct Prediction {
    pub response: String,
    pub token_ids: Vec<u32>,
    pub boxes: Vec<[f32; 4]>,
    /// One 8-bit grayscale PNG data URI per OBJ, sigmoid before resize.
    pub alpha_png: Vec<String>,
    pub width: u32,
    pub height: u32,
}

pub fn validate_components(
    text: &dyn TensorSource,
    vision: &dyn TensorSource,
    decoder: Option<&dyn TensorSource>,
) -> Result<(), String> {
    let revision = |s: &dyn TensorSource| {
        s.metadata("wevisone_vl.source_revision")
            .and_then(|v| v.to_string_val())
            .map(str::to_owned)
    };
    if crate::app::arch_of(text) != "qwen3vl"
        || revision(text).as_deref() != Some("855d2a73ac0b929f668e6ac4f5deccc667aba99b")
        || revision(text) != revision(vision)
        || decoder.is_some_and(|s| revision(s) != revision(text))
    {
        return Err("expected matching released WeVisOne text/vision/decoder components".into());
    }
    crate::app::media::validate_mmproj_capabilities(
        "qwen3vl",
        vision,
        crate::app::media::MediaKind::Image,
    )?;
    Ok(())
}

pub fn predict(
    source: Arc<dyn TensorSource>,
    mmproj: &dyn TensorSource,
    decoder_source: Option<&dyn TensorSource>,
    pool: Arc<ComputePool>,
    tokenizer: Arc<BPETokenizer>,
    options: PredictionOptions,
    request: &PredictRequest,
    image: &image::DynamicImage,
) -> Result<Prediction, String> {
    let policy = options.compute;
    let max_new_tokens = request.max_new_tokens.min(options.max_new_tokens);
    request.validate()?;
    validate_components(source.as_ref(), mmproj, decoder_source)?;
    // Full Vulkan currently cannot export DeepStack inputs and raw hidden rows.
    policy.validate_text_arch("qwen3vl")?;
    let _scope = policy.enter_legacy_scope();
    let task = request.task.trim().to_lowercase();
    let needs_masks = matches!(task.as_str(), "segmentation" | "matting");
    let decoder = if needs_masks || decoder_source.is_some() {
        Some(Decoder::new(
            decoder_source.ok_or("segmentation/matting require --wevisone-decoder")?,
            &pool,
        )?)
    } else {
        None
    };
    let (width, height) = (image.width(), image.height());
    let instruction = request.instruction(width, height)?;
    let mut encoder = VisionEncoder::from_source(mmproj, pool.clone())?;
    encoder.precompute_lossless();
    let grid = qwen_smart_resize(width as usize, height as usize, &encoder.config)?;
    let pixels = crate::app::media::normalize_resized_image(
        image,
        grid.image_width(),
        grid.image_height(),
        &[0.5; 3],
        &[0.5; 3],
    )?;
    let mut scratch = VisionScratchpad::new(&encoder.config);
    let mut captured = Vec::new();
    encoder.encode_pair_capture(
        &pixels,
        &pixels,
        grid.image_width(),
        grid.image_height(),
        &mut scratch,
        &mut |layer, values| {
            if decoder.is_some() && [5, 11, 17, 23].contains(&layer) {
                captured.push(values.to_vec());
            }
        },
    )?;
    let features = captured
        .iter()
        .map(|v| Feature::from_premerge(v, 1024, grid.grid_h * 2, grid.grid_w * 2, 2))
        .collect::<Result<Vec<_>, _>>()?;
    let model = Qwen3Model::from_source(source, tokenizer.clone(), pool.clone())?;
    let mut ids = prompt_tokens(&tokenizer, &instruction, grid.token_count())?;
    let pad = tokenizer
        .token_id("<|image_pad|>")
        .ok_or("missing image pad")?;
    let prepare = |ids: &[u32]| -> Result<_, String> {
        let mut embeddings = model.embed_tokens(ids)?;
        let deep = inject_qwen_media_embeddings(
            ids,
            pad,
            &mut embeddings,
            &scratch.projected,
            &scratch.deepstack,
            2560,
        )?;
        let positions = build_qwen3_media_positions(ids, pad, &[(grid.grid_h, grid.grid_w)])?;
        Ok((embeddings, deep, positions))
    };
    let capacity = ids
        .len()
        .checked_add(max_new_tokens)
        .ok_or("context overflow")?;
    if capacity > options.max_context.min(model.config().n_ctx) {
        return Err(format!(
            "WeVisOne requires {capacity} tokens; context limit is {}",
            options.max_context.min(model.config().n_ctx)
        ));
    }
    let mut session = Qwen3Session::new_with_compute(
        &model,
        capacity,
        options.kv_format,
        KvLifecycle::Ephemeral,
        policy,
    )?;
    let (embeddings, deep, positions) = prepare(&ids)?;
    let prompt_len = ids.len();
    let generated = session.generate_streaming(
        Qwen3Input {
            token_ids: &ids,
            positions: &positions,
            embeddings: Some(&embeddings),
            deepstack_embeddings: Some(&deep),
        },
        Qwen3GenerateOptions {
            max_new_tokens,
            temperature: options.temperature,
            prefill_batch_size: options.prefill_batch_size,
        },
        options.repetition_penalty,
        |_| {},
    )?;
    drop(session);
    let response = tokenizer.decode(&generated.token_ids, true);
    let boxes = parse_seg_boxes(&response);
    let mut alpha_png = Vec::new();
    if !boxes.is_empty() {
        if decoder.is_none() {
            return Err("response contains OBJ masks; provide --wevisone-decoder".into());
        }
        let output_pixels = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(boxes.len()))
            .ok_or("alpha dimensions overflow")?;
        if boxes.len() > 64 || output_pixels > 64 * 1024 * 1024 {
            return Err("WeVisOne alpha output exceeds 64 masks / 64M pixels".into());
        }
        ids.extend(&generated.token_ids);
        // Causal OBJ states do not need a trailing EOS/newline or extra context.
        if ids.len() > options.max_context.min(model.config().n_ctx) {
            return Err("teacher-forced response exceeds context limit".into());
        }
        let (embeddings, deep, positions) = prepare(&ids)?;
        let mut teacher = Qwen3Session::new_with_compute(
            &model,
            ids.len(),
            KvFormat::F32,
            KvLifecycle::Ephemeral,
            policy,
        )?;
        let input = Qwen3Input {
            token_ids: &ids,
            positions: &positions,
            embeddings: Some(&embeddings),
            deepstack_embeddings: Some(&deep),
        };
        let raw = if policy == ComputePolicy::Auto {
            teacher.forward_hidden_sequence_raw_with_linear_offload(input, policy)?
        } else {
            teacher.forward_hidden_sequence_raw(input)?
        };
        let mut hidden = Vec::new();
        for (i, _) in ids
            .iter()
            .enumerate()
            .skip(prompt_len)
            .filter(|(_, id)| **id == 152678)
            .take(boxes.len())
        {
            hidden.extend_from_slice(&raw[i * 2560..(i + 1) * 2560]);
        }
        if hidden.len() != boxes.len() * 2560 {
            return Err("box/OBJ hidden row count mismatch".into());
        }
        let canvas =
            crate::app::media::normalize_resized_image(image, 1024, 1024, &[0.0; 3], &[1.0; 3])?;
        let mut chw = vec![0.0; canvas.len()];
        for (i, pixel) in canvas.chunks_exact(3).enumerate() {
            for c in 0..3 {
                chw[c * 1024 * 1024 + i] = pixel[c];
            }
        }
        let masks = decoder.as_ref().ok_or("missing decoder")?.forward(
            &Feature::new(chw, 3, 1024, 1024)?,
            &features,
            &hidden,
            &boxes,
        )?;
        for mask in masks {
            let alpha = mask.sigmoid().resize(height as usize, width as usize)?;
            let bytes = alpha
                .values
                .iter()
                .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect::<Vec<_>>();
            let mut png = Vec::new();
            image::codecs::png::PngEncoder::new(&mut png)
                .write_image(&bytes, width, height, image::ExtendedColorType::L8)
                .map_err(|e| e.to_string())?;
            alpha_png.push(format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(png)
            ));
        }
    }
    Ok(Prediction {
        response,
        token_ids: generated.token_ids,
        boxes,
        alpha_png,
        width,
        height,
    })
}

pub(crate) fn prompt_tokens(
    tokenizer: &BPETokenizer,
    instruction: &str,
    rows: usize,
) -> Result<Vec<u32>, String> {
    let mut ids = Vec::new();
    append_qwen_message_tokens(
        &mut ids,
        tokenizer,
        "system",
        &tokenizer.encode("You are a helpful assistant.", Default::default()),
    )?;
    let mut content = Vec::new();
    for marker in ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"] {
        let id = tokenizer
            .token_id(marker)
            .ok_or_else(|| format!("missing {marker}"))?;
        content.extend(std::iter::repeat_n(
            id,
            if marker == "<|image_pad|>" { rows } else { 1 },
        ));
    }
    content.extend(tokenizer.encode(
        instruction,
        crate::core::tokenizer::EncodeOptions {
            add_special: false,
            parse_special: true,
        },
    ));
    append_qwen_message_tokens(&mut ids, tokenizer, "user", &content)?;
    append_qwen_assistant_prefix(&mut ids, tokenizer, true)?;
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::*;
    #[test]
    fn request_boundary_and_visual_prompt() {
        let mut r: PredictRequest = serde_json::from_str(
            r#"{"image":"a.png","task":"visual_prompt","reference_boxes":[[10,20,90,80]]}"#,
        )
        .unwrap();
        r.validate().unwrap();
        assert!(r
            .instruction(100, 100)
            .unwrap()
            .contains("<100><200><900><800>"));
        r.max_new_tokens = 0;
        assert!(r.validate().is_err());
        r.max_new_tokens = 1;
        r.reference_boxes[0][0] = f32::NAN;
        assert!(r.validate().is_err());
        assert!(
            serde_json::from_str::<PredictRequest>(r#"{"image":"a","task":"ocr","typo":1}"#)
                .is_err()
        );
    }
    #[test]
    #[ignore = "requires real model weights and Transformers CPU fixtures"]
    fn official_backbone_fixture() {
        use crate::format::ggufrs::{open_model_source, ComponentRole};
        let root = std::path::PathBuf::from(std::env::var("WEVISONE_BACKBONE").unwrap());
        let model_path = std::env::var("WEVISONE_MODEL").unwrap();
        let vision_path = std::env::var("WEVISONE_VISION").unwrap();
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("fixture.json")).unwrap()).unwrap();
        let source: Arc<dyn TensorSource> = Arc::from(
            open_model_source(std::path::Path::new(&model_path), ComponentRole::Llm).unwrap(),
        );
        let mmproj =
            open_model_source(std::path::Path::new(&vision_path), ComponentRole::Mmproj).unwrap();
        let tokenizer =
            Arc::new(BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned()).unwrap());
        for sample in fixture["tokenizer_samples"].as_array().unwrap() {
            let expected: Vec<u32> = serde_json::from_value(sample["ids"].clone()).unwrap();
            assert_eq!(
                tokenizer.encode(
                    sample["text"].as_str().unwrap(),
                    crate::core::tokenizer::EncodeOptions {
                        add_special: false,
                        parse_special: true
                    }
                ),
                expected
            );
        }
        let pool = Arc::new(ComputePool::new(8));
        let mut encoder = VisionEncoder::from_source(mmproj.as_ref(), pool.clone()).unwrap();
        encoder.precompute();
        let image = image::open(root.join("image.png")).unwrap();
        let grid = qwen_smart_resize(
            image.width() as usize,
            image.height() as usize,
            &encoder.config,
        )
        .unwrap();
        let ids = prompt_tokens(
            &tokenizer,
            fixture["instruction"].as_str().unwrap(),
            grid.token_count(),
        )
        .unwrap();
        assert_eq!(
            ids,
            serde_json::from_value::<Vec<u32>>(fixture["token_ids"].clone()).unwrap()
        );
        let positions =
            build_qwen3_media_positions(&ids, 151655, &[(grid.grid_h, grid.grid_w)]).unwrap();
        let expected: Vec<[usize; 3]> =
            serde_json::from_value(fixture["positions"].clone()).unwrap();
        assert_eq!(
            positions
                .iter()
                .map(|p| [p[0], p[1], p[2]])
                .collect::<Vec<_>>(),
            expected
        );
        let approximate = std::env::var_os("WEVISONE_APPROXIMATE").is_some();
        let compare = |name: &str, actual: &[f32], limit: f32| {
            let bytes = std::fs::read(root.join(format!("{name}.f32"))).unwrap();
            let expected = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(expected.len(), actual.len(), "{name}");
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let rmse = (actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).powi(2) as f64)
                .sum::<f64>()
                / actual.len() as f64)
                .sqrt();
            let worst = actual
                .iter()
                .zip(&expected)
                .position(|(a, b)| (*a - *b).abs() == max)
                .unwrap();
            eprintln!(
                "{name}: max_abs={max} rmse={rmse} worst_actual={} worst_expected={}",
                actual[worst], expected[worst]
            );
            let reference_rms = (expected.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>()
                / expected.len() as f64)
                .sqrt();
            let dot = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>();
            let norm = actual
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                .sqrt()
                * reference_rms
                * (expected.len() as f64).sqrt();
            let cosine = dot / norm;
            eprintln!(
                "{name}: relative_rmse={} cosine={cosine}",
                rmse / reference_rms
            );
            let aligned = if approximate && matches!(name, "raw" | "logits") {
                if name == "raw" {
                    rmse / reference_rms <= 0.06 && cosine >= 0.998
                } else {
                    rmse / reference_rms <= 0.03 && cosine >= 0.999
                }
            } else {
                max <= limit && rmse <= f64::from(limit) / 50.0
            };
            assert!(
                actual.iter().all(|v| v.is_finite()) && aligned,
                "{name} max_abs={max}"
            );
        };
        let pixels = crate::app::media::normalize_resized_image(
            &image,
            grid.image_width(),
            grid.image_height(),
            &[0.5; 3],
            &[0.5; 3],
        )
        .unwrap();
        let mut features = Vec::new();
        let mut scratch = VisionScratchpad::new(&encoder.config);
        encoder
            .encode_pair_capture(
                &pixels,
                &pixels,
                grid.image_width(),
                grid.image_height(),
                &mut scratch,
                &mut |layer, values| {
                    if [5, 11, 17, 23].contains(&layer) {
                        compare(
                            &format!("vit{layer}"),
                            values,
                            if layer == 23 { 0.02 } else { 0.0005 },
                        );
                        features.push(
                            Feature::from_premerge(
                                values,
                                1024,
                                grid.grid_h * 2,
                                grid.grid_w * 2,
                                2,
                            )
                            .unwrap(),
                        );
                    }
                },
            )
            .unwrap();
        compare("projected", &scratch.projected, 0.001);
        compare("deepstack", &scratch.deepstack, 0.001);
        let model = Qwen3Model::from_source(source, tokenizer.clone(), pool.clone()).unwrap();
        let mut embeddings = model.embed_tokens(&ids).unwrap();
        let deep = inject_qwen_media_embeddings(
            &ids,
            151655,
            &mut embeddings,
            &scratch.projected,
            &scratch.deepstack,
            2560,
        )
        .unwrap();
        let policy = if std::env::var_os("WEVISONE_AUTO").is_some() {
            ComputePolicy::Auto
        } else {
            ComputePolicy::Cpu
        };
        let mut session = Qwen3Session::new_with_compute(
            &model,
            ids.len() + 32,
            KvFormat::F32,
            KvLifecycle::Ephemeral,
            policy,
        )
        .unwrap();
        let input = Qwen3Input {
            token_ids: &ids,
            positions: &positions,
            embeddings: Some(&embeddings),
            deepstack_embeddings: Some(&deep),
        };
        let raw = if policy == ComputePolicy::Auto {
            session
                .forward_hidden_sequence_raw_with_linear_offload(input.clone(), policy)
                .unwrap()
        } else {
            session.forward_hidden_sequence_raw(input.clone()).unwrap()
        };
        compare("raw", &raw, if approximate { 25.0 } else { 0.04 });
        let mut norm = vec![0.0; 2560];
        crate::ops::rms_norm(
            &raw[raw.len() - 2560..],
            model.output_norm(),
            &mut norm,
            model.config().eps,
        );
        let mut logits = vec![0.0; model.config().vocab];
        model
            .output
            .kernel
            .forward(&norm, &mut logits, 2560, model.config().vocab);
        compare("logits", &logits, if approximate { 5.0 } else { 0.003 });
        let mut session = Qwen3Session::new_with_compute(
            &model,
            ids.len() + 32,
            KvFormat::F32,
            KvLifecycle::Ephemeral,
            policy,
        )
        .unwrap();
        let result = session
            .generate_streaming(
                input,
                Qwen3GenerateOptions {
                    max_new_tokens: 32,
                    temperature: 0.0,
                    prefill_batch_size: 64,
                },
                1.05,
                |_| {},
            )
            .unwrap();
        let mut expected: Vec<u32> = serde_json::from_value(fixture["greedy_ids"].clone()).unwrap();
        if expected.last() == Some(&151645) {
            expected.pop();
        }
        eprintln!("greedy expected={expected:?} actual={:?}", result.token_ids);
        if !approximate {
            assert_eq!(result.token_ids, expected);
        }
        if !approximate && std::env::var_os("WEVISONE_MASKS").is_some() {
            drop(session);
            let mut teacher_ids = ids.clone();
            teacher_ids.extend(&result.token_ids);
            teacher_ids.push(151645);
            teacher_ids.extend(tokenizer.encode("\n", Default::default()));
            assert_eq!(
                teacher_ids,
                serde_json::from_value::<Vec<u32>>(fixture["teacher_ids"].clone()).unwrap()
            );
            let mut embeds = model.embed_tokens(&teacher_ids).unwrap();
            let deep = inject_qwen_media_embeddings(
                &teacher_ids,
                151655,
                &mut embeds,
                &scratch.projected,
                &scratch.deepstack,
                2560,
            )
            .unwrap();
            let positions =
                build_qwen3_media_positions(&teacher_ids, 151655, &[(grid.grid_h, grid.grid_w)])
                    .unwrap();
            let mut teacher = Qwen3Session::new_with_compute(
                &model,
                teacher_ids.len(),
                KvFormat::F32,
                KvLifecycle::Ephemeral,
                policy,
            )
            .unwrap();
            let raw = teacher
                .forward_hidden_sequence_raw(Qwen3Input {
                    token_ids: &teacher_ids,
                    positions: &positions,
                    embeddings: Some(&embeds),
                    deepstack_embeddings: Some(&deep),
                })
                .unwrap();
            let hidden = teacher_ids
                .iter()
                .enumerate()
                .filter(|(_, id)| **id == 152678)
                .flat_map(|(i, _)| raw[i * 2560..(i + 1) * 2560].to_vec())
                .collect::<Vec<_>>();
            compare("obj_hidden", &hidden, 0.04);
            let path = std::env::var("WEVISONE_DECODER").unwrap();
            let decoder_source =
                open_model_source(std::path::Path::new(&path), ComponentRole::Llm).unwrap();
            let decoder = Decoder::new(decoder_source.as_ref(), &pool).unwrap();
            let rgb = crate::app::media::normalize_resized_image(
                &image, 1024, 1024, &[0.0; 3], &[1.0; 3],
            )
            .unwrap();
            let mut chw = vec![0.0; rgb.len()];
            for (i, p) in rgb.chunks_exact(3).enumerate() {
                for c in 0..3 {
                    chw[c * 1024 * 1024 + i] = p[c];
                }
            }
            let boxes: Vec<[f32; 4]> = serde_json::from_value(fixture["boxes"].clone()).unwrap();
            let masks = decoder
                .forward(
                    &Feature::new(chw, 3, 1024, 1024).unwrap(),
                    &features,
                    &hidden,
                    &boxes,
                )
                .unwrap();
            compare(
                "full_mask_logits",
                &masks
                    .iter()
                    .flat_map(|f| f.values.clone())
                    .collect::<Vec<_>>(),
                0.02,
            );
            let alpha = masks
                .into_iter()
                .flat_map(|f| f.sigmoid().resize(128, 128).unwrap().values)
                .collect::<Vec<_>>();
            compare("alpha", &alpha, 0.002);
        }
    }

    #[test]
    fn task_prompts_match_released_instructions() {
        assert_eq!(
            build_prompt("detection", " cat\n dog ").unwrap(),
            "Locate all the instances that match the following description: cat. dog."
        );
        assert_eq!(build_prompt("matting", "the person").unwrap(), "Locate the instances that match the following description and provide the alpha mattes: the person.");
        assert!(build_prompt("unknown", "cat").is_err());
    }

    #[test]
    fn segmentation_boxes_require_valid_obj_delimiters() {
        assert_eq!(
            parse_seg_boxes("<box><1><2><900><999><OBJ>,<10><20><30><40><OBJ></box>"),
            vec![[0.001, 0.002, 0.9, 0.999], [0.01, 0.02, 0.03, 0.04]]
        );
        for text in [
            "None",
            "<box><1><2><3><4></box>",
            "<box><9><2><3><4><OBJ></box>",
            "<box><0><0><1001><999><OBJ></box>",
        ] {
            assert!(parse_seg_boxes(text).is_empty());
        }
    }
}
