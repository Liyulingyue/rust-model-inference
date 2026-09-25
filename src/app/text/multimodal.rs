use super::generation::{sample_token, validate_gemma4_temperature};
use super::vision::{
    build_qwen3_media_positions, inject_qwen_media_embeddings, inject_vision_embeddings,
    validate_single_qwen_media,
};
use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::app::media::{decode_image, normalize_resized_image};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::models::qwen3::vision::{
    qwen_smart_resize as qwen3vl_smart_resize, VisionEncoder as VisionEncoder3vl,
    VisionScratchpad as VisionScratchpad3vl,
};
use crate::models::qwen3::{Qwen3GenerateOptions, Qwen3Input, Qwen3Model};
use crate::models::qwen35::vision::{
    qwen_smart_resize as qwen35_smart_resize, VisionEncoder as VisionEncoder35, VisionGrid,
    VisionScratchpad as VisionScratchpad35,
};
use crate::models::qwen35::{build_qwen35_positions, Qwen35Model, Qwen35Session};
use crate::prompt::{append_qwen_assistant_prefix, append_qwen_message_tokens};
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;

/// Multimodal forward (vision + audio + text → generated text). The
/// HTTP server's `/v1/jev/image` and `/v1/jev/image_grouped` paths
/// call this directly with `max_new_tokens=1` to approximate the
/// argmax-over-labels behaviour of JEV on multimodal inputs; the
/// per-arch multimodal logits-only forward (which would let us
/// apply the same scoring as text JEV) is not yet implemented for
/// Qwen3.5, so the 1-token-generation approach is the pragmatic
/// fallback. `run_qwen3_family_multimodal_capture_text` wraps this
/// for the gemma4 path which has a separate forward signature.
pub fn run_qwen3_family_multimodal(
    llm_source: &dyn TensorSource,
    model_source: Arc<dyn TensorSource>,
    mmproj_path: &Path,
    image_path: Option<&Path>,
    video_path: Option<&Path>,
    audio_path: Option<&Path>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    prefill_batch_size: usize,
) -> Result<String, String> {
    validate_single_qwen_media(
        image_path.is_some(),
        video_path.is_some(),
        audio_path.is_some(),
    )?;
    // Shared ComputePool for vision-encoder matmuls (Qwen2.5-Omni vision
    // encode is otherwise single-threaded BF16; see `matmul_weight_batch_pooled`).
    let pool = Arc::new(ComputePool::new(resolve_thread_count(
        n_threads_arg,
        std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1),
    )));
    let arch = llm_source
        .metadata("general.architecture")
        .and_then(|value| value.to_string_val())
        .unwrap_or_default();
    let mmproj: Arc<dyn TensorSource> = Arc::from(
        open_model_source(mmproj_path, ComponentRole::Mmproj)
            .map_err(|error| format!("Failed to load mmproj {}: {error}", mmproj_path.display()))?,
    );
    let media_kind = if audio_path.is_some() {
        crate::app::media::MediaKind::Audio
    } else if video_path.is_some() {
        crate::app::media::MediaKind::Video
    } else {
        crate::app::media::MediaKind::Image
    };
    let family =
        crate::app::media::validate_mmproj_capabilities(arch, mmproj.as_ref(), media_kind)?;
    let mut media = Vec::new();
    let mut media_deepstack_layers: Vec<Vec<f32>> = Vec::new();
    let mut media_grid_shapes = Vec::new();
    if let Some(audio_path) = audio_path {
        let samples = crate::app::media::decode_audio(audio_path)?;
        media =
            crate::models::qwen3::omni::encode_audio(Arc::clone(&mmproj), &samples, n_threads_arg)?;
    } else {
        let mut frames = if let Some(path) = image_path {
            vec![decode_image(path)?]
        } else {
            crate::app::media::decode_video(video_path.ok_or("missing image or video input")?)?
        };
        let is_video = video_path.is_some();
        let (first_w, first_h) = {
            let first = frames.first().ok_or("media produced no frames")?;
            (first.width() as usize, first.height() as usize)
        };
        let (grid_w, grid_h) = match family {
            crate::app::media::ProjectorFamily::Qwen3VlMerger => {
                let encoder = VisionEncoder3vl::from_source(mmproj.as_ref())
                    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
                let grid = qwen3vl_smart_resize(first_w, first_h, &encoder.config)?;
                (grid.image_width(), grid.image_height())
            }
            crate::app::media::ProjectorFamily::Qwen25Omni => {
                let mut encoder = VisionEncoder35::from_source(mmproj.as_ref())
                    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
                if is_video {
                    encoder.config.image_min_pixels = encoder.config.video_min_pixels;
                    encoder.config.image_max_pixels = encoder.config.video_max_pixels;
                }
                let grid = qwen35_smart_resize(first_w, first_h, &encoder.config)?;
                (grid.image_width(), grid.image_height())
            }
        };
        let (mean, std) = match family {
            crate::app::media::ProjectorFamily::Qwen3VlMerger => {
                let encoder = VisionEncoder3vl::from_source(mmproj.as_ref())
                    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
                (encoder.config.image_mean, encoder.config.image_std)
            }
            crate::app::media::ProjectorFamily::Qwen25Omni => {
                let encoder = VisionEncoder35::from_source(mmproj.as_ref())
                    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
                (encoder.config.image_mean, encoder.config.image_std)
            }
        };
        let normalized = frames
            .drain(..)
            .map(|frame| normalize_resized_image(&frame, grid_w, grid_h, &mean, &std))
            .collect::<Result<Vec<_>, _>>()?;
        let pairs = if is_video {
            (0..normalized.len())
                .step_by(2)
                .map(|index| (index, (index + 1).min(normalized.len() - 1)))
                .collect::<Vec<_>>()
        } else {
            vec![(0, 0)]
        };
        match family {
            crate::app::media::ProjectorFamily::Qwen3VlMerger => {
                let mut encoder = VisionEncoder3vl::from_source(mmproj.as_ref())
                    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
                encoder.precompute();
                let grid = qwen3vl_smart_resize(first_w, first_h, &encoder.config)?;
                let mut scratch = VisionScratchpad3vl::new(&encoder.config);
                for (a, b) in pairs {
                    encoder.encode_pair(
                        &normalized[a],
                        &normalized[b],
                        grid.image_width(),
                        grid.image_height(),
                        &mut scratch,
                    )?;
                    media.extend_from_slice(&scratch.projected);
                    let deepstack_layers = encoder
                        .config
                        .has_deepstack_layers
                        .iter()
                        .filter(|enabled| **enabled)
                        .count();
                    if deepstack_layers > 0 {
                        if scratch.deepstack.len() % deepstack_layers != 0 {
                            return Err("Vision deepstack output is not layer aligned".into());
                        }
                        if media_deepstack_layers.is_empty() {
                            media_deepstack_layers.resize_with(deepstack_layers, Vec::new);
                        } else if media_deepstack_layers.len() != deepstack_layers {
                            return Err(
                                "Vision deepstack layer count changed between frames".into()
                            );
                        }
                        let per_layer = scratch.deepstack.len() / deepstack_layers;
                        for (layer, output) in media_deepstack_layers.iter_mut().enumerate() {
                            output.extend_from_slice(
                                &scratch.deepstack[layer * per_layer..(layer + 1) * per_layer],
                            );
                        }
                    }
                    media_grid_shapes.push((grid.grid_h, grid.grid_w));
                }
            }
            crate::app::media::ProjectorFamily::Qwen25Omni => {
                let mut encoder = VisionEncoder35::from_source(mmproj.as_ref())
                    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
                encoder.precompute();
                if is_video {
                    encoder.config.image_min_pixels = encoder.config.video_min_pixels;
                    encoder.config.image_max_pixels = encoder.config.video_max_pixels;
                }
                let grid = qwen35_smart_resize(first_w, first_h, &encoder.config)?;
                let mut scratch = VisionScratchpad35::new(&encoder.config);
                for (a, b) in pairs {
                    encoder.encode_pair(
                        &normalized[a],
                        &normalized[b],
                        grid.image_width(),
                        grid.image_height(),
                        &mut scratch,
                        &pool,
                    )?;
                    media.extend_from_slice(&scratch.projected);
                    media_grid_shapes.push((grid.grid_h, grid.grid_w));
                }
            }
        }
    }

    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|key| llm_source.metadata(key).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?,
    );
    let model = Qwen3Model::from_source(model_source, Arc::clone(&tokenizer), Arc::clone(&pool))?;
    let width = model.config().n_embd;
    if media.len() % width != 0 {
        return Err(format!(
            "Media projector width does not match model width {width}"
        ));
    }
    let (start_name, pad_name, end_name) = match media_kind {
        crate::app::media::MediaKind::Audio => ("audio_start", "audio_pad", "audio_end"),
        crate::app::media::MediaKind::Image => ("vision_start", "image_pad", "vision_end"),
        crate::app::media::MediaKind::Video => ("vision_start", "video_pad", "vision_end"),
    };
    let start = tokenizer
        .special_token_id(start_name)
        .ok_or_else(|| format!("Required token missing: {start_name}"))?;
    let pad = tokenizer
        .special_token_id(pad_name)
        .ok_or_else(|| format!("Required token missing: {pad_name}"))?;
    let end = tokenizer
        .special_token_id(end_name)
        .ok_or_else(|| format!("Required token missing: {end_name}"))?;
    let rows = media.len() / width;
    let media_deepstack = media_deepstack_layers.concat();
    let deepstack_layers = if media_deepstack.is_empty() {
        0
    } else {
        let per_layer = rows
            .checked_mul(width)
            .ok_or("Media deepstack shape overflow")?;
        if per_layer == 0 || media_deepstack.len() % per_layer != 0 {
            return Err("Media deepstack output is not row aligned".into());
        }
        media_deepstack.len() / per_layer
    };
    if deepstack_layers != model.config().n_deepstack_layers {
        return Err(format!(
            "Deepstack layer mismatch: model={}, projector={deepstack_layers}",
            model.config().n_deepstack_layers
        ));
    }
    let mut content = vec![start];
    content.extend(std::iter::repeat_n(pad, rows));
    content.push(end);
    content.extend(tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: false,
            parse_special: false,
        },
    ));
    let mut token_ids = Vec::new();
    // Qwen2.5-Omni (qwen2vl arch with `qwen2.5o` projector) requires a
    // modality-aware system prompt per the upstream README—without it,
    // the assistant role can drift (audio output only works with the
    // exact prompt; for text-only multimodal a generic variant still helps
    // the model behave as a virtual-human assistant). We only inject the
    // system turn when the projector family matches `Qwen25Omni`, so
    // qwen3vl / qwen3vlmoe (Qwen3-VL family) keep their existing
    // system-less behaviour.
    if matches!(family, crate::app::media::ProjectorFamily::Qwen25Omni) {
        let system_text = match media_kind {
            crate::app::media::MediaKind::Audio => {
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            }
            crate::app::media::MediaKind::Video => {
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            }
            crate::app::media::MediaKind::Image => {
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            }
        };
        append_qwen_message_tokens(
            &mut token_ids,
            &tokenizer,
            "system",
            &tokenizer.encode(
                system_text,
                EncodeOptions {
                    add_special: false,
                    parse_special: false,
                },
            ),
        )?;
    }
    append_qwen_message_tokens(&mut token_ids, &tokenizer, "user", &content)?;
    append_qwen_assistant_prefix(&mut token_ids, &tokenizer, false)?;
    let mut embeddings = model.embed_tokens(&token_ids)?;
    let deepstack_embeddings = inject_qwen_media_embeddings(
        &token_ids,
        pad,
        &mut embeddings,
        &media,
        &media_deepstack,
        width,
    )?;
    let positions = build_qwen3_media_positions(&token_ids, pad, &media_grid_shapes)?;
    let generation = model.generate(
        Qwen3Input {
            token_ids: &token_ids,
            positions: &positions,
            embeddings: Some(&embeddings),
            deepstack_embeddings: (!deepstack_embeddings.is_empty())
                .then_some(deepstack_embeddings.as_slice()),
        },
        Qwen3GenerateOptions {
            max_new_tokens: max_tokens,
            temperature,
            prefill_batch_size,
        },
    )?;
    print!("{}", generation.text);
    io::stdout().flush().map_err(|error| error.to_string())?;
    println!();
    Ok(generation.text)
}

/// Logits-only multimodal forward for qwen3 / qwen3vl. Mirrors
/// [`run_qwen3_family_multimodal`] but skips autoregressive
/// generation and returns the final-position logits vector
/// (length = vocab_size). Used by the `/v1/jev/image` and
/// `/v1/jev/image_grouped` HTTP endpoints so callers can apply
/// true JEV argmax-over-labels scoring on multimodal inputs
/// without paying the cost of generation + parsing.
pub fn run_qwen3_family_multimodal_logits(
    llm_source: &dyn TensorSource,
    model_source: Arc<dyn TensorSource>,
    mmproj_path: &Path,
    image_path: Option<&Path>,
    video_path: Option<&Path>,
    audio_path: Option<&Path>,
    prompt: &str,
    n_threads_arg: usize,
    prefill_batch_size: usize,
) -> Result<(Vec<f32>, std::time::Duration), String> {
    use crate::models::qwen3::Qwen3Session;
    use crate::core::scratchpad::{KvFormat as Qwen3KvFormat, KvLifecycle};
    use crate::app::media::{frame_pairs, normalize_resized_image};

    validate_single_qwen_media(
        image_path.is_some(),
        video_path.is_some(),
        audio_path.is_some(),
    )?;
    let pool = Arc::new(ComputePool::new(resolve_thread_count(
        n_threads_arg,
        std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1),
    )));
    let arch = llm_source
        .metadata("general.architecture")
        .and_then(|value| value.to_string_val())
        .unwrap_or_default();
    let mmproj: Arc<dyn TensorSource> = Arc::from(
        open_model_source(mmproj_path, ComponentRole::Mmproj)
            .map_err(|error| format!("Failed to load mmproj {}: {error}", mmproj_path.display()))?,
    );
    let media_kind = if audio_path.is_some() {
        crate::app::media::MediaKind::Audio
    } else if video_path.is_some() {
        crate::app::media::MediaKind::Video
    } else {
        crate::app::media::MediaKind::Image
    };
    let family =
        crate::app::media::validate_mmproj_capabilities(arch, mmproj.as_ref(), media_kind)?;
    let mut media = Vec::new();
    let mut media_deepstack_layers: Vec<Vec<f32>> = Vec::new();
    let mut media_grid_shapes = Vec::new();
    if let Some(audio_path) = audio_path {
        let samples = crate::app::media::decode_audio(audio_path)?;
        media =
            crate::models::qwen3::omni::encode_audio(Arc::clone(&mmproj), &samples, n_threads_arg)?;
    } else {
        let mut frames = if let Some(path) = image_path {
            vec![decode_image(path)?]
        } else {
            crate::app::media::decode_video(video_path.ok_or("missing image or video input")?)?
        };
        let is_video = video_path.is_some();
        let (first_w, first_h) = {
            let first = frames.first().ok_or("media produced no frames")?;
            (first.width() as usize, first.height() as usize)
        };
        let (grid_w, grid_h) = match family {
            crate::app::media::ProjectorFamily::Qwen3VlMerger => {
                let encoder = VisionEncoder3vl::from_source(mmproj.as_ref())
                    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
                let grid = qwen3vl_smart_resize(first_w, first_h, &encoder.config)?;
                (grid.image_width(), grid.image_height())
            }
            crate::app::media::ProjectorFamily::Qwen25Omni => {
                let mut encoder = VisionEncoder35::from_source(mmproj.as_ref())
                    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
                if is_video {
                    encoder.config.image_min_pixels = encoder.config.video_min_pixels;
                    encoder.config.image_max_pixels = encoder.config.video_max_pixels;
                }
                let grid = qwen35_smart_resize(first_w, first_h, &encoder.config)?;
                (grid.image_width(), grid.image_height())
            }
        };
        let pairs = frame_pairs(frames.len());
        let mut normalized = Vec::with_capacity(frames.len());
        for frame in &frames {
            normalized.push(normalize_resized_image(
                frame,
                grid_w,
                grid_h,
                &[0.5, 0.5, 0.5],
                &[0.5, 0.5, 0.5],
            )?);
        }
        match family {
            crate::app::media::ProjectorFamily::Qwen3VlMerger => {
                let mut encoder = VisionEncoder3vl::from_source(mmproj.as_ref())
                    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
                encoder.precompute();
                let grid = qwen3vl_smart_resize(first_w, first_h, &encoder.config)?;
                let mut scratch = VisionScratchpad3vl::new(&encoder.config);
                for (a, b) in pairs {
                    encoder.encode_pair(
                        &normalized[a],
                        &normalized[b],
                        grid_w,
                        grid_h,
                        &mut scratch,
                    )?;
                    media.extend_from_slice(&scratch.projected);
                    let deepstack_layers = scratch
                        .deepstack
                        .len()
                        / (scratch.projected.len() / encoder.config.n_embd).max(1)
                        .max(1);
                    if scratch.deepstack.len() % deepstack_layers != 0 {
                        return Err("Vision deepstack output is not layer aligned".into());
                    }
                    if media_deepstack_layers.is_empty() {
                        media_deepstack_layers.resize_with(deepstack_layers, Vec::new);
                    } else if media_deepstack_layers.len() != deepstack_layers {
                        return Err(
                            "Vision deepstack layer count changed between frames".into()
                        );
                    }
                    let per_layer = scratch.deepstack.len() / deepstack_layers;
                    for (layer, output) in media_deepstack_layers.iter_mut().enumerate() {
                        output.extend_from_slice(
                            &scratch.deepstack[layer * per_layer..(layer + 1) * per_layer],
                        );
                    }
                    media_grid_shapes.push((grid.grid_h, grid.grid_w));
                }
            }
            crate::app::media::ProjectorFamily::Qwen25Omni => {
                let mut encoder = VisionEncoder35::from_source(mmproj.as_ref())
                    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
                encoder.precompute();
                if is_video {
                    encoder.config.image_min_pixels = encoder.config.video_min_pixels;
                    encoder.config.image_max_pixels = encoder.config.video_max_pixels;
                }
                let grid = qwen35_smart_resize(first_w, first_h, &encoder.config)?;
                let mut scratch = VisionScratchpad35::new(&encoder.config);
                for (a, b) in pairs {
                    encoder.encode_pair(
                        &normalized[a],
                        &normalized[b],
                        grid.image_width(),
                        grid.image_height(),
                        &mut scratch,
                        &pool,
                    )?;
                    media.extend_from_slice(&scratch.projected);
                    media_grid_shapes.push((grid.grid_h, grid.grid_w));
                }
            }
        }
    }

    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|key| llm_source.metadata(key).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?,
    );
    let model = Qwen3Model::from_source(model_source, Arc::clone(&tokenizer), Arc::clone(&pool))?;
    let width = model.config().n_embd;
    if media.len() % width != 0 {
        return Err(format!(
            "Media projector width does not match model width {width}"
        ));
    }
    let (start_name, pad_name, end_name) = match media_kind {
        crate::app::media::MediaKind::Audio => ("audio_start", "audio_pad", "audio_end"),
        crate::app::media::MediaKind::Image => ("vision_start", "image_pad", "vision_end"),
        crate::app::media::MediaKind::Video => ("vision_start", "video_pad", "vision_end"),
    };
    let start = tokenizer
        .special_token_id(start_name)
        .ok_or_else(|| format!("Required token missing: {start_name}"))?;
    let pad = tokenizer
        .special_token_id(pad_name)
        .ok_or_else(|| format!("Required token missing: {pad_name}"))?;
    let end = tokenizer
        .special_token_id(end_name)
        .ok_or_else(|| format!("Required token missing: {end_name}"))?;
    let rows = media.len() / width;
    let mut content = vec![start];
    content.extend(std::iter::repeat_n(pad, rows));
    content.push(end);
    content.extend(tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: false,
            parse_special: false,
        },
    ));
    let mut token_ids = Vec::new();
    if family == crate::app::media::ProjectorFamily::Qwen25Omni {
        let system_text = if media_kind == crate::app::media::MediaKind::Audio {
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        } else {
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        };
        append_qwen_message_tokens(
            &mut token_ids,
            &tokenizer,
            "system",
            &tokenizer.encode(
                system_text,
                EncodeOptions {
                    add_special: false,
                    parse_special: false,
                },
            ),
        )?;
    }
    append_qwen_message_tokens(&mut token_ids, &tokenizer, "user", &content)?;
    append_qwen_assistant_prefix(&mut token_ids, &tokenizer, false)?;
    let mut embeddings = model.embed_tokens(&token_ids)?;
    // qwen3vl deepstack embeddings currently unused here — vision
    // embeddings flow through `model.embed_tokens` + `inject_qwen_media_embeddings`.
    let positions = build_qwen3_media_positions(&token_ids, pad, &media_grid_shapes)?;
    let capacity = token_ids.len() + 1;
    let mut session = Qwen3Session::new_with_kv_state(
        &model,
        capacity.min(model.config().n_ctx),
        Qwen3KvFormat::F16,
        KvLifecycle::Ephemeral,
    )?;
    let t0 = std::time::Instant::now();
    let input = Qwen3Input {
        token_ids: &token_ids,
        positions: &positions,
        embeddings: Some(&embeddings),
        deepstack_embeddings: None,
    };
    let (logits, _dur) = session.forward_logits(input, prefill_batch_size)?;
    Ok((logits, t0.elapsed()))
}

/// Logits-only multimodal forward for qwen35 (Qwen2.5-Omni vision
/// encoder path). Mirrors the qwen35 multimodal setup inside
/// `run_multimodal_with_video_ref` but ends with a single
/// `session.step` call to return the final-position logits vector.
/// Used by the `/v1/jev/image` and `/v1/jev/image_grouped` HTTP
/// endpoints so callers can apply true JEV argmax-over-labels
/// scoring on multimodal inputs.
pub fn run_qwen35_family_multimodal_logits(
    llm_source: &dyn TensorSource,
    mmproj_path: &Path,
    image_path: Option<&Path>,
    video_path: Option<&Path>,
    audio_path: Option<&Path>,
    prompt: &str,
    n_threads_arg: usize,
    prefill_batch_size: usize,
    max_context: usize,
) -> Result<(Vec<f32>, std::time::Duration), String> {
    use crate::models::qwen35::vision::{
        qwen_smart_resize as qwen35_smart_resize, VisionEncoder as VisionEncoder35,
        VisionGrid, VisionScratchpad as VisionScratchpad35,
    };
    use crate::models::qwen35::{Qwen35Model, Qwen35Session};
    use crate::app::media::frame_pairs;

    if audio_path.is_some() {
        return Err(format!(
            "Only gemma4 architecture is supported for multimodal audio, got: qwen35"
        ));
    }
    let pool = Arc::new(ComputePool::new(resolve_thread_count(
        n_threads_arg,
        std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1),
    )));
    let mmproj: Arc<dyn TensorSource> = Arc::from(
        open_model_source(mmproj_path, ComponentRole::Mmproj)
            .map_err(|error| format!("Failed to load mmproj {}: {error}", mmproj_path.display()))?,
    );

    // Decode image / video.
    let frames: Vec<image::DynamicImage> = if let Some(path) = image_path {
        vec![decode_image(path)?]
    } else if let Some(path) = video_path {
        crate::app::media::decode_video(path)?
    } else {
        return Err("qwen35 multimodal requires --image or --video".into());
    };
    let is_video = video_path.is_some();

    let (first_w, first_h) = {
        let first = frames.first().ok_or("media produced no frames")?;
        (first.width() as usize, first.height() as usize)
    };
    let mut encoder = VisionEncoder35::from_source(mmproj.as_ref())
        .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
    encoder.precompute();
    if is_video {
        encoder.config.image_min_pixels = encoder.config.video_min_pixels;
        encoder.config.image_max_pixels = encoder.config.video_max_pixels;
    }
    let grid = qwen35_smart_resize(first_w, first_h, &encoder.config)?;
    let mut scratch = VisionScratchpad35::new(&encoder.config);
    let pairs = frame_pairs(frames.len());
    let mut vis_embeddings: Vec<f32> = Vec::new();
    for (a, b) in pairs {
        let normalized_a = crate::app::media::normalize_resized_image(
            &frames[a],
            grid.image_width(),
            grid.image_height(),
            &[0.5, 0.5, 0.5],
            &[0.5, 0.5, 0.5],
        )?;
        let normalized_b = crate::app::media::normalize_resized_image(
            &frames[b],
            grid.image_width(),
            grid.image_height(),
            &[0.5, 0.5, 0.5],
            &[0.5, 0.5, 0.5],
        )?;
        encoder.encode_pair(
            &normalized_a,
            &normalized_b,
            grid.image_width(),
            grid.image_height(),
            &mut scratch,
            &pool,
        )?;
        vis_embeddings.extend_from_slice(&scratch.projected);
    }
    let n_vis_tokens = grid.token_count();

    let tokenizer = BPETokenizer::from_gguf_metadata(|k| llm_source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    let image_token_id = tokenizer
        .special_token_id("image_pad")
        .ok_or("Required token missing: <|image_pad|>")?;
    let vision_start = tokenizer
        .special_token_id("vision_start")
        .ok_or("Required token missing: <|vision_start|>")?;
    let vision_end = tokenizer
        .special_token_id("vision_end")
        .ok_or("Required token missing: <|vision_end|>")?;
    let mut content_tokens = vec![vision_start];
    content_tokens.extend(std::iter::repeat(image_token_id).take(n_vis_tokens));
    content_tokens.push(vision_end);
    content_tokens.extend(tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: false,
            parse_special: false,
        },
    ));
    let mut prompt_ids = Vec::new();
    append_qwen_message_tokens(&mut prompt_ids, &tokenizer, "user", &content_tokens)?;
    append_qwen_assistant_prefix(&mut prompt_ids, &tokenizer, false)?;
    let image_grids: Vec<VisionGrid> = vec![VisionGrid {
        grid_t: grid.grid_t,
        grid_h: grid.grid_h,
        grid_w: grid.grid_w,
        patch_size: grid.patch_size,
        merge_size: grid.merge_size,
    }];
    let image_token_id_u32 = image_token_id;
    let image_token_id_i32 = i32::try_from(image_token_id_u32)
        .map_err(|_| format!("Token ID {image_token_id_u32} exceeds i32"))?;
    let (prompt_positions, _next_text_position) = build_qwen35_positions(
        &prompt_ids,
        Some(image_token_id_u32),
        &image_grids,
    )?;
    let prompt_tokens: Vec<i32> = prompt_ids
        .iter()
        .copied()
        .map(|id| i32::try_from(id).map_err(|_| format!("Token ID {id} exceeds i32")))
        .collect::<Result<_, _>>()?;

    let mut llm = Qwen35Model::from_source(llm_source)
        .map_err(|error| format!("Failed to parse Qwen3.5 model: {error}"))?;
    let max_seq = (prompt_tokens.len() + 1).min(llm.config.n_ctx).min(max_context);
    let prompt_embd = inject_vision_embeddings(
        &llm,
        &prompt_tokens,
        Some(image_token_id_i32),
        &vis_embeddings,
        n_vis_tokens,
        llm.config.n_embd,
    )?;
    let t0 = std::time::Instant::now();
    let mut session = Qwen35Session::new_with_prefill_batch_size(
        &mut llm,
        max_seq,
        prefill_batch_size,
        std::sync::Arc::clone(&pool),
    )?;
    let logits = session.step(&prompt_embd, prompt_tokens.len(), &prompt_positions)?;
    Ok((logits, t0.elapsed()))
}

pub fn run_multimodal(
    llm_source: &dyn TensorSource,
    model_path: &Path,
    mmproj_path: Option<&Path>,
    image_path: Option<&Path>,
    audio_path: Option<&Path>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    prefill_batch_size: usize,
    max_context: usize,
    repetition_penalty: f32,
) -> Result<(), String> {
    run_multimodal_with_video_ref(
        llm_source,
        model_path,
        mmproj_path,
        image_path,
        None,
        audio_path,
        prompt,
        max_tokens,
        temperature,
        n_threads_arg,
        prefill_batch_size,
        max_context,
        repetition_penalty,
        None,
    )
}

pub fn run_multimodal_with_video(
    llm_source: Arc<dyn TensorSource>,
    model_path: &Path,
    mmproj_path: Option<&Path>,
    image_path: Option<&Path>,
    video_path: Option<&Path>,
    audio_path: Option<&Path>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    prefill_batch_size: usize,
    max_context: usize,
    repetition_penalty: f32,
) -> Result<(), String> {
    let owned_source = Arc::clone(&llm_source);
    run_multimodal_with_video_ref(
        llm_source.as_ref(),
        model_path,
        mmproj_path,
        image_path,
        video_path,
        audio_path,
        prompt,
        max_tokens,
        temperature,
        n_threads_arg,
        prefill_batch_size,
        max_context,
        repetition_penalty,
        Some(owned_source),
    )
}

pub(super) fn run_gemma4_capture_text(
    _model_path: &Path,
    _mmproj_path: Option<&Path>,
    _image_path: Option<&Path>,
    _audio_path: Option<&Path>,
    _prompt: &str,
    _max_tokens: usize,
    _n_threads_arg: usize,
    _prefill_batch_size: usize,
) -> Result<String, String> {
    // Gemma4 multimodal path is implemented in `crate::models::gemma4`;
    // for now, the TTS post-processor is opt-in and the gemma4 capture
    // path is left as a TODO. The pipeline above documents this gap.
    Err("Omni → TTS capture-text is not yet implemented for gemma4".into())
}

pub(super) fn run_multimodal_with_video_ref(
    llm_source: &dyn TensorSource,
    model_path: &Path,
    mmproj_path: Option<&Path>,
    image_path: Option<&Path>,
    video_path: Option<&Path>,
    audio_path: Option<&Path>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    prefill_batch_size: usize,
    max_context: usize,
    _repetition_penalty: f32,
    model_source: Option<Arc<dyn TensorSource>>,
) -> Result<(), String> {
    let arch = llm_source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();
    validate_gemma4_temperature(arch, temperature)?;
    if arch == "gemma4" {
        if video_path.is_some() {
            return Err("Gemma4 multimodal generation does not support --video".into());
        }
        return crate::models::gemma4::run_gemma4(crate::models::gemma4::Gemma4Request {
            model: model_path,
            mmproj: mmproj_path,
            image: image_path,
            audio: audio_path,
            prompt,
            max_tokens,
            threads: n_threads_arg,
            kv_format: KvFormat::F32,
            prefill_batch_size,
        });
    }
    if matches!(arch, "qwen2vl" | "qwen3vl" | "qwen3vlmoe")
        && (image_path.is_some() || video_path.is_some() || audio_path.is_some())
    {
        return run_qwen3_family_multimodal(
            llm_source,
            model_source.ok_or("Qwen multimodal routing requires an owned model source")?,
            mmproj_path.ok_or("multimodal Qwen models require --mmproj")?,
            image_path,
            video_path,
            audio_path,
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            prefill_batch_size,
        )
        .map(|text| {
            // Multimodal text is already printed inside the function;
            // this branch swallows the value for the caller that only
            // expects a unit result.
            let _ = text;
        });
    }
    if audio_path.is_some() {
        return Err(format!(
            "Only gemma4 architecture is supported for multimodal audio, got: {arch}"
        ));
    }
    println!("LLM arch: {}", arch);
    if arch == "lfm2" {
        // LFM2.5-VL: mmproj carries the SigLIP encoder + LFM2 projector.
        let mmproj = mmproj_path.ok_or("LFM2.5-VL requires --mmproj")?;
        let mmproj_source = open_model_source(mmproj, ComponentRole::Mmproj)
            .map_err(|e| format!("Failed to load mmproj {}: {e}", mmproj.display()))?;
        let image = image_path.ok_or("LFM2.5-VL requires --image")?;
        return crate::models::lfm2::vision::run_multimodal(
            llm_source,
            mmproj_source.as_ref(),
            image,
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            crate::core::scratchpad::KvFormat::F16,
            max_context,
        );
    }
    if arch != "qwen35" && arch != "qwen3vl" {
        return Err(format!(
            "Only qwen35 and qwen3vl architectures are supported for multimodal, got: {arch}"
        ));
    }

    let t_img_start = std::time::Instant::now();
    let (image_grid, vis_embeddings_vec) = if let Some(image_path) = image_path {
        let projector_path = mmproj_path.unwrap_or(model_path);
        println!("Loading mmproj {} ...", projector_path.display());
        let mmproj_source =
            open_model_source(projector_path, ComponentRole::Mmproj).map_err(|error| {
                if mmproj_path.is_none() {
                    format!(
                        "Model {} has no bundled mmproj; pass --mmproj: {error}",
                        model_path.display()
                    )
                } else {
                    format!(
                        "Failed to load mmproj {}: {error}",
                        projector_path.display()
                    )
                }
            })?;

        if arch == "qwen3vl" {
            let mut encoder = VisionEncoder3vl::from_source(mmproj_source.as_ref())
                .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
            encoder.precompute();
            println!(
                "Vision encoder loaded: {} layers, n_embd={}, image_size={}, patch_size={}, merge={}",
                encoder.config.n_layer,
                encoder.config.n_embd,
                encoder.config.image_size,
                encoder.config.patch_size,
                encoder.config.spatial_merge_size
            );
            let t_load = std::time::Instant::now();
            let image = decode_image(image_path)?;
            let t_load = t_load.elapsed();
            let original_w = usize::try_from(image.width())
                .map_err(|_| "Original image width does not fit usize")?;
            let original_h = usize::try_from(image.height())
                .map_err(|_| "Original image height does not fit usize")?;
            let grid = qwen3vl_smart_resize(original_w, original_h, &encoder.config)?;
            let t_preproc = std::time::Instant::now();
            let pixels = normalize_resized_image(
                &image,
                grid.image_width(),
                grid.image_height(),
                &encoder.config.image_mean,
                &encoder.config.image_std,
            )?;
            let t_preproc = t_preproc.elapsed();
            println!(
                "Image resized to {}x{} ({} vision tokens)",
                grid.image_width(),
                grid.image_height(),
                grid.token_count()
            );
            let projection_dim = encoder.config.projection_dim;
            let mut scratch = VisionScratchpad3vl::new(&encoder.config);
            println!("Encoding image...");
            let t_venc = std::time::Instant::now();
            let encoded_grid = encoder.encode_image(
                &pixels,
                grid.image_width(),
                grid.image_height(),
                &mut scratch,
            )?;
            let t_venc = t_venc.elapsed();
            if encoded_grid != grid {
                return Err(format!(
                    "Vision grid mismatch: preprocess={grid:?}, encoder={encoded_grid:?}"
                ));
            }
            let projected_len = grid
                .token_count()
                .checked_mul(projection_dim)
                .ok_or("Projected vision length overflow")?;
            if scratch.projected.len() != projected_len {
                return Err(format!(
                    "Projected vision length mismatch: expected {projected_len}, got {}",
                    scratch.projected.len()
                ));
            }
            let t_img_total = t_img_start.elapsed();
            eprintln!(
                "[pipeline-timing] image_total={:.3}s  image_load={:.3}s ({:.0}%)  preprocess={:.3}s ({:.0}%)  vision_encode={:.3}s ({:.0}%)",
                t_img_total.as_secs_f64(),
                t_load.as_secs_f64(), t_load.as_secs_f64()/t_img_total.as_secs_f64()*100.0,
                t_preproc.as_secs_f64(), t_preproc.as_secs_f64()/t_img_total.as_secs_f64()*100.0,
                t_venc.as_secs_f64(), t_venc.as_secs_f64()/t_img_total.as_secs_f64()*100.0,
            );
            println!(
                "Vision tokens: {} (dim={})",
                grid.token_count(),
                projection_dim
            );
            (
                Some(VisionGrid {
                    grid_t: grid.grid_t,
                    grid_h: grid.grid_h,
                    grid_w: grid.grid_w,
                    patch_size: grid.patch_size,
                    merge_size: grid.merge_size,
                }),
                scratch.projected.clone(),
            )
        } else {
            let mut encoder = VisionEncoder35::from_source(mmproj_source.as_ref())
                .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
            encoder.precompute();
            println!(
                "Vision encoder loaded: {} layers, n_embd={}, image_size={}, patch_size={}, merge={}",
                encoder.config.n_layer,
                encoder.config.n_embd,
                encoder.config.image_size,
                encoder.config.patch_size,
                encoder.config.spatial_merge_size
            );
            let t_load = std::time::Instant::now();
            let image = decode_image(image_path)?;
            let t_load = t_load.elapsed();
            let original_w = usize::try_from(image.width())
                .map_err(|_| "Original image width does not fit usize")?;
            let original_h = usize::try_from(image.height())
                .map_err(|_| "Original image height does not fit usize")?;
            let grid = qwen35_smart_resize(original_w, original_h, &encoder.config)?;
            let t_preproc = std::time::Instant::now();
            let venc_pool = Arc::new(ComputePool::new(resolve_thread_count(
                n_threads_arg,
                std::thread::available_parallelism()
                    .map(|value| value.get())
                    .unwrap_or(1),
            )));
            let pixels = normalize_resized_image(
                &image,
                grid.image_width(),
                grid.image_height(),
                &encoder.config.image_mean,
                &encoder.config.image_std,
            )?;
            let t_preproc = t_preproc.elapsed();
            println!(
                "Image resized to {}x{} ({} vision tokens)",
                grid.image_width(),
                grid.image_height(),
                grid.token_count()
            );
            let projection_dim = encoder.config.projection_dim;
            let mut scratch = VisionScratchpad35::new(&encoder.config);
            println!("Encoding image...");
            let t_venc = std::time::Instant::now();
            let encoded_grid = encoder.encode_image(
                &pixels,
                grid.image_width(),
                grid.image_height(),
                &mut scratch,
                &venc_pool,
            )?;
            let t_venc = t_venc.elapsed();
            if encoded_grid != grid {
                return Err(format!(
                    "Vision grid mismatch: preprocess={grid:?}, encoder={encoded_grid:?}"
                ));
            }
            let projected_len = grid
                .token_count()
                .checked_mul(projection_dim)
                .ok_or("Projected vision length overflow")?;
            if scratch.projected.len() != projected_len {
                return Err(format!(
                    "Projected vision length mismatch: expected {projected_len}, got {}",
                    scratch.projected.len()
                ));
            }
            let t_img_total = t_img_start.elapsed();
            eprintln!(
                "[pipeline-timing] image_total={:.3}s  image_load={:.3}s ({:.0}%)  preprocess={:.3}s ({:.0}%)  vision_encode={:.3}s ({:.0}%)",
                t_img_total.as_secs_f64(),
                t_load.as_secs_f64(), t_load.as_secs_f64()/t_img_total.as_secs_f64()*100.0,
                t_preproc.as_secs_f64(), t_preproc.as_secs_f64()/t_img_total.as_secs_f64()*100.0,
                t_venc.as_secs_f64(), t_venc.as_secs_f64()/t_img_total.as_secs_f64()*100.0,
            );
            println!(
                "Vision tokens: {} (dim={})",
                grid.token_count(),
                projection_dim
            );
            (Some(grid), scratch.projected.clone())
        }
    } else {
        (None, Vec::new())
    };
    let n_vis_tokens = image_grid.map(|g| g.token_count()).unwrap_or(0);
    let vis_embeddings = &vis_embeddings_vec[..];
    if image_grid.is_some() {
        println!(
            "First 5 vision embedding values: {:?}",
            &vis_embeddings[..5.min(vis_embeddings.len())]
        );
    }

    let mut llm = Qwen35Model::from_source(llm_source)
        .map_err(|error| format!("Failed to parse Qwen3.5 model: {error}"))?;
    let model_name = llm_source
        .metadata("general.name")
        .and_then(|value| value.to_string_val())
        .unwrap_or("Qwen3.5-family");
    println!("{model_name} model loaded: {} layers, n_embd={}, n_head={}, n_ff={}, rope_freq_base={}, rope_sections={:?}, rope_dim_count={}", llm.config.n_layer, llm.config.n_embd, llm.config.n_head, llm.config.n_ff, llm.config.rope_freq_base, llm.config.rope_dimension_sections, llm.config.rope_dimension_count);

    let tokenizer = BPETokenizer::from_gguf_metadata(|k| llm_source.metadata(k).cloned())
        .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?;
    let image_token_id = if image_grid.is_some() {
        Some(
            tokenizer
                .special_token_id("image_pad")
                .ok_or("Required token missing: <|image_pad|>")?,
        )
    } else {
        None
    };

    let mut content_tokens = Vec::new();
    if let Some(image_token_id) = image_token_id {
        content_tokens.push(
            tokenizer
                .special_token_id("vision_start")
                .ok_or("Required token missing: <|vision_start|>")?,
        );
        content_tokens.extend(std::iter::repeat(image_token_id).take(n_vis_tokens));
        content_tokens.push(
            tokenizer
                .special_token_id("vision_end")
                .ok_or("Required token missing: <|vision_end|>")?,
        );
    }
    content_tokens.extend(tokenizer.encode(
        prompt,
        EncodeOptions {
            add_special: false,
            parse_special: false,
        },
    ));

    let mut prompt_ids = Vec::new();
    append_qwen_message_tokens(&mut prompt_ids, &tokenizer, "user", &content_tokens)?;
    append_qwen_assistant_prefix(&mut prompt_ids, &tokenizer, false)?;
    let image_grids: Vec<crate::models::qwen35::vision::VisionGrid> =
        image_grid.iter().copied().collect();
    let (prompt_positions, mut next_text_position) =
        build_qwen35_positions(&prompt_ids, image_token_id, &image_grids)?;
    let prompt_tokens: Vec<i32> = prompt_ids
        .iter()
        .copied()
        .map(|id| i32::try_from(id).map_err(|_| format!("Token ID {id} exceeds i32")))
        .collect::<Result<_, _>>()?;

    let projected_count = if vis_embeddings.is_empty() {
        0
    } else {
        let projection_dim = llm.config.n_embd;
        if vis_embeddings.len() % projection_dim != 0 {
            return Err("Projected vision embeddings are not row aligned".into());
        }
        vis_embeddings.len() / projection_dim
    };
    if projected_count != n_vis_tokens || prompt_positions.len() != prompt_tokens.len() {
        return Err(format!(
            "Vision/position count mismatch: placeholders={n_vis_tokens}, projected={projected_count}, positions={}, tokens={}",
            prompt_positions.len(),
            prompt_tokens.len()
        ));
    }
    let image_token_id = image_token_id
        .map(|id| i32::try_from(id).map_err(|_| format!("Token ID {id} exceeds i32")))
        .transpose()?;

    println!(
        "Prompt tokens: {} (including {} vision placeholders)",
        prompt_tokens.len(),
        n_vis_tokens
    );
    eprintln!(
        "[RUST_TOKENS] n={} ids={:?}",
        prompt_tokens.len(),
        prompt_tokens
    );

    let max_seq = (prompt_tokens.len() + max_tokens).min(llm.config.n_ctx);
    let prompt_embd = inject_vision_embeddings(
        &llm,
        &prompt_tokens,
        image_token_id,
        vis_embeddings,
        n_vis_tokens,
        llm.config.n_embd,
    )?;
    #[cfg(feature = "parity-trace")]
    {
        let flat_positions = prompt_positions
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        crate::parity_trace::report(crate::parity_trace::token_ids(
            "qwen35.prompt_ids",
            &prompt_ids,
        ));
        crate::parity_trace::report(crate::parity_trace::usize_values(
            "qwen35.mrope_positions",
            &[prompt_positions.len(), 4],
            &flat_positions,
        ));
        crate::parity_trace::report(crate::parity_trace::bool_values(
            "qwen35.layer_is_recurrent",
            &llm.config.is_recurrent,
        ));
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "qwen35.embedding",
            None,
            &[prompt_tokens.len(), llm.config.n_embd],
            &prompt_embd,
        ));
    }

    let n_prompt = prompt_tokens.len();
    let mut all_tokens = prompt_tokens.clone();

    let n_threads = if n_threads_arg > 0 { n_threads_arg } else { 8 };
    let pool = std::sync::Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());
    let mut session = Qwen35Session::new_with_prefill_batch_size(
        &mut llm,
        max_seq,
        prefill_batch_size,
        std::sync::Arc::clone(&pool),
    )?;

    let mut generated = String::new();
    #[cfg(feature = "parity-trace")]
    let mut greedy_token_ids = Vec::with_capacity(max_tokens);
    let mut decoder = tokenizer.streaming_decoder(false);
    println!("\n--- Generation ---");
    let t_gen_start = std::time::Instant::now();
    let mut t_prompt = 0.0;
    let mut t_decode = 0.0;
    let mut prefill_evals = 0usize;
    let mut decode_evals = 0usize;

    for step in 0..max_tokens {
        let t0 = std::time::Instant::now();
        let tokens = if step == 0 {
            &prompt_tokens
        } else {
            &all_tokens[all_tokens.len() - 1..all_tokens.len() - 1 + 1]
        };
        let n_tok = tokens.len();

        let decode_embedding;
        let embeddings = if step == 0 {
            prompt_embd.as_slice()
        } else {
            let token_id = u32::try_from(tokens[0])
                .map_err(|_| format!("invalid negative token id {}", tokens[0]))?;
            decode_embedding = session.embed_tokens(&[token_id])?;
            decode_embedding.as_slice()
        };

        let decode_position = [[
            next_text_position,
            next_text_position,
            next_text_position,
            0,
        ]];
        let positions = if step == 0 {
            &prompt_positions[..]
        } else {
            &decode_position[..]
        };
        let logits = session.step(embeddings, n_tok, positions)?;
        // Parity debugging: top-10 logits per step when RUST_QWEN35_DEBUG_LOGITS
        // is set (mirrors the other trunks).
        if std::env::var("RUST_QWEN35_DEBUG_LOGITS").is_ok() {
            let mut idxs: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            idxs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let tag = if step == 0 {
                prompt_tokens.len() - 1
            } else {
                prompt_tokens.len() - 1 + step
            };
            let mut line = format!("RUST_LOGITS step={} top10:", tag);
            for k in 0..10 {
                line.push_str(&format!(" {}:{:.5}", idxs[k].0, idxs[k].1));
            }
            line.push('\n');
            let _ = io::stderr().write_all(line.as_bytes());
            let _ = io::stderr().flush();
        }
        let t_step = t0.elapsed().as_secs_f64();
        if step == 0 {
            t_prompt += t_step;
            prefill_evals += 1;
        } else {
            t_decode += t_step;
            decode_evals += 1;
            next_text_position = next_text_position
                .checked_add(1)
                .ok_or("Qwen3.5 decode position overflow")?;
        }

        let next_token = if temperature <= 0.0 {
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as i32)
                .unwrap_or(0)
        } else {
            sample_token(&logits, temperature)
        };
        #[cfg(feature = "parity-trace")]
        greedy_token_ids.push(next_token as u32);

        // Stop on EOS, on qwen-style `<|im_end|>`, and on K2-Horizon's
        // `<|ifm|im_end|>`. The K2-Horizon literal is registered via
        // `K2_HORIZON_SEMANTIC_TOKENS` so `special_token_id("ifm|im_end")`
        // returns the right id; for other architectures that lookup is
        // a no-op (the token isn't in their vocab).
        if next_token >= 0 {
            let nt = next_token as u32;
            if tokenizer.eos_id() == Some(nt)
                || tokenizer.special_token_id("im_end") == Some(nt)
                || tokenizer.special_token_id("ifm|im_end") == Some(nt)
            {
                break;
            }
        }

        let token_str = decoder.push(next_token as u32);
        generated.push_str(&token_str);
        print!("{}", token_str);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        all_tokens.push(next_token);
    }
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::token_ids(
        "qwen35.greedy_token_ids",
        &greedy_token_ids,
    ));

    let tail = decoder.finish();
    generated.push_str(&tail);
    if !tail.is_empty() {
        print!("{}", tail);
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }

    let gen_ms = t_gen_start.elapsed().as_millis();
    let n_gen = all_tokens.len() - n_prompt;
    let tok_s = if gen_ms > 0 {
        n_gen as f64 / gen_ms as f64 * 1000.0
    } else {
        0.0
    };
    let per_second = |count: usize, secs: f64| {
        if secs > 0.0 {
            count as f64 / secs
        } else {
            0.0
        }
    };
    println!("\n--- End ---");
    eprintln!(
        "Prompt: {:.1} t/s | Generation: {:.1} t/s | end-to-end: {:.1} tok/s",
        per_second(prefill_evals, t_prompt),
        per_second(decode_evals, t_decode),
        tok_s
    );
    Ok(())
}
