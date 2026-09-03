use crate::app::cli::{resolve_thread_count, KvFormat};
use crate::core::scratchpad::{ExecutionScratchpad, KvCache};
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
use crate::models::qwen35::{build_qwen35_positions, Qwen35Model};
use crate::ops::embedding_lookup;
use crate::ops::kernel::Kernel;
use crate::prompt::{
    append_qwen_assistant_prefix, append_qwen_message_tokens, build_hunyuan_chat_prompt,
    build_lfm2_chat_prompt, build_qwen_chat_prompt, HunyuanMessage, Lfm2Message, QwenMessage,
};
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn validate_gemma4_temperature(arch: &str, temperature: f32) -> Result<(), String> {
    if arch == "gemma4" && temperature != 0.0 {
        return Err("Gemma4 requires greedy decoding; --temp must be 0".into());
    }
    Ok(())
}

pub fn run_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    thinking: bool,
    bench: bool,
    profile: bool,
    kv_format: KvFormat,
) -> Result<(), String> {
    let arch = source
        .metadata("general.architecture")
        .and_then(|v| v.to_string_val())
        .unwrap_or_default();

    if arch == "hunyuan-dense" {
        crate::models::qwen3::hunyuan::run_inference(
            source.clone(),
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            profile,
            kv_format,
        )
    } else if arch == "lfm2" {
        let is_lfm25 = source
            .metadata("general.basename")
            .and_then(|v| v.to_string_val())
            .map(|v| v.contains("2.5"))
            .unwrap_or(false);

        if is_lfm25 {
            crate::models::lfm25::run_inference(
                source.as_ref(),
                prompt,
                max_tokens,
                temperature,
                n_threads_arg,
                profile,
                kv_format,
            )
        } else {
            crate::models::lfm2::run_inference(
                source.as_ref(),
                prompt,
                max_tokens,
                temperature,
                n_threads_arg,
                profile,
                kv_format,
            )
        }
    } else if arch == "lfm2moe" {
        crate::models::lfm2moe::run_inference(
            source.as_ref(),
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            profile,
            kv_format,
        )
    } else if arch == "llama" || arch == "granite" || arch == "nanbeige" {
        crate::models::llama::run_inference(
            source.as_ref(),
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            bench,
            profile,
            kv_format,
        )
    } else {
        crate::models::qwen3::text::run_inference(
            source.clone(),
            prompt,
            max_tokens,
            temperature,
            n_threads_arg,
            thinking,
            bench,
            profile,
            kv_format,
        )
    }
}

pub fn run_interactive(
    source: Arc<dyn TensorSource>,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
) -> Result<(), String> {
    println!("=== RustModelInference Interactive Mode ===");
    println!("Type your prompt and press Enter. Ctrl+C to exit.\n");

    loop {
        print!("> ");
        io::stdout()
            .flush()
            .map_err(|error| format!("Failed to flush prompt: {error}"))?;
        let mut line = String::new();
        if io::stdin()
            .read_line(&mut line)
            .map_err(|error| format!("Failed to read prompt: {error}"))?
            == 0
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        run_inference(
            source.clone(),
            line,
            max_tokens,
            temperature,
            n_threads_arg,
            false,
            false,
            false,
            KvFormat::F16,
        )?;
        println!();
    }
    Ok(())
}

pub fn run_shared_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    thinking: bool,
) -> Result<(), String> {
    crate::models::qwen3::run_shared_inference(
        source,
        prompt,
        max_tokens,
        temperature,
        n_threads_arg,
        thinking,
    )
}

pub fn inject_vision_embeddings(
    llm: &Qwen35Model,
    tokens: &[i32],
    image_token_id: Option<i32>,
    vis_embd: &[f32],
    _n_vis_tokens: usize,
    proj_dim: usize,
) -> Vec<f32> {
    let n_embd = llm.config.n_embd;
    let n_tokens = tokens.len();
    let mut embeddings = vec![0.0f32; n_tokens * n_embd];

    let mut vis_idx = 0;

    for t in 0..n_tokens {
        if image_token_id == Some(tokens[t]) && vis_idx * proj_dim < vis_embd.len() {
            let embd_off = t * n_embd;
            let vis_off = vis_idx * proj_dim;
            if proj_dim == n_embd {
                embeddings[embd_off..embd_off + n_embd]
                    .copy_from_slice(&vis_embd[vis_off..vis_off + n_embd]);
            } else {
                for e in 0..n_embd.min(proj_dim) {
                    embeddings[embd_off + e] = vis_embd[vis_off + e];
                }
            }
            vis_idx += 1;
        } else {
            let tok = tokens[t] as usize;
            let tok_off = tok * n_embd;
            let embd_off = t * n_embd;
            for e in 0..n_embd {
                if tok_off + e < llm.tok_embd.len() {
                    embeddings[embd_off + e] = llm.tok_embd[tok_off + e];
                }
            }
        }
    }

    embeddings
}

pub fn sample_token(logits: &[f32], temperature: f32) -> i32 {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut probs = vec![0.0f32; logits.len()];
    for (i, l) in logits.iter().enumerate() {
        probs[i] = ((l - max_logit) / temperature).exp();
        sum += probs[i];
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let r = 0.5f32;
    let mut cumsum = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i as i32;
        }
    }
    (logits.len() - 1) as i32
}

pub fn decode_image(path: &Path) -> Result<image::DynamicImage, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("Failed to read image {}: {error}", path.display()))?;
    image::load_from_memory(&bytes)
        .map_err(|error| format!("Failed to decode image {}: {error}", path.display()))
}

pub fn normalize_resized_image(
    image: &image::DynamicImage,
    target_w: usize,
    target_h: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Result<Vec<f32>, String> {
    if std.iter().any(|value| *value == 0.0) {
        return Err("Vision normalization std must be nonzero".into());
    }
    let source = image.to_rgb8();
    let resized = crate::models::gemma4::vision::resize_bicubic_pillow(
        source.as_raw(),
        source.width() as usize,
        source.height() as usize,
        target_w,
        target_h,
    )?;
    let output_len = target_w
        .checked_mul(target_h)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("Normalized image length overflow")?;
    let mut output = vec![0.0f32; output_len];
    for y in 0..target_h {
        for x in 0..target_w {
            let offset = (y * target_w + x) * 3;
            for channel in 0..3 {
                output[offset + channel] =
                    (f32::from(resized[offset + channel]) / 255.0 - mean[channel]) / std[channel];
            }
        }
    }
    Ok(output)
}

fn build_qwen3_media_positions(
    token_ids: &[u32],
    placeholder_id: u32,
    grid_shapes: &[(usize, usize)],
) -> Result<Vec<[usize; 4]>, String> {
    let mut positions = Vec::with_capacity(token_ids.len());
    let mut next = 0usize;
    let mut token = 0usize;
    let mut grid_index = 0usize;
    while token < token_ids.len() {
        if token_ids[token] != placeholder_id {
            positions.push([next, next, next, 0]);
            next = next.checked_add(1).ok_or("Qwen media position overflow")?;
            token += 1;
            continue;
        }
        if grid_shapes.is_empty() {
            positions.push([next, next, next, 0]);
            next = next.checked_add(1).ok_or("Qwen audio position overflow")?;
            token += 1;
            continue;
        }
        let (grid_h, grid_w) = *grid_shapes
            .get(grid_index)
            .ok_or("Media placeholder has no matching vision grid")?;
        let count = grid_h
            .checked_mul(grid_w)
            .ok_or("Qwen media grid token count overflow")?;
        let end = token
            .checked_add(count)
            .ok_or("Qwen media placeholder range overflow")?;
        if count == 0
            || end > token_ids.len()
            || token_ids[token..end].iter().any(|id| *id != placeholder_id)
        {
            return Err(format!(
                "Vision grid {grid_index} requires {count} contiguous placeholders"
            ));
        }
        let base = next;
        for index in 0..count {
            let row = index / grid_w;
            let column = index % grid_w;
            positions.push([
                base,
                base.checked_add(row)
                    .ok_or("Qwen media row position overflow")?,
                base.checked_add(column)
                    .ok_or("Qwen media column position overflow")?,
                0,
            ]);
        }
        next = base
            .checked_add(grid_h.max(grid_w))
            .ok_or("Qwen media logical position overflow")?;
        token = end;
        grid_index += 1;
    }
    if !grid_shapes.is_empty() && grid_index != grid_shapes.len() {
        return Err(format!(
            "Unused vision grids: consumed {grid_index}, provided {}",
            grid_shapes.len()
        ));
    }
    Ok(positions)
}

fn run_qwen3_family_multimodal(
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
) -> Result<(), String> {
    validate_single_qwen_media(
        image_path.is_some(),
        video_path.is_some(),
        audio_path.is_some(),
    )?;
    let arch = llm_source
        .metadata("general.architecture")
        .and_then(|value| value.to_string_val())
        .unwrap_or_default();
    let mmproj: Arc<dyn TensorSource> = Arc::from(
        open_model_source(mmproj_path, ComponentRole::Mmproj)
            .map_err(|error| format!("Failed to load mmproj {}: {error}", mmproj_path.display()))?,
    );
    let media_kind = if audio_path.is_some() {
        crate::app::omni::MediaKind::Audio
    } else if video_path.is_some() {
        crate::app::omni::MediaKind::Video
    } else {
        crate::app::omni::MediaKind::Image
    };
    let family = crate::app::omni::validate_mmproj_capabilities(arch, mmproj.as_ref(), media_kind)?;
    let mut media = Vec::new();
    let mut media_deepstack_layers: Vec<Vec<f32>> = Vec::new();
    let mut media_grid_shapes = Vec::new();
    if let Some(audio_path) = audio_path {
        let samples = crate::app::omni::decode_audio(audio_path)?;
        media =
            crate::models::qwen3::omni::encode_audio(Arc::clone(&mmproj), &samples, n_threads_arg)?;
    } else {
        let mut frames = if let Some(path) = image_path {
            vec![decode_image(path)?]
        } else {
            crate::app::omni::decode_video(video_path.ok_or("missing image or video input")?)?
        };
        let is_video = video_path.is_some();
        let (first_w, first_h) = {
            let first = frames.first().ok_or("media produced no frames")?;
            (first.width() as usize, first.height() as usize)
        };
        let (grid_w, grid_h) = match family {
            crate::app::omni::ProjectorFamily::Qwen3VlMerger => {
                let encoder = VisionEncoder3vl::from_source(mmproj.as_ref())
                    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
                let grid = qwen3vl_smart_resize(first_w, first_h, &encoder.config)?;
                (grid.image_width(), grid.image_height())
            }
            crate::app::omni::ProjectorFamily::Qwen25Omni => {
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
            crate::app::omni::ProjectorFamily::Qwen3VlMerger => {
                let encoder = VisionEncoder3vl::from_source(mmproj.as_ref())
                    .map_err(|error| format!("Failed to parse vision encoder: {error}"))?;
                (encoder.config.image_mean, encoder.config.image_std)
            }
            crate::app::omni::ProjectorFamily::Qwen25Omni => {
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
            crate::app::omni::ProjectorFamily::Qwen3VlMerger => {
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
            crate::app::omni::ProjectorFamily::Qwen25Omni => {
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
    let pool = Arc::new(ComputePool::new(resolve_thread_count(
        n_threads_arg,
        std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1),
    )));
    let model = Qwen3Model::from_source(model_source, Arc::clone(&tokenizer), pool)?;
    let width = model.config().n_embd;
    if media.len() % width != 0 {
        return Err(format!(
            "Media projector width does not match model width {width}"
        ));
    }
    let (start_name, pad_name, end_name) = match media_kind {
        crate::app::omni::MediaKind::Audio => ("audio_start", "audio_pad", "audio_end"),
        crate::app::omni::MediaKind::Image => ("vision_start", "image_pad", "vision_end"),
        crate::app::omni::MediaKind::Video => ("vision_start", "video_pad", "vision_end"),
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
        },
    )?;
    print!("{}", generation.text);
    io::stdout().flush().map_err(|error| error.to_string())?;
    println!();
    Ok(())
}

fn inject_qwen_media_embeddings(
    token_ids: &[u32],
    pad: u32,
    embeddings: &mut [f32],
    media: &[f32],
    media_deepstack: &[f32],
    width: usize,
) -> Result<Vec<f32>, String> {
    if width == 0 || embeddings.len() != token_ids.len().saturating_mul(width) {
        return Err("Prompt embedding shape mismatch".into());
    }
    if media.len() % width != 0 {
        return Err("Media embeddings are not row aligned".into());
    }
    let media_rows = media.len() / width;
    let per_layer = media.len();
    if !media_deepstack.is_empty() && (per_layer == 0 || media_deepstack.len() % per_layer != 0) {
        return Err("Media deepstack embeddings are not layer aligned".into());
    }
    let deepstack_layers = if per_layer == 0 {
        0
    } else {
        media_deepstack.len() / per_layer
    };
    let mut deepstack = vec![0.0; deepstack_layers * embeddings.len()];
    let mut media_row = 0;
    for (token_index, (&token, row)) in token_ids
        .iter()
        .zip(embeddings.chunks_exact_mut(width))
        .enumerate()
    {
        if token != pad {
            continue;
        }
        if media_row >= media_rows {
            return Err("Media placeholder count exceeds projector rows".into());
        }
        row.copy_from_slice(&media[media_row * width..(media_row + 1) * width]);
        for layer in 0..deepstack_layers {
            let src = (layer * media_rows + media_row) * width;
            let dst = (layer * token_ids.len() + token_index) * width;
            deepstack[dst..dst + width].copy_from_slice(&media_deepstack[src..src + width]);
        }
        media_row += 1;
    }
    if media_row != media_rows {
        return Err(format!(
            "Media placeholder count mismatch: placeholders={media_row}, rows={media_rows}"
        ));
    }
    Ok(deepstack)
}

fn validate_single_qwen_media(image: bool, video: bool, audio: bool) -> Result<(), String> {
    if usize::from(image) + usize::from(video) + usize::from(audio) != 1 {
        return Err(
            "Qwen multimodal generation requires exactly one of --image, --video, or --audio"
                .into(),
        );
    }
    Ok(())
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
        Some(owned_source),
    )
}

fn run_multimodal_with_video_ref(
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
        );
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

    let llm = Qwen35Model::from_source(llm_source)
        .map_err(|error| format!("Failed to parse Qwen3.5 model: {error}"))?;
    println!("Qwen3.5 model loaded: {} layers, n_embd={}, n_head={}, n_ff={}, rope_freq_base={}, rope_sections={:?}, rope_dim_count={}", llm.config.n_layer, llm.config.n_embd, llm.config.n_head, llm.config.n_ff, llm.config.rope_freq_base, llm.config.rope_dimension_sections, llm.config.rope_dimension_count);

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
    let mut kv_cache = crate::core::scratchpad::KvCache::new_f32(
        llm.config.n_layer_impl(),
        max_seq,
        llm.config.n_embd_head() * llm.config.n_head_kv,
    );
    let mut llm_scratch = crate::models::qwen35::Qwen35Scratchpad::new(
        &llm.config,
        prompt_tokens.len().max(max_tokens),
    );

    let prompt_embd = inject_vision_embeddings(
        &llm,
        &prompt_tokens,
        image_token_id,
        vis_embeddings,
        n_vis_tokens,
        llm.config.n_embd,
    );

    let n_prompt = prompt_tokens.len();
    let mut all_tokens = prompt_tokens.clone();

    let n_threads = if n_threads_arg > 0 { n_threads_arg } else { 8 };
    let pool = std::sync::Arc::new(ComputePool::new(n_threads));
    eprintln!("compute pool: {} threads", pool.n_threads());

    let mut generated = String::new();
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

        if step == 0 {
            for t in 0..n_prompt {
                let embd_off = t * llm.config.n_embd;
                llm_scratch.x[embd_off..embd_off + llm.config.n_embd]
                    .copy_from_slice(&prompt_embd[embd_off..embd_off + llm.config.n_embd]);
            }
        } else {
            let tok = tokens[0] as usize;
            let tok_off = tok * llm.config.n_embd;
            for e in 0..llm.config.n_embd {
                if tok_off + e < llm.tok_embd.len() {
                    llm_scratch.x[e] = llm.tok_embd[tok_off + e];
                }
            }
        }

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
        let logits = llm.forward(n_tok, &mut kv_cache, &mut llm_scratch, &pool, positions)?;
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

        if next_token >= 0
            && (tokenizer.eos_id() == Some(next_token as u32)
                || tokenizer.special_token_id("im_end") == Some(next_token as u32))
        {
            break;
        }

        let token_str = decoder.push(next_token as u32);
        generated.push_str(&token_str);
        print!("{}", token_str);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        all_tokens.push(next_token);
    }

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

#[cfg(test)]
mod tests {
    use super::{
        build_qwen3_media_positions, inject_qwen_media_embeddings, run_multimodal,
        validate_single_qwen_media,
    };
    use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};
    use std::path::Path;

    struct ArchSource(MetaValue);

    impl TensorSource for ArchSource {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            (key == "general.architecture").then_some(&self.0)
        }

        fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
            None
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    #[test]
    fn gemma4_cli_rejects_nonzero_temperature() {
        let source = ArchSource(MetaValue::String("gemma4".into()));
        let error = run_multimodal(
            &source,
            Path::new("missing.gguf"),
            None,
            None,
            None,
            "hello",
            1,
            0.1,
            1,
        )
        .unwrap_err();
        assert!(error.contains("--temp"), "{error}");
    }

    #[test]
    fn qwen_media_positions_expand_grid_rows_and_audio_rows() {
        assert_eq!(
            build_qwen3_media_positions(&[10, 99, 99, 99, 99, 11], 99, &[(2, 2)]).unwrap(),
            vec![
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 2, 0],
                [1, 2, 1, 0],
                [1, 2, 2, 0],
                [3, 3, 3, 0]
            ]
        );
        assert_eq!(
            build_qwen3_media_positions(&[10, 99, 99, 11], 99, &[]).unwrap(),
            vec![[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 0], [3, 3, 3, 0]]
        );
        assert!(build_qwen3_media_positions(&[99, 99], 99, &[(1, 1)]).is_err());
    }

    #[test]
    fn qwen_multimodal_generation_rejects_combined_media() {
        assert!(validate_single_qwen_media(true, false, false).is_ok());
        assert!(validate_single_qwen_media(false, true, true).is_err());
    }

    #[test]
    fn qwen_media_injection_preserves_deepstack_layer_and_prompt_order() {
        let tokens = [1, 99, 2, 99];
        let mut embeddings = vec![0.0; tokens.len() * 2];
        let media = [10.0, 11.0, 20.0, 21.0];
        let media_deepstack = [
            100.0, 101.0, 200.0, 201.0, // layer 0 media rows
            300.0, 301.0, 400.0, 401.0, // layer 1 media rows
        ];

        let deepstack =
            inject_qwen_media_embeddings(&tokens, 99, &mut embeddings, &media, &media_deepstack, 2)
                .unwrap();

        assert_eq!(embeddings, [0.0, 0.0, 10.0, 11.0, 0.0, 0.0, 20.0, 21.0]);
        assert_eq!(
            deepstack,
            [
                0.0, 0.0, 100.0, 101.0, 0.0, 0.0, 200.0, 201.0, // layer 0
                0.0, 0.0, 300.0, 301.0, 0.0, 0.0, 400.0, 401.0, // layer 1
            ]
        );
    }
}
