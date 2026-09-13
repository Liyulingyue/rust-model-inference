use super::cli::{PlanningMode, QwenDriveCliOptions, QwenDriveHead};
use super::text::{decode_image, inject_vision_embeddings};
use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::models::qwen35::vision::{
    qwen_smart_resize, VisionEncoder, VisionGrid, VisionScratchpad,
};
use crate::models::qwen35::{build_qwen35_positions, Qwen35Model, Qwen35Session};
use crate::models::qwen_drive::perception::fpn::ViewGeometry;
use crate::models::qwen_drive::perception::ops::Tensor4;
use crate::models::qwen_drive::perception::QwenDrivePerception;
use crate::models::qwen_drive::planning::{PlanningExpert, TrajectoryBatch};
use crate::models::qwen_drive::scene::{
    read_perception_frame, read_planning_scenes, PerceptionContent, PerceptionFrame, PlanningScene,
};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const REASONING_REQUEST: &str =
    "\n\nGive a one-sentence brief reasoning of the ego's future driving decision ONLY.";
const PLANNER_LAYERS: [usize; 8] = [3, 7, 11, 15, 19, 23, 27, 31];
const HISTORY_IMAGE_PIXELS: usize = 174_080;
const CURRENT_IMAGE_PIXELS: usize = 921_600;

#[derive(Serialize)]
struct PlanningPrediction<'a> {
    token: &'a str,
    trajectories: Vec<Vec<[f32; 3]>>,
    reasoning: Option<&'a str>,
    reasoning_token_ids: &'a [u32],
}

#[derive(Serialize)]
struct PerceptionPrediction<'a> {
    token: &'a str,
    #[serde(flatten)]
    result: &'a crate::models::qwen_drive::perception::heads::PerceptionResult,
}

struct PrefillOutput {
    cache: Vec<crate::models::qwen35::Qwen35DenseKvSnapshot>,
    anchor: [usize; 3],
    reasoning: Option<String>,
    reasoning_token_ids: Vec<u32>,
}

fn architecture(source: &dyn TensorSource, expected: &str, label: &str) -> Result<(), String> {
    let actual = source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .ok_or_else(|| format!("{label} has no string general.architecture"))?;
    if actual != expected {
        return Err(format!(
            "Expected {label} architecture {expected}, got {actual}"
        ));
    }
    Ok(())
}

fn plain(tokenizer: &BPETokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(
        text,
        EncodeOptions {
            add_special: false,
            parse_special: false,
        },
    )
}

fn special(tokenizer: &BPETokenizer, name: &str) -> Result<u32, String> {
    tokenizer
        .special_token_id(name)
        .ok_or_else(|| format!("Qwen-Drive tokenizer is missing {name}"))
}

fn build_prompt(
    tokenizer: &BPETokenizer,
    scene: &PlanningScene,
    grids: &[VisionGrid],
    mode: PlanningMode,
) -> Result<Vec<u32>, String> {
    let image = special(tokenizer, "image_pad")?;
    let vision_start = special(tokenizer, "vision_start")?;
    let vision_end = special(tokenizer, "vision_end")?;
    let im_start = special(tokenizer, "im_start")?;
    let im_end = special(tokenizer, "im_end")?;
    if grids.len()
        != scene
            .views
            .iter()
            .map(|view| view.frames.len())
            .sum::<usize>()
    {
        return Err("Qwen-Drive image grid count does not match the scene".into());
    }

    let mut body = Vec::new();
    let mut grid = 0usize;
    for view in &scene.views {
        body.extend(plain(tokenizer, view.label));
        for frame in 0..view.frames.len() {
            body.extend(plain(tokenizer, &format!("frame: {frame}")));
            body.push(vision_start);
            body.extend(std::iter::repeat_n(image, grids[grid].token_count()));
            body.push(vision_end);
            grid += 1;
        }
    }
    body.extend(plain(tokenizer, &scene.instruction));
    if mode == PlanningMode::Reasoning {
        body.extend(plain(tokenizer, REASONING_REQUEST));
    }

    let mut tokens = vec![im_start];
    tokens.extend(plain(tokenizer, "user\n"));
    tokens.extend(body);
    tokens.push(im_end);
    tokens.extend(plain(tokenizer, "\n"));
    tokens.push(im_start);
    tokens.extend(plain(tokenizer, "assistant\n"));
    if mode == PlanningMode::Direct {
        tokens.push(im_end);
        tokens.extend(plain(tokenizer, "\n"));
    }
    Ok(tokens)
}

fn normalize_rgb(rgb: &[u8], mean: [f32; 3], std: [f32; 3]) -> Result<Vec<f32>, String> {
    if rgb.len() % 3 != 0 || std.contains(&0.0) {
        return Err("Invalid Qwen-Drive RGB normalization input".into());
    }
    let mut output = vec![0.0; rgb.len()];
    for (pixel, normalized) in rgb.chunks_exact(3).zip(output.chunks_exact_mut(3)) {
        for channel in 0..3 {
            normalized[channel] =
                (f32::from(pixel[channel]) / 255.0 - mean[channel]) / std[channel];
        }
    }
    Ok(output)
}

fn encode_images(
    encoder: &VisionEncoder<'_>,
    scene: &PlanningScene,
) -> Result<(Vec<f32>, Vec<VisionGrid>), String> {
    let per_view = scene
        .views
        .first()
        .map(|view| view.frames.len())
        .ok_or("Qwen-Drive scene has no camera views")?;
    let factor = encoder
        .config
        .patch_size
        .checked_mul(encoder.config.spatial_merge_size)
        .ok_or("Qwen-Drive vision factor overflow")?;
    let grid_pixel_limit = 12_800usize
        .checked_mul(factor)
        .and_then(|value| value.checked_mul(factor))
        .ok_or("Qwen-Drive vision pixel limit overflow")?;
    let min_pixels = 4usize
        .checked_mul(factor)
        .and_then(|value| value.checked_mul(factor))
        .ok_or("Qwen-Drive minimum pixel count overflow")?;
    let mut projected = Vec::new();
    let mut grids = Vec::new();
    let mut scratch = VisionScratchpad::new(&encoder.config);
    for view in &scene.views {
        if view.frames.len() != per_view {
            return Err("Qwen-Drive camera views have different frame counts".into());
        }
        for (frame_index, frame) in view.frames.iter().enumerate() {
            let image = decode_image(&frame.image)?.to_rgb8();
            let mut width = image.width() as usize;
            let mut height = image.height() as usize;
            let mut rgb = image.into_raw();
            let mut resize_config = encoder.config.clone();
            resize_config.image_min_pixels = min_pixels;
            resize_config.image_max_pixels = if let (Some(target_width), Some(target_height)) =
                (frame.resized_width, frame.resized_height)
            {
                rgb = crate::models::gemma4::vision::resize_bicubic_pillow(
                    &rgb,
                    width,
                    height,
                    target_width,
                    target_height,
                )?;
                width = target_width;
                height = target_height;
                grid_pixel_limit
            } else if frame_index + 1 == per_view {
                CURRENT_IMAGE_PIXELS
            } else {
                HISTORY_IMAGE_PIXELS
            };
            let grid = qwen_smart_resize(width, height, &resize_config)?;
            rgb = crate::models::gemma4::vision::resize_bicubic_pillow(
                &rgb,
                width,
                height,
                grid.image_width(),
                grid.image_height(),
            )?;
            let normalized =
                normalize_rgb(&rgb, encoder.config.image_mean, encoder.config.image_std)?;
            let actual = encoder.encode_image(
                &normalized,
                grid.image_width(),
                grid.image_height(),
                &mut scratch,
            )?;
            if actual != grid {
                return Err("Qwen-Drive vision encoder returned an unexpected grid".into());
            }
            projected.extend_from_slice(&scratch.projected);
            grids.push(grid);
        }
    }
    Ok((projected, grids))
}

fn build_perception_prompt(
    tokenizer: &BPETokenizer,
    frame: &PerceptionFrame,
    grids: &[VisionGrid],
) -> Result<Vec<u32>, String> {
    let image = special(tokenizer, "image_pad")?;
    let vision_start = special(tokenizer, "vision_start")?;
    let vision_end = special(tokenizer, "vision_end")?;
    let im_start = special(tokenizer, "im_start")?;
    let im_end = special(tokenizer, "im_end")?;
    let mut body = Vec::new();
    let mut image_index = 0usize;
    for item in &frame.content {
        match item {
            PerceptionContent::Text(text) => body.extend(plain(tokenizer, text)),
            PerceptionContent::Image { .. } => {
                let grid = grids
                    .get(image_index)
                    .ok_or("Qwen-Drive perception image grid count mismatch")?;
                body.push(vision_start);
                body.extend(std::iter::repeat_n(image, grid.token_count()));
                body.push(vision_end);
                image_index += 1;
            }
        }
    }
    if image_index != grids.len() {
        return Err("Qwen-Drive perception image grid count mismatch".into());
    }
    let mut tokens = vec![im_start];
    tokens.extend(plain(tokenizer, "user\n"));
    tokens.extend(body);
    tokens.push(im_end);
    tokens.extend(plain(tokenizer, "\n"));
    tokens.push(im_start);
    tokens.extend(plain(tokenizer, "assistant\n"));
    Ok(tokens)
}

fn encode_perception_images(
    encoder: &VisionEncoder<'_>,
    frame: &PerceptionFrame,
    image_size: [usize; 2],
    vit_dim: usize,
) -> Result<(Tensor4, Vec<f32>, Vec<VisionGrid>), String> {
    if encoder.config.n_embd != vit_dim {
        return Err(format!(
            "Qwen-Drive perception expects ViT width {vit_dim}, got {}",
            encoder.config.n_embd
        ));
    }
    let images = frame
        .content
        .iter()
        .filter_map(|item| match item {
            PerceptionContent::Image { path, .. } => Some(path),
            PerceptionContent::Text(_) => None,
        })
        .collect::<Vec<_>>();
    let cameras = images.len();
    let mut projected = Vec::new();
    let mut grids = Vec::with_capacity(cameras);
    let mut vit = Vec::new();
    let mut scratch = VisionScratchpad::new(&encoder.config);
    for (camera, path) in images.into_iter().enumerate() {
        let image = decode_image(path)?.to_rgb8();
        let rgb = crate::models::gemma4::vision::resize_bicubic_pillow(
            image.as_raw(),
            image.width() as usize,
            image.height() as usize,
            image_size[0],
            image_size[1],
        )?;
        let normalized = normalize_rgb(&rgb, encoder.config.image_mean, encoder.config.image_std)?;
        let grid = encoder.encode_image(&normalized, image_size[0], image_size[1], &mut scratch)?;
        if grids.first().is_some_and(|first| *first != grid) {
            return Err("Qwen-Drive perception cameras produced different vision grids".into());
        }
        let patch_height = grid.grid_h * grid.merge_size;
        let patch_width = grid.grid_w * grid.merge_size;
        let image_values = patch_height
            .checked_mul(patch_width)
            .and_then(|value| value.checked_mul(vit_dim))
            .ok_or("Qwen-Drive perception ViT feature length overflow")?;
        if scratch.merged.len() != image_values {
            return Err("Qwen-Drive perception pre-merge feature shape mismatch".into());
        }
        let start = vit.len();
        vit.resize(start + image_values, 0.0);
        for block_y in 0..grid.grid_h {
            for block_x in 0..grid.grid_w {
                for dy in 0..grid.merge_size {
                    for dx in 0..grid.merge_size {
                        let y = block_y * grid.merge_size + dy;
                        let x = block_x * grid.merge_size + dx;
                        let source = (((block_y * grid.grid_w + block_x) * grid.merge_size + dy)
                            * grid.merge_size
                            + dx)
                            * vit_dim;
                        for channel in 0..vit_dim {
                            let destination =
                                start + ((channel * patch_height + y) * patch_width + x);
                            vit[destination] = scratch.merged[source + channel];
                        }
                    }
                }
            }
        }
        projected.extend_from_slice(&scratch.projected);
        grids.push(grid);
        debug_assert_eq!(camera + 1, grids.len());
    }
    let grid = grids
        .first()
        .copied()
        .ok_or("Qwen-Drive perception frame has no images")?;
    Ok((
        Tensor4::new(
            vit,
            [
                cameras,
                vit_dim,
                grid.grid_h * grid.merge_size,
                grid.grid_w * grid.merge_size,
            ],
        )?,
        projected,
        grids,
    ))
}

fn prefill_perception(
    model: &mut Qwen35Model<'_>,
    tokenizer: &BPETokenizer,
    pool: Arc<ComputePool>,
    frame: &PerceptionFrame,
    projected: &[f32],
    grids: &[VisionGrid],
) -> Result<Tensor4, String> {
    let tokens = build_perception_prompt(tokenizer, frame, grids)?;
    let image = special(tokenizer, "image_pad")?;
    let signed_tokens = tokens
        .iter()
        .map(|&token| i32::try_from(token).map_err(|_| "Qwen-Drive token id exceeds i32"))
        .collect::<Result<Vec<_>, _>>()?;
    let embeddings = inject_vision_embeddings(
        model,
        &signed_tokens,
        Some(image as i32),
        projected,
        projected.len() / model.config.n_embd,
        model.config.n_embd,
    )?;
    let (positions, _) = build_qwen35_positions(&tokens, Some(image), grids)?;
    let width = model.config.n_embd;
    let mut session = Qwen35Session::new(model, tokens.len(), pool)?;
    session.step(&embeddings, tokens.len(), &positions)?;
    let hidden = session.last_hidden(tokens.len())?;
    let grid = grids
        .last()
        .copied()
        .ok_or("Qwen-Drive perception has no image grids")?;
    if grids.iter().any(|current| *current != grid) {
        return Err("Qwen-Drive perception image grids differ".into());
    }
    let per_image = grid.token_count();
    let cameras = frame.cam_order.len();
    let image_rows = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, &token)| (token == image).then_some(index))
        .collect::<Vec<_>>();
    let keep = cameras
        .checked_mul(per_image)
        .ok_or("Qwen-Drive perception hidden length overflow")?;
    let start = image_rows
        .len()
        .checked_sub(keep)
        .ok_or("Qwen-Drive perception prompt has too few image tokens")?;
    let image_rows = image_rows
        .get(start..)
        .ok_or("Qwen-Drive perception image-token slice is invalid")?;
    let mut values = vec![0.0; keep * width];
    for (row, &source_row) in image_rows.iter().enumerate() {
        for channel in 0..width {
            values[(row / per_image * width + channel) * per_image + row % per_image] =
                hidden[source_row * width + channel];
        }
    }
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::token_ids(
        "qwen_drive.perception_prompt_ids",
        &tokens,
    ));
    Tensor4::new(values, [cameras, width, grid.grid_h, grid.grid_w])
}

fn single_frame_lidar2ego(frame: &PerceptionFrame) -> [[f32; 16]; 1] {
    [frame.lidar2ego]
}

fn argmax(logits: &[f32]) -> Result<u32, String> {
    let (&first, tail) = logits
        .split_first()
        .ok_or("Qwen-Drive VLM returned empty logits")?;
    if !first.is_finite() {
        return Err("Qwen-Drive VLM returned non-finite logits".into());
    }
    let mut best = (0usize, first);
    for (index, &value) in tail.iter().enumerate() {
        if !value.is_finite() {
            return Err("Qwen-Drive VLM returned non-finite logits".into());
        }
        if value > best.1 {
            best = (index + 1, value);
        }
    }
    u32::try_from(best.0).map_err(|_| "Qwen-Drive token id does not fit u32".into())
}

fn prefill_scene(
    model: &mut Qwen35Model<'_>,
    tokenizer: &BPETokenizer,
    pool: Arc<ComputePool>,
    scene: &PlanningScene,
    projected: &[f32],
    grids: &[VisionGrid],
    mode: PlanningMode,
    max_reasoning_tokens: usize,
) -> Result<PrefillOutput, String> {
    let tokens = build_prompt(tokenizer, scene, grids, mode)?;
    let image = special(tokenizer, "image_pad")?;
    let signed_tokens = tokens
        .iter()
        .map(|&token| i32::try_from(token).map_err(|_| "Qwen-Drive token id exceeds i32"))
        .collect::<Result<Vec<_>, _>>()?;
    let embeddings = inject_vision_embeddings(
        model,
        &signed_tokens,
        Some(image as i32),
        projected,
        projected.len() / model.config.n_embd,
        model.config.n_embd,
    )?;
    let (positions, _) = build_qwen35_positions(&tokens, Some(image), grids)?;
    let newline = plain(tokenizer, "\n");
    let extra = if mode == PlanningMode::Reasoning {
        max_reasoning_tokens
            .checked_add(newline.len())
            .and_then(|value| value.checked_add(1))
            .ok_or("Qwen-Drive session capacity overflow")?
    } else {
        0
    };
    let capacity = tokens
        .len()
        .checked_add(extra)
        .ok_or("Qwen-Drive session capacity overflow")?;
    if capacity > model.config.n_ctx {
        return Err(format!(
            "Qwen-Drive prompt requires {capacity} tokens, model context is {}",
            model.config.n_ctx
        ));
    }
    let mut session = Qwen35Session::new(model, capacity, pool)?;
    let mut logits = session.step(&embeddings, tokens.len(), &positions)?;
    let mut cached = tokens.len();
    let mut reasoning_ids = Vec::new();

    if mode == PlanningMode::Reasoning {
        let im_end = special(tokenizer, "im_end")?;
        let eos = tokenizer.eos_id();
        let min_reasoning_tokens = 10usize;
        for index in 0..max_reasoning_tokens {
            let token = argmax(&logits)?;
            let terminator = token == im_end || eos == Some(token);
            if terminator && index + 1 >= min_reasoning_tokens {
                break;
            }
            reasoning_ids.push(token);
            if index + 1 < max_reasoning_tokens {
                let position = session.next_position();
                logits =
                    session.step_with_tokens(&[token], &[[position, position, position, 0]])?;
                cached += 1;
            }
        }
        let already_cached_reasoning = cached - tokens.len();
        let mut closing = reasoning_ids[already_cached_reasoning..].to_vec();
        closing.push(im_end);
        closing.extend_from_slice(&newline);
        if !closing.is_empty() {
            let start = session.next_position();
            let closing_positions = (0..closing.len())
                .map(|offset| {
                    let position = start + offset;
                    [position, position, position, 0]
                })
                .collect::<Vec<_>>();
            session.step_with_tokens(&closing, &closing_positions)?;
            cached += closing.len();
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::token_ids(
            "qwen_drive.reasoning_token_ids",
            &reasoning_ids,
        ));
    }
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::token_ids(
        "qwen_drive.prompt_ids",
        &tokens,
    ));

    let last = session.next_position().saturating_sub(1);
    let cache = session.dense_kv_snapshots(&PLANNER_LAYERS, cached)?;
    let reasoning = (mode == PlanningMode::Reasoning).then(|| {
        let decoded = tokenizer.decode(&reasoning_ids, false);
        decoded
            .rsplit_once("</think>")
            .map_or(decoded.as_str(), |(_, answer)| answer)
            .trim()
            .to_owned()
    });
    Ok(PrefillOutput {
        cache,
        anchor: [last; 3],
        reasoning,
        reasoning_token_ids: reasoning_ids,
    })
}

fn trajectories(batch: &TrajectoryBatch) -> Result<Vec<Vec<[f32; 3]>>, String> {
    if batch.values.len() != batch.samples * batch.points * 3 {
        return Err("Qwen-Drive trajectory output shape mismatch".into());
    }
    Ok(batch
        .values
        .chunks_exact(batch.points * 3)
        .map(|sample| {
            sample
                .chunks_exact(3)
                .map(|point| [point[0], point[1], point[2]])
                .collect()
        })
        .collect())
}

fn temp_output_path(output: &Path) -> Result<PathBuf, String> {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("Qwen-Drive output must have a UTF-8 file name")?;
    Ok(output.with_file_name(format!(".{name}.{}.tmp", std::process::id())))
}

pub fn run_qwen_drive_cli(options: QwenDriveCliOptions, threads: usize) -> Result<(), String> {
    let vlm = open_model_source(&options.model, ComponentRole::Llm)
        .map_err(|error| format!("Failed to load Qwen-Drive VLM: {error}"))?;
    let mmproj = open_model_source(&options.mmproj, ComponentRole::Mmproj)
        .map_err(|error| format!("Failed to load Qwen-Drive mmproj: {error}"))?;
    let (head_path, expected_architecture, head_label) = match &options.head {
        QwenDriveHead::Planner(path) => (path, "qwen_drive_planner", "planner"),
        QwenDriveHead::Perception(path) => (path, "qwen_drive_perception", "perception"),
    };
    let head = open_model_source(head_path, ComponentRole::Llm)
        .map_err(|error| format!("Failed to load Qwen-Drive {head_label}: {error}"))?;
    architecture(vlm.as_ref(), "qwen35", "Qwen-Drive VLM")?;
    architecture(mmproj.as_ref(), "clip", "Qwen-Drive mmproj")?;
    architecture(head.as_ref(), expected_architecture, "Qwen-Drive head")?;

    let tokenizer = BPETokenizer::from_gguf_metadata(|key| vlm.metadata(key).cloned())
        .map_err(|error| format!("Failed to initialize Qwen-Drive tokenizer: {error}"))?;
    let mut model = Qwen35Model::from_source(vlm.as_ref())?;
    let encoder = VisionEncoder::from_source(mmproj.as_ref())?;
    let pool = Arc::new(ComputePool::new(threads.max(1)));
    let temporary = temp_output_path(&options.output)?;
    let result = (|| {
        let file = File::create(&temporary)
            .map_err(|error| format!("Cannot create {}: {error}", temporary.display()))?;
        let mut writer = BufWriter::new(file);
        match &options.head {
            QwenDriveHead::Planner(_) => {
                let planner = PlanningExpert::from_source(head.as_ref())?;
                let scenes = read_planning_scenes(
                    options.scenes.as_deref().expect("validated planner scenes"),
                    options
                        .image_root
                        .as_deref()
                        .expect("validated planner image root"),
                    None,
                )?;
                for scene in &scenes {
                    let (projected, grids) = encode_images(&encoder, scene)?;
                    let prefill = prefill_scene(
                        &mut model,
                        &tokenizer,
                        Arc::clone(&pool),
                        scene,
                        &projected,
                        &grids,
                        options.mode,
                        planner.config().max_reasoning_tokens,
                    )?;
                    let batch = planner.sample(
                        &prefill.cache,
                        prefill.anchor,
                        scene,
                        options.samples,
                        options.steps,
                        options.seed as u64,
                        pool.as_ref(),
                    )?;
                    serde_json::to_writer(
                        &mut writer,
                        &PlanningPrediction {
                            token: &scene.token,
                            trajectories: trajectories(&batch)?,
                            reasoning: prefill.reasoning.as_deref(),
                            reasoning_token_ids: &prefill.reasoning_token_ids,
                        },
                    )
                    .map_err(|error| format!("Cannot serialize Qwen-Drive prediction: {error}"))?;
                    writer.write_all(b"\n").map_err(|error| {
                        format!("Cannot write {}: {error}", temporary.display())
                    })?;
                }
            }
            QwenDriveHead::Perception(_) => {
                let perception = QwenDrivePerception::from_source(head.as_ref())?;
                if perception.config().llm_dim != model.config.n_embd {
                    return Err(format!(
                        "Qwen-Drive perception expects LLM width {}, got {}",
                        perception.config().llm_dim,
                        model.config.n_embd
                    ));
                }
                let frame = read_perception_frame(
                    options
                        .frames
                        .as_deref()
                        .expect("validated perception frame"),
                )?;
                let (vit, projected, grids) = encode_perception_images(
                    &encoder,
                    &frame,
                    perception.config().image_size,
                    perception.config().vit_dim,
                )?;
                let llm = prefill_perception(
                    &mut model,
                    &tokenizer,
                    Arc::clone(&pool),
                    &frame,
                    &projected,
                    &grids,
                )?;
                let lidar2ego = single_frame_lidar2ego(&frame);
                let config = perception.config();
                let geometry = ViewGeometry {
                    frustum_range: config.frustum_range,
                    frustum_size: config.frustum_size,
                    pc_range: config.det_pc_range,
                    voxel_size: [
                        config.det_voxel_size[0],
                        config.det_voxel_size[1],
                        (config.det_pc_range[5] - config.det_pc_range[2])
                            / config.occ_pillar_h as f32,
                    ],
                    voxel_shape: [config.bev[1], config.bev[0], config.occ_pillar_h],
                    lidar2img: &frame.lidar2img,
                    lidar2ego: &lidar2ego,
                };
                let output = perception.infer(
                    &vit,
                    &llm,
                    &geometry,
                    &frame.dataset_type,
                    frame.box_coord_system_ego,
                    pool.as_ref(),
                )?;
                serde_json::to_writer(
                    &mut writer,
                    &PerceptionPrediction {
                        token: &frame.token,
                        result: &output,
                    },
                )
                .map_err(|error| {
                    format!("Cannot serialize Qwen-Drive perception result: {error}")
                })?;
                writer
                    .write_all(b"\n")
                    .map_err(|error| format!("Cannot write {}: {error}", temporary.display()))?;
            }
        }
        writer
            .flush()
            .map_err(|error| format!("Cannot flush {}: {error}", temporary.display()))?;
        fs::rename(&temporary, &options.output).map_err(|error| {
            format!(
                "Cannot publish {} as {}: {error}",
                temporary.display(),
                options.output.display()
            )
        })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct PromptIds {
        token_counts: Vec<usize>,
        direct: Vec<u32>,
        reasoning: Vec<u32>,
    }

    #[derive(serde::Deserialize)]
    struct PerceptionPromptIds {
        prompt_ids: Vec<u32>,
    }

    #[test]
    fn perception_geometry_keeps_one_lidar_pose_for_six_cameras() {
        let frame = PerceptionFrame {
            token: "fixture".into(),
            dataset_type: "nuscenes".into(),
            cam_order: vec![String::new(); 6],
            content: Vec::new(),
            image_shapes: vec![[512, 896, 3]; 6],
            lidar2img: vec![[0.0; 16]; 6],
            lidar2ego: [1.0; 16],
            box_coord_system_ego: true,
        };

        assert_eq!(single_frame_lidar2ego(&frame), [[1.0; 16]]);
    }

    #[test]
    #[ignore = "requires RMI_QWEN_DRIVE_VLM and RMI_QWEN_DRIVE_OFFICIAL"]
    fn planning_prompt_token_ids_match_official_tokenizer() {
        let model = std::env::var_os("RMI_QWEN_DRIVE_VLM").unwrap();
        let official = PathBuf::from(std::env::var_os("RMI_QWEN_DRIVE_OFFICIAL").unwrap());
        let source = crate::core::loader::GGUFLoader::from_file(model).unwrap();
        let tokenizer =
            BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
        let scene = read_planning_scenes(
            &official.join("data/demo/planning_scenes.jsonl"),
            &official.join("data/demo"),
            Some(1),
        )
        .unwrap()
        .remove(0);
        let fixture: PromptIds = serde_json::from_str(
            &std::fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/qwen_drive/planner-prompt-ids.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let grids = fixture
            .token_counts
            .iter()
            .map(|&count| VisionGrid {
                grid_t: 1,
                grid_h: 1,
                grid_w: count,
                patch_size: 1,
                merge_size: 1,
            })
            .collect::<Vec<_>>();

        assert_eq!(
            build_prompt(&tokenizer, &scene, &grids, PlanningMode::Direct).unwrap(),
            fixture.direct
        );
        assert_eq!(
            build_prompt(&tokenizer, &scene, &grids, PlanningMode::Reasoning).unwrap(),
            fixture.reasoning
        );
    }

    #[test]
    #[ignore = "requires RMI_QWEN_DRIVE_VLM"]
    fn perception_prompt_token_ids_match_official_tokenizer() {
        let source = crate::core::loader::GGUFLoader::from_file(
            std::env::var_os("RMI_QWEN_DRIVE_VLM").unwrap(),
        )
        .unwrap();
        let tokenizer =
            BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned()).unwrap();
        let cameras = [
            ("<FRONT VIEW>", "CAM_FRONT"),
            ("<FRONT RIGHT VIEW>", "CAM_FRONT_RIGHT"),
            ("<BACK RIGHT VIEW>", "CAM_BACK_RIGHT"),
            ("<BACK VIEW>", "CAM_BACK"),
            ("<BACK LEFT VIEW>", "CAM_BACK_LEFT"),
            ("<FRONT LEFT VIEW>", "CAM_FRONT_LEFT"),
        ];
        let mut content = Vec::new();
        for (label, camera) in cameras {
            content.push(PerceptionContent::Text(label.into()));
            content.push(PerceptionContent::Image {
                camera: camera.into(),
                path: PathBuf::from(format!("{camera}.jpg")),
            });
        }
        content.push(PerceptionContent::Text("Analyze the scene.".into()));
        let frame = PerceptionFrame {
            token: "fixture".into(),
            dataset_type: "nuscenes".into(),
            cam_order: cameras
                .into_iter()
                .map(|(_, camera)| camera.into())
                .collect(),
            content,
            image_shapes: vec![[512, 896, 3]; 6],
            lidar2img: Vec::new(),
            lidar2ego: [0.0; 16],
            box_coord_system_ego: true,
        };
        let grids = vec![
            VisionGrid {
                grid_t: 1,
                grid_h: 16,
                grid_w: 28,
                patch_size: 16,
                merge_size: 2,
            };
            6
        ];
        let fixture: PerceptionPromptIds = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen_drive/perception-frame.json"
        )))
        .unwrap();
        assert_eq!(
            build_perception_prompt(&tokenizer, &frame, &grids).unwrap(),
            fixture.prompt_ids
        );
    }

    #[test]
    fn trajectory_json_shape_is_sample_point_coordinate() {
        let batch = TrajectoryBatch {
            samples: 2,
            points: 2,
            values: (0..12).map(|value| value as f32).collect(),
        };
        assert_eq!(
            trajectories(&batch).unwrap(),
            vec![
                vec![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                vec![[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            ]
        );
    }
}
