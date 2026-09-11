use super::cli::{PlanningMode, QwenDriveCliOptions, QwenDriveHead};
use super::text::{decode_image, inject_vision_embeddings};
use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::models::qwen35::vision::{qwen_smart_resize, VisionEncoder, VisionGrid, VisionScratchpad};
use crate::models::qwen35::{build_qwen35_positions, Qwen35Model, Qwen35Session};
use crate::models::qwen_drive::planning::{PlanningExpert, TrajectoryBatch};
use crate::models::qwen_drive::scene::{read_planning_scenes, PlanningScene};
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
        return Err(format!("Expected {label} architecture {expected}, got {actual}"));
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
    if grids.len() != scene.views.iter().map(|view| view.frames.len()).sum::<usize>() {
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
            let normalized = normalize_rgb(
                &rgb,
                encoder.config.image_mean,
                encoder.config.image_std,
            )?;
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
                logits = session.step_with_tokens(&[token], &[[position, position, position, 0]])?;
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
    let QwenDriveHead::Planner(planner_path) = &options.head else {
        return Err("Qwen-Drive perception runtime is not implemented yet".into());
    };
    let vlm = open_model_source(&options.model, ComponentRole::Llm)
        .map_err(|error| format!("Failed to load Qwen-Drive VLM: {error}"))?;
    let mmproj = open_model_source(&options.mmproj, ComponentRole::Mmproj)
        .map_err(|error| format!("Failed to load Qwen-Drive mmproj: {error}"))?;
    let planner = open_model_source(planner_path, ComponentRole::Llm)
        .map_err(|error| format!("Failed to load Qwen-Drive planner: {error}"))?;
    architecture(vlm.as_ref(), "qwen35", "Qwen-Drive VLM")?;
    architecture(mmproj.as_ref(), "clip", "Qwen-Drive mmproj")?;
    architecture(
        planner.as_ref(),
        "qwen_drive_planner",
        "Qwen-Drive planner",
    )?;

    let tokenizer = BPETokenizer::from_gguf_metadata(|key| vlm.metadata(key).cloned())
        .map_err(|error| format!("Failed to initialize Qwen-Drive tokenizer: {error}"))?;
    let mut model = Qwen35Model::from_source(vlm.as_ref())?;
    let encoder = VisionEncoder::from_source(mmproj.as_ref())?;
    let planner = PlanningExpert::from_source(planner.as_ref())?;
    let scenes = read_planning_scenes(
        options.scenes.as_deref().expect("validated planner scenes"),
        options
            .image_root
            .as_deref()
            .expect("validated planner image root"),
        None,
    )?;
    let pool = Arc::new(ComputePool::new(threads.max(1)));
    let temporary = temp_output_path(&options.output)?;
    let result = (|| {
        let file = File::create(&temporary)
            .map_err(|error| format!("Cannot create {}: {error}", temporary.display()))?;
        let mut writer = BufWriter::new(file);
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
            writer
                .write_all(b"\n")
                .map_err(|error| format!("Cannot write {}: {error}", temporary.display()))?;
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

    #[test]
    #[ignore = "requires RMI_QWEN_DRIVE_VLM and RMI_QWEN_DRIVE_OFFICIAL"]
    fn planning_prompt_token_ids_match_official_tokenizer() {
        let model = std::env::var_os("RMI_QWEN_DRIVE_VLM").unwrap();
        let official = PathBuf::from(std::env::var_os("RMI_QWEN_DRIVE_OFFICIAL").unwrap());
        let source = crate::core::loader::GGUFLoader::from_file(model).unwrap();
        let tokenizer = BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
            .unwrap();
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
