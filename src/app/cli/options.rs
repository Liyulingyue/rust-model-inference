use super::types::{
    CliOptions, DreamXCliOptions, DreamXOptions, DreamXRefinerOptions, EmbeddingOutput,
    LongCatCliOptions, PlanningMode, QwenDriveCliOptions, QwenDriveHead, YuE2CliOptions,
    ZImageCliOptions,
};
use crate::models::diffusion::longcat::LongCatKind;

/// Parse the LongCat Image Edit flags, or `None` when the invocation is not
/// for LongCat. Selected by `--kind` / `--components`, which no other model
/// uses, so the discriminator never collides.
pub fn longcat_cli_options(options: &CliOptions) -> Result<Option<LongCatCliOptions>, String> {
    let kind = match options.longcat_kind.as_deref() {
        None => {
            for (present, flag) in [
                (options.components.is_some(), "--components"),
                (options.input.is_some(), "--input"),
                (options.side.is_some(), "--side"),
                (options.guidance.is_some(), "--guidance"),
            ] {
                if present {
                    return Err(format!("{flag} requires --kind edit|turbo"));
                }
            }
            return Ok(None);
        }
        Some("edit") => LongCatKind::Edit,
        Some("turbo") => LongCatKind::EditTurbo,
        Some(other) => return Err(format!("Invalid --kind {other}: expected edit or turbo")),
    };

    let model = options.model.clone();
    if model.as_os_str().is_empty() {
        return Err("LongCat requires --model <transformer.gguf>".into());
    }
    let components = options
        .components
        .clone()
        .ok_or("LongCat requires --components <dir>")?;
    let input = options
        .input
        .clone()
        .ok_or("LongCat requires --input <image.png>")?;
    let out = options
        .out
        .clone()
        .ok_or("LongCat requires --out <image.png>")?;
    let instruction = options
        .instruction
        .clone()
        .ok_or("LongCat requires --instruction <text>")?;
    if instruction.trim().is_empty() {
        return Err("LongCat --instruction must not be empty".into());
    }
    let side = options.side.unwrap_or(1024);
    if side == 0 || side % 16 != 0 {
        return Err(format!(
            "LongCat --side must be a positive multiple of 16, got {side}"
        ));
    }
    // The two checkpoints ship different Flux schedules, so the defaults are
    // per-kind rather than shared.
    let steps = options
        .steps
        .unwrap_or(if kind == LongCatKind::Edit { 50 } else { 8 });
    if steps == 0 {
        return Err("LongCat --steps must be positive".into());
    }
    let guidance = options
        .guidance
        .unwrap_or(if kind == LongCatKind::Edit { 4.5 } else { 1.0 });
    if !(guidance.is_finite() && guidance >= 0.0) {
        return Err(format!(
            "LongCat --guidance must be finite and >= 0, got {guidance}"
        ));
    }
    let seed = u64::try_from(options.seed.unwrap_or(42))
        .map_err(|_| "LongCat --seed must be non-negative".to_string())?;

    Ok(Some(LongCatCliOptions {
        kind,
        model,
        components,
        input,
        out,
        instruction,
        side,
        steps,
        guidance,
        seed,
        overwrite: options.overwrite,
    }))
}

pub fn yue2_cli_options(options: &CliOptions) -> Result<Option<YuE2CliOptions>, String> {
    if !options.yue2 {
        return if options.lyrics.is_some() {
            Err("--lyrics requires --yue2".into())
        } else {
            Ok(None)
        };
    }

    let conflict = if options.tts {
        Some("--tts")
    } else if options.edit {
        Some("--edit")
    } else if options.audio.is_some() {
        Some("--audio")
    } else if options.ref_audio.is_some() {
        Some("--ref-audio")
    } else if options.image.is_some() {
        Some("--image")
    } else if options.video.is_some() {
        Some("--video")
    } else if options.mmproj.is_some() {
        Some("--mmproj")
    } else if options.text_encoder.is_some() {
        Some("--text-encoder")
    } else if options.embedding {
        Some("--embedding")
    } else if options.embedding_output != EmbeddingOutput::Summary {
        Some("--embedding-output")
    } else if options.jev
        || options.jev_context.is_some()
        || !options.jev_questions.is_empty()
        || options.jev_positive.is_some()
        || options.jev_output_json
        || options.jev_multi
        || !options.jev_blocks.is_empty()
    {
        Some("--jev")
    } else if options.dreamx {
        Some("--dreamx")
    } else if options.planner.is_some()
        || options.perception.is_some()
        || options.scenes.is_some()
        || options.image_root.is_some()
        || options.frames.is_some()
        || options.planning_mode.is_some()
        || options.num_samples.is_some()
        || options.num_steps.is_some()
        || options.output.is_some()
    {
        Some("Qwen-Drive flags")
    } else if options.negative_prompt.is_some()
        || options.duration_seconds.is_some()
        || options.fps.is_some()
        || options.target_spatial_tokens.is_some()
        || options.refine.is_some()
        || options.refiner_kv_len.is_some()
        || options.latent_upsample.is_some()
        || options.refiner_decoder.is_some()
        || options.dry_run
        || options.overwrite
        || options.allow_memory_overcommit
    {
        Some("DreamX flags")
    } else if options.bench {
        Some("--bench")
    } else if options.profile {
        Some("--profile")
    } else if options.dump_logits {
        Some("--dump-logits")
    } else if options.tts_model.is_some()
        || options.tts_mmproj.is_some()
        || options.ref_text.is_some()
        || options.source_audio.is_some()
        || options.source_text.is_some()
        || options.target_text.is_some()
        || options.instruction.is_some()
        || options.use_xvector_supplied
        || options.language.is_some()
    {
        Some("TTS reference/instruction flags")
    } else if options.max_context.is_some() {
        Some("--max-context")
    } else if options.chat_template.is_some() {
        Some("--chat-template")
    } else if options.prefill_batch_size.is_some() {
        Some("--prefill-batch-size")
    } else if options.repetition_penalty.is_some() {
        Some("--repetition-penalty")
    } else if options.resolution.is_some() {
        Some("--resolution")
    } else if options.cfg_scale.is_some() {
        Some("--cfg-scale")
    } else if options.thinking {
        Some("--thinking")
    } else {
        None
    };
    if let Some(conflict) = conflict {
        return Err(format!("--yue2 cannot be used with {conflict}"));
    }

    let model = (!options.model.as_os_str().is_empty())
        .then(|| options.model.clone())
        .ok_or("--yue2 requires a non-empty --model")?;
    let vae = options
        .vae
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("--yue2 requires --vae")?;
    let style = options
        .prompt
        .clone()
        .filter(|value| !value.trim().is_empty())
        .ok_or("--yue2 requires a non-empty --prompt style")?;
    let lyrics = options
        .lyrics
        .clone()
        .filter(|value| !value.trim().is_empty())
        .ok_or("--yue2 requires non-empty --lyrics")?;
    let out = options
        .out
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("--yue2 requires --out")?;
    if !out
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("wav"))
    {
        return Err("--yue2 requires a .wav --out path".into());
    }

    let seed = options.seed.unwrap_or(831_001);
    let seed = u64::try_from(seed).map_err(|_| "--yue2 requires a non-negative --seed")?;
    let steps = options.steps.unwrap_or(32);
    if steps == 0 {
        return Err("--yue2 requires --steps greater than 0".into());
    }
    let mut semantic = crate::models::yue2::SamplingConfig::semantic();
    semantic.max_tokens = options.max_tokens.unwrap_or(semantic.max_tokens);
    semantic.temperature = options.temperature.unwrap_or(semantic.temperature);
    semantic.top_k = options.top_k.unwrap_or(semantic.top_k);
    semantic.top_p = options.top_p.unwrap_or(semantic.top_p);
    semantic.validate()?;

    Ok(Some(YuE2CliOptions {
        model,
        vae,
        style,
        lyrics,
        out,
        seed,
        semantic,
        steps,
    }))
}

pub fn qwen_drive_cli_options(options: &CliOptions) -> Result<Option<QwenDriveCliOptions>, String> {
    let requested = options.planner.is_some()
        || options.perception.is_some()
        || options.scenes.is_some()
        || options.image_root.is_some()
        || options.frames.is_some()
        || options.planning_mode.is_some()
        || options.num_samples.is_some()
        || options.num_steps.is_some()
        || options.output.is_some();
    if !requested {
        return Ok(None);
    }
    let head = match (&options.planner, &options.perception) {
        (Some(_), Some(_)) => {
            return Err("--planner and --perception are mutually exclusive".into())
        }
        (Some(path), None) => QwenDriveHead::Planner(path.clone()),
        (None, Some(path)) => QwenDriveHead::Perception(path.clone()),
        (None, None) => return Err("Qwen-Drive requires --planner or --perception".into()),
    };
    if options.model.as_os_str().is_empty() {
        return Err("Qwen-Drive requires --model".into());
    }
    let mmproj = options
        .mmproj
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("Qwen-Drive requires --mmproj")?;
    let output = options
        .output
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("Qwen-Drive requires --output")?;
    let conflict = if options.dreamx {
        Some("--dreamx")
    } else if options.tts {
        Some("--tts")
    } else if options.edit {
        Some("--edit")
    } else if options.audio.is_some() {
        Some("--audio")
    } else if options.image.is_some() {
        Some("--image")
    } else if options.video.is_some() {
        Some("--video")
    } else if options.vae.is_some() || options.text_encoder.is_some() {
        Some("Z-Image flags")
    } else if options.embedding {
        Some("--embedding")
    } else if options.out.is_some() {
        Some("--out")
    } else {
        None
    };
    if let Some(conflict) = conflict {
        return Err(format!("Qwen-Drive cannot be used with {conflict}"));
    }

    let (mode, scenes, image_root, frames, samples, steps, seed) = match head {
        QwenDriveHead::Planner(_) => {
            let mode = match options.planning_mode.as_deref() {
                Some("direct_planning") => PlanningMode::Direct,
                Some("reasoning_planning") => PlanningMode::Reasoning,
                Some(value) => {
                    return Err(format!(
                        "Invalid --mode {value:?}; expected direct_planning or reasoning_planning"
                    ));
                }
                None => return Err("--planner requires --mode".into()),
            };
            let scenes = options
                .scenes
                .clone()
                .filter(|path| !path.as_os_str().is_empty())
                .ok_or("--planner requires --scenes")?;
            let image_root = options
                .image_root
                .clone()
                .filter(|path| !path.as_os_str().is_empty())
                .ok_or("--planner requires --image-root")?;
            if options.frames.is_some() {
                return Err("--frames requires --perception".into());
            }
            let samples = options
                .num_samples
                .ok_or("--planner requires --num-samples")?;
            let steps = options.num_steps.ok_or("--planner requires --num-steps")?;
            let seed = options.seed.ok_or("--planner requires --seed")?;
            if samples == 0 || steps == 0 {
                return Err("--num-samples and --num-steps must be greater than zero".into());
            }
            (
                mode,
                Some(scenes),
                Some(image_root),
                None,
                samples,
                steps,
                seed,
            )
        }
        QwenDriveHead::Perception(_) => {
            if options.scenes.is_some()
                || options.image_root.is_some()
                || options.planning_mode.is_some()
                || options.num_samples.is_some()
                || options.num_steps.is_some()
                || options.seed.is_some()
            {
                return Err("Planning flags require --planner".into());
            }
            let frames = options
                .frames
                .clone()
                .filter(|path| !path.as_os_str().is_empty())
                .ok_or("--perception requires --frames")?;
            (PlanningMode::Direct, None, None, Some(frames), 1, 10, 42)
        }
    };

    Ok(Some(QwenDriveCliOptions {
        model: options.model.clone(),
        mmproj,
        head,
        mode,
        scenes,
        image_root,
        frames,
        output,
        samples,
        steps,
        seed,
    }))
}

pub fn dreamx_cli_options(options: &CliOptions) -> Result<Option<DreamXCliOptions>, String> {
    let unique_option = if options.negative_prompt.is_some() {
        Some("--negative-prompt")
    } else if options.duration_seconds.is_some() {
        Some("--duration")
    } else if options.fps.is_some() {
        Some("--fps")
    } else if options.target_spatial_tokens.is_some() {
        Some("--target-spatial-tokens")
    } else if options.refine.is_some() {
        Some("--refine/--no-refine")
    } else if options.refiner_kv_len.is_some() {
        Some("--refiner-kv-len")
    } else if options.latent_upsample.is_some() {
        Some("--latent-upsample")
    } else if options.refiner_decoder.is_some() {
        Some("--refiner-decoder")
    } else if options.dry_run {
        Some("--dry-run")
    } else if options.overwrite {
        Some("--overwrite")
    } else if options.allow_memory_overcommit {
        Some("--allow-memory-overcommit")
    } else {
        None
    };
    if !options.dreamx {
        return match unique_option {
            Some(option) => Err(format!("{option} requires --dreamx")),
            None => Ok(None),
        };
    }

    let conflict = if options.tts {
        Some("--tts")
    } else if options.edit {
        Some("--edit")
    } else if options.source_audio.is_some() {
        Some("--source-audio")
    } else if options.source_text.is_some() {
        Some("--source-text")
    } else if options.target_text.is_some() {
        Some("--target-text")
    } else if options.instruction.is_some() {
        Some("--instruction")
    } else if options.use_xvector_supplied {
        Some("--use-xvector")
    } else if options.audio.is_some() {
        Some("--audio")
    } else if options.video.is_some() {
        Some("--video")
    } else if options.vae.is_some() {
        Some("--vae")
    } else if options.text_encoder.is_some() {
        Some("--text-encoder")
    } else if options.ref_audio.is_some() {
        Some("--ref-audio")
    } else if options.ref_text.is_some() {
        Some("--ref-text")
    } else if options.embedding {
        Some("--embedding")
    } else if options.dump_logits {
        Some("--dump-logits")
    } else if options.bench {
        Some("--bench")
    } else if options.profile {
        Some("--profile")
    } else if options.compute_policy() != crate::compute::ComputePolicy::Cpu {
        Some("--gpu/--compute")
    } else if options.thinking {
        Some("--thinking")
    } else if options.language.is_some() {
        Some("--language")
    } else if options.max_tokens.is_some() {
        Some("--max-tokens")
    } else if options.temperature.is_some() {
        Some("--temp")
    } else if options.resolution.is_some() {
        Some("--resolution")
    } else {
        None
    };
    if let Some(conflict) = conflict {
        return Err(format!("--dreamx cannot be used with {conflict}"));
    }

    if options.model.as_os_str().is_empty() {
        return Err("--dreamx requires --model".into());
    }
    let model = options.model.clone();
    let mmproj = options
        .mmproj
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("--dreamx requires --mmproj")?;
    let image = options
        .image
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("--dreamx requires --image")?;
    let prompt = options
        .prompt
        .clone()
        .filter(|prompt| !prompt.trim().is_empty())
        .ok_or("--dreamx requires a non-empty --prompt")?;
    let out = options
        .out
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("--dreamx requires --out")?;
    if options
        .negative_prompt
        .as_deref()
        .is_some_and(|prompt| prompt.trim().is_empty())
    {
        return Err("--negative-prompt must not be empty".into());
    }

    let defaults = DreamXOptions::default();
    let duration_seconds = options
        .duration_seconds
        .unwrap_or(defaults.duration_seconds);
    let fps = options.fps.unwrap_or(defaults.fps);
    let steps = options.steps.unwrap_or(defaults.steps);
    let target_spatial_tokens = options
        .target_spatial_tokens
        .unwrap_or(defaults.target_spatial_tokens);
    let kv_len = options.refiner_kv_len.unwrap_or(defaults.refiner.kv_len);
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Err("--dreamx requires a finite positive --duration".into());
    }
    if fps == 0 || steps == 0 || target_spatial_tokens == 0 || kv_len == 0 {
        return Err(
            "--dreamx requires positive --fps, --steps, --target-spatial-tokens, and --refiner-kv-len"
                .into(),
        );
    }

    Ok(Some(DreamXCliOptions {
        model,
        mmproj,
        image,
        prompt,
        negative_prompt: options.negative_prompt.clone(),
        out,
        options: DreamXOptions {
            duration_seconds,
            fps,
            steps,
            seed: options.seed.unwrap_or(defaults.seed),
            target_spatial_tokens,
            refine: options.refine.unwrap_or(defaults.refine),
            refiner: DreamXRefinerOptions {
                kv_len,
                latent_upsample: options
                    .latent_upsample
                    .unwrap_or(defaults.refiner.latent_upsample),
                decoder: options.refiner_decoder.unwrap_or(defaults.refiner.decoder),
            },
        },
        dry_run: options.dry_run,
        overwrite: options.overwrite,
        allow_memory_overcommit: options.allow_memory_overcommit,
    }))
}

pub fn z_image_cli_options(options: &CliOptions) -> Result<Option<ZImageCliOptions>, String> {
    if options.text_encoder.is_none() && options.vae.is_none() {
        return if options.seed.is_some()
            && options.planner.is_none()
            && options.perception.is_none()
            && !options.tts
            && !options.dreamx
        {
            Err("--seed requires Z-Image components or --tts".into())
        } else {
            Ok(None)
        };
    }
    let conflict = if options.tts {
        Some("--tts")
    } else if options.audio.is_some() {
        Some("--audio")
    } else if options.ref_audio.is_some() {
        Some("--ref-audio")
    } else if options.image.is_some() {
        Some("--image")
    } else if options.video.is_some() {
        Some("--video")
    } else if options.mmproj.is_some() {
        Some("--mmproj")
    } else if options.embedding {
        Some("--embedding")
    } else if options.dump_logits {
        Some("--dump-logits")
    } else if options.bench {
        Some("--bench")
    } else if options.profile {
        Some("--profile")
    } else if options.thinking {
        Some("--thinking")
    } else if options.language.is_some() {
        Some("--language")
    } else if options.max_tokens.is_some() {
        Some("--max-tokens")
    } else if options.temperature.is_some() {
        Some("--temp")
    } else if options.embedding_output != EmbeddingOutput::Summary {
        Some("--embedding-output")
    } else {
        None
    };
    // `--gpu` is deliberately not in the conflict list. The DiT's projections
    // go through `matmul_q8_0_quantized_parallel_rows`, which hands supported
    // shapes to the Vulkan backend and falls back per-matmul otherwise, so the
    // flag is meaningful here. Everything outside that kernel -- attention,
    // norms, the VAE convolutions -- stays on the CPU either way.
    if let Some(conflict) = conflict {
        return Err(format!("Z-Image cannot be used with {conflict}"));
    }
    if options.model.as_os_str().is_empty() {
        return Err("Z-Image requires --model for the diffusion component".into());
    }
    options
        .text_encoder
        .as_ref()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("Z-Image requires --text-encoder")?;
    options
        .vae
        .as_ref()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("Z-Image requires --vae")?;
    let out = options
        .out
        .clone()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or("Z-Image requires --out")?;
    if options
        .prompt
        .as_deref()
        .is_none_or(|prompt| prompt.trim().is_empty())
    {
        return Err("Z-Image requires a non-empty --prompt".into());
    }
    let steps = options.steps.unwrap_or(8);
    let resolution = options.resolution.unwrap_or(512);
    if steps == 0 || resolution == 0 || resolution % 16 != 0 {
        return Err("Z-Image requires positive --steps and --resolution divisible by 16".into());
    }
    Ok(Some(ZImageCliOptions {
        steps,
        resolution,
        seed: options.seed.unwrap_or(0),
        out,
    }))
}
