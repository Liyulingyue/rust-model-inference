use super::options::{
    dreamx_cli_options, qwen_drive_cli_options, yue2_cli_options, z_image_cli_options,
};
use super::types::{normalize_tts_language, CliOptions};

pub fn validate_cli_options(options: &CliOptions) -> Result<(), String> {
    if options.width.is_some()
        || options.height.is_some()
        || options.cfg.is_some()
        || options.noise.is_some()
        || !options.references.is_empty()
    {
        return Err("--width/--height/--cfg/--noise/--reference require a Mage-Flow model".into());
    }
    if options.qwen_latent_width.is_some_and(|value| value == 0)
        || options.qwen_latent_height.is_some_and(|value| value == 0)
    {
        return Err("--qwen-latent-width and --qwen-latent-height must be positive".into());
    }
    if options
        .qwen_timestep
        .is_some_and(|value| !value.is_finite())
    {
        return Err("--qwen-timestep must be finite".into());
    }
    if options.gliner2_decide {
        if !options.jev {
            return Err("--gliner2-decide requires --jev".into());
        }
        if options.clm_head.is_some() {
            return Err("--gliner2-decide and --clm-head select different backends".into());
        }
        if options.gliner2_schema.is_some() && options.jev_multi {
            return Err("--gliner2-schema cannot be combined with --jev-multi".into());
        }
    } else if options.gliner2_schema.is_some() {
        return Err("--gliner2-schema requires --gliner2-decide".into());
    }
    if options.laya_request.is_some() {
        if options.model.as_os_str().is_empty() {
            return Err("--laya-request requires --model".into());
        }
        if options.jev
            || options.prompt.is_some()
            || options.embedding
            || options.tts
            || options.yue2
        {
            return Err("--laya-request cannot be combined with generation, --jev, --embedding, --tts, or --yue2".into());
        }
        return Ok(());
    }
    if yue2_cli_options(options)?.is_some() {
        return Ok(());
    }
    if (options.top_k.is_some() || options.top_p.is_some()) && (!options.tts || options.edit) {
        return Err("--top-k/--top-p require Breeze --tts without --edit".into());
    }
    if options
        .top_p
        .is_some_and(|p| !p.is_finite() || p <= 0.0 || p > 1.0)
    {
        return Err("--top-p must be finite and in (0, 1]".into());
    }
    if let Some(scale) = options.cfg_scale {
        let diffusion =
            options.text_encoder.is_some() && options.vae.is_some() && options.out.is_some();
        if (!options.tts || options.edit) && !diffusion {
            return Err("--cfg-scale requires Breeze --tts without --edit".into());
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err("--cfg-scale must be finite and greater than zero".into());
        }
    }
    if qwen_drive_cli_options(options)?.is_some() {
        return Ok(());
    }
    if dreamx_cli_options(options)?.is_some() {
        return Ok(());
    }
    if options.instruction.is_some() && !options.tts {
        return Err("--instruction requires --tts".into());
    }
    if !options.edit
        && (options.source_audio.is_some()
            || options.source_text.is_some()
            || options.target_text.is_some()
            || options.use_xvector_supplied)
    {
        return Err(
            "--source-audio, --source-text, --target-text, and --use-xvector require --tts --edit"
                .into(),
        );
    }
    if options.edit && !options.tts {
        return Err("--edit requires --tts".into());
    }
    z_image_cli_options(options)?;
    if options.tts {
        if options.model.as_os_str().is_empty() {
            return Err("--tts requires --model".into());
        }
        if options
            .mmproj
            .as_deref()
            .is_none_or(|path| path.as_os_str().is_empty())
        {
            return Err("--tts requires --mmproj".into());
        }
        if options
            .out
            .as_deref()
            .is_none_or(|path| path.as_os_str().is_empty())
        {
            return Err("--tts requires --out".into());
        }
        if options.max_tokens == Some(0) {
            return Err("--tts requires --max-tokens greater than 0".into());
        }
        if options.steps == Some(0) {
            return Err("--tts requires --steps greater than 0".into());
        }
        let conflict = if options.audio.is_some() {
            Some("--audio")
        } else if options.image.is_some() {
            Some("--image")
        } else if options.video.is_some() {
            Some("--video")
        } else if options.embedding {
            Some("--embedding")
        } else if options.dump_logits {
            Some("--dump-logits")
        } else if options.bench {
            Some("--bench")
        } else if options.profile {
            Some("--profile")
        } else {
            None
        };
        if let Some(conflict) = conflict {
            return Err(format!("--tts cannot be used with {conflict}"));
        }
        if options.edit {
            if options.source_audio.is_none() {
                return Err("--tts --edit requires --source-audio".into());
            }
            if options
                .instruction
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
            {
                return Err("--tts --edit requires a non-empty --instruction".into());
            }
            let conflict = if options.prompt.is_some() {
                Some("--prompt")
            } else if options.ref_audio.is_some() {
                Some("--ref-audio")
            } else if options.ref_text.is_some() {
                Some("--ref-text")
            } else if options.language.is_some() {
                Some("--language")
            } else {
                None
            };
            if let Some(conflict) = conflict {
                return Err(format!("--tts --edit cannot be used with {conflict}"));
            }
        } else {
            if options
                .prompt
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
            {
                return Err("--tts requires a non-empty --prompt".into());
            }
            normalize_tts_language(options.language.as_deref())?;
            if options.ref_text.is_some() && options.ref_audio.is_none() {
                return Err("--ref-text requires --ref-audio".into());
            }
        }
        return Ok(());
    }
    if options.ref_audio.is_some() || options.ref_text.is_some() {
        return Err("--ref-audio/--ref-text require --tts".into());
    }
    let media_count = usize::from(options.image.is_some())
        + usize::from(options.video.is_some())
        + usize::from(options.audio.is_some());
    if options.embedding {
        if media_count > 1 {
            return Err("--embedding accepts only one of --image, --video, or --audio".into());
        }
        if media_count == 1 {
            if options
                .mmproj
                .as_deref()
                .is_none_or(|path| path.as_os_str().is_empty())
            {
                return Err("media embedding requires --mmproj".into());
            }
            return Ok(());
        }
    }
    if options.video.is_some() {
        if options
            .mmproj
            .as_deref()
            .is_none_or(|path| path.as_os_str().is_empty())
        {
            return Err("--video requires --mmproj".into());
        }
        return Ok(());
    }
    // ---- --rerank: cross-encoder query/document scoring ----
    // Mirrors the conflict structure of --tts / --audio / --embedding above:
    // a single dispatch mode wins; --rerank is mutually exclusive with the
    // other heavyweight modes. Lighter inspection flags (--bench / --profile)
    // are still allowed because they don't pick a forward path of their own.
    if options.rerank {
        if options.model.as_os_str().is_empty() {
            return Err("--rerank requires --model".into());
        }
        let conflict = if options.tts {
            Some("--tts")
        } else if options.embedding {
            Some("--embedding")
        } else if options.jev {
            Some("--jev")
        } else if options.dreamx {
            Some("--dreamx")
        } else if options.yue2 {
            Some("--yue2")
        } else if options.gliner2_decide {
            Some("--gliner2-decide")
        } else if options.clm_head.is_some() {
            Some("--clm-head")
        } else if options.dump_logits {
            Some("--dump-logits")
        } else if options.audio.is_some() {
            Some("--audio")
        } else if options.image.is_some() {
            Some("--image")
        } else if options.video.is_some() {
            Some("--video")
        } else {
            None
        };
        if let Some(conflict) = conflict {
            return Err(format!("--rerank cannot be combined with {conflict}"));
        }
        if options
            .rerank_query
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err("--rerank requires --rerank-query <TEXT>".into());
        }
        if options.rerank_documents.is_empty() && options.rerank_documents_file.is_none() {
            return Err(
                "--rerank requires at least one document (use --rerank-doc <TEXT> repeatedly, \
                 or --rerank-documents <FILE>)"
                    .into(),
            );
        }
        return Ok(());
    }
    if options.audio.is_none() {
        return if options.language.is_some() {
            Err("--language requires --audio".into())
        } else {
            Ok(())
        };
    }
    // --audio requires either --mmproj (for encoder+LLM split models)
    // or a standalone ASR model (sensevoice-small, paraformer).
    // The actual routing happens in app::asr::run_asr_cli.
    // Skip the mmproj requirement here; it's enforced by the ASR dispatcher.
    let conflict = if options.embedding {
        Some("--embedding")
    } else if options.dump_logits {
        Some("--dump-logits")
    } else if options.bench {
        Some("--bench")
    } else if options.profile {
        Some("--profile")
    } else {
        None
    };
    if let Some(conflict) = conflict {
        return Err(format!("--audio cannot be used with {conflict}"));
    }
    if options.max_tokens == Some(0) {
        return Err("--audio requires --max-tokens greater than 0".into());
    }
    Ok(())
}

pub fn resolve_cli_generation_options(options: &CliOptions) -> (usize, f32) {
    (
        options
            .max_tokens
            .unwrap_or(if options.audio.is_some() { 256 } else { 128 }),
        options
            .temperature
            .unwrap_or(if options.audio.is_some() { 0.0 } else { 0.6 }),
    )
}

pub fn transcription_options(
    options: &CliOptions,
) -> crate::models::qwen3::asr::model::TranscriptionOptions {
    let language = options
        .language
        .as_ref()
        .filter(|language| !language.eq_ignore_ascii_case("auto"))
        .cloned();
    crate::models::qwen3::asr::model::TranscriptionOptions {
        language,
        prompt: options.prompt.clone(),
        max_new_tokens: resolve_cli_generation_options(options).0,
    }
}
