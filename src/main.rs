use std::path::Path;
use std::sync::Arc;

use rust_model_inference::app;
use rust_model_inference::format::ggufrs::ComponentRole;
use rust_model_inference::open_model_source;
use rust_model_inference::ops;
use rust_model_inference::MetaValue;
use rust_model_inference::TensorSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenericDispatchMode {
    Tts,
    Model,
}

fn generic_dispatch_mode(options: &app::CliOptions) -> GenericDispatchMode {
    if options.tts {
        GenericDispatchMode::Tts
    } else {
        GenericDispatchMode::Model
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioRoute {
    Asr,
    Gemma4,
}

fn dispatch_audio_arch(arch: &str) -> Result<AudioRoute, String> {
    match arch {
        "qwen3vl" => Ok(AudioRoute::Asr),
        "gemma4" => Ok(AudioRoute::Gemma4),
        _ => Err(format!(
            "--audio is not supported for architecture {arch:?}"
        )),
    }
}

fn validate_audio_route(route: AudioRoute, has_image: bool) -> Result<(), String> {
    if route == AudioRoute::Asr && has_image {
        return Err("Qwen3-ASR --audio cannot be used with --image".into());
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let options = app::parse_cli_options(&args).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    app::validate_cli_options(&options).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let z_image_options = app::z_image_cli_options(&options).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });

    // Resolved thread count for both LLM ComputePool and rayon global pool.
    let available_threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let n_threads = app::resolve_thread_count(options.threads, available_threads);
    app::init_rayon_global_pool(n_threads);

    if options.model.as_os_str().is_empty() {
        app::run_self_test();
        return;
    }

    if let Some(z_image_options) = z_image_options {
        let diffusion: Arc<dyn TensorSource> =
            Arc::from(open_or_exit(&options.model, ComponentRole::Llm));
        let text: Arc<dyn TensorSource> = Arc::from(open_or_exit(
            options
                .text_encoder
                .as_deref()
                .expect("validated Z-Image text encoder"),
            ComponentRole::Llm,
        ));
        let vae: Arc<dyn TensorSource> = Arc::from(open_or_exit(
            options.vae.as_deref().expect("validated Z-Image VAE"),
            ComponentRole::Llm,
        ));
        app::run_or_exit(app::run_z_image_cli(
            diffusion,
            text,
            vae,
            options.prompt.as_deref().expect("validated Z-Image prompt"),
            z_image_options,
            n_threads,
        ));
        return;
    }

    match generic_dispatch_mode(&options) {
        GenericDispatchMode::Tts => {
            app::run_or_exit(app::run_tts_cli(&options));
            return;
        }
        GenericDispatchMode::Model => {}
    }

    let model_path = options.model.as_path();
    let source: Arc<dyn TensorSource> = Arc::from(open_or_exit(model_path, ComponentRole::Llm));
    let arch = source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .unwrap_or_default();
    if let Err(error) = app::reject_incomplete_z_image_architecture(&arch) {
        app::run_or_exit(Err(error));
        return;
    }

    if options.gpu {
        ops::enable_gpu();
    }

    let (max_tokens, temperature) = app::resolve_cli_generation_options(&options);
    let prompt = options.prompt.as_deref().unwrap_or_default();

    let explicit_mmproj = options
        .mmproj
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty());
    let image = options
        .image
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty());
    let audio = options
        .audio
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty());
    let video = options
        .video
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty());

    if options.embedding && (image.is_some() || video.is_some() || audio.is_some()) {
        app::run_or_exit(app::run_omni_embedding(
            source.as_ref(),
            explicit_mmproj.expect("validated media embedding mmproj"),
            image,
            video,
            audio,
            prompt,
            options.threads,
            options.embedding_output,
        ));
        return;
    }

    if audio.is_some() {
        let route = dispatch_audio_arch(&arch).unwrap_or_else(|error| {
            eprintln!("Inference error: {error}");
            std::process::exit(1);
        });
        app::run_or_exit(validate_audio_route(route, image.is_some()));
        if route == AudioRoute::Asr {
            app::run_or_exit(app::run_asr_cli(&options));
            return;
        }
    }

    if arch == "gemma4" {
        app::run_or_exit(app::run_multimodal(
            source.as_ref(),
            model_path,
            explicit_mmproj,
            image,
            audio,
            prompt,
            max_tokens,
            options.temperature.unwrap_or(0.0),
            n_threads,
        ));
    } else if explicit_mmproj.is_some() || image.is_some() {
        app::run_or_exit(app::run_multimodal(
            source.as_ref(),
            model_path,
            explicit_mmproj,
            image,
            None,
            prompt,
            max_tokens,
            temperature,
            options.threads,
        ));
    } else if !prompt.is_empty() {
        if arch == "qwen35" {
            app::run_or_exit(app::run_multimodal(
                source.as_ref(),
                model_path,
                None,
                None,
                None,
                prompt,
                max_tokens,
                temperature,
                options.threads,
            ));
        } else if options.embedding {
            app::run_embedding(
                source.as_ref(),
                prompt,
                options.threads,
                options.kv_format,
                options.embedding_output,
            );
        } else if arch == "qwen3vl" {
            app::run_or_exit(app::validate_qwen3vl_decoder_mode(
                &arch,
                options.dump_logits,
                options.bench,
                options.profile,
                options.kv_format,
                false,
            ));
            app::run_or_exit(app::run_shared_inference(
                std::sync::Arc::clone(&source),
                prompt,
                max_tokens,
                temperature,
                options.threads,
                options.thinking,
            ));
        } else if options.bench || options.profile || options.kv_format == app::KvFormat::F32 {
            app::run_or_exit(app::run_inference(
                source.clone(),
                prompt,
                max_tokens,
                temperature,
                options.threads,
                options.thinking,
                options.bench,
                options.profile,
                options.kv_format,
            ));
        } else {
            app::run_or_exit(app::run_inference(
                source.clone(),
                prompt,
                max_tokens,
                temperature,
                options.threads,
                options.thinking,
                false,
                false,
                options.kv_format,
            ));
        }
    } else {
        app::run_or_exit(app::validate_qwen3vl_decoder_mode(
            &arch,
            options.dump_logits,
            options.bench,
            options.profile,
            options.kv_format,
            true,
        ));
        app::run_or_exit(app::run_interactive(
            source.clone(),
            max_tokens,
            temperature,
            options.threads,
        ));
    }
}

fn open_or_exit(path: &Path, role: ComponentRole) -> Box<dyn TensorSource> {
    open_model_source(path, role).unwrap_or_else(|error| {
        eprintln!(
            "Failed to load {} component from {}: {error}",
            match role {
                ComponentRole::Llm => "LLM",
                ComponentRole::Mmproj => "mmproj",
            },
            path.display(),
        );
        std::process::exit(1);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pig_architecture_requires_complete_z_image_route() {
        assert_eq!(
            app::reject_incomplete_z_image_architecture("pig").unwrap_err(),
            "Z-Image model requires --text-encoder, --vae, --prompt, and --out"
        );
    }

    #[test]
    fn only_tts_dispatches_before_main_model_open() {
        assert_eq!(
            generic_dispatch_mode(&app::CliOptions {
                tts: true,
                ..app::CliOptions::default()
            }),
            GenericDispatchMode::Tts
        );
        assert_eq!(
            generic_dispatch_mode(&app::CliOptions {
                audio: Some("speech.wav".into()),
                ..app::CliOptions::default()
            }),
            GenericDispatchMode::Model
        );
    }

    #[test]
    fn architecture_aware_audio_dispatch_preserves_asr() {
        assert_eq!(dispatch_audio_arch("qwen3vl").unwrap(), AudioRoute::Asr);
        assert_eq!(dispatch_audio_arch("gemma4").unwrap(), AudioRoute::Gemma4);
        assert!(dispatch_audio_arch("llama").is_err());
    }

    #[test]
    fn qwen_asr_audio_route_rejects_images() {
        assert!(validate_audio_route(AudioRoute::Asr, true).is_err());
        assert!(validate_audio_route(AudioRoute::Asr, false).is_ok());
        assert!(validate_audio_route(AudioRoute::Gemma4, true).is_ok());
    }

    #[test]
    fn ordinary_model_dispatch_stays_in_main() {
        assert_eq!(
            generic_dispatch_mode(&app::CliOptions::default()),
            GenericDispatchMode::Model
        );
    }
}
