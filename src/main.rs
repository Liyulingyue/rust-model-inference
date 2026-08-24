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
    Asr,
    Model,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GenericDispatchPlan {
    enable_gpu: bool,
    mode: GenericDispatchMode,
}

fn generic_dispatch_plan(
    arch: &str,
    options: &app::CliOptions,
) -> Result<GenericDispatchPlan, String> {
    if arch == "pig" {
        return Err("Z-Image model requires --text-encoder, --vae, --prompt, and --out".into());
    }
    let mode = if options.tts {
        GenericDispatchMode::Tts
    } else if options.audio.is_some() {
        GenericDispatchMode::Asr
    } else {
        GenericDispatchMode::Model
    };
    Ok(GenericDispatchPlan {
        enable_gpu: options.gpu,
        mode,
    })
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

    let model_path = options.model.as_path();
    let source: Arc<dyn TensorSource> = Arc::from(open_or_exit(model_path, ComponentRole::Llm));
    let arch = source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .unwrap_or_default();
    let dispatch = match generic_dispatch_plan(&arch, &options) {
        Ok(dispatch) => dispatch,
        Err(error) => {
            app::run_or_exit(Err(error));
            return;
        }
    };

    if dispatch.enable_gpu {
        ops::enable_gpu();
    }

    match dispatch.mode {
        GenericDispatchMode::Tts => {
            drop(source);
            app::run_or_exit(app::run_tts_cli(&options));
            return;
        }
        GenericDispatchMode::Asr => {
            drop(source);
            app::run_or_exit(app::run_asr_cli(&options));
            return;
        }
        GenericDispatchMode::Model => {}
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

    if explicit_mmproj.is_some() || image.is_some() {
        app::run_or_exit(app::run_multimodal(
            source.as_ref(),
            model_path,
            explicit_mmproj,
            image,
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
                source.as_ref(),
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
                source.as_ref(),
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
            source.as_ref(),
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

    fn assert_pig_route_is_rejected(options: app::CliOptions) {
        assert_eq!(
            generic_dispatch_plan("pig", &options).unwrap_err(),
            "Z-Image model requires --text-encoder, --vae, --prompt, and --out"
        );
    }

    #[test]
    fn pig_only_tts_is_rejected_before_tts_dispatch() {
        assert_pig_route_is_rejected(app::CliOptions {
            tts: true,
            ..app::CliOptions::default()
        });
    }

    #[test]
    fn pig_only_asr_is_rejected_before_asr_dispatch() {
        assert_pig_route_is_rejected(app::CliOptions {
            audio: Some("speech.wav".into()),
            ..app::CliOptions::default()
        });
    }

    #[test]
    fn pig_only_gpu_is_rejected_before_gpu_enablement() {
        assert_pig_route_is_rejected(app::CliOptions {
            gpu: true,
            ..app::CliOptions::default()
        });
    }
}
