use std::path::Path;
use std::sync::Arc;

use rust_model_inference::app;
use rust_model_inference::format::ggufrs::ComponentRole;
use rust_model_inference::open_model_source;
use rust_model_inference::ops;
use rust_model_inference::MetaValue;
use rust_model_inference::TensorSource;

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

    if options.gpu {
        ops::enable_gpu();
    }

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

    if options.tts {
        app::run_or_exit(app::run_tts_cli(&options));
        return;
    }

    if options.audio.is_some() {
        app::run_or_exit(app::run_asr_cli(&options));
        return;
    }

    let (max_tokens, temperature) = app::resolve_cli_generation_options(&options);
    let prompt = options.prompt.as_deref().unwrap_or_default();

    let model_path = options.model.as_path();
    let source: Arc<dyn TensorSource> =
        std::sync::Arc::from(open_or_exit(model_path, ComponentRole::Llm));
    let arch = source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .unwrap_or_default();
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
        } else if arch == "pig" {
            let vae_source = if let Some(ref vae_path) = options.vae {
                Some(std::sync::Arc::from(open_or_exit(vae_path, ComponentRole::Llm)) as std::sync::Arc<dyn TensorSource>)
            } else {
                None
            };
            let text_encoder_source = if let Some(ref te_path) = options.text_encoder {
                Some(std::sync::Arc::from(open_or_exit(te_path, ComponentRole::Llm)) as std::sync::Arc<dyn TensorSource>)
            } else {
                None
            };
            app::run_or_exit(app::run_pig_image(
                std::sync::Arc::clone(&source),
                vae_source,
                text_encoder_source,
                prompt,
                options.steps.unwrap_or(20),
                options.resolution.unwrap_or(512),
                options.threads,
            ));
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
    } else if arch == "pig" {
        app::run_or_exit(app::run_pig_image(
            std::sync::Arc::from(source),
            None,
            None,
            prompt,
            options.steps.unwrap_or(20),
            options.resolution.unwrap_or(512),
            options.threads,
        ));
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
