use std::path::Path;
use std::sync::Arc;

use rust_model_inference::app;
use rust_model_inference::format::ggufrs::ComponentRole;
use rust_model_inference::open_model_source;
use rust_model_inference::ops;
use rust_model_inference::DreamXConfig;
use rust_model_inference::MetaValue;
use rust_model_inference::TensorSource;

const USAGE: &str = "Usage: rust-model-inference --model <path.gguf-or-ggufrs> [--prompt ...] [--threads N] [--kv-cache f16|f32] [--prefill-batch-size N (default 64)] [--max-context N (default 8192)] [--repetition-penalty α (default 1.0 = disabled)] [--serve [--host 0.0.0.0] [--port 8080]]\n\nJEV mode: --jev --jev-context <text> --jev-question <text> --jev-option <a> [--jev-option <b> ...] | single-forward-pass decision scoring over candidate labels A/B/C/…\n\nJEV grouped: --jev --jev-multi [--jev-option <pos> --jev-option <neg> ...] (pairs) or --jev-block <label> --jev-option <a> [--jev-option <b> ...] (blocks)\n\nServer mode: --serve [--host 0.0.0.0] [--port 8080] --model <path> [--mmproj ...] [--tts] [--embedding]";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DispatchMode {
    DreamX,
    QwenDrive,
    Tts,
    Model,
}

fn dispatch_mode(options: &app::CliOptions) -> DispatchMode {
    if options.dreamx {
        DispatchMode::DreamX
    } else if options.planner.is_some() || options.perception.is_some() {
        DispatchMode::QwenDrive
    } else if options.tts {
        DispatchMode::Tts
    } else {
        DispatchMode::Model
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioRoute {
    Asr,
    Gemma4,
    Multimodal,
}

fn dispatch_audio_arch(arch: &str) -> Result<AudioRoute, String> {
    match arch {
        // qwen3 covers Fun-ASR-Nano (qwen3-0.6b LLM + funasr encoder mmproj);
        // qwen3vl covers Qwen3-Audio ASR; qwen2 covers VibeVoice ASR;
        // sensevoice-small and paraformer are standalone ASR models.
        "qwen3" | "qwen3vl" | "qwen2" | "sensevoice-small" | "paraformer" => Ok(AudioRoute::Asr),
        "qwen2vl" | "qwen3vlmoe" => Ok(AudioRoute::Multimodal),
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

fn validate_audio_temperature(route: AudioRoute, temperature: Option<f32>) -> Result<(), String> {
    if route == AudioRoute::Asr && temperature.is_some_and(|value| value != 0.0) {
        return Err("Qwen3-ASR requires greedy decoding; --temp must be 0".into());
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return;
    }
    // Server mode: --serve delegates to app::server::run_server which has its
    // own --host/--port pre-parser and reuses the shared CLI parser for the rest.
    if args.iter().any(|arg| arg == "--serve") {
        app::server::run_server();
        return;
    }
    let options = app::parse_cli_options(&args).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    app::validate_cli_options(&options).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let prefill_batch_size = options
        .effective_prefill_batch_size()
        .unwrap_or_else(|error| {
            eprintln!("{error}");
            std::process::exit(2);
        });
    let dreamx_options = app::dreamx_cli_options(&options).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let z_image_options = app::z_image_cli_options(&options).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let qwen_drive_options = app::qwen_drive_cli_options(&options).unwrap_or_else(|error| {
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

    match dispatch_mode(&options) {
        DispatchMode::DreamX => {
            let dreamx = dreamx_options.expect("validated DreamX options");
            let main: Arc<dyn TensorSource> =
                Arc::from(open_or_exit(&dreamx.model, ComponentRole::Llm));
            let mmproj: Arc<dyn TensorSource> =
                Arc::from(open_or_exit(&dreamx.mmproj, ComponentRole::Mmproj));
            app::run_or_exit(
                DreamXConfig::from_sources(main.as_ref(), mmproj.as_ref()).map(|_| ()),
            );
            app::run_or_exit(app::run_dreamx_cli(main, mmproj, dreamx, n_threads));
            return;
        }
        DispatchMode::Tts => {
            app::run_or_exit(app::run_tts_cli(&options));
            return;
        }
        DispatchMode::QwenDrive => {
            app::run_or_exit(app::run_qwen_drive_cli(
                qwen_drive_options.expect("validated Qwen-Drive options"),
                n_threads,
            ));
            return;
        }
        DispatchMode::Model => {}
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
    let tts_model = options
        .tts_model
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty());
    let tts_mmproj = options
        .tts_mmproj
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
        app::run_or_exit(validate_audio_temperature(route, options.temperature));
        if route == AudioRoute::Asr {
            app::run_or_exit(app::run_asr_cli(&options, prefill_batch_size));
            return;
        }
    }

    if arch == "gemma4" {
        app::run_or_exit(app::run_multimodal_with_video(
            Arc::clone(&source),
            model_path,
            explicit_mmproj,
            image,
            video,
            audio,
            prompt,
            max_tokens,
            options.temperature.unwrap_or(0.0),
            n_threads,
            prefill_batch_size,
            options.effective_max_context(),
            options.effective_repetition_penalty(),
        ));
    } else if explicit_mmproj.is_some() || image.is_some() || video.is_some() || audio.is_some() {
        // Omni → TTS post-processor pipeline: when both the multimodal
        // media path AND a TTS model + mmproj are present, route the
        // generated reply through Qwen3-TTS to produce a 24 kHz WAV
        // alongside the text. The TTS layer is opt-in so the existing
        // text-only multimodal flow is unchanged for users without a
        // bundled TTS model.
        if let (Some(tts_model_path), Some(tts_mmproj_path)) = (tts_model, tts_mmproj) {
            let wav_out = match options
                .out
                .as_deref()
                .filter(|path| !path.as_os_str().is_empty())
            {
                Some(path) => path,
                None => {
                    eprintln!("Inference error: --tts-model and --tts-mmproj require --out <wav>");
                    std::process::exit(1);
                }
            };
            let language = options.language.as_deref().unwrap_or("en");
            app::run_or_exit(app::run_multimodal_with_tts_postproc(
                Arc::clone(&source),
                model_path,
                explicit_mmproj,
                image,
                video,
                audio,
                prompt,
                max_tokens,
                temperature,
                options.threads,
                prefill_batch_size,
                options.effective_max_context(),
                options.effective_repetition_penalty(),
                tts_model_path,
                tts_mmproj_path,
                wav_out,
                language,
            ));
            return;
        }
        app::run_or_exit(app::run_multimodal_with_video(
            Arc::clone(&source),
            model_path,
            explicit_mmproj,
            image,
            video,
            audio,
            prompt,
            max_tokens,
            temperature,
            options.threads,
            prefill_batch_size,
            options.effective_max_context(),
            options.effective_repetition_penalty(),
        ));
    } else if options.jev {
        match app::build_jev_inputs(&options) {
            Ok(Some(app::JevInputs::Grouped {
                context,
                questions,
                mode,
            })) => {
                app::run_or_exit(app::run_jev_grouped_decision(
                    source.clone(),
                    &context,
                    &questions,
                    mode,
                    n_threads,
                    prefill_batch_size,
                    options.jev_output_json,
                ));
            }
            Ok(Some(app::JevInputs::Single {
                context,
                questions,
                positive,
            })) => {
                app::run_or_exit(app::run_jev_decision(
                    source.clone(),
                    &context,
                    &questions,
                    positive.as_deref(),
                    n_threads,
                    prefill_batch_size,
                    options.jev_output_json,
                ));
            }
            Ok(None) => {}
            Err(e) => {
                app::run_or_exit(Err(e));
                return;
            }
        }
        return;
    } else if !prompt.is_empty() {
        if arch == "qwen35" {
            app::run_or_exit(app::run_multimodal_with_video(
                Arc::clone(&source),
                model_path,
                None,
                None,
                video,
                None,
                prompt,
                max_tokens,
                temperature,
                options.threads,
                prefill_batch_size,
                options.effective_max_context(),
                options.effective_repetition_penalty(),
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
                prefill_batch_size,
            ));
        } else {
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
                prefill_batch_size,
                options.effective_max_context(),
                options.effective_repetition_penalty(),
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
        // Qwen3.5 (qwen35) ships its own dense transformer; the generic
        // qwen3 text dispatch would fail to load it because the
        // per-layer tensor names differ (`blk.{i}.attn_norm` only, no
        // `ffn_norm`).  Mirror `run_interactive` so the chat template,
        // tokenizer and prompt token ids line up with the rest of the
        // qwen35 family; the qwen35 multimodal stack handles the
        // image-less case (it just skips the vision stage).
        if arch == "qwen35" {
            app::run_or_exit(app::run_interactive_qwen35(
                Arc::clone(&source),
                model_path,
                max_tokens,
                temperature,
                options.threads,
                prefill_batch_size,
                options.effective_max_context(),
                options.effective_repetition_penalty(),
            ));
            return;
        }
        app::run_or_exit(app::run_interactive(
            source.clone(),
            max_tokens,
            temperature,
            options.threads,
            prefill_batch_size,
            options.effective_repetition_penalty(),
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
    fn dreamx_and_tts_dispatch_before_main_model_open() {
        assert_eq!(
            dispatch_mode(&app::CliOptions {
                dreamx: true,
                ..app::CliOptions::default()
            }),
            DispatchMode::DreamX
        );
        assert_eq!(
            dispatch_mode(&app::CliOptions {
                tts: true,
                ..app::CliOptions::default()
            }),
            DispatchMode::Tts
        );
        assert_eq!(
            dispatch_mode(&app::CliOptions {
                audio: Some("speech.wav".into()),
                ..app::CliOptions::default()
            }),
            DispatchMode::Model
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
    fn only_qwen_asr_keeps_greedy_audio_validation() {
        assert!(validate_audio_temperature(AudioRoute::Asr, Some(0.1)).is_err());
        assert!(validate_audio_temperature(AudioRoute::Multimodal, Some(0.1)).is_ok());
    }

    #[test]
    fn ordinary_model_dispatch_stays_in_main() {
        assert_eq!(
            dispatch_mode(&app::CliOptions::default()),
            DispatchMode::Model
        );
    }
}
