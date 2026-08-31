pub(crate) mod audio;
pub mod cli;
pub(crate) mod embedding;
pub mod gemma4;
pub(crate) mod image;
pub(crate) mod omni;
pub(crate) mod selftest;
pub(crate) mod text;
pub mod tts;

pub use audio::run_asr_cli;
pub use cli::{parse_cli_options, validate_cli_options, normalize_tts_language, resolve_cli_generation_options, transcription_options, resolve_thread_count, init_rayon_global_pool, validate_qwen3vl_decoder_mode, z_image_cli_options, CliOptions, EmbeddingOutput, KvFormat, ZImageCliOptions, DEFAULT_THREAD_CAP, per_second, inference_step_budget};
pub use embedding::{compute_embedding, run_embedding};
pub use gemma4::{run_gemma4, Gemma4Request};
pub use image::{run_pig_image, run_z_image_cli, write_png_atomically};
pub use omni::run_omni_embedding;
pub use selftest::run_self_test;
pub use text::{run_inference, run_interactive, run_shared_inference, run_multimodal};
pub use tts::{run_tts_cli, synthesize_tts_to_wav};

use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::core::tensor::TensorSource;
use std::path::Path;

pub fn reject_incomplete_z_image_architecture(arch: &str) -> Result<(), String> {
    if arch == "pig" {
        return Err("Z-Image model requires --text-encoder, --vae, --prompt, and --out".into());
    }
    Ok(())
}

pub fn open_or_exit(path: &Path, role: ComponentRole) -> Box<dyn TensorSource> {
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

pub fn run_or_exit(result: Result<(), String>) {
    if let Err(error) = result {
        eprintln!("Inference error: {error}");
        std::process::exit(1);
    }
}
