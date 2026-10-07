pub(crate) mod asr;
pub mod cli;
pub(crate) mod diffusion;
pub(crate) mod jev;
pub(crate) mod media;
pub(crate) mod omni;
pub(crate) mod qwen_drive;
pub(crate) mod selftest;
pub mod server;
pub mod text;
pub(crate) mod tts;
pub(crate) mod yue2;

pub use crate::models::qwen3::embedding::{
    compute_embedding as qwen3_compute_embedding, run_embedding as qwen3_run_embedding,
};
pub use asr::run_asr_cli;
pub use cli::{
    dreamx_cli_options, inference_step_budget, init_rayon_global_pool, longcat_cli_options,
    normalize_tts_language, parse_cli_options, per_second, qwen_drive_cli_options,
    resolve_cli_generation_options, resolve_thread_count, transcription_options,
    validate_cli_options, validate_qwen3vl_decoder_mode, yue2_cli_options, z_image_cli_options,
    CliOptions, DreamXCliOptions, DreamXOptions, DreamXRefinerOptions, EmbeddingOutput,
    JevBlockInput, KvFormat, LatentUpsampleKind, LongCatCliOptions, PlanningMode,
    QwenDriveCliOptions, QwenDriveHead, RefinerDecoderKind, YuE2CliOptions, ZImageCliOptions,
    DEFAULT_THREAD_CAP,
};
pub use diffusion::{
    read_f32_file, run_auk_cli, run_dreamx_cli, run_ernie_image_cli, run_longcat_image_edit,
    run_mage_flow_cli, run_pig_image, run_qwen_image_2_1, run_z_image_cli, write_output_atomically,
    write_png_atomically, QwenImage21Request,
};
pub use jev::{
    build_grouped_payload, build_grouped_system, build_jev_inputs, gliner2_schema,
    image_supported_arch, jev_payload_json, jev_system_prompt, load_gliner2_source, parse_schema,
    prepare_jev_grouped_questions, prepare_jev_questions, run_clm_decision, run_clm_scoring,
    run_gliner2_decision, run_gliner2_scoring, run_jev_decision, run_jev_decision_data,
    run_jev_grouped_decision, run_jev_grouped_decision_data, schema_from_label_sets,
    schema_from_questions, JevGroupInput, JevGroupedQuestionInput, JevInputs, JevMode,
    JevQuestionInput, JevResult, LabelSet,
};
pub use jev::{
    extract_long_document, parse_boundary_schema, run_gliner2_boundary,
    run_gliner2_boundary_extract, BoundaryDecodeOptions, BoundarySchemaOptions,
    LongDocumentOptions,
};
pub use media::{validate_mmproj_capabilities, MediaKind, ProjectorFamily};
pub use omni::run_omni_embedding;
pub use qwen_drive::run_qwen_drive_cli;
pub use selftest::run_self_test;
pub use text::{
    run_inference, run_interactive, run_interactive_qwen35, run_multimodal,
    run_multimodal_with_tts_postproc, run_multimodal_with_video,
    run_multimodal_with_video_capture_text, run_qwen35_family_multimodal_logits,
    run_qwen3_family_multimodal, run_qwen3_family_multimodal_logits, run_shared_inference,
};
pub use tts::{run_tts_cli, synthesize_tts_to_wav};
pub use yue2::run_yue2_cli;

use crate::core::tensor::TensorSource;
use crate::format::ggufrs::{open_model_source, ComponentRole};
use std::path::Path;
use std::sync::Arc;

/// `general.architecture` of an already-open source.
pub fn arch_of(source: &dyn TensorSource) -> String {
    source
        .metadata("general.architecture")
        .and_then(crate::core::tensor::MetaValue::to_string_val)
        .unwrap_or_default()
        .to_string()
}

/// Embedding entry point dispatched on `general.architecture`.
///
/// Only archs with a dedicated, byte-level verified implementation are routed;
/// an unknown arch is an error rather than a silent fallback (per
/// `.agents/skills/adapting-new-models`).
pub fn compute_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    if crate::models::bitnet::detect_is_bitnet(source) {
        return crate::models::bitnet::compute_embedding(source, prompt, n_threads_arg);
    }
    match arch_of(source).as_str() {
        "gemma-embedding" => {
            crate::models::gemma_embedding::compute_embedding(source, prompt, n_threads_arg)
        }
        // gemma3 arch: BitNet b1.58 270M (file_type=40) hits the
        // detect_is_bitnet gate above; the standard gemma3 270M-it
        // (and future 1B/4B/12B/27B) falls through to the standard
        // path here. Both paths live in `models::gemma3` and share
        // the same SPM tokenizer (`tokenizer.ggml.model = "llama"`,
        // `pre = "default"`, no merges).
        "gemma3" => crate::models::gemma3::compute_embedding(source, prompt, n_threads_arg),
        // gemma2 (standard, non-BitNet). Shares tensor layout with
        // the llama trunk; arch-specific GeGLU + softcap + sliding
        // window is wired inside `models::llama::trunk`.
        "gemma2" => crate::models::gemma2::compute_embedding(source, prompt, n_threads_arg),
        "bert" | "jina-bert-v2" | "nomic-bert" | "nomic-bert-moe" => {
            crate::models::bert_family::compute_embedding(source, prompt, n_threads_arg)
        }
        _ => qwen3_compute_embedding(source, prompt, n_threads_arg),
    }
}

/// CLI-facing embedding entry point, same dispatch rule as [`compute_embedding`].
pub fn run_embedding(
    source: &dyn TensorSource,
    prompt: &str,
    n_threads_arg: usize,
    kv_format: KvFormat,
    output: EmbeddingOutput,
) {
    let arch = arch_of(source);
    match arch.as_str() {
        "gemma-embedding" => crate::models::gemma_embedding::run_embedding(
            source,
            prompt,
            n_threads_arg,
            kv_format,
            output,
        ),
        // BitNet b1.58 (file_type=40 + per-projection `*_norm_in`
        // tensors). Architecture-neutral dispatch inside the BitNet
        // trunk family: `bitnet::compute_embedding` reads
        // `general.architecture` and routes to `qwen3_arch` /
        // `gemma3_arch` accordingly. The qwen3 / gemma3 standard
        // trunks below are BitNet-free.
        arch if crate::models::bitnet::detect_is_bitnet(source) => {
            crate::models::bitnet::run_embedding(source, prompt, n_threads_arg, kv_format, output)
        }
        "bert" | "jina-bert-v2" | "nomic-bert" | "nomic-bert-moe" => {
            crate::models::bert_family::run_embedding(
                source,
                prompt,
                n_threads_arg,
                kv_format,
                output,
            )
        }
        "gemma3" => {
            crate::models::gemma3::run_embedding(source, prompt, n_threads_arg, kv_format, output)
        }
        _ => qwen3_run_embedding(source, prompt, n_threads_arg, kv_format, output),
    }
}

/// Cross-encoder rerank entry point. Dispatches on architecture:
///
/// - `jina-bert-v2` with `cls.weight` + `cls.bias`: bidirectional BERT
///   forward + CLS-row projection (`src/models/bert_family::compute_rerank_score`).
/// - `qwen3` with `pooling_type = 4` + `cls.output.weight`: causal prefill
///   of the standard ChatML prompt + last-token classification head
///   (`src/models/qwen3::trunk`).
///
/// Anything else returns an error rather than falling back to a generic
/// embedder — rerank is a deliberately arch-locked contract and silently
/// degrading to embedding would be the wrong failure mode.
pub fn run_rerank(
    source: Arc<dyn TensorSource>,
    query: &str,
    documents: &[String],
    n_threads_arg: usize,
    instruction: Option<&str>,
) -> Result<Vec<f32>, String> {
    let arch = arch_of(source.as_ref());
    match arch.as_str() {
        "jina-bert-v2" => crate::models::bert_family::compute_rerank_score(
            source.as_ref(),
            query,
            documents,
            n_threads_arg,
        ),
        "qwen3" => crate::models::qwen3::trunk::score_qwen3_rerank(
            source,
            query,
            documents,
            n_threads_arg,
            instruction,
        ),
        other => Err(format!(
            "--rerank does not support architecture {other:?}; \
             expected jina-bert-v2 (with cls.weight + cls.bias) or qwen3 \
             (with pooling_type=4 + cls.output.weight)"
        )),
    }
}

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

/// Unwrap a startup-time value or exit with the usual error prefix, for the
/// call sites inside a `match` that already returned.
pub fn unwrap_or_exit<T>(result: Result<T, String>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            eprintln!("Inference error: {error}");
            std::process::exit(1);
        }
    }
}

pub fn run_or_exit(result: Result<(), String>) {
    if let Err(error) = result {
        eprintln!("Inference error: {error}");
        std::process::exit(1);
    }
}
