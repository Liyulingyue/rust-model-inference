use std::path::PathBuf;

pub use crate::core::scratchpad::KvFormat;
pub use crate::models::diffusion::dreamx::{
    DreamXOptions, DreamXRefinerOptions, LatentUpsampleKind, RefinerDecoderKind,
};
pub use crate::models::dots::XVectorMode;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmbeddingOutput {
    #[default]
    Summary,
    Raw,
}

#[derive(Debug, Default)]
pub struct CliOptions {
    pub dreamx: bool,
    pub planner: Option<PathBuf>,
    pub perception: Option<PathBuf>,
    pub scenes: Option<PathBuf>,
    pub image_root: Option<PathBuf>,
    pub frames: Option<PathBuf>,
    pub planning_mode: Option<String>,
    pub num_samples: Option<usize>,
    pub num_steps: Option<usize>,
    pub output: Option<PathBuf>,
    pub model: PathBuf,
    pub mmproj: Option<PathBuf>,
    pub audio: Option<PathBuf>,
    pub ref_audio: Option<PathBuf>,
    pub ref_text: Option<String>,
    pub image: Option<PathBuf>,
    pub video: Option<PathBuf>,
    pub vae: Option<PathBuf>,
    pub text_encoder: Option<PathBuf>,
    pub prompt: Option<String>,
    pub negative_prompt: Option<String>,
    pub system: Option<String>,
    pub chat_mode: bool,
    pub language: Option<String>,
    pub max_tokens: Option<usize>,
    pub max_context: Option<usize>,
    pub prefill_batch_size: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub steps: Option<usize>,
    pub resolution: Option<usize>,
    pub seed: Option<i64>,
    pub duration_seconds: Option<f32>,
    pub fps: Option<usize>,
    pub target_spatial_tokens: Option<usize>,
    pub refine: Option<bool>,
    pub refiner_kv_len: Option<usize>,
    pub latent_upsample: Option<LatentUpsampleKind>,
    pub refiner_decoder: Option<RefinerDecoderKind>,
    pub dry_run: bool,
    pub overwrite: bool,
    pub allow_memory_overcommit: bool,
    pub temperature: Option<f32>,
    pub cfg_scale: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub threads: usize,
    pub thinking: bool,
    pub embedding: bool,
    pub embedding_output: EmbeddingOutput,
    pub dump_logits: bool,
    pub bench: bool,
    pub profile: bool,
    pub kv_format: KvFormat,
    pub gpu: bool,
    pub tts: bool,
    pub edit: bool,
    pub source_audio: Option<PathBuf>,
    pub source_text: Option<String>,
    pub target_text: Option<String>,
    pub instruction: Option<String>,
    pub use_xvector: XVectorMode,
    pub use_xvector_supplied: bool,
    pub out: Option<PathBuf>,
    pub tts_model: Option<PathBuf>,
    pub tts_mmproj: Option<PathBuf>,
    pub jev: bool,
    pub jev_context: Option<String>,
    pub jev_questions: Vec<JevQuestion>,
    pub jev_positive: Option<String>,
    pub jev_output_json: bool,
    pub jev_multi: bool,
    pub jev_blocks: Vec<JevBlockInput>,
    pub chunk_seconds: Option<f64>,
    pub srt: bool,
    pub vad: Option<PathBuf>,
    pub vad_maxseg: usize,
}

#[derive(Clone, Debug, Default)]
pub struct JevQuestion {
    pub text: String,
    pub options: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct JevBlockInput {
    pub label: String,
    pub options: Vec<String>,
}

impl CliOptions {
    pub fn effective_prefill_batch_size(&self) -> Result<usize, String> {
        crate::core::prefill::checked_prefill_batch_size(self.prefill_batch_size)
    }

    /// Default max-context cap. Most chat workloads fit in 8K; this
    /// guards against models that declare an over-large
    /// `context_length` (e.g. K2-Horizon-4B claims 524288, which would
    /// require ~77 GB of F32 KV cache).
    pub const DEFAULT_MAX_CONTEXT: usize = 8192;

    pub fn effective_max_context(&self) -> usize {
        self.max_context.unwrap_or(Self::DEFAULT_MAX_CONTEXT)
    }

    /// Effective repetition penalty for sampling. `None` / `Some(1.0)` means
    /// disabled; values > 1.0 suppress already-generated tokens (Hugging Face
    /// / llama.cpp definition), values < 1.0 encourage repeats.  We don't
    /// validate against `< 1.0` because users may intentionally want
    /// repetition in some prompts.
    pub fn effective_repetition_penalty(&self) -> f32 {
        self.repetition_penalty.unwrap_or(1.0)
    }
}

#[derive(Debug)]
pub struct ZImageCliOptions {
    pub steps: usize,
    pub resolution: usize,
    pub seed: i64,
    pub out: PathBuf,
}

#[derive(Debug, PartialEq)]
pub struct DreamXCliOptions {
    pub model: PathBuf,
    pub mmproj: PathBuf,
    pub image: PathBuf,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub out: PathBuf,
    pub options: DreamXOptions,
    pub dry_run: bool,
    pub overwrite: bool,
    pub allow_memory_overcommit: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum QwenDriveHead {
    Planner(PathBuf),
    Perception(PathBuf),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanningMode {
    Direct,
    Reasoning,
}

#[derive(Debug, PartialEq, Eq)]
pub struct QwenDriveCliOptions {
    pub model: PathBuf,
    pub mmproj: PathBuf,
    pub head: QwenDriveHead,
    pub mode: PlanningMode,
    pub scenes: Option<PathBuf>,
    pub image_root: Option<PathBuf>,
    pub frames: Option<PathBuf>,
    pub output: PathBuf,
    pub samples: usize,
    pub steps: usize,
    pub seed: i64,
}

pub fn parse_embedding_output(value: Option<&str>) -> Result<EmbeddingOutput, String> {
    match value {
        Some("summary") => Ok(EmbeddingOutput::Summary),
        Some("raw") => Ok(EmbeddingOutput::Raw),
        Some(value) => Err(format!(
            "Invalid --embedding-output {value:?}; expected summary or raw"
        )),
        None => Err("Missing value for --embedding-output".into()),
    }
}

pub fn normalize_tts_language(language: Option<&str>) -> Result<&'static str, String> {
    match language.unwrap_or("en").to_ascii_lowercase().as_str() {
        "cn" | "zh" | "chinese" => Ok("chinese"),
        "en" | "english" => Ok("english"),
        "ge" | "de" | "german" => Ok("german"),
        "it" | "italian" => Ok("italian"),
        "po" | "pt" | "portuguese" => Ok("portuguese"),
        "sp" | "es" | "spanish" => Ok("spanish"),
        "ja" | "japanese" => Ok("japanese"),
        "ko" | "korean" => Ok("korean"),
        "fr" | "french" => Ok("french"),
        "ru" | "russian" => Ok("russian"),
        value => Err(format!("Unsupported TTS language {value:?}")),
    }
}
