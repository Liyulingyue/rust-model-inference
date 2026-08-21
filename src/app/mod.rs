pub(crate) mod audio;
pub(crate) mod cli;
pub(crate) mod embedding;
pub(crate) mod image;
pub(crate) mod logits;
pub(crate) mod selftest;
pub(crate) mod text;
pub(crate) mod tts;

pub use audio::run_asr_cli;
pub use cli::{parse_cli_options, validate_cli_options, resolve_cli_generation_options, transcription_options, resolve_thread_count, init_rayon_global_pool, validate_qwen3vl_decoder_mode, CliOptions, EmbeddingOutput, KvFormat, DEFAULT_THREAD_CAP, per_second, inference_step_budget};
pub use embedding::run_embedding;
pub use image::run_pig_image;
pub use logits::run_dump_logits;
pub use selftest::run_self_test;
pub use text::{run_inference, run_interactive, run_shared_inference, run_multimodal};
pub use tts::run_tts_cli;

use crate::format::ggufrs::{open_model_source, ComponentRole};
use crate::core::tensor::{GGMLType, TensorSource};
use crate::ops;
use std::path::Path;

pub struct LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub wq: Box<dyn ops::kernel::Kernel + 'a>,
    pub wk: Box<dyn ops::kernel::Kernel + 'a>,
    pub wv: Box<dyn ops::kernel::Kernel + 'a>,
    pub wo: Box<dyn ops::kernel::Kernel + 'a>,
    pub w_gate: Box<dyn ops::kernel::Kernel + 'a>,
    pub w_up: Box<dyn ops::kernel::Kernel + 'a>,
    pub w_down: Box<dyn ops::kernel::Kernel + 'a>,
}

pub fn get_f32_tensor<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    expected_len: usize,
) -> Vec<f32> {
    let info = source
        .tensor_info(name)
        .unwrap_or_else(|| panic!("tensor {name} not found"));
    let bytes = source
        .tensor_slice(name)
        .unwrap_or_else(|| panic!("slice {name} not found"));
    let mut output = vec![0.0; expected_len];
    if info.ggml_type == GGMLType::F32 {
        for (value, chunk) in output.iter_mut().zip(bytes.chunks_exact(4)) {
            *value = f32::from_le_bytes(chunk.try_into().unwrap());
        }
    }
    output
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

macro_rules! slice_from_mut {
    ($ptr:expr, $len:expr) => {
        unsafe { std::slice::from_raw_parts_mut($ptr, $len) }
    };
}

macro_rules! slice_from_ref {
    ($ptr:expr, $len:expr) => {
        unsafe { std::slice::from_raw_parts($ptr, $len) }
    };
}

macro_rules! raw_parts {
    ($ptr:expr, $len:expr) => {
        unsafe { std::slice::from_raw_parts($ptr, $len) }
    };
}

pub(crate) use slice_from_mut;
pub(crate) use slice_from_ref;
pub(crate) use raw_parts;
