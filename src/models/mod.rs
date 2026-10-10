pub mod audio8;
pub mod bert_family;
pub mod bitnet;
pub mod breeze;
/// Renamed to [`crate::prompt::legacy`].
///
/// Kept as a deprecated re-export so moving the module was not a breaking
/// change for anyone importing `rust_model_inference::models::chat_template`.
#[deprecated(note = "use crate::prompt::legacy instead")]
pub mod chat_template {
    pub use crate::prompt::legacy::*;
}

pub mod clm;
pub mod diffusion;
pub mod dots;
pub mod edge0;
pub mod falcon_h1;
pub mod funasr;
pub mod gemma2;
pub mod gemma3;
pub mod gemma4;
pub mod gemma_embedding;
pub mod gliner;
pub mod gliner_boundary;
pub mod gliner_ettin;
pub mod hybrid;
pub mod laya;
pub mod lfm2;
pub mod lfm25;
pub mod lfm2moe;
pub mod llama;
pub mod nemotron_h;
pub mod phi3;
pub mod qwen3;
pub mod qwen35;
pub mod qwen_drive;
pub mod spark;
pub mod vibevoice_asr;
pub mod xing4_0;
pub mod yue2;
pub mod zdt;
