//! Gemma 4 transformer decoder trunk.

pub mod config;
pub mod forward;
pub mod scratch;
pub mod session;
pub mod weights;

pub use config::Gemma4Config;
pub use forward::Gemma4InputRow;
pub use session::Gemma4Session;
pub use weights::Gemma4Model;

#[cfg(test)]
use config::{
    BASE_FFN_LAYERS, EMBED, FULL_HEAD_DIM, HEADS, LAYERS, MAX_FFN, PER_LAYER, PER_LAYER_ALL,
    SWA_HEAD_DIM, VOCAB,
};
#[cfg(test)]
use forward::{assemble_input_rows, attend, ggml_geglu_fp16_inplace, matmul, softcap};
#[cfg(test)]
use session::{require_f32_kv, KvLayer};
#[cfg(test)]
use weights::{head_dim, kv_source_layer, load_weight, Gemma4Layer};
#[cfg(test)]
mod tests;
