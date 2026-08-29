//! # Qwen3Model: Qwen3 文本模型（VL/ASR/TTS 路径专用）
//!
//! **注意**：此模块**不是**普通 CLI Qwen3 文本推理的执行路径。
//!
//! ## 代码路径说明
//!
//! Qwen3 推理存在**两套**代码路径：
//!
//! ### 路径 1：普通 CLI 文本推理（`app/text.rs`）
//! ```text
//! main.rs -> app::run_inference() -> app/text.rs
//! ```
//! - **推荐用于纯文本生成任务**
//!
//! ### 路径 2：Qwen3Model 路径（`models/qwen3.rs`）
//! ```text
//! main.rs -> app::run_qwen3vl / run_asr / run_tts
//!   -> Qwen3Model::from_source()
//!   -> Qwen3Model::text_encode()
//! ```
//! - 支持多量化格式（Q4_K、Q6_K、Q8_0 等）
//! - 支持 RoPE：Neox 和 Interleaved (mrope)
//! - 用于：Qwen3-VL 解码器、Qwen3-ASR、Qwen3-TTS、Z-Image 文本编码器

use crate::app::cli::resolve_thread_count;
use crate::core::loader::{
    check_qwen3_allowed_dimensions, model_config_from_source, qwen3_arch_knobs,
};
use crate::core::tensor::{
    GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource,
};
use crate::ops::kernel::{Kernel, QuantizedTensor, Weight};
use crate::ops::*;
#[cfg(feature = "parity-trace")]
use crate::parity_trace;
use crate::prompt::{build_qwen_chat_prompt, QwenMessage};
use crate::core::scratchpad::{ExecutionScratchpad, KvArch, KvCache, KvCacheF16, KvFormat, KvLifecycle, KvState};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
pub use crate::models::qwen3::static_weight;
use crate::models::qwen3::{load_layers_static, Qwen3LayerWeights};
use crate::models::qwen3::util::{
    check_allocation, checked_decoder_steps, checked_generated_position, checked_product,
    checked_session_capacity, greedy_token, load_f32_tensor, optional_usize,
    sample_token, usize_to_u64, validate_generation, validate_input_shapes, validate_token_ids,
};
use crate::models::qwen3::qwen_text_positions;
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3Rope {
    Neox,
    Interleaved { sections: [i32; 4], n_dims: usize },
}

pub struct Qwen3Config {
    pub architecture: String,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_ff: usize,
    pub vocab: usize,
    pub n_ctx: usize,
    pub eps: f32,
    pub freq_base: f32,
    pub has_qk_norm: bool,
    pub rope: Qwen3Rope,
}

impl Qwen3Config {
    /// Build a `Qwen3Config` from a GGUF tensor source.
    ///
    /// Phase 4c: architecture dispatch (which architectures are accepted, which
    /// rope flavor they use, which dimensional configurations are allowed)
    /// lives in [`crate::core::loader::qwen3_arch_knobs`]. This function now
    /// only composes the per-arch knobs with the dimensional configuration
    /// produced by [`model_config_from_source`].
    pub(crate) fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let config = model_config_from_source(source)?;
        let knobs = qwen3_arch_knobs(source)?;

        // Per-arch head dimensions (Qwen3 / Qwen3-VL expose these explicitly;
        // other architectures fall back to n_embd / n_head).
        let n_embd_head_k = optional_usize(source, &format!("{}.attention.key_length", knobs.arch))?
            .unwrap_or(config.n_embd_head);
        let n_embd_head_v =
            optional_usize(source, &format!("{}.attention.value_length", knobs.arch))?
                .unwrap_or(config.n_embd_head);
        if n_embd_head_k == 0 || n_embd_head_v == 0 {
            return Err(format!(
                "Invalid {} attention head lengths: key={n_embd_head_k}, value={n_embd_head_v}",
                knobs.arch
            ));
        }

        // Optional dimensional whitelist (e.g. Qwen3-VL).
        if let Some(allowed) = knobs.allowed_dimensions {
            check_qwen3_allowed_dimensions(allowed, &config, n_embd_head_k, n_embd_head_v)?;
        }

        let rope = match knobs.rope_sections {
            Some(sections) => Qwen3Rope::Interleaved {
                sections,
                n_dims: n_embd_head_k,
            },
            None => Qwen3Rope::Neox,
        };

        Ok(Self {
            architecture: knobs.arch,
            n_embd: config.n_embd,
            n_layer: config.n_layer,
            n_head: config.n_head,
            n_head_kv: config.n_head_kv,
            n_embd_head_k,
            n_embd_head_v,
            n_ff: config.n_ff,
            vocab: config.vocab_size,
            n_ctx: config.n_ctx,
            eps: config.norm_eps,
            freq_base: config.rope_freq_base,
            has_qk_norm: knobs.has_qk_norm,
            rope,
        })
    }
}

pub struct Qwen3Input<'a> {
    pub token_ids: &'a [u32],
    pub positions: &'a [[usize; 4]],
    pub embeddings: Option<&'a [f32]>,
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen3GenerateOptions {
    pub max_new_tokens: usize,
    pub temperature: f32,
}

pub struct Qwen3Generation {
    pub text: String,
    pub rendered_tokens: Vec<String>,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
}

pub struct Qwen3Model {
    pub(crate) source: Arc<dyn TensorSource>,
    pub(crate) tokenizer: Arc<BPETokenizer>,
    pub(crate) pool: Arc<ComputePool>,
    pub(crate) config: Qwen3Config,
    pub(crate) layers: Vec<Qwen3LayerWeights<'static>>,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) token_embedding: Weight<'static>,
    pub(crate) output: Weight<'static>,
}

pub struct Qwen3Session<'model> {
    pub(crate) model: &'model Qwen3Model,
    pub(crate) kv_state: KvState,
    pub(crate) scratch: ExecutionScratchpad,
    pub(crate) capacity: usize,
}

/// 携带格式信息的 KV cache 指针。
///
/// 让 attention 循环可以根据格式选择对应的写入/读取路径。
/// - `F16`: 高性能路径，使用 F16 特定的 dot/f32_to_f16 优化（生产路径）
/// - `F32`: 通用路径，直接读写 f32（用于调试/精确推理）
#[derive(Clone, Copy)]
pub(crate) enum KvPtrs {
    F16 { k: *mut u16, v: *mut u16 },
    F32 { k: *mut f32, v: *mut f32 },
}



