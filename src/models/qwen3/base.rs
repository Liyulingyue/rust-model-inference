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



pub fn text_encode(
    model: &Qwen3Model,
    token_ids: &[u32],
    positions: &[[usize; 4]],
) -> Result<Vec<f32>, String> {
    validate_token_ids(token_ids, model.config.vocab)?;
    let n_tokens = token_ids.len();
    if positions.len() != n_tokens {
        return Err(format!(
            "positions length {} != token_ids length {}",
            positions.len(),
            n_tokens
        ));
    }
    if n_tokens == 0 {
        return Ok(Vec::new());
    }

    let cfg = &model.config;
    let n_embd_q = checked_product("query width", cfg.n_head, cfg.n_embd_head_k)?;
    let n_embd_k = checked_product("key width", cfg.n_head_kv, cfg.n_embd_head_k)?;
    let n_embd_v = checked_product("value width", cfg.n_head_kv, cfg.n_embd_head_v)?;
    let n_attn = checked_product("attn width", cfg.n_head, cfg.n_embd_head_v)?;
    let group_size = cfg.n_head / cfg.n_head_kv;
    let kq_scale = 1.0 / (cfg.n_embd_head_k as f32).sqrt();

    let embeddings = model.embed_tokens(token_ids)?;
    let mut hidden = embeddings;

    for layer_idx in 0..cfg.n_layer {
        let layer = &model.layers[layer_idx];

        let mut normed = vec![0.0; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            rms_norm(
                &hidden[off..off + cfg.n_embd],
                &layer.attn_norm,
                &mut normed[off..off + cfg.n_embd],
                cfg.eps,
            );
        }

        let mut q_all = vec![0.0; n_tokens * n_embd_q];
        let mut k_all = vec![0.0; n_tokens * n_embd_k];
        let mut v_all = vec![0.0; n_tokens * n_embd_v];
        for tok in 0..n_tokens {
            let norm_row = &normed[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd];
            let q_off = tok * n_embd_q;
            let k_off = tok * n_embd_k;
            let v_off = tok * n_embd_v;

            let blocks = (cfg.n_embd + 31) / 32;
            let mut q8_buf = vec![0u8; cfg.n_embd];
            let mut scale_buf = vec![0.0f32; blocks];
            quantize_q8_0_into(norm_row, cfg.n_embd, &mut q8_buf, &mut scale_buf);

            layer.wq.kernel.forward_prepared(
                norm_row,
                &q8_buf,
                &scale_buf,
                None,
                &mut q_all[q_off..q_off + n_embd_q],
                cfg.n_embd,
                n_embd_q,
                0,
                1,
            );
            layer.wk.kernel.forward_prepared(
                norm_row,
                &q8_buf,
                &scale_buf,
                None,
                &mut k_all[k_off..k_off + n_embd_k],
                cfg.n_embd,
                n_embd_k,
                0,
                1,
            );
            layer.wv.kernel.forward_prepared(
                norm_row,
                &q8_buf,
                &scale_buf,
                None,
                &mut v_all[v_off..v_off + n_embd_v],
                cfg.n_embd,
                n_embd_v,
                0,
                1,
            );
        }

        if let (Some(q_norm), Some(k_norm)) = (layer.q_norm.as_deref(), layer.k_norm.as_deref()) {
            for head in 0..cfg.n_head {
                let off = head * cfg.n_embd_head_k;
                rms_norm_inplace(&mut q_all[off..off + cfg.n_embd_head_k], q_norm, cfg.eps);
            }
            for head in 0..cfg.n_head_kv {
                let off = head * cfg.n_embd_head_k;
                rms_norm_inplace(&mut k_all[off..off + cfg.n_embd_head_k], k_norm, cfg.eps);
            }
        }

        for tok in 0..n_tokens {
            let pos = positions[tok];
            for head in 0..cfg.n_head {
                let off = head * cfg.n_embd_head_k;
                let q_slice = &mut q_all[off..off + cfg.n_embd_head_k];
                match cfg.rope {
                    Qwen3Rope::Neox => {
                        rope_neox(q_slice, pos[0], cfg.n_embd_head_k, cfg.freq_base);
                    }
                    Qwen3Rope::Interleaved { sections, n_dims } => {
                        rope_mrope_interleaved(
                            q_slice,
                            pos,
                            sections,
                            cfg.n_embd_head_k,
                            cfg.freq_base,
                            n_dims,
                        );
                    }
                }
            }
            for head in 0..cfg.n_head_kv {
                let off = head * cfg.n_embd_head_k;
                let k_slice = &mut k_all[off..off + cfg.n_embd_head_k];
                match cfg.rope {
                    Qwen3Rope::Neox => {
                        rope_neox(k_slice, pos[0], cfg.n_embd_head_k, cfg.freq_base);
                    }
                    Qwen3Rope::Interleaved { sections, n_dims } => {
                        rope_mrope_interleaved(
                            k_slice,
                            pos,
                            sections,
                            cfg.n_embd_head_k,
                            cfg.freq_base,
                            n_dims,
                        );
                    }
                }
            }
        }

        let mut attn_out = vec![0.0; n_tokens * n_attn];
        for head in 0..cfg.n_head {
            let kv_head = head / group_size;
            let q_off = head * cfg.n_embd_head_k;
            let k_off = kv_head * cfg.n_embd_head_k;
            let v_off = kv_head * cfg.n_embd_head_v;
            let attn_off = head * cfg.n_embd_head_v;

            for i in 0..n_tokens {
                let mut max_val = f32::NEG_INFINITY;
                let mut scores = vec![0.0; n_tokens];
                for j in 0..=i {
                    let q_row = &q_all[i * n_embd_q + q_off..i * n_embd_q + q_off + cfg.n_embd_head_k];
                    let k_row = &k_all[j * n_embd_k + k_off..j * n_embd_k + k_off + cfg.n_embd_head_k];
                    scores[j] = dot_f32(q_row, k_row, cfg.n_embd_head_k) * kq_scale;
                    if scores[j] > max_val {
                        max_val = scores[j];
                    }
                }
                let mut exp_sum = 0.0f32;
                for j in 0..=i {
                    scores[j] = (scores[j] - max_val).exp();
                    exp_sum += scores[j];
                }
                for j in 0..=i {
                    scores[j] /= exp_sum;
                }
                for dim in 0..cfg.n_embd_head_v {
                    let mut sum = 0.0f32;
                    for j in 0..=i {
                        let v_row = &v_all[j * n_embd_v + v_off..j * n_embd_v + v_off + cfg.n_embd_head_v];
                        sum += scores[j] * v_row[dim];
                    }
                    attn_out[i * n_attn + attn_off + dim] = sum;
                }
            }
        }

        let mut attn_proj_out = vec![0.0; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let attn_row = &attn_out[tok * n_attn..tok * n_attn + n_attn];
            let blocks = (n_attn + 31) / 32;
            let mut q8_buf = vec![0u8; n_attn];
            let mut scale_buf = vec![0.0f32; blocks];
            quantize_q8_0_into(attn_row, n_attn, &mut q8_buf, &mut scale_buf);
            layer.wo.kernel.forward_prepared(
                attn_row,
                &q8_buf,
                &scale_buf,
                None,
                &mut attn_proj_out[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd],
                n_attn,
                cfg.n_embd,
                0,
                1,
            );
        }

        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            for j in 0..cfg.n_embd {
                hidden[off + j] += attn_proj_out[off + j];
            }
        }

        let mut ffn_normed = vec![0.0; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            rms_norm(
                &hidden[off..off + cfg.n_embd],
                &layer.ffn_norm,
                &mut ffn_normed[off..off + cfg.n_embd],
                cfg.eps,
            );
        }

        let mut gate_buf = vec![0.0; n_tokens * cfg.n_ff];
        let mut up_buf = vec![0.0; n_tokens * cfg.n_ff];
        for tok in 0..n_tokens {
            let ffn_row = &ffn_normed[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd];
            let blocks = (cfg.n_embd + 31) / 32;
            let mut q8_buf = vec![0u8; cfg.n_embd];
            let mut scale_buf = vec![0.0f32; blocks];
            quantize_q8_0_into(ffn_row, cfg.n_embd, &mut q8_buf, &mut scale_buf);
            layer.w_gate.kernel.forward_prepared(
                ffn_row,
                &q8_buf,
                &scale_buf,
                None,
                &mut gate_buf[tok * cfg.n_ff..tok * cfg.n_ff + cfg.n_ff],
                cfg.n_embd,
                cfg.n_ff,
                0,
                1,
            );
            layer.w_up.kernel.forward_prepared(
                ffn_row,
                &q8_buf,
                &scale_buf,
                None,
                &mut up_buf[tok * cfg.n_ff..tok * cfg.n_ff + cfg.n_ff],
                cfg.n_embd,
                cfg.n_ff,
                0,
                1,
            );
        }

        for tok in 0..n_tokens {
            let off = tok * cfg.n_ff;
            for i in 0..cfg.n_ff {
                gate_buf[off + i] = silu(gate_buf[off + i]) * up_buf[off + i];
            }
        }

        let mut down_buf = vec![0.0; n_tokens * cfg.n_embd];
        for tok in 0..n_tokens {
            let blocks = (cfg.n_ff + 31) / 32;
            let mut q8_buf = vec![0u8; cfg.n_ff];
            let mut scale_buf = vec![0.0f32; blocks];
            quantize_q8_0_into(
                &gate_buf[tok * cfg.n_ff..tok * cfg.n_ff + cfg.n_ff],
                cfg.n_ff,
                &mut q8_buf,
                &mut scale_buf,
            );
            layer.w_down.kernel.forward_prepared(
                &gate_buf[tok * cfg.n_ff..tok * cfg.n_ff + cfg.n_ff],
                &q8_buf,
                &scale_buf,
                None,
                &mut down_buf[tok * cfg.n_embd..tok * cfg.n_embd + cfg.n_embd],
                cfg.n_ff,
                cfg.n_embd,
                0,
                1,
            );
        }

        for tok in 0..n_tokens {
            let off = tok * cfg.n_embd;
            for i in 0..cfg.n_embd {
                hidden[off + i] += down_buf[off + i];
            }
        }
    }

    let mut output = vec![0.0; n_tokens * cfg.n_embd];
    for tok in 0..n_tokens {
        let off = tok * cfg.n_embd;
        rms_norm(
            &hidden[off..off + cfg.n_embd],
            &model.output_norm,
            &mut output[off..off + cfg.n_embd],
            cfg.eps,
        );
    }

    Ok(output)
}

pub fn run_shared_inference(
    source: Arc<dyn TensorSource>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    n_threads_arg: usize,
    thinking: bool,
) -> Result<(), String> {
    let started = Instant::now();
    let tokenizer = Arc::new(
        BPETokenizer::from_gguf_metadata(|key| source.metadata(key).cloned())
            .map_err(|error| format!("Failed to initialize tokenizer: {error}"))?,
    );
    let available_threads = std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(4);
    let pool = Arc::new(ComputePool::new(resolve_thread_count(
        n_threads_arg,
        available_threads,
    )));
    let model = Qwen3Model::from_source(source, Arc::clone(&tokenizer), pool)?;
    let input_tokens = build_qwen_chat_prompt(
        &tokenizer,
        &[QwenMessage {
            role: "user",
            content: prompt,
        }],
        thinking,
    )?;
    let positions = qwen_text_positions(input_tokens.len());
    println!(
        "Model: {} | n_embd={} n_layer={} n_head={} n_head_kv={} n_ff={} | loaded in {}ms",
        model.config().architecture,
        model.config().n_embd,
        model.config().n_layer,
        model.config().n_head,
        model.config().n_head_kv,
        model.config().n_ff,
        started.elapsed().as_millis(),
    );
    eprintln!("compute pool: {} threads", model.pool().n_threads());
    println!("Prompt: {} ({} tokens)", prompt, input_tokens.len());
    print!("Output: ");
    io::stdout().flush().map_err(|error| error.to_string())?;
    let inference_started = Instant::now();
    let generation = model.generate(
        Qwen3Input {
            token_ids: &input_tokens,
            positions: &positions,
            embeddings: None,
        },
        Qwen3GenerateOptions {
            max_new_tokens: max_tokens,
            temperature,
        },
    )?;
    print!("{}", generation.text);
    io::stdout().flush().map_err(|error| error.to_string())?;
    let elapsed_ms = inference_started.elapsed().as_millis();
    let tokens_per_second = if elapsed_ms > 0 {
        generation.token_ids.len() as f64 / elapsed_ms as f64 * 1000.0
    } else {
        0.0
    };
    println!();
    println!(
        "[end-to-end: {} output tokens in {}ms | {:.1} tok/s]",
        generation.token_ids.len(),
        elapsed_ms,
        tokens_per_second,
    );
    Ok(())
}

