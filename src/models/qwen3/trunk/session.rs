//! `Qwen3Session` — the decode-step driver.
//!
//! A session owns the scratchpad (buffer pool + score buffer), the KV
//! cache (F16 or F32, ephemeral / timed / persistent), and the prompt /
//! generation position cursor. Every call to `generate_with_asr_trace`
//! runs the prompt through the decoder once, then samples one token at
//! a time until `max_new_tokens` is reached.
//!
//! The 877-line `impl` block was lifted out of `base.rs` verbatim during
//! the architectural split; behaviour is unchanged.

use super::config::Qwen3Rope;
use super::forward::{forward_moe_token, Qwen3GenerateOptions, Qwen3Generation, Qwen3Input};
use super::util::{
    check_allocation, checked_decoder_steps, checked_generated_position, checked_product,
    checked_session_capacity, sample_token, validate_generation, validate_input_shapes,
};
use super::weights::Qwen3Model;
use crate::core::scratchpad::{
    ExecutionScratchpad, KvArch, KvCache, KvCacheF16, KvFormat, KvLifecycle, KvState,
};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::Kernel;
use crate::ops::*;
#[cfg(feature = "parity-trace")]
use crate::parity_trace;
#[cfg(feature = "vulkan")]
use crate::vulkan::qwen3::{commit_shadow_kv, Qwen3VulkanSession};
use std::io::{self, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

// =============================================================================
// Qwen3Session struct (per-request decode state)
// =============================================================================

pub struct Qwen3Session<'model> {
    pub(crate) model: &'model Qwen3Model,
    pub(crate) kv_state: KvState,
    pub(crate) scratch: ExecutionScratchpad,
    pub(crate) capacity: usize,
    #[cfg(feature = "vulkan")]
    pub(crate) gpu: Option<Qwen3VulkanSession>,
    #[cfg(feature = "vulkan")]
    pub(crate) full_model_gpu_failed: bool,
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

fn add_deepstack_embedding(
    hidden: &mut [f32],
    deepstack: &[f32],
    layer: usize,
    token: usize,
    token_count: usize,
    width: usize,
) {
    let start = (layer * token_count + token) * width;
    for (value, addition) in hidden.iter_mut().zip(&deepstack[start..start + width]) {
        *value += *addition;
    }
}

impl<'model> Qwen3Session<'model> {
    /// 创建新的会话（默认 F16 KV cache, Ephemeral 生命周期）。
    pub fn new(model: &'model Qwen3Model, capacity: usize) -> Result<Self, String> {
        Self::new_with_kv_state(model, capacity, KvFormat::F16, KvLifecycle::Ephemeral)
    }

    /// 使用指定的 KV 格式和生命周期创建会话（推荐入口）。
    ///
    /// 这是统一 KV 缓存设计的标准入口，支持：
    /// - `KvFormat::F16/F32`：选择 KV 精度
    /// - `KvLifecycle::Ephemeral/Timed/Persistent`：选择生命周期策略
    ///
    /// 当前阶段 `base` 主要需要 F16（与原代码一致）。
    /// 未来可以无缝扩展到 F32 或其他格式。
    pub fn new_with_kv_state(
        model: &'model Qwen3Model,
        capacity: usize,
        kv_format: KvFormat,
        lifecycle: KvLifecycle,
    ) -> Result<Self, String> {
        if capacity == 0 || capacity > model.config.n_ctx {
            return Err(format!(
                "Session capacity {capacity} must be within 1..={}",
                model.config.n_ctx
            ));
        }
        let config = &model.config;
        let n_embd_q = checked_product("query width", config.n_head, config.n_embd_head_k)?;
        let n_embd_k = checked_product("key width", config.n_head_kv, config.n_embd_head_k)?;
        let n_embd_v = checked_product("value width", config.n_head_kv, config.n_embd_head_v)?;
        let n_attn = checked_product(
            "attention output width",
            config.n_head,
            config.n_embd_head_v,
        )?;
        let kv_stride = n_embd_k.max(n_embd_v);
        let kv_size = checked_product(
            "KV cache values",
            checked_product("KV cache rows", config.n_layer, capacity)?,
            kv_stride,
        )?;
        let kv_bytes = match kv_format {
            KvFormat::F16 => check_allocation("KV cache", kv_size, std::mem::size_of::<u16>())?,
            KvFormat::F32 => check_allocation("KV cache", kv_size, std::mem::size_of::<f32>())?,
        };

        let max_n_in = n_embd_q.max(n_attn).max(config.n_ff);
        let score_stride = capacity
            .checked_add(255)
            .map(|value| value / 256 * 256)
            .ok_or_else(|| "Attention score stride overflow".to_string())?;
        let score_values =
            checked_product("attention scores", model.pool.n_threads(), score_stride)?;
        for (name, len, bytes) in [
            ("hidden state", config.n_embd, std::mem::size_of::<f32>()),
            (
                "normalized state",
                config.n_embd,
                std::mem::size_of::<f32>(),
            ),
            ("queries", n_embd_q, std::mem::size_of::<f32>()),
            ("keys", kv_stride, std::mem::size_of::<f32>()),
            ("values", kv_stride, std::mem::size_of::<f32>()),
            ("attention output", n_attn, std::mem::size_of::<f32>()),
            (
                "attention projection",
                config.n_embd,
                std::mem::size_of::<f32>(),
            ),
            ("down projection", config.n_embd, std::mem::size_of::<f32>()),
            ("gate projection", config.n_ff, std::mem::size_of::<f32>()),
            ("up projection", config.n_ff, std::mem::size_of::<f32>()),
            ("logits", config.vocab, std::mem::size_of::<f32>()),
            ("quantized activations", max_n_in, std::mem::size_of::<u8>()),
            (
                "quantization scales",
                max_n_in / 32,
                std::mem::size_of::<f32>(),
            ),
            ("attention scores", score_values, std::mem::size_of::<f32>()),
        ] {
            check_allocation(name, len, bytes)?;
        }

        // 构造标准化的 KvState
        let arch = Arc::new(KvArch::new(
            config.n_layer,
            config.n_head_kv,
            config.n_embd_head_k,
            config.n_embd_head_v,
            model.config.n_ctx,
        ));
        let mut kv_state = KvState::new(arch, kv_format, capacity).with_lifecycle(lifecycle);
        // KvState 自带的 KvCache 已分配好容量，但我们需要重新分配为正确的 stride
        // （KvState::new 用的是 max(k,v) 而非实际的 kv_stride）
        match (kv_format, &mut kv_state.cache) {
            (KvFormat::F16, KvCache::F16(c)) => {
                c.k = vec![0u16; kv_size];
                c.v = vec![0u16; kv_size];
            }
            (KvFormat::F32, KvCache::F32(c)) => {
                c.k = vec![0f32; kv_size];
                c.v = vec![0f32; kv_size];
            }
            _ => unreachable!("format and cache variant mismatch"),
        }
        let _ = kv_bytes; // 当前未直接使用，保留用于将来分配校验

        #[cfg(feature = "vulkan")]
        let (gpu, full_model_gpu_failed) = match crate::ops::get_vulkan_context() {
            Some(context) => match Qwen3VulkanSession::try_new(model, capacity, context) {
                Ok(gpu) => (gpu, false),
                Err(error) => {
                    eprintln!(
                        "[GPU] Qwen3 Vulkan session unavailable: {error}. Falling back to CPU."
                    );
                    (None, true)
                }
            },
            None => (None, false),
        };

        Ok(Self {
            model,
            kv_state,
            scratch: ExecutionScratchpad {
                x: vec![0.0; config.n_embd],
                normed: vec![0.0; config.n_embd],
                q: vec![0.0; n_embd_q],
                k_new: vec![0.0; kv_stride],
                v_new: vec![0.0; kv_stride],
                attn_out: vec![0.0; n_attn],
                attn_proj: vec![0.0; config.n_embd],
                down_buf: vec![0.0; config.n_embd],
                gate_buf: vec![0.0; config.n_ff],
                up_buf: vec![0.0; config.n_ff],
                logits: vec![0.0; config.vocab],
                q8_buf: vec![0; max_n_in],
                scale_buf: vec![0.0; max_n_in / 32],
                // Q8_K pre-quantization buffer for K-quant kernels (Q4_K / Q6_K).
                // See TODO in docs/TODO.md: "Q8_0 与 Q8_K 量化路径按需量化".
                // - `q8_buf` + `scale_buf` (Q8_0): consumed by the default kernel
                //   path (`forward_prequantized`, see ops/kernel/mod.rs:70) and by
                //   kernels with Q8_0 / f16 weights.
                // - `q8k_buf` (Q8_K): consumed by K-quant kernels
                //   (q4_k.rs:90, q6_k.rs:73) via `forward_prepared(.., Some(q8_k), ..)`;
                //   those overrides name the Q8_0 args `_input_q8` / `_input_scales`
                //   and discard them.
                // A single layer's kernel uses exactly one of the two paths, but the
                // model as a whole can be heterogeneous (some layers Q8_0, others
                // Q4_K/Q6_K) so both buffers must stay allocated. The waste today is
                // that we re-quantize the same input into BOTH formats on every
                // forward even when a given layer only needs one — fix is to dispatch
                // per-layer on weight format and skip the unused quantize call.
                q8k_buf: vec![
                    crate::ops::quant::BlockQ8K {
                        d: 0.0,
                        qs: [0; 256],
                        bsums: [0; 16],
                    };
                    max_n_in / 256
                ],
                score_stride,
                scores: vec![0.0; score_values],
            },
            capacity,
            #[cfg(feature = "vulkan")]
            gpu,
            #[cfg(feature = "vulkan")]
            full_model_gpu_failed,
        })
    }

    /// 获取对底层 KV state 的引用（只读），供外部监控或跨 session 共享。
    pub fn kv_state(&self) -> &KvState {
        &self.kv_state
    }

    pub fn last_logits(&self) -> &[f32] {
        &self.scratch.logits
    }

    /// 重置 KV 缓存（用于多轮对话中的上下文清理）。
    pub fn reset_kv(&mut self) {
        self.kv_state.reset();
        #[cfg(feature = "vulkan")]
        if let Some(gpu) = &mut self.gpu {
            gpu.reset();
        }
    }

    pub fn generate(
        &mut self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
    ) -> Result<Qwen3Generation, String> {
        self.generate_with_asr_trace(input, options, false)
    }

    /// Streaming 生成版本：每个 token 渲染后立即回调 `on_token`。
    ///
    /// 这是 CLI 实时输出、多轮对话"打字机"效果的关键接口。
    /// 与 `generate()` 的区别仅在 streaming callback；最终 `Qwen3Generation`
    /// 内容一致。
    ///
    /// # Example
    ///
    /// ```ignore
    /// session.generate_streaming(input, options, |text| print!("{text}"))?;
    /// ```
    pub fn generate_streaming(
        &mut self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
        mut on_token: impl FnMut(&str),
    ) -> Result<Qwen3Generation, String> {
        validate_generation(self.model, &input, &options)?;
        let required = checked_session_capacity(
            input.token_ids.len(),
            options.max_new_tokens,
            self.model.config.n_ctx,
        )?;
        if required > self.capacity {
            return Err(format!(
                "Generation requires capacity {required}; session has {}",
                self.capacity
            ));
        }
        // 包装 callback 为 dyn FnMut（一次性，不需要 Box<dyn FnMut>）
        let mut cb = |text: &str| on_token(text);
        self.generate_inner(input, options, false, Some(&mut cb))
    }

    pub(crate) fn generate_with_asr_trace(
        &mut self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
        asr_trace: bool,
    ) -> Result<Qwen3Generation, String> {
        validate_generation(self.model, &input, &options)?;
        let required = checked_session_capacity(
            input.token_ids.len(),
            options.max_new_tokens,
            self.model.config.n_ctx,
        )?;
        if required > self.capacity {
            return Err(format!(
                "Generation requires capacity {required}; session has {}",
                self.capacity
            ));
        }
        self.generate_inner(input, options, asr_trace, None)
    }

    fn generate_inner(
        &mut self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
        asr_trace: bool,
        mut on_token: Option<&mut dyn FnMut(&str)>,
    ) -> Result<Qwen3Generation, String> {
        let _source = &self.model.source;
        let model = self.model;
        let config = &model.config;
        let capacity = self.capacity;
        let n_prompt = input.token_ids.len();
        let n_embd_q = checked_product("query width", config.n_head, config.n_embd_head_k)?;
        let n_embd_k = checked_product("key width", config.n_head_kv, config.n_embd_head_k)?;
        let n_embd_v = checked_product("value width", config.n_head_kv, config.n_embd_head_v)?;
        let n_attn = checked_product(
            "attention output width",
            config.n_head,
            config.n_embd_head_v,
        )?;
        let kv_stride = n_embd_k.max(n_embd_v);
        let kv_cache_size = checked_product(
            "KV cache values",
            checked_product("KV cache rows", config.n_layer, capacity)?,
            kv_stride,
        )?;
        let max_n_in = n_embd_q.max(n_attn).max(config.n_ff);
        let group_size = config.n_head / config.n_head_kv;
        let kq_scale = 1.0 / (config.n_embd_head_k as f32).sqrt();
        // 统一 KV cache 指针（同时支持 F16/F32），由 `KvPtrs` 携带格式信息
        // 让 attention 循环可以根据格式选择对应的写入/读取路径。
        let kv_ptrs = match &mut self.kv_state.cache {
            KvCache::F16(cache) => KvPtrs::F16 {
                k: cache.k.as_mut_ptr(),
                v: cache.v.as_mut_ptr(),
            },
            KvCache::F32(cache) => KvPtrs::F32 {
                k: cache.k.as_mut_ptr(),
                v: cache.v.as_mut_ptr(),
            },
        };
        self.kv_state.update_access();

        #[cfg(feature = "parity-trace")]
        {
            if asr_trace {
                parity_trace::report(parity_trace::token_ids("asr.prompt_ids", input.token_ids));
                let position_values =
                    checked_product("ASR position values", input.positions.len(), 4)?;
                let mut positions = Vec::new();
                positions
                    .try_reserve_exact(position_values)
                    .map_err(|error| format!("Failed to allocate ASR positions: {error}"))?;
                for position in input.positions {
                    positions.extend_from_slice(position);
                }
                parity_trace::report(parity_trace::usize_values(
                    "asr.positions",
                    &[input.positions.len(), 4],
                    &positions,
                ));
            } else {
                parity_trace::report(parity_trace::token_ids("prompt_ids", input.token_ids));
                let text_positions: Vec<usize> =
                    input.positions.iter().map(|value| value[0]).collect();
                parity_trace::report(parity_trace::usize_values(
                    "qwen3.positions",
                    &[text_positions.len()],
                    &text_positions,
                ));
            }
        }
        #[cfg(not(feature = "parity-trace"))]
        let _ = asr_trace;

        let mut generated_tokens = Vec::new();
        generated_tokens
            .try_reserve_exact(options.max_new_tokens)
            .map_err(|error| format!("Failed to allocate generated tokens: {error}"))?;
        let mut rendered_tokens = Vec::new();
        rendered_tokens
            .try_reserve_exact(options.max_new_tokens)
            .map_err(|error| format!("Failed to allocate rendered tokens: {error}"))?;
        let mut decoder = model.tokenizer.streaming_decoder(false);
        let mut prompt_duration = Duration::ZERO;
        let mut decode_duration = Duration::ZERO;

        let decoder_steps = checked_decoder_steps(n_prompt, options.max_new_tokens, config.n_ctx)?;
        for step in 0..decoder_steps {
            let eval_start = Instant::now();
            let position = if step < n_prompt {
                input.positions[step]
            } else {
                checked_generated_position(input.positions, step - n_prompt)?
            };
            if step < n_prompt {
                if let Some(embeddings) = input.embeddings {
                    let start = step * config.n_embd;
                    self.scratch
                        .x
                        .copy_from_slice(&embeddings[start..start + config.n_embd]);
                } else {
                    model
                        .token_embedding
                        .embedding_lookup(input.token_ids[step], &mut self.scratch.x);
                }
            } else {
                let token_id = *generated_tokens
                    .last()
                    .ok_or_else(|| "Missing generated token for decoder step".to_string())?;
                model
                    .token_embedding
                    .embedding_lookup(token_id, &mut self.scratch.x);
            }
            #[cfg(feature = "parity-trace")]
            parity_trace::report(parity_trace::checkpoint(
                "model.input_embed",
                None,
                &[1, config.n_embd],
                &self.scratch.x,
            ));

            #[cfg(feature = "vulkan")]
            let used_vulkan = {
                let mut disable_reason = None;
                let used = match self.gpu.as_mut() {
                    Some(gpu) => match gpu.forward_token(&self.scratch.x, position[0]) {
                        Ok(result) => {
                            if let Err(error) = commit_shadow_kv(
                                &mut self.kv_state,
                                step,
                                result.k_delta,
                                result.v_delta,
                            ) {
                                gpu.abort_token();
                                disable_reason = Some(error);
                                false
                            } else {
                                self.scratch.logits.copy_from_slice(result.logits);
                                gpu.commit_token();
                                true
                            }
                        }
                        Err(error) => {
                            disable_reason = Some(error.to_string());
                            false
                        }
                    },
                    None => false,
                };
                if let Some(reason) = disable_reason {
                    eprintln!(
                        "[GPU] Qwen3 Vulkan session disabled after error: {reason}. Falling back to CPU."
                    );
                    self.gpu = None;
                    self.full_model_gpu_failed = true;
                }
                used
            };
            #[cfg(not(feature = "vulkan"))]
            let used_vulkan = false;

            if !used_vulkan {
                #[cfg(feature = "vulkan")]
                let _gpu_matmul_scope = self
                    .full_model_gpu_failed
                    .then(ComputePool::disable_gpu_matmul_for_scope);
                for layer in 0..config.n_layer {
                    let weights = &model.layers[layer];
                    let x_ptr = self.scratch.x.as_mut_ptr();
                    let normed_ptr = self.scratch.normed.as_mut_ptr();
                    let q_ptr = self.scratch.q.as_mut_ptr();
                    let k_ptr = self.scratch.k_new.as_mut_ptr();
                    let v_ptr = self.scratch.v_new.as_mut_ptr();
                    let attn_out_ptr = self.scratch.attn_out.as_mut_ptr();
                    let attn_proj_ptr = self.scratch.attn_proj.as_mut_ptr();
                    let down_buf_ptr = self.scratch.down_buf.as_mut_ptr();
                    let gate_buf_ptr = self.scratch.gate_buf.as_mut_ptr();
                    let up_buf_ptr = self.scratch.up_buf.as_mut_ptr();
                    let q8_buf_ptr = self.scratch.q8_buf.as_mut_ptr();
                    let scale_buf_ptr = self.scratch.scale_buf.as_mut_ptr();

                    let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, config.n_embd) };
                    let normed =
                        unsafe { std::slice::from_raw_parts_mut(normed_ptr, config.n_embd) };
                    let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
                    let scale_buf =
                        unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };

                    rms_norm(x, &weights.attn_norm, normed, config.eps);
                    #[cfg(feature = "parity-trace")]
                    if layer == 0 {
                        parity_trace::report(parity_trace::checkpoint(
                            "attn_norm-0",
                            Some(0),
                            &[1, config.n_embd],
                            normed,
                        ));
                    }
                    quantize_q8_0_into(
                        normed,
                        config.n_embd,
                        &mut q8_buf[..config.n_embd],
                        &mut scale_buf[..config.n_embd / 32],
                    );
                    let q8 = q8_buf[..config.n_embd].as_ptr();
                    let scales = scale_buf[..config.n_embd / 32].as_ptr();
                    let pool = Arc::clone(&model.pool);
                    pool.compute(move |thread, threads| {
                        let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_embd) };
                        let scales =
                            unsafe { std::slice::from_raw_parts(scales, config.n_embd / 32) };
                        let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
                        let k = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_k) };
                        let v = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_v) };
                        weights.wq.kernel.forward_prepared(
                            normed,
                            q8,
                            scales,
                            None,
                            q,
                            config.n_embd,
                            n_embd_q,
                            thread,
                            threads,
                        );
                        weights.wk.kernel.forward_prepared(
                            normed,
                            q8,
                            scales,
                            None,
                            k,
                            config.n_embd,
                            n_embd_k,
                            thread,
                            threads,
                        );
                        weights.wv.kernel.forward_prepared(
                            normed,
                            q8,
                            scales,
                            None,
                            v,
                            config.n_embd,
                            n_embd_v,
                            thread,
                            threads,
                        );
                    });

                    {
                        let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
                        let k = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_k) };
                        let v = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_v) };
                        if let Some(bias) = weights.q_bias.as_deref() {
                            for (value, bias) in q.iter_mut().zip(bias) {
                                *value += *bias;
                            }
                        }
                        if let Some(bias) = weights.k_bias.as_deref() {
                            for (value, bias) in k.iter_mut().zip(bias) {
                                *value += *bias;
                            }
                        }
                        if let Some(bias) = weights.v_bias.as_deref() {
                            for (value, bias) in v.iter_mut().zip(bias) {
                                *value += *bias;
                            }
                        }
                        if let (Some(q_norm), Some(k_norm)) =
                            (weights.q_norm.as_deref(), weights.k_norm.as_deref())
                        {
                            for head in q.chunks_exact_mut(config.n_embd_head_k) {
                                rms_norm_inplace(head, q_norm, config.eps);
                            }
                            for head in k.chunks_exact_mut(config.n_embd_head_k) {
                                rms_norm_inplace(head, k_norm, config.eps);
                            }
                        }
                        #[cfg(feature = "parity-trace")]
                        if layer == 0 {
                            parity_trace::report(parity_trace::checkpoint(
                                "Qcur_normed-0",
                                Some(0),
                                &[config.n_head, config.n_embd_head_k],
                                q,
                            ));
                            parity_trace::report(parity_trace::checkpoint(
                                "Kcur_normed-0",
                                Some(0),
                                &[config.n_head_kv, config.n_embd_head_k],
                                k,
                            ));
                        }
                        for head in q.chunks_exact_mut(config.n_embd_head_k) {
                            match config.rope {
                                Qwen3Rope::Neox => rope_neox(
                                    head,
                                    position[0],
                                    config.n_embd_head_k,
                                    config.freq_base,
                                ),
                                Qwen3Rope::Interleaved { sections, n_dims } => {
                                    rope_mrope_interleaved(
                                        head,
                                        position,
                                        sections,
                                        config.n_embd_head_k,
                                        config.freq_base,
                                        n_dims,
                                    )
                                }
                            }
                        }
                        for head in k.chunks_exact_mut(config.n_embd_head_k) {
                            match config.rope {
                                Qwen3Rope::Neox => rope_neox(
                                    head,
                                    position[0],
                                    config.n_embd_head_k,
                                    config.freq_base,
                                ),
                                Qwen3Rope::Interleaved { sections, n_dims } => {
                                    rope_mrope_interleaved(
                                        head,
                                        position,
                                        sections,
                                        config.n_embd_head_k,
                                        config.freq_base,
                                        n_dims,
                                    )
                                }
                            }
                        }
                        #[cfg(feature = "parity-trace")]
                        if layer == 0 {
                            parity_trace::report(parity_trace::checkpoint(
                                "Qcur-0",
                                Some(0),
                                &[config.n_head, config.n_embd_head_k],
                                q,
                            ));
                            parity_trace::report(parity_trace::checkpoint(
                                "Kcur-0",
                                Some(0),
                                &[config.n_head_kv, config.n_embd_head_k],
                                k,
                            ));
                        }

                        let layer_base = layer * capacity * kv_stride;
                        // 根据格式选择写入路径
                        match kv_ptrs {
                            KvPtrs::F16 { k: k_ptr, v: v_ptr } => {
                                let k_cache =
                                    unsafe { std::slice::from_raw_parts_mut(k_ptr, kv_cache_size) };
                                let v_cache =
                                    unsafe { std::slice::from_raw_parts_mut(v_ptr, kv_cache_size) };
                                for head in 0..config.n_head_kv {
                                    let k_offset = head * config.n_embd_head_k;
                                    let v_offset = head * config.n_embd_head_v;
                                    let cache_row = layer_base + step * kv_stride;
                                    f32_slice_to_f16(
                                        &k[k_offset..k_offset + config.n_embd_head_k],
                                        &mut k_cache[cache_row + k_offset
                                            ..cache_row + k_offset + config.n_embd_head_k],
                                    );
                                    f32_slice_to_f16(
                                        &v[v_offset..v_offset + config.n_embd_head_v],
                                        &mut v_cache[cache_row + v_offset
                                            ..cache_row + v_offset + config.n_embd_head_v],
                                    );
                                }
                            }
                            KvPtrs::F32 { k: k_ptr, v: v_ptr } => {
                                let k_cache =
                                    unsafe { std::slice::from_raw_parts_mut(k_ptr, kv_cache_size) };
                                let v_cache =
                                    unsafe { std::slice::from_raw_parts_mut(v_ptr, kv_cache_size) };
                                for head in 0..config.n_head_kv {
                                    let k_offset = head * config.n_embd_head_k;
                                    let v_offset = head * config.n_embd_head_v;
                                    let cache_row = layer_base + step * kv_stride;
                                    k_cache[cache_row + k_offset
                                        ..cache_row + k_offset + config.n_embd_head_k]
                                        .copy_from_slice(
                                            &k[k_offset..k_offset + config.n_embd_head_k],
                                        );
                                    v_cache[cache_row + v_offset
                                        ..cache_row + v_offset + config.n_embd_head_v]
                                        .copy_from_slice(
                                            &v[v_offset..v_offset + config.n_embd_head_v],
                                        );
                                }
                            }
                        }
                    }

                    let pool = Arc::clone(&model.pool);
                    let scores_ptr = self.scratch.scores.as_mut_ptr();
                    let score_stride = self.scratch.score_stride;
                    // 把 kv_ptrs (轻量 enum, 含 *mut) 移入闭包，避免在闭包中重复解引用
                    let kv_ptrs_inner = kv_ptrs;
                    pool.compute(move |thread, threads| {
                        let q = unsafe { std::slice::from_raw_parts(q_ptr, n_embd_q) };
                        let attn_out =
                            unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_attn) };
                        let scores = unsafe {
                            std::slice::from_raw_parts_mut(
                                scores_ptr.add(thread * score_stride),
                                score_stride,
                            )
                        };
                        let f16_scratch = scores.as_mut_ptr().cast::<u16>();
                        let head_start = thread * config.n_head / threads;
                        let head_end = (thread + 1) * config.n_head / threads;
                        let layer_base = layer * capacity * kv_stride;
                        let n_padded = (step + 1).div_ceil(256) * 256;
                        match kv_ptrs_inner {
                            KvPtrs::F16 { k: k_ptr, v: v_ptr } => {
                                let k_cache =
                                    unsafe { std::slice::from_raw_parts(k_ptr, kv_cache_size) };
                                let v_cache =
                                    unsafe { std::slice::from_raw_parts(v_ptr, kv_cache_size) };
                                for head in head_start..head_end {
                                    let kv_head = head / group_size;
                                    let q_offset = head * config.n_embd_head_k;
                                    let output_offset = head * config.n_embd_head_v;
                                    let output = &mut attn_out
                                        [output_offset..output_offset + config.n_embd_head_v];
                                    let query = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            output.as_mut_ptr().cast::<u16>(),
                                            config.n_embd_head_k,
                                        )
                                    };
                                    f32_slice_to_f16(
                                        &q[q_offset..q_offset + config.n_embd_head_k],
                                        query,
                                    );
                                    scores[..n_padded].fill(f32::NEG_INFINITY);
                                    for token in 0..=step {
                                        let row = layer_base + token * kv_stride;
                                        let key_offset = row + kv_head * config.n_embd_head_k;
                                        scores[token] = dot_f16(
                                            query,
                                            &k_cache[key_offset..key_offset + config.n_embd_head_k],
                                            config.n_embd_head_k,
                                        ) * kq_scale;
                                    }
                                    softmax_inplace(&mut scores[..n_padded]);
                                    for index in 0..n_padded {
                                        unsafe {
                                            *f16_scratch.add(index) = f32_to_f16(scores[index])
                                        };
                                    }
                                    let weights = unsafe {
                                        std::slice::from_raw_parts(f16_scratch, n_padded)
                                    };
                                    let values = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            f16_scratch.add(score_stride),
                                            n_padded,
                                        )
                                    };
                                    values[step + 1..].fill(0);
                                    for dimension in 0..config.n_embd_head_v {
                                        for token in 0..=step {
                                            let row = layer_base + token * kv_stride;
                                            values[token] = v_cache
                                                [row + kv_head * config.n_embd_head_v + dimension];
                                        }
                                        output[dimension] = dot_f16(values, weights, n_padded);
                                    }
                                }
                            }
                            KvPtrs::F32 { k: k_ptr, v: v_ptr } => {
                                // F32 路径：直接用 f32 计算，无 F16 优化
                                let k_cache =
                                    unsafe { std::slice::from_raw_parts(k_ptr, kv_cache_size) };
                                let v_cache =
                                    unsafe { std::slice::from_raw_parts(v_ptr, kv_cache_size) };
                                for head in head_start..head_end {
                                    let kv_head = head / group_size;
                                    let q_offset = head * config.n_embd_head_k;
                                    let output_offset = head * config.n_embd_head_v;
                                    let output = &mut attn_out
                                        [output_offset..output_offset + config.n_embd_head_v];
                                    let query = &q[q_offset..q_offset + config.n_embd_head_k];
                                    scores[..n_padded].fill(f32::NEG_INFINITY);
                                    for token in 0..=step {
                                        let row = layer_base + token * kv_stride;
                                        let key_offset = row + kv_head * config.n_embd_head_k;
                                        scores[token] = dot_f32(
                                            query,
                                            &k_cache[key_offset..key_offset + config.n_embd_head_k],
                                            config.n_embd_head_k,
                                        ) * kq_scale;
                                    }
                                    softmax_inplace(&mut scores[..n_padded]);
                                    // F32 路径：values 直接存 f32， weights 也存 f32
                                    // 用 scores 复用区（scores[..n_padded] 已经 softmax 过）
                                    let weights = &scores[..n_padded];
                                    for dimension in 0..config.n_embd_head_v {
                                        let mut acc = 0.0f32;
                                        for token in 0..=step {
                                            let row = layer_base + token * kv_stride;
                                            let v = v_cache
                                                [row + kv_head * config.n_embd_head_v + dimension];
                                            acc += weights[token] * v;
                                        }
                                        output[dimension] = acc;
                                    }
                                }
                            }
                        }
                    });

                    let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_attn) };
                    let attn_input_ptr = attn_out.as_ptr();
                    #[cfg(feature = "parity-trace")]
                    if layer == 0 {
                        parity_trace::report(parity_trace::checkpoint(
                            "kqv_out-0",
                            Some(0),
                            &[config.n_head, config.n_embd_head_v],
                            attn_out,
                        ));
                    }
                    let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
                    let scale_buf =
                        unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
                    quantize_q8_0_into(
                        attn_out,
                        n_attn,
                        &mut q8_buf[..n_attn],
                        &mut scale_buf[..n_attn / 32],
                    );
                    let q8 = q8_buf[..n_attn].as_ptr();
                    let scales = scale_buf[..n_attn / 32].as_ptr();
                    let pool = Arc::clone(&model.pool);
                    pool.compute(move |thread, threads| {
                        let q8 = unsafe { std::slice::from_raw_parts(q8, n_attn) };
                        let scales = unsafe { std::slice::from_raw_parts(scales, n_attn / 32) };
                        let attn_input =
                            unsafe { std::slice::from_raw_parts(attn_input_ptr, n_attn) };
                        let output =
                            unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, config.n_embd) };
                        weights.wo.kernel.forward_prepared(
                            attn_input,
                            q8,
                            scales,
                            None,
                            output,
                            n_attn,
                            config.n_embd,
                            thread,
                            threads,
                        );
                    });

                    let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, config.n_embd) };
                    let normed =
                        unsafe { std::slice::from_raw_parts_mut(normed_ptr, config.n_embd) };
                    let attn_projection =
                        unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, config.n_embd) };
                    for (hidden, projection) in x.iter_mut().zip(attn_projection) {
                        *hidden += *projection;
                    }
                    rms_norm(x, &weights.ffn_norm, normed, config.eps);
                    if weights.moe_router.is_some() {
                        forward_moe_token(normed, weights, config, unsafe {
                            std::slice::from_raw_parts_mut(down_buf_ptr, config.n_embd)
                        })?;
                    } else {
                        let norm_input_ptr = normed.as_ptr();
                        quantize_q8_0_into(
                            normed,
                            config.n_embd,
                            &mut q8_buf[..config.n_embd],
                            &mut scale_buf[..config.n_embd / 32],
                        );
                        let q8 = q8_buf[..config.n_embd].as_ptr();
                        let scales = scale_buf[..config.n_embd / 32].as_ptr();
                        let pool = Arc::clone(&model.pool);
                        pool.compute(move |thread, threads| {
                            let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_embd) };
                            let scales =
                                unsafe { std::slice::from_raw_parts(scales, config.n_embd / 32) };
                            let norm_input = unsafe {
                                std::slice::from_raw_parts(norm_input_ptr, config.n_embd)
                            };
                            let gate = unsafe {
                                std::slice::from_raw_parts_mut(gate_buf_ptr, config.n_ff)
                            };
                            let up =
                                unsafe { std::slice::from_raw_parts_mut(up_buf_ptr, config.n_ff) };
                            weights.w_gate.kernel.forward_prepared(
                                norm_input,
                                q8,
                                scales,
                                None,
                                up,
                                config.n_embd,
                                config.n_ff,
                                thread,
                                threads,
                            );
                            weights.w_up.kernel.forward_prepared(
                                norm_input,
                                q8,
                                scales,
                                None,
                                gate,
                                config.n_embd,
                                config.n_ff,
                                thread,
                                threads,
                            );
                            if crate::ops::gpu_matmul_active() {
                                // The matmul ran as one fenced GPU dispatch owned by
                                // thread 0 — per-thread row slices have no data
                                // dependency to hang off, so thread 0 applies the
                                // epilogue over the whole buffer.
                                if thread == 0 {
                                    silu_mul_approx_inplace(
                                        &up[..config.n_ff],
                                        &mut gate[..config.n_ff],
                                    );
                                }
                            } else {
                                let start = thread * config.n_ff / threads;
                                let end = (thread + 1) * config.n_ff / threads;
                                silu_mul_approx_inplace(&up[start..end], &mut gate[start..end]);
                            }
                        });

                        let gate =
                            unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, config.n_ff) };
                        let gate_input_ptr = gate.as_ptr();
                        quantize_q8_0_into(
                            gate,
                            config.n_ff,
                            &mut q8_buf[..config.n_ff],
                            &mut scale_buf[..config.n_ff / 32],
                        );
                        let q8 = q8_buf[..config.n_ff].as_ptr();
                        let scales = scale_buf[..config.n_ff / 32].as_ptr();
                        let pool = Arc::clone(&model.pool);
                        pool.compute(move |thread, threads| {
                            let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_ff) };
                            let scales =
                                unsafe { std::slice::from_raw_parts(scales, config.n_ff / 32) };
                            let gate_input =
                                unsafe { std::slice::from_raw_parts(gate_input_ptr, config.n_ff) };
                            let down = unsafe {
                                std::slice::from_raw_parts_mut(down_buf_ptr, config.n_embd)
                            };
                            weights.w_down.kernel.forward_prepared(
                                gate_input,
                                q8,
                                scales,
                                None,
                                down,
                                config.n_ff,
                                config.n_embd,
                                thread,
                                threads,
                            );
                        });
                    }

                    let down =
                        unsafe { std::slice::from_raw_parts_mut(down_buf_ptr, config.n_embd) };
                    #[cfg(feature = "parity-trace")]
                    if layer == 0 {
                        parity_trace::report(parity_trace::checkpoint(
                            "ffn_out-0",
                            Some(0),
                            &[1, config.n_embd],
                            down,
                        ));
                    }
                    let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, config.n_embd) };
                    for (hidden, projection) in x.iter_mut().zip(down) {
                        *hidden += *projection;
                    }
                    if step < n_prompt && layer < config.n_deepstack_layers {
                        if let Some(deepstack) = input.deepstack_embeddings {
                            add_deepstack_embedding(
                                x,
                                deepstack,
                                layer,
                                step,
                                n_prompt,
                                config.n_embd,
                            );
                        }
                    }
                }

                rms_norm(
                    &self.scratch.x,
                    &model.output_norm,
                    &mut self.scratch.normed,
                    config.eps,
                );
                let output_input_ptr = self.scratch.normed.as_ptr();
                #[cfg(feature = "parity-trace")]
                parity_trace::report(parity_trace::checkpoint(
                    "result_norm",
                    None,
                    &[1, config.n_embd],
                    &self.scratch.normed,
                ));
                quantize_q8_0_into(
                    &self.scratch.normed,
                    config.n_embd,
                    &mut self.scratch.q8_buf[..config.n_embd],
                    &mut self.scratch.scale_buf[..config.n_embd / 32],
                );
                let q8 = self.scratch.q8_buf[..config.n_embd].as_ptr();
                let scales = self.scratch.scale_buf[..config.n_embd / 32].as_ptr();
                let logits_ptr = self.scratch.logits.as_mut_ptr();
                let pool = Arc::clone(&model.pool);
                pool.compute(move |thread, threads| {
                    let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_embd) };
                    let scales = unsafe { std::slice::from_raw_parts(scales, config.n_embd / 32) };
                    let output_input =
                        unsafe { std::slice::from_raw_parts(output_input_ptr, config.n_embd) };
                    let logits =
                        unsafe { std::slice::from_raw_parts_mut(logits_ptr, config.vocab) };
                    model.output.kernel.forward_prepared(
                        output_input,
                        q8,
                        scales,
                        None,
                        logits,
                        config.n_embd,
                        config.vocab,
                        thread,
                        threads,
                    );
                });
            }
            let elapsed = eval_start.elapsed();
            if step < n_prompt {
                prompt_duration += elapsed;
            } else {
                decode_duration += elapsed;
            }
            #[cfg(feature = "parity-trace")]
            parity_trace::report(parity_trace::checkpoint(
                "result_output",
                None,
                &[config.vocab],
                &self.scratch.logits,
            ));

            #[cfg(feature = "parity-trace")]
            if asr_trace && step == n_prompt - 1 {
                parity_trace::report(parity_trace::checkpoint(
                    "asr.decoder_first_logits",
                    None,
                    &[config.vocab],
                    &self.scratch.logits,
                ));
            }

            if step < n_prompt - 1 {
                continue;
            }
            let token_id = sample_token(&self.scratch.logits, options.temperature)?;
            if model.tokenizer.eos_id() == Some(token_id)
                || model.tokenizer.special_token_id("im_end") == Some(token_id)
            {
                break;
            }
            if generated_tokens.len() >= options.max_new_tokens {
                break;
            }
            let text = decoder.push(token_id);
            if !text.is_empty() {
                if let Some(cb) = on_token.as_mut() {
                    cb(&text);
                }
                rendered_tokens.push(text);
            }
            generated_tokens.push(token_id);
        }

        #[cfg(feature = "parity-trace")]
        parity_trace::report(parity_trace::token_ids(
            if asr_trace {
                "asr.generated_ids"
            } else {
                "generated_ids"
            },
            &generated_tokens,
        ));
        let tail = decoder.finish();
        if !tail.is_empty() {
            rendered_tokens.push(tail);
        }
        Ok(Qwen3Generation {
            text: rendered_tokens.concat(),
            rendered_tokens,
            token_ids: generated_tokens,
            prompt_tokens: n_prompt,
            prompt_duration,
            decode_duration,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::add_deepstack_embedding;

    #[test]
    fn deepstack_injection_uses_layer_major_token_rows() {
        let deepstack = [
            1.0, 2.0, 3.0, 4.0, // decoder layer 0, prompt tokens 0 and 1
            5.0, 6.0, 7.0, 8.0, // decoder layer 1, prompt tokens 0 and 1
        ];
        let mut hidden = [10.0, 20.0];

        add_deepstack_embedding(&mut hidden, &deepstack, 1, 0, 2, 2);

        assert_eq!(hidden, [15.0, 26.0]);
    }
}
