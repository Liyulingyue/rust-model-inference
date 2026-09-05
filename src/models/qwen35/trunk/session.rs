//! High-level inference state for a `Qwen35Model`.
//!
//! `Qwen35Model` only owns weights and config. `Qwen35Session` wraps a
//! reference to a model with the per-request state (KV cache, scratchpad,
//! thread pool) and exposes the standard forward API:
//!   - `embed_tokens`: lookup token embeddings
//!   - `step`: run one forward pass, return next-token logits
//!   - `reset`: clear cache + state for a new request
//!
//! This mirrors `qwen3::Qwen3Session` and replaces the ad-hoc per-call-site
//! kv_cache + scratchpad construction in `app/text.rs` and `bin/server.rs`.
//!
//! Existing call sites that use `Qwen35Model::forward` directly keep working;
//! `Session::step` is additive.

use std::sync::Arc;

use super::config::Qwen35Config;
use super::scratch::Qwen35Scratchpad;
use super::weights::Qwen35Model;
use crate::core::scratchpad::KvCache;
use crate::core::thread_pool::ComputePool;
#[cfg(feature = "vulkan")]
use crate::vulkan::qwen35::{commit_shadow_state, Qwen35VulkanSession};

/// Per-request inference state for a `Qwen35Model`.
///
/// Holds:
/// - the KV cache for dense attention layers
/// - the scratchpad (Mamba SSM states, FFN/attention buffers)
/// - the thread pool
/// - the next logical position for incremental decode
///
/// Construction validates the model is loaded (no extra work) and allocates
/// the cache and scratchpad sized for `model.config.n_ctx`.
pub struct Qwen35Session<'a> {
    model: &'a Qwen35Model<'a>,
    capacity: usize,
    kv_cache: KvCache,
    scratch: Qwen35Scratchpad,
    pool: Arc<ComputePool>,
    /// Next logical mrope position to assign during incremental decode.
    /// Updated by callers that manage their own generation loop
    /// (see `step_with_tokens`). Vision-augmented flows set this explicitly
    /// via `step(embeddings, n_tokens, positions)` because image tokens
    /// consume multiple positions.
    next_position: usize,
    processed_tokens: usize,
    #[cfg(feature = "vulkan")]
    gpu: Option<Qwen35VulkanSession>,
    #[cfg(feature = "vulkan")]
    full_model_gpu_failed: bool,
    #[cfg(all(test, feature = "vulkan"))]
    gpu_failure_for_test: Option<String>,
}

pub(super) fn required_token_count(
    processed_tokens: usize,
    incoming_tokens: usize,
    capacity: usize,
) -> Result<usize, String> {
    let required = processed_tokens
        .checked_add(incoming_tokens)
        .ok_or_else(|| "Qwen3.5 processed token count overflow".to_string())?;
    if required > capacity {
        return Err(format!(
            "Qwen3.5 step requires {required} tokens; session capacity is {capacity}"
        ));
    }
    Ok(required)
}

impl<'a> Qwen35Session<'a> {
    /// Build a session with cache and scratch sized for `model.config.n_ctx`.
    /// `pool` is shared across sessions (typical) so a single `Arc<ComputePool>`
    /// is sufficient.
    pub fn new(
        model: &'a Qwen35Model<'a>,
        capacity: usize,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let cfg = &model.config;
        if capacity == 0 || capacity > cfg.n_ctx {
            return Err(format!(
                "Qwen3.5 session capacity {capacity} must be within 1..={}",
                cfg.n_ctx
            ));
        }
        let kv_cache = KvCache::new_f32(cfg.n_layer_impl(), capacity, cfg.n_embd_gqa());
        let scratch = Qwen35Scratchpad::new(cfg, capacity);
        #[cfg(feature = "vulkan")]
        let (gpu, full_model_gpu_failed) = match crate::ops::get_vulkan_context() {
            Some(context) => match Qwen35VulkanSession::try_new(model, capacity, context) {
                Ok(gpu) => (gpu, false),
                Err(error) => {
                    eprintln!(
                        "[GPU] Qwen3.5 Vulkan session unavailable: {error}. Falling back to CPU."
                    );
                    (None, true)
                }
            },
            None => (None, false),
        };
        Ok(Self {
            model,
            capacity,
            kv_cache,
            scratch,
            pool,
            next_position: 0,
            processed_tokens: 0,
            #[cfg(feature = "vulkan")]
            gpu,
            #[cfg(feature = "vulkan")]
            full_model_gpu_failed,
            #[cfg(all(test, feature = "vulkan"))]
            gpu_failure_for_test: None,
        })
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.model.config
    }
    pub fn model(&self) -> &Qwen35Model<'a> {
        self.model
    }
    pub fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }
    pub fn kv_cache_mut(&mut self) -> &mut KvCache {
        &mut self.kv_cache
    }
    pub fn scratch(&self) -> &Qwen35Scratchpad {
        &self.scratch
    }
    pub fn scratch_mut(&mut self) -> &mut Qwen35Scratchpad {
        &mut self.scratch
    }
    pub fn pool(&self) -> &ComputePool {
        &self.pool
    }
    pub fn next_position(&self) -> usize {
        self.next_position
    }
    pub fn set_next_position(&mut self, position: usize) {
        self.next_position = position;
    }

    /// Clear KV cache and scratch state. Use this between unrelated requests
    /// to free the previous prompt's attention state. Does NOT reallocate
    /// (just zeros in place).
    pub fn reset(&mut self) {
        let cfg = &self.model.config;
        self.kv_cache = KvCache::new_f32(cfg.n_layer_impl(), self.capacity, cfg.n_embd_gqa());
        let mut fresh = Qwen35Scratchpad::new(cfg, self.capacity);
        std::mem::swap(&mut self.scratch, &mut fresh);
        // `fresh` is dropped here, freeing its buffers
        self.next_position = 0;
        self.processed_tokens = 0;
        #[cfg(feature = "vulkan")]
        if let Some(gpu) = &mut self.gpu {
            if let Err(error) = gpu.reset() {
                eprintln!(
                    "[GPU] Qwen3.5 Vulkan session disabled after reset error: {error}. Falling back to CPU."
                );
                self.gpu = None;
                self.full_model_gpu_failed = true;
            }
        }
    }

    /// Look up a single token's embedding row. Returns an error if the token
    /// id exceeds the embedding table.
    pub fn embed_token(&self, token_id: u32) -> Result<Vec<f32>, String> {
        self.model.embed_tokens(&[token_id])
    }

    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        self.model.embed_tokens(token_ids)
    }

    /// Run one forward pass over pre-computed embeddings of shape
    /// `[n_tokens, n_embd]`. Returns the logits for predicting the next
    /// token after the last input position (shape `[vocab_size]`).
    ///
    /// `positions` has shape `[n_tokens, 4]` (mrope time/row/column).
    /// The caller is responsible for setting `next_position` (typically
    /// to `positions.last().map(|p| p[0] + 1).unwrap_or(0)`).
    pub fn step(
        &mut self,
        embeddings: &[f32],
        n_tokens: usize,
        positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.model.config;
        let n_embd = cfg.n_embd;
        if n_tokens == 0 {
            return Err("Qwen3.5 step requires at least one token".into());
        }
        let expected_embeddings = n_tokens
            .checked_mul(n_embd)
            .ok_or_else(|| "Qwen3.5 embedding length overflow".to_string())?;
        if embeddings.len() != expected_embeddings {
            return Err(format!(
                "Qwen3.5 embeddings length {} != n_tokens * n_embd = {}",
                embeddings.len(),
                expected_embeddings
            ));
        }
        if positions.len() != n_tokens {
            return Err(format!(
                "Qwen3.5 positions length {} != n_tokens = {}",
                positions.len(),
                n_tokens
            ));
        }
        let required = required_token_count(self.processed_tokens, n_tokens, self.capacity)?;

        #[cfg(feature = "vulkan")]
        let mut gpu_available = self.gpu.is_some();
        #[cfg(all(test, feature = "vulkan"))]
        {
            gpu_available |= self.gpu_failure_for_test.is_some();
        }
        #[cfg(feature = "vulkan")]
        if gpu_available {
            let mut logits = vec![0.0; cfg.vocab_size];
            for (token_index, (token, position)) in
                embeddings.chunks_exact(n_embd).zip(positions).enumerate()
            {
                let cache_position = self.processed_tokens;
                #[cfg(test)]
                let forced_failure = self.gpu_failure_for_test.take();
                #[cfg(not(test))]
                let forced_failure: Option<String> = None;
                let attempt = if let Some(error) = forced_failure {
                    Err(error)
                } else {
                    let gpu = self.gpu.as_mut().expect("checked above");
                    match gpu.forward_token(token, cache_position, *position) {
                        Ok(result) => {
                            let shadow_commit = commit_shadow_state(
                                &mut self.kv_cache,
                                &mut self.scratch.conv_states,
                                &mut self.scratch.ssm_states,
                                cache_position,
                                self.capacity,
                                cfg.n_embd_head() * cfg.n_head_kv,
                                result.k_delta,
                                result.v_delta,
                                result.conv_state,
                                result.ssm_state,
                            );
                            match shadow_commit {
                                Ok(()) => {
                                    logits.copy_from_slice(result.logits);
                                    gpu.commit_token();
                                    Ok(())
                                }
                                Err(error) => {
                                    gpu.abort_token();
                                    Err(error)
                                }
                            }
                        }
                        Err(error) => Err(error.to_string()),
                    }
                };

                match attempt {
                    Ok(()) => {
                        self.processed_tokens += 1;
                        self.next_position = position[0].saturating_add(1);
                    }
                    Err(error) => {
                        eprintln!(
                            "[GPU] Qwen3.5 Vulkan session disabled after error: {error}. Falling back to CPU."
                        );
                        self.gpu = None;
                        self.full_model_gpu_failed = true;
                        let remaining_embeddings = &embeddings[token_index * n_embd..];
                        let remaining_positions = &positions[token_index..];
                        self.scratch.x[..remaining_embeddings.len()]
                            .copy_from_slice(remaining_embeddings);
                        let _gpu_matmul_scope = self
                            .full_model_gpu_failed
                            .then(ComputePool::disable_gpu_matmul_for_scope);
                        logits = self.model.forward(
                            remaining_positions.len(),
                            &mut self.kv_cache,
                            &mut self.scratch,
                            &self.pool,
                            remaining_positions,
                        )?;
                        self.processed_tokens = required;
                        if let Some(last) = remaining_positions.last() {
                            self.next_position = last[0].saturating_add(1);
                        }
                        break;
                    }
                }
            }
            return Ok(logits);
        }

        for t in 0..n_tokens {
            let off = t * n_embd;
            self.scratch.x[off..off + n_embd].copy_from_slice(&embeddings[off..off + n_embd]);
        }
        #[cfg(feature = "vulkan")]
        let _gpu_matmul_scope = self
            .full_model_gpu_failed
            .then(ComputePool::disable_gpu_matmul_for_scope);
        let logits = self.model.forward(
            n_tokens,
            &mut self.kv_cache,
            &mut self.scratch,
            &self.pool,
            positions,
        )?;
        self.processed_tokens = required;
        if let Some(last) = positions.last() {
            self.next_position = last[0].saturating_add(1);
        }
        Ok(logits)
    }

    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn fail_gpu_once_for_test(&mut self, reason: &str) {
        self.gpu_failure_for_test = Some(reason.into());
    }

    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn gpu_enabled_for_test(&self) -> bool {
        self.gpu.is_some() || self.gpu_failure_for_test.is_some()
    }

    /// Embed + step in a single call (the common prefill/decode pattern for
    /// text-only inference). Returns next-token logits.
    pub fn step_with_tokens(
        &mut self,
        token_ids: &[u32],
        positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        let embeddings = self.embed_tokens(token_ids)?;
        self.step(&embeddings, token_ids.len(), positions)
    }
}
