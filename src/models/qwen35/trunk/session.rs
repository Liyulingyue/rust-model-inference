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
        Ok(Self {
            model,
            capacity,
            kv_cache,
            scratch,
            pool,
            next_position: 0,
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
        if n_tokens > self.capacity {
            return Err(format!(
                "Qwen3.5 token count {n_tokens} exceeds session capacity {}",
                self.capacity
            ));
        }
        if embeddings.len() != n_tokens * n_embd {
            return Err(format!(
                "Qwen3.5 embeddings length {} != n_tokens * n_embd = {}",
                embeddings.len(),
                n_tokens * n_embd
            ));
        }
        if positions.len() != n_tokens {
            return Err(format!(
                "Qwen3.5 positions length {} != n_tokens = {}",
                positions.len(),
                n_tokens
            ));
        }
        for t in 0..n_tokens {
            let off = t * n_embd;
            self.scratch.x[off..off + n_embd].copy_from_slice(&embeddings[off..off + n_embd]);
        }
        let logits = self.model.forward(
            n_tokens,
            &mut self.kv_cache,
            &mut self.scratch,
            &self.pool,
            positions,
        )?;
        if let Some(last) = positions.last() {
            self.next_position = last[0] + 1;
        }
        Ok(logits)
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
