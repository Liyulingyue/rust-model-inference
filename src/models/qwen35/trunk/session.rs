//! High-level inference state for the shared hybrid trunk.
//!
//! `Qwen35Model` only owns weights and config. `HybridSession` wraps a
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

use crate::compute::ComputePolicy;
use std::marker::PhantomData;
use std::sync::Arc;

use super::config::Qwen35Config;
use super::scratch::Qwen35Scratchpad;
use super::weights::{HybridTrunk, Qwen35Model};
use crate::core::prefill::{
    checked_prefill_batch_size, prefill_chunks, DEFAULT_PREFILL_BATCH_SIZE,
};
use crate::core::scratchpad::KvCache;
use crate::core::thread_pool::ComputePool;
#[cfg(feature = "vulkan")]
use crate::vulkan::qwen35::{commit_shadow_state_chunk, Qwen35VulkanSession};

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35DenseKvSnapshot {
    pub layer: usize,
    pub tokens: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

impl Qwen35DenseKvSnapshot {
    pub fn shape(&self) -> [usize; 3] {
        [self.tokens, self.kv_heads, self.head_dim]
    }
}

/// Per-request inference state for Qwen3.5 or Edge0.
///
/// Holds:
/// - the KV cache for dense attention layers
/// - the scratchpad (Mamba SSM states, FFN/attention buffers)
/// - the thread pool
/// - the next logical position for incremental decode
///
/// Construction validates the model is loaded (no extra work) and allocates
/// the cache and scratchpad sized for `model.config.n_ctx`.
pub struct HybridSession<'a, 'm, M: HybridTrunkModel<'m>> {
    model: &'a mut M,
    source_lifetime: PhantomData<&'m ()>,
    capacity: usize,
    compute_policy: ComputePolicy,
    kv_cache: KvCache,
    /// The MTP draft head's own KV cache. One layer, as in llama.cpp: the
    /// draft context is separate from the target's, so the draft block
    /// attends over the drafted positions and not over the target's keys.
    /// `None` when the checkpoint has no MTP head.
    mtp_cache: Option<KvCache>,
    /// The hidden state of the position before the next committed one: the
    /// MTP head pairs a token with the hidden state of the *previous*
    /// position, so the first row of a batch needs the last row of the
    /// previous one. llama.cpp keeps the same value as `pending_h`.
    mtp_pending_hidden: Vec<f32>,
    /// A copy of the recurrent state, taken before a speculative
    /// verification. The GDN and short-convolution layers carry their state
    /// from token to token, so a rejected draft cannot be undone by
    /// truncating the KV cache alone: the state has to be put back.
    ///
    /// It is flat (the layers back to back) because that is the layout the
    /// device keeps, so restoring it there is a straight upload.
    mtp_state_backup: Option<(Vec<f32>, Vec<f32>)>,
    /// Whether the next forward should hand back a hidden state per row.
    /// The MTP head chains its drafts from the row before each position and
    /// fills its cache from every prompt row, so both need the rows.
    mtp_collect_hidden: bool,
    /// Whether the next forward should project every row's logits. The
    /// verification reads the target's answer at each drafted position.
    mtp_collect_logits: bool,
    /// Whether the next forward should capture the recurrent layers'
    /// projections. The device rollback replays the recurrent state from
    /// them.
    mtp_collect_capture: bool,
    /// Whether the last forward projected the logits of every row, so that
    /// the verification can read them instead of projecting them again.
    mtp_logits_ready: bool,
    scratch: Qwen35Scratchpad,
    prefill_batch_size: usize,
    pool: Arc<ComputePool>,
    /// Next logical mrope position to assign during incremental decode.
    /// Updated by callers that manage their own generation loop
    /// (see `step_with_tokens`). Vision-augmented flows set this explicitly
    /// via `step(embeddings, n_tokens, positions)` because image tokens
    /// consume multiple positions.
    next_position: usize,
    processed_tokens: usize,
    last_step_tokens: usize,
    #[cfg(feature = "vulkan")]
    gpu: Option<Qwen35VulkanSession>,
    #[cfg(feature = "vulkan")]
    full_model_gpu_failed: bool,
    #[cfg(all(test, feature = "vulkan"))]
    gpu_failure_for_test: Option<String>,
    #[cfg(test)]
    fail_cpu_chunk_after_row: Option<usize>,
}

pub type Qwen35Session<'a, 'm> = HybridSession<'a, 'm, Qwen35Model<'m>>;

/// What a speculative decoding run produced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MtpGeneration {
    /// The generated tokens.
    pub tokens: Vec<u32>,
    /// How many speculative rounds ran.
    pub rounds: usize,
    /// How many of the drafted tokens the target accepted.
    pub accepted: usize,
    /// How many tokens were drafted.
    pub drafted: usize,
}

/// What one speculative decoding step produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpStep {
    /// The tokens the step commits, in order: the bonus token the target
    /// had already sampled and the drafts it accepted.
    pub tokens: Vec<u32>,
    /// The token the target sampled after the committed ones. It is the
    /// bonus token of the next round: llama.cpp samples it from the
    /// verification logits as well, so it costs no extra forward pass.
    pub next: u32,
    /// How many of the drafted tokens the target accepted.
    pub accepted: usize,
    /// How many tokens were drafted.
    pub drafted: usize,
}

/// Shared session operations for the Qwen3.5 and Edge0 hybrid trunks.
pub trait HybridTrunkModel<'m> {
    fn trunk(&self) -> &HybridTrunk<'m>;
    fn forward_at(
        &mut self,
        n_tokens: usize,
        base_position: usize,
        kv_cache: &mut KvCache,
        scratch: &mut Qwen35Scratchpad,
        pool: &ComputePool,
        positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String>;
    fn forward_chunk(
        &mut self,
        n_tokens: usize,
        base_position: usize,
        kv_cache: &mut KvCache,
        scratch: &mut Qwen35Scratchpad,
        conv_states: &mut [Vec<f32>],
        ssm_states: &mut [Vec<f32>],
        pool: &ComputePool,
        mrope_positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        if conv_states.len() != scratch.conv_states.len()
            || ssm_states.len() != scratch.ssm_states.len()
        {
            return Err("Qwen3.5 recurrent state layer count mismatch".into());
        }
        for (persistent, working) in scratch.conv_states.iter_mut().zip(conv_states.iter_mut()) {
            std::mem::swap(persistent, working);
        }
        for (persistent, working) in scratch.ssm_states.iter_mut().zip(ssm_states.iter_mut()) {
            std::mem::swap(persistent, working);
        }
        let result = self.forward_at(
            n_tokens,
            base_position,
            kv_cache,
            scratch,
            pool,
            mrope_positions,
        );
        for (persistent, working) in scratch.conv_states.iter_mut().zip(conv_states.iter_mut()) {
            std::mem::swap(persistent, working);
        }
        for (persistent, working) in scratch.ssm_states.iter_mut().zip(ssm_states.iter_mut()) {
            std::mem::swap(persistent, working);
        }
        result
    }
    #[cfg(feature = "vulkan")]
    fn supports_vulkan(&self) -> bool;
    #[cfg(feature = "vulkan")]
    fn stream_routing(&self) -> Option<crate::models::sensenova_loop::StreamRouting<'_, 'm>> {
        None
    }
}

impl<'m> HybridTrunkModel<'m> for HybridTrunk<'m> {
    fn trunk(&self) -> &HybridTrunk<'m> {
        self
    }
    fn forward_at(
        &mut self,
        n_tokens: usize,
        base_position: usize,
        kv_cache: &mut KvCache,
        scratch: &mut Qwen35Scratchpad,
        pool: &ComputePool,
        positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        Qwen35Model::forward_at(
            self,
            n_tokens,
            base_position,
            kv_cache,
            scratch,
            pool,
            positions,
        )
    }
    #[cfg(feature = "vulkan")]
    fn supports_vulkan(&self) -> bool {
        true
    }
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

pub(super) fn dense_kv_chunk_is_finite(
    cache: &KvCache,
    config: &super::config::Qwen35Config,
    capacity: usize,
    base_position: usize,
    rows: usize,
) -> bool {
    let KvCache::F32(values) = cache else {
        return false;
    };
    let stride = config.n_embd_gqa();
    let layer_len = capacity * stride;
    for (layer, recurrent) in config.is_recurrent.iter().copied().enumerate() {
        if recurrent {
            continue;
        }
        let key_start = layer * layer_len + base_position * stride;
        let key_end = key_start + rows * stride;
        if !values
            .k
            .get(key_start..key_end)
            .is_some_and(|chunk| chunk.iter().all(|value| value.is_finite()))
        {
            return false;
        }
        let value_base = layer * layer_len;
        for dimension in 0..stride {
            let start = value_base + dimension * capacity + base_position;
            if !values
                .v
                .get(start..start + rows)
                .is_some_and(|chunk| chunk.iter().all(|value| value.is_finite()))
            {
                return false;
            }
        }
    }
    true
}

impl<'a, 'm, M: HybridTrunkModel<'m>> HybridSession<'a, 'm, M> {
    /// Build a session with cache and scratch sized for `model.config.n_ctx`.
    /// `pool` is shared across sessions (typical) so a single `Arc<ComputePool>`
    /// is sufficient.
    pub fn new(model: &'a mut M, capacity: usize, pool: Arc<ComputePool>) -> Result<Self, String> {
        Self::new_with_prefill_batch_size(model, capacity, DEFAULT_PREFILL_BATCH_SIZE, pool)
    }

    pub fn new_with_prefill_batch_size(
        model: &'a mut M,
        capacity: usize,
        prefill_batch_size: usize,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        Self::new_with_compute(
            model,
            capacity,
            prefill_batch_size,
            pool,
            ComputePolicy::legacy(),
        )
    }

    pub fn new_with_compute(
        model: &'a mut M,
        capacity: usize,
        prefill_batch_size: usize,
        pool: Arc<ComputePool>,
        compute_policy: ComputePolicy,
    ) -> Result<Self, String> {
        compute_policy.check_build()?;
        let _scope = compute_policy.cpu_scope();
        let prefill_batch_size = checked_prefill_batch_size(Some(prefill_batch_size))?;
        let cfg = &model.trunk().config;
        if capacity == 0 || capacity > cfg.n_ctx {
            return Err(format!(
                "Qwen3.5 session capacity {capacity} must be within 1..={}",
                cfg.n_ctx
            ));
        }
        let kv_cache = KvCache::new_f32(cfg.n_layer_impl(), capacity, cfg.n_embd_gqa());
        let mtp_cache = model
            .trunk()
            .mtp
            .as_ref()
            .map(|_| KvCache::new_f32(1, capacity, cfg.n_embd_gqa()));
        let mtp_pending_hidden = vec![0.0f32; cfg.n_embd];
        let mtp_state_backup = None;
        let scratch = Qwen35Scratchpad::new(cfg, capacity.min(prefill_batch_size));
        #[cfg(feature = "vulkan")]
        let (gpu, full_model_gpu_failed) = match if model.supports_vulkan() {
            compute_policy
                .context()
                .map_err(|error| error.to_string())?
        } else if compute_policy == ComputePolicy::Vulkan {
            return Err("This hybrid model does not support full-model Vulkan".into());
        } else {
            None
        } {
            Some(context) => match Qwen35VulkanSession::try_new_rows_with_routing(
                model.trunk(),
                capacity,
                capacity.min(prefill_batch_size),
                context,
                model.stream_routing(),
            ) {
                Ok(None) if compute_policy == ComputePolicy::Vulkan => {
                    return Err("Qwen3.5 configuration is unsupported for full-model Vulkan".into());
                }
                Ok(gpu) => (gpu, false),
                Err(error) if compute_policy == ComputePolicy::Vulkan => {
                    return Err(error.to_string())
                }
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
            source_lifetime: PhantomData,
            capacity,
            compute_policy,
            kv_cache,
            mtp_cache,
            mtp_pending_hidden,
            mtp_state_backup,
            mtp_collect_hidden: false,
            mtp_collect_logits: false,
            mtp_collect_capture: false,
            mtp_logits_ready: false,
            scratch,
            prefill_batch_size,
            pool,
            next_position: 0,
            processed_tokens: 0,
            last_step_tokens: 0,
            #[cfg(feature = "vulkan")]
            gpu,
            #[cfg(feature = "vulkan")]
            full_model_gpu_failed,
            #[cfg(all(test, feature = "vulkan"))]
            gpu_failure_for_test: None,
            #[cfg(test)]
            fail_cpu_chunk_after_row: None,
        })
    }

    #[cfg(feature = "vulkan")]
    fn gpu_available(&self) -> bool {
        #[cfg(test)]
        if self.gpu_failure_for_test.is_some() {
            return true;
        }
        self.gpu.is_some()
    }

    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn force_vulkan_policy_for_test(&mut self) {
        self.compute_policy = ComputePolicy::Vulkan;
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.model.trunk().config
    }
    pub fn model(&self) -> &M {
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
    pub fn last_hidden(&self, tokens: usize) -> Result<&[f32], String> {
        if tokens > self.last_step_tokens {
            return Err(format!(
                "Qwen3.5 hidden request for {tokens} rows exceeds the last step of {} rows",
                self.last_step_tokens
            ));
        }
        let len = tokens
            .checked_mul(self.model.trunk().config.n_embd)
            .ok_or_else(|| "Qwen3.5 hidden length overflow".to_string())?;
        self.scratch
            .normed_buf
            .get(..len)
            .ok_or_else(|| "Qwen3.5 hidden request exceeds the last step".into())
    }
    /// Fill the MTP head's KV cache over the tokens the target just
    /// committed.
    ///
    /// `tokens` are the committed tokens, `positions` their logical
    /// positions and `hidden` the target's normalized hidden rows (one row
    /// per token, the row of the token's own position).
    ///
    /// The head pairs the token at position `p` with the hidden state of
    /// position `p - 1`: it predicts the token *after* the one it is given,
    /// so it still needs the state of the position before it. llama.cpp
    /// shifts the target's hidden rows by one for the same reason and keeps
    /// the row that has no predecessor in `pending_h`
    /// (`common/speculative.cpp`, `common_speculative_impl_draft_mtp`).
    /// This session keeps the same carry-over row, so the head sees the
    /// committed positions in its attention exactly as upstream does.
    pub fn mtp_catch_up(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        hidden: &[f32],
    ) -> Result<(), String> {
        let n_embd = self.model.trunk().config.n_embd;
        if tokens.len() != positions.len() {
            return Err(format!(
                "Qwen3.5 MTP catch-up: {} tokens but {} positions",
                tokens.len(),
                positions.len()
            ));
        }
        if hidden.len() < tokens.len() * n_embd {
            return Err(format!(
                "Qwen3.5 MTP catch-up: {} hidden rows for {} tokens",
                hidden.len() / n_embd.max(1),
                tokens.len()
            ));
        }
        let Some(mtp) = self.model.trunk().mtp.as_ref() else {
            return Err("this checkpoint has no MTP head".into());
        };
        // On the device the head has a KV cache of its own in the device's
        // arena, and the catch-up is one batch of rows over it. A head the
        // device cannot hold stays on the CPU.
        #[cfg(feature = "vulkan")]
        if self.gpu.as_ref().is_some_and(|gpu| gpu.has_mtp_head()) {
            return self.mtp_catch_up_device(tokens, positions, hidden);
        }
        let Some(cache) = self.mtp_cache.as_mut() else {
            return Err("the MTP head has no KV cache".into());
        };
        let _scope = self.compute_policy.cpu_scope();
        // The head has no device recorder and reads the host copy of its
        // weights, so its matmuls stay on the CPU even when the target runs
        // on the device.
        #[cfg(feature = "vulkan")]
        let _cpu_only = ComputePool::disable_gpu_matmul_for_scope();
        let mut pending = std::mem::take(&mut self.mtp_pending_hidden);
        for (index, (&token, &position)) in tokens.iter().zip(positions).enumerate() {
            // The first row has no predecessor inside this batch, so it uses
            // the carry-over row; every later row uses the row before it.
            let row: &[f32] = if index == 0 {
                &pending
            } else {
                &hidden[(index - 1) * n_embd..index * n_embd]
            };
            let mrope = [position, position, position, 0];
            super::mtp::forward_mtp(
                self.model.trunk(),
                mtp,
                row,
                token,
                position,
                mrope,
                cache,
                &mut self.scratch,
                &self.pool,
                // The row only fills the head's cache: nothing samples it,
                // so neither the head's output nor its vocabulary
                // projection is needed.
                false,
            )?;
        }
        // The last row of the batch becomes the carry-over row for the next
        // batch, which is how the head chains across the batch boundary.
        if !tokens.is_empty() {
            pending.copy_from_slice(&hidden[(tokens.len() - 1) * n_embd..tokens.len() * n_embd]);
        }
        self.mtp_pending_hidden = pending;
        Ok(())
    }

    /// The hidden state of the position before the next committed one.
    ///
    /// The MTP head pairs the token at position `p` with the hidden state of
    /// position `p - 1`, so the first draft of a round needs this row.
    pub fn mtp_pending_hidden(&self) -> Result<&[f32], String> {
        Ok(&self.mtp_pending_hidden)
    }

    /// Fill the MTP head's KV cache over the committed tokens, on the device.
    ///
    /// The head's cache lives in the device's arena, next to the target's
    /// own: it is the layer slot after the main stack. The rows are the
    /// committed tokens, and the head pairs each of them with the hidden
    /// state of the position before it, so the first row of the batch uses
    /// the carry-over row and every later row the row before it.
    ///
    /// The rows are one batch when their positions are consecutive, which is
    /// the usual case: the attention reads the cache row by row, so a batch
    /// of consecutive positions is one call.
    #[cfg(feature = "vulkan")]
    fn mtp_catch_up_device(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        hidden: &[f32],
    ) -> Result<(), String> {
        let n_embd = self.model.trunk().config.n_embd;
        let embeddings = self.embed_tokens(tokens)?;
        let mut pending = std::mem::take(&mut self.mtp_pending_hidden);
        if tokens.is_empty() {
            self.mtp_pending_hidden = pending;
            return Ok(());
        }
        let mut rows = Vec::with_capacity(tokens.len() * n_embd);
        rows.extend_from_slice(&pending);
        for index in 1..tokens.len() {
            rows.extend_from_slice(&hidden[(index - 1) * n_embd..index * n_embd]);
        }
        let outcome = {
            let gpu = match self.gpu.as_mut() {
                Some(gpu) => gpu,
                None => {
                    self.mtp_pending_hidden = pending;
                    return Err("the Vulkan session went away".into());
                }
            };
            let consecutive = positions.windows(2).all(|pair| pair[1] == pair[0] + 1);
            if consecutive {
                gpu.mtp_head(&embeddings, &rows, positions[0], false)
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            } else {
                let mut outcome = Ok(());
                for (index, &position) in positions.iter().enumerate() {
                    let row = &rows[index * n_embd..(index + 1) * n_embd];
                    if let Err(error) = gpu
                        .mtp_head(
                            &embeddings[index * n_embd..(index + 1) * n_embd],
                            row,
                            position,
                            false,
                        )
                        .map_err(|error| error.to_string())
                    {
                        outcome = Err(error);
                        break;
                    }
                }
                outcome
            }
        };
        // The last row of the batch becomes the carry-over row for the next
        // batch, which is how the head chains across the batch boundary.
        if !tokens.is_empty() {
            pending.copy_from_slice(&hidden[(tokens.len() - 1) * n_embd..tokens.len() * n_embd]);
        }
        self.mtp_pending_hidden = pending;
        outcome
    }

    /// The target's logits for one row of the last forward pass.
    ///
    /// A speculative verification needs the logits of the rows of the
    /// verification batch, not only the last one: the target has to be able
    /// to say what it would have produced at each drafted position. The
    /// rows are the target's normalized hidden states of the last chunk, so
    /// this only works after a forward that kept them.
    fn logits_row(&mut self, row: usize) -> Result<Vec<f32>, String> {
        let trunk = self.model.trunk();
        let n_embd = trunk.config.n_embd;
        let vocab = trunk.config.vocab_size;
        if row >= self.last_step_tokens {
            return Err("Qwen3.5 MTP verification needs a row of the last forward".into());
        }
        // The forward projected every row at once: read them from there.
        if self.mtp_logits_ready && self.scratch.logits_rows.len() >= (row + 1) * vocab {
            return Ok(self.scratch.logits_rows[row * vocab..(row + 1) * vocab].to_vec());
        }
        let normed = &self.scratch.normed_buf[row * n_embd..(row + 1) * n_embd];
        trunk.output_weight.quantize_and_matmul_with_scratch(
            normed,
            &mut self.scratch.q8k_buf,
            &mut self.scratch.q8_buf,
            &mut self.scratch.scale_buf,
            &mut self.scratch.matmul_out,
            &self.pool,
        );
        let n = vocab.min(self.scratch.matmul_out.len());
        Ok(self.scratch.matmul_out[..n].to_vec())
    }

    /// One speculative decoding step with the MTP draft head.
    ///
    /// `token` is the token the target sampled at `position`, `n` the number
    /// of tokens to draft. The head proposes `n` tokens, the target checks
    /// them in a single forward pass over `[token, draft_0, ...]`, and the
    /// step returns the tokens to commit: the accepted draft prefix plus
    /// the token the target itself would have produced next.
    ///
    /// The draft head runs on the CPU: the target may run on the device, in
    /// which case the head consumes the hidden states the device hands back.
    pub fn mtp_speculative_step(
        &mut self,
        token: u32,
        position: usize,
        n: usize,
    ) -> Result<MtpStep, String> {
        if self.model.trunk().mtp.is_none() {
            return Err("this checkpoint has no MTP head".into());
        }
        #[cfg(feature = "vulkan")]
        let on_device = self.gpu.is_some();
        #[cfg(not(feature = "vulkan"))]
        let on_device = false;
        if n == 0 {
            return Ok(MtpStep {
                tokens: vec![token],
                next: token,
                accepted: 0,
                drafted: 0,
            });
        }

        // The draft: the head runs at `position` with the token the target
        // just sampled and proposes the tokens after it.
        let drafts = self.mtp_draft_tokens(token, position, n)?;

        // The verification: the target runs the bonus token and the whole
        // draft in one forward pass.
        let mut batch = Vec::with_capacity(drafts.len() + 1);
        batch.push(token);
        batch.extend_from_slice(&drafts);
        let positions: Vec<[usize; 4]> = (0..batch.len())
            .map(|offset| {
                let p = position + offset;
                [p, p, p, 0]
            })
            .collect();

        // The GDN and short-convolution layers carry their state from token
        // to token, so the state has to be saved before the verification:
        // a rejected draft cannot be undone by truncating the KV cache alone.
        self.save_mtp_state();

        let embeddings = self.embed_tokens(&batch)?;
        let processed_before = self.processed_tokens;
        // Capture the recurrent layers' projections: the CPU rollback replays
        // only those layers instead of the whole model. The device keeps its
        // own capture of the same projections and replays the state update
        // from them.
        if !on_device {
            self.prepare_recurrent_capture(batch.len());
        }
        // The verification needs the target's logits for every row of the
        // batch, so the forward projects them all in one pass: the
        // vocabulary matrix is 1.3 GB, and projecting the rows one at a time
        // reads it once per row. The rows also have to survive the forward:
        // the head chains its drafts from the row before each position, so
        // the whole batch has to fit in one chunk.
        if batch.len() > self.prefill_batch_size {
            return Err(format!(
                "the Qwen3.5 MTP draft batch of {} rows does not fit in the chunk size {}",
                batch.len(),
                self.prefill_batch_size
            ));
        }
        self.scratch.collect_all_logits = !on_device;
        self.mtp_collect_hidden = on_device;
        self.mtp_collect_logits = on_device;
        // The device rollback replays the recurrent state, which needs the
        // recurrent layers' projections: the chunk captures them.
        self.mtp_collect_capture = on_device;
        let last_logits = self.step(&embeddings, batch.len(), &positions)?;
        self.scratch.collect_all_logits = false;
        self.mtp_collect_hidden = false;
        self.mtp_collect_logits = false;
        self.mtp_collect_capture = false;
        self.mtp_logits_ready = self.scratch.logits_rows_collected;
        self.scratch.capture_recurrent_replay = false;

        // The target's own continuation: the logits of row `i` say what the
        // target would produce at the position of draft `i`.
        //
        // The rows are read one at a time and only as far as the round
        // needs them: the first row that does not match its draft ends the
        // round, so the rows after it are never looked at. Every row is a
        // vocabulary projection, and the last row's projection is the one
        // the forward pass already did.
        let mut accepted = 0usize;
        let mut next = 0u32;
        loop {
            let logits = if accepted + 1 == batch.len() {
                last_logits.clone()
            } else {
                self.logits_row(accepted)?
            };
            let sampled = crate::ops::sampling::argmax(&logits) as u32;
            if accepted < drafts.len() && sampled == drafts[accepted] {
                accepted += 1;
                continue;
            }
            next = sampled;
            break;
        }
        // The round commits the bonus token the target had already sampled
        // and the drafts it accepted. The token the target produces after
        // them is the bonus token of the next round.
        let mut tokens = Vec::with_capacity(accepted + 1);
        tokens.push(token);
        tokens.extend_from_slice(&drafts[..accepted]);

        // The draft context has to see the same positions as the target: the
        // head's KV cache is refreshed with the target's hidden states, which
        // is what llama.cpp's draft context does after every verification.
        let n_embd = self.model.trunk().config.n_embd;
        let hidden = self.scratch.normed_buf[..batch.len() * n_embd].to_vec();
        self.mtp_catch_up(&batch, &positions_as_usize(&positions), &hidden)?;
        // The carry-over row is the hidden state of the last accepted
        // position, so the next round starts from the target's state and not
        // from a rejected draft's.
        self.mtp_pending_hidden
            .copy_from_slice(&hidden[accepted * n_embd..(accepted + 1) * n_embd]);

        // The rejected drafts left their KV and their recurrent state behind:
        // both have to go, or the next round would continue from a state the
        // target never produced.
        if accepted < drafts.len() {
            self.restore_mtp_state();
            let cfg = &self.model.trunk().config;
            let k_dim = cfg.n_embd_head() * cfg.n_head_kv;
            // The committed rows of the verification batch are already
            // correct -- every row of a forward pass sees exactly the state
            // its own position has. Only the rows after them are wrong, and
            // the free slot of a layer is the first all-zero key row.
            super::scratch::kv_cache_truncate(
                &mut self.kv_cache,
                cfg.n_layer_impl(),
                k_dim,
                position + accepted + 1,
                position + batch.len(),
            );
            // The recurrent state is the one thing that cannot be
            // truncated: the GDN and short-convolution layers carry it from
            // token to token. The CPU path restores it and replays only those
            // layers over the accepted rows; the device path replays the
            // committed rows through the model.
            let keep = accepted + 1;
            #[cfg(feature = "vulkan")]
            if on_device {
                self.mtp_replay_device(position, keep, processed_before)?;
                return Ok(MtpStep {
                    tokens,
                    next,
                    accepted,
                    drafted: drafts.len(),
                });
            }
            self.model
                .trunk()
                .replay_recurrent_state(keep, &mut self.scratch, &self.pool)?;
            self.processed_tokens = processed_before + keep;
            self.next_position = position + keep;
        }

        Ok(MtpStep {
            tokens,
            next,
            accepted,
            drafted: drafts.len(),
        })
    }

    /// Put the device back where the committed tokens leave it.
    ///
    /// The verification advanced the device's recurrent state over every
    /// drafted row, accepted or not, and that state cannot be truncated the
    /// way the KV cache can. So the round puts the saved copy back and
    /// replays the committed rows over the recurrent layers: the chunk
    /// captured each recurrent layer's projections, and the state update
    /// consumes nothing else. The projections are the expensive half -- they
    /// read the recurrent layers' weights -- so replaying only the state
    /// update leaves a fraction of the cost of running the model again.
    #[cfg(feature = "vulkan")]
    fn mtp_replay_device(
        &mut self,
        position: usize,
        keep: usize,
        processed_before: usize,
    ) -> Result<(), String> {
        let cfg = &self.model.trunk().config;
        let conv_len: usize = self.scratch.conv_states.iter().map(Vec::len).sum();
        let ssm_len: usize = self.scratch.ssm_states.iter().map(Vec::len).sum();
        {
            // The host shadow was just restored; hand the same values to the
            // device.
            let (conv, ssm) = self
                .mtp_state_backup
                .as_ref()
                .filter(|(conv, ssm)| conv.len() == conv_len && ssm.len() == ssm_len)
                .ok_or("the Qwen3.5 MTP state backup does not match the recurrent state")?;
            let gpu = self.gpu.as_mut().ok_or("the Vulkan session went away")?;
            gpu.restore_recurrent_state(conv, ssm)
                .map_err(|error| error.to_string())?;
            let (conv_state, ssm_state) = gpu
                .replay_recurrent_rows(keep)
                .map_err(|error| error.to_string())?;
            // The keys of the committed rows are already in the cache: the
            // verification wrote them, and they do not depend on the state
            // the replay puts back.
            crate::vulkan::qwen35::commit_shadow_recurrent_state(
                &mut self.scratch.conv_states,
                &mut self.scratch.ssm_states,
                conv_state,
                ssm_state,
            )?;
            // The next chunk starts at the committed rows and overwrites
            // the rejected ones before anything reads them.
            gpu.rewind(position + keep)
                .map_err(|error| error.to_string())?;
        }
        self.processed_tokens = processed_before + keep;
        self.next_position = position + keep;
        self.last_step_tokens = keep;
        Ok(())
    }

    /// Prepare the capture of the recurrent layers' projections.
    ///
    /// The capture is what lets a rollback replay the recurrent layers
    /// alone: their state is a function of the incoming state and their
    /// projected rows, and the rows of the verification batch are the ones
    /// the forward actually used.
    fn prepare_recurrent_capture(&mut self, n_rows: usize) {
        let cfg = &self.model.trunk().config;
        let n_layer = cfg.n_layer_impl();
        let qkv_width = n_rows * cfg.conv_dim();
        let head_width = n_rows * cfg.ssm_dt_rank;
        let slots = &mut self.scratch.recurrent_replay;
        if slots.len() < n_layer {
            slots.resize_with(n_layer, Default::default);
        }
        for slot in slots.iter_mut().take(n_layer) {
            slot.qkv.clear();
            slot.beta.clear();
            slot.alpha.clear();
            slot.qkv.reserve(qkv_width);
            slot.beta.reserve(head_width);
            slot.alpha.reserve(head_width);
        }
        self.scratch.capture_recurrent_replay = true;
    }

    /// Save the recurrent state so a rejected draft can be undone.
    ///
    /// The state is tens of megabytes, so the backup keeps its buffers
    /// between rounds and only the contents are copied. It is flat (the
    /// layers back to back) because that is the layout the device keeps.
    fn save_mtp_state(&mut self) {
        let conv_len: usize = self.scratch.conv_states.iter().map(Vec::len).sum();
        let ssm_len: usize = self.scratch.ssm_states.iter().map(Vec::len).sum();
        let backup = self
            .mtp_state_backup
            .get_or_insert_with(|| (Vec::new(), Vec::new()));
        if backup.0.len() != conv_len || backup.1.len() != ssm_len {
            backup.0.resize(conv_len, 0.0);
            backup.1.resize(ssm_len, 0.0);
        }
        let mut offset = 0;
        for slot in &self.scratch.conv_states {
            backup.0[offset..offset + slot.len()].copy_from_slice(slot);
            offset += slot.len();
        }
        let mut offset = 0;
        for slot in &self.scratch.ssm_states {
            backup.1[offset..offset + slot.len()].copy_from_slice(slot);
            offset += slot.len();
        }
    }

    /// Put the recurrent state back after a rejected draft.
    fn restore_mtp_state(&mut self) {
        let Some((conv, ssm)) = self.mtp_state_backup.as_ref() else {
            return;
        };
        let conv_len: usize = self.scratch.conv_states.iter().map(Vec::len).sum();
        let ssm_len: usize = self.scratch.ssm_states.iter().map(Vec::len).sum();
        if conv.len() != conv_len || ssm.len() != ssm_len {
            return;
        }
        let mut offset = 0;
        for slot in self.scratch.conv_states.iter_mut() {
            let len = slot.len();
            slot.copy_from_slice(&conv[offset..offset + len]);
            offset += len;
        }
        let mut offset = 0;
        for slot in self.scratch.ssm_states.iter_mut() {
            let len = slot.len();
            slot.copy_from_slice(&ssm[offset..offset + len]);
            offset += len;
        }
    }

    /// Propose up to `n` draft tokens.
    ///
    /// `token` is the token the target just sampled and `position` its
    /// logical position. The head pairs that token with the hidden state of
    /// the position before it -- the carry-over row [`Self::mtp_pending_hidden`]
    /// -- and each draft step feeds the head's own hidden state into the
    /// next one, which is how llama.cpp chains the drafts.
    ///
    /// The head keeps its own KV cache, so the draft block attends over the
    /// committed and the drafted positions, exactly as llama.cpp's separate
    /// draft context does.
    pub fn mtp_draft_tokens(
        &mut self,
        token: u32,
        position: usize,
        n: usize,
    ) -> Result<Vec<u32>, String> {
        let Some(mtp) = self.model.trunk().mtp.as_ref() else {
            return Err("this checkpoint has no MTP head".into());
        };
        // On the device the head has weights and a KV cache of its own, so
        // the draft steps run there. A head the device cannot hold stays on
        // the CPU.
        #[cfg(feature = "vulkan")]
        if self.gpu.as_ref().is_some_and(|gpu| gpu.has_mtp_head()) {
            return self.mtp_draft_tokens_device(token, position, n);
        }
        let Some(cache) = self.mtp_cache.as_mut() else {
            return Err("the MTP head has no KV cache".into());
        };
        let _scope = self.compute_policy.cpu_scope();
        // The head has no device recorder and reads the host copy of its
        // weights, so its matmuls stay on the CPU even when the target runs
        // on the device.
        #[cfg(feature = "vulkan")]
        let _cpu_only = ComputePool::disable_gpu_matmul_for_scope();
        let mut drafts = Vec::with_capacity(n);
        let mut token = token;
        let mut position = position;
        let mut hidden = self.mtp_pending_hidden.clone();
        for _ in 0..n {
            let mrope = [position, position, position, 0];
            let draft = super::mtp::forward_mtp(
                self.model.trunk(),
                mtp,
                &hidden,
                token,
                position,
                mrope,
                cache,
                &mut self.scratch,
                &self.pool,
                // The draft is sampled, so it needs the head's logits.
                true,
            )?;
            let next = crate::ops::sampling::argmax(&draft.logits) as u32;
            drafts.push(next);
            token = next;
            position += 1;
            hidden = draft.hidden;
        }
        Ok(drafts)
    }

    /// Propose the draft tokens, on the device.
    ///
    /// The head's weights and its KV cache are on the device, so each draft
    /// step is one call there: the projection, the attention over the head's
    /// own cache, the feed-forward and the vocabulary projection.
    ///
    /// The chain is the same as on the CPU: the first step takes the
    /// carry-over row, and every later step takes the head's own hidden
    /// state.
    #[cfg(feature = "vulkan")]
    fn mtp_draft_tokens_device(
        &mut self,
        token: u32,
        position: usize,
        n: usize,
    ) -> Result<Vec<u32>, String> {
        let mut drafts = Vec::with_capacity(n);
        let mut token = token;
        let mut position = position;
        let mut hidden = self.mtp_pending_hidden.clone();
        for _ in 0..n {
            let embedding = self.embed_tokens(&[token])?;
            let gpu = self.gpu.as_mut().ok_or("the Vulkan session went away")?;
            let draft = gpu
                .mtp_head(&embedding, &hidden, position, true)
                .map_err(|error| error.to_string())?;
            let next = crate::ops::sampling::argmax(&draft.logits) as u32;
            drafts.push(next);
            token = next;
            position += 1;
            hidden = draft.hidden;
        }
        Ok(drafts)
    }

    /// The main model's last hidden state after the output norm: what the
    /// MTP draft head consumes.
    ///
    /// The scratchpad keeps one row per token of the last forward, so the
    /// row of the *last* token is the one the MTP head needs -- not the
    /// first row of the chunk.
    pub fn last_hidden_state(&self) -> Result<&[f32], String> {
        let n_embd = self.model.trunk().config.n_embd;
        let rows = self.last_step_tokens.max(1);
        let start = (rows - 1) * n_embd;
        let end = start + n_embd;
        if end > self.scratch.normed_buf.len() {
            return Err("Qwen3.5 hidden state exceeds the scratchpad".into());
        }
        Ok(&self.scratch.normed_buf[start..end])
    }

    pub fn dense_kv_snapshots(
        &self,
        layers: &[usize],
        tokens: usize,
    ) -> Result<Vec<Qwen35DenseKvSnapshot>, String> {
        if tokens > self.processed_tokens {
            return Err(format!(
                "Qwen3.5 KV snapshot requests {tokens} tokens after {} processed",
                self.processed_tokens
            ));
        }
        let cfg = &self.model.trunk().config;
        let head_dim = cfg.n_embd_head();
        let kv_heads = cfg.n_head_kv;
        let kv_width = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Qwen3.5 KV width overflow".to_string())?;
        let layer_stride = self
            .capacity
            .checked_mul(kv_width)
            .ok_or_else(|| "Qwen3.5 KV layer stride overflow".to_string())?;
        let KvCache::F32(cache) = &self.kv_cache else {
            return Err("Qwen3.5 dense KV snapshots require F32 cache".into());
        };
        let mut seen = std::collections::HashSet::with_capacity(layers.len());
        let mut snapshots = Vec::with_capacity(layers.len());
        for &layer in layers {
            if layer >= cfg.n_layer_impl() {
                return Err(format!("Qwen3.5 KV snapshot layer {layer} is out of range"));
            }
            if cfg.is_recurrent[layer] {
                return Err(format!("Qwen3.5 KV snapshot layer {layer} is recurrent"));
            }
            if !seen.insert(layer) {
                return Err(format!("Duplicate Qwen3.5 KV snapshot layer {layer}"));
            }
            let values = tokens
                .checked_mul(kv_width)
                .ok_or_else(|| "Qwen3.5 KV snapshot length overflow".to_string())?;
            let base = layer
                .checked_mul(layer_stride)
                .ok_or_else(|| "Qwen3.5 KV snapshot offset overflow".to_string())?;
            let mut key = vec![0.0; values];
            let mut value = vec![0.0; values];
            key.copy_from_slice(&cache.k[base..base + values]);
            for token in 0..tokens {
                for head in 0..kv_heads {
                    for dim in 0..head_dim {
                        let dst = (token * kv_heads + head) * head_dim + dim;
                        let src = base + (head * head_dim + dim) * self.capacity + token;
                        value[dst] = cache.v[src];
                    }
                }
            }
            snapshots.push(Qwen35DenseKvSnapshot {
                layer,
                tokens,
                kv_heads,
                head_dim,
                key,
                value,
            });
        }
        Ok(snapshots)
    }
    pub fn pool(&self) -> &ComputePool {
        &self.pool
    }
    pub fn scratch_bytes(&self) -> usize {
        self.scratch.bytes()
    }
    pub fn next_position(&self) -> usize {
        self.next_position
    }
    pub fn set_next_position(&mut self, position: usize) {
        self.next_position = position;
    }

    pub(crate) fn processed_tokens(&self) -> usize {
        self.processed_tokens
    }

    /// Clear KV cache and scratch state. Use this between unrelated requests
    /// to free the previous prompt's attention state. Does NOT reallocate
    /// (just zeros in place).
    pub fn reset(&mut self) {
        let cfg = &self.model.trunk().config;
        self.kv_cache = KvCache::new_f32(cfg.n_layer_impl(), self.capacity, cfg.n_embd_gqa());
        let mut fresh = Qwen35Scratchpad::new(cfg, self.capacity.min(self.prefill_batch_size));
        std::mem::swap(&mut self.scratch, &mut fresh);
        // `fresh` is dropped here, freeing its buffers
        self.next_position = 0;
        self.processed_tokens = 0;
        self.last_step_tokens = 0;
        // The draft context starts empty as well: its KV cache covers the
        // committed positions, and the carry-over row belongs to the row
        // before the next committed one.
        if let Some(cache) = self.mtp_cache.as_mut() {
            cache.clear();
        }
        self.mtp_pending_hidden.fill(0.0);
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
        self.model.trunk().embed_tokens(&[token_id])
    }

    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        self.model.trunk().embed_tokens(token_ids)
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
        let _scope = self.compute_policy.cpu_scope();
        #[cfg(feature = "vulkan")]
        if self.compute_policy == ComputePolicy::Vulkan && !self.gpu_available() {
            return Err("Vulkan session unavailable; create a new session to retry".into());
        }
        let n_embd = self.model.trunk().config.n_embd;
        #[cfg(feature = "vulkan")]
        let kv_stride =
            self.model.trunk().config.n_embd_head() * self.model.trunk().config.n_head_kv;
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
        if embeddings.iter().any(|value| !value.is_finite()) {
            return Err("Qwen3.5 embeddings contain NaN or infinity".into());
        }
        let required = required_token_count(self.processed_tokens, n_tokens, self.capacity)?;
        self.last_step_tokens = 0;

        // The trace schema is token-major and contains every prompt row's logits.
        #[cfg(feature = "parity-trace")]
        let trace_each_token = std::env::var_os("RMI_PARITY_TRACE").is_some();
        #[cfg(not(feature = "parity-trace"))]
        let trace_each_token = false;
        // GPU tracing flushes each layer and records the same token-major schema.
        let batch_size = self.prefill_batch_size;
        #[cfg(feature = "vulkan")]
        let batch_size = if trace_each_token && self.gpu_available() {
            1
        } else {
            batch_size
        };
        let mut logits = Vec::new();
        for range in prefill_chunks(n_tokens, batch_size) {
            let rows = range.len();
            #[cfg(feature = "parity-trace")]
            let _trace = crate::parity_trace::TokenMajorTrace::new(rows);
            let base = self.processed_tokens;
            let chunk_embeddings = &embeddings[range.start * n_embd..range.end * n_embd];
            let chunk_positions = &positions[range.clone()];
            #[cfg(all(test, feature = "vulkan"))]
            let forced_failure = self.gpu_failure_for_test.take();
            #[cfg(all(feature = "vulkan", not(test)))]
            let forced_failure: Option<String> = None;
            #[cfg(feature = "vulkan")]
            let gpu_error = if self.gpu.is_some() || forced_failure.is_some() {
                let collect_hidden = self.mtp_collect_hidden;
                let collect_logits = self.mtp_collect_logits;
                let collect_capture = self.mtp_collect_capture;
                let vocab_size = self.model.trunk().config.vocab_size;
                let attempt = (|| {
                    if let Some(error) = forced_failure {
                        return Err(error);
                    }
                    let gpu = self.gpu.as_mut().expect("checked above");
                    gpu.set_collect(collect_hidden || collect_logits, collect_logits);
                    gpu.set_collect_capture(collect_capture);
                    let result = if rows == 1 {
                        gpu.forward_token(chunk_embeddings, base, chunk_positions[0])
                    } else {
                        gpu.forward_chunk(chunk_embeddings, base, chunk_positions, rows)
                    }
                    .map_err(|error| error.to_string())?;
                    if result.logits.len() != self.model.trunk().config.vocab_size
                        || result.logits.iter().any(|value| !value.is_finite())
                    {
                        return Err("invalid Qwen3.5 Vulkan logits".into());
                    }
                    commit_shadow_state_chunk(
                        &mut self.kv_cache,
                        &mut self.scratch.conv_states,
                        &mut self.scratch.ssm_states,
                        base,
                        rows,
                        self.capacity,
                        kv_stride,
                        result.k_delta,
                        result.v_delta,
                        result.conv_state,
                        result.ssm_state,
                    )?;
                    logits = result.logits.to_vec();
                    // The device path keeps a hidden state per row, like the
                    // CPU path does: the MTP head chains its drafts from the
                    // row before each one, and a catch-up refreshes its cache
                    // from every committed row.
                    if collect_hidden || collect_logits {
                        self.scratch.normed_buf[..rows * n_embd]
                            .copy_from_slice(&result.normed[..rows * n_embd]);
                    }
                    if collect_logits {
                        let count = rows * vocab_size;
                        self.scratch.logits_rows.clear();
                        self.scratch
                            .logits_rows
                            .extend_from_slice(&result.logits_rows[..count]);
                        self.scratch.logits_rows_collected = true;
                    } else {
                        self.scratch.logits_rows_collected = false;
                    }
                    gpu.commit_token();
                    Ok(())
                })();
                match attempt {
                    Ok(()) => {
                        self.processed_tokens += rows;
                        self.next_position = chunk_positions[rows - 1][0].saturating_add(1);
                        // The verification reads the target's answer at every
                        // row of the chunk, so the session has to know how
                        // many rows the last forward covered.
                        self.last_step_tokens = rows;
                        self.compute_policy.trace(
                            "resident_hybrid_decoder",
                            crate::compute::UsedBackend::Vulkan,
                            rows,
                        );
                        continue;
                    }
                    Err(error) => {
                        if let Some(gpu) = &mut self.gpu {
                            gpu.abort_token();
                        }
                        self.gpu = None;
                        self.full_model_gpu_failed = true;
                        if self.compute_policy == ComputePolicy::Vulkan {
                            return Err(error);
                        }
                        eprintln!("[GPU] Qwen3.5 Vulkan chunk {base}..{} failed: {error}. Recomputing the whole chunk on CPU.", base + rows);
                        Some(error)
                    }
                }
            } else {
                None
            };

            #[cfg(feature = "vulkan")]
            let _gpu_matmul_scope = ComputePool::disable_gpu_matmul_for_scope();
            self.scratch.x[..rows * n_embd].copy_from_slice(chunk_embeddings);
            // Preserve the existing CPU decode path; only a failed GPU chunk needs a retry snapshot.
            let direct_decode = n_tokens == 1;
            #[cfg(feature = "vulkan")]
            let direct_decode = direct_decode && gpu_error.is_none();
            #[cfg(test)]
            let direct_decode = direct_decode && self.fail_cpu_chunk_after_row.is_none();
            if direct_decode {
                logits = self.model.forward_at(
                    1,
                    base,
                    &mut self.kv_cache,
                    &mut self.scratch,
                    &self.pool,
                    chunk_positions,
                )?;
                self.processed_tokens += 1;
                self.next_position = chunk_positions[0][0].saturating_add(1);
                self.last_step_tokens = 1;
                continue;
            }
            let mut working_conv = self.scratch.conv_states.clone();
            let mut working_ssm = self.scratch.ssm_states.clone();
            let result = (|| {
                #[cfg(test)]
                super::forward::set_cpu_scan_failure(self.fail_cpu_chunk_after_row.take());
                let result = self.model.forward_chunk(
                    rows,
                    base,
                    &mut self.kv_cache,
                    &mut self.scratch,
                    &mut working_conv,
                    &mut working_ssm,
                    &self.pool,
                    chunk_positions,
                );
                #[cfg(test)]
                super::forward::set_cpu_scan_failure(None);
                let logits = result?;
                if logits.iter().any(|value| !value.is_finite())
                    || !dense_kv_chunk_is_finite(
                        &self.kv_cache,
                        &self.model.trunk().config,
                        self.capacity,
                        base,
                        rows,
                    )
                    || working_conv
                        .iter()
                        .flatten()
                        .chain(working_ssm.iter().flatten())
                        .any(|value| !value.is_finite())
                {
                    return Err("Qwen3.5 prefill produced non-finite state".into());
                }
                Ok(logits)
            })();
            logits = match result {
                Ok(logits) => logits,
                Err(error) => {
                    #[cfg(feature = "vulkan")]
                    if let Some(gpu_error) = gpu_error {
                        return Err(format!("{error}; original Vulkan error: {gpu_error}"));
                    }
                    return Err(error);
                }
            };
            self.scratch.conv_states = working_conv;
            self.scratch.ssm_states = working_ssm;
            self.compute_policy
                .trace("hybrid_decoder", crate::compute::UsedBackend::Cpu, rows);
            self.processed_tokens += rows;
            self.next_position = chunk_positions[rows - 1][0].saturating_add(1);
            self.last_step_tokens = rows;
        }
        debug_assert_eq!(self.processed_tokens, required);
        Ok(logits)
    }

    #[cfg(test)]
    pub(crate) fn fail_cpu_chunk_after_row_for_test(&mut self, row: usize) {
        self.fail_cpu_chunk_after_row = Some(row);
    }

    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn fail_gpu_once_for_test(&mut self, reason: &str) {
        self.gpu_failure_for_test = Some(reason.into());
    }

    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn fail_gpu_after_row_for_test(&mut self, row: usize) {
        self.gpu
            .as_mut()
            .expect("Vulkan session required")
            .fail_after_row = Some(row);
    }

    #[cfg(all(test, feature = "vulkan"))]
    pub(crate) fn gpu_enabled_for_test(&self) -> bool {
        self.gpu.is_some() || self.gpu_failure_for_test.is_some()
    }

    /// Prefill the prompt and keep the MTP draft context in sync with it.
    ///
    /// The prompt is chunked like a normal prefill, but every chunk also
    /// runs through the MTP head: the draft block attends over the
    /// committed positions, so its KV cache has to cover the prompt before
    /// the first draft.
    ///
    /// The MTP head embeds the tokens itself, so this path needs the token
    /// ids -- the image path, whose prompt rows are projected embeddings,
    /// cannot use it.
    pub fn prefill_with_mtp(
        &mut self,
        token_ids: &[u32],
        positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        let n_embd = self.model.trunk().config.n_embd;
        let mut logits = Vec::new();
        for range in prefill_chunks(token_ids.len(), self.prefill_batch_size) {
            let ids = &token_ids[range.clone()];
            let chunk_positions = &positions[range.clone()];
            // The head refreshes its cache from every prompt row, so the
            // chunk has to hand back a hidden state per row. Its logits are
            // not used: projecting a whole prompt batch would read the
            // vocabulary matrix for rows nobody looks at.
            self.mtp_collect_hidden = true;
            let stepped = self.step_with_tokens(ids, chunk_positions);
            self.mtp_collect_hidden = false;
            logits = stepped?;
            let hidden = self.scratch.normed_buf[..ids.len() * n_embd].to_vec();
            let flat: Vec<usize> = chunk_positions.iter().map(|p| p[0]).collect();
            self.mtp_catch_up(ids, &flat, &hidden)?;
        }
        Ok(logits)
    }

    /// Generate `max_tokens` tokens with the MTP draft head.
    ///
    /// The head proposes `n_drafts` tokens per round and the target checks
    /// them in one forward pass, so a round can commit more than one token.
    /// The tokens are the target's own greedy continuation: the draft head
    /// only proposes, the target decides.
    ///
    /// Greedy only. A sampled token would need the rejection scheme of
    /// speculative sampling, which is a different piece of work.
    pub fn mtp_generate(
        &mut self,
        prompt_tokens: &[u32],
        prompt_positions: &[[usize; 4]],
        max_tokens: usize,
        n_drafts: usize,
    ) -> Result<MtpGeneration, String> {
        if self.model.trunk().mtp.is_none() {
            return Err("this checkpoint has no MTP head".into());
        }
        if max_tokens == 0 {
            return Ok(MtpGeneration::default());
        }
        let logits = self.prefill_with_mtp(prompt_tokens, prompt_positions)?;
        self.mtp_generate_after_prefill(logits, prompt_positions, max_tokens, n_drafts)
    }

    /// Generate `max_tokens` tokens with the MTP draft head, starting from a
    /// prefill that already ran.
    ///
    /// The prompt's logits are the ones the prefill produced: the first
    /// round's bonus token comes from them. A caller that wants to time the
    /// prefill apart from the rounds uses this instead of
    /// [`Self::mtp_generate`].
    pub fn mtp_generate_after_prefill(
        &mut self,
        logits: Vec<f32>,
        prompt_positions: &[[usize; 4]],
        max_tokens: usize,
        n_drafts: usize,
    ) -> Result<MtpGeneration, String> {
        if self.model.trunk().mtp.is_none() {
            return Err("this checkpoint has no MTP head".into());
        }
        if max_tokens == 0 {
            return Ok(MtpGeneration::default());
        }
        let mut position = prompt_positions
            .last()
            .map(|p| p[0].saturating_add(1))
            .unwrap_or(0);
        let mut bonus = crate::ops::sampling::argmax(&logits) as u32;
        let mut out: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut rounds = 0usize;
        let mut accepted = 0usize;
        let mut drafted = 0usize;
        while out.len() < max_tokens {
            let step = self.mtp_speculative_step(bonus, position, n_drafts)?;
            rounds += 1;
            accepted += step.accepted;
            drafted += step.drafted;
            position += step.tokens.len();
            bonus = step.next;
            for token in step.tokens {
                out.push(token);
                if out.len() >= max_tokens {
                    break;
                }
            }
        }
        Ok(MtpGeneration {
            tokens: out,
            rounds,
            accepted,
            drafted,
        })
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

    /// Single forward pass: prefill a prompt and return the last-position
    /// logits. Used by JEV / classification modes that do not need
    /// autoregressive decoding. The caller is responsible for building
    /// mrope `positions`; this method does NOT reset KV state, so reuse
    /// the session for at most one call or recreate it between questions.
    pub fn forward_logits(
        &mut self,
        token_ids: &[u32],
        positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() {
            return Err("Qwen3.5 prompt must contain at least one token".into());
        }
        self.step_with_tokens(token_ids, positions)
    }

    /// Prefill a prompt, then classify the last non-pad token via the
    /// model's `cls.output.weight` head. Returns one logit per class
    /// (no softmax — callers apply softmax themselves, mirroring
    /// `Qwen3Model::score_logits`). Errors if the head isn't loaded.
    ///
    /// The session must have `is_classifier()` true. The caller is
    /// responsible for building mrope `positions`; this method does
    /// NOT reset KV state, so reuse the session for at most one call or
    /// recreate it between questions.
    pub fn forward_classify(
        &mut self,
        token_ids: &[u32],
        positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        if !self.model.trunk().is_classifier() {
            return Err("forward_classify requires cls.output.weight".into());
        }
        if token_ids.is_empty() {
            return Err("Qwen3.5 classify prompt must contain at least one token".into());
        }
        // Drive the full forward pass (this also produces the lm_head
        // logits, which we discard — for the 0.8B NLI head the matmul
        // is a 1024 × 1024 × 248320 step; cheap enough to throw away).
        let _logits = self.step_with_tokens(token_ids, positions)?;
        // The classification head expects the **last non-pad token**'s
        // hidden state (HF `modeling_openjev.py` does the same thing via
        // `attention_mask.sum(1) - 1`). `Qwen35Session::last_hidden(N)`
        // returns the FIRST N rows of `normed_buf`, which is wrong here —
        // for a fixed NLI prompt the first row is dominated by the
        // shared `Premise:` prefix and is essentially identical across
        // every premise/hypothesis pair, which would force the head to
        // produce the same logits regardless of the actual text.
        let n_embd = self.model.trunk().config.n_embd;
        let last_offset = (self.last_step_tokens.saturating_sub(1))
            .checked_mul(n_embd)
            .ok_or_else(|| "Qwen3.5 last_hidden offset overflow".to_string())?;
        let scratch_len = self.scratch().normed_buf.len();
        let hidden = self
            .scratch()
            .normed_buf
            .get(last_offset..last_offset + n_embd)
            .ok_or_else(|| {
                format!(
                    "Qwen3.5 classify last_hidden: scratch len {scratch_len}, \
                     wanted rows [{}, {})",
                    last_offset / n_embd,
                    last_offset / n_embd + 1,
                )
            })?;
        self.model.trunk().score_logits(hidden)
    }
}

/// The flat (non-mrope) positions of an mrope position list.
fn positions_as_usize(positions: &[[usize; 4]]) -> Vec<usize> {
    positions.iter().map(|p| p[0]).collect()
}
