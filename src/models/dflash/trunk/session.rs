//! The session of a DFlash / DSpark draft model: its KV cache, its
//! scratchpad, and the two driving operations.
//!
//! * [`DflashSession::inject_target_features`] is the encoder pass: it turns
//!   the target model's per-layer hidden states into the draft's KV cache
//!   (llama.cpp's `graph<false>` with `ubatch.embd` set).
//! * [`DflashSession::draft_block`] is the decode pass: it denoises one
//!   `[anchor, MASK, ...]` block and applies the DSpark heads — the Markov
//!   head's bias, the confidence head's `p_min` gate, and the greedy pick.
//!
//! The block layout follows `common/speculative.cpp`'s
//! `common_speculative_impl_draft_dflash::draft`:
//!
//! ```text
//! sample_from_anchor  (this repo's files): slot 0 is the anchor *and* a
//!     prediction slot, so a `block_size` block drafts `block_size` tokens.
//! !sample_from_anchor (SpecForge exports): slot 0 is a bonus token, not a
//!     prediction slot; `block_size - 1` slots draft `block_size - 1` tokens.
//! ```

use super::forward::{decode_block, inject_target_features};
use super::scratch::DflashScratchpad;
use super::weights::DflashWeights;
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::Weight;

/// The draft's own KV cache.
///
/// The layout is `[layer][position][kv head][dimension]` — one contiguous
/// block per layer, positions linear (the caller sizes the cache to the
/// target's context, so positions never wrap).
pub struct DflashKvState {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub n_layer: usize,
    pub capacity: usize,
    /// One row of one layer: `n_head_kv * n_embd_head`.
    pub kv_stride: usize,
    /// One past the highest position ever written.
    pub n_filled: usize,
}

impl DflashKvState {
    fn new(n_layer: usize, capacity: usize, kv_stride: usize) -> Self {
        Self {
            k: vec![0.0f32; n_layer * capacity * kv_stride],
            v: vec![0.0f32; n_layer * capacity * kv_stride],
            n_layer,
            capacity,
            kv_stride,
            n_filled: 0,
        }
    }
}

/// One round's draft: the proposed tokens and, when the file carries the
/// confidence head, each position's acceptance estimate.
#[derive(Debug, Default, Clone)]
pub struct DsparkDraft {
    pub tokens: Vec<u32>,
    pub confidences: Vec<f32>,
}

pub struct DflashSession<'d, 't> {
    pub weights: DflashWeights<'d>,
    /// The target's token embedding table, shared with the draft exactly as
    /// llama.cpp shares it through `ctx_other`.
    token_embd: Weight<'t>,
    kv: DflashKvState,
    scratch: DflashScratchpad,
    mask_token: u32,
    /// The confidence gate: positions whose acceptance estimate falls below
    /// it are cut from the draft. Zero disables the gate.
    p_min: f32,
}

impl<'d, 't> DflashSession<'d, 't> {
    /// Build the session.
    ///
    /// `capacity` is the number of positions the KV cache holds — the
    /// target's context, since the draft runs over the same positions.
    /// `max_rows` is the row count the scratchpad is sized for: the larger
    /// of the block size and one encoder-injection chunk.
    pub fn new(
        weights: DflashWeights<'d>,
        token_embd: Weight<'t>,
        n_threads: usize,
        capacity: usize,
        max_rows: usize,
    ) -> Self {
        let config = &weights.config;
        let mask_token = config.mask_token;
        let n_layer = config.n_layer;
        let kv_stride = config.n_head_kv * config.n_embd_head_k;
        let n_vocab = config.n_vocab.max(token_embd.n_out as usize);
        let block_size = config.block_size;
        let scratch = DflashScratchpad::new(
            config,
            n_vocab,
            max_rows.max(block_size),
            n_threads,
            capacity,
        );
        Self {
            mask_token,
            weights,
            token_embd,
            kv: DflashKvState::new(n_layer, capacity, kv_stride),
            scratch,
            p_min: 0.0,
        }
    }

    /// The confidence gate: draft positions whose acceptance estimate falls
    /// below it are cut. Zero disables the gate.
    pub fn p_min(&self) -> f32 {
        self.p_min
    }

    /// Set the confidence gate.
    pub fn set_p_min(&mut self, p_min: f32) {
        self.p_min = p_min;
    }

    /// The encoder's input width: `target_layers.len()` times the target's
    /// hidden size.
    pub fn n_embd_enc(&self) -> usize {
        self.weights.n_embd_enc
    }

    /// The vocabulary the sampler reads: the target's.
    pub fn n_vocab(&self) -> usize {
        self.scratch.n_vocab
    }

    /// The mask token, in the target's vocabulary.
    pub fn mask_token(&self) -> u32 {
        self.mask_token
    }

    /// One past the highest position ever written into the KV cache.
    pub fn kv_n_filled(&self) -> usize {
        self.kv.n_filled
    }

    /// The maximum draft one round can propose.
    pub fn n_draft_max(&self) -> usize {
        if self.weights.config.sample_from_anchor {
            self.weights.config.block_size
        } else {
            self.weights.config.block_size - 1
        }
    }

    /// The encoder pass: write the target's per-layer hidden states into
    /// the draft's KV cache.
    ///
    /// `hidden` is `[rows][target_layers.len() * n_embd_target]` — each row
    /// the target's hidden state at each `target_layers` entry, concatenated
    /// in layer order.
    pub fn inject_target_features(
        &mut self,
        hidden: &[f32],
        positions: &[usize],
        pool: &ComputePool,
    ) -> Result<(), String> {
        inject_target_features(
            &self.weights,
            hidden,
            positions,
            &mut self.kv,
            &mut self.scratch,
            pool,
        )
    }

    /// Draft one block after `anchor` (the last committed token, at absolute
    /// position `position`).
    ///
    /// Returns at most `n_draft` tokens, in position order — the tokens the
    /// decoder predicts for `position + 1 ..= position + n_draft`, each gated
    /// by the confidence head when `p_min > 0`.
    pub fn draft_block(
        &mut self,
        anchor: u32,
        position: usize,
        n_draft: usize,
        p_min: f32,
        pool: &ComputePool,
    ) -> Result<DsparkDraft, String> {
        let config = &self.weights.config;
        let n_block = if config.sample_from_anchor {
            n_draft
        } else {
            n_draft + 1
        };
        if n_block > config.block_size {
            return Err(format!(
                "DFlash draft block {n_block} exceeds the trained block size {}",
                config.block_size
            ));
        }
        if position + n_block > self.kv.capacity {
            return Err(format!(
                "DFlash draft position {} + block {n_block} exceeds the cache capacity {}",
                position, self.kv.capacity
            ));
        }

        // The noise block: the anchor followed by masks.
        let mut tokens = vec![self.mask_token; n_block];
        tokens[0] = anchor;
        let positions: Vec<usize> = (position..position + n_block).collect();
        decode_block(
            &self.weights,
            &self.token_embd,
            &tokens,
            &positions,
            &mut self.kv,
            &mut self.scratch,
            pool,
        )?;

        let n_vocab = self.scratch.n_vocab;
        let n_embd = self.scratch.n_embd;
        // With `sample_from_anchor` every slot — the anchor's included — is a
        // prediction slot; without it slot 0 is a bonus token and the
        // predictions start at slot 1.
        let i_draft_beg = if config.sample_from_anchor { 0 } else { 1 };
        let markov_w1 = self
            .weights
            .markov_w1
            .as_ref()
            .ok_or_else(|| "DFlash draft (no Markov head) is not supported here".to_string())?;
        let markov_w2 = self
            .weights
            .markov_w2
            .as_ref()
            .ok_or_else(|| "DSpark draft is missing markov_w2".to_string())?;
        let has_conf = self.weights.conf_proj.is_some();

        let mut draft = DsparkDraft::default();
        // The Markov chain's first state: the anchor.
        let mut prev = anchor as usize;
        let debug = std::env::var("RMI_DFLASH_DEBUG").is_ok();
        for slot in i_draft_beg..n_block {
            let w1_prev = markov_w1.row(prev)?;
            if debug {
                eprintln!(
                    "dflash[{slot}]: w1_prev[0..4] = {:?} (prev={prev})",
                    &w1_prev[..4.min(w1_prev.len())]
                );
            }

            // bias = markov_w2 @ markov_w1[prev], scattered into the
            // sampler's vocabulary (zero outside the draft's rows).
            let bias = &mut self.scratch.markov_bias[..n_vocab];
            match &self.weights.d2t {
                Some(map) => {
                    bias.fill(0.0);
                    let n_draft_vocab = self.scratch.n_vocab_draft;
                    self.scratch.prepared.prepare(
                        &w1_prev,
                        1,
                        markov_w2.n_in,
                        markov_w2.needs_q8_0_activation(),
                        markov_w2.uses_q8_k(),
                    )?;
                    let raw = &mut self.scratch.head[..n_draft_vocab];
                    self.scratch
                        .prepared
                        .matmul(markov_w2, &w1_prev, raw, pool)?;
                    for (draft_row, value) in raw.iter().enumerate() {
                        let target = map[draft_row] as usize;
                        if target < n_vocab {
                            bias[target] = *value;
                        }
                    }
                }
                None => {
                    if self.scratch.n_vocab_draft != n_vocab {
                        return Err(
                            "DSpark draft has no d2t but a reduced head vocabulary".to_string()
                        );
                    }
                    self.scratch.prepared.prepare(
                        &w1_prev,
                        1,
                        markov_w2.n_in,
                        markov_w2.needs_q8_0_activation(),
                        markov_w2.uses_q8_k(),
                    )?;
                    self.scratch
                        .prepared
                        .matmul(markov_w2, &w1_prev, &mut bias[..n_vocab], pool)?;
                }
            }

            // The confidence head: sigmoid(conf_proj . [hidden; markov] + b).
            let mut confidence = 1.0f32;
            if has_conf {
                let conf_proj = self.weights.conf_proj.as_ref().unwrap();
                let feat = &mut self.scratch.conf_feat[..n_embd + markov_w1.rank];
                feat[..n_embd]
                    .copy_from_slice(&self.scratch.hidden[slot * n_embd..(slot + 1) * n_embd]);
                feat[n_embd..].copy_from_slice(&w1_prev);
                let mut conf_out = [0.0f32; 1];
                self.scratch.prepared.prepare(
                    feat,
                    1,
                    conf_proj.n_in,
                    conf_proj.needs_q8_0_activation(),
                    conf_proj.uses_q8_k(),
                )?;
                self.scratch
                    .prepared
                    .matmul(conf_proj, feat, &mut conf_out, pool)?;
                confidence = 1.0 / (1.0 + (-conf_out[0]
                    + self.weights.conf_proj_bias.first().copied().unwrap_or(0.0))
                    .exp());
                if debug {
                    eprintln!(
                        "dflash[{slot}]: hidden[0..4] = {:?} conf_out = {conf_out:?} bias = {:?}",
                        &self.scratch.hidden[slot * n_embd..slot * n_embd + 4],
                        self.weights.conf_proj_bias.as_slice()
                    );
                }
            }

            // Truncate below the confidence threshold, before the pick.
            if p_min > 0.0 && confidence < p_min {
                break;
            }

            // The greedy pick over the biased logits — llama.cpp samples
            // through a top-k=10 chain and takes the first candidate, which
            // is the argmax.
            let base = &self.scratch.logits[slot * n_vocab..(slot + 1) * n_vocab];
            let mut best = 0usize;
            let mut best_value = f32::NEG_INFINITY;
            for (id, (logit, bias)) in base.iter().zip(bias.iter()).enumerate() {
                let value = logit + bias;
                if value > best_value {
                    best_value = value;
                    best = id;
                }
            }
            draft.tokens.push(best as u32);
            draft.confidences.push(confidence);
            prev = best;
        }
        Ok(draft)
    }
}
