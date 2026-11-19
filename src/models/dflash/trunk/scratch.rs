//! The scratchpad of a DFlash / DSpark draft model.
//!
//! The draft is small (a handful of blocks) and runs one block at a time, so
//! the buffers here are sized for the block (and for one chunk of encoder
//! injection), not for a whole prefill.

use super::config::DflashConfig;
use crate::core::tensor::BlockQ8K;
use crate::ops::kernel::PreparedRows;

pub struct DflashScratchpad {
    /// The row count the buffers are sized for: the larger of the block size
    /// and one encoder-injection chunk.
    pub max_rows: usize,
    pub n_embd: usize,
    pub n_embd_q: usize,
    pub n_embd_gqa: usize,
    pub n_ff: usize,
    /// The width of the draft's own head.
    pub n_vocab_draft: usize,
    /// The width the logits are scattered into: the vocabulary the sampler
    /// reads, i.e. the target's.
    pub n_vocab: usize,
    /// The Markov head's rank (the length of one `markov_w1` row).
    pub markov_rank: usize,

    /// The residual stream.
    pub x: Vec<f32>,
    pub normed: Vec<f32>,
    /// The final-normed hidden rows the confidence head reads.
    pub hidden: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub proj: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
    /// The draft's head, before the vocabulary scatter.
    pub head: Vec<f32>,
    /// The base logits, after the scatter: one row per block position.
    pub logits: Vec<f32>,
    /// The Markov head's bias for one block position, scattered into the
    /// sampler's vocabulary.
    pub markov_bias: Vec<f32>,
    /// The confidence head's input: one position's `[hidden; markov]`.
    pub conf_feat: Vec<f32>,

    pub q8_buf: Vec<u8>,
    pub scale_buf: Vec<f32>,
    pub q8k_buf: Vec<BlockQ8K>,
    pub(crate) prepared: PreparedRows,

    /// Per-thread attention scores, `n_threads * score_stride` in total.
    pub scores: Vec<f32>,
    pub score_stride: usize,
}

impl DflashScratchpad {
    /// Build the scratchpad. `capacity` is the KV cache capacity (tokens);
    /// the score rows are padded to 256 so the softmax tail is aligned.
    pub fn new(
        config: &DflashConfig,
        n_vocab: usize,
        max_rows: usize,
        n_threads: usize,
        capacity: usize,
    ) -> Self {
        let max_rows = max_rows.max(1);
        let n_embd = config.n_embd;
        let n_embd_q = config.n_head * config.n_embd_head_k;
        let n_embd_gqa = config.n_head_kv * config.n_embd_head_k;
        let n_ff = config.n_ff;
        let n_vocab_draft = config.n_vocab_draft;
        let markov_rank = config.markov_rank.max(1);
        // The largest width any matmul reads: the encoder's fusion input,
        // the block's projections, the head and the Markov projection, and
        // the confidence head's `[hidden; markov]` pair.
        let max_n_in = n_embd
            .max(n_embd_q)
            .max(n_ff)
            .max(config.target_layers.len() * n_embd)
            .max(markov_rank)
            .max(n_embd + config.markov_rank);
        let score_stride = capacity.div_ceil(256) * 256;
        let zero_q8k = BlockQ8K {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16],
        };
        Self {
            max_rows,
            n_embd,
            n_embd_q,
            n_embd_gqa,
            n_ff,
            n_vocab_draft,
            n_vocab,
            markov_rank,
            x: vec![0.0f32; n_embd * max_rows],
            normed: vec![0.0f32; n_embd * max_rows],
            hidden: vec![0.0f32; n_embd * max_rows],
            q: vec![0.0f32; n_embd_q * max_rows],
            k: vec![0.0f32; n_embd_gqa * max_rows],
            v: vec![0.0f32; n_embd_gqa * max_rows],
            attn_out: vec![0.0f32; n_embd_q * max_rows],
            proj: vec![0.0f32; n_embd * max_rows],
            gate: vec![0.0f32; n_ff * max_rows],
            up: vec![0.0f32; n_ff * max_rows],
            down: vec![0.0f32; n_embd * max_rows],
            head: vec![0.0f32; n_vocab_draft * max_rows],
            logits: vec![0.0f32; n_vocab * max_rows],
            markov_bias: vec![0.0f32; n_vocab],
            conf_feat: vec![0.0f32; n_embd + markov_rank],
            q8_buf: vec![0u8; max_n_in * max_rows],
            scale_buf: vec![0.0f32; max_n_in.div_ceil(32) * max_rows],
            q8k_buf: vec![zero_q8k; max_n_in.div_ceil(256) * max_rows],
            prepared: PreparedRows::new(max_rows, max_n_in),
            scores: vec![0.0f32; n_threads.max(1) * score_stride],
            score_stride,
        }
    }
}
