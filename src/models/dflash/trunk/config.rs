//! DflashConfig — hyperparameters for a DFlash / DSpark draft model.
//!
//! A DFlash file is a *sidecar*: a small model that drafts for another one.
//! Its metadata lives under the `dflash.` prefix (the architecture name), not
//! under the target's.
//!
//! The contract is llama.cpp's (`src/models/dflash.cpp::load_arch_hparams`):
//!
//! | key | meaning |
//! |---|---|
//! | `dflash.block_count` | how many blocks the draft has |
//! | `dflash.embedding_length` | the draft's hidden size |
//! | `dflash.target_layers` | the *target* model's layers whose input hidden state the draft reads |
//! | `dflash.block_size` | how many tokens one draft block holds |
//! | `dflash.sample_from_anchor` | whether slot 0 of a block is the anchor (a prediction slot) |
//! | `dflash.has_confidence_head` | whether the confidence head is present |
//!
//! A file that also carries `markov_w1.weight` is a **DSpark** draft: the
//! Markov head biases the draft logits by the previously committed token.

use crate::core::tensor::{MetaValue, TensorSource};

#[derive(Debug, Clone)]
pub struct DflashConfig {
    /// The draft's own depth.
    pub n_layer: usize,
    /// The draft's hidden size.
    pub n_embd: usize,
    /// The draft's feed-forward width.
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_ctx: usize,
    pub norm_eps: f32,
    pub rope_freq_base: f32,
    /// How many head dimensions the rotary embedding touches.
    pub rope_dimension_count: usize,
    /// The four rotary axes (`[T, H, W, E]`), in rotary *pairs*.
    pub rope_sections: [i32; 4],
    /// The sliding window, or zero for full attention.
    pub sliding_window: usize,
    /// Which blocks use the sliding window.
    pub sliding_window_pattern: Vec<bool>,
    /// How many tokens one draft block holds.
    pub block_size: usize,
    /// The target model's layers whose input hidden state the draft reads.
    pub target_layers: Vec<usize>,
    /// Whether slot 0 of a block is the anchor *and* a prediction slot.
    ///
    /// With `sample_from_anchor` the block is `[anchor, MASK, ...]` and every
    /// slot, including the anchor's, is a draft candidate.
    pub sample_from_anchor: bool,
    /// Whether the file carries the confidence head.
    pub has_confidence_head: bool,
    /// The *target's* vocabulary size: the Markov head indexes it.
    pub n_vocab: usize,
    /// The draft's own vocabulary, which may be a reduced one.
    pub n_vocab_draft: usize,
    /// The Markov head's rank.
    pub markov_rank: usize,
    /// The mask token, in the target's vocabulary: the noise block's filler.
    pub mask_token: u32,
}

impl DflashConfig {
    /// The number of rotary pairs the four axes share.
    pub fn rope_pairs(&self) -> usize {
        self.rope_dimension_count / 2
    }

    /// Whether the draft's attention sees the whole context or a window.
    pub fn is_sliding(&self) -> bool {
        self.sliding_window > 0
    }

    /// Whether block `il` uses the sliding window.
    pub fn layer_is_sliding(&self, il: usize) -> bool {
        if !self.is_sliding() {
            return false;
        }
        match self.sliding_window_pattern.get(il) {
            Some(&flag) => flag,
            // llama.cpp's `set_swa_pattern` marks every layer when the file
            // carries no pattern at all.
            None => true,
        }
    }
}

fn u64_of(value: Option<&MetaValue>) -> Option<u64> {
    value.and_then(|value| value.to_u64())
}

fn read_u64_array<S: TensorSource + ?Sized>(source: &S, key: &str) -> Option<Vec<u64>> {
    let values = source.metadata(key)?.to_arr()?;
    Some(values.iter().filter_map(|value| value.to_u64()).collect())
}

impl DflashConfig {
    /// Read the draft's hyperparameters out of its GGUF metadata.
    ///
    /// `n_vocab` (the target's) and `n_vocab_draft` come from the Markov
    /// head's two matrices rather than from metadata: `markov_w1` is indexed
    /// by target token id and `markov_w2` by draft row.
    pub fn from_source<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let prefix = "dflash";
        let get = |key: &str| -> Result<u64, String> {
            u64_of(source.metadata(&format!("{prefix}.{key}")))
                .ok_or_else(|| format!("Missing {prefix} metadata: {key}"))
        };

        let n_layer = get("block_count")? as usize;
        let n_embd = get("embedding_length")? as usize;
        let n_ff = get("feed_forward_length")? as usize;
        let n_head = get("attention.head_count")? as usize;
        let n_head_kv = get("attention.head_count_kv")? as usize;
        let n_embd_head_k = get("attention.key_length")? as usize;
        let n_embd_head_v = get("attention.value_length")? as usize;
        let n_ctx = get("context_length").unwrap_or(2048) as usize;

        // Floating-point metadata (F64 in these files) must go through
        // `to_f64`: `to_u64` truncates `1e-6` to `0`, and reading the bits of
        // that as an f32 would make the norm epsilon zero — which turns any
        // all-zero row into `0/sqrt(0)` = NaN.
        let norm_eps = source
            .metadata(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .and_then(|value| value.to_f64())
            .map(|value| value as f32)
            .unwrap_or(1e-6);
        let rope_freq_base = source
            .metadata(&format!("{prefix}.rope.freq_base"))
            .and_then(|value| value.to_f64())
            .map(|value| value as f32)
            .unwrap_or(10_000.0);
        let rope_dimension_count =
            u64_of(source.metadata(&format!("{prefix}.rope.dimension_count")))
                .unwrap_or(n_embd_head_k as u64) as usize;

        // The rotary axes are given in pairs and must add up to half of the
        // rotated dimensions. A file with no sections at all is a plain
        // (non-multimodal) draft.
        let rope_sections =
            match read_u64_array(source, &format!("{prefix}.rope.dimension_sections")) {
                Some(values) if values.len() == 4 => [
                    values[0] as i32,
                    values[1] as i32,
                    values[2] as i32,
                    values[3] as i32,
                ],
                _ => {
                    let pairs = (rope_dimension_count / 2) as i32;
                    [pairs, 0, 0, 0]
                }
            };

        let sliding_window = u64_of(source.metadata(&format!("{prefix}.attention.sliding_window")))
            .unwrap_or(0) as usize;
        let sliding_window_pattern = read_u64_array(
            source,
            &format!("{prefix}.attention.sliding_window_pattern"),
        )
        .map(|values| values.iter().map(|&value| value != 0).collect())
        .unwrap_or_default();

        let block_size = get("block_size")? as usize;
        let target_layers = read_u64_array(source, &format!("{prefix}.target_layers"))
            .filter(|values| !values.is_empty())
            .map(|values| values.iter().map(|&value| value as usize).collect())
            .ok_or_else(|| "DFlash draft requires 'dflash.target_layers'".to_string())?;

        let sample_from_anchor = match source.metadata(&format!("{prefix}.sample_from_anchor")) {
            Some(MetaValue::Bool(value)) => *value,
            Some(MetaValue::String(value)) => value == "true",
            _ => true,
        };
        // The mask token: the noise block is [anchor, MASK, ...]. It is an id
        // in the target's vocabulary, whose embedding table the draft shares.
        let mask_token =
            u64_of(source.metadata("tokenizer.ggml.mask_token_id")).ok_or_else(|| {
                "DFlash draft requires 'tokenizer.ggml.mask_token_id' in its GGUF".to_string()
            })? as u32;
        let has_confidence_head = match source.metadata(&format!("{prefix}.has_confidence_head")) {
            Some(MetaValue::Bool(value)) => *value,
            Some(MetaValue::String(value)) => value == "true",
            _ => false,
        };

        if n_layer == 0 || n_embd == 0 || n_head == 0 || n_head_kv == 0 {
            return Err("DFlash draft has an empty shape".to_string());
        }
        if n_head % n_head_kv != 0 {
            return Err(format!(
                "DFlash draft head count {n_head} is not a multiple of the kv head count {n_head_kv}"
            ));
        }
        if block_size == 0 {
            return Err("DFlash draft requires a positive block size".to_string());
        }

        // The vocabulary sizes live in the Markov head's matrices.
        let (n_vocab, n_vocab_draft, markov_rank) = match source.tensor_info("markov_w1.weight") {
            Some(info) => {
                let rank = info.dims.first().copied().unwrap_or(0) as usize;
                let target_vocab = info.dims.get(1).copied().unwrap_or(0) as usize;
                let draft_vocab = source
                    .tensor_info("markov_w2.weight")
                    .and_then(|info| info.dims.get(1).copied())
                    .unwrap_or(target_vocab as u64) as usize;
                (target_vocab, draft_vocab, rank)
            }
            // A plain DFlash draft (no Markov head) projects onto the target's
            // vocabulary directly.
            None => (0, 0, 0),
        };

        Ok(Self {
            n_layer,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            n_embd_head_k,
            n_embd_head_v,
            n_ctx,
            norm_eps,
            rope_freq_base,
            rope_dimension_count,
            rope_sections,
            sliding_window,
            sliding_window_pattern,
            block_size,
            target_layers,
            sample_from_anchor,
            has_confidence_head,
            n_vocab,
            n_vocab_draft,
            markov_rank,
            mask_token,
        })
    }
}
