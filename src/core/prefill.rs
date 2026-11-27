//! Prefill batching abstraction.
//!
//! Decoding is "one token in, one token out" — every token gets its own
//! attention pass over the full KV cache. Prefill is fundamentally a
//! different shape: a chunk of `B` tokens comes in together, the chunk's
//! rows can share most of the FFN matmul / RMSNorm / KV-cache-append
//! work, and the attention computation for each row only has to look at
//! `prev_seq_len + row` cached positions (causal mask).
//!
//! In the limit `B = 1` this degenerates to the legacy "one token at a
//! time" decode loop, which is what every trunk implemented before the
//! batched path was introduced. So a chunked prefill is a *generalisation*
//! of the existing path, not a replacement — trunks that still only
//! know how to process one row at a time can implement
//! [`ChunkedPrefill::forward_chunk`] by looping with `B = rows`.
//!
//! Each trunk decides for itself how to amortise the inner work; this
//! module only owns the **dispatch skeleton**:
//!
//! 1. Split the input range into chunks of `batch_size` rows
//!    ([`prefill_chunks`] / [`checked_prefill_batch_size`]).
//! 2. For each chunk, call [`ChunkedPrefill::forward_chunk`], passing
//!    the chunk's `rows` and the absolute `base_position` so the trunk
//!    can write KV entries at the right slot.
//! 3. Only project logits for the **last** chunk's last row.
//! 4. Advance [`ChunkedPrefill::seq_len`] by the chunk's row count.
//!
//! Concrete trunks (qwen3, gemma4, …) keep their own scratch, KV cache
//! type, and mathematical kernels; the trait is deliberately
//! `forward_chunk(rows, base, project_logits)` rather than something
//! richer so trunks don't have to bend their internal data layout to
//! fit an abstraction.

use std::ops::Range;

pub const DEFAULT_PREFILL_BATCH_SIZE: usize = 64;

/// Validate a user-supplied (or default) prefill batch size.
///
/// Returns `Err` for `0` (would cause an infinite chunk loop), otherwise
/// passes the value through unchanged so callers can use `1` as the
/// "process one token at a time" fallback.
pub fn checked_prefill_batch_size(value: Option<usize>) -> Result<usize, String> {
    match value.unwrap_or(DEFAULT_PREFILL_BATCH_SIZE) {
        0 => Err("prefill batch size must be at least 1".into()),
        value => Ok(value),
    }
}

/// Yield contiguous `[start..end)` ranges covering `[0, len)`, in
/// `batch_size`-sized chunks (the last chunk may be shorter).
///
/// `batch_size == 1` reduces to `0..1, 1..2, 2..3, …` — the legacy
/// per-token loop.
pub(crate) fn prefill_chunks(len: usize, batch_size: usize) -> impl Iterator<Item = Range<usize>> {
    (0..len)
        .step_by(batch_size)
        .map(move |start| start..(start + batch_size).min(len))
}

/// Dispatch skeleton for trunks that can process a contiguous chunk of
/// `rows` prefill tokens at consecutive positions.
///
/// Implementations are responsible for:
/// - looking up embeddings for each row in the chunk,
/// - running every layer for every row (RMSNorm → QKV matmul → optional
///   q/k norm → RoPE → KV cache append → causal attention → wo → FFN
///   → residual),
/// - appending per-row K/V into the trunk's KV cache at
///   `base_position + row` (causal),
/// - *only when* `project_logits` is true and the chunk is the final
///   one, projecting the last row's final hidden state through
///   output_norm + LM head and returning `Some(logits)`.
///
/// When the chunk is **not** the final one (or `project_logits` is
/// explicitly false), the implementation should return `Ok(None)` —
/// the trunk only pays for the LM-head projection once per prefill.
pub trait ChunkedPrefill {
    /// Input bundle that carries the tokens (and any per-row metadata,
    /// embeddings, mrope positions, deepstack hooks, …) for the
    /// whole prefill pass. Implementations decide what this is — a
    /// bare `&[u32]` for the simplest token-only trunks, a richer
    /// `Qwen3Input<'a>` for Qwen3, etc.
    type Input;

    /// Total number of prefill rows the input represents. The default
    /// [`prefill`](Self::prefill) uses this to drive the chunk loop and
    /// to decide which chunk is the *final* one (and therefore the one
    /// that pays for the LM-head projection).
    fn input_len(input: &Self::Input) -> usize;

    /// Forward `rows` tokens of `input` whose absolute positions are
    /// `[base_position, base_position + rows)`.
    ///
    /// `project_logits` is `true` only on the final chunk of the final
    /// prefill pass (so LM-head cost is paid exactly once). On every
    /// other call it is `false`.
    ///
    /// When `project_logits` is true the implementation should return
    /// `Some(logits)`; otherwise `Ok(None)`.
    fn forward_chunk(
        &mut self,
        input: &Self::Input,
        rows: usize,
        base_position: usize,
        project_logits: bool,
    ) -> Result<Option<Vec<f32>>, String>;

    /// Largest chunk the trunk's scratch can hold. A chunk larger than
    /// this returns an error rather than silently overflowing.
    fn max_chunk_size(&self) -> usize;

    /// Current sequence length (number of tokens already cached in KV).
    /// The default [`prefill`](Self::prefill) implementation reads this
    /// to compute `base_position` for each chunk.
    fn seq_len(&self) -> usize;

    /// Set the sequence length. Called by the default
    /// [`prefill`](Self::prefill) implementation after each chunk.
    fn set_seq_len(&mut self, len: usize);

    /// Convenience: prefill `input` with the given `batch_size`. Walks
    /// [`prefill_chunks`] and forwards each chunk through
    /// [`forward_chunk`](Self::forward_chunk). The default
    /// implementation lives in the trait so every trunk gets the same
    /// chunking semantics; trunks are free to override if they need
    /// finer control (e.g. qwen3 captures per-layer intermediates for
    /// DSpark on the final chunk, gemma4 owns its own GPU/CPU
    /// dispatch).
    fn prefill(
        &mut self,
        input: &Self::Input,
        batch_size: usize,
    ) -> Result<Option<Vec<f32>>, String> {
        let batch_size = checked_prefill_batch_size(Some(batch_size))?;
        let total_tokens = Self::input_len(input);
        if total_tokens == 0 {
            return Ok(None);
        }
        let mut last_logits: Option<Vec<f32>> = None;
        for chunk in prefill_chunks(total_tokens, batch_size) {
            let rows = chunk.len();
            let max = self.max_chunk_size();
            if rows > max {
                return Err(format!("prefill chunk rows {rows} exceed trunk max {max}"));
            }
            let base = self.seq_len();
            let is_last = chunk.end == total_tokens;
            last_logits = self.forward_chunk(input, rows, base, is_last)?;
            self.set_seq_len(base + rows);
        }
        Ok(last_logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_batch_size_defaults_to_64_and_rejects_zero() {
        assert_eq!(checked_prefill_batch_size(None).unwrap(), 64);
        assert_eq!(checked_prefill_batch_size(Some(1)).unwrap(), 1);
        assert_eq!(checked_prefill_batch_size(Some(128)).unwrap(), 128);
        let zero_err = checked_prefill_batch_size(Some(0)).unwrap_err();
        assert!(
            zero_err.contains("at least 1"),
            "unexpected error message: {zero_err}"
        );
    }

    #[test]
    fn prefill_chunks_cover_input_without_padding() {
        let ranges = prefill_chunks(130, 64).collect::<Vec<_>>();
        assert_eq!(ranges, [0..64, 64..128, 128..130]);
    }

    #[test]
    fn prefill_chunks_with_batch_one_is_per_token() {
        let ranges = prefill_chunks(4, 1).collect::<Vec<_>>();
        assert_eq!(ranges, [0..1, 1..2, 2..3, 3..4]);
    }

    #[test]
    fn prefill_chunks_with_batch_larger_than_input_yields_single_chunk() {
        let ranges = prefill_chunks(3, 16).collect::<Vec<_>>();
        assert_eq!(ranges, [0..3]);
    }

    // -- Smoke test: a toy ChunkedPrefill implementation that just
    //    accumulates the row indices and the chunk-bounded "logits"
    //    payload — exercises the trait's default `prefill` loop and the
    //    `project_logits` argument only firing on the final chunk.

    struct ToyChunked {
        seq_len: usize,
        max_rows: usize,
        last_logits: Option<Vec<f32>>,
        seen_rows: Vec<usize>,
    }

    impl ToyChunked {
        fn new(max_rows: usize) -> Self {
            Self {
                seq_len: 0,
                max_rows,
                last_logits: None,
                seen_rows: Vec::new(),
            }
        }
    }

    impl ChunkedPrefill for ToyChunked {
        type Input = Vec<u32>;

        fn input_len(input: &Self::Input) -> usize {
            input.len()
        }

        fn forward_chunk(
            &mut self,
            _input: &Self::Input,
            rows: usize,
            base_position: usize,
            project_logits: bool,
        ) -> Result<Option<Vec<f32>>, String> {
            if rows > self.max_rows {
                return Err(format!("toy chunk rows {rows} > max {}", self.max_rows));
            }
            for offset in 0..rows {
                self.seen_rows.push(base_position + offset);
            }
            if project_logits {
                let logits: Vec<f32> = (0..rows).map(|r| (base_position + r) as f32).collect();
                self.last_logits = Some(logits.clone());
                Ok(Some(logits))
            } else {
                Ok(None)
            }
        }

        fn max_chunk_size(&self) -> usize {
            self.max_rows
        }

        fn seq_len(&self) -> usize {
            self.seq_len
        }

        fn set_seq_len(&mut self, len: usize) {
            self.seq_len = len;
        }
    }

    fn toy_input() -> Vec<u32> {
        (0..10u32).collect()
    }

    #[test]
    fn chunked_prefill_default_loop_pays_logits_only_for_final_chunk() {
        let mut toy = ToyChunked::new(3);
        let logits = toy.prefill(&toy_input(), 3).unwrap().expect("final logits");
        assert_eq!(toy.seen_rows, (0..10).collect::<Vec<_>>());
        assert_eq!(toy.seq_len, 10);
        // The toy encodes base_position+r into logits[r]; the final
        // chunk spans rows 9..10 so the chunk's first row projects
        // base=9 → 9.0.
        assert_eq!(logits, vec![9.0]);
    }

    #[test]
    fn chunked_prefill_default_loop_rejects_oversized_chunk() {
        let mut toy = ToyChunked::new(2);
        let err = toy.prefill(&toy_input(), 3).unwrap_err();
        assert!(err.contains("exceed"), "unexpected error message: {err}");
    }

    #[test]
    fn chunked_prefill_with_zero_tokens_is_a_noop() {
        let mut toy = ToyChunked::new(4);
        let empty: Vec<u32> = Vec::new();
        let logits = toy.prefill(&empty, 4).unwrap();
        assert!(logits.is_none());
        assert_eq!(toy.seq_len, 0);
        assert!(toy.seen_rows.is_empty());
    }

    #[test]
    fn chunked_prefill_with_batch_one_emulates_legacy_per_token_loop() {
        let mut toy = ToyChunked::new(1);
        toy.prefill(&toy_input(), 1).unwrap();
        assert_eq!(toy.seen_rows, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(toy.seq_len, 10);
    }
}
