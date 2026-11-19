//! The n-gram self-speculative draft (llama.cpp's `ngram-simple`).
//!
//! The draft needs no second model and no weights: it looks up the last
//! `n` tokens (history tail + the token the target just sampled) in the
//! committed token history and proposes the `m` tokens that followed a
//! previous occurrence. The target then verifies the proposal with the
//! same batched-greedy machinery the MTP and DSpark paths use, so the
//! output is always the greedy output — the n-gram only changes how many
//! forward passes the tokens take.
//!
//! This is a port of `common_ngram_simple_draft` from the pinned
//! llama.cpp `b96806d` (`common/ngram-map.cpp`), including its
//! boundaries: the search skips position 0 (a match at position 0 is
//! treated as no match, because 0 is that function's "not found"
//! sentinel), the search runs backwards from the occurrence before the
//! current one, and a match whose remaining history is shorter than the
//! lookup size produces no draft.

/// Upstream's default lookup n-gram size (`--spec-ngram-simple-size-n`,
/// default 12).
pub const NGRAM_SIMPLE_SIZE_N: usize = 12;

/// Upstream's default draft m-gram size (`--spec-ngram-simple-size-m`,
/// default 48).
pub const NGRAM_SIMPLE_SIZE_M: usize = 48;

/// The `ngram-simple` configuration: `size_n` is the lookup n-gram size,
/// `size_m` the draft m-gram size.
#[derive(Clone, Copy, Debug)]
pub struct NgramSimpleConfig {
    pub size_n: usize,
    pub size_m: usize,
}

impl Default for NgramSimpleConfig {
    fn default() -> Self {
        Self {
            size_n: NGRAM_SIMPLE_SIZE_N,
            size_m: NGRAM_SIMPLE_SIZE_M,
        }
    }
}

/// Draft tokens from the committed history, the `ngram-simple` contract.
///
/// `history` is every committed token (the prompt plus the generated
/// tokens so far) **without** the just-sampled `token`; `token` is the
/// bonus token the target already sampled and the round is about to
/// commit. The lookup pattern is the last `size_n` tokens ending with
/// `token`; the draft is the `size_m` tokens that followed the nearest
/// previous occurrence of that pattern.
pub fn ngram_simple_draft(config: &NgramSimpleConfig, history: &[u32], token: u32) -> Vec<u32> {
    let cur_len = history.len();
    let n = config.size_n;
    let m = config.size_m;
    // Upstream: "we need at least n + m + 1 tokens". A zero lookup size
    // would slice out of bounds below (upstream tolerates it; the CLI and
    // the runtime gate both require at least 1), so it drafts nothing here
    // instead of panicking.
    if n == 0 || cur_len <= n + m + 1 {
        return Vec::new();
    }

    // The pattern: the history's last n-1 tokens plus the sampled token.
    let pattern = &history[cur_len - n + 1..];
    let pattern_matches = |start: usize| -> bool {
        history[start..start + n - 1] == pattern[..n - 1] && history[start + n - 1] == token
    };

    // Search backwards, skipping the occurrence the round is currently
    // sitting on; position 0 is the "not found" sentinel, so a match at
    // position 0 is not found (upstream's contract).
    let mut match_pos = 0usize;
    let mut j = cur_len - n - 1;
    while j > 0 {
        if pattern_matches(j) {
            match_pos = j;
            break;
        }
        j -= 1;
    }
    if match_pos == 0 {
        return Vec::new();
    }

    // The draft: the tokens that followed the matched occurrence. A match
    // with less than `n` tokens behind it drafts nothing (upstream's
    // `copy_max < n_draft_min` guard).
    let copy_max = m.min(cur_len - (match_pos + n));
    if copy_max < n {
        return Vec::new();
    }
    history[match_pos + n..match_pos + n + copy_max].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(n: usize, m: usize) -> NgramSimpleConfig {
        NgramSimpleConfig {
            size_n: n,
            size_m: m,
        }
    }

    #[test]
    fn drafts_the_tokens_that_followed_a_previous_occurrence() {
        // The pattern is the last n-1 history tokens plus the sampled one:
        // [1, 2, 4] over history [1,2,3,4,5,6,7,8,9,2,3,4,10,11,2,3] with
        // sampled 4 occurs at index 1 and at index 9; the backwards search
        // takes the latest previous occurrence (9), so the draft is what
        // followed *that* one: [10, 11, 2].
        let history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 10, 11, 2, 3];
        let draft = ngram_simple_draft(&config(3, 3), &history, 4);
        assert_eq!(draft, vec![10, 11, 2]);
    }

    #[test]
    fn no_previous_occurrence_drafts_nothing() {
        let history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let draft = ngram_simple_draft(&config(3, 4), &history, 12);
        assert!(draft.is_empty());
    }

    #[test]
    fn a_match_only_at_position_zero_is_not_found() {
        // [1, 2, 3] occurs only at index 0 — upstream's "not found"
        // sentinel — so the draft is empty even though a match exists.
        let history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1, 2];
        let draft = ngram_simple_draft(&config(3, 3), &history, 3);
        assert!(draft.is_empty());
    }

    #[test]
    fn too_short_history_drafts_nothing() {
        // Upstream needs cur_len > n + m + 1; a history exactly that long
        // (8 = 3 + 4 + 1) is still too short.
        let history = [1u32; 8];
        assert!(ngram_simple_draft(&config(3, 4), &history, 1).is_empty());
        // One token longer is enough for the length guard, and this
        // history has a match with room after it: pattern [2, 2, 2] occurs
        // at index 1, whose followers are [5, 5, 5, 2].
        let history = [9, 2, 2, 2, 5, 5, 5, 2, 2];
        let draft = ngram_simple_draft(&config(3, 4), &history, 2);
        assert_eq!(draft, vec![5, 5, 5, 2]);
    }

    #[test]
    fn a_match_too_close_to_the_end_drafts_nothing() {
        // Pattern [9, 9, 9]: the latest previous occurrence (index 9) has
        // only one token behind it, so upstream's `copy_max < size_n`
        // guard fires even though a match exists — the earlier occurrence
        // at index 3 is passed over because the search takes the latest.
        let history = [1, 9, 9, 9, 9, 9, 4, 5, 9, 9, 9, 9, 9];
        assert!(ngram_simple_draft(&config(3, 8), &history, 9).is_empty());
        // The same history with the late occurrence removed: the match at
        // index 3 has 7 followers, all of which are drafted.
        let history = [1, 9, 9, 9, 9, 9, 4, 5, 6, 7, 8, 9, 9];
        let draft = ngram_simple_draft(&config(3, 8), &history, 9);
        assert_eq!(draft, vec![4, 5, 6, 7, 8, 9, 9]);
    }

    #[test]
    fn the_search_skips_the_current_occurrence() {
        // Pattern [1, 2, 9] occurs at index 1 and at index 6; the current
        // occurrence (the one the round sits on) would be at index 11,
        // past history's end. The draft comes from index 6, the latest
        // previous one — not from the earlier occurrence at index 1
        // (whose followers [5, 5, 1, 2] it must not draft) and not from
        // the current tail.
        let history = [6, 1, 2, 9, 5, 5, 1, 2, 9, 8, 7, 1, 2];
        let draft = ngram_simple_draft(&config(3, 4), &history, 9);
        assert_eq!(draft, vec![8, 7, 1, 2]);
    }

    #[test]
    fn the_draft_is_capped_by_the_remaining_history() {
        // The match at index 5 has only 6 tokens behind it, so the draft
        // takes all 6 even though size_m is 8.
        let history = [1, 2, 3, 4, 5, 1, 2, 3, 6, 7, 8, 9, 1, 2];
        let draft = ngram_simple_draft(&config(3, 8), &history, 3);
        assert_eq!(draft, vec![6, 7, 8, 9, 1, 2]);
    }

    #[test]
    fn upstream_defaults_need_a_long_history() {
        // n = 12, m = 48 (upstream defaults): the guard needs 62 tokens.
        let history = [42u32; 61];
        assert!(ngram_simple_draft(&NgramSimpleConfig::default(), &history, 42).is_empty());
        // At 62 tokens the guard passes; a [100; 12] pattern at index 1
        // drafts 48 tokens: the 38 filler tokens and the first 10 of the
        // trailing run.
        let mut history = vec![9u32];
        history.extend(std::iter::repeat(100u32).take(12));
        history.extend(std::iter::repeat(7u32).take(38));
        history.extend(std::iter::repeat(100u32).take(11));
        let draft = ngram_simple_draft(&NgramSimpleConfig::default(), &history, 100);
        assert_eq!(draft.len(), 48);
        assert!(draft[..38].iter().all(|&t| t == 7));
        assert!(draft[38..].iter().all(|&t| t == 100));
    }

    #[test]
    fn a_zero_lookup_size_drafts_nothing() {
        // Degenerate config the entries never produce: no panic, no draft.
        let history = [1, 2, 3, 4, 5, 6];
        assert!(ngram_simple_draft(&config(0, 4), &history, 3).is_empty());
    }
}
