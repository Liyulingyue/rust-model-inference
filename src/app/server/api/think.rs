/// Hold back a leading reasoning block so it never reaches the client.
///
/// Reasoning models (`LFM2.5-Thinking` and friends) sometimes start thinking
/// even when the request asked for a direct answer, emitting
/// `<|think|>\n…\n<|/think|>\n` before the real answer. Unlike
/// [`super::stop::StopFilter`] this is a *leading-only* filter: it only acts
/// while the block is still at the very start of the output, so a body that
/// merely mentions ` reasoning` later is untouched.
///
/// Latency notes: text is only buffered while it could still turn into the
/// opener. Anything that cannot be an opener prefix is flushed immediately,
/// so the common case (model answers directly) adds no first-token delay.
/// An opener that never closes is dropped entirely — it is reasoning that
/// ran to the end of generation without producing an answer.
pub struct ThinkFilter {
    pending: String,
    /// None until the opener has been confirmed; Some(true) inside a block,
    /// Some(false) once no block is coming and we are in passthrough.
    state: Option<bool>,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

impl ThinkFilter {
    pub fn new() -> Self {
        Self {
            pending: String::new(),
            state: None,
        }
    }

    pub fn push(&mut self, text: &str) -> String {
        // Passthrough: no leading reasoning block.
        if self.state == Some(false) {
            return text.to_string();
        }
        self.pending.push_str(text);
        // State still unknown: decide as soon as we can.
        if self.state.is_none() {
            if let Some(rest) = self.pending.strip_prefix(THINK_OPEN) {
                self.pending = rest.to_string();
                self.state = Some(true);
            } else if THINK_OPEN.starts_with(&self.pending) {
                // Could still become the opener once more text arrives.
                return String::new();
            } else {
                // Cannot be an opener: never will strip anything.
                self.state = Some(false);
                return std::mem::take(&mut self.pending);
            }
        }
        // Inside a block: wait for the closer, emit what follows it.
        if self.state == Some(true) {
            if let Some(offset) = self.pending.find(THINK_CLOSE) {
                let output = self.pending[offset + THINK_CLOSE.len()..].to_string();
                self.pending.clear();
                self.state = Some(false);
                return output;
            }
            // Keep only a possible suffix of the closer; drop reasoning text.
            let keep = closer_prefix_keep(&self.pending);
            self.pending = self.pending.split_off(self.pending.len() - keep);
            return String::new();
        }
        String::new()
    }

    /// Flush at end of generation. Returns anything still worth emitting;
    /// an unterminated reasoning block is discarded.
    pub fn finish(&mut self) -> String {
        if self.state == Some(true) {
            self.state = Some(false);
            self.pending.clear();
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }
}

impl Default for ThinkFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Longest suffix of `pending` that is also a non-empty prefix of the closer,
/// so a closer split across chunks is still matched.
///
/// `pending` is a `String` of arbitrary UTF-8 text — the closer ``
/// itself is ASCII, but the surrounding text may carry multi-byte code
/// points (e.g. Chinese). `THINK_CLOSE.len()` is a byte length, so we
/// must step through `pending`'s *chars* and snap each candidate offset
/// to the nearest char boundary; otherwise `pending[..len]` panics when
/// the byte index lands inside a multi-byte sequence (e.g. ``用``,
/// bytes 0..3 — the suffix ``len=1`` would slice bytes [2..3]).
fn closer_prefix_keep(pending: &str) -> usize {
    let close_chars = THINK_CLOSE.chars().count();
    (1..=close_chars)
        .map(|n| {
            // The last `n` chars of `pending`, in bytes.
            let mut byte_len = 0usize;
            let mut char_count = 0usize;
            for ch in pending.chars().rev() {
                if char_count == n {
                    break;
                }
                byte_len += ch.len_utf8();
                char_count += 1;
            }
            byte_len
        })
        .filter(|&byte_len| byte_len <= pending.len())
        .filter(|&byte_len| THINK_CLOSE.starts_with(&pending[pending.len() - byte_len..]))
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_answer_passes_through_without_buffering() {
        let mut filter = ThinkFilter::new();
        assert_eq!(filter.push("Hello"), "Hello");
        assert_eq!(filter.push(" there"), " there");
        assert_eq!(filter.finish(), "");
    }

    #[test]
    fn terminated_block_is_removed() {
        let open = THINK_OPEN;
        let close = THINK_CLOSE;
        let mut filter = ThinkFilter::new();
        let mut out = String::new();
        for chunk in [open, " secret reasoning\n", close, "real answer"] {
            out.push_str(&filter.push(chunk));
        }
        out.push_str(&filter.finish());
        assert_eq!(out, "real answer");
    }

    #[test]
    fn unterminated_block_is_dropped() {
        let open = THINK_OPEN;
        let mut filter = ThinkFilter::new();
        let mut out = String::new();
        for chunk in [open, " thinking forever", " still thinking"] {
            out.push_str(&filter.push(chunk));
        }
        out.push_str(&filter.finish());
        assert!(
            out.is_empty(),
            "unterminated reasoning must not leak: {out:?}"
        );
    }

    #[test]
    fn closer_split_across_chunks_still_matches() {
        let open = THINK_OPEN;
        let close = THINK_CLOSE;
        let mut filter = ThinkFilter::new();
        let mut out = String::new();
        // Break the closer into two halves so the split-across-chunks path runs.
        let head = &close[..close.len() - 4];
        for chunk in [open, "x", head, &close[close.len() - 4..], "answer"] {
            out.push_str(&filter.push(chunk));
        }
        out.push_str(&filter.finish());
        assert_eq!(out, "answer");
    }

    #[test]
    fn mention_of_marker_later_is_preserved() {
        let mut filter = ThinkFilter::new();
        let mut out = String::new();
        for chunk in ["Sure! ", "You said \"re", "asoning", "\" earlier."] {
            out.push_str(&filter.push(chunk));
        }
        out.push_str(&filter.finish());
        assert_eq!(out, "Sure! You said \"reasoning\" earlier.");
    }

    /// Regression: a `pending` whose tail is a multi-byte UTF-8 code point
    /// used to panic at `closer_prefix_keep` because the old code indexed
    /// `pending` by raw `len()` (byte offset) inside the closer loop. With
    /// Chinese / emoji in the reasoning text, the suffix `<` of an in-flight
    /// `` opener would never appear, but `pending.len() - 1` could land on
    /// the 2nd/3rd byte of a 3-byte CJK char and trigger
    /// `start byte index N is not a char boundary`.
    #[test]
    fn multibyte_suffix_does_not_panic() {
        let open = THINK_OPEN;
        let mut filter = ThinkFilter::new();
        // Open a block and feed multi-byte text; never close.
        let mut out = String::new();
        for chunk in [open, "用中文推理: ", "巴黎是", "法国的首都"] {
            out.push_str(&filter.push(chunk));
        }
        // The unterminated reasoning block is dropped.
        let tail = filter.finish();
        out.push_str(&tail);
        assert!(
            out.is_empty(),
            "unterminated multi-byte reasoning must not leak: {out:?}"
        );
    }

    /// Same regression at a finer grain: a 3-byte CJK char that *ends* the
    /// pending buffer (so the byte-1-of-3 offset is the boundary) plus
    /// a partially-formed closer byte after it. The old code sliced
    /// `pending[len-1..]` and tripped the boundary check.
    #[test]
    fn cjk_then_partial_closer_does_not_panic() {
        let open = THINK_OPEN;
        let mut filter = ThinkFilter::new();
        // Build pending = "<think>x用" — 3-byte CJK at the tail.
        let mut out = String::new();
        for chunk in [open, "x用"] {
            out.push_str(&filter.push(chunk));
        }
        // Now feed half of `` so the filter must keep the suffix
        // while inside the block.
        for chunk in ["</", "thin", "k>answer"] {
            out.push_str(&filter.push(chunk));
        }
        out.push_str(&filter.finish());
        assert_eq!(out, "answer");
    }
}
