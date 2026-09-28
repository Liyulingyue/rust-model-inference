/// Hold back a leading reasoning block so it never reaches the client.
///
/// Reasoning models (`LFM2.5-Thinking` and friends) sometimes start thinking
/// even when the request asked for a direct answer, emitting
/// `<|think|>\n…\n<|/think|>\n` before the real answer. Unlike
/// [`super::stop::StopFilter`] this is a *leading-only* filter: it only acts
/// while the block is still at the very start of the output, so a body that
/// merely mentions ` reasoning` later is untouched.
///
/// **Out-of-order variants.** LFM2.5-8B-A1B and similar reasoning models
/// sometimes emit `</think>` *before* `<think>` — there is no opening tag
/// at all, the close is just an orphan (e.g. an over-eager early stop in
/// the model's own prompt conditioning), and the actual reasoning block
/// follows. The state machine recognises this as `State::OrphanClose` so
/// the orphan close + opening tag are stripped as a unit and the body
/// resumes after the matching closer.
///
/// Latency notes: text is only buffered while it could still turn into the
/// opener. Anything that cannot be an opener prefix is flushed immediately,
/// so the common case (model answers directly) adds no first-token delay.
/// An opener that never closes is dropped entirely — it is reasoning that
/// ran to the end of generation without producing an answer.
pub struct ThinkFilter {
    pending: String,
    state: State,
}

/// Current mode of the filter. `Open` is the leading-only think-block path;
/// `OrphanClose` tracks the pathological case where the model emits ``
/// without a preceding `<think>` and we are waiting for the actual opener.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// No decision yet — `pending` may still grow into the opener.
    Unknown,
    /// Inside a `<think>...` block, waiting for ``.
    Open,
    /// Saw `` without a preceding opener. Will drop it together with the
    /// next `<think>` block (and its closer) and resume passthrough.
    OrphanClose,
    /// Past the leading reasoning: any further text is the real answer.
    Passthrough,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

impl ThinkFilter {
    pub fn new() -> Self {
        Self {
            pending: String::new(),
            state: State::Unknown,
        }
    }

    pub fn push(&mut self, text: &str) -> String {
        // Passthrough: emit verbatim, no state work needed.
        if self.state == State::Passthrough {
            return text.to_string();
        }
        self.pending.push_str(text);

        // State machine: try to make progress on the leading-prefix problem.
        match self.state {
            State::Unknown => self.advance_unknown(),
            State::Open => self.advance_open(),
            State::OrphanClose => self.advance_orphan_close(),
            State::Passthrough => unreachable!(),
        }
    }

    /// `Unknown` -> either `Open` (saw `<think>`), `OrphanClose` (saw
    /// ``), or `Passthrough` (any leading text that isn't either).
    fn advance_unknown(&mut self) -> String {
        // Fast paths: full tag at the very start.
        if let Some(rest) = self.pending.strip_prefix(THINK_OPEN) {
            self.pending = rest.to_string();
            self.state = State::Open;
            return self.advance_open();
        }
        if self.pending.starts_with(THINK_CLOSE) {
            // Pathological: model emitted a `` with no matching opener.
            // Eat the close and wait for the real `<think>`.
            self.pending.clear();
            self.state = State::OrphanClose;
            return String::new();
        }

        // The model often emits `...` as a single unit,
        // possibly split across chunks. Trim leading whitespace
        // and see whether the trimmed start still has a chance of becoming
        // either tag once more text arrives.
        let leading_ws = self.pending.len() - self.pending.trim_start().len();
        let trimmed = &self.pending[leading_ws..];

        if trimmed.is_empty() {
            // All whitespace so far. Buffer; could still become
            // `<think>` or `` after the whitespace.
            return String::new();
        }
        if THINK_OPEN.starts_with(trimmed) || THINK_CLOSE.starts_with(trimmed) {
            // Still building toward a tag — wait.
            return String::new();
        }
        if trimmed.starts_with('<') {
            // First non-WS char is `<` but no prefix match yet.
            // Look for an embedded orphan close → open pair within the
            // pending buffer (model emits "...</think>\n<think>" as
            // a unit). If we find one, the whole leading sequence
            // including the orphan close and the following opener are
            // junk we should eat before the real answer.
            if let Some(close_idx) = trimmed.find(THINK_CLOSE) {
                let after_close = &trimmed[close_idx + THINK_CLOSE.len()..];
                if let Some(rest) = after_close.strip_prefix(THINK_OPEN) {
                    self.pending = rest.to_string();
                    self.state = State::Open;
                    return self.advance_open();
                }
                // The orphan close is there but the opener hasn't shown
                // up yet — switch to OrphanClose mode and let the
                // orphan-close handler wait for `<think>`.
                if THINK_OPEN.starts_with(after_close.trim_start()) {
                    self.pending.clear();
                    self.state = State::OrphanClose;
                    return String::new();
                }
            }
            // No complete orphan close yet — keep buffering in case the
            // next chunk brings more of the tag.
            return String::new();
        }
        // First non-WS char is plain text. This is the real answer;
        // whatever leading whitespace is here belongs to it.
        self.state = State::Passthrough;
        std::mem::take(&mut self.pending)
    }

    /// `Open` -> emit text after ``, switch to `Passthrough`.
    fn advance_open(&mut self) -> String {
        if let Some(offset) = self.pending.find(THINK_CLOSE) {
            let output = self.pending[offset + THINK_CLOSE.len()..].to_string();
            self.pending.clear();
            self.state = State::Passthrough;
            return output;
        }
        // Keep only a possible suffix of the closer; drop reasoning text.
        let keep = closer_prefix_keep(&self.pending);
        self.pending = self.pending.split_off(self.pending.len() - keep);
        String::new()
    }

    /// `OrphanClose` -> if we now see `<think>`, drop it and the matching
    /// closer; otherwise accumulate until we can decide.
    fn advance_orphan_close(&mut self) -> String {
        // Need to skip any whitespace/newlines between the orphan ``
        // and the actual opener. Check whether the pending buffer starts
        // with the opener (possibly preceded by whitespace).
        let trimmed = self.pending.trim_start();
        if let Some(rest) = trimmed.strip_prefix(THINK_OPEN) {
            // Found the real opener. Drop everything up to and including it,
            // then walk forward to find the matching closer.
            self.pending = rest.to_string();
            self.state = State::Open;
            return self.advance_open();
        }
        if THINK_OPEN.starts_with(trimmed) {
            // Still could become the opener once more text arrives.
            return String::new();
        }
        // No opener ever came. The orphan `` is meaningless on its own;
        // whatever we have buffered is just leading text and should pass
        // through. (In practice this only fires if the model never thinks
        // at all and we saw `` early — typically as part of the template
        // boilerplate echo.)
        self.state = State::Passthrough;
        std::mem::take(&mut self.pending)
    }

    /// Flush at end of generation. Returns anything still worth emitting;
    /// an unterminated reasoning block is discarded.
    pub fn finish(&mut self) -> String {
        // An `Open` block that never closes: discard (model went straight
        // to thinking with no answer). An `OrphanClose` that never met its
        // opener: passthrough whatever was buffered.
        match self.state {
            State::Open => {
                self.state = State::Passthrough;
                self.pending.clear();
                String::new()
            }
            _ => std::mem::take(&mut self.pending),
        }
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

    /// LFM2.5-8B-A1B emits an orphan `` *before* `<think>` (its own
    /// training-data quirk). The filter must eat the orphan close, then
    /// also strip the leading `<think>` block that follows, and only emit
    /// the trailing real answer.
    #[test]
    fn orphan_close_then_open_is_stripped() {
        let open = THINK_OPEN;
        let close = THINK_CLOSE;
        let mut filter = ThinkFilter::new();
        let mut out = String::new();
        for chunk in [close, "\n", open, "secret reasoning\n", close, "Paris"] {
            out.push_str(&filter.push(chunk));
        }
        out.push_str(&filter.finish());
        assert_eq!(out, "Paris");
    }

    /// Same path, but the orphan close and the opener are split across
    /// streaming chunks. The leading whitespace between them must not be
    /// emitted prematurely.
    #[test]
    fn orphan_close_then_open_split_across_chunks() {
        let open = THINK_OPEN;
        let close = THINK_CLOSE;
        let mut filter = ThinkFilter::new();
        let mut out = String::new();
        // ``close``, then 1 char of opener, then rest of opener + think content.
        for chunk in [close, "\n", &open[..1], &open[1..], "think body\n", close, "answer"] {
            out.push_str(&filter.push(chunk));
        }
        out.push_str(&filter.finish());
        assert_eq!(out, "answer");
    }

    /// An orphan `` with no `<think>` ever appearing must NOT swallow the
    /// leading answer text — the filter should fall through to passthrough.
    #[test]
    fn orphan_close_without_open_passes_through() {
        let close = THINK_CLOSE;
        let mut filter = ThinkFilter::new();
        let mut out = String::new();
        for chunk in [close, "\n", "just plain text"] {
            out.push_str(&filter.push(chunk));
        }
        out.push_str(&filter.finish());
        assert_eq!(out, "\njust plain text");
    }

    /// LFM2.5-8B-A1B emits the entire `...` unit as the
    /// very first chunk(s). The leading whitespace must not leak through
    /// before we can confirm the close+open pair — we hold it back until
    /// the leading tag becomes unambiguous.
    #[test]
    fn newline_before_close_then_open_strips_whitespace() {
        let open = THINK_OPEN;
        let close = THINK_CLOSE;
        let mut filter = ThinkFilter::new();
        let mut out = String::new();
        // Feed `` as one chunk, `` followed by opener.
        for chunk in ["\n", close, "\n", open, "think\n", close, "answer"] {
            out.push_str(&filter.push(chunk));
        }
        out.push_str(&filter.finish());
        assert_eq!(out, "answer");
    }
}
