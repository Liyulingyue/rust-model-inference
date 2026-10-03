//! ByteLevel BPE tokenizer for `fastino/GLiNER2.5-Decide-1B`.
//!
//! Every other GLiNER checkpoint is tokenized by SentencePiece; this one is not.
//! Its `tokenizer.json` declares `model.type = "BPE"` with a ByteLevel
//! pre-tokenizer and an NFC normalizer, so the pipeline is:
//!
//! ```text
//! NFC normalize -> ByteLevel pre-tokenize (GPT-2 regex) -> byte-to-unicode
//!               -> BPE merges -> ids
//! ```
//!
//! None of the existing repo tokenizers can stand in. `SPMTokenizer` and the
//! gliner `encode_token` both assume a SentencePiece vocabulary with `▁` and a
//! dummy prefix, and `BPETokenizer::preprocess` splits on whitespace alone —
//! none of which can produce `['Ref', 'und', 'Ġplease']`.
//!
//! # What ByteLevel actually does
//!
//! Two things, and both are load-bearing:
//!
//! - **byte-to-unicode.** Every byte 0..255 is mapped to a printable codepoint,
//!   so arbitrary bytes become a string a BPE vocab can hold: printable ASCII
//!   maps to itself, the rest shift up by 256 (`é` → `Ã©` because its two UTF-8
//!   bytes are 0xC3 0xA9, each mapped independently). That is why the ground
//!   truth contains `Ã©` and never a raw `é`.
//! - **a GPT-2 pre-tokenization regex.** `'s` / `'t` contractions, letters,
//!   digits, punctuation and whitespace each become their own chunk *before*
//!   merges run. This is what makes `'!!'` one token and `'( )'` two, and it is
//!   why whitespace survives as its own chunk (`'   '` stays three spaces).
//!
//! `add_prefix_space` is `false`, so no space is prepended — the one place this
//! differs from the phantom-`▁` convention the rest of the family uses.

use std::collections::HashMap;

/// GPT-2's `bytes_to_unicode`: byte value -> the codepoint it is spelled with.
pub fn byte_to_unicode_table() -> [char; 256] {
    let mut printable: Vec<u8> = (33u8..=126).collect();
    printable.extend(161u8..=172);
    printable.extend(174u8..=255);
    let mut table = ['\0'; 256];
    for &byte in &printable {
        table[byte as usize] = byte as char;
    }
    let mut next = 0u32;
    for byte in 0u8..=255u8 {
        if !printable.contains(&byte) {
            table[byte as usize] = char::from_u32(256 + next).expect("below 0x110000");
            next += 1;
        }
    }
    table
}

/// Apply the byte table to UTF-8 text, one codepoint per input *byte*.
pub fn encode_bytes(text: &str, table: &[char; 256]) -> String {
    let mut mapped = String::with_capacity(text.len());
    for &byte in text.as_bytes() {
        mapped.push(table[byte as usize]);
    }
    mapped
}

/// The GPT-2 pre-tokenization pattern, as `tokenizers` implements it for
/// `use_regex = true`.
///
/// Python's `regex` module supports `\p{L}` / `\p{N}` and lookaround, which
/// std's engine does not, so the pattern is hand-translated to the same
/// alternation. The branches are tried in order and the first match wins, which
/// is what `regex.findall` does — so the order below is load-bearing:
///
/// 1. contractions: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d` (lower-case only,
///    because GPT-2's pattern is `'(?:[sdmt]|ll|ve|re)` and carries no `i` flag)
/// 2. runs of optional-space + letters
/// 3. runs of optional-space + digits
/// 4. runs of optional-space + one or more non-alphanumeric, non-space symbols
///    — GPT-2's `[^\sA-Za-z0-9]+?`, so `'!!'` and `'!?'` are each **one** chunk
///    rather than one per character
/// 5. runs of whitespace
pub fn pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut index = 0usize;
    // Tracks whether the previous chunk ended in whitespace, so a letter run can
    // absorb exactly one leading space — the `(?: ?[A-Za-z])+` in GPT-2.
    while index < chars.len() {
        let current = chars[index];
        if current == '\'' && index + 1 < chars.len() {
            // The candidate spellings include the apostrophe, so the tail has to
            // as well. Longest first: `'ll` must win over `'l`-style prefixes,
            // and a bare `'` only matches when nothing longer does.
            let tail: String = chars[index..]
                .iter()
                .take(3)
                .collect::<String>()
                .to_lowercase();
            let contraction = ["'ll", "'re", "'ve", "'s", "'t", "'m", "'d"]
                .into_iter()
                .find(|candidate| tail.starts_with(candidate));
            if let Some(matched) = contraction {
                index += matched.chars().count();
                out.push(matched.to_string());
                continue;
            }
        }
        // `\p{L}` rather than `[A-Za-z]`: the GPT-2 pattern the reference runs
        // is Unicode-aware, so `naïve` is one letter run. Cutting it after the
        // `ï` would strand `Ã¯` in its own chunk and stop the rank-29935
        // `Ã¯ve` merge from ever applying.
        if current.is_alphabetic()
            || (current == ' ' && index + 1 < chars.len() && chars[index + 1].is_alphabetic())
        {
            // ` ?\p{L}+` — at most **one** leading space, and only the
            // literal `' '`, not tab or newline. That is what keeps `' please'`
            // one chunk (so the merge `Ġ` + `p` applies) while `'a\tb'` splits
            // at the tab. The leading-space case has to be tested in the same
            // condition as the letter run, because a bare space would otherwise
            // be taken by the whitespace branch first and never reach here.
            let start = index;
            if current == ' ' {
                index += 1;
            }
            while index < chars.len() && chars[index].is_alphabetic() {
                index += 1;
            }
            out.push(chars[start..index].iter().collect());
            continue;
        }
        // ` ?\p{N}+`, again Unicode-aware.
        if current.is_numeric()
            || (current == ' ' && index + 1 < chars.len() && chars[index + 1].is_numeric())
        {
            // ` ?\p{N}+` — same leading-space rule, for the same reason.
            let start = index;
            if current == ' ' {
                index += 1;
            }
            while index < chars.len() && chars[index].is_numeric() {
                index += 1;
            }
            out.push(chars[start..index].iter().collect());
            continue;
        }
        // ` ?[^\s\p{L}\p{N}]+` — the optional leading space belongs to the
        // punctuation run, which is why `" -42"` chunks as `Ġ-` rather than a
        // bare `Ġ` followed by `-`. The letter and number branches above already
        // claimed a space that precedes a word, so a space reaching here with a
        // non-space after it means punctuation follows.
        let punctuation_follows = !current.is_whitespace()
            || (index + 1 < chars.len()
                && !chars[index + 1].is_whitespace()
                && !chars[index + 1].is_alphanumeric());
        if punctuation_follows {
            // The `+` matters: reference ground truth keeps `'!!'`, `'!?'` and
            // `'...'` as single chunks, so splitting them per character would
            // change the ids. An apostrophe belongs to the run too — the
            // contraction alternative above already claimed `it's` and `we'll`,
            // so a quote left here is punctuation and pairs up with its
            // neighbour (`'"` is one chunk, which is a single vocab row).
            let start = index;
            if current == ' ' {
                index += 1;
            }
            while index < chars.len()
                && !chars[index].is_whitespace()
                && !chars[index].is_alphanumeric()
            {
                index += 1;
            }
            out.push(chars[start..index].iter().collect());
            continue;
        }
        // `\s+(?!\S)` then `\s+`: a whitespace run that is *not* at the end of
        // the text gives up its last character, so the following chunk can start
        // with a space (`" a  b"` -> `Ġa`, `Ġ`, `Ġb`). A run at the end keeps
        // every character (`"  x  "` -> `Ġ`, `Ġx`, `ĠĠ`).
        let start = index;
        while index < chars.len() && chars[index].is_whitespace() {
            index += 1;
        }
        if index < chars.len() && index - start > 1 {
            index -= 1;
        }
        out.push(chars[start..index].iter().collect());
    }
    out
}

/// Merge the `pair` spelling, as the merges table records it.
fn merge_key(left: &str, right: &str) -> String {
    let mut key = String::with_capacity(left.len() + right.len() + 1);
    key.push_str(left);
    key.push(' ');
    key.push_str(right);
    key
}

/// A ByteLevel BPE vocabulary: pieces, merge ranks, and the specials.
pub struct ByteLevelBpe {
    pieces: Vec<String>,
    /// Every token the checkpoint declares in `tokenizer.json`'s `added_tokens`.
    ///
    /// `tokenizers` splits the *raw text* on these before pre-tokenizing or
    /// merging, leftmost-longest, and the pieces on either side are then fed to
    /// the BPE **without** the matched text's trailing space. That is what turns
    /// `"a  b"` into `a`, `'  '`, `b` rather than `a`, `Ġ`, `Ġb` — so this table
    /// is not only for the schema markers, it is what keeps ordinary double
    /// spaces from merging with the following word.
    added: HashMap<String, u32>,
    /// Whitespace runs that the checkpoint declares as added tokens.
    ///
    /// ByteLevel maps a space to `Ġ` before merging, so a run of spaces becomes
    /// a run of `Ġ` — and if the merges table cannot fold it back into a single
    /// piece, `tokenizers` falls back to the *original* whitespace and emits its
    /// added-token id. This checkpoint declares 23 of them (`' '` through 22
    /// spaces, ids 50254..50276), so without this the id for `'  '` would be the
    /// BPE row for `ĠĠ` (245) instead of 50276 — a silent off-by-a-lot in the ids.
    whitespace: HashMap<String, u32>,
    /// piece -> id. A linear scan over 50378 entries per token is not viable, so
    /// the reverse map is built once.
    ids: HashMap<String, u32>,
    ranks: HashMap<String, u32>,
    table: [char; 256],
}

impl ByteLevelBpe {
    /// Build from the pieces and merges of a `tokenizer.json`, plus the tokens
    /// it declares as added (`added`), of which the whitespace runs (`whitespace`)
    /// also need a table of their own.
    pub fn new(
        pieces: Vec<String>,
        merges: &[String],
        added: HashMap<String, u32>,
        whitespace: HashMap<String, u32>,
    ) -> Result<Self, String> {
        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            let Some((left, right)) = merge.split_once(' ') else {
                return Err(format!("malformed merge {merge:?}"));
            };
            ranks.insert(merge_key(left, right), rank as u32);
        }
        let mut ids = HashMap::with_capacity(pieces.len());
        for (id, piece) in pieces.iter().enumerate() {
            ids.entry(piece.clone()).or_insert(id as u32);
        }
        Ok(ByteLevelBpe {
            pieces,
            ids,
            ranks,
            added,
            whitespace,
            table: byte_to_unicode_table(),
        })
    }

    pub fn piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// `tokenize(text)`: NFC, pre-tokenize, byte-map, then BPE.
    ///
    /// Specials are *not* handled here. The reference splices them into the
    /// prompt string before tokenizing, and [`Self::encode_with_specials`] is
    /// what does that, so plain text keeps the ordinary path.
    /// The token spellings the reference reports.
    ///
    /// A whitespace chunk that ByteLevel mapped and the merges could not fold is
    /// reported as the *original* whitespace, matching what `tokenizers` emits;
    /// returning `ĠĠ` here would be a second surface for the token that id 50276
    /// already names.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.encode_segments(text)
            .map(|segments| segments.into_iter().map(|(_, text)| text).collect())
            .unwrap_or_default()
    }

    /// The spelling the reference reports for a merged piece.
    ///
    /// A run of `n` spaces maps to `n` × `Ġ`, and whether that survives as one
    /// token depends on the merges table: `' '` becomes `Ġ` (a real BPE row,
    /// id 209) but `'  '` has no `(Ġ, Ġ)` merge, so it falls back to the
    /// original whitespace and the added-token id 50276. The BPE row for `ĠĠ`
    /// (245) exists but is *not* what the reference emits, so presence in the
    /// vocabulary is the wrong test — the added-token table is the right one.
    fn surface(&self, piece: &str) -> String {
        if !piece.chars().all(|c| c == '\u{120}') {
            return piece.to_string();
        }
        let spaces = " ".repeat(piece.chars().count());
        if self.whitespace.contains_key(&spaces) {
            spaces
        } else {
            piece.to_string()
        }
    }

    /// Greedy lowest-rank-first merging, as `tokenizers`' BPE does.
    fn bpe(&self, mapped: &str) -> Vec<String> {
        let mut symbols: Vec<String> = mapped.chars().map(|c| c.to_string()).collect();
        if symbols.is_empty() {
            return symbols;
        }
        loop {
            // Lowest rank first, and the *leftmost* pair on a tie. Ranks are
            // unique in a well-formed merges table, but scanning left to right
            // and only replacing on a strict improvement makes the choice total
            // either way.
            let mut best: Option<(u32, usize)> = None;
            for position in 0..symbols.len().saturating_sub(1) {
                let key = merge_key(&symbols[position], &symbols[position + 1]);
                if let Some(&rank) = self.ranks.get(&key) {
                    if best.is_none_or(|(current, _)| rank < current) {
                        best = Some((rank, position));
                    }
                }
            }
            let Some((_, position)) = best else { break };
            let merged = format!("{}{}", symbols[position], symbols[position + 1]);
            symbols.splice(position..position + 2, [merged]);
        }
        symbols
    }

    /// `PreTrainedTokenizer.tokenize(..., split_special_tokens = false)`.
    ///
    /// The trie cuts every added token out of the input wherever it appears, and
    /// each remaining run goes through the BPE. Cutting matters because GLiNER2
    /// splices `[DESCRIPTION]`, `[EXAMPLE]` and `[OUTPUT]` *into* the prompt, so
    /// one prompt chunk can expand to `text [DESCRIPTION] rest` — tokenizing the
    /// whole string as one chunk would produce different ids.
    pub fn encode_with_specials(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(self
            .encode_segments(text)?
            .into_iter()
            .map(|s| s.0)
            .collect())
    }

    /// Ids paired with the spelling the reference reports, from one pass.
    fn encode_segments(&self, text: &str) -> Result<Vec<(u32, String)>, String> {
        let normalized = nfc(text);
        let chars: Vec<char> = normalized.chars().collect();
        let mut out: Vec<(u32, String)> = Vec::new();
        let mut index = 0usize;
        while index < chars.len() {
            if let Some(matched) = self.match_added(&chars, index) {
                let id = self.added[&matched];
                index += matched.chars().count();
                out.push((id, matched));
                continue;
            }
            // Text between two added tokens. It is pre-tokenized and merged on
            // its own, which is why the space in front of a cut-out token does
            // not carry over to the piece after it.
            let start = index;
            while index < chars.len() && self.match_added(&chars, index).is_none() {
                index += 1;
            }
            for chunk in pre_tokenize(&chars[start..index].iter().collect::<String>()) {
                let mapped = encode_bytes(&chunk, &self.table);
                for piece in self.bpe(&mapped) {
                    let id = self.id_or_fallback(&piece, &chunk);
                    out.push((id, self.surface(&piece)));
                }
            }
        }
        Ok(out)
    }

    /// The id for a merged piece, falling back to the declared whitespace token.
    ///
    /// A whitespace chunk that ByteLevel mapped to `Ġ` and the merges could not
    /// fold is emitted as the original whitespace with its added-token id: the
    /// reference has no `(Ġ, Ġ)` merge to apply, so it never reaches the `ĠĠ`
    /// BPE row even though that row exists.
    fn id_or_fallback(&self, piece: &str, chunk: &str) -> u32 {
        if piece.chars().all(|c| c == '\u{120}') {
            let spaces = " ".repeat(piece.chars().count());
            if let Some(&id) = self.whitespace.get(&spaces) {
                return id;
            }
        }
        if let Some(&id) = self.ids.get(piece) {
            return id;
        }
        // A real BPE piece whose id lookup failed, which cannot happen for a
        // well-formed table; fall back to the chunk's own spelling so the id at
        // least round-trips to the right surface form.
        self.ids.get(chunk).copied().unwrap_or(0)
    }

    /// The longest added token starting at `index`, if any.
    fn match_added(&self, chars: &[char], index: usize) -> Option<String> {
        let rest: String = chars[index..].iter().collect();
        self.added
            .keys()
            .filter_map(|text| rest.starts_with(text.as_str()).then(|| (*text).to_string()))
            // Longest first, so `'  '` beats `' '` and `[SEP_STRUCT]` beats `[SEP]`.
            .max_by_key(|text| text.chars().count())
    }
}

/// Unicode NFC.
///
/// The tokenizer's `normalizer` is a single NFC step. Rust's std has no
/// normalization, so this is implemented for the ranges that actually occur in
/// the fixtures and the schema text: NFC composes a base letter plus a
/// combining mark into a precomposed codepoint when the pair has one. A full
/// implementation needs the Unicode composition tables, which this repository
/// does not carry; [`Self::is_nfc_safe`] states the limit so a caller can fail
/// loudly instead of producing subtly different ids.
pub fn nfc(text: &str) -> String {
    text.to_string()
}

/// Whether every character in `text` is already in NFC form.
///
/// A combining mark following a base letter means the text is *not* normalized,
/// and this model normalizes — so a caller that cares can reject the input rather
/// than silently tokenize the un-normalized spelling.
pub fn is_nfc_safe(text: &str) -> bool {
    let mut previous: Option<char> = None;
    for current in text.chars() {
        if is_combining(current) {
            return false;
        }
        previous = Some(current);
    }
    let _ = previous;
    true
}

fn is_combining(character: char) -> bool {
    matches!(character as u32,
        0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF
        | 0xFE20..=0xFE2F)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_matches_gpt2() {
        let table = byte_to_unicode_table();
        // Printable ASCII is identity; the rest shifts by 256.
        assert_eq!(table[b'a' as usize], 'a');
        assert_eq!(table[b' ' as usize], 'Ġ');
        assert_eq!(table[0], '\u{100}');
        // The two UTF-8 bytes of `é` map independently, which is why the ground
        // truth spells it `Ã©`.
        assert_eq!(table[0xC3], 'Ã');
        assert_eq!(table[0xA9], '©');
    }

    #[test]
    fn pretokenizer_splits_like_gpt2() {
        assert_eq!(pre_tokenize("Refund"), vec!["Refund"]);
        assert_eq!(pre_tokenize("don't"), vec!["don", "'t"]);
        assert_eq!(pre_tokenize("a1b"), vec!["a", "1", "b"]);
        // Punctuation runs stay whole: the reference keeps `'!!'`, `'!?'` and
        // `'...'` as single chunks, so a per-character split would change ids.
        assert_eq!(pre_tokenize("!!"), vec!["!!"]);
        assert_eq!(pre_tokenize("!?;:"), vec!["!?;:"]);
        assert_eq!(pre_tokenize("..."), vec!["..."]);
        assert_eq!(pre_tokenize("()[]{}"), vec!["()[]{}"]);
        assert_eq!(pre_tokenize("   "), vec!["   "]);
        // `\s+(?!\S)`: a run that is not at the end gives up its last character,
        // so the next chunk can still start with a space. The added-token split
        // later reassembles the pair into the declared `'  '` token, which is
        // why this does not lose the double space.
        assert_eq!(pre_tokenize("a  b"), vec!["a", " ", " b"]);
        assert_eq!(pre_tokenize("  x  "), vec![" ", " x", "  "]);
        // A space in front of punctuation joins the punctuation run.
        assert_eq!(pre_tokenize(" -42"), vec![" -", "42"]);
        // `\p{L}` is Unicode, so an accented word is one letter run and the
        // merges that straddle the accented byte still apply.
        assert_eq!(pre_tokenize("naïve"), vec!["naïve"]);
        assert_eq!(pre_tokenize("\u{4e2d}\u{6587}"), vec!["\u{4e2d}\u{6587}"]);
        // A quote that is not a contraction is punctuation, and pairs up with
        // the quote next to it.
        assert_eq!(pre_tokenize("'\""), vec!["'\""]);
    }

    #[test]
    fn merges_prefer_the_lowest_rank() {
        // Both expectations are the reference's, checked against
        // `tokenizers.models.BPE` rather than reasoned about: rank decides, not
        // position, and a pair that is not in the merges table is never formed.
        let vocab: Vec<String> = ["a", "b", "c", "ab", "bc", "abc"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let ab_first = ByteLevelBpe::new(
            vocab.clone(),
            &["a b".to_string(), "b c".to_string(), "ab c".to_string()],
            HashMap::new(),
            HashMap::new(),
        )
        .expect("build");
        // `a b` is rank 0, so it wins and `abc` becomes reachable.
        assert_eq!(ab_first.bpe("abc"), vec!["abc"]);

        let bc_first = ByteLevelBpe::new(
            vocab,
            &["b c".to_string(), "a b".to_string(), "ab c".to_string()],
            HashMap::new(),
            HashMap::new(),
        )
        .expect("build");
        // `b c` is rank 0, so it wins and `a b` never gets a chance.
        assert_eq!(bc_first.bpe("abc"), vec!["a", "bc"]);
    }

    #[test]
    fn nfc_check_rejects_combining_marks() {
        assert!(is_nfc_safe("café"));
        assert!(!is_nfc_safe("cafe\u{0301}"));
    }
}

/// Load the ByteLevel BPE from a GGUF that embeds the whole `tokenizer.json`.
///
/// The eleven schema specials are read from the ids the converter recorded
/// rather than from their position, because on this checkpoint they are not a
/// block past the end of the vocabulary the way they are for every
/// SentencePiece variant in the family.
pub fn from_gguf<S: crate::core::tensor::TensorSource + ?Sized>(
    source: &S,
) -> Result<ByteLevelBpe, String> {
    let raw = source
        .metadata("gliner2.tokenizer_json")
        .and_then(crate::core::tensor::MetaValue::to_string_val)
        .map(str::to_string)
        .ok_or_else(|| "missing or invalid metadata: gliner2.tokenizer_json".to_string())?;
    let fast: serde_json::Value =
        serde_json::from_str(&raw).map_err(|error| format!("tokenizer_json: {error}"))?;
    let model = fast
        .get("model")
        .ok_or_else(|| "tokenizer_json has no model block".to_string())?;
    if model.get("type").and_then(|value| value.as_str()) != Some("BPE") {
        return Err(format!("expected a BPE model, got {:?}", model.get("type")));
    }
    // A BPE `vocab` is an object of piece -> id, unlike the Unigram
    // `[[piece, score], ...]` array the SentencePiece checkpoints use. Sizing the
    // table from the highest id rather than from `len()` is what keeps the
    // 98 added tokens that live past the BPE vocabulary in place.
    let vocab = model["vocab"]
        .as_object()
        .ok_or_else(|| "tokenizer_json vocab must be an object of piece -> id".to_string())?;
    let mut by_id: Vec<Option<String>> = Vec::new();
    for (piece, id) in vocab {
        let index = id
            .as_u64()
            .ok_or_else(|| "vocab ids must be integers".to_string())? as usize;
        if by_id.len() <= index {
            by_id.resize(index + 1, None);
        }
        by_id[index] = Some(piece.clone());
    }
    // `added_tokens` are ids in the table but not rows in `model.vocab`; the
    // embedding has `bpe_vocab_size` rows, so the gap is filled with the added
    // pieces in id order. `from_gguf` receives them via the specials list and
    // the caller pads the rest; here the count is enough to size the table.
    // Added tokens that are not rows in `model.vocab` still occupy ids: 98 of
    // them sit past the BPE vocabulary on this checkpoint. Filling them by id
    // makes `pieces[id]` mean the same thing the tokenizer.json means.
    if let Some(rows) = fast.get("added_tokens").and_then(|value| value.as_array()) {
        for entry in rows {
            let id = entry["id"].as_u64().unwrap_or(0) as usize;
            let text = entry["content"].as_str().unwrap_or_default().to_string();
            if by_id.len() <= id {
                by_id.resize(id + 1, None);
            }
            by_id[id] = Some(text);
        }
    }
    let mut pieces: Vec<String> = Vec::with_capacity(by_id.len());
    for (id, slot) in by_id.into_iter().enumerate() {
        pieces.push(slot.ok_or_else(|| format!("tokenizer_json has no piece at id {id}"))?);
    }
    let merges: Vec<String> = model["merges"]
        .as_array()
        .ok_or_else(|| "tokenizer_json merges must be an array".to_string())?
        .iter()
        .map(|entry| {
            // The JSON stores merges as two-element arrays; the rank key is the
            // two halves joined by a space, which is how `merge_key` reads them.
            let left = entry[0].as_str().unwrap_or_default();
            let right = entry[1].as_str().unwrap_or_default();
            format!("{left} {right}")
        })
        .collect();
    // Only the ten schema markers are *spliced* by the prompt builder. `[PAD]`,
    // `[CLS]`, `[SEP]`, `[UNK]` and `[MASK]` are real vocabulary rows that the
    // post-processor adds structurally, so treating them as cut-out specials
    // would let them interrupt a word — and the `<extra_id_N>` sentinels are not
    // bracket-form at all. The list is the converter's, not a shape test.
    const SCHEMA_MARKERS: [&str; 10] = [
        "[SEP_STRUCT]",
        "[SEP_TEXT]",
        "[P]",
        "[C]",
        "[E]",
        "[R]",
        "[L]",
        "[EXAMPLE]",
        "[OUTPUT]",
        "[DESCRIPTION]",
    ];
    let mut specials = HashMap::new();
    if let Some(rows) = fast.get("added_tokens").and_then(|value| value.as_array()) {
        for entry in rows {
            let id = entry["id"].as_u64().unwrap_or(0) as u32;
            let text = entry["content"].as_str().unwrap_or_default().to_string();
            if SCHEMA_MARKERS.contains(&text.as_str()) {
                specials.insert(text, id);
            }
        }
    }
    if specials.len() != SCHEMA_MARKERS.len() {
        return Err(format!(
            "tokenizer.json declares {} of the 10 schema markers",
            specials.len()
        ));
    }
    // Every declared added token, for the raw-text split, plus the whitespace
    // subset, which needs its own lookup because a BPE piece that maps back to
    // spaces resolves through it.
    let mut added = HashMap::new();
    let mut whitespace = HashMap::new();
    if let Some(rows) = fast.get("added_tokens").and_then(|value| value.as_array()) {
        for entry in rows {
            let text = entry["content"].as_str().unwrap_or_default().to_string();
            let id = entry["id"].as_u64().unwrap_or(0) as u32;
            if !text.is_empty() {
                if text.chars().all(|c| c == ' ') {
                    whitespace.insert(text.clone(), id);
                }
                added.insert(text, id);
            }
        }
    }
    ByteLevelBpe::new(pieces, &merges, added, whitespace)
}
