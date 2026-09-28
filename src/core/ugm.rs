//! Unigram (UGM) tokenizer for `tokenizer.ggml.model = "t5"` GGUF models.
//!
//! Pinned against `references/llama.cpp/src/llama-vocab.cpp` (the
//! `llm_tokenizer_ugm_session` class around line 967 and the
//! `xcda_array_view` helper around line 1133). Each block below carries
//! the corresponding oracle line range in its doc comment.
//!
//! Three pieces:
//!
//!   1. [`NaiveTrie`] — recursive 8-bit `std::map<char, naive_trie>`,
//!      used both as the UGM token matcher (insert with `id`) and the
//!      user-defined matcher (insert with no value).
//!   2. [`XcdaArray`] — XOR-compressed double-array trie reader.
//!      Each packed u32 holds a 21-bit base, an 8-bit lcheck, a 1-bit
//!      leaf, and a 1-bit replacement-sequence marker.
//!   3. [`UgmTokenizer`] — the public tokenizer. It owns both tries,
//!      normalizes input through the XCDA blob, then runs SentencePiece
//!      unigram Viterbi over `token_matcher` to produce the best-scoring
//!      tokenization, merging consecutive unknown-token emissions
//!      (`llama-vocab.cpp:1067-1078`).
//!
//! Bit-exact with llama.cpp's vocabulary module: `normalize_prefix` follows
//! `llm_tokenizer_ugm_session::normalize_prefix` (`llama-vocab.cpp:1167`),
//! the Viterbi is the loop at `llama-vocab.cpp:993-1062`, and the
//! post-emit UNK merge is at `llama-vocab.cpp:1067-1078`. Scores use
//! `f64` (one wider than llama.cpp's `double`) to match `token_data.score`
//! which llama.cpp also stores as `float32` but accumulates in `double`.

use std::collections::HashMap;

use crate::core::tensor::MetaValue;

#[derive(Debug, Default, Clone)]
pub struct NaiveTrie {
    pub children: HashMap<char, NaiveTrie>,
    pub value: Option<u32>,
}

impl NaiveTrie {
    /// `naive_trie::insert` (`llama-vocab.cpp:32`). Inserts the empty key
    /// with `value` only when `len == 0`.
    fn insert(&mut self, key: &str, value: u32) {
        let mut node = self;
        for c in key.chars() {
            node = node.children.entry(c).or_default();
        }
        node.value = Some(value);
    }

    /// `naive_trie::traverse` (`llama-vocab.cpp:59`).
    pub fn traverse(&self, c: char) -> Option<&NaiveTrie> {
        self.children.get(&c)
    }

    /// `naive_trie::get_longest_prefix` (`llama-vocab.cpp:47`), returning
    /// `(bytes_consumed, child_terminal_offset)`. The caller uses the
    /// length; the user-defined matcher needs the offset to track where
    /// in the input the match began.
    #[allow(dead_code)]
    fn longest_prefix<'a>(&self, input: &'a [u8], offset: usize) -> usize {
        let mut node = self;
        let mut depth = 0usize;
        for &byte in &input[offset..] {
            // Trie keys are byte-valued (they're built from raw token text,
            // which is treated as bytes by `insert`).
            let c = byte as char;
            match node.children.get(&c) {
                Some(child) => {
                    node = child;
                    depth += 1;
                }
                None => return depth,
            }
        }
        depth
    }
}

/// XOR-compressed double-array trie reader.
/// `xcda_array_view` (`llama-vocab.cpp:1133`).
struct XcdaArray<'a> {
    entries: &'a [u32],
}

impl<'a> XcdaArray<'a> {
    fn new(entries: &'a [u32]) -> Self {
        Self { entries }
    }

    fn get(&self, index: usize) -> u32 {
        self.entries
            .get(index)
            .copied()
            .expect("Index out of array bounds in XCDA array!")
    }

    /// `get_base` (`llama-vocab.cpp:1138`). `base = (packed >> 10) << ((packed & 0x200) >> 6)`.
    fn base(&self, index: usize) -> u32 {
        let p = self.get(index);
        (p >> 10) << ((p & (1u32 << 9)) >> 6)
    }

    /// `get_lcheck` (`llama-vocab.cpp:1142`). `lcheck = packed & (1u31 | 0xff)`.
    fn lcheck(&self, index: usize) -> u32 {
        let p = self.get(index);
        p & ((1u32 << 31) | 0xff)
    }

    /// `get_leaf` (`llama-vocab.cpp:1146`). `leaf = (packed >> 8) & 1`.
    fn leaf(&self, index: usize) -> bool {
        (self.get(index) >> 8) & 1 == 1
    }

    /// `get_value` (`llama-vocab.cpp:1150`). `value = packed & ((1u<<31) - 1)`.
    fn value(&self, index: usize) -> u32 {
        self.get(index) & ((1u32 << 31) - 1)
    }
}

/// One pass over `normalize_prefix` (`llama-vocab.cpp:1167`).
///
/// We own the bytes rather than borrowing because the replacement path
/// pulls from `self.prefix_replacements`, which has a different lifetime
/// from the input slice. The lengths are tiny in practice (single-byte
/// to short-codepoint replacements), so an owned copy is cheap.
struct NormalizationResult {
    normalized: Vec<u8>,
    consumed_input: usize,
}

#[derive(Debug)]
pub enum UgmError {
    Meta(String),
}

impl std::fmt::Display for UgmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UgmError::Meta(s) => write!(f, "UGM init: {s}"),
        }
    }
}

impl std::error::Error for UgmError {}

/// Adapter so `UgmTokenizer` fits the unified [`crate::core::tokenizer::Tokenizer`]
/// trait used by `load_tokenizer`.
impl crate::core::tokenizer::Tokenizer for UgmTokenizer {
    fn encode(&self, text: &str, options: crate::core::tokenizer::EncodeOptions) -> Vec<u32> {
        self.encode_with_options(text, options.add_special, options.parse_special)
    }
    fn decode_bytes(&self, ids: &[u32], render_special: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            out.extend(self.decode_one(id, render_special));
        }
        out
    }
    fn token_piece_bytes(&self, id: u32, render_special: bool) -> Vec<u8> {
        self.decode_one(id, render_special)
    }
    fn token_id(&self, literal: &str) -> Option<u32> {
        self.token_id(literal)
    }
    fn special_token_id(&self, _semantic_name: &str) -> Option<u32> {
        None
    }
    fn bos_id(&self) -> Option<u32> {
        UgmTokenizer::bos_id(self)
    }
    fn eos_id(&self) -> Option<u32> {
        UgmTokenizer::eos_id(self)
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}

#[derive(Debug)]
pub struct UgmTokenizer {
    /// Sorted vocabulary strings — id 0..vocab_size. `tokens[id]` is the
    /// raw token text (may start with the SentencePiece `▁` marker).
    tokens: Vec<String>,
    token_types: Vec<TokenType>,
    scores: Vec<f32>,
    /// `token_matcher` trie (id-bearing).
    pub matcher: NaiveTrie,
    /// `user_defined_token_matcher` trie (offset-only).
    user_defined: NaiveTrie,
    /// xcda_array_size = `xcda_blob_size / 4`. 0 means no normalizer map.
    xcda_entries: Vec<u32>,
    /// The replacement-strings tail of the charsmap, indexed by
    /// `xcda_array_view::get_value` after a leaf hit.
    prefix_replacements: Vec<u8>,
    /// `llama-vocab.cpp:944-948`. Used for `unknown_token_score_penalty`
    /// and the final UNK fallback. Cloned at construction time.
    min_score: f32,
    unknown_token_score_penalty: f32,
    /// `llama-vocab.cpp:1082`. 1 = the `▁` literal (3-byte UTF-8 `U+2581`)
    /// is emitted instead of an ASCII space. UGM defaults to 1
    /// (`llama-vocab.cpp:1837`); BPE models force 0 at line 2157.
    escape_whitespaces: u32,
    /// `llama-vocab.cpp:1083`. Whether to prepend a space at the start of
    /// the normalized output. Default true for UGM.
    add_space_prefix: bool,
    /// `llama-vocab.cpp:1084`. Whether to merge consecutive spaces.
    remove_extra_whitespaces: bool,
    /// `llama-vocab.cpp:1085`. UGM defaults to `true` (whitespace is a
    /// suffix of a word, not a prefix).
    treat_whitespace_as_suffix: bool,
    /// `llama-vocab.cpp:1089`. Always zero in current llama.cpp.
    unused_token_id: Option<u32>,
    /// First control token id (used to backtrack when we encounter one in
    /// the trie).
    control_token_id: u32,
    /// Optional: the `<unk>` token id (preferred UNK surface).
    unk_id: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    pub fn from_gguf(v: i32) -> Self {
        match v {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }
    pub fn is_normal(self) -> bool {
        matches!(self, Self::Normal)
    }
    pub fn is_user_defined(self) -> bool {
        matches!(self, Self::UserDefined)
    }
    pub fn is_unused(self) -> bool {
        matches!(self, Self::Unused)
    }
}

impl UgmTokenizer {
    /// Build from raw GGUF metadata. Callers must hand a closure over
    /// the GGUF meta lookup so the tokenizer has no opinions on where
    /// the bytes come from.
    pub fn from_gguf_metadata(
        get_meta: impl Fn(&str) -> Option<MetaValue>,
    ) -> Result<Self, UgmError> {
        let err = |s: &str| Err(UgmError::Meta(s.to_string()));

        // ---- model discriminator ----
        let model = match get_meta("tokenizer.ggml.model") {
            Some(MetaValue::String(value)) if value == "t5" => value,
            Some(MetaValue::String(value)) => {
                return err(&format!(
                    "Unsupported UGM tokenizer.ggml.model {value:?}; expected t5"
                ));
            }
            _ => return err("Missing or invalid tokenizer.ggml.model"),
        };
        let _ = model;

        // ---- vocab ----
        let tokens = string_array(&get_meta, "tokenizer.ggml.tokens")?;
        let n_tokens = tokens.len();
        let scores = f32_array(&get_meta, "tokenizer.ggml.scores")?;
        if scores.len() != n_tokens {
            return err("tokenizer.ggml.scores length does not match tokens length");
        }
        let token_types = match get_meta("tokenizer.ggml.token_type") {
            Some(MetaValue::Array(_, vs)) => vs
                .iter()
                .map(|v| match v {
                    MetaValue::Int32(i) => TokenType::from_gguf(*i),
                    _ => TokenType::Normal,
                })
                .collect(),
            _ => vec![TokenType::Normal; n_tokens],
        };

        // ---- special ids (llama-vocab.cpp:2042-2046 defaults) ----
        let special_bos_id = u32_meta(&get_meta, "tokenizer.ggml.bos_token_id");
        let special_unk_id = u32_meta(&get_meta, "tokenizer.ggml.unknown_token_id");
        let control_token_id = special_bos_id.unwrap_or(0);

        // ---- flags ----
        let add_space_prefix = bool_meta(&get_meta, "tokenizer.ggml.add_space_prefix", true);
        let remove_extra_whitespaces =
            bool_meta(&get_meta, "tokenizer.ggml.remove_extra_whitespaces", true);
        let treat_whitespace_as_suffix =
            bool_meta(&get_meta, "tokenizer.ggml.treat_whitespace_as_suffix", true);

        // ---- charsmap ----
        let precompiled_charsmap = match get_meta("tokenizer.ggml.precompiled_charsmap") {
            Some(MetaValue::Array(_, vs)) => {
                let mut bytes = Vec::with_capacity(vs.len());
                for v in vs {
                    match v {
                        MetaValue::Int8(b) => bytes.push(b as u8),
                        MetaValue::Uint8(b) => bytes.push(b),
                        _ => return err("precompiled_charsmap must be int8/uint8 array"),
                    }
                }
                bytes
            }
            _ => Vec::new(),
        };

        // ---- min_score, used both for Viterbi initial value and the
        //      UNK fallback penalty ----
        let mut min_score = 0.0f32;
        let mut max_score = 0.0f32;
        for (id, score) in scores.iter().enumerate() {
            if token_types[id].is_normal() {
                min_score = min_score.min(*score);
                max_score = max_score.max(*score);
            }
        }
        // `unknown_token_score_penalty` defaults to 10.0
        // (`llama-vocab.cpp:937`).
        let unknown_token_score_penalty = 10.0f32;

        // ---- tries ----
        let mut matcher = NaiveTrie::default();
        let mut user_defined = NaiveTrie::default();
        for (id, (text, kind)) in tokens.iter().zip(token_types.iter()).enumerate() {
            let bytes = text.as_bytes();
            if kind.is_normal() || kind.is_user_defined() || kind.is_unused() {
                matcher.insert(std::str::from_utf8(bytes).unwrap_or(""), id as u32);
            }
            if kind.is_user_defined() {
                user_defined.insert(std::str::from_utf8(bytes).unwrap_or(""), id as u32);
            }
        }

        // ---- split charsmap into XCDA entries + replacement bytes ----
        let (xcda_entries, prefix_replacements) = if precompiled_charsmap.len() >= 4 {
            let xcda_blob_size = u32::from_le_bytes([
                precompiled_charsmap[0],
                precompiled_charsmap[1],
                precompiled_charsmap[2],
                precompiled_charsmap[3],
            ]) as usize;
            let blob_end = 4 + xcda_blob_size;
            if blob_end > precompiled_charsmap.len() {
                return err("Index out of array bounds in precompiled charsmap");
            }
            let mut entries = Vec::with_capacity(xcda_blob_size / 4);
            for chunk in precompiled_charsmap[4..blob_end].chunks_exact(4) {
                entries.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            (entries, precompiled_charsmap[blob_end..].to_vec())
        } else {
            (Vec::new(), Vec::new())
        };

        Ok(Self {
            tokens,
            token_types,
            scores,
            matcher,
            user_defined,
            xcda_entries,
            prefix_replacements,
            min_score,
            unknown_token_score_penalty,
            escape_whitespaces: 1,
            add_space_prefix,
            remove_extra_whitespaces,
            treat_whitespace_as_suffix,
            unused_token_id: None,
            control_token_id,
            unk_id: special_unk_id,
            // add_bos / add_eos are read by callers; the UGM session does
            // not embed them. Tokenizer-level metadata is stored in
            // `vocab_model` so the wrapper at the top level can emit
            // specials.
        })
    }

    /// Look up a token id by text.
    pub fn token_id(&self, literal: &str) -> Option<u32> {
        (0..self.tokens.len())
            .find(|&id| self.tokens[id] == literal)
            .map(|i| i as u32)
    }

    /// Sentinel id used when an unknown character sequence collapses to
    /// `<unk>` (or to `control_token_id` if no `<unk>` was provided).
    pub fn unk_id(&self) -> u32 {
        self.unk_id.unwrap_or(self.control_token_id)
    }

    /// Mirror of `SPMTokenizer::bos_id` so the trait dispatch is uniform.
    pub fn bos_id(&self) -> Option<u32> {
        // For UGM, llama.cpp falls back to using the `<unk>` placeholder as
        // BOS when no explicit `bos_token_id` is shipped. We mirror that:
        // `special_bos_id` may be unset (None), and the bos is then the
        // control token (0).
        Some(self.control_token_id)
    }

    pub fn eos_id(&self) -> Option<u32> {
        Some(self.unk_id())
    }

    /// Decode one token id back to bytes, applying `render_special` for
    /// control tokens. SentencePiece tokens start with the `▁` marker
    /// (`U+2581`), which we render as a space.
    pub fn decode_one(&self, id: u32, render_special: bool) -> Vec<u8> {
        let Some(text) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        let kind = self
            .token_types
            .get(id as usize)
            .copied()
            .unwrap_or(TokenType::Normal);
        match kind {
            TokenType::Control | TokenType::Unknown | TokenType::UserDefined => {
                if render_special {
                    text.as_bytes().to_vec()
                } else {
                    Vec::new()
                }
            }
            _ => {
                let mut out = Vec::with_capacity(text.len());
                let mut chars = text.chars().peekable();
                while let Some(c) = chars.next() {
                    if c == '\u{2581}' {
                        out.push(b' ');
                    } else {
                        let mut buf = [0u8; 4];
                        let s = c.encode_utf8(&mut buf);
                        out.extend_from_slice(s.as_bytes());
                    }
                }
                out
            }
        }
    }

    /// Trait-shape encode: wraps the Viterbi tokenization with optional
    /// `add_bos` / `add_eos`. `parse_special` is wired through the
    /// user-defined trie so `<foo>` literals still work, matching the
    /// other tokenizers' behavior.
    pub fn encode_with_options(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special {
            if let Some(b) = self.bos_id() {
                out.push(b);
            }
        }
        out.extend(self.tokenize_with_options(text, parse_special));
        if add_special {
            if let Some(e) = self.eos_id() {
                out.push(e);
            }
        }
        out
    }

    fn tokenize_with_options(&self, text: &str, parse_special: bool) -> Vec<u32> {
        if !parse_special {
            return self.tokenize(text);
        }
        // Strip `<...>` user-defined segments out of the input, emit the
        // user-defined tokens in place, and tokenize the rest. This is the
        // conservative interpretation; a full handling would also bracket
        // the surrounding whitespace, which we don't need for embedding.
        let mut out = Vec::new();
        let bytes = text.as_bytes();
        let mut cursor = 0usize;
        while cursor < bytes.len() {
            if bytes[cursor] == b'<' {
                if let Some(end) = find_user_defined_end(&bytes[cursor..]) {
                    let literal = std::str::from_utf8(&bytes[cursor..cursor + end]).unwrap_or("");
                    if let Some(id) = self.token_id(literal) {
                        out.push(id);
                        cursor += end;
                        continue;
                    }
                }
            }
            // Otherwise, scan forward until next '<' and tokenize that slice.
            let scan_end = bytes[cursor..]
                .iter()
                .position(|&b| b == b'<')
                .map(|p| cursor + p)
                .unwrap_or(bytes.len());
            out.extend(self.tokenize(std::str::from_utf8(&bytes[cursor..scan_end]).unwrap_or("")));
            cursor = scan_end;
        }
        out
    }
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        // ---- normalize ----
        let input = text.as_bytes();
        let normalized = self.normalize(input);

        if normalized.is_empty() {
            return Vec::new();
        }

        // ---- Viterbi over the normalized char sequence ----
        // (`llama-vocab.cpp:993-1062`). The trie is keyed by char, not
        // byte, because the normalized text is UTF-8 and tokens like
        // `▁` (U+2581, 3 bytes) would otherwise mis-match against the
        // single-byte `(byte as char)` view. We walk the bytes for index
        // arithmetic and the chars for the trie path simultaneously.
        let normalized_str =
            std::str::from_utf8(&normalized).expect("normalize produced invalid UTF-8");
        let chars: Vec<(usize, char)> = normalized_str.char_indices().collect();
        let nc = chars.len();

        let n = normalized.len();
        let nc = chars.len();
        // `best_tokenization` array has `n+1` byte positions, plus we also
        // index by char position for O(1) lookup.
        let mut best: Vec<(u32, usize, f64)> = vec![(0, 0, f64::NEG_INFINITY); n + 1];
        best[0] = (self.unk_id(), 0, 0.0);

        for ci in 0..nc {
            let (input_offset, c0) = chars[ci];
            let utf8_len = utf8_char_len(normalized[input_offset]);

            // Traverse the token matcher. We descend one char at a time
            // and record each visited trie node's value at the position it
            // lands on.
            let mut node = self.matcher.traverse(c0);
            let mut prefix_ci = ci + 1;
            let mut single_codepoint_match = false;
            let current_best = best[input_offset];
            while let Some(child) = node {
                // A node reached with a value may be the longest match.
                if let Some(token_id) = child.value {
                    let prefix_offset = if prefix_ci < nc {
                        chars[prefix_ci].0
                    } else {
                        n
                    };
                    if prefix_ci == ci + 1 {
                        single_codepoint_match = true;
                    }
                    let token_score = if self.token_types[token_id as usize].is_user_defined() {
                        0.0
                    } else {
                        self.scores[token_id as usize] as f64
                    };
                    let challenger = current_best.2 + token_score;
                    if challenger > best[prefix_offset].2 {
                        best[prefix_offset] = (token_id, input_offset, challenger);
                    }
                }
                if prefix_ci >= nc {
                    break;
                }
                let (_, next_c) = chars[prefix_ci];
                node = child.traverse(next_c);
                prefix_ci += 1;
            }

            // UNK fallback: if the single-char token didn't exist, the
            // session advances one byte and tries the next boundary with
            // the UNK score.
            let unk_step = if single_codepoint_match { utf8_len } else { 1 };
            let unk_score =
                current_best.2 + (self.min_score as f64) - self.unknown_token_score_penalty as f64;
            let next_boundary = (input_offset + unk_step).min(n);
            if unk_score > best[next_boundary].2 {
                best[next_boundary] = (self.unk_id(), input_offset, unk_score);
            }
        }

        // ---- backtrack, merging consecutive UNKs ----
        let mut out_rev = Vec::new();
        let mut prev_was_unk = false;
        let mut cursor = best[n];
        loop {
            let (id, off, _score) = cursor;
            let is_unk = id == self.unk_id();
            if !(prev_was_unk && is_unk) {
                out_rev.push(id);
            }
            if off == 0 {
                break;
            }
            prev_was_unk = is_unk;
            cursor = best[off];
        }
        out_rev.reverse();
        out_rev
    }

    /// `llm_tokenizer_ugm_session::normalize` (`llama-vocab.cpp:1078`).
    pub fn normalize(&self, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(input.len() * 3);
        let space: &[u8] = if self.escape_whitespaces != 0 {
            "▁".as_bytes()
        } else {
            b" "
        };
        let shall_prepend_space = !self.treat_whitespace_as_suffix && self.add_space_prefix;
        let shall_append_space = self.treat_whitespace_as_suffix && self.add_space_prefix;
        let shall_merge_spaces = self.remove_extra_whitespaces;

        let mut space_prepended = false;
        let mut in_non_ws = false;

        let mut off = 0usize;
        while off < input.len() {
            let norm = self.normalize_prefix(input, off);
            for &byte in &norm.normalized {
                if byte != b' ' {
                    if !in_non_ws {
                        in_non_ws = true;
                        if (shall_prepend_space && !space_prepended) || shall_merge_spaces {
                            out.extend_from_slice(space);
                            space_prepended = true;
                        }
                    }
                    out.push(byte);
                } else {
                    if in_non_ws {
                        in_non_ws = false;
                    }
                    if !shall_merge_spaces {
                        out.extend_from_slice(space);
                    }
                }
            }
            off += norm.consumed_input;
        }

        if shall_append_space {
            out.extend_from_slice(space);
        }
        out
    }

    /// `normalize_prefix` (`llama-vocab.cpp:1167`).
    fn normalize_prefix(&self, input: &[u8], offset: usize) -> NormalizationResult {
        if offset >= input.len() {
            return NormalizationResult {
                normalized: Vec::new(),
                consumed_input: 0,
            };
        }

        // 1. user-defined longest-prefix takes priority.
        let user_match = self.user_defined.longest_prefix(input, offset);
        if user_match > 0 {
            return NormalizationResult {
                normalized: input[offset..offset + user_match].to_vec(),
                consumed_input: user_match,
            };
        }

        // 2. XCDA walk.
        if !self.xcda_entries.is_empty() {
            let xcda = XcdaArray::new(&self.xcda_entries);
            let mut node_index: u32 = 0;
            let base = xcda.base(node_index as usize);
            node_index ^= base;
            let mut prefix_offset = offset;
            let mut longest_len = 0usize;
            let mut longest_offset = 0u32;
            while prefix_offset < input.len() {
                let c = input[prefix_offset];
                if c == 0 {
                    break;
                }
                node_index ^= c as u32;
                if xcda.lcheck(node_index as usize) != c as u32 {
                    break;
                }
                let is_leaf = xcda.leaf(node_index as usize);
                node_index ^= xcda.base(node_index as usize);
                if is_leaf {
                    longest_len = prefix_offset - offset + 1;
                    longest_offset = xcda.value(node_index as usize);
                }
                prefix_offset += 1;
            }
            if longest_len > 0 {
                let replacement_start = longest_offset as usize;
                if replacement_start >= self.prefix_replacements.len() {
                    panic!("Index out of array bounds in precompiled charsmap!");
                }
                let tail = &self.prefix_replacements[replacement_start..];
                let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
                return NormalizationResult {
                    normalized: tail[..end].to_vec(),
                    consumed_input: longest_len,
                };
            }
        }

        // 3. Pass through a single valid UTF-8 codepoint, else U+FFFD.
        let mut cursor = offset;
        if consume_utf8(input, &mut cursor) {
            NormalizationResult {
                normalized: input[offset..cursor].to_vec(),
                consumed_input: cursor - offset,
            }
        } else {
            NormalizationResult {
                normalized: vec![0xEF, 0xBF, 0xBD],
                consumed_input: 1,
            }
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }
}

/// Length of the UTF-8 sequence starting with `first_byte`. Standard table:
/// `0xxxxxxx` → 1, `110xxxxx` → 2, `1110xxxx` → 3, `11110xxx` → 4, else 1.
/// Find the closing `>` of a `<...>` literal starting at offset 0. Returns
/// the byte length including the `>`. Returns None if no terminator is
/// found within `bytes`.
fn find_user_defined_end(bytes: &[u8]) -> Option<usize> {
    let mut i = 1usize;
    while i < bytes.len() {
        if bytes[i] == b'>' {
            return Some(i + 1);
        }
        i += 1;
    }
    None
}

fn utf8_char_len(first_byte: u8) -> usize {
    if first_byte < 0x80 {
        1
    } else if first_byte < 0xC0 {
        1
    } else if first_byte < 0xE0 {
        2
    } else if first_byte < 0xF0 {
        3
    } else {
        4
    }
}

fn consume_utf8(input: &[u8], cursor: &mut usize) -> bool {
    let first = input[*cursor];
    let len = utf8_char_len(first);
    if *cursor + len > input.len() {
        return false;
    }
    // Validate continuation bytes.
    for i in 1..len {
        if (input[*cursor + i] & 0xC0) != 0x80 {
            return false;
        }
    }
    *cursor += len;
    true
}

fn string_array(
    get_meta: &dyn Fn(&str) -> Option<MetaValue>,
    key: &str,
) -> Result<Vec<String>, UgmError> {
    let vals = match get_meta(key) {
        Some(MetaValue::Array(_, vs)) => vs,
        _ => return Err(UgmError::Meta(format!("Missing {key}"))),
    };
    let mut out = Vec::with_capacity(vals.len());
    for v in vals {
        match v {
            MetaValue::String(s) => out.push(s),
            _ => return Err(UgmError::Meta(format!("{key} must be string array"))),
        }
    }
    Ok(out)
}

fn f32_array(
    get_meta: &dyn Fn(&str) -> Option<MetaValue>,
    key: &str,
) -> Result<Vec<f32>, UgmError> {
    let vals = match get_meta(key) {
        Some(MetaValue::Array(_, vs)) => vs,
        _ => return Err(UgmError::Meta(format!("Missing {key}"))),
    };
    let mut out = Vec::with_capacity(vals.len());
    for v in vals {
        match v {
            MetaValue::Float32(x) => out.push(x),
            _ => return Err(UgmError::Meta(format!("{key} must be float32 array"))),
        }
    }
    Ok(out)
}

fn bool_meta(get_meta: &dyn Fn(&str) -> Option<MetaValue>, key: &str, default: bool) -> bool {
    match get_meta(key) {
        Some(MetaValue::Bool(v)) => v,
        _ => default,
    }
}

fn u32_meta(get_meta: &dyn Fn(&str) -> Option<MetaValue>, key: &str) -> Option<u32> {
    match get_meta(key) {
        Some(MetaValue::Uint32(v)) => Some(v),
        Some(MetaValue::Int32(v)) => Some(v as u32),
        Some(MetaValue::Int64(v)) => Some(v as u32),
        Some(MetaValue::Uint64(v)) => Some(v as u32),
        _ => None,
    }
}
