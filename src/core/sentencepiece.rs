//! SentencePiece **unigram** tokenizer, byte-exact with `sentencepiece`.
//!
//! Only the `ModelInterface::Encode` path is implemented — the one
//! `DebertaV2Tokenizer` (and therefore GLiNER2) goes through:
//! `spm.encode(text, out_type=str)`. Concretely that is
//! `Normalizer::Normalize` → `unigram::Model::PopulateNodes` → `Lattice::Viterbi`
//! → the byte-fallback expansion in `PopulateSentencePieceText`.
//!
//! Reference sources (sentencepiece 0.1.99 / 0.2.2):
//! - `src/normalizer.cc` — the `nmt_nfkc` charsmap transducer
//! - `src/unigram_model.cc` — lattice construction and Viterbi
//! - `src/sentencepiece_processor.cc` — byte-fallback expansion
//! - `third_party/darts_clone/darts.h` — the double-array trie format
//!
//! ## Deviations from the C++ (and why they are safe)
//!
//! - **Byte offsets instead of character offsets in the lattice.** The lattice
//!   is indexed by *character* position in C++, which matters in exactly one
//!   way: the three bytes of `▁` (U+2581) are a single position, so the byte
//!   offsets 1 and 2 inside it must never receive an UNK fallback node. This
//!   port therefore iterates only character starts while storing byte offsets.
//!   Everything else is safe: every piece in the trie is a valid UTF-8 string
//!   (BYTE pieces live in `reserved_id_map_` and never enter the trie), so
//!   matches always land on character boundaries. The other place that reads
//!   character counts — `has_single_node` and the USER_DEIGNED score — is
//!   reconstructed from the UTF-8 length of the character at the start.
//! - **Darts "results" array.** The normalizer only ever keeps the *longest*
//!   match, so the search tracks it inline instead of filling a 64-entry buffer.
//! - **A plain byte trie instead of a double array.** Darts exists to make the
//!   lookup compact; the prefix semantics are identical.
//! - **No `PrefixMatcher`.** It is built from USER_DEFINED pieces only, and the
//!   DeBERTa-v3 vocabulary has none.

use std::collections::HashMap;
use std::ops::Range;

/// Piece type tags from `sentencepiece_model.proto`.
const TYPE_NORMAL: u8 = 1;
const TYPE_UNKNOWN: u8 = 2;
const TYPE_USER_DEFINED: u8 = 4;
const TYPE_UNUSED: u8 = 5;

/// `kUnkPenalty` from `unigram_model.cc:40`.
const UNK_PENALTY: f32 = 10.0;

/// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece's whitespace escape.
const SPACE_SYMBOL: [u8; 3] = [0xE2, 0x96, 0x81];
/// U+FFFD REPLACEMENT CHARACTER, substituted for malformed UTF-8.
const REPLACEMENT_CHAR: [u8; 3] = [0xEF, 0xBF, 0xBD];

/// A SentencePiece unigram model plus its normalizer.
#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    pieces: Vec<Vec<u8>>,
    scores: Vec<f32>,
    types: Vec<u8>,
    /// piece bytes -> id for `NORMAL` / `USER_DEFINED` / `UNUSED` (trie source).
    normal_ids: HashMap<Vec<u8>, u32>,
    /// piece bytes -> id for `CONTROL` / `UNKNOWN` / `BYTE`.
    reserved_ids: HashMap<Vec<u8>, u32>,
    trie: PieceTrie,
    unk_id: u32,
    min_score: f32,
    max_score: f32,
    byte_fallback: bool,
    normalizer: Normalizer,
}

impl SentencePieceTokenizer {
    /// Build from the raw `ModelProto` bytes of a `*.model` file.
    ///
    /// The runtime path is [`SentencePieceTokenizer::from_parts`] because the
    /// converter bakes the model into GGUF metadata; this constructor is what
    /// the parity tests read directly.
    pub fn from_model_proto(bytes: &[u8]) -> Result<Self, String> {
        let mut pieces: Vec<Vec<u8>> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        let mut types: Vec<u8> = Vec::new();
        let mut normalizer = Normalizer::parse(&[])?;
        let mut byte_fallback = false;
        for (field, value) in ProtoReader::new(bytes) {
            match (field, value) {
                (1, Value::Bytes(msg)) => {
                    let (piece, score, kind) = parse_sentence_piece(msg)?;
                    pieces.push(piece);
                    scores.push(score);
                    types.push(kind);
                }
                (2, Value::Bytes(trainer)) => {
                    if let Some(Value::U64(flag)) = find_field(trainer, TRAINER_BYTE_FALLBACK) {
                        byte_fallback = flag != 0;
                    }
                }
                (3, Value::Bytes(spec)) => normalizer = Normalizer::parse(spec)?,
                _ => {}
            }
        }
        if pieces.is_empty() {
            return Err("sentencepiece model has no pieces".into());
        }
        Self::from_parts(pieces, scores, types, byte_fallback, normalizer)
    }

    /// Build from explicit parts, as stored in GGUF metadata.
    pub fn from_parts(
        pieces: Vec<Vec<u8>>,
        scores: Vec<f32>,
        types: Vec<u8>,
        byte_fallback: bool,
        normalizer: Normalizer,
    ) -> Result<Self, String> {
        if pieces.len() != scores.len() || pieces.len() != types.len() {
            return Err("sentencepiece pieces/scores/types length mismatch".into());
        }
        let mut normal_ids = HashMap::with_capacity(pieces.len());
        let mut reserved_ids = HashMap::new();
        let mut unk_id = u32::MAX;
        let mut min_score = f32::MAX;
        let mut max_score = -f32::MAX;
        for (index, piece) in pieces.iter().enumerate() {
            if piece.is_empty() {
                return Err(format!("sentencepiece piece {index} is empty"));
            }
            let id = index as u32;
            match types[index] {
                TYPE_NORMAL | TYPE_USER_DEFINED | TYPE_UNUSED => {
                    if normal_ids.insert(piece.clone(), id).is_some() {
                        return Err(format!(
                            "duplicate piece {:?}",
                            String::from_utf8_lossy(piece)
                        ));
                    }
                    if types[index] == TYPE_NORMAL {
                        min_score = min_score.min(scores[index]);
                        max_score = max_score.max(scores[index]);
                    }
                }
                _ => {
                    if reserved_ids.insert(piece.clone(), id).is_some() {
                        return Err(format!(
                            "duplicate piece {:?}",
                            String::from_utf8_lossy(piece)
                        ));
                    }
                    if types[index] == TYPE_UNKNOWN {
                        if unk_id != u32::MAX {
                            return Err("sentencepiece model has more than one UNK piece".into());
                        }
                        unk_id = id;
                    }
                }
            }
        }
        if unk_id == u32::MAX {
            return Err("sentencepiece model has no UNK piece".into());
        }
        if min_score == f32::MAX {
            return Err("sentencepiece model has no NORMAL piece".into());
        }
        let mut trie = PieceTrie::new();
        for (piece, &id) in &normal_ids {
            trie.insert(piece, id);
        }
        Ok(Self {
            pieces,
            scores,
            types,
            normal_ids,
            reserved_ids,
            trie,
            unk_id,
            min_score,
            max_score,
            byte_fallback,
            normalizer,
        })
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    pub fn piece(&self, id: u32) -> Option<&[u8]> {
        self.pieces.get(id as usize).map(Vec::as_slice)
    }

    pub fn score(&self, id: u32) -> Option<f32> {
        self.scores.get(id as usize).copied()
    }

    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// `ModelInterface::PieceToId` — reserved ids win, then trie pieces, then UNK.
    pub fn piece_to_id(&self, piece: &[u8]) -> u32 {
        if let Some(&id) = self.reserved_ids.get(piece) {
            return id;
        }
        if let Some(&id) = self.normal_ids.get(piece) {
            return id;
        }
        self.unk_id
    }

    /// `SentencePieceProcessor::Encode(text, out_type=str)` -> piece strings.
    pub fn encode_pieces(&self, text: &str) -> Vec<String> {
        self.encode_ids(text)
            .into_iter()
            .map(|id| String::from_utf8_lossy(self.pieces[id as usize].as_slice()).into_owned())
            .collect()
    }

    /// `SentencePieceProcessor::Encode(text, out_type=int)`.
    pub fn encode_ids(&self, text: &str) -> Vec<u32> {
        let normalized = self.normalizer.normalize(text.as_bytes());
        if normalized.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(normalized.len());
        for (id, surface) in self.viterbi(&normalized) {
            if self.types[id as usize] == TYPE_UNKNOWN && self.byte_fallback {
                for &byte in &normalized[surface] {
                    out.push(self.byte_piece_id(byte));
                }
            } else {
                out.push(id);
            }
        }
        out
    }

    fn byte_piece_id(&self, byte: u8) -> u32 {
        self.piece_to_id(format!("<0x{byte:02X}>").as_bytes())
    }

    /// `PopulateNodes` + `Lattice::Viterbi`, returning `(piece id, surface)`.
    ///
    /// The reference lattice is indexed by *character* position; here it is
    /// indexed by byte offset, so `begin` only ever visits character starts.
    /// Byte offsets inside a multi-byte character (e.g. the three bytes of
    /// `▁`) are not lattice positions and must not get UNK fallback nodes.
    fn viterbi(&self, sentence: &[u8]) -> Vec<(u32, Range<usize>)> {
        let len = sentence.len();
        let mut nodes: Vec<Node> = Vec::new();
        let mut begin_nodes: Vec<Vec<usize>> = vec![Vec::new(); len + 1];
        let mut end_nodes: Vec<Vec<usize>> = vec![Vec::new(); len + 1];
        let unk_score = self.min_score - UNK_PENALTY;

        let mut char_starts = Vec::with_capacity(len);
        let mut cursor = 0usize;
        while cursor < len {
            char_starts.push(cursor);
            cursor += utf8_char_len(sentence, cursor);
        }

        for &begin in &char_starts {
            let mut has_single_node = false;
            for (id, piece_len) in self.trie.common_prefix(&sentence[begin..]) {
                let kind = self.types[id as usize];
                if kind == TYPE_UNUSED {
                    continue;
                }
                let score = if kind == TYPE_USER_DEFINED {
                    // `length * max_score_ - 0.1`, with `length` in characters.
                    char_len(sentence, begin, piece_len) as f32 * self.max_score - 0.1
                } else {
                    self.scores[id as usize]
                };
                push_node(
                    &mut nodes,
                    &mut begin_nodes,
                    &mut end_nodes,
                    begin,
                    piece_len,
                    id,
                    score,
                );
                if piece_len == utf8_char_len(sentence, begin) {
                    has_single_node = true;
                }
            }
            if !has_single_node {
                // `lattice->Insert(begin_pos, 1)` with the UNK id: one character.
                let piece_len = utf8_char_len(sentence, begin);
                push_node(
                    &mut nodes,
                    &mut begin_nodes,
                    &mut end_nodes,
                    begin,
                    piece_len,
                    self.unk_id,
                    unk_score,
                );
            }
        }

        // BOS lands in `end_nodes[0]` only and EOS in `begin_nodes[len]` only.
        // `Viterbi` walks the path through `begin_nodes(len)[0]`, so EOS must be
        // pushed before anything else can land at `len` — and nothing else does,
        // because `PopulateNodes` never starts a node at `len`.
        let bos = nodes.len();
        nodes.push(Node {
            begin: 0,
            end: 0,
            id: 0,
            score: 0.0,
            backtrace: 0.0,
            prev: None,
        });
        end_nodes[0].push(bos);
        let eos = nodes.len();
        nodes.push(Node {
            begin: len,
            end: len,
            id: 0,
            score: 0.0,
            backtrace: 0.0,
            prev: None,
        });
        begin_nodes[len].push(eos);

        for pos in 0..=len {
            for &right in &begin_nodes[pos] {
                let mut best: Option<usize> = None;
                let mut best_score = 0.0f32;
                for &left in &end_nodes[pos] {
                    let score = nodes[left].backtrace + nodes[right].score;
                    // `best_node == nullptr || score > best_score`: the first
                    // candidate always wins, later ones must be strictly better.
                    if best.is_none() || score > best_score {
                        best = Some(left);
                        best_score = score;
                    }
                }
                // Unreachable: BOS covers position 0 and the UNK fallback keeps
                // `end_nodes` non-empty everywhere else.
                let Some(best) = best else {
                    return Vec::new();
                };
                nodes[right].prev = Some(best);
                nodes[right].backtrace = best_score;
            }
        }

        let mut path = Vec::new();
        let mut cursor = nodes[eos].prev;
        while let Some(index) = cursor {
            if nodes[index].prev.is_none() {
                break;
            }
            path.push(index);
            cursor = nodes[index].prev;
        }
        path.reverse();
        path.into_iter()
            .map(|index| {
                let node = &nodes[index];
                (node.id, node.begin..node.end)
            })
            .collect()
    }
}

struct Node {
    /// Byte offset of the first byte of this node.
    begin: usize,
    /// Byte offset one past the last byte.
    end: usize,
    id: u32,
    score: f32,
    backtrace: f32,
    prev: Option<usize>,
}

fn push_node(
    nodes: &mut Vec<Node>,
    begin_nodes: &mut [Vec<usize>],
    end_nodes: &mut [Vec<usize>],
    begin: usize,
    len: usize,
    id: u32,
    score: f32,
) {
    let index = nodes.len();
    nodes.push(Node {
        begin,
        end: begin + len,
        id,
        score,
        backtrace: 0.0,
        prev: None,
    });
    begin_nodes[begin].push(index);
    end_nodes[begin + len].push(index);
}

/// Length in bytes of the UTF-8 character starting at `pos`; a malformed byte
/// counts as one, matching `string_util::IsValidDecodeUTF8`.
fn utf8_char_len(bytes: &[u8], pos: usize) -> usize {
    let Some(first) = bytes.get(pos) else {
        return 1;
    };
    let len = match first {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => 1,
    };
    if pos + len > bytes.len() || std::str::from_utf8(&bytes[pos..pos + len]).is_err() {
        return 1;
    }
    len
}

/// Character count of the `byte_len`-byte slice starting at `pos`.
fn char_len(bytes: &[u8], pos: usize, byte_len: usize) -> usize {
    let end = pos + byte_len;
    let mut count = 0;
    let mut cursor = pos;
    while cursor < end {
        cursor += utf8_char_len(bytes, cursor);
        count += 1;
    }
    count
}

/// Byte trie over the model's normal pieces, replacing Darts' double array.
#[derive(Debug, Clone)]
struct PieceTrie {
    /// `(node << 8) | byte` -> child node.
    edges: HashMap<u64, u32>,
    /// Piece id per node, `u32::MAX` when the node is not a piece.
    ///
    /// Index 0 is the root and is never a piece, so [`PieceTrie::new`] seeds it
    /// with `u32::MAX`. Letting the first child take index 0 instead aliases it
    /// with the root: any later piece whose walk returns to node 0 on its final
    /// byte then overwrites the root's sentinel with its own id. Which piece
    /// lands there depends on `normal_ids`' HashMap iteration order, so the
    /// corruption was per-process and showed up as an occasional wrong
    /// segmentation — e.g. `down. Can` cutting to `666.` instead of `.`.
    values: Vec<u32>,
}

impl PieceTrie {
    /// An empty trie holding only the root.
    fn new() -> Self {
        PieceTrie {
            edges: HashMap::new(),
            values: vec![u32::MAX],
        }
    }

    fn insert(&mut self, piece: &[u8], id: u32) {
        let mut node = 0u32;
        for &byte in piece {
            let key = (u64::from(node) << 8) | u64::from(byte);
            node = match self.edges.get(&key) {
                Some(&child) => child,
                None => {
                    let child = self.values.len() as u32;
                    self.values.push(u32::MAX);
                    self.edges.insert(key, child);
                    child
                }
            };
        }
        self.values[node as usize] = id;
    }

    /// All `(piece id, byte length)` whose piece is a prefix of `input`,
    /// shortest first — the order `commonPrefixSearch` reports.
    fn common_prefix(&self, input: &[u8]) -> Vec<(u32, usize)> {
        let mut found = Vec::new();
        let mut node = 0u32;
        for (index, &byte) in input.iter().enumerate() {
            let key = (u64::from(node) << 8) | u64::from(byte);
            match self.edges.get(&key) {
                Some(&child) => node = child,
                None => break,
            }
            let value = self.values[node as usize];
            if value != u32::MAX {
                found.push((value, index + 1));
            }
        }
        found
    }
}

// ---------------------------------------------------------------------------
// Normalizer
// ---------------------------------------------------------------------------

const NORM_PRECOMPILED_CHARSMAP: u32 = 2;
const NORM_ADD_DUMMY_PREFIX: u32 = 3;
const NORM_REMOVE_EXTRA_WHITESPACES: u32 = 4;
const NORM_ESCAPE_WHITESPACES: u32 = 5;
const TRAINER_BYTE_FALLBACK: u32 = 35;

/// The `NormalizerSpec` half of a SentencePiece model.
#[derive(Debug, Clone)]
pub struct Normalizer {
    charsmap: Option<CharsMap>,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    treat_whitespace_as_suffix: bool,
}

impl Default for Normalizer {
    /// Protobuf defaults: the three `true` flags are the spec defaults, and an
    /// absent `NormalizerSpec` leaves them on.
    fn default() -> Self {
        Normalizer {
            charsmap: None,
            add_dummy_prefix: true,
            remove_extra_whitespaces: true,
            escape_whitespaces: true,
            treat_whitespace_as_suffix: false,
        }
    }
}

impl Normalizer {
    /// Parse a serialized `NormalizerSpec`.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut normalizer = Normalizer::default();
        for (field, value) in ProtoReader::new(bytes) {
            match (field, value) {
                (NORM_PRECOMPILED_CHARSMAP, Value::Bytes(blob)) => {
                    normalizer.charsmap = Some(CharsMap::parse(blob)?);
                }
                (NORM_ADD_DUMMY_PREFIX, Value::U64(flag)) => {
                    normalizer.add_dummy_prefix = flag != 0;
                }
                (NORM_REMOVE_EXTRA_WHITESPACES, Value::U64(flag)) => {
                    normalizer.remove_extra_whitespaces = flag != 0;
                }
                (NORM_ESCAPE_WHITESPACES, Value::U64(flag)) => {
                    normalizer.escape_whitespaces = flag != 0;
                }
                _ => {}
            }
        }
        Ok(normalizer)
    }

    /// Build from the parts the GGUF converter stores, so the runtime never
    /// needs the `spm.model` sidecar.
    pub fn from_parts(
        charsmap: &[u8],
        add_dummy_prefix: bool,
        remove_extra_whitespaces: bool,
        escape_whitespaces: bool,
        treat_whitespace_as_suffix: bool,
    ) -> Result<Self, String> {
        Ok(Normalizer {
            charsmap: if charsmap.is_empty() {
                None
            } else {
                Some(CharsMap::parse(charsmap)?)
            },
            add_dummy_prefix,
            remove_extra_whitespaces,
            escape_whitespaces,
            treat_whitespace_as_suffix,
        })
    }

    /// Builder hook for `treat_whitespace_as_suffix`, which lives on the
    /// `TrainerSpec` rather than the `NormalizerSpec`.
    pub fn with_treat_whitespace_as_suffix(mut self, value: bool) -> Self {
        self.treat_whitespace_as_suffix = value;
        self
    }

    /// `Normalizer::Normalize`.
    pub fn normalize(&self, input: &[u8]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::with_capacity(input.len() * 3);
        let mut rest = input;
        if !input.is_empty() && self.remove_extra_whitespaces {
            while !rest.is_empty() {
                let (replacement, len) = self.normalize_prefix(rest);
                if replacement != b" " {
                    break;
                }
                rest = &rest[len..];
            }
        }
        if rest.is_empty() {
            return out;
        }

        if !self.treat_whitespace_as_suffix && self.add_dummy_prefix {
            self.push_space(&mut out);
        }

        let mut is_prev_space = self.remove_extra_whitespaces;
        while !rest.is_empty() {
            let (replacement, len) = self.normalize_prefix(rest);
            let mut piece = replacement;
            while is_prev_space && piece.starts_with(b" ") {
                piece = &piece[1..];
            }
            if !piece.is_empty() {
                if self.escape_whitespaces {
                    for &byte in piece {
                        if byte == b' ' {
                            out.extend_from_slice(&SPACE_SYMBOL);
                        } else {
                            out.push(byte);
                        }
                    }
                } else {
                    out.extend_from_slice(piece);
                }
                is_prev_space = piece.ends_with(b" ");
            }
            rest = &rest[len..];
            if !self.remove_extra_whitespaces {
                is_prev_space = false;
            }
        }

        if self.remove_extra_whitespaces {
            let marker: &[u8] = if self.escape_whitespaces {
                &SPACE_SYMBOL
            } else {
                b" "
            };
            while out.len() >= marker.len() && out.ends_with(marker) {
                let new_len = out.len() - marker.len();
                out.truncate(new_len);
            }
        }

        if self.treat_whitespace_as_suffix && self.add_dummy_prefix {
            self.push_space(&mut out);
        }
        out
    }

    fn push_space(&self, out: &mut Vec<u8>) {
        if self.escape_whitespaces {
            out.extend_from_slice(&SPACE_SYMBOL);
        } else {
            out.push(b' ');
        }
    }

    /// `Normalizer::NormalizePrefix` — the longest matching charsmap rule, else
    /// the input's first valid UTF-8 character.
    fn normalize_prefix<'a>(&'a self, input: &'a [u8]) -> (&'a [u8], usize) {
        if input.is_empty() {
            return (&[], 0);
        }
        if let Some(charsmap) = &self.charsmap {
            if let Some((value, len)) = charsmap.common_prefix(input) {
                if let Some(replacement) = charsmap.segment(value) {
                    return (replacement, len);
                }
            }
        }
        let len = utf8_char_len(input, 0);
        if std::str::from_utf8(&input[..len]).is_err() {
            (&REPLACEMENT_CHAR, 1)
        } else {
            (&input[..len], len)
        }
    }
}

/// The precompiled `nmt_nfkc` char map: a Darts double array over input bytes
/// plus the NUL-delimited pool of replacement strings it points into.
#[derive(Debug, Clone)]
struct CharsMap {
    units: Vec<u32>,
    pool: Vec<u8>,
}

impl CharsMap {
    /// `Normalizer::DecodePrecompiledCharsMap` + `Darts::DoubleArray::set_array`.
    fn parse(blob: &[u8]) -> Result<Self, String> {
        if blob.len() <= 4 {
            return Err("precompiled charsmap is truncated".into());
        }
        let trie_size = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
        if trie_size >= blob.len() || trie_size % 4 != 0 {
            return Err("precompiled charsmap has a bad trie size".into());
        }
        let units: Vec<u32> = blob[4..4 + trie_size]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let pool = blob[4 + trie_size..].to_vec();
        Ok(CharsMap { units, pool })
    }

    /// `Darts::DoubleArray::commonPrefixSearch`, reduced to the longest hit.
    ///
    /// The value is a byte offset into the replacement pool, exactly as
    /// `NormalizePrefix` uses it (`string_view(&normalized_[longest_value])`),
    /// so it is resolved by scanning to the next NUL.
    fn common_prefix(&self, input: &[u8]) -> Option<(usize, usize)> {
        let mut node = unit_offset(self.units[0]) as usize;
        let mut best: Option<(usize, usize)> = None;
        for (index, &byte) in input.iter().enumerate() {
            node ^= usize::from(byte);
            let unit = *self.units.get(node)?;
            if unit_label(unit) != u32::from(byte) {
                return best;
            }
            node ^= unit_offset(unit) as usize;
            if unit_has_leaf(unit) {
                let value = self.units.get(node).map_or(0, |unit| unit_value(*unit)) as usize;
                if best.is_none_or(|(_, len)| index + 1 > len) {
                    best = Some((value, index + 1));
                }
            }
        }
        best
    }

    fn segment(&self, offset: usize) -> Option<&[u8]> {
        let rest = self.pool.get(offset..)?;
        let len = rest
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(rest.len());
        Some(&rest[..len])
    }
}

/// `DoubleArrayUnit::offset()`: `(unit >> 10) << ((unit & (1 << 9)) >> 6)`.
fn unit_offset(unit: u32) -> u32 {
    (unit >> 10) << ((unit & (1 << 9)) >> 6)
}

/// `DoubleArrayUnit::label()`: `unit & ((1 << 31) | 0xFF)`.
fn unit_label(unit: u32) -> u32 {
    unit & ((1 << 31) | 0xFF)
}

/// `DoubleArrayUnit::has_leaf()`: `((unit >> 8) & 1) == 1`.
fn unit_has_leaf(unit: u32) -> bool {
    (unit >> 8) & 1 == 1
}

/// `DoubleArrayUnit::value()`: `unit & ((1 << 31) - 1)`.
fn unit_value(unit: u32) -> u32 {
    unit & ((1 << 31) - 1)
}

// ---------------------------------------------------------------------------
// Minimal protobuf wire-format reader
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Value<'a> {
    U64(u64),
    Bytes(&'a [u8]),
    /// A 32-bit little-endian scalar, used for `SentencePiece.score`.
    F32(f32),
}

struct ProtoReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        ProtoReader { buf, pos: 0 }
    }
}

impl<'a> Iterator for ProtoReader<'a> {
    type Item = (u32, Value<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.buf.len() {
            return None;
        }
        let (key, next) = read_varint(self.buf, self.pos).ok()?;
        let field = (key >> 3) as u32;
        match key & 7 {
            0 => {
                let (value, next) = read_varint(self.buf, next).ok()?;
                self.pos = next;
                Some((field, Value::U64(value)))
            }
            1 => {
                let end = next + 8;
                if end > self.buf.len() {
                    return None;
                }
                self.pos = end;
                Some((field, Value::U64(0)))
            }
            2 => {
                let (len, after_len) = read_varint(self.buf, next).ok()?;
                let end = after_len.checked_add(len as usize)?;
                if end > self.buf.len() {
                    return None;
                }
                let slice = &self.buf[after_len..end];
                self.pos = end;
                Some((field, Value::Bytes(slice)))
            }
            5 => {
                let end = next + 4;
                if end > self.buf.len() {
                    return None;
                }
                let raw = &self.buf[next..end];
                self.pos = end;
                Some((
                    field,
                    Value::F32(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
                ))
            }
            // Groups (3, 4) and the deprecated 6/7 wire types never appear in a
            // SentencePiece model; stop rather than mis-parse.
            _ => {
                self.pos = self.buf.len();
                None
            }
        }
    }
}

fn find_field(buf: &[u8], want: u32) -> Option<Value<'_>> {
    ProtoReader::new(buf)
        .find(|(field, _)| *field == want)
        .map(|(_, value)| value)
}

fn read_varint(buf: &[u8], mut pos: usize) -> Result<(u64, usize), String> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *buf.get(pos).ok_or("truncated varint")?;
        pos += 1;
        result |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, pos));
        }
        shift += 7;
        if shift >= 64 {
            return Err("varint overflows 64 bits".into());
        }
    }
}

/// `ModelProto.SentencePiece` -> `(piece, score, type)`.
fn parse_sentence_piece(buf: &[u8]) -> Result<(Vec<u8>, f32, u8), String> {
    let mut piece = Vec::new();
    let mut score = 0.0f32;
    let mut kind = TYPE_NORMAL;
    for (field, value) in ProtoReader::new(buf) {
        match (field, value) {
            (1, Value::Bytes(raw)) => piece = raw.to_vec(),
            (2, Value::F32(raw)) => score = raw,
            (3, Value::U64(raw)) => kind = raw as u8,
            _ => {}
        }
    }
    Ok((piece, score, kind))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_char_len_handles_multibyte() {
        assert_eq!(utf8_char_len("abc".as_bytes(), 0), 1);
        assert_eq!(utf8_char_len("café".as_bytes(), 3), 2);
        assert_eq!(utf8_char_len("日本".as_bytes(), 0), 3);
        assert_eq!(utf8_char_len("😀x".as_bytes(), 0), 4);
        assert_eq!(utf8_char_len(&[0xFF], 0), 1);
    }

    #[test]
    fn piece_trie_reports_prefixes_in_order() {
        let mut trie = PieceTrie::new();
        trie.insert(b"ab", 7);
        trie.insert(b"abc", 8);
        trie.insert(b"abd", 9);
        assert_eq!(trie.common_prefix(b"abcde"), vec![(7, 2), (8, 3)]);
        assert_eq!(trie.common_prefix(b"abx"), vec![(7, 2)]);
        assert!(trie.common_prefix(b"zz").is_empty());
    }

    #[test]
    fn piece_trie_root_never_holds_a_piece() {
        // The regression that let the root alias the first child. `root-first`
        // takes node 0's only key, so a later piece *ending* on that byte at the
        // root level lands back on node 0. Before the root was seeded, that
        // overwrote the root sentinel with the later piece's id, and which piece
        // won depended on the caller's iteration order.
        let mut trie = PieceTrie::new();
        trie.insert(b"u", 1);
        trie.insert(b"uu", 2);
        trie.insert(b"bu", 3);
        assert_eq!(trie.values[0], u32::MAX, "root must stay empty");
        assert_eq!(trie.common_prefix(b"u"), vec![(1, 1)]);
        assert_eq!(trie.common_prefix(b"uu"), vec![(1, 1), (2, 2)]);
        assert_eq!(trie.common_prefix(b"bu"), vec![(3, 2)]);
        // A byte sharing the root's key must still resolve to its own node.
        assert_eq!(trie.common_prefix(b"b"), Vec::new());
    }

    #[test]
    fn piece_trie_build_is_order_independent() {
        // Same pieces, different insertion order, identical answers. The aliasing
        // above only showed up because the real caller inserts from a HashMap,
        // whose order is randomised per process.
        let pieces: [&[u8]; 6] = [b"u", b"uu", b"bu", b"a", b"ab", b"abc"];
        let build = |order: &[usize]| {
            let mut trie = PieceTrie::new();
            for &i in order {
                trie.insert(pieces[i], i as u32 + 1);
            }
            trie
        };
        let forward = build(&[0, 1, 2, 3, 4, 5]);
        let backward = build(&[5, 4, 3, 2, 1, 0]);
        for probe in [&b"uu"[..], b"bu", b"abc", b"abx", b"u"] {
            assert_eq!(
                forward.common_prefix(probe),
                backward.common_prefix(probe),
                "prefix search for {probe:?} depends on insertion order"
            );
        }
    }

    #[test]
    fn normalizer_defaults_escape_whitespace() {
        let normalizer = Normalizer::default();
        assert_eq!(
            normalizer.normalize(b"hello world"),
            "\u{2581}hello\u{2581}world".as_bytes()
        );
        assert_eq!(normalizer.normalize(b"  hi  "), "\u{2581}hi".as_bytes());
    }
}
