//! GLiNER2 input construction: the word splitter, the schema prompt, and the
//! marker positions the classifier reads.
//!
//! Ported from `gliner2/processing/word_splitter.py` (`WhitespaceTokenSplitter`)
//! and `gliner2/processor.py` (`_transform_schema` / `_format_input_with_mapping`).
//! Both are load-bearing for parity, down to the punctuation of the prompt:
//!
//! ```text
//! ▁(  [P]  ▁intent  ▁(  [L]  ▁order _ status  [L]  ▁refund _ request  ...  ▁)  ▁)  [SEP_TEXT]  text…
//! ```
//!
//! Three details are easy to get wrong and are asserted by the tests:
//!
//! 1. **No `[CLS]`/`[SEP]`.** `input_ids` is `convert_tokens_to_ids(subwords)`
//!    straight onto the concatenation; `build_inputs_with_special_tokens` is
//!    never called, so the encoder sees the bare schema-then-text sequence.
//! 2. **`[SEP_STRUCT]` disappears for a single task.** The combiner appends it
//!    after every schema and then pops the last one, so a one-task prompt has
//!    no separator at all and only a multi-task prompt carries one.
//! 3. **The label rows are the `[L]` markers themselves.** `embs[1:]` drops the
//!    `[P]` prompt row and keeps the marker rows, in declaration order.

use regex::Regex;
use std::sync::OnceLock;

/// `SchemaTransformer.P_TOKEN`.
pub const P_TOKEN: &str = "[P]";
/// `SchemaTransformer.L_TOKEN`.
pub const L_TOKEN: &str = "[L]";
/// `SchemaTransformer.E_TOKEN` — the child marker for extractive (`entities`)
/// schema groups, i.e. one query per field.
pub const E_TOKEN: &str = "[E]";
/// `SchemaTransformer.C_TOKEN` — the child marker for classification groups.
pub const C_TOKEN: &str = "[C]";
/// `SchemaTransformer.R_TOKEN` — the child marker for relation groups.
pub const R_TOKEN: &str = "[R]";
/// `SchemaTransformer.SEP_TEXT`.
pub const SEP_TEXT: &str = "[SEP_TEXT]";
/// `SchemaTransformer.SEP_STRUCT`.
pub const SEP_STRUCT: &str = "[SEP_STRUCT]";
/// `SchemaTransformer.DESC_TOKEN`.
pub const DESC_TOKEN: &str = "[DESCRIPTION]";
/// `SchemaTransformer.EXAMPLE_TOKEN`.
pub const EXAMPLE_TOKEN: &str = "[EXAMPLE]";
/// `SchemaTransformer.OUTPUT_TOKEN`.
pub const OUTPUT_TOKEN: &str = "[OUTPUT]";

/// The ten schema markers GLiNER2 registers as additional special tokens.
pub const SPECIAL_TOKENS: [&str; 10] = [
    SEP_STRUCT,
    SEP_TEXT,
    P_TOKEN,
    "[C]",
    "[E]",
    "[R]",
    L_TOKEN,
    EXAMPLE_TOKEN,
    OUTPUT_TOKEN,
    DESC_TOKEN,
];

/// `_RESERVED` from `gliner2/classification/schema.py`. A label or prompt
/// containing one of these would be mis-parsed as a marker, silently shifting
/// every logit onto the wrong label.
pub const RESERVED: [&str; 10] = [
    P_TOKEN,
    L_TOKEN,
    "[C]",
    "[E]",
    "[R]",
    DESC_TOKEN,
    EXAMPLE_TOKEN,
    OUTPUT_TOKEN,
    "(",
    ")",
];

fn word_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        // `\w` is spelled `[\p{L}\p{N}_]` rather than left as `\w` because the
        // `regex` crate's `\w` is `[\p{Alphabetic}\p{M}\p{Nd}\p{Join_Control}\p{Pc}]`
        // and Python's is "alphanumeric as `str.isalnum()` reports it, plus the
        // underscore" — that is, categories L* and N* plus `_`. The two differ in
        // both directions: `\p{M}` matches combining and spacing marks that
        // Python rejects (so `cafe` + U+0301 stays one token here and splits into
        // two there), and `\p{Nd}` misses Nl and No that Python accepts.
        const PY_WORD: &str = r"[\p{L}\p{N}_]";
        Regex::new(&format!(
            concat!(
                r"(?:https?://[^\s]+|www\.[^\s]+)",
                r"|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{{2,}}",
                r"|@[a-z0-9_]+",
                r"|{PY_WORD}+(?:[-_]{PY_WORD}+)*",
                r"|\S",
            ),
            PY_WORD = PY_WORD,
        ))
        .expect("word splitter pattern")
    })
}

fn char_level_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN
        .get_or_init(|| Regex::new(r"[A-Za-z0-9@._\-+]+|\S").expect("char-level splitter pattern"))
}

/// Which word segmentation to run, as `word_splitter` names them.
///
/// The two differ on one axis: which characters may share a token. The
/// whitespace splitter's `\w` is Unicode-aware, so it keeps `café` and CJK runs
/// whole; the char splitter's class is ASCII-only, so `café` becomes `caf` +
/// `é` and CJK becomes one token per character. That is the point of it — a
/// whitespace splitter cannot find word boundaries in a language that does not
/// delimit them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WordSplitter {
    /// `word_splitter="whitespace"`, and the default.
    #[default]
    Whitespace,
    /// `word_splitter="char"`.
    CharLevel,
}

impl WordSplitter {
    /// `resolve_word_splitter`: the built-in names, with `whitespace` as the
    /// default for an absent setting.
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "whitespace" => Ok(WordSplitter::Whitespace),
            "char" => Ok(WordSplitter::CharLevel),
            other => Err(format!(
                "Unknown word_splitter {other:?}. Supported names: 'char', 'whitespace'."
            )),
        }
    }

    fn pattern(self) -> &'static Regex {
        match self {
            WordSplitter::Whitespace => word_pattern(),
            WordSplitter::CharLevel => char_level_pattern(),
        }
    }
}

/// One word plus half-open **code point** offsets into the original string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordSpan {
    /// The matched text, lower-cased when the caller asked for it.
    pub token: String,
    pub start: usize,
    pub end: usize,
}

/// `SchemaTransformer._normalize_text`: collation expects terminal punctuation.
pub fn normalize_text(text: &str) -> String {
    if text.is_empty() {
        return ".".to_string();
    }
    if text.ends_with('.') || text.ends_with('!') || text.ends_with('?') {
        text.to_string()
    } else {
        format!("{text}.")
    }
}

/// `WhitespaceTokenSplitter`: the first alternative that matches at each
/// position wins, and the token value is lower-cased.
pub fn split_words(text: &str) -> Vec<String> {
    split_words_with(text, WordSplitter::Whitespace)
}

/// Word tokens under `splitter`, lower-cased.
pub fn split_words_with(text: &str, splitter: WordSplitter) -> Vec<String> {
    word_spans(text, splitter, true)
        .into_iter()
        .map(|span| span.token)
        .collect()
}

/// Word tokens with their offsets, as both reference splitters yield them.
///
/// The offsets are **code point** indices, matching Python `str` slicing, not
/// the byte offsets the `regex` crate reports. For `中华人民共和国` the
/// reference's second token is `(1, 2)`; the byte range for the same token is
/// `3..6`. Every caller-visible offset in the reference — span character
/// offsets, chunk boundaries — is a code point index, so the conversion is not
/// optional.
///
/// Lower-casing is applied to the token value only, never to the source text
/// first: Unicode case folding can change length (`"İ".lower()` is `"i̇"`, two
/// code points), which would shift every later offset.
pub fn word_spans(text: &str, splitter: WordSplitter, lower: bool) -> Vec<WordSpan> {
    let pattern = splitter.pattern();
    let mut spans = Vec::new();
    // Byte position of the last match end, so the code point offset advances by
    // counting only the text between matches.
    let mut cursor = 0usize;
    let mut code_point = 0usize;
    for mat in pattern.find_iter(text) {
        code_point += text[cursor..mat.start()].chars().count();
        let token = &text[mat.start()..mat.end()];
        let width = token.chars().count();
        spans.push(WordSpan {
            token: if lower {
                token.to_lowercase()
            } else {
                token.to_string()
            },
            start: code_point,
            end: code_point + width,
        });
        cursor = mat.end();
        code_point += width;
    }
    spans
}

/// One label of a classification task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Label {
    pub name: String,
    pub description: Option<String>,
    /// Few-shot `(input, label)` pairs. At inference `example_mode == "both"`,
    /// so these are appended whenever they exist.
    pub examples: Vec<(String, String)>,
}

impl Label {
    pub fn new(name: impl Into<String>) -> Self {
        Label {
            name: name.into(),
            description: None,
            examples: Vec::new(),
        }
    }
}

/// One classification head, i.e. one `classify_text` key.
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    pub name: String,
    /// Compiled into the `[P]` prompt as `"{name}: {prompt}"`.
    pub prompt: Option<String>,
    pub labels: Vec<Label>,
    /// `class_act`: sigmoid instead of softmax, or `None` for the
    /// `multi_label` default.
    pub activation: Option<String>,
    pub multi_label: bool,
    pub cls_threshold: f32,
    /// Logit calibration; the reference defaults to 1.0 and divides by it.
    pub temperature: f32,
}

impl Task {
    pub fn new(name: impl Into<String>, labels: Vec<Label>) -> Self {
        Task {
            name: name.into(),
            prompt: None,
            labels,
            activation: None,
            multi_label: false,
            cls_threshold: 0.5,
            temperature: 1.0,
        }
    }

    /// Parse one entry of a `classify_text` task mapping.
    ///
    /// Accepts exactly the shapes the reference `runtime._classification_schema`
    /// takes, so a documented snippet runs unchanged:
    ///
    /// ```json
    /// ["a", "b"]
    /// {"labels": ["a", "b"], "multi_label": true, "cls_threshold": 0.4}
    /// {"labels": {"a": "the a label", "b": "the b label"}}
    /// {"labels": ["yes", "no"], "prompt": "Did it work?"}
    /// ```
    pub fn from_json(name: &str, value: &serde_json::Value) -> Result<Self, String> {
        let mut task = Task::new(name, Vec::new());
        let spec = match value {
            serde_json::Value::Array(items) => return Self::labels_from_array(name, items),
            serde_json::Value::Object(map) if map.contains_key("labels") => map,
            _ => {
                return Err(format!(
                    "task {name:?} must be a label array or an object with a \"labels\" key"
                ))
            }
        };
        task.labels = match spec.get("labels").expect("checked above") {
            serde_json::Value::Array(items) => Self::labels_from_array(name, items)?.labels,
            serde_json::Value::Object(entries) => {
                let mut labels = Vec::with_capacity(entries.len());
                for (label, description) in entries {
                    let description = description.as_str().ok_or_else(|| {
                        format!("task {name:?}: description of {label:?} must be a string")
                    })?;
                    labels.push(Label {
                        name: label.clone(),
                        description: Some(description.to_string()),
                        examples: Vec::new(),
                    });
                }
                labels
            }
            _ => {
                return Err(format!(
                    "task {name:?}: \"labels\" must be an array or a {{label: description}} object"
                ))
            }
        };
        if let Some(prompt) = spec.get("prompt") {
            task.prompt = Some(
                prompt
                    .as_str()
                    .ok_or_else(|| format!("task {name:?}: \"prompt\" must be a string"))?
                    .to_string(),
            );
        }
        if let Some(multi_label) = spec.get("multi_label") {
            task.multi_label = multi_label
                .as_bool()
                .ok_or_else(|| format!("task {name:?}: \"multi_label\" must be a boolean"))?;
        }
        if let Some(threshold) = spec.get("cls_threshold") {
            task.cls_threshold = threshold
                .as_f64()
                .ok_or_else(|| format!("task {name:?}: \"cls_threshold\" must be a number"))?
                as f32;
        }
        if let Some(activation) = spec.get("class_act") {
            task.activation = Some(
                activation
                    .as_str()
                    .ok_or_else(|| format!("task {name:?}: \"class_act\" must be a string"))?
                    .to_string(),
            );
        }
        if let Some(temperature) = spec.get("temperature") {
            task.temperature = temperature
                .as_f64()
                .ok_or_else(|| format!("task {name:?}: \"temperature\" must be a number"))?
                as f32;
        }
        if let Some(examples) = spec.get("examples") {
            let pairs = examples.as_array().ok_or_else(|| {
                format!("task {name:?}: \"examples\" must be an array of [input, label] pairs")
            })?;
            for pair in pairs {
                let items = pair.as_array().ok_or_else(|| {
                    format!("task {name:?}: each example must be a [input, label] pair")
                })?;
                if items.len() != 2 {
                    return Err(format!(
                        "task {name:?}: each example must have exactly 2 items"
                    ));
                }
                let read = |index: usize, field: &str| {
                    items[index]
                        .as_str()
                        .map(str::to_string)
                        .ok_or_else(|| format!("task {name:?}: example {field} must be a string"))
                };
                let (input, output) = (read(0, "input")?, read(1, "label")?);
                // Few-shot examples attach to their own label, and the
                // reference only renders examples whose output is declared.
                let target = task
                    .labels
                    .iter_mut()
                    .find(|label| label.name == output)
                    .ok_or_else(|| {
                        format!("task {name:?}: example label {output:?} is not one of its labels")
                    })?;
                target.examples.push((input, output));
            }
        }
        task.validate()?;
        Ok(task)
    }

    fn labels_from_array(name: &str, items: &[serde_json::Value]) -> Result<Self, String> {
        let mut labels = Vec::with_capacity(items.len());
        for item in items {
            labels.push(Label::new(item.as_str().ok_or_else(|| {
                format!("task {name:?}: every label must be a string")
            })?));
        }
        let task = Task::new(name, labels);
        task.validate()?;
        Ok(task)
    }

    /// Reject strings that would corrupt marker parsing, matching `_clean`.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("task name must be a non-empty string".into());
        }
        if self.labels.is_empty() {
            return Err(format!(
                "task {:?} must declare at least one label",
                self.name
            ));
        }
        let mut seen: Vec<&str> = Vec::with_capacity(self.labels.len());
        for value in std::iter::once(&self.name)
            .chain(self.labels.iter().map(|label| &label.name))
            .chain(
                self.labels
                    .iter()
                    .filter_map(|label| label.description.as_ref()),
            )
            .chain(self.prompt.iter())
            .chain(
                self.labels
                    .iter()
                    .flat_map(|label| label.examples.iter())
                    .flat_map(|(input, output)| [input, output]),
            )
        {
            for token in RESERVED {
                if value.contains(token) {
                    return Err(format!(
                        "task {:?}: {value:?} may not contain {token:?}; it is a reserved marker",
                        self.name
                    ));
                }
            }
        }
        for label in &self.labels {
            if seen.contains(&label.name.as_str()) {
                return Err(format!(
                    "task {:?} has duplicate label {:?}",
                    self.name, label.name
                ));
            }
            seen.push(&label.name);
            for (_, output) in &label.examples {
                // The processor drops examples whose output is not a declared
                // label; refusing them here keeps the prompt honest.
                if !self
                    .labels
                    .iter()
                    .any(|candidate| candidate.name == *output)
                {
                    return Err(format!(
                        "task {:?}: example label {output:?} is not one of its labels",
                        self.name
                    ));
                }
            }
        }
        if !(0.0..=1.0).contains(&self.cls_threshold) {
            return Err(format!(
                "task {:?}: cls_threshold must be in [0, 1]",
                self.name
            ));
        }
        if self.temperature <= 0.0 {
            return Err(format!(
                "task {:?}: temperature must be positive",
                self.name
            ));
        }
        match self.activation.as_deref() {
            None | Some("auto") | Some("softmax") | Some("sigmoid") => Ok(()),
            Some(other) => Err(format!(
                "task {:?}: unknown activation {other:?}",
                self.name
            )),
        }
    }

    /// `SchemaTransformer._transform_schema`.
    pub fn schema_tokens(&self) -> Vec<String> {
        self.schema_tokens_with(L_TOKEN)
    }

    /// The reference's `_transform_schema` layout with an explicit child
    /// marker: `[L]` for classification (Decide), `[E]` for extractive entities,
    /// `[R]` for relations. The rest of the token stream is identical, so this
    /// is the only thing that distinguishes the two prompt families.
    pub fn schema_tokens_with(&self, child_marker: &str) -> Vec<String> {
        let mut prompt = match &self.prompt {
            Some(text) => format!("{}: {text}", self.name),
            None => self.name.clone(),
        };
        for label in &self.labels {
            if let Some(description) = &label.description {
                prompt.push_str(&format!(" {DESC_TOKEN} {}: {description}", label.name));
            }
        }
        for label in &self.labels {
            for (input, output) in &label.examples {
                prompt.push_str(&format!(" {EXAMPLE_TOKEN} {input} {OUTPUT_TOKEN} {output}"));
            }
        }
        let mut tokens = vec![
            "(".to_string(),
            P_TOKEN.to_string(),
            prompt,
            "(".to_string(),
        ];
        for label in &self.labels {
            tokens.push(child_marker.to_string());
            tokens.push(label.name.clone());
        }
        tokens.push(")".to_string());
        tokens.push(")".to_string());
        tokens
    }
}

/// Marker rows of one task, as subword indices into the encoded sequence.
/// Element 0 is the `[P]` prompt row; the rest are the `[L]` label rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskMarkers {
    /// Subword index of each `[P]` / `[L]` marker, in that order.
    pub positions: Vec<usize>,
    pub labels: Vec<String>,
}

/// The encoded prompt + text, ready for the encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedPrompt {
    pub input_ids: Vec<u32>,
    /// Per task: the `[P]` group marker followed by one row per child marker.
    /// The classification head reads this with `.skip(1)`.
    pub markers: Vec<TaskMarkers>,
    /// The text, lowercased and split into words. Span indices the boundary
    /// path reports are offsets into this list, so the caller needs it to turn
    /// a span back into text.
    pub words: Vec<String>,
    /// Subword index of the first subword of each entry of `words`
    /// (`token_pooling = "first"`). `_encode_core` gathers `text_states` here.
    /// Kept 1:1 with `words`: a word that tokenizes to nothing still gets a
    /// placeholder row, because dropping it would shift every later index.
    pub text_word_first_positions: Vec<usize>,
    /// How many leading entries of the text stream are the `choices` prefix
    /// rather than document words — `len_prefix` in the reference.
    ///
    /// The prefix is prepended to the **text** token stream, not the schema
    /// stream (`processor.py:645`), so `schema_tokens` and the `[C]` marker
    /// stride are untouched. It is word-routed like any other text token, which
    /// is what lets a choice be scored as a one-token span — and it means every
    /// candidate index the pool produces is a *text-stream* index, so a document
    /// index is that index minus this.
    ///
    /// Zero when the schema declares no `choices`.
    pub text_prefix_len: usize,
    /// Subword index of each child marker, `[P]` dropped, tasks in order. This
    /// is `schema_special_positions[group][1:]` flattened, which is what
    /// `_encode_core` routes into `query_states` — the group marker itself is
    /// not scored.
    pub query_positions: Vec<usize>,
    /// Field name per entry of `query_positions`.
    pub query_names: Vec<String>,
    /// Subword index of each *classification* choice's `[C]` marker, tasks in
    /// order. `_encode_core` routes these to `cls_marker_indices` rather than
    /// `query_marker_indices` (`processor.py:712-719`), because they are scored
    /// by the shared classifier instead of the boundary pool.
    pub classification_positions: Vec<usize>,
    /// Label per entry of `classification_positions`.
    pub classification_names: Vec<String>,
}

/// The four base specials, whose ids the SentencePiece convention fixes at 0..3.
pub const BASE_SPECIALS: [(&str, u32); 4] =
    [("[PAD]", 0), ("[CLS]", 1), ("[SEP]", 2), ("[UNK]", 3)];

/// The eleven tokens GLiNER2 appends past the SentencePiece vocabulary, in id
/// order: `[MASK]` first, then `additional_special_tokens`.
///
/// Their ids are **not** fixed — they start at the piece count, which is 128000
/// for the 128k DeBERTa-v3 vocabularies but 250101 for mDeBERTa-v3's 250k
/// multilingual one. Pinning 128000 encoded multilingual prompts with
/// out-of-vocab ids: the surrounding text still tokenized correctly, so the only
/// symptom was every marker landing on the wrong row of the embedding table.
pub const APPENDED_SPECIALS: [&str; 11] = [
    "[MASK]",
    SEP_STRUCT,
    SEP_TEXT,
    P_TOKEN,
    "[C]",
    "[E]",
    "[R]",
    L_TOKEN,
    EXAMPLE_TOKEN,
    OUTPUT_TOKEN,
    DESC_TOKEN,
];

/// Every added token with its id, for a tokenizer whose vocabulary ends at
/// `piece_count` — the same number the converter used to lay the block out.
pub fn added_tokens(piece_count: u32) -> impl Iterator<Item = (&'static str, u32)> + use<> {
    BASE_SPECIALS
        .iter()
        .copied()
        .map(|(text, id)| (text, id))
        .chain(
            APPENDED_SPECIALS
                .iter()
                .enumerate()
                .map(move |(offset, text)| (*text, piece_count + offset as u32)),
        )
}

/// `PreTrainedTokenizer.tokenize` with `split_special_tokens = false`: the
/// `tokens_trie` cuts every added token out of the input, wherever it appears,
/// and each remaining run goes through SentencePiece.
///
/// The cut matters because GLiNER2 splices `[DESCRIPTION]`, `[EXAMPLE]` and
/// `[OUTPUT]` *into* the prompt string, so a single prompt token can expand to
/// `[SP ...] [DESCRIPTION] [SP ...]`. Passing the whole string to SentencePiece
/// instead would silently produce different ids.
pub fn encode_token(
    token: &str,
    spm: &crate::core::sentencepiece::SentencePieceTokenizer,
) -> Vec<u32> {
    // The appended block's base is this tokenizer's own piece count, so the ids
    // can never drift from the vocabulary that does the encoding.
    let special = added_tokens(spm.len() as u32).collect::<Vec<_>>();
    let mut ids = Vec::new();
    let mut rest = token;
    while !rest.is_empty() {
        let mut at = rest.len();
        let mut hit: Option<&'static str> = None;
        let mut offset = 0usize;
        for candidate in rest.char_indices() {
            let (index, _) = candidate;
            for (text, _) in &special {
                if rest[index..].starts_with(text)
                    && (index < at
                        || (index == at && hit.is_some_and(|current| text.len() > current.len())))
                {
                    at = index;
                    hit = Some(text);
                }
            }
            let _ = offset;
            offset = index;
        }
        let Some(found) = hit else {
            ids.extend(spm.encode_ids(rest));
            return ids;
        };
        if at > 0 {
            ids.extend(spm.encode_ids(&rest[..at]));
        }
        ids.push(id_of(found, spm.len() as u32));
        rest = &rest[at + found.len()..];
    }
    ids
}

fn id_of(token: &str, piece_count: u32) -> u32 {
    added_tokens(piece_count)
        .find(|(text, _)| *text == token)
        .map(|(_, id)| id)
        .expect("token comes from the added-token set")
}

/// `_transform_record` + `_format_input_with_mapping` for classification tasks.
pub fn build_prompt(
    tasks: &[Task],
    text: &str,
    spm: &crate::core::sentencepiece::SentencePieceTokenizer,
) -> Result<EncodedPrompt, String> {
    build_prompt_with(tasks, text, |part| Ok(encode_token(part, spm)))
}

/// The same schema builder with the checkpoint's `tokenizer.json` encoder.
pub fn build_boundary_prompt(
    tasks: &[Task],
    text: &str,
    child_marker: &str,
    spm: &crate::core::sentencepiece::SentencePieceTokenizer,
) -> Result<EncodedPrompt, String> {
    build_boundary_prompt_with(tasks, text, child_marker, |part| {
        Ok(encode_token(part, spm))
    })
}

/// The same schema builder with the checkpoint's `tokenizer.json` encoder.
pub fn build_prompt_with(
    tasks: &[Task],
    text: &str,
    encode: impl FnMut(&str) -> Result<Vec<u32>, String>,
) -> Result<EncodedPrompt, String> {
    build_with_child_marker(tasks, text, L_TOKEN, encode, false, None)
}

/// Prompt assembly for the boundary architecture, with an explicit child
/// marker ([E] / [C] / [R]).
///
/// Identical token stream to [`build_prompt_with`] — the reference's
/// `_format_input_with_mapping` does not know about architectures — but it also
/// reports the two routing index sets `_encode_core` gathers from
/// `last_hidden_state`: the text words and the query markers. Passing
/// `require_query_count` checks that every task produced one query per field,
/// which is the same mis-alignment guard the classification path applies to its
/// label count.
pub fn build_boundary_prompt_with(
    tasks: &[Task],
    text: &str,
    child_marker: &str,
    encode: impl FnMut(&str) -> Result<Vec<u32>, String>,
) -> Result<EncodedPrompt, String> {
    build_with_child_marker(tasks, text, child_marker, encode, true, None)
}

/// Which marker a boundary task group uses, and therefore which head scores it.
///
/// The reference picks the child token from the schema's task type
/// (`processor.py`: `_process_entities` / `_process_json_structures` /
/// `_process_classifications` / `_process_relations`). Note that
/// **classifications use `[L]`, not `[C]`** — `[C]` belongs to
/// `json_structures`. Getting this backwards would route a classification
/// group's markers into the document pool, where they are scored as span
/// queries and emit spans nobody asked for.
///
/// The Rust `Task` type does not carry a task type, so the caller states it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryTaskKind {
    /// `[E]` — one boundary query per field, scored by the document pool.
    Entities,
    /// `[L]` — one classifier logit per choice, scored by `classifier.0`/`3`.
    /// Same marker Decide uses, which is why the classification path is the
    /// shared one.
    Classification,
    /// `[C]` — `json_structures`. Routed like `Entities` but decoded by the
    /// structure decoder, which is not implemented yet.
    JsonStructure,
    /// `[R]` — relation roles, scored by `relation_scorer`. The group's first
    /// two fields are the head and tail roles.
    Relation,
}

impl BoundaryTaskKind {
    pub fn child_marker(self) -> &'static str {
        match self {
            Self::Entities => E_TOKEN,
            Self::Classification => L_TOKEN,
            Self::JsonStructure => C_TOKEN,
            Self::Relation => R_TOKEN,
        }
    }

    /// Whether the group contributes boundary queries to the document pool.
    ///
    /// Only `Entities` does today. `JsonStructure` would also (its markers are
    /// routed as queries and scored for spans), but its decode is a nested
    /// structure rather than a flat span list, so it is left out rather than
    /// half-supported.
    /// Whether this group's `[L]`-position children are boundary queries rather
    /// than classifier choices.
    ///
    /// The reference's split is on the task *type* being `"classifications"`
    /// (`processor.py:712-719`), so it is everything-else that goes to the
    /// query side — not just `Entities`. Matching only `Entities` routed
    /// `[C]` and `[R]` children to the classifier, which then failed the
    /// "routed but not consumed" check with a message that pointed at the
    /// classification head rather than at the routing.
    pub fn yields_boundary_queries(self) -> bool {
        !matches!(self, Self::Classification)
    }
}

/// Build a prompt for a mix of [`BoundaryTaskKind`] groups.
///
/// `kinds[i]` describes `tasks[i]`, and groups are laid out in that order.
/// `text_prefix` is the rendered `choices` prefix; see
/// [`render_choice_prefix`].
pub fn build_mixed_boundary_prompt(
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    text_prefix: &[String],
    text: &str,
    spm: &crate::core::sentencepiece::SentencePieceTokenizer,
) -> Result<EncodedPrompt, String> {
    build_mixed_boundary_prompt_with(tasks, kinds, text_prefix, text, |part| {
        Ok(encode_token(part, spm))
    })
}

/// The SPM-free form of [`build_mixed_boundary_prompt`].
/// `text_prefix` is the rendered `choices` prefix; pass `&[]` for a schema with
/// no choice field.
pub fn build_mixed_boundary_prompt_with(
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    text_prefix: &[String],
    text: &str,
    encode: impl FnMut(&str) -> Result<Vec<u32>, String>,
) -> Result<EncodedPrompt, String> {
    if kinds.len() != tasks.len() {
        return Err(format!("{} tasks but {} kinds", tasks.len(), kinds.len()));
    }
    build_with_child_marker_mixed(tasks, kinds, text_prefix, text, encode)
}

fn build_with_child_marker(
    tasks: &[Task],
    text: &str,
    child_marker: &str,
    encode: impl FnMut(&str) -> Result<Vec<u32>, String>,
    check_query_count: bool,
    kinds: Option<&[BoundaryTaskKind]>,
) -> Result<EncodedPrompt, String> {
    // `kinds` is absent for the single-marker callers, which pass the marker
    // directly. It has to stay authoritative: defaulting to `Entities` here
    // silently turns the classification path's `[L]` markers into `[E]`, and
    // every label then routes to the wrong position.
    let schema_tokens: Vec<Vec<String>> = match kinds {
        Some(kinds) => tasks
            .iter()
            .zip(kinds)
            .map(|(task, kind)| task.schema_tokens_with(kind.child_marker()))
            .collect(),
        None => tasks
            .iter()
            .map(|task| task.schema_tokens_with(child_marker))
            .collect(),
    };
    let owned;
    let kinds = match kinds {
        Some(kinds) => kinds,
        None => {
            // Only consulted to split the routing below, which the
            // classification path never reaches (`check_query_count` is false).
            owned = vec![BoundaryTaskKind::Entities; tasks.len()];
            &owned
        }
    };
    assemble(
        tasks,
        kinds,
        &schema_tokens,
        // The classification path has no `choices` concept; only the boundary
        // `json_structures` path renders one.
        &[],
        text,
        encode,
        check_query_count,
    )
}

/// Render the `choices` prefix for a schema's literal-enum fields.
///
/// `_build_classification_prefix` (`processor.py:825-858`) emits, per group that
/// has at least one choice field:
///
/// ```text
/// ( <parent>: <field> ( <c1> | <c2> ) , <field2> ( <c3> ) )
/// ```
///
/// and returns `[]` when no field carries choices, which is what keeps every
/// other schema byte-identical. Field order and choice order are the schema's
/// declaration order; the reference shuffles both when training and not at
/// inference.
///
/// Choice literals are emitted verbatim — the reference does not lower-case them,
/// and `_find_choice_idx` lower-cases both sides when it looks them up — so
/// `"Happy"` reaches the encoder as `Happy`.
pub fn render_choice_prefix(schema: &serde_json::Value) -> Vec<String> {
    let mut prefix = Vec::new();
    let Some(groups) = schema
        .get("json_structures")
        .and_then(|value| value.as_array())
    else {
        return prefix;
    };
    for group in groups {
        let Some(fields) = group.as_object() else {
            continue;
        };
        for (parent, occurrences) in fields {
            let Some(occurrences) = occurrences.as_object() else {
                continue;
            };
            // `processor.py:830-835`: only a field whose value is a dict carrying
            // both `value` and `choices` is a choice field. A plain `[]` field
            // contributes no parenthesised run, so a group mixes both shapes.
            let choice_fields: Vec<(&String, &Vec<serde_json::Value>)> = occurrences
                .iter()
                .filter_map(|(name, value)| {
                    let object = value.as_object()?;
                    if !object.contains_key("value") {
                        return None;
                    }
                    let choices = object.get("choices")?.as_array()?;
                    if choices.is_empty() {
                        return None;
                    }
                    Some((name, choices))
                })
                .collect();
            if choice_fields.is_empty() {
                continue;
            }
            // `processor.py:848-852`: each field contributes
            // `[name, "(", *choices, ")", ","]` — comma *last* — and the
            // trailing comma is then dropped from the whole run. Emitting the
            // separator as a leading token instead would need the same pop to
            // land on a different element.
            let mut inner: Vec<String> = Vec::new();
            for (name, choices) in &choice_fields {
                inner.push((*name).clone());
                inner.push("(".to_string());
                for (choice_index, choice) in choices.iter().enumerate() {
                    if choice_index > 0 {
                        inner.push("|".to_string());
                    }
                    inner.push(choice.as_str().unwrap_or_default().to_string());
                }
                inner.push(")".to_string());
                inner.push(",".to_string());
            }
            // `if inner: inner = inner[:-1]`
            inner.pop();
            prefix.push("(".to_string());
            prefix.push(format!("{parent}:"));
            prefix.extend(inner);
            prefix.push(")".to_string());
        }
    }
    prefix
}

/// `text_prefix` is the rendered `choices` prefix for this schema; see
/// [`assemble`].
fn build_with_child_marker_mixed(
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    text_prefix: &[String],
    text: &str,
    encode: impl FnMut(&str) -> Result<Vec<u32>, String>,
) -> Result<EncodedPrompt, String> {
    let schema_tokens: Vec<Vec<String>> = tasks
        .iter()
        .zip(kinds)
        .map(|(task, kind)| task.schema_tokens_with(kind.child_marker()))
        .collect();
    assemble(
        tasks,
        kinds,
        &schema_tokens,
        text_prefix,
        text,
        encode,
        true,
    )
}

/// `text_prefix` is the rendered `choices` prefix, prepended to the text stream
/// after `[SEP_TEXT]`. Pass `&[]` when the schema declares no choice field, which
/// is every schema on the span path.
fn assemble(
    tasks: &[Task],
    kinds: &[BoundaryTaskKind],
    schema_tokens: &[Vec<String>],
    text_prefix: &[String],
    text: &str,
    mut encode: impl FnMut(&str) -> Result<Vec<u32>, String>,
    check_query_count: bool,
) -> Result<EncodedPrompt, String> {
    for task in tasks {
        task.validate()?;
    }
    if tasks.is_empty() {
        return Err("no classification task was given".into());
    }

    // Combined token stream: every schema followed by [SEP_STRUCT], then the
    // final [SEP_STRUCT] popped, then [SEP_TEXT] and the text words.
    let mut combined: Vec<String> = Vec::new();
    for (index, tokens) in schema_tokens.iter().enumerate() {
        combined.extend(tokens.iter().cloned());
        // `_format_input_with_mapping` walks `combined` with a running offset,
        // which is what the marker bookkeeping below mirrors.
        if index + 1 < schema_tokens.len() {
            combined.push(SEP_STRUCT.to_string());
        }
    }
    combined.push(SEP_TEXT.to_string());
    // `_format_input_with_mapping` does `combined.extend(text_tokens)` and
    // `text_tokens` is `prefix + words`, so the prefix sits between `[SEP_TEXT]`
    // and the document words.
    combined.extend(text_prefix.iter().cloned());
    let words = split_words(&normalize_text(text));
    combined.extend(words.iter().cloned());
    let sep_index = combined.len() - 1 - words.len();

    // Which combined-token indices are structural markers, per task. The
    // reference computes this on the un-popped stream, where every struct is
    // followed by [SEP_STRUCT]; with the pop applied the arithmetic is the
    // same because the popped slot is the last one.
    let mut marker_orig: Vec<Vec<usize>> = Vec::with_capacity(schema_tokens.len());
    let mut offset = 0usize;
    for (index, tokens) in schema_tokens.iter().enumerate() {
        let mut slots = Vec::with_capacity(tokens.len());
        if tokens.len() > 1 {
            slots.push(offset + 1); // [P]
        }
        // range(4, len(struct) - 2, 2) is every child marker.
        let mut cursor = 4;
        while cursor + 2 < tokens.len() {
            slots.push(offset + cursor);
            cursor += 2;
        }
        marker_orig.push(slots);
        if index + 1 < schema_tokens.len() {
            offset += tokens.len() + 1; // tokens plus [SEP_STRUCT]
        } else {
            offset += tokens.len();
        }
    }

    let mut input_ids: Vec<u32> = Vec::new();
    let mut markers: Vec<TaskMarkers> = Vec::with_capacity(schema_tokens.len());
    for (task_index, task) in tasks.iter().enumerate() {
        markers.push(TaskMarkers {
            positions: Vec::new(),
            labels: task.labels.iter().map(|l| l.name.clone()).collect(),
        });
    }
    let mut text_word_first_positions: Vec<usize> = Vec::with_capacity(words.len());
    for (orig_index, token) in combined.iter().enumerate() {
        let sub = encode(token)?;
        let base = input_ids.len();
        input_ids.extend_from_slice(&sub);
        for (task_index, slots) in marker_orig.iter().enumerate() {
            if slots.contains(&orig_index) {
                markers[task_index].positions.push(base);
            }
        }
        // One entry per text word, recorded at the word's first subword even if
        // the word produced no subwords at all — that is what keeps word indices
        // and subword positions aligned 1:1.
        if orig_index > sep_index {
            text_word_first_positions.push(base);
        }
    }

    if check_query_count {
        let expected: usize = tasks.iter().map(|task| task.labels.len()).sum();
        let mut query_positions = Vec::with_capacity(expected);
        let mut query_names = Vec::with_capacity(expected);
        let mut classification_positions = Vec::new();
        let mut classification_names = Vec::new();
        for (task_index, task) in tasks.iter().enumerate() {
            let found = markers[task_index].positions.len();
            if found != task.labels.len() + 1 {
                return Err(format!(
                    "task {:?}: {} markers for {} fields",
                    task.name,
                    found,
                    task.labels.len()
                ));
            }
            // `[P]` is not routed; `schema_special_positions[group][1:]` is.
            // Classification groups go to the classifier's routing instead,
            // matching `processor.py:712-719`, which splits the two on the
            // group's task type before padding them separately.
            let target: (&mut Vec<usize>, &mut Vec<String>) =
                if kinds[task_index].yields_boundary_queries() {
                    (&mut query_positions, &mut query_names)
                } else {
                    (&mut classification_positions, &mut classification_names)
                };
            for (label, position) in task
                .labels
                .iter()
                .zip(markers[task_index].positions.iter().skip(1))
            {
                target.0.push(*position);
                target.1.push(label.name.clone());
            }
        }
        return Ok(EncodedPrompt {
            input_ids,
            markers,
            words,
            text_word_first_positions,
            text_prefix_len: text_prefix.len(),
            query_positions,
            query_names,
            classification_positions,
            classification_names,
        });
    }

    for (task_index, task) in tasks.iter().enumerate() {
        // `_score_document` raises when the recovered label count and the logit
        // count disagree; a label that tokenizes to nothing is the only way to
        // get there, and it is a silent mis-alignment if left alone.
        if markers[task_index].positions.len() != task.labels.len() + 1 {
            return Err(format!(
                "task {:?}: {} markers for {} labels",
                task.name,
                markers[task_index].positions.len(),
                task.labels.len()
            ));
        }
    }
    Ok(EncodedPrompt {
        input_ids,
        markers,
        words,
        text_word_first_positions,
        text_prefix_len: text_prefix.len(),
        query_positions: Vec::new(),
        query_names: Vec::new(),
        classification_positions: Vec::new(),
        classification_names: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        split_words(text)
    }

    /// The two prompt families differ in exactly one token, and getting it
    /// wrong routes every label to the wrong position while leaving all the
    /// per-stage boundary fixtures green — the boundary path passes its marker
    /// explicitly, so only the Decide path is affected. Lock it here.
    #[test]
    fn the_two_prompt_families_use_their_own_child_marker() {
        let task = Task::new("entities", vec![Label::new("person")]);
        let decide = task.schema_tokens();
        assert!(
            decide.contains(&L_TOKEN.to_string()),
            "the classification path must use [L]"
        );
        assert!(
            !decide.contains(&E_TOKEN.to_string()),
            "the classification path must not emit [E]"
        );
        for (kind, marker) in [
            (BoundaryTaskKind::Entities, E_TOKEN),
            (BoundaryTaskKind::Classification, L_TOKEN),
            (BoundaryTaskKind::JsonStructure, C_TOKEN),
            (BoundaryTaskKind::Relation, R_TOKEN),
        ] {
            assert_eq!(kind.child_marker(), marker, "{kind:?} marker");
            let tokens = task.schema_tokens_with(kind.child_marker());
            assert!(tokens.contains(&marker.to_string()), "{kind:?} tokens");
        }
    }

    #[test]
    fn splits_urls_emails_and_handles() {
        assert_eq!(
            words("mail me at a@b.co"),
            vec!["mail", "me", "at", "a@b.co"]
        );
        assert_eq!(words("ping @handle_now"), vec!["ping", "@handle_now"]);
        assert_eq!(
            words("see https://x.dev/a?b=c now"),
            vec!["see", "https://x.dev/a?b=c", "now"]
        );
        assert_eq!(
            words("go to www.example.com"),
            vec!["go", "to", "www.example.com"]
        );
    }

    #[test]
    fn splits_underscores_and_dashes_within_words() {
        assert_eq!(words("order_status"), vec!["order_status"]);
        assert_eq!(words("e-mail here"), vec!["e-mail", "here"]);
        assert_eq!(words("5,400"), vec!["5", ",", "400"]);
        assert_eq!(words("don't"), vec!["don", "'", "t"]);
    }

    #[test]
    fn lowercases_only_the_token_value() {
        assert_eq!(words("Hello WORLD"), vec!["hello", "world"]);
    }

    #[test]
    fn normalizes_terminal_punctuation() {
        assert_eq!(normalize_text(""), ".");
        assert_eq!(normalize_text("hi?"), "hi?");
        assert_eq!(normalize_text("hi"), "hi.");
        assert_eq!(normalize_text("wow!"), "wow!");
    }

    #[test]
    fn schema_prompt_spells_the_marker_layout() {
        let task = Task::new("intent", vec![Label::new("a"), Label::new("b")]);
        assert_eq!(
            task.schema_tokens(),
            vec!["(", "[P]", "intent", "(", "[L]", "a", "[L]", "b", ")", ")"]
        );
    }

    #[test]
    fn schema_prompt_appends_instruction_descriptions_and_examples() {
        let mut task = Task::new("answer", vec![Label::new("yes"), Label::new("no")]);
        task.prompt = Some("Did it work?".into());
        task.labels[0].description = Some("it worked".into());
        task.labels[0].examples = vec![("it worked".into(), "yes".into())];
        let tokens = task.schema_tokens();
        assert_eq!(
            tokens[2],
            "answer: Did it work? [DESCRIPTION] yes: it worked [EXAMPLE] it worked [OUTPUT] yes"
        );
    }

    #[test]
    fn reserved_markers_are_rejected() {
        let mut task = Task::new("intent", vec![Label::new("ok")]);
        task.labels[0].name = "[L]".into();
        assert!(task.validate().is_err());
    }

    #[test]
    fn duplicate_labels_are_rejected() {
        let task = Task::new("intent", vec![Label::new("a"), Label::new("a")]);
        assert!(task.validate().is_err());
    }
}
