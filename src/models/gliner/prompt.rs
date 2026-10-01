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
        Regex::new(concat!(
            r"(?:https?://[^\s]+|www\.[^\s]+)",
            r"|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",
            r"|@[a-z0-9_]+",
            r"|\w+(?:[-_]\w+)*",
            r"|\S",
        ))
        .expect("word splitter pattern")
    })
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
    word_pattern()
        .find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect()
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
            tokens.push(L_TOKEN.to_string());
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
    pub markers: Vec<TaskMarkers>,
}

/// Every token the reference registers in `_added_tokens_encoder`, with its id.
///
/// The four base specials come from the checkpoint's `added_tokens_decoder`
/// (transformers replays that dict through `add_tokens`); the eleven that follow
/// are GLiNER2's `additional_special_tokens` plus `[MASK]`.
pub const ADDED_TOKENS: [(&str, u32); 15] = [
    ("[PAD]", 0),
    ("[CLS]", 1),
    ("[SEP]", 2),
    ("[UNK]", 3),
    ("[MASK]", 128000),
    (SEP_STRUCT, 128001),
    (SEP_TEXT, 128002),
    (P_TOKEN, 128003),
    ("[C]", 128004),
    ("[E]", 128005),
    ("[R]", 128006),
    (L_TOKEN, 128007),
    (EXAMPLE_TOKEN, 128008),
    (OUTPUT_TOKEN, 128009),
    (DESC_TOKEN, 128010),
];

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
    let mut ids = Vec::new();
    let mut rest = token;
    while !rest.is_empty() {
        let mut at = rest.len();
        let mut hit: Option<&'static str> = None;
        let mut offset = 0usize;
        for candidate in rest.char_indices() {
            let (index, _) = candidate;
            for (text, _) in ADDED_TOKENS {
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
        ids.push(id_of(found));
        rest = &rest[at + found.len()..];
    }
    ids
}

fn id_of(token: &str) -> u32 {
    ADDED_TOKENS
        .iter()
        .find(|(text, _)| *text == token)
        .map(|(_, id)| *id)
        .expect("token comes from ADDED_TOKENS")
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
pub fn build_prompt_with(
    tasks: &[Task],
    text: &str,
    mut encode: impl FnMut(&str) -> Result<Vec<u32>, String>,
) -> Result<EncodedPrompt, String> {
    for task in tasks {
        task.validate()?;
    }
    if tasks.is_empty() {
        return Err("no classification task was given".into());
    }

    let schema_tokens: Vec<Vec<String>> = tasks.iter().map(Task::schema_tokens).collect();
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
    let text_tokens = split_words(&normalize_text(text));
    combined.extend(text_tokens.iter().cloned());

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
        // range(4, len(struct) - 2, 2) is every [L].
        let mut cursor = 4;
        while cursor + 2 < tokens.len() {
            slots.push(offset + cursor);
            cursor += 2;
        }
        if index + 1 < schema_tokens.len() {
            offset += tokens.len() + 1; // tokens plus [SEP_STRUCT]
        } else {
            offset += tokens.len();
        }
        marker_orig.push(slots);
    }

    let mut input_ids: Vec<u32> = Vec::new();
    let mut markers: Vec<TaskMarkers> = Vec::with_capacity(schema_tokens.len());
    for (task_index, task) in tasks.iter().enumerate() {
        markers.push(TaskMarkers {
            positions: Vec::new(),
            labels: task.labels.iter().map(|l| l.name.clone()).collect(),
        });
    }
    for (orig_index, token) in combined.iter().enumerate() {
        let sub = encode(token)?;
        let base = input_ids.len();
        input_ids.extend_from_slice(&sub);
        for (task_index, slots) in marker_orig.iter().enumerate() {
            if slots.contains(&orig_index) {
                markers[task_index].positions.push(base);
            }
        }
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
    Ok(EncodedPrompt { input_ids, markers })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        split_words(text)
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
