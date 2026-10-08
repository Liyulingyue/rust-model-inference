//! Jinja2 chat-template rendering.
//!
//! GGUF files ship the model's own chat template in the
//! `tokenizer.chat_template` metadata key as a Jinja2 source string. The
//! hand-written builders in [`crate::prompt`] and
//! [`crate::models::chat_template`] can only cover a handful of formats,
//! and they drift: LFM2.5's real template is a 4.6 kB Jinja program with
//! macros, `namespace()` and `tojson`, which no hand-rolled `format!` can
//! track. This module renders the shipped template instead.
//!
//! Variable names and semantics follow llama.cpp's
//! `common_chat_templates_apply`, because the templates were written for
//! it:
//!
//! - `messages` — list of `{role, content}` objects
//! - `add_generation_prompt` — bool; whether to open the assistant turn
//! - `enable_thinking` / `preserve_thinking` — bool, read by Qwen3 and
//!   LFM2.5 respectively. We pass both; unused ones are inert.
//! - `bos_token` / `eos_token` — the literal control strings, so a
//!   template that emits `{{ bos_token }}` produces a token the encoder
//!   can resolve back to the BOS id.
//! - `tools` — always an empty list. The `{% if tools %}` branches then
//!   go down the plain path. Tool calling is deliberately out of scope.
//!
//! A template that references no BOS still gets one from the tokenizer's
//! `add_bos_token`, so callers must not prepend a second one.

use std::borrow::Cow;

use minijinja::{Environment, UndefinedBehavior};
use serde_json::{json, Value};

/// One chat turn. `content` may be an array of parts for multimodal
/// templates (the LFM2.5 template branches on `content is string`);
/// callers that only have text pass a plain string.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: Value,
}

impl ChatMessage {
    pub fn text(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: Value::String(content.to_string()),
        }
    }
}

/// Special-token strings the templates are allowed to interpolate.
///
/// These are the *literals* (e.g. `<|im_start|>`), not ids: the rendered
/// string is re-encoded with `parse_special: true` so the encoder maps
/// them back to their control ids.
#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    pub bos: String,
    pub eos: String,
}

/// A compiled chat template.
///
/// minijinja's `Template` borrows its `Environment`, so the environment is
/// kept as a field and the template is looked up by name on each render.
/// The environment is `'static` because the source is registered with
/// `add_template_owned` (a borrowed `Cow<'_, str>` would tie it to a
/// lifetime we do not have here).
pub struct JinjaChatTemplate {
    env: minijinja::Environment<'static>,
    origin: String,
}

const TEMPLATE_NAME: &str = "chat";

/// Tags that are llama.cpp extensions rather than Jinja2.
const LLAMACPP_TAGS: [&str; 2] = ["generation", "endgeneration"];

/// Replace `x.get("k")` with a lookup that keeps Python's `None` result.
///
/// minijinja has no `dict.get`, and it cannot be registered: the extension
/// points are `add_filter` / `add_test` / `add_function` / `add_global`, none
/// of which add a method to a map. So the call has to be rewritten.
///
/// `x["k"]` is *not* an equivalent substitution. Jinja2 gives `None` for a
/// missing key, so `x.get("k") is none` is true; minijinja's `x["k"]` yields
/// Undefined, and `Undefined is none` is **false**. Rewriting to an index
/// silently flips `is none` branches, which is why this injects
/// `| default(__py_none)` instead: that restores the real `None` the template
/// expects. (`__py_none` rather than `none`, because minijinja's `none` is a
/// *test*, not a value.)
///
/// The scan is string-literal aware. A previous version matched `.get(`
/// anywhere in the source and rewrote it inside string literals too, so
/// `{% set s = 'api .get("k")' %}` became `{% set s = 'api ["k"]' %}`.
/// Find where the receiver expression of a trailing method call starts.
///
/// Handles `foo`, `foo.bar`, `foo[0]`, `messages[i].content` and any chain of
/// those. Returns `None` when the receiver is not recognisable, so the caller
/// can leave the call alone and let minijinja raise a real error instead of
/// mangling the template.
fn is_expr_byte(c: u8) -> bool {
    matches!(c,
        b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'.' | b'|'
        | b'\'' | b'"' | b' ' | b'\t' | b'\n' | b'\r'
        | b'(' | b')' | b'[' | b']')
}

fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

/// The identifier starting at `at`, if there is one.
fn leading_word(text: &str, at: usize) -> Option<&str> {
    let bytes = text.as_bytes();
    let mut end = at;
    while end < text.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
        end += 1;
    }
    if end == at {
        return None;
    }
    Some(&text[at..end])
}

/// Tag words that can sit immediately before an expression.
const JINJA_KEYWORDS: &[&str] = &[
    "if", "elif", "else", "endif", "for", "endfor", "in", "is", "not", "and", "or", "set",
    "endset", "when", "endwhen", "with", "as", "by", "macro", "endmacro", "call", "filter",
    "block", "include", "extends", "import", "from", "do", "true", "false", "none",
];

/// True when every bracket in `text` is closed, in order.
fn is_balanced(text: &str) -> bool {
    let mut depth = 0i32;
    for c in text.bytes() {
        match c {
            b'(' | b'[' => depth += 1,
            b')' | b']' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

/// Where the receiver of the method call ending at `text` starts.
///
/// The capture must be bracket-balanced; an unbalanced prefix means the scan
/// landed inside an expression it cannot describe, and rewriting it would
/// corrupt the template, so it is rejected and the call is left verbatim.
fn receiver_start(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut start = text.len();
    let mut depth = 0i32;
    while start > 0 {
        let c = bytes[start - 1];
        if c == b')' || c == b']' {
            depth += 1;
            start -= 1;
            continue;
        }
        if c == b'(' || c == b'[' {
            if depth == 0 {
                break;
            }
            depth -= 1;
            start -= 1;
            continue;
        }
        // Step back over a whole string literal: a comma inside `'a,b'` would
        // otherwise look like an operator and truncate the receiver.
        if depth == 0 && (c == b'\'' || c == b'"') {
            let mut k = start - 2;
            let mut found = false;
            while k > 0 {
                k -= 1;
                match bytes[k] {
                    d if d == c => {
                        start = k;
                        found = true;
                        break;
                    }
                    b'\\' if k > 0 => k -= 1,
                    _ => {}
                }
            }
            if !found {
                return None;
            }
            continue;
        }
        // `{`, `}`, `%` and operators end the receiver. In practice `%}` stops
        // the scan at the end of the enclosing `{% ... %}` block, so a capture
        // cannot escape the current expression.
        if depth == 0 && !is_expr_byte(c) {
            break;
        }
        start -= 1;
    }
    let mut head = start;
    while head < text.len() && (is_space(bytes[head]) || bytes[head] == b'.') {
        head += 1;
    }
    if head >= text.len() || head == 0 {
        return None;
    }
    // `{% if s.startswith(p) %}`: the scan stops at `%}` and the capture would
    // otherwise start at the `if`. Keywords belong to the tag, not the
    // receiver.
    while let Some(word) = leading_word(text, head) {
        if !JINJA_KEYWORDS.contains(&word) {
            break;
        }
        let mut next = head + word.len();
        while next < text.len() && is_space(bytes[next]) {
            next += 1;
        }
        if next >= text.len() {
            return None;
        }
        head = next;
    }
    if !is_balanced(&text[head..]) {
        return None;
    }
    let first = bytes[head];
    if !(first.is_ascii_alphanumeric()
        || first == b'_'
        || first == b'\''
        || first == b'"'
        || first == b'(')
    {
        return None;
    }
    Some(head)
}

/// Remove fully enclosing parentheses so nesting does not grow on each pass.
fn strip_outer_parens(text: &str) -> &str {
    let mut s = text.trim();
    loop {
        let Some(inner) = s.strip_prefix('(') else {
            break;
        };
        let Some(rest) = inner.strip_suffix(')') else {
            break;
        };
        let _ = rest;
        let body = &inner[..inner.len() - 1];
        if !is_balanced(body) {
            break;
        }
        s = body.trim();
    }
    s
}

/// Python's `str.lstrip`/`str.rstrip`: with no argument they strip whitespace,
/// with one they strip any of the given characters.
fn trim_py<'a>(s: &'a str, chars: Option<&str>, left: bool) -> &'a str {
    match chars {
        Some(set) => {
            let set: Vec<char> = set.chars().collect();
            let pred = |c: char| set.contains(&c);
            if left {
                s.trim_start_matches(pred)
            } else {
                s.trim_end_matches(pred)
            }
        }
        None => {
            if left {
                s.trim_start()
            } else {
                s.trim_end()
            }
        }
    }
}

/// Python string methods that Jinja2 templates call but that minijinja has no
/// method for. Each is rewritten to the identically named filter.
const STRING_METHODS: &[&str] = &["startswith", "endswith", "lstrip", "rstrip", "split"];

/// Rewrite the Python-flavoured constructs that minijinja cannot run.
///
/// minijinja has no `dict.get`, and it cannot be registered as a method: the
/// extension points are `add_filter` / `add_test` / `add_function` /
/// `add_global`, none of which add a method to a map or string. So the call
/// has to be rewritten.
///
/// `x.get("k")` must NOT become `x["k"]`: Jinja2 returns `None` for a missing
/// key, so `x.get("k") is none` is true, while minijinja's `x["k"]` yields
/// Undefined and `Undefined is none` is **false**. Rewriting to an index
/// silently flips `is none` branches, so this emits
/// `x["k"]|default(__py_none)` instead, which restores the `None`.
///
/// The scan is string-literal aware, because a naive `find(".get(")` also
/// rewrote inside string literals -- `{% set s = 'api .get("k")' %}` became
/// `{% set s = 'api ["k"]' %}`. Only `{# #}` comments and `{% raw %}` blocks
/// are skipped wholesale; `{% ... %}` and `{{ ... }}` are code and are
/// scanned, since `{% if d.get("k") %}` needs the rewrite too.
fn rewrite_python_compat(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        // `{# ... #}` is a Jinja comment: verbatim.
        if b == b'{' && i + 1 < bytes.len() && bytes[i + 1] == b'#' {
            let end = source[i..]
                .find("#}")
                .map(|p| i + p + 2)
                .unwrap_or(bytes.len());
            out.push_str(&source[i..end]);
            i = end;
            continue;
        }
        // `{% raw %}` ... `{% endraw %}` is literal text, not code.
        if source[i..].starts_with("{% raw")
            || source[i..].starts_with("{%- raw")
            || source[i..].starts_with("{%+ raw")
        {
            let end = source[i..]
                .find("{% endraw")
                .or_else(|| source[i..].find("{%- endraw"))
                .map(|p| i + p)
                .and_then(|p| source[p..].find("%}").map(|q| p + q + 2))
                .unwrap_or(bytes.len());
            out.push_str(&source[i..end]);
            i = end;
            continue;
        }
        // String literals are verbatim wherever they appear.
        if b == b'\'' || b == b'"' {
            let quote = b;
            let start = i;
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == quote {
                    i += 1;
                    break;
                }
                i += 1;
            }
            out.push_str(&source[start..i]);
            continue;
        }
        if b == b'.' {
            // `x.get("k")` -> `x["k"]|default(__py_none)`
            if source[i..].starts_with(".get(") {
                let arg_start = i + ".get(".len();
                if let Some(rel) = source[arg_start..].find(')') {
                    let close = arg_start + rel;
                    let arg = source[arg_start..close].trim();
                    let quoted = match arg.as_bytes().first() {
                        Some(q @ (b'"' | b'\'')) if arg.len() >= 2 && arg.ends_with(*q as char) => {
                            !arg[1..arg.len() - 1].contains(*q as char)
                        }
                        _ => false,
                    };
                    if quoted {
                        // The filter must sit outside the brackets: a filter
                        // inside them decorates the *key*, leaving `x["k"]`
                        // and `is none` false again.
                        out.push('[');
                        out.push_str(arg);
                        out.push_str("]|default(__py_none)");
                        i = close + 1;
                        continue;
                    }
                    // Multi-arg `.get(k, d)` really does change meaning; leave
                    // it for minijinja to report rather than rewrite it.
                }
            }
        }
        let ch = source[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Rewrite one `x.method(args)` call into `x|filter(args)`, if `name` is a
/// supported Python string method and the receiver can be identified.
fn rewrite_one_string_method(
    text: &str,
    start: usize,
    dot: usize,
    name: &str,
    arg_start: usize,
    close: usize,
) -> Option<String> {
    let receiver = &text[start..dot];
    let expr = strip_outer_parens(receiver);
    if expr.is_empty() {
        return None;
    }
    let args = rewrite_string_methods(&text[arg_start..close]);
    let mut out = String::with_capacity(text.len() + 8);
    // The receiver goes in parentheses: `a|b[0]` does not parse, so a filter
    // has to bind to a grouped expression when the chain also subscripts.
    out.push_str(&text[..start]);
    // The group must close *after* the filter call. `(a)|split(p)[0]` does not
    // parse -- minijinja binds `[0]` to the filter name -- so a following
    // subscript has to apply to `(a|split(p))`.
    out.push('(');
    out.push_str(expr);
    out.push('|');
    out.push_str(name);
    out.push('(');
    out.push_str(&args);
    out.push_str("))");
    out.push_str(&text[close + 1..]);
    Some(out)
}

/// One left-to-right pass that rewrites the first resolvable string-method
/// call it finds, leaving everything else untouched.
fn rewrite_string_methods_once(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'{' && i + 1 < bytes.len() && bytes[i + 1] == b'#' {
            i = text[i..]
                .find("#}")
                .map(|p| i + p + 2)
                .unwrap_or(bytes.len());
            continue;
        }
        if text[i..].starts_with("{% raw") || text[i..].starts_with("{%- raw") {
            i = text[i..]
                .find("{% endraw")
                .or_else(|| text[i..].find("{%- endraw"))
                .map(|p| i + p)
                .and_then(|p| text[p..].find("%}").map(|q| p + q + 2))
                .unwrap_or(bytes.len());
            continue;
        }
        if b == b'\'' || b == b'"' {
            let quote = b;
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == quote {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }
        if b == b'.' {
            if let Some(name) = STRING_METHODS
                .iter()
                .find(|name| text[i..].starts_with(&format!(".{name}(")))
            {
                let arg_start = i + name.len() + 2;
                if let Some(rel) = text[arg_start..].find(')') {
                    let close = arg_start + rel;
                    if let Some(start) = receiver_start(&text[..i]) {
                        return rewrite_one_string_method(text, start, i, name, arg_start, close);
                    }
                }
            }
        }
        i += 1;
    }
    None
}

/// Rewrite every supported Python string method call into a filter call.
///
/// This has to be iterative. Handlers chain -- Edge0 ships
/// `content.split('</think>')[0].rstrip('\n').split('<think>')[-1]` -- and each
/// rewrite re-scans the result, so the receiver is always read from text whose
/// rewritten parts are already parenthesised. Every pass strictly reduces the
/// number of remaining method calls, so it terminates.
fn rewrite_string_methods(source: &str) -> String {
    let mut text = source.to_string();
    // Bounded: each pass removes at least one call, and templates are small.
    for _ in 0..256 {
        match rewrite_string_methods_once(&text) {
            Some(next) => text = next,
            None => break,
        }
    }
    text
}

fn strip_llamacpp_extensions(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    loop {
        let Some(open) = rest.find("{%") else {
            out.push_str(rest);
            return out;
        };
        let Some(close_rel) = rest[open + 2..].find("%}") else {
            out.push_str(rest);
            return out;
        };
        let close = open + 2 + close_rel;
        out.push_str(&rest[..open]);

        // Read the trim markers off the *raw* tag body before stripping
        // them, otherwise `{%- generation -%}` loses its left trim.
        let raw_inner = rest[open + 2..close].trim();
        let ltrim = raw_inner.starts_with('-');
        let rtrim = raw_inner.ends_with('-');
        let name = raw_inner
            .trim_start_matches('-')
            .trim_end_matches('-')
            .trim()
            .split_whitespace()
            .next()
            .unwrap_or("");

        if LLAMACPP_TAGS.contains(&name) {
            // Mirror the original trim markers onto a comment.
            match (ltrim, rtrim) {
                (true, true) => out.push_str("{#- -#}"),
                (true, false) => out.push_str("{#- #}"),
                (false, true) => out.push_str("{# -#}"),
                (false, false) => out.push_str("{# #}"),
            }
        } else {
            out.push_str(&rest[open..=close + 1]);
        }
        rest = &rest[close + 2..];
    }
}

impl JinjaChatTemplate {
    /// Compile a Jinja2 template source string.
    pub fn compile(source: &str, origin: &str) -> Result<Self, String> {
        let mut env = Environment::new();
        // Templates in the wild test `is defined` before touching optional
        // fields, so lenient undefined is the behaviour they expect.
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        // `fuel` is enabled in Cargo.toml; bound template work so a
        // pathological template cannot hang a server request.
        env.set_fuel(Some(2_000_000));
        env.set_keep_trailing_newline(true);
        // A genuine `None` for `.get()` rewrites to fall back to. minijinja's
        // `none` is a *test*, not a value, so it cannot be used here.
        env.add_global("__py_none", minijinja::Value::from(()));
        // String methods that minijinja lacks; `rewrite_python_compat` turns
        // `x.startswith(p)` into `x|startswith(p)`.
        env.add_filter("startswith", |s: Cow<'_, str>, p: Cow<'_, str>| {
            s.starts_with(&*p)
        });
        env.add_filter("endswith", |s: Cow<'_, str>, p: Cow<'_, str>| {
            s.ends_with(&*p)
        });
        // minijinja's builtin `split` yields a lazy sequence, which cannot be
        // indexed negatively, so Python's `s.split(p)[-1]` would silently
        // return the first element. A real list keeps list semantics.
        env.add_filter(
            "split",
            |s: Cow<'_, str>, pat: Option<Cow<'_, str>>| -> Vec<String> {
                match pat {
                    Some(p) => s.split(&*p).map(str::to_string).collect(),
                    None => s.split_whitespace().map(str::to_string).collect(),
                }
            },
        );
        // Python's `lstrip`/`rstrip` take an optional set of characters, so
        // `rstrip('\n')` must be accepted as well as a bare `rstrip()`.
        env.add_filter("lstrip", |s: Cow<'_, str>, chars: Option<Cow<'_, str>>| {
            trim_py(&s, chars.as_deref(), true).to_string()
        });
        env.add_filter("rstrip", |s: Cow<'_, str>, chars: Option<Cow<'_, str>>| {
            trim_py(&s, chars.as_deref(), false).to_string()
        });
        let source = rewrite_python_compat(&strip_llamacpp_extensions(source));
        let source = rewrite_string_methods(&source);
        env.add_template_owned(TEMPLATE_NAME, source)
            .map_err(|e| format!("{origin}: invalid chat template: {e}"))?;
        Ok(Self {
            env,
            origin: origin.to_string(),
        })
    }

    /// Where this template came from, for error messages.
    pub fn origin(&self) -> &str {
        &self.origin
    }

    /// Render the conversation.
    ///
    /// `add_generation_prompt` must be `false` when reproducing a stored
    /// transcript for parity comparison, and `true` for live generation —
    /// getting it backwards silently changes every token after the last
    /// user turn.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: bool,
        special: &SpecialTokens,
    ) -> Result<String, String> {
        let messages: Vec<Value> = messages
            .iter()
            .map(|m| json!({ "role": m.role, "content": m.content }))
            .collect();
        let ctx = json!({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
            "preserve_thinking": enable_thinking,
            "bos_token": special.bos,
            "eos_token": special.eos,
            // Always empty: `{% if tools %}` must be falsy. Tool calling
            // is not wired up, and pretending otherwise would render a
            // tools preamble the model has no way to satisfy.
            "tools": [],
        });
        let template = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| format!("{}: template lost: {e}", self.origin))?;
        template
            .render(&ctx)
            .map_err(|e| format!("{}: render failed: {e}", self.origin))
    }
}

/// Extract `tokenizer.chat_template` from a GGUF source.
///
/// Takes the same `Fn(&str) -> Option<MetaValue>` shape the tokenizer
/// constructors use, so callers can pass a closure straight over
/// `TensorSource::metadata`.
pub fn template_from_source(
    metadata: &dyn Fn(&str) -> Option<crate::core::tensor::MetaValue>,
) -> Option<Result<JinjaChatTemplate, String>> {
    let value = metadata("tokenizer.chat_template")?;
    let raw = value.to_string_val()?;
    Some(JinjaChatTemplate::compile(
        &raw,
        "GGUF tokenizer.chat_template",
    ))
}

/// Load a template from an explicit file path.
pub fn template_from_file(path: &std::path::Path) -> Result<JinjaChatTemplate, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("chat template file {}: {e}", path.display()))?;
    JinjaChatTemplate::compile(&raw, &path.display().to_string())
}

/// Pick the literal control strings for `bos_token` / `eos_token` out of
/// a tokenizer's special-token table, falling back to the conventional
/// Qwen spellings when the tokenizer does not name them.
pub fn special_tokens(lookup: &dyn Fn(&str) -> Option<String>) -> SpecialTokens {
    SpecialTokens {
        bos: lookup("bos_token").unwrap_or_else(|| "<|endoftext|>".to_string()),
        eos: lookup("eos_token").unwrap_or_else(|| "<|im_end|>".to_string()),
    }
}

/// Render a conversation straight to token ids.
///
/// `add_special` must be `false` whenever the template emits
/// `{{ bos_token }}` itself (LFM2.5 does), otherwise the prompt gets two
/// BOS tokens. The rendered string is encoded with `parse_special: true`
/// so `<|im_start|>` and friends map back to their control ids instead of
/// being shredded into `<`, `|`, `im_start`, ... — this is the step that
/// silently corrupts a prompt when it is forgotten.
///
/// Takes `&dyn Tokenizer` rather than `&BPETokenizer` because the llama
/// trunk goes through `load_tokenizer`, which hands back a boxed trait
/// object.
pub fn render_tokens(
    tokenizer: &dyn crate::core::tokenizer::Tokenizer,
    template: &JinjaChatTemplate,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Result<Vec<u32>, String> {
    let special = special_tokens_from_tokenizer(tokenizer);
    let text = template.render(messages, add_generation_prompt, enable_thinking, &special)?;
    let options = crate::core::tokenizer::EncodeOptions {
        // The template owns the BOS decision; see the doc comment.
        add_special: false,
        parse_special: true,
    };
    Ok(tokenizer.encode(&text, options))
}

/// Look up the literal BOS/EOS spellings a template may interpolate, from
/// the vocabulary itself rather than a hardcoded guess.
///
/// `token_piece_bytes(id, true)` is how the trait exposes a token's text —
/// there is no `token_str` on `Tokenizer`, only on `BPETokenizer`.
fn special_tokens_from_tokenizer(
    tokenizer: &dyn crate::core::tokenizer::Tokenizer,
) -> SpecialTokens {
    let literal = |id: Option<u32>| {
        id.map(|id| String::from_utf8_lossy(&tokenizer.token_piece_bytes(id, true)).into_owned())
            .unwrap_or_default()
    };
    SpecialTokens {
        bos: literal(tokenizer.bos_id()),
        eos: literal(tokenizer.eos_id()),
    }
}

/// Command-line switches that select a chat template.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// `--jinja`: render the GGUF's own template.
    pub jinja: bool,
    /// `--chat-template-file <path>`: render this file. Implies Jinja2.
    pub file: Option<std::path::PathBuf>,
}

impl Options {
    /// Whether any Jinja2 rendering was requested at all.
    pub fn enabled(&self) -> bool {
        self.jinja || self.file.is_some()
    }

    /// Resolve to the template that applies, or `None` when the caller should
    /// fall back to its hand-written builder.
    ///
    /// Precedence matches llama.cpp: an explicit `--chat-template-file` wins,
    /// then the GGUF's own `tokenizer.chat_template`. Jinja2 is never used
    /// unless asked for: turning it on by default would change the token ids
    /// of every model whose template disagrees with our builder, which is
    /// exactly the kind of silent change the A/B test exists to catch.
    pub fn resolve(
        &self,
        metadata: &dyn Fn(&str) -> Option<crate::core::tensor::MetaValue>,
    ) -> Result<Option<JinjaChatTemplate>, String> {
        if !self.enabled() {
            return Ok(None);
        }
        if let Some(path) = self.file.as_deref() {
            return template_from_file(path).map(Some);
        }
        // `template_from_source` reports "the model ships no template" as
        // `None`, and a broken one as `Err`.
        match template_from_source(metadata) {
            Some(Ok(template)) => Ok(Some(template)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        }
    }
}

/// Render a one-turn conversation to token ids, or `None` when Jinja2 was
/// not requested (or the model ships no template) and the caller should use
/// its own builder.
///
/// This is the shape every CLI text path wants, so each model site stays a
/// four-line `match` instead of re-deriving the precedence rules.
pub fn single_turn_tokens(
    tokenizer: &dyn crate::core::tokenizer::Tokenizer,
    opts: &Options,
    metadata: &dyn Fn(&str) -> Option<crate::core::tensor::MetaValue>,
    user: &str,
    enable_thinking: bool,
) -> Result<Option<Vec<u32>>, String> {
    let Some(template) = opts.resolve(metadata)? else {
        return Ok(None);
    };
    Ok(Some(render_tokens(
        tokenizer,
        &template,
        &[ChatMessage::text("user", user)],
        true,
        enable_thinking,
    )?))
}

/// Same as [`single_turn_tokens`] but returns the rendered text, for models
/// that render to a `String` and tokenize it themselves (nemotron_h,
/// falcon-h1).
pub fn single_turn_text(
    opts: &Options,
    metadata: &dyn Fn(&str) -> Option<crate::core::tensor::MetaValue>,
    bos: &str,
    eos: &str,
    user: &str,
    enable_thinking: bool,
) -> Result<Option<String>, String> {
    let Some(template) = opts.resolve(metadata)? else {
        return Ok(None);
    };
    let special = SpecialTokens {
        bos: bos.to_string(),
        eos: eos.to_string(),
    };
    Ok(Some(template.render(
        &[ChatMessage::text("user", user)],
        true,
        enable_thinking,
        &special,
    )?))
}

/// Render a multi-turn conversation to token ids.
///
/// `add_generation_prompt` must be `false` for scoring / transcript work —
/// the template is reproducing a fixed conversation, not asking the model
/// to write the next turn.
pub fn conversation_tokens(
    tokenizer: &dyn crate::core::tokenizer::Tokenizer,
    opts: &Options,
    metadata: &dyn Fn(&str) -> Option<crate::core::tensor::MetaValue>,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Result<Option<Vec<u32>>, String> {
    let Some(template) = opts.resolve(metadata)? else {
        return Ok(None);
    };
    let special = special_tokens_from_tokenizer(tokenizer);
    let text = template.render(messages, add_generation_prompt, enable_thinking, &special)?;
    let options = crate::core::tokenizer::EncodeOptions {
        add_special: false,
        parse_special: true,
    };
    Ok(Some(tokenizer.encode(&text, options)))
}

/// Expand the single vision placeholder a template emits into the
/// contiguous run the position builders require.
///
/// A chat template writes one `<|image_pad|>` per image, but the vision
/// encoder produced `count` grid tokens for that image, so the sequence has
/// to carry `count` copies. Both position builders
/// (`build_qwen3_media_positions` and `build_qwen35_positions`) document the
/// same contract: on hitting `placeholder_id` they consume `grid_shapes[i]`
/// contiguous copies and refuse anything else.
///
/// `counts` is per media item, in the order the template emitted the
/// placeholders. Counts and placeholders must line up exactly — a mismatch
/// means the template and the projector disagree about how much media there
/// is, which would silently misalign every position id, so it is an error.
pub fn expand_vision_placeholders(
    ids: &[u32],
    placeholder_id: u32,
    counts: &[usize],
) -> Result<Vec<u32>, String> {
    let found = ids.iter().filter(|&&id| id == placeholder_id).count();
    if found != counts.len() {
        return Err(format!(
            "chat template emitted {found} vision placeholder(s) but the projector \
             produced {} grid(s); positions would be misaligned",
            counts.len()
        ));
    }
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut next = 0usize;
    for &id in ids {
        if id != placeholder_id {
            out.push(id);
            continue;
        }
        let count = counts[next];
        next += 1;
        if count == 0 {
            return Err("vision grid produced zero placeholder tokens".into());
        }
        out.extend(std::iter::repeat_n(placeholder_id, count));
    }
    Ok(out)
}

/// Build the `content` field for the user turn.
///
/// With no media this is a plain string, the shape templates treat as "no
/// media at all". With media it is always an array: collapsing a lone part back
/// to a bare string used to drop an image-only message entirely, because the
/// single part was `{"type": "image"}`, `.get("text")` was absent, and it
/// silently became `""`.
fn build_user_content(media: &[MediaPart], user_text: &str) -> Value {
    if media.is_empty() {
        return Value::String(user_text.to_string());
    }
    let mut parts: Vec<Value> = Vec::with_capacity(media.len() + 1);
    for m in media {
        parts.push(json!({ "type": m.kind.as_str() }));
    }
    if !user_text.is_empty() {
        parts.push(json!({ "type": "text", "text": user_text }));
    }
    Value::Array(parts)
}

/// What kind of attachment a content part carries.
///
/// Templates branch on this: Qwen's emit an image block, while video models
/// expect `type == "video"`. Every part used to be hard-coded to `image`, so a
/// non-image model could never render its own template correctly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
    Audio,
}

impl MediaKind {
    /// The value used in the content part's `type` field.
    pub fn as_str(self) -> &'static str {
        match self {
            MediaKind::Image => "image",
            MediaKind::Video => "video",
            MediaKind::Audio => "audio",
        }
    }
}

/// One media attachment in a multimodal prompt.
#[derive(Clone, Copy, Debug)]
pub struct MediaPart {
    pub kind: MediaKind,
    /// How many vision placeholders this attachment expands to.
    pub token_count: usize,
}

impl MediaPart {
    pub fn image(token_count: usize) -> Self {
        Self {
            kind: MediaKind::Image,
            token_count,
        }
    }
}

/// Render one user turn carrying media, expanding the vision placeholder.
///
/// `media[i].token_count` is the number of grid tokens the projector produced
/// for the i-th media item. The template emits one placeholder per item, so
/// [`expand_vision_placeholders`] lines them up.
///
/// Returns `None` when Jinja2 is off, letting the caller keep its
/// hand-built token layout.
pub fn media_conversation_tokens(
    tokenizer: &dyn crate::core::tokenizer::Tokenizer,
    opts: &Options,
    metadata: &dyn Fn(&str) -> Option<crate::core::tensor::MetaValue>,
    placeholder_id: u32,
    media: &[MediaPart],
    user_text: &str,
    system_text: Option<&str>,
    enable_thinking: bool,
) -> Result<Option<Vec<u32>>, String> {
    let media_counts: Vec<usize> = media.iter().map(|m| m.token_count).collect();
    let Some(mut ids) = text_conversation_tokens(
        tokenizer,
        opts,
        metadata,
        system_text,
        user_text,
        media,
        enable_thinking,
    )?
    else {
        return Ok(None);
    };
    if media_counts.is_empty() {
        return Ok(Some(ids));
    }
    ids = expand_vision_placeholders(&ids, placeholder_id, &media_counts)?;
    Ok(Some(ids))
}

/// Render `[system?, user]` and open the assistant turn.
///
/// The JEV scorers use this directly: they author the system message and
/// the payload themselves, but the *shape* is identical to generation —
/// `[system, user]` plus an open assistant turn, because the token after the
/// prompt is the decision being scored. That is `add_generation_prompt =
/// true`, the same value live generation uses.
pub fn text_conversation_tokens(
    tokenizer: &dyn crate::core::tokenizer::Tokenizer,
    opts: &Options,
    metadata: &dyn Fn(&str) -> Option<crate::core::tensor::MetaValue>,
    system_text: Option<&str>,
    user_text: &str,
    media: &[MediaPart],
    enable_thinking: bool,
) -> Result<Option<Vec<u32>>, String> {
    let Some(template) = opts.resolve(metadata)? else {
        return Ok(None);
    };
    let special = special_tokens_from_tokenizer(tokenizer);
    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(system) = system_text.filter(|s| !s.trim().is_empty()) {
        messages.push(ChatMessage::text("system", system));
    }
    let content = build_user_content(media, user_text);
    messages.push(ChatMessage {
        role: "user".into(),
        content,
    });

    let text = template.render(&messages, true, enable_thinking, &special)?;
    let options = crate::core::tokenizer::EncodeOptions {
        add_special: false,
        parse_special: true,
    };
    Ok(Some(tokenizer.encode(&text, options)))
}

/// Resolve a template once, for callers that render repeatedly.
///
/// The JEV scorers resolve in their constructor (the only place with the
/// `TensorSource` to hand) and then render per question, so the template is
/// stored rather than re-resolved.
/// Render `[system?, user]` with an already-resolved template, opening the
/// assistant turn.
///
/// JEV scoring uses this: the scorer authored the system message and the
/// payload itself, but the shape is identical to generation — `[system,
/// user]` plus an open assistant turn, because the next token is the decision
/// being scored. That is `add_generation_prompt = true`.
pub fn render_text_conversation(
    tokenizer: &dyn crate::core::tokenizer::Tokenizer,
    template: &JinjaChatTemplate,
    system_text: Option<&str>,
    user_text: &str,
    enable_thinking: bool,
) -> Result<Vec<u32>, String> {
    let special = special_tokens_from_tokenizer(tokenizer);
    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(system) = system_text.filter(|s| !s.trim().is_empty()) {
        messages.push(ChatMessage::text("system", system));
    }
    messages.push(ChatMessage::text("user", user_text));
    let text = template.render(&messages, true, enable_thinking, &special)?;
    let options = crate::core::tokenizer::EncodeOptions {
        add_special: false,
        parse_special: true,
    };
    Ok(tokenizer.encode(&text, options))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::MetaValue;

    fn render(src: &str, agp: bool, thinking: bool) -> String {
        let t = JinjaChatTemplate::compile(src, "test").unwrap();
        t.render(
            &[ChatMessage::text("user", "hi")],
            agp,
            thinking,
            &SpecialTokens::default(),
        )
        .unwrap()
    }

    #[test]
    fn chatml_shape() {
        let src = "{% for m in messages %}<|im_start|>{{ m.role }}\n\
                   {{ m.content }}<|im_end|>\n{% endfor %}\
                   {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        assert_eq!(
            render(src, true, false),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn add_generation_prompt_false_omits_assistant_turn() {
        let src = "{% for m in messages %}<|im_start|>{{ m.role }}\n\
                   {{ m.content }}<|im_end|>\n{% endfor %}\
                   {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        assert_eq!(
            render(src, false, false),
            "<|im_start|>user\nhi<|im_end|>\n"
        );
    }

    #[test]
    fn tools_branch_is_skipped_when_empty() {
        let src = "{% if tools %}TOOLS{% else %}PLAIN{% endif %}";
        assert_eq!(render(src, true, false), "PLAIN");
    }

    #[test]
    fn whitespace_control_is_honoured() {
        // `{%-` / -%}` trimming is what real templates rely on to avoid
        // stray newlines between turns.
        let src = "A\n{%- if true %}B{% endif %}";
        assert_eq!(render(src, true, false), "AB");
    }

    #[test]
    fn namespace_and_macro_supported() {
        // Both are used by the shipped LFM2.5 template.
        let src = "{% macro m(x) %}[{{ x }}]{% endmacro %}\
                   {% set ns = namespace(v='') %}\
                   {% for i in [1,2,3] %}{% set ns.v = ns.v + m(i) %}{% endfor %}\
                   {{ ns.v }}";
        assert_eq!(render(src, true, false), "[1][2][3]");
    }

    #[test]
    fn enable_thinking_reaches_template() {
        let src = "{% if enable_thinking %}T{% else %}N{% endif %}";
        assert_eq!(render(src, true, true), "T");
        assert_eq!(render(src, true, false), "N");
    }

    #[test]
    fn is_defined_on_absent_field_is_falsy() {
        // The shipped LFM2.5 template branches on
        // `message.thinking is defined`, where the message object has no
        // `thinking` key at all. This is the exact shape that must not
        // blow up or flip the branch.
        let src = "{% for m in messages %}\
                   {% if m.thinking is defined %}yes{% else %}no{% endif %}{% endfor %}";
        assert_eq!(render(src, true, false), "no");
    }

    #[test]
    fn is_defined_sees_present_field() {
        let src = "{% for m in messages %}\
                   {% if m.thinking is defined %}yes{% else %}no{% endif %}{% endfor %}";
        let t = JinjaChatTemplate::compile(src, "test").unwrap();
        let msgs = [ChatMessage {
            role: "assistant".into(),
            content: Value::String(String::new()),
        }];
        // A message carrying `thinking` must take the defined branch.
        let ctx =
            json!({ "messages": [ { "role": "assistant", "content": "", "thinking": "hm" } ] });
        assert_eq!(
            t.env
                .get_template(TEMPLATE_NAME)
                .unwrap()
                .render(&ctx)
                .unwrap(),
            "yes"
        );
        let _ = msgs;
    }

    #[test]
    fn broken_template_reports_origin() {
        match JinjaChatTemplate::compile("{% for %}", "myfile.jinja") {
            Ok(_) => panic!("expected a syntax error"),
            Err(e) => assert!(e.contains("myfile.jinja"), "{e}"),
        }
    }

    fn eval_with_content(template: &str, content: Option<&str>) -> String {
        let t = JinjaChatTemplate::compile(template, "t").unwrap();
        let mut msg = serde_json::Map::new();
        msg.insert("role".into(), json!("user"));
        match content {
            Some(c) => {
                msg.insert("content".into(), json!(c));
            }
            None => {}
        }
        let ctx = json!({ "messages": [Value::Object(msg)] });
        t.env
            .get_template(TEMPLATE_NAME)
            .unwrap()
            .render(&ctx)
            .unwrap()
    }

    #[test]
    fn map_get_preserves_none_semantics() {
        // The old test only checked truthiness, which made it look like the
        // rewrite was equivalent. It is not: `x["k"] is none` is false in
        // minijinja while `x.get("k") is none` is true in Jinja2. This pins
        // the identity test that actually changed.
        let via_get = "{% set d = {} %}{% if d.get(\"k\") is none %}NONE{% else %}NOT{% endif %}";
        let via_index = "{% set d = {} %}{% if d[\"k\"]|default(__py_none) is none %}NONE{% else %}NOT{% endif %}";
        assert_eq!(render(via_get, true, false), render(via_index, true, false));
        assert_eq!(render(via_get, true, false), "NONE");
    }

    #[test]
    fn map_get_truthiness_is_still_equivalent() {
        let via_get = "{% set d = {} %}{% if d.get(\"k\") %}T{% else %}F{% endif %}";
        let via_index =
            "{% set d = {} %}{% if d[\"k\"]|default(__py_none) %}T{% else %}F{% endif %}";
        assert_eq!(render(via_get, true, false), render(via_index, true, false));
    }

    #[test]
    fn get_inside_a_string_literal_is_not_rewritten() {
        // Regression: a naive `find(".get(")` scan rewrote string literals,
        // so this template rendered `the api ["k"] call`.
        let src = "{% set s = 'the api .get(\"k\") call' %}{{ s }}";
        assert_eq!(render(src, true, false), "the api .get(\"k\") call");
    }

    #[test]
    fn get_with_single_quotes_is_handled() {
        // Gemma4's template uses single-quoted keys; the first version of the
        // rewrite only recognised double quotes and left these to fail with
        // "unknown method: map has no method named get".
        let src = "{% set d = {'k': 'v'} %}{{ d.get('k') }}";
        assert_eq!(render(src, true, false), "v");
    }

    #[test]
    fn startswith_on_a_variable_works() {
        // Qwen3.5's template calls `.startswith(...)`, which minijinja
        // rejects with "unknown method: string has no method named
        // startswith". It has no `add_method`, so the call is rewritten to the
        // registered `startswith` filter.
        let src = "{% set s = '<think>x' %}{% if s.startswith('<think>') %}T{% else %}F{% endif %}";
        assert_eq!(render(src, true, false), "T");
    }

    #[test]
    fn startswith_on_an_indexed_field_works() {
        let src =
            "{% set m = [{'c': 'hello'}] %}{% if m[0].c.startswith('he') %}T{% else %}F{% endif %}";
        assert_eq!(render(src, true, false), "T");
    }

    #[test]
    fn startswith_false_branch_is_reachable() {
        let src = "{% set s = 'nope' %}{% if s.startswith('yes') %}T{% else %}F{% endif %}";
        assert_eq!(render(src, true, false), "F");
    }

    #[test]
    fn endswith_lstrip_rstrip_and_split_work() {
        // Each of these is a Python string method that minijinja has no method
        // for; they are rewritten onto registered filters.
        assert_eq!(
            render(
                "{% set s = 'x.py' %}{% if s.endswith('.py') %}T{% else %}F{% endif %}",
                true,
                false
            ),
            "T"
        );
        assert_eq!(
            render("{% set s = '  a,b  ' %}[{{ s.lstrip() }}]", true, false),
            "[a,b  ]"
        );
        assert_eq!(
            render("{% set s = '  a,b  ' %}[{{ s.rstrip() }}]", true, false),
            "[  a,b]"
        );
        assert_eq!(
            render(
                "{% set s = '  a,b  ' %}{{ s.split(',')|join('|') }}",
                true,
                false
            ),
            "  a|b  "
        );
    }

    #[test]
    fn get_inside_a_raw_block_is_left_alone() {
        let src = "{% raw %}d.get(\"k\"){% endraw %}";
        assert_eq!(render(src, true, false), "d.get(\"k\")");
    }

    #[test]
    fn tmp_dbg2() {
        let src =
            std::fs::read_to_string("models/Edge0-35B-A3B-preview/chat_template.jinja").unwrap();
        let out = rewrite_python_compat(&strip_llamacpp_extensions(&src));
        for (n, l) in out.lines().enumerate() {
            if n + 1 == 95 || n + 1 == 96 {
                println!("L{}: {}", n + 1, l);
            }
        }
    }

    #[test]
    fn tmp_dbg3() {
        for src in [
            r#"{% set s = '<think>x' %}{% if s.startswith('<think>') %}T{% else %}F{% endif %}"#,
            r#"{% set m = [{'c': 'hello'}] %}{% if m[0].c.startswith('he') %}T{% endif %}"#,
            r#"{{ 'a,b'.split(',')[0] }}"#,
        ] {
            println!("IN : {}", src);
            println!("MID: {}", rewrite_python_compat(src));
            println!(
                "OUT: {}",
                rewrite_string_methods(&rewrite_python_compat(src))
            );
        }
    }

    #[test]
    fn chained_string_methods_keep_python_semantics() {
        // Edge0 ships `content.split('</think>')[0].rstrip('\n').split('<think>')[-1]`,
        // so the rewrite has to survive a chain *and* keep the value right, not
        // just parse. The expected values are what CPython produces for
        // c = 'a</think>b<think>c': 'a' and 'b<think>c'.
        let src = "{% set c = 'a</think>b<think>c' %}\
                   {% set r = c.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}\
                   {% set t = c.split('</think>')[-1].lstrip('\n') %}\
                   [{{ r }}][{{ t }}]";
        assert_eq!(render(src, true, false), "[a][b<think>c]");
    }

    #[test]
    fn split_returns_a_real_list() {
        // minijinja's builtin `split` yields a lazy sequence whose `[-1]`
        // returns the *first* element; Python needs the last.
        assert_eq!(render("{{ ['a','b','c'][-1] }}", true, false), "c");
        assert_eq!(
            render("{% set s = 'a,b,c' %}{{ (s|split(','))[-1] }}", true, false),
            "c"
        );
    }

    #[test]
    fn lstrip_and_rstrip_accept_a_character_set() {
        assert_eq!(render("{{ 'a\n\n'|rstrip('\n') }}", true, false), "a");
        assert_eq!(render("{{ '\n\na'|lstrip('\n') }}", true, false), "a");
        assert_eq!(render("{{ '  a  '|rstrip() }}", true, false), "  a");
    }

    /// A media part whose `type` the template can branch on.
    const MEDIA_ECHO: &str = "{% for m in messages %}{% for c in m.content %}\
        [{{ c.type }}]{% endfor %}{% endfor %}";

    #[test]
    fn media_kind_is_preserved() {
        let t = JinjaChatTemplate::compile(MEDIA_ECHO, "t").unwrap();
        let rendered = t
            .render(
                &[ChatMessage {
                    role: "user".into(),
                    content: serde_json::json!([
                        { "type": "image" },
                        { "type": "video" },
                        { "type": "text", "text": "hi" }
                    ]),
                }],
                true,
                false,
                &SpecialTokens::default(),
            )
            .unwrap();
        assert_eq!(rendered, "[image][video][text]");
    }

    #[test]
    fn image_only_turn_keeps_its_image() {
        // Regression: a single image with no user text used to collapse to `""`
        // and the attachment disappeared from the prompt entirely.
        let content = build_user_content(&[MediaPart::image(4)], "");
        assert_eq!(content, serde_json::json!([{ "type": "image" }]));
    }

    #[test]
    fn content_shapes() {
        assert_eq!(build_user_content(&[], "hi"), Value::String("hi".into()));
        assert_eq!(build_user_content(&[], ""), Value::String("".into()));
        assert_eq!(
            build_user_content(&[MediaPart::image(4)], "hi"),
            serde_json::json!([{ "type": "image" }, { "type": "text", "text": "hi" }])
        );
        assert_eq!(
            build_user_content(
                &[
                    MediaPart::image(4),
                    MediaPart {
                        kind: MediaKind::Video,
                        token_count: 9,
                    },
                ],
                "",
            ),
            serde_json::json!([{ "type": "image" }, { "type": "video" }])
        );
    }

    #[test]
    fn media_kind_labels_are_stable() {
        assert_eq!(MediaKind::Image.as_str(), "image");
        assert_eq!(MediaKind::Video.as_str(), "video");
        assert_eq!(MediaKind::Audio.as_str(), "audio");
    }

    #[test]
    fn two_arg_get_is_left_alone() {
        // `.get(k, default)` really does change meaning; we must not
        // rewrite it, and minijinja should be the one to complain.
        let src = rewrite_python_compat(r#"{% if m.get("k", "d") %}x{% endif %}"#);
        assert!(src.contains(r#".get("k", "d")"#), "{src}");
    }

    #[test]
    fn generation_tag_becomes_a_noop_comment() {
        let src = strip_llamacpp_extensions("A{% generation %}B{% endgeneration %}C");
        assert!(!src.contains("generation"), "{src}");
        assert_eq!(render(&src, true, false), "ABC");
    }

    #[test]
    fn generation_tag_keeps_whitespace_control() {
        // `{%- generation -%}` trims both sides; the replacement must too,
        // or every turn in the LFM2.5 template gains a blank line.
        let src = strip_llamacpp_extensions("A\n  {%- generation -%}  \nB");
        assert_eq!(render(&src, true, false), "AB");
    }

    #[test]
    fn options_off_by_default_never_renders_jinja() {
        // The whole no-regression argument rests on this: with no flags the
        // Jinja2 path must not be taken, or every model's token ids move.
        let opts = Options::default();
        assert!(!opts.enabled());
        let got = opts
            .resolve(&|_| panic!("metadata must not be read when jinja is off"))
            .unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn jinja_without_a_shipped_template_falls_back() {
        // `--jinja` on a GGUF that has no `tokenizer.chat_template` must
        // fall back to the hand-written builder, not fail.
        let opts = Options {
            jinja: true,
            file: None,
        };
        let got = opts.resolve(&|_| None).unwrap();
        assert!(got.is_none(), "expected fallback, got a template");
    }

    #[test]
    fn jinja_reads_the_shipped_template() {
        let opts = Options {
            jinja: true,
            file: None,
        };
        let got = opts
            .resolve(&|k| {
                (k == "tokenizer.chat_template")
                    .then(|| MetaValue::String("{{ messages[0].content }}".into()))
            })
            .unwrap();
        let t = got.expect("expected a template");
        assert_eq!(
            t.render(
                &[ChatMessage::text("user", "hello")],
                true,
                false,
                &SpecialTokens::default()
            )
            .unwrap(),
            "hello"
        );
    }

    #[test]
    fn a_file_override_implies_jinja() {
        let dir = std::env::temp_dir().join("rust_model_inference_jinja_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.jinja");
        std::fs::write(&path, "FILE:{{ messages[0].content }}").unwrap();
        let opts = Options {
            jinja: false, // deliberately not set
            file: Some(path.clone()),
        };
        assert!(opts.enabled(), "a template file must imply jinja");
        let t = opts
            .resolve(&|_| None)
            .unwrap()
            .expect("template from file");
        assert_eq!(
            t.render(
                &[ChatMessage::text("user", "x")],
                true,
                false,
                &SpecialTokens::default()
            )
            .unwrap(),
            "FILE:x"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn vision_placeholder_expands_to_a_contiguous_run() {
        // Template emits one pad per image; the position builder wants a run.
        assert_eq!(
            expand_vision_placeholders(&[1, 99, 2], 99, &[4]).unwrap(),
            vec![1, 99, 99, 99, 99, 2]
        );
    }

    #[test]
    fn two_media_items_expand_in_order() {
        assert_eq!(
            expand_vision_placeholders(&[1, 99, 2, 99, 3], 99, &[2, 3]).unwrap(),
            vec![1, 99, 99, 2, 99, 99, 99, 3]
        );
    }

    #[test]
    fn placeholder_count_mismatch_is_an_error() {
        // A mismatch would misalign every position id, so it must not be
        // papered over.
        assert!(expand_vision_placeholders(&[1, 99, 2], 99, &[4, 4]).is_err());
        assert!(expand_vision_placeholders(&[1, 2], 99, &[4]).is_err());
    }

    #[test]
    fn zero_count_grid_is_an_error() {
        assert!(expand_vision_placeholders(&[1, 99], 99, &[0]).is_err());
    }

    #[test]
    fn no_media_leaves_the_sequence_untouched() {
        assert_eq!(
            expand_vision_placeholders(&[1, 2, 3], 99, &[]).unwrap(),
            vec![1, 2, 3]
        );
    }

    /// A vision template rendered with an image content part must contain
    /// exactly one placeholder, so expansion has a well-defined target.
    /// Uses the real Qwen3-VL template shape (vision control tokens).
    #[test]
    fn image_content_part_renders_one_placeholder() {
        let src = "{% for m in messages %}{% for c in m.content %}\
                   {%- if c.type == 'image' -%}<|vision_start|><|image_pad|><|vision_end|>\
                   {%- else -%}{{ c.text }}{%- endif -%}{% endfor %}{% endfor %}";
        let t = JinjaChatTemplate::compile(src, "t").unwrap();
        let msgs = [ChatMessage {
            role: "user".into(),
            content: serde_json::json!([
                { "type": "image" },
                { "type": "text", "text": "hi" }
            ]),
        }];
        let out = t
            .render(&msgs, true, false, &SpecialTokens::default())
            .unwrap();
        assert_eq!(out, "<|vision_start|><|image_pad|><|vision_end|>hi");
    }

    #[test]
    fn unknown_statement_is_not_mangled() {
        // We only rewrite what we understand; anything else must reach
        // minijinja so the error names the real problem.
        let err = match JinjaChatTemplate::compile("{% frobnicate %}", "t") {
            Ok(_) => panic!("expected an error"),
            Err(e) => e,
        };
        assert!(
            err.contains("frobnicate") || err.contains("invalid chat template"),
            "{err}"
        );
    }

    /// Render the template a real GGUF ships, for eyeballing the output
    /// against the hand-written builders. Not a correctness gate: the
    /// assertion only checks that something sensible came out.
    ///
    /// `JINJA_PROBE_GGUF=path/to.gguf cargo test -- --ignored jinja_probe`
    #[test]
    #[ignore = "needs a real GGUF via JINJA_PROBE_GGUF"]
    fn jinja_probe() {
        let Ok(path) = std::env::var("JINJA_PROBE_GGUF") else {
            eprintln!("skipped: set JINJA_PROBE_GGUF=<file.gguf>");
            return;
        };
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skipped: cannot read {path}");
            return;
        };
        // Minimal GGUF metadata walk: find `tokenizer.chat_template`.
        let mut i = 8usize;
        let u64at =
            |b: &[u8], o: usize| -> u64 { u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) };
        let u32at =
            |b: &[u8], o: usize| -> u32 { u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) };
        let _ = u32at(bytes.as_slice(), 4);
        let n_tensors = u64at(&bytes, 8);
        let n_kv = u64at(&bytes, 16);
        assert!(n_tensors > 0 && n_kv > 0, "not a GGUF");
        i = 24;
        let mut src: Option<String> = None;
        for _ in 0..n_kv {
            let kl = u64at(&bytes, i) as usize;
            let key = String::from_utf8_lossy(&bytes[i + 8..i + 8 + kl]).into_owned();
            i += 8 + kl;
            let ty = u32at(&bytes, i);
            i += 4;
            match ty {
                8 => {
                    let vl = u64at(&bytes, i) as usize;
                    let val = &bytes[i + 8..i + 8 + vl];
                    if key == "tokenizer.chat_template" {
                        src = Some(String::from_utf8_lossy(val).into_owned());
                    }
                    i += 8 + vl;
                }
                9 => {
                    let et = u32at(&bytes, i);
                    let cnt = u64at(&bytes, i + 4) as usize;
                    i += 12;
                    match et {
                        8 => {
                            for _ in 0..cnt {
                                let l = u64at(&bytes, i) as usize;
                                i += 8 + l;
                            }
                        }
                        4 | 5 | 6 | 7 => i += cnt * 4,
                        10 | 11 => i += cnt * 8,
                        _ => panic!("nested array"),
                    }
                }
                0 | 1 | 7 => i += 1,
                2 | 3 => i += 2,
                4 | 5 | 6 => i += 4,
                10 | 11 => i += 8,
                _ => panic!("type {ty}"),
            }
        }
        let Some(src) = src else {
            eprintln!("skipped: {path} has no tokenizer.chat_template");
            return;
        };
        eprintln!("template: {} chars", src.len());
        let t = JinjaChatTemplate::compile(&src, &path).expect("compile");
        let special = SpecialTokens {
            bos: "<|endoftext|>".into(),
            eos: "<|im_end|>".into(),
        };
        for thinking in [true, false] {
            let out = t
                .render(
                    &[ChatMessage::text("user", "Capital of France?")],
                    true,
                    thinking,
                    &special,
                )
                .expect("render");
            eprintln!("--- thinking={thinking} ---\n{out}\n---");
        }
        assert!(!src.is_empty());
    }
}
