//! Jinja2 chat-template rendering.
//!
//! GGUF files ship the model's own chat template in the
//! `tokenizer.chat_template` metadata key as a Jinja2 source string. The
//! hand-written builders in [`super`] and [`super::legacy`] can only cover
//! a handful of formats, and they drift: LFM2.5's real template is a 4.6 kB Jinja program with
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

use minijinja::{Environment, UndefinedBehavior};
use serde_json::{json, Value};

use super::jinja_compat::{python_method, strip_llamacpp_extensions};

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
        // Python method compatibility. minijinja has no `add_method`, but it
        // does have this callback, which fires whenever a method call on a map
        // or string would otherwise raise UnknownMethod. Handling the methods
        // here instead of rewriting the template *source* is what makes this
        // safe: a rewriter has to guess where the receiver expression ends,
        // and every guess that is wrong either mangles the template or panics
        // on a multibyte character. With the callback the source reaches
        // minijinja untouched.
        env.set_unknown_method_callback(python_method);
        // Own the source: `add_template_owned` takes a `Cow`, and a borrowed
        // `&str` would tie the template's storage to this function's lifetime.
        env.add_template_owned(TEMPLATE_NAME, strip_llamacpp_extensions(source))
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
    // `None` means the key is absent: the caller falls back. A key that is
    // present but not a string is malformed metadata, and quietly treating it
    // as absent would make an explicit `--jinja` silently use the hand-written
    // builder instead of reporting the problem.
    let value = metadata("tokenizer.chat_template")?;
    let Some(raw) = value.to_string_val() else {
        return Some(Err(
            "GGUF tokenizer.chat_template is present but is not a string".to_string(),
        ));
    };
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
        // Never let `add_special` decide: it would add BOS to models whose
        // template already emitted one. The tokenizer's intent is honoured
        // below instead, and only when the template left it out.
        add_special: false,
        parse_special: true,
    };
    let mut ids = tokenizer.encode(&text, options);
    // A template that renders `{{ bos_token }}` has already produced it, so
    // only add it when the tokenizer asks for it *and* the result does not
    // start with it. Ministral3 is the case that matters: its GGUF sets
    // `add_bos_token = true` and its `[INST]` template never mentions BOS, so
    // with a blanket `add_special: false` the prompt lost its `<s>`.
    if tokenizer.add_bos() && ids.first().copied() != tokenizer.bos_id() {
        if let Some(bos) = tokenizer.bos_id() {
            ids.insert(0, bos);
        }
    }
    Ok(ids)
}

/// The literal BOS/EOS spellings, for callers that render text and then encode
/// it themselves.
///
/// Templates may interpolate `{{ bos_token }}` / `{{ eos_token }}`, so handing
/// them empty strings silently drops those control tokens. Public because
/// three call sites used to each open-code this lookup.
pub fn special_token_literals(tokenizer: &dyn crate::core::tokenizer::Tokenizer) -> SpecialTokens {
    special_tokens_from_tokenizer(tokenizer)
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

/// Render a conversation with an already-resolved template.
///
/// `conversation_tokens` resolves `Options` on every call, which re-reads the
/// GGUF metadata and re-compiles the template. A long-lived caller such as the
/// server's `TextBackend` should resolve once at load time and use this.
pub fn conversation_tokens_with(
    tokenizer: &dyn crate::core::tokenizer::Tokenizer,
    template: Option<&JinjaChatTemplate>,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Result<Option<Vec<u32>>, String> {
    let Some(template) = template else {
        return Ok(None);
    };
    Ok(Some(render_tokens(
        tokenizer,
        template,
        messages,
        add_generation_prompt,
        enable_thinking,
    )?))
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
    let template = opts.resolve(metadata)?;
    // Routed through `render_tokens` so the tokenizer's BOS contract applies
    // here too. Encoding inline duplicated the path and skipped the fixup,
    // which cost the server chat route its `<s>` on models like Ministral.
    conversation_tokens_with(
        tokenizer,
        template.as_ref(),
        messages,
        add_generation_prompt,
        enable_thinking,
    )
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
        parts.push(json!({ "type": m.content_type }));
    }
    if !user_text.is_empty() {
        parts.push(json!({ "type": "text", "text": user_text }));
    }
    Value::Array(parts)
}

/// One media attachment in a multimodal prompt.
///
/// `content_type` is the `type` a template branches on -- `"image"`, `"video"`
/// or `"audio"`. It is a plain string rather than an enum on purpose: the
/// attachment kind is decided once, where the media is decoded
/// (`crate::app::media::MediaKind`), and duplicating that enum here only
/// created a second thing that could drift out of sync.
#[derive(Clone, Copy, Debug)]
pub struct MediaPart {
    pub content_type: &'static str,
    /// How many placeholders this attachment expands to.
    pub token_count: usize,
}

impl MediaPart {
    pub fn new(content_type: &'static str, token_count: usize) -> Self {
        Self {
            content_type,
            token_count,
        }
    }

    pub fn image(token_count: usize) -> Self {
        Self::new("image", token_count)
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
    template: Option<&JinjaChatTemplate>,
    placeholder_id: u32,
    media: &[MediaPart],
    user_text: &str,
    system_text: Option<&str>,
    enable_thinking: bool,
) -> Result<Option<Vec<u32>>, String> {
    let media_counts: Vec<usize> = media.iter().map(|m| m.token_count).collect();
    let Some(mut ids) = text_conversation_tokens(
        tokenizer,
        template,
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
/// The template is passed already resolved rather than as `Options`.
///
/// Resolution reads GGUF metadata or a file, so it belongs once at config time.
/// Taking `Options` here would let a caller re-resolve with different inputs,
/// which is how the multimodal path lost `--chat-template-file`: the JEV scorer
/// kept only the resolved template and then rebuilt `Options { file: None }` to
/// satisfy this signature, silently falling back to the GGUF's own template.
pub fn text_conversation_tokens(
    tokenizer: &dyn crate::core::tokenizer::Tokenizer,
    template: Option<&JinjaChatTemplate>,
    system_text: Option<&str>,
    user_text: &str,
    media: &[MediaPart],
    enable_thinking: bool,
) -> Result<Option<Vec<u32>>, String> {
    let Some(template) = template else {
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
    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(system) = system_text.filter(|s| !s.trim().is_empty()) {
        messages.push(ChatMessage::text("system", system));
    }
    messages.push(ChatMessage::text("user", user_text));
    // Same reason as `conversation_tokens`: the BOS fixup lives in
    // `render_tokens`. This path is what every JEV scorer uses, including the
    // mistral3 one, so an inline encode silently diverged from the tokenizer
    // contract.
    render_tokens(tokenizer, template, &messages, true, enable_thinking)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::MetaValue;
    use crate::core::tokenizer::MockTokenizer;

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
            build_user_content(&[MediaPart::image(4), MediaPart::new("video", 9)], "",),
            serde_json::json!([{ "type": "image" }, { "type": "video" }])
        );
    }

    #[test]
    fn media_part_carries_its_content_type() {
        assert_eq!(MediaPart::image(4).content_type, "image");
        assert_eq!(MediaPart::new("video", 4).content_type, "video");
        assert_eq!(MediaPart::new("audio", 4).content_type, "audio");
    }

    /// A tokenizer that asks for BOS must still end up with exactly one, whether
    /// the template emits `{{ bos_token }}` itself or not.
    ///
    /// Falcon-H1 forces `add_bos`, and its Jinja text used to be encoded with
    /// `add_special: true`, so a template emitting `{{ bos_token }}` produced two
    /// BOS tokens. All four combinations are pinned here.
    #[test]
    fn bos_is_added_at_most_once() {
        let asks = MockTokenizer::with_bos(1, true);
        let silent = MockTokenizer::with_bos(1, false);
        let emitting = JinjaChatTemplate::compile("<s>{{ messages[0].content }}", "t").unwrap();
        let plain = JinjaChatTemplate::compile("{{ messages[0].content }}", "t").unwrap();
        let count_bos = |tok: &MockTokenizer, tpl: &JinjaChatTemplate| {
            let ids =
                render_tokens(tok, tpl, &[ChatMessage::text("user", "hi")], true, false).unwrap();
            ids.iter().filter(|&&t| t == 1).count()
        };

        // Template emits BOS: one, regardless of what the tokenizer wants.
        assert_eq!(count_bos(&silent, &emitting), 1);
        assert_eq!(count_bos(&asks, &emitting), 1);
        // Template is silent: only added when the tokenizer asks.
        assert_eq!(count_bos(&silent, &plain), 0);
        assert_eq!(count_bos(&asks, &plain), 1);
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
