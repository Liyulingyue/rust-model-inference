//! Per-architecture default chat templates.
//!
//! Each model family ships with its own chat format encoded in the GGUF
//! metadata (`tokenizer.chat_template` is a Jinja2 source string we can't
//! render without a template engine). When the user passes `--system`,
//! we wrap the prompt with the model's *expected* template so the
//! underlying weights interpret the input correctly.
//!
//! Returns `None` from [`format_chat`] when no system prompt is given,
//! which keeps the inference in base-model continuation mode for raw
//! text comparison (e.g. parity tests against llama.cpp).

/// Render the chat-mode prompt for the given architecture.
///
/// - `arch` is the GGUF `general.architecture` value (e.g. `"nemotron_h"`,
///   `"qwen3"`, `"llama"`).
/// - `preset_override` lets the caller force a specific template name
///   (e.g. `"chatml"`, `"llama3"`, `"gemma"`, `"lfm2"`, `"none"`).
///   `"none"` or empty string forces base mode; `"auto"` (or `None`)
///   falls back to `default_template(arch)`.
/// - `user_msg` is the raw user prompt (system messages should be
///   embedded in `user_msg` by the caller, mirroring llama.cpp's
///   `--chat-template` flow without a separate `--system` flag).
///
/// Returns `Some(formatted)` when a template is selected; `None` when
/// the architecture is unknown AND no override is supplied (caller
/// keeps the raw prompt for parity tests).
pub fn format_chat(arch: &str, preset_override: Option<&str>, user_msg: &str) -> Option<String> {
    let template = resolve_template(arch, preset_override)?;
    Some(template.render(user_msg))
}

/// The set of supported chat-template formats. Each variant knows how
/// to render `(system, user_msg) -> formatted prompt`.
///
/// We deliberately keep the variants small and explicit instead of
/// pulling in `minijinja` / `tera` — the full Jinja2 templates in
/// GGUF files add features (thinking, tool calls) that the base
/// models in this repo don't have instruct fine-tunes for, so the
/// extra surface area isn't useful here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Qwen-family ChatML: `<|im_start|>role\n...<|im_end|>\n`.
    /// Used by Qwen2 / Qwen3 / Qwen35 / Nemotron-3 Nano (built on Qwen3).
    ChatML,
    /// Llama 3.x: `<|begin_of_text|><|start_header_id|>...<|end_header_id|>...`.
    Llama3,
    /// Gemma: `<start_of_turn>role\n...<end_of_turn>\n`.
    GemmaTurn,
    /// LFM2 / LFM2.5: `<|start_of_role|>role<|end_of_role|>...<|end_of_text|>`.
    Lfm2,
}

impl ChatTemplate {
    /// Render `user_msg` using this template. The user message is
    /// expected to contain any system-level instructions (the caller
    /// baked them into the prompt if needed).
    pub fn render(self, user_msg: &str) -> String {
        match self {
            Self::ChatML => format!(
                "<|im_start|>user\n{user_msg}<|im_end|>\n\
                 <|im_start|>assistant\n"
            ),
            Self::Llama3 => format!(
                "<|begin_of_text|>\
                 <|start_header_id|>user<|end_header_id|>\n\n\
                 {user_msg}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            Self::GemmaTurn => format!(
                "<start_of_turn>user\n{user_msg}<end_of_turn>\n\
                 <start_of_turn>model\n"
            ),
            Self::Lfm2 => format!(
                "<|start_of_role|>user<|end_of_role|>{user_msg}<|end_of_text|>\
                 <|start_of_role|>assistant<|end_of_role|>"
            ),
        }
    }

    /// Static string for `--help` / debugging.
    pub fn name(self) -> &'static str {
        match self {
            Self::ChatML => "ChatML (<|im_start|>/<|im_end|>)",
            Self::Llama3 => "Llama-3 (<|start_header_id|>)",
            Self::GemmaTurn => "Gemma (<start_of_turn>)",
            Self::Lfm2 => "LFM2 (<|start_of_role|>)",
        }
    }
}

/// Map a `general.architecture` value to its canonical chat template.
///
/// New architectures default to `None` (raw base-model mode). When the
/// GGUF gains a new arch, add a match arm here rather than hardcoding
/// the wrapping at the inference site.
pub fn default_template(arch: &str) -> Option<ChatTemplate> {
    Some(match arch {
        // Qwen family — ChatML is the standard format and matches the
        // GGUF `tokenizer.chat_template` field for these archs.
        "qwen2" | "qwen2vl" | "qwen3" | "qwen3vl" | "qwen3vlmoe" | "qwen35" | "qwen3tts"
        | "nemotron_h" => ChatTemplate::ChatML,

        // Llama family.
        "llama" | "k2-horizon" | "granite" | "nanbeige" => ChatTemplate::Llama3,

        // Gemma — `<start_of_turn>...<end_of_turn>`.
        "gemma4" => ChatTemplate::GemmaTurn,

        // LFM2 / LFM2.5 — Liquid AI's own role markers.
        "lfm2" | "lfm2moe" => ChatTemplate::Lfm2,

        // Hunyuan-Dense and others: no chat template baked in; fall
        // back to base-mode (caller keeps the raw prompt).
        _ => return None,
    })
}

/// Parse a `--chat-template <preset>` value into a `ChatTemplate`.
///
/// Accepted presets:
/// - `chatml` / `qwen` / `qwen3` → ChatML
/// - `nemotron` / `nemotron_h` → ChatML (Nemotron-3 Nano is built on Qwen3
///   so the marker set is shared; if/when we add Jinja2 rendering this
///   preset will load the GGUF's bespoke 10 KB Jinja template with
///   thinking + tool-call rendering)
/// - `llama3` / `llama` → Llama-3
/// - `gemma` / `gemma4` → Gemma turn
/// - `lfm2` → LFM2
/// - `none` / `off` / `base` → `None` (forces base mode)
/// - `auto` (or unknown) → `None` here; the caller then falls back to
///   `default_template(arch)`.
///
/// Returns `None` for `none/off/base/auto`, `Some(ChatTemplate)` for
/// the explicit presets. Names are case-insensitive.
pub fn parse_preset(name: &str) -> Option<ChatTemplate> {
    match name.trim().to_ascii_lowercase().as_str() {
        "chatml" | "qwen" | "qwen3" => Some(ChatTemplate::ChatML),
        "nemotron" | "nemotron_h" => Some(ChatTemplate::ChatML),
        "llama3" | "llama" => Some(ChatTemplate::Llama3),
        "gemma" | "gemma4" => Some(ChatTemplate::GemmaTurn),
        "lfm2" => Some(ChatTemplate::Lfm2),
        // "none" / "off" / "base" / "auto" all return None — the caller
        // treats these as "use default or skip wrapping" depending on
        // context.
        _ => None,
    }
}

/// Resolve the template to use, honouring an explicit `--chat-template`
/// preset override and falling back to the per-arch default.
///
/// `override_preset` is the raw `--chat-template` argument string:
/// - `None` / empty / `"auto"` → use `default_template(arch)`
/// - `"none"` / `"off"` → return `None` (force base mode)
/// - explicit preset name → that template, ignoring the per-arch default
///
/// Returns `Some(template)` to wrap, or `None` to keep the raw prompt.
fn resolve_template(arch: &str, override_preset: Option<&str>) -> Option<ChatTemplate> {
    let preset = override_preset.map(str::trim).filter(|s| !s.is_empty());
    match preset.map(str::to_ascii_lowercase).as_deref() {
        Some("none") | Some("off") | Some("base") => None,
        Some("auto") => default_template(arch),
        Some(name) => parse_preset(name).or_else(|| default_template(arch)),
        None => default_template(arch),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_wraps_user_and_assistant() {
        let s = ChatTemplate::ChatML
            .render("Capital of France?")
            .replace('\n', "\\n");
        assert_eq!(
            s,
            "<|im_start|>user\\nCapital of France?<|im_end|>\\n\
             <|im_start|>assistant\\n"
        );
    }

    #[test]
    fn llama3_uses_header_id_tokens() {
        let s = ChatTemplate::Llama3.render("hi");
        assert!(s.starts_with("<|begin_of_text|><|start_header_id|>user"));
        assert!(s.contains("<|eot_id|>"));
    }

    #[test]
    fn parse_preset_recognises_known_names() {
        assert_eq!(parse_preset("chatml"), Some(ChatTemplate::ChatML));
        assert_eq!(parse_preset("ChatML"), Some(ChatTemplate::ChatML));
        assert_eq!(parse_preset("nemotron"), Some(ChatTemplate::ChatML));
        assert_eq!(parse_preset("llama3"), Some(ChatTemplate::Llama3));
        assert_eq!(parse_preset("gemma"), Some(ChatTemplate::GemmaTurn));
        assert_eq!(parse_preset("lfm2"), Some(ChatTemplate::Lfm2));
        assert_eq!(parse_preset("none"), None);
        assert_eq!(parse_preset("auto"), None);
    }

    #[test]
    fn format_chat_uses_per_arch_default() {
        let s = format_chat("nemotron_h", None, "hi").unwrap();
        assert!(s.contains("<|im_start|>user"));
        assert!(s.contains("<|im_start|>assistant"));
    }

    #[test]
    fn format_chat_override_wins_over_per_arch() {
        // nemotron normally uses ChatML, but `llama3` override should
        // force Llama-3 markers.
        let s = format_chat("nemotron_h", Some("llama3"), "hi").unwrap();
        assert!(s.contains("<|begin_of_text|>"));
    }

    #[test]
    fn format_chat_none_override_returns_raw() {
        assert!(format_chat("nemotron_h", Some("none"), "hi").is_none());
        assert!(format_chat("llama", Some("off"), "hi").is_none());
    }

    #[test]
    fn format_chat_unknown_arch_no_override() {
        assert!(format_chat("unknown_arch", None, "hi").is_none());
        assert!(format_chat("unknown_arch", Some("chatml"), "hi").is_some());
    }
}
