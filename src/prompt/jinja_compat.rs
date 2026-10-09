//! Python-flavoured Jinja2 constructs that minijinja does not implement.
//!
//! This used to rewrite template *source*. It no longer does, and that is the
//! whole point of the file existing in this shape.
//!
//! Rewriting the receiver of `x.foo(...)` requires knowing where the receiver
//! expression ends. Every version of that guess was wrong somewhere: it
//! panicked on a multibyte character, it glued `for` and `part` together, it
//! rewrote `dict.get("k")` inside prose, and it produced unbalanced brackets
//! for chained calls. minijinja's `Environment::set_unknown_method_callback`
//! hands us the receiver as an already-evaluated value, so there is nothing
//! left to parse and the source reaches minijinja untouched.
//!
//! What remains is `strip_llamacpp_extensions`, which removes tags rather
//! than expressions: llama.cpp's `{% generation %}` is not Jinja2 and must go,
//! but there is no expression to recover, only a tag to neutralise.

use minijinja::value::Value;
use minijinja::value::ValueKind;
use minijinja::{Error, ErrorKind, State};

/// Tags that are llama.cpp extensions rather than Jinja2.
const LLAMACPP_TAGS: [&str; 2] = ["generation", "endgeneration"];

/// Neutralise llama.cpp's `{% generation %}` / `{% endgeneration %}` tags.
///
/// These are not Jinja2, so minijinja rejects them. They only ever wrap
/// assistant output that we discard anyway, so replacing them with a comment
/// is enough. The whitespace-control markers are preserved, because a template
/// that relies on `{%- generation -%}` to trim surrounding whitespace must
/// keep trimming it.
///
/// This walks the source with `find` and slices at the resulting boundaries,
/// so multibyte text is safe: every index here comes from a substring search
/// and therefore lands on a char boundary.
pub(super) fn strip_llamacpp_extensions(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    // Everything in `source[..copied]` has already been emitted.
    let mut copied = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            // `{# ... #}` is already a comment.
            b'{' if bytes.get(i + 1) == Some(&b'#') => {
                i = match source[i..].find("#}") {
                    Some(p) => i + p + 2,
                    None => bytes.len(),
                };
                continue;
            }
            // `{% raw %} ... {% endraw %}` is literal text, tags included.
            b'{' if is_raw_open(&source[i..]) => {
                // `raw_block_end` is relative to the slice it is given, so the
                // offset has to be added back: assigning it directly rewound
                // the cursor and looped forever on any raw block that did not
                // start at offset 0.
                i = match raw_block_end(&source[i..]) {
                    Some(end) => i + end,
                    None => bytes.len(),
                };
                continue;
            }
            b'{' if bytes.get(i + 1) == Some(&b'%') => {
                let Some(close_rel) = source[i + 2..].find("%}") else {
                    break;
                };
                let close = i + 2 + close_rel;
                let tag_start = i;
                let after_tag = close + 2;
                // Read the whitespace-control markers off the raw tag body
                // before stripping them, otherwise `{%- generation -%}` loses
                // its trim.
                let raw_inner = source[i + 2..close].trim();
                let ltrim = raw_inner.starts_with('-') || raw_inner.starts_with('+');
                let rtrim = raw_inner.ends_with('-') || raw_inner.ends_with('+');
                let name = raw_inner
                    .trim_matches(|c: char| c == '-' || c == '+' || c.is_whitespace())
                    .split_whitespace()
                    .next()
                    .unwrap_or("");
                i = after_tag;
                if LLAMACPP_TAGS.contains(&name) {
                    out.push_str(&source[copied..tag_start]);
                    out.push_str(match (ltrim, rtrim) {
                        (true, true) => "{#- -#}",
                        (true, false) => "{#- #}",
                        (false, true) => "{# -#}",
                        (false, false) => "{# #}",
                    });
                    copied = after_tag;
                }
                continue;
            }
            // Quotes delimit strings only inside Jinja markup. Treating every
            // apostrophe in rendered prose as an opening quote made
            // `It's fine{% generation %}` skip past the tag, which then
            // reached minijinja unstripped and failed to compile.
            b'\'' | b'"' if inside_jinja_markup(source, i) => {
                let quote = b;
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        // Step over the escaped character whole, so `i` stays
                        // on a char boundary for multibyte content.
                        i += 1;
                        match source.get(i..).and_then(|rest| rest.chars().next()) {
                            Some(c) => i += c.len_utf8(),
                            None => break,
                        }
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
            _ => {}
        }
        // Advance a whole character so `i` stays on a char boundary.
        match source[i..].chars().next() {
            Some(c) => i += c.len_utf8(),
            None => break,
        }
    }
    out.push_str(&source[copied..]);
    out
}

fn is_raw_open(slice: &str) -> bool {
    slice.starts_with("{% raw") || slice.starts_with("{%- raw") || slice.starts_with("{%+ raw")
}

/// Whether `at` sits inside a Jinja block, i.e. a `{{ ... }}` or `{% ... %}`
/// region that has not been closed yet.
///
/// Only the *most recent* opener matters. Checking each delimiter kind
/// independently let an older closed `{% ... %}` mask a newer open `{{ ... }}`,
/// so in `{% set x = 1 %}{{ '{% generation %}' }}` the quote was treated as
/// prose and the tag-shaped string literal was rewritten.
fn inside_jinja_markup(source: &str, at: usize) -> bool {
    let before = &source[..at];
    let mut best: Option<(usize, bool)> = None; // (offset, is_comment)
    for (open, close, is_comment) in [("{{", "}}", false), ("{%", "%}", false), ("{#", "#}", true)]
    {
        let Some(o) = before.rfind(open) else {
            continue;
        };
        let c = before.rfind(close).unwrap_or(0);
        // Already closed before this position, so this opener does not span
        // `at`.
        if c > o {
            continue;
        }
        if best.is_none_or(|(b, _)| o > b) {
            best = Some((o, is_comment));
        }
    }
    // Inside a comment is not inside markup we need to skip strings for.
    best.is_some_and(|(_, is_comment)| !is_comment)
}

/// End offset, relative to the slice, just past the `{% endraw %}` tag.
fn raw_block_end(slice: &str) -> Option<usize> {
    let start = slice
        .find("{% endraw")
        .or_else(|| slice.find("{%- endraw"))
        .or_else(|| slice.find("{%+ endraw"))?;
    let close = slice[start..].find("%}")?;
    Some(start + close + 2)
}

/// Python's `str.lstrip` / `str.rstrip`: with no argument they strip
/// whitespace, with one they strip any of the given characters.
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

fn as_str(value: &Value) -> Result<&str, Error> {
    value
        .as_str()
        .ok_or_else(|| Error::from(ErrorKind::InvalidOperation))
}

/// A single string argument, as Python would take it.
fn one_str(args: &[Value]) -> Result<&str, Error> {
    let [arg] = args else {
        return Err(Error::from(ErrorKind::InvalidOperation));
    };
    as_str(arg)
}

/// Exactly two string arguments, e.g. `s.replace(old, new)`.
fn two_strs(args: &[Value]) -> Result<(&str, &str), Error> {
    let [first, second] = args else {
        return Err(Error::from(ErrorKind::InvalidOperation));
    };
    Ok((as_str(first)?, as_str(second)?))
}

/// Dispatch a Python method call that minijinja does not implement.
///
/// `value` is the already-evaluated receiver, which is why this can exist at
/// all: no source-level parsing is involved.
///
/// `dict.get` is the subtle one. Python returns `None` for a missing key, so
/// `x.get("k") is none` is true. minijinja distinguishes None from Undefined,
/// and `Undefined is none` is false, so the missing case must produce a real
/// `None` -- returning Undefined here would silently flip every `is none`
/// branch in the template.
pub(super) fn python_method(
    _state: &State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    match value.kind() {
        ValueKind::Map => {
            if method != "get" {
                return Err(Error::from(ErrorKind::UnknownMethod));
            }
            // Python's `get` takes a key, and optionally a default.
            let (key, default) = match args {
                [key] => (key, None),
                [key, default] => (key, Some(default)),
                _ => return Err(Error::from(ErrorKind::InvalidOperation)),
            };
            // Templates key maps with strings; anything else goes through
            // `get_item`, which hashes arbitrary values.
            let found = match key.as_str() {
                Some(name) => value.get_attr(name).ok(),
                None => value.get_item(key).ok(),
            }
            .filter(|v| !v.is_undefined());
            match found {
                Some(found) => Ok(found),
                // Python's `.get(key, default)` only applies the default on a
                // miss; a present key returns the value even if it is falsy.
                None if let Some(default) = default => Ok(default.clone()),
                // `Value::from(())` is a genuine None. minijinja's `none` is a
                // test, not a value, so it cannot be used for this.
                None => Ok(Value::from(())),
            }
        }
        ValueKind::String => {
            let s = as_str(value)?;
            Ok(match method {
                "startswith" => Value::from(s.starts_with(one_str(args)?)),
                "endswith" => Value::from(s.ends_with(one_str(args)?)),
                "lstrip" => Value::from(trim_py(s, args.first().and_then(|v| v.as_str()), true)),
                "rstrip" => Value::from(trim_py(s, args.first().and_then(|v| v.as_str()), false)),
                // A real list, not minijinja's lazy `split`: Python's
                // `s.split(p)[-1]` has to give the last element. Python also
                // takes an optional maxsplit, which has to be honoured or the
                // tail is joined into one element.
                "split" => {
                    let (sep, maxsplit) = match args {
                        [] => (None, None),
                        // A non-string separator is a type error in Python, not
                        // a request to split on whitespace.
                        [sep] => (Some(as_str(sep)?), None),
                        [sep, limit] => (
                            Some(as_str(sep)?),
                            Some(
                                limit
                                    .as_usize()
                                    .ok_or_else(|| Error::from(ErrorKind::InvalidOperation))?,
                            ),
                        ),
                        _ => return Err(Error::from(ErrorKind::InvalidOperation)),
                    };
                    let parts: Vec<String> = match (sep, maxsplit) {
                        (Some(pat), Some(limit)) => {
                            s.splitn(limit + 1, pat).map(str::to_string).collect()
                        }
                        (Some(pat), None) => s.split(pat).map(str::to_string).collect(),
                        (None, None) => s.split_whitespace().map(str::to_string).collect(),
                        // Unreachable: a separator is required whenever a
                        // maxsplit is given, but keep the arm total.
                        (None, Some(_)) => {
                            return Err(Error::from(ErrorKind::InvalidOperation));
                        }
                    };
                    Value::from_iter(parts)
                }
                // Common enough in templates, and trivial while we are here.
                "lower" => Value::from(s.to_lowercase()),
                "upper" => Value::from(s.to_uppercase()),
                // Python's `strip(chars)` takes an optional character set.
                // Ignoring it and calling `trim()` would also eat spaces: the
                // real Qwen3 template does `reasoning_content.strip('\n')`,
                // so `"\n  foo"` must keep its two leading spaces.
                "strip" => {
                    let set = args.first().and_then(|v| v.as_str());
                    let front = trim_py(s, set, true);
                    Value::from(trim_py(front, set, false))
                }
                // Python requires both arguments; dispatching through
                // `one_str` here would reject every valid call, since it
                // matches exactly one argument.
                "replace" => {
                    let (old, new) = two_strs(args)?;
                    Value::from(s.replace(old, new))
                }
                "count" => Value::from(s.matches(one_str(args)?).count()),
                _ => return Err(Error::from(ErrorKind::UnknownMethod)),
            })
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

#[cfg(test)]
mod tests {
    use super::{python_method, strip_llamacpp_extensions};
    use minijinja::{Environment, UndefinedBehavior};

    /// Compile and render with `context` (JSON) as the whole variable scope.
    fn render(source: &str, context: &str) -> Result<String, String> {
        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        env.set_unknown_method_callback(python_method);
        let ctx: minijinja::Value = serde_json::from_str(context).map_err(|e| e.to_string())?;
        env.add_template_owned("t", strip_llamacpp_extensions(source))
            .map_err(|e| e.to_string())?;
        env.get_template("t")
            .map_err(|e| e.to_string())?
            .render(&ctx)
            .map_err(|e| e.to_string())
    }

    #[test]
    fn generation_tag_becomes_a_comment_and_keeps_trim_markers() {
        assert_eq!(
            strip_llamacpp_extensions("A{% generation %}B{% endgeneration %}C"),
            "A{# #}B{# #}C"
        );
        assert_eq!(
            strip_llamacpp_extensions("A\n  {%- generation -%}  \nB"),
            "A\n  {#- -#}  \nB"
        );
    }

    /// Regression: the old source rewriter sliced at byte offsets that could
    /// land inside a multibyte character and panic. Chinese prompts are the
    /// common case, so this has to hold.
    #[test]
    fn multibyte_text_around_tags_does_not_panic() {
        for (src, ctx) in [
            ("你好{{ m }}", "{}"),
            ("{{ m }}中文", "{}"),
            ("{% generation %}中文内容{% endgeneration %}", "{}"),
            ("🎉🎉{{ m }}🎉", "{}"),
            ("Grüße {{ m }}", "{}"),
            ("{{ m }}", r#"{"m":"日本語"}"#),
        ] {
            assert!(render(src, ctx).is_ok(), "{src:?} failed");
        }
    }

    /// Regression: `dict.get("k")` in prose used to be rewritten, because the
    /// scan did not know which parts of the source were template code.
    #[test]
    fn plain_prose_containing_get_is_untouched() {
        assert_eq!(
            render(r#"{{ 'Use dict.get("key");' }}"#, "{}").unwrap(),
            "Use dict.get(\"key\");"
        );
    }

    #[test]
    fn get_on_an_indexed_map_can_still_be_indexed() {
        // Regression: the rewrite produced `...|default(none)[0]`, which does
        // not parse.
        assert_eq!(
            render(
                "{{ messages[0].get('content')[0] }}",
                r#"{"messages":[{"content":"ab"}]}"#
            )
            .unwrap(),
            "a"
        );
    }

    #[test]
    fn get_keeps_python_none_semantics() {
        // A missing key must satisfy `is none`, which Undefined does not.
        assert_eq!(
            render(
                "{% if m.get('k') is none %}NONE{% else %}NOT{% endif %}",
                r#"{"m":{}}"#
            )
            .unwrap(),
            "NONE"
        );
        assert_eq!(
            render(
                "{% if m.get('k') is none %}NONE{% else %}NOT{% endif %}",
                r#"{"m":{"k":1}}"#
            )
            .unwrap(),
            "NOT"
        );
    }

    #[test]
    fn get_honours_an_explicit_default() {
        assert_eq!(
            render("{{ m.get('k', 'fallback') }}", r#"{"m":{}}"#).unwrap(),
            "fallback"
        );
        // A present key wins even when the default differs.
        assert_eq!(
            render("{{ m.get('k', 'fallback') }}", r#"{"m":{"k":"real"}}"#).unwrap(),
            "real"
        );
    }

    /// Regression: the rewriter ate the space in `for part in ...`, producing
    /// `forpart in ...`.
    #[test]
    fn a_method_call_in_a_for_target_compiles() {
        assert_eq!(
            render(
                "{% for part in text.split(',') %}{{ part }};{% endfor %}",
                r#"{"text":"a,b"}"#
            )
            .unwrap(),
            "a;b;"
        );
    }

    #[test]
    fn chained_methods_keep_python_semantics() {
        // CPython: 'a</think>b<think>c'.split('</think>')[-1] -> 'b<think>c'
        assert_eq!(
            render(
                "{{ c.split('</think>')[-1].lstrip('\\n') }}",
                r#"{"c":"a</think>b<think>c"}"#
            )
            .unwrap(),
            "b<think>c"
        );
    }

    #[test]
    fn string_methods_dispatch() {
        // minijinja 2.24 renders booleans Python-style, as "True"/"False".
        let cases: [(&str, &str, &str); 9] = [
            ("{{ s.replace('a', 'Z') }}", "Zbc", "abc"),
            ("{{ s.startswith('ab') }}", "True", "abc"),
            ("{{ s.endswith('bc') }}", "True", "abc"),
            ("{{ s.upper() }}", "AB", "ab"),
            ("{{ s.lower() }}", "ab", "AB"),
            ("{{ s.strip() }}", "a", " a "),
            ("{{ s.rstrip('\\n') }}", "a", "a\n\n"),
            ("{{ s.lstrip('x') }}", "a", "xxa"),
            ("{{ s.split(',')|join('|') }}", "a|b", "a,b"),
        ];
        for (tpl, want, input) in cases {
            let ctx = serde_json::json!({ "s": input }).to_string();
            assert_eq!(render(tpl, &ctx).unwrap(), want, "{tpl} on {input:?}");
        }
        // The false branch has to be reachable too.
        assert_eq!(
            render("{{ s.startswith('zz') }}", r#"{"s":"abc"}"#).unwrap(),
            "False"
        );
    }

    /// Regression: `raw_block_end` returns a slice-relative offset and it was
    /// assigned to the absolute cursor, so any raw block after a non-empty
    /// prefix rewound the scan and looped forever. The earlier tests all put
    /// `{% raw %}` at offset 0, which hid it.
    #[test]
    fn a_raw_block_after_a_prefix_compiles() {
        for src in [
            "0123456789{% raw %}{% generation %}{% endraw %}tail",
            "{{ m }}{% raw %}a{% generation %}b{% endraw %}{{ m }}",
            "{%- if m -%}x{% endif %}{% raw %}{% generation %}{% endraw %}",
        ] {
            assert!(render(src, "{\"m\":\"v\"}").is_ok(), "{src} failed");
        }
        assert_eq!(
            render("0123456789{% raw %}{% generation %}{% endraw %}tail", "{}").unwrap(),
            "0123456789{% generation %}tail"
        );
    }

    /// Regression: every apostrophe in rendered prose used to open a Jinja
    /// string, hiding the tags that followed it.
    #[test]
    fn an_apostrophe_in_prose_does_not_hide_a_tag() {
        assert_eq!(
            render("It's fine{% generation %}x{% endgeneration %}", "{}").unwrap(),
            "It's finex"
        );
        assert_eq!(
            render(r#"He said "hi"{% generation %}y{% endgeneration %}"#, "{}").unwrap(),
            r#"He said "hi"y"#
        );
        // Inside markup the quote still delimits, so a tag-shaped string there
        // must survive.
        assert_eq!(
            render("{{ '{% generation %}' }}", "{}").unwrap(),
            "{% generation %}"
        );
    }

    #[test]
    fn methods_with_the_wrong_arity_error() {
        // Regression: `replace` dispatched through a one-argument helper, so
        // every valid `s.replace(old, new)` call failed.
        assert!(render("{{ s.replace('a') }}", r#"{"s":"abc"}"#).is_err());
        assert!(render("{{ s.startswith() }}", r#"{"s":"abc"}"#).is_err());
        assert!(render("{{ s.count() }}", r#"{"s":"abc"}"#).is_err());
    }

    /// The real Qwen3 template does `reasoning_content.strip('\n')`. Ignoring
    /// the character set and calling `trim()` also removed leading spaces,
    /// which changes the prompt for any reasoning text that starts with them.
    #[test]
    fn strip_honours_its_character_set() {
        // CPython: '\thi'.strip(' ') == '\thi', '\n hi '.strip('\n') == ' hi '
        assert_eq!(
            render("[{{ s.strip(' ') }}]", "{\"s\":\"\\thi\"}").unwrap(),
            "[\thi]"
        );
        assert_eq!(
            render("[{{ s.strip('\\n') }}]", "{\"s\":\"\\n hi \"}").unwrap(),
            "[ hi ]"
        );
        // Bare `strip()` still means whitespace.
        assert_eq!(
            render("[{{ s.strip() }}]", "{\"s\":\" \\thi \"}").unwrap(),
            "[hi]"
        );
    }

    #[test]
    fn split_honours_maxsplit() {
        // CPython: 'a,b,c'.split(',', 1) == ['a', 'b,c']
        assert_eq!(
            render("{{ s.split(',',1)|join('|') }}", r#"{"s":"a,b,c"}"#).unwrap(),
            "a|b,c"
        );
        assert_eq!(
            render("{{ s.split(',')|join('|') }}", r#"{"s":"a,b,c"}"#).unwrap(),
            "a|b|c"
        );
        // `str.split(None, n)` is not valid Python.
        assert!(render("{{ s.split()|length }}", r#"{"s":"a b"}"#).is_ok());
        assert!(render("{{ s.split(1) }}", r#"{"s":"a b"}"#).is_err());
    }

    /// Regression: an older closed block masked a newer open one, so the quote
    /// was read as prose and the tag-shaped literal was rewritten.
    #[test]
    fn an_older_closed_block_does_not_mask_a_newer_open_one() {
        assert_eq!(
            render("{% set x = 1 %}{{ '{% generation %}' }}", "{}").unwrap(),
            "{% generation %}"
        );
        assert_eq!(
            render("{{ '{% generation %}' }}{% set x = 1 %}", "{}").unwrap(),
            "{% generation %}"
        );
        // A tag inside markup is still rewritten; only literals are spared.
        assert_eq!(
            render("{% set x = 1 %}{% generation %}y{% endgeneration %}", "{}").unwrap(),
            "y"
        );
    }

    /// Regression: the tag search did not know a string literal is data, so
    /// `{{ '{% generation %}' }}` rendered a comment instead of the text.
    #[test]
    fn a_generation_tag_inside_a_string_literal_survives() {
        assert_eq!(
            render("{{ '{% generation %}' }}", "{}").unwrap(),
            "{% generation %}"
        );
        assert_eq!(
            render(r#"{{ "{%- generation -%}" }}"#, "{}").unwrap(),
            "{%- generation -%}"
        );
    }

    /// Regression: `{% raw %}` is literal text, so a tag inside it must not be
    /// rewritten either.
    #[test]
    fn a_generation_tag_inside_a_raw_block_survives() {
        assert_eq!(
            render("{% raw %}{% generation %}{% endraw %}", "{}").unwrap(),
            "{% generation %}"
        );
        assert_eq!(
            render("{% raw %}{%- endgeneration -%}{% endraw %}", "{}").unwrap(),
            "{%- endgeneration -%}"
        );
    }

    /// The scan must not mistake a quote inside a comment or a tag for the
    /// start of a string literal.
    #[test]
    fn a_quote_inside_a_comment_does_not_hide_a_tag() {
        assert_eq!(
            render("{# it's fine #}{% generation %}x{% endgeneration %}", "{}").unwrap(),
            "x"
        );
    }

    /// Multibyte text must survive the scan; the previous implementation
    /// advanced by bytes and could land inside a character.
    #[test]
    fn multibyte_text_around_generation_tags_survives() {
        assert_eq!(
            render("你好{% generation %}世界{% endgeneration %}尾", "{}").unwrap(),
            "你好世界尾"
        );
    }

    #[test]
    fn tmp_review_repro() {
        let show = |label: &str, tpl: &str, ctx: &str| {
            println!("{label} => {:?}", render(tpl, ctx));
        };
        show(
            "[1a replace 2 args]",
            "{{ s.replace('a','x') }}",
            r#"{"s":"banana"}"#,
        );
        show(
            "[1b strip set]     ",
            "[{{ s.strip(' ') }}]",
            r#"{"s":"  hi  "}"#,
        );
        show(
            "[1b strip newline] ",
            "[{{ s.strip('\n') }}]",
            r#"{"s":"\n hi "}"#,
        );
        show(
            "[1c split maxsplit]",
            "{{ s.split(',',1)|join('|') }}",
            r#"{"s":"a,b,c"}"#,
        );
        show("[2a literal tag]   ", "{{ '{% generation %}' }}", "{}");
        show(
            "[2b raw block]     ",
            "{% raw %}{% generation %}{% endraw %}",
            "{}",
        );
    }

    #[test]
    fn an_unimplemented_method_still_errors() {
        // Silently answering anything would hide real gaps.
        assert!(render("{{ s.nope() }}", r#"{"s":"a"}"#).is_err());
        assert!(render("{{ m.get('k') }}", "{}").is_err());
    }
}
