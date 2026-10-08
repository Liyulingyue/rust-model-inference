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
                // `s.split(p)[-1]` has to give the last element.
                "split" => Value::from_iter(match args.first().and_then(|v| v.as_str()) {
                    Some(pat) => s.split(pat).map(str::to_string).collect::<Vec<_>>(),
                    None => s.split_whitespace().map(str::to_string).collect::<Vec<_>>(),
                }),
                // Common enough in templates, and trivial while we are here.
                "lower" => Value::from(s.to_lowercase()),
                "upper" => Value::from(s.to_uppercase()),
                "strip" => Value::from(s.trim()),
                "replace" => Value::from(s.replace(one_str(args)?, &from_str(args, 1)?)),
                "count" => Value::from(match args.first().and_then(|v| v.as_str()) {
                    Some(pat) => s.matches(pat).count(),
                    None => 0,
                }),
                _ => return Err(Error::from(ErrorKind::UnknownMethod)),
            })
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

/// Optional second string argument, for `replace(old, new)`.
fn from_str(args: &[Value], index: usize) -> Result<String, Error> {
    match args.get(index).and_then(|v| v.as_str()) {
        Some(s) => Ok(s.to_string()),
        None => Err(Error::from(ErrorKind::InvalidOperation)),
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
        let cases: [(&str, &str, &str); 8] = [
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

    #[test]
    fn an_unimplemented_method_still_errors() {
        // Silently answering anything would hide real gaps.
        assert!(render("{{ s.nope() }}", r#"{"s":"a"}"#).is_err());
        assert!(render("{{ m.get('k') }}", "{}").is_err());
    }
}
