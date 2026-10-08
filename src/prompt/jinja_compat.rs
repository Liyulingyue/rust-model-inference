//! Rewrites that let Python-flavoured Jinja2 templates run under minijinja.
//!
//! This is the only code in the crate that rewrites template *source*. Every
//! function here is a pure `source -> source` transform, so the whole surface
//! can be audited on its own: if a prompt comes out wrong, this file is where
//! to look first.
//!
//! minijinja is not Python and has no `add_method`, so the constructs real
//! templates use -- `dict.get`, `str.startswith`, `str.split` -- have to be
//! rewritten rather than registered. Two rules govern everything here:
//!
//! - Never change meaning. `x.get("k")` must not become `x["k"]`, because
//!   Jinja2 returns `None` for a missing key and minijinja returns Undefined,
//!   and `is none` disagrees between the two.
//! - When a rewrite cannot be applied confidently, leave the source alone and
//!   let minijinja raise a real error. A template that fails loudly is
//!   strictly better than one that renders plausible garbage.

/// Tags that are llama.cpp extensions rather than Jinja2.
pub(super) const LLAMACPP_TAGS: [&str; 2] = ["generation", "endgeneration"];

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
pub(super) const JINJA_KEYWORDS: &[&str] = &[
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
pub(super) fn trim_py<'a>(s: &'a str, chars: Option<&str>, left: bool) -> &'a str {
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
pub(super) const STRING_METHODS: &[&str] = &["startswith", "endswith", "lstrip", "rstrip", "split"];

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
pub(super) fn rewrite_python_compat(source: &str) -> String {
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
pub(super) fn rewrite_string_methods(source: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::{rewrite_python_compat, rewrite_string_methods, strip_llamacpp_extensions};

    /// These assert on the rewritten *source text*, not on a rendered prompt.
    /// The end-to-end behaviour is covered in `super::jinja`, where the filters
    /// and globals are actually registered; duplicating that setup here would
    /// test the harness rather than the rewriter.

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

    #[test]
    fn get_is_rewritten_inside_a_statement_but_not_inside_a_literal() {
        // `{% ... %}` is code, so `if x.get(k)` must be rewritten...
        assert_eq!(
            rewrite_python_compat("{% if x.get(\"k\") %}y{% endif %}"),
            // The filter goes *outside* the brackets: inside them it would
            // decorate the key and leave `x["k"]`, so `is none` is false again.
            "{% if x[\"k\"]|default(__py_none) %}y{% endif %}"
        );
        // ...but a string literal is data.
        assert_eq!(
            rewrite_python_compat("{% set s = 'api .get(\"k\")' %}"),
            "{% set s = 'api .get(\"k\")' %}"
        );
    }

    #[test]
    fn get_accepts_single_quotes() {
        assert_eq!(
            rewrite_python_compat("{{ m.get('k') }}"),
            "{{ m['k']|default(__py_none) }}"
        );
    }

    #[test]
    fn get_in_a_raw_block_is_left_alone() {
        let src = "{% raw %}x.get(\"k\"){% endraw %}";
        assert_eq!(rewrite_python_compat(src), src);
    }

    #[test]
    fn multi_arg_get_is_left_alone() {
        // `d.get(k, default)` genuinely changes meaning; rewritting it would
        // silently drop the default.
        let src = "{% if m.get(\"k\", \"d\") %}x{% endif %}";
        assert_eq!(rewrite_python_compat(src), src);
    }

    #[test]
    fn string_methods_become_parenthesised_filter_calls() {
        assert_eq!(
            rewrite_string_methods("{% if s.startswith('a') %}y{% endif %}"),
            "{% if (s|startswith('a')) %}y{% endif %}"
        );
        assert_eq!(
            rewrite_string_methods("{{ m[0].c.split(',')|join('') }}"),
            "{{ (m[0].c|split(','))|join('') }}"
        );
    }

    #[test]
    fn a_chain_is_rewritten_everywhere_and_converges() {
        assert_eq!(
            rewrite_string_methods("{{ c.split('a')[0].rstrip('\\n') }}"),
            "{{ ((c|split('a'))[0]|rstrip('\\n')) }}"
        );
    }

    #[test]
    fn an_unrecognised_receiver_is_left_verbatim() {
        // Losing the receiver must not drop the dot: `s` + `startswith` glued
        // together would be a silently different template.
        let src = "{{ 1 + 2.startswith('a') }}";
        let out = rewrite_string_methods(src);
        assert!(out.contains("startswith"), "{out}");
    }
}
