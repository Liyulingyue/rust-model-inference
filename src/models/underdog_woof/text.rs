//! English text frontend matching the pinned official Underdog Woof rules.
//! Source and license provenance: tools/oracle/underdog_woof/README.md.

use fancy_regex::{Captures, Regex};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use unicode_categories::UnicodeCategories;
use unicode_normalization::{char::canonical_combining_class, UnicodeNormalization};

include!("text_data.rs");

pub const CHARS: &str =
    " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"-()&/%$€£@#+=*_<>[]…—";

pub fn encode_text(text: &str) -> Vec<u32> {
    std::iter::once(1)
        .chain(text.chars().map(|c| {
            CHARS
                .chars()
                .position(|x| x == c)
                .map_or(2, |i| i as u32 + 3)
        }))
        .collect()
}

fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
fn trim(s: &str) -> &str {
    s.trim_matches(py_space)
}
fn py_word(c: char) -> bool {
    c.is_letter() || c.is_number() || c == '_'
}
fn alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_letter())
}
fn upper(s: &str) -> bool {
    s.chars().any(char::is_uppercase) && !s.chars().any(char::is_lowercase)
}
fn lower(s: &str) -> bool {
    s.chars().any(char::is_lowercase) && !s.chars().any(char::is_uppercase)
}
fn first_upper(s: &str) -> bool {
    s.chars().next().is_some_and(char::is_uppercase)
}
fn capitalize(s: &str) -> String {
    let mut cs = s.chars();
    match cs.next() {
        Some(c) => c.to_uppercase().collect::<String>() + &cs.as_str().to_lowercase(),
        None => String::new(),
    }
}
fn capitalize_first(s: &str) -> String {
    let mut cs = s.chars();
    match cs.next() {
        Some(c) => c.to_uppercase().collect::<String>() + cs.as_str(),
        None => String::new(),
    }
}
fn table<'a>(items: &[(&str, &'a str)], key: &str) -> &'a str {
    items
        .iter()
        .find_map(|(k, v)| (*k == key).then_some(*v))
        .unwrap_or("")
}
fn escaped(s: &str) -> String {
    regex::escape(s)
}
fn keys(items: &[(&str, &str)], sorted: bool) -> String {
    let mut v: Vec<_> = items.iter().map(|(k, _)| *k).collect();
    if sorted {
        v.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
    }
    v.into_iter().map(escaped).collect::<Vec<_>>().join("|")
}

// Python's \w omits combining marks and \s also includes the four information separators.
fn python_pattern(pattern: &str) -> String {
    let pattern = pattern.replace(r"[^\W\d_]", r"[\pL\p{Nl}\p{No}]");
    let mut out = String::new();
    let mut chars = pattern.chars().peekable();
    let mut class = false;
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('w') => out.push_str(if class { r"\pL\pN_" } else { r"[\pL\pN_]" }),
                Some('W') => out.push_str(r"[^\pL\pN_]"),
                Some('s') => out.push_str(if class {
                    r"\s\x1c-\x1f"
                } else {
                    r"[\s\x1c-\x1f]"
                }),
                Some('S') => out.push_str(r"[^\s\x1c-\x1f]"),
                Some('b') if !class => {
                    out.push_str(r"(?:(?<=[\pL\pN_])(?![\pL\pN_])|(?<![\pL\pN_])(?=[\pL\pN_]))")
                }
                Some(c) => {
                    out.push('\\');
                    out.push(c);
                }
                None => out.push('\\'),
            }
        } else {
            if c == '[' {
                class = true;
            } else if c == ']' {
                class = false;
            }
            out.push(c);
        }
    }
    out
}
fn rx(pattern: &str) -> Result<Arc<Regex>, String> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<Regex>>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "Woof regex cache lock poisoned")?;
    if let Some(r) = cache.get(pattern) {
        return Ok(r.clone());
    }
    let r = Arc::new(
        Regex::new(&python_pattern(pattern))
            .map_err(|e| format!("Woof text rule {pattern:?}: {e}"))?,
    );
    cache.insert(pattern.to_owned(), r.clone());
    Ok(r)
}
fn test(pattern: &str, s: &str) -> Result<bool, String> {
    rx(pattern)?
        .is_match(s)
        .map_err(|e| format!("Woof text matching: {e}"))
}
fn full(pattern: &str, s: &str) -> Result<bool, String> {
    test(&format!(r"\A(?:{pattern})\z"), s)
}
fn g<'a>(m: &Captures<'a>, i: usize) -> &'a str {
    m.get(i).map_or("", |x| x.as_str())
}
fn before<'a>(m: &Captures<'_>, s: &'a str) -> &'a str {
    &s[..m.get(0).unwrap().start()]
}
fn after<'a>(m: &Captures<'_>, s: &'a str) -> &'a str {
    &s[m.get(0).unwrap().end()..]
}
fn expand(m: &Captures<'_>, replacement: &str) -> String {
    let mut out = String::new();
    let mut cs = replacement.chars().peekable();
    while let Some(c) = cs.next() {
        if c == '\\' {
            if let Some(d) = cs.peek().and_then(|c| c.to_digit(10)) {
                cs.next();
                out.push_str(g(m, d as usize));
            } else if cs.peek() == Some(&'\\') {
                cs.next();
                out.push('\\');
            } else {
                out.push(c);
            }
        } else {
            out.push(c);
        }
    }
    out
}
fn replace(
    s: &mut String,
    pattern: &str,
    rep: impl Fn(&Captures<'_>, &str) -> Result<String, String>,
) -> Result<(), String> {
    let mut out = String::with_capacity(s.len());
    let mut pos = 0;
    for m in rx(pattern)?.captures_iter(s) {
        let m = m.map_err(|e| format!("Woof text matching: {e}"))?;
        let span = m.get(0).unwrap();
        out.push_str(&s[pos..span.start()]);
        out.push_str(&rep(&m, s)?);
        pos = span.end();
    }
    out.push_str(&s[pos..]);
    *s = out;
    Ok(())
}
macro_rules! rule {
    ($t:ident, $pat:expr, |$m:ident, $s:ident| $body:expr) => {
        replace(&mut $t, $pat, |$m, $s| Ok($body))?;
    };
    ($t:ident, $pat:expr, $rep:expr) => {
        replace(&mut $t, $pat, |m, _| Ok(expand(m, $rep)))?;
    };
}
fn decimal(s: &str) -> String {
    s.chars()
        .map(|c| {
            let n = c as u32;
            DECIMAL_ZEROES
                .iter()
                .find_map(|z| (n >= *z && n < *z + 10).then(|| char::from_u32(48 + n - z).unwrap()))
                .unwrap_or(c)
        })
        .collect()
}
fn uint(s: &str) -> u64 {
    decimal(s).parse().unwrap_or(0)
}
fn canonical(s: &str) -> String {
    let s = decimal(s);
    let s = s.trim_start_matches('0');
    if s.is_empty() {
        "0".into()
    } else {
        s.into()
    }
}
fn cardinal(n: u64) -> String {
    if n < 20 {
        return ONES[n as usize].into();
    }
    if n < 100 {
        return TENS[n as usize / 10].to_owned()
            + &if n % 10 == 0 {
                String::new()
            } else {
                format!(" {}", ONES[n as usize % 10])
            };
    }
    if n < 1000 {
        return format!(
            "{} hundred{}",
            ONES[n as usize / 100],
            if n % 100 == 0 {
                String::new()
            } else {
                format!(" and {}", cardinal(n % 100))
            }
        );
    }
    for (v, name) in [
        (1_000_000_000_000, "trillion"),
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1000, "thousand"),
    ] {
        if n >= v {
            let rest = n % v;
            return format!(
                "{} {name}{}",
                cardinal(n / v),
                if rest == 0 {
                    String::new()
                } else {
                    format!(
                        "{}{}",
                        if rest < 100 { " and " } else { " " },
                        cardinal(rest)
                    )
                }
            );
        }
    }
    n.to_string()
}
fn cardinal_str(s: &str) -> String {
    let s = canonical(s);
    if let Ok(n) = s.parse::<u64>() {
        return cardinal(n);
    }
    let cut = s.len() - 12;
    let rest = uint(&s[cut..]);
    format!(
        "{} trillion{}",
        cardinal_str(&s[..cut]),
        if rest == 0 {
            String::new()
        } else {
            format!(
                "{}{}",
                if rest < 100 { " and " } else { " " },
                cardinal(rest)
            )
        }
    )
}
fn ordinal_str(s: &str) -> String {
    let mut words = cardinal_str(s);
    let i = words.rfind(' ').map_or(0, |i| i + 1);
    let last = &words[i..];
    let mapped = table(ORD, last);
    let last = if !mapped.is_empty() {
        mapped.into()
    } else if let Some(stem) = last.strip_suffix('y') {
        format!("{stem}ieth")
    } else {
        format!("{last}th")
    };
    words.truncate(i);
    words.push_str(&last);
    words
}
fn ordinal(n: u64) -> String {
    ordinal_str(&n.to_string())
}
fn year(n: u64) -> String {
    if (2000..=2009).contains(&n) {
        return cardinal(n);
    }
    let (hi, lo) = (n / 100, n % 100);
    cardinal(hi)
        + &if lo == 0 {
            " hundred".into()
        } else if lo < 10 {
            format!(" oh {}", ONES[lo as usize])
        } else {
            format!(" {}", cardinal(lo))
        }
}
fn number(s: &str) -> String {
    let s = decimal(&s.replace(',', ""));
    if let Some((a, b)) = s.split_once('.') {
        format!(
            "{} point {}",
            cardinal_str(if a.is_empty() { "0" } else { a }),
            digits(b)
        )
    } else {
        cardinal_str(&s)
    }
}
fn digits(s: &str) -> String {
    decimal(s)
        .chars()
        .filter_map(|c| c.to_digit(10))
        .map(|n| ONES[n as usize])
        .collect::<Vec<_>>()
        .join(" ")
}
fn spell(s: &str) -> String {
    decimal(s)
        .chars()
        .filter_map(|c| {
            c.to_digit(10)
                .map(|n| ONES[n as usize].to_owned())
                .or_else(|| c.is_letter().then(|| c.to_uppercase().collect()))
        })
        .collect::<Vec<_>>()
        .join(" ")
}
fn pronounceable(s: &str) -> Result<bool, String> {
    let s = s.to_lowercase();
    if s.chars().count() < 4 || !test("[aeiouy]", &s)? {
        return Ok(false);
    }
    if test("[^aeiouy]{3,}", &s)? && !test("(str|sch|thr|chr|spr|scr|ght|nds|nts)", &s)? {
        return Ok(false);
    }
    full("(?:[^aeiouy]{0,3}[aeiouy]{1,2})+[^aeiouy]{0,3}", &s)
}
fn nread(s: &str) -> String {
    let n = uint(s);
    let len = s.chars().count();
    if (s.starts_with('0') && len > 1) || len > 4 {
        return digits(s);
    }
    if n < 100 || (n % 100 == 0 && len == 3) || n % 1000 == 0 {
        return cardinal(n);
    }
    cardinal(n / 100)
        + &if n % 100 == 0 {
            " hundred".into()
        } else if n % 100 < 10 {
            format!(" oh {}", ONES[n as usize % 100])
        } else {
            format!(" {}", cardinal(n % 100))
        }
}
fn code_read(s: &str) -> Result<String, String> {
    let mut out = Vec::new();
    for p in s.split(['-', '/']).filter(|s| !s.is_empty()) {
        out.push(if full(r"\d+", p)? {
            if p.chars().count() <= 2 && !p.starts_with('0') {
                p.into()
            } else {
                digits(p)
            }
        } else if alpha(p) && lower(p) && p.chars().count() > 3 && pronounceable(p)? {
            p.into()
        } else if alpha(p) && pronounceable(p)? {
            capitalize(p)
        } else {
            spell(p)
        });
    }
    Ok(out.join(" "))
}
fn split_ident(s: &str) -> Result<String, String> {
    let mut t = s.replace('_', " ");
    rule!(t, r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ");
    Ok(t.split(py_space)
        .filter(|s| !s.is_empty())
        .map(|s| {
            if upper(s) && s.chars().count() <= 4 {
                spell(s)
            } else {
                s.into()
            }
        })
        .collect::<Vec<_>>()
        .join(" "))
}
fn handle(s: &str) -> Result<String, String> {
    let mut out = Vec::new();
    for p in s.split('_') {
        let r = rx(r"\A([a-z]{3,}[nlyex])(ai|hq)\z")?;
        let text = if let Some(m) = r.captures(p).map_err(|e| e.to_string())? {
            format!("{} {}", g(&m, 1), spell(g(&m, 2)))
        } else {
            split_ident(p)?
        };
        if !text.is_empty() {
            out.push(text);
        }
    }
    Ok(out.join(" underscore "))
}
fn roman(s: &str) -> u64 {
    fn v(c: char) -> u64 {
        match c {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => 0,
        }
    }
    let cs: Vec<_> = s.chars().collect();
    let mut total = 0i64;
    for (i, c) in cs.iter().enumerate() {
        let n = v(*c) as i64;
        total += if cs.get(i + 1).is_some_and(|c| v(*c) > n as u64) {
            -n
        } else {
            n
        };
    }
    total as u64
}
fn frac_words(a: u64, b: u64, lead: bool) -> String {
    let name = if b == 2 {
        if a == 1 {
            "half".into()
        } else {
            "halves".into()
        }
    } else {
        (if b == 4 { "quarter".into() } else { ordinal(b) }) + if a == 1 { "" } else { "s" }
    };
    format!(
        "{} {name}",
        if lead && a == 1 {
            "a".into()
        } else {
            cardinal(a)
        }
    )
}
fn frac_den(b: u64) -> bool {
    [2, 3, 4, 5, 6, 8, 10, 16, 32, 64, 100].contains(&b)
}
fn sing(s: &str) -> String {
    let (head, rest) = s.split_once(" per ").unwrap_or((s, ""));
    if head.contains(" of ") {
        return s.into();
    }
    let (lead, last) = head
        .rsplit_once(' ')
        .map_or(("".into(), head), |(a, b)| (format!("{a} "), b));
    let mapped = table(SING, last);
    lead + if mapped.is_empty() {
        last.strip_suffix('s').unwrap_or(last)
    } else {
        mapped
    } + if rest.is_empty() { "" } else { " per " }
        + rest
}
fn unit(k: &str, n: &str) -> String {
    let s = table(UNITS, k);
    if n == "1" {
        sing(s)
    } else {
        s.into()
    }
}
fn plural(n: &str, name: &str, numeric: bool) -> String {
    let c = if numeric { canonical(n) } else { n.into() };
    format!("{c} {name}{}", if c == "1" { "" } else { "s" })
}
fn tail_chars(s: &str, count: usize) -> &str {
    let start = s
        .char_indices()
        .rev()
        .nth(count.saturating_sub(1))
        .map_or(0, |(i, _)| i);
    &s[start..]
}
fn head_chars(s: &str, count: usize) -> &str {
    &s[..s.char_indices().nth(count).map_or(s.len(), |(i, _)| i)]
}

#[derive(Debug)]
pub struct Frontend {
    lexicon: HashMap<String, (String, bool)>,
    max_words: usize,
}
impl Frontend {
    pub fn new(lexicon: &str) -> Result<Self, String> {
        let mut out = Self {
            lexicon: HashMap::new(),
            max_words: 1,
        };
        for (i, line) in lexicon.split('\n').enumerate() {
            if trim(line).is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() < 2 {
                return Err(format!(
                    "Woof respelling lexicon row {} needs written and spoken forms",
                    i + 1
                ));
            }
            let key = fold(fields[0]);
            out.max_words = out.max_words.max(key.split(' ').count());
            out.lexicon.insert(
                key,
                (
                    fields[1].into(),
                    fields.get(2).is_some_and(|f| f.contains('c')),
                ),
            );
        }
        Ok(out)
    }
    pub fn normalize(&self, text: &str) -> Result<String, String> {
        let mut t = norm_pre(text)?;
        if test(r"[0-9#%$€£@&/_+=<>±≥≤×°]", &t)? {
            t = fallback(t)?;
        }
        rule!(t, r"\s+([.,!?;:])", r"\1");
        rule!(t, r"\s{2,}", " ");
        t = self.respell(trim(&t))?;
        Ok(if first_upper(text) {
            capitalize_first(&t)
        } else {
            t
        })
    }
    fn respell(&self, text: &str) -> Result<String, String> {
        if self.lexicon.is_empty() {
            return Ok(text.into());
        }
        let text = text.replace('’', "'");
        let r = rx(r"[^\W\d_]+(?:'[^\W\d_]+)*")?;
        let words: Vec<_> = r
            .find_iter(&text)
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        let (mut out, mut pos, mut i) = (String::new(), 0, 0);
        while i < words.len() {
            let mut found = false;
            for n in (1..=self.max_words.min(words.len() - i)).rev() {
                let span = &words[i..i + n];
                if span
                    .windows(2)
                    .any(|w| ![" ", "-"].contains(&&text[w[0].end()..w[1].start()]))
                {
                    continue;
                }
                let (a, b) = (span[0].start(), span[n - 1].end());
                let src = &text[a..b];
                let left: Vec<_> = text[..a].chars().rev().take(2).collect();
                let right: Vec<_> = text[b..].chars().take(2).collect();
                let left_joined =
                    left.first() == Some(&'-') && left.get(1).is_some_and(|c| c.is_letter());
                let right_joined =
                    right.first() == Some(&'-') && right.get(1).is_some_and(|c| c.is_letter());
                if left_joined {
                    let before_dash = &text[..a - 1];
                    let other = before_dash
                        .split(py_space)
                        .filter(|s| !s.is_empty())
                        .next_back()
                        .unwrap_or("");
                    if !(first_upper(src) && other.chars().next().is_some_and(char::is_lowercase)) {
                        continue;
                    }
                }
                if right_joined {
                    let other = text[b + 1..]
                        .split(py_space)
                        .find(|s| !s.is_empty())
                        .unwrap_or("");
                    if !(first_upper(src) && other.chars().next().is_some_and(char::is_lowercase)) {
                        continue;
                    }
                }
                let key = fold(src);
                let (hit, tail) = if let Some(hit) = self.lexicon.get(&key) {
                    (Some(hit), "")
                } else if let Some(base) = key.strip_suffix("'s") {
                    (self.lexicon.get(base), tail_chars(src, 2))
                } else {
                    (None, "")
                };
                let Some((rep, caps)) = hit else {
                    continue;
                };
                if *caps && !span.iter().all(|w| first_upper(w.as_str())) {
                    continue;
                }
                out.push_str(&text[pos..a]);
                if first_upper(src) && rep.chars().next().is_some_and(char::is_lowercase) {
                    out.push_str(&capitalize_first(rep));
                } else {
                    out.push_str(rep);
                }
                out.push_str(tail);
                pos = b;
                i += n;
                found = true;
                break;
            }
            if !found {
                i += 1;
            }
        }
        out.push_str(&text[pos..]);
        Ok(out)
    }
}
fn fold(s: &str) -> String {
    let s: String = s
        .replace(['’', '‘'], "'")
        .nfkd()
        .filter(|c| canonical_combining_class(*c) == 0)
        .collect::<String>()
        .to_lowercase();
    let s = s
        .replace('ł', "l")
        .replace('ø', "o")
        .replace('æ', "ae")
        .replace('œ', "oe")
        .replace('ß', "ss")
        .replace('đ', "d")
        .replace(['-', '‐', '–'], " ");
    s.split(py_space)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn norm_tokens(text: &str) -> Result<String, String> {
    let mut t = text
        .replace('−', "-")
        .replace(['’', '‘'], "'")
        .replace(['“', '”'], "\"");
    rule!(t, r"(?<=\d)([½¼¾⅓⅔⅛])", r" \1");
    rule!(t, r"[½¼¾⅓⅔⅛]", |m, _s| table(
        &[
            ("½", "1/2"),
            ("¼", "1/4"),
            ("¾", "3/4"),
            ("⅓", "1/3"),
            ("⅔", "2/3"),
            ("⅛", "1/8")
        ],
        g(m, 0)
    )
    .into());
    rule!(t, r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+", |m, _s| format!(
        "^{}",
        g(m, 0)
            .chars()
            .map(|c| "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"
                .chars()
                .position(|x| x == c)
                .map(|i| "0123456789-".chars().nth(i).unwrap())
                .unwrap())
            .collect::<String>()
    ));
    rule!(t, r"(?<![\w.])([A-Za-z]*)\.NET\b", |m, _s| format!(
        "{}dot net",
        if g(m, 1).is_empty() {
            String::new()
        } else {
            format!("{} ", spell(g(m, 1)))
        }
    ));
    rule!(t, r"\b([A-Za-z])#(?!\w)", r"\1 sharp");
    rule!(t, r"\b([A-Za-z])\+\+(?!\w)", r"\1 plus plus");
    rule!(t, r"\b([A-Za-z_][A-Za-z0-9_]*)\(\)", |m, _s| split_ident(
        g(m, 1)
    )?);
    rule!(t, r"\b[a-z]{2,}(?:[A-Z][a-z]+){2,}\b", |m, _s| split_ident(
        g(m, 0)
    )?);
    rule!(t, r"\bv(\d+(?:\.\d+)+)\b", |m, _s| format!(
        "version {}",
        g(m, 1).split('.').collect::<Vec<_>>().join(" point ")
    ));
    rule!(
        t,
        r"(?<![\w&#])#([A-Za-z][A-Za-z_]*)\b(?![\d-])",
        |m, _s| format!("hashtag {}", split_ident(g(m, 1))?)
    );
    rule!(
        t,
        r"(?<![\w.@])@([A-Za-z_][A-Za-z0-9_]{1,30})\b",
        |m, _s| format!("at {}", handle(g(m, 1))?)
    );
    rule!(
        t,
        r"\b([A-Z]{1,2}\d[A-Z\d]?|[A-Z]\d[A-Z])\s(\d[A-Z]{2}|\d[A-Z]\d)\b",
        |m, _s| format!("{}, {}", spell(g(m, 1)), spell(g(m, 2)))
    );
    rule!(
        t,
        r"\b((?:ending|ends) in|last (?:four|4)(?: digits)?(?: of)?|PIN|[Pp]in|passcode|OTP|[Cc]ode)(\s+)(\d{3,8}(?:\s\d{2,4})*)\b(?![.,]\d|-\w)",
        |m, _s| format!(
            "{}{}{}",
            g(m, 1),
            g(m, 2),
            g(m, 3)
                .split(py_space)
                .filter(|s| !s.is_empty())
                .map(digits)
                .collect::<Vec<_>>()
                .join(", ")
        )
    );
    rule!(t, r"\b([A-Z]):\\", r"\1 colon backslash ");
    rule!(t, r"\\", " backslash ");
    rule!(t, r"(?<=\s)\*(?=\s)(?!\s*\d)", "star");
    rule!(
        t,
        r"(?<![\w.])\.(env|gitignore|bashrc|zshrc|npmrc|htaccess|dockerignore)\b",
        r"dot \1"
    );
    rule!(
        t,
        r"\b(?=\w*_)(?![\w]*\.\w)[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b",
        |m, _s| {
            let mut parts = Vec::new();
            for p in g(m, 0).split('_') {
                parts.push(if test(r"\d", p)? && test("[A-Za-z]", p)? {
                    spell(p)
                } else {
                    p.into()
                });
            }
            parts.join(" underscore ")
        }
    );
    rule!(
        t,
        r"(?<![\w+])1[-. ](\d{3})[-. ](?:(\d{3})[-. ](\d{4})|([A-Z]{3,}(?:-[A-Z]+)*))\b",
        |m, _s| format!(
            "one {}, {}",
            if ["800", "900"].contains(&g(m, 1)) {
                cardinal(uint(g(m, 1)))
            } else {
                digits(g(m, 1))
            },
            if !g(m, 2).is_empty() {
                format!("{}, {}", digits(g(m, 2)), digits(g(m, 3)))
            } else {
                g(m, 4)
                    .split('-')
                    .map(capitalize)
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        )
    );
    let num = r"(\d[\d,]*(?:\.\d+)?)";
    let sc = r"(\s?(?:thousand|million|billion|trillion)\b|(?:k|K|mn|m|M|bn|b|B|tn|T)\b)?";
    rule!(
        t,
        &format!(r"(?<![\w$])({})\s?{num}{sc}", keys(CURCODE, true)),
        |m, _s| currency(g(m, 1), g(m, 2), g(m, 3))
    );
    rule!(
        t,
        &format!(
            r"{num}{sc}\s?(USD|EUR|GBP|JPY|CNY|INR|CAD|AUD|CHF|NZD|HKD|SGD|KRW|MXN|BRL|RUB|SEK|NOK|DKK|ZAR|BTC|ETH)\b"
        ),
        |m, _s| currency(g(m, 3), g(m, 1), g(m, 2))
    );
    rule!(t, r"(?<!\w)(\d+)\s?¢", r"\1 cents");
    rule!(
        t,
        &format!(r"([$€£]){num}{sc}\s?[-–]\s?[$€£]?{num}{sc}"),
        |m, _s| {
            let (c, a, s1, b, s2) = (g(m, 1), g(m, 2), g(m, 3), g(m, 4), g(m, 5));
            if s1.is_empty() && s2.is_empty() && test(r"\.\d\d\b", &format!("{a} {b}"))? {
                format!("{c}{a} to {c}{b}")
            } else {
                let s1 = if s1.is_empty() {
                    String::new()
                } else {
                    format!(" {}", table(SCALE, trim(s1)))
                };
                let s2 = if s2.is_empty() {
                    String::new()
                } else {
                    format!(" {}", table(SCALE, trim(s2)))
                };
                format!(
                    "{a}{} to {b}{s2} {}",
                    if s1 == s2 { "" } else { &s1 },
                    table(&[("$", "dollars"), ("€", "euros"), ("£", "pounds")], c)
                )
            }
        }
    );
    rule!(
        t,
        &format!(
            r"([$€£]){num}(\s?(?:thousand|million|billion|trillion)\b|(?:k|K|mn|m|M|bn|b|B|tn|T)\b)"
        ),
        |m, _s| format!(
            "{} {} {}",
            g(m, 2),
            table(SCALE, trim(g(m, 3))),
            table(
                &[("$", "dollars"), ("€", "euros"), ("£", "pounds")],
                g(m, 1)
            )
        )
    );
    rule!(
        t,
        r"\b(\d+)\s?h(?:rs?)?\s?(\d{1,2})(?:\s?m(?:ins?)?)?\b",
        |m, _s| format!(
            "{} {}",
            plural(g(m, 1), "hour", false),
            plural(g(m, 2), "minute", false)
        )
    );
    rule!(
        t,
        r"\b(\d+)\s?m(?:ins?)?\s?(\d{1,2})\s?s(?:ecs?)?\b",
        |m, _s| format!(
            "{} {}",
            plural(g(m, 1), "minute", false),
            plural(g(m, 2), "second", false)
        )
    );
    rule!(
        t,
        r"(\b(?:for|last|lasts|lasted|lasting|takes?|took|taking|every|within|wait|waited|remaining|runtime|ETA|blocked)\s+(?:\w+\s+){0,2}?)(\d{1,3})m\b(?!\s*(?:dash|sprint|race|swim|hurdles|pool|freestyle|relay))",
        |m, _s| format!("{}{}", g(m, 1), plural(g(m, 2), "minute", false))
    );
    rule!(
        t,
        r"(?<![\w$€£.,'])(')?(\d{2}|1\d{3}|20\d{2})'?s\b",
        |m, _s| {
            let s = g(m, 2);
            let n = uint(s);
            if n % 10 != 0 {
                g(m, 0).into()
            } else {
                let w = if s.chars().count() == 4 {
                    year(n)
                } else if n == 0 && !g(m, 1).is_empty() {
                    "two thousand".into()
                } else if n != 0 {
                    cardinal(n)
                } else {
                    String::new()
                };
                if w.is_empty() {
                    g(m, 0).into()
                } else if let Some(stem) = w.strip_suffix('y') {
                    format!("{stem}ies")
                } else {
                    w + "s"
                }
            }
        }
    );
    rule!(
        t,
        r"(?:(?<=\s)|^)'(\d{2})\b(?!['\w])",
        |m, _s| if g(m, 1).starts_with('0') {
            format!("oh {}", ONES[uint(g(m, 1)) as usize])
        } else {
            g(m, 1).into()
        }
    );
    rule!(t, r"\bWW\s?(II|I|2|1)\b", |m, _s| format!(
        "World War {}",
        if ["II", "2"].contains(&g(m, 1)) {
            "Two"
        } else {
            "One"
        }
    ));
    rule!(
        t,
        &format!(r"\b([A-Za-z][A-Za-z']*\.?)(\s+)({ROMAN_RE})\b(?!'\w|-\d)"),
        |m, s| roman_sub(m, s)?
    );
    rule!(t, r"(?<=[a-z])\((I|II|III|IV|V|VI)\)", |m, _s| format!(
        " {}",
        cardinal(roman(g(m, 1)))
    ));
    rule!(t, r"(\d)\s?[x×*]\s?(?=10\s?\^)", r"\1 times ");
    rule!(
        t,
        &format!(r"{UNIT_PRE}\s?(km|m|cm|mm|ft|mi|yd|in)\^([23])\b"),
        |m, _s| format!(
            "{} {} {}",
            g(m, 1),
            if g(m, 3) == "2" { "square" } else { "cubic" },
            table(UNITS, g(m, 2))
        )
    );
    rule!(
        t,
        r"(\b[A-Za-z]{1,3}|\b\d+(?:\.\d+)?|\))\s?\^\s?\(?([+-]?(?:\d+|[a-z]))\)?",
        |m, _s| {
            let (b, e) = (g(m, 1), g(m, 2));
            let neg = e.starts_with('-');
            let e = e.trim_start_matches(['+', '-']);
            let b = if alpha(b) && b.chars().count() > 1 {
                b.chars()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                b.into()
            };
            if !full(r"\d+", e)? {
                format!("{b} to the {e}")
            } else if canonical(e) == "2" && !neg {
                format!("{b} squared")
            } else if canonical(e) == "3" && !neg {
                format!("{b} cubed")
            } else {
                format!(
                    "{b} to the {}{}",
                    if neg { "minus " } else { "" },
                    ordinal_str(e)
                )
            }
        }
    );
    if test(r"[=^+]|\bsolve\b|\bequation", &t)? {
        rule!(t, r"(?<=[\w)])\s+[-–]\s+(?=[\w(])", " minus ");
        rule!(t, r"\b(\d+)([a-fi-jn-rt-z])\b", r"\1 \2");
    }
    rule!(
        t,
        r"\b(\d+(?:\.\d+)?)x(?=\s+(?:faster|slower|more|less|larger|smaller|bigger|higher|lower|cheaper|better|worse|longer|shorter|the|as|over|speed|zoom)\b)",
        r"\1 times"
    );
    rule!(t, r"(\d)\s+[x×*]\s+(\d+(?:\.\d+)?)(?=\s*=)", r"\1 times \2");
    rule!(
        t,
        r"(?<![\w$€£.])(\d+\.\d+|\d+(?=\s?\w+\s+[a-z]{3,}s\b))\s?(M|bn|mn|tn)\b(?=\s+[a-z]{3,})",
        |m, _s| format!("{} {}", g(m, 1), table(SCALE, g(m, 2)))
    );
    Ok(t)
}
fn currency(code: &str, num: &str, scale: &str) -> String {
    let name = table(CURCODE, code);
    if name.chars().count() == 1 {
        return format!("{name}{num}{scale}");
    }
    let (whole, cents) = num.split_once('.').unwrap_or((num, ""));
    if scale.is_empty() && cents.chars().count() == 2 && name.ends_with("dollars") {
        format!(
            "{whole} {name}{}",
            if uint(cents) == 0 {
                String::new()
            } else {
                format!(" and {} cents", uint(cents))
            }
        )
    } else {
        format!(
            "{num}{} {name}",
            if scale.is_empty() {
                String::new()
            } else {
                format!(" {}", table(SCALE, trim(scale)))
            }
        )
    }
}
fn roman_sub(m: &Captures<'_>, s: &str) -> Result<String, String> {
    let (prev, num) = (g(m, 1), g(m, 3));
    if num.is_empty() {
        return Ok(g(m, 0).into());
    }
    let b = before(m, s);
    let prev2 = b
        .split(py_space)
        .filter(|s| !s.is_empty())
        .next_back()
        .unwrap_or("");
    let prev2 = if test(r#"[.,;:!?)\"]$"#, prev2)? {
        ""
    } else {
        prev2
    };
    let key = prev.to_lowercase().trim_end_matches('.').to_owned();
    let n = roman(num);
    if ["through", "thru", "to", "and", "or"].contains(&key.as_str())
        && !test(&format!(r"(?i)\b(?:{})\s+\S+\s+$", ROMAN_CARD.join("|")), b)?
    {
        return Ok(g(m, 0).into());
    }
    if ROMAN_CARD.contains(&key.as_str())
        && (num.len() > 1
            || num == "I" && first_upper(prev)
            || ["V", "X"].contains(&num) && !ROMAN_LABEL.contains(&key.as_str()))
    {
        return Ok(format!("{prev}{}{}", g(m, 2), cardinal(n)));
    }
    if !full("[IVXL]+", num)? {
        return Ok(g(m, 0).into());
    }
    let title = full("[A-Z][a-z]+", prev)?;
    if REGNAL.contains(&prev) || TITLES.contains(&prev2) && title && num != "I" {
        return Ok(format!("{prev}{}the {}", g(m, 2), ordinal(n)));
    }
    if num.len() > 1 && title && !full("[IVXL]*L[IVXL]*", num)? {
        let full_name =
            full("[A-Z][a-z]+", prev2)? && !ROMAN_STOP.contains(&prev2.to_lowercase().as_str());
        return Ok(format!(
            "{prev}{}{}",
            g(m, 2),
            if full_name {
                format!("the {}", ordinal(n))
            } else {
                cardinal(n)
            }
        ));
    }
    Ok(g(m, 0).into())
}
fn norm_pre(text: &str) -> Result<String, String> {
    let mut t = norm_tokens(text)?;
    rule!(
        t,
        r"\b(localhost|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}):(\d{2,5})\b",
        |m, _s| format!("{} colon {}", g(m, 1), nread(g(m, 2)))
    );
    rule!(
        t,
        r#"(?<![\w.+-])([A-Za-z][A-Za-z0-9+.-]{1,9})://(?=[\s,.;:!?)\"']|$)"#,
        |m, _s| format!("{} colon slash slash", scheme(g(m, 1))?)
    );
    rule!(
        t,
        r"(?<![\w.+-])(?!https?://)([A-Za-z][A-Za-z0-9+.-]{1,9})://(?=[\w/@\[])((?![\w.-]*@)[A-Za-z][\w-]*(?:\.[\w-]+)+\b)?",
        |m, _s| format!(
            "{} colon slash slash {}",
            scheme(g(m, 1))?,
            g(m, 2).replace('.', " dot ").replace('-', " dash ")
        )
    );
    rule!(t, r"\bmailto:(?=[\w.+-]+@)", "mail to ");
    rule!(t, r"\b([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)\b", |m, _s| format!(
        "{} at {}",
        g(m, 1)
            .replace('.', " dot ")
            .replace('_', " underscore ")
            .replace('-', " dash "),
        g(m, 2).replace('.', " dot ").replace('-', " dash ")
    ));
    let tld = "com|org|net|io|ai|co|uk|dev|app|gov|edu|ly|me|tv|us|eu|de|fr|info|biz";
    rule!(
        t,
        r"\bhttps?://(?![\w.-]*[:@])([A-Za-z][\w-]*(?:\.[\w-]+)+\b)?",
        |m, _s| {
            let host = g(m, 1);
            if !host.is_empty() && !test(&format!(r"\.(?:{tld})$"), host)? {
                let host = host.replace('.', " dot ").replace('-', " dash ");
                if let Some(tail) = host.strip_prefix("www ") {
                    format!("w w w {tail}")
                } else {
                    host
                }
            } else {
                host.into()
            }
        }
    );
    rule!(t, r"\bwww\.", "w w w dot ");
    rule!(t, &format!(r"\b((?:[\w-]+\.)+)({tld})\b"), |m, _s| format!(
        "{}{}",
        g(m, 1).replace('.', " dot ").replace('-', " dash "),
        g(m, 2)
    ));
    rule!(
        t,
        r"\b([\w-]+)\.(pdf|png|jpe?g|gif|docx?|xlsx?|csv|txt|zip|mp3|mp4|wav|json|py|js|html?|gradle|ipynb|rtf|pptx?|webm|bin|dat|bat|ps1|vue|jsx|kts|cs|dart|lua|r)\b",
        |m, _s| format!("{} dot {}", g(m, 1).replace('_', " underscore "), g(m, 2))
    );
    rule!(
        t,
        r"(?<![\w@])((?:/?[\w-]+/)*[\w-]*[a-z_][\w-]*)\.(conf|log|cfg|ini|yaml|yml|toml|md|sh|ts|tsx|css|xml|sql|db|bak|tmp|exe|dmg|apk|iso|tar|gz|rar|7z|svg|webp|heic|mov|avi|mkv|flac|ogg|m4a|key|pem|crt|env|lock|rs|go|java|kt|swift|rb|php|c|cpp|h|git)\b",
        |m, _s| format!(
            "{} dot {}",
            g(m, 1)
                .replace('_', " underscore ")
                .replace('/', " slash ")
                .replace('-', " dash ")
                .replace('.', " dot "),
            g(m, 2)
        )
    );
    rule!(t, r"(?<![\w])/((?:[\w.-]+/)*[\w.-]+)", |m, _s| format!(
        " slash {}",
        g(m, 1).replace('/', " slash ")
    ));
    rule!(t, r"\b(\d{1,2})-(\d{1,2})-((?:19|20)\d\d)\b", r"\1/\2/\3");
    rule!(t, r"\b(\d{1,2})\.(\d{1,2})\.((?:19|20)\d\d)\b", |m, _s| {
        let (d, mo) = (uint(g(m, 1)), uint(g(m, 2)));
        if (1..=31).contains(&d) && (1..=12).contains(&mo) {
            format!(
                "the {} of {} {}",
                ordinal(d),
                MONTHS[mo as usize - 1],
                g(m, 3)
            )
        } else {
            g(m, 0).into()
        }
    });
    let days = keys(DAYS, false);
    let months = keys(MON_ABBR, false);
    rule!(
        t,
        &format!(r"\b({days})\.?\s*[-–]\s*({days})\.?"),
        |m, _s| format!("{} to {}", table(DAYS, g(m, 1)), table(DAYS, g(m, 2)))
    );
    rule!(
        t,
        &format!(r"(?<!\bthe )(?<!\bThe )\b({days})\.(?=[\s,]|$)"),
        |m, s| format!(
            "{}{}",
            table(DAYS, g(m, 1)),
            if test(r"\A\s*(?:$|[A-Z])", after(m, s))? {
                "."
            } else {
                ""
            }
        )
    );
    rule!(
        t,
        &format!(r"\b(on|by|until|till|every|from|next|this|last) ({days})\b(?!')"),
        |m, _s| format!("{} {}", g(m, 1), table(DAYS, g(m, 2)))
    );
    rule!(t, &format!(r"\b({months})\.(?=\s*\d)"), |m, _s| table(
        MON_ABBR,
        g(m, 1)
    )
    .into());
    rule!(
        t,
        &format!(r"\b({months})\b(?=\s+\d{{1,4}}\b(?!:))"),
        |m, _s| table(MON_ABBR, g(m, 1)).into()
    );
    rule!(t, &format!(r"(?<=\d )({months})\b\.?"), |m, _s| table(
        MON_ABBR,
        g(m, 1)
    )
    .into());
    rule!(
        t,
        &format!(
            r"\b(\d{{1,4}})([A-Za-z]?)\s+((?:(?:North|South|East|West|N\.?|S\.?|E\.?|W\.?)\s+)?(?:[A-Z][\w']*|\d+(?:st|nd|rd|th))(?:\s+[A-Z][\w']*){{0,2}})\s+{STREET_RE}"
        ),
        |m, s| {
            let kind = table(STREET, g(m, 4));
            format!(
                "{}{} {} {}{}",
                nread(g(m, 1)),
                if g(m, 2).is_empty() {
                    String::new()
                } else {
                    format!(" {}", g(m, 2).to_uppercase())
                },
                g(m, 3),
                if kind.is_empty() { g(m, 4) } else { kind },
                if !g(m, 5).is_empty() && test(r"\A\s*(?:$|[A-Z\n])", after(m, s))? {
                    "."
                } else {
                    ""
                }
            )
        }
    );
    rule!(
        t,
        r"(?<!Mount )(?<!Port )(?<!Fort )(?<!Lake )(?<!Cape )\b([A-Z][a-z]+|\d+(?:st|nd|rd|th))\s+(St|Rd|Ave|Blvd|Ln|Hwy|Pkwy|Cres|Terr?)\b(\.?)",
        |m, s| {
            let key = g(m, 1).to_lowercase();
            if ROMAN_STOP.contains(&key.as_str())
                || ["on", "at", "visit", "see"].contains(&key.as_str())
                || g(m, 2) == "St" && test(r"\A\s+[A-Z]", after(m, s))?
            {
                g(m, 0).into()
            } else {
                format!(
                    "{} {}{}",
                    g(m, 1),
                    table(STREET, g(m, 2)),
                    if !g(m, 3).is_empty() && test(r"\A\s*(?:$|\n)", after(m, s))? {
                        "."
                    } else {
                        ""
                    }
                )
            }
        }
    );
    rule!(
        t,
        r"\b((?:I|US|SR|Route|Highway|Hwy|Interstate)[- ]\d+)\s([NSEW])\b",
        |m, _s| format!(
            "{} {}",
            g(m, 1),
            table(
                &[("N", "North"), ("S", "South"), ("E", "East"), ("W", "West")],
                g(m, 2)
            )
        )
    );
    for (pat, rep) in ABBR {
        rule!(t, pat, rep);
    }
    rule!(t, r"\b(\d{4})-(\d{2})-(\d{2})\b", |m, _s| {
        let mo = uint(g(m, 2));
        if (1..=12).contains(&mo) {
            format!("{} {}, {}", MONTHS[mo as usize - 1], uint(g(m, 3)), g(m, 1))
        } else {
            g(m, 0).into()
        }
    });
    rule!(
        t,
        &format!(
            r"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*[-–]\s*(\d{{1,2}})(?:st|nd|rd|th)?\s+{MON_RE}\b"
        ),
        |m, _s| {
            let (a, b) = (uint(g(m, 1)), uint(g(m, 2)));
            if a < b && b <= 31 {
                format!("the {} to the {} of {}", ordinal(a), ordinal(b), g(m, 3))
            } else {
                g(m, 0).into()
            }
        }
    );
    rule!(
        t,
        &format!(r"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MON_RE}\b"),
        |m, s| {
            let d = uint(g(m, 1));
            if (1..=31).contains(&d) {
                format!(
                    "{}{} of {}",
                    if test(r"(?i)\bthe\s+$", before(m, s))? {
                        ""
                    } else {
                        "the "
                    },
                    ordinal(d),
                    g(m, 2)
                )
            } else {
                g(m, 0).into()
            }
        }
    );
    rule!(
        t,
        r"\b(1[5-9]\d\d|20\d\d)\s*[-–]\s*(1[5-9]\d\d|20\d\d|\d\d)\b(?![-/.]\d)",
        |m, _s| {
            if g(m, 2).chars().count() == 4 || uint(g(m, 2)) > uint(tail_chars(g(m, 1), 2)) {
                format!("{} to {}", g(m, 1), g(m, 2))
            } else {
                g(m, 0).into()
            }
        }
    );
    rule!(t, r"(?<![\d/])\b(\d{1,2})/((?:19|20)\d\d)\b", |m, _s| {
        let mo = uint(g(m, 1));
        if (1..=12).contains(&mo) {
            format!("{} {}", MONTHS[mo as usize - 1], g(m, 2))
        } else {
            g(m, 0).into()
        }
    });
    rule!(t, r"(?<![\d/])\b(24/7|50/50|9/11)\b(?!/)", |m, _s| table(
        &[
            ("24/7", "24 7"),
            ("50/50", "fifty fifty"),
            ("9/11", "nine eleven")
        ],
        g(m, 1)
    )
    .into());
    rule!(
        t,
        r"(?<![\d/.,])\b(\d+)\s+(\d{1,2})/(\d{1,2})\b(?!/\d)",
        |m, _s| {
            let (a, b) = (uint(g(m, 2)), uint(g(m, 3)));
            if a > 0 && a < b && frac_den(b) {
                format!("{} and {}", g(m, 1), frac_words(a, b, true))
            } else {
                g(m, 0).into()
            }
        }
    );
    rule!(t, r"(?<![\d/])\b(\d{1,2})/(\d{1,2})\b(?!/\d)", |m, s| {
        day_month(m, s)?
    });
    rule!(
        t,
        r"\b(\d+(?:\.\d+){2,})(?:-(alpha|beta|rc\d*|preview))?\b",
        |m, _s| g(m, 1).split('.').collect::<Vec<_>>().join(" point ")
            + &if g(m, 2).is_empty() {
                String::new()
            } else {
                format!(" {}", g(m, 2))
            }
    );
    rule!(t, r"\b(\d{1,2}):(\d{2}):(\d{2})\b(?!:)", |m, s| {
        if test(r"\b(at|by|until|from)\s+$", before(m, s))? {
            format!(
                "{}:{} and {}",
                g(m, 1),
                g(m, 2),
                plural(g(m, 3), "second", true)
            )
        } else {
            (1..=3)
                .zip(["hour", "minute", "second"])
                .filter(|(i, name)| uint(g(m, *i)) != 0 || *name == "second")
                .map(|(i, name)| plural(g(m, i), name, true))
                .collect::<Vec<_>>()
                .join(", ")
        }
    });
    rule!(
        t,
        r"\b(\d{1,2}(?::\d{2})?\s?(?:[AaPp]\.?[Mm]\.?)?)\s?[-–]\s?(\d{1,2}(?::\d{2})?\s?[AaPp]\.?[Mm]\b\.?)",
        r"\1 to \2"
    );
    rule!(
        t,
        r"\b(\d{1,2}:\d{2})\s?[-–]\s?(\d{1,2}:\d{2})\b",
        r"\1 to \2"
    );
    rule!(t, r"\b(\d{1,2}):(\d)\b(?!\d)", r"\1 to \2");
    rule!(
        t,
        r"(?<=[Rr]atio of |[Rr]atio is )(\d{1,2}):(\d{1,2})\b",
        r"\1 to \2"
    );
    rule!(
        t,
        r"\b[A-Z]{1,4}(?:-[A-Z0-9]{1,6})*-\d[A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b",
        |m, _s| code_read(g(m, 0))?
    );
    rule!(
        t,
        r"(?<![\w.,])\+?\(?\d{2,4}\)?(?:[\s.-]\(?\d{2,4}\)?){1,4}(?![\w.,]*\d)",
        |m, _s| {
            let s = g(m, 0);
            let groups = rx(r"\d+")?
                .find_iter(s)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            if groups
                .iter()
                .map(|m| m.as_str().chars().count())
                .sum::<usize>()
                >= 7
            {
                format!(
                    " {}{} ",
                    if trim(s).starts_with('+') {
                        "plus "
                    } else {
                        ""
                    },
                    groups
                        .iter()
                        .map(|m| digits(m.as_str()))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            } else {
                s.into()
            }
        }
    );
    rule!(
        t,
        r"\b((?:[Cc]all|[Dd]ial|[Tt]ext|[Rr]ing)\s+)(\d{3,6})\b",
        |m, _s| format!("{}{}", g(m, 1), digits(g(m, 2)))
    );
    rule!(
        t,
        r"\b(\d+)\s?ft\.?\s?(\d{1,2})\s?in\b\.?",
        |m, _s| format!(
            "{} {} {} {}",
            g(m, 1),
            if g(m, 1) == "1" { "foot" } else { "feet" },
            g(m, 2),
            if g(m, 2) == "1" { "inch" } else { "inches" }
        )
    );
    rule!(t, r#"\b(\d)'\s?(\d{1,2})(?:\"|'')"#, r"\1 foot \2");
    let units: Vec<_> = UNITS
        .iter()
        .copied()
        .filter(|(k, _)| !["in", "s"].contains(k))
        .collect();
    let ukeys = keys(&units, true);
    rule!(
        t,
        &format!(r"{UNIT_PRE}\s?({ukeys})/({ukeys}|s|in)(?![\w/])"),
        |m, _s| {
            if g(m, 3) == "in" {
                g(m, 0).into()
            } else {
                format!(
                    "{} {} per {}",
                    g(m, 1),
                    unit(g(m, 2), g(m, 1)),
                    sing(table(UNITS, g(m, 3)))
                )
            }
        }
    );
    rule!(
        t,
        &format!(r"{UNIT_PRE}\s?({ukeys})(?![\w/'])"),
        |m, _s| format!("{} {}", g(m, 1), unit(g(m, 2), g(m, 1)))
    );
    rule!(
        t,
        &format!(
            r"(\b(?:squared|cubed|to the (?:minus )?(?:\w+ )?\w+(?:th|st|nd|rd))\s)({ukeys})(?:/({ukeys}|s))?(?![\w/'])"
        ),
        |m, _s| format!(
            "{}{}{}",
            g(m, 1),
            table(UNITS, g(m, 2)),
            if g(m, 3).is_empty() {
                String::new()
            } else {
                format!(" per {}", sing(table(UNITS, g(m, 3))))
            }
        )
    );
    rule!(
        t,
        &format!(
            r"\b({ID_WORDS})(\s+(?:No\.?\s*|#\s*)?)([A-Z]{{1,3}}\s?)?(\d{{2,4}})([A-Z]{{0,2}})\b(?![.,]\d|\s?[-–]\s?\d)"
        ),
        |m, _s| format!(
            "{}{}{}{}{}",
            g(m, 1),
            g(m, 2),
            if g(m, 3).is_empty() {
                String::new()
            } else {
                format!("{} ", spell(g(m, 3)))
            },
            nread(g(m, 4)),
            if g(m, 5).is_empty() {
                String::new()
            } else {
                format!(" {}", spell(g(m, 5)))
            }
        )
    );
    rule!(
        t,
        r"\b([A-Z]{2,3})\s(\d{3,4})\b(?![.,:]\d|\s\d)",
        |m, _s| format!("{} {}", g(m, 1), nread(g(m, 2)))
    );
    rule!(
        t,
        r"(?<![-/.\w])([A-Z]{1,3})(\d{1,4})([A-Z]{0,2})(?![-/\w])",
        |m, s| {
            if pair(m, s)?
                && (g(m, 2).chars().count() <= 2 || g(m, 1).len() == 2 && g(m, 3).is_empty())
            {
                [spell(g(m, 1)), nread(g(m, 2)), spell(g(m, 3))]
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                g(m, 0).into()
            }
        }
    );
    rule!(
        t,
        r"(?<![-/.\w])(\d{1,4})([A-Z]{1,2})(?![-/\w])",
        |m, s| if pair(m, s)? {
            format!("{} {}", nread(g(m, 1)), spell(g(m, 2)))
        } else {
            g(m, 0).into()
        }
    );
    rule!(
        t,
        r"(?<![-/.\w])(?=[A-Za-z]*\d)(?:[A-Z][a-z]?\d{0,3}){2,}(?![-/\w])",
        |m, s| {
            let r = rx(r"([A-Z][a-z]?)(\d*)")?;
            let parts = r
                .captures_iter(g(m, 0))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            if parts.iter().all(|p| ELEMENTS.contains(&g(p, 1))) && pair(m, s)? {
                parts
                    .iter()
                    .map(|p| {
                        format!(
                            "{}{}",
                            spell(g(p, 1)),
                            if g(p, 2).is_empty() {
                                String::new()
                            } else {
                                format!(" {}", cardinal(uint(g(p, 2))))
                            }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                g(m, 0).into()
            }
        }
    );
    rule!(t, r"#(\d+(?:-\d+)*)\b", |m, _s| format!(
        "number {}",
        g(m, 1)
            .split('-')
            .map(|g| if g.chars().count() <= 3 {
                g.into()
            } else {
                digits(g)
            })
            .collect::<Vec<_>>()
            .join(", ")
    ));
    rule!(
        t,
        r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b",
        |m, _s| code_read(g(m, 0))?
    );
    rule!(t, r"\b(?=\w*\d)(?=\w*[A-Z])[A-Z0-9]{3,}\b", |m, _s| spell(
        g(m, 0)
    ));
    rule!(
        t,
        r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{4,}\b",
        |m, _s| {
            if full(
                r"(?i)\d+(?:st|nd|rd|th|am|pm|s|kg|km|mm|cm|ml|mg|gb|mb|tb|kb|ghz|mhz|kw|kwh|mah|hrs?|mins?|h|m|k|x|g|l|w|v)",
                g(m, 0),
            )? {
                g(m, 0).into()
            } else {
                spell(g(m, 0))
            }
        }
    );
    rule!(t, r"(?<![\d$€£.,])\b(\d{5,})\b(?![.,]\d)", |m, _s| {
        digits(g(m, 1))
    });
    rule!(
        t,
        r"(?<![\d/:.-])\b(\d{1,4})\s*[-–]\s*(\d{1,4})\b(?![/:.-]\d)",
        r"\1 to \2"
    );
    rule!(t, r"\b(\d+)\s*h\s*(\d{1,2})\b", |m, _s| format!(
        "{} {}",
        plural(g(m, 1), "hour", false),
        g(m, 2)
    ));
    rule!(t, r"\b(\d+)\s*hrs?\b", |m, _s| plural(
        g(m, 1),
        "hour",
        false
    ));
    rule!(t, r"\b(\d+)\s*mins?\b", |m, _s| plural(
        g(m, 1),
        "minute",
        false
    ));
    rule!(
        t,
        &format!(r"{UNIT_PRE}\s?({ukeys})(?![\w/'])"),
        |m, _s| format!("{} {}", g(m, 1), unit(g(m, 2), g(m, 1)))
    );
    rule!(t, r"\s*×\s*", " by ");
    rule!(t, r"(?<=\d)\s+x\s+(?=\d)", " by ");
    for (pat, rep) in SYMBOLS {
        rule!(t, pat, rep);
    }
    rule!(t, r"£(\d[\d,]*)\.(\d{2})\b", |m, _s| format!(
        "{} pounds{}",
        g(m, 1),
        if g(m, 2) == "00" {
            String::new()
        } else {
            format!(" {}", uint(g(m, 2)))
        }
    ));
    rule!(t, r"\s+–\s+", " — ");
    t = t.replace('–', "-");
    rule!(t, r"\s{2,}", " ");
    Ok(trim(&t).into())
}
fn scheme(s: &str) -> Result<String, String> {
    let l = s.to_lowercase();
    let known = table(SCHEME_SAY, &l);
    Ok(if !known.is_empty() {
        known.into()
    } else if pronounceable(s)? {
        l
    } else {
        spell(s)
    })
}
fn pair(m: &Captures<'_>, s: &str) -> Result<bool, String> {
    Ok(!test(CODE_CTX, tail_chars(before(m, s), 40))?)
}
fn day_month(m: &Captures<'_>, s: &str) -> Result<String, String> {
    let (a, b) = (uint(g(m, 1)), uint(g(m, 2)));
    let (after, before) = (
        head_chars(after(m, s), 12).to_lowercase(),
        tail_chars(before(m, s), 16),
    );
    if test(r"\A\s*(stars?|points?|marks?|votes?)\b", &after)?
        || test(r"(rated|scored?|rating|got)\s*$", &before.to_lowercase())?
    {
        return Ok(format!("{a} out of {b}"));
    }
    if a > 0
        && a < b
        && frac_den(b)
        && !g(m, 1).starts_with('0')
        && !g(m, 2).starts_with('0')
        && (!test(&format!("(?i){DATE_BEFORE}"), before)?
            || test(&format!(r"\A{FRAC_AFTER}"), &after)?)
    {
        return Ok(frac_words(a, b, false));
    }
    let suffix = |n: u64| {
        if (10..=20).contains(&(n % 100)) {
            "th"
        } else {
            match n % 10 {
                1 => "st",
                2 => "nd",
                3 => "rd",
                _ => "th",
            }
        }
    };
    if (1..=12).contains(&b) && (1..=31).contains(&a) {
        return Ok(format!(
            "{}{a}{} of {}",
            if test(r"(?i)\bthe\s+$", before)? {
                ""
            } else {
                "the "
            },
            suffix(a),
            MONTHS[b as usize - 1]
        ));
    }
    if (1..=12).contains(&a) && (1..=31).contains(&b) {
        return Ok(format!("{} {b}{}", MONTHS[a as usize - 1], suffix(b)));
    }
    Ok(g(m, 0).into())
}
fn fallback(mut t: String) -> Result<String, String> {
    rule!(t, r"([$€£])(\d[\d,]*)(?:\.(\d+))?\b", |m, _s| {
        let (one, many, c1, cn) = match g(m, 1) {
            "$" => ("dollar", "dollars", "cent", "cents"),
            "€" => ("euro", "euros", "cent", "cents"),
            _ => ("pound", "pounds", "penny", "pence"),
        };
        let whole = canonical(&g(m, 2).replace(',', ""));
        let frac = g(m, 3);
        if !frac.is_empty() && frac.chars().count() != 2 {
            format!("{} {many}", number(&format!("{}.{frac}", g(m, 2))))
        } else {
            let c = uint(frac);
            if whole == "0" && c != 0 {
                format!("{} {}", cardinal(c), if c == 1 { c1 } else { cn })
            } else {
                format!(
                    "{} {}{}",
                    cardinal_str(&whole),
                    if whole == "1" { one } else { many },
                    if c == 0 {
                        String::new()
                    } else {
                        format!(" and {} {}", cardinal(c), if c == 1 { c1 } else { cn })
                    }
                )
            }
        }
    });
    rule!(
        t,
        r"\b(\d{1,2}):(\d{2})\s*([aApP]\.?[mM])?(?:\.(?=\s+[a-z]))?(?=\W|$)",
        |m, s| {
            let (h, mi, am) = (uint(g(m, 1)), uint(g(m, 2)), g(m, 3));
            if mi == 0 && am.is_empty() && test(r"\A\s*(noon|midnight)", after(m, s))? {
                cardinal(h)
            } else if mi == 0 && am.is_empty() && (h > 12 || h == 0) {
                format!("{} hundred", cardinal(h))
            } else {
                format!(
                    "{}{}{}",
                    cardinal(h),
                    if mi == 0 {
                        if am.is_empty() {
                            " o'clock".into()
                        } else {
                            String::new()
                        }
                    } else if mi < 10 {
                        format!(" oh {}", ONES[mi as usize])
                    } else {
                        format!(" {}", cardinal(mi))
                    },
                    if am.is_empty() {
                        String::new()
                    } else {
                        format!(" {}", am.replace('.', "").to_uppercase())
                    }
                )
            }
        }
    );
    rule!(
        t,
        r"\b(\d{1,2})\s*([aApP])\.?[mM](?:\.(?=\s+[a-z]))?(?=\W|$)",
        |m, _s| format!("{} {}M", cardinal(uint(g(m, 1))), g(m, 2).to_uppercase())
    );
    rule!(t, r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b", |m, _s| {
        let (mo, d, y) = (
            uint(g(m, 1)),
            uint(g(m, 2)),
            uint(g(m, 3))
                + if g(m, 3).chars().count() == 2 {
                    2000
                } else {
                    0
                },
        );
        if (13..=31).contains(&mo) && (1..=12).contains(&d) {
            format!(
                "the {} of {} {}",
                ordinal(mo),
                MONTHS[d as usize - 1],
                year(y)
            )
        } else if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
            g(m, 0).into()
        } else {
            format!("{} {} {}", MONTHS[mo as usize - 1], ordinal(d), year(y))
        }
    });
    rule!(
        t,
        &format!(r"\b({})\.? (\d{{1,2}})\b(?!:)", MONTHS.join("|")),
        |m, _s| format!("{} {}", g(m, 1), ordinal(uint(g(m, 2))))
    );
    rule!(t, r"(\d)\s?°\s?C\b", r"\1 degrees Celsius");
    rule!(t, r"(\d)\s?°\s?F\b", r"\1 degrees Fahrenheit");
    rule!(t, r"(\d)\s?°", r"\1 degrees");
    rule!(t, r"(?<=\w)/(?=\w)", " slash ");
    rule!(t, r"\b(\d+)(st|nd|rd|th)\b", |m, _s| ordinal_str(g(m, 1)));
    rule!(t, r"\b(1[1-9]\d\d|20\d\d)\b(?![.,]\d)", |m, _s| year(uint(
        g(m, 1)
    )));
    rule!(t, r"(?<![\w-])-(?=\d)", "minus ");
    rule!(t, r"(?<=[A-Za-z])(?=\d)", " ");
    rule!(t, r"(\d+(?:\.\d+)?)\s?%", |m, _s| format!(
        "{} percent",
        number(g(m, 1))
    ));
    rule!(
        t,
        r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b",
        |m, _s| number(g(m, 0))
    );
    Ok(t.replace('&', " and ")
        .replace('+', " plus ")
        .replace('=', " equals ")
        .replace('@', " at ")
        .replace('#', " number "))
}

/// Default official sentence/clause splitting (CLAUSE_MAX=120, FIRST_MAX=0).
pub fn split_text(text: &str) -> Vec<String> {
    let text = trim(text);
    let mut sentences = Vec::new();
    let mut start = 0;
    let mut previous = '\0';
    let mut splitting = false;
    for (i, c) in text.char_indices() {
        if py_space(c) && (splitting || ['.', '!', '?', '…'].contains(&previous)) {
            if !splitting {
                sentences.push(&text[start..i]);
                splitting = true;
            }
            start = i + c.len_utf8();
        } else {
            splitting = false;
        }
        previous = c;
    }
    if start < text.len() {
        sentences.push(&text[start..]);
    }
    let mut out = Vec::new();
    for s in sentences.into_iter().filter(|s| !s.is_empty()) {
        if s.chars().count() <= 120 {
            out.push(s.to_owned());
            continue;
        }
        let mut clauses = Vec::new();
        let mut start = 0;
        let mut cursor = 0;
        while cursor < s.len() {
            let c = s[cursor..].chars().next().unwrap();
            let end = cursor + c.len_utf8();
            let dash = c == '—' || c == '-' && s[..cursor].ends_with('-');
            let punct_space =
                [',', ';', ':'].contains(&c) && s[end..].chars().next().is_some_and(py_space);
            if dash || punct_space {
                if start < end {
                    clauses.push(&s[start..end]);
                }
                start = end;
                while start < s.len() && s[start..].chars().next().is_some_and(py_space) {
                    start += s[start..].chars().next().unwrap().len_utf8();
                }
                cursor = start;
            } else {
                cursor = end;
            }
        }
        if start < s.len() {
            clauses.push(&s[start..]);
        }
        let mut merged = Vec::new();
        let mut current = String::new();
        for c in clauses {
            if !current.is_empty() && current.chars().count() + 1 + c.chars().count() > 120 {
                merged.push(current);
                current = c.into();
            } else if current.is_empty() {
                current = c.into();
            } else {
                if !(current.ends_with('—') || current.ends_with("--")) {
                    current.push(' ');
                }
                current.push_str(c);
            }
        }
        if !current.is_empty() {
            merged.push(current);
        }
        for s in merged {
            split_conj(&s, &mut out);
        }
    }
    out
}
fn split_conj(s: &str, out: &mut Vec<String>) {
    let len = s.chars().count();
    if len <= 120 {
        out.push(s.into());
        return;
    }
    const CONJ: &[&str] = &[
        "and", "but", "or", "nor", "that", "which", "who", "whom", "whose", "when", "where",
        "while", "because", "although", "though", "so", "if", "as", "than", "until",
    ];
    let mut candidate = None;
    for (n, (i, c)) in s.char_indices().enumerate() {
        if n < 30 || len - n < 30 || !py_space(c) {
            continue;
        }
        let after = &s[i + c.len_utf8()..];
        if CONJ.iter().any(|w| {
            after
                .strip_prefix(w)
                .is_some_and(|tail| !tail.chars().next().is_some_and(py_word))
        }) {
            let distance = (2 * n).abs_diff(len);
            if candidate.is_none_or(|(_, d)| distance < d) {
                candidate = Some((i, distance));
            }
        }
    }
    if let Some((i, _)) = candidate {
        split_conj(trim(&s[..i]), out);
        split_conj(trim(&s[i..]), out);
    } else {
        out.push(s.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_official_frontend_fixtures() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("text_fixtures.json")).unwrap();
        let frontend = Frontend::new(fixtures["lexicon"].as_str().unwrap()).unwrap();
        let mut mismatches = Vec::new();
        for case in fixtures["normalization"].as_array().unwrap() {
            let input = case[0].as_str().unwrap();
            let expected = case[1].as_str().unwrap();
            match frontend.normalize(input) {
                Ok(actual) if actual == expected => {}
                actual => mismatches.push(format!(
                    "input: {input:?}\nexpected: {expected:?}\nactual: {actual:?}"
                )),
            }
        }
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n\n"));
        for case in fixtures["splitting"].as_array().unwrap() {
            let input = case[0].as_str().unwrap();
            let expected: Vec<_> = case[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap().to_owned())
                .collect();
            assert_eq!(split_text(input), expected, "input: {input:?}");
        }
    }
    #[test]
    fn exact_character_vocabulary() {
        let encoded = encode_text(CHARS);
        assert_eq!(encoded[0], 1);
        assert_eq!(
            &encoded[1..],
            &(3..3 + CHARS.chars().count() as u32).collect::<Vec<_>>()
        );
        assert_eq!(encode_text("a🙂é\n"), vec![1, 4, 2, 2, 2]);
        assert_eq!(encode_text(""), vec![1]);
    }
    #[test]
    fn respelling_boundaries_caps_accents_and_possessives() {
        let frontend = Frontend::new("Eoin\tOwen\ncrème brûlée\tkrem broolay\nWorcestershire\twoostersher\nNew York\tnoo york\tc\n").unwrap();
        assert_eq!(
            frontend
                .respell("Eoin's CRÈME BRÛLÉE Worcestershire-glazed")
                .unwrap(),
            "Owen's Krem broolay Woostersher-glazed"
        );
        assert_eq!(
            frontend
                .respell("new York New York Worcestershire-Sauce")
                .unwrap(),
            "new York Noo york Worcestershire-Sauce"
        );
        assert_eq!(
            frontend.respell("crème  brûlée and eoinish").unwrap(),
            "crème  brûlée and eoinish"
        );
        assert!(Frontend::new("broken row").is_err());
    }
}
