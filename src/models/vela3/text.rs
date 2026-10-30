use regex::Regex;
use serde_json::{json, Value};
use std::sync::OnceLock;

const CJK: &str = "⺀-⿿　-〿぀-ヿ㄀-ㇿ㐀-䶿一-鿿ꀀ-꓏가-힯豈-﫿︰-﹏＀-￯";
const FAST_CJK: &str = "⺀-⿿　-〿぀-ヿ㄀-ㇿ㐀-䶿一-鿿ꀀ-꓏가-힯豈-﫿︰-﹏＀-￯";
const EDGE: &str = "[](){}<>「」『』【】《》〈〉\"'“”‘’«»‹›";

pub(crate) fn threshold_value(value: &Value) -> Result<f64, String> {
    match value {
        Value::String(s) => s.trim().parse::<f64>().ok().or_else(|| {
            static DECIMAL: OnceLock<Regex> = OnceLock::new();
            let decimal = DECIMAL.get_or_init(|| Regex::new(r"^\p{Nd}$").unwrap());
            let is_digit = |c: char| decimal.is_match(c.encode_utf8(&mut [0; 4]));
            let mut chars = s.trim().chars().peekable();
            let mut normalized = String::new();
            let mut previous_digit = false;
            while let Some(c) = chars.next() {
                if c == '_' {
                    if !previous_digit || !chars.peek().is_some_and(|&c| is_digit(c)) {
                        return None;
                    }
                    continue;
                }
                previous_digit = is_digit(c);
                let c = if previous_digit {
                    let mut start = c as u32;
                    while char::from_u32(start - 1).is_some_and(is_digit) {
                        start -= 1;
                    }
                    // Unicode decimal runs contain one or more consecutive sets of ten digits.
                    char::from(b'0' + ((c as u32 - start) % 10) as u8)
                } else {
                    c
                };
                normalized.push(c);
            }
            normalized.parse::<f64>().ok()
        }),
        Value::Bool(b) => Some(if *b { 1. } else { 0. }),
        _ => value.as_f64(),
    }
    .filter(|v| v.is_finite())
    .ok_or_else(|| "span threshold must be a finite number".into())
}

pub(crate) fn split_words(text: &str) -> Vec<(usize, usize)> {
    static WORD: OnceLock<Regex> = OnceLock::new();
    let re = WORD.get_or_init(|| Regex::new(&format!(
        r"(?i)(?:https?://|www\.)[^\s{FAST_CJK}]+|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{{2,}}|@[a-z0-9_]+|[{FAST_CJK}]|[\p{{L}}\p{{N}}_&&[^{FAST_CJK}]]+(?:[-_][\p{{L}}\p{{N}}_&&[^{FAST_CJK}]]+)*|\S"
    )).expect("static Vela word pattern"));
    let mut words = Vec::new();
    for m in re.find_iter(text) {
        let lower = m.as_str().to_ascii_lowercase();
        let end = if lower.starts_with("http://")
            || lower.starts_with("https://")
            || lower.starts_with("www.")
        {
            m.start() + url_unit_end(m.as_str(), url_prefix(&lower), ".,;:!?)\\]}>\"'”’»")
        } else {
            m.end()
        };
        if end > m.start() {
            words.push((m.start(), end));
        }
        for (i, c) in text[end..m.end()].char_indices() {
            words.push((end + i, end + i + c.len_utf8()));
        }
    }
    words
}

pub(crate) fn decode_spans(
    text: &str,
    offsets: &[(usize, usize)],
    probs: &[f32],
    names: &[String],
    threshold: f32,
) -> Result<Value, String> {
    decode_profile(text, offsets, probs, names, threshold, false)
}
fn decode_profile(
    text: &str,
    offsets: &[(usize, usize)],
    probs: &[f32],
    names: &[String],
    threshold: f32,
    fast: bool,
) -> Result<Value, String> {
    if names.is_empty()
        || offsets.len().checked_mul(names.len()) != Some(probs.len())
        || !threshold.is_finite()
        || probs.iter().any(|p| !p.is_finite())
    {
        return Err("invalid span probability shape or non-finite value".into());
    }
    let mut previous = 0;
    for &(a, b) in offsets {
        if a > b || a < previous || !text.is_char_boundary(a) || !text.is_char_boundary(b) {
            return Err("invalid or unordered UTF-8 span offsets".into());
        }
        previous = a;
    }
    static UNIT: OnceLock<Regex> = OnceLock::new();
    static FAST_UNIT: OnceLock<Regex> = OnceLock::new();
    let unit = if fast {
        FAST_UNIT.get_or_init(||Regex::new(&format!(
        r#"(?:https?://|www\.)[^\s\]\)\}}>「」『』"'“”{FAST_CJK}]+|[\p{{L}}\p{{N}}@._+\-&&[^{FAST_CJK}]]+|\S"#
    )).expect("static fast unit pattern"))
    } else {
        UNIT.get_or_init(|| {
            Regex::new(&format!(
        r#"(?:https?://|www\.)[^\s\]\)\}}>「」『』"'“”{CJK}]+|[\p{{L}}\p{{N}}@._+\-&&[^{CJK}]]+|\S"#
    )).expect("static Vela unit pattern")
        })
    };
    let mut units = Vec::new();
    for m in unit.find_iter(text) {
        let prefix = url_prefix(m.as_str());
        let end = if fast && prefix > 0 {
            url_unit_end(m.as_str(), prefix, ".,;:!?") + m.start()
        } else {
            m.end()
        };
        if end > m.start() {
            units.push((m.start(), end));
        }
        for (i, c) in text[end..m.end()].char_indices() {
            units.push((end + i, end + i + c.len_utf8()));
        }
    }
    let mut labels = vec![None; offsets.len()];
    let mut maxima = vec![0.0; offsets.len()];
    let mut token_units = vec![None; offsets.len()];
    for (i, (&(a, b), row)) in offsets
        .iter()
        .zip(probs.chunks_exact(names.len()))
        .enumerate()
    {
        if a == b {
            continue;
        }
        let mut best = 0;
        for j in 1..row.len() {
            if row[j] > row[best] {
                best = j;
            }
        }
        maxima[i] = row[best];
        labels[i] = (row[best] > threshold).then_some(best);
        let first = a + text[a..b].len() - text[a..b].trim_start().len();
        token_units[i] = units.iter().position(|&(s, e)| s <= first && first < e);
    }
    let mut i = 0;
    while i < offsets.len() {
        let Some(unit) = token_units[i] else {
            i += 1;
            continue;
        };
        let mut end = i + 1;
        while end < offsets.len()
            && (token_units[end] == Some(unit) || offsets[end].0 == offsets[end].1)
        {
            end += 1;
        }
        let mut order = Vec::new();
        let mut scores = vec![0.0f64; names.len()];
        for t in i..end {
            if let Some(c) = labels[t] {
                if !order.contains(&c) {
                    order.push(c);
                }
                scores[c] += maxima[t] as f64;
            }
        }
        if let Some(&first) = order.first() {
            let mut winner = first;
            for &c in &order[1..] {
                if scores[c] > scores[winner] {
                    winner = c;
                }
            }
            let lo = (i..end).find(|&t| labels[t].is_some()).unwrap();
            let hi = (i..end).rfind(|&t| labels[t].is_some()).unwrap();
            for t in lo..=hi {
                if offsets[t].0 < offsets[t].1 {
                    labels[t] = Some(winner);
                }
            }
        }
        i = end;
    }
    let mut runs: Vec<(usize, usize, usize, Vec<usize>)> = Vec::new();
    let mut current: Option<(usize, usize, usize, Vec<usize>)> = None;
    for (i, &(a, b)) in offsets.iter().enumerate() {
        if a == b {
            continue;
        }
        if let Some(c) = labels[i] {
            if let Some(run) = current.as_mut() {
                if run.0 == c {
                    run.2 = b;
                    run.3.push(i);
                    continue;
                }
            }
            if let Some(run) = current.take() {
                runs.push(run);
            }
            current = Some((c, a, b, vec![i]));
        } else if let Some(run) = current.take() {
            runs.push(run);
        }
    }
    if let Some(run) = current {
        runs.push(run);
    }
    let mut spans = Vec::new();
    for (c, a, b, indices) in runs {
        let trimmed = text[a..b].trim_matches(|ch: char| ch.is_whitespace() || EDGE.contains(ch));
        if trimmed.is_empty() {
            continue;
        }
        let start = a + text[a..b].len()
            - text[a..b]
                .trim_start_matches(|ch: char| ch.is_whitespace() || EDGE.contains(ch))
                .len();
        let mut end = start + trimmed.len();
        if fast {
            let lower = text[start..end].to_ascii_lowercase();
            if url_prefix(&lower) > 0 {
                static CJK_CHAR: OnceLock<Regex> = OnceLock::new();
                if let Some(m) = CJK_CHAR
                    .get_or_init(|| Regex::new(&format!("[{FAST_CJK}]")).unwrap())
                    .find(&text[start..end])
                {
                    end = start + m.start();
                }
                while end > start {
                    let ch = text[start..end].chars().next_back().unwrap();
                    if ch.is_whitespace() || EDGE.contains(ch) || ".,;:!?)".contains(ch) {
                        end -= ch.len_utf8();
                    } else {
                        break;
                    }
                }
            }
        }
        if end <= start {
            continue;
        }
        let trimmed = &text[start..end];
        let mut score = 0.0f64;
        let mut count = 0;
        for (i, &(s, e)) in offsets.iter().enumerate() {
            if s < end && e > start {
                score += probs[i * names.len() + c] as f64;
                count += 1;
            }
        }
        let mut span = json!({"start":text[..start].chars().count(),"end":text[..end].chars().count(),"label":names[c],"text":trimmed,"score":if count==0 {0.0} else {score/count as f64}});
        if fast {
            let probability = indices
                .iter()
                .map(|&i| probs[i * names.len() + c])
                .sum::<f32>()
                / indices.len() as f32;
            span["probability"] = json!(probability);
        }
        spans.push(span);
    }
    Ok(Value::Array(spans))
}
fn url_prefix(text: &str) -> usize {
    if text.starts_with("https://") {
        8
    } else if text.starts_with("http://") {
        7
    } else if text.starts_with("www.") {
        4
    } else {
        0
    }
}
fn url_unit_end(text: &str, prefix: usize, trail: &str) -> usize {
    let trimmed = text.trim_end_matches(|c| trail.contains(c)).len();
    let first = prefix
        + text[prefix..]
            .chars()
            .next()
            .map(char::len_utf8)
            .unwrap_or(0);
    trimmed.max(first)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decoder_url_units_are_case_sensitive_like_the_official_python_regex() {
        let got = decode_spans(
            "HTTPS://x.org/a",
            &[(0, 5), (8, 15)],
            &[0.9, 0.1, 0.1, 0.8],
            &["a".into(), "b".into()],
            0.5,
        )
        .unwrap();
        assert_eq!(got.as_array().unwrap().len(), 2);
        assert_eq!(got[0]["label"], "a");
        assert_eq!(got[1]["label"], "b");
    }
    #[test]
    fn words_preserve_urls_email_cjk_and_utf8_byte_offsets() {
        let text = "é-mail 张伟 x@y.org https://x.org/a. @alice";
        let words: Vec<_> = split_words(text)
            .into_iter()
            .map(|(a, b)| &text[a..b])
            .collect();
        assert_eq!(
            words,
            [
                "é-mail",
                "张",
                "伟",
                "x@y.org",
                "https://x.org/a",
                ".",
                "@alice"
            ]
        );
    }
    #[test]
    fn unit_vote_fills_inner_holes_trims_edges_and_returns_codepoints() {
        let text = "中 [a_b] x";
        let offsets = vec![(4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (10, 11)];
        let probs = [0.8, 0.1, 0.9, 0.1, 0.2, 0.1, 0.1, 0.7, 0.8, 0.1, 0.1, 0.1];
        let got = decode_spans(
            text,
            &offsets,
            &probs,
            &["first".into(), "second".into()],
            0.5,
        )
        .unwrap();
        assert_eq!(got[0]["start"], 3);
        assert_eq!(got[0]["end"], 6);
        assert_eq!(got[0]["text"], "a_b");
        assert_eq!(got[0]["label"], "first");
        assert_eq!(got.as_array().unwrap().len(), 1);
    }
    #[test]
    fn threshold_is_strict_and_invalid_shapes_are_rejected() {
        assert_eq!(
            decode_spans("x", &[(0, 1)], &[0.5], &["x".into()], 0.5).unwrap(),
            json!([])
        );
        assert!(decode_spans("é", &[(1, 2)], &[0.9], &["x".into()], 0.5).is_err());
        assert!(decode_spans("x", &[(0, 1)], &[], &["x".into()], 0.5).is_err());
        assert!(decode_spans("x", &[(0, 1)], &[f32::NAN], &["x".into()], 0.5).is_err());
    }

    #[test]
    fn span_thresholds_accept_finite_python_numeric_values() {
        for (value, expected) in [
            (json!(" 0.5 "), 0.5),
            (json!("０.５"), 0.5),
            (json!("٠.٥"), 0.5),
            (json!("0_0.5"), 0.5),
            (json!("1_0e-2"), 0.1),
            (json!(true), 1.),
            (json!(false), 0.),
            (json!(-1), -1.),
            (json!(2), 2.),
            (json!("1e300"), 1e300),
            (json!(-1e300), -1e300),
        ] {
            let threshold = threshold_value(&value).unwrap();
            assert_eq!(threshold, expected);
            let spans = decode_spans(
                "word",
                &[(0, 4)],
                &[0.5],
                &["label".into()],
                threshold.clamp(-1., 2.) as f32,
            )
            .unwrap();
            assert_eq!(spans.as_array().unwrap().is_empty(), expected >= 0.5);
        }
        for value in [
            json!(null),
            json!([]),
            json!({}),
            json!("NaN"),
            json!("inf"),
            json!("0__5"),
            json!("_0.5"),
            json!("0_.5"),
            json!("0._5"),
            json!("1e_2"),
        ] {
            assert!(threshold_value(&value).is_err());
        }
    }
}

pub(crate) fn decode_fast_spans(
    text: &str,
    offsets: &[(usize, usize)],
    probs: &[f32],
    names: &[String],
    threshold: f32,
) -> Result<Value, String> {
    decode_profile(text, offsets, probs, names, threshold, true)
}
#[cfg(test)]
mod fast_tests {
    use super::*;
    #[test]
    fn fast_url_keeps_original_run_probability_and_trims_the_score_range() {
        let text = "[https://x.org/a).]";
        let offsets = vec![(0, 1), (1, 16), (16, 17), (17, 18), (18, 19)];
        let out = decode_fast_spans(
            text,
            &offsets,
            &[0.9, 0.8, 0.7, 0.6, 0.5],
            &["url".into()],
            0.4,
        )
        .unwrap();
        assert_eq!(out[0]["text"], "https://x.org/a");
        assert_eq!(out[0]["start"], 1);
        assert_eq!(out[0]["end"], 16);
        assert!((out[0]["probability"].as_f64().unwrap() - 0.7).abs() < 1e-6);
        assert!((out[0]["score"].as_f64().unwrap() - 0.8).abs() < 1e-6);
    }
    #[test]
    fn word_units_use_the_official_broad_cjk_range() {
        assert_eq!(split_words("ꔀꔁ abc"), vec![(0, 3), (3, 6), (7, 10)]);
    }
}
