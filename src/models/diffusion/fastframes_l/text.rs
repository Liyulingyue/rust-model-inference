use super::{math::*, weights::Graph};
use crate::compute::linear::LinearExecutor;
use crate::core::tensor::{MetaValue, TensorSource};
use crate::ops::nn::{
    attention_online_with_bias, gelu_approx_inplace, rms_norm_rows, AttentionSpec, Linear,
};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

pub(super) fn tokenizer(source: &dyn TensorSource) -> Result<tokenizers::Tokenizer, String> {
    // Both variants share the pinned umT5 tokenizer; either key is accepted.
    let json = ["fastframes_l", "fastframes_s"]
        .iter()
        .find_map(|variant| {
            source
                .metadata(&format!("{variant}.tokenizer_json"))
                .and_then(MetaValue::to_string_val)
        })
        .ok_or("missing FastFrames tokenizer JSON")?;
    if format!("{:x}", Sha256::digest(json.as_bytes()))
        != "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"
    {
        return Err("FastFrames tokenizer SHA256 mismatch".into());
    }
    let mut tokenizer =
        tokenizers::Tokenizer::from_bytes(json.as_bytes()).map_err(|e| e.to_string())?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(tokenizers::utils::truncation::TruncationParams {
            max_length: 512,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
    if tokenizer.get_vocab_size(true) != 256300 {
        return Err("FastFrames tokenizer vocabulary mismatch".into());
    }
    Ok(tokenizer)
}

pub(super) fn ids(tokenizer: &tokenizers::Tokenizer, prompt: &str) -> Result<Vec<u32>, String> {
    let encoding = tokenizer
        .encode(clean(prompt), true)
        .map_err(|e| e.to_string())?;
    let ids = encoding.get_ids().to_vec();
    if ids.is_empty()
        || ids.len() > 512
        || ids.iter().any(|&id| id >= 256384)
        || ids.last() != Some(&1)
    {
        return Err("invalid FastFrames-L tokenizer output (expected terminal EOS)".into());
    }
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::token_ids("fastframes.token_ids", &ids));
    Ok(ids)
}

pub(super) fn encode<'a>(
    g: &'a Graph<'a>,
    ex: &mut LinearExecutor<'a, 'a>,
    ids: &[u32],
) -> Result<Vec<f32>, String> {
    const D: usize = 4096;
    let n = ids.len();
    let mut x = vec![0.; n * D];
    for (&id, row) in ids.iter().zip(x.chunks_exact_mut(D)) {
        g.linear["text.token_embedding"].embedding_lookup(id, row)?;
    }
    for layer in 0..24 {
        let p = format!("text.blocks.{layer}");
        let h = rms_norm_rows(&x, n, &g.floats[&format!("{p}.norm1.weight")], 1e-6)?;
        let [q, k, v] = Linear::forward_group(
            [
                &g.linear[&format!("{p}.attn.q")],
                &g.linear[&format!("{p}.attn.k")],
                &g.linear[&format!("{p}.attn.v")],
            ],
            ex,
            &h,
            n,
        )?;
        let embedding = &g.floats[&format!("{p}.pos_embedding.embedding.weight")];
        let mut bias = vec![0.; 64 * n * n];
        for head in 0..64 {
            for i in 0..n {
                for j in 0..n {
                    bias[(head * n + i) * n + j] =
                        embedding[bucket(j as isize - i as isize) * 64 + head];
                }
            }
        }
        let h = attention_online_with_bias(
            &q,
            &k,
            &v,
            &bias,
            AttentionSpec {
                query_tokens: n,
                key_tokens: n,
                query_heads: 64,
                key_value_heads: 64,
                head_dim: 64,
                causal: false,
                scale: 1.,
            },
        )?;
        let h = linear(g, ex, &format!("{p}.attn.o"), &h)?;
        residual(&mut x, &h, None);
        let h = rms_norm_rows(&x, n, &g.floats[&format!("{p}.norm2.weight")], 1e-6)?;
        let [mut gate, up] = Linear::forward_group(
            [
                &g.linear[&format!("{p}.ffn.gate.0")],
                &g.linear[&format!("{p}.ffn.fc1")],
            ],
            ex,
            &h,
            n,
        )?;
        gelu_approx_inplace(&mut gate);
        for (a, b) in gate.iter_mut().zip(up) {
            *a *= b;
        }
        let h = linear(g, ex, &format!("{p}.ffn.fc2"), &gate)?;
        residual(&mut x, &h, None);
        trace(&format!("fastframes.text.{layer}"), &x);
    }
    let mut x = rms_norm_rows(&x, n, &g.floats["text.norm.weight"], 1e-6)?;
    x.resize(512 * D, 0.);
    trace("fastframes.text", &x);
    Ok(x)
}

pub(super) fn bucket(relative: isize) -> usize {
    let n = relative.unsigned_abs();
    let sign = if relative > 0 { 16 } else { 0 };
    sign + if n < 8 {
        n
    } else {
        (8 + ((n as f32 / 8.).ln() / 16_f32.ln() * 8.) as usize).min(15)
    }
}

fn unescape(text: &str) -> String {
    static ENTITIES: OnceLock<HashMap<String, String>> = OnceLock::new();
    let entities = ENTITIES.get_or_init(|| {
        serde_json::from_str(include_str!("html_entities.json")).expect("HTML5 entities")
    });
    let mut out = String::new();
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp + 1..];
        if let Some(number) = rest.strip_prefix('#') {
            let (radix, number, prefix) = if let Some(n) = number.strip_prefix(['x', 'X']) {
                (16, n, 2)
            } else {
                (10, number, 1)
            };
            let len = number
                .bytes()
                .take_while(|b| b.is_ascii_digit() || (radix == 16 && b.is_ascii_hexdigit()))
                .count();
            if len > 0 {
                let value = u32::from_str_radix(&number[..len], radix).unwrap_or(0xFFFD);
                out.push_str(&html_numeric(value));
                let count = prefix + len + usize::from(number[len..].starts_with(';'));
                rest = &rest[count..];
                continue;
            }
        }
        let len = rest
            .bytes()
            .take(33)
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b';')
            .count();
        let found = (1..=len)
            .rev()
            .find_map(|n| entities.get(&rest[..n]).map(|s| (n, s)));
        if let Some((n, s)) = found {
            out.push_str(s);
            rest = &rest[n..];
        } else {
            out.push('&');
        }
    }
    out.push_str(rest);
    out
}

const CP1252: [char; 32] = [
    '€', '\u{81}', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u{8D}', 'Ž', '\u{8F}',
    '\u{90}', '‘', '’', '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '\u{9D}', 'ž', 'Ÿ',
];

fn cp1252_byte(c: char) -> Option<u8> {
    u8::try_from(c as u32)
        .ok()
        .or_else(|| CP1252.iter().position(|&x| x == c).map(|i| 0x80 + i as u8))
}

fn repair_encoding(text: &str) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let first = cp1252_byte(chars[i]);
        let count = match first {
            Some(0xC2..=0xDF) => 2,
            Some(0xE0..=0xEF) => 3,
            Some(0xF0..=0xF4) => 4,
            _ => 0,
        };
        if count > 0 && i + count <= chars.len() {
            let bytes = chars[i..i + count]
                .iter()
                .map(|&c| cp1252_byte(c))
                .collect::<Option<Vec<_>>>();
            if let Some(bytes) = bytes {
                if let Ok(value) = std::str::from_utf8(&bytes) {
                    out.push_str(value);
                    i += count;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn html_numeric(value: u32) -> String {
    if (0x80..=0x9F).contains(&value) {
        return CP1252[(value - 0x80) as usize].to_string();
    }
    if matches!(value,1..=8|11|14..=31|127)
        || (0xFDD0..=0xFDEF).contains(&value)
        || (value <= 0x10FFFF && value & 0xFFFF >= 0xFFFE)
    {
        return String::new();
    }
    char::from_u32(value)
        .filter(|&c| c != '\0')
        .unwrap_or('\u{FFFD}')
        .to_string()
}

pub(super) fn clean(text: &str) -> String {
    let mut text = text.to_owned();
    // ftfy's automatic HTML repair is disabled when the original text contains markup.
    if !text.contains('<') {
        loop {
            let decoded = unescape(&text);
            if decoded == text {
                break;
            }
            text = decoded;
        }
    }
    for _ in 0..3 {
        let repaired = repair_encoding(&text);
        if repaired == text {
            break;
        }
        text = repaired;
    }
    static ANSI: OnceLock<regex::Regex> = OnceLock::new();
    let text = ANSI
        .get_or_init(|| regex::Regex::new(r"\x1b\[[0-9;]*[A-Za-z]").unwrap())
        .replace_all(&text, "");
    let mut normalized = String::new();
    for c in text.chars() {
        match c {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => normalized.push('\''),
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => normalized.push('"'),
            '\u{FF00}'..='\u{FFEF}' => normalized.extend(c.to_string().nfkc()),
            '\u{FB00}' => normalized.push_str("ff"),
            '\u{FB01}' => normalized.push_str("fi"),
            '\u{FB02}' => normalized.push_str("fl"),
            '\u{FB03}' => normalized.push_str("ffi"),
            '\u{FB04}' => normalized.push_str("ffl"),
            '\u{FB05}' | '\u{FB06}' => normalized.push_str("st"),
            '\u{80}'..='\u{9F}' => normalized.push(CP1252[(c as u32 - 0x80) as usize]),
            c if c.is_control() && !c.is_whitespace() => {}
            c => normalized.push(c),
        }
    }
    unescape(&unescape(&normalized))
        .nfc()
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
