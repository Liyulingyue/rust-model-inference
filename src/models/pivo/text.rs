use regex::{Captures, Regex};
use std::collections::HashMap;
use std::sync::OnceLock;
use tokenizers::{NormalizedString, Normalizer};

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "zero", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap()
}

pub(super) fn symbols() -> &'static [String] {
    static SYMBOLS: OnceLock<Vec<String>> = OnceLock::new();
    SYMBOLS.get_or_init(|| {
        let mut values: Vec<String> = "<pad> <bos> <eos> <space> . , ? ! ; : B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split_whitespace().map(str::to_owned).collect();
        for vowel in "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split_whitespace() {
            for stress in 0..3 { values.push(format!("{vowel}{stress}")); }
        }
        values
    })
}

fn cardinal(raw: &str) -> String {
    let stripped = raw.trim_start_matches('0');
    let raw = if stripped.is_empty() { "0" } else { stripped };
    if raw.len() >= 16 {
        return raw
            .bytes()
            .map(|c| ONES[(c - b'0') as usize])
            .collect::<Vec<_>>()
            .join(" ");
    }
    let n: u64 = raw.parse().unwrap();
    if n < 20 {
        return ONES[n as usize].into();
    }
    if n < 100 {
        return format!(
            "{}{}",
            TENS[(n / 10) as usize],
            if n % 10 == 0 {
                String::new()
            } else {
                format!(" {}", ONES[(n % 10) as usize])
            }
        );
    }
    for (scale, label) in [
        (1_000_000_000_000, "trillion"),
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1000, "thousand"),
        (100, "hundred"),
    ] {
        if n >= scale {
            let suffix = if n % scale == 0 {
                String::new()
            } else {
                format!(" {}", cardinal(&(n % scale).to_string()))
            };
            return format!("{} {label}{suffix}", cardinal(&(n / scale).to_string()));
        }
    }
    unreachable!()
}

fn ordinal(raw: &str) -> String {
    let words = cardinal(raw);
    let (prefix, last) = words
        .rsplit_once(' ')
        .map_or(("", words.as_str()), |(a, b)| (a, b));
    let last = match last {
        "one" => "first".into(),
        "two" => "second".into(),
        "three" => "third".into(),
        "five" => "fifth".into(),
        "eight" => "eighth".into(),
        "nine" => "ninth".into(),
        "twelve" => "twelfth".into(),
        value if value.ends_with('y') => format!("{}ieth", &value[..value.len() - 1]),
        value => format!("{value}th"),
    };
    if prefix.is_empty() {
        last
    } else {
        format!("{prefix} {last}")
    }
}

pub(super) fn normalize(input: &str) -> Result<String, String> {
    let mut normalized = NormalizedString::from(input);
    tokenizers::normalizers::unicode::NFKC
        .normalize(&mut normalized)
        .map_err(|e| e.to_string())?;
    let mut text: String = normalized
        .get()
        .chars()
        .map(|c| match c {
            '’' | '‘' => '\'',
            '“' | '”' => '"',
            '—' | '–' => ',',
            c => c,
        })
        .collect();
    let digits = re(r"\d");
    text = digits
        .replace_all(&text, |c: &Captures| {
            let value = c[0].chars().next().unwrap() as u32;
            let mut start = value;
            // Unicode decimal runs contain whole 0..9 sets, sometimes adjacent sets.
            while char::from_u32(start - 1).is_some_and(|v| digits.is_match(&v.to_string())) {
                start -= 1;
            }
            char::from(b'0' + ((value - start) % 10) as u8).to_string()
        })
        .into_owned();
    let months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let mut bad_date = false;
    text = re(r"\b([0-9]{4})-([0-9]{2})-([0-9]{2})\b")
        .replace_all(&text, |c: &Captures| {
            let (year, month, day): (u32, usize, u32) = (
                c[1].parse().unwrap(),
                c[2].parse().unwrap(),
                c[3].parse().unwrap(),
            );
            let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
            let days = [
                31,
                if leap { 29 } else { 28 },
                31,
                30,
                31,
                30,
                31,
                31,
                30,
                31,
                30,
                31,
            ];
            if year == 0 || !(1..=12).contains(&month) || day == 0 || day > days[month - 1] {
                bad_date = true;
                return c[0].to_string();
            }
            format!(
                "{} {}, {}",
                months[month - 1],
                ordinal(&c[3]),
                cardinal(&c[1])
            )
        })
        .into_owned();
    if bad_date {
        return Err("Invalid PiVo ISO date".into());
    }
    if re(r"[0-9]/[0-9]").is_match(&text) {
        return Err("Write slash dates/fractions in words, or use an ISO date".into());
    }
    text = re(r"(?i)\b(Mr|Mrs|Dr|Prof|St|vs|etc)\.(\s|$)")
        .replace_all(&text, |c: &Captures| {
            let word = match c[1].to_ascii_lowercase().as_str() {
                "mr" => "Mister",
                "mrs" => "Missus",
                "dr" => "Doctor",
                "prof" => "Professor",
                "st" => "Saint",
                "vs" => "versus",
                _ => "et cetera",
            };
            format!("{word}{}", &c[2])
        })
        .into_owned();
    if re(r"[$£€][0-9][0-9,]*\.[0-9]{3,}").is_match(&text) {
        return Err("Currency amounts must have at most two decimal places".into());
    }
    text = re(r"([$£€])([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
        .replace_all(&text, |c: &Captures| {
            let raw = c[2].replace(',', "");
            let (whole, fraction) = raw.split_once('.').unwrap_or((&raw, ""));
            let one = whole.trim_start_matches('0') == "1";
            let unit = match &c[1] {
                "$" => "dollar",
                "£" => "pound",
                _ => "euro",
            };
            let mut result = format!("{} {unit}{}", cardinal(whole), if one { "" } else { "s" });
            let cents: usize = format!("{fraction:0<2}").parse().unwrap();
            if cents > 0 {
                let unit = match (&c[1], cents) {
                    ("£", 1) => "penny",
                    ("£", _) => "pence",
                    (_, 1) => "cent",
                    _ => "cents",
                };
                result.push_str(&format!(" and {} {unit}", cardinal(&cents.to_string())));
            }
            result
        })
        .into_owned();
    text = re(r"(?i)\b([0-9]+)(?:st|nd|rd|th)\b")
        .replace_all(&text, |c: &Captures| ordinal(&c[1]))
        .into_owned();
    text = re(r"\b[0-9]{1,3}(?:,[0-9]{3})+\b")
        .replace_all(&text, |c: &Captures| c[0].replace(',', ""))
        .into_owned();
    text = re(r"-?(?:[0-9]+\.[0-9]+|[0-9]+)")
        .replace_all(&text, |c: &Captures| {
            let value = c[0].trim_start_matches('-');
            let mut result = if c[0].starts_with('-') {
                "minus ".to_string()
            } else {
                String::new()
            };
            if let Some((whole, fraction)) = value.split_once('.') {
                result.push_str(&format!(
                    "{} point {}",
                    cardinal(whole),
                    fraction
                        .bytes()
                        .map(|b| ONES[(b - b'0') as usize])
                        .collect::<Vec<_>>()
                        .join(" ")
                ));
            } else {
                result.push_str(&cardinal(value));
            }
            result
        })
        .into_owned();
    text = text
        .replace('%', " percent ")
        .replace('&', " and ")
        .replace('+', " plus ");
    text = re(r"\bNASA\b").replace_all(&text, "nasa").into_owned();
    text = re(r"\b[A-Z]{2,5}\b")
        .replace_all(&text, |c: &Captures| {
            c[0].chars()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .into_owned();
    text = text
        .chars()
        .map(|c| if "-()[]{}\"/".contains(c) { ' ' } else { c })
        .collect();
    if text
        .chars()
        .any(|c| !c.is_ascii_alphabetic() && !c.is_whitespace() && !"'.,?!;:".contains(c))
    {
        return Err("Unsupported text characters after PiVo normalization".into());
    }
    Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
}

pub(super) struct Frontend {
    lexicon: HashMap<String, Vec<usize>>,
}
pub(super) struct Front {
    pub normalized: String,
    pub tokens: Vec<usize>,
}

impl Frontend {
    pub fn new(raw: &str) -> Result<Self, String> {
        let mut lexicon = HashMap::new();
        for line in raw.lines() {
            let mut parts = line.split('#').next().unwrap().split_whitespace();
            let Some(word) = parts.next() else {
                continue;
            };
            let word = word.split('(').next().unwrap().to_lowercase();
            if lexicon.contains_key(&word) {
                continue;
            }
            let phones = parts
                .map(|p| {
                    symbols()
                        .iter()
                        .position(|s| s == p)
                        .ok_or_else(|| format!("Invalid PiVo lexicon phone {p}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if !phones.is_empty() {
                lexicon.insert(word, phones);
            }
        }
        if lexicon.is_empty() {
            return Err("Empty PiVo pronunciation dictionary".into());
        }
        Ok(Self { lexicon })
    }

    pub fn phonemize(
        &self,
        input: &str,
        overrides: &HashMap<String, Vec<String>>,
    ) -> Result<Front, String> {
        if input.trim().is_empty() || input.chars().count() > 2000 {
            return Err("PiVo requires spoken text of at most 2000 characters".into());
        }
        if overrides.len() > 128
            || overrides
                .iter()
                .any(|(w, p)| w.chars().count() > 64 || p.len() > 64)
        {
            return Err("Invalid PiVo pronunciation overrides".into());
        }
        let mut lowered = HashMap::new();
        for (word, phones) in overrides {
            if lowered.insert(word.to_lowercase(), phones).is_some() {
                return Err(format!("Duplicate case-insensitive PiVo override: {word}"));
            }
        }
        let overrides = lowered;
        let normalized = normalize(input)?;
        let mut tokens = vec![1];
        for m in re(r"[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]").find_iter(&normalized) {
            let item = m.as_str();
            if let Some(id) = symbols()[4..10].iter().position(|s| s == item) {
                tokens.push(id + 4);
                continue;
            }
            let word = item.to_ascii_lowercase();
            let phones = if let Some(phones) = overrides.get(&word) {
                phones
                    .iter()
                    .map(|p| {
                        symbols()
                            .iter()
                            .position(|s| s == p)
                            .ok_or_else(|| format!("Unknown ARPAbet phone {p} in {word}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                self.lexicon
                    .get(&word)
                    .ok_or_else(|| {
                        format!("Out-of-vocabulary word requires a pronunciation override: {word}")
                    })?
                    .clone()
            };
            if !matches!(tokens.last(), Some(1 | 3)) {
                tokens.push(3);
            }
            tokens.extend(phones);
        }
        tokens.push(2);
        if !tokens.iter().any(|&id| id >= 10) {
            return Err("PiVo input must contain spoken text".into());
        }
        if tokens.len() > 512 {
            return Err("PiVo input exceeds 512 frontend tokens; split it into clauses".into());
        }
        Ok(Front { normalized, tokens })
    }
}
