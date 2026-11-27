//! `RegexValidator` — the schema-level span filter.
//!
//! `gliner2/inference/schema.py:27-55` is a four-field post-processing filter
//! over one span's surface text:
//!
//! ```text
//! pattern: str
//! mode:    "full" | "partial"   (default "full")
//! exclude: bool = False
//! flags:   int  = re.IGNORECASE  (default case-insensitive)
//!
//! validate(text) = (fullmatch | search)(text) is not None, negated if exclude
//! ```
//!
//! A span whose surface fails every check is dropped before it reaches the
//! output, so this changes which entities a schema reports rather than how they
//! are scored.
//!
//! # Where the reference exposes it
//!
//! Only through its Python builder (`schema.field(..., validators=[...])`). It is
//! not reachable from JSON: `to_dict` does not serialize it, `from_dict` does
//! not forward it, `schema_model`'s pydantic layer has no such field, and the
//! HTTP client warns and drops it (`api_client.py:90-97`). So the JSON shape
//! below is this port's own, named after the dataclass fields, and it is a
//! superset of nothing — there is no wire format to match.
//!
//! # Engine differences from Python's `re`
//!
//! These are measured, not assumed; `tools/oracle/gliner_boundary/
//! dump_regex_validator.py` records the reference's answers for each.
//!
//! - **Anchoring.** `fullmatch` is the whole string. `\A(?:…)\z` is used rather
//!   than `^(?:…)$` because `$` is engine-relative in some dialects and `\A`/`\z`
//!   are not. Both engines reject `"a\n"` for `a$`, so the two agree here.
//! - **Case folding: the Turkish dotless and dotted i.** Python's
//!   `re.IGNORECASE` matches `İ` (U+0130) and `ı` (U+0131) against a pattern
//!   `i`; Rust's `(?i)` does not, because Unicode simple case folding treats them
//!   as distinct letters. Rust is right and Python is the outlier, but parity is
//!   parity, so this is a known gap rather than a silent one — see
//!   `IGNORECASE_DIVERGENCES` and the test that asserts it.
//! - **`\w` in a user-written pattern.** Python's `\w` is `L* + N* + _` and
//!   rejects a combining mark; the `regex` crate's `\w` includes `\p{M}`, so
//!   `\w+` matches `"a" + U+0301` here and not in Python. The word splitter
//!   spells its own class out for exactly this reason; rewriting a *user's*
//!   pattern would need a real parser, so this port does not and the gap is
//!   documented instead.
//!
//! Kelvin sign (U+212A), long s (U+017F) and the Angstrom sign all agree, so
//! only the two cases above diverge.

use std::collections::BTreeMap;

use regex::Regex;
use serde::{Deserialize, Serialize};

/// Characters on which Python's `re.IGNORECASE` and Rust's `(?i)` disagree
/// against an ASCII letter, as `(letter, extra case variants)`.
///
/// Recorded rather than worked around: closing it would mean rewriting the
/// user's pattern, which needs a parser this port does not have. A validator
/// that has to be exact for Turkish text should spell the variants out in its
/// own pattern.
pub const IGNORECASE_DIVERGENCES: [(&str, [char; 2]); 1] = [("i", ['\u{0130}', '\u{131}'])];

/// Whether the pattern must match the whole surface or merely appear in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ValidatorMode {
    /// `re.fullmatch`.
    #[default]
    Full,
    /// `re.search`.
    Partial,
}

/// One `RegexValidator`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegexValidator {
    /// The pattern, in this crate's dialect (`regex`), not Python's `re`.
    pub pattern: String,
    #[serde(default)]
    pub mode: ValidatorMode,
    #[serde(default)]
    pub exclude: bool,
    /// Defaults to `true`, matching the reference's `flags = re.IGNORECASE`.
    #[serde(default = "default_true")]
    pub ignore_case: bool,
    /// Equivalent to `re.DOTALL`. Not a reference field; the reference exposes
    /// arbitrary `re` flags as an integer, and this port spells the one that
    /// changes matching out.
    #[serde(default)]
    pub dot_all: bool,
}

fn default_true() -> bool {
    true
}

impl RegexValidator {
    /// Compile the pattern, reporting the reference's construction-time errors.
    ///
    /// `RegexValidator.__post_init__` raises `ValueError` for a mode outside its
    /// two values and for an uncompilable pattern, so neither can be reached
    /// from a decode — the port rejects them where the reference does.
    pub fn compile(&self) -> Result<Regex, String> {
        if self.pattern.is_empty() {
            return Err("validator pattern must not be empty".into());
        }
        let mut body = String::with_capacity(self.pattern.len() + 16);
        // `(?i)`/`(?s)` are only emitted when requested: a bare `(?)\A…\z` is
        // not a valid pattern, so a case-sensitive validator would fail to
        // compile at all.
        if self.ignore_case || self.dot_all {
            body.push_str("(?");
            if self.ignore_case {
                body.push('i');
            }
            if self.dot_all {
                body.push('s');
            }
            body.push(')');
        }
        let pattern = &self.pattern;
        let body = match self.mode {
            // `\A`/`\z` are absolute in both engines; `^`/`$` are not, which is
            // why the reference's `$`-before-a-trailing-newline behaviour is not
            // something to imitate.
            ValidatorMode::Full => format!("{body}\\A(?:{pattern})\\z"),
            ValidatorMode::Partial => format!("{body}(?:{pattern})"),
        };
        Regex::new(&body).map_err(|_| format!("Invalid regex: {:?}", self.pattern))
    }

    /// `validate(text)`.
    pub fn validate(&self, text: &str) -> Result<bool, String> {
        let regex = self.compile()?;
        Ok(self.validate_with(&regex, text))
    }

    /// `validate` against an already-compiled pattern, so a schema with many
    /// validators on one span does not recompile per span.
    pub fn validate_with(&self, regex: &Regex, text: &str) -> bool {
        // `is_match` on the anchored form already encodes `full` vs `search`.
        regex.is_match(text) != self.exclude
    }
}

/// A schema's validators, compiled once.
///
/// `all(...)` over the validator list is the reference's rule at all three call
/// sites (`engine.py:293`, `:577`, `:1114`), so an empty list admits everything
/// and a span must satisfy every validator to survive.
#[derive(Clone, Debug, Default)]
pub struct CompiledValidators {
    entries: Vec<(RegexValidator, Regex)>,
}

impl CompiledValidators {
    /// Compile a validator list. `None` and `[]` both mean "admit everything",
    /// which is what the reference's `if validators and not all(...)` does.
    pub fn compile(list: Option<&[RegexValidator]>) -> Result<Self, String> {
        let Some(list) = list else {
            return Ok(Self::default());
        };
        let mut entries = Vec::with_capacity(list.len());
        for validator in list {
            entries.push((validator.clone(), validator.compile()?));
        }
        Ok(Self { entries })
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `text` survives every validator.
    pub fn accepts(&self, text: &str) -> bool {
        self.entries
            .iter()
            .all(|(validator, regex)| validator.validate_with(regex, text))
    }
}

/// Read a `validators` list out of a schema metadata entry.
///
/// The reference stores live `RegexValidator` objects (`schema.py:201`), which
/// JSON cannot carry; this is the port's own spelling, named after the
/// dataclass fields. An entry that is not an object, or whose `pattern` is
/// missing, is an error rather than a silently ignored filter — a validator that
/// does not apply is a span that was never filtered.
pub fn parse_validators(
    metadata: Option<&serde_json::Value>,
) -> Result<Vec<RegexValidator>, String> {
    let Some(list) = metadata.and_then(|value| value.get("validators")) else {
        return Ok(Vec::new());
    };
    let list = list
        .as_array()
        .ok_or_else(|| "validators must be an array".to_string())?;
    list.iter()
        .map(|entry| {
            serde_json::from_value::<RegexValidator>(entry.clone())
                .map_err(|error| format!("invalid validator entry: {error}"))
        })
        .collect()
}

/// Every `entity_metadata` / `field_metadata` entry's validators, keyed the same
/// way the thresholds are, so one lookup serves both features.
pub fn parse_metadata_validators(
    entity_metadata: Option<&serde_json::Value>,
    field_metadata: Option<&serde_json::Value>,
    entities: impl Iterator<Item = String>,
    structures: impl Iterator<Item = (String, String)>,
) -> Result<BTreeMap<String, CompiledValidators>, String> {
    let mut out = BTreeMap::new();
    for name in entities {
        let validators = parse_validators(entity_metadata.and_then(|value| value.get(&name)))?;
        if !validators.is_empty() {
            out.insert(
                name.clone(),
                CompiledValidators::compile(Some(&validators))?,
            );
        }
    }
    for (group, field) in structures {
        let key = format!("{group}.{field}");
        let validators = parse_validators(field_metadata.and_then(|value| value.get(&key)))?;
        if !validators.is_empty() {
            out.insert(key, CompiledValidators::compile(Some(&validators))?);
        }
    }
    Ok(out)
}
