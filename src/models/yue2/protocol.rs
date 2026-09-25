use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};

pub const EOD: u32 = 151_643;
pub const ABC_START: u32 = 151_847;
pub const ABC_END: u32 = 151_848;
pub const MUSIC_START: u32 = 151_851;
pub const MUSIC_END: u32 = 151_852;
pub const CODEC_OFFSET: u32 = 151_853;
pub const CODEC_SIZE: usize = 32_768;
pub const LATENT_START: u32 = 184_621;
pub const LATENT_END: u32 = 184_622;
pub const LATENT_PAD: u32 = 184_623;
pub const VOCAB_SIZE: usize = 184_704;
pub const CONTEXT: usize = 24_576;
pub const PROTOCOL_VERSION: &str = "yue2-native-v1";

const FULL_COT_INSTRUCTION: &str =
    "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub penalty_window: usize,
    pub min_tokens: usize,
    pub max_tokens: usize,
}

impl SamplingConfig {
    pub const fn abc() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 30,
            repetition_penalty: 1.005,
            penalty_window: 100,
            min_tokens: 32,
            max_tokens: 4096,
        }
    }

    pub const fn semantic() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 100,
            repetition_penalty: 1.2,
            penalty_window: 50,
            min_tokens: 200,
            max_tokens: 9000,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.temperature.is_finite()
            || !self.top_p.is_finite()
            || !self.repetition_penalty.is_finite()
        {
            return Err("YuE2 sampling numbers must be finite".into());
        }
        if !(0.0..=5.0).contains(&self.temperature)
            || !(0.0 < self.top_p && self.top_p <= 1.0)
            || self.top_k == 0
        {
            return Err("Invalid YuE2 sampling temperature/top_p/top_k".into());
        }
        if self.repetition_penalty <= 0.0 || !(1..=100).contains(&self.penalty_window) {
            return Err("Invalid YuE2 repetition penalty/window".into());
        }
        if self.min_tokens > self.max_tokens || self.max_tokens == 0 {
            return Err("YuE2 sampling requires 0 <= min_tokens <= max_tokens".into());
        }
        Ok(())
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self::semantic()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YuE2Request {
    pub style: String,
    pub lyrics: String,
    pub seed: u64,
}

impl YuE2Request {
    pub fn new(
        style: impl Into<String>,
        lyrics: impl Into<String>,
        seed: u64,
    ) -> Result<Self, String> {
        let style = style.into();
        let lyrics = lyrics.into();
        if style.trim().is_empty() {
            return Err("YuE2 style must not be empty".into());
        }
        if lyrics.trim().is_empty() {
            return Err("YuE2 lyrics must not be empty".into());
        }
        Ok(Self {
            style,
            lyrics,
            seed,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct YuE2Protocol {
    pub abc: SamplingConfig,
    pub semantic: SamplingConfig,
}

impl YuE2Protocol {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        require_string(source, "general.architecture", "yue2")?;
        require_string(source, "yue2.protocol_version", PROTOCOL_VERSION)?;
        for (key, expected) in [
            ("yue2.context_length", CONTEXT as u64),
            ("yue2.vocab_size", VOCAB_SIZE as u64),
            ("yue2.eod_token_id", EOD as u64),
            ("yue2.abc_start_token_id", ABC_START as u64),
            ("yue2.abc_end_token_id", ABC_END as u64),
            ("yue2.music_start_token_id", MUSIC_START as u64),
            ("yue2.music_end_token_id", MUSIC_END as u64),
            ("yue2.codec_offset", CODEC_OFFSET as u64),
            ("yue2.codec_size", CODEC_SIZE as u64),
            ("yue2.latent_start_token_id", LATENT_START as u64),
            ("yue2.latent_end_token_id", LATENT_END as u64),
            ("yue2.latent_pad_token_id", LATENT_PAD as u64),
            ("yue2.abc.top_k", 30),
            ("yue2.abc.penalty_window", 100),
            ("yue2.abc.min_tokens", 32),
            ("yue2.abc.max_tokens", 4096),
            ("yue2.semantic.top_k", 100),
            ("yue2.semantic.penalty_window", 50),
            ("yue2.semantic.min_tokens", 200),
            ("yue2.semantic.max_tokens", 9000),
        ] {
            require_u64(source, key, expected)?;
        }
        for (key, expected) in [
            ("yue2.abc.temperature", 0.7),
            ("yue2.abc.top_p", 0.9),
            ("yue2.abc.repetition_penalty", 1.005),
            ("yue2.semantic.temperature", 1.0),
            ("yue2.semantic.top_p", 0.95),
            ("yue2.semantic.repetition_penalty", 1.2),
        ] {
            require_f64(source, key, expected)?;
        }
        let protocol = Self {
            abc: SamplingConfig::abc(),
            semantic: SamplingConfig::semantic(),
        };
        protocol.abc.validate()?;
        protocol.semantic.validate()?;
        Ok(protocol)
    }

    pub fn prompt_text(&self, request: &YuE2Request) -> String {
        format!(
            "{FULL_COT_INSTRUCTION}\n[Tags]\n{}\n[Lyrics]\n{}\n",
            request.style, request.lyrics
        )
    }

    pub fn abc_prefix(
        &self,
        tokenizer: &BPETokenizer,
        request: &YuE2Request,
    ) -> Result<Vec<u32>, String> {
        let mut prefix = vec![EOD];
        prefix.extend(tokenizer.encode(
            &self.prompt_text(request),
            EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        ));
        prefix.push(ABC_START);
        Ok(prefix)
    }

    pub fn semantic_prefix(
        &self,
        tokenizer: &BPETokenizer,
        request: &YuE2Request,
        abc_ids: &[u32],
    ) -> Result<Vec<u32>, String> {
        if abc_ids.iter().any(|&token| token >= EOD) {
            return Err("YuE2 ABC IDs must remain inside the ordinary text vocabulary".into());
        }
        let mut prefix = self.abc_prefix(tokenizer, request)?;
        prefix.extend_from_slice(abc_ids);
        prefix.extend([ABC_END, MUSIC_START]);
        Ok(prefix)
    }
}

fn require_string(source: &dyn TensorSource, key: &str, expected: &str) -> Result<(), String> {
    match source.metadata(key) {
        Some(MetaValue::String(value)) if value == expected => Ok(()),
        Some(value) => Err(format!(
            "Invalid {key}: expected {expected:?}, got {value:?}"
        )),
        None => Err(format!("Missing {key}: expected {expected:?}")),
    }
}

fn require_u64(source: &dyn TensorSource, key: &str, expected: u64) -> Result<(), String> {
    match source.metadata(key).and_then(MetaValue::to_u64) {
        Some(value) if value == expected => Ok(()),
        Some(value) => Err(format!("Invalid {key}: expected {expected}, got {value}")),
        None => Err(format!("Missing or invalid {key}: expected {expected}")),
    }
}

fn require_f64(source: &dyn TensorSource, key: &str, expected: f64) -> Result<(), String> {
    let actual = match source.metadata(key) {
        Some(MetaValue::Float32(value)) => Some(f64::from(*value)),
        Some(MetaValue::Float64(value)) => Some(*value),
        _ => None,
    };
    match actual {
        Some(value) if value == expected => Ok(()),
        Some(value) => Err(format!("Invalid {key}: expected {expected}, got {value}")),
        None => Err(format!("Missing or invalid {key}: expected {expected}")),
    }
}
