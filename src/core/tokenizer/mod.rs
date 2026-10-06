use std::collections::HashMap;
use std::ops::Range;

use unicode_categories::UnicodeCategories;

use crate::core::tensor::{MetaValue, MetaValueType};

pub mod ugm;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeOptions {
    pub add_special: bool,
    pub parse_special: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            add_special: false,
            parse_special: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreTokenizer {
    Qwen2,
    Qwen35,
    HunyuanDense,
    Lfm2,
    LlamaBpe,
    Gemma4,
    K2Horizon,
    /// Xunfei Spark 2.5 (and other Chinese-tokenizer dialects that share
    /// the same byte-level BPE setup but use a different regex split).
    /// Differs from `Qwen2` in that punctuation immediately preceding a
    /// letter (`!hello`) is split off as its own token.
    Spark2_5,
    /// MiniCPM5 (openbmb/MiniCPM5-1B, MiniCPM5-2.6B). GPT-2 style BPE
    /// (`tokenizer.ggml.model = "gpt2"`) with a pre-split regex that
    /// chunks numbers into 1-3 digit runs before the standard GPT-2
    /// split. Mirrors llama.cpp's `LLAMA_VOCAB_PRE_TYPE_MINICPM5`.
    Minicpm5,
    /// GPT-4o / Phi-4 / tiktoken-cl100k style. Like LlamaBpe but the
    /// word regex separates uppercase and lowercase runs (`iPhone` ->
    /// `i` + `Phone`) and digits are chunked 1-3 at a time. Mirrors
    /// llama.cpp's `LLAMA_VOCAB_PRE_TYPE_GPT4O`.
    Gpt4o,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

#[derive(Debug, Clone)]
struct SpecialToken {
    text: String,
    id: u32,
    kind: TokenType,
}

#[derive(Debug)]
pub struct BPETokenizer {
    tokens: Vec<String>,
    token_types: Vec<TokenType>,
    token_to_id: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), u32>,
    byte_encoder: Vec<String>,
    byte_decoder: HashMap<char, u8>,
    pre: PreTokenizer,
    special_tokens: Vec<SpecialToken>,
    semantic_tokens: HashMap<String, u32>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    byte_fallback: bool,
    normalize_nfc: bool,
}

const QWEN_SEMANTIC_TOKENS: &[(&str, &str)] = &[
    ("<|im_start|>", "im_start"),
    ("<|im_end|>", "im_end"),
    ("<|image_pad|>", "image_pad"),
    ("<|video_pad|>", "video_pad"),
    ("<|vision_pad|>", "vision_pad"),
    ("<|vision_start|>", "vision_start"),
    ("<|vision_end|>", "vision_end"),
    ("<|vision_bos|>", "vision_start"),
    ("<|vision_eos|>", "vision_end"),
    ("<|audio_start|>", "audio_start"),
    ("<|audio_pad|>", "audio_pad"),
    ("<|audio_end|>", "audio_end"),
    ("<|audio_bos|>", "audio_start"),
    ("<|audio_eos|>", "audio_end"),
    ("<|IMAGE|>", "image_pad"),
    ("<|VIDEO|>", "video_pad"),
    ("<|AUDIO|>", "audio_pad"),
    ("<asr_text>", "asr_text"),
    ("<|endoftext|>", "endoftext"),
];

const QWEN2_ORACLE_SPECIAL_TOKENS: &[&str] = &[
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<|boi_token|>",
    "<|bor_token|>",
    "<|eor_token|>",
    "<|bot_token|>",
    "<|tms_token|>",
];

/// Qwen2.5-Omni-3B special tokens at ids 151643..151664. Replaces the
/// standard Qwen2 oracle list and adds the audio-related tokens
/// (`<|AUDIO|>` = 151646, `<|audio_bos|>` = 151647, `<|audio_eos|>` = 151648)
/// that are required for AuK CFMEdit reference-audio conditioning.
///
/// Verified against the Qwen/Qwen2.5-Omni-3B `added_tokens.json`. The
/// named vocab size shrinks from 32 to 22 tokens because the Qwen2.5-Omni
/// tokenizer drops 10 of the standard Qwen2 oracle tokens (no
/// `<|object_ref_*|>`, no `<|image_pad|>`/`<|video_pad|>`, no
/// `<tool_response>`, no `<|boi_token|>` etc.).
const QWEN25_OMNI_SPECIAL_TOKENS: &[&str] = &[
    "<|endoftext|>",  // 151643
    "<|im_start|>",   // 151644
    "<|im_end|>",     // 151645
    "<|AUDIO|>",      // 151646  (CFMEdit audio placeholder)
    "<|audio_bos|>",  // 151647  (CFMEdit audio begin-of-stream)
    "<|audio_eos|>",  // 151648  (CFMEdit audio end-of-stream)
    "<|box_end|>",    // 151649
    "<|quad_start|>", // 151650
    "<|quad_end|>",   // 151651
    "<|vision_bos|>", // 151652
    "<|vision_eos|>", // 151653
    "<|vision_pad|>", // 151654
    "<|IMAGE|>",      // 151655
    "<|VIDEO|>",      // 151656
    "<tool_call>",    // 151657
    "</tool_call>",   // 151658
    "<|fim_prefix|>", // 151659
    "<|fim_middle|>", // 151660
    "<|fim_suffix|>", // 151661
    "<|fim_pad|>",    // 151662
    "<|repo_name|>",  // 151663
    "<|file_sep|>",   // 151664
];

const HUNYUAN_SEMANTIC_TOKENS: &[(&str, &str)] = &[
    ("<｜hy_begin▁of▁sentence｜>", "hy_begin"),
    ("<｜hy_User｜>", "hy_user"),
    ("<｜hy_Assistant｜>", "hy_assistant"),
    ("<｜hy_place▁holder▁no▁2｜>", "hy_placeholder_2"),
    ("<｜hy_place▁holder▁no▁3｜>", "hy_placeholder_3"),
    ("<｜hy_place▁holder▁no▁8｜>", "hy_placeholder_8"),
];

const LLAMA_BPE_SEMANTIC_TOKENS: &[(&str, &str)] = &[
    ("<s>", "bos_token"),
    ("</s>", "eos_token"),
    ("<|im_start|>", "im_start"),
    ("<|im_end|>", "im_end"),
    ("<|im_sep|>", "im_sep"),
    ("<|thought_begin|>", "think_start"),
    ("<|thought_end|>", "think_end"),
];

const K2_HORIZON_SEMANTIC_TOKENS: &[(&str, &str)] = &[
    // Chat turn delimiters (mirror Qwen's <|im_start|> / <|im_end|>).
    ("<|ifm|im_start|>", "ifm|im_start"),
    ("<|ifm|im_end|>", "ifm|im_end"),
    // System segment delimiters.
    ("<|sys_start|>", "ifm|sys_start"),
    ("<|sys_end|>", "ifm|sys_end"),
    // Reasoning block tags. The chat template renders the open tag at
    // the assistant turn start and the model is expected to close with
    // the matching end tag.
    ("<|begin_of_thought|>", "ifm|begin_of_thought"),
    ("<|end_of_thought|>", "ifm|end_of_thought"),
    ("<ifm|think>", "ifm|think_start"),
    ("</ifm|think>", "ifm|think_end"),
    // Solution block tags (alternate answer segment).
    ("<|begin_of_solution|>", "ifm|begin_of_solution"),
    ("<|end_of_solution|>", "ifm|end_of_solution"),
    // Multimodal placeholders (not used by 1B text-only path, but
    // registered for tokenizer consistency).
    ("<image>", "ifm|image"),
    ("</image>", "ifm|/image"),
    ("<video>", "ifm|video"),
    ("</video>", "ifm|/video"),
    ("<audio>", "ifm|audio"),
    ("</audio>", "ifm|/audio"),
    // Per-role turn delimiters (the chat template uses these too).
    ("<|user_start|>", "ifm|user_start"),
    ("<|user_end|>", "ifm|user_end"),
    ("<|assistant_start|>", "ifm|assistant_start"),
    ("<|assistant_end|>", "ifm|assistant_end"),
];

fn string_array(value: Option<MetaValue>, key: &str) -> Result<Vec<String>, String> {
    let Some(MetaValue::Array(MetaValueType::String, values)) = value else {
        return Err(format!("Missing or invalid {key}: expected string array"));
    };
    values
        .into_iter()
        .map(|value| match value {
            MetaValue::String(value) => Ok(value),
            _ => Err(format!("Invalid {key}: expected string array entries")),
        })
        .collect()
}

fn integer_array(value: Option<MetaValue>, key: &str) -> Result<Vec<u64>, String> {
    let Some(MetaValue::Array(kind, values)) = value else {
        return Err(format!("Missing or invalid {key}: expected integer array"));
    };
    if !matches!(
        kind,
        MetaValueType::Uint8
            | MetaValueType::Int8
            | MetaValueType::Uint16
            | MetaValueType::Int16
            | MetaValueType::Uint32
            | MetaValueType::Int32
            | MetaValueType::Uint64
            | MetaValueType::Int64
    ) {
        return Err(format!("Invalid {key}: expected integer array"));
    }
    values
        .into_iter()
        .map(|value| {
            value
                .to_u64()
                .ok_or_else(|| format!("Invalid {key}: expected integer array entries"))
        })
        .collect()
}

fn token_type(value: u64) -> Result<TokenType, String> {
    match value {
        1 => Ok(TokenType::Normal),
        2 => Ok(TokenType::Unknown),
        3 => Ok(TokenType::Control),
        4 => Ok(TokenType::UserDefined),
        5 => Ok(TokenType::Unused),
        6 => Ok(TokenType::Byte),
        _ => Err(format!("Unsupported tokenizer token type: {value}")),
    }
}

fn optional_token_id(value: Option<MetaValue>, key: &str) -> Result<Option<u32>, String> {
    match value {
        None => Ok(None),
        Some(value) => value
            .to_u64()
            .and_then(|id| u32::try_from(id).ok())
            .map(Some)
            .ok_or_else(|| format!("Invalid {key}")),
    }
}

fn validate_token_id(id: Option<u32>, vocab_size: usize, key: &str) -> Result<(), String> {
    if id.is_some_and(|id| id as usize >= vocab_size) {
        return Err(format!("Invalid {key}: token ID is outside the vocabulary"));
    }
    Ok(())
}

fn bool_meta(value: Option<MetaValue>, key: &str) -> Result<bool, String> {
    match value {
        None => Ok(false),
        Some(MetaValue::Bool(value)) => Ok(value),
        Some(_) => Err(format!("Invalid {key}: expected bool")),
    }
}

impl BPETokenizer {
    pub fn from_qwen3_embedded_merges() -> Result<Self, String> {
        const ORACLE_NAMED_VOCAB_SIZE: usize = 151_674;
        const MODEL_VOCAB_SIZE: usize = 151_936;
        let byte_encoder = build_byte_encoder();
        let mut tokens = byte_encoder.clone();
        tokens.sort_by_key(|token| token.chars().next());
        let mut token_types = vec![TokenType::Normal; tokens.len()];
        let mut merge_ranks = HashMap::new();

        for (rank, merge) in include_str!("../../models/diffusion/z_image/qwen_merges.txt")
            .lines()
            .enumerate()
        {
            let (left, right) = merge
                .split_once(' ')
                .ok_or_else(|| format!("Invalid embedded Qwen merge at line {}", rank + 1))?;
            if left.is_empty() || right.is_empty() {
                return Err(format!("Invalid embedded Qwen merge at line {}", rank + 1));
            }
            merge_ranks.insert((left.into(), right.into()), rank as u32);
            tokens.push(format!("{left}{right}"));
            token_types.push(TokenType::Normal);
        }

        for token in QWEN2_ORACLE_SPECIAL_TOKENS {
            tokens.push((*token).into());
            token_types.push(TokenType::Control);
        }
        if tokens.len() != ORACLE_NAMED_VOCAB_SIZE {
            return Err(format!(
                "Embedded Qwen oracle vocabulary has {} named IDs; expected {ORACLE_NAMED_VOCAB_SIZE}",
                tokens.len()
            ));
        }
        // The pinned oracle names IDs through 151673, while the supplied model has
        // 151936 embedding rows. Preserve the remaining rows as deliberately
        // unencodable placeholders instead of inventing tokenizer semantics.
        while tokens.len() < MODEL_VOCAB_SIZE {
            tokens.push(format!("<|reserved_{}|>", tokens.len()));
            token_types.push(TokenType::Unused);
        }

        let token_to_id: HashMap<String, u32> = tokens
            .iter()
            .zip(&token_types)
            .enumerate()
            .filter(|(_, (_, kind))| **kind != TokenType::Unused)
            .map(|(id, (token, _))| (token.clone(), id as u32))
            .collect();
        let mut byte_decoder = HashMap::new();
        for (byte, token) in byte_encoder.iter().enumerate() {
            let value = token
                .chars()
                .next()
                .ok_or_else(|| "Invalid embedded Qwen byte symbol".to_string())?;
            byte_decoder.insert(value, byte as u8);
        }
        let mut special_tokens: Vec<_> = QWEN2_ORACLE_SPECIAL_TOKENS
            .iter()
            .map(|text| SpecialToken {
                text: (*text).into(),
                id: *token_to_id
                    .get(*text)
                    .expect("embedded special token was just inserted"),
                kind: TokenType::Control,
            })
            .collect();
        special_tokens.sort_by(|left, right| right.text.len().cmp(&left.text.len()));
        let semantic_tokens = QWEN_SEMANTIC_TOKENS
            .iter()
            .filter_map(|(literal, name)| {
                token_to_id
                    .get(*literal)
                    .map(|id| ((*name).to_string(), *id))
            })
            .collect();
        let eos_id = token_to_id.get("<|endoftext|>").copied();

        let tokenizer = Self {
            tokens,
            token_types,
            token_to_id,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            pre: PreTokenizer::Qwen2,
            special_tokens,
            semantic_tokens,
            bos_id: None,
            eos_id,
            add_bos: false,
            add_eos: false,
            byte_fallback: false,
            normalize_nfc: false,
        };
        if tokenizer.vocab_size() != MODEL_VOCAB_SIZE {
            return Err(format!(
                "Embedded Qwen vocabulary has {} IDs; expected {MODEL_VOCAB_SIZE}",
                tokenizer.vocab_size()
            ));
        }
        Ok(tokenizer)
    }

    /// Qwen2.5-Omni-3B tokenizer. Same BPE merges as the Qwen2 base but
    /// the named-vocab special token list is the Qwen2.5-Omni one (22
    /// tokens at ids 151643..151664) which includes the audio-related
    /// markers (`<|AUDIO|>`, `<|audio_bos|>`, `<|audio_eos|>`) needed for
    /// AuK CFMEdit reference-audio conditioning.
    pub fn from_qwen25_omni_embedded_merges() -> Result<Self, String> {
        const ORACLE_NAMED_VOCAB_SIZE: usize = 151_665;
        const MODEL_VOCAB_SIZE: usize = 151_936;
        let byte_encoder = build_byte_encoder();
        let mut tokens = byte_encoder.clone();
        tokens.sort_by_key(|token| token.chars().next());
        let mut token_types = vec![TokenType::Normal; tokens.len()];
        let mut merge_ranks = HashMap::new();

        for (rank, merge) in include_str!("../../models/diffusion/z_image/qwen_merges.txt")
            .lines()
            .enumerate()
        {
            let (left, right) = merge
                .split_once(' ')
                .ok_or_else(|| format!("Invalid embedded Qwen merge at line {}", rank + 1))?;
            if left.is_empty() || right.is_empty() {
                return Err(format!("Invalid embedded Qwen merge at line {}", rank + 1));
            }
            merge_ranks.insert((left.into(), right.into()), rank as u32);
            tokens.push(format!("{left}{right}"));
            token_types.push(TokenType::Normal);
        }

        for token in QWEN25_OMNI_SPECIAL_TOKENS {
            tokens.push((*token).into());
            token_types.push(TokenType::Control);
        }
        if tokens.len() != ORACLE_NAMED_VOCAB_SIZE {
            return Err(format!(
                "Embedded Qwen2.5-Omni vocabulary has {} named IDs; expected {ORACLE_NAMED_VOCAB_SIZE}",
                tokens.len()
            ));
        }
        // The pinned oracle names IDs through 151663, while the supplied
        // model has 151936 embedding rows. Preserve the remaining rows as
        // deliberately unencodable placeholders instead of inventing
        // tokenizer semantics.
        while tokens.len() < MODEL_VOCAB_SIZE {
            tokens.push(format!("<|reserved_{}|>", tokens.len()));
            token_types.push(TokenType::Unused);
        }

        let token_to_id: HashMap<String, u32> = tokens
            .iter()
            .zip(&token_types)
            .enumerate()
            .filter(|(_, (_, kind))| **kind != TokenType::Unused)
            .map(|(id, (token, _))| (token.clone(), id as u32))
            .collect();
        let mut byte_decoder = HashMap::new();
        for (byte, token) in byte_encoder.iter().enumerate() {
            let value = token
                .chars()
                .next()
                .ok_or_else(|| "Invalid embedded Qwen byte symbol".to_string())?;
            byte_decoder.insert(value, byte as u8);
        }
        let mut special_tokens: Vec<_> = QWEN25_OMNI_SPECIAL_TOKENS
            .iter()
            .map(|text| SpecialToken {
                text: (*text).into(),
                id: *token_to_id
                    .get(*text)
                    .expect("embedded special token was just inserted"),
                kind: TokenType::Control,
            })
            .collect();
        special_tokens.sort_by(|left, right| right.text.len().cmp(&left.text.len()));
        let semantic_tokens = QWEN_SEMANTIC_TOKENS
            .iter()
            .filter_map(|(literal, name)| {
                token_to_id
                    .get(*literal)
                    .map(|id| ((*name).to_string(), *id))
            })
            .collect();
        let eos_id = token_to_id.get("<|endoftext|>").copied();
        let mut semantic_token_names: HashMap<String, u32> = HashMap::new();
        semantic_token_names.insert("im_start".into(), token_to_id["<|im_start|>"]);
        semantic_token_names.insert("im_end".into(), token_to_id["<|im_end|>"]);
        semantic_token_names.insert("endoftext".into(), token_to_id["<|endoftext|>"]);

        let tokenizer = BPETokenizer {
            tokens,
            token_types,
            token_to_id,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            pre: PreTokenizer::Qwen2,
            special_tokens,
            semantic_tokens,
            bos_id: eos_id,
            eos_id,
            add_bos: false,
            add_eos: false,
            byte_fallback: true,
            normalize_nfc: false,
        };
        // Sanity check: every QWEN25_OMNI_SPECIAL_TOKENS text must round-trip.
        for (i, text) in QWEN25_OMNI_SPECIAL_TOKENS.iter().enumerate() {
            let id = tokenizer.token_to_id.get(*text).copied();
            assert_eq!(
                id,
                Some(151_643 + i as u32),
                "QWEN25_OMNI_SPECIAL_TOKENS[{i}] = {text:?} expected id {} got {id:?}",
                151_643 + i as u32,
            );
        }
        Ok(tokenizer)
    }

    pub fn from_gguf_metadata(
        get_meta: impl Fn(&str) -> Option<MetaValue>,
    ) -> Result<Self, String> {
        let gemma4 = match get_meta("tokenizer.ggml.model") {
            Some(MetaValue::String(value)) if value == "gpt2" => false,
            Some(MetaValue::String(value)) if value == "gemma4" => true,
            Some(MetaValue::String(value)) => {
                return Err(format!(
                    "Unsupported tokenizer.ggml.model {value:?}; expected gpt2 or gemma4"
                ));
            }
            _ => return Err("Missing or invalid tokenizer.ggml.model".into()),
        };

        let mut forced_add_bos = false;
        let pre = if gemma4 {
            PreTokenizer::Gemma4
        } else {
            // llama.cpp forces `add_bos = true` for the llama3 pre family
            // (llama3 / falcon3 / falcon-h1 / lfm2 / pixtral ...) even when
            // the GGUF omits `tokenizer.ggml.add_bos_token`. Falcon-H1 is
            // the one arch in this repo whose GGUF relies on that override,
            // so mirror it explicitly (merged into `add_bos` below).
            let mut force_add_bos = false;
            let pre = match get_meta("tokenizer.ggml.pre") {
                Some(MetaValue::String(value)) if value == "qwen2" => PreTokenizer::Qwen2,
                Some(MetaValue::String(value)) if value == "qwen35" => PreTokenizer::Qwen35,
                Some(MetaValue::String(value)) if value == "spark2_5" => PreTokenizer::Spark2_5,
                Some(MetaValue::String(value)) if value == "hunyuan-dense" => {
                    PreTokenizer::HunyuanDense
                }
                Some(MetaValue::String(value)) if value == "hunyuan" => PreTokenizer::HunyuanDense,
                Some(MetaValue::String(value)) if value == "lfm2" => PreTokenizer::Lfm2,
                Some(MetaValue::String(value)) if value == "llama-bpe" => PreTokenizer::LlamaBpe,
                Some(MetaValue::String(value)) if value == "dbrx" => PreTokenizer::LlamaBpe,
                Some(MetaValue::String(value)) if value == "pixtral" => PreTokenizer::LlamaBpe,
                Some(MetaValue::String(value)) if value == "exaone" => PreTokenizer::LlamaBpe,
                // jina-v1-en / jina-v2-code / roberta-bpe all map to standard
                // GPT-2 byte-level BPE in llama.cpp
                // (`llama-vocab.cpp:2239-2245`) with `add_sep = true`. For our
                // BPETokenizer that maps to `LlamaBpe` (the same regex
                // LlamaBpe uses) plus `add_bos = add_eos = true`, which the
                // GGUF ships via `tokenizer.ggml.add_bos_token` /
                // `tokenizer.ggml.add_eos_token`; the SEP token itself is
                // identical to the EOS token (`seperator_token_id = 2` in
                // jina-reranker-v1-turbo-en), so `add_eos` does the right thing
                // without a separate `add_sep` arm.
                Some(MetaValue::String(value)) if value == "jina-v1-en" => PreTokenizer::LlamaBpe,
                Some(MetaValue::String(value)) if value == "jina-v2-code" => PreTokenizer::LlamaBpe,
                Some(MetaValue::String(value)) if value == "roberta-bpe" => PreTokenizer::LlamaBpe,
                Some(MetaValue::String(value)) if value == "falcon-h1" => {
                    force_add_bos = true;
                    PreTokenizer::LlamaBpe
                }
                Some(MetaValue::String(value)) if value == "k2-horizon" => PreTokenizer::K2Horizon,
                Some(MetaValue::String(value)) if value == "minicpm5" => PreTokenizer::Minicpm5,
                Some(MetaValue::String(value)) if value == "gpt-4o" => PreTokenizer::Gpt4o,
                // GLM-4 (THUDM) uses standard GPT-2 byte-level BPE
                // (`tokenizer.ggml.model = "gpt2"`). The chunked-digit
                // regex from Minicpm5 happens to be a superset, so
                // reusing it is safe — the extra digit-chunking never
                // fires on GLM-4 inputs that Minicpm5 would handle
                // differently.
                Some(MetaValue::String(value)) if value == "glm4" => PreTokenizer::Minicpm5,
                // Mistral 3 / Tekken (`harshatheg/Ministral-3-3B-Instruct-2512-GGUF`):
                // vocab is GPT-2 byte-level BPE with a Mistral-flavored
                // pretokenizer regex. The byte-level BPE encoding is the
                // same as Llama-3 / pixtral / dbrx, so reusing
                // `LlamaBpe` is correct. Special-token boundaries
                // (BOS=1, EOS=2, pad=11) are pinned by the GGUF, no extra
                // `force_add_bos` needed.
                Some(MetaValue::String(value)) if value == "tekken" => PreTokenizer::LlamaBpe,
                Some(MetaValue::String(value)) => {
                    return Err(format!(
                        "Unsupported tokenizer.ggml.pre {value:?}; expected qwen2 or qwen35, hunyuan, hunyuan-dense, lfm2, llama-bpe, pixtral, falcon-h1, exaone, jina-v1-en, jina-v2-code, roberta-bpe, k2-horizon, minicpm5, gpt-4o, tekken, or glm4"
                    ));
                }
                _ => return Err("Missing or invalid tokenizer.ggml.pre".into()),
            };
            forced_add_bos = force_add_bos;
            pre
        };

        let tokens = string_array(get_meta("tokenizer.ggml.tokens"), "tokenizer.ggml.tokens")?;

        let token_type_values = integer_array(
            get_meta("tokenizer.ggml.token_type"),
            "tokenizer.ggml.token_type",
        )?;
        if token_type_values.len() != tokens.len() {
            return Err(format!(
                "tokenizer.ggml.token_type length {} does not match token length {}",
                token_type_values.len(),
                tokens.len()
            ));
        }
        let token_types = token_type_values
            .into_iter()
            .map(token_type)
            .collect::<Result<Vec<_>, _>>()?;

        let merges = string_array(get_meta("tokenizer.ggml.merges"), "tokenizer.ggml.merges")?;

        let mut token_to_id = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let mut merge_ranks = HashMap::new();
        for (rank, merge) in merges.into_iter().enumerate() {
            let Some((left, right)) = merge.split_once(' ') else {
                return Err(format!("Invalid tokenizer.ggml.merges entry: {merge:?}"));
            };
            merge_ranks.insert((left.to_string(), right.to_string()), rank as u32);
        }

        let byte_encoder = build_byte_encoder();
        let mut byte_decoder = HashMap::new();
        for (b, s) in byte_encoder.iter().enumerate() {
            if !s.is_empty() {
                if let Some(ch) = s.chars().next() {
                    byte_decoder.insert(ch, b as u8);
                }
            }
        }

        let bos_id = optional_token_id(
            get_meta("tokenizer.ggml.bos_token_id"),
            "tokenizer.ggml.bos_token_id",
        )?;
        let eos_id = optional_token_id(
            get_meta("tokenizer.ggml.eos_token_id"),
            "tokenizer.ggml.eos_token_id",
        )?;
        validate_token_id(bos_id, tokens.len(), "tokenizer.ggml.bos_token_id")?;
        validate_token_id(eos_id, tokens.len(), "tokenizer.ggml.eos_token_id")?;
        if pre == PreTokenizer::Gemma4 && bos_id.is_none() {
            return Err("tokenizer.ggml.model gemma4 requires tokenizer.ggml.bos_token_id".into());
        }

        let add_bos = bool_meta(
            get_meta("tokenizer.ggml.add_bos_token"),
            "tokenizer.ggml.add_bos_token",
        )? || forced_add_bos;
        let add_eos = bool_meta(
            get_meta("tokenizer.ggml.add_eos_token"),
            "tokenizer.ggml.add_eos_token",
        )?;
        if add_bos && bos_id.is_none() {
            return Err("tokenizer.ggml.add_bos_token requires tokenizer.ggml.bos_token_id".into());
        }
        if add_eos && eos_id.is_none() {
            return Err("tokenizer.ggml.add_eos_token requires tokenizer.ggml.eos_token_id".into());
        }

        let mut special_tokens: Vec<SpecialToken> = tokens
            .iter()
            .zip(&token_types)
            .enumerate()
            .filter_map(|(id, (text, kind))| {
                matches!(
                    kind,
                    TokenType::Control | TokenType::Unknown | TokenType::UserDefined
                )
                .then(|| SpecialToken {
                    text: text.clone(),
                    id: id as u32,
                    kind: *kind,
                })
            })
            .collect();

        // MiniCPM5's GGUF is the one outlier: it tags chat-marker tokens
        // like `` as `token_type = Normal` instead of Control, so the
        // semantic-token list has to re-scan the vocab and pull those
        // literals in regardless of the declared type. For every other
        // pretokenizer the declared `token_type` stays authoritative — a
        // token typed Normal is treated as ordinary text even if its
        // literal happens to spell a ChatML marker (that's the
        // `normal_control_looking_literal_has_no_chatml_semantic_name`
        // contract: literals alone do not confer special semantics).
        let force_semantic_registration = matches!(pre, PreTokenizer::Minicpm5);
        let semantic_literals: &[(&str, &str)] = match pre {
            PreTokenizer::HunyuanDense => HUNYUAN_SEMANTIC_TOKENS,
            PreTokenizer::LlamaBpe | PreTokenizer::Minicpm5 => LLAMA_BPE_SEMANTIC_TOKENS,
            PreTokenizer::K2Horizon => K2_HORIZON_SEMANTIC_TOKENS,
            _ => QWEN_SEMANTIC_TOKENS,
        };
        for (literal, _) in semantic_literals {
            if let Some(&id) = token_to_id.get(*literal) {
                if !special_tokens.iter().any(|t| t.text == *literal)
                    && (force_semantic_registration
                        || token_types.get(id as usize) == Some(&TokenType::Control))
                {
                    special_tokens.push(SpecialToken {
                        text: (*literal).into(),
                        id,
                        kind: TokenType::Control,
                    });
                }
            }
        }
        special_tokens.sort_by(|left, right| right.text.len().cmp(&left.text.len()));

        let semantic_tokens = semantic_literals
            .iter()
            .filter_map(|(literal, name)| {
                special_tokens
                    .iter()
                    .find(|token| token.text == *literal)
                    .map(|token| ((*name).to_string(), token.id))
            })
            .collect();

        let byte_fallback = match get_meta("tokenizer.ggml.byte_fallback") {
            Some(MetaValue::Bool(value)) => value,
            None => tokens.iter().any(|token| hex_byte(token).is_some()),
            Some(_) => return Err("Invalid tokenizer.ggml.byte_fallback: expected bool".into()),
        };

        Ok(Self {
            tokens,
            token_types,
            token_to_id,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            pre,
            special_tokens,
            semantic_tokens,
            bos_id,
            eos_id,
            add_bos,
            add_eos,
            byte_fallback,
            normalize_nfc: bool_meta(
                get_meta("tokenizer.ggml.normalizer.nfc"),
                "tokenizer.ggml.normalizer.nfc",
            )?,
        })
    }

    fn encode_partitioned(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let mut output = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            let best = self
                .special_tokens
                .iter()
                .filter(|token| {
                    parse_special || !matches!(token.kind, TokenType::Control | TokenType::Unknown)
                })
                .filter_map(|token| {
                    remaining
                        .find(&token.text)
                        .map(|position| (position, token))
                })
                .min_by(|(left_pos, left), (right_pos, right)| {
                    left_pos
                        .cmp(right_pos)
                        .then_with(|| right.text.len().cmp(&left.text.len()))
                });

            let Some((position, token)) = best else {
                output.extend(self.encode_ordinary(remaining));
                break;
            };
            output.extend(self.encode_ordinary(&remaining[..position]));
            output.push(token.id);
            remaining = &remaining[position + token.text.len()..];
        }

        output
    }

    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let mut normalized;
        let text = if self.normalize_nfc {
            normalized = tokenizers::NormalizedString::from(text);
            normalized.nfc();
            normalized.get()
        } else {
            text
        };
        let mut ids = Vec::new();
        if self.pre == PreTokenizer::Gemma4 {
            for fragment in text.split_inclusive('\n') {
                let (ordinary, newline) = fragment
                    .strip_suffix('\n')
                    .map_or((fragment, false), |ordinary| (ordinary, true));
                ids.extend(self.encode_bpe_segment(&ordinary.replace(' ', "▁")));
                if newline {
                    ids.extend(self.encode_bpe_segment("\n"));
                }
            }
            return ids;
        }
        for range in scan_qwen_ranges(text, self.pre) {
            ids.extend(self.encode_bpe_segment(&text[range]));
        }
        ids
    }

    pub fn encode(&self, text: &str, options: EncodeOptions) -> Vec<u32> {
        let mut ids = self.encode_partitioned(text, options.parse_special);
        if self.pre == PreTokenizer::Gemma4 {
            ids.insert(0, self.bos_id.expect("Gemma4 requires bos_id"));
        } else if options.add_special && self.add_bos {
            ids.insert(0, self.bos_id.expect("validated add_bos requires bos_id"));
        }
        if options.add_special && self.add_eos {
            ids.push(self.eos_id.expect("validated add_eos requires eos_id"));
        }
        ids
    }

    fn encode_bpe_segment(&self, text: &str) -> Vec<u32> {
        let mut symbols: Vec<Symbol> = if self.pre == PreTokenizer::Gemma4 {
            text.chars()
                .map(|value| Symbol {
                    text: value.to_string(),
                    prev: 0,
                    next: 0,
                    n: 1,
                })
                .collect()
        } else {
            text.as_bytes()
                .iter()
                .map(|&byte| Symbol {
                    text: self.byte_encoder[byte as usize].clone(),
                    prev: 0,
                    next: 0,
                    n: 1,
                })
                .collect()
        };

        for i in 0..symbols.len() {
            symbols[i].prev = if i > 0 { i - 1 } else { usize::MAX };
            symbols[i].next = if i + 1 < symbols.len() {
                i + 1
            } else {
                usize::MAX
            };
        }

        loop {
            let mut best_rank = u32::MAX;
            let mut best_idx = 0;
            let mut found = false;

            let mut i = 0;
            while i < symbols.len() {
                if symbols[i].n == 0 {
                    i += 1;
                    continue;
                }
                let next = symbols[i].next;
                if next == usize::MAX || symbols[next].n == 0 {
                    i += 1;
                    continue;
                }

                if let Some(&rank) = self
                    .merge_ranks
                    .get(&(symbols[i].text.clone(), symbols[next].text.clone()))
                {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                        found = true;
                    }
                }
                i += 1;
            }

            if !found {
                break;
            }

            let next = symbols[best_idx].next;
            symbols[best_idx].text = format!("{}{}", symbols[best_idx].text, symbols[next].text);
            symbols[best_idx].n += symbols[next].n;
            symbols[next].n = 0;
            symbols[best_idx].next = symbols[next].next;
            if symbols[best_idx].next != usize::MAX {
                let nn = symbols[best_idx].next;
                symbols[nn].prev = best_idx;
            }
        }

        let mut ids = Vec::new();
        for sym in &symbols {
            if sym.n == 0 {
                continue;
            }
            if let Some(&id) = self.token_to_id.get(&sym.text) {
                ids.push(id);
            } else if self.byte_fallback {
                for &b in sym.text.as_bytes() {
                    let hex_token = format!("<0x{:02X}>", b);
                    if let Some(&id) = self.token_to_id.get(&hex_token) {
                        ids.push(id);
                    }
                }
            }
        }
        ids
    }

    pub fn token_piece_bytes(&self, id: u32, render_special: bool) -> Vec<u8> {
        let Some(token) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        let kind = self.token_types[id as usize];
        if matches!(kind, TokenType::Control | TokenType::Unknown) && !render_special {
            return Vec::new();
        }
        if let Some(byte) = hex_byte(token) {
            return vec![byte];
        }
        if self.pre == PreTokenizer::Gemma4 && kind == TokenType::Normal {
            // Gemma4 token pieces are raw UTF-8 but use the SentencePiece `▁`
            // (U+2581) marker at word starts. Replace it with a literal space
            // so decoded text reads naturally.
            return token.replace('\u{2581}', " ").into_bytes();
        }

        let mut bytes = Vec::new();
        for value in token.chars() {
            if let Some(byte) = self.byte_decoder.get(&value) {
                bytes.push(*byte);
            } else {
                let mut encoded = [0u8; 4];
                bytes.extend_from_slice(value.encode_utf8(&mut encoded).as_bytes());
            }
        }
        bytes
    }

    pub fn decode_bytes(&self, ids: &[u32], render_special: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend(self.token_piece_bytes(id, render_special));
        }
        bytes
    }

    pub fn decode(&self, ids: &[u32], render_special: bool) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, render_special)).into_owned()
    }

    pub fn streaming_decoder(&self, render_special: bool) -> StreamingDecoder<'_> {
        StreamingDecoder {
            tokenizer: self,
            render_special,
            pending: Vec::new(),
        }
    }

    pub fn token_id(&self, literal: &str) -> Option<u32> {
        self.token_to_id.get(literal).copied()
    }

    pub fn special_token_id(&self, semantic_name: &str) -> Option<u32> {
        self.semantic_tokens.get(semantic_name).copied()
    }

    pub fn contains_special_literal(&self, text: &str) -> bool {
        self.special_tokens
            .iter()
            .any(|token| text.contains(&token.text))
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    pub fn add_bos(&self) -> bool {
        self.add_bos
    }

    pub fn add_eos(&self) -> bool {
        self.add_eos
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_str(&self, id: u32) -> &str {
        self.tokens
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }
}

pub struct StreamingDecoder<'a> {
    tokenizer: &'a dyn Tokenizer,
    render_special: bool,
    pending: Vec<u8>,
}

impl<'a> StreamingDecoder<'a> {
    pub fn new(tokenizer: &'a dyn Tokenizer, render_special: bool) -> Self {
        Self {
            tokenizer,
            render_special,
            pending: Vec::new(),
        }
    }
}

impl StreamingDecoder<'_> {
    pub fn push(&mut self, id: u32) -> String {
        self.pending
            .extend(self.tokenizer.token_piece_bytes(id, self.render_special));
        let mut output = String::new();

        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    output.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    output.push_str(
                        std::str::from_utf8(&self.pending[..valid])
                            .expect("valid_up_to prefix is UTF-8"),
                    );
                    self.pending.drain(..valid);
                    let Some(invalid_len) = error.error_len() else {
                        break;
                    };
                    output.push('\u{FFFD}');
                    self.pending.drain(..invalid_len.min(self.pending.len()));
                }
            }
        }
        output
    }

    pub fn finish(self) -> String {
        String::from_utf8_lossy(&self.pending).into_owned()
    }
}

// ============================================================================
// SentencePiece tokenizer (tokenizer.ggml.model = "llama")
//
// Port of llama.cpp's `llm_tokenizer_spm` + `llm_tokenizer_spm_session` from
// `src/llama-vocab.cpp` (commit `074bea2eb1f1349a0118239c4152914aecaa1be4`).
//
// Algorithm summary (per llama.cpp):
//   1. Pretokenizer: split input on any CONTROL/USER_DEFINED tokens. For each
//      remaining text fragment:
//        a) prepend ' ' if the previous fragment was a special token
//           (`add_space_prefix`)
//        b) escape ' ' (0x20) to U+2581 (`▁`)
//   2. Tokenizer: BPE merge via a max-heap seeded with all adjacent UTF-8
//      char pairs. Each pair's score is the vocab score of the merged token.
//      Ties broken by `left asc`. After all merges, walk the linked list and
//      emit tokens via direct vocab lookup; if missing, fall back through
//      `rev_merge` (which two symbols originally produced this span), and
//      finally byte-by-byte via `<0xXX>` tokens.
//
// Defaults when tokenizer.ggml.model == "llama":
//   - bos_id = 1 (<s>), eos_id = 2 (</s>), unk_id = 0 (<unk>)
//   - add_space_prefix = true
//   - add_bos = true, add_eos = false (unless overridden in GGUF)
// ============================================================================

/// SentencePiece tokenizer for GGUF models with `tokenizer.ggml.model = "llama"`.
pub struct SPMTokenizer {
    tokens: Vec<String>,
    token_types: Vec<TokenType>,
    token_to_id: HashMap<String, u32>,
    scores: Vec<f32>,
    /// `<0x00>` … `<0xFF>` token ids for byte fallback. Slot for byte `b` is
    /// at index `b` (default 0 / unk_id if not present).
    byte_tokens: [u32; 256],
    /// CONTROL / USER_DEFINED tokens used by the partition step. Sorted by
    /// descending text length so longer matches win.
    special_tokens: Vec<SpecialToken>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    unk_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    /// Semantic alias table — `<s>` → "bos", `</s>` → "eos", `<unk>` → "unk".
    semantic_tokens: HashMap<String, u32>,
}

impl SPMTokenizer {
    pub fn from_gguf_metadata(
        get_meta: impl Fn(&str) -> Option<MetaValue>,
    ) -> Result<Self, String> {
        // --- model string ---
        let model = match get_meta("tokenizer.ggml.model") {
            Some(MetaValue::String(value)) if value == "llama" => value,
            Some(MetaValue::String(value)) => {
                return Err(format!(
                    "Unsupported SPM tokenizer.ggml.model {value:?}; expected llama"
                ));
            }
            _ => return Err("Missing or invalid tokenizer.ggml.model".into()),
        };
        let _ = model;

        // --- pretokenizer (informational; SPM path doesn't use regex) ---
        if let Some(MetaValue::String(value)) = get_meta("tokenizer.ggml.pre") {
            if value != "default" {
                return Err(format!(
                    "Unsupported SPM tokenizer.ggml.pre {value:?}; expected default"
                ));
            }
        }

        // --- vocab tokens ---
        let tokens = string_array(get_meta("tokenizer.ggml.tokens"), "tokenizer.ggml.tokens")?;
        let n_tokens = tokens.len();

        // --- scores (REQUIRED for SPM, can be F32 or I32 array) ---
        let scores: Vec<f32> = match get_meta("tokenizer.ggml.scores") {
            Some(MetaValue::Array(_, values)) => match values.first() {
                Some(MetaValue::Float32(_)) => values
                    .iter()
                    .map(|v| match v {
                        MetaValue::Float32(x) => Some(*x),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| "Invalid scores: mixed types".to_string())?,
                Some(MetaValue::Uint32(_)) | Some(MetaValue::Int32(_)) => values
                    .iter()
                    .map(|v| match v {
                        MetaValue::Uint32(x) => Some(*x as f32),
                        MetaValue::Int32(x) => Some(*x as f32),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| "Invalid scores: mixed types".to_string())?,
                _ => return Err("tokenizer.ggml.scores: unsupported inner type".into()),
            },
            _ => return Err("Missing tokenizer.ggml.scores (required for SPM)".into()),
        };
        if scores.len() < n_tokens {
            return Err(format!(
                "tokenizer.ggml.scores length {} < vocab length {}",
                scores.len(),
                n_tokens
            ));
        }

        // --- token types (optional but recommended) ---
        let token_types = if let Some(meta) = get_meta("tokenizer.ggml.token_type") {
            let raw = integer_array(Some(meta), "tokenizer.ggml.token_type")?;
            if raw.len() < n_tokens {
                return Err(format!(
                    "tokenizer.ggml.token_type length {} < vocab length {}",
                    raw.len(),
                    n_tokens
                ));
            }
            raw.into_iter()
                .take(n_tokens)
                .map(token_type)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![TokenType::Normal; n_tokens]
        };

        // --- build vocab lookup ---
        let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(n_tokens);
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        // --- byte fallback table: find <0xXX> tokens ---
        let mut byte_tokens = [0u32; 256];
        for b in 0..u16::from(u8::MAX) + 1 {
            let piece = format!("<0x{:02X}>", b);
            if let Some(&id) = token_to_id.get(&piece) {
                byte_tokens[b as usize] = id;
            }
        }

        // --- special tokens ---
        let mut special_tokens: Vec<SpecialToken> = tokens
            .iter()
            .zip(&token_types)
            .enumerate()
            .filter_map(|(id, (text, kind))| {
                matches!(
                    kind,
                    TokenType::Control | TokenType::Unknown | TokenType::UserDefined
                )
                .then(|| SpecialToken {
                    text: text.clone(),
                    id: id as u32,
                    kind: *kind,
                })
            })
            .collect();
        special_tokens.sort_by(|left, right| right.text.len().cmp(&left.text.len()));

        // --- bos/eos/unk ---
        let bos_id = optional_token_id(
            get_meta("tokenizer.ggml.bos_token_id"),
            "tokenizer.ggml.bos_token_id",
        )?;
        let eos_id = optional_token_id(
            get_meta("tokenizer.ggml.eos_token_id"),
            "tokenizer.ggml.eos_token_id",
        )?;
        let unk_id = optional_token_id(
            get_meta("tokenizer.ggml.unknown_token_id"),
            "tokenizer.ggml.unknown_token_id",
        )?;
        validate_token_id(bos_id, n_tokens, "tokenizer.ggml.bos_token_id")?;
        validate_token_id(eos_id, n_tokens, "tokenizer.ggml.eos_token_id")?;
        validate_token_id(unk_id, n_tokens, "tokenizer.ggml.unknown_token_id")?;

        let add_bos = match get_meta("tokenizer.ggml.add_bos_token") {
            Some(MetaValue::Bool(v)) => v,
            // llama.cpp SPM default: true (vocab.cpp default for SPM vocab)
            _ => true,
        };
        let add_eos = bool_meta(
            get_meta("tokenizer.ggml.add_eos_token"),
            "tokenizer.ggml.add_eos_token",
        )?;
        if add_bos && bos_id.is_none() {
            return Err("tokenizer.ggml.add_bos_token requires tokenizer.ggml.bos_token_id".into());
        }
        if add_eos && eos_id.is_none() {
            return Err("tokenizer.ggml.add_eos_token requires tokenizer.ggml.eos_token_id".into());
        }

        let add_space_prefix = match get_meta("tokenizer.ggml.add_space_prefix") {
            Some(MetaValue::Bool(v)) => v,
            _ => true, // SPM default per llama.cpp
        };

        // --- semantic aliases: only bos/eos/unk for SPM ---
        let mut semantic_tokens: HashMap<String, u32> = HashMap::new();
        if let Some(id) = bos_id {
            semantic_tokens.insert("bos".into(), id);
        }
        if let Some(id) = eos_id {
            semantic_tokens.insert("eos".into(), id);
        }
        if let Some(id) = unk_id {
            semantic_tokens.insert("unk".into(), id);
        }

        Ok(Self {
            tokens,
            token_types,
            token_to_id,
            scores,
            byte_tokens,
            special_tokens,
            bos_id,
            eos_id,
            unk_id,
            add_bos,
            add_eos,
            add_space_prefix,
            semantic_tokens,
        })
    }

    /// Top-level encode entry point matching the BPE surface.
    pub fn encode(&self, text: &str, options: EncodeOptions) -> Vec<u32> {
        let mut output = Vec::new();

        if options.add_special && self.add_bos {
            if let Some(bos) = self.bos_id {
                output.push(bos);
            }
        }

        let fragments = partition_special(&self.special_tokens, text, options.parse_special);
        let mut prev_was_special = true; // mirror llama.cpp: BOS counts as special
        for frag in fragments {
            match frag {
                Fragment::Special(id) => {
                    output.push(id);
                    prev_was_special = true;
                }
                Fragment::Text(span) => {
                    self.encode_fragment(span, prev_was_special, &mut output);
                    prev_was_special = false;
                }
            }
        }

        if options.add_special && self.add_eos {
            if let Some(eos) = self.eos_id {
                output.push(eos);
            }
        }

        output
    }

    /// Encode a single text fragment after applying prefix-space and
    /// whitespace-escape, then running the SPM merge algorithm.
    fn encode_fragment(&self, text: &str, prefix_space: bool, output: &mut Vec<u32>) {
        // 1) prefix space + 2) escape ' ' → '▁'
        let mut buffer = String::with_capacity(text.len() + 1);
        if prefix_space && self.add_space_prefix {
            buffer.push(' ');
        }
        buffer.push_str(text);
        // SPM's "meta-space" marker replaces each ASCII space (matches
        // llama.cpp's `llama_escape_whitespace`).
        buffer = buffer.replace(' ', "\u{2581}");

        if buffer.is_empty() {
            return;
        }

        // 3) Build linked list of UTF-8 byte spans (one per char).
        //    Note: `idx` is the symbol's position in `symbols`, NOT its byte
        //    offset into the buffer — the linked list must traverse by symbol
        //    index, not by byte position.
        let mut symbols: Vec<SpmSymbol> = Vec::with_capacity(buffer.len());
        {
            let mut offset = 0usize;
            for (byte_idx, ch) in buffer.char_indices() {
                let len = ch.len_utf8();
                let sym_idx = symbols.len() as i32;
                symbols.push(SpmSymbol {
                    start: offset,
                    len,
                    prev: sym_idx - 1,
                    next: sym_idx + 1,
                });
                debug_assert_eq!(byte_idx, offset);
                offset += len;
            }
            if let Some(last) = symbols.last_mut() {
                last.next = -1;
            }
        }
        if symbols.is_empty() {
            return;
        }
        let n_symbols = symbols.len() as i32;

        // 4) Seed priority queue with all adjacent pairs.
        let mut work: std::collections::BinaryHeap<SpmBigram> =
            std::collections::BinaryHeap::with_capacity(symbols.len());
        let mut rev_merge: HashMap<String, (i32, i32)> = HashMap::new();
        for i in 1..symbols.len() {
            try_add_bigram(
                i as i32 - 1,
                i as i32,
                &symbols,
                &buffer,
                &self.token_to_id,
                &self.scores,
                &mut work,
                &mut rev_merge,
            );
        }

        // 5) Greedy merge.
        while let Some(bigram) = work.pop() {
            let left = &symbols[bigram.left as usize];
            let right = &symbols[bigram.right as usize];
            if left.len == 0
                || right.len == 0
                || left.len + right.len != bigram.size
                || left.len != bigram.size - right.len
            {
                continue;
            }

            // Merge right into left.
            symbols[bigram.left as usize].len += symbols[bigram.right as usize].len;
            symbols[bigram.right as usize].len = 0;
            symbols[bigram.left as usize].next = symbols[bigram.right as usize].next;
            if let Some(next_idx) = symbols[bigram.right as usize].next_checked() {
                symbols[next_idx].prev = bigram.left;
            }

            // Re-seed at the new boundary.
            try_add_bigram(
                symbols[bigram.left as usize].prev,
                bigram.left,
                &symbols,
                &buffer,
                &self.token_to_id,
                &self.scores,
                &mut work,
                &mut rev_merge,
            );
            try_add_bigram(
                bigram.left,
                symbols[bigram.left as usize].next,
                &symbols,
                &buffer,
                &self.token_to_id,
                &self.scores,
                &mut work,
                &mut rev_merge,
            );
        }

        // 6) Walk linked list, emit tokens via resegment.
        let mut i = 0i32;
        while i != -1 {
            let next = symbols[i as usize].next;
            self.resegment(&symbols, i, &buffer, &rev_merge, output);
            i = next;
        }
        let _ = n_symbols;
    }

    fn resegment(
        &self,
        symbols: &[SpmSymbol],
        idx: i32,
        buffer: &str,
        rev_merge: &HashMap<String, (i32, i32)>,
        output: &mut Vec<u32>,
    ) {
        let sym = &symbols[idx as usize];
        if sym.len == 0 {
            return;
        }
        let text = &buffer[sym.start..sym.start + sym.len];

        if let Some(&id) = self.token_to_id.get(text) {
            output.push(id);
            return;
        }

        if let Some(&(l, r)) = rev_merge.get(text) {
            self.resegment(symbols, l, buffer, rev_merge, output);
            self.resegment(symbols, r, buffer, rev_merge, output);
            return;
        }

        // Byte fallback: emit one <0xXX> per UTF-8 byte. For multi-byte
        // chars each byte is sent independently (matches llama.cpp).
        for &b in text.as_bytes() {
            output.push(self.byte_tokens[b as usize]);
        }
    }

    pub fn decode_bytes(&self, ids: &[u32], render_special: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend(self.token_piece_bytes(id, render_special));
        }
        bytes
    }

    pub fn decode(&self, ids: &[u32], render_special: bool) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, render_special)).into_owned()
    }

    fn token_piece_bytes(&self, id: u32, render_special: bool) -> Vec<u8> {
        let id = id as usize;
        if id >= self.tokens.len() {
            return Vec::new();
        }
        let text = &self.tokens[id];
        let kind = self
            .token_types
            .get(id)
            .copied()
            .unwrap_or(TokenType::Normal);

        match kind {
            TokenType::Control | TokenType::UserDefined | TokenType::Unknown => {
                if render_special {
                    text.as_bytes().to_vec()
                } else {
                    Vec::new()
                }
            }
            TokenType::Byte => {
                if let Some(b) = hex_byte(text) {
                    vec![b]
                } else {
                    Vec::new()
                }
            }
            TokenType::Normal | TokenType::Unused => {
                // Unescape ▁ → ' '
                const META_SPACE: &str = "\u{2581}";
                let meta_len = META_SPACE.len();
                let mut out = String::with_capacity(text.len());
                let bytes = text.as_bytes();
                let mut i = 0;
                while i < bytes.len() {
                    if bytes[i..].starts_with(META_SPACE.as_bytes()) {
                        out.push(' ');
                        i += meta_len;
                    } else {
                        // find next ▁ or end
                        let mut j = i + 1;
                        while j < bytes.len() && !bytes[j..].starts_with(META_SPACE.as_bytes()) {
                            // advance by one UTF-8 char
                            j += 1;
                            while j < bytes.len() && (bytes[j] & 0xC0) == 0x80 {
                                j += 1;
                            }
                        }
                        // SAFETY: we're walking the same valid `text` UTF-8
                        if let Ok(s) = std::str::from_utf8(&bytes[i..j]) {
                            out.push_str(s);
                        }
                        i = j;
                    }
                }
                out.into_bytes()
            }
        }
    }

    pub fn token_id(&self, literal: &str) -> Option<u32> {
        self.token_to_id.get(literal).copied()
    }

    pub fn special_token_id(&self, semantic_name: &str) -> Option<u32> {
        self.semantic_tokens.get(semantic_name).copied()
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    pub fn unk_id(&self) -> Option<u32> {
        self.unk_id
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

enum Fragment<'a> {
    Text(&'a str),
    Special(u32),
}

struct SpmSymbol {
    /// Byte offset into the escaped text buffer.
    start: usize,
    /// Current byte length. Doubles as a "deleted" flag (== 0 means removed).
    len: usize,
    prev: i32,
    next: i32,
}

impl SpmSymbol {
    fn next_checked(&self) -> Option<usize> {
        if self.next < 0 {
            None
        } else {
            Some(self.next as usize)
        }
    }
}

#[derive(Clone, Copy)]
struct SpmBigram {
    left: i32,
    right: i32,
    /// Combined byte length of (left..right) at the time the bigram was seeded.
    /// Used to invalidate stale heap entries after further merges.
    size: usize,
    score: f32,
}

impl PartialEq for SpmBigram {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.left == other.left
    }
}
impl Eq for SpmBigram {}
impl PartialOrd for SpmBigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
// Max-heap by (score desc, left asc), matching llama.cpp's SPM queue.
impl Ord for SpmBigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(other.left.cmp(&self.left))
    }
}

#[allow(clippy::too_many_arguments)]
fn try_add_bigram(
    left: i32,
    right: i32,
    symbols: &[SpmSymbol],
    buffer: &str,
    token_to_id: &HashMap<String, u32>,
    scores: &[f32],
    work: &mut std::collections::BinaryHeap<SpmBigram>,
    rev_merge: &mut HashMap<String, (i32, i32)>,
) {
    if left < 0 || right < 0 {
        return;
    }
    let l = &symbols[left as usize];
    let r = &symbols[right as usize];
    if l.len == 0 || r.len == 0 {
        return;
    }
    let combined_len = l.len + r.len;
    let combined = &buffer[l.start..l.start + combined_len];
    let Some(&token) = token_to_id.get(combined) else {
        return;
    };

    work.push(SpmBigram {
        left,
        right,
        size: combined_len,
        score: scores[token as usize],
    });
    rev_merge.insert(combined.to_string(), (left, right));
}

// ============================================================================
// Top-level dispatch: read the tokenizer model type from GGUF and return the
// appropriate tokenizer implementation behind a `Box<dyn Tokenizer>`.
// ============================================================================

/// Common surface shared by `BPETokenizer` and `SPMTokenizer`. Implemented for
/// both so call sites that only need encoding/decoding/special-token lookups
/// can be polymorphic.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, options: EncodeOptions) -> Vec<u32>;
    fn decode_bytes(&self, ids: &[u32], render_special: bool) -> Vec<u8>;
    fn decode(&self, ids: &[u32], render_special: bool) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, render_special)).into_owned()
    }
    fn token_piece_bytes(&self, id: u32, render_special: bool) -> Vec<u8>;
    fn token_id(&self, literal: &str) -> Option<u32>;
    fn special_token_id(&self, semantic_name: &str) -> Option<u32>;
    fn bos_id(&self) -> Option<u32>;
    fn eos_id(&self) -> Option<u32>;
    fn vocab_size(&self) -> usize;
}

impl Tokenizer for BPETokenizer {
    fn encode(&self, text: &str, options: EncodeOptions) -> Vec<u32> {
        BPETokenizer::encode(self, text, options)
    }
    fn decode_bytes(&self, ids: &[u32], render_special: bool) -> Vec<u8> {
        BPETokenizer::decode_bytes(self, ids, render_special)
    }
    fn token_piece_bytes(&self, id: u32, render_special: bool) -> Vec<u8> {
        BPETokenizer::token_piece_bytes(self, id, render_special)
    }
    fn token_id(&self, literal: &str) -> Option<u32> {
        BPETokenizer::token_id(self, literal)
    }
    fn special_token_id(&self, semantic_name: &str) -> Option<u32> {
        BPETokenizer::special_token_id(self, semantic_name)
    }
    fn bos_id(&self) -> Option<u32> {
        BPETokenizer::bos_id(self)
    }
    fn eos_id(&self) -> Option<u32> {
        BPETokenizer::eos_id(self)
    }
    fn vocab_size(&self) -> usize {
        BPETokenizer::vocab_size(self)
    }
}

impl Tokenizer for SPMTokenizer {
    fn encode(&self, text: &str, options: EncodeOptions) -> Vec<u32> {
        SPMTokenizer::encode(self, text, options)
    }
    fn decode_bytes(&self, ids: &[u32], render_special: bool) -> Vec<u8> {
        SPMTokenizer::decode_bytes(self, ids, render_special)
    }
    fn token_piece_bytes(&self, id: u32, render_special: bool) -> Vec<u8> {
        SPMTokenizer::token_piece_bytes(self, id, render_special)
    }
    fn token_id(&self, literal: &str) -> Option<u32> {
        SPMTokenizer::token_id(self, literal)
    }
    fn special_token_id(&self, semantic_name: &str) -> Option<u32> {
        SPMTokenizer::special_token_id(self, semantic_name)
    }
    fn bos_id(&self) -> Option<u32> {
        SPMTokenizer::bos_id(self)
    }
    fn eos_id(&self) -> Option<u32> {
        SPMTokenizer::eos_id(self)
    }
    fn vocab_size(&self) -> usize {
        SPMTokenizer::vocab_size(self)
    }
}

/// Read `tokenizer.ggml.model` from GGUF metadata and dispatch to the
/// right implementation. Returns a trait object so callers can stay
/// agnostic. The set of supported model strings:
///
///   * `gpt2` / `gemma4`        — BPE
///   * `llama`                  — SentencePiece (SPM)
///   * `bert`                   — WordPiece (separate construction
///                                function, `WPMTokenizer`)
///   * `t5`                     — SentencePiece unigram with a
///                                precompiled XCDA normalization map
///                                (`UgmTokenizer`)
pub fn load_tokenizer(
    get_meta: impl Fn(&str) -> Option<MetaValue>,
) -> Result<Box<dyn Tokenizer>, String> {
    match get_meta("tokenizer.ggml.model") {
        Some(MetaValue::String(value)) if value == "llama" => {
            Ok(Box::new(SPMTokenizer::from_gguf_metadata(get_meta)?))
        }
        Some(MetaValue::String(value)) if value == "t5" => Ok(Box::new(
            ugm::UgmTokenizer::from_gguf_metadata(get_meta).map_err(|e| e.to_string())?,
        )),
        _ => Ok(Box::new(BPETokenizer::from_gguf_metadata(get_meta)?)),
    }
}

fn hex_byte(piece: &str) -> Option<u8> {
    (piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>'))
        .then(|| u8::from_str_radix(&piece[3..5], 16).ok())
        .flatten()
}

fn is_word_char(value: char, pre: PreTokenizer) -> bool {
    value.is_letter()
        || (matches!(pre, PreTokenizer::Qwen35 | PreTokenizer::K2Horizon) && value.is_mark())
        || (pre == PreTokenizer::K2Horizon && matches!(value, '\u{200c}' | '\u{200d}'))
}

fn is_number(value: char) -> bool {
    value.is_number()
}

fn is_punctuation_run_char(value: char, pre: PreTokenizer) -> bool {
    !value.is_whitespace() && !is_word_char(value, pre) && !is_number(value)
}

/// Spark2_5-specific punctuation rule: returns true if `pos` is at a
/// punctuation char followed by a letter (which Spark2_5 splits into
/// two tokens). Mirrors llama.cpp's
/// `[!"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~][A-Za-z]+` regex.
fn spark2_5_splits_punct_before_letter(values: &[char], pos: usize) -> bool {
    let here = match values.get(pos).copied() {
        Some(c) => c,
        None => return false,
    };
    if here.is_whitespace() || is_word_char(here, PreTokenizer::Spark2_5) || is_number(here) {
        return false;
    }
    match values.get(pos + 1).copied() {
        Some(next) => is_word_char(next, PreTokenizer::Spark2_5),
        None => false,
    }
}

fn contraction_len(chars: &[char], pos: usize) -> Option<usize> {
    if chars.get(pos) != Some(&'\'') {
        return None;
    }
    let one = chars.get(pos + 1)?.to_ascii_lowercase();
    if matches!(one, 's' | 't' | 'm' | 'd') {
        return Some(2);
    }
    let two = chars.get(pos + 2)?.to_ascii_lowercase();
    if matches!((one, two), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
        return Some(3);
    }
    None
}

fn scan_qwen_ranges(text: &str, pre: PreTokenizer) -> Vec<Range<usize>> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let values: Vec<char> = chars.iter().map(|(_, value)| *value).collect();
    let byte_at = |index: usize| {
        chars
            .get(index)
            .map(|(byte, _)| *byte)
            .unwrap_or(text.len())
    };
    let mut ranges = Vec::new();
    let mut pos = 0usize;

    while pos < values.len() {
        let start = pos;
        let current = values[pos];

        if let Some(length) = contraction_len(&values, pos) {
            pos += length;
            ranges.push(byte_at(start)..byte_at(pos));
            continue;
        }

        if current != '\r' && current != '\n' && !is_number(current) {
            // Spark2_5 only suppresses word-start at *punctuation* chars, not
            // at whitespace (whitespace + letter is still one piece, e.g.
            // ` hello`). This is the practical reading of llama.cpp's
            // `[punct][A-Za-z]+` rule.
            let current_is_punct_only =
                !current.is_whitespace() && !is_word_char(current, pre) && !is_number(current);
            let starts_word = if pre == PreTokenizer::Spark2_5 && current_is_punct_only {
                false
            } else {
                is_word_char(current, pre)
                    || values
                        .get(pos + 1)
                        .copied()
                        .is_some_and(|value| is_word_char(value, pre))
            };
            if starts_word {
                if pre == PreTokenizer::Gpt4o {
                    // GPT-4O / tiktoken-cl100k regex (simplified):
                    //   [^\r\n\p{L}\p{N}]?\p{L}+
                    // i.e. an OPTIONAL single non-letter/non-number/non-newline
                    // char (typically a leading space, but also tabs,
                    // commas, etc.) followed by 1+ letters. The leading
                    // space MUST be folded into the same pre-token as
                    // the word, otherwise the BPE merges a separate
                    // `" "` (or `"Ġ"`) token between every word and
                    // every model trained on cl100k_base vocab (Phi-4,
                    // GPT-4o, etc.) sees a doubled-length input and
                    // produces degenerate output.
                    let mut p = pos;
                    // Optional leading single non-letter/non-number
                    // char. The regex matches one char in
                    // `[^\r\n\p{L}\p{N}]` — that includes whitespace,
                    // punctuation, and similar, but NOT `\r`, `\n`,
                    // letters, or numbers. Skip at most ONE char.
                    if current != '\r'
                        && current != '\n'
                        && !is_word_char(current, PreTokenizer::Gpt4o)
                        && !is_number(current)
                    {
                        p += 1;
                    }
                    let first_is_lower = values.get(p).copied().is_some_and(|c| c.is_lowercase());
                    if first_is_lower {
                        // Alt 1: 0+ uppercase, 1+ lowercase.
                        while values.get(p).copied().is_some_and(|c| c.is_uppercase()) {
                            p += 1;
                        }
                        while values.get(p).copied().is_some_and(|c| c.is_lowercase()) {
                            p += 1;
                        }
                    } else {
                        // Alt 2: 1+ uppercase, 0+ lowercase.
                        while values.get(p).copied().is_some_and(|c| c.is_uppercase()) {
                            p += 1;
                        }
                        while values.get(p).copied().is_some_and(|c| c.is_lowercase()) {
                            p += 1;
                        }
                    }
                    if p == start || (p == start + 1 && start != pos) {
                        // No letter consumed (e.g., lone punctuation
                        // not handled by the lowercase/uppercase
                        // branches). Fall through to the standard
                        // single-char consume path.
                        pos += 1;
                        ranges.push(byte_at(start)..byte_at(pos));
                    } else {
                        pos = p;
                        ranges.push(byte_at(start)..byte_at(pos));
                    }
                    continue;
                }
                pos += 1;
                while values
                    .get(pos)
                    .copied()
                    .is_some_and(|value| is_word_char(value, pre))
                {
                    pos += 1;
                }
                ranges.push(byte_at(start)..byte_at(pos));
                continue;
            }
        }

        if is_number(current) {
            pos += 1;
            // K2Horizon, Minicpm5, and Gpt4o all chunk digit runs into 1-3
            // digit pieces. llama.cpp implements Minicpm5 by first
            // running `\\p{N}{1,3}` over the text, then the standard
            // GPT-2 split; the chunk-size cap is equivalent for the
            // second pass. Gpt4o is `\\p{N}{1,3}` directly.
            if matches!(
                pre,
                PreTokenizer::K2Horizon | PreTokenizer::Minicpm5 | PreTokenizer::Gpt4o
            ) {
                while pos - start < 3 && values.get(pos).copied().is_some_and(is_number) {
                    pos += 1;
                }
            }
            ranges.push(byte_at(start)..byte_at(pos));
            continue;
        }

        let punctuation_start = if current == ' ' { pos + 1 } else { pos };
        if values
            .get(punctuation_start)
            .copied()
            .is_some_and(|value| is_punctuation_run_char(value, pre))
        {
            pos = punctuation_start;
            // Always consume at least the first punctuation char so we
            // make forward progress even if Spark2_5 immediately splits
            // (in which case the loop body below terminates after one step).
            if values
                .get(pos)
                .copied()
                .is_some_and(|value| is_punctuation_run_char(value, pre))
            {
                pos += 1;
            }
            while values
                .get(pos)
                .copied()
                .is_some_and(|value| is_punctuation_run_char(value, pre))
            {
                // Spark2_5: stop the punctuation run before a letter so
                // `!hello` tokenizes as `!` + `hello`, matching the
                // `[punct][A-Za-z]+` regex in llama.cpp.
                if pre == PreTokenizer::Spark2_5
                    && spark2_5_splits_punct_before_letter(&values, pos)
                {
                    break;
                }
                pos += 1;
            }
            while matches!(values.get(pos), Some('\r' | '\n')) {
                pos += 1;
            }
            ranges.push(byte_at(start)..byte_at(pos));
            continue;
        }

        let mut whitespace_count = 0usize;
        let mut last_newline_end = None;
        while let Some(value) = values.get(pos + whitespace_count) {
            if !value.is_whitespace() {
                break;
            }
            whitespace_count += 1;
            if matches!(value, '\r' | '\n') {
                last_newline_end = Some(pos + whitespace_count);
            }
        }
        if let Some(end) = last_newline_end {
            pos = end;
        } else if whitespace_count > 1 && pos + whitespace_count < values.len() {
            pos += whitespace_count - 1;
        } else if whitespace_count > 0 {
            pos += whitespace_count;
        } else {
            pos += 1;
        }
        ranges.push(byte_at(start)..byte_at(pos));
    }

    ranges
}

fn scan_qwen_words(text: &str, pre: PreTokenizer) -> Vec<&str> {
    scan_qwen_ranges(text, pre)
        .into_iter()
        .map(|range| &text[range])
        .collect()
}

struct Symbol {
    text: String,
    prev: usize,
    next: usize,
    n: usize,
}

fn build_byte_encoder() -> Vec<String> {
    let mut bs: Vec<u8> = Vec::new();
    for b in 33u8..=126 {
        bs.push(b);
    }
    for b in 161u8..=172 {
        bs.push(b);
    }
    for b in 174u8..=255 {
        bs.push(b);
    }
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0u8..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut encoder = vec![String::new(); 256];
    for i in 0..bs.len() {
        if (bs[i] as usize) < 256 {
            encoder[bs[i] as usize] = char::from_u32(cs[i]).unwrap_or('?').to_string();
        }
    }
    encoder
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValue, MetaValueType};
    use std::collections::HashMap;

    fn tokenizer_with_pre(pre: Option<&str>) -> Result<BPETokenizer, String> {
        let mut metadata: HashMap<String, MetaValue> = HashMap::from([
            (
                "tokenizer.ggml.model".to_string(),
                MetaValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                MetaValue::Array(
                    MetaValueType::String,
                    ["a", "b", "Ġ", "Ġa", "Ġb"]
                        .into_iter()
                        .map(|value| MetaValue::String(value.into()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                MetaValue::Array(
                    MetaValueType::Uint32,
                    std::iter::repeat(MetaValue::Uint32(1)).take(5).collect(),
                ),
            ),
            (
                "tokenizer.ggml.merges".to_string(),
                MetaValue::Array(MetaValueType::String, vec![]),
            ),
        ]);
        if let Some(pre) = pre {
            metadata.insert("tokenizer.ggml.pre".into(), MetaValue::String(pre.into()));
        }
        BPETokenizer::from_gguf_metadata(|key| metadata.get(key).cloned())
    }

    #[test]
    fn rejects_missing_or_unknown_qwen_pre() {
        assert!(tokenizer_with_pre(None)
            .unwrap_err()
            .contains("tokenizer.ggml.pre"));
        assert!(tokenizer_with_pre(Some("default"))
            .unwrap_err()
            .contains("qwen2 or qwen35"));
    }

    #[test]
    fn qwen2_scanner_preserves_reference_boundaries() {
        assert_eq!(
            scan_qwen_words("hello   world", PreTokenizer::Qwen2),
            vec!["hello", "  ", " world"]
        );
        assert_eq!(scan_qwen_words("  a", PreTokenizer::Qwen2), vec![" ", " a"]);
        assert_eq!(
            scan_qwen_words("a  b", PreTokenizer::Qwen2),
            vec!["a", " ", " b"]
        );
    }

    #[test]
    fn embedded_qwen3_tokenizer_matches_z_image_chatml() {
        let tokenizer = BPETokenizer::from_qwen3_embedded_merges().unwrap();
        assert_eq!(tokenizer.vocab_size(), 151_936);
        assert_eq!(tokenizer.token_types[151_674], TokenType::Unused);
        assert!(!tokenizer.token_to_id.contains_key("<|reserved_151674|>"));
        assert_eq!(
            tokenizer.special_tokens.len(),
            QWEN2_ORACLE_SPECIAL_TOKENS.len()
        );
        assert_eq!(tokenizer.special_token_id("reserved_151674"), None);
        assert_eq!(
            tokenizer.encode("hello   world", EncodeOptions::default()),
            vec![14_990, 256, 1_879]
        );
        assert_eq!(
            tokenizer.encode(
                "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
                EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            ),
            vec![151_644, 872, 198, 9_707, 151_645, 198, 151_644, 77_091, 198]
        );
    }

    #[test]
    fn qwen35_scanner_keeps_combining_marks_in_letter_runs() {
        assert_eq!(
            scan_qwen_words("e\u{301}", PreTokenizer::Qwen35),
            vec!["e\u{301}"]
        );
        assert_eq!(
            scan_qwen_words("re\u{301}sume\u{301}", PreTokenizer::Qwen35),
            vec!["re\u{301}sume\u{301}"]
        );
        assert_eq!(
            scan_qwen_words("Vieết Nam", PreTokenizer::Qwen35),
            vec!["Vieết", " Nam"]
        );
        assert_eq!(
            scan_qwen_words("e\u{301}", PreTokenizer::Qwen2),
            vec!["e", "\u{301}"]
        );
    }

    #[test]
    fn k2_horizon_scanner_matches_reference_boundaries() {
        assert_eq!(
            scan_qwen_words("abc1234", PreTokenizer::K2Horizon),
            vec!["abc", "123", "4"]
        );
        assert_eq!(
            scan_qwen_words("re\u{301}sume\u{200d}now", PreTokenizer::K2Horizon),
            vec!["re\u{301}sume\u{200d}now"]
        );
        assert!(tokenizer_with_pre(Some("k2-horizon")).is_ok());
    }

    #[test]
    fn spark2_5_splits_punctuation_before_letter() {
        // `[!"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~][A-Za-z]+` pattern:
        // a single punctuation char immediately followed by ASCII letters
        // is split into two tokens.
        assert_eq!(
            scan_qwen_words("!hello", PreTokenizer::Spark2_5),
            vec!["!", "hello"]
        );
        assert_eq!(
            scan_qwen_words("/world", PreTokenizer::Spark2_5),
            vec!["/", "world"]
        );
        // Qwen2 keeps the run together (its regex has no such rule).
        assert_eq!(
            scan_qwen_words("!hello", PreTokenizer::Qwen2),
            vec!["!hello"]
        );
        // Two punctuation chars: the first splits before the second (non-letter),
        // the second splits before the letter run — yielding three tokens.
        // (llama.cpp's full regex would not match `!!hello`; BPE merges
        // would then split, which is functionally similar.)
        assert_eq!(
            scan_qwen_words("!!hello", PreTokenizer::Spark2_5),
            vec!["!", "!", "hello"]
        );
        // Letter run after a space is unaffected.
        assert_eq!(
            scan_qwen_words("! hello", PreTokenizer::Spark2_5),
            vec!["!", " hello"]
        );
    }

    fn tokenizer_from_parts(
        tokens: &[&str],
        token_types: &[u32],
        bos_id: Option<u32>,
        eos_id: Option<u32>,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<BPETokenizer, String> {
        let mut metadata: HashMap<String, MetaValue> = HashMap::from([
            (
                "tokenizer.ggml.model".into(),
                MetaValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.pre".into(),
                MetaValue::String("qwen2".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                MetaValue::Array(
                    MetaValueType::String,
                    tokens
                        .iter()
                        .map(|value| MetaValue::String((*value).into()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                MetaValue::Array(
                    MetaValueType::Uint32,
                    token_types.iter().copied().map(MetaValue::Uint32).collect(),
                ),
            ),
            (
                "tokenizer.ggml.merges".into(),
                MetaValue::Array(MetaValueType::String, Vec::new()),
            ),
            (
                "tokenizer.ggml.add_bos_token".into(),
                MetaValue::Bool(add_bos),
            ),
            (
                "tokenizer.ggml.add_eos_token".into(),
                MetaValue::Bool(add_eos),
            ),
        ]);
        if let Some(id) = bos_id {
            metadata.insert("tokenizer.ggml.bos_token_id".into(), MetaValue::Uint32(id));
        }
        if let Some(id) = eos_id {
            metadata.insert("tokenizer.ggml.eos_token_id".into(), MetaValue::Uint32(id));
        }
        BPETokenizer::from_gguf_metadata(|key| metadata.get(key).cloned())
    }

    fn special_test_tokenizer() -> BPETokenizer {
        tokenizer_from_parts(
            &[
                "a",
                "b",
                "_",
                "<|im_start|>",
                "<tool_call>",
                "<|im_end|>",
                "<|image_pad|>",
                "<|vision_start|>",
                "<|vision_end|>",
                "<|audio_start|>",
                "<|audio_pad|>",
                "<|audio_end|>",
                "<asr_text>",
                "<|endoftext|>",
            ],
            &[1, 1, 1, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            None,
            Some(13),
            false,
            false,
        )
        .unwrap()
    }

    fn overlapping_special_test_tokenizer() -> BPETokenizer {
        tokenizer_from_parts(
            &["a", "b", "c", "d", "e", "f", "<tool>", "<tool_call>"],
            &[1, 1, 1, 1, 1, 1, 4, 4],
            None,
            None,
            false,
            false,
        )
        .unwrap()
    }

    fn bos_eos_test_tokenizer(add_bos: bool, add_eos: bool) -> BPETokenizer {
        tokenizer_from_parts(
            &["a", "b", "c", "d", "e", "f", "g", "h", "<s>", "</s>"],
            &[1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
            Some(8),
            Some(9),
            add_bos,
            add_eos,
        )
        .unwrap()
    }

    fn tokenizer_with_token_types(types: &[u32]) -> Result<BPETokenizer, String> {
        tokenizer_from_parts(&["a", "b"], types, None, None, false, false)
    }

    fn normal_control_literal_test_tokenizer() -> BPETokenizer {
        tokenizer_from_parts(&["a", "<|im_start|>"], &[1, 1], None, None, false, false).unwrap()
    }

    fn gemma4_test_tokenizer() -> BPETokenizer {
        let mut tokens = vec!["<unused>".to_string(); 258_884];
        let mut token_types = vec![MetaValue::Uint32(5); tokens.len()];
        for (id, token, kind) in [
            (0, "<bos>", 3),
            (1, "▁", 1),
            (2, "h", 1),
            (3, "e", 1),
            (4, "l", 1),
            (5, "o", 1),
            (6, "\n", 1),
            (7, "w", 1),
            (8, "r", 1),
            (9, "d", 1),
            (10, "Ā", 1),
            (255_999, "<|image>", 3),
            (256_000, "<audio>", 3),
            (258_882, "</image>", 3),
            (258_883, "<audio|>", 3),
        ] {
            tokens[id] = token.into();
            token_types[id] = MetaValue::Uint32(kind);
        }
        let metadata: HashMap<String, MetaValue> = HashMap::from([
            (
                "tokenizer.ggml.model".into(),
                MetaValue::String("gemma4".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                MetaValue::Array(
                    MetaValueType::String,
                    tokens.into_iter().map(MetaValue::String).collect(),
                ),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                MetaValue::Array(MetaValueType::Uint32, token_types),
            ),
            (
                "tokenizer.ggml.merges".into(),
                MetaValue::Array(MetaValueType::String, Vec::new()),
            ),
            ("tokenizer.ggml.bos_token_id".into(), MetaValue::Uint32(0)),
        ]);
        BPETokenizer::from_gguf_metadata(|key| metadata.get(key).cloned()).unwrap()
    }

    #[test]
    fn gemma4_bpe_normalizes_spaces_splits_newlines_and_forces_bos() {
        let tokenizer = gemma4_test_tokenizer();
        let ids = tokenizer.encode(
            " hello\nworld",
            EncodeOptions {
                add_special: false,
                parse_special: true,
            },
        );
        assert_eq!(ids[0], tokenizer.bos_id().unwrap());
        // The encoder normalizes a leading space to the SPM `▁` (U+2581)
        // marker; the decoder unescapes it back to a literal space.
        assert_eq!(
            tokenizer.decode_bytes(&ids[1..], true),
            " hello\nworld".as_bytes()
        );
    }

    #[test]
    fn gemma4_media_controls_are_single_controls_after_bos() {
        let tokenizer = gemma4_test_tokenizer();
        assert_eq!(
            tokenizer.encode(
                "<|image>",
                EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            ),
            vec![tokenizer.bos_id().unwrap(), 255_999]
        );
        assert_eq!(
            tokenizer.encode(
                "<audio|>",
                EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            ),
            vec![tokenizer.bos_id().unwrap(), 258_883]
        );
    }

    #[test]
    fn gemma4_raw_utf8_token_piece_does_not_use_gpt2_byte_decoder() {
        let tokenizer = gemma4_test_tokenizer();
        assert_eq!(tokenizer.decode_bytes(&[10], true), "Ā".as_bytes());
    }

    #[test]
    fn parse_special_controls_and_user_defined_match_llama() {
        let tokenizer = special_test_tokenizer();
        assert_eq!(
            tokenizer.encode(
                "a<|im_start|>b<tool_call>",
                EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            ),
            vec![0, 3, 1, 4],
        );
        assert_ne!(
            tokenizer.encode(
                "<|im_start|>",
                EncodeOptions {
                    add_special: false,
                    parse_special: false,
                },
            ),
            vec![3],
        );
        assert_eq!(
            tokenizer.encode(
                "<tool_call>",
                EncodeOptions {
                    add_special: false,
                    parse_special: false,
                },
            ),
            vec![4],
        );
    }

    #[test]
    fn special_partition_prefers_longest_literal() {
        let tokenizer = overlapping_special_test_tokenizer();
        assert_eq!(
            tokenizer.encode(
                "<tool_call>",
                EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            ),
            vec![7],
        );
    }

    #[test]
    fn add_special_follows_flags_not_id_presence() {
        let tokenizer = bos_eos_test_tokenizer(false, true);
        assert_eq!(
            tokenizer.encode(
                "a",
                EncodeOptions {
                    add_special: true,
                    parse_special: true,
                },
            ),
            vec![0, 9],
        );
        assert_eq!(tokenizer.bos_id(), Some(8));
        assert_eq!(tokenizer.eos_id(), Some(9));
        assert!(!tokenizer.add_bos());
        assert!(tokenizer.add_eos());
    }

    #[test]
    fn malformed_token_type_is_rejected() {
        assert!(tokenizer_with_token_types(&[1])
            .unwrap_err()
            .contains("length"));
        assert!(tokenizer_with_token_types(&[1, 7])
            .unwrap_err()
            .contains("token type"));
    }

    #[test]
    fn semantic_literals_and_plain_names_are_distinct() {
        let tokenizer = special_test_tokenizer();
        let plain = EncodeOptions {
            add_special: false,
            parse_special: false,
        };
        let special = EncodeOptions {
            add_special: false,
            parse_special: true,
        };
        assert_eq!(tokenizer.encode("<|im_start|>", special), vec![3]);
        assert_ne!(tokenizer.encode("im_start", plain), vec![3]);
        assert_eq!(tokenizer.encode("<|endoftext|>", special), vec![13]);
        assert_eq!(tokenizer.encode("<tool_call>", plain), vec![4]);
        assert_eq!(tokenizer.special_token_id("im_start"), Some(3));
        assert_eq!(tokenizer.special_token_id("im_end"), Some(5));
        assert_eq!(tokenizer.special_token_id("image_pad"), Some(6));
        assert_eq!(tokenizer.special_token_id("vision_start"), Some(7));
        assert_eq!(tokenizer.special_token_id("vision_end"), Some(8));
        assert_eq!(tokenizer.special_token_id("endoftext"), Some(13));
    }

    #[test]
    fn asr_semantic_literals_resolve_to_token_ids() {
        let tokenizer = special_test_tokenizer();
        let audio_start_id = tokenizer.token_id("<|audio_start|>").unwrap();
        let audio_pad_id = tokenizer.token_id("<|audio_pad|>").unwrap();
        let audio_end_id = tokenizer.token_id("<|audio_end|>").unwrap();
        let asr_text_id = tokenizer.token_id("<asr_text>").unwrap();

        assert_eq!(
            tokenizer.special_token_id("audio_start"),
            Some(audio_start_id)
        );
        assert_eq!(tokenizer.special_token_id("audio_pad"), Some(audio_pad_id));
        assert_eq!(tokenizer.special_token_id("audio_end"), Some(audio_end_id));
        assert_eq!(tokenizer.special_token_id("asr_text"), Some(asr_text_id));
    }

    #[test]
    fn qwen25_omni_semantic_aliases_resolve_to_media_names() {
        let tokenizer = tokenizer_from_parts(
            &[
                "<|vision_bos|>",
                "<|vision_eos|>",
                "<|vision_pad|>",
                "<|IMAGE|>",
                "<|VIDEO|>",
                "<|audio_bos|>",
                "<|audio_eos|>",
                "<|AUDIO|>",
            ],
            &[3, 3, 3, 3, 3, 3, 3, 3],
            None,
            None,
            false,
            false,
        )
        .unwrap();

        assert_eq!(tokenizer.special_token_id("vision_start"), Some(0));
        assert_eq!(tokenizer.special_token_id("vision_end"), Some(1));
        assert_eq!(tokenizer.special_token_id("vision_pad"), Some(2));
        assert_eq!(tokenizer.special_token_id("image_pad"), Some(3));
        assert_eq!(tokenizer.special_token_id("video_pad"), Some(4));
        assert_eq!(tokenizer.special_token_id("audio_start"), Some(5));
        assert_eq!(tokenizer.special_token_id("audio_end"), Some(6));
        assert_eq!(tokenizer.special_token_id("audio_pad"), Some(7));
    }

    #[test]
    fn qwen_video_pad_semantic_resolves_to_its_token_id() {
        let tokenizer = BPETokenizer::from_qwen3_embedded_merges().unwrap();
        assert_eq!(tokenizer.special_token_id("video_pad"), Some(151_656));
    }

    #[test]
    fn normal_control_looking_literal_has_no_chatml_semantic_name() {
        let tokenizer = normal_control_literal_test_tokenizer();
        assert_eq!(tokenizer.token_id("<|im_start|>"), Some(1));
        assert_eq!(tokenizer.special_token_id("im_start"), None);
    }

    fn byte_fallback_test_tokenizer(piece: &str) -> BPETokenizer {
        let metadata: HashMap<String, MetaValue> = HashMap::from([
            (
                "tokenizer.ggml.model".into(),
                MetaValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.pre".into(),
                MetaValue::String("qwen2".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                MetaValue::Array(MetaValueType::String, vec![MetaValue::String(piece.into())]),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                MetaValue::Array(MetaValueType::Uint32, vec![MetaValue::Uint32(6)]),
            ),
            (
                "tokenizer.ggml.merges".into(),
                MetaValue::Array(MetaValueType::String, Vec::new()),
            ),
            ("tokenizer.ggml.byte_fallback".into(), MetaValue::Bool(true)),
        ]);
        BPETokenizer::from_gguf_metadata(|key| metadata.get(key).cloned()).unwrap()
    }

    fn tokenizer_from_env(name: &str) -> BPETokenizer {
        let path = std::env::var(name).unwrap();
        let loader = crate::core::loader::GGUFLoader::from_file(&path).unwrap();
        BPETokenizer::from_gguf_metadata(|key| loader.metadata(key).cloned()).unwrap()
    }

    #[test]
    fn six_character_byte_fallback_decodes() {
        let tokenizer = byte_fallback_test_tokenizer("<0xF0>");
        assert_eq!(tokenizer.decode_bytes(&[0], false), vec![0xF0]);
    }

    #[test]
    fn render_special_controls_only_when_requested() {
        let tokenizer = special_test_tokenizer();
        let im_start = tokenizer.special_token_id("im_start").unwrap();
        assert_eq!(tokenizer.decode_bytes(&[im_start], false), b"");
        assert_eq!(tokenizer.decode_bytes(&[im_start], true), b"<|im_start|>");
        let tool = tokenizer.token_id("<tool_call>").unwrap();
        assert_eq!(tokenizer.decode_bytes(&[tool], false), b"<tool_call>");
    }

    #[test]
    #[ignore = "requires RMI_QWEN3_MODEL"]
    fn decode_bytes_joins_split_utf8_pieces() {
        let tokenizer = tokenizer_from_env("RMI_QWEN3_MODEL");
        assert_eq!(tokenizer.decode(&[9284, 104, 254], false), "🫠");
        assert_eq!(tokenizer.decode(&[124596, 252], false), "𝄞");
    }

    #[test]
    #[ignore = "requires RMI_QWEN3_MODEL"]
    fn streaming_decoder_buffers_incomplete_utf8() {
        let tokenizer = tokenizer_from_env("RMI_QWEN3_MODEL");
        let mut decoder = tokenizer.streaming_decoder(false);
        assert_eq!(decoder.push(9284), "");
        assert_eq!(decoder.push(104), "");
        assert_eq!(decoder.push(254), "🫠");
        assert_eq!(decoder.finish(), "");

        let mut decoder = tokenizer.streaming_decoder(false);
        assert_eq!(decoder.push(124596), "");
        assert_eq!(decoder.push(252), "𝄞");
        assert_eq!(decoder.finish(), "");
    }
}

// ============================================================================
// WordPiece tokenizer (tokenizer.ggml.model = "bert")
//
// Port of llama.cpp's `llm_tokenizer_wpm` from `src/llama-vocab.cpp`.
//
// Algorithm (per llama.cpp):
//   1. Normalize: Unicode NFD, drop combining marks when strip_accents,
//      lowercase when `lowercase`, split whitespace, punctuation and CJK
//      (dropping NUL / U+FFFD / control code points).
//   2. Prepend the phantom space U+2581 ("▁") to each word.
//   3. Greedy longest-match left-to-right against the vocab, advancing one
//      character per failed position. A word that matches nothing at all
//      emits `unk`.
//   4. `add_special` wraps [CLS]/[SEP], independently of add_bos/add_eos.
//      CONTROL/USER_DEFINED literals are partitioned before normalization.
//
// Unlike a HF BERT loader this never emits `##` continuation pieces: the
// converted GGUF vocab uses the `▁`-prefixed SentencePiece-style spelling
// (verified: 23 701 of 30 528 tokens in embeddinggemma/jina-v2 start with
// `▁`, zero start with `##`).
// ============================================================================

fn partition_special<'a>(
    special_tokens: &[SpecialToken],
    text: &'a str,
    parse_special: bool,
) -> Vec<Fragment<'a>> {
    let mut fragments = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        let best = special_tokens
            .iter()
            .filter(|token| {
                parse_special || !matches!(token.kind, TokenType::Control | TokenType::Unknown)
            })
            .filter_map(|token| {
                remaining
                    .find(&token.text)
                    .map(|position| (position, token))
            })
            .min_by(|(left_pos, left), (right_pos, right)| {
                left_pos
                    .cmp(right_pos)
                    .then_with(|| right.text.len().cmp(&left.text.len()))
            });

        let Some((position, token)) = best else {
            fragments.push(Fragment::Text(remaining));
            break;
        };
        if position > 0 {
            fragments.push(Fragment::Text(&remaining[..position]));
        }
        fragments.push(Fragment::Special(token.id));
        remaining = &remaining[position + token.text.len()..];
    }

    fragments
}

/// WordPiece tokenizer for `tokenizer.ggml.model = "bert"` GGUFs.
pub struct WPMTokenizer {
    tokens: Vec<String>,
    special_tokens: Vec<SpecialToken>,
    token_to_id: std::collections::HashMap<String, u32>,
    /// Longest token length in bytes, used to bound the match loop.
    max_token_len: usize,
    bos_id: Option<u32>,
    unk_id: Option<u32>,
    cls_id: Option<u32>,
    sep_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    add_sep: bool,
    lowercase: bool,
    strip_accents: bool,
}

impl WPMTokenizer {
    pub fn from_gguf_metadata(
        get_meta: impl Fn(&str) -> Option<MetaValue>,
    ) -> Result<Self, String> {
        match get_meta("tokenizer.ggml.model") {
            Some(MetaValue::String(value)) if value == "bert" => {}
            Some(MetaValue::String(value)) => {
                return Err(format!(
                    "Unsupported WPM tokenizer.ggml.model {value:?}; expected bert"
                ));
            }
            _ => return Err("Missing or invalid tokenizer.ggml.model".into()),
        }

        let tokens = string_array(get_meta("tokenizer.ggml.tokens"), "tokenizer.ggml.tokens")?;
        let n_tokens = tokens.len();
        let kinds = match get_meta("tokenizer.ggml.token_type") {
            Some(meta) => integer_array(Some(meta), "tokenizer.ggml.token_type")?,
            None => vec![1; n_tokens],
        };
        if kinds.len() != n_tokens {
            return Err("tokenizer.ggml.token_type length must match vocabulary".into());
        }
        let mut special_tokens = Vec::new();
        for (id, (text, kind)) in tokens.iter().zip(kinds).enumerate() {
            let kind = token_type(kind)?;
            if !text.is_empty()
                && matches!(
                    kind,
                    TokenType::Control | TokenType::Unknown | TokenType::UserDefined
                )
            {
                special_tokens.push(SpecialToken {
                    text: text.clone(),
                    id: id as u32,
                    kind,
                });
            }
        }
        let token_to_id: std::collections::HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(id, text)| (text.clone(), id as u32))
            .collect();
        let max_token_len = tokens
            .iter()
            .map(|token| token.chars().count())
            .max()
            .unwrap_or(0);

        // Defaults come from llama.cpp's `bert` vocabulary branch
        // (`llama-vocab.cpp:1982-1996`): bos=[CLS]=101, sep=[SEP]=102,
        // unk=100, and `add_sep = true`. The GGUF's `cls_token_id` happens to
        // agree (101) but is NOT read into `special_bos_id` — the WPM branch
        // uses the model-type default, so it must not be used as the turn-on.
        let bos_id = optional_token_id(
            get_meta("tokenizer.ggml.bos_token_id"),
            "tokenizer.ggml.bos_token_id",
        )?
        .or(Some(101));
        let unk_id = optional_token_id(
            get_meta("tokenizer.ggml.unknown_token_id"),
            "tokenizer.ggml.unknown_token_id",
        )?
        .or(Some(100));
        let sep_id = optional_token_id(
            get_meta("tokenizer.ggml.seperator_token_id")
                .or_else(|| get_meta("tokenizer.ggml.sep_token_id")),
            "tokenizer.ggml.sep_token_id",
        )?
        .or(Some(102));
        let cls_id = optional_token_id(
            get_meta("tokenizer.ggml.cls_token_id"),
            "tokenizer.ggml.cls_token_id",
        )?
        // `tokenizer.ggml.cls_token_id` is deprecated — current llama.cpp no
        // longer reads it at all. For the bert family the `[CLS]` id lives in
        // `special_bos_id`, whose model-type default is 101
        // (`llama-vocab.cpp:1982-1996`), so that is the fallback. Without it a
        // GGUF that omits the key (nomic-embed-text-v1.5 does) leaves
        // `cls_id = None` and `[CLS]` leaks back out of `decode`.
        .or(bos_id);
        validate_token_id(bos_id, n_tokens, "tokenizer.ggml.bos_token_id")?;
        validate_token_id(unk_id, n_tokens, "tokenizer.ggml.unknown_token_id")?;
        validate_token_id(sep_id, n_tokens, "tokenizer.ggml.sep_token_id")?;
        validate_token_id(cls_id, n_tokens, "tokenizer.ggml.cls_token_id")?;

        let add_sep = match get_meta("tokenizer.ggml.add_sep_token") {
            Some(MetaValue::Bool(v)) => v,
            _ => true,
        };
        let _ = add_sep;
        // The BERT WPM path ignores `add_bos`/`add_eos` when wrapping specials
        // (`llama-vocab.cpp:3521-3543`), but keep them for introspection.
        let add_bos = match get_meta("tokenizer.ggml.add_bos_token") {
            Some(MetaValue::Bool(v)) => v,
            _ => false,
        };
        let add_eos = bool_meta(
            get_meta("tokenizer.ggml.add_eos_token"),
            "tokenizer.ggml.add_eos_token",
        )?;

        // `normalizer_lowercase` defaults to true in llama.cpp and drives
        // `strip_accents` unless the latter is set explicitly.
        let lowercase = match get_meta("tokenizer.ggml.normalizer_lowercase") {
            Some(MetaValue::Bool(v)) => v,
            _ => true,
        };
        let strip_accents = match get_meta("tokenizer.ggml.normalizer_strip_accents") {
            Some(MetaValue::Bool(v)) => v,
            _ => lowercase,
        };

        Ok(Self {
            tokens,
            special_tokens,
            token_to_id,
            max_token_len,
            bos_id,
            unk_id,
            cls_id,
            sep_id,
            add_bos,
            add_eos,
            add_sep,
            lowercase,
            strip_accents,
        })
    }

    /// Leading special: `special_bos_id`, which for the bert family is
    /// `[CLS]` (`llama-vocab.cpp:1982-1996`).
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    pub fn unk_id(&self) -> Option<u32> {
        self.unk_id
    }

    pub fn cls_id(&self) -> Option<u32> {
        self.cls_id
    }

    pub fn sep_id(&self) -> Option<u32> {
        self.sep_id
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Encode `text`, applying the [CLS]…[SEP] wrapper the encoder expects.
    /// `llama-vocab.cpp:3521-3543` gates both the leading `special_bos_id`
    /// (=[CLS]) and the trailing `special_sep_id` (=[SEP]) on `add_special`
    /// alone — not on `add_bos` / `add_eos`, which the BERT family never sets.
    pub fn encode(&self, text: &str, options: EncodeOptions) -> Vec<u32> {
        let mut output = Vec::new();
        if options.add_special {
            if let Some(bos) = self.bos_id {
                output.push(bos);
            }
        }
        for fragment in partition_special(&self.special_tokens, text, options.parse_special) {
            match fragment {
                Fragment::Special(id) => output.push(id),
                Fragment::Text(text) => {
                    for word in self.preprocess(text) {
                        self.encode_word(&word, &mut output);
                    }
                }
            }
        }
        if options.add_special {
            if let Some(sep) = self.sep_id {
                output.push(sep);
            }
        }
        output
    }

    /// Decode back to text, dropping the phantom-space prefix.
    pub fn decode(&self, ids: &[u32], render_special: bool) -> String {
        let mut text = String::new();
        for &id in ids {
            let token = match self.tokens.get(id as usize) {
                Some(token) => token.as_str(),
                None => continue,
            };
            if !render_special
                && (Some(id) == self.cls_id || Some(id) == self.sep_id || Some(id) == self.unk_id)
            {
                continue;
            }
            match token.strip_prefix('\u{2581}') {
                Some(rest) => {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(rest);
                }
                None => text.push_str(token),
            }
        }
        text
    }

    /// Reuse the installed BERT normalizer and pre-tokenizer (NFD, CJK and punctuation).
    /// This is the production path — it matches `llama.cpp` b96806d9 exactly.
    fn preprocess(&self, text: &str) -> Vec<String> {
        use tokenizers::{
            NormalizedString, Normalizer, OffsetReferential, OffsetType, PreTokenizedString,
            PreTokenizer,
        };
        let mut normalized = NormalizedString::from(text);
        tokenizers::normalizers::bert::BertNormalizer::new(
            true,
            true,
            Some(self.strip_accents),
            self.lowercase,
        )
        .normalize(&mut normalized)
        .expect("BERT normalization");
        let mut split = PreTokenizedString::from(normalized);
        tokenizers::pre_tokenizers::bert::BertPreTokenizer
            .pre_tokenize(&mut split)
            .expect("BERT pre-tokenization");
        split
            .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
            .into_iter()
            .map(|(word, _, _)| word.to_owned())
            .collect()
    }

    /// Pre-`tokenizers`-crate BERT pre-tokenizer (HF-free escape hatch reference).
    ///
    /// **NOT production-ready**: this implementation has known gaps vs the HF path
    /// above and vs `llama.cpp` b96806d9. It is kept here as a reference for
    /// future maintainers, **not** as a drop-in replacement.
    ///
    /// ## What this does correctly
    /// - Lowercase (when `self.lowercase == true`).
    /// - Strip combining marks after NFD-like reduction (when `self.strip_accents`).
    /// - Split on Unicode whitespace.
    /// - Drop NUL / U+FFFD / control code points.
    ///
    /// ## What this gets wrong (vs `preprocess` / `llama.cpp`)
    /// - **No punctuation splitting**: `"France?"` becomes one word, not two.
    ///   The HF `BertPreTokenizer` splits on punctuation; this version does not.
    /// - **No CJK per-character splitting**: `"世界"` stays as one word, not two.
    ///   The HF `BertNormalizer` wraps CJK in spaces via `handle_chinese_chars`,
    ///   then the pre-tokenizer splits per character.
    /// - **NFD is implicit**: it relies on `is_combining_mark` to drop accents
    ///   without an explicit `unicode-normalization` NFD pass first. For most
    ///   ASCII-with-diacritics input this matches, but precomposed sequences
    ///   that decompose to multiple non-combining codepoints are not folded.
    ///
    /// ## When to reach for this
    /// Only if the `tokenizers` crate dependency must be removed entirely for
    /// licensing / supply-chain / compile-time reasons. In that case, the right
    /// fix is to **rewrite** this function (or its replacement) with proper
    /// punctuation + CJK splitting, then swap `preprocess` to call it. Do not
    /// simply uncomment the call — the output diverges from `llama.cpp`.
    ///
    /// `#[allow(dead_code)]` is intentional: the production encoder never calls
    /// this. The function and its helper `is_combining_mark` exist purely so
    /// the historical algorithm stays reviewable.
    #[allow(dead_code)]
    fn preprocess_legacy(&self, text: &str) -> Vec<String> {
        let mut words: Vec<String> = Vec::new();
        let mut current = String::new();
        for value in text.chars() {
            let mapped = if self.lowercase {
                value.to_lowercase().to_string()
            } else {
                value.to_string()
            };
            for mapped_value in mapped.chars() {
                if self.strip_accents && is_combining_mark(mapped_value) {
                    continue;
                }
                if mapped_value.is_whitespace() {
                    if !current.is_empty() {
                        words.push(std::mem::take(&mut current));
                    }
                    continue;
                }
                if mapped_value == '\0' || mapped_value == '\u{FFFD}' || mapped_value.is_control() {
                    continue;
                }
                current.push(mapped_value);
            }
        }
        if !current.is_empty() {
            words.push(current);
        }
        words
    }

    /// Greedy longest-match one word, prefixed with the phantom space.
    fn encode_word(&self, word: &str, output: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }
        let pieces: Vec<char> = std::iter::once('\u{2581}').chain(word.chars()).collect();
        let n = pieces.len();
        let before = output.len();
        let mut i = 0usize;
        while i < n {
            let mut matched = None;
            let limit = (i + self.max_token_len + 1).min(n);
            for j in (i + 1..=limit).rev() {
                let candidate: String = pieces[i..j].iter().collect();
                if let Some(&id) = self.token_to_id.get(&candidate) {
                    matched = Some((id, j));
                    break;
                }
            }
            match matched {
                Some((id, next)) => {
                    output.push(id);
                    i = next;
                }
                None => {
                    // `llama-vocab.cpp:829-831`: on the first missed position
                    // the oracle does `output.resize(current_tokens); break;`
                    // — it rolls the word back and stops scanning it, but it
                    // does NOT return. Control falls through to the check
                    // below, where `current_tokens == output.size()` is now
                    // true, so the word becomes exactly one `[UNK]`.
                    //
                    // Rust's `return` here would skip that check and drop the
                    // word entirely, which is what this did until
                    // 2026-09-29: `"🎉🎊"` tokenized to `[CLS, SEP]` and
                    // `"hello 🎉 world"` to `hello world`, silently losing the
                    // out-of-vocabulary word instead of emitting `[UNK]`.
                    output.truncate(before);
                    break;
                }
            }
        }
        if output.len() == before {
            if let Some(unk) = self.unk_id {
                output.push(unk);
            }
        }
    }
}

/// Helper for [`WPMTokenizer::preprocess_legacy`]: does the codepoint classify
/// as a Unicode combining mark? Used only by the HF-free escape-hatch path;
/// the production [`WPMTokenizer::preprocess`] delegates to
/// `tokenizers::BertNormalizer`, which performs its own NFD + combining-mark
/// stripping internally. Kept here so `preprocess_legacy` can stay bit-for-bit
/// equivalent to the algorithm it replaced.
#[allow(dead_code)]
fn is_combining_mark(value: char) -> bool {
    use unicode_categories::UnicodeCategories;
    value.is_mark_nonspacing() || value.is_mark_enclosing()
}

#[cfg(test)]
mod wpm_unk_tests {
    use super::{EncodeOptions, SpecialToken, TokenType, WPMTokenizer};

    /// Minimal WordPiece vocab covering the ASCII leaves plus the two marker
    /// tokens, so the out-of-vocabulary path can be exercised without a GGUF.
    fn tokenizer() -> WPMTokenizer {
        let tokens: Vec<String> = [
            "[UNK]", "[CLS]", "[SEP]", "▁hello", "▁world", "hello", "world",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let token_to_id = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        WPMTokenizer {
            max_token_len: tokens.iter().map(|t| t.len()).max().unwrap_or(0),
            tokens,
            special_tokens: vec![SpecialToken {
                text: "[SEP]".into(),
                id: 2,
                kind: TokenType::Control,
            }],
            token_to_id,
            bos_id: Some(1),
            unk_id: Some(0),
            cls_id: Some(1),
            sep_id: Some(2),
            add_bos: true,
            add_eos: true,
            add_sep: true,
            lowercase: true,
            strip_accents: false,
        }
    }

    fn encode(text: &str) -> Vec<u32> {
        tokenizer().encode(
            text,
            EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        )
    }

    #[test]
    fn a_word_with_no_matching_piece_becomes_a_single_unk() {
        // `llama-vocab.cpp:829-833`: the oracle rolls the word back with
        // `output.resize(current_tokens)` then *falls through* to the
        // `current_tokens == output.size()` check and pushes exactly one
        // `token_unk()`. Our version used to `return` there, dropping the word
        // entirely, so "🎉🎊" came out as just [CLS, SEP].
        let unk = 0u32;
        let cls = 1u32;
        let sep = 2u32;
        assert_eq!(encode("🎉🎊"), vec![cls, unk, sep]);
    }

    #[test]
    fn an_oov_word_between_matching_words_still_gets_an_unk() {
        // hello (matches) + 🎉 (no match -> one UNK) + world (matches). Before
        // the fix this was `[CLS, ▁hello, ▁world, SEP]`, silently losing the
        // emoji instead of emitting [UNK].
        assert_eq!(encode("hello 🎉 world"), vec![1, 3, 0, 4, 2]);
    }

    #[test]
    fn a_partially_matching_word_collapses_to_one_unk() {
        // "hello🎉" has a matching prefix then an unmatchable character. The
        // oracle truncates the partial match and emits one UNK for the whole
        // word, not a mix of real tokens plus UNK.
        assert_eq!(encode("hello🎉"), vec![1, 0, 2]);
    }

    #[test]
    fn whitespace_only_and_empty_inputs_produce_no_unk() {
        // Whitespace never forms a word in the first place
        // (`llama-vocab.cpp:849-853` skips empty words), so neither case has
        // anything to fall back to.
        assert_eq!(encode("   "), vec![1, 2]);
        assert_eq!(encode(""), vec![1, 2]);
    }

    #[test]
    fn a_fully_matching_word_gets_no_extra_unk() {
        assert_eq!(encode("hello"), vec![1, 3, 2]);
        assert_eq!(encode("hello world"), vec![1, 3, 4, 2]);
    }

    #[test]
    fn wordpiece_normalizes_accents_and_splits_punctuation_and_cjk() {
        let mut tok = tokenizer();
        tok.strip_accents = true;
        assert_eq!(
            tok.preprocess("  Héllo\t世界! Café e\u{301}\n🙂"),
            ["hello", "世", "界", "!", "cafe", "e", "🙂"]
        );
    }
}

/// Re-export the UGM (SentencePiece unigram) tokenizer at the same flat
/// path level as BPE/SPM/WPM. `load_tokenizer` above routes
/// `tokenizer.ggml.model = "t5"` GGUFs here.
pub use ugm::{UgmError, UgmTokenizer};
