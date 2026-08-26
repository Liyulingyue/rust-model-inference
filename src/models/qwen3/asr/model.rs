use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::format::ggufrs::{ComponentRole, GgufrsFile};
use crate::core::tensor::{MetaValue, TensorSource};
use crate::models::qwen3::asr::audio_processor::{decode_pcm16_wav, log_mel_windows, AsrAudioError};
use crate::models::qwen3::asr::mel_encoder::{validate_qwen3a_source, AudioEmbeddings, Qwen3AudioModel};
use crate::models::qwen3::base_multimodal::{Qwen3GenerateOptions, Qwen3Input, Qwen3Model};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

const LANGUAGES: &[(&str, &str)] = &[
    ("Chinese", "zh"),
    ("English", "en"),
    ("Cantonese", "yue"),
    ("Arabic", "ar"),
    ("German", "de"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("Portuguese", "pt"),
    ("Indonesian", "id"),
    ("Italian", "it"),
    ("Korean", "ko"),
    ("Russian", "ru"),
    ("Thai", "th"),
    ("Vietnamese", "vi"),
    ("Japanese", "ja"),
    ("Turkish", "tr"),
    ("Hindi", "hi"),
    ("Malay", "ms"),
    ("Dutch", "nl"),
    ("Swedish", "sv"),
    ("Danish", "da"),
    ("Finnish", "fi"),
    ("Polish", "pl"),
    ("Czech", "cs"),
    ("Filipino", "fil"),
    ("Persian", "fa"),
    ("Greek", "el"),
    ("Romanian", "ro"),
    ("Hungarian", "hu"),
    ("Macedonian", "mk"),
];

const DETECTED_ONLY_LANGUAGES: &[&str] = &[
    "Anhui",
    "Dongbei",
    "Fujian",
    "Gansu",
    "Guizhou",
    "Hebei",
    "Henan",
    "Hubei",
    "Hunan",
    "Jiangxi",
    "Ningxia",
    "Shandong",
    "Shaanxi",
    "Shanxi",
    "Sichuan",
    "Tianjin",
    "Yunnan",
    "Zhejiang",
    "Cantonese (Hong Kong accent)",
    "Cantonese (Guangdong accent)",
    "Wu language",
    "Minnan language",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsrErrorKind {
    UnsupportedAudio,
    Unprocessable,
    Internal,
}

#[derive(Debug)]
pub struct AsrError {
    pub kind: AsrErrorKind,
    pub message: String,
}

impl std::fmt::Display for AsrError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for AsrError {}

#[derive(Debug, Clone)]
pub struct TranscriptionOptions {
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub max_new_tokens: usize,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            language: None,
            prompt: None,
            max_new_tokens: 256,
        }
    }
}

pub struct Transcription {
    pub text: String,
    pub language: Option<String>,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub audio_tokens: usize,
}

pub struct AsrRuntime {
    decoder: Arc<Qwen3Model>,
    audio: Qwen3AudioModel,
}

pub fn open_bundled_audio_source(
    model_path: &Path,
) -> Result<Option<Arc<dyn TensorSource>>, String> {
    let mut file = File::open(model_path)
        .map_err(|error| format!("Failed to open {}: {error}", model_path.display()))?;
    let mut magic = [0; 8];
    file.read_exact(&mut magic)
        .map_err(|error| format!("Failed to read {} magic: {error}", model_path.display()))?;
    if &magic != b"GGUFRS\0\0" {
        return Ok(None);
    }

    let package = GgufrsFile::open(model_path).map_err(|error| error.to_string())?;
    if package.component_id(ComponentRole::Mmproj).is_none() {
        return Ok(None);
    }
    let source = package
        .load_component(ComponentRole::Mmproj)
        .map_err(|error| error.to_string())?;
    match source.metadata("clip.has_audio_encoder") {
        None | Some(MetaValue::Bool(false)) => return Ok(None),
        Some(MetaValue::Bool(true)) => {}
        Some(_) => return Err("Invalid clip.has_audio_encoder: expected bool".into()),
    }
    match source.metadata("clip.audio.projector_type") {
        Some(MetaValue::String(value)) if value == "qwen3a" => {}
        _ => return Err("Invalid clip.audio.projector_type: expected qwen3a".into()),
    }
    validate_qwen3a_source(&source)?;
    Ok(Some(Arc::new(source)))
}

impl AsrRuntime {
    pub fn new(
        decoder: Arc<Qwen3Model>,
        audio_source: Arc<dyn TensorSource>,
    ) -> Result<Self, AsrError> {
        let audio = Qwen3AudioModel::from_source(audio_source, decoder.pool()).map_err(internal)?;
        if audio.config().projection != decoder.config().n_embd {
            return Err(internal(format!(
                "Audio projection width {} does not match decoder embedding width {}",
                audio.config().projection,
                decoder.config().n_embd
            )));
        }
        Ok(Self { decoder, audio })
    }

    pub fn transcribe_wav(
        &self,
        wav: &[u8],
        options: &TranscriptionOptions,
    ) -> Result<Transcription, AsrError> {
        let forced_language = normalize_language(options.language.as_deref())?;
        if options.max_new_tokens == 0 {
            return Err(unprocessable("max_new_tokens must be greater than zero"));
        }
        if options
            .prompt
            .as_deref()
            .is_some_and(|prompt| self.decoder.tokenizer().contains_special_literal(prompt))
        {
            return Err(unprocessable(
                "System prompt contains a tokenizer control literal",
            ));
        }

        let t0 = std::time::Instant::now();
        let samples = decode_pcm16_wav(wav).map_err(map_audio_error)?;
        let t1 = std::time::Instant::now();
        let windows = log_mel_windows(&samples).map_err(map_audio_error)?;
        let t2 = std::time::Instant::now();
        eprintln!(
            "    [asr-timing] audio: {} mel windows for {} samples",
            windows.len(),
            samples.len(),
        );
        let audio = self.audio.encode(&windows).map_err(internal)?;
        let t3 = std::time::Instant::now();
        let prompt = build_asr_prompt(
            self.decoder.tokenizer(),
            self.decoder.config().n_ctx,
            audio.tokens,
            options.prompt.as_deref(),
            forced_language,
        )?;
        validate_generation_context(
            prompt.token_ids.len(),
            options.max_new_tokens,
            self.decoder.config().n_ctx,
        )?;
        let embeddings = replace_audio_embeddings(&self.decoder, &prompt, &audio)?;
        let t4 = std::time::Instant::now();
        let generation = self
            .decoder
            .generate_asr(
                Qwen3Input {
                    token_ids: &prompt.token_ids,
                    positions: &prompt.positions,
                    embeddings: Some(&embeddings),
                },
                Qwen3GenerateOptions {
                    max_new_tokens: options.max_new_tokens,
                    temperature: 0.0,
                },
            )
            .map_err(internal)?;
        let t5 = std::time::Instant::now();
        let (text, language) = parse_model_output(&generation.text, forced_language)?;
        eprintln!(
            "    [asr-timing] decode_wav={:.3}s mel={:.3}s audio_encode={:.3}s prompt+embed={:.3}s llm_generate={:.3}s (prefill+decode)",
            (t1 - t0).as_secs_f64(),
            (t2 - t1).as_secs_f64(),
            (t3 - t2).as_secs_f64(),
            (t4 - t3).as_secs_f64(),
            (t5 - t4).as_secs_f64(),
        );
        Ok(Transcription {
            text,
            language,
            token_ids: generation.token_ids,
            prompt_tokens: generation.prompt_tokens,
            audio_tokens: audio.tokens,
        })
    }
}

fn validate_generation_context(
    prompt_tokens: usize,
    max_new_tokens: usize,
    decoder_context: usize,
) -> Result<(), AsrError> {
    let required = prompt_tokens
        .checked_add(max_new_tokens)
        .ok_or_else(|| unprocessable("ASR context length overflow"))?;
    if required > decoder_context {
        return Err(unprocessable(format!(
            "ASR requires {required} tokens; decoder context is {decoder_context}"
        )));
    }
    Ok(())
}

pub fn normalize_language(value: Option<&str>) -> Result<Option<&'static str>, AsrError> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    LANGUAGES
        .iter()
        .find(|(name, code)| value.eq_ignore_ascii_case(name) || value.eq_ignore_ascii_case(code))
        .map(|(name, _)| Some(*name))
        .ok_or_else(|| unprocessable(format!("Unsupported ASR language: {value}")))
}

struct AsrPrompt {
    token_ids: Vec<u32>,
    positions: Vec<[usize; 4]>,
}

fn build_asr_prompt(
    tokenizer: &BPETokenizer,
    decoder_context: usize,
    audio_tokens: usize,
    system_prompt: Option<&str>,
    forced_language: Option<&'static str>,
) -> Result<AsrPrompt, AsrError> {
    if audio_tokens >= decoder_context {
        return Err(unprocessable(format!(
            "Audio token count {audio_tokens} must be below decoder context {decoder_context}"
        )));
    }
    let system_prompt = system_prompt.unwrap_or_default();
    if tokenizer.contains_special_literal(system_prompt) {
        return Err(unprocessable(
            "System prompt contains a tokenizer control literal",
        ));
    }

    let pad_len = "<|audio_pad|>"
        .len()
        .checked_mul(audio_tokens)
        .ok_or_else(|| unprocessable("Audio placeholder length overflow"))?;
    let audio_pads = "<|audio_pad|>".repeat(audio_tokens);
    if audio_pads.len() != pad_len {
        return Err(unprocessable("Audio placeholder length mismatch"));
    }
    let assistant_prefill = forced_language
        .map(|name| format!("language {name}<asr_text>"))
        .unwrap_or_default();
    let fixed_len = "<|im_start|>system\n<|im_end|>\n\n<|im_start|>user\n<|audio_start|><|audio_end|><|im_end|>\n<|im_start|>assistant\n".len();
    fixed_len
        .checked_add(system_prompt.len())
        .and_then(|len| len.checked_add(audio_pads.len()))
        .and_then(|len| len.checked_add(assistant_prefill.len()))
        .ok_or_else(|| unprocessable("ASR prompt length overflow"))?;
    let system_text = format!("<|im_start|>system\n{system_prompt}<|im_end|>\n");
    let user_text = format!(
        "\n<|im_start|>user\n<|audio_start|>{}<|audio_end|><|im_end|>\n\
         <|im_start|>assistant\n{}",
        audio_pads, assistant_prefill,
    );

    let semantic = |name| {
        tokenizer
            .special_token_id(name)
            .ok_or_else(|| internal(format!("Tokenizer is missing required {name} token")))
    };
    let im_start = semantic("im_start")?;
    let im_end = semantic("im_end")?;
    let audio_start = semantic("audio_start")?;
    let audio_pad = semantic("audio_pad")?;
    let audio_end = semantic("audio_end")?;
    let asr_text = semantic("asr_text")?;
    let encode_options = EncodeOptions {
        add_special: false,
        parse_special: true,
    };
    let mut token_ids = tokenizer.encode(&system_text, encode_options);
    token_ids.extend(tokenizer.encode(&user_text, encode_options));
    let count = |token| {
        token_ids
            .iter()
            .filter(|&&candidate| candidate == token)
            .count()
    };
    let start = token_ids.iter().position(|&token| token == audio_start);
    let end = token_ids.iter().position(|&token| token == audio_end);
    if count(im_start) != 3
        || count(im_end) != 2
        || count(audio_start) != 1
        || count(audio_end) != 1
        || count(audio_pad) != audio_tokens
        || count(asr_text) != usize::from(forced_language.is_some())
        || !matches!((start, end), (Some(start), Some(end)) if start < end
            && token_ids[start + 1..end].iter().all(|&token| token == audio_pad)
            && end - start - 1 == audio_tokens)
    {
        return Err(internal("Tokenizer violated the ASR prompt protocol"));
    }
    if token_ids.is_empty() || token_ids.len() > decoder_context {
        return Err(unprocessable("ASR prompt exceeds decoder context"));
    }
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(token_ids.len())
        .map_err(|_| unprocessable("ASR position allocation failed"))?;
    positions.extend((0..token_ids.len()).map(|index| {
        if matches!((start, end), (Some(start), Some(end)) if start < index && index < end) {
            [index; 4]
        } else {
            [index, index, index, 0]
        }
    }));
    Ok(AsrPrompt {
        token_ids,
        positions,
    })
}

fn replace_audio_embeddings(
    decoder: &Qwen3Model,
    prompt: &AsrPrompt,
    audio: &AudioEmbeddings,
) -> Result<Vec<f32>, AsrError> {
    let dim = decoder.config().n_embd;
    let expected_audio_values = audio
        .tokens
        .checked_mul(audio.dim)
        .ok_or_else(|| internal("Audio embedding shape overflow"))?;
    if audio.dim != dim
        || audio.values.len() != expected_audio_values
        || audio.values.iter().any(|value| !value.is_finite())
        || prompt.positions.len() != prompt.token_ids.len()
    {
        return Err(internal("Invalid audio embedding protocol"));
    }
    let mut embeddings = decoder.embed_tokens(&prompt.token_ids).map_err(internal)?;
    let audio_pad = decoder
        .tokenizer()
        .special_token_id("audio_pad")
        .ok_or_else(|| internal("Tokenizer is missing required audio_pad token"))?;
    let pad_rows: Vec<usize> = prompt
        .token_ids
        .iter()
        .enumerate()
        .filter_map(|(index, &token)| (token == audio_pad).then_some(index))
        .collect();
    if pad_rows.len() != audio.tokens {
        return Err(internal(format!(
            "Audio pad count {} does not match audio token count {}",
            pad_rows.len(),
            audio.tokens
        )));
    }
    for (audio_row, prompt_row) in pad_rows.into_iter().enumerate() {
        let source_start = audio_row
            .checked_mul(dim)
            .ok_or_else(|| internal("Audio embedding offset overflow"))?;
        let source_end = source_start
            .checked_add(dim)
            .ok_or_else(|| internal("Audio embedding range overflow"))?;
        let destination_start = prompt_row
            .checked_mul(dim)
            .ok_or_else(|| internal("Decoder embedding offset overflow"))?;
        let destination_end = destination_start
            .checked_add(dim)
            .ok_or_else(|| internal("Decoder embedding range overflow"))?;
        let source = audio
            .values
            .get(source_start..source_end)
            .ok_or_else(|| internal("Invalid audio embedding range"))?;
        let destination = embeddings
            .get_mut(destination_start..destination_end)
            .ok_or_else(|| internal("Invalid decoder embedding range"))?;
        destination.copy_from_slice(source);
    }
    Ok(embeddings)
}

fn parse_model_output(
    output: &str,
    forced_language: Option<&'static str>,
) -> Result<(String, Option<String>), AsrError> {
    let output = trim_framing(output);
    if let Some(language) = forced_language {
        return Ok((output.to_string(), Some(language.to_string())));
    }
    let protocol = output
        .strip_prefix("language ")
        .ok_or_else(|| internal("ASR output is missing the language prefix"))?;
    let (language, transcript) = protocol
        .split_once("<asr_text>")
        .ok_or_else(|| internal("ASR output is missing the transcript marker"))?;
    let language = language.trim();
    let transcript = trim_framing(transcript).to_string();
    if language == "None" {
        if transcript.is_empty() {
            return Ok((transcript, None));
        }
        return Err(internal(
            "ASR returned language None with a nonempty transcript",
        ));
    }
    if !LANGUAGES.iter().any(|(name, _)| *name == language)
        && !DETECTED_ONLY_LANGUAGES.contains(&language)
    {
        return Err(internal(format!(
            "ASR returned unknown language: {language}"
        )));
    }
    Ok((transcript, Some(language.to_string())))
}

fn trim_framing(mut output: &str) -> &str {
    output = output.trim();
    loop {
        let trimmed = ["<|im_end|>", "<|endoftext|>"]
            .into_iter()
            .find_map(|marker| output.strip_suffix(marker));
        let Some(trimmed) = trimmed else {
            return output;
        };
        output = trimmed.trim_end();
    }
}

fn map_audio_error(error: AsrAudioError) -> AsrError {
    match error {
        AsrAudioError::Unsupported(message) => AsrError {
            kind: AsrErrorKind::UnsupportedAudio,
            message,
        },
        AsrAudioError::Invalid(message) => AsrError {
            kind: AsrErrorKind::Unprocessable,
            message,
        },
    }
}

fn unprocessable(message: impl Into<String>) -> AsrError {
    AsrError {
        kind: AsrErrorKind::Unprocessable,
        message: message.into(),
    }
}

fn internal(message: impl Into<String>) -> AsrError {
    AsrError {
        kind: AsrErrorKind::Internal,
        message: message.into(),
    }
}