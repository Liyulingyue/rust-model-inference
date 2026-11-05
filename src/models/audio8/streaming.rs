//! Audio8 ASR streaming schedule and chunk-by-chunk transcriber.
//!
//! The encoder/decoder live in [`super::text`] and [`super`]; this module owns
//! the simulated-streaming schedule (chunk size, pad tokens, mel-window slice)
//! so that the CLI and any future HTTP entry point share one implementation.

use super::mel::compute_voxtral_log_mel;
use super::text::{Audio8TextDecoder, Audio8TextSession};
use super::{time_condition, Audio8Encoder, Audio8State, TEXT_WIDTH};
use crate::core::tokenizer::BPETokenizer;
use crate::models::qwen3::asr::audio_processor::decode_pcm16_wav_any;
use std::collections::VecDeque;

pub const SAMPLE_RATE: usize = 16_000;
pub const SAMPLES_PER_TOKEN: usize = 1280;
pub const LEFT_PAD_TOKENS: usize = 18;
pub const DELAY_TOKENS: usize = 6;
pub const RIGHT_TEXT_TOKENS: usize = 10;
pub const LOOK_BACK_SAMPLES: usize = 840;
pub const LOOK_AHEAD_SAMPLES: usize = 40;

#[derive(Clone, Copy, Debug)]
pub struct Schedule;

impl Schedule {
    pub const fn new() -> Self {
        Self
    }

    pub const fn sample_rate(self) -> usize {
        SAMPLE_RATE
    }

    pub const fn samples_per_token(self) -> usize {
        SAMPLES_PER_TOKEN
    }

    pub const fn left_pad_tokens(self) -> usize {
        LEFT_PAD_TOKENS
    }

    pub const fn delay_tokens(self) -> usize {
        DELAY_TOKENS
    }

    pub const fn right_text_tokens(self) -> usize {
        RIGHT_TEXT_TOKENS
    }

    pub const fn look_back_samples(self) -> usize {
        LOOK_BACK_SAMPLES
    }

    pub const fn look_ahead_samples(self) -> usize {
        LOOK_AHEAD_SAMPLES
    }

    /// BOS + `(LEFT_PAD + DELAY)` streaming-pad tokens that prime the decoder
    /// before the first real audio chunk arrives.
    pub const fn initial_tokens(self) -> usize {
        LEFT_PAD_TOKENS + DELAY_TOKENS + 1
    }

    /// Trailing silence pad so the final decode step still has right-context.
    pub const fn right_tokens(self) -> usize {
        DELAY_TOKENS + 1 + RIGHT_TEXT_TOKENS
    }

    /// Decoder steps consumed by a `samples` buffer at this schedule.
    pub fn capacity_for(self, samples: &[f32]) -> usize {
        let initial = self.initial_tokens();
        let remaining = samples
            .len()
            .saturating_sub(initial * SAMPLES_PER_TOKEN + LOOK_AHEAD_SAMPLES)
            / SAMPLES_PER_TOKEN
            + 1;
        initial + remaining
    }

    /// Build the padded streaming buffer: leading silence, audio, trailing silence.
    pub fn padded_stream(self, samples: &[f32]) -> Vec<f32> {
        let mut stream = vec![0.0; LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN];
        stream.extend_from_slice(samples);
        stream.resize(stream.len() + self.right_tokens() * SAMPLES_PER_TOKEN, 0.0);
        stream
    }
}

impl Default for Schedule {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode a 16 kHz mono PCM stream from raw WAV bytes, with channel
/// downmix and `linear_resample` for non-16 kHz sources. Used by both the
/// CLI (`--audio` path) and the HTTP `/v1/audio/transcriptions` endpoint
/// so the audio contract is identical across entry points.
pub fn decode_samples(wav_bytes: &[u8]) -> Result<Vec<f32>, String> {
    let decoded = decode_pcm16_wav_any(wav_bytes).map_err(|error| format!("{error:?}"))?;
    let samples = if decoded.channels == 1 {
        decoded.samples
    } else {
        decoded
            .samples
            .chunks(decoded.channels as usize)
            .map(|channels| channels.iter().sum::<f32>() / channels.len() as f32)
            .collect()
    };
    if samples.is_empty() {
        return Err("Audio8 WAV contains no audio".into());
    }
    let samples = if decoded.sample_rate as usize == SAMPLE_RATE {
        samples
    } else {
        crate::models::funasr::model::linear_resample(
            &samples,
            decoded.sample_rate as usize,
            SAMPLE_RATE,
        )
    };
    Ok(samples)
}

/// Audio8 tokenizer specials resolved once per model load.
#[derive(Clone)]
pub struct SpecialTokens {
    pub bos: u32,
    pub eos: u32,
    pub stream_pad: u32,
    pub stream_word: u32,
    pub pad: u32,
    pub language_zh: u32,
    pub language_en: u32,
}

impl SpecialTokens {
    pub fn lookup(tokenizer: &BPETokenizer) -> Result<Self, String> {
        let find = |literal: &str| {
            tokenizer
                .token_id(literal)
                .ok_or_else(|| format!("Audio8 tokenizer is missing {literal}"))
        };
        // Special tokens are emitted as raw bytes to avoid double-encoding in
        // source files; concat! builds the four `|im_*| ` markers at compile time.
        Ok(Self {
            bos: find(concat!("<", "|im_", "start", "|", ">"))?,
            eos: find(concat!("<", "|im_", "end", "|", ">"))?,
            stream_pad: find("[STREAMING_PAD]")?,
            stream_word: find("[STREAMING_WORD]")?,
            pad: find(concat!("<", "|endoftext", "|", ">"))?,
            language_zh: find("[LANGUAGE_ZH]")?,
            language_en: find("[LANGUAGE_EN]")?,
        })
    }

    pub fn language(&self, code: &str) -> Result<u32, String> {
        match code {
            "zh" => Ok(self.language_zh),
            "en" => Ok(self.language_en),
            other => Err(format!("Audio8 language must be zh or en, got {other:?}")),
        }
    }
}

/// Stateful Audio8 transcriber: build once, drive `next_chunk` until `None`.
pub struct StreamingTranscriber<'a> {
    encoder: &'a Audio8Encoder,
    decoder: &'a Audio8TextDecoder,
    session: Audio8TextSession<'a>,
    audio_state: Audio8State,
    stream: Vec<f32>,
    queue: VecDeque<u32>,
    tokens: SpecialTokens,
    start: usize,
    end: usize,
    generated: Vec<u32>,
    max_tokens: usize,
}

impl<'a> StreamingTranscriber<'a> {
    pub fn new(
        encoder: &'a Audio8Encoder,
        decoder: &'a Audio8TextDecoder,
        stream: Vec<f32>,
        tokens: SpecialTokens,
        language: u32,
        max_tokens: usize,
    ) -> Result<Self, String> {
        let schedule = Schedule::new();
        let initial = schedule.initial_tokens();
        let capacity = schedule.capacity_for(&stream);
        if capacity > decoder.context() {
            return Err("Audio8 input exceeds the decoder context".into());
        }
        let condition = time_condition(DELAY_TOKENS, &encoder.frame_embedding)?;
        let session = Audio8TextSession::new(decoder, capacity, &condition)?;
        let mut queue = VecDeque::with_capacity(initial);
        queue.push_back(tokens.bos);
        queue.push_back(language);
        queue.extend(std::iter::repeat_n(tokens.stream_pad, initial - 2));
        Ok(Self {
            encoder,
            decoder,
            session,
            audio_state: Audio8State::new(),
            stream,
            queue,
            tokens,
            start: 0,
            end: initial * SAMPLES_PER_TOKEN,
            generated: Vec::new(),
            max_tokens,
        })
    }

    /// Decode one 80 ms audio chunk. Returns `Some(next_token)` until the
    /// buffer is exhausted or `max_tokens` is reached, then `None`.
    pub fn next_chunk(&mut self) -> Result<Option<u32>, String> {
        if self.end + LOOK_AHEAD_SAMPLES > self.stream.len()
            || self.generated.len() >= self.max_tokens
        {
            return Ok(None);
        }
        let count = (self.end - self.start) / SAMPLES_PER_TOKEN;
        let window = &self.stream
            [self.start.saturating_sub(LOOK_BACK_SAMPLES)..self.end + LOOK_AHEAD_SAMPLES];
        let mel =
            compute_voxtral_log_mel(window).map_err(|error| format!("Audio8 mel: {error:?}"))?;
        let audio_embeddings = self.encoder.encode_window(
            &mel.normalized,
            mel.frames,
            count * 4,
            &mut self.audio_state,
        )?;
        let input_ids: Vec<u32> = (0..count)
            .map(|_| self.queue.pop_front().unwrap_or(self.tokens.stream_pad))
            .collect();
        #[cfg(feature = "parity-trace")]
        {
            crate::parity_trace::report(crate::parity_trace::token_ids(
                "audio8.input_ids",
                &input_ids,
            ));
            crate::parity_trace::report(crate::parity_trace::checkpoint_at(
                "audio8.audio_embeddings",
                None,
                Some(self.generated.len()),
                &[count, TEXT_WIDTH],
                &audio_embeddings,
            ));
        }
        let embeddings = self
            .decoder
            .embed_audio_tokens(&input_ids, &audio_embeddings)?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint_at(
            "audio8.text_embeddings",
            None,
            Some(self.generated.len()),
            &[count, TEXT_WIDTH],
            &embeddings,
        ));
        let mut logits = self.session.forward_logits(&embeddings)?;
        if logits.iter().any(|value| value.is_nan()) {
            return Err("Audio8 decoder produced NaN logits".into());
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint_at(
            "audio8.logits",
            None,
            Some(self.generated.len()),
            &[logits.len()],
            &logits,
        ));
        logits[self.tokens.eos as usize] = f32::NEG_INFINITY;
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(id, _)| id as u32)
            .ok_or("Audio8 decoder returned no logits")?;
        self.generated.push(next);
        self.queue.push_back(next);
        self.start = self.end;
        self.end += SAMPLES_PER_TOKEN;
        Ok(Some(next))
    }

    /// Finalize: emit the user-visible token stream with specials and EOS stripped.
    pub fn finish(self) -> Vec<u32> {
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::token_ids(
            "audio8.generated_ids",
            &self.generated,
        ));
        let hidden = [
            self.tokens.bos,
            self.tokens.stream_pad,
            self.tokens.stream_word,
            self.tokens.pad,
        ];
        self.generated
            .into_iter()
            .take_while(|&id| id != self.tokens.eos)
            .filter(|&id| !hidden.contains(&id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_capacity_matches_byte_window() {
        let schedule = Schedule::new();
        assert_eq!(schedule.initial_tokens(), 25);
        assert_eq!(schedule.right_tokens(), 17);
        let stream = vec![0.0f32; 25 * SAMPLES_PER_TOKEN];
        assert_eq!(schedule.capacity_for(&stream), 25 + 1);
    }

    #[test]
    fn padded_stream_prepends_silence_and_appends_right_pad() {
        let schedule = Schedule::new();
        let stream = schedule.padded_stream(&[1.0, 2.0, 3.0]);
        assert_eq!(
            stream.len(),
            LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN + 3 + schedule.right_tokens() * SAMPLES_PER_TOKEN
        );
        assert!(stream[..LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN]
            .iter()
            .all(|&value| value == 0.0));
    }

    #[test]
    fn special_tokens_rejects_unknown_language() {
        let tokens = SpecialTokens {
            bos: 0,
            eos: 1,
            stream_pad: 2,
            stream_word: 3,
            pad: 4,
            language_zh: 5,
            language_en: 6,
        };
        assert_eq!(tokens.language("zh").unwrap(), 5);
        assert_eq!(tokens.language("en").unwrap(), 6);
        assert!(tokens.language("ja").is_err());
    }
}
