//! Generation-schedule assembly for dots.tts (TTS and Edit variants).
//!
//! The schedule is a token sequence with audio "span" placeholders
//! (`<|audio_gen_span|>` / `<|audio_comp_span|>`). During prefill the LLM
//! embedding rows at span positions are replaced by patch-encoder embeddings,
//! and at decode time each span position generates one latent patch.
//! Ported from `dots_tts.data.pipelines.tokenizing` (template
//! `"[文本]{text}[文本对应语音]{audio}"`).

use crate::core::tokenizer::BPETokenizer;

pub const TTS_TEXT_PREFIX: &str = "[文本]";
pub const TTS_AUDIO_PREFIX: &str = "[文本对应语音]";
pub const TTA_TEXT_PREFIX: &str = "[声音描述]";
pub const TTA_AUDIO_PREFIX: &str = "[描述对应声音]";
pub const EDIT_SOURCE_TEXT_PREFIX: &str = "[原文本]";
pub const EDIT_SOURCE_AUDIO_PREFIX: &str = "[原语音]";
pub const EDIT_INSTRUCTION_PREFIX: &str = "[编辑指令]";
pub const EDIT_TARGET_TEXT_PREFIX: &str = "[编辑文本]";
pub const EDIT_TARGET_AUDIO_PREFIX: &str = "[编辑后语音]";

pub const AUDIO_GEN_START_TOKEN: &str = "<|audio_gen_start|>";
pub const AUDIO_GEN_SPAN_TOKEN: &str = "<|audio_gen_span|>";
pub const AUDIO_GEN_END_TOKEN: &str = "<|audio_gen_end|>";
pub const AUDIO_COMP_START_TOKEN: &str = "<|audio_comp_start|>";
pub const AUDIO_COMP_SPAN_TOKEN: &str = "<|audio_comp_span|>";
pub const AUDIO_COMP_END_TOKEN: &str = "<|audio_comp_end|>";
pub const TEXT_COND_END_TOKEN: &str = "<|text_cond_end|>";

pub fn token_id(tokenizer: &BPETokenizer, literal: &str) -> Result<u32, String> {
    tokenizer
        .token_id(literal)
        .ok_or_else(|| format!("tokenizer is missing required special token {literal}"))
}

fn encode_literal(tokenizer: &BPETokenizer, text: &str) -> Result<Vec<u32>, String> {
    Ok(tokenizer.encode(text, Default::default()))
}

#[derive(Debug, Clone)]
pub struct DotsSchedule {
    /// Full schedule token ids.
    pub ids: Vec<u32>,
    /// Positions of the audio span tokens (gen/comp spans), ascending.
    pub span_positions: Vec<usize>,
}

impl DotsSchedule {
    /// Audio span token ids (both gen and comp spans).
    pub fn audio_span_ids(tokenizer: &BPETokenizer) -> Result<Vec<u32>, String> {
        Ok(vec![
            token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)?,
            token_id(tokenizer, AUDIO_COMP_SPAN_TOKEN)?,
        ])
    }

    fn finalize(ids: Vec<u32>, span_ids: &[u32]) -> Self {
        let span_positions = ids
            .iter()
            .enumerate()
            .filter(|(_, id)| span_ids.contains(id))
            .map(|(pos, _)| pos)
            .collect();
        Self {
            ids,
            span_positions,
        }
    }
}

/// TTS generation schedule:
/// `[文本]{text}[文本对应语音]<|audio_gen_start|><|audio_gen_span|>×max_patches`
pub fn build_generation_schedule(
    tokenizer: &BPETokenizer,
    text: &str,
    max_audio_patches: usize,
) -> Result<DotsSchedule, String> {
    let gen_start = token_id(tokenizer, AUDIO_GEN_START_TOKEN)?;
    let gen_span = token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)?;
    let mut ids = Vec::new();
    ids.extend(encode_literal(tokenizer, TTS_TEXT_PREFIX)?);
    ids.extend(tokenizer.encode(text, Default::default()));
    ids.extend(encode_literal(tokenizer, TTS_AUDIO_PREFIX)?);
    ids.push(gen_start);
    ids.extend(std::iter::repeat(gen_span).take(max_audio_patches));
    Ok(DotsSchedule::finalize(ids, &[gen_span]))
}