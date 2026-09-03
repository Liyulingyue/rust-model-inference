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
    /// Audio span positions filled during prefill, ascending.
    pub fill_span_positions: Vec<usize>,
    /// Audio span positions decoded after prefill, ascending.
    pub decode_span_positions: Vec<usize>,
}

impl DotsSchedule {
    /// Audio span token ids (both gen and comp spans).
    pub fn audio_span_ids(tokenizer: &BPETokenizer) -> Result<Vec<u32>, String> {
        Ok(vec![
            token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)?,
            token_id(tokenizer, AUDIO_COMP_SPAN_TOKEN)?,
        ])
    }
}

fn checked_total_len(parts: &[usize]) -> Result<usize, String> {
    parts.iter().try_fold(0usize, |total, &part| {
        total
            .checked_add(part)
            .ok_or_else(|| "dots schedule token count overflow".into())
    })
}

/// TTS generation schedule:
/// `[文本]{text}[文本对应语音]<|audio_gen_start|><|audio_gen_span|>×max_patches`
pub fn build_generation_schedule(
    tokenizer: &BPETokenizer,
    text: &str,
    fill_patch_count: usize,
    target_patch_count: usize,
) -> Result<DotsSchedule, String> {
    let gen_start = token_id(tokenizer, AUDIO_GEN_START_TOKEN)?;
    let gen_span = token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)?;
    let text_prefix = encode_literal(tokenizer, TTS_TEXT_PREFIX)?;
    let text_ids = tokenizer.encode(text, Default::default());
    let audio_prefix = encode_literal(tokenizer, TTS_AUDIO_PREFIX)?;
    let span_count = fill_patch_count
        .checked_add(target_patch_count)
        .ok_or_else(|| "dots schedule patch count overflow".to_string())?;
    let capacity = checked_total_len(&[
        text_prefix.len(),
        text_ids.len(),
        audio_prefix.len(),
        1,
        span_count,
    ])?;
    let mut ids = Vec::with_capacity(capacity);
    ids.extend(text_prefix);
    ids.extend(text_ids);
    ids.extend(audio_prefix);
    ids.push(gen_start);
    let fill_start = ids.len();
    ids.extend(std::iter::repeat_n(gen_span, fill_patch_count));
    let decode_start = ids.len();
    ids.extend(std::iter::repeat_n(gen_span, target_patch_count));
    Ok(DotsSchedule {
        fill_span_positions: (fill_start..decode_start).collect(),
        decode_span_positions: (decode_start..ids.len()).collect(),
        ids,
    })
}

pub fn build_edit_generation_schedule(
    tokenizer: &BPETokenizer,
    source_text: &str,
    instruction: &str,
    target_text: &str,
    source_patch_count: usize,
    target_patch_count: usize,
) -> Result<DotsSchedule, String> {
    assemble_edit_schedule(
        &encode_literal(tokenizer, EDIT_SOURCE_TEXT_PREFIX)?,
        &tokenizer.encode(source_text, Default::default()),
        &encode_literal(tokenizer, EDIT_SOURCE_AUDIO_PREFIX)?,
        &encode_literal(tokenizer, EDIT_INSTRUCTION_PREFIX)?,
        &tokenizer.encode(instruction, Default::default()),
        &encode_literal(tokenizer, EDIT_TARGET_TEXT_PREFIX)?,
        &tokenizer.encode(target_text, Default::default()),
        &encode_literal(tokenizer, EDIT_TARGET_AUDIO_PREFIX)?,
        token_id(tokenizer, AUDIO_GEN_START_TOKEN)?,
        token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)?,
        token_id(tokenizer, AUDIO_GEN_END_TOKEN)?,
        source_patch_count,
        target_patch_count,
    )
}

fn assemble_edit_schedule(
    source_prefix: &[u32],
    source_text: &[u32],
    source_audio_prefix: &[u32],
    instruction_prefix: &[u32],
    instruction: &[u32],
    target_prefix: &[u32],
    target_text: &[u32],
    target_audio_prefix: &[u32],
    gen_start: u32,
    gen_span: u32,
    gen_end: u32,
    source_count: usize,
    target_count: usize,
) -> Result<DotsSchedule, String> {
    if source_count == 0 || target_count == 0 {
        return Err("dots edit source and target patch counts must be positive".into());
    }
    source_count
        .checked_add(target_count)
        .ok_or_else(|| "dots edit schedule patch count overflow".to_string())?;
    let capacity = checked_total_len(&[
        source_prefix.len(),
        source_text.len(),
        source_audio_prefix.len(),
        1,
        source_count,
        1,
        instruction_prefix.len(),
        instruction.len(),
        target_prefix.len(),
        target_text.len(),
        target_audio_prefix.len(),
        1,
        target_count,
        1,
    ])?;
    let mut ids = Vec::with_capacity(capacity);
    ids.extend_from_slice(source_prefix);
    ids.extend_from_slice(source_text);
    ids.extend_from_slice(source_audio_prefix);
    ids.push(gen_start);
    let fill_start = ids.len();
    ids.extend(std::iter::repeat_n(gen_span, source_count));
    let fill_end = ids.len();
    ids.push(gen_end);
    ids.extend_from_slice(instruction_prefix);
    ids.extend_from_slice(instruction);
    ids.extend_from_slice(target_prefix);
    ids.extend_from_slice(target_text);
    ids.extend_from_slice(target_audio_prefix);
    ids.push(gen_start);
    let decode_start = ids.len();
    ids.extend(std::iter::repeat_n(gen_span, target_count));
    let decode_end = ids.len();
    ids.push(gen_end);
    Ok(DotsSchedule {
        ids,
        fill_span_positions: (fill_start..fill_end).collect(),
        decode_span_positions: (decode_start..decode_end).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_schedule_separates_source_fill_from_target_decode() {
        let schedule = assemble_edit_schedule(
            &[10],
            &[20],
            &[30],
            &[40],
            &[50],
            &[60],
            &[70],
            &[80],
            90,
            91,
            92,
            2,
            3,
        )
        .unwrap();
        assert_eq!(
            schedule.ids,
            vec![10, 20, 30, 90, 91, 91, 92, 40, 50, 60, 70, 80, 90, 91, 91, 91, 92]
        );
        assert_eq!(schedule.fill_span_positions, vec![3, 4]);
        assert_eq!(schedule.decode_span_positions, vec![12, 13, 14]);
    }

    #[test]
    fn edit_schedule_requires_positive_source_and_target_counts() {
        assert!(
            assemble_edit_schedule(&[], &[], &[], &[], &[], &[], &[], &[], 1, 2, 3, 0, 1).is_err()
        );
        assert!(
            assemble_edit_schedule(&[], &[], &[], &[], &[], &[], &[], &[], 1, 2, 3, 1, 0).is_err()
        );
    }
}
