use super::{trace, HyAudioEncoder};
use crate::compute::ComputePolicy;
use crate::core::scratchpad::{KvFormat, KvLifecycle};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::qwen3::{Qwen3Input, Qwen3Model, Qwen3Session};
use std::sync::Arc;

const USER: &str = "<｜hy_User｜>";
const AUDIO_START: &str = "<｜hy_place▁holder▁no▁6｜>";
const AUDIO_END: &str = "<｜hy_place▁holder▁no▁8｜>";
const SOURCE: &str = "<｜hy_place▁holder▁no▁790｜>";

pub struct HySimulMt {
    decoder: Qwen3Model,
    encoder: HyAudioEncoder,
    interleave: bool,
}

pub struct Translation {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub turns: Vec<Vec<u32>>,
    pub prompt_tokens: usize,
    pub audio_tokens: usize,
}

impl HySimulMt {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        if source
            .metadata("general.architecture")
            .and_then(|v| v.to_string_val())
            != Some("hy_v3")
        {
            return Err("Hy-SimulMT requires general.architecture=hy_v3".into());
        }
        let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|k| {
            source.metadata(k).cloned()
        })?);
        for piece in [
            USER,
            AUDIO_START,
            AUDIO_END,
            SOURCE,
            "<｜hy_place▁holder▁no▁791｜>",
            "<｜hy_EOT｜>",
        ] {
            if tokenizer.token_id(piece).is_none() {
                return Err(format!("missing Hy-SimulMT protocol token {piece}"));
            }
        }
        let interleave = source
            .metadata("general.name")
            .and_then(|v| v.to_string_val())
            .unwrap_or("")
            .to_ascii_lowercase()
            .contains("interleave");
        let encoder = HyAudioEncoder::from_source(source.clone(), pool.clone())?;
        let decoder = Qwen3Model::from_source(source, tokenizer, pool)?;
        if decoder.config().n_embd != 2048 {
            return Err("Hy-SimulMT audio projection requires decoder width 2048".into());
        }
        Ok(Self {
            decoder,
            encoder,
            interleave,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn translate(
        &self,
        samples: &[f32],
        language: &str,
        instruction: Option<&str>,
        max_tokens: usize,
        max_context: usize,
        batch_size: usize,
        kv_format: KvFormat,
    ) -> Result<Translation, String> {
        if samples.is_empty()
            || samples.iter().any(|v| !v.is_finite())
            || max_tokens == 0
            || batch_size == 0
        {
            return Err(
                "Hy-SimulMT requires finite non-empty audio and positive token/batch limits".into(),
            );
        }
        let _cpu = ComputePolicy::Cpu.cpu_scope();
        let tokenizer = self.decoder.tokenizer();
        let encode = |s: &str| {
            tokenizer.encode(
                s,
                EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            )
        };
        let (prefix, suffix) = prompt(language, instruction, self.interleave)?;
        let suffix = encode(&suffix);
        let mut session = Qwen3Session::new_with_compute(
            &self.decoder,
            max_context.min(self.decoder.config().n_ctx),
            kv_format,
            KvLifecycle::Ephemeral,
            ComputePolicy::Cpu,
        )?;
        let mut result = Translation {
            text: String::new(),
            token_ids: Vec::new(),
            turns: Vec::new(),
            prompt_tokens: 0,
            audio_tokens: 0,
        };
        let prefix = encode(&prefix);
        feed(&mut session, &prefix, None, batch_size)?;
        result.prompt_tokens += prefix.len();
        let eot = tokenizer
            .token_id("<｜hy_EOT｜>")
            .ok_or("missing EOT token")?;
        let eos = tokenizer.eos_id().ok_or("missing EOS token")?;
        let mut offset = 0;
        let mut done = false;
        while offset < samples.len() && !done && result.token_ids.len() < max_tokens {
            let count = (if offset == 0 { 25_600 } else { 12_800 }).min(samples.len() - offset);
            let audio = self.encoder.encode(&samples[offset..offset + count])?;
            let n_audio = audio.len() / 2048;
            let mut ids = if offset == 0 {
                Vec::new()
            } else {
                encode(USER)
            };
            ids.extend(encode(AUDIO_START));
            let audio_offset = ids.len();
            ids.resize(
                ids.len() + n_audio,
                tokenizer
                    .token_id("<｜hy_place▁holder▁no▁5｜>")
                    .ok_or("missing audio placeholder")?,
            );
            ids.extend(encode(AUDIO_END));
            ids.extend_from_slice(&suffix);
            let mut embeddings = self.decoder.embed_tokens(&ids)?;
            embeddings[audio_offset * 2048..(audio_offset + n_audio) * 2048]
                .copy_from_slice(&audio);
            let mut logits = feed(&mut session, &ids, Some(&embeddings), batch_size)?;
            result.prompt_tokens += ids.len();
            result.audio_tokens += n_audio;
            let mut turn = Vec::new();
            loop {
                if logits.iter().any(|v| !v.is_finite()) {
                    return Err("non-finite Hy-SimulMT logits".into());
                }
                trace("hy.logits", &[logits.len()], &logits);
                let next = logits
                    .iter()
                    .enumerate()
                    .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
                    .ok_or("empty logits")?
                    .0 as u32;
                result.token_ids.push(next);
                turn.push(next);
                done = next == eos || result.token_ids.len() == max_tokens;
                // A completed assistant boundary is retained in KV for the next audio turn.
                if !done {
                    logits = feed(&mut session, &[next], None, batch_size)?;
                }
                if next == eot || done {
                    break;
                }
            }
            let visible: Vec<u32> = turn
                .iter()
                .copied()
                .filter(|&t| t != eot && t != eos)
                .collect();
            if !result.text.is_empty() {
                result.text.push('\n');
            }
            result.text.push_str(&tokenizer.decode(&visible, true));
            result.turns.push(turn);
            offset += count;
        }
        Ok(result)
    }
}

fn feed(
    session: &mut Qwen3Session<'_>,
    ids: &[u32],
    embeddings: Option<&[f32]>,
    batch: usize,
) -> Result<Vec<f32>, String> {
    let start = session.kv_state().seq_len;
    let positions: Vec<[usize; 4]> = (start..start + ids.len()).map(|p| [p, p, p, 0]).collect();
    session
        .forward_logits(
            Qwen3Input {
                token_ids: ids,
                positions: &positions,
                embeddings,
                deepstack_embeddings: None,
            },
            batch,
        )
        .map(|(logits, _)| logits)
}

fn prompt(
    language: &str,
    instruction: Option<&str>,
    interleave: bool,
) -> Result<(String, String), String> {
    let language = language.trim();
    if language.is_empty() {
        return Err("Hy-SimulMT target language must not be empty".into());
    }
    let language = match language.to_ascii_lowercase().as_str() {
        "zh" | "chinese" => "中文",
        "en" | "english" => "英文",
        _ => language,
    };
    let instruction=instruction.map(str::to_owned).unwrap_or_else(||format!("请完成下面的同声传译任务，将陆续收到的语音翻译成{language}。请结合上下文边听边译，输出能够确定的译文，不必等待整句结束。"));
    let prefix=format!("<｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>reasoning_effort:no_think{USER}{instruction}");
    let suffix = format!(
        "<｜hy_Assistant｜><think></think>{}",
        if interleave { SOURCE } else { "" }
    );
    Ok((prefix, suffix))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hy_protocol_only_interleave_injects_source_marker() {
        let (prefix, suffix) = prompt("en", None, false).unwrap();
        assert!(prefix.contains("翻译成英文"));
        assert_eq!(suffix, "<｜hy_Assistant｜><think></think>");
        assert_eq!(
            prompt("zh", Some("custom"), true).unwrap().1,
            format!("{suffix}{SOURCE}")
        );
        assert!(prompt(" ", None, false).is_err());
    }
}
