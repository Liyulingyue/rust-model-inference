//! Tag-aware CosyVoice3 speech-token decoder.

use crate::compute::ComputePolicy;
use crate::core::scratchpad::{KvFormat, KvLifecycle};
use crate::core::tensor::{require_tensor_meta, GGMLType, MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::qwen3::trunk::{
    get_f32_tensor, load_layers_static, static_weight, Qwen3Config, Qwen3Input, Qwen3Model,
    Qwen3Rope, Qwen3Session,
};
use crate::ops::kernel::Weight;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

const WIDTH: usize = 896;
const VOCAB: usize = 6761;
const TEXT_VOCAB: usize = 151936;
pub const SPEECH: u32 = 6561;
pub const SOS: u32 = SPEECH;
pub const EOS: u32 = SPEECH + 1;
pub const TASK: u32 = SPEECH + 2;
pub const FILL: u32 = SPEECH + 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Feed {
    Text(u32),
    Speech(u32),
    Repeat,
}

struct Schedule {
    text: Vec<u32>,
    consumed: usize,
    out: Vec<u32>,
    next_fill: usize,
    final_phase: bool,
    finished: bool,
}

impl Schedule {
    fn new(text: Vec<u32>) -> Self {
        Self {
            text,
            consumed: 0,
            out: Vec::new(),
            next_fill: 15,
            final_phase: false,
            finished: false,
        }
    }

    fn forced(&self) -> Option<u32> {
        (!self.final_phase && !self.finished && self.out.len() == self.next_fill).then_some(FILL)
    }

    fn done(&self) -> bool {
        self.finished
    }

    fn accept(&mut self, token: u32) -> Result<Vec<Feed>, String> {
        if self.finished || (token >= SPEECH && token != if self.final_phase { EOS } else { FILL })
        {
            return Err(format!("invalid talker token {token} in current phase"));
        }
        if token == FILL {
            self.next_fill = self.out.len() + 16;
        }
        self.out.push(token);
        if token == EOS {
            self.finished = true;
            return Ok(Vec::new());
        }
        if token != FILL {
            return Ok(vec![Feed::Speech(token)]);
        }
        let remaining = &self.text[self.consumed..];
        if remaining.len() >= 5 {
            let feeds = remaining[..5].iter().copied().map(Feed::Text).collect();
            self.consumed += 5;
            return Ok(feeds);
        }
        self.final_phase = true;
        // The official bistream decoder feeds the input that produced fill twice.
        let feeds = std::iter::once(Feed::Repeat)
            .chain(remaining.iter().copied().map(Feed::Text))
            .chain(std::iter::once(Feed::Speech(TASK)))
            .collect();
        self.consumed = self.text.len();
        Ok(feeds)
    }
}

pub struct Talker {
    decoder: Qwen3Model,
    text_embedding: Weight<'static>,
}

pub struct SpeechTokens {
    pub tokens: Vec<u32>,
    pub terminated: bool,
}

impl Talker {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        super::require_string(
            source.as_ref(),
            "general.architecture",
            "sophon_realtime_talker",
        )?;
        super::require_string(source.as_ref(), "sophon_realtime.component", "talker")?;
        super::require_string(
            source.as_ref(),
            "sophon_realtime.source_revision",
            super::SOURCE_REVISION,
        )?;
        for (key, value) in [
            ("embedding_length", WIDTH),
            ("block_count", 24),
            ("attention.head_count", 14),
            ("attention.head_count_kv", 2),
            ("attention.key_length", 64),
            ("attention.value_length", 64),
            ("feed_forward_length", 4864),
            ("context_length", 32768),
            ("rope.dimension_count", 64),
        ] {
            super::require_u64(source.as_ref(), &format!("qwen2.{key}"), value as u64)?;
        }
        for (key, expected) in [
            ("rope.freq_base", 1_000_000.0f32),
            ("attention.layer_norm_rms_epsilon", 1e-6),
        ] {
            if source
                .metadata(&format!("qwen2.{key}"))
                .and_then(MetaValue::to_f64)
                .map(|v| v as f32)
                != Some(expected)
            {
                return Err(format!("invalid Sophon talker qwen2.{key}"));
            }
        }
        for (key, expected) in [
            ("speech_token_size", SPEECH),
            ("speech_vocab_size", VOCAB as u32),
            ("sos_token_id", SOS),
            ("eos_token_id", EOS),
            ("task_token_id", TASK),
            ("fill_token_id", FILL),
        ] {
            super::require_u64(
                source.as_ref(),
                &format!("sophon_realtime.talker.{key}"),
                expected as u64,
            )?;
        }
        for (key, expected) in [("has_qkv_bias", true), ("has_qk_norm", false)] {
            if !matches!(source.metadata(&format!("sophon_realtime.talker.{key}")), Some(MetaValue::Bool(v)) if *v == expected)
            {
                return Err(format!("invalid Sophon talker {key}"));
            }
        }
        let types = [
            GGMLType::F32,
            GGMLType::F16,
            GGMLType::BF16,
            GGMLType::Q8_0,
            GGMLType::Q4_0,
            GGMLType::Q4_1,
            GGMLType::Q4K,
            GGMLType::Q5K,
            GGMLType::Q6K,
        ];
        for (name, rows) in [
            ("token_embd.weight", TEXT_VOCAB),
            ("speech_embd.weight", VOCAB),
            ("output.weight", VOCAB),
        ] {
            require_tensor_meta(source.as_ref(), name, &[WIDTH as u64, rows as u64], &types)?;
        }
        require_tensor_meta(
            source.as_ref(),
            "output_norm.weight",
            &[WIDTH as u64],
            &[GGMLType::F32],
        )?;
        for layer in 0..24 {
            for name in ["attn_norm", "ffn_norm"] {
                require_tensor_meta(
                    source.as_ref(),
                    &format!("blk.{layer}.{name}.weight"),
                    &[WIDTH as u64],
                    &[GGMLType::F32],
                )?;
            }
            for (name, input, output) in [
                ("attn_q", WIDTH, WIDTH),
                ("attn_k", WIDTH, 128),
                ("attn_v", WIDTH, 128),
                ("attn_output", WIDTH, WIDTH),
                ("ffn_gate", WIDTH, 4864),
                ("ffn_up", WIDTH, 4864),
                ("ffn_down", 4864, WIDTH),
            ] {
                require_tensor_meta(
                    source.as_ref(),
                    &format!("blk.{layer}.{name}.weight"),
                    &[input as u64, output as u64],
                    &types,
                )?;
            }
            for (name, width) in [("attn_q", WIDTH), ("attn_k", 128), ("attn_v", 128)] {
                require_tensor_meta(
                    source.as_ref(),
                    &format!("blk.{layer}.{name}.bias"),
                    &[width as u64],
                    &[GGMLType::F32],
                )?;
            }
        }
        let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
            source.metadata(key).cloned()
        })?);
        if tokenizer.vocab_size() != TEXT_VOCAB
            || tokenizer.token_id("<|endofprompt|>") != Some(151646)
            || tokenizer.token_id("[breath]") != Some(151647)
            || tokenizer.eos_id() != Some(151643)
        {
            return Err("invalid Sophon talker text tokenizer".into());
        }
        let text_embedding = static_weight(source.as_ref(), "token_embd.weight", WIDTH, TEXT_VOCAB);
        let decoder = Qwen3Model {
            config: Qwen3Config {
                architecture: "qwen2".into(),
                n_embd: WIDTH,
                n_layer: 24,
                n_head: 14,
                n_head_kv: 2,
                n_embd_head_k: 64,
                n_embd_head_v: 64,
                n_ff: 4864,
                vocab: VOCAB,
                n_ctx: 32768,
                eps: 1e-6,
                freq_base: 1_000_000.0,
                has_qk_norm: false,
                has_qkv_bias: true,
                n_deepstack_layers: 0,
                moe: None,
                rope: Qwen3Rope::Neox,
            },
            layers: load_layers_static(
                Arc::clone(&source),
                24,
                WIDTH,
                WIDTH,
                128,
                4864,
                64,
                false,
                true,
                None,
            )?,
            output_norm: get_f32_tensor(source.as_ref(), "output_norm.weight", WIDTH),
            token_embedding: static_weight(source.as_ref(), "speech_embd.weight", WIDTH, VOCAB),
            output: static_weight(source.as_ref(), "output.weight", WIDTH, VOCAB),
            cls_score: None,
            tokenizer,
            pool,
            source,
        };
        Ok(Self {
            decoder,
            text_embedding,
        })
    }

    pub fn text_tokens(&self, text: &str) -> Vec<u32> {
        self.decoder.tokenizer.encode(
            text,
            EncodeOptions {
                add_special: false,
                parse_special: true,
            },
        )
    }

    fn embedding(&self, feed: Feed, last: &[f32]) -> Result<Vec<f32>, String> {
        if feed == Feed::Repeat {
            return Ok(last.to_vec());
        }
        let (weight, id) = match feed {
            Feed::Text(id) => (&self.text_embedding, id),
            Feed::Speech(id) => (&self.decoder.token_embedding, id),
            Feed::Repeat => unreachable!(),
        };
        if id as usize >= weight.n_out {
            return Err(format!("talker embedding id {id} out of range"));
        }
        let mut row = vec![0.0; WIDTH];
        weight.embedding_lookup(id, &mut row);
        Ok(row)
    }

    pub fn generate_tokens(
        &self,
        text: &str,
        max_speech_tokens: usize,
        temperature: f32,
        seed: u64,
        compute: ComputePolicy,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<SpeechTokens, String> {
        if text.trim().is_empty()
            || max_speech_tokens == 0
            || max_speech_tokens > 1250
            || !temperature.is_finite()
            || temperature < 0.0
        {
            return Err(
                "talker requires text, 1..=1250 speech tokens and finite nonnegative temperature"
                    .into(),
            );
        }
        let _scope = compute.enter_legacy_scope();
        let ids = self.text_tokens(&tag_clean(text, true));
        if ids.is_empty() {
            return Err("talker text is empty after tag cleaning".into());
        }
        let prompt = self.text_tokens("You are a helpful assistant.<|endofprompt|>");
        #[cfg(feature = "parity-trace")]
        {
            crate::parity_trace::report(crate::parity_trace::token_ids(
                "sophon.talker.text_ids",
                &ids,
            ));
            crate::parity_trace::report(crate::parity_trace::token_ids(
                "sophon.talker.prompt_ids",
                &prompt,
            ));
        }
        let capacity = prompt
            .len()
            .checked_add(ids.len())
            .and_then(|n| n.checked_add(max_speech_tokens * 2 + 8))
            .filter(|&n| n <= self.decoder.config.n_ctx)
            .ok_or("talker context budget exceeded")?;
        let mut session = Qwen3Session::new_with_compute(
            &self.decoder,
            capacity,
            KvFormat::F32,
            KvLifecycle::Ephemeral,
            compute,
        )?;
        let mut embeds = self.embedding(Feed::Speech(SOS), &[])?;
        for id in prompt {
            embeds.extend(self.embedding(Feed::Text(id), &[])?);
        }
        let mut last = embeds[embeds.len() - WIDTH..].to_vec();
        let mut logits = forward(&mut session, &embeds)?;
        let mut state = Schedule::new(ids);
        let mut tokens = Vec::new();
        let mut rng = StdRng::seed_from_u64(seed);
        while tokens.len() < max_speech_tokens {
            trace_logits(state.out.len(), &logits);
            let token = state
                .forced()
                .unwrap_or(sample(&logits, &state, temperature, &mut rng)?);
            let feeds = state.accept(token)?;
            if state.done() {
                break;
            }
            if token < SPEECH {
                tokens.push(token);
                if !on_token(token) {
                    break;
                }
            }
            embeds.clear();
            for feed in feeds {
                embeds.extend(self.embedding(feed, &last)?);
            }
            last.copy_from_slice(&embeds[embeds.len() - WIDTH..]);
            logits = forward(&mut session, &embeds)?;
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::token_ids(
            "sophon.talker.sampled_tokens",
            &state.out,
        ));
        Ok(SpeechTokens {
            tokens,
            terminated: state.done(),
        })
    }
}

fn forward(session: &mut Qwen3Session<'_>, embeddings: &[f32]) -> Result<Vec<f32>, String> {
    let rows = embeddings.len() / WIDTH;
    let offset = session.kv_state().seq_len;
    let positions: Vec<_> = (offset..offset + rows).map(|p| [p, p, p, 0]).collect();
    session
        .forward_logits(
            Qwen3Input {
                token_ids: &vec![0; rows],
                positions: &positions,
                embeddings: Some(embeddings),
                deepstack_embeddings: None,
            },
            32,
        )
        .map(|(logits, _)| logits)
}

fn sample(
    logits: &[f32],
    state: &Schedule,
    temperature: f32,
    rng: &mut impl Rng,
) -> Result<u32, String> {
    if logits.len() != VOCAB || logits.iter().any(|v| !v.is_finite()) {
        return Err("invalid talker logits".into());
    }
    let control = if state.final_phase { EOS } else { FILL };
    let mut scores = logits.to_vec();
    scores[SPEECH as usize..].fill(f32::NEG_INFINITY);
    scores[control as usize] = logits[control as usize];
    if temperature < 1e-5 {
        return Ok(scores
            .iter()
            .enumerate()
            .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
            .unwrap()
            .0 as u32);
    }
    for score in &mut scores {
        *score /= temperature;
    }
    crate::ops::softmax_inplace(&mut scores);
    let mut sorted: Vec<_> = (0..VOCAB).collect();
    sorted.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
    let mut before = 0.0;
    let mut candidates = Vec::new();
    for id in sorted.into_iter().take(25) {
        if before >= 0.8 {
            break;
        }
        candidates.push(id);
        before += scores[id];
    }
    let chosen = draw(&scores, &candidates, rng)?;
    if state
        .out
        .iter()
        .rev()
        .take(10)
        .any(|&id| id == chosen as u32)
    {
        scores[chosen] = 0.0;
        return draw(&scores, &(0..VOCAB).collect::<Vec<_>>(), rng).map(|id| id as u32);
    }
    Ok(chosen as u32)
}

fn draw(scores: &[f32], ids: &[usize], rng: &mut impl Rng) -> Result<usize, String> {
    let sum: f32 = ids.iter().map(|&id| scores[id]).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("empty talker sampling distribution".into());
    }
    let mut target = rng.gen::<f32>() * sum;
    for &id in ids {
        target -= scores[id];
        if target < 0.0 {
            return Ok(id);
        }
    }
    ids.iter()
        .copied()
        .rfind(|&id| scores[id] > 0.0)
        .ok_or("empty talker sampling distribution".into())
}

fn trace_logits(step: usize, logits: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint_at(
        "sophon.talker.logits",
        None,
        Some(step),
        &[logits.len()],
        logits,
    ));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (step, logits);
}

pub fn tag_clean(mut text: &str, final_chunk: bool) -> String {
    let mut output = String::new();
    let mut open: Vec<&str> = Vec::new();
    fn emit(text: &str, output: &mut String, open: &mut Vec<&str>) {
        let text = text
            .replace(['→', '⇒'], ", so ")
            .replace('≥', " at least ")
            .replace('≤', " at most ");
        if open.contains(&"strong") {
            if let Some((index, _)) = text
                .char_indices()
                .find(|(_, c)| "，。！？、,.!?;；:：".contains(*c))
            {
                output.push_str(&text[..index]);
                output.push_str("</strong>");
                open.retain(|&name| name != "strong");
                output.push_str(&text[index..]);
                return;
            }
        }
        output.push_str(&text);
    }
    while !text.is_empty() {
        let found = text.char_indices().find(|(_, c)| "[<(（【*#`".contains(*c));
        let Some((index, first)) = found else {
            emit(text, &mut output, &mut open);
            break;
        };
        emit(&text[..index], &mut output, &mut open);
        text = &text[index + first.len_utf8()..];
        let close = match first {
            '[' => ']',
            '<' => '>',
            '(' => ')',
            '（' => '）',
            '【' => '】',
            _ => continue,
        };
        let Some(end) = text.find(close) else {
            if text.chars().count() + 1 > 40 {
                continue;
            }
            break;
        };
        let content = &text[..end];
        text = &text[end + close.len_utf8()..];
        if first == '['
            && ["laughter", "breath", "sigh", "cough", "quick_breath"].contains(&content)
        {
            output.push('[');
            output.push_str(content);
            output.push(']');
        } else if first == '<' {
            let name = content.strip_prefix('/').unwrap_or(content);
            if !["laughter", "strong"].contains(&name) {
                continue;
            }
            let name = if name == "strong" {
                "strong"
            } else {
                "laughter"
            };
            if content.starts_with('/') {
                if !open.contains(&name) {
                    continue;
                }
                open.retain(|&v| v != name);
            } else {
                if open.contains(&name) {
                    continue;
                }
                open.push(name);
            }
            output.push('<');
            output.push_str(content);
            output.push('>');
        }
    }
    if final_chunk {
        for name in open.into_iter().rev() {
            output.push_str("</");
            output.push_str(name);
            output.push('>');
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_masks_controls_and_sampling_rejects_invalid_distributions() {
        let mut state = Schedule::new(Vec::new());
        let mut logits = vec![-100.0; VOCAB];
        logits[SOS as usize] = 1000.0;
        logits[EOS as usize] = 100.0;
        logits[FILL as usize] = 10.0;
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(sample(&logits, &state, 0.0, &mut rng).unwrap(), FILL);
        state.accept(FILL).unwrap();
        assert_eq!(sample(&logits, &state, 0.0, &mut rng).unwrap(), EOS);
        logits[0] = f32::NAN;
        assert!(sample(&logits, &state, 1.0, &mut rng).is_err());
        assert!(sample(&[], &state, 1.0, &mut rng).is_err());
    }

    #[test]
    fn tags_follow_the_official_whitelist_and_cumulative_partial_contract() {
        assert_eq!(
            tag_clean(
                "**Hello** [unknown] [breath]<strong>你好，朋友</strong>→x≥2",
                true
            ),
            "Hello  [breath]<strong>你好</strong>，朋友, so x at least 2"
        );
        assert_eq!(tag_clean("a[laugh", false), "a");
        assert_eq!(
            tag_clean("<laughter>哈<strong>好", true),
            "<laughter>哈<strong>好</strong></laughter>"
        );
        assert_eq!(
            tag_clean("<strong><strong>x</strong></strong>（skip）z", true),
            "<strong>x</strong>z"
        );
        assert_eq!(tag_clean("a[".to_owned().as_str(), true), "a");
    }

    #[test]
    #[ignore = "requires the remote pinned talker GGUF"]
    fn real_talker_checkpoints() {
        let path = std::env::var("RMI_SOPHON_TALKER").expect("set RMI_SOPHON_TALKER");
        let source = Arc::new(crate::core::loader::GGUFLoader::from_file(&path).unwrap());
        let model = Talker::from_source(source, Arc::new(ComputePool::new(4))).unwrap();
        let text = std::env::var("RMI_SOPHON_TEXT")
            .unwrap_or_else(|_| "你好，欢迎使用语音助手。[breath]".into());
        let tokens = model
            .generate_tokens(&text, 128, 0.0, 0, ComputePolicy::Cpu, |_| true)
            .unwrap();
        assert!(!tokens.tokens.is_empty());
        assert!(tokens.tokens.iter().all(|&v| v < SPEECH));
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::token_ids(
            "sophon.talker.speech",
            &tokens.tokens,
        ));
        eprintln!(
            "speech tokens={} terminated={}",
            tokens.tokens.len(),
            tokens.terminated
        );
    }

    #[test]
    fn bistream_fill_feeds_five_text_tokens_then_repeats_the_final_input() {
        let mut state = Schedule::new((1..=7).collect());
        assert_eq!(
            state.accept(FILL).unwrap(),
            (1..=5).map(Feed::Text).collect::<Vec<_>>()
        );
        for _ in 0..15 {
            assert_eq!(state.forced(), None);
            assert_eq!(state.accept(42).unwrap(), [Feed::Speech(42)]);
        }
        assert_eq!(state.forced(), Some(FILL));
        assert_eq!(
            state.accept(FILL).unwrap(),
            [
                Feed::Repeat,
                Feed::Text(6),
                Feed::Text(7),
                Feed::Speech(TASK)
            ]
        );
        let history = state.out.clone();
        for token in [SOS, FILL, SPEECH + 199] {
            assert!(state.accept(token).is_err());
            assert_eq!(state.out, history);
        }
        assert!(state.accept(EOS).unwrap().is_empty());
        assert!(state.done());
        assert!(state.accept(0).is_err());
    }

    #[test]
    fn exact_text_chunk_still_requires_final_repeat_and_task() {
        let mut state = Schedule::new(vec![10; 5]);
        assert_eq!(state.accept(FILL).unwrap(), [Feed::Text(10); 5]);
        for _ in 0..3 {
            state.accept(0).unwrap();
        }
        // An early sampled fill resets the next fifteen-speech-token boundary.
        assert_eq!(
            state.accept(FILL).unwrap(),
            [Feed::Repeat, Feed::Speech(TASK)]
        );
        assert!(!state.done());
        assert_eq!(state.forced(), None);
    }
}
