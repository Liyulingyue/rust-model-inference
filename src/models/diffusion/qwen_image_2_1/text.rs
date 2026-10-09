use super::{pipeline::trace, Condition};
use crate::core::scratchpad::{KvFormat, KvLifecycle};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::qwen3::{Qwen3Input, Qwen3Model, Qwen3Session};
use std::sync::Arc;

const SYSTEM: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

pub(crate) struct ReferenceFeatures {
    pub embeddings: Vec<f32>,
    pub deepstack: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
}

pub(crate) struct TextConditioner {
    model: Qwen3Model,
}

impl TextConditioner {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let tokenizer = Arc::new(BPETokenizer::from_gguf_metadata(|key| {
            source.metadata(key).cloned()
        })?);
        let model = Qwen3Model::from_source(source, tokenizer, pool)?;
        let config = model.config();
        if config.architecture != "qwen3vl" || config.n_embd != 4096 || config.n_layer != 36 {
            return Err("Qwen-Image-2.1 requires the released Qwen3-VL-8B text encoder".into());
        }
        Ok(Self { model })
    }

    pub(crate) fn encode(
        &self,
        prompt: &str,
        references: &[ReferenceFeatures],
    ) -> Result<Condition, String> {
        let options = EncodeOptions {
            add_special: false,
            parse_special: true,
        };
        let tokenizer = self.model.tokenizer();
        let drop = tokenizer.encode(SYSTEM, options).len();
        let mut prefix = String::from(SYSTEM);
        prefix.push_str("<|im_start|>user\n");
        for i in 0..references.len() {
            if i > 0 {
                prefix.push(' ');
            }
            prefix.push_str(&format!(
                "<image{}><|vision_start|><|image_pad|><|vision_end|>",
                i + 1
            ));
        }
        let mut raw = tokenizer.encode(&prefix, options);
        raw.extend(tokenizer.encode(if prompt.is_empty() { " " } else { prompt }, options));
        raw.extend(tokenizer.encode(SUFFIX, options));
        let mut tokens = Vec::new();
        let mut slots = Vec::new();
        let mut image = 0;
        let mut image_ranges = Vec::new();
        for token in raw {
            if token == 151655 {
                let r = references
                    .get(image)
                    .ok_or("Image placeholder without a reference")?;
                let count = r
                    .grid_h
                    .checked_mul(r.grid_w)
                    .filter(|&n| n > 0)
                    .ok_or("Invalid vision grid")?;
                if count.checked_mul(4096) != Some(r.embeddings.len())
                    || r.deepstack.len() != 3 * r.embeddings.len()
                {
                    return Err("Invalid Qwen3-VL reference features".into());
                }
                image_ranges.push((tokens.len(), count));
                tokens.extend(std::iter::repeat_n(token, count));
                slots.extend(std::iter::repeat_n(image + 1, count));
                image += 1;
            } else {
                tokens.push(token);
                slots.push(0);
            }
        }
        if image != references.len() {
            return Err("Missing reference placeholder".into());
        }
        if tokens.len() > 2048 {
            return Err("Qwen-Image-2.1 prompt exceeds 2048 tokens".into());
        }
        trace(
            "qwen.text.tokens",
            &[tokens.len()],
            &tokens.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        )?;
        let mut positions: Vec<_> = (0..tokens.len()).map(|p| [p, p, p, 0]).collect();
        let mut embeddings = self.model.embed_tokens(&tokens)?;
        let mut deepstack = vec![0.0; 3 * embeddings.len()];
        let mut offset = 0isize;
        for ((start, count), r) in image_ranges.iter().copied().zip(references) {
            let base = (start as isize + offset) as usize;
            let next = base + r.grid_h.max(r.grid_w);
            for i in 0..count {
                positions[start + i] = [base, base + i / r.grid_w, base + i % r.grid_w, 0];
            }
            for (i, p) in positions.iter_mut().enumerate().skip(start + count) {
                let v = next + i - start - count;
                *p = [v, v, v, 0];
            }
            offset += r.grid_h.max(r.grid_w) as isize - count as isize;
            let begin = start * 4096;
            let len = count * 4096;
            embeddings[begin..begin + len].copy_from_slice(&r.embeddings);
            for level in 0..3 {
                let dst = level * embeddings.len() + begin;
                deepstack[dst..dst + len]
                    .copy_from_slice(&r.deepstack[level * len..(level + 1) * len]);
            }
        }
        trace("qwen.text.embeddings", &[4096, tokens.len()], &embeddings)?;
        let mut session = Qwen3Session::new_with_kv_state(
            &self.model,
            tokens.len(),
            KvFormat::F32,
            KvLifecycle::Ephemeral,
        )?;
        let hidden = session.forward_hidden_sequence_raw(Qwen3Input {
            token_ids: &tokens,
            positions: &positions,
            embeddings: Some(&embeddings),
            deepstack_embeddings: (!references.is_empty()).then_some(deepstack.as_slice()),
        })?;
        let context = hidden[drop * 4096..].to_vec();
        trace("qwen.text.context", &[4096, tokens.len() - drop], &context)?;
        if !references.is_empty() {
            trace(
                "qwen.text.image_slots",
                &[tokens.len() - drop],
                &slots[drop..].iter().map(|&v| v as f32).collect::<Vec<_>>(),
            )?;
        }
        Ok(Condition {
            values: context,
            image_slots: if references.is_empty() {
                Vec::new()
            } else {
                slots[drop..].to_vec()
            },
        })
    }
}
