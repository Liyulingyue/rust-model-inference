//! Qwen2.5-Omni-3B text encoder for AuK-Base 1.5B.
//!
//! Architecture: `qwen2vl` (per GGUF `general.architecture`), 36 layers,
//! hidden=2048, n_head=16, n_kv=2 (GQA), ffn=11008, RoPE base 1_000_000.
//! No QK-norm (Qwen2.5 style), QKV biases present.
//!
//! `qwen3::trunk::Qwen3Model` already supports the `qwen2vl` arch via
//! `qwen3_arch_knobs` (`has_qk_norm=false`, `has_qkv_bias=true`, tensor
//! naming `blk.{i}.{attn_q,attn_k,attn_v,attn_output,attn_norm,
//! ffn_gate,ffn_up,ffn_down,ffn_norm}.weight`). We just wrap the model
//! here and expose `encode(prompt) -> Vec<f32>` returning per-token
//! hidden states of width `TEXT_IN = 2048` (matches AuK's
//! `transformer.txt_proj` input dim).
//!
//! The AukPipeline then projects these to AuK's hidden via `txt_proj`
//! inside `AukDit::denoise`.

use std::sync::Arc;

use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::models::qwen3::trunk::positions::qwen_text_positions;
use crate::models::qwen3::trunk::{
    text_encode as trunk_text_encode, text_encode_with_audio as trunk_text_encode_with_audio,
    Qwen3Model,
};

const TEXT_IN: usize = 2_048;
/// Qwen2.5-Omni `<|AUDIO|>` placeholder token id; the audio tower's output
/// gets substituted for this token's position in the embedding sequence.
const AUDIO_PLACEHOLDER_TOKEN: u32 = 151_646;

pub(crate) struct AukTextEncoder {
    model: Qwen3Model,
    pool: Arc<ComputePool>,
}

impl AukTextEncoder {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        // Qwen2.5-Omni-3B tokenizer (adds <|AUDIO|>, <|audio_bos|>, <|audio_eos|>
        // to the standard Qwen2 oracle list). Required for CFMEdit audio
        // conditioning via the Qwen text encoder.
        let tokenizer = Arc::new(BPETokenizer::from_qwen25_omni_embedded_merges()?);
        let model = Qwen3Model::from_source(source, tokenizer, Arc::clone(&pool))?;
        Ok(Self { model, pool })
    }

    /// Encode `prompt` to per-token hidden states of width `TEXT_IN = 2048`.
    /// Output is a single contiguous `Vec<f32>` of shape `[seq_len * TEXT_IN]`,
    /// suitable for direct consumption by `AukDit::denoise`.
    pub(crate) fn encode(&self, prompt: &str) -> Result<Vec<f32>, String> {
        if prompt.is_empty() {
            return Err("AuK prompt is empty".into());
        }
        let ids = self.model.tokenizer().encode(
            prompt,
            crate::core::tokenizer::EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        );
        if ids.is_empty() {
            return Err("AuK prompt produced no tokens".into());
        }
        let positions = qwen_text_positions(ids.len());
        let hidden = trunk_text_encode(&self.model, &ids, &positions)?;
        if hidden.len() % TEXT_IN != 0 {
            return Err(format!(
                "AukTextEncoder produced malformed hidden: len {} not divisible by TEXT_IN={}",
                hidden.len(),
                TEXT_IN,
            ));
        }
        if !hidden.iter().all(|v| v.is_finite()) {
            return Err("AukTextEncoder produced non-finite hidden states".into());
        }
        Ok(hidden)
    }

    /// Encode the CFMEdit template:
    /// `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n
    /// {prompt}<|audio_bos|><|AUDIO|>...<|audio_eos|><|im_end|>\n<|im_start|>assistant\n`
    ///
    /// The audio tower's per-frame embeddings (length `audio_count * TEXT_IN`)
    /// are substituted for the `<|AUDIO|>` token positions before the
    /// Qwen transformer forward, so text and audio attend to each other
    /// inside the trunk -- matching audio.cpp's `conditioning.cpp`.
    ///
    /// `audio_count` must equal the number of `<|AUDIO|>` placeholders
    /// that will be inserted; if `audio_count == 0`, the encoder behaves
    /// like `encode(prompt)` with a no-prompt marker appended (per
    /// audio.cpp's "Zero audio token + reference voice" mode).
    ///
    /// When `instruct` is `Some`, the user message is built as
    /// `<instruct=...>\n{prompt}` per audio.cpp's instruct-TTS path.
    pub(crate) fn encode_with_audio(
        &self,
        prompt: &str,
        audio_count: usize,
        audio_embeddings: &[f32],
        instruct: Option<&str>,
    ) -> Result<Vec<f32>, String> {
        if audio_count == 0 {
            // No reference audio: just encode the bare prompt with the
            // "<no_prompt_audio>" marker (per audio.cpp's TTS path).
            // Instruct prefix is applied here too.
            let full_prompt = match instruct {
                Some(ins) if !ins.is_empty() => format!("<instruct={}>\n{}", ins, prompt),
                _ => prompt.to_string(),
            };
            return self.encode(&full_prompt);
        }
        if audio_embeddings.len() != audio_count * TEXT_IN {
            return Err(format!(
                "AukTextEncoder::encode_with_audio: audio embeddings length {} != audio_count*TEXT_IN={}",
                audio_embeddings.len(),
                audio_count * TEXT_IN
            ));
        }
        if prompt.is_empty() {
            return Err("AuK prompt is empty".into());
        }

        // Build the CFMEdit template.
        let mut formatted = String::with_capacity(256 + prompt.len() + audio_count * 12);
        formatted.push_str(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\n",
        );
        // Instruct prefix: per audio.cpp's instruct-TTS format
        // (`<instruct=...>\n<text>`) when an instruct string is provided.
        if let Some(ins) = instruct {
            if !ins.is_empty() {
                formatted.push_str("<instruct=");
                formatted.push_str(ins);
                formatted.push_str(">\n");
            }
        }
        formatted.push_str(prompt);
        formatted.push_str("<|audio_bos|>");
        for _ in 0..audio_count {
            formatted.push_str("<|AUDIO|>");
        }
        formatted.push_str("<|audio_eos|><|im_end|>\n<|im_start|>assistant\n");

        // Tokenize with parse_special=true so <|AUDIO|> etc. resolve to
        // their token ids. The <|AUDIO|> token id is 151_646.
        let ids = self.model.tokenizer().encode(
            &formatted,
            crate::core::tokenizer::EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        );
        if ids.is_empty() {
            return Err("AuK CFMEdit prompt produced no tokens".into());
        }

        // Collect (token_position, audio_index) pairs for each <|AUDIO|> token.
        let mut replacements: Vec<(usize, &[f32])> = Vec::with_capacity(audio_count);
        let mut audio_index = 0usize;
        for (i, &tok) in ids.iter().enumerate() {
            if tok == AUDIO_PLACEHOLDER_TOKEN {
                if audio_index >= audio_count {
                    return Err(format!(
                        "AukTextEncoder: token sequence contains more <|AUDIO|> placeholders \
                         ({}) than supplied audio embeddings ({})",
                        audio_index + 1,
                        audio_count
                    ));
                }
                replacements.push((
                    i,
                    &audio_embeddings[audio_index * TEXT_IN..(audio_index + 1) * TEXT_IN],
                ));
                audio_index += 1;
            }
        }
        if audio_index != audio_count {
            return Err(format!(
                "AukTextEncoder: token sequence has {} <|AUDIO|> placeholders, expected {}",
                audio_index, audio_count
            ));
        }

        let positions = qwen_text_positions(ids.len());
        let hidden = trunk_text_encode_with_audio(&self.model, &ids, &positions, &replacements)?;
        if hidden.len() % TEXT_IN != 0 {
            return Err(format!(
                "AukTextEncoder CFMEdit produced malformed hidden: len {} not divisible by TEXT_IN={}",
                hidden.len(),
                TEXT_IN,
            ));
        }
        if !hidden.iter().all(|v| v.is_finite()) {
            return Err("AukTextEncoder CFMEdit produced non-finite hidden states".into());
        }
        Ok(hidden)
    }

    /// Number of tokens in the last encoded sequence.
    pub(crate) fn last_token_count(&self, hidden: &[f32]) -> usize {
        hidden.len() / TEXT_IN
    }
}
