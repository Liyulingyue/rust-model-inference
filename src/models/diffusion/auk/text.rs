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

use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
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
        let source = if source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            == Some("audiocpp")
        {
            Arc::new(NativeQwenSource::new(source)?) as Arc<dyn TensorSource>
        } else {
            source
        };
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

/// The audio.cpp GGUF retains HF tensor names but omits the Thinker config.
/// This view preserves its bytes; the fixed 3B signature is checked before use.
struct NativeQwenSource {
    source: Arc<dyn TensorSource>,
    metadata: std::collections::HashMap<String, MetaValue>,
    tensors: std::collections::HashMap<String, (String, TensorInfo)>,
}

impl NativeQwenSource {
    fn new(source: Arc<dyn TensorSource>) -> Result<Self, String> {
        let mut view = Self {
            source,
            metadata: Default::default(),
            tensors: Default::default(),
        };
        view.alias(
            "token_embd.weight",
            "thinker.model.embed_tokens.weight",
            &[2048, 151936],
        )?;
        // Encoder-only exports omit the LM head; text_encode never computes logits.
        if view.source.tensor_info("thinker.lm_head.weight").is_some() {
            view.alias("output.weight", "thinker.lm_head.weight", &[2048, 151936])?;
        }
        view.alias("output_norm.weight", "thinker.model.norm.weight", &[2048])?;
        for layer in 0..36 {
            for (alias, native, dims) in [
                ("attn_norm.weight", "input_layernorm.weight", vec![2048]),
                (
                    "ffn_norm.weight",
                    "post_attention_layernorm.weight",
                    vec![2048],
                ),
                ("attn_q.weight", "self_attn.q_proj.weight", vec![2048, 2048]),
                ("attn_k.weight", "self_attn.k_proj.weight", vec![2048, 256]),
                ("attn_v.weight", "self_attn.v_proj.weight", vec![2048, 256]),
                (
                    "attn_output.weight",
                    "self_attn.o_proj.weight",
                    vec![2048, 2048],
                ),
                ("attn_q.bias", "self_attn.q_proj.bias", vec![2048]),
                ("attn_k.bias", "self_attn.k_proj.bias", vec![256]),
                ("attn_v.bias", "self_attn.v_proj.bias", vec![256]),
                ("ffn_gate.weight", "mlp.gate_proj.weight", vec![2048, 11008]),
                ("ffn_up.weight", "mlp.up_proj.weight", vec![2048, 11008]),
                ("ffn_down.weight", "mlp.down_proj.weight", vec![11008, 2048]),
            ] {
                view.alias(
                    &format!("blk.{layer}.{alias}"),
                    &format!("thinker.model.layers.{layer}.{native}"),
                    &dims,
                )?;
            }
        }
        if view
            .source
            .tensor_info("thinker.model.layers.36.self_attn.q_proj.weight")
            .is_some()
        {
            return Err("AuK native text encoder must have exactly 36 layers".into());
        }
        // Qwen/Qwen2.5-Omni-3B config.json -> thinker_config.text_config.
        view.metadata.insert(
            "general.architecture".into(),
            MetaValue::String("qwen2vl".into()),
        );
        for (key, value) in [
            ("embedding_length", 2048),
            ("block_count", 36),
            ("attention.head_count", 16),
            ("attention.head_count_kv", 2),
            ("feed_forward_length", 11008),
            ("context_length", 32768),
            ("vocab_size", 151936),
        ] {
            view.metadata
                .insert(format!("qwen2vl.{key}"), MetaValue::Uint32(value));
        }
        view.metadata.insert(
            "qwen2vl.rope.freq_base".into(),
            MetaValue::Float32(1_000_000.0),
        );
        view.metadata.insert(
            "qwen2vl.attention.layer_norm_rms_epsilon".into(),
            MetaValue::Float32(1e-6),
        );
        view.metadata.insert(
            "qwen2vl.rope.dimension_sections".into(),
            MetaValue::Array(
                MetaValueType::Int32,
                [16, 24, 24, 0].into_iter().map(MetaValue::Int32).collect(),
            ),
        );
        Ok(view)
    }

    fn alias(&mut self, alias: &str, native: &str, dims: &[u64]) -> Result<(), String> {
        let mut info = self
            .source
            .tensor_info(native)
            .ok_or_else(|| format!("Missing AuK native Qwen tensor: {native}"))?
            .clone();
        let supported = if dims.len() == 1 {
            matches!(info.ggml_type, GGMLType::F32 | GGMLType::BF16)
        } else {
            matches!(
                info.ggml_type,
                GGMLType::F16 | GGMLType::BF16 | GGMLType::Q8_0
            )
        };
        if info.dims != dims || !supported {
            return Err(format!(
                "Unsupported AuK native Qwen tensor {native}: {:?} {:?}",
                info.dims, info.ggml_type
            ));
        }
        let bytes = self
            .source
            .tensor_slice(native)
            .ok_or_else(|| format!("Missing AuK native Qwen bytes: {native}"))?;
        if info.checked_nbytes() != Some(bytes.len() as u64) {
            return Err(format!("Invalid AuK native Qwen byte length: {native}"));
        }
        info.name = alias.into();
        self.tensors.insert(alias.into(), (native.into(), info));
        Ok(())
    }
}

impl TensorSource for NativeQwenSource {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key).or_else(|| self.source.metadata(key))
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name).map(|(_, info)| info)
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.tensors
            .get(name)
            .and_then(|(native, _)| self.source.tensor_slice(native))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the published audio.cpp Qwen GGUF in RMI_AUK_NATIVE_TEXT"]
    fn native_auk_qwen_signature_preserves_original_tensor_bytes() {
        let path = std::env::var("RMI_AUK_NATIVE_TEXT").expect("native Qwen GGUF required");
        let source: Arc<dyn TensorSource> = Arc::from(
            crate::open_model_source(std::path::Path::new(&path), crate::ComponentRole::Llm)
                .unwrap(),
        );
        let view = NativeQwenSource::new(source.clone()).unwrap();
        let config = crate::models::qwen3::trunk::Qwen3Config::from_source(&view).unwrap();
        assert_eq!(
            (
                config.n_embd,
                config.n_layer,
                config.n_head,
                config.n_head_kv
            ),
            (2048, 36, 16, 2)
        );
        for (alias, (native, info)) in &view.tensors {
            assert_eq!(&info.name, alias);
            assert_eq!(
                view.tensor_slice(alias).unwrap().as_ptr(),
                source.tensor_slice(native).unwrap().as_ptr()
            );
        }
    }
}
