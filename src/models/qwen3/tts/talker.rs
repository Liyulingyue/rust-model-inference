//! Qwen3-TTS-12Hz 1.7B Base "Talker" LLM.
//!
//! The Talker is a 28-layer Qwen3 transformer with M-RoPE
//! (`[24, 20, 20, 0]`) and per-head Q/K RMSNorm, identical in shape to the
//! 1.7B Qwen3 dense configuration. It differs from a vanilla Qwen3 in two
//! ways:
//!
//! 1. The vocabulary mixes text tokens (151936) and audio-codebook tokens
//!    (3072) into a single embedding table of 155008 entries — used for the
//!    input side.
//! 2. The output head is a small `output.weight` tensor with shape
//!    `[n_embd, audio_codebook_size]` (= `[2048, 3072]` for this 1.7B
//!    variant). After the audio-start marker, the Talker only emits audio
//!    codebook tokens, so the head is intentionally narrow.
//!
//! The forward loop mirrors `crate::models::qwen3`'s `Qwen3Session::generate_inner`
//! — single-token decode against an F16 KV cache — substituting a
//! `rope_mrope_interleaved` call and a narrow output head for the standard
//! vocabulary projection.

use std::sync::Arc;

use rand::Rng;

use crate::core::scratchpad::{ExecutionScratchpad, KvCache};
use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::qwen3::base::{Qwen3Config};
use crate::models::qwen3::util::{
    check_allocation, checked_product, load_f32_tensor, usize_to_u64,
};
use crate::models::qwen3::static_weight;
use crate::models::qwen3::tts::AUDIO_CODEBOOK_SIZE;
use crate::ops::kernel::Weight;
use crate::ops::{
    dot_f16, f16_to_f32, f32_slice_to_f16, f32_to_f16,
    quantize_q8_0_into, rms_norm, rms_norm_inplace,
    rope_mrope_interleaved, rope_neox, silu_mul_approx_inplace, softmax_inplace, vec_scale_f32,
};

const SEMANTIC_CODEBOOK_SIZE: usize = 2048;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub(crate) unsafe fn scale_f16_inplace(values: &mut [u16], scale: f32) {
    use std::arch::aarch64::*;

    let scale: float16x8_t = std::mem::transmute(vdupq_n_u16(f32_to_f16(scale)));
    for values in values.chunks_exact_mut(8) {
        let vector: float16x8_t = std::mem::transmute(vld1q_u16(values.as_ptr()));
        vst1q_u16(
            values.as_mut_ptr(),
            std::mem::transmute(vmulq_f16(vector, scale)),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub(crate) unsafe fn mad_f16_inplace(output: &mut [u16], values: &[u16], scale: f32) {
    use std::arch::aarch64::*;

    let scale: float16x8_t = std::mem::transmute(vdupq_n_u16(f32_to_f16(scale)));
    for (output, values) in output.chunks_exact_mut(8).zip(values.chunks_exact(8)) {
        let accumulator: float16x8_t = std::mem::transmute(vld1q_u16(output.as_ptr()));
        let values: float16x8_t = std::mem::transmute(vld1q_u16(values.as_ptr()));
        vst1q_u16(
            output.as_mut_ptr(),
            std::mem::transmute(vfmaq_f16(accumulator, values, scale)),
        );
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn scale_f16_inplace(values: &mut [u16], scale: f32) {
    for value in values {
        *value = f32_to_f16(f16_to_f32(*value) * scale);
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn mad_f16_inplace(output: &mut [u16], values: &[u16], scale: f32) {
    for (output, value) in output.iter_mut().zip(values) {
        *output = f32_to_f16(f16_to_f32(*output) + f16_to_f32(*value) * scale);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TtsSpecialTokens {
    pub codec_0: u32,
    pub codec_bos: u32,
    pub codec_eos: u32,
    pub codec_pad: u32,
    pub codec_think: u32,
    pub codec_think_bos: u32,
    pub codec_think_eos: u32,
    pub codec_language: u32,
    pub tts_pad: u32,
    pub tts_text_bos: u32,
    pub tts_text_eod: u32,
}

impl TtsSpecialTokens {
    pub fn resolve(tokenizer: &BPETokenizer, language: &str) -> Result<Self, String> {
        resolve_tts_special_tokens(language, |literal| tokenizer.token_id(literal))
    }

    fn narrow_eos(self, output_rows: usize) -> Result<u32, String> {
        let eos = self.codec_eos.checked_sub(self.codec_0).ok_or_else(|| {
            format!(
                "TTS codec EOS token {} precedes codec base {}",
                self.codec_eos, self.codec_0
            )
        })?;
        if eos as usize >= output_rows {
            return Err(format!(
                "TTS codec EOS row {eos} exceeds output rows {output_rows}"
            ));
        }
        Ok(eos)
    }
}

fn resolve_tts_special_tokens<F>(
    language: &str,
    mut token_id: F,
) -> Result<TtsSpecialTokens, String>
where
    F: FnMut(&str) -> Option<u32>,
{
    let language_literal = format!("<|codec_language_{language}|>");
    let mut required = |literal: &str| {
        token_id(literal).ok_or_else(|| format!("Missing TTS vocabulary token {literal}"))
    };
    Ok(TtsSpecialTokens {
        codec_0: required("<|codec_0|>")?,
        codec_bos: required("<|codec_bos|>")?,
        codec_eos: required("<|codec_eos_token|>")?,
        codec_pad: required("<|codec_pad|>")?,
        codec_think: required("<|codec_think|>")?,
        codec_think_bos: required("<|codec_think_bos|>")?,
        codec_think_eos: required("<|codec_think_eos|>")?,
        codec_language: required(&language_literal)?,
        tts_pad: required("<tts_pad>")?,
        tts_text_bos: required("<tts_text_bos>")?,
        tts_text_eod: required("<tts_text_eod>")?,
    })
}

#[derive(Debug, Clone)]
pub struct TtsPrompt {
    pub wrapper_ids: Vec<u32>,
    pub embeddings: Vec<f32>,
    pub positions: Vec<[usize; 4]>,
    pub overlay: Vec<Vec<f32>>,
}

/// Resolved configuration of a Qwen3-TTS Base Talker.
#[derive(Debug, Clone)]
pub struct Qwen3TtsTalkerConfig {
    pub architecture: String,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_ff: usize,
    pub vocab_size: usize,
    pub n_ctx: usize,
    pub eps: f32,
    pub freq_base: f32,
    pub rope_sections: [i32; 4],
    pub audio_codebook_size: usize,
    pub eos_token_id: u32,
}

impl Qwen3TtsTalkerConfig {
    fn from_source(source: &dyn TensorSource, eos_token_id: u32) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .ok_or_else(|| "Missing metadata: general.architecture".to_string())?;
        if arch != "qwen3tts" {
            return Err(format!("Unsupported TTS architecture: {arch}"));
        }

        if source.tensor_info("blk.0.attn_norm.weight").is_none() {
            return Err("TTS talker is missing blk.0.attn_norm.weight".into());
        }

        // The TTS model carries M-RoPE sections at `qwen3tts.rope.dimension_sections`
        // (= [24, 20, 20, 0] for this 1.7B variant). Fall back to that constant
        // if the metadata is missing for whatever reason.
        let rope_sections =
            read_i32_array(source, "qwen3tts.rope.dimension_sections").unwrap_or([24, 20, 20, 0]);

        let qwen3 = Qwen3Config::from_source(source)?;

        let vocab_size = source
            .metadata("tokenizer.ggml.tokens")
            .and_then(MetaValue::to_arr)
            .map(Vec::len)
            .unwrap_or(0);

        Ok(Self {
            architecture: "qwen3tts".into(),
            n_embd: qwen3.n_embd,
            n_layer: qwen3.n_layer,
            n_head: qwen3.n_head,
            n_head_kv: qwen3.n_head_kv,
            n_embd_head_k: qwen3.n_embd_head_k,
            n_embd_head_v: qwen3.n_embd_head_v,
            n_ff: qwen3.n_ff,
            vocab_size,
            n_ctx: qwen3.n_ctx,
            eps: qwen3.eps,
            freq_base: qwen3.freq_base,
            rope_sections,
            audio_codebook_size: AUDIO_CODEBOOK_SIZE,
            eos_token_id,
        })
    }
}

fn read_i32_array(source: &dyn TensorSource, key: &str) -> Result<[i32; 4], String> {
    let value = source
        .metadata(key)
        .ok_or_else(|| format!("Missing metadata: {key}"))?;
    let arr = value
        .to_arr()
        .ok_or_else(|| format!("{key} is not an array"))?;
    if arr.len() != 4 {
        return Err(format!("{key} expected 4 entries, got {}", arr.len()));
    }
    let mut out = [0i32; 4];
    for (slot, value) in out.iter_mut().zip(arr.iter()) {
        let v = value
            .to_u64()
            .ok_or_else(|| format!("{key} contains a non-integer entry"))?;
        *slot = i32::try_from(v).map_err(|_| format!("{key} entry {v} does not fit i32"))?;
    }
    Ok(out)
}

/// Per-layer weights for the Talker.
pub(crate) struct TtsLayerWeights {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) wq: Weight<'static>,
    pub(crate) wk: Weight<'static>,
    pub(crate) wv: Weight<'static>,
    pub(crate) wo: Weight<'static>,
    pub(crate) w_gate: Weight<'static>,
    pub(crate) w_up: Weight<'static>,
    pub(crate) w_down: Weight<'static>,
}

/// One generation result from the Talker.
#[derive(Debug, Clone)]
pub struct Qwen3TtsGeneration {
    pub prompt_token_ids: Vec<u32>,
    pub audio_token_ids: Vec<u32>,
    pub finished_reason: TtsFinishReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsFinishReason {
    Eos,
    Length,
    Invalid,
}

/// Loaded Qwen3-TTS Talker.
pub struct Qwen3TtsTalker {
    source: Arc<dyn TensorSource>,
    tokenizer: Arc<BPETokenizer>,
    pool: Arc<ComputePool>,
    config: Qwen3TtsTalkerConfig,
    layers: Vec<TtsLayerWeights>,
    output_norm: Vec<f32>,
    token_embedding: Weight<'static>,
    audio_output_head: Weight<'static>,
}

impl Qwen3TtsTalker {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        tokenizer: Arc<BPETokenizer>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let eos_token_id =
            TtsSpecialTokens::resolve(&tokenizer, "english")?.narrow_eos(AUDIO_CODEBOOK_SIZE)?;
        let config = Qwen3TtsTalkerConfig::from_source(source.as_ref(), eos_token_id)?;
        if config.vocab_size != tokenizer.vocab_size() {
            return Err(format!(
                "{} vocabulary size {} does not match tokenizer vocab {}",
                config.architecture,
                config.vocab_size,
                tokenizer.vocab_size(),
            ));
        }

        let n_embd_q = checked_product("query width", config.n_head, config.n_embd_head_k)?;
        let n_embd_k = checked_product("key width", config.n_head_kv, config.n_embd_head_k)?;
        let n_embd_v = checked_product("value width", config.n_head_kv, config.n_embd_head_v)?;
        let n_attn = checked_product(
            "attention output width",
            config.n_head,
            config.n_embd_head_v,
        )?;

        let output_norm = load_f32_tensor(
            source.as_ref(),
            "output_norm.weight",
            &[usize_to_u64(config.n_embd, "embedding width")?],
        )?;

        let token_embedding = static_weight(
            source.as_ref(),
            "token_embd.weight",
            config.vocab_size,
            config.n_embd,
        );

        let audio_output_head = static_weight(
            source.as_ref(),
            "output.weight",
            config.audio_codebook_size,
            config.n_embd,
        );

        check_allocation(
            "Talker decoder layers",
            config.n_layer,
            std::mem::size_of::<TtsLayerWeights>(),
        )?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.n_layer)
            .map_err(|error| format!("Failed to allocate Talker decoder layers: {error}"))?;
        for layer in 0..config.n_layer {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}");
            let n_embd_dim = [usize_to_u64(config.n_embd, "embedding width")?];
            let head_dim = [usize_to_u64(config.n_embd_head_k, "key head width")?];
            layers.push(TtsLayerWeights {
                attn_norm: load_f32_tensor(
                    source.as_ref(),
                    &name("attn_norm.weight"),
                    &n_embd_dim,
                )?,
                ffn_norm: load_f32_tensor(source.as_ref(), &name("ffn_norm.weight"), &n_embd_dim)?,
                q_norm: load_f32_tensor(source.as_ref(), &name("attn_q_norm.weight"), &head_dim)?,
                k_norm: load_f32_tensor(source.as_ref(), &name("attn_k_norm.weight"), &head_dim)?,
                wq: static_weight(
                    source.as_ref(),
                    &name("attn_q.weight"),
                    n_embd_q,
                    config.n_embd,
                ),
                wk: static_weight(
                    source.as_ref(),
                    &name("attn_k.weight"),
                    n_embd_k,
                    config.n_embd,
                ),
                wv: static_weight(
                    source.as_ref(),
                    &name("attn_v.weight"),
                    n_embd_v,
                    config.n_embd,
                ),
                wo: static_weight(
                    source.as_ref(),
                    &name("attn_output.weight"),
                    config.n_embd,
                    n_attn,
                ),
                w_gate: static_weight(
                    source.as_ref(),
                    &name("ffn_gate.weight"),
                    config.n_ff,
                    config.n_embd,
                ),
                w_up: static_weight(
                    source.as_ref(),
                    &name("ffn_up.weight"),
                    config.n_ff,
                    config.n_embd,
                ),
                w_down: static_weight(
                    source.as_ref(),
                    &name("ffn_down.weight"),
                    config.n_embd,
                    config.n_ff,
                ),
            });
        }

        Ok(Self {
            source,
            tokenizer,
            pool,
            config,
            layers,
            output_norm,
            token_embedding,
            audio_output_head,
        })
    }

    pub fn config(&self) -> &Qwen3TtsTalkerConfig {
        &self.config
    }

    pub fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }

    pub fn pool(&self) -> Arc<ComputePool> {
        Arc::clone(&self.pool)
    }

    pub fn prepare_prompt(
        &self,
        text: &str,
        language: &str,
        speaker: Option<&[f32]>,
    ) -> Result<TtsPrompt, String> {
        if text.trim().is_empty() {
            return Err("TTS prompt must not be empty".into());
        }
        let specials = TtsSpecialTokens::resolve(&self.tokenizer, language)?;
        if specials.narrow_eos(self.config.audio_codebook_size)? != self.config.eos_token_id {
            return Err("TTS codec EOS token changed across language resolution".into());
        }
        let wrapper_ids = self.tokenizer.encode(
            &format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"),
            EncodeOptions {
                add_special: false,
                parse_special: true,
            },
        );
        let prompt = compose_prompt_embeddings(
            &wrapper_ids,
            &specials,
            speaker,
            self.config.n_embd,
            |token_id| self.embedding_row(token_id),
        )?;
        #[cfg(feature = "parity-trace")]
        {
            crate::parity_trace::report(crate::parity_trace::token_ids(
                "tts.prompt_ids",
                &prompt.wrapper_ids,
            ));
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "tts.prompt_embeddings",
                None,
                &[prompt.positions.len(), self.config.n_embd],
                &prompt.embeddings,
            ));
        }
        Ok(prompt)
    }

    fn embedding_row(&self, token_id: u32) -> Result<Vec<f32>, String> {
        if token_id as usize >= self.config.vocab_size {
            return Err(format!(
                "TTS embedding token {token_id} exceeds vocabulary {}",
                self.config.vocab_size
            ));
        }
        let mut row = vec![0.0; self.config.n_embd];
        self.token_embedding.embedding_lookup(token_id, &mut row);
        if row.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "TTS embedding token {token_id} contains non-finite values"
            ));
        }
        Ok(row)
    }

    /// Diagnostic generation without the mmproj predictor. Full synthesis
    /// uses the predictor feedback path in `app::tts`.
    pub fn synthesize(
        &self,
        text: &str,
        _positions: Option<&[[usize; 4]]>,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Qwen3TtsGeneration, String> {
        let specials = TtsSpecialTokens::resolve(&self.tokenizer, "english")?;
        let prompt = self.prepare_prompt(text, "english", None)?;
        let mut session = TtsSession::new(self)?;
        session.prefill_prompt(&prompt)?;
        let total = prompt
            .positions
            .len()
            .checked_add(max_new_tokens)
            .ok_or_else(|| "TTS prompt+generation length overflow".to_string())?;
        if total > self.config.n_ctx {
            return Err(format!(
                "prompt+generation={total} exceeds TTS context {}",
                self.config.n_ctx
            ));
        }

        let mut rng = rand::thread_rng();
        let mut audio_token_ids = Vec::with_capacity(max_new_tokens);
        let mut finished_reason = TtsFinishReason::Length;
        for frame in 0..max_new_tokens {
            let Some(semantic) = session.sample_semantic(temperature, &mut rng)? else {
                finished_reason = TtsFinishReason::Eos;
                break;
            };
            audio_token_ids.push(semantic);
            let full_token = specials
                .codec_0
                .checked_add(semantic)
                .ok_or_else(|| "TTS semantic token ID overflow".to_string())?;
            let mut feedback = self.embedding_row(full_token)?;
            let overlay = &prompt.overlay[frame.min(prompt.overlay.len() - 1)];
            for (value, text) in feedback.iter_mut().zip(overlay) {
                *value += *text;
            }
            let position = prompt.positions.len() + frame;
            session.forward_step_with_embedding(&feedback, [position; 4])?;
        }
        Ok(Qwen3TtsGeneration {
            prompt_token_ids: prompt.wrapper_ids,
            audio_token_ids,
            finished_reason,
        })
    }

    /// Allocate a fresh empty `TtsSession` for the talker, ready to be
    /// driven step-by-step from outside (e.g. by the codec pipeline that
    /// interleaves talker forwards with code-predictor passes).
    pub fn new_session(&self) -> Result<TtsSession<'_>, String> {
        TtsSession::new(self)
    }
}

pub(crate) struct TtsSession<'model> {
    pub(crate) model: &'model Qwen3TtsTalker,
    pub(crate) kv_cache: KvCache,
    scratch: ExecutionScratchpad,
    capacity: usize,
    next_step: usize,
}

impl<'model> TtsSession<'model> {
    pub(crate) fn new(model: &'model Qwen3TtsTalker) -> Result<Self, String> {
        let config = &model.config;
        let n_embd_q = checked_product("query width", config.n_head, config.n_embd_head_k)?;
        let n_embd_k = checked_product("key width", config.n_head_kv, config.n_embd_head_k)?;
        let n_embd_v = checked_product("value width", config.n_head_kv, config.n_embd_head_v)?;
        let n_attn = checked_product(
            "attention output width",
            config.n_head,
            config.n_embd_head_v,
        )?;
        let scratch = ExecutionScratchpad::new(
            config.n_embd,
            n_embd_q,
            n_embd_k.max(n_embd_v),
            config.n_ff,
            config.audio_codebook_size,
            model.pool.n_threads(),
            config.n_ctx,
        );
        let kv_cache = KvCache::new_f16(config.n_layer, config.n_ctx, n_embd_k.max(n_embd_v));
        Ok(Self {
            model,
            kv_cache,
            scratch,
            capacity: config.n_ctx,
            next_step: 0,
        })
    }

    pub(crate) fn prefill_prompt(&mut self, prompt: &TtsPrompt) -> Result<(), String> {
        if prompt.embeddings.is_empty() || prompt.embeddings.len() % self.model.config.n_embd != 0 {
            return Err("TTS prompt embeddings have an invalid shape".into());
        }
        let rows = prompt.embeddings.len() / self.model.config.n_embd;
        if prompt.positions.len() != rows {
            return Err(format!(
                "TTS prompt positions {} != embedding rows {rows}",
                prompt.positions.len()
            ));
        }
        let end = self
            .next_step
            .checked_add(rows)
            .ok_or_else(|| "TTS prompt step overflow".to_string())?;
        if end > self.capacity {
            return Err(format!(
                "TTS prompt rows {rows} exceed remaining context {}",
                self.capacity - self.next_step
            ));
        }
        for (embedding, &position) in prompt
            .embeddings
            .chunks_exact(self.model.config.n_embd)
            .zip(&prompt.positions)
        {
            self.forward_step_inner(None, Some(embedding), position, true)?;
        }
        Ok(())
    }

    pub(crate) fn sample_semantic<R: Rng + ?Sized>(
        &mut self,
        temperature: f32,
        rng: &mut R,
    ) -> Result<Option<u32>, String> {
        let logits = self.compute_logits()?;
        sample_semantic_logits(
            &logits,
            self.model.config.eos_token_id as usize,
            temperature,
            rng.gen(),
        )
    }

    /// Single-token forward pass: writes one token into the KV cache and
    /// updates `scratch.x` with the resulting hidden state.
    pub(crate) fn forward_step(
        &mut self,
        token_id: u32,
        position: [usize; 4],
    ) -> Result<(), String> {
        self.forward_step_inner(Some(token_id), None, position, true)
    }
}

impl TtsSession<'_> {
    /// Like [`TtsSession::forward_step`] but uses a precomputed 2048-dim
    /// embedding instead of looking up `token_embedding`. Used by the codec
    /// pipeline to feed `out_embd` from the previous frame's code predictor
    /// back into the talker as the next frame's input.
    pub(crate) fn forward_step_with_embedding(
        &mut self,
        embedding: &[f32],
        position: [usize; 4],
    ) -> Result<(), String> {
        self.forward_step_inner(None, Some(embedding), position, true)
    }

    /// Borrow the talker's normalized output state used by the code predictor.
    pub(crate) fn hidden_state(&self) -> &[f32] {
        &self.scratch.normed
    }

    /// Run the output norm + LM head on `scratch.x` and return the audio-
    /// codebook logits. Same as the internal `compute_logits` but exposed.
    pub(crate) fn compute_audio_logits(&mut self) -> Result<Vec<f32>, String> {
        self.compute_logits()
    }

    /// Sample a single token id from the supplied logits (greedy if
    /// temperature <= 0, else temperature-scaled categorical).
    pub(crate) fn sample_from_logits(
        &self,
        logits: &[f32],
        temperature: f32,
    ) -> Result<u32, String> {
        let sampled = sample_semantic_logits(
            logits,
            self.model.config.eos_token_id as usize,
            temperature,
            rand::random(),
        )?;
        Ok(sampled.unwrap_or(self.model.config.eos_token_id))
    }
}

impl TtsSession<'_> {
    fn forward_step_inner(
        &mut self,
        token_id: Option<u32>,
        precomputed_embedding: Option<&[f32]>,
        position: [usize; 4],
        online_attention: bool,
    ) -> Result<(), String> {
        let model = self.model;
        let config = &model.config;
        let n_embd_q = checked_product("query width", config.n_head, config.n_embd_head_k)?;
        let n_embd_k = checked_product("key width", config.n_head_kv, config.n_embd_head_k)?;
        let n_embd_v = checked_product("value width", config.n_head_kv, config.n_embd_head_v)?;
        let n_attn = checked_product(
            "attention output width",
            config.n_head,
            config.n_embd_head_v,
        )?;
        let group_size = config.n_head / config.n_head_kv;
        let kq_scale = 1.0 / (config.n_embd_head_k as f32).sqrt();
        let step = self.next_step;
        if step >= self.capacity {
            return Err(format!("TTS session exceeds context {}", self.capacity));
        }
        self.next_step = self
            .next_step
            .checked_add(1)
            .ok_or_else(|| "TTS session step overflow".to_string())?;

        // Embed the input into scratch.x — either from token_embedding
        // lookup or from a precomputed 2048-dim vector.
        match (token_id, precomputed_embedding) {
            (Some(tid), _) => {
                model.token_embedding.embedding_lookup(tid, &mut self.scratch.x);
            }
            (None, Some(emb)) => {
                if emb.len() != config.n_embd {
                    return Err(format!(
                        "forward_step_with_embedding: embedding length {} != {}",
                        emb.len(),
                        config.n_embd
                    ));
                }
                self.scratch.x.copy_from_slice(emb);
            }
            (None, None) => {
                return Err("forward_step_inner: no token_id or embedding".into());
            }
        }
        let (k_cache_ptr, v_cache_ptr) = match &mut self.kv_cache {
            KvCache::F16(cache) => (cache.k.as_mut_ptr(), cache.v.as_mut_ptr()),
            KvCache::F32(_) => {
                return Err("TtsSession requires an F16 KV cache".into());
            }
        };
        let kv_stride = n_embd_k.max(n_embd_v);
        let kv_cache_size = checked_product(
            "KV cache values",
            checked_product("KV cache rows", config.n_layer, self.capacity)?,
            kv_stride,
        )?;
        let max_n_in = n_embd_q.max(n_attn).max(config.n_ff);

        for layer in 0..config.n_layer {
            let t_layer_start = std::time::Instant::now();
            let weights = &model.layers[layer];
            let x_ptr = self.scratch.x.as_mut_ptr();
            let normed_ptr = self.scratch.normed.as_mut_ptr();
            let q_ptr = self.scratch.q.as_mut_ptr();
            let k_ptr = self.scratch.k_new.as_mut_ptr();
            let v_ptr = self.scratch.v_new.as_mut_ptr();
            let attn_out_ptr = self.scratch.attn_out.as_mut_ptr();
            let attn_proj_ptr = self.scratch.attn_proj.as_mut_ptr();
            let down_buf_ptr = self.scratch.down_buf.as_mut_ptr();
            let gate_buf_ptr = self.scratch.gate_buf.as_mut_ptr();
            let up_buf_ptr = self.scratch.up_buf.as_mut_ptr();
            let q8_buf_ptr = self.scratch.q8_buf.as_mut_ptr();
            let scale_buf_ptr = self.scratch.scale_buf.as_mut_ptr();

            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, config.n_embd) };
            let normed = unsafe { std::slice::from_raw_parts_mut(normed_ptr, config.n_embd) };
            let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
            let scale_buf = unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };

            rms_norm(x, &weights.attn_norm, normed, config.eps);
            quantize_q8_0_into(
                normed,
                config.n_embd,
                &mut q8_buf[..config.n_embd],
                &mut scale_buf[..config.n_embd / 32],
            );
            let q8 = q8_buf[..config.n_embd].as_ptr();
            let scales = scale_buf[..config.n_embd / 32].as_ptr();
            let pool = Arc::clone(&model.pool);
            let wq = &weights.wq;
            let wk = &weights.wk;
            let wv = &weights.wv;
            pool.compute(move |thread, threads| {
                let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_embd) };
                let scales = unsafe { std::slice::from_raw_parts(scales, config.n_embd / 32) };
                let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
                let k = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_k) };
                let v = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_v) };
                wq.kernel.forward_prepared(
                    &[],
                    q8,
                    scales,
                    None,
                    q,
                    config.n_embd,
                    n_embd_q,
                    thread,
                    threads,
                );
                wk.kernel.forward_prepared(
                    &[],
                    q8,
                    scales,
                    None,
                    k,
                    config.n_embd,
                    n_embd_k,
                    thread,
                    threads,
                );
                wv.kernel.forward_prepared(
                    &[],
                    q8,
                    scales,
                    None,
                    v,
                    config.n_embd,
                    n_embd_v,
                    thread,
                    threads,
                );
            });

            {
                let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
                let k = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_k) };
                let v = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_v) };
                for head in q.chunks_exact_mut(config.n_embd_head_k) {
                    rms_norm_inplace(head, &weights.q_norm, config.eps);
                }
                for head in k.chunks_exact_mut(config.n_embd_head_k) {
                    rms_norm_inplace(head, &weights.k_norm, config.eps);
                }
                apply_rope_to_heads(q, position, config);
                apply_rope_to_heads(k, position, config);

                let layer_base = layer * self.capacity * kv_stride;
                let k_cache = unsafe { std::slice::from_raw_parts_mut(k_cache_ptr, kv_cache_size) };
                let v_cache = unsafe { std::slice::from_raw_parts_mut(v_cache_ptr, kv_cache_size) };
                for head in 0..config.n_head_kv {
                    let k_offset = head * config.n_embd_head_k;
                    let v_offset = head * config.n_embd_head_v;
                    let cache_row = layer_base + step * kv_stride;
                    f32_slice_to_f16(
                        &k[k_offset..k_offset + config.n_embd_head_k],
                        &mut k_cache
                            [cache_row + k_offset..cache_row + k_offset + config.n_embd_head_k],
                    );
                    f32_slice_to_f16(
                        &v[v_offset..v_offset + config.n_embd_head_v],
                        &mut v_cache
                            [cache_row + v_offset..cache_row + v_offset + config.n_embd_head_v],
                    );
                }
            }

            let pool = Arc::clone(&model.pool);
            let scores_ptr = self.scratch.scores.as_mut_ptr();
            let score_stride = self.scratch.score_stride;
            let layer_capacity = self.capacity;
            pool.compute(move |thread, threads| {
                let q = unsafe { std::slice::from_raw_parts(q_ptr, n_embd_q) };
                let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_attn) };
                let k_cache = unsafe { std::slice::from_raw_parts(k_cache_ptr, kv_cache_size) };
                let v_cache = unsafe { std::slice::from_raw_parts(v_cache_ptr, kv_cache_size) };
                let head_start = thread * config.n_head / threads;
                let head_end = (thread + 1) * config.n_head / threads;
                let layer_base = layer * layer_capacity * kv_stride;
                if online_attention {
                    let f16_scratch = unsafe {
                        std::slice::from_raw_parts_mut(
                            scores_ptr.add(thread * score_stride).cast::<u16>(),
                            score_stride * 2,
                        )
                    };
                    let (query, accumulator) = f16_scratch.split_at_mut(config.n_embd_head_k);
                    let accumulator = &mut accumulator[..config.n_embd_head_v];
                    for head in head_start..head_end {
                        let kv_head = head / group_size;
                        let q_offset = head * config.n_embd_head_k;
                        let output_offset = head * config.n_embd_head_v;
                        let output =
                            &mut attn_out[output_offset..output_offset + config.n_embd_head_v];
                        f32_slice_to_f16(&q[q_offset..q_offset + config.n_embd_head_k], query);
                        accumulator.fill(0);
                        let mut sum = 0.0f32;
                        let mut max = f32::NEG_INFINITY;
                        for token in 0..=step {
                            let row = layer_base + token * kv_stride;
                            let key_offset = row + kv_head * config.n_embd_head_k;
                            let score = dot_f16(
                                query,
                                &k_cache[key_offset..key_offset + config.n_embd_head_k],
                                config.n_embd_head_k,
                            ) * kq_scale;
                            let mut rescale = 1.0f32;
                            let mut weight = 1.0f32;
                            if score > max {
                                rescale = (max - score).exp();
                                max = score;
                                #[cfg(target_arch = "aarch64")]
                                unsafe {
                                    scale_f16_inplace(accumulator, rescale);
                                }
                                #[cfg(not(target_arch = "aarch64"))]
                                scale_f16_inplace(accumulator, rescale);
                            } else {
                                weight = (score - max).exp();
                            }
                            let value_offset = row + kv_head * config.n_embd_head_v;
                            #[cfg(target_arch = "aarch64")]
                            unsafe {
                                mad_f16_inplace(
                                    accumulator,
                                    &v_cache[value_offset..value_offset + config.n_embd_head_v],
                                    weight,
                                );
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            mad_f16_inplace(
                                accumulator,
                                &v_cache[value_offset..value_offset + config.n_embd_head_v],
                                weight,
                            );
                            sum = sum.mul_add(rescale, weight);
                        }
                        for (output, &value) in output.iter_mut().zip(accumulator.iter()) {
                            *output = f16_to_f32(value);
                        }
                        vec_scale_f32(output, if sum == 0.0 { 0.0 } else { sum.recip() });
                    }
                } else {
                    let scores = unsafe {
                        std::slice::from_raw_parts_mut(
                            scores_ptr.add(thread * score_stride),
                            score_stride,
                        )
                    };
                    let f16_scratch = scores.as_mut_ptr().cast::<u16>();
                    let n_padded = (step + 1).div_ceil(256) * 256;
                    for head in head_start..head_end {
                        let kv_head = head / group_size;
                        let q_offset = head * config.n_embd_head_k;
                        let output_offset = head * config.n_embd_head_v;
                        let output =
                            &mut attn_out[output_offset..output_offset + config.n_embd_head_v];
                        let query = unsafe {
                            std::slice::from_raw_parts_mut(
                                output.as_mut_ptr().cast::<u16>(),
                                config.n_embd_head_k,
                            )
                        };
                        f32_slice_to_f16(&q[q_offset..q_offset + config.n_embd_head_k], query);
                        scores[..n_padded].fill(f32::NEG_INFINITY);
                        for token in 0..=step {
                            let row = layer_base + token * kv_stride;
                            let key_offset = row + kv_head * config.n_embd_head_k;
                            scores[token] = dot_f16(
                                query,
                                &k_cache[key_offset..key_offset + config.n_embd_head_k],
                                config.n_embd_head_k,
                            ) * kq_scale;
                        }
                        softmax_inplace(&mut scores[..n_padded]);
                        for index in 0..n_padded {
                            unsafe { *f16_scratch.add(index) = f32_to_f16(scores[index]) };
                        }
                        let weights = unsafe { std::slice::from_raw_parts(f16_scratch, n_padded) };
                        let values = unsafe {
                            std::slice::from_raw_parts_mut(f16_scratch.add(score_stride), n_padded)
                        };
                        values[step + 1..].fill(0);
                        for dimension in 0..config.n_embd_head_v {
                            for token in 0..=step {
                                let row = layer_base + token * kv_stride;
                                values[token] =
                                    v_cache[row + kv_head * config.n_embd_head_v + dimension];
                            }
                            output[dimension] = dot_f16(values, weights, n_padded);
                        }
                    }
                }
            });

            let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_attn) };
            let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
            let scale_buf = unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
            quantize_q8_0_into(
                attn_out,
                n_attn,
                &mut q8_buf[..n_attn],
                &mut scale_buf[..n_attn / 32],
            );
            let q8 = q8_buf[..n_attn].as_ptr();
            let scales = scale_buf[..n_attn / 32].as_ptr();
            let pool = Arc::clone(&model.pool);
            let wo = &weights.wo;
            pool.compute(move |thread, threads| {
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_attn) };
                let scales = unsafe { std::slice::from_raw_parts(scales, n_attn / 32) };
                let output =
                    unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, config.n_embd) };
                wo.kernel.forward_prepared(
                    &[],
                    q8,
                    scales,
                    None,
                    output,
                    n_attn,
                    config.n_embd,
                    thread,
                    threads,
                );
            });

            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, config.n_embd) };
            let attn_projection =
                unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, config.n_embd) };
            for (hidden, projection) in x.iter_mut().zip(attn_projection) {
                *hidden += *projection;
            }
            rms_norm(x, &weights.ffn_norm, normed, config.eps);
            quantize_q8_0_into(
                normed,
                config.n_embd,
                &mut q8_buf[..config.n_embd],
                &mut scale_buf[..config.n_embd / 32],
            );
            let q8 = q8_buf[..config.n_embd].as_ptr();
            let scales = scale_buf[..config.n_embd / 32].as_ptr();
            let pool = Arc::clone(&model.pool);
            let w_gate = &weights.w_gate;
            let w_up = &weights.w_up;
            pool.compute(move |thread, threads| {
                let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_embd) };
                let scales = unsafe { std::slice::from_raw_parts(scales, config.n_embd / 32) };
                let gate = unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, config.n_ff) };
                let up = unsafe { std::slice::from_raw_parts_mut(up_buf_ptr, config.n_ff) };
                w_gate.kernel.forward_prepared(
                    &[],
                    q8,
                    scales,
                    None,
                    up,
                    config.n_embd,
                    config.n_ff,
                    thread,
                    threads,
                );
                w_up.kernel.forward_prepared(
                    &[],
                    q8,
                    scales,
                    None,
                    gate,
                    config.n_embd,
                    config.n_ff,
                    thread,
                    threads,
                );
                let start = thread * config.n_ff / threads;
                let end = (thread + 1) * config.n_ff / threads;
                silu_mul_approx_inplace(&up[start..end], &mut gate[start..end]);
            });

            let gate = unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, config.n_ff) };
            quantize_q8_0_into(
                gate,
                config.n_ff,
                &mut q8_buf[..config.n_ff],
                &mut scale_buf[..config.n_ff / 32],
            );
            let q8 = q8_buf[..config.n_ff].as_ptr();
            let scales = scale_buf[..config.n_ff / 32].as_ptr();
            let pool = Arc::clone(&model.pool);
            let w_down = &weights.w_down;
            pool.compute(move |thread, threads| {
                let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_ff) };
                let scales = unsafe { std::slice::from_raw_parts(scales, config.n_ff / 32) };
                let down = unsafe { std::slice::from_raw_parts_mut(down_buf_ptr, config.n_embd) };
                w_down.kernel.forward_prepared(
                    &[],
                    q8,
                    scales,
                    None,
                    down,
                    config.n_ff,
                    config.n_embd,
                    thread,
                    threads,
                );
            });

            let down = unsafe { std::slice::from_raw_parts_mut(down_buf_ptr, config.n_embd) };
            let x = unsafe { std::slice::from_raw_parts_mut(x_ptr, config.n_embd) };
            for (hidden, projection) in x.iter_mut().zip(down) {
                *hidden += *projection;
            }
            eprintln!("  [layer {}] took {:.3}ms (step={})", layer, t_layer_start.elapsed().as_secs_f64() * 1000.0, step);
        }
        Ok(())
    }

    fn compute_logits(&mut self) -> Result<Vec<f32>, String> {
        let model = self.model;
        let config = &model.config;
        rms_norm(
            &self.scratch.x,
            &model.output_norm,
            &mut self.scratch.normed,
            config.eps,
        );
        let n_embd = config.n_embd;
        let blocks = (n_embd + 31) / 32;
        if self.scratch.q8_buf.len() < n_embd {
            self.scratch.q8_buf.resize(n_embd, 0);
            self.scratch.scale_buf.resize(blocks, 0.0);
        }
        quantize_q8_0_into(
            &self.scratch.normed,
            n_embd,
            &mut self.scratch.q8_buf[..n_embd],
            &mut self.scratch.scale_buf[..blocks],
        );
        let mut logits = vec![0.0f32; config.audio_codebook_size];
        model.audio_output_head.kernel.forward_prepared(
            &[],
            &self.scratch.q8_buf[..n_embd],
            &self.scratch.scale_buf[..blocks],
            None,
            &mut logits,
            n_embd,
            config.audio_codebook_size,
            0,
            1,
        );
        Ok(logits)
    }
}

fn compose_prompt_embeddings<F>(
    wrapper_ids: &[u32],
    specials: &TtsSpecialTokens,
    speaker: Option<&[f32]>,
    n_embd: usize,
    mut embedding_row: F,
) -> Result<TtsPrompt, String>
where
    F: FnMut(u32) -> Result<Vec<f32>, String>,
{
    if wrapper_ids.len() < 8 {
        return Err(format!(
            "TTS ChatML wrapper produced {} tokens; expected at least 8",
            wrapper_ids.len()
        ));
    }
    if n_embd == 0 {
        return Err("TTS embedding width must be nonzero".into());
    }
    if speaker.is_some_and(|row| row.len() != n_embd || row.iter().any(|v| !v.is_finite())) {
        return Err(format!(
            "TTS speaker embedding must contain {n_embd} finite values"
        ));
    }

    let load = |id: u32, embedding_row: &mut F| {
        let row = embedding_row(id)?;
        if row.len() != n_embd || row.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "TTS token {id} embedding must contain {n_embd} finite values"
            ));
        }
        Ok(row)
    };
    let sum = |left: Vec<f32>, right: &[f32]| -> Result<Vec<f32>, String> {
        if right.len() != n_embd {
            return Err(format!(
                "TTS embedding sum width {} != {n_embd}",
                right.len()
            ));
        }
        let output: Vec<f32> = left.into_iter().zip(right).map(|(a, b)| a + b).collect();
        if output.iter().any(|value| !value.is_finite()) {
            return Err("TTS composed embedding contains non-finite values".into());
        }
        Ok(output)
    };

    let mut rows = Vec::new();
    for &token_id in &wrapper_ids[..3] {
        rows.push(load(token_id, &mut embedding_row)?);
    }
    let tts_pad = load(specials.tts_pad, &mut embedding_row)?;
    for token_id in [
        specials.codec_think,
        specials.codec_think_bos,
        specials.codec_language,
        specials.codec_think_eos,
    ] {
        rows.push(sum(tts_pad.clone(), &load(token_id, &mut embedding_row)?)?);
    }
    if let Some(speaker) = speaker {
        rows.push(sum(tts_pad.clone(), speaker)?);
    }

    let codec_pad = load(specials.codec_pad, &mut embedding_row)?;
    let text_bos = load(specials.tts_text_bos, &mut embedding_row)?;
    rows.push(sum(text_bos, &codec_pad)?);

    let body_ids = &wrapper_ids[3..wrapper_ids.len() - 5];
    let mut overlay = Vec::with_capacity(body_ids.len() + 2);
    for &token_id in body_ids {
        let text = load(token_id, &mut embedding_row)?;
        rows.push(sum(text.clone(), &codec_pad)?);
        overlay.push(text);
    }
    let text_eod = load(specials.tts_text_eod, &mut embedding_row)?;
    rows.push(sum(text_eod.clone(), &codec_pad)?);
    overlay.push(text_eod);
    rows.push(sum(
        tts_pad.clone(),
        &load(specials.codec_bos, &mut embedding_row)?,
    )?);
    overlay.push(tts_pad);

    let positions = (0..rows.len()).map(|index| [index; 4]).collect();
    Ok(TtsPrompt {
        wrapper_ids: wrapper_ids.to_vec(),
        embeddings: rows.into_iter().flatten().collect(),
        positions,
        overlay,
    })
}

fn sample_semantic_logits(
    logits: &[f32],
    eos_row: usize,
    temperature: f32,
    draw: f32,
) -> Result<Option<u32>, String> {
    if logits.len() <= SEMANTIC_CODEBOOK_SIZE || eos_row >= logits.len() {
        return Err(format!(
            "TTS logits length {} cannot address semantic rows and EOS {eos_row}",
            logits.len()
        ));
    }
    if eos_row < SEMANTIC_CODEBOOK_SIZE {
        return Err(format!("TTS EOS row {eos_row} overlaps semantic rows"));
    }
    if !temperature.is_finite() {
        return Err("TTS temperature must be finite".into());
    }

    let candidates = (0..SEMANTIC_CODEBOOK_SIZE).chain(std::iter::once(eos_row));
    if temperature <= 0.0 {
        let mut best = None;
        for index in candidates {
            let logit = logits[index];
            if logit.is_nan() || logit == f32::INFINITY {
                return Err("Cannot sample non-finite TTS logits".into());
            }
            if best.is_none_or(|(_, value)| logit > value) {
                best = Some((index, logit));
            }
        }
        let (index, value) = best.expect("semantic candidates are non-empty");
        if value == f32::NEG_INFINITY {
            return Err("All legal TTS logits are suppressed".into());
        }
        return Ok((index != eos_row).then_some(index as u32));
    }

    if !(0.0..1.0).contains(&draw) || !draw.is_finite() {
        return Err(format!("TTS sampling draw {draw} is outside [0, 1)"));
    }
    let max = candidates.clone().map(|index| logits[index]).try_fold(
        f32::NEG_INFINITY,
        |max, logit| {
            if logit.is_nan() || logit == f32::INFINITY {
                Err("Cannot sample non-finite TTS logits".to_string())
            } else {
                Ok(max.max(logit))
            }
        },
    )?;
    if max == f32::NEG_INFINITY {
        return Err("All legal TTS logits are suppressed".into());
    }
    let sum: f32 = candidates
        .clone()
        .map(|index| ((logits[index] - max) / temperature).exp())
        .sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("TTS sampling probability sum is not finite and positive".into());
    }
    let target = draw * sum;
    let mut cumulative = 0.0;
    let mut last = None;
    for index in candidates {
        let probability = ((logits[index] - max) / temperature).exp();
        if probability == 0.0 {
            continue;
        }
        cumulative += probability;
        last = Some(index);
        if cumulative > target {
            return Ok((index != eos_row).then_some(index as u32));
        }
    }
    let index = last.expect("positive sampling sum has a candidate");
    Ok((index != eos_row).then_some(index as u32))
}

fn apply_rope_to_heads(heads: &mut [f32], position: [usize; 4], config: &Qwen3TtsTalkerConfig) {
    let n_dims = config.n_embd_head_k;
    if config.rope_sections.iter().any(|&value| value > 0) {
        rope_mrope_interleaved(
            heads,
            position,
            config.rope_sections,
            n_dims,
            config.freq_base,
            n_dims,
        );
    } else {
        for head in heads.chunks_exact_mut(n_dims) {
            rope_neox(head, position[0], n_dims, config.freq_base);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_rows_follow_qwen3_tts_reference_order() {
        let specials = TtsSpecialTokens {
            codec_0: 100,
            codec_bos: 101,
            codec_eos: 102,
            codec_pad: 103,
            codec_think: 104,
            codec_think_bos: 105,
            codec_think_eos: 106,
            codec_language: 107,
            tts_pad: 108,
            tts_text_bos: 109,
            tts_text_eod: 110,
        };
        let wrapper = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let row = |id: u32| Ok(vec![id as f32, 1.0]);
        let prompt =
            compose_prompt_embeddings(&wrapper, &specials, Some(&[0.5, 1.5]), 2, row).unwrap();
        let rows: Vec<[f32; 2]> = prompt
            .embeddings
            .chunks_exact(2)
            .map(|row| [row[0], row[1]])
            .collect();
        assert_eq!(
            rows,
            vec![
                [1.0, 1.0],
                [2.0, 1.0],
                [3.0, 1.0],
                [212.0, 2.0],
                [213.0, 2.0],
                [215.0, 2.0],
                [214.0, 2.0],
                [108.5, 2.5],
                [212.0, 2.0],
                [107.0, 2.0],
                [213.0, 2.0],
                [209.0, 2.0],
            ]
        );
        assert_eq!(
            prompt.overlay,
            vec![vec![4.0, 1.0], vec![110.0, 1.0], vec![108.0, 1.0]]
        );
        assert!(prompt
            .positions
            .iter()
            .enumerate()
            .all(|(index, position)| *position == [index, index, index, index]));
    }

    #[test]
    fn semantic_sampling_never_returns_control_rows() {
        let mut logits = vec![f32::NEG_INFINITY; 3072];
        logits[2047] = 3.0;
        logits[2048] = 100.0;
        logits[2150] = 2.0;
        assert_eq!(
            sample_semantic_logits(&logits, 2150, 0.0, 0.5).unwrap(),
            Some(2047)
        );
        logits[2150] = 4.0;
        assert_eq!(
            sample_semantic_logits(&logits, 2150, 0.0, 0.5).unwrap(),
            None
        );

        logits.fill(f32::NEG_INFINITY);
        logits[0] = 0.0;
        logits[2048] = 100.0;
        assert_eq!(
            sample_semantic_logits(&logits, 2150, 1.0, 0.999).unwrap(),
            Some(0)
        );
    }

    #[test]
    fn missing_special_token_names_the_required_literal() {
        let error = resolve_tts_special_tokens("english", |literal| {
            (literal != "<|codec_eos_token|>").then_some(7)
        })
        .unwrap_err();
        assert!(error.contains("<|codec_eos_token|>"));
    }
}
