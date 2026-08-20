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

use crate::core::scratchpad::{ExecutionScratchpad, KvCache};
use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::qwen3::{
    check_allocation, checked_product, load_f32_tensor, qwen_text_positions, static_q8_matrix,
    static_q8_tensor, static_tensor, usize_to_u64, validate_token_ids, Qwen3Config,
};
use crate::ops::{
    dot_f16, dot_f32, embedding_lookup, f32_slice_to_f16, f32_to_f16,
    matmul_q8_0_quantized_parallel_rows, quantize_q8_0_into, rms_norm, rms_norm_inplace,
    rope_mrope_interleaved, rope_neox, silu_mul_inplace, softmax,
};
use crate::models::qwen3::sample_token;
use crate::models::tts::{AUDIO_CODEBOOK_SIZE, TTS_DEFAULT_TEMP, TTS_EOS_TOKEN_ID};

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
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
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
        let rope_sections = read_i32_array(source, "qwen3tts.rope.dimension_sections")
            .unwrap_or([24, 20, 20, 0]);

        let qwen3 = Qwen3Config::from_source(source)?;

        let vocab_size = source
            .metadata("tokenizer.ggml.tokens")
            .and_then(MetaValue::to_arr)
            .map(Vec::len)
            .unwrap_or(0);

        // The talker emits tokens over the full vocab (text + audio), so the EOS
        // id from the tokenizer metadata is a full-vocab id. Convert it to
        // the narrow audio-codebook range by subtracting the audio offset
        // (= full vocab size - audio codebook size).
        let raw_eos = source
            .metadata("tokenizer.ggml.eos_token_id")
            .and_then(MetaValue::to_u64)
            .map(|v| v as u32)
            .unwrap_or(TTS_EOS_TOKEN_ID);
        let vocab_size = source
            .metadata("tokenizer.ggml.tokens")
            .and_then(MetaValue::to_arr)
            .map(Vec::len)
            .unwrap_or(0);
        let eos_token_id = raw_eos.saturating_sub((vocab_size - AUDIO_CODEBOOK_SIZE) as u32);

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

fn read_i32_array(
    source: &dyn TensorSource,
    key: &str,
) -> Result<[i32; 4], String> {
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
    pub(crate) wq: &'static [u8],
    pub(crate) wk: &'static [u8],
    pub(crate) wv: &'static [u8],
    pub(crate) wo: &'static [u8],
    pub(crate) w_gate: &'static [u8],
    pub(crate) w_up: &'static [u8],
    pub(crate) w_down: &'static [u8],
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
    token_embedding: &'static [u8],
    audio_output_head: &'static [u8],
    embedding_type: GGMLType,
}

impl Qwen3TtsTalker {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        tokenizer: Arc<BPETokenizer>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let config = Qwen3TtsTalkerConfig::from_source(source.as_ref())?;
        if config.vocab_size != tokenizer.vocab_size() {
            return Err(format!(
                "{} vocabulary size {} does not match tokenizer vocab {}",
                config.architecture, config.vocab_size, tokenizer.vocab_size(),
            ));
        }

        let n_embd_q = checked_product("query width", config.n_head, config.n_embd_head_k)?;
        let n_embd_k = checked_product("key width", config.n_head_kv, config.n_embd_head_k)?;
        let n_embd_v = checked_product("value width", config.n_head_kv, config.n_embd_head_v)?;
        let n_attn =
            checked_product("attention output width", config.n_head, config.n_embd_head_v)?;

        let output_norm = load_f32_tensor(
            source.as_ref(),
            "output_norm.weight",
            &[usize_to_u64(config.n_embd, "embedding width")?],
        )?;

        let embedding_dims = [
            usize_to_u64(config.n_embd, "embedding width")?,
            usize_to_u64(config.vocab_size, "vocabulary size")?,
        ];
        let embedding_type = source
            .tensor_info("token_embd.weight")
            .map(|info| info.ggml_type)
            .unwrap_or(GGMLType::Q8_0);
        let token_embedding = static_tensor(
            source.as_ref(),
            "token_embd.weight",
            &embedding_dims,
            embedding_type,
        )?;

        let audio_head_dims = [
            usize_to_u64(config.n_embd, "embedding width")?,
            usize_to_u64(config.audio_codebook_size, "audio codebook")?,
        ];
        let audio_output_head = static_q8_tensor(
            source.as_ref(),
            "output.weight",
            &audio_head_dims,
        )?;

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
                ffn_norm: load_f32_tensor(
                    source.as_ref(),
                    &name("ffn_norm.weight"),
                    &n_embd_dim,
                )?,
                q_norm: load_f32_tensor(
                    source.as_ref(),
                    &name("attn_q_norm.weight"),
                    &head_dim,
                )?,
                k_norm: load_f32_tensor(
                    source.as_ref(),
                    &name("attn_k_norm.weight"),
                    &head_dim,
                )?,
                wq: static_q8_matrix(
                    source.as_ref(),
                    &name("attn_q.weight"),
                    config.n_embd,
                    n_embd_q,
                )?,
                wk: static_q8_matrix(
                    source.as_ref(),
                    &name("attn_k.weight"),
                    config.n_embd,
                    n_embd_k,
                )?,
                wv: static_q8_matrix(
                    source.as_ref(),
                    &name("attn_v.weight"),
                    config.n_embd,
                    n_embd_v,
                )?,
                wo: static_q8_matrix(
                    source.as_ref(),
                    &name("attn_output.weight"),
                    n_attn,
                    config.n_embd,
                )?,
                w_gate: static_q8_matrix(
                    source.as_ref(),
                    &name("ffn_gate.weight"),
                    config.n_embd,
                    config.n_ff,
                )?,
                w_up: static_q8_matrix(
                    source.as_ref(),
                    &name("ffn_up.weight"),
                    config.n_embd,
                    config.n_ff,
                )?,
                w_down: static_q8_matrix(
                    source.as_ref(),
                    &name("ffn_down.weight"),
                    config.n_ff,
                    config.n_embd,
                )?,
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
            embedding_type,
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

    /// Encode `text` and feed it through the Talker, sampling up to
    /// `max_new_tokens` audio-codebook tokens. Generation terminates early
    /// when the EOS audio codebook id is sampled.
    pub fn synthesize(
        &self,
        text: &str,
        positions: Option<&[[usize; 4]]>,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Qwen3TtsGeneration, String> {
        let token_ids: Vec<u32> = self.tokenizer.encode(
            text,
            EncodeOptions {
                add_special: false,
                parse_special: false,
            },
        );
        validate_token_ids(&token_ids, self.config.vocab_size)?;

        let positions_owned;
        let positions_ref = match positions {
            Some(p) => p,
            None => {
                positions_owned = qwen_text_positions(token_ids.len());
                &positions_owned
            }
        };

        let mut session = TtsSession::new(self)?;
        session.synthesize(&token_ids, positions_ref, max_new_tokens, temperature)
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
        let n_attn =
            checked_product("attention output width", config.n_head, config.n_embd_head_v)?;
        let scratch = ExecutionScratchpad::new(
            config.n_embd,
            n_embd_q,
            n_embd_k.max(n_embd_v),
            config.n_ff,
            config.audio_codebook_size,
            model.pool.n_threads(),
            config.n_ctx,
        );
        let kv_cache = KvCache::new_f16(
            config.n_layer,
            config.n_ctx,
            n_embd_k.max(n_embd_v),
        );
        Ok(Self {
            model,
            kv_cache,
            scratch,
            capacity: config.n_ctx,
            next_step: 0,
        })
    }

    pub(crate) fn synthesize(
        &mut self,
        prompt_ids: &[u32],
        positions: &[[usize; 4]],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Qwen3TtsGeneration, String> {
        if prompt_ids.is_empty() {
            return Err("TTS prompt must contain at least one token".into());
        }
        let config = &self.model.config;
        if positions.len() != prompt_ids.len() {
            return Err(format!(
                "positions length {} != prompt length {}",
                positions.len(),
                prompt_ids.len(),
            ));
        }
        let prompt_len = prompt_ids.len();
        let total = prompt_len
            .checked_add(max_new_tokens)
            .ok_or_else(|| "TTS prompt+generation length overflow".to_string())?;
        if total > config.n_ctx {
            return Err(format!(
                "prompt+generation={total} exceeds TTS context {}",
                config.n_ctx,
            ));
        }

        // Prefill: run each prompt token through the single-token forward.
        let mut next_position: Option<[usize; 4]> = None;
        for (&token_id, &pos) in prompt_ids.iter().zip(positions.iter()) {
            self.forward_step(token_id, pos)?;
            next_position = Some(pos);
        }
        let mut next_position = next_position.expect("non-empty prompt");

        // Sample first audio token from the post-prefill hidden state.
        let mut logits = self.compute_logits()?;
        let temperature = if temperature <= 0.0 { 0.0 } else { temperature };
        let mut next_token = sample_token(&logits, temperature)?;
        let mut audio_ids = Vec::with_capacity(max_new_tokens);
        let mut finished = TtsFinishReason::Length;
        if next_token == self.model.config.eos_token_id {
            finished = TtsFinishReason::Eos;
        } else if (next_token as usize) >= self.model.config.audio_codebook_size {
            finished = TtsFinishReason::Invalid;
        } else {
            audio_ids.push(next_token);
        }

        for _step in 1..max_new_tokens {
            if !matches!(finished, TtsFinishReason::Length) {
                break;
            }
            next_position[0] = next_position[0].saturating_add(1);
            self.forward_step(next_token, next_position)?;
            logits = self.compute_logits()?;
            next_token = sample_token(&logits, temperature)?;
            if next_token == self.model.config.eos_token_id {
                finished = TtsFinishReason::Eos;
                break;
            }
            if (next_token as usize) >= self.model.config.audio_codebook_size {
                finished = TtsFinishReason::Invalid;
                break;
            }
            audio_ids.push(next_token);
        }

        Ok(Qwen3TtsGeneration {
            prompt_token_ids: prompt_ids.to_vec(),
            audio_token_ids: audio_ids,
            finished_reason: finished,
        })
    }

    /// Single-token forward pass: writes one token into the KV cache and
    /// updates `scratch.x` with the resulting hidden state.
    pub(crate) fn forward_step(&mut self, token_id: u32, position: [usize; 4]) -> Result<(), String> {
        self.forward_step_inner(Some(token_id), None, position)
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
        self.forward_step_inner(None, Some(embedding), position)
    }

    /// Borrow the talker's current hidden state (length = `n_embd`, set by
    /// the most recent forward step). Used by the code predictor as `h_state`.
    pub(crate) fn hidden_state(&self) -> &[f32] {
        &self.scratch.x
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
        let temp = if temperature <= 0.0 { 0.0 } else { temperature };
        sample_token(logits, temp)
    }
}

impl TtsSession<'_> {
    fn forward_step_inner(
        &mut self,
        token_id: Option<u32>,
        precomputed_embedding: Option<&[f32]>,
        position: [usize; 4],
    ) -> Result<(), String> {
        let model = self.model;
        let config = &model.config;
        let n_embd_q = checked_product("query width", config.n_head, config.n_embd_head_k)?;
        let n_embd_k = checked_product("key width", config.n_head_kv, config.n_embd_head_k)?;
        let n_embd_v = checked_product("value width", config.n_head_kv, config.n_embd_head_v)?;
        let n_attn =
            checked_product("attention output width", config.n_head, config.n_embd_head_v)?;
        let group_size = config.n_head / config.n_head_kv;
        let kq_scale = 1.0 / (config.n_embd_head_k as f32).sqrt();
        let step = self.next_step;
        self.next_step = self
            .next_step
            .checked_add(1)
            .ok_or_else(|| "TTS session step overflow".to_string())?;

        // Embed the input into scratch.x — either from token_embedding
        // lookup or from a precomputed 2048-dim vector.
        match (token_id, precomputed_embedding) {
            (Some(tid), _) => {
                embedding_lookup(
                    model.token_embedding,
                    tid,
                    config.n_embd,
                    model.embedding_type,
                    &mut self.scratch.x,
                );
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
            let scale_buf =
                unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };

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
            pool.compute(move |thread, threads| {
                let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_embd) };
                let scales = unsafe { std::slice::from_raw_parts(scales, config.n_embd / 32) };
                let q = unsafe { std::slice::from_raw_parts_mut(q_ptr, n_embd_q) };
                let k = unsafe { std::slice::from_raw_parts_mut(k_ptr, n_embd_k) };
                let v = unsafe { std::slice::from_raw_parts_mut(v_ptr, n_embd_v) };
                matmul_q8_0_quantized_parallel_rows(
                    weights.wq,
                    q8,
                    scales,
                    q,
                    config.n_embd,
                    n_embd_q,
                    thread,
                    threads,
                );
                matmul_q8_0_quantized_parallel_rows(
                    weights.wk,
                    q8,
                    scales,
                    k,
                    config.n_embd,
                    n_embd_k,
                    thread,
                    threads,
                );
                matmul_q8_0_quantized_parallel_rows(
                    weights.wv,
                    q8,
                    scales,
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
                let k_cache =
                    unsafe { std::slice::from_raw_parts_mut(k_cache_ptr, kv_cache_size) };
                let v_cache =
                    unsafe { std::slice::from_raw_parts_mut(v_cache_ptr, kv_cache_size) };
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
                let scores = unsafe {
                    std::slice::from_raw_parts_mut(
                        scores_ptr.add(thread * score_stride),
                        score_stride,
                    )
                };
                let f16_scratch = scores.as_mut_ptr().cast::<u16>();
                let head_start = thread * config.n_head / threads;
                let head_end = (thread + 1) * config.n_head / threads;
                let layer_base = layer * layer_capacity * kv_stride;
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
                    softmax(&mut scores[..n_padded]);
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
            });

            let attn_out = unsafe { std::slice::from_raw_parts_mut(attn_out_ptr, n_attn) };
            let q8_buf = unsafe { std::slice::from_raw_parts_mut(q8_buf_ptr, max_n_in) };
            let scale_buf =
                unsafe { std::slice::from_raw_parts_mut(scale_buf_ptr, max_n_in / 32) };
            quantize_q8_0_into(
                attn_out,
                n_attn,
                &mut q8_buf[..n_attn],
                &mut scale_buf[..n_attn / 32],
            );
            let q8 = q8_buf[..n_attn].as_ptr();
            let scales = scale_buf[..n_attn / 32].as_ptr();
            let pool = Arc::clone(&model.pool);
            pool.compute(move |thread, threads| {
                let q8 = unsafe { std::slice::from_raw_parts(q8, n_attn) };
                let scales = unsafe { std::slice::from_raw_parts(scales, n_attn / 32) };
                let output =
                    unsafe { std::slice::from_raw_parts_mut(attn_proj_ptr, config.n_embd) };
                matmul_q8_0_quantized_parallel_rows(
                    weights.wo,
                    q8,
                    scales,
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
            pool.compute(move |thread, threads| {
                let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_embd) };
                let scales = unsafe { std::slice::from_raw_parts(scales, config.n_embd / 32) };
                let gate = unsafe { std::slice::from_raw_parts_mut(gate_buf_ptr, config.n_ff) };
                let up = unsafe { std::slice::from_raw_parts_mut(up_buf_ptr, config.n_ff) };
                matmul_q8_0_quantized_parallel_rows(
                    weights.w_gate,
                    q8,
                    scales,
                    up,
                    config.n_embd,
                    config.n_ff,
                    thread,
                    threads,
                );
                matmul_q8_0_quantized_parallel_rows(
                    weights.w_up,
                    q8,
                    scales,
                    gate,
                    config.n_embd,
                    config.n_ff,
                    thread,
                    threads,
                );
                let start = thread * config.n_ff / threads;
                let end = (thread + 1) * config.n_ff / threads;
                silu_mul_inplace(&up[start..end], &mut gate[start..end]);
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
            pool.compute(move |thread, threads| {
                let q8 = unsafe { std::slice::from_raw_parts(q8, config.n_ff) };
                let scales = unsafe { std::slice::from_raw_parts(scales, config.n_ff / 32) };
                let down =
                    unsafe { std::slice::from_raw_parts_mut(down_buf_ptr, config.n_embd) };
                matmul_q8_0_quantized_parallel_rows(
                    weights.w_down,
                    q8,
                    scales,
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
        matmul_q8_0_quantized_parallel_rows(
            model.audio_output_head,
            &self.scratch.q8_buf[..n_embd],
            &self.scratch.scale_buf[..blocks],
            &mut logits,
            n_embd,
            config.audio_codebook_size,
            0,
            1,
        );
        Ok(logits)
    }
}

fn apply_rope_to_heads(
    heads: &mut [f32],
    position: [usize; 4],
    config: &Qwen3TtsTalkerConfig,
) {
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