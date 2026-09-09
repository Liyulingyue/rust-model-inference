//! Breeze TTS 2: lossless BF16 text/backbone/depth inference and native 24 kHz codec.
mod bf16_math;
pub mod codec;
#[cfg(test)]
mod tests;
mod transformer;

use crate::core::tensor::{load_f32_tensor, MetaValue, TensorSource};
use crate::ops::kernel::Weight;
pub use codec::BreezeCodec;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde_json::Value;
use transformer::{linear, matrix, Cache, Kind, Transformer};

const HIDDEN: usize = 2048;
const TEXT_HIDDEN: usize = 1152;
const TEXT_VOCAB: usize = 262158;
const CODEBOOKS: usize = 16;
const VOCAB: usize = 2051;
const CODE_SIZE: usize = 2048;
const AUDIO: u32 = 262144;
const AUDIO_EOS: u32 = 262145;

pub struct BreezeModel<'a> {
    tokenizer: tokenizers::Tokenizer,
    text_embeddings: Weight<'a>,
    eoi: Vec<f32>,
    text: Transformer<'a>,
    projection: Weight<'a>,
    backbone: Transformer<'a>,
    lm_head: Weight<'a>,
    audio_embeddings: Weight<'a>,
    depth: Transformer<'a>,
    depth_projection: Weight<'a>,
    depth_heads: Vec<u8>,
    pool: rayon::ThreadPool,
}

impl<'a> BreezeModel<'a> {
    pub fn from_source(source: &'a dyn TensorSource, threads: usize) -> Result<Self, String> {
        // These tensors use the shared F32/BF16 loader, but the original checkpoint
        // stores all of them as BF16. Missing tensors, shapes and bytes are checked below.
        let require_bf16 = |name: &str| -> Result<(), String> {
            if source
                .tensor_info(name)
                .is_some_and(|info| info.ggml_type != crate::core::tensor::GGMLType::BF16)
            {
                return Err(format!("{name}: Breeze requires original BF16 weights"));
            }
            Ok(())
        };
        require_bf16("depth_decoder.codebooks_head.weight")?;
        require_bf16("text_encoder.embed_tokens.eoi_embedding")?;
        for (prefix, count, norms) in [
            (
                "text_encoder",
                26,
                &[
                    "pre_self_attn_layernorm",
                    "post_self_attn_layernorm",
                    "pre_feedforward_layernorm",
                    "post_feedforward_layernorm",
                    "self_attn.q_norm",
                    "self_attn.k_norm",
                ][..],
            ),
            (
                "backbone_model",
                28,
                &[
                    "input_layernorm",
                    "post_attention_layernorm",
                    "self_attn.q_norm",
                    "self_attn.k_norm",
                ][..],
            ),
            (
                "depth_decoder.model",
                12,
                &["input_layernorm", "post_attention_layernorm"][..],
            ),
        ] {
            require_bf16(&format!("{prefix}.norm.weight"))?;
            for index in 0..count {
                for norm in norms {
                    require_bf16(&format!("{prefix}.layers.{index}.{norm}.weight"))?;
                }
            }
        }
        validate_config(source)?;
        if threads == 0 {
            return Err("Breeze threads must be positive".into());
        }
        let tokenizer = tokenizers::Tokenizer::from_bytes(
            meta_string(source, "breeze.tokenizer_json")?.as_bytes(),
        )
        .map_err(|e| format!("Invalid Breeze tokenizer: {e}"))?;
        if tokenizer.get_vocab_size(true) != TEXT_VOCAB
            || tokenizer.token_to_id("<|AUDIO|>") != Some(AUDIO)
            || tokenizer.token_to_id("[S0]") != Some(262146)
        {
            return Err(
                "Breeze tokenizer vocabulary/special tokens do not match the checkpoint".into(),
            );
        }
        let heads = load_f32_tensor(
            source,
            "depth_decoder.codebooks_head.weight",
            &[VOCAB as u64, 1024, 15],
        )?;
        // The published head is [15, hidden, vocab], while native dot kernels need contiguous input rows.
        let mut depth_heads = vec![0u8; heads.len() * 2];
        for h in 0..15 {
            for d in 0..1024 {
                for v in 0..VOCAB {
                    let off = ((h * VOCAB + v) * 1024 + d) * 2;
                    depth_heads[off..off + 2].copy_from_slice(
                        &half::bf16::from_f32(heads[(h * 1024 + d) * VOCAB + v])
                            .to_bits()
                            .to_le_bytes(),
                    );
                }
            }
        }
        Ok(Self {
            tokenizer,
            text_embeddings: matrix(
                source,
                "text_encoder.embed_tokens.weight",
                TEXT_HIDDEN,
                TEXT_VOCAB,
            )?,
            eoi: load_f32_tensor(
                source,
                "text_encoder.embed_tokens.eoi_embedding",
                &[TEXT_HIDDEN as u64],
            )?,
            text: Transformer::load(source, Kind::Text)?,
            projection: matrix(source, "text_encoder_proj.weight", TEXT_HIDDEN, HIDDEN)?,
            backbone: Transformer::load(source, Kind::Backbone)?,
            lm_head: matrix(source, "lm_head.weight", HIDDEN, VOCAB + 1)?,
            audio_embeddings: matrix(
                source,
                "depth_decoder.model.embed_tokens.weight",
                HIDDEN,
                CODEBOOKS * VOCAB,
            )?,
            depth: Transformer::load(source, Kind::Depth)?,
            depth_projection: matrix(
                source,
                "depth_decoder.model.inputs_embeds_projector.weight",
                HIDDEN,
                1024,
            )?,
            depth_heads,
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| e.to_string())?,
        })
    }

    pub fn generate(
        &self,
        prompt: &str,
        instruction: Option<&str>,
        reference: Option<(&str, &[[u32; 16]])>,
        max_frames: usize,
        cfg_scale: f32,
    ) -> Result<Vec<[u32; 16]>, String> {
        self.generate_with_sampling(
            prompt,
            instruction,
            reference,
            max_frames,
            cfg_scale,
            0.0,
            0,
            1.0,
            42,
        )
    }

    pub fn generate_with_sampling(
        &self,
        prompt: &str,
        instruction: Option<&str>,
        reference: Option<(&str, &[[u32; 16]])>,
        max_frames: usize,
        cfg_scale: f32,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        seed: u64,
    ) -> Result<Vec<[u32; 16]>, String> {
        let mut sampler = Sampling::new(temperature, top_k, top_p, seed)?;
        self.pool.install(|| {
            self.generate_inner(
                prompt,
                instruction,
                reference,
                max_frames,
                cfg_scale,
                &mut sampler,
            )
        })
    }

    fn generate_inner(
        &self,
        prompt: &str,
        instruction: Option<&str>,
        reference: Option<(&str, &[[u32; 16]])>,
        max_frames: usize,
        cfg_scale: f32,
        sampler: &mut Sampling,
    ) -> Result<Vec<[u32; 16]>, String> {
        if prompt.trim().is_empty() || max_frames == 0 || !cfg_scale.is_finite() || cfg_scale < 0.0
        {
            return Err(
                "Breeze requires nonempty text, positive max frames and finite nonnegative CFG"
                    .into(),
            );
        }
        let instruction = instruction.filter(|s| !s.trim().is_empty());
        if cfg_scale != 1.0 && instruction.is_none() {
            return Err("Breeze CFG requires --instruction".into());
        }
        if let Some((text, codes)) = reference {
            if text.trim().is_empty()
                || codes.is_empty()
                || codes.iter().flatten().any(|&id| id >= CODE_SIZE as u32)
            {
                return Err(
                    "Breeze reference needs a transcript and nonempty valid codec frames".into(),
                );
            }
        }
        let positive = self.prepare_prompt(prompt, instruction, reference)?;
        if max_frames > 40960usize.saturating_sub(positive.ids.len()) {
            return Err(
                "Breeze prompt plus requested frames exceeds the 40960-token context".into(),
            );
        }
        let negative = if cfg_scale != 1.0 {
            Some(self.prepare_prompt(prompt, None, reference)?)
        } else {
            None
        };
        tokens("breeze.prompt_ids", &positive.ids)?;
        if let Some(negative) = &negative {
            tokens("breeze.cfg_negative_prompt_ids", &negative.ids)?;
        }
        if let Some((_, codes)) = reference {
            tokens(
                "breeze.reference_codes",
                &codes.iter().flatten().copied().collect::<Vec<_>>(),
            )?;
        }
        let mut cond_cache = Cache::default();
        let mut neg_cache = Cache::default();
        let mut frames = Vec::new();
        for step in 0..max_frames {
            let cond_input = if step == 0 {
                self.prompt_embeddings(&positive, reference)?
            } else {
                self.audio_embedding(frames.last().unwrap())
            };
            let (cond_hidden, mut scores) =
                self.backbone_step(cond_input, &mut cond_cache, step)?;
            let negative_hidden = if let Some(negative) = &negative {
                let input = if step == 0 {
                    self.prompt_embeddings(negative, reference)?
                } else {
                    self.audio_embedding(frames.last().unwrap())
                };
                let (hidden, neg_scores) = self.backbone_step(input, &mut neg_cache, step)?;
                guide(&mut scores, &neg_scores, cfg_scale);
                Some(hidden)
            } else {
                None
            };
            let first = sampler.draw(&scores, true)?;
            if first == VOCAB as u32 {
                // The official stream represents EOS as an all-pad frame; it is not audio.
                tokens("breeze.frame", &[2050; 16])?;
                break;
            }
            let mut frame = [0u32; 16];
            frame[0] = first;
            let mut depth_cache = Cache::default();
            for codebook in 1..CODEBOOKS {
                let depth_step = step * 15 + codebook - 1;
                let mut logits = self.depth_step(
                    &cond_hidden,
                    &frame[..codebook],
                    if negative_hidden.is_some() {
                        None
                    } else {
                        Some(&mut depth_cache)
                    },
                    depth_step,
                )?;
                if let Some(hidden) = &negative_hidden {
                    let neg_logits =
                        self.depth_step(hidden, &frame[..codebook], None, depth_step)?;
                    guide(&mut logits, &neg_logits, cfg_scale);
                }
                frame[codebook] = sampler.draw(&logits, false)?;
            }
            tokens("breeze.frame", &frame)?;
            frames.push(frame);
        }
        Ok(frames)
    }

    fn backbone_step(
        &self,
        input: Vec<f32>,
        cache: &mut Cache,
        step: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let n = input.len() / HIDDEN;
        trace(
            "breeze.backbone.input",
            None,
            Some(step),
            &[n, HIDDEN],
            &input,
        )?;
        let hidden = self
            .backbone
            .forward(input, &[n], Some(cache), Some(step))?;
        let hidden = hidden[hidden.len() - HIDDEN..].to_vec();
        let logits = linear(&self.lm_head, &hidden);
        trace(
            "breeze.backbone.logits",
            None,
            Some(step),
            &[1, VOCAB + 1],
            &logits,
        )?;
        Ok((hidden, logits))
    }

    fn depth_step(
        &self,
        backbone_hidden: &[f32],
        codes: &[u32],
        mut cache: Option<&mut Cache>,
        step: usize,
    ) -> Result<Vec<f32>, String> {
        let cached = cache.as_ref().map_or(0, |c| c.len);
        let mut embeddings = Vec::new();
        if cached == 0 {
            embeddings.extend_from_slice(backbone_hidden);
        }
        for i in cached.saturating_sub(1)..codes.len() {
            let mut row = vec![0.0; HIDDEN];
            self.audio_embeddings
                .embedding_lookup(i as u32 * VOCAB as u32 + codes[i], &mut row);
            embeddings.extend(row);
        }
        let input = linear(&self.depth_projection, &embeddings);
        let n = input.len() / 1024;
        trace("breeze.depth.input", None, Some(step), &[n, 1024], &input)?;
        let hidden = self
            .depth
            .forward(input, &[n], cache.as_deref_mut(), Some(step))?;
        // CFG's official eager path recomputes the full depth prefix, including all output heads.
        let first = if cache.is_none() { 1 } else { n - 1 };
        let mut all_logits = Vec::new();
        for row in first..n {
            let head = if cache.is_none() {
                row - 1
            } else {
                codes.len() - 1
            };
            let weights = &self.depth_heads[head * VOCAB * 1024 * 2..(head + 1) * VOCAB * 1024 * 2];
            let h = &hidden[row * 1024..(row + 1) * 1024];
            let mut logits = vec![0.0; VOCAB];
            logits
                .par_iter_mut()
                .zip(weights.par_chunks_exact(1024 * 2))
                .for_each(|(out, w)| {
                    *out = bf16_math::dot_bf16_gemm(w, h);
                });
            all_logits.extend(logits);
        }
        trace(
            "breeze.depth.logits",
            None,
            Some(step),
            &[all_logits.len() / VOCAB, VOCAB],
            &all_logits,
        )?;
        Ok(all_logits[all_logits.len() - VOCAB..].to_vec())
    }

    fn audio_embedding(&self, codes: &[u32; 16]) -> Vec<f32> {
        let mut out = vec![0.0; HIDDEN];
        let mut row = vec![0.0; HIDDEN];
        for (i, &id) in codes.iter().enumerate() {
            self.audio_embeddings
                .embedding_lookup(i as u32 * VOCAB as u32 + id, &mut row);
            for (o, v) in out.iter_mut().zip(&row) {
                *o += v;
            }
        }
        for o in &mut out {
            *o = bf(*o);
        }
        out
    }

    fn prepare_prompt(
        &self,
        text: &str,
        instruction: Option<&str>,
        reference: Option<(&str, &[[u32; 16]])>,
    ) -> Result<Prompt, String> {
        let target = if let Some(ins) = instruction {
            format!("[S0]<ins_bos>{ins}<ins_eos>{text}")
        } else {
            format!("[S0]{text}")
        };
        let mut segments = Vec::new();
        let mut ids = Vec::new();
        if let Some((text, codes)) = reference {
            let segment = encode_segment(&self.tokenizer, &format!("[S0]{text}"))?;
            ids.extend_from_slice(&segment);
            segments.push(segment);
            ids.extend(std::iter::repeat_n(AUDIO, codes.len()));
            ids.push(AUDIO_EOS);
        }
        let segment = encode_segment(&self.tokenizer, &target)?;
        ids.extend_from_slice(&segment);
        segments.push(segment);
        if ids.len() > 40960 {
            return Err("Breeze prompt exceeds the backbone context (40960 tokens)".into());
        }
        Ok(Prompt { ids, segments })
    }

    fn prompt_embeddings(
        &self,
        prompt: &Prompt,
        reference: Option<(&str, &[[u32; 16]])>,
    ) -> Result<Vec<f32>, String> {
        let mut encoded = self.encode_text(&prompt.segments)?;
        let mut out = encoded.remove(0);
        if let Some((_, codes)) = reference {
            for frame in codes {
                out.extend(self.audio_embedding(frame));
            }
            out.extend(self.audio_embedding(&[0; 16]));
            out.extend(encoded.remove(0));
        }
        Ok(out)
    }

    fn encode_text(&self, segments: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, String> {
        let mut sorted = (0..segments.len()).collect::<Vec<_>>();
        sorted.sort_by_key(|&i| segments[i].len());
        let mut buckets: Vec<Vec<usize>> = Vec::new();
        for i in sorted {
            if let Some(bucket) = buckets.last_mut() {
                if segments[i].len() <= 2 * segments[bucket[0]].len() {
                    bucket.push(i);
                    continue;
                }
            }
            buckets.push(vec![i]);
        }
        let mut hidden = vec![Vec::new(); segments.len()];
        for bucket in buckets {
            let n = bucket.iter().map(|&i| segments[i].len()).max().unwrap();
            let lengths = bucket
                .iter()
                .map(|&i| segments[i].len())
                .collect::<Vec<_>>();
            let mut input = vec![0.0; bucket.len() * n * TEXT_HIDDEN];
            for (b, &i) in bucket.iter().enumerate() {
                for t in 0..n {
                    let id = segments[i].get(t).copied().unwrap_or(0);
                    let row = &mut input[(b * n + t) * TEXT_HIDDEN..(b * n + t + 1) * TEXT_HIDDEN];
                    if id == 256000 {
                        row.copy_from_slice(&self.eoi);
                    } else {
                        self.text_embeddings.embedding_lookup(id, row);
                        for v in row {
                            *v = bf(*v * bf((TEXT_HIDDEN as f32).sqrt()));
                        }
                    }
                }
            }
            let shape = if bucket.len() == 1 {
                vec![n, TEXT_HIDDEN]
            } else {
                vec![bucket.len(), n, TEXT_HIDDEN]
            };
            trace("breeze.text.embedding", None, None, &shape, &input)?;
            let output = self.text.forward(input, &lengths, None, None)?;
            for (b, &i) in bucket.iter().enumerate() {
                hidden[i] =
                    output[b * n * TEXT_HIDDEN..(b * n + lengths[b]) * TEXT_HIDDEN].to_vec();
            }
        }
        let mut projected = Vec::new();
        for segment in hidden {
            let output = linear(&self.projection, &segment);
            trace(
                "breeze.text.projected",
                None,
                None,
                &[output.len() / HIDDEN, HIDDEN],
                &output,
            )?;
            projected.push(output);
        }
        Ok(projected)
    }
}

struct Prompt {
    ids: Vec<u32>,
    segments: Vec<Vec<u32>>,
}

fn encode_segment(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    let encoded = tokenizer.encode(text, true).map_err(|e| e.to_string())?;
    let rendered = tokenizer
        .decode(encoded.get_ids(), false)
        .map_err(|e| e.to_string())?;
    let ids = tokenizer
        .encode(rendered, false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    if ids.is_empty()
        || ids
            .iter()
            .any(|&id| id >= TEXT_VOCAB as u32 || id == AUDIO || id == AUDIO_EOS)
    {
        return Err("Invalid Breeze text segment (audio markers are reserved)".into());
    }
    Ok(ids)
}

fn guide(cond: &mut [f32], uncond: &[f32], scale: f32) {
    for (c, &u) in cond.iter_mut().zip(uncond) {
        *c = u + scale * (*c - u);
    }
}

fn greedy(logits: &[f32], backbone: bool) -> Result<u32, String> {
    if logits.len() != VOCAB + usize::from(backbone) || logits.iter().any(|x| !x.is_finite()) {
        return Err("Breeze logits have invalid shape or non-finite values".into());
    }
    let mut best = 0;
    for i in 1..logits.len() {
        if (i < CODE_SIZE || (backbone && i == VOCAB)) && logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(best as u32)
}

struct Sampling {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: rand::rngs::StdRng,
}
impl Sampling {
    fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Result<Self, String> {
        if !temperature.is_finite()
            || temperature < 0.0
            || !top_p.is_finite()
            || top_p <= 0.0
            || top_p > 1.0
        {
            return Err(
                "Breeze sampling requires finite temperature >= 0 and 0 < top_p <= 1".into(),
            );
        }
        Ok(Self {
            temperature,
            top_k,
            top_p,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        })
    }
    fn draw(&mut self, logits: &[f32], backbone: bool) -> Result<u32, String> {
        let best = greedy(logits, backbone)?;
        if self.temperature == 0.0 {
            return Ok(best);
        }
        // Shift before temperature scaling so subnormal positive temperatures cannot overflow every logit.
        let max = logits[best as usize];
        let mut logits = logits
            .iter()
            .map(|&v| ((v as f64 - max as f64) / self.temperature as f64) as f32)
            .collect::<Vec<_>>();
        logits[CODE_SIZE..VOCAB].fill(f32::NEG_INFINITY);
        let k = if self.top_k == 0 {
            logits.len()
        } else {
            self.top_k
        };
        let mut candidates = crate::ops::sample_top_k(&logits, k);
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut sum = 0.0;
        let mut keep = 0;
        for &(_, p) in &candidates {
            sum += p;
            keep += 1;
            if sum >= self.top_p {
                break;
            }
        }
        candidates.truncate(keep);
        let target = self.rng.gen::<f32>() * sum;
        let mut cumulative = 0.0;
        for &(id, p) in &candidates {
            cumulative += p;
            if cumulative > target {
                return Ok(id as u32);
            }
        }
        Ok(candidates.last().map_or(best, |&(id, _)| id as u32))
    }
}

fn meta_string<'a>(source: &'a dyn TensorSource, key: &str) -> Result<&'a str, String> {
    source
        .metadata(key)
        .and_then(MetaValue::to_string_val)
        .ok_or_else(|| format!("Missing string metadata: {key}"))
}

fn validate_config(source: &dyn TensorSource) -> Result<(), String> {
    if meta_string(source, "general.architecture")? != "breeze" {
        return Err("Expected explicit breeze architecture".into());
    }
    let c: Value = serde_json::from_str(meta_string(source, "breeze.config")?)
        .map_err(|e| format!("Invalid breeze.config: {e}"))?;
    let required: Value = serde_json::from_str(r#"{
        "/model_type":"breeze", "/backbone_model_type":"qwen3", "/num_codebooks":16, "/vocab_size":2051, "/audio_embed_size":2048,
        "/audio_token_id":262144,"/audio_eos_token_id":262145,"/codebook_eos_token_id":0,"/text_vocab_size":262158,
        "/tie_codebooks_embeddings":true,"/text_encoder_proj_type":"linear", "/text_encoder_lora_config/enabled":false,
        "/backbone_config/hidden_size":2048,"/backbone_config/num_hidden_layers":28,"/backbone_config/num_attention_heads":16,"/backbone_config/num_key_value_heads":8,"/backbone_config/head_dim":128,"/backbone_config/intermediate_size":6144,
        "/backbone_config/rms_norm_eps":0.000001,"/backbone_config/rope_theta":1000000,"/backbone_config/rope_scaling":null,"/backbone_config/attention_bias":false,"/backbone_config/use_sliding_window":false,"/backbone_config/hidden_act":"silu",
        "/depth_decoder_config/hidden_size":1024,"/depth_decoder_config/num_hidden_layers":12,"/depth_decoder_config/num_attention_heads":8,"/depth_decoder_config/num_key_value_heads":2,"/depth_decoder_config/head_dim":128,"/depth_decoder_config/intermediate_size":8192,
        "/depth_decoder_config/rms_norm_eps":0.00001,"/depth_decoder_config/rope_theta":500000,"/depth_decoder_config/rope_scaling/rope_type":"llama3","/depth_decoder_config/rope_scaling/factor":32.0,
        "/depth_decoder_config/rope_scaling/high_freq_factor":0.0078125,"/depth_decoder_config/rope_scaling/low_freq_factor":0.001953125,"/depth_decoder_config/rope_scaling/original_max_position_embeddings":16,
        "/depth_decoder_config/attention_bias":false,"/depth_decoder_config/mlp_bias":false,"/depth_decoder_config/hidden_act":"silu",
        "/text_encoder_config/hidden_size":1152,"/text_encoder_config/num_hidden_layers":26,"/text_encoder_config/num_attention_heads":4,"/text_encoder_config/num_key_value_heads":1,"/text_encoder_config/head_dim":256,"/text_encoder_config/intermediate_size":6912,
        "/text_encoder_config/rms_norm_eps":0.000001,"/text_encoder_config/hidden_activation":"gelu_pytorch_tanh","/text_encoder_config/attention_bias":false,"/text_encoder_config/sliding_window":512,"/text_encoder_config/query_pre_attn_scalar":256,
        "/text_encoder_config/rope_parameters/full_attention/rope_type":"linear","/text_encoder_config/rope_parameters/full_attention/rope_theta":1000000,"/text_encoder_config/rope_parameters/full_attention/factor":8.0,
        "/text_encoder_config/rope_parameters/sliding_attention/rope_type":"default","/text_encoder_config/rope_parameters/sliding_attention/rope_theta":10000,"/text_encoder_config/attn_logit_softcapping":null,
        "/text_encoder_config/vocab_size":262158,"/text_encoder_config/eoi_token_index":256000
    }"#).expect("Breeze contract JSON");
    for (path, expected) in required.as_object().unwrap() {
        let actual = c.pointer(path);
        let matches = actual == Some(expected)
            || actual
                .and_then(Value::as_f64)
                .zip(expected.as_f64())
                .is_some_and(|(a, b)| a == b);
        if !matches {
            return Err(format!(
                "Unsupported Breeze config {path}: {actual:?}; expected {expected}"
            ));
        }
    }
    let types = c
        .pointer("/text_encoder_config/layer_types")
        .and_then(Value::as_array)
        .ok_or("Missing Breeze text layer types")?;
    if types.len() != 26
        || types.iter().enumerate().any(|(i, v)| {
            v != if (i + 1) % 6 == 0 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
    {
        return Err("Unsupported Breeze text attention schedule".into());
    }
    Ok(())
}

#[inline]
fn bf(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

fn trace(
    name: &str,
    layer: Option<usize>,
    step: Option<usize>,
    shape: &[usize],
    values: &[f32],
) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    if std::env::var_os("RMI_PARITY_TRACE").is_some() {
        crate::parity_trace::checkpoint_at(name, layer, step, shape, values)
            .map_err(|e| e.to_string())?;
    }
    let _ = (name, layer, step, shape, values);
    Ok(())
}
fn tokens(name: &str, ids: &[u32]) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    if std::env::var_os("RMI_PARITY_TRACE").is_some() {
        crate::parity_trace::token_ids(name, ids).map_err(|e| e.to_string())?;
    }
    let _ = (name, ids);
    Ok(())
}
