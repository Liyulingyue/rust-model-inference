//! Dedicated Audio8 conditional Qwen2 decoder; no Qwen3 model state is shared.

use super::{audio8_rope, audio_dot, audio_rms_norm, matrix, vector, Linear};
use crate::core::tensor::{MetaValue, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::Weight;
use crate::ops::silu;
use std::sync::Arc;
use std::sync::Mutex;

const WIDTH: usize = 2048;
const HEAD_WIDTH: usize = 128;
const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const FFN_WIDTH: usize = 11008;
const LAYERS: usize = 36;
const VOCAB: usize = 151936;

/// Projections are held behind `&'static` so the executor, which is built
/// after the layers, can borrow them for the model's lifetime.
struct TextLayer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q: &'static Weight<'static>,
    k: &'static Weight<'static>,
    v: &'static Weight<'static>,
    o: &'static Weight<'static>,
    gate: &'static Weight<'static>,
    up: &'static Weight<'static>,
    down: &'static Weight<'static>,
    ada1: &'static Weight<'static>,
    ada2: &'static Weight<'static>,
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    v_bias: Vec<f32>,
}

pub struct Audio8TextDecoder {
    _source: Arc<dyn TensorSource>,
    embedding: &'static Weight<'static>,
    norm: Vec<f32>,
    layers: Vec<TextLayer>,
    context: usize,
    /// Every projection above, bound to the CPU/GPU executor. The LM head is
    /// the tied embedding, so it is bound as well.
    linear: Mutex<Linear>,
}

pub struct Audio8TextSession<'a> {
    model: &'a Audio8TextDecoder,
    scales: Vec<Vec<f32>>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    capacity: usize,
    position: usize,
}

impl Audio8TextDecoder {
    /// Borrow the executor. A poisoned lock only means an earlier projection
    /// panicked, so the model is unusable either way.
    fn linear(&self) -> Result<std::sync::MutexGuard<'_, Linear>, String> {
        self.linear
            .lock()
            .map_err(|_| "Audio8 linear executor poisoned".to_string())
    }

    pub fn from_source(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        if source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            != Some("audio8_asr_infinite")
        {
            return Err("expected audio8_asr_infinite GGUF".into());
        }
        let raw = source
            .metadata("audio8_asr_infinite.config_json")
            .and_then(MetaValue::to_string_val)
            .ok_or("missing Audio8 config JSON")?;
        let config: serde_json::Value =
            serde_json::from_str(raw).map_err(|error| format!("invalid Audio8 config: {error}"))?;
        let text = &config["text_config"];
        for (field, expected) in [
            ("hidden_size", WIDTH),
            ("intermediate_size", FFN_WIDTH),
            ("num_hidden_layers", LAYERS),
            ("num_attention_heads", QUERY_HEADS),
            ("num_key_value_heads", KV_HEADS),
            ("vocab_size", VOCAB),
        ] {
            if text[field].as_u64() != Some(expected as u64) {
                return Err(format!("unsupported Audio8 text config {field}"));
            }
        }
        if text["model_type"].as_str() != Some("qwen2")
            || text["tie_word_embeddings"].as_bool() != Some(true)
            || text["rope_parameters"]["rope_theta"].as_f64() != Some(1_000_000.0)
            || text["rms_norm_eps"].as_f64() != Some(1e-6)
        {
            return Err("unsupported Audio8 text decoder contract".into());
        }
        let context = text["max_position_embeddings"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or("invalid Audio8 text context")?;
        let embedding = super::leak(matrix(source.as_ref(), "token_embd.weight", WIDTH, VOCAB)?);
        let norm = vector(source.as_ref(), "output_norm.weight", WIDTH)?;
        let mut layers = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            let name = |suffix: &str| format!("blk.{index}.{suffix}");
            layers.push(TextLayer {
                attn_norm: vector(source.as_ref(), &name("attn_norm.weight"), WIDTH)?,
                ffn_norm: vector(source.as_ref(), &name("ffn_norm.weight"), WIDTH)?,
                q: super::leak(matrix(
                    source.as_ref(),
                    &name("attn_q.weight"),
                    WIDTH,
                    WIDTH,
                )?),
                k: super::leak(matrix(
                    source.as_ref(),
                    &name("attn_k.weight"),
                    WIDTH,
                    KV_HEADS * HEAD_WIDTH,
                )?),
                v: super::leak(matrix(
                    source.as_ref(),
                    &name("attn_v.weight"),
                    WIDTH,
                    KV_HEADS * HEAD_WIDTH,
                )?),
                o: super::leak(matrix(
                    source.as_ref(),
                    &name("attn_output.weight"),
                    WIDTH,
                    WIDTH,
                )?),
                gate: super::leak(matrix(
                    source.as_ref(),
                    &name("ffn_gate.weight"),
                    WIDTH,
                    FFN_WIDTH,
                )?),
                up: super::leak(matrix(
                    source.as_ref(),
                    &name("ffn_up.weight"),
                    WIDTH,
                    FFN_WIDTH,
                )?),
                down: super::leak(matrix(
                    source.as_ref(),
                    &name("ffn_down.weight"),
                    FFN_WIDTH,
                    WIDTH,
                )?),
                ada1: super::leak(matrix(
                    source.as_ref(),
                    &name("ada_linear1.weight"),
                    WIDTH,
                    32,
                )?),
                ada2: super::leak(matrix(
                    source.as_ref(),
                    &name("ada_linear2.weight"),
                    32,
                    WIDTH,
                )?),
                q_bias: vector(source.as_ref(), &name("attn_q.bias"), WIDTH)?,
                k_bias: vector(source.as_ref(), &name("attn_k.bias"), KV_HEADS * HEAD_WIDTH)?,
                v_bias: vector(source.as_ref(), &name("attn_v.bias"), KV_HEADS * HEAD_WIDTH)?,
            });
        }
        let mut bindings: Vec<(&'static str, &'static Weight<'static>)> =
            Vec::with_capacity(1 + LAYERS * 9);
        bindings.push(("token_embd.weight", embedding));
        for (index, layer) in layers.iter().enumerate() {
            let names: &'static [&'static str] = Box::leak(Box::new(
                ["q", "k", "v", "o", "gate", "up", "down", "ada1", "ada2"].map(|suffix| {
                    Box::leak(format!("blk.{index}.{suffix}").into_boxed_str()) as &'static str
                }),
            ));
            for (name, weight) in names.iter().zip([
                layer.q, layer.k, layer.v, layer.o, layer.gate, layer.up, layer.down, layer.ada1,
                layer.ada2,
            ]) {
                bindings.push((name, weight));
            }
        }
        let linear = Linear::new(&bindings, pool)?;
        Ok(Self {
            _source: source,
            embedding,
            norm,
            layers,
            context,
            linear: Mutex::new(linear),
        })
    }

    pub fn context(&self) -> usize {
        self.context
    }

    pub fn embed_audio_tokens(&self, token_ids: &[u32], audio: &[f32]) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() || audio.len() != token_ids.len() * WIDTH {
            return Err("invalid Audio8 token/audio input length".into());
        }
        let mut embeddings = vec![0.0; audio.len()];
        for ((&token, audio), output) in token_ids
            .iter()
            .zip(audio.chunks_exact(WIDTH))
            .zip(embeddings.chunks_exact_mut(WIDTH))
        {
            if token as usize >= VOCAB {
                return Err("Audio8 token exceeds vocabulary".into());
            }
            self.embedding.kernel.embedding_lookup(token, WIDTH, output);
            for (value, &audio) in output.iter_mut().zip(audio) {
                *value += audio;
            }
        }
        Ok(embeddings)
    }
}

impl<'a> Audio8TextSession<'a> {
    pub fn new(
        model: &'a Audio8TextDecoder,
        capacity: usize,
        condition: &[f32],
    ) -> Result<Self, String> {
        if capacity == 0 || capacity > model.context || condition.len() != WIDTH {
            return Err("invalid Audio8 text session capacity or condition".into());
        }
        let mut scales = Vec::with_capacity(model.layers.len());
        for index in 0..model.layers.len() {
            let mut hidden = model
                .linear()?
                .run(&format!("blk.{index}.ada1"), condition, 32)?;
            for value in &mut hidden {
                *value = crate::ops::gelu_erf(*value);
            }
            let mut scale = model
                .linear()?
                .run(&format!("blk.{index}.ada2"), &hidden, WIDTH)?;
            for value in &mut scale {
                *value += 1.0;
            }
            scales.push(scale);
        }
        Ok(Self {
            model,
            scales,
            keys: vec![Vec::new(); LAYERS],
            values: vec![Vec::new(); LAYERS],
            capacity,
            position: 0,
        })
    }

    pub fn forward_logits(&mut self, embeddings: &[f32]) -> Result<Vec<f32>, String> {
        if embeddings.is_empty()
            || embeddings.len() % WIDTH != 0
            || self
                .position
                .checked_add(embeddings.len() / WIDTH)
                .is_none_or(|end| end > self.capacity)
        {
            return Err("invalid Audio8 text/audio input length".into());
        }
        let end_position = self.position + embeddings.len() / WIDTH;
        #[cfg(feature = "parity-trace")]
        let trace_all = crate::parity_trace::enabled("result_norm")
            || crate::parity_trace::enabled("result_output");
        #[cfg(not(feature = "parity-trace"))]
        let trace_all = false;
        let mut logits = Vec::new();
        for embedding in embeddings.chunks_exact(WIDTH) {
            let position = self.position;
            let mut hidden = embedding.to_vec();
            trace("model.input_embed", None, &[1, WIDTH], &hidden);
            for (index, layer) in self.model.layers.iter().enumerate() {
                let mut normed = vec![0.0; WIDTH];
                audio_rms_norm(&hidden, &layer.attn_norm, &mut normed, 1e-6);
                if index == 0 {
                    trace("attn_norm-0", Some(0), &[1, WIDTH], &normed);
                }
                let mut linear = self.model.linear()?;
                let mut q = linear.run(&format!("blk.{index}.q"), &normed, WIDTH)?;
                for (value, bias) in q.iter_mut().zip(&layer.q_bias) {
                    *value += *bias;
                }
                let mut k =
                    linear.run(&format!("blk.{index}.k"), &normed, KV_HEADS * HEAD_WIDTH)?;
                for (value, bias) in k.iter_mut().zip(&layer.k_bias) {
                    *value += *bias;
                }
                let mut v =
                    linear.run(&format!("blk.{index}.v"), &normed, KV_HEADS * HEAD_WIDTH)?;
                for (value, bias) in v.iter_mut().zip(&layer.v_bias) {
                    *value += *bias;
                }
                audio8_rope(&mut q, HEAD_WIDTH, position);
                audio8_rope(&mut k, HEAD_WIDTH, position);
                if index == 0 {
                    trace("Qcur-0", Some(0), &[QUERY_HEADS, HEAD_WIDTH], &q);
                    trace("Kcur-0", Some(0), &[KV_HEADS, HEAD_WIDTH], &k);
                }
                self.keys[index].extend_from_slice(&k);
                self.values[index].extend_from_slice(&v);
                let mut attention = vec![0.0f32; WIDTH];
                let scale = 1.0 / (HEAD_WIDTH as f32).sqrt();
                for head in 0..QUERY_HEADS {
                    let kv_head = head / (QUERY_HEADS / KV_HEADS);
                    let head_start = head * HEAD_WIDTH;
                    let query = &q[head_start..head_start + HEAD_WIDTH];
                    let mut scores = Vec::with_capacity(position + 1);
                    let mut maximum = f32::NEG_INFINITY;
                    for past in 0..=position {
                        let offset = past * KV_HEADS * HEAD_WIDTH + kv_head * HEAD_WIDTH;
                        let score =
                            audio_dot(query, &self.keys[index][offset..offset + HEAD_WIDTH])
                                * scale;
                        maximum = maximum.max(score);
                        scores.push(score);
                    }
                    let mut sum = 0.0f32;
                    for score in &mut scores {
                        *score = (*score - maximum).exp();
                        sum += *score;
                    }
                    for (past, score) in scores.iter().enumerate() {
                        let offset = past * KV_HEADS * HEAD_WIDTH + kv_head * HEAD_WIDTH;
                        for dimension in 0..HEAD_WIDTH {
                            attention[head_start + dimension] +=
                                (*score / sum) * self.values[index][offset + dimension];
                        }
                    }
                }
                if index == 0 {
                    trace("kqv_out-0", Some(0), &[QUERY_HEADS, HEAD_WIDTH], &attention);
                }
                let update = linear.run(&format!("blk.{index}.o"), &attention, WIDTH)?;
                for (value, update) in hidden.iter_mut().zip(update) {
                    *value += update;
                }
                audio_rms_norm(&hidden, &layer.ffn_norm, &mut normed, 1e-6);
                for (value, scale) in normed.iter_mut().zip(&self.scales[index]) {
                    *value *= *scale;
                }
                let mut gate = linear.run(&format!("blk.{index}.gate"), &normed, FFN_WIDTH)?;
                let up = linear.run(&format!("blk.{index}.up"), &normed, FFN_WIDTH)?;
                for (gate, up) in gate.iter_mut().zip(up) {
                    *gate = silu(*gate) * up;
                }
                let down = linear.run(&format!("blk.{index}.down"), &gate, WIDTH)?;
                if index == 0 {
                    trace("ffn_out-0", Some(0), &[1, WIDTH], &down);
                }
                for (value, down) in hidden.iter_mut().zip(down) {
                    *value += down;
                }
                trace(
                    "audio8.text_layer_output",
                    Some(index),
                    &[1, WIDTH],
                    &hidden,
                );
            }
            if position + 1 == end_position || trace_all {
                let mut normed = vec![0.0; WIDTH];
                audio_rms_norm(&hidden, &self.model.norm, &mut normed, 1e-6);
                trace("result_norm", None, &[1, WIDTH], &normed);
                logits = self
                    .model
                    .linear()?
                    .run("token_embd.weight", &normed, VOCAB)?;
                trace("result_output", None, &[VOCAB], &logits);
            }
            self.position += 1;
        }
        Ok(logits)
    }
}

#[inline]
fn trace(name: &str, layer: Option<usize>, shape: &[usize], values: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(name, layer, shape, values));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, shape, values);
}
