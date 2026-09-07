use std::sync::Arc;

use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use super::kernels::{
    attention_online_with_bias, gelu_inplace, load_float_values, rms_norm_rows, AttentionSpec,
    Linear,
};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;

const TEXT_CONTEXT: usize = 512;
const TEXT_DIM: usize = 4096;
const TEXT_FFN: usize = 10240;
const TEXT_HEADS: usize = 64;
const TEXT_LAYERS: usize = 24;
const TEXT_VOCAB: usize = 256384;
const TEXT_BUCKETS: usize = 32;
const TEXT_EPSILON: f32 = 1e-6;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizedText {
    pub ids: Vec<u32>,
    pub real_len: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextConditioning {
    pub positive: Vec<f32>,
    pub negative: Vec<f32>,
    pub positive_len: usize,
    pub negative_len: usize,
}

#[derive(Clone, Copy)]
struct TextDimensions {
    context: usize,
    dim: usize,
    ffn: usize,
    heads: usize,
    layers: usize,
    vocab: usize,
    buckets: usize,
}

impl TextDimensions {
    const RELEASED: Self = Self {
        context: TEXT_CONTEXT,
        dim: TEXT_DIM,
        ffn: TEXT_FFN,
        heads: TEXT_HEADS,
        layers: TEXT_LAYERS,
        vocab: TEXT_VOCAB,
        buckets: TEXT_BUCKETS,
    };

    fn validate(self) -> Result<(), String> {
        if self.context == 0
            || self.dim == 0
            || self.ffn == 0
            || self.heads == 0
            || self.layers == 0
            || self.vocab == 0
            || self.buckets < 4
            || self.buckets % 2 != 0
            || self.dim % self.heads != 0
        {
            return Err("Invalid DreamX text dimensions".into());
        }
        Ok(())
    }
}

struct TextBlock {
    norm1: Vec<f32>,
    query: Linear<'static>,
    key: Linear<'static>,
    value: Linear<'static>,
    output: Linear<'static>,
    norm2: Vec<f32>,
    gate: Linear<'static>,
    up: Linear<'static>,
    down: Linear<'static>,
    position: Vec<f32>,
}

impl TextBlock {
    fn load(
        source: &'static dyn TensorSource,
        layer: usize,
        dimensions: TextDimensions,
    ) -> Result<Self, String> {
        let name = |suffix: &str| format!("dreamx.text.blocks.{layer}.{suffix}");
        Ok(Self {
            norm1: load_float_values(source, &name("norm1.weight"), &[dimensions.dim as u64])?,
            query: Linear::from_source(
                source,
                &name("attn.q.weight"),
                None,
                dimensions.dim,
                dimensions.dim,
            )?,
            key: Linear::from_source(
                source,
                &name("attn.k.weight"),
                None,
                dimensions.dim,
                dimensions.dim,
            )?,
            value: Linear::from_source(
                source,
                &name("attn.v.weight"),
                None,
                dimensions.dim,
                dimensions.dim,
            )?,
            output: Linear::from_source(
                source,
                &name("attn.o.weight"),
                None,
                dimensions.dim,
                dimensions.dim,
            )?,
            norm2: load_float_values(source, &name("norm2.weight"), &[dimensions.dim as u64])?,
            gate: Linear::from_source(
                source,
                &name("ffn.gate.0.weight"),
                None,
                dimensions.dim,
                dimensions.ffn,
            )?,
            up: Linear::from_source(
                source,
                &name("ffn.fc1.weight"),
                None,
                dimensions.dim,
                dimensions.ffn,
            )?,
            down: Linear::from_source(
                source,
                &name("ffn.fc2.weight"),
                None,
                dimensions.ffn,
                dimensions.dim,
            )?,
            position: load_float_values(
                source,
                &name("pos_embedding.embedding.weight"),
                &[dimensions.heads as u64, dimensions.buckets as u64],
            )?,
        })
    }

    fn forward(
        &self,
        pool: &ComputePool,
        hidden: &mut [f32],
        tokens: usize,
        dimensions: TextDimensions,
    ) -> Result<(), String> {
        let normed = rms_norm_rows(hidden, tokens, &self.norm1, TEXT_EPSILON)?;
        let query = self.query.forward(pool, &normed, tokens)?;
        let key = self.key.forward(pool, &normed, tokens)?;
        let value = self.value.forward(pool, &normed, tokens)?;
        let bias = relative_attention_bias(&self.position, tokens, dimensions)?;
        let attention = attention_online_with_bias(
            &query,
            &key,
            &value,
            &bias,
            AttentionSpec {
                query_tokens: tokens,
                key_tokens: tokens,
                query_heads: dimensions.heads,
                key_value_heads: dimensions.heads,
                head_dim: dimensions.dim / dimensions.heads,
                causal: false,
                scale: 1.0,
            },
        )?;
        let projected = self.output.forward(pool, &attention, tokens)?;
        for (hidden, residual) in hidden.iter_mut().zip(projected) {
            *hidden += residual;
        }

        let normed = rms_norm_rows(hidden, tokens, &self.norm2, TEXT_EPSILON)?;
        let mut gate = self.gate.forward(pool, &normed, tokens)?;
        gelu_inplace(&mut gate);
        let up = self.up.forward(pool, &normed, tokens)?;
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate *= up;
        }
        let projected = self.down.forward(pool, &gate, tokens)?;
        for (hidden, residual) in hidden.iter_mut().zip(projected) {
            *hidden += residual;
        }
        Ok(())
    }
}

pub struct DreamXTextEncoder {
    _source: Option<Arc<dyn TensorSource>>,
    tokenizer: Tokenizer,
    dimensions: TextDimensions,
    embedding: Linear<'static>,
    blocks: Vec<TextBlock>,
    norm: Vec<f32>,
    pool: Arc<ComputePool>,
}

impl DreamXTextEncoder {
    pub fn load(source: Arc<dyn TensorSource>, pool: Arc<ComputePool>) -> Result<Self, String> {
        let tokenizer_json = source
            .metadata("dreamx.tokenizer.json")
            .and_then(|value| value.to_string_val())
            .ok_or("Invalid DreamX metadata dreamx.tokenizer.json: expected string")?
            .to_owned();
        // The Arc is retained by the returned model, so these mmap-backed slices
        // outlive every Weight created below.
        let source_ref: &'static dyn TensorSource = unsafe { &*Arc::as_ptr(&source) };
        Self::load_with_dimensions(
            Some(source),
            source_ref,
            &tokenizer_json,
            pool,
            TextDimensions::RELEASED,
        )
    }

    fn load_with_dimensions(
        source_owner: Option<Arc<dyn TensorSource>>,
        source: &'static dyn TensorSource,
        tokenizer_json: &str,
        pool: Arc<ComputePool>,
        dimensions: TextDimensions,
    ) -> Result<Self, String> {
        dimensions.validate()?;
        let tokenizer = configured_tokenizer(tokenizer_json, dimensions.context)?;
        let embedding = Linear::from_source(
            source,
            "dreamx.text.token_embedding.weight",
            None,
            dimensions.dim,
            dimensions.vocab,
        )?;
        let mut blocks = Vec::with_capacity(dimensions.layers);
        for layer in 0..dimensions.layers {
            blocks.push(TextBlock::load(source, layer, dimensions)?);
        }
        let norm = load_float_values(source, "dreamx.text.norm.weight", &[dimensions.dim as u64])?;
        Ok(Self {
            _source: source_owner,
            tokenizer,
            dimensions,
            embedding,
            blocks,
            norm,
            pool,
        })
    }

    pub fn tokenize(&self, text: &str) -> Result<TokenizedText, String> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| format!("DreamX tokenization failed: {error}"))?;
        if encoding.get_ids().len() != self.dimensions.context
            || encoding.get_attention_mask().len() != self.dimensions.context
        {
            return Err("DreamX tokenizer did not produce the configured context length".into());
        }
        let real_len = encoding
            .get_attention_mask()
            .iter()
            .map(|&value| value as usize)
            .sum();
        Ok(TokenizedText {
            ids: encoding.get_ids().to_vec(),
            real_len,
        })
    }

    pub fn encode(&self, prompt: &str, negative_prompt: &str) -> Result<TextConditioning, String> {
        let positive = self.tokenize(prompt)?;
        let negative = self.tokenize(negative_prompt)?;
        Ok(TextConditioning {
            positive: self.encode_tokens(&positive)?,
            negative: self.encode_tokens(&negative)?,
            positive_len: positive.real_len,
            negative_len: negative.real_len,
        })
    }

    fn encode_tokens(&self, tokens: &TokenizedText) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0; self.dimensions.context * self.dimensions.dim];
        if tokens.real_len == 0 {
            return Ok(output);
        }
        let mut hidden = vec![0.0; tokens.real_len * self.dimensions.dim];
        for (row, &token_id) in tokens.ids[..tokens.real_len].iter().enumerate() {
            self.embedding.embedding_lookup(
                token_id,
                &mut hidden[row * self.dimensions.dim..(row + 1) * self.dimensions.dim],
            )?;
        }
        for block in &self.blocks {
            block.forward(&self.pool, &mut hidden, tokens.real_len, self.dimensions)?;
        }
        let hidden = rms_norm_rows(&hidden, tokens.real_len, &self.norm, TEXT_EPSILON)?;
        output[..hidden.len()].copy_from_slice(&hidden);
        Ok(output)
    }

    #[cfg(test)]
    fn testing(tokenizer_json: &str) -> Result<Self, String> {
        let dimensions = TextDimensions {
            context: TEXT_CONTEXT,
            dim: 4,
            ffn: 8,
            heads: 2,
            layers: 1,
            vocab: 7,
            buckets: 4,
        };
        let linear = |n_in: usize, n_out: usize, seed: usize| {
            let values: Vec<f32> = (0..n_in * n_out)
                .map(|index| (((index + seed) * 13 % 17) as f32 - 8.0) / 32.0)
                .collect();
            let mut weight = crate::ops::kernel::Weight::from_quantized(
                crate::ops::kernel::QuantizedTensor::F32(values),
            );
            weight.n_in = n_in;
            weight.n_out = n_out;
            Linear::from_weight(weight, None)
        };
        let block = TextBlock {
            norm1: vec![1.0; dimensions.dim],
            query: linear(dimensions.dim, dimensions.dim, 1)?,
            key: linear(dimensions.dim, dimensions.dim, 2)?,
            value: linear(dimensions.dim, dimensions.dim, 3)?,
            output: linear(dimensions.dim, dimensions.dim, 4)?,
            norm2: vec![1.0; dimensions.dim],
            gate: linear(dimensions.dim, dimensions.ffn, 5)?,
            up: linear(dimensions.dim, dimensions.ffn, 6)?,
            down: linear(dimensions.ffn, dimensions.dim, 7)?,
            position: vec![0.0; dimensions.buckets * dimensions.heads],
        };
        Ok(Self {
            _source: None,
            tokenizer: configured_tokenizer(tokenizer_json, dimensions.context)?,
            dimensions,
            embedding: linear(dimensions.dim, dimensions.vocab, 0)?,
            blocks: vec![block],
            norm: vec![1.0; dimensions.dim],
            pool: Arc::new(ComputePool::new(2)),
        })
    }
}

fn configured_tokenizer(json: &str, context: usize) -> Result<Tokenizer, String> {
    let mut tokenizer = Tokenizer::from_bytes(json.as_bytes())
        .map_err(|error| format!("Invalid DreamX tokenizer JSON: {error}"))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: context,
            ..TruncationParams::default()
        }))
        .map_err(|error| format!("Invalid DreamX tokenizer truncation: {error}"))?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(context),
        pad_id: 0,
        pad_token: "<pad>".into(),
        ..PaddingParams::default()
    }));
    Ok(tokenizer)
}

fn relative_attention_bias(
    embedding: &[f32],
    tokens: usize,
    dimensions: TextDimensions,
) -> Result<Vec<f32>, String> {
    if embedding.len() != dimensions.buckets * dimensions.heads || tokens == 0 {
        return Err("Invalid DreamX relative-position embedding".into());
    }
    let mut bias = vec![0.0; dimensions.heads * tokens * tokens];
    for head in 0..dimensions.heads {
        for query in 0..tokens {
            for key in 0..tokens {
                let bucket = relative_position_bucket(
                    key as isize - query as isize,
                    dimensions.buckets,
                    128,
                );
                bias[(head * tokens + query) * tokens + key] =
                    embedding[bucket * dimensions.heads + head];
            }
        }
    }
    Ok(bias)
}

fn relative_position_bucket(position: isize, buckets: usize, max_distance: usize) -> usize {
    let half = buckets / 2;
    let direction = usize::from(position > 0) * half;
    let distance = position.unsigned_abs();
    let max_exact = half / 2;
    if distance < max_exact {
        return direction + distance;
    }
    let logarithmic = max_exact
        + ((distance as f32 / max_exact as f32).ln()
            / (max_distance as f32 / max_exact as f32).ln()
            * (half - max_exact) as f32) as usize;
    direction + logarithmic.min(half - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY_TEXT_DIM: usize = 4;
    const TOKENIZER_JSON: &str = r#"{
        "version":"1.0",
        "truncation":null,
        "padding":null,
        "added_tokens":[],
        "normalizer":null,
        "pre_tokenizer":{"type":"Whitespace"},
        "post_processor":null,
        "decoder":null,
        "model":{"type":"WordLevel","vocab":{"<pad>":0,"<unk>":1,"a":2,"person":3,"speaking":4,"rain":5,"static":6},"unk_token":"<unk>"}
    }"#;

    fn tiny_text_encoder() -> DreamXTextEncoder {
        DreamXTextEncoder::testing(TOKENIZER_JSON).unwrap()
    }

    #[test]
    fn tokenizer_encodes_and_pads_to_512() {
        let encoder = tiny_text_encoder();
        let encoded = encoder.tokenize("a person speaking").unwrap();
        assert_eq!(encoded.ids.len(), 512);
        assert_eq!(encoded.real_len, 3);
        assert!(encoded.ids[encoded.real_len..].iter().all(|&id| id == 0));
    }

    #[test]
    fn text_encoder_returns_positive_and_negative_context() {
        let out = tiny_text_encoder().encode("rain", "static").unwrap();
        assert_eq!(out.positive.len(), 512 * TINY_TEXT_DIM);
        assert_eq!(out.negative.len(), 512 * TINY_TEXT_DIM);
        assert_eq!((out.positive_len, out.negative_len), (1, 1));
        assert!(out.positive[TINY_TEXT_DIM..]
            .iter()
            .all(|&value| value == 0.0));
        assert!(out.negative[TINY_TEXT_DIM..]
            .iter()
            .all(|&value| value == 0.0));
    }
}
