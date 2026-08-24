use std::sync::Arc;

use crate::core::scratchpad::KvCacheF16;
use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::ops::{
    dot_f16_f32, embedding_lookup, f16_to_f32, f32_slice_to_f16, rms_norm, rms_norm_inplace,
    rope_neox, silu_mul_inplace, softmax_inplace,
};

use super::{linear_into, validate_component, Component, Q8Scratch};

const HIDDEN: usize = 2_560;
const QUERY_HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_WIDTH: usize = 128;
const QUERY_WIDTH: usize = QUERY_HEADS * HEAD_WIDTH;
const KV_WIDTH: usize = KV_HEADS * HEAD_WIDTH;
const FFN_WIDTH: usize = 9_728;
const LAYERS: usize = 36;
const LAYER_35_BLOCKS: usize = 35;
const RMS_EPSILON: f32 = 1e-6;
const ROPE_BASE: f32 = 1_000_000.0;

struct TextLayer {
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    q_proj: String,
    k_proj: String,
    v_proj: String,
    o_proj: String,
    gate_proj: String,
    up_proj: String,
    down_proj: String,
}

pub(crate) struct Qwen3TextEncoder {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    tokenizer: BPETokenizer,
    layers: Vec<TextLayer>,
}

impl Qwen3TextEncoder {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        validate_component(source.as_ref(), Component::Text)?;
        let tokenizer = BPETokenizer::from_qwen3_embedded_merges()?;
        let mut layers = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let prefix = format!("model.layers.{layer}");
            layers.push(TextLayer {
                input_norm: load_f32(
                    source.as_ref(),
                    &format!("{prefix}.input_layernorm.weight"),
                    HIDDEN,
                )?,
                post_attention_norm: load_f32(
                    source.as_ref(),
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    HIDDEN,
                )?,
                q_norm: load_f32(
                    source.as_ref(),
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    HEAD_WIDTH,
                )?,
                k_norm: load_f32(
                    source.as_ref(),
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    HEAD_WIDTH,
                )?,
                q_proj: format!("{prefix}.self_attn.q_proj.weight"),
                k_proj: format!("{prefix}.self_attn.k_proj.weight"),
                v_proj: format!("{prefix}.self_attn.v_proj.weight"),
                o_proj: format!("{prefix}.self_attn.o_proj.weight"),
                gate_proj: format!("{prefix}.mlp.gate_proj.weight"),
                up_proj: format!("{prefix}.mlp.up_proj.weight"),
                down_proj: format!("{prefix}.mlp.down_proj.weight"),
            });
        }
        Ok(Self {
            source,
            pool,
            tokenizer,
            layers,
        })
    }

    pub(crate) fn encode_layer_35(&self, prompt: &str) -> Result<Vec<f32>, String> {
        let ids = self.tokenizer.encode(
            &z_image_prompt(prompt),
            EncodeOptions {
                add_special: false,
                parse_special: true,
            },
        );
        if ids.is_empty() {
            return Err("Z-Image prompt produced no tokens".into());
        }
        self.forward_to_block(&ids, LAYER_35_BLOCKS)
    }

    fn forward_to_block(&self, ids: &[u32], blocks: usize) -> Result<Vec<f32>, String> {
        if ids.is_empty() || blocks == 0 || blocks > self.layers.len() {
            return Err("Invalid Qwen3 layer-35 request".into());
        }
        if ids
            .iter()
            .any(|id| *id as usize >= self.tokenizer.vocab_size())
        {
            return Err("Qwen3 token ID is outside the vocabulary".into());
        }
        let token_count = ids.len();
        let output_len = token_count
            .checked_mul(HIDDEN)
            .ok_or("Qwen3 layer-35 output size overflow")?;
        let cache_len = blocks
            .checked_mul(token_count)
            .and_then(|value| value.checked_mul(KV_WIDTH))
            .ok_or("Qwen3 KV cache size overflow")?;
        let mut output = reserve_f32("Qwen3 layer-35 output", output_len)?;
        let mut cache = KvCacheF16 {
            k: reserve_u16("Qwen3 key cache", cache_len)?,
            v: reserve_u16("Qwen3 value cache", cache_len)?,
        };
        let embedding = self
            .source
            .tensor_slice("model.embed_tokens.weight")
            .ok_or("Missing tensor data: model.embed_tokens.weight")?;
        let mut scratch = TextScratch::new(token_count);

        for (position, &id) in ids.iter().enumerate() {
            embedding_lookup(embedding, id, HIDDEN, GGMLType::Q8_0, &mut scratch.hidden);
            for (layer_index, layer) in self.layers[..blocks].iter().enumerate() {
                rms_norm(
                    &scratch.hidden,
                    &layer.input_norm,
                    &mut scratch.normed,
                    RMS_EPSILON,
                );
                linear_into(
                    self.source.as_ref(),
                    &layer.q_proj,
                    HIDDEN,
                    QUERY_WIDTH,
                    &scratch.normed,
                    &mut scratch.q,
                    &mut scratch.q8,
                    self.pool.as_ref(),
                )?;
                linear_into(
                    self.source.as_ref(),
                    &layer.k_proj,
                    HIDDEN,
                    KV_WIDTH,
                    &scratch.normed,
                    &mut scratch.k,
                    &mut scratch.q8,
                    self.pool.as_ref(),
                )?;
                linear_into(
                    self.source.as_ref(),
                    &layer.v_proj,
                    HIDDEN,
                    KV_WIDTH,
                    &scratch.normed,
                    &mut scratch.v,
                    &mut scratch.q8,
                    self.pool.as_ref(),
                )?;

                for head in scratch.q.chunks_exact_mut(HEAD_WIDTH) {
                    rms_norm_inplace(head, &layer.q_norm, RMS_EPSILON);
                }
                for head in scratch.k.chunks_exact_mut(HEAD_WIDTH) {
                    rms_norm_inplace(head, &layer.k_norm, RMS_EPSILON);
                }
                rope_neox(&mut scratch.q, position, HEAD_WIDTH, ROPE_BASE);
                rope_neox(&mut scratch.k, position, HEAD_WIDTH, ROPE_BASE);

                let cache_row = (layer_index * token_count + position) * KV_WIDTH;
                f32_slice_to_f16(&scratch.k, &mut cache.k[cache_row..cache_row + KV_WIDTH]);
                f32_slice_to_f16(&scratch.v, &mut cache.v[cache_row..cache_row + KV_WIDTH]);
                attention(
                    &scratch.q,
                    &cache,
                    layer_index,
                    position,
                    token_count,
                    &mut scratch.scores,
                    &mut scratch.attention,
                );
                linear_into(
                    self.source.as_ref(),
                    &layer.o_proj,
                    QUERY_WIDTH,
                    HIDDEN,
                    &scratch.attention,
                    &mut scratch.projected,
                    &mut scratch.q8,
                    self.pool.as_ref(),
                )?;
                for (hidden, projected) in scratch.hidden.iter_mut().zip(&scratch.projected) {
                    *hidden += projected;
                }

                rms_norm(
                    &scratch.hidden,
                    &layer.post_attention_norm,
                    &mut scratch.normed,
                    RMS_EPSILON,
                );
                linear_into(
                    self.source.as_ref(),
                    &layer.gate_proj,
                    HIDDEN,
                    FFN_WIDTH,
                    &scratch.normed,
                    &mut scratch.gate,
                    &mut scratch.q8,
                    self.pool.as_ref(),
                )?;
                linear_into(
                    self.source.as_ref(),
                    &layer.up_proj,
                    HIDDEN,
                    FFN_WIDTH,
                    &scratch.normed,
                    &mut scratch.up,
                    &mut scratch.q8,
                    self.pool.as_ref(),
                )?;
                silu_mul_inplace(&scratch.gate, &mut scratch.up);
                linear_into(
                    self.source.as_ref(),
                    &layer.down_proj,
                    FFN_WIDTH,
                    HIDDEN,
                    &scratch.up,
                    &mut scratch.projected,
                    &mut scratch.q8,
                    self.pool.as_ref(),
                )?;
                for (hidden, projected) in scratch.hidden.iter_mut().zip(&scratch.projected) {
                    *hidden += projected;
                }
            }
            output[position * HIDDEN..(position + 1) * HIDDEN].copy_from_slice(&scratch.hidden);
        }
        validate_hidden_output(output, token_count)
    }
}

struct TextScratch {
    hidden: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention: Vec<f32>,
    projected: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    scores: Vec<f32>,
    q8: Q8Scratch,
}

impl TextScratch {
    fn new(token_count: usize) -> Self {
        Self {
            hidden: vec![0.0; HIDDEN],
            normed: vec![0.0; HIDDEN],
            q: vec![0.0; QUERY_WIDTH],
            k: vec![0.0; KV_WIDTH],
            v: vec![0.0; KV_WIDTH],
            attention: vec![0.0; QUERY_WIDTH],
            projected: vec![0.0; HIDDEN],
            gate: vec![0.0; FFN_WIDTH],
            up: vec![0.0; FFN_WIDTH],
            scores: vec![0.0; token_count],
            q8: Q8Scratch::new(FFN_WIDTH),
        }
    }
}

fn attention(
    query: &[f32],
    cache: &KvCacheF16,
    layer: usize,
    position: usize,
    token_count: usize,
    scores: &mut [f32],
    output: &mut [f32],
) {
    output.fill(0.0);
    let scale = 1.0 / (HEAD_WIDTH as f32).sqrt();
    let active_scores = &mut scores[..=position];
    for query_head in 0..QUERY_HEADS {
        let kv_head = query_head / (QUERY_HEADS / KV_HEADS);
        let query = &query[query_head * HEAD_WIDTH..(query_head + 1) * HEAD_WIDTH];
        for (key_position, score) in active_scores.iter_mut().enumerate() {
            let key_start = (layer * token_count + key_position) * KV_WIDTH + kv_head * HEAD_WIDTH;
            *score = dot_f16_f32(
                query,
                &cache.k[key_start..key_start + HEAD_WIDTH],
                HEAD_WIDTH,
            ) * scale;
        }
        softmax_inplace(active_scores);
        let output_head = &mut output[query_head * HEAD_WIDTH..(query_head + 1) * HEAD_WIDTH];
        for (value_position, &weight) in active_scores.iter().enumerate() {
            let value_start =
                (layer * token_count + value_position) * KV_WIDTH + kv_head * HEAD_WIDTH;
            for (result, value) in output_head
                .iter_mut()
                .zip(&cache.v[value_start..value_start + HEAD_WIDTH])
            {
                *result += weight * f16_to_f32(*value);
            }
        }
    }
}

fn load_f32(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != len.checked_mul(4).ok_or("F32 tensor size overflow")? {
        return Err(format!("Invalid {name} byte length"));
    }
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("Non-finite tensor: {name}"));
    }
    Ok(values)
}

fn reserve_f32(name: &str, len: usize) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| format!("Failed to allocate {name}: {error}"))?;
    values.resize(len, 0.0);
    Ok(values)
}

fn reserve_u16(name: &str, len: usize) -> Result<Vec<u16>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| format!("Failed to allocate {name}: {error}"))?;
    values.resize(len, 0);
    Ok(values)
}

fn validate_hidden_output(output: Vec<f32>, token_count: usize) -> Result<Vec<f32>, String> {
    let expected = token_count
        .checked_mul(HIDDEN)
        .ok_or("Qwen3 layer-35 output size overflow")?;
    if token_count == 0 || output.is_empty() || output.len() != expected {
        return Err("Invalid Qwen3 layer-35 output shape".into());
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err("Qwen3 layer-35 output contains NaN or infinity".into());
    }
    Ok(output)
}

pub(crate) fn z_image_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

#[cfg(test)]
mod tests {
    use super::{validate_hidden_output, z_image_prompt, Qwen3TextEncoder};
    use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};
    use crate::core::thread_pool::ComputePool;
    use std::sync::Arc;

    struct EmptySource;

    impl TensorSource for EmptySource {
        fn metadata(&self, _key: &str) -> Option<&MetaValue> {
            None
        }

        fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
            None
        }

        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    #[test]
    fn z_image_prompt_enables_special_token_parsing() {
        assert_eq!(
            z_image_prompt("Hello"),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn text_loader_rejects_an_incomplete_graph() {
        let error = Qwen3TextEncoder::load(Arc::new(EmptySource), Arc::new(ComputePool::new(1)))
            .err()
            .unwrap();
        assert_eq!(error, "Missing tensor: model.embed_tokens.weight");
    }

    #[test]
    fn layer_35_output_must_have_finite_token_rows() {
        assert!(validate_hidden_output(Vec::new(), 1).is_err());
        assert!(validate_hidden_output(vec![0.0; 2_559], 1).is_err());
        let mut non_finite = vec![0.0; 2_560];
        non_finite[17] = f32::NAN;
        assert!(validate_hidden_output(non_finite, 1).is_err());
        assert_eq!(
            validate_hidden_output(vec![0.0; 2_560], 1).unwrap().len(),
            2_560
        );
    }

    #[test]
    #[ignore = "requires Z_IMAGE_TEXT_GGUF"]
    fn supplied_qwen3_loader_matches_the_fixed_signature() {
        let source: Arc<dyn TensorSource> = Arc::new(
            crate::core::loader::GGUFLoader::from_file(
                std::env::var("Z_IMAGE_TEXT_GGUF").expect("missing Z_IMAGE_TEXT_GGUF"),
            )
            .unwrap(),
        );
        Qwen3TextEncoder::load(source, Arc::new(ComputePool::new(1))).unwrap();
    }

    #[test]
    #[ignore = "full supplied Qwen3 encode; requires Z_IMAGE_TEXT_GGUF"]
    fn supplied_qwen3_encodes_layer_35() {
        let source: Arc<dyn TensorSource> = Arc::new(
            crate::core::loader::GGUFLoader::from_file(
                std::env::var("Z_IMAGE_TEXT_GGUF").expect("missing Z_IMAGE_TEXT_GGUF"),
            )
            .unwrap(),
        );
        let encoder = Qwen3TextEncoder::load(source, Arc::new(ComputePool::new(1))).unwrap();
        let hidden = encoder.encode_layer_35("Hello").unwrap();
        assert!(!hidden.is_empty());
        assert_eq!(hidden.len() % 2_560, 0);
        assert!(hidden.iter().all(|value| value.is_finite()));
    }
}
