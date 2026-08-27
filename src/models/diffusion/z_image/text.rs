use std::sync::Arc;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::ops::dot_f32;
use crate::ops::silu_mul_inplace;
use crate::ops::softmax_inplace;
use crate::ops::{
    attention_value_f32, embedding_lookup, rms_norm, rms_norm_inplace, rope_neox,
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
        let total_start = std::time::Instant::now();
        let t_tok = std::time::Instant::now();
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
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::token_ids("z_image.prompt_ids", &ids));
        let t_tok = t_tok.elapsed();
        let t_fwd = std::time::Instant::now();
        let output = self.forward_to_block(&ids, LAYER_35_BLOCKS)?;
        let t_fwd = t_fwd.elapsed();
        eprintln!(
            "[text-profile] n_tokens={}  forward={:.1}ms  tokenize={:.1}ms  total={:.1}ms",
            ids.len(),
            t_fwd.as_secs_f64() * 1000.0,
            t_tok.as_secs_f64() * 1000.0,
            total_start.elapsed().as_secs_f64() * 1000.0,
        );
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "z_image.text_layer_35",
            Some(LAYER_35_BLOCKS),
            &[HIDDEN, ids.len()],
            &output,
        ));
        Ok(output)
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
        let mut cache = TextKvCache {
            k: reserve_f32("Qwen3 key cache", cache_len)?,
            v: reserve_f32("Qwen3 value cache", cache_len)?,
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
                cache.k[cache_row..cache_row + KV_WIDTH].copy_from_slice(&scratch.k);
                cache.v[cache_row..cache_row + KV_WIDTH].copy_from_slice(&scratch.v);
                attention(
                    &scratch.q,
                    &cache,
                    layer_index,
                    position,
                    token_count,
                    &mut scratch.scores,
                    &mut scratch.value_column,
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

struct TextKvCache {
    k: Vec<f32>,
    v: Vec<f32>,
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
    value_column: Vec<f32>,
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
            value_column: vec![0.0; token_count],
            q8: Q8Scratch::new(FFN_WIDTH),
        }
    }
}

fn attention(
    query: &[f32],
    cache: &TextKvCache,
    layer: usize,
    position: usize,
    token_count: usize,
    scores: &mut [f32],
    value_column: &mut [f32],
    output: &mut [f32],
) {
    output.fill(0.0);
    let scale = 1.0 / (HEAD_WIDTH as f32).sqrt();
    for query_head in 0..QUERY_HEADS {
        let kv_head = query_head / (QUERY_HEADS / KV_HEADS);
        let query = &query[query_head * HEAD_WIDTH..(query_head + 1) * HEAD_WIDTH];
        {
            let active_scores = &mut scores[..=position];
            for (key_position, score) in active_scores.iter_mut().enumerate() {
                let key_start =
                    (layer * token_count + key_position) * KV_WIDTH + kv_head * HEAD_WIDTH;
                *score = dot_f32(query, &cache.k[key_start..key_start + HEAD_WIDTH], HEAD_WIDTH) * scale;
            }
        }
        softmax_inplace(&mut scores[..=position]);
        let output_head = &mut output[query_head * HEAD_WIDTH..(query_head + 1) * HEAD_WIDTH];
        #[cfg(target_arch = "aarch64")]
        for dimension in 0..HEAD_WIDTH {
            for (value_position, value) in value_column.iter_mut().enumerate() {
                let value_start = (layer * token_count + value_position) * KV_WIDTH
                    + kv_head * HEAD_WIDTH
                    + dimension;
                *value = cache.v[value_start];
            }
            output_head[dimension] =
                attention_value_f32(value_column, scores, value_column.len(), scores.len());
        }
        #[cfg(not(target_arch = "aarch64"))]
        for (value_position, &weight) in scores[..=position].iter().enumerate() {
            let value_start =
                (layer * token_count + value_position) * KV_WIDTH + kv_head * HEAD_WIDTH;
            for (result, &value) in output_head
                .iter_mut()
                .zip(&cache.v[value_start..value_start + HEAD_WIDTH])
            {
                *result += weight * value;
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
    use crate::ops::attention_value_f32;
    use crate::ops::dot_f32;
    use crate::ops::silu_mul_inplace;
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

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn silu_mul_inplace_matches_pinned_ggml_neon_activation() {
        let gate = [0x3f6c_76fc, 0x3fe2_6ebc, 0xbf66_8824, 0xc009_429b].map(f32::from_bits);
        let mut up = [0xbfa1_5cef, 0x4013_8811, 0x3f8f_08d4, 0xbfe4_c88a].map(f32::from_bits);

        silu_mul_inplace(&gate, &mut up);

        assert_eq!(
            up.map(f32::to_bits),
            [0xbf55_6098, 0x405e_f7a5, 0xbe94_dea6, 0x3ecd_be95],
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn qwen_attention_dot_matches_pinned_ggml_neon_reduction() {
        let left = (0..super::HEAD_WIDTH)
            .map(|index| (((index * 37) % 101) as f32 - 50.0) / 7.0)
            .collect::<Vec<_>>();
        let right = (0..super::HEAD_WIDTH)
            .map(|index| (((index * 53 + 11) % 97) as f32 - 48.0) / 11.0)
            .collect::<Vec<_>>();

        assert_eq!(
            dot_f32(&left, &right, super::HEAD_WIDTH).to_bits(),
            0x41d3_f2b4
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[test]
    fn x86_dot_f32_matches_pinned_ggml_neon_reduction() {
        // 这是仓库里**唯一** x86 dot_f32 跨 AVX2/no-AVX2 的 bit-pinned 测试。
        // 关键正确性保证：两 arch 都产出 0x41d3_f2b4（与 ggml NEON 参考完全一致）。
        let left = (0..super::HEAD_WIDTH)
            .map(|index| (((index * 37) % 101) as f32 - 50.0) / 7.0)
            .collect::<Vec<_>>();
        let right = (0..super::HEAD_WIDTH)
            .map(|index| (((index * 53 + 11) % 97) as f32 - 48.0) / 11.0)
            .collect::<Vec<_>>();

        assert_eq!(
            dot_f32(&left, &right, super::HEAD_WIDTH).to_bits(),
            0x41d3_f2b4
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn qwen_attention_softmax_matches_the_pinned_ggml_neon_reduction() {
        let mut scores = vec![0.0; 16];
        scores[..5].copy_from_slice(
            &[
                0x40d4_779d,
                0x40a0_90be,
                0x40fc_8756,
                0x40f4_4f84,
                0x40ee_1fbb,
            ]
            .map(f32::from_bits),
        );

        softmax_inplace(&mut scores[..=4]);

        assert_eq!(
            scores.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
            [
                0x3dd4_b07f,
                0x3ca8_09ef,
                0x3eb9_f242,
                0x3e8f_d4ec,
                0x3e6d_1827,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
            ],
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

    /// `attention_value_f32` 与标量参考的 bit-level 一致性。
    /// Z-Image 文本编码器 attention value 聚合直接调这个函数（之前还有
    /// `qwen_attention_value` 包装，已删——wrapper 变成1 行透传，没意义）。
    /// 实测：len ≤ 129 时与标量 1 ULP 以内（LLVM 不自动向量化）；
    ///       len ≥ 255 时 LLVM 自动向量化 reference，与手写 SIMD 走不同
    ///       指令序列，最多 ~3 ULP。HEAD_WIDTH=128 是真实调用长度。
    /// 关键正确性保证见 `attention_value_f32_matches_pinned_ggml_reduction`：
    /// 两 arch 都产出 0xbbf2_4ce4（与 ggml NEON 参考完全一致）。
    #[test]
    fn attention_value_f32_matches_scalar_reference() {
        let mut max_diff_overall = 0u32;
        for &len in &[0usize, 1, 3, 4, 5, 7, 8, 9, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 384] {
            let values: Vec<f32> = (0..len).map(|i| (i as f32 * 0.013).sin() * 2.5).collect();
            let weights: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.029).cos() * 1.3).collect();
            let expected: f32 = values
                .iter()
                .zip(&weights)
                .fold(0.0f32, |acc, (v, w)| acc + v * w);
            let actual = attention_value_f32(&values, &weights, values.len(), values.len());
            let diff_bits = (actual.to_bits() as i32).wrapping_sub(expected.to_bits() as i32).abs() as u32;
            if diff_bits > max_diff_overall {
                max_diff_overall = diff_bits;
            }
            assert!(
                diff_bits <= 8,
                "attention_value_f32 mismatch at len={len}: \
                 actual={actual:?} (bits={:#x}), expected={expected:?} (bits={:#x}), diff_bits={diff_bits}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
        eprintln!("max_diff_overall={max_diff_overall} ULP across all lengths");
    }

    /// 复用 ggml NEON pinned 测试的输入，验证两 arch 都产出 `0xbbf2_4ce4`。
    /// `dot_f32`（x86 AVX2）与 `dot_f32_neon` 的 reduce 顺序必须等价，否则
    /// 整个 DiT 文本编码会逐 token 偏离 ggml 参考。
    #[test]
    fn attention_value_f32_matches_pinned_ggml_reduction() {
        let values = [
            0xbc43_7f80u32, 0x3dcf_0c16, 0x3ab2_6fac, 0xbc9e_b19e,
            0xbc68_338b, 0x3e83_1f48, 0xbd8d_9b7b, 0xbea6_6c9f,
            0xbe7e_b316, 0x3dcd_159e, 0xbd1d_9b7f, 0xbe02_e230,
            0x3d0c_95ad, 0xbe0a_58fc, 0xbe55_ef52, 0x3d88_c14e,
        ]
        .map(f32::from_bits);
        let weights = [
            0x3dd4_b07fu32, 0x3ca8_09ef, 0x3eb9_f242, 0x3e8f_d4ec,
            0x3e6d_1827, 0x0000_0000, 0x0000_0000, 0x0000_0000,
            0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
            0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
        ]
        .map(f32::from_bits);
        // 两 arch 都跑：aarch64 → dot_f32_neon，x86 → dot_f32 (AVX2)。
        // 结果应与 ggml NEON 参考一致。
        assert_eq!(
            attention_value_f32(&values, &weights, values.len(), values.len()).to_bits(),
            0xbbf2_4ce4
        );
    }

    /// 空切片不 panic。
    #[test]
    fn attention_value_f32_empty_input_is_zero() {
        let values: Vec<f32> = vec![];
        let weights: Vec<f32> = vec![];
        let result = attention_value_f32(&values, &weights, values.len(), values.len());
        assert_eq!(result, 0.0);
    }
}
