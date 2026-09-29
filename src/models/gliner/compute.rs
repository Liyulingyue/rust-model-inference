//! DeBERTa-v3 encoder forward for `arch = "gliner2"`.
//!
//! Mirrors `transformers` 4.48.1
//! `models/deberta_v2/modeling_deberta_v2.py`, which is what
//! `SpanExtractorModel._load_encoder` instantiates. Four things differ from the
//! `bert`/`jina-bert-v2` encoder in [`crate::models::bert_family`], and each is
//! easy to get wrong:
//!
//! 1. **Disentangled attention.** The score is
//!    `content·content + content·position + position·content`, all three
//!    divided by the *same* `sqrt(head_dim * scale_factor)` where
//!    `scale_factor = 1 + |pos_att_type| = 3` — not by `sqrt(head_dim)`.
//! 2. **The position keys and queries come from the same Q/K projections as the
//!    content ones** (`share_att_key = true`), applied to the LayerNorm'd
//!    relative-position table.
//! 3. **ST-transposed residuals.** Each sublayer is `LayerNorm(f(x) + x)`, so
//!    the norm sees the *sum*, not `f(x)` alone.
//! 4. **`transpose_for_scores` on the position table** splits the last axis into
//!    `(n_head, head_dim)` and moves the head axis forward, so head `h` of a
//!    position row is `rel[h * head_dim .. (h + 1) * head_dim]`.
//!
//! ## Relative position bucketing
//!
//! `make_log_bucket_position` with `bucket_size = 256`, `max_position = 512`,
//! then `att_span = pos_ebd_size = position_buckets = 256`:
//!
//! ```text
//! mid = 128
//! abs_pos = if -128 < rel < 128 { 127 } else { |rel| }
//! log_pos = ceil(ln(abs_pos / 128) / ln(511 / 128) * 127) + 128
//! bucket  = if abs_pos <= 128 { rel } else { log_pos * sign(rel) }
//! c2p_pos = clamp(bucket + 256, 0, 511)
//! p2c_pos = clamp(-bucket + 256, 0, 511)
//! ```
//!
//! `bucket` is cast back to a long (truncation toward zero) before the clamp.
//! `abs_pos == 0` would divide by zero, but `abs_pos <= mid` picks the `rel`
//! branch first, so the result is discarded.

use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::Weight;
use crate::ops::{gelu_erf_inplace, layer_norm, softmax_inplace};
use std::sync::Arc;

use super::weights::{LayerWeights, ModelWeights};

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_ff: usize,
    /// `attention_head_size`; DeBERTa-v3 derives it as
    /// `hidden_size / num_attention_heads`.
    pub head_dim: usize,
    pub eps: f32,
    /// `att_span`, the shift applied to the bucketed relative position. Not the
    /// same as `bucket_size / 2`.
    pub att_span: usize,
    /// Rows in the relative-position table, `2 * att_span`.
    pub pos_ebd_size: usize,
    pub bucket_size: usize,
    pub max_relative_positions: usize,
    pub norm_rel_embeddings: bool,
    pub vocab_size: usize,
}

/// `make_log_bucket_position` for a single signed offset.
fn relative_bucket(rel: i32, mid: usize, max_position: usize) -> i32 {
    let abs_pos = if rel > -(mid as i32) && rel < mid as i32 {
        (mid - 1) as f32
    } else {
        rel.unsigned_abs() as f32
    };
    if abs_pos <= mid as f32 {
        // The `where(abs_pos <= mid, rel, ...)` branch. `rel` is zero here only
        // when the first arm fired, so the `ln(0)` below is never reached.
        return rel;
    }
    let log_pos = (abs_pos / mid as f32).ln()
        / ((max_position - 1) as f32 / mid as f32).ln()
        * (mid - 1) as f32
        + mid as f32;
    (log_pos * rel.signum() as f32) as i32
}

/// Index into the relative-position table for the (query, key) pair.
///
/// The reference builds two tables — `c2p_pos = clamp(relative_pos + att_span)`
/// and `p2c_pos = clamp(-r_pos + att_span)` — but the p2c one is
/// `gather(...).transpose(-1, -2)`-ed into place, which swaps the roles of the
/// two axes: the result at `(query t, key s)` reads `raw_p2c[s, p2c_pos[s, t]]`,
/// i.e. `clamp(-(s - t) + att_span)` = `clamp((t - s) + att_span)`. Both tables
/// therefore index the *same* bucket; only the dotted vectors differ (Q against
/// the position keys, K against the position queries). Keeping one table makes
/// that collapse explicit instead of leaving a double negation to get wrong.
fn position_index_table(
    n_tokens: usize,
    bucket_size: usize,
    max_position: usize,
    att_span: usize,
) -> Vec<usize> {
    let mid = bucket_size / 2;
    let limit = 2 * att_span - 1;
    let mut table = Vec::with_capacity(n_tokens * n_tokens);
    for query in 0..n_tokens {
        for key in 0..n_tokens {
            let bucket = relative_bucket(query as i32 - key as i32, mid, max_position);
            table.push((bucket + att_span as i32).clamp(0, limit as i32) as usize);
        }
    }
    table
}

/// Quantization scratch shared by every matmul in the forward.
struct MatmulScratch {
    q8k: Vec<crate::ops::quant::BlockQ8K>,
    q8: Vec<u8>,
    scales: Vec<f32>,
}

impl MatmulScratch {
    fn new(max_width: usize) -> Self {
        MatmulScratch {
            q8k: vec![
                crate::ops::quant::BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] };
                max_width.div_ceil(crate::ops::quant::QK_K)
            ],
            q8: vec![0u8; max_width],
            scales: vec![0.0; max_width.div_ceil(32)],
        }
    }
}

/// One `weight @ input` with an explicit output width.
///
/// `Weight::n_out` is *not* usable for F32: `QuantizedTensor::n_rows` reports
/// `!data.is_empty()` for that variant, so every F32 `Weight` claims an output
/// width of 1. `quantize_and_matmul_with_scratch` reads `self.n_out`, so
/// calling it on an F32 weight would silently compute a single column. The F32
/// path therefore calls the kernel directly with the real width, which is the
/// same call the prepared path makes (`input_q8` empty ⇒ the F32 branch of
/// `forward_prepared`), and keeps the SIMD + thread-pool dispatch.
pub fn matmul_into(
    weight: &Weight<'_>,
    input: &[f32],
    n_in: usize,
    n_out: usize,
    output: &mut [f32],
    pool: &Arc<ComputePool>,
) {
    debug_assert_eq!(input.len(), n_in);
    debug_assert_eq!(output.len(), n_out);
    if !matches!(weight.ggml_type, crate::core::tensor::GGMLType::F32) {
        let mut scratch = MatmulScratch::new(n_in.max(n_out));
        weight.quantize_and_matmul_with_scratch(
            input,
            &mut scratch.q8k,
            &mut scratch.q8,
            &mut scratch.scales,
            output,
            pool,
        );
        return;
    }
    let output_ptr = output.as_mut_ptr();
    pool.compute(|ith, nth| {
        // Same disjointness contract as `quantize_and_matmul_with_scratch`:
        // the kernel writes only `output[ith * n_out .. (ith + 1) * n_out]`.
        let out = unsafe { std::slice::from_raw_parts_mut(output_ptr, n_out) };
        weight
            .kernel
            .forward_prepared(input, &[], &[], None, out, n_in, n_out, ith, nth);
    });
}

/// `weight @ input` with a fresh output buffer.
pub fn matmul_vec(
    weight: &Weight<'_>,
    input: &[f32],
    n_out: usize,
    pool: &Arc<ComputePool>,
) -> Vec<f32> {
    let mut output = vec![0.0f32; n_out];
    matmul_into(weight, input, input.len(), n_out, &mut output, pool);
    output
}

/// `weight @ input` for a row-major `[rows, n_in] -> [rows, n_out]` matmul.
fn matmul_rows(
    weight: &Weight<'_>,
    input: &[f32],
    n_in: usize,
    n_out: usize,
    output: &mut [f32],
    pool: &Arc<ComputePool>,
) {
    debug_assert_eq!(input.len() % n_in, 0);
    let rows = input.len() / n_in;
    debug_assert_eq!(output.len(), rows * n_out);
    for row in 0..rows {
        let start = row * n_in;
        matmul_into(
            weight,
            &input[start..start + n_in],
            n_in,
            n_out,
            &mut output[row * n_out..(row + 1) * n_out],
            pool,
        );
    }
}

fn add_bias(values: &mut [f32], bias: &[f32]) {
    for (value, extra) in values.iter_mut().zip(bias) {
        *value += *extra;
    }
}

/// `add_bias` for a row-major `[rows, width]` buffer. Zipping the whole buffer
/// against the bias would only ever reach the first row.
fn add_bias_rows(values: &mut [f32], width: usize, bias: &[f32]) {
    debug_assert_eq!(values.len() % width, 0);
    for row in values.chunks_exact_mut(width) {
        add_bias(row, bias);
    }
}

/// Decode one F32 row. The converter only emits F32 for this architecture.
fn decode_row(
    bytes: &[u8],
    ggml_type: crate::core::tensor::GGMLType,
    offset: usize,
    out: &mut [f32],
) -> Result<(), String> {
    if !matches!(ggml_type, crate::core::tensor::GGMLType::F32) {
        return Err(format!("gliner2 tensors must be F32, found {ggml_type:?}"));
    }
    let needed = (offset + out.len()) * 4;
    if bytes.len() < needed {
        return Err(format!("tensor row at {offset} (+{}) exceeds its data", out.len()));
    }
    for (index, slot) in out.iter_mut().enumerate() {
        let base = (offset + index) * 4;
        *slot = f32::from_le_bytes([bytes[base], bytes[base + 1], bytes[base + 2], bytes[base + 3]]);
    }
    Ok(())
}

/// Full-sequence encoder output, `[n_tokens, n_embd]` row-major.
pub fn encode(
    weights: &ModelWeights<'_>,
    config: &EncoderConfig,
    input_ids: &[u32],
    n_threads_arg: usize,
) -> Result<Vec<f32>, String> {
    let n_tokens = input_ids.len();
    if n_tokens == 0 {
        return Err("encoder input produced no tokens".into());
    }
    let d = config.n_embd;
    if d != config.n_head * config.head_dim {
        return Err(format!(
            "embedding length {d} is not n_head {} * head_dim {}",
            config.n_head, config.head_dim
        ));
    }
    if n_tokens > config.max_relative_positions.max(config.pos_ebd_size) {
        return Err(format!(
            "sequence of {n_tokens} tokens exceeds the relative-position range of {}",
            config.max_relative_positions.max(config.pos_ebd_size)
        ));
    }
    for (position, id) in input_ids.iter().enumerate() {
        if *id as usize >= config.vocab_size {
            return Err(format!(
                "token id {id} at position {position} is outside the vocabulary of {}",
                config.vocab_size
            ));
        }
    }

    let available = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(4);
    let n_threads = crate::app::resolve_thread_count(n_threads_arg, available);
    let pool = Arc::new(ComputePool::new(n_threads));

    // `DebertaV2Encoder.get_rel_embedding()` LayerNorms the whole table once,
    // before any projection. Only rows `[0, 2 * att_span)` are ever indexed and
    // `pos_ebd_size` is exactly that, so this is the entire table.
    let rel_rows = config.pos_ebd_size;
    let mut rel_table = vec![0.0f32; rel_rows * d];
    let mut row = vec![0.0f32; d];
    for index in 0..rel_rows {
        let start = index * d;
        let slice = &mut rel_table[start..start + d];
        decode_row(weights.rel_embeddings, weights.rel_embeddings_type, start, slice)?;
        if config.norm_rel_embeddings {
            layer_norm(
                slice,
                &weights.rel_norm.weight,
                &weights.rel_norm.bias,
                config.eps,
                &mut row,
            );
            slice.copy_from_slice(&row);
        }
    }

    // The position keys/queries are per layer: `share_att_key` means the
    // position table goes through *this layer's* `query_proj` / `key_proj`
    // rather than a second pair of projections, and `DebertaV2Encoder.forward`
    // hands the same `rel_embeddings` to every layer. Recomputing it per layer
    // is what the reference does, and it costs two extra
    // `pos_ebd_size x n_embd` matmuls per layer.
    let mut pos_query = vec![0.0f32; rel_rows * d];
    let mut pos_key = vec![0.0f32; rel_rows * d];

    // Embeddings: word lookup then LayerNorm. No position or token_type
    // embeddings (`position_biased_input = false`, `type_vocab_size = 0`).
    let mut hidden = vec![0.0f32; n_tokens * d];
    for (token, slot) in input_ids.iter().zip(hidden.chunks_exact_mut(d)) {
        decode_row(
            weights.token_embd,
            weights.token_embd_type,
            *token as usize * d,
            slot,
        )?;
    }
    let mut normed = vec![0.0f32; d];
    for slot in hidden.chunks_exact_mut(d) {
        layer_norm(slot, &weights.tok_norm.weight, &weights.tok_norm.bias, config.eps, &mut normed);
        slot.copy_from_slice(&normed);
    }

    let hd = config.head_dim;
    // `scale_factor = 1 + |pos_att_type|`; all three terms share the scale.
    let scale = 1.0f32 / ((hd * 3) as f32).sqrt();
    let position_table = position_index_table(
        n_tokens,
        config.bucket_size,
        config.max_relative_positions,
        config.att_span,
    );

    let mut q = vec![0.0f32; n_tokens * d];
    let mut k = vec![0.0f32; n_tokens * d];
    let mut v = vec![0.0f32; n_tokens * d];
    let mut context = vec![0.0f32; n_tokens * d];
    let mut projected = vec![0.0f32; d];
    let mut ffn = vec![0.0f32; config.n_ff];
    let mut scores = vec![0.0f32; n_tokens];

    for layer in &weights.layers {
        matmul_rows(&layer.attn_q, &rel_table, d, d, &mut pos_query, &pool);
        matmul_rows(&layer.attn_k, &rel_table, d, d, &mut pos_key, &pool);
        add_bias_rows(&mut pos_query, d, &layer.attn_q_bias);
        add_bias_rows(&mut pos_key, d, &layer.attn_k_bias);
        // Q/K/V per token.
        project_all(&layer.attn_q, &layer.attn_q_bias, &hidden, &mut q, d, d, &pool);
        project_all(&layer.attn_k, &layer.attn_k_bias, &hidden, &mut k, d, d, &pool);
        project_all(&layer.attn_v, &layer.attn_v_bias, &hidden, &mut v, d, d, &pool);

        // Bidirectional attention; batch is always 1 so no key is masked.
        for t in 0..n_tokens {
            let row_base = t * n_tokens;
            for head in 0..config.n_head {
                let head_offset = head * hd;
                for s in 0..n_tokens {
                    let query = &q[t * d + head_offset..t * d + head_offset + hd];
                    let key = &k[s * d + head_offset..s * d + head_offset + hd];
                    let mut score = dot(query, key);
                    let c2p = position_table[row_base + s] * d + head_offset;
                    let c2p_raw = dot(query, &pos_key[c2p..c2p + hd]);
                    score += c2p_raw;
                    let p2c = position_table[row_base + s] * d + head_offset;
                    let p2c_raw = dot(key, &pos_query[p2c..p2c + hd]);
                    score += p2c_raw;
                    scores[s] = score * scale;
                }
                softmax_inplace(&mut scores[..n_tokens]);
                let out_base = t * d + head_offset;
                for i in 0..hd {
                    let mut acc = 0.0f32;
                    for s in 0..n_tokens {
                        acc += v[s * d + head_offset + i] * scores[s];
                    }
                    context[out_base + i] = acc;
                }
            }
        }

        // Attention output projection, then the ST-transposed residual:
        // `LayerNorm(attn_output(context) + hidden)`.
        for t in 0..n_tokens {
            let offset = t * d;
            matmul_into(
                &layer.attn_output,
                &context[offset..offset + d],
                d,
                d,
                &mut projected,
                &pool,
            );
            add_bias(&mut projected, &layer.attn_output_bias);
            for i in 0..d {
                projected[i] += hidden[offset + i];
            }
            layer_norm(
                &projected,
                &layer.attn_out_norm.weight,
                &layer.attn_out_norm.bias,
                config.eps,
                &mut normed,
            );
            hidden[offset..offset + d].copy_from_slice(&normed);
        }

        // FFN: GELU (erf form, `ACT2FN["gelu"]`) up, then down, then the same
        // ST-transposed residual.
        for t in 0..n_tokens {
            let x = &hidden[t * d..(t + 1) * d];
            matmul_into(&layer.ffn_up, x, d, config.n_ff, &mut ffn, &pool);
            add_bias(&mut ffn, &layer.ffn_up_bias);
            gelu_erf_inplace(&mut ffn);
            matmul_into(&layer.ffn_down, &ffn, config.n_ff, d, &mut projected, &pool);
            add_bias(&mut projected, &layer.ffn_down_bias);
            for i in 0..d {
                projected[i] += hidden[t * d + i];
            }
            layer_norm(
                &projected,
                &layer.output_norm.weight,
                &layer.output_norm.bias,
                config.eps,
                &mut normed,
            );
            hidden[t * d..(t + 1) * d].copy_from_slice(&normed);
        }
    }
    Ok(hidden)
}

/// `dense(x) + bias` for every row of a `[n_tokens, n_in]` batch.
fn project_all(
    weight: &Weight<'_>,
    bias: &[f32],
    hidden: &[f32],
    out: &mut [f32],
    n_in: usize,
    n_out: usize,
    pool: &Arc<ComputePool>,
) {
    matmul_rows(weight, hidden, n_in, n_out, out, pool);
    for row in out.chunks_exact_mut(n_out) {
        add_bias(row, bias);
    }
}

#[inline]
fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_matches_the_reference_formula() {
        // Inside the near band the bucket is the raw offset.
        assert_eq!(relative_bucket(0, 128, 512), 0);
        assert_eq!(relative_bucket(5, 128, 512), 5);
        assert_eq!(relative_bucket(-5, 128, 512), -5);
        assert_eq!(relative_bucket(127, 128, 512), 127);
        assert_eq!(relative_bucket(-127, 128, 512), -127);
        // |rel| == mid keeps the raw offset via `abs_pos <= mid`.
        assert_eq!(relative_bucket(128, 128, 512), 128);
        assert_eq!(relative_bucket(-128, 128, 512), -128);
        // Beyond it, the log buckets saturate at mid - 1.
        assert_eq!(relative_bucket(511, 128, 512), 255);
        assert_eq!(relative_bucket(-511, 128, 512), -255);
        let beyond = relative_bucket(4096, 128, 512);
        assert!(beyond >= 255, "far offsets clamp, got {beyond}");
    }

    #[test]
    fn position_table_clamps_into_the_att_span_range() {
        let table = position_index_table(3, 256, 512, 256);
        assert_eq!(table.len(), 9);
        // Diagonal: rel 0 -> bucket 0 + att_span.
        assert_eq!(table[0], 256);
        assert_eq!(table[4], 256);
        assert_eq!(table[8], 256);
        // query 2, key 0: rel 2 -> 258; query 0, key 2: rel -2 -> 254.
        assert_eq!(table[6], 258);
        assert_eq!(table[2], 254);
        assert!(table.iter().all(|&value| value <= 511));
    }

    #[test]
    fn far_offsets_saturate_at_the_edge_of_the_table() {
        let table = position_index_table(600, 256, 512, 256);
        // The widest reachable bucket is 255, so the largest index is 511.
        assert!(table.iter().all(|&value| value <= 511));
        assert_eq!(table[0], 256);
    }
}
