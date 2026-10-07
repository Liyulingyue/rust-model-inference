//! ModernBERT encoder for `fastino/GLiNER2.5-Decide-1B`.
//!
//! This is the one checkpoint in the GLiNER family that is not a DeBERTa-v2, and
//! every architectural decision below is a difference from the encoder the other
//! eleven share. The reference reaches it through the same generic
//! `AutoModel.from_pretrained` path as everything else — `model_name` is
//! `jhu-clsp/ettin-enc-from-dec-1b`, whose `architectures` is
//! `ModernBertForMaskedLM` — so `transformers` *is* the reference here, and this
//! port is measured against it rather than against a GLiNER-side reimplementation.
//!
//! # What differs from the DeBERTa path
//!
//! | | DeBERTa-v2 (11 models) | ModernBERT (this one) |
//! |---|---|---|
//! | norm | LayerNorm with bias | **LayerNorm, `bias = False`**, eps 1e-5 |
//! | attention | disentangled, relative buckets | **plain, sliding-window hybrid** |
//! | QKV | three separate projections | **fused `[3d, d]`, interleaved on dim 2** |
//! | MLP | `Linear(d, 4d)` + GELU | **`Linear(d, 2f)` chunked into a GeLU GLU** |
//! | position | relative embeddings | **none** (`sans_pos`); RoPE only |
//! | layer 0 | — | **`attn_norm` is `Identity`** |
//!
//! `src/ops/norm.rs`'s `rms_norm` and `silu_mul_inplace` are the *wrong*
//! operators for this checkpoint: ModernBERT normalizes with LayerNorm and
//! gates with GeLU, not SiLU.
//!
//! # The hybrid attention schedule
//!
//! `modeling_modernbert.py:484`:
//!
//! ```python
//! if layer_id % config.global_attn_every_n_layers != 0:
//!     self.local_attention = (config.local_attention // 2, config.local_attention // 2)
//! else:
//!     self.local_attention = (-1, -1)
//! ```
//!
//! so layers 0, 3, 6, … are global and every other layer sees a 128-wide window
//! (64 left, 64 right). All 28 layers' tensors are named identically, so nothing
//! in the checkpoint distinguishes them — the schedule is the *only* source of
//! that fact, which is why the converter writes `attention.local_window` and
//! `attention.global_every_n_layers` into the GGUF rather than leaving the
//! loader to assume the DeBERTa shape.
//!
//! # Why `interleaved` QKV matters
//!
//! `Wqkv` is `[3 * d, d]`, which invites reading it as three contiguous
//! `[d, d]` blocks — the layout LLaMA uses. ModernBERT does not:
//! `qkv.view(bs, seq, 3, heads, head_dim)`, then `unbind(dim=2)`. So the rows
//! interleave as `q0 k0 v0 q1 k1 v1 …` per head rather than `q… q… k… k… v… v…`.
//! Getting this wrong produces a plausible encoder that is simply wrong, and
//! the logits stay finite, so only the oracle catches it.
//!
//! # RoPE
//!
//! `position_embedding_type = "sans_pos"`, so position enters only through
//! RoPE with `theta = 160000.0`, applied to Q and K in NeoX style
//! (`rotate_half`, i.e. not interleaved adjacent pairs). Scaling is plain
//! `1/sqrt(head_dim)` from `scaled_dot_product_attention` — there is no
//! `scale_divisor` here, unlike DeBERTa's `1 + |pos_att_type|`.

pub mod bpe;

use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};

/// The encoder's dimensions and hybrid-attention schedule, all from GGUF metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct EttinConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub norm_eps: f32,
    /// Sliding-window width for local layers. `None` when the model is all-global.
    pub local_window: Option<usize>,
    /// A layer is local when `layer_id % global_every_n_layers != 0`.
    pub global_every_n_layers: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    /// Layers whose `attn_norm` is `Identity` and therefore has no tensor.
    pub identity_attn_norm: Vec<usize>,
}

impl EttinConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let key = |suffix: &str| format!("gliner2.{suffix}");
        let usize_meta = |name: &str| -> Result<usize, String> {
            source
                .metadata(&key(name))
                .and_then(|value| value.to_u64())
                .map(|number| number as usize)
                .ok_or_else(|| format!("missing or non-integer metadata gliner2.{name}"))
        };
        let f32_meta = |name: &str| -> Result<f32, String> {
            source
                .metadata(&key(name))
                .and_then(|value| value.to_f64())
                .map(|number| number as f32)
                .ok_or_else(|| format!("missing or non-numeric metadata gliner2.{name}"))
        };
        // The converter only ever writes a positive window, and a zero here
        // would silently turn "local" into "one token wide".
        let window = usize_meta("attention.local_window")?;
        let local_window = (window > 0).then_some(window);
        let identity: Vec<usize> = source
            .metadata(&key("identity_attn_norm_layers"))
            .and_then(|value| value.to_arr())
            .map(|rows| {
                rows.iter()
                    .filter_map(|row| row.to_u64())
                    .map(|n| n as usize)
                    .collect()
            })
            .unwrap_or_default();
        Ok(EttinConfig {
            n_layer: usize_meta("block_count")?,
            n_embd: usize_meta("embedding_length")?,
            n_head: usize_meta("attention.head_count")?,
            head_dim: usize_meta("attention.head_dim")?,
            n_ff: usize_meta("feed_forward_length")?,
            norm_eps: f32_meta("attention.norm_epsilon")?,
            local_window,
            global_every_n_layers: usize_meta("attention.global_every_n_layers")?,
            rope_theta: f32_meta("rope.theta")?,
            max_position_embeddings: usize_meta("rope.max_position_embeddings")?,
            vocab_size: usize_meta("vocab_size")?,
            identity_attn_norm: identity,
        })
    }

    /// `modeling_modernbert.py:484`: a layer is local when its index is not a
    /// multiple of `global_attn_every_n_layers`.
    pub fn is_local_layer(&self, layer: usize) -> bool {
        self.local_window.is_some() && !layer.is_multiple_of(self.global_every_n_layers)
    }

    /// The `(left, right)` reach of a local layer's window, or `None` for global.
    pub fn local_attention(&self, layer: usize) -> Option<(usize, usize)> {
        if !self.is_local_layer(layer) {
            return None;
        }
        let half = self.local_window.expect("checked by is_local_layer") / 2;
        Some((half, half))
    }

    /// Sanity-check the relationships the forward relies on, once, at load time.
    pub fn validate(&self) -> Result<(), String> {
        if self.n_embd != self.n_head * self.head_dim {
            return Err(format!(
                "embedding_length {} is not head_count {} x head_dim {}",
                self.n_embd, self.n_head, self.head_dim
            ));
        }
        if self.global_every_n_layers == 0 {
            return Err("global_every_n_layers must be positive".into());
        }
        if let Some(window) = self.local_window {
            if window % 2 != 0 {
                return Err(format!(
                    "local window {window} is odd; the reference halves it into \
                     (left, right) and an odd width would lose a token"
                ));
            }
        }
        Ok(())
    }
}

/// One encoder block's weights, held as `Weight` so the F32 path goes through
/// the repo's own kernel (no hand-rolled reinterpretation) and a quantized
/// checkpoint would still dispatch correctly. `attn_norm` is `None` for the
/// layers whose `attn_norm` is `Identity` (layer 0 here), which is also why the
/// checkpoint stores no such tensor.
pub struct EttinLayer<'a> {
    pub attn_qkv: Weight<'a>,
    pub attn_out: Weight<'a>,
    pub attn_norm: Option<Weight<'a>>,
    pub mlp_in: Weight<'a>,
    pub mlp_out: Weight<'a>,
    pub mlp_norm: Weight<'a>,
}

pub struct EttinWeights<'a> {
    /// The embedding table, kept row-major as `[vocab, d]` so a token is one
    /// contiguous slice. GGUF stores the reversed dims `[d, vocab]`, so this is
    /// loaded by a dedicated check rather than as a matrix.
    pub token_embd: Vec<f32>,
    pub embd_norm: Weight<'a>,
    pub final_norm: Weight<'a>,
    pub layers: Vec<EttinLayer<'a>>,
    pub classifier0_weight: Weight<'a>,
    pub classifier0_bias: Vec<f32>,
    pub classifier2_weight: Weight<'a>,
    pub classifier2_bias: f32,
}

fn load<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    // GGUF stores torch dims reversed, so a weight matrix is `[n_in, n_out]`.
    // Norms and biases are genuinely 1-D, and `[3584]` vs `[1, 3584]` are
    // different tensors — ModernBERT's classifier output projection is the
    // matrix — so `load` always expects a matrix and `load_vec` below is the
    // only route to a vector.
    let expected: &[u64] = &[n_in as u64, n_out as u64];
    if info.dims != expected {
        return Err(format!(
            "tensor {name} has dims {:?}, expected {expected:?}",
            info.dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("missing tensor data {name}"))?;
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in.max(1),
        n_out.max(1),
    )))
}

fn f32_of<'a>(weight: &'a Weight<'_>) -> Result<&'a [f32], String> {
    weight
        .kernel
        .f32_slice()
        .ok_or_else(|| "expected an F32 weight; the Ettin converter only emits F32".to_string())
}

/// A 1-D tensor (a norm weight or a bias), wrapped so callers still see a
/// `Weight`. The dims are checked as a single dimension, not as `[1, len]`.
fn load_vec<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    len: usize,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if info.dims != [len as u64] {
        return Err(format!(
            "tensor {name} has dims {:?}, expected [{len}] (1-D)",
            info.dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("missing tensor data {name}"))?;
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        len,
        1,
    )))
}

/// The `[vocab, d]` embedding table. GGUF reverses torch dims, so the on-disk
/// shape is `[d, vocab]`; the payload is still row-major, i.e. token-major.
fn load_embedding<S: TensorSource + ?Sized>(
    source: &S,
    vocab: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info("token_embd.weight")
        .ok_or_else(|| "missing tensor token_embd.weight".to_string())?;
    if info.dims != [width as u64, vocab as u64] {
        return Err(format!(
            "tensor token_embd.weight has dims {:?}, expected [{width}, {vocab}]",
            info.dims
        ));
    }
    let bytes = source
        .tensor_slice("token_embd.weight")
        .ok_or_else(|| "missing tensor data token_embd.weight".to_string())?;
    if bytes.len() != vocab * width * 4 {
        return Err(format!(
            "token_embd.weight payload is {} bytes, expected {}",
            bytes.len(),
            vocab * width * 4
        ));
    }
    let mut table = Vec::with_capacity(vocab * width);
    for chunk in bytes.chunks_exact(4) {
        table.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(table)
}

pub fn load_weights<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    config: &EttinConfig,
) -> Result<EttinWeights<'a>, String> {
    config.validate()?;
    let d = config.n_embd;
    let f = config.n_ff;
    let wide = d * 2;
    let mut layers = Vec::with_capacity(config.n_layer);
    for index in 0..config.n_layer {
        let prefix = format!("blk.{index}");
        layers.push(EttinLayer {
            attn_qkv: load(source, &format!("{prefix}.attn_qkv.weight"), d, 3 * d)?,
            attn_out: load(source, &format!("{prefix}.attn_out.weight"), d, d)?,
            attn_norm: if config.identity_attn_norm.contains(&index) {
                None
            } else {
                Some(load_vec(source, &format!("{prefix}.attn_norm.weight"), d)?)
            },
            mlp_in: load(source, &format!("{prefix}.mlp_in.weight"), d, 2 * f)?,
            mlp_out: load(source, &format!("{prefix}.mlp_out.weight"), f, d)?,
            mlp_norm: load_vec(source, &format!("{prefix}.mlp_norm.weight"), d)?,
        });
    }
    let classifier0_bias = f32_of(&load_vec(source, "classifier.0.bias", wide)?)?.to_vec();
    let classifier2_bias = f32_of(&load_vec(source, "classifier.2.bias", 1)?)?[0];
    let token_embd = load_embedding(source, config.vocab_size, d)?;
    Ok(EttinWeights {
        token_embd,
        embd_norm: load_vec(source, "embd_norm.weight", d)?,
        final_norm: load_vec(source, "final_norm.weight", d)?,
        layers,
        classifier0_weight: load(source, "classifier.0.weight", d, wide)?,
        classifier0_bias,
        classifier2_weight: load(source, "classifier.2.weight", wide, 1)?,
        classifier2_bias,
    })
}

/// `qkv @ x` for a `[rows, cols]` row-major F32 weight, with no bias.
fn linear_no_bias(input: &[f32], weight: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; n_out];
    for (out_index, row) in weight.chunks_exact(n_in).take(n_out).enumerate() {
        output[out_index] = crate::ops::dot_f32(row, input, n_in);
    }
    output
}

/// The same projection over a `[rows, n_in]` matrix.
///
/// A whole sequence goes through this rather than through [`linear_no_bias`],
/// which projects a single row: passing `seq * n_in` values to that one would
/// silently produce the projection of the first token and then broadcast it.
fn linear_rows(input: &[f32], weight: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let rows = input.len() / n_in;
    let mut output = vec![0.0f32; rows * n_out];
    let table: Vec<&[f32]> = weight.chunks_exact(n_in).take(n_out).collect();
    for row in 0..rows {
        let source = &input[row * n_in..(row + 1) * n_in];
        let target = &mut output[row * n_out..(row + 1) * n_out];
        for (out_index, weights_row) in table.iter().enumerate() {
            target[out_index] = crate::ops::dot_f32(weights_row, source, n_in);
        }
    }
    output
}

/// `Linear` with bias, for the two-layer classifier head.
fn linear_bias(input: &[f32], weight: &[f32], bias: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let mut output = linear_no_bias(input, weight, n_in, n_out);
    for (value, b) in output.iter_mut().zip(bias.iter()) {
        *value += b;
    }
    output
}

/// LayerNorm with no bias, which is all ModernBERT uses (`norm_bias = False`),
/// over a `[rows, width]` matrix — one normalization per row.
fn layer_norm_no_bias(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let width = weight.len();
    assert_eq!(
        input.len() % width,
        0,
        "hidden state of {} rows is not a multiple of width {width}",
        input.len()
    );
    let mut output = vec![0.0f32; input.len()];
    for (source, target) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        crate::ops::layer_norm(source, weight, &[], eps, target);
    }
    output
}

/// NeoX-style rotary embedding: `rotate_half` over the first and second halves
/// of `head_dim`, not interleaved adjacent pairs.
fn rope_tables(head_dim: usize, positions: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; positions * head_dim];
    let mut sin = vec![0.0f32; positions * head_dim];
    for (position, (cos_row, sin_row)) in cos
        .chunks_exact_mut(head_dim)
        .zip(sin.chunks_exact_mut(head_dim))
        .enumerate()
    {
        for (index, slot) in cos_row.iter_mut().enumerate() {
            // `emb = cat((freqs, freqs), -1)`: the table is `head_dim` wide but
            // carries only `head_dim / 2` distinct frequencies, repeated. Letting
            // the exponent keep climbing across the second half would rotate the
            // upper half at the wrong rate.
            let exponent = (index % half) as f32 / half as f32;
            let angle = position as f32 * (1.0f32 / theta).powf(exponent);
            *slot = angle.cos();
            sin_row[index] = angle.sin();
        }
    }
    (cos, sin)
}

/// Split `Wqkv`'s output into per-head Q, K and V.
///
/// The fused projection is read as `[seq, 3, heads, head_dim]`, so the three
/// roles interleave with stride `heads * head_dim` rather than sitting in three
/// contiguous blocks. The output is head-major (`[heads, seq, head_dim]`), which
/// is the layout the per-head attention loop slices contiguously.
fn split_interleaved_qkv(
    qkv: &[f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut query = vec![0.0f32; seq * heads * head_dim];
    let mut key = vec![0.0f32; seq * heads * head_dim];
    let mut value = vec![0.0f32; seq * heads * head_dim];
    let head_width = head_dim;
    for position in 0..seq {
        for head in 0..heads {
            let destination = (head * seq + position) * head_width;
            for component in 0..head_width {
                // `view(seq, 3, heads, head_dim)` flattens to
                // `position * (3 * heads * head_dim) + role * (heads * head_dim)
                //  + head * head_dim + component`: the role axis is the second,
                // so it strides by `heads * head_dim` within one position, and
                // `component` is the innermost index and does not stride at all.
                let source = position * 3 * heads * head_dim + head * head_width + component;
                query[destination + component] = qkv[source];
                key[destination + component] = qkv[source + heads * head_dim];
                value[destination + component] = qkv[source + 2 * heads * head_dim];
            }
        }
    }
    (query, key, value)
}

fn apply_rope(
    values: &mut [f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = head_dim / 2;
    for position in 0..seq {
        for head in 0..heads {
            let base = (head * seq + position) * head_dim;
            for index in 0..half {
                let first = values[base + index];
                let second = values[base + index + half];
                let cos_value = cos[position * head_dim + index];
                let sin_value = sin[position * head_dim + index];
                values[base + index] = first * cos_value - second * sin_value;
                values[base + index + half] = second * cos_value + first * sin_value;
            }
        }
    }
}

/// One attention head over `seq` tokens, honouring the local window.
/// One head's sliced Q/K/V, already narrowed to this head's `head_dim` stride.
struct HeadSlices<'a> {
    query: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
}

/// `scaled_dot_product_attention` for one head over `seq` tokens, honouring the
/// local window. The window is applied while scanning rather than as a post-hoc
/// mask, so the softmax only ever sums the keys it is allowed to use.
fn attend_head(
    slices: HeadSlices<'_>,
    seq: usize,
    head_dim: usize,
    window: Option<(usize, usize)>,
    output: &mut [f32],
) {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut scores = vec![f32::NEG_INFINITY; seq];
    for (position, row) in output.chunks_exact_mut(head_dim).enumerate() {
        let query_base = position * head_dim;
        let mut best = f32::NEG_INFINITY;
        let mut allowed_any = false;
        for (source, slot) in scores.iter_mut().enumerate() {
            if let Some((left, right)) = window {
                let reach = if position >= source { left } else { right };
                if position.abs_diff(source) > reach {
                    *slot = f32::NEG_INFINITY;
                    continue;
                }
            }
            let key_base = source * head_dim;
            let mut total = 0.0f32;
            for index in 0..head_dim {
                total += slices.query[query_base + index] * slices.key[key_base + index];
            }
            *slot = total * scale;
            best = best.max(*slot);
            allowed_any = true;
        }
        if !allowed_any {
            // Only reachable for a degenerate window; zeros rather than NaN so a
            // downstream comparison fails on the value instead of propagating.
            row.fill(0.0);
            continue;
        }
        let mut denominator = 0.0f32;
        for score in scores.iter_mut() {
            if score.is_finite() {
                *score = (*score - best).exp();
                denominator += *score;
            } else {
                *score = 0.0;
            }
        }
        let inverse = if denominator > 0.0 {
            1.0 / denominator
        } else {
            0.0
        };
        for (index, slot) in row.iter_mut().enumerate() {
            let mut total = 0.0f32;
            for (source, score) in scores.iter().enumerate() {
                total += score * slices.value[source * head_dim + index];
            }
            *slot = total * inverse;
        }
    }
}

/// `ModernBertModel.forward` for a single sequence: embeddings, the 28 blocks,
/// then `final_norm`.
///
/// `attention_mask` marks real tokens; padding is excluded from every softmax so
/// a padded slot cannot contribute to a real one.
pub fn encode(
    weights: &EttinWeights<'_>,
    config: &EttinConfig,
    input_ids: &[u32],
    attention_mask: &[bool],
) -> Result<Vec<f32>, String> {
    let seq = input_ids.len();
    if seq == 0 {
        return Err("empty input".into());
    }
    if attention_mask.len() != seq {
        return Err(format!(
            "attention_mask has {} entries for {seq} tokens",
            attention_mask.len()
        ));
    }
    if seq > config.max_position_embeddings {
        return Err(format!(
            "sequence of {seq} exceeds max_position_embeddings {}",
            config.max_position_embeddings
        ));
    }
    let d = config.n_embd;

    // Embedding lookup followed by `embeddings.norm`.
    let token_embd = &weights.token_embd;
    let mut hidden = vec![0.0f32; seq * d];
    for (position, &id) in input_ids.iter().enumerate() {
        let id = id as usize;
        if id >= config.vocab_size {
            return Err(format!(
                "token id {id} is outside the {}-row embedding",
                config.vocab_size
            ));
        }
        let source = id * d;
        let destination = position * d;
        hidden[destination..destination + d].copy_from_slice(&token_embd[source..source + d]);
    }
    let normalized = layer_norm_no_bias(&hidden, f32_of(&weights.embd_norm)?, config.norm_eps);
    hidden.copy_from_slice(&normalized);

    let (cos, sin) = rope_tables(config.head_dim, seq, config.rope_theta);
    let rows = seq;

    for (index, layer) in weights.layers.iter().enumerate() {
        // x + attn(attn_norm(x)), with `attn_norm` skipped where it is Identity.
        let attention_input = match &layer.attn_norm {
            Some(norm) => layer_norm_no_bias(&hidden, f32_of(norm)?, config.norm_eps),
            None => hidden.clone(),
        };
        let qkv = linear_rows(&attention_input, f32_of(&layer.attn_qkv)?, d, 3 * d);
        let (mut query, mut key, value) =
            split_interleaved_qkv(&qkv, seq, config.n_head, config.head_dim);
        apply_rope(&mut query, seq, config.n_head, config.head_dim, &cos, &sin);
        apply_rope(&mut key, seq, config.n_head, config.head_dim, &cos, &sin);

        let mut context = vec![0.0f32; seq * d];
        let window = config.local_attention(index);
        for head in 0..config.n_head {
            let head_width = config.head_dim;
            // Head-major: head `h` owns a contiguous `[seq, head_dim]` block.
            let offset = head * seq * head_width;
            let mut head_out = vec![0.0f32; seq * head_width];
            attend_head(
                HeadSlices {
                    query: &query[offset..offset + seq * head_width],
                    key: &key[offset..offset + seq * head_width],
                    value: &value[offset..offset + seq * head_width],
                },
                seq,
                head_width,
                window,
                &mut head_out,
            );
            // Scatter back into the `[seq, d]` context, where a head's columns
            // are strided by `d` rather than contiguous.
            for position in 0..seq {
                let destination = position * d + head * head_width;
                context[destination..destination + head_width]
                    .copy_from_slice(&head_out[position * head_width..(position + 1) * head_width]);
            }
        }
        let projected = linear_rows(&context, f32_of(&layer.attn_out)?, d, d);
        for (value, addend) in hidden.iter_mut().zip(projected.iter()) {
            *value += addend;
        }

        // x + mlp(mlp_norm(x)), where mlp is a GeLU GLU: `chunk(2)` then
        // `act(first) * second`.
        let mlp_input = layer_norm_no_bias(&hidden, f32_of(&layer.mlp_norm)?, config.norm_eps);
        let gate = linear_rows(&mlp_input, f32_of(&layer.mlp_in)?, d, 2 * config.n_ff);
        // `chunk(2)` then `act(first) * second`, per token. The gate is
        // `[rows, 2 * n_ff]`, so a token's two halves sit `2 * n_ff` apart and
        // the gated result has to be gathered into a fresh `[rows, n_ff]`
        // buffer. Leaving it in place would look right for row 0 — whose halves
        // are contiguous anyway — and pair every later token's gate with the
        // previous token's projection.
        let mut activated = vec![0.0f32; rows * config.n_ff];
        for row in 0..rows {
            let base = row * 2 * config.n_ff;
            let (first, second) = gate[base..base + 2 * config.n_ff].split_at(config.n_ff);
            let target = &mut activated[row * config.n_ff..(row + 1) * config.n_ff];
            target.copy_from_slice(first);
            crate::ops::gelu_inplace(target);
            for index in 0..config.n_ff {
                target[index] *= second[index];
            }
        }
        let mlp_out = linear_rows(&activated, f32_of(&layer.mlp_out)?, config.n_ff, d);
        for (value, addend) in hidden.iter_mut().zip(mlp_out.iter()) {
            *value += addend;
        }

        // Padding must not leak into a later token: ModernBERT masks the input
        // to every block, not just the attention.
        for position in 0..seq {
            if !attention_mask[position] {
                for component in 0..d {
                    hidden[position * d + component] = 0.0;
                }
            }
        }
    }

    // `final_norm` is applied to the whole sequence, as in the reference.
    Ok(layer_norm_no_bias(
        &hidden,
        f32_of(&weights.final_norm)?,
        config.norm_eps,
    ))
}

/// The task head: `Linear(2d, d)` + ReLU + `Linear(2d, 1)`, the Decide contract.
pub fn classify(
    weights: &EttinWeights<'_>,
    config: &EttinConfig,
    hidden: &[f32],
) -> Result<f32, String> {
    let d = config.n_embd;
    if hidden.len() != d {
        return Err(format!(
            "hidden state has {} entries, expected {d}",
            hidden.len()
        ));
    }
    let mut wide = linear_bias(
        hidden,
        f32_of(&weights.classifier0_weight)?,
        &weights.classifier0_bias,
        d,
        2 * d,
    );
    for value in wide.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
    let logit = linear_no_bias(&wide, f32_of(&weights.classifier2_weight)?, 2 * d, 1);
    Ok(logit[0] + weights.classifier2_bias)
}
